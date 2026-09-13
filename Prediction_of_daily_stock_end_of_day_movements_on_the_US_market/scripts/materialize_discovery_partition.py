"""Materialize the target-blind, physically isolated Discovery partition.

This script is a controlled raw-partitioning operation.  It may parse the
original training input and label files only to construct the immutable
Discovery files.  It performs no analytical statistics, feature selection,
target summary, modelling, or evaluation.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.exp000_validation import N_DEV_EQUITIES, N_HOLDOUT_EQUITIES, validate_equity_partition  # noqa: E402


DATA_DIR = PROJECT_ROOT / "data"
DISCOVERY_DIR = DATA_DIR / "discovery"
FULL_INPUT_PATH = DATA_DIR / "input_training.csv"
PARTITION_PATH = PROJECT_ROOT / "experiments" / "EXP_000_equity_partition.csv"
DISCOVERY_INPUT_FILENAME = "discovery_input_training.csv"
DISCOVERY_LABEL_FILENAME = "discovery_output_training.csv"
MANIFEST_FILENAME = "discovery_partition_manifest.json"
EXPECTED_DISCOVERY_ROWS = 472_816
DISCOVERY_DAY_MIN = 0
DISCOVERY_DAY_MAX = 352
HASH_ALGORITHM = "sha256"
CHUNK_SIZE = 100_000


def sha256_file(path: Path) -> str:
    """Return the content SHA-256 of one file without interpreting it."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def label_source_path(data_dir: Path = DATA_DIR) -> Path:
    """Locate the single raw training-label file; no test paths are considered."""
    candidates = sorted(data_dir.glob("output_training_*.csv"))
    if len(candidates) != 1:
        raise FileNotFoundError("Partition materialization requires exactly one training-label source.")
    return candidates[0]


def development_equities(partition: pd.DataFrame) -> set[int]:
    """Return E_dev IDs from the frozen, validated EXP_000 partition."""
    if partition["equity"].duplicated().any():
        raise AssertionError("Frozen equity partition contains duplicate equity IDs.")
    if int(partition["partition"].eq("E_dev").sum()) != N_DEV_EQUITIES:
        raise AssertionError("Frozen partition must contain exactly the approved E_dev count.")
    if int(partition["partition"].eq("E_holdout").sum()) != N_HOLDOUT_EQUITIES:
        raise AssertionError("Frozen partition must contain exactly the approved E_holdout count.")
    return set(partition.loc[partition["partition"].eq("E_dev"), "equity"])


def authorized_membership(frame: pd.DataFrame, development_ids: Iterable[int]) -> pd.Series:
    """Apply the only permitted, target-blind Discovery membership rule."""
    required = {"ID", "day", "equity"}
    if not required.issubset(frame.columns):
        raise AssertionError("Membership requires ID, day, and equity only.")
    return frame["day"].between(DISCOVERY_DAY_MIN, DISCOVERY_DAY_MAX) & frame["equity"].isin(set(development_ids))


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _path_reference(path: Path) -> str:
    """Use a repository-relative reference when possible, otherwise an absolute test path."""
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _validate_partition_frames(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    development_ids: set[int],
    *,
    expected_rows: int,
) -> None:
    if len(features) != expected_rows:
        raise AssertionError(f"Discovery input row count must be {expected_rows}.")
    if features["ID"].duplicated().any() or labels["ID"].duplicated().any():
        raise AssertionError("Discovery input and labels must both have unique IDs.")
    if not features["day"].between(DISCOVERY_DAY_MIN, DISCOVERY_DAY_MAX).all():
        raise AssertionError("Discovery partition contains an Internal Confirmation day.")
    if not features["equity"].isin(development_ids).all():
        raise AssertionError("Discovery partition contains an E_holdout equity.")
    if set(features["ID"]) != set(labels["ID"]):
        raise AssertionError("Discovery input and label ID sets must be exactly equal.")
    if not features["ID"].equals(labels["ID"]):
        raise AssertionError("Discovery labels must be deterministically aligned to input order.")


def _existing_partition_is_valid(
    discovery_dir: Path,
    development_ids: set[int],
    *,
    expected_rows: int,
    source_input_hash: str,
    source_label_hash: str,
    partition_hash: str,
) -> bool:
    input_path = discovery_dir / DISCOVERY_INPUT_FILENAME
    label_path = discovery_dir / DISCOVERY_LABEL_FILENAME
    manifest_path = discovery_dir / MANIFEST_FILENAME
    if not all(path.exists() for path in (input_path, label_path, manifest_path)):
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_hashes = {
        "source_input_sha256": source_input_hash,
        "source_label_sha256": source_label_hash,
        "equity_partition_sha256": partition_hash,
        "input_sha256": sha256_file(input_path),
        "label_sha256": sha256_file(label_path),
    }
    if any(manifest.get(key) != value for key, value in required_hashes.items()):
        return False
    features = pd.read_csv(input_path)
    labels = pd.read_csv(label_path)
    _validate_partition_frames(features, labels, development_ids, expected_rows=expected_rows)
    return True


def materialize_discovery_partition(
    *,
    data_dir: Path = DATA_DIR,
    partition_path: Path = PARTITION_PATH,
    discovery_dir: Path = DISCOVERY_DIR,
    expected_rows: int = EXPECTED_DISCOVERY_ROWS,
) -> dict[str, object]:
    """Build or validate the physical partition using only the frozen rule.

    The input membership filter is intentionally applied before labels are
    read.  Label values are copied solely by the resulting authorized ID set.
    """
    full_input_path = data_dir / FULL_INPUT_PATH.name
    full_label_path = label_source_path(data_dir)
    if not full_input_path.exists():
        raise FileNotFoundError("Controlled partitioning requires the raw training input source.")
    if not partition_path.exists():
        raise FileNotFoundError("Controlled partitioning requires the frozen EXP_000 partition.")
    partition = pd.read_csv(partition_path)
    # This validates the entire frozen partition structure without examining labels or returns.
    validate_equity_partition(partition, partition["equity"].unique(), n_holdout=N_HOLDOUT_EQUITIES)
    development_ids = development_equities(partition)
    source_input_hash = sha256_file(full_input_path)
    source_label_hash = sha256_file(full_label_path)
    partition_hash = sha256_file(partition_path)
    if _existing_partition_is_valid(
        discovery_dir,
        development_ids,
        expected_rows=expected_rows,
        source_input_hash=source_input_hash,
        source_label_hash=source_label_hash,
        partition_hash=partition_hash,
    ):
        return json.loads((discovery_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))

    retained_inputs: list[pd.DataFrame] = []
    for chunk in pd.read_csv(full_input_path, chunksize=CHUNK_SIZE):
        # Selection is deliberately computed from the three permitted columns only.
        allowed = authorized_membership(chunk[["ID", "day", "equity"]], development_ids)
        if allowed.any():
            retained_inputs.append(chunk.loc[allowed].copy())
    if not retained_inputs:
        raise AssertionError("Target-blind Discovery selection retained no input rows.")
    features = pd.concat(retained_inputs, ignore_index=True)
    if features["ID"].duplicated().any():
        raise AssertionError("Raw training input unexpectedly contains duplicate authorized IDs.")
    authorized_ids = pd.Index(features["ID"], name="ID")

    retained_labels: list[pd.DataFrame] = []
    authorized_id_set = set(authorized_ids)
    for chunk in pd.read_csv(full_label_path, usecols=["ID", "reod"], chunksize=CHUNK_SIZE):
        selected = chunk.loc[chunk["ID"].isin(authorized_id_set)].copy()
        if not selected.empty:
            retained_labels.append(selected)
    labels_by_id = pd.concat(retained_labels, ignore_index=True)
    if labels_by_id["ID"].duplicated().any():
        raise AssertionError("Raw training labels unexpectedly contain duplicate authorized IDs.")
    labels = labels_by_id.set_index("ID").reindex(authorized_ids).reset_index()
    if labels["reod"].isna().any():
        raise AssertionError("An authorized input ID has no corresponding training label.")
    _validate_partition_frames(features, labels, development_ids, expected_rows=expected_rows)

    discovery_dir.mkdir(parents=True, exist_ok=True)
    input_path = discovery_dir / DISCOVERY_INPUT_FILENAME
    label_path = discovery_dir / DISCOVERY_LABEL_FILENAME
    _atomic_csv(features, input_path)
    _atomic_csv(labels, label_path)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "controlled_raw_partitioning_target_blind_membership",
        "membership_rule": "0 <= day <= 352 AND equity in frozen E_dev",
        "source_input_filename": full_input_path.name,
        "source_label_filename": full_label_path.name,
        "authorized_day_min": DISCOVERY_DAY_MIN,
        "authorized_day_max": DISCOVERY_DAY_MAX,
        "equity_partition_reference": _path_reference(partition_path),
        "e_dev_count": len(development_ids),
        "row_count": int(len(features)),
        "unique_id_count": int(features["ID"].nunique()),
        "unique_day_count": int(features["day"].nunique()),
        "unique_equity_count": int(features["equity"].nunique()),
        "hash_algorithm": HASH_ALGORITHM,
        "source_input_sha256": source_input_hash,
        "source_label_sha256": source_label_hash,
        "equity_partition_sha256": partition_hash,
        "input_sha256": sha256_file(input_path),
        "label_sha256": sha256_file(label_path),
        "construction_script": "scripts/materialize_discovery_partition.py",
    }
    _atomic_json(manifest, discovery_dir / MANIFEST_FILENAME)
    return manifest


def main() -> None:
    manifest = materialize_discovery_partition()
    print(
        "Discovery partition validated: "
        f"rows={manifest['row_count']}, IDs={manifest['unique_id_count']}, "
        f"days={manifest['unique_day_count']}, equities={manifest['unique_equity_count']}"
    )


if __name__ == "__main__":
    main()
