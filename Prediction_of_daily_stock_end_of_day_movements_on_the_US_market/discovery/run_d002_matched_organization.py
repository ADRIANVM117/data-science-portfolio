"""Execute the one authorized target-conditioned D002 paired description."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from discovery.d001_path_structure_beyond_recent_rms import prepare_complete_w_population  # noqa: E402
from discovery.run_d001_path_structure_beyond_recent_rms import RESULTS_DIR, load_discovery_rows  # noqa: E402
from discovery.run_d001_shap_interpretation import validate_physical_discovery_boundary  # noqa: E402
from discovery.d002_local_organization import DISCOVERY_BLOCKS, local_organization_statistics  # noqa: E402


PAIR_PATH = RESULTS_DIR / "D002_rms_matched_pairs.csv"
QUALITY_PATH = RESULTS_DIR / "D002_rms_matching_quality.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_delta(values: pd.Series, *, block: str, statistic: str) -> dict[str, float | int | str]:
    """Return the exact descriptive summary approved for a paired D002 delta."""
    if values.empty or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise AssertionError("D002 paired delta values must be non-empty and finite.")
    return {
        "block": block,
        "statistic": statistic,
        "n_matched_pairs": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std_ddof_1": float(values.std(ddof=1)),
        "p05": float(values.quantile(0.05)),
        "p25": float(values.quantile(0.25)),
        "p75": float(values.quantile(0.75)),
        "p95": float(values.quantile(0.95)),
        "fraction_positive": float(values.gt(0.0).mean()),
        "fraction_zero": float(values.eq(0.0).mean()),
        "fraction_negative": float(values.lt(0.0).mean()),
    }


def block_for_days(days: pd.Series, *, block: str) -> None:
    boundaries = {name: (start, end) for name, start, end in DISCOVERY_BLOCKS}
    start, end = boundaries[block]
    if not days.between(start, end).all():
        raise AssertionError(f"D002 frozen pairs escaped chronological {block}.")


def main() -> None:
    validate_physical_discovery_boundary()
    if not PAIR_PATH.exists() or not QUALITY_PATH.exists():
        raise FileNotFoundError("D002 scientific analysis requires the frozen matching artifacts.")
    pairs = pd.read_csv(PAIR_PATH)
    frozen_quality = pd.read_csv(QUALITY_PATH)
    required_pair_columns = {"block", "neutral_ID", "directional_ID", "neutral_I_recent", "directional_I_recent", "abs_rms_difference"}
    if not required_pair_columns.issubset(pairs.columns) or pairs["neutral_ID"].duplicated().any():
        raise AssertionError("D002 frozen pair artifact has invalid membership or schema.")
    if len(pairs) != 125_961 or set(pairs["block"]) != {name for name, _, _ in DISCOVERY_BLOCKS}:
        raise AssertionError("D002 frozen pairs differ from the approved matching audit.")

    features, target, development_equities = load_discovery_rows()
    complete_features, complete_target, _ = prepare_complete_w_population(features, target, development_equities)
    statistics = local_organization_statistics(complete_features)
    lookup = pd.DataFrame(
        {
            "ID": complete_features["ID"].to_numpy(),
            "day": complete_features["day"].to_numpy(),
            "Z": complete_target.to_numpy(),
            "E": statistics["E"].to_numpy(),
            "A_adj": statistics["A_adj"].to_numpy(),
            "H_peak": statistics["H_peak"].to_numpy(),
        }
    ).set_index("ID", verify_integrity=True)
    if not set(pairs["neutral_ID"]).issubset(lookup.index) or not set(pairs["directional_ID"]).issubset(lookup.index):
        raise AssertionError("D002 frozen pair IDs are absent from the complete-W physical population.")
    neutral = lookup.loc[pairs["neutral_ID"]].reset_index().add_prefix("neutral_")
    directional = lookup.loc[pairs["directional_ID"]].reset_index().add_prefix("directional_")
    if not neutral["neutral_Z"].eq(0).all() or not directional["directional_Z"].eq(1).all():
        raise AssertionError("D002 pair target identities differ from frozen matching membership.")
    assembled = pd.concat([pairs.reset_index(drop=True), neutral.drop(columns="neutral_ID"), directional.drop(columns="directional_ID")], axis=1)
    for block, _, _ in DISCOVERY_BLOCKS:
        subset = assembled.loc[assembled["block"].eq(block)]
        block_for_days(subset["neutral_day"], block=block)
        block_for_days(subset["directional_day"], block=block)
    assembled["Delta_A_adj"] = assembled["directional_A_adj"] - assembled["neutral_A_adj"]
    assembled["Delta_H_peak"] = assembled["directional_H_peak"] - assembled["neutral_H_peak"]
    if not np.isfinite(assembled[["Delta_A_adj", "Delta_H_peak"]].to_numpy(dtype=float)).all():
        raise AssertionError("D002 organization deltas must be finite.")

    a_summary: list[dict[str, float | int | str]] = []
    h_summary: list[dict[str, float | int | str]] = []
    zero_records: list[dict[str, int | str]] = []
    for block, _, _ in DISCOVERY_BLOCKS:
        subset = assembled.loc[assembled["block"].eq(block)]
        a_summary.append(summarize_delta(subset["Delta_A_adj"], block=block, statistic="Delta_A_adj"))
        h_summary.append(summarize_delta(subset["Delta_H_peak"], block=block, statistic="Delta_H_peak"))
        zero_records.append(
            {
                "block": block,
                "n_pairs": int(len(subset)),
                "zero_energy_neutral": int(subset["neutral_E"].eq(0.0).sum()),
                "zero_energy_directional": int(subset["directional_E"].eq(0.0).sum()),
                "zero_energy_both": int((subset["neutral_E"].eq(0.0) & subset["directional_E"].eq(0.0)).sum()),
            }
        )
    a_summary.append(summarize_delta(assembled["Delta_A_adj"], block="pooled_discovery", statistic="Delta_A_adj"))
    h_summary.append(summarize_delta(assembled["Delta_H_peak"], block="pooled_discovery", statistic="Delta_H_peak"))
    zero_records.append(
        {
            "block": "pooled_discovery",
            "n_pairs": int(len(assembled)),
            "zero_energy_neutral": int(assembled["neutral_E"].eq(0.0).sum()),
            "zero_energy_directional": int(assembled["directional_E"].eq(0.0).sum()),
            "zero_energy_both": int((assembled["neutral_E"].eq(0.0) & assembled["directional_E"].eq(0.0)).sum()),
        }
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    assembled.to_csv(RESULTS_DIR / "D002_matched_organization_pairs.csv", index=False)
    pd.DataFrame(a_summary).to_csv(RESULTS_DIR / "D002_delta_a_adj_summary.csv", index=False)
    pd.DataFrame(h_summary).to_csv(RESULTS_DIR / "D002_delta_h_peak_summary.csv", index=False)
    pd.DataFrame(zero_records).to_csv(RESULTS_DIR / "D002_zero_energy_pair_diagnostics.csv", index=False)
    metadata = {
        "candidate": "D002",
        "analysis": "one_authorized_descriptive_paired_local_organization_pass",
        "matching_artifact_sha256": sha256_file(PAIR_PATH),
        "matching_quality_artifact_sha256": sha256_file(QUALITY_PATH),
        "pair_count": int(len(assembled)),
        "matching_recomputed": False,
        "caliper_applied": False,
        "predictive_model_trained": False,
        "internal_confirmation_accessed": False,
        "e_holdout_accessed": False,
        "competition_test_accessed": False,
    }
    (RESULTS_DIR / "D002_organization_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(a_summary).to_string(index=False))
    print(pd.DataFrame(h_summary).to_string(index=False))
    print(pd.DataFrame(zero_records).to_string(index=False))
    print(f"Frozen matching quality rows: {len(frozen_quality)}")


if __name__ == "__main__":
    main()
