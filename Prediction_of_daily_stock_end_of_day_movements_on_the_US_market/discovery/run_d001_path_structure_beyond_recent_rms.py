"""Run D001 only when separately authorized; training data are Discovery-filtered."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import sklearn
import xgboost


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from discovery.d001_path_structure_beyond_recent_rms import (  # noqa: E402
    DISCOVERY_DAY_END,
    DISCOVERY_FOLDS,
    RECENT_WINDOW_COLUMNS,
    assert_both_binary_classes,
    assert_identical_representation_rows,
    build_d001_representations,
    incremental_auc_records,
    make_d001_classifier,
    prepare_complete_w_population,
    split_discovery_fold,
)


PARTITION_PATH = PROJECT_ROOT / "experiments" / "EXP_000_equity_partition.csv"
RESULTS_DIR = PROJECT_ROOT / "discovery" / "results"
DISCOVERY_DATA_DIR = PROJECT_ROOT / "data" / "discovery"
DISCOVERY_INPUT_PATH = DISCOVERY_DATA_DIR / "discovery_input_training.csv"
DISCOVERY_LABEL_PATH = DISCOVERY_DATA_DIR / "discovery_output_training.csv"
DISCOVERY_MANIFEST_PATH = DISCOVERY_DATA_DIR / "discovery_partition_manifest.json"
EXPECTED_DISCOVERY_ROWS = 472_816


def load_discovery_rows() -> tuple[pd.DataFrame, pd.Series, set[int]]:
    """Load only the validated physical Discovery partition for D001.

    Ordinary Discovery execution must never open the complete training files.
    Those files are permitted only to the separately authorized controlled
    materialization operation.
    """
    required_paths = (DISCOVERY_INPUT_PATH, DISCOVERY_LABEL_PATH, DISCOVERY_MANIFEST_PATH)
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "D001 requires the validated physical Discovery partition; missing: " + ", ".join(missing)
        )
    partition = pd.read_csv(PARTITION_PATH)
    development_equities = set(partition.loc[partition["partition"].eq("E_dev"), "equity"])
    features = pd.read_csv(DISCOVERY_INPUT_PATH)
    labels = pd.read_csv(DISCOVERY_LABEL_PATH, usecols=["ID", "reod"])
    if len(features) != EXPECTED_DISCOVERY_ROWS or len(labels) != EXPECTED_DISCOVERY_ROWS:
        raise AssertionError("D001 physical Discovery partition has an unexpected row count.")
    if features["ID"].duplicated().any() or labels["ID"].duplicated().any():
        raise AssertionError("D001 physical Discovery IDs must be unique.")
    if not features["ID"].equals(labels["ID"]):
        raise AssertionError("D001 physical Discovery labels must align exactly to input IDs.")
    if not features["day"].between(0, DISCOVERY_DAY_END).all():
        raise AssertionError("D001 physical Discovery partition contains a non-Discovery day.")
    if not features["equity"].isin(development_equities).all():
        raise AssertionError("D001 physical Discovery partition contains an E_holdout equity.")
    target = pd.Series(np.where(labels["reod"].eq(0), 0, 1), index=features.index, name="Z", dtype="int8")
    return features, target, development_equities


def main() -> None:
    started = time.perf_counter()
    features, target, development_equities = load_discovery_rows()
    metrics: list[dict[str, float | int | str]] = []
    manifest: list[dict[str, float | int | str]] = []
    fold_deltas: dict[int, float] = {}
    for fold in DISCOVERY_FOLDS:
        (fit_features, fit_target), (validation_features, validation_target) = split_discovery_fold(
            features, target, development_equities, fold
        )
        fit_features, fit_target, _ = prepare_complete_w_population(fit_features, fit_target, development_equities)
        validation_features, validation_target, _ = prepare_complete_w_population(
            validation_features, validation_target, development_equities
        )
        assert_both_binary_classes(fit_target, context=f"fit fold {fold.fold}")
        assert_both_binary_classes(validation_target, context=f"validation fold {fold.fold}")
        fit_representations = build_d001_representations(fit_features)
        validation_representations = build_d001_representations(validation_features)
        assert_identical_representation_rows(fit_representations, fit_features, fit_target)
        assert_identical_representation_rows(validation_representations, validation_features, validation_target)
        for subset_name, subset_features, subset_target in (
            ("fit", fit_features, fit_target),
            ("validation", validation_features, validation_target),
        ):
            manifest.append(
                {
                    "fold": fold.fold,
                    "subset": subset_name,
                    "n_rows": len(subset_target),
                    "n_neutral": int(subset_target.eq(0).sum()),
                    "n_directional": int(subset_target.eq(1).sum()),
                    "n_days": int(subset_features["day"].nunique()),
                    "n_equities": int(subset_features["equity"].nunique()),
                    "min_day": int(subset_features["day"].min()),
                    "max_day": int(subset_features["day"].max()),
                }
            )
        scores: dict[str, tuple[float, float, float]] = {}
        for condition in ("C", "P"):
            classifier = make_d001_classifier()
            model_started = time.perf_counter()
            classifier.fit(fit_representations[condition], fit_target)
            runtime_seconds = time.perf_counter() - model_started
            if tuple(classifier.classes_) != (0, 1):
                raise AssertionError("D001 classifier must fit both binary classes.")
            fit_probability = classifier.predict_proba(fit_representations[condition])[:, 1]
            validation_probability = classifier.predict_proba(validation_representations[condition])[:, 1]
            scores[condition] = (
                float(roc_auc_score(fit_target, fit_probability)),
                float(roc_auc_score(validation_target, validation_probability)),
                runtime_seconds,
            )
            metrics.append(
                {
                    "fold": fold.fold,
                    "condition": condition,
                    "auc_fit": scores[condition][0],
                    "auc_validation": scores[condition][1],
                    "fit_minus_validation_auc": scores[condition][0] - scores[condition][1],
                    "training_runtime_seconds": runtime_seconds,
                }
            )
        fold_deltas[fold.fold] = scores["P"][1] - scores["C"][1]

    metrics_frame = pd.DataFrame(metrics).sort_values(["fold", "condition"])
    summary = metrics_frame.groupby("condition")[["auc_fit", "auc_validation", "fit_minus_validation_auc", "training_runtime_seconds"]].agg(
        ["mean", lambda values: values.std(ddof=1)]
    )
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    summary = summary.rename(columns=lambda name: name.replace("<lambda_0>", "std_ddof_1")).reset_index()
    deltas = incremental_auc_records(fold_deltas)
    metadata = {
        "discovery_candidate": "D001",
        "discovery_scope": "days 0-352 x E_dev",
        "model_parameters": dict(make_d001_classifier().get_params()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "runtime_seconds": time.perf_counter() - started,
        "internal_confirmation_accessed": False,
        "e_holdout_accessed": False,
        "competition_test_accessed": False,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest).sort_values(["fold", "subset"]).to_csv(RESULTS_DIR / "D001_fold_manifest.csv", index=False)
    metrics_frame.to_csv(RESULTS_DIR / "D001_metrics.csv", index=False)
    summary.to_csv(RESULTS_DIR / "D001_summary.csv", index=False)
    deltas.to_csv(RESULTS_DIR / "D001_incremental_auc.csv", index=False)
    with (RESULTS_DIR / "D001_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=str)
    print(metrics_frame.to_string(index=False))


if __name__ == "__main__":
    main()
