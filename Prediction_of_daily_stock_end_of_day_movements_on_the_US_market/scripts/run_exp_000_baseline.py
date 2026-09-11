"""Run the majority-class baseline specified by EXP_000."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.exp000_validation import (  # noqa: E402
    CLASS_ORDER,
    FOLDS,
    N_HOLDOUT_EQUITIES,
    PARTITION_SEED,
    build_equity_partition,
    class_counts,
    evaluate_constant_baseline,
    fold_manifest_rows,
    majority_class,
    split_fold,
    validate_equity_partition,
)
from src.preprocessing import get_return_columns, load_training_data  # noqa: E402


DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = EXPERIMENT_DIR / "results"
PARTITION_PATH = EXPERIMENT_DIR / "EXP_000_equity_partition.csv"


def load_or_create_partition(features: pd.DataFrame) -> pd.DataFrame:
    """Persist the single frozen entity split mandated by EXP_000."""
    expected_equities = features["equity"].unique()
    if PARTITION_PATH.exists():
        partition = pd.read_csv(PARTITION_PATH)
        validate_equity_partition(partition, expected_equities, n_holdout=N_HOLDOUT_EQUITIES)
        return partition

    partition = build_equity_partition(
        expected_equities,
        seed=PARTITION_SEED,
        n_holdout=N_HOLDOUT_EQUITIES,
    )
    partition.to_csv(PARTITION_PATH, index=False)
    return partition


def main() -> None:
    features, target = load_training_data(DATA_DIR)
    return_columns = get_return_columns(features.columns)
    if return_columns != [f"r{index}" for index in range(53)]:
        raise AssertionError("EXP_000 requires exactly r0 through r52 as initial predictive inputs.")
    if set(features["day"].unique()) != set(range(503)):
        raise AssertionError("Training day identifiers do not match the EXP_000 contract.")
    if features["equity"].nunique() != 1829:
        raise AssertionError("Training equity count does not match the EXP_000 contract.")

    partition = load_or_create_partition(features)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_records: list[dict[str, int | str]] = []
    metric_records: list[dict[str, int | float | str]] = []
    confusion_records: dict[str, object] = {
        "labels": list(CLASS_ORDER),
        "matrices": {},
    }

    for fold in FOLDS:
        subsets = split_fold(features, target, partition, fold)
        manifest_records.extend(fold_manifest_rows(subsets))
        predicted_class = majority_class(subsets.fit_target)

        for subset_name, subset_target in (
            ("temporal_oos", subsets.temporal_target),
            ("joint_oos", subsets.joint_target),
        ):
            metrics, matrix = evaluate_constant_baseline(subset_target, predicted_class)
            record: dict[str, int | float | str] = {
                "experiment": "EXP_000",
                "fold": fold.fold,
                "subset": subset_name,
                "n_rows": int(len(subset_target)),
                "fit_majority_class": predicted_class,
            }
            record.update(class_counts(subset_target))
            record.update(metrics)
            metric_records.append(record)
            confusion_records["matrices"][f"fold_{fold.fold}_{subset_name}"] = matrix

    manifest = pd.DataFrame(manifest_records).sort_values(["fold", "subset"])
    metrics = pd.DataFrame(metric_records).sort_values(["subset", "fold"])
    summary = (
        metrics.groupby("subset")[["accuracy", "macro_f1", "balanced_accuracy"]]
        .agg(["mean", lambda values: values.std(ddof=1)])
        .rename(columns={"<lambda_0>": "std_ddof_1"})
    )
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    summary = summary.reset_index()

    manifest.to_csv(RESULTS_DIR / "EXP_000_fold_manifest.csv", index=False)
    metrics.to_csv(RESULTS_DIR / "EXP_000_baseline_metrics.csv", index=False)
    summary.to_csv(RESULTS_DIR / "EXP_000_baseline_summary.csv", index=False)
    with (RESULTS_DIR / "EXP_000_baseline_confusion_matrices.json").open("w", encoding="utf-8") as file:
        json.dump(confusion_records, file, indent=2)

    print("EXP_000 completed using training data only.")
    print("\nFold manifest:")
    print(manifest.to_string(index=False))
    print("\nOOS metrics:")
    print(metrics.to_string(index=False))
    print("\nFold summary (standard deviation uses ddof=1):")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
