"""Execute the pre-specified EXP_001 missingness-only experiment."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import sklearn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.exp000_validation import (  # noqa: E402
    CLASS_ORDER,
    FOLDS,
    N_HOLDOUT_EQUITIES,
    class_counts,
    fold_manifest_rows,
    split_fold,
    validate_equity_partition,
)
from src.exp001_missingness import (  # noqa: E402
    LOGISTIC_REGRESSION_PARAMS,
    RETURN_COLUMNS,
    VARIANT_COLUMNS,
    build_missingness_representations,
    delta_vs_baseline,
    evaluate_predictions,
    joint_decision,
    make_logistic_regression,
    strictly_greater,
    validate_representation,
)
from src.preprocessing import get_return_columns, load_training_data  # noqa: E402


DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = EXPERIMENT_DIR / "results"
PARTITION_PATH = EXPERIMENT_DIR / "EXP_000_equity_partition.csv"
BASELINE_METRICS_PATH = RESULTS_DIR / "EXP_000_baseline_metrics.csv"


def load_exp000_baseline() -> pd.DataFrame:
    """Load and validate the executed EXP_000 fold-level comparison artifact."""
    if not BASELINE_METRICS_PATH.exists():
        raise FileNotFoundError(f"Required EXP_000 artifact is missing: {BASELINE_METRICS_PATH}")
    baseline = pd.read_csv(BASELINE_METRICS_PATH)
    required = {"experiment", "fold", "subset", "n_rows", "accuracy", "macro_f1", "balanced_accuracy"}
    if not required.issubset(baseline.columns):
        raise AssertionError("EXP_000 baseline artifact lacks required comparison columns.")
    if set(baseline["experiment"]) != {"EXP_000"}:
        raise AssertionError("Baseline comparison must use only EXP_000 results.")
    expected_keys = {(fold.fold, subset) for fold in FOLDS for subset in ("temporal_oos", "joint_oos")}
    observed_keys = set(zip(baseline["fold"], baseline["subset"], strict=True))
    if observed_keys != expected_keys or baseline.duplicated(["fold", "subset"]).any():
        raise AssertionError("EXP_000 baseline artifact does not have exactly one row per fold/subset.")
    return baseline.set_index(["fold", "subset"]).sort_index()


def validate_training_contract(features: pd.DataFrame) -> None:
    """Check the fixed training-universe facts needed by EXP_001."""
    if tuple(get_return_columns(features.columns)) != RETURN_COLUMNS:
        raise AssertionError("EXP_001 requires exactly r0 through r52.")
    if set(features["day"].unique()) != set(range(503)):
        raise AssertionError("Training day identifiers do not match the frozen protocol.")
    if features["equity"].nunique() != 1829:
        raise AssertionError("Training equity count does not match the frozen protocol.")


def check_baseline_row(baseline_row: pd.Series, target: pd.Series, fold: int, subset: str) -> None:
    """Ensure the reused baseline describes exactly the evaluated OOS rows."""
    if int(baseline_row["n_rows"]) != len(target):
        raise AssertionError(f"EXP_000 baseline row count differs for fold {fold} / {subset}.")
    expected_counts = class_counts(target)
    for name, count in expected_counts.items():
        if int(baseline_row[name]) != count:
            raise AssertionError(f"EXP_000 baseline class counts differ for fold {fold} / {subset}.")


def main() -> None:
    features, target = load_training_data(DATA_DIR)
    validate_training_contract(features)
    if not PARTITION_PATH.exists():
        raise FileNotFoundError(f"Frozen EXP_000 partition is missing: {PARTITION_PATH}")
    partition = pd.read_csv(PARTITION_PATH)
    validate_equity_partition(partition, features["equity"].unique(), n_holdout=N_HOLDOUT_EQUITIES)
    baseline = load_exp000_baseline()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_records: list[dict[str, int | str]] = []
    metric_records: list[dict[str, int | float | str]] = []
    decision_records: list[dict[str, int | float | str | bool]] = []
    confusion_records: dict[str, object] = {"labels": list(CLASS_ORDER), "matrices": {}}

    for fold in FOLDS:
        subsets = split_fold(features, target, partition, fold)
        manifest_records.extend(fold_manifest_rows(subsets))
        named_subsets = {
            "fit": (subsets.fit_features, subsets.fit_target),
            "temporal_oos": (subsets.temporal_features, subsets.temporal_target),
            "joint_oos": (subsets.joint_features, subsets.joint_target),
        }
        representations = {
            subset_name: build_missingness_representations(subset_features)
            for subset_name, (subset_features, _) in named_subsets.items()
        }

        for variant, expected_columns in VARIANT_COLUMNS.items():
            fit_features, fit_target = named_subsets["fit"]
            fit_matrix = representations["fit"][variant]
            validate_representation(variant, fit_matrix, expected_index=fit_features.index)
            if list(fit_matrix.columns) != list(expected_columns) or not fit_matrix.index.equals(fit_target.index):
                raise AssertionError("EXP_001 fit matrix violates its feature or target boundary.")
            if set(fit_target.unique()) != set(CLASS_ORDER):
                raise AssertionError(f"Fit target lacks a required class in fold {fold.fold}.")

            classifier = make_logistic_regression()
            classifier.fit(fit_matrix, fit_target)
            if tuple(classifier.classes_) != CLASS_ORDER:
                raise AssertionError("Fixed classifier was not fitted on all three contract classes.")

            for subset_name in ("temporal_oos", "joint_oos"):
                subset_features, subset_target = named_subsets[subset_name]
                matrix = representations[subset_name][variant]
                validate_representation(variant, matrix, expected_index=subset_features.index)
                if not matrix.index.equals(subset_target.index):
                    raise AssertionError("OOS matrix and target have different row identities.")
                baseline_row = baseline.loc[(fold.fold, subset_name)]
                check_baseline_row(baseline_row, subset_target, fold.fold, subset_name)

                metrics, matrix_confusion = evaluate_predictions(subset_target, classifier.predict(matrix))
                record: dict[str, int | float | str] = {
                    "experiment": "EXP_001",
                    "variant": variant,
                    "fold": fold.fold,
                    "subset": subset_name,
                    "n_rows": int(len(subset_target)),
                }
                record.update(class_counts(subset_target))
                record.update(metrics)
                for metric_name in ("accuracy", "macro_f1", "balanced_accuracy"):
                    baseline_value = float(baseline_row[metric_name])
                    record[f"baseline_{metric_name}"] = baseline_value
                    record[f"{metric_name}_delta_vs_exp000"] = delta_vs_baseline(metrics[metric_name], baseline_value)
                metric_records.append(record)
                confusion_records["matrices"][f"{variant}_fold_{fold.fold}_{subset_name}"] = matrix_confusion

                if subset_name == "joint_oos":
                    fold_decision: dict[str, int | float | str | bool] = {
                        "record_type": "fold",
                        "variant": variant,
                        "fold": fold.fold,
                        "baseline_accuracy": float(baseline_row["accuracy"]),
                        "accuracy": metrics["accuracy"],
                        "accuracy_delta_vs_exp000": delta_vs_baseline(
                            metrics["accuracy"], float(baseline_row["accuracy"])
                        ),
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "recall_class_-1": metrics["recall_class_-1"],
                        "recall_class_1": metrics["recall_class_1"],
                    }
                    fold_decision.update(joint_decision(metrics, float(baseline_row["accuracy"])))
                    decision_records.append(fold_decision)

    metrics_frame = pd.DataFrame(metric_records).sort_values(["variant", "subset", "fold"])
    manifest_frame = pd.DataFrame(manifest_records).sort_values(["fold", "subset"])
    decision_frame = pd.DataFrame(decision_records).sort_values(["variant", "fold"])

    summary_metrics = [
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "recall_class_-1",
        "recall_class_0",
        "recall_class_1",
        "accuracy_delta_vs_exp000",
        "macro_f1_delta_vs_exp000",
        "balanced_accuracy_delta_vs_exp000",
    ]
    summary = metrics_frame.groupby(["variant", "subset"])[summary_metrics].agg(["mean", lambda values: values.std(ddof=1)])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    summary = summary.rename(columns=lambda name: name.replace("<lambda_0>", "std_ddof_1")).reset_index()

    aggregate_records: list[dict[str, int | float | str | bool]] = []
    for variant, variant_decisions in decision_frame.groupby("variant", sort=True):
        all_fold_conditions = bool(variant_decisions["all_fold_conditions_met"].all())
        mean_delta = float(variant_decisions["accuracy_delta_vs_exp000"].mean())
        aggregate_records.append(
            {
                "record_type": "aggregate",
                "variant": variant,
                "fold": "all",
                "n_joint_folds": int(len(variant_decisions)),
                "all_four_accuracy_above_baseline": bool(variant_decisions["accuracy_above_baseline"].all()),
                "all_four_balanced_accuracy_above_one_third": bool(variant_decisions["balanced_accuracy_above_one_third"].all()),
                "all_four_recall_negative_positive": bool(variant_decisions["recall_negative_positive"].all()),
                "all_four_recall_positive_positive": bool(variant_decisions["recall_positive_positive"].all()),
                "mean_joint_accuracy_delta_vs_exp000": mean_delta,
                "mean_joint_accuracy_delta_positive": strictly_greater(mean_delta, 0.0),
                "promising": all_fold_conditions and strictly_greater(mean_delta, 0.0),
            }
        )
    decision_output = pd.concat([decision_frame, pd.DataFrame(aggregate_records)], ignore_index=True, sort=False)

    manifest_frame.to_csv(RESULTS_DIR / "EXP_001_fold_manifest.csv", index=False)
    metrics_frame.to_csv(RESULTS_DIR / "EXP_001_metrics.csv", index=False)
    summary.to_csv(RESULTS_DIR / "EXP_001_summary.csv", index=False)
    decision_output.to_csv(RESULTS_DIR / "EXP_001_joint_decision.csv", index=False)
    with (RESULTS_DIR / "EXP_001_confusion_matrices.json").open("w", encoding="utf-8") as file:
        json.dump(confusion_records, file, indent=2)

    print("EXP_001 completed using training data only; competition test was not accessed.")
    print(f"scikit-learn version: {sklearn.__version__}")
    print("\nFold manifest:")
    print(manifest_frame.to_string(index=False))
    print("\nMetrics:")
    print(metrics_frame.to_string(index=False))
    print("\nJoint OOS decision criteria:")
    print(decision_output.to_string(index=False))


if __name__ == "__main__":
    main()
