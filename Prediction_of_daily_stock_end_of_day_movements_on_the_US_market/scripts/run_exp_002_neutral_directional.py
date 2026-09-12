"""Execute the frozen EXP_002 neutral-versus-directional missingness experiment."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import sklearn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.exp000_validation import FOLDS, N_HOLDOUT_EQUITIES, split_fold, validate_equity_partition  # noqa: E402
from src.exp001_missingness import RETURN_COLUMNS  # noqa: E402
from src.exp002_neutral_directional import (  # noqa: E402
    STRUCTURAL_SESSION_DAYS,
    aggregate_primary_decision,
    aggregate_secondary_decision,
    assert_both_binary_classes,
    assert_subset_matches_frozen_split,
    binary_class_counts,
    binary_majority_class,
    build_binary_target,
    build_missing_ratio,
    delta_vs_baseline,
    directional_probability,
    evaluate_binary_majority_baseline,
    evaluate_binary_predictions,
    make_binary_logistic_regression,
    primary_fold_decision,
    secondary_fold_decision,
    threshold_predictions,
    validate_missing_ratio,
)
from src.preprocessing import get_return_columns, load_training_data  # noqa: E402


DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = EXPERIMENT_DIR / "results"
PARTITION_PATH = EXPERIMENT_DIR / "EXP_000_equity_partition.csv"


def validate_training_contract(features: pd.DataFrame) -> None:
    if tuple(get_return_columns(features.columns)) != RETURN_COLUMNS:
        raise AssertionError("EXP_002 requires exactly r0 through r52.")
    if set(features["day"].unique()) != set(range(503)):
        raise AssertionError("Training day identifiers do not match the frozen protocol.")
    if features["equity"].nunique() != 1829:
        raise AssertionError("Training equity count does not match the frozen protocol.")


def binary_manifest_rows(subsets) -> list[dict[str, int | str]]:
    records = []
    for subset_name, features, target in (
        ("fit", subsets.fit_features, subsets.fit_target),
        ("temporal_oos", subsets.temporal_features, subsets.temporal_target),
        ("joint_oos", subsets.joint_features, subsets.joint_target),
    ):
        record: dict[str, int | str] = {"fold": subsets.fold.fold, "subset": subset_name, "n_rows": len(target)}
        record.update(binary_class_counts(target))
        record["n_structural_session_rows"] = int(features["day"].isin(STRUCTURAL_SESSION_DAYS).sum())
        records.append(record)
    return records


def main() -> None:
    features, reod = load_training_data(DATA_DIR)
    validate_training_contract(features)
    target = build_binary_target(reod)
    if not PARTITION_PATH.exists():
        raise FileNotFoundError(f"Frozen EXP_000 partition is missing: {PARTITION_PATH}")
    partition = pd.read_csv(PARTITION_PATH)
    validate_equity_partition(partition, features["equity"].unique(), n_holdout=N_HOLDOUT_EQUITIES)

    manifest_records: list[dict[str, int | str]] = []
    metric_records: list[dict[str, int | float | str]] = []
    fold_decisions: list[dict[str, int | float | str | bool]] = []
    confusion_records: dict[str, object] = {"labels": [0, 1], "matrices": {}}

    for fold in FOLDS:
        subsets = split_fold(features, target, partition, fold)
        manifest_records.extend(binary_manifest_rows(subsets))
        named_subsets = {
            "fit": (subsets.fit_features, subsets.fit_target),
            "temporal_oos": (subsets.temporal_features, subsets.temporal_target),
            "joint_oos": (subsets.joint_features, subsets.joint_target),
        }
        expected_features = {
            "temporal_oos": subsets.temporal_features,
            "joint_oos": subsets.joint_features,
        }

        fit_features, fit_target = named_subsets["fit"]
        assert_both_binary_classes(fit_target, context=f"fit fold {fold.fold}")
        fit_matrix = build_missing_ratio(fit_features)
        validate_missing_ratio(fit_matrix, expected_index=fit_features.index)
        classifier = make_binary_logistic_regression()
        classifier.fit(fit_matrix, fit_target)
        if tuple(classifier.classes_) != (0, 1):
            raise AssertionError("EXP_002 classifier was not fitted on both binary classes.")
        baseline_class = binary_majority_class(fit_target)

        for subset_name in ("temporal_oos", "joint_oos"):
            subset_features, subset_target = named_subsets[subset_name]
            assert_subset_matches_frozen_split(subset_features, expected_features[subset_name], subset_name=subset_name)
            assert_both_binary_classes(subset_target, context=f"{subset_name} fold {fold.fold}")
            matrix = build_missing_ratio(subset_features)
            validate_missing_ratio(matrix, expected_index=subset_features.index)
            probability = directional_probability(classifier, matrix)
            prediction = threshold_predictions(probability)
            model_metrics, model_confusion = evaluate_binary_predictions(
                subset_target, prediction, directional_probability_values=probability
            )
            baseline_metrics, baseline_confusion = evaluate_binary_majority_baseline(subset_target, baseline_class)
            record: dict[str, int | float | str] = {
                "experiment": "EXP_002",
                "fold": fold.fold,
                "subset": subset_name,
                "n_rows": len(subset_target),
                "baseline_class_from_fit": baseline_class,
            }
            record.update(binary_class_counts(subset_target))
            record.update(model_metrics)
            for metric_name, value in baseline_metrics.items():
                record[f"baseline_{metric_name}"] = value
            record["accuracy_delta_vs_baseline"] = delta_vs_baseline(model_metrics["accuracy"], baseline_metrics["accuracy"])
            record["balanced_accuracy_delta_vs_baseline"] = delta_vs_baseline(
                model_metrics["balanced_accuracy"], baseline_metrics["balanced_accuracy"]
            )
            metric_records.append(record)
            confusion_records["matrices"][f"model_fold_{fold.fold}_{subset_name}"] = model_confusion
            confusion_records["matrices"][f"baseline_fold_{fold.fold}_{subset_name}"] = baseline_confusion

            if subset_name == "joint_oos":
                primary = primary_fold_decision(model_metrics["roc_auc"])
                secondary = secondary_fold_decision(model_metrics)
                fold_decisions.append(
                    {
                        "record_type": "fold",
                        "fold": fold.fold,
                        "roc_auc": model_metrics["roc_auc"],
                        "balanced_accuracy": model_metrics["balanced_accuracy"],
                        "recall_neutral": model_metrics["recall_neutral"],
                        "recall_directional": model_metrics["recall_directional"],
                        "balanced_accuracy_delta_vs_baseline": record["balanced_accuracy_delta_vs_baseline"],
                        **primary,
                        **secondary,
                    }
                )

    metrics_frame = pd.DataFrame(metric_records).sort_values(["subset", "fold"])
    manifest_frame = pd.DataFrame(manifest_records).sort_values(["fold", "subset"])
    decisions_frame = pd.DataFrame(fold_decisions).sort_values("fold")
    primary_aggregate = aggregate_primary_decision(decisions_frame["roc_auc"].tolist())
    secondary_aggregate = aggregate_secondary_decision(
        decisions_frame.to_dict("records"), decisions_frame["balanced_accuracy_delta_vs_baseline"].tolist()
    )
    aggregate_rows = [
        {"record_type": "aggregate_primary", "fold": "all", **primary_aggregate},
        {"record_type": "aggregate_secondary", "fold": "all", **secondary_aggregate},
    ]
    decision_output = pd.concat([decisions_frame, pd.DataFrame(aggregate_rows)], ignore_index=True, sort=False)

    summary_columns = [
        "accuracy", "balanced_accuracy", "recall_neutral", "recall_directional", "roc_auc",
        "accuracy_delta_vs_baseline", "balanced_accuracy_delta_vs_baseline",
    ]
    summary = metrics_frame.groupby("subset")[summary_columns].agg(["mean", lambda values: values.std(ddof=1)])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    summary = summary.rename(columns=lambda name: name.replace("<lambda_0>", "std_ddof_1")).reset_index()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_frame.to_csv(RESULTS_DIR / "EXP_002_fold_manifest.csv", index=False)
    metrics_frame.to_csv(RESULTS_DIR / "EXP_002_metrics.csv", index=False)
    summary.to_csv(RESULTS_DIR / "EXP_002_summary.csv", index=False)
    decision_output.to_csv(RESULTS_DIR / "EXP_002_joint_decision.csv", index=False)
    with (RESULTS_DIR / "EXP_002_confusion_matrices.json").open("w", encoding="utf-8") as file:
        json.dump(confusion_records, file, indent=2)

    print("EXP_002 completed using training data only; competition test was not accessed.")
    print(f"scikit-learn version: {sklearn.__version__}")
    print("\nFold manifest:")
    print(manifest_frame.to_string(index=False))
    print("\nMetrics:")
    print(metrics_frame.to_string(index=False))
    print("\nJoint OOS decisions:")
    print(decision_output.to_string(index=False))


if __name__ == "__main__":
    main()
