"""Execute frozen EXP_009 using training data only."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import sklearn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.exp000_validation import FOLDS, N_HOLDOUT_EQUITIES, split_fold, validate_equity_partition  # noqa: E402
from src.exp001_missingness import RETURN_COLUMNS  # noqa: E402
from src.exp002_neutral_directional import (  # noqa: E402
    aggregate_secondary_decision,
    assert_both_binary_classes,
    binary_class_counts,
    binary_majority_class,
    build_binary_target,
    build_missing_ratio,
    delta_vs_baseline,
    directional_probability,
    evaluate_binary_majority_baseline,
    evaluate_binary_predictions,
    make_binary_logistic_regression,
    secondary_fold_decision,
    threshold_predictions,
    validate_missing_ratio,
)
from src.exp009_recency_weighted_movement_intensity import (  # noqa: E402
    RECENCY_WEIGHTS,
    RECENCY_WEIGHT_SUM,
    RECENT_WINDOW_COLUMNS,
    CompleteWindowIntensityTransformer,
    aggregate_primary_decision,
    assert_identical_representation_rows,
    auc_delta,
    build_complete_window_structure,
    build_exp009_representations,
    build_raw_intensities,
    filter_complete_window_rows,
    primary_fold_decision,
)
from src.preprocessing import get_return_columns, load_training_data  # noqa: E402


DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = EXPERIMENT_DIR / "results"
PARTITION_PATH = EXPERIMENT_DIR / "EXP_000_equity_partition.csv"


def validate_training_contract(features: pd.DataFrame) -> None:
    if tuple(get_return_columns(features.columns)) != RETURN_COLUMNS:
        raise AssertionError("EXP_009 requires exactly r0 through r52.")
    if set(features["day"].unique()) != set(range(503)):
        raise AssertionError("Training day identifiers do not match the frozen protocol.")
    if features["equity"].nunique() != 1829:
        raise AssertionError("Training equity count does not match the frozen protocol.")


def _retain_complete_window(
    features: pd.DataFrame, target: pd.Series
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    structure = build_complete_window_structure(features)
    retained_features, retained_target = filter_complete_window_rows(features, target, structure)
    expected_ids = features.loc[structure["complete_w"], "ID"]
    if not retained_features["ID"].equals(expected_ids):
        raise AssertionError("EXP_009 complete-W filtering changed expected source row ordering.")
    return retained_features, retained_target, structure


def manifest_rows(
    fold: int,
    subset: str,
    before_target: pd.Series,
    structure: pd.DataFrame,
    retained_target: pd.Series,
) -> dict[str, int | str]:
    row: dict[str, int | str] = {"fold": fold, "subset": subset, "n_rows_before_complete_w": len(before_target)}
    row.update({f"before_{key}": value for key, value in binary_class_counts(before_target).items()})
    row["n_complete_w"] = int(structure["complete_w"].sum())
    row["n_incomplete_w"] = int((~structure["complete_w"]).sum())
    row["n_rows_retained"] = len(retained_target)
    row.update(binary_class_counts(retained_target))
    return row


def main() -> None:
    started = time.perf_counter()
    features, reod = load_training_data(DATA_DIR)
    validate_training_contract(features)
    target = build_binary_target(reod)
    if not PARTITION_PATH.exists():
        raise FileNotFoundError(f"Frozen EXP_000 partition is missing: {PARTITION_PATH}")
    partition = pd.read_csv(PARTITION_PATH)
    validate_equity_partition(partition, features["equity"].unique(), n_holdout=N_HOLDOUT_EQUITIES)

    manifest_records: list[dict[str, int | str]] = []
    parameter_records: list[dict[str, int | float]] = []
    metric_records: list[dict[str, int | float | str]] = []
    delta_records: list[dict[str, int | float | str]] = []
    joint_decisions: list[dict[str, int | float | bool | str]] = []
    confusion_records: dict[str, object] = {"labels": [0, 1], "matrices": {}}

    for fold in FOLDS:
        subsets = split_fold(features, target, partition, fold)
        raw_subsets = {
            "fit": (subsets.fit_features, subsets.fit_target),
            "temporal_oos": (subsets.temporal_features, subsets.temporal_target),
            "joint_oos": (subsets.joint_features, subsets.joint_target),
        }
        retained: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
        for name, (subset_features, subset_target) in raw_subsets.items():
            retained_features, retained_target, structure = _retain_complete_window(subset_features, subset_target)
            manifest_records.append(manifest_rows(fold.fold, name, subset_target, structure, retained_target))
            retained[name] = (retained_features, retained_target)
            assert_both_binary_classes(retained_target, context=f"retained {name} fold {fold.fold}")

        fit_features, fit_target = retained["fit"]
        fit_raw = build_raw_intensities(fit_features)
        transformer = CompleteWindowIntensityTransformer().fit(fit_raw)
        parameter_records.append(
            {
                "fold": fold.fold,
                "scaler_recent_mean": float(transformer.scaler_recent_.mean_[0]),
                "scaler_recent_scale": float(transformer.scaler_recent_.scale_[0]),
                "scaler_recency_mean": float(transformer.scaler_recency_.mean_[0]),
                "scaler_recency_scale": float(transformer.scaler_recency_.scale_[0]),
                "n_fit_complete_w": len(fit_target),
            }
        )
        baseline_class = binary_majority_class(fit_target)
        matrices: dict[str, dict[str, pd.DataFrame]] = {}
        for name, (subset_features, subset_target) in retained.items():
            ratio = build_missing_ratio(subset_features)
            validate_missing_ratio(ratio, expected_index=subset_features.index)
            raw = build_raw_intensities(subset_features)
            transformed = transformer.transform(raw)
            matrices[name] = build_exp009_representations(ratio, transformed)
            assert_identical_representation_rows(matrices[name], subset_features, subset_target)

        fitted_models = {}
        for condition, matrix in matrices["fit"].items():
            classifier = make_binary_logistic_regression()
            classifier.fit(matrix, fit_target)
            if tuple(classifier.classes_) != (0, 1):
                raise AssertionError("EXP_009 classifier was not fitted on both binary classes.")
            fitted_models[condition] = classifier

        for subset_name in ("temporal_oos", "joint_oos"):
            subset_features, subset_target = retained[subset_name]
            baseline_metrics, baseline_confusion = evaluate_binary_majority_baseline(subset_target, baseline_class)
            confusion_records["matrices"][f"baseline_fold_{fold.fold}_{subset_name}"] = baseline_confusion
            condition_metrics: dict[str, dict[str, float]] = {}
            for condition in ("C", "D"):
                probability = directional_probability(fitted_models[condition], matrices[subset_name][condition])
                prediction = threshold_predictions(probability)
                model_metrics, model_confusion = evaluate_binary_predictions(
                    subset_target, prediction, directional_probability_values=probability
                )
                condition_metrics[condition] = model_metrics
                record: dict[str, int | float | str] = {
                    "experiment": "EXP_009",
                    "condition": condition,
                    "fold": fold.fold,
                    "subset": subset_name,
                    "n_rows": len(subset_target),
                    "baseline_class_from_retained_fit": baseline_class,
                }
                record.update(binary_class_counts(subset_target))
                record.update(model_metrics)
                for metric_name, value in baseline_metrics.items():
                    record[f"baseline_{metric_name}"] = value
                record["accuracy_delta_vs_baseline"] = delta_vs_baseline(
                    model_metrics["accuracy"], baseline_metrics["accuracy"]
                )
                record["balanced_accuracy_delta_vs_baseline"] = delta_vs_baseline(
                    model_metrics["balanced_accuracy"], baseline_metrics["balanced_accuracy"]
                )
                metric_records.append(record)
                confusion_records["matrices"][f"{condition}_fold_{fold.fold}_{subset_name}"] = model_confusion

            recency_delta = auc_delta(condition_metrics["D"]["roc_auc"], condition_metrics["C"]["roc_auc"])
            delta_records.append(
                {
                    "fold": fold.fold,
                    "subset": subset_name,
                    "auc_c": condition_metrics["C"]["roc_auc"],
                    "auc_d": condition_metrics["D"]["roc_auc"],
                    "delta_recency_auc_d_minus_c": recency_delta,
                }
            )
            if subset_name == "joint_oos":
                d_record = next(
                    record for record in reversed(metric_records)
                    if record["condition"] == "D" and record["fold"] == fold.fold and record["subset"] == subset_name
                )
                joint_decisions.append(
                    {
                        "record_type": "fold",
                        "fold": fold.fold,
                        "d_roc_auc": condition_metrics["D"]["roc_auc"],
                        "recency_auc_delta_d_minus_c": recency_delta,
                        "balanced_accuracy_d": condition_metrics["D"]["balanced_accuracy"],
                        "recall_neutral_d": condition_metrics["D"]["recall_neutral"],
                        "recall_directional_d": condition_metrics["D"]["recall_directional"],
                        "balanced_accuracy_delta_d_vs_baseline": d_record["balanced_accuracy_delta_vs_baseline"],
                        **primary_fold_decision(condition_metrics["D"]["roc_auc"], recency_delta),
                        **secondary_fold_decision(condition_metrics["D"]),
                    }
                )

    metrics_frame = pd.DataFrame(metric_records).sort_values(["condition", "subset", "fold"])
    manifest_frame = pd.DataFrame(manifest_records).sort_values(["fold", "subset"])
    parameters_frame = pd.DataFrame(parameter_records).sort_values("fold")
    deltas_frame = pd.DataFrame(delta_records).sort_values(["subset", "fold"])
    fold_decisions = pd.DataFrame(joint_decisions).sort_values("fold")
    primary_aggregate = aggregate_primary_decision(
        fold_decisions["d_roc_auc"].tolist(), fold_decisions["recency_auc_delta_d_minus_c"].tolist()
    )
    secondary_aggregate = aggregate_secondary_decision(
        fold_decisions.to_dict("records"), fold_decisions["balanced_accuracy_delta_d_vs_baseline"].tolist()
    )
    decision_frame = pd.concat(
        [
            fold_decisions,
            pd.DataFrame(
                [
                    {"record_type": "aggregate_primary", "fold": "all", **primary_aggregate},
                    {"record_type": "aggregate_secondary", "fold": "all", **secondary_aggregate},
                ]
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    summary_columns = [
        "accuracy", "balanced_accuracy", "recall_neutral", "recall_directional", "roc_auc",
        "accuracy_delta_vs_baseline", "balanced_accuracy_delta_vs_baseline",
    ]
    summary = metrics_frame.groupby(["condition", "subset"])[summary_columns].agg(["mean", lambda values: values.std(ddof=1)])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    summary = summary.rename(columns=lambda name: name.replace("<lambda_0>", "std_ddof_1")).reset_index()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_frame.to_csv(RESULTS_DIR / "EXP_009_fold_manifest.csv", index=False)
    parameters_frame.to_csv(RESULTS_DIR / "EXP_009_preprocessing_parameters.csv", index=False)
    metrics_frame.to_csv(RESULTS_DIR / "EXP_009_metrics.csv", index=False)
    deltas_frame.to_csv(RESULTS_DIR / "EXP_009_incremental_auc.csv", index=False)
    summary.to_csv(RESULTS_DIR / "EXP_009_summary.csv", index=False)
    decision_frame.to_csv(RESULTS_DIR / "EXP_009_joint_decision.csv", index=False)
    with (RESULTS_DIR / "EXP_009_confusion_matrices.json").open("w", encoding="utf-8") as file:
        json.dump(confusion_records, file, indent=2)
    metadata = {
        "experiment": "EXP_009",
        "return_window": list(RECENT_WINDOW_COLUMNS),
        "recency_weights": RECENCY_WEIGHTS.tolist(),
        "recency_weight_sum": RECENCY_WEIGHT_SUM,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "runtime_seconds": time.perf_counter() - started,
        "training_data_only": True,
    }
    with (RESULTS_DIR / "EXP_009_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print("EXP_009 completed using training data only; competition test was not accessed.")


if __name__ == "__main__":
    main()
