"""Execute frozen EXP_008 using training data only."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import sklearn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.exp000_validation import FOLDS, N_HOLDOUT_EQUITIES, split_fold, validate_equity_partition  # noqa: E402
from src.exp001_missingness import RETURN_COLUMNS  # noqa: E402
from src.exp002_neutral_directional import (  # noqa: E402
    STRUCTURAL_SESSION_DAYS,
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
    secondary_fold_decision,
    threshold_predictions,
    validate_missing_ratio,
)
from src.exp008_recent_movement_intensity import (  # noqa: E402
    RECENT_WINDOW_COLUMNS,
    RecentIntensityTransformer,
    aggregate_primary_decision,
    assert_identical_representation_rows,
    auc_delta,
    build_exp008_representations,
    build_recent_window_structure,
    primary_fold_decision,
)
from src.preprocessing import get_return_columns, load_training_data  # noqa: E402


DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = EXPERIMENT_DIR / "results"
PARTITION_PATH = EXPERIMENT_DIR / "EXP_000_equity_partition.csv"


def validate_training_contract(features: pd.DataFrame) -> None:
    """Assert the training schema required by frozen EXP_000/EXP_002."""
    if tuple(get_return_columns(features.columns)) != RETURN_COLUMNS:
        raise AssertionError("EXP_008 requires exactly r0 through r52.")
    if set(features["day"].unique()) != set(range(503)):
        raise AssertionError("Training day identifiers do not match the frozen protocol.")
    if features["equity"].nunique() != 1829:
        raise AssertionError("Training equity count does not match the frozen protocol.")


def manifest_row(fold: int, subset: str, features: pd.DataFrame, target: pd.Series, structure: pd.DataFrame) -> dict[str, int | str]:
    """Record frozen subset size/classes and target-free W structure."""
    row: dict[str, int | str] = {"fold": fold, "subset": subset, "n_rows": len(target)}
    row.update(binary_class_counts(target))
    row["n_q_one"] = int(structure["q"].sum())
    row["n_q_zero"] = int((structure["q"] == 0).sum())
    row["n_structural_session_rows"] = int(features["day"].isin(STRUCTURAL_SESSION_DAYS).sum())
    return row


def main() -> None:
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
        named_subsets = {
            "fit": (subsets.fit_features, subsets.fit_target),
            "temporal_oos": (subsets.temporal_features, subsets.temporal_target),
            "joint_oos": (subsets.joint_features, subsets.joint_target),
        }
        structures = {name: build_recent_window_structure(subset_features) for name, (subset_features, _) in named_subsets.items()}
        for name, (subset_features, subset_target) in named_subsets.items():
            manifest_records.append(manifest_row(fold.fold, name, subset_features, subset_target, structures[name]))

        fit_features, fit_target = named_subsets["fit"]
        assert_both_binary_classes(fit_target, context=f"fit fold {fold.fold}")
        transformer = RecentIntensityTransformer().fit(structures["fit"]["raw_I_recent"])
        parameter_records.append(
            {
                "fold": fold.fold,
                "fit_intensity_median": transformer.fit_intensity_median_,
                "scaler_mean": float(transformer.scaler_.mean_[0]),
                "scaler_scale": float(transformer.scaler_.scale_[0]),
                "n_fit_defined_intensity": int((structures["fit"]["q"] == 0).sum()),
                "n_fit_undefined_intensity": int((structures["fit"]["q"] == 1).sum()),
            }
        )
        baseline_class = binary_majority_class(fit_target)
        matrices: dict[str, dict[str, pd.DataFrame]] = {}
        for subset_name, (subset_features, subset_target) in named_subsets.items():
            missing_ratio = build_missing_ratio(subset_features)
            validate_missing_ratio(missing_ratio, expected_index=subset_features.index)
            intensity_z = transformer.transform(structures[subset_name]["raw_I_recent"])
            matrices[subset_name] = build_exp008_representations(missing_ratio, structures[subset_name], intensity_z)
            assert_identical_representation_rows(matrices[subset_name], subset_features, subset_target)

        fitted_models = {}
        for condition, matrix in matrices["fit"].items():
            classifier = make_binary_logistic_regression()
            classifier.fit(matrix, fit_target)
            if tuple(classifier.classes_) != (0, 1):
                raise AssertionError("EXP_008 classifier was not fitted on both binary classes.")
            fitted_models[condition] = classifier

        for subset_name in ("temporal_oos", "joint_oos"):
            subset_features, subset_target = named_subsets[subset_name]
            assert_subset_matches_frozen_split(subset_features, getattr(subsets, f"{subset_name.split('_')[0]}_features"), subset_name=subset_name)
            assert_both_binary_classes(subset_target, context=f"{subset_name} fold {fold.fold}")
            baseline_metrics, baseline_confusion = evaluate_binary_majority_baseline(subset_target, baseline_class)
            confusion_records["matrices"][f"baseline_fold_{fold.fold}_{subset_name}"] = baseline_confusion
            condition_metrics: dict[str, dict[str, float]] = {}
            for condition in ("A", "C", "B"):
                probability = directional_probability(fitted_models[condition], matrices[subset_name][condition])
                prediction = threshold_predictions(probability)
                model_metrics, model_confusion = evaluate_binary_predictions(
                    subset_target, prediction, directional_probability_values=probability
                )
                condition_metrics[condition] = model_metrics
                record: dict[str, int | float | str] = {
                    "experiment": "EXP_008",
                    "condition": condition,
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
                confusion_records["matrices"][f"{condition}_fold_{fold.fold}_{subset_name}"] = model_confusion

            q_delta = auc_delta(condition_metrics["C"]["roc_auc"], condition_metrics["A"]["roc_auc"])
            intensity_delta = auc_delta(condition_metrics["B"]["roc_auc"], condition_metrics["C"]["roc_auc"])
            delta_records.append(
                {
                    "fold": fold.fold,
                    "subset": subset_name,
                    "auc_a": condition_metrics["A"]["roc_auc"],
                    "auc_c": condition_metrics["C"]["roc_auc"],
                    "auc_b": condition_metrics["B"]["roc_auc"],
                    "delta_q_auc_c_minus_a": q_delta,
                    "delta_intensity_auc_b_minus_c": intensity_delta,
                }
            )
            if subset_name == "joint_oos":
                b_record = next(record for record in reversed(metric_records) if record["condition"] == "B" and record["fold"] == fold.fold and record["subset"] == subset_name)
                primary = primary_fold_decision(condition_metrics["B"]["roc_auc"], intensity_delta)
                secondary = secondary_fold_decision(condition_metrics["B"])
                joint_decisions.append(
                    {
                        "record_type": "fold",
                        "fold": fold.fold,
                        "b_roc_auc": condition_metrics["B"]["roc_auc"],
                        "intensity_auc_delta_b_minus_c": intensity_delta,
                        "q_auc_delta_c_minus_a": q_delta,
                        "balanced_accuracy_b": condition_metrics["B"]["balanced_accuracy"],
                        "recall_neutral_b": condition_metrics["B"]["recall_neutral"],
                        "recall_directional_b": condition_metrics["B"]["recall_directional"],
                        "balanced_accuracy_delta_b_vs_baseline": b_record["balanced_accuracy_delta_vs_baseline"],
                        **primary,
                        **secondary,
                    }
                )

    metrics_frame = pd.DataFrame(metric_records).sort_values(["condition", "subset", "fold"])
    manifest_frame = pd.DataFrame(manifest_records).sort_values(["fold", "subset"])
    parameters_frame = pd.DataFrame(parameter_records).sort_values("fold")
    deltas_frame = pd.DataFrame(delta_records).sort_values(["subset", "fold"])
    fold_decisions = pd.DataFrame(joint_decisions).sort_values("fold")
    primary_aggregate = aggregate_primary_decision(
        fold_decisions["b_roc_auc"].tolist(), fold_decisions["intensity_auc_delta_b_minus_c"].tolist()
    )
    secondary_aggregate = aggregate_secondary_decision(
        fold_decisions.to_dict("records"), fold_decisions["balanced_accuracy_delta_b_vs_baseline"].tolist()
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
    manifest_frame.to_csv(RESULTS_DIR / "EXP_008_fold_manifest.csv", index=False)
    parameters_frame.to_csv(RESULTS_DIR / "EXP_008_preprocessing_parameters.csv", index=False)
    metrics_frame.to_csv(RESULTS_DIR / "EXP_008_metrics.csv", index=False)
    deltas_frame.to_csv(RESULTS_DIR / "EXP_008_incremental_auc.csv", index=False)
    summary.to_csv(RESULTS_DIR / "EXP_008_summary.csv", index=False)
    decision_frame.to_csv(RESULTS_DIR / "EXP_008_joint_decision.csv", index=False)
    with (RESULTS_DIR / "EXP_008_confusion_matrices.json").open("w", encoding="utf-8") as file:
        json.dump(confusion_records, file, indent=2)
    metadata = {
        "experiment": "EXP_008",
        "return_window": list(RECENT_WINDOW_COLUMNS),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "training_data_only": True,
    }
    with (RESULTS_DIR / "EXP_008_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print("EXP_008 completed using training data only; competition test was not accessed.")


if __name__ == "__main__":
    main()
