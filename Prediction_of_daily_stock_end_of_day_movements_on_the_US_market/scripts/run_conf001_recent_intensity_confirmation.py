"""Execute frozen CONF_001 only after separate execution authorization."""

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

from src.conf001_recent_intensity_confirmation import (  # noqa: E402
    CONFIRMATION_BLOCKS,
    assert_confirmation_row_identity,
    confirmation_primary_decision,
    fit_confirmation_transformer,
    prepare_confirmation_matrices,
)
from src.exp000_validation import N_HOLDOUT_EQUITIES, split_fold, validate_equity_partition  # noqa: E402
from src.exp001_missingness import RETURN_COLUMNS  # noqa: E402
from src.exp002_neutral_directional import (  # noqa: E402
    assert_both_binary_classes,
    binary_class_counts,
    binary_majority_class,
    build_binary_target,
    delta_vs_baseline,
    directional_probability,
    evaluate_binary_majority_baseline,
    evaluate_binary_predictions,
    make_binary_logistic_regression,
    threshold_predictions,
)
from src.preprocessing import get_return_columns, load_training_data  # noqa: E402


DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
PARTITION_PATH = PROJECT_ROOT / "experiments" / "EXP_000_equity_partition.csv"


def validate_training_contract(features: pd.DataFrame) -> None:
    """Validate the immutable training schema required by CONF_001."""
    if tuple(get_return_columns(features.columns)) != RETURN_COLUMNS:
        raise AssertionError("CONF_001 requires exactly r0 through r52.")
    if set(features["day"].unique()) != set(range(503)):
        raise AssertionError("CONF_001 requires frozen day identifiers 0 through 502.")
    if features["equity"].nunique() != 1829:
        raise AssertionError("CONF_001 requires the frozen 1,829-equity training universe.")


def manifest_row(block: int, subset: str, target: pd.Series, structure: pd.DataFrame) -> dict[str, int | str]:
    """Record population, class, and target-free availability counts."""
    row: dict[str, int | str] = {"block": block, "subset": subset, "n_rows": int(len(target))}
    row.update(binary_class_counts(target))
    row["n_q_one"] = int(structure["q"].sum())
    row["n_q_zero"] = int((structure["q"] == 0).sum())
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
    confusion_records: dict[str, object] = {"labels": [0, 1], "matrices": {}}

    for block in CONFIRMATION_BLOCKS:
        subsets = split_fold(features, target, partition, block)
        named_subsets = {
            "fit": (subsets.fit_features, subsets.fit_target),
            "temporal_confirmation": (subsets.temporal_features, subsets.temporal_target),
            "joint_confirmation": (subsets.joint_features, subsets.joint_target),
        }
        fit_features, fit_target = named_subsets["fit"]
        assert_both_binary_classes(fit_target, context=f"CONF_001 fit block {block.fold}")
        fit_structure, transformer = fit_confirmation_transformer(fit_features)
        parameter_records.append({
            "block": block.fold,
            "fit_intensity_median": transformer.fit_intensity_median_,
            "scaler_mean": float(transformer.scaler_.mean_[0]),
            "scaler_scale": float(transformer.scaler_.scale_[0]),
            "n_fit_defined_intensity": int((fit_structure["q"] == 0).sum()),
            "n_fit_undefined_intensity": int((fit_structure["q"] == 1).sum()),
        })

        matrices: dict[str, dict[str, pd.DataFrame]] = {}
        structures: dict[str, pd.DataFrame] = {}
        for subset_name, (subset_features, subset_target) in named_subsets.items():
            structure, representations = prepare_confirmation_matrices(
                subset_features, subset_target, transformer
            )
            structures[subset_name] = structure
            matrices[subset_name] = representations
            assert_confirmation_row_identity(representations, subset_features, subset_target)
            manifest_records.append(manifest_row(block.fold, subset_name, subset_target, structure))

        baseline_class = binary_majority_class(fit_target)
        models = {}
        for condition in ("C", "B"):
            classifier = make_binary_logistic_regression()
            classifier.fit(matrices["fit"][condition], fit_target)
            if tuple(classifier.classes_) != (0, 1):
                raise AssertionError("CONF_001 classifier must be fitted on both binary classes.")
            models[condition] = classifier

        for subset_name in ("temporal_confirmation", "joint_confirmation"):
            subset_features, subset_target = named_subsets[subset_name]
            assert_both_binary_classes(subset_target, context=f"CONF_001 {subset_name} block {block.fold}")
            baseline_metrics, baseline_confusion = evaluate_binary_majority_baseline(subset_target, baseline_class)
            confusion_records["matrices"][f"baseline_block_{block.fold}_{subset_name}"] = baseline_confusion
            condition_metrics: dict[str, dict[str, float]] = {}
            for condition in ("C", "B"):
                probability = directional_probability(models[condition], matrices[subset_name][condition])
                prediction = threshold_predictions(probability)
                metrics, matrix = evaluate_binary_predictions(
                    subset_target, prediction, directional_probability_values=probability
                )
                condition_metrics[condition] = metrics
                record: dict[str, int | float | str] = {
                    "confirmation": "CONF_001", "condition": condition,
                    "block": block.fold, "subset": subset_name, "n_rows": int(len(subset_target)),
                    "baseline_class_from_fit": baseline_class,
                }
                record.update(binary_class_counts(subset_target))
                record.update(metrics)
                for name, value in baseline_metrics.items():
                    record[f"baseline_{name}"] = value
                record["accuracy_delta_vs_baseline"] = delta_vs_baseline(metrics["accuracy"], baseline_metrics["accuracy"])
                record["balanced_accuracy_delta_vs_baseline"] = delta_vs_baseline(
                    metrics["balanced_accuracy"], baseline_metrics["balanced_accuracy"]
                )
                metric_records.append(record)
                confusion_records["matrices"][f"{condition}_block_{block.fold}_{subset_name}"] = matrix
            delta_records.append({
                "block": block.fold, "subset": subset_name,
                "auc_c": condition_metrics["C"]["roc_auc"],
                "auc_b": condition_metrics["B"]["roc_auc"],
                "delta_auc_b_minus_c": delta_vs_baseline(
                    condition_metrics["B"]["roc_auc"], condition_metrics["C"]["roc_auc"]
                ),
            })

    metrics_frame = pd.DataFrame(metric_records).sort_values(["condition", "subset", "block"])
    manifest_frame = pd.DataFrame(manifest_records).sort_values(["block", "subset"])
    parameter_frame = pd.DataFrame(parameter_records).sort_values("block")
    delta_frame = pd.DataFrame(delta_records).sort_values(["subset", "block"])
    joint_deltas = delta_frame.loc[delta_frame["subset"].eq("joint_confirmation"), "delta_auc_b_minus_c"].tolist()
    decision = confirmation_primary_decision(joint_deltas)
    decision_frame = pd.DataFrame([{**decision, "record_type": "aggregate_primary"}])
    metrics = ["roc_auc", "balanced_accuracy", "accuracy", "recall_neutral", "recall_directional"]
    summary = metrics_frame.groupby(["condition", "subset"])[metrics].agg(["mean", lambda x: x.std(ddof=1)])
    summary.columns = [f"{name}_{stat}".replace("<lambda_0>", "std_ddof_1") for name, stat in summary.columns]
    summary = summary.reset_index()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_frame.to_csv(RESULTS_DIR / "CONF_001_fold_manifest.csv", index=False)
    parameter_frame.to_csv(RESULTS_DIR / "CONF_001_preprocessing_parameters.csv", index=False)
    metrics_frame.to_csv(RESULTS_DIR / "CONF_001_metrics.csv", index=False)
    delta_frame.to_csv(RESULTS_DIR / "CONF_001_incremental_auc.csv", index=False)
    summary.to_csv(RESULTS_DIR / "CONF_001_summary.csv", index=False)
    decision_frame.to_csv(RESULTS_DIR / "CONF_001_joint_decision.csv", index=False)
    with (RESULTS_DIR / "CONF_001_confusion_matrices.json").open("w", encoding="utf-8") as file:
        json.dump(confusion_records, file, indent=2)
    metadata = {
        "confirmation": "CONF_001", "return_window": [f"r{i}" for i in range(41, 53)],
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "training_data_only": True,
    }
    with (RESULTS_DIR / "CONF_001_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print("CONF_001 completed using training data only; competition test was not accessed.")


if __name__ == "__main__":
    main()
