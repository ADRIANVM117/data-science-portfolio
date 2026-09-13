"""Execute frozen EXP_007 using training data only after explicit approval."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.exp000_validation import FOLDS, N_HOLDOUT_EQUITIES, split_fold, validate_equity_partition  # noqa: E402
from src.exp001_missingness import RETURN_COLUMNS  # noqa: E402
from src.exp003_conditional_directional_sign import (  # noqa: E402
    assert_both_sign_classes,
    evaluate_sign_majority_baseline,
    evaluate_sign_predictions,
    sign_majority_class,
    threshold_predictions,
)
from src.exp005_positional_path_signal import PositionalPathTransformer, prepare_evaluable_directional_rows  # noqa: E402
from src.exp007_sequential_chronology_signal import (  # noqa: E402
    BATCH_ORDER_SEED,
    BATCH_SIZE,
    MODEL_SEED,
    N_EPOCHS,
    N_TRAINABLE_PARAMETERS,
    PERMUTATION,
    aggregate_primary_decision,
    aggregate_secondary_decision,
    assert_frozen_permutation,
    assert_paired_rows,
    build_paired_models,
    build_permuted_sequences,
    build_real_sequences,
    configure_deterministic_cpu,
    permutation_diagnostics,
    predict_positive_probability,
    primary_fold_decision,
    secondary_fold_decision,
    train_paired_models,
)
from src.exp005_positional_path_signal import delta_vs_baseline  # noqa: E402
from src.preprocessing import get_return_columns, load_training_data  # noqa: E402


DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
PARTITION_PATH = PROJECT_ROOT / "experiments" / "EXP_000_equity_partition.csv"


def validate_training_contract(features: pd.DataFrame) -> None:
    if tuple(get_return_columns(features.columns)) != RETURN_COLUMNS:
        raise AssertionError("EXP_007 requires exactly r0 through r52.")
    if set(features["day"].unique()) != set(range(503)) or features["equity"].nunique() != 1829:
        raise AssertionError("Training universe does not match the frozen EXP_000 protocol.")


def _sequence_pair(transformer: PositionalPathTransformer, raw_returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    real, index = build_real_sequences(transformer, raw_returns)
    return real, build_permuted_sequences(real), index


def _evaluate_condition(
    *,
    model: torch.nn.Module,
    sequence: np.ndarray,
    target: pd.Series,
    baseline_metrics: dict[str, float],
) -> tuple[dict[str, float], list[list[int]]]:
    probability = predict_positive_probability(model, sequence)
    prediction = threshold_predictions(probability)
    metrics, matrix = evaluate_sign_predictions(target, prediction, probability=probability)
    metrics["accuracy_delta_vs_baseline"] = delta_vs_baseline(metrics["accuracy"], baseline_metrics["accuracy"])
    metrics["balanced_accuracy_delta_vs_baseline"] = delta_vs_baseline(
        metrics["balanced_accuracy"], baseline_metrics["balanced_accuracy"]
    )
    return metrics, matrix


def main() -> None:
    configure_deterministic_cpu()
    assert_frozen_permutation()
    started = time.perf_counter()

    # Only this labelled training-data loader is used by the runner.
    features, reod = load_training_data(DATA_DIR)
    validate_training_contract(features)
    partition = pd.read_csv(PARTITION_PATH)
    validate_equity_partition(partition, features["equity"].unique(), n_holdout=N_HOLDOUT_EQUITIES)

    manifests: list[dict[str, object]] = []
    preprocessing_rows: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []
    metrics_rows: list[dict[str, object]] = []
    chronology_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    confusion: dict[str, object] = {"labels": [0, 1], "matrices": {}}
    integrity: dict[str, object] = {"global": {"frozen_permutation": True, "competition_test_accessed": False}, "folds": {}}

    for fold in FOLDS:
        normal = split_fold(features, reod, partition, fold)
        prepared = {
            "fit": prepare_evaluable_directional_rows(normal.fit_features, normal.fit_target),
            "temporal_oos": prepare_evaluable_directional_rows(normal.temporal_features, normal.temporal_target),
            "joint_oos": prepare_evaluable_directional_rows(normal.joint_features, normal.joint_target),
        }
        for subset_name, subset in prepared.items():
            manifests.append({"fold": fold.fold, "subset": subset_name, **subset.manifest})

        fit = prepared["fit"]
        assert_both_sign_classes(fit.sign_target, context=f"EXP_007 fit fold {fold.fold}")
        baseline_class = sign_majority_class(fit.sign_target)
        transformer = PositionalPathTransformer().fit(fit.raw_returns)
        preprocessing_rows.extend(transformer.parameter_rows(fold=fold.fold))

        # Training receives only the fit pair. OOS tensors are built after epoch 10.
        fit_real, fit_permuted, fit_index = _sequence_pair(transformer, fit.raw_returns)
        assert_paired_rows(fit_index, fit_index, fit.sign_target, fit.sign_target.copy())
        real_model, permuted_model = build_paired_models()
        diagnostics_rows.extend(
            train_paired_models(
                real_model, permuted_model, fit_real, fit_permuted, fit.sign_target, fold_number=fold.fold
            )
        )

        fold_integrity: dict[str, object] = {
            "fit_medians_finite": True,
            "paired_initialization": True,
            "identical_minibatch_schedule": True,
            "n_epochs": N_EPOCHS,
            "n_trainable_parameters": N_TRAINABLE_PARAMETERS,
            "subsets": {},
        }
        for subset_name in ("temporal_oos", "joint_oos"):
            subset = prepared[subset_name]
            assert_both_sign_classes(subset.sign_target, context=f"EXP_007 {subset_name} fold {fold.fold}")
            real_sequence, permuted_sequence, row_index = _sequence_pair(transformer, subset.raw_returns)
            assert_paired_rows(row_index, row_index, subset.sign_target, subset.sign_target.copy())
            baseline_metrics, baseline_matrix = evaluate_sign_majority_baseline(subset.sign_target, baseline_class)

            outputs: dict[str, dict[str, object]] = {}
            for condition, model, sequence in (
                ("real", real_model, real_sequence),
                ("permuted", permuted_model, permuted_sequence),
            ):
                model_metrics, model_matrix = _evaluate_condition(
                    model=model, sequence=sequence, target=subset.sign_target, baseline_metrics=baseline_metrics
                )
                record: dict[str, object] = {
                    "experiment": "EXP_007",
                    "condition": condition,
                    "fold": fold.fold,
                    "subset": subset_name,
                    "n_rows": int(len(subset.sign_target)),
                    "baseline_class_from_fit": baseline_class,
                    **model_metrics,
                }
                for metric_name, metric_value in baseline_metrics.items():
                    record[f"baseline_{metric_name}"] = metric_value
                metrics_rows.append(record)
                outputs[condition] = {"metrics": model_metrics, "record": record}
                matrices = confusion["matrices"]
                assert isinstance(matrices, dict)
                matrices[f"{condition}_fold_{fold.fold}_{subset_name}"] = model_matrix
            matrices = confusion["matrices"]
            assert isinstance(matrices, dict)
            matrices[f"baseline_fold_{fold.fold}_{subset_name}"] = baseline_matrix

            real_auc = float(outputs["real"]["metrics"]["roc_auc"])
            permuted_auc = float(outputs["permuted"]["metrics"]["roc_auc"])
            delta_auc = delta_vs_baseline(real_auc, permuted_auc)
            chronology_rows.append(
                {
                    "fold": fold.fold,
                    "subset": subset_name,
                    "real_roc_auc": real_auc,
                    "permuted_roc_auc": permuted_auc,
                    "real_minus_permuted_roc_auc": delta_auc,
                }
            )
            fold_integrity["subsets"][subset_name] = {
                "identical_rows_and_targets": True,
                "both_sign_classes_present": True,
                "real_permuted_pairing": True,
            }
            if subset_name == "joint_oos":
                real_metrics = outputs["real"]["metrics"]
                decision_rows.append(
                    {
                        "record_type": "fold",
                        "fold": fold.fold,
                        "real_roc_auc": real_auc,
                        "permuted_roc_auc": permuted_auc,
                        "real_minus_permuted_roc_auc": delta_auc,
                        "balanced_accuracy": real_metrics["balanced_accuracy"],
                        "recall_negative": real_metrics["recall_negative"],
                        "recall_positive": real_metrics["recall_positive"],
                        "balanced_accuracy_delta_vs_baseline": real_metrics[
                            "balanced_accuracy_delta_vs_baseline"
                        ],
                        **primary_fold_decision(real_auc, delta_auc),
                        **secondary_fold_decision(real_metrics),
                    }
                )
        integrity["folds"][str(fold.fold)] = fold_integrity

    manifest = pd.DataFrame(manifests).sort_values(["fold", "subset"])
    preprocessing = pd.DataFrame(preprocessing_rows).sort_values(["fold", "return_column"])
    diagnostics = pd.DataFrame(diagnostics_rows).sort_values(["fold", "condition", "epoch"])
    metrics = pd.DataFrame(metrics_rows).sort_values(["condition", "subset", "fold"])
    chronology = pd.DataFrame(chronology_rows).sort_values(["subset", "fold"])
    decisions = pd.DataFrame(decision_rows).sort_values("fold")
    aggregate = pd.DataFrame(
        [
            {
                "record_type": "aggregate_primary",
                "fold": "all",
                **aggregate_primary_decision(
                    decisions["real_roc_auc"].tolist(), decisions["real_minus_permuted_roc_auc"].tolist()
                ),
            },
            {
                "record_type": "aggregate_secondary",
                "fold": "all",
                **aggregate_secondary_decision(
                    decisions.to_dict("records"), decisions["balanced_accuracy_delta_vs_baseline"].tolist()
                ),
            },
        ]
    )
    decision_output = pd.concat([decisions, aggregate], ignore_index=True, sort=False)
    summary_columns = [
        "accuracy", "balanced_accuracy", "recall_negative", "recall_positive", "roc_auc",
        "accuracy_delta_vs_baseline", "balanced_accuracy_delta_vs_baseline",
    ]
    summary = metrics.groupby(["condition", "subset"])[summary_columns].agg(["mean", lambda series: series.std(ddof=1)])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    summary = summary.rename(columns=lambda name: name.replace("<lambda_0>", "std_ddof_1")).reset_index()
    delta_summary = chronology.groupby("subset")["real_minus_permuted_roc_auc"].agg(["mean", lambda series: series.std(ddof=1)])
    delta_summary = delta_summary.rename(columns={"<lambda_0>": "std_ddof_1"}).reset_index()
    delta_summary.insert(0, "condition", "real_minus_permuted")
    summary = pd.concat([summary, delta_summary], ignore_index=True, sort=False)

    integrity["global"]["all_integrity_assertions_passed"] = True
    metadata = {
        "experiment": "EXP_007",
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pytorch": torch.__version__,
        "device": "cpu",
        "deterministic_algorithms": True,
        "cpu_threads": 1,
        "interop_threads": 1,
        "model_seed": MODEL_SEED,
        "batch_order_seed": BATCH_ORDER_SEED,
        "epoch_seed_formula": "20260909 + 1000 * fold_number + epoch_number",
        "n_epochs": N_EPOCHS,
        "batch_size": BATCH_SIZE,
        "n_trainable_parameters": N_TRAINABLE_PARAMETERS,
        "permutation": list(PERMUTATION),
        "permutation_diagnostics": permutation_diagnostics(),
        "competition_test_accessed": False,
        "runtime_seconds": time.perf_counter() - started,
        "integrity_checks": integrity,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(RESULTS_DIR / "EXP_007_evaluability_manifest.csv", index=False)
    preprocessing.to_csv(RESULTS_DIR / "EXP_007_preprocessing_parameters.csv", index=False)
    diagnostics.to_csv(RESULTS_DIR / "EXP_007_training_diagnostics.csv", index=False)
    metrics.to_csv(RESULTS_DIR / "EXP_007_metrics.csv", index=False)
    chronology.to_csv(RESULTS_DIR / "EXP_007_chronology_auc.csv", index=False)
    summary.to_csv(RESULTS_DIR / "EXP_007_summary.csv", index=False)
    decision_output.to_csv(RESULTS_DIR / "EXP_007_joint_decision.csv", index=False)
    with (RESULTS_DIR / "EXP_007_confusion_matrices.json").open("w", encoding="utf-8") as file:
        json.dump(confusion, file, indent=2)
    with (RESULTS_DIR / "EXP_007_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print("EXP_007 completed using training data only; competition test was not accessed.")


if __name__ == "__main__":
    main()
