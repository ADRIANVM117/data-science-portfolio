"""Execute frozen EXP_005 positional-path signal experiment."""

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
from src.exp003_conditional_directional_sign import assert_both_sign_classes  # noqa: E402
from src.exp005_positional_path_signal import (  # noqa: E402
    PositionalPathTransformer,
    aggregate_primary_decision,
    aggregate_secondary_decision,
    assert_identical_representation_rows,
    build_original_mask,
    build_path_mask,
    delta_vs_baseline,
    evaluate_sign_majority_baseline,
    evaluate_sign_predictions,
    make_positional_logistic_regression,
    positive_probability,
    prepare_evaluable_directional_rows,
    primary_fold_decision,
    secondary_fold_decision,
    sign_majority_class,
    threshold_predictions,
)
from src.preprocessing import get_return_columns, load_training_data  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
PARTITION_PATH = PROJECT_ROOT / "experiments" / "EXP_000_equity_partition.csv"


def validate_training_contract(features: pd.DataFrame) -> None:
    if tuple(get_return_columns(features.columns)) != RETURN_COLUMNS:
        raise AssertionError("EXP_005 requires exactly r0 through r52.")
    if set(features["day"].unique()) != set(range(503)) or features["equity"].nunique() != 1829:
        raise AssertionError("Training universe does not match the frozen EXP_000 protocol.")


def main() -> None:
    features, reod = load_training_data(DATA_DIR)
    validate_training_contract(features)
    partition = pd.read_csv(PARTITION_PATH)
    validate_equity_partition(partition, features["equity"].unique(), n_holdout=N_HOLDOUT_EQUITIES)

    manifests: list[dict[str, object]] = []
    preprocessing_rows: list[dict[str, object]] = []
    metrics_rows: list[dict[str, object]] = []
    comparisons_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    confusion: dict[str, object] = {"labels": [0, 1], "matrices": {}}

    for fold in FOLDS:
        # Frozen order: normal split -> D -> N_obs >= 1 -> S -> A/B matrices.
        normal = split_fold(features, reod, partition, fold)
        prepared = {
            "fit": prepare_evaluable_directional_rows(normal.fit_features, normal.fit_target),
            "temporal_oos": prepare_evaluable_directional_rows(normal.temporal_features, normal.temporal_target),
            "joint_oos": prepare_evaluable_directional_rows(normal.joint_features, normal.joint_target),
        }
        for subset_name, subset in prepared.items():
            manifests.append({"fold": fold.fold, "subset": subset_name, **subset.manifest})

        transformer = PositionalPathTransformer().fit(prepared["fit"].raw_returns)
        preprocessing_rows.extend(transformer.parameter_rows(fold=fold.fold))
        matrices: dict[str, dict[str, pd.DataFrame]] = {}
        for subset_name, subset in prepared.items():
            a_mask = build_original_mask(subset.raw_returns)
            b_path_mask = build_path_mask(transformer, subset.raw_returns)
            assert_identical_representation_rows(a_mask, b_path_mask, subset.sign_target)
            matrices[subset_name] = {"A_mask": a_mask, "B_path_mask": b_path_mask}

        fit = prepared["fit"]
        assert_both_sign_classes(fit.sign_target, context=f"fit fold {fold.fold}")
        baseline_class = sign_majority_class(fit.sign_target)
        classifiers = {}
        for variant in ("A_mask", "B_path_mask"):
            classifier = make_positional_logistic_regression()
            classifier.fit(matrices["fit"][variant], fit.sign_target)
            if tuple(classifier.classes_) != (0, 1):
                raise AssertionError(f"EXP_005 {variant} classifier was not fitted on both sign classes.")
            classifiers[variant] = classifier

        for subset_name in ("temporal_oos", "joint_oos"):
            subset = prepared[subset_name]
            assert_both_sign_classes(subset.sign_target, context=f"{subset_name} fold {fold.fold}")
            variant_outputs: dict[str, dict[str, object]] = {}
            baseline_metrics, baseline_matrix = evaluate_sign_majority_baseline(subset.sign_target, baseline_class)
            for variant in ("A_mask", "B_path_mask"):
                probability = positive_probability(classifiers[variant], matrices[subset_name][variant])
                prediction = threshold_predictions(probability)
                model_metrics, model_matrix = evaluate_sign_predictions(
                    subset.sign_target, prediction, probability=probability
                )
                record: dict[str, object] = {
                    "experiment": "EXP_005",
                    "variant": variant,
                    "fold": fold.fold,
                    "subset": subset_name,
                    "n_rows": len(subset.sign_target),
                    "baseline_class_from_fit": baseline_class,
                    **model_metrics,
                }
                for name, value in baseline_metrics.items():
                    record[f"baseline_{name}"] = value
                record["accuracy_delta_vs_baseline"] = delta_vs_baseline(
                    model_metrics["accuracy"], baseline_metrics["accuracy"]
                )
                record["balanced_accuracy_delta_vs_baseline"] = delta_vs_baseline(
                    model_metrics["balanced_accuracy"], baseline_metrics["balanced_accuracy"]
                )
                metrics_rows.append(record)
                variant_outputs[variant] = {"metrics": model_metrics, "record": record}
                matrices_json = confusion["matrices"]
                assert isinstance(matrices_json, dict)
                matrices_json[f"{variant}_fold_{fold.fold}_{subset_name}"] = model_matrix
                matrices_json[f"baseline_fold_{fold.fold}_{subset_name}"] = baseline_matrix

            a_auc = float(variant_outputs["A_mask"]["metrics"]["roc_auc"])
            b_auc = float(variant_outputs["B_path_mask"]["metrics"]["roc_auc"])
            auc_delta = delta_vs_baseline(b_auc, a_auc)
            comparisons_rows.append({
                "fold": fold.fold,
                "subset": subset_name,
                "a_mask_roc_auc": a_auc,
                "b_path_mask_roc_auc": b_auc,
                "b_minus_a_roc_auc": auc_delta,
            })
            if subset_name == "joint_oos":
                b_metrics = variant_outputs["B_path_mask"]["metrics"]
                b_record = variant_outputs["B_path_mask"]["record"]
                decision_rows.append({
                    "record_type": "fold",
                    "fold": fold.fold,
                    "a_mask_roc_auc": a_auc,
                    "b_path_mask_roc_auc": b_auc,
                    "b_minus_a_roc_auc": auc_delta,
                    "balanced_accuracy": b_metrics["balanced_accuracy"],
                    "recall_negative": b_metrics["recall_negative"],
                    "recall_positive": b_metrics["recall_positive"],
                    "balanced_accuracy_delta_vs_baseline": b_record[
                        "balanced_accuracy_delta_vs_baseline"
                    ],
                    **primary_fold_decision(b_auc, auc_delta),
                    **secondary_fold_decision(b_metrics),
                })

    manifest = pd.DataFrame(manifests).sort_values(["fold", "subset"])
    preprocessing = pd.DataFrame(preprocessing_rows).sort_values(["fold", "return_column"])
    metrics = pd.DataFrame(metrics_rows).sort_values(["variant", "subset", "fold"])
    comparisons = pd.DataFrame(comparisons_rows).sort_values(["subset", "fold"])
    decisions = pd.DataFrame(decision_rows).sort_values("fold")
    aggregate = pd.DataFrame([
        {
            "record_type": "aggregate_primary",
            "fold": "all",
            **aggregate_primary_decision(
                decisions["b_path_mask_roc_auc"].tolist(), decisions["b_minus_a_roc_auc"].tolist()
            ),
        },
        {
            "record_type": "aggregate_secondary",
            "fold": "all",
            **aggregate_secondary_decision(
                decisions.to_dict("records"),
                decisions["balanced_accuracy_delta_vs_baseline"].tolist(),
            ),
        },
    ])
    decision_output = pd.concat([decisions, aggregate], ignore_index=True, sort=False)
    summary_columns = [
        "accuracy",
        "balanced_accuracy",
        "recall_negative",
        "recall_positive",
        "roc_auc",
        "accuracy_delta_vs_baseline",
        "balanced_accuracy_delta_vs_baseline",
    ]
    summary = metrics.groupby(["variant", "subset"])[summary_columns].agg(
        ["mean", lambda value: value.std(ddof=1)]
    )
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    summary = summary.rename(columns=lambda name: name.replace("<lambda_0>", "std_ddof_1")).reset_index()
    comparison_summary = comparisons.groupby("subset")["b_minus_a_roc_auc"].agg(
        ["mean", lambda value: value.std(ddof=1)]
    )
    comparison_summary = comparison_summary.rename(columns={"<lambda_0>": "std_ddof_1"}).reset_index()
    comparison_summary.insert(0, "variant", "B_path_mask_minus_A_mask")
    summary = pd.concat([summary, comparison_summary], ignore_index=True, sort=False)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(RESULTS_DIR / "EXP_005_evaluability_manifest.csv", index=False)
    preprocessing.to_csv(RESULTS_DIR / "EXP_005_preprocessing_parameters.csv", index=False)
    metrics.to_csv(RESULTS_DIR / "EXP_005_metrics.csv", index=False)
    comparisons.to_csv(RESULTS_DIR / "EXP_005_path_vs_mask_auc.csv", index=False)
    summary.to_csv(RESULTS_DIR / "EXP_005_summary.csv", index=False)
    decision_output.to_csv(RESULTS_DIR / "EXP_005_joint_decision.csv", index=False)
    with (RESULTS_DIR / "EXP_005_confusion_matrices.json").open("w", encoding="utf-8") as file:
        json.dump(confusion, file, indent=2)
    print("EXP_005 completed using training data only; competition test was not accessed.")
    print(f"scikit-learn version: {sklearn.__version__}")


if __name__ == "__main__":
    main()
