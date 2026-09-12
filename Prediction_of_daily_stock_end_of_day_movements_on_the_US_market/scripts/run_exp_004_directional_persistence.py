"""Execute the frozen EXP_004 conditional directional-persistence experiment."""

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
from src.exp003_conditional_directional_sign import (  # noqa: E402
    assert_both_sign_classes,
    coefficient_sign,
    coefficient_sign_consistent,
    evaluate_sign_majority_baseline,
    evaluate_sign_predictions,
    positive_probability,
    sign_majority_class,
    threshold_predictions,
)
from src.exp004_directional_persistence import (  # noqa: E402
    aggregate_primary_decision,
    aggregate_secondary_decision,
    delta_vs_baseline,
    make_persistence_logistic_regression,
    persistence_beta,
    prepare_evaluable_persistence_subset,
    primary_fold_decision,
    secondary_fold_decision,
)
from src.preprocessing import get_return_columns, load_training_data  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
PARTITION_PATH = PROJECT_ROOT / "experiments" / "EXP_000_equity_partition.csv"


def validate_training_contract(features: pd.DataFrame) -> None:
    """Confirm that the permitted training universe matches frozen EXP_000."""
    if tuple(get_return_columns(features.columns)) != RETURN_COLUMNS:
        raise AssertionError("EXP_004 requires exactly r0 through r52.")
    if set(features["day"].unique()) != set(range(503)) or features["equity"].nunique() != 1829:
        raise AssertionError("Training universe does not match the frozen EXP_000 protocol.")


def main() -> None:
    features, reod = load_training_data(DATA_DIR)
    validate_training_contract(features)
    partition = pd.read_csv(PARTITION_PATH)
    validate_equity_partition(partition, features["equity"].unique(), n_holdout=N_HOLDOUT_EQUITIES)

    manifests: list[dict[str, object]] = []
    metrics_rows: list[dict[str, object]] = []
    decisions_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    confusion: dict[str, object] = {"labels": [0, 1], "matrices": {}}

    for fold in FOLDS:
        # Frozen order: shared split -> D -> N_obs >= 1 -> construct P.
        normal = split_fold(features, reod, partition, fold)
        prepared = {
            "fit": prepare_evaluable_persistence_subset(normal.fit_features, normal.fit_target),
            "temporal_oos": prepare_evaluable_persistence_subset(normal.temporal_features, normal.temporal_target),
            "joint_oos": prepare_evaluable_persistence_subset(normal.joint_features, normal.joint_target),
        }
        for subset_name, subset in prepared.items():
            manifests.append({"fold": fold.fold, "subset": subset_name, **subset.manifest})

        fit = prepared["fit"]
        assert_both_sign_classes(fit.sign_target, context=f"fit fold {fold.fold}")
        classifier = make_persistence_logistic_regression()
        classifier.fit(fit.feature_matrix, fit.sign_target)
        if tuple(classifier.classes_) != (0, 1):
            raise AssertionError("EXP_004 classifier was not fitted on both sign classes.")
        beta = persistence_beta(classifier)
        coefficient_rows.append({"fold": fold.fold, "beta_P": beta, "beta_sign": coefficient_sign(beta)})
        baseline_class = sign_majority_class(fit.sign_target)

        for subset_name in ("temporal_oos", "joint_oos"):
            subset = prepared[subset_name]
            assert_both_sign_classes(subset.sign_target, context=f"{subset_name} fold {fold.fold}")
            probability = positive_probability(classifier, subset.feature_matrix)
            prediction = threshold_predictions(probability)
            model_metrics, model_matrix = evaluate_sign_predictions(
                subset.sign_target, prediction, probability=probability
            )
            baseline_metrics, baseline_matrix = evaluate_sign_majority_baseline(
                subset.sign_target, baseline_class
            )
            record: dict[str, object] = {
                "experiment": "EXP_004",
                "fold": fold.fold,
                "subset": subset_name,
                "n_rows": len(subset.sign_target),
                "baseline_class_from_fit": baseline_class,
                "beta_P": beta,
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
            matrices = confusion["matrices"]
            assert isinstance(matrices, dict)
            matrices[f"model_fold_{fold.fold}_{subset_name}"] = model_matrix
            matrices[f"baseline_fold_{fold.fold}_{subset_name}"] = baseline_matrix
            if subset_name == "joint_oos":
                decisions_rows.append({
                    "record_type": "fold",
                    "fold": fold.fold,
                    "roc_auc": model_metrics["roc_auc"],
                    "balanced_accuracy": model_metrics["balanced_accuracy"],
                    "recall_negative": model_metrics["recall_negative"],
                    "recall_positive": model_metrics["recall_positive"],
                    "balanced_accuracy_delta_vs_baseline": record[
                        "balanced_accuracy_delta_vs_baseline"
                    ],
                    **primary_fold_decision(model_metrics["roc_auc"]),
                    **secondary_fold_decision(model_metrics),
                })

    metrics = pd.DataFrame(metrics_rows).sort_values(["subset", "fold"])
    manifest = pd.DataFrame(manifests).sort_values(["fold", "subset"])
    decisions = pd.DataFrame(decisions_rows).sort_values("fold")
    coefficients = pd.DataFrame(coefficient_rows).sort_values("fold")
    coefficients = pd.concat(
        [
            coefficients,
            pd.DataFrame([{
                "fold": "all",
                "beta_sign_consistent": coefficient_sign_consistent(coefficients["beta_P"].tolist()),
            }]),
        ],
        ignore_index=True,
        sort=False,
    )
    aggregate = pd.DataFrame([
        {
            "record_type": "aggregate_primary",
            "fold": "all",
            **aggregate_primary_decision(decisions["roc_auc"].tolist()),
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
    summary = metrics.groupby("subset")[summary_columns].agg(["mean", lambda value: value.std(ddof=1)])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    summary = summary.rename(columns=lambda name: name.replace("<lambda_0>", "std_ddof_1")).reset_index()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(RESULTS_DIR / "EXP_004_evaluability_manifest.csv", index=False)
    metrics.to_csv(RESULTS_DIR / "EXP_004_metrics.csv", index=False)
    summary.to_csv(RESULTS_DIR / "EXP_004_summary.csv", index=False)
    decision_output.to_csv(RESULTS_DIR / "EXP_004_joint_decision.csv", index=False)
    coefficients.to_csv(RESULTS_DIR / "EXP_004_coefficients.csv", index=False)
    with (RESULTS_DIR / "EXP_004_confusion_matrices.json").open("w", encoding="utf-8") as file:
        json.dump(confusion, file, indent=2)
    print("EXP_004 completed using training data only; competition test was not accessed.")
    print(f"scikit-learn version: {sklearn.__version__}")


if __name__ == "__main__":
    main()
