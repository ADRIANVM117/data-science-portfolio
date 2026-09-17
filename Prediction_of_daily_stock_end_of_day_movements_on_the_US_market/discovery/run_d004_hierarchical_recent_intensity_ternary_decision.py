"""Execute frozen D004 only after separate real-execution authorization."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from discovery.d003_incremental_cross_sectional_relative_path import (  # noqa: E402
    assert_e_dev_only,
    split_discovery_fold,
    validate_discovery_features,
)
from discovery.d004_hierarchical_recent_intensity_ternary_decision import (  # noqa: E402
    DISCOVERY_FOLDS,
    build_d004_matrix,
    calibration_diagnostics,
    directional_probability,
    fit_directional_priors,
    fit_intensity_transformer,
    fit_majority_predictions,
    hierarchical_predictions,
    ternary_metrics,
)
from src.exp002_neutral_directional import build_binary_target, make_binary_logistic_regression  # noqa: E402


DISCOVERY_DATA_DIR = PROJECT_ROOT / "data" / "discovery"
INPUT_PATH = DISCOVERY_DATA_DIR / "discovery_input_training.csv"
OUTPUT_PATH = DISCOVERY_DATA_DIR / "discovery_output_training.csv"
MANIFEST_PATH = DISCOVERY_DATA_DIR / "discovery_partition_manifest.json"
PARTITION_PATH = PROJECT_ROOT / "experiments" / "EXP_000_equity_partition.csv"
RESULTS_DIR = PROJECT_ROOT / "discovery" / "results"


def load_physical_discovery_data() -> tuple[pd.DataFrame, pd.Series]:
    """Fail closed unless the authorized materialized Discovery files are valid."""
    if not all(path.is_file() for path in (INPUT_PATH, OUTPUT_PATH, MANIFEST_PATH, PARTITION_PATH)):
        raise FileNotFoundError("D004 requires the physical Discovery files and frozen equity partition; no fallback is allowed.")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    features = pd.read_csv(INPUT_PATH)
    labels = pd.read_csv(OUTPUT_PATH)
    validate_discovery_features(features)
    assert_e_dev_only(features, pd.read_csv(PARTITION_PATH))
    if len(features) != int(manifest["row_count"]) or features["ID"].nunique() != len(features):
        raise AssertionError("D004 physical Discovery input disagrees with its manifest.")
    if set(labels.columns) != {"ID", "reod"} or labels["ID"].duplicated().any():
        raise AssertionError("D004 physical Discovery labels have an unexpected schema.")
    aligned = labels.set_index("ID").reindex(features["ID"])
    if aligned["reod"].isna().any():
        raise AssertionError("D004 labels do not align to physical Discovery IDs.")
    reod = pd.Series(aligned["reod"].to_numpy(), index=features.index, name="reod")
    if not set(reod.unique()).issubset({-1, 0, 1}):
        raise AssertionError("D004 labels must be ternary.")
    return features, reod


def main() -> None:
    features, reod = load_physical_discovery_data()
    metric_rows: list[dict[str, int | float | str]] = []
    routing_rows: list[dict[str, int | float | bool]] = []
    calibration_rows: list[dict[str, int | float]] = []
    reliability_rows: list[dict[str, int | float | str]] = []
    confusion: dict[str, object] = {"labels": [-1, 0, 1], "matrices": {}}

    for fold in DISCOVERY_FOLDS:
        subsets = split_discovery_fold(features, reod, fold)
        fit_structure, transformer = fit_intensity_transformer(subsets.fit_features)
        fit_matrix, _ = build_d004_matrix(subsets.fit_features, subsets.fit_reod, transformer)
        validation_matrix, _ = build_d004_matrix(subsets.validation_features, subsets.validation_reod, transformer)
        model: LogisticRegression = make_binary_logistic_regression().fit(fit_matrix, build_binary_target(subsets.fit_reod))
        p_d = directional_probability(model, validation_matrix)
        priors = fit_directional_priors(subsets.fit_reod)
        candidate_prediction = hierarchical_predictions(p_d, priors)
        baseline_class, baseline_prediction = fit_majority_predictions(subsets.fit_reod, len(subsets.validation_reod))
        candidate_metrics, candidate_confusion = ternary_metrics(subsets.validation_reod, candidate_prediction)
        baseline_metrics, baseline_confusion = ternary_metrics(subsets.validation_reod, baseline_prediction)
        z_validation = build_binary_target(subsets.validation_reod)
        calibration, reliability = calibration_diagnostics(z_validation, p_d)
        nvd_auc = float(roc_auc_score(z_validation, p_d))

        for condition, metrics, matrix in (("hierarchical", candidate_metrics, candidate_confusion), ("baseline", baseline_metrics, baseline_confusion)):
            metric_rows.append({"fold": fold.fold, "condition": condition, "n_rows": len(subsets.validation_reod), **metrics})
            confusion["matrices"][f"{condition}_fold_{fold.fold}_validation"] = matrix
        directional = candidate_prediction != 0
        routing_rows.append({
            "fold": fold.fold, "pi_minus": priors.pi_minus, "pi_plus": priors.pi_plus,
            "p_d_star": priors.threshold, "baseline_class_from_fit": baseline_class,
            "fraction_neutral": float((~directional).mean()), "fraction_directional": float(directional.mean()),
            "fraction_predicted_minus_among_directional": float((candidate_prediction[directional] == -1).mean()) if directional.any() else 0.0,
            "fraction_predicted_plus_among_directional": float((candidate_prediction[directional] == 1).mean()) if directional.any() else 0.0,
            "both_directional_classes_predicted": bool((candidate_prediction == -1).any() and (candidate_prediction == 1).any()),
            "nvd_roc_auc": nvd_auc,
            "delta_accuracy": candidate_metrics["accuracy"] - baseline_metrics["accuracy"],
        })
        calibration_rows.append({"fold": fold.fold, **calibration})
        reliability_rows.extend([{"fold": fold.fold, **record} for record in reliability.to_dict("records")])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(RESULTS_DIR / "D004_metrics.csv", index=False)
    pd.DataFrame(routing_rows).to_csv(RESULTS_DIR / "D004_routing.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(RESULTS_DIR / "D004_calibration.csv", index=False)
    pd.DataFrame(reliability_rows).to_csv(RESULTS_DIR / "D004_reliability.csv", index=False)
    with (RESULTS_DIR / "D004_confusion_matrices.json").open("w", encoding="utf-8") as handle:
        json.dump(confusion, handle, indent=2)
    print("D004 completed using only the physical Discovery partition; competition test was not accessed.")


if __name__ == "__main__":
    main()
