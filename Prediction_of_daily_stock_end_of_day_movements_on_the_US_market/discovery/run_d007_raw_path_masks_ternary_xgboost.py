"""Execute frozen D007 only after separate real-execution authorization."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from discovery.d003_incremental_cross_sectional_relative_path import split_discovery_fold  # noqa: E402
from discovery.run_d004_hierarchical_recent_intensity_ternary_decision import RESULTS_DIR, load_physical_discovery_data  # noqa: E402
from discovery.d007_raw_path_masks_ternary_xgboost import (  # noqa: E402
    DISCOVERY_FOLDS, XGBOOST_PARAMS, build_d007_representations, encode_ternary_target,
    evaluate_d007_probabilities, evaluate_majority_baseline, make_d007_classifier,
    primary_screen, ternary_probabilities,
)

ARTIFACT_NAMES = ("D007_fold_manifest.csv", "D007_model_spec.json", "D007_metrics.csv", "D007_incremental_log_loss.csv", "D007_summary.json", "D007_confusion_matrices.json", "D007_majority_baseline.csv", "D007_metadata.json")


def assert_fresh_artifacts() -> None:
    existing = [name for name in ARTIFACT_NAMES if (RESULTS_DIR / name).exists()]
    if existing:
        raise FileExistsError(f"D007 refuses to overwrite existing result artifacts: {existing}")


def main() -> None:
    assert_fresh_artifacts()
    features, reod = load_physical_discovery_data()
    manifests, metric_rows, baseline_rows, deltas = [], [], [], []
    confusion: dict[str, object] = {"labels": [-1, 0, 1], "matrices": {}}
    for fold in DISCOVERY_FOLDS:
        subsets = split_discovery_fold(features, reod, fold)
        fit_matrices = build_d007_representations(subsets.fit_features, subsets.fit_reod)
        validation_matrices = build_d007_representations(subsets.validation_features, subsets.validation_reod)
        for name, frame, target in (("fit", subsets.fit_features, subsets.fit_reod), ("validation", subsets.validation_features, subsets.validation_reod)):
            manifests.append({"fold": fold.fold, "subset": name, "n_rows": len(target), "class_minus": int(target.eq(-1).sum()), "class_zero": int(target.eq(0).sum()), "class_plus": int(target.eq(1).sum()), "n_unique_days": int(frame.day.nunique()), "n_unique_equities": int(frame.equity.nunique())})
        losses: dict[str, float] = {}
        for representation in ("M", "R+M"):
            model = make_d007_classifier().fit(fit_matrices[representation], encode_ternary_target(subsets.fit_reod))
            probabilities = ternary_probabilities(model, validation_matrices[representation])
            metrics, matrix = evaluate_d007_probabilities(subsets.validation_reod, probabilities)
            metric_rows.append({"fold": fold.fold, "representation": representation, "n_rows": len(subsets.validation_reod), **metrics})
            confusion["matrices"][f"{representation}_fold_{fold.fold}_validation"] = matrix
            losses[representation] = metrics["multiclass_log_loss"]
        deltas.append({"fold": fold.fold, "M_log_loss": losses["M"], "R_plus_M_log_loss": losses["R+M"], "delta_log_loss": losses["M"] - losses["R+M"]})
        selected, metrics, matrix = evaluate_majority_baseline(subsets.fit_reod, subsets.validation_reod)
        baseline_rows.append({"fold": fold.fold, "fit_majority_class": selected, "n_rows": len(subsets.validation_reod), **metrics})
        confusion["matrices"][f"majority_fold_{fold.fold}_validation"] = matrix
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifests).to_csv(RESULTS_DIR / "D007_fold_manifest.csv", index=False)
    (RESULTS_DIR / "D007_model_spec.json").write_text(json.dumps(dict(XGBOOST_PARAMS), indent=2), encoding="utf-8")
    pd.DataFrame(metric_rows).to_csv(RESULTS_DIR / "D007_metrics.csv", index=False)
    pd.DataFrame(deltas).to_csv(RESULTS_DIR / "D007_incremental_log_loss.csv", index=False)
    (RESULTS_DIR / "D007_summary.json").write_text(json.dumps(primary_screen([row["delta_log_loss"] for row in deltas]), indent=2), encoding="utf-8")
    (RESULTS_DIR / "D007_confusion_matrices.json").write_text(json.dumps(confusion, indent=2), encoding="utf-8")
    pd.DataFrame(baseline_rows).to_csv(RESULTS_DIR / "D007_majority_baseline.csv", index=False)
    (RESULTS_DIR / "D007_metadata.json").write_text(json.dumps({"experiment": "D007", "scope": "physical Discovery days 0-352 x E_dev", "competition_test_accessed": False}, indent=2), encoding="utf-8")
    print("D007 completed only after separate real-execution authorization.")


if __name__ == "__main__":
    main()
