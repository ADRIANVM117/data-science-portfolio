"""Execute frozen D006 only after separate real-execution approval."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from discovery.d003_incremental_cross_sectional_relative_path import split_discovery_fold  # noqa: E402
from discovery.d004_hierarchical_recent_intensity_ternary_decision import fit_intensity_transformer  # noqa: E402
from discovery.run_d004_hierarchical_recent_intensity_ternary_decision import RESULTS_DIR, load_physical_discovery_data  # noqa: E402
from discovery.d006_ternary_tail_intensity_interaction import (  # noqa: E402
    DISCOVERY_FOLDS,
    build_d006_representations,
    candidate_screen,
    evaluate_d006_probabilities,
    make_d006_multinomial_logistic_regression,
    orientation_consistency,
    orientation_shift,
    ternary_probabilities,
)


def main() -> None:
    features, reod = load_physical_discovery_data()
    manifests, metrics, parameters, orientation_records = [], [], [], []
    confusion: dict[str, object] = {"labels": [-1, 0, 1], "matrices": {}}
    fold_metric: dict[str, list[float]] = {name: [] for name in ("C0", "C1", "C2")}
    c1_deltas: list[float] = []
    c2_deltas: list[float] = []
    shifts: dict[str, list[float]] = {"C1": [], "C2": []}
    for fold in DISCOVERY_FOLDS:
        subsets = split_discovery_fold(features, reod, fold)
        fit_structure, transformer = fit_intensity_transformer(subsets.fit_features)
        fit_matrices = build_d006_representations(subsets.fit_features, subsets.fit_reod, transformer)
        validation_matrices = build_d006_representations(subsets.validation_features, subsets.validation_reod, transformer)
        for subset_name, subset_features, subset_reod in (("fit", subsets.fit_features, subsets.fit_reod), ("validation", subsets.validation_features, subsets.validation_reod)):
            manifests.append({"fold": fold.fold, "subset": subset_name, "n_rows": len(subset_reod), "class_minus": int(subset_reod.eq(-1).sum()), "class_zero": int(subset_reod.eq(0).sum()), "class_plus": int(subset_reod.eq(1).sum()), "n_q": int((fit_structure["q"] if subset_name == "fit" else build_d006_representations(subset_features, subset_reod, transformer)["C0"]["q"]).sum()), "n_unique_days": int(subset_features["day"].nunique()), "n_unique_equities": int(subset_features["equity"].nunique())})
        parameters.append({"fold": fold.fold, "fit_intensity_median": transformer.fit_intensity_median_, "scaler_mean": float(transformer.scaler_.mean_[0]), "scaler_scale": float(transformer.scaler_.scale_[0])})
        current: dict[str, float] = {}
        for name in ("C0", "C1", "C2"):
            model = make_d006_multinomial_logistic_regression().fit(fit_matrices[name], subsets.fit_reod)
            probabilities = ternary_probabilities(model, validation_matrices[name])
            observed, matrix = evaluate_d006_probabilities(subsets.validation_reod, probabilities)
            current[name] = observed["macro_tail_auc"]
            fold_metric[name].append(current[name])
            metrics.append({"fold": fold.fold, "representation": name, "n_rows": len(subsets.validation_reod), **observed})
            confusion["matrices"][f"{name}_fold_{fold.fold}_validation"] = matrix
            if name in shifts:
                value = orientation_shift(model, name, validation_matrices[name])
                shifts[name].append(value)
                orientation_records.append({"fold": fold.fold, "representation": name, "mean_signed_tail_shift": value})
        c1_deltas.append(current["C1"] - current["C0"])
        c2_deltas.append(current["C2"] - current["C1"])
    increments = pd.DataFrame({"fold": [1, 2, 3], "C1_minus_C0_macro_tail_auc": c1_deltas, "C2_minus_C1_macro_tail_auc": c2_deltas})
    summary = {
        "C1_allocation": candidate_screen(fold_metric["C1"], c1_deltas),
        "C2_interaction": candidate_screen(fold_metric["C2"], c2_deltas),
        "C1_orientation": orientation_consistency(shifts["C1"]),
        "C2_orientation": orientation_consistency(shifts["C2"]),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifests).to_csv(RESULTS_DIR / "D006_fold_manifest.csv", index=False)
    pd.DataFrame(parameters).to_csv(RESULTS_DIR / "D006_preprocessing_parameters.csv", index=False)
    pd.DataFrame(metrics).to_csv(RESULTS_DIR / "D006_metrics.csv", index=False)
    increments.to_csv(RESULTS_DIR / "D006_incremental_macro_tail_auc.csv", index=False)
    pd.DataFrame(orientation_records).to_csv(RESULTS_DIR / "D006_orientation.csv", index=False)
    (RESULTS_DIR / "D006_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (RESULTS_DIR / "D006_confusion_matrices.json").write_text(json.dumps(confusion, indent=2), encoding="utf-8")
    (RESULTS_DIR / "D006_metadata.json").write_text(json.dumps({"experiment": "D006", "scope": "physical Discovery days 0-352 x E_dev", "competition_test_accessed": False}, indent=2), encoding="utf-8")
    print("D006 completed only after separate real-execution authorization.")


if __name__ == "__main__":
    main()
