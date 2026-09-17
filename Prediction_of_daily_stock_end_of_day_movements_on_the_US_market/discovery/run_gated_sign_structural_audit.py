"""Post-D004 structural feasibility audit; no conditional-sign model or metric."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from discovery.d003_incremental_cross_sectional_relative_path import split_discovery_fold  # noqa: E402
from discovery.d004_hierarchical_recent_intensity_ternary_decision import (  # noqa: E402
    DISCOVERY_FOLDS,
    build_d004_matrix,
    directional_probability,
    fit_directional_priors,
    fit_intensity_transformer,
)
from discovery.run_d004_hierarchical_recent_intensity_ternary_decision import (  # noqa: E402
    RESULTS_DIR,
    load_physical_discovery_data,
)
from src.exp002_neutral_directional import build_binary_target, make_binary_logistic_regression  # noqa: E402


def distribution_summary(values: pd.Series) -> dict[str, float | int]:
    """Compact target-free coverage summary for a positive-count Series."""
    if values.empty or (values <= 0).any():
        raise AssertionError("Coverage summaries require positive represented-group counts.")
    quantiles = values.quantile([0.05, 0.50, 0.95])
    return {
        "min": int(values.min()), "p05": float(quantiles.loc[0.05]),
        "median": float(quantiles.loc[0.50]), "p95": float(quantiles.loc[0.95]), "max": int(values.max()),
    }


def coverage_record(frame: pd.DataFrame, gate: np.ndarray) -> dict[str, float | int]:
    """Return target-free gate coverage over day/equity identifiers."""
    selected = frame.loc[np.asarray(gate, dtype=bool)]
    if selected.empty:
        return {"n_unique_days": 0, "n_unique_equities": 0, **{f"rows_per_day_{key}": np.nan for key in ("min", "p05", "median", "p95", "max")}, **{f"rows_per_equity_{key}": np.nan for key in ("min", "p05", "median", "p95", "max")}}
    day = distribution_summary(selected.groupby("day").size())
    equity = distribution_summary(selected.groupby("equity").size())
    return {"n_unique_days": int(selected["day"].nunique()), "n_unique_equities": int(selected["equity"].nunique()), **{f"rows_per_day_{key}": value for key, value in day.items()}, **{f"rows_per_equity_{key}": value for key, value in equity.items()}}


def main() -> None:
    features, reod = load_physical_discovery_data()
    gate_rows: list[dict[str, int | float | str]] = []
    eligibility_rows: list[dict[str, int | float | str]] = []
    eligible_coverage_rows: list[dict[str, int | float | str]] = []

    for fold in DISCOVERY_FOLDS:
        subsets = split_discovery_fold(features, reod, fold)
        _, transformer = fit_intensity_transformer(subsets.fit_features)
        fit_matrix, _ = build_d004_matrix(subsets.fit_features, subsets.fit_reod, transformer)
        validation_matrix, _ = build_d004_matrix(subsets.validation_features, subsets.validation_reod, transformer)
        model = make_binary_logistic_regression().fit(fit_matrix, build_binary_target(subsets.fit_reod))
        priors = fit_directional_priors(subsets.fit_reod)

        for subset_name, subset_features, subset_reod, matrix in (
            ("fit", subsets.fit_features, subsets.fit_reod, fit_matrix),
            ("validation", subsets.validation_features, subsets.validation_reod, validation_matrix),
        ):
            p_d = directional_probability(model, matrix)
            gate = p_d > priors.threshold  # exact frozen D004 gate; equality is not routed.
            coverage = coverage_record(subset_features, gate)
            gate_rows.append({
                "fold": fold.fold, "subset": subset_name, "n_rows": len(subset_features),
                "n_gate_1": int(gate.sum()), "n_gate_0": int((~gate).sum()), "fraction_gate_1": float(gate.mean()),
                "pi_minus": priors.pi_minus, "pi_plus": priors.pi_plus, "pi_max": priors.pi_max,
                "p_d_star": priors.threshold, **coverage,
            })

            truth = subset_reod.to_numpy(dtype=np.int8)
            directional = np.isin(truth, [-1, 1])
            eligible = gate & directional
            n_minus, n_zero, n_plus = (int(((truth == label) & gate).sum()) for label in (-1, 0, 1))
            n_eligible = int(eligible.sum())
            eligibility_rows.append({
                "fold": fold.fold, "subset": subset_name,
                "n_gate_true_minus": n_minus, "n_gate_true_zero": n_zero, "n_gate_true_plus": n_plus,
                "n_gate_directional": n_minus + n_plus,
                "fraction_gate_directional": (n_minus + n_plus) / int(gate.sum()) if gate.any() else np.nan,
                "n_eligible_sign": n_eligible,
                "n_eligible_minus": n_minus, "n_eligible_plus": n_plus,
                "fraction_eligible_minus": n_minus / n_eligible if n_eligible else np.nan,
                "fraction_eligible_plus": n_plus / n_eligible if n_eligible else np.nan,
                "n_original_directional": int(directional.sum()),
                "fraction_original_directional_retained": n_eligible / int(directional.sum()) if directional.any() else np.nan,
            })
            eligible_coverage_rows.append({"fold": fold.fold, "subset": subset_name, "n_eligible_sign": n_eligible, **coverage_record(subset_features, eligible)})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(gate_rows).to_csv(RESULTS_DIR / "GATED_SIGN_STRUCTURAL_AUDIT_gate_coverage.csv", index=False)
    pd.DataFrame(eligibility_rows).to_csv(RESULTS_DIR / "GATED_SIGN_STRUCTURAL_AUDIT_eligibility.csv", index=False)
    pd.DataFrame(eligible_coverage_rows).to_csv(RESULTS_DIR / "GATED_SIGN_STRUCTURAL_AUDIT_eligible_coverage.csv", index=False)
    metadata = {
        "audit": "GATED_SIGN_STRUCTURAL_AUDIT", "target_free_gate_with_respect_to_validation_sign": True,
        "conditional_sign_model_fitted": False, "conditional_sign_predictive_metric_evaluated": False,
        "alternative_gate_or_threshold_tested": False, "days": "0-352", "equity_partition": "E_dev",
        "competition_test_accessed": False,
    }
    (RESULTS_DIR / "GATED_SIGN_STRUCTURAL_AUDIT_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Gated-sign structural audit completed; no conditional-sign model or predictive statistic was evaluated.")


if __name__ == "__main__":
    main()
