"""Descriptive post-result diagnosis of frozen D004; no alternative decision rule."""

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
    fit_majority_predictions,
    hierarchical_predictions,
    ternary_metrics,
)
from discovery.run_d004_hierarchical_recent_intensity_ternary_decision import (  # noqa: E402
    RESULTS_DIR,
    load_physical_discovery_data,
)
from src.exp002_neutral_directional import build_binary_target, make_binary_logistic_regression  # noqa: E402


MARGIN_REGIONS: tuple[tuple[str, float | None, float | None, bool, bool], ...] = (
    ("(-inf, -0.20)", None, -0.20, False, False),
    ("[-0.20, -0.10)", -0.20, -0.10, True, False),
    ("[-0.10, -0.05)", -0.10, -0.05, True, False),
    ("[-0.05, 0.00]", -0.05, 0.00, True, True),
    ("(0.00, 0.05]", 0.00, 0.05, False, True),
    ("(0.05, 0.10]", 0.05, 0.10, False, True),
    ("(0.10, 0.20]", 0.10, 0.20, False, True),
    ("(0.20, +inf)", 0.20, None, False, False),
)
POSITIVE_MARGIN_LABELS: tuple[str, ...] = tuple(label for label, lower, _, _, _ in MARGIN_REGIONS if lower is not None and lower >= 0.0)


def margin_region_masks(margin: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Assign each finite margin once to the frozen, non-data-dependent regions."""
    values = np.asarray(margin, dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError("D004 diagnostic margins must be finite.")
    masks: list[tuple[str, np.ndarray]] = []
    for label, lower, upper, include_lower, include_upper in MARGIN_REGIONS:
        member = np.ones(len(values), dtype=bool)
        if lower is not None:
            member &= values >= lower if include_lower else values > lower
        if upper is not None:
            member &= values <= upper if include_upper else values < upper
        masks.append((label, member))
    stacked = np.vstack([mask for _, mask in masks])
    if not np.all(stacked.sum(axis=0) == 1):
        raise AssertionError("Frozen D004 margin regions must partition validation rows exactly once.")
    return masks


def _class_counts(truth: np.ndarray, member: np.ndarray) -> dict[str, int]:
    return {"n_true_minus": int((truth[member] == -1).sum()), "n_true_zero": int((truth[member] == 0).sum()), "n_true_plus": int((truth[member] == 1).sum())}


def _fractions(counts: dict[str, int], n_rows: int) -> dict[str, float]:
    return {
        "fraction_true_minus": counts["n_true_minus"] / n_rows if n_rows else np.nan,
        "fraction_true_zero": counts["n_true_zero"] / n_rows if n_rows else np.nan,
        "fraction_true_plus": counts["n_true_plus"] / n_rows if n_rows else np.nan,
    }


def main() -> None:
    required = (RESULTS_DIR / "D004_metrics.csv", RESULTS_DIR / "D004_routing.csv")
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("D004 post-result diagnosis requires persisted D004 results; no fallback is allowed.")
    persisted_metrics = pd.read_csv(required[0])
    persisted_routing = pd.read_csv(required[1]).set_index("fold")
    features, reod = load_physical_discovery_data()

    decomposition_rows: list[dict[str, int | float | str | bool]] = []
    margin_rows: list[dict[str, int | float | str]] = []
    positive_rows: list[dict[str, int | float | str]] = []
    fold_rows: list[dict[str, int | float | bool]] = []

    for fold in DISCOVERY_FOLDS:
        subsets = split_discovery_fold(features, reod, fold)
        _, transformer = fit_intensity_transformer(subsets.fit_features)
        fit_matrix, _ = build_d004_matrix(subsets.fit_features, subsets.fit_reod, transformer)
        validation_matrix, _ = build_d004_matrix(subsets.validation_features, subsets.validation_reod, transformer)
        model = make_binary_logistic_regression().fit(fit_matrix, build_binary_target(subsets.fit_reod))
        p_d = directional_probability(model, validation_matrix)
        priors = fit_directional_priors(subsets.fit_reod)
        candidate = hierarchical_predictions(p_d, priors)
        baseline_class, baseline = fit_majority_predictions(subsets.fit_reod, len(subsets.validation_reod))
        if baseline_class != 0:
            raise AssertionError("D004 decomposition identity requires the persisted neutral majority baseline.")
        candidate_metrics, _ = ternary_metrics(subsets.validation_reod, candidate)
        baseline_metrics, _ = ternary_metrics(subsets.validation_reod, baseline)
        persisted_candidate = persisted_metrics.query("fold == @fold.fold and condition == 'hierarchical'").iloc[0]
        persisted_baseline = persisted_metrics.query("fold == @fold.fold and condition == 'baseline'").iloc[0]
        if not np.isclose(candidate_metrics["accuracy"], float(persisted_candidate["accuracy"]), rtol=0.0, atol=1e-12):
            raise AssertionError("D004 diagnostic reproduction does not match persisted candidate accuracy.")
        if not np.isclose(baseline_metrics["accuracy"], float(persisted_baseline["accuracy"]), rtol=0.0, atol=1e-12):
            raise AssertionError("D004 diagnostic reproduction does not match persisted baseline accuracy.")
        if not np.isclose(priors.threshold, float(persisted_routing.loc[fold.fold, "p_d_star"]), rtol=0.0, atol=1e-12):
            raise AssertionError("D004 diagnostic threshold does not match the persisted frozen threshold.")

        truth = subsets.validation_reod.to_numpy(dtype=np.int8)
        routed = candidate != 0
        non_routed = ~routed
        routed_counts = _class_counts(truth, routed)
        neutral_counts = _class_counts(truth, non_routed)
        gain = routed_counts["n_true_minus"]
        cost = routed_counts["n_true_zero"]
        unchanged_error = routed_counts["n_true_plus"]
        candidate_correct = int((candidate == truth).sum())
        baseline_correct = int((baseline == truth).sum())
        net = gain - cost
        if candidate_correct - baseline_correct != net:
            raise AssertionError("D004 routed accuracy decomposition does not reconcile exactly.")
        delta_accuracy = net / len(truth)
        if not np.isclose(delta_accuracy, candidate_metrics["accuracy"] - baseline_metrics["accuracy"], rtol=0.0, atol=1e-12):
            raise AssertionError("D004 routed delta accuracy does not reconcile exactly.")
        decomposition_rows.extend([
            {"fold": fold.fold, "region": "routed", "n_rows": int(routed.sum()), **routed_counts, **_fractions(routed_counts, int(routed.sum())), "gain_count": gain, "cost_count": cost, "unchanged_error_count": unchanged_error, "net_correct_gain": net, "delta_accuracy": delta_accuracy, "gain_cost_ratio": gain / cost if cost else np.nan},
            {"fold": fold.fold, "region": "non_routed_neutral", "n_rows": int(non_routed.sum()), **neutral_counts, **_fractions(neutral_counts, int(non_routed.sum())), "gain_count": np.nan, "cost_count": np.nan, "unchanged_error_count": np.nan, "net_correct_gain": np.nan, "delta_accuracy": np.nan, "gain_cost_ratio": np.nan},
        ])

        margin = p_d - priors.threshold
        cumulative_net = 0
        for label, member in margin_region_masks(margin):
            n_rows = int(member.sum())
            counts = _class_counts(truth, member)
            regional_candidate = float((candidate[member] == truth[member]).mean()) if n_rows else np.nan
            regional_baseline = float((baseline[member] == truth[member]).mean()) if n_rows else np.nan
            predicted_values = np.unique(candidate[member])
            predicted_class = int(predicted_values[0]) if len(predicted_values) == 1 else "mixed"
            margin_rows.append({
                "fold": fold.fold, "margin_region": label, "n_rows": n_rows,
                "fraction_validation": n_rows / len(truth), "mean_p_d": float(p_d[member].mean()) if n_rows else np.nan,
                **counts, **_fractions(counts, n_rows),
                "empirical_directional_frequency": (counts["n_true_minus"] + counts["n_true_plus"]) / n_rows if n_rows else np.nan,
                "predicted_class": predicted_class, "candidate_accuracy": regional_candidate,
                "baseline_accuracy": regional_baseline,
                "regional_delta_accuracy": regional_candidate - regional_baseline if n_rows else np.nan,
            })
            if label in POSITIVE_MARGIN_LABELS:
                region_net = counts["n_true_minus"] - counts["n_true_zero"]
                cumulative_net += region_net
                positive_rows.append({"fold": fold.fold, "margin_region": label, "n_rows": n_rows, "n_true_minus": counts["n_true_minus"], "n_true_zero": counts["n_true_zero"], "n_true_plus": counts["n_true_plus"], "net_correct_contribution": region_net, "cumulative_net_from_boundary": cumulative_net})

        fold_rows.append({
            "fold": fold.fold, "n_validation": len(truth), "n_routed": int(routed.sum()), "n_non_routed": int(non_routed.sum()),
            "n_routed_true_minus": gain, "n_routed_true_zero": cost, "n_routed_true_plus": unchanged_error,
            "gain_cost_ratio": gain / cost if cost else np.nan, "net_correct_gain": net,
            "delta_accuracy": delta_accuracy, "candidate_accuracy": candidate_metrics["accuracy"], "baseline_accuracy": baseline_metrics["accuracy"],
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(RESULTS_DIR / "D004_post_result_fold_decomposition.csv", index=False)
    pd.DataFrame(decomposition_rows).to_csv(RESULTS_DIR / "D004_post_result_routing_composition.csv", index=False)
    pd.DataFrame(margin_rows).to_csv(RESULTS_DIR / "D004_post_result_margin_regions.csv", index=False)
    pd.DataFrame(positive_rows).to_csv(RESULTS_DIR / "D004_post_result_positive_margin_contribution.csv", index=False)
    (RESULTS_DIR / "D004_post_result_metadata.json").write_text(json.dumps({"diagnostic": "post_D004_descriptive", "reproduced_frozen_predictions": True, "alternative_threshold_tested": False, "calibration_method_fitted": False, "sign_model_fitted": False, "days": "0-352", "equity_partition": "E_dev", "competition_test_accessed": False}, indent=2), encoding="utf-8")
    print("D004 post-result diagnostic completed with exact frozen-result reconciliation.")


if __name__ == "__main__":
    main()
