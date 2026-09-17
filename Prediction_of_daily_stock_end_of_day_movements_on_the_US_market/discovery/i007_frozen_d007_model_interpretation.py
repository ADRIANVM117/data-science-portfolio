"""Frozen I007A utilities for descriptive main-effect SHAP of D007 R+M.

The module is deliberately data-loading free.  A separately authorized runner
may reconstruct D007 first; no function here changes its learner, features, or
scientific metrics.
"""

from __future__ import annotations

from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

from discovery.d007_raw_path_masks_ternary_xgboost import (
    ENCODED_CLASS_ORDER,
    MASK_COLUMNS,
    RM_COLUMNS,
    RETURN_COLUMNS,
    ternary_probabilities,
)
from src.exp000_validation import CLASS_ORDER


RECONSTRUCTION_LOG_LOSS_ATOL = 1e-8
SHAP_ADDITIVITY_RTOL = 1e-5
SHAP_ADDITIVITY_ATOL = 1e-5
GLOBAL_SHAP_SAMPLE_MAX = 2_000
GLOBAL_SHAP_SEED = 20260908
N_OUTPUTS = 3
TOP_K = 10
OUTPUT_LABELS = (-1, 0, 1)


def expected_d007_r_plus_m_log_loss(metrics: pd.DataFrame) -> dict[int, float]:
    """Extract the exact persisted D007 R+M log-loss references."""
    required = {"fold", "representation", "multiclass_log_loss"}
    if not required.issubset(metrics.columns):
        raise AssertionError("I007A D007 metrics lack required reconstruction fields.")
    rows = metrics.loc[metrics["representation"].eq("R+M"), ["fold", "multiclass_log_loss"]]
    if len(rows) != 3 or rows["fold"].duplicated().any() or set(rows["fold"]) != {1, 2, 3}:
        raise AssertionError("I007A requires exactly three persisted D007 R+M fold log losses.")
    values = {int(row.fold): float(row.multiclass_log_loss) for row in rows.itertuples(index=False)}
    if not np.isfinite(list(values.values())).all():
        raise AssertionError("I007A persisted D007 R+M log losses must be finite.")
    return values


def reconstruction_log_loss(target: pd.Series, model: XGBClassifier, matrix: pd.DataFrame) -> float:
    """Compute only the frozen reconstruction gate statistic."""
    probabilities = ternary_probabilities(model, matrix)
    if not target.index.equals(probabilities.index) or set(target.unique()) != set(CLASS_ORDER):
        raise AssertionError("I007A reconstruction requires aligned complete ternary validation labels.")
    return float(log_loss(target.to_numpy(dtype=int), probabilities.to_numpy(), labels=list(CLASS_ORDER)))


def assert_reconstruction_matches(*, fold: int, reconstructed: float, expected: float) -> dict[str, float | int | bool]:
    """Fail closed before SHAP if an exact D007 model cannot be reconstructed."""
    difference = float(reconstructed - expected)
    passed = bool(np.isclose(reconstructed, expected, rtol=0.0, atol=RECONSTRUCTION_LOG_LOSS_ATOL))
    record: dict[str, float | int | bool] = {
        "fold": fold,
        "persisted_r_plus_m_log_loss": float(expected),
        "reconstructed_r_plus_m_log_loss": float(reconstructed),
        "absolute_difference": abs(difference),
        "tolerance_atol": RECONSTRUCTION_LOG_LOSS_ATOL,
        "passed": passed,
    }
    if not passed:
        raise AssertionError(f"I007A D007 reconstruction mismatch in fold {fold}: {difference:.16g}")
    return record


def deterministic_validation_positions(n_rows: int, *, fold: int) -> np.ndarray:
    """Select I007A Validation positions without labels, then restore source order."""
    if n_rows < 1 or fold not in (1, 2, 3):
        raise AssertionError("I007A sampling requires a nonempty frozen Discovery fold.")
    count = min(GLOBAL_SHAP_SAMPLE_MAX, n_rows)
    positions = np.random.default_rng(GLOBAL_SHAP_SEED + fold).choice(n_rows, size=count, replace=False)
    result = np.sort(positions.astype(np.int64, copy=False))
    if len(result) != count or len(np.unique(result)) != count:
        raise AssertionError("I007A deterministic Validation sample is not unique.")
    return result


def sampled_validation_matrix(matrix: pd.DataFrame, features: pd.DataFrame, *, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return identically ordered R+M rows and target-blind sample manifest."""
    if tuple(matrix.columns) != RM_COLUMNS or not matrix.index.equals(features.index):
        raise AssertionError("I007A sampled R+M matrix must preserve source index and frozen schema.")
    positions = deterministic_validation_positions(len(matrix), fold=fold)
    sample = matrix.iloc[positions].copy()
    manifest = pd.DataFrame({
        "fold": fold,
        "source_row_position": positions,
        "source_index": matrix.index.to_numpy()[positions],
        "ID": features.iloc[positions]["ID"].to_numpy(),
    })
    if not sample.index.equals(pd.Index(manifest["source_index"].to_numpy())):
        raise AssertionError("I007A sample manifest changed source row order.")
    return sample, manifest


def make_i007_tree_explainer(model: XGBClassifier) -> shap.TreeExplainer:
    """Build exactly the frozen TreeSHAP main-effect configuration."""
    if tuple(model.classes_) != ENCODED_CLASS_ORDER:
        raise AssertionError("I007A requires reconstructed D007 encoded classes [0, 1, 2].")
    return shap.TreeExplainer(
        model,
        data=None,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
        feature_names=list(RM_COLUMNS),
    )


def explain_and_validate_additivity(model: XGBClassifier, matrix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute main-effect SHAP and fail closed on shape or raw-margin mismatch."""
    if tuple(matrix.columns) != RM_COLUMNS:
        raise AssertionError("I007A SHAP requires the frozen 106-column R+M schema.")
    explanation = make_i007_tree_explainer(model)(matrix, check_additivity=True)
    values = np.asarray(explanation.values, dtype=float)
    base_values = np.asarray(explanation.base_values, dtype=float)
    margins = np.asarray(model.predict(matrix, output_margin=True), dtype=float)
    expected = (len(matrix), len(RM_COLUMNS), N_OUTPUTS)
    if values.shape != expected or base_values.shape != (len(matrix), N_OUTPUTS) or margins.shape != (len(matrix), N_OUTPUTS):
        raise AssertionError("I007A SHAP raw values, base values, or margins have an unexpected shape.")
    if not (np.isfinite(values).all() and np.isfinite(base_values).all() and np.isfinite(margins).all()):
        raise AssertionError("I007A SHAP raw values, base values, and margins must be finite.")
    reconstructed = base_values + values.sum(axis=1)
    if not np.allclose(reconstructed, margins, rtol=SHAP_ADDITIVITY_RTOL, atol=SHAP_ADDITIVITY_ATOL):
        maximum = float(np.abs(reconstructed - margins).max())
        raise AssertionError(f"I007A SHAP raw-margin additivity failed: maximum difference {maximum:.16g}")
    return values, base_values, margins


def additivity_record(*, fold: int, values: np.ndarray, base_values: np.ndarray, margins: np.ndarray) -> dict[str, float | int | bool]:
    reconstructed = np.asarray(base_values, dtype=float) + np.asarray(values, dtype=float).sum(axis=1)
    difference = np.abs(reconstructed - np.asarray(margins, dtype=float))
    return {
        "fold": fold, "n_rows": int(len(values)), "n_features": int(values.shape[1]), "n_outputs": int(values.shape[2]),
        "max_abs_difference": float(difference.max()), "rtol": SHAP_ADDITIVITY_RTOL,
        "atol": SHAP_ADDITIVITY_ATOL, "passed": bool(np.allclose(reconstructed, margins, rtol=SHAP_ADDITIVITY_RTOL, atol=SHAP_ADDITIVITY_ATOL)),
    }


def allocation_by_fold(values: np.ndarray, *, fold: int) -> pd.DataFrame:
    """Mean absolute SHAP and raw/mask allocation shares for one fold."""
    if values.shape[1:] != (len(RM_COLUMNS), N_OUTPUTS) or not np.isfinite(values).all():
        raise AssertionError("I007A allocation requires finite (n, 106, 3) SHAP values.")
    mean_abs = np.abs(values).mean(axis=0)
    rows: list[dict[str, object]] = []
    for output_index, output_label in enumerate(OUTPUT_LABELS):
        total = float(mean_abs[:, output_index].sum())
        if total <= 0.0 or not np.isfinite(total):
            raise AssertionError("I007A cannot allocate a zero or non-finite total attribution.")
        raw_total = float(mean_abs[:len(RETURN_COLUMNS), output_index].sum())
        mask_total = float(mean_abs[len(RETURN_COLUMNS):, output_index].sum())
        for feature_index, feature in enumerate(RM_COLUMNS):
            rows.append({
                "fold": fold, "output_index": output_index, "output_label": output_label,
                "feature": feature, "feature_group": "raw_return" if feature_index < len(RETURN_COLUMNS) else "mask",
                "mean_abs_shap": float(mean_abs[feature_index, output_index]),
                "raw_allocation_share": raw_total / total, "mask_allocation_share": mask_total / total,
            })
    return pd.DataFrame(rows)


def rank_stability(allocation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build per-output ranks, rank correlations, and deterministic Top-10 records."""
    needed = {"fold", "output_index", "output_label", "feature", "mean_abs_shap"}
    if not needed.issubset(allocation.columns) or set(allocation["fold"]) != {1, 2, 3}:
        raise AssertionError("I007A rank stability requires all three fold allocations.")
    ranked = allocation.copy()
    feature_order = {feature: position for position, feature in enumerate(RM_COLUMNS)}
    ranked["_feature_order"] = ranked["feature"].map(feature_order)
    ranked = ranked.sort_values(["fold", "output_index", "mean_abs_shap", "_feature_order"], ascending=[True, True, False, True], kind="stable")
    ranked["rank_within_fold"] = ranked.groupby(["fold", "output_index"]).cumcount().add(1).astype(float)
    ranked = ranked.drop(columns="_feature_order")
    summary = ranked.groupby(["output_index", "output_label", "feature"], as_index=False).agg(
        mean_abs_shap_macro=("mean_abs_shap", "mean"), median_rank=("rank_within_fold", "median"),
        min_rank=("rank_within_fold", "min"), max_rank=("rank_within_fold", "max"),
    )
    correlations: list[dict[str, object]] = []
    top_records: list[dict[str, object]] = []
    for output_index, output_label in enumerate(OUTPUT_LABELS):
        subset = ranked.loc[ranked.output_index.eq(output_index)]
        pivot = subset.pivot(index="feature", columns="fold", values="rank_within_fold").reindex(index=RM_COLUMNS, columns=[1, 2, 3])
        for left, right in combinations((1, 2, 3), 2):
            correlations.append({"output_index": output_index, "output_label": output_label, "fold_left": left, "fold_right": right, "spearman_rank_correlation": float(pivot[left].corr(pivot[right], method="spearman"))})
        top_by_fold = {fold: set(subset.loc[subset.fold.eq(fold) & subset.rank_within_fold.le(TOP_K), "feature"]) for fold in (1, 2, 3)}
        membership = {feature: sum(feature in top_by_fold[fold] for fold in (1, 2, 3)) for feature in RM_COLUMNS}
        for left, right in combinations((1, 2, 3), 2):
            intersection = top_by_fold[left] & top_by_fold[right]
            union = top_by_fold[left] | top_by_fold[right]
            top_records.append({"record_type": "pairwise_overlap", "output_index": output_index, "output_label": output_label, "fold_left": left, "fold_right": right, "feature": "", "top10_membership": np.nan, "intersection_count": len(intersection), "jaccard_overlap": len(intersection) / len(union)})
        intersection_all = set.intersection(*top_by_fold.values())
        for fold in (1, 2, 3):
            for feature in sorted(top_by_fold[fold], key=lambda name: (float(subset.loc[(subset.fold.eq(fold)) & (subset.feature.eq(name)), "rank_within_fold"].iloc[0]), name)):
                top_records.append({"record_type": "top10_membership", "output_index": output_index, "output_label": output_label, "fold_left": fold, "fold_right": fold, "feature": feature, "top10_membership": membership[feature], "intersection_count": len(intersection_all), "jaccard_overlap": np.nan})
        summary.loc[summary.output_index.eq(output_index), "top10_membership_count"] = summary.loc[summary.output_index.eq(output_index), "feature"].map(membership)
        summary.loc[summary.output_index.eq(output_index), "stable_top10"] = summary.loc[summary.output_index.eq(output_index), "feature"].isin(intersection_all)
    ranked = ranked.merge(
        summary[["output_index", "output_label", "feature", "top10_membership_count", "stable_top10"]],
        on=["output_index", "output_label", "feature"], how="left", validate="many_to_one",
    )
    if ranked[["top10_membership_count", "stable_top10"]].isna().any().any():
        raise AssertionError("I007A Top-10 stability did not align to fold rankings.")
    return ranked.sort_values(["output_index", "fold", "rank_within_fold", "feature"]), pd.DataFrame(correlations), pd.DataFrame(top_records)


def stable_raw_features(rank_summary: pd.DataFrame, *, output_label: int) -> tuple[str, ...]:
    """Return only raw features stable Top-10 for the matching tail output."""
    output_index = OUTPUT_LABELS.index(output_label)
    selected = rank_summary.loc[(rank_summary.output_index.eq(output_index)) & rank_summary.stable_top10.astype(bool), "feature"]
    return tuple(feature for feature in selected if feature in RETURN_COLUMNS)


def orientation_diagnostics(sample_matrix: pd.DataFrame, values: np.ndarray, manifest: pd.DataFrame, *, fold: int, output_label: int, features: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Observed-value-only SHAP orientation summaries, without directional labels."""
    if not sample_matrix.index.equals(pd.Index(manifest["source_index"].to_numpy())) or tuple(sample_matrix.columns) != RM_COLUMNS:
        raise AssertionError("I007A orientation requires the unchanged sampled R+M order.")
    output_index = OUTPUT_LABELS.index(output_label)
    if values.shape != (len(sample_matrix), len(RM_COLUMNS), N_OUTPUTS):
        raise AssertionError("I007A orientation SHAP values have an unexpected shape.")
    correlations: list[dict[str, object]] = []
    bins: list[dict[str, object]] = []
    for feature in features:
        if feature not in RETURN_COLUMNS:
            raise AssertionError("I007A orientation diagnostics permit raw returns only.")
        index = RM_COLUMNS.index(feature)
        raw = sample_matrix[feature].to_numpy(dtype=float)
        observed = ~np.isnan(raw)
        if not np.isfinite(raw[observed]).all():
            raise AssertionError("I007A observed orientation returns must be finite.")
        x, y = raw[observed], values[observed, index, output_index]
        status = "defined" if len(x) >= 2 and np.unique(x).size >= 2 else "undefined_constant"
        spearman = float(pd.Series(x).corr(pd.Series(y), method="spearman")) if status == "defined" else np.nan
        correlations.append({"fold": fold, "output_label": output_label, "feature": feature, "n_observed": len(x), "spearman_value_vs_same_output_shap": spearman, "status": status})
        if len(x) == 0:
            continue
        order = np.lexsort((manifest.loc[observed, "source_row_position"].to_numpy(), x))
        for bin_number, selected in enumerate(np.array_split(order, min(10, len(order))), start=1):
            bins.append({"fold": fold, "output_label": output_label, "feature": feature, "bin": bin_number, "n_bins": min(10, len(order)), "count": len(selected), "raw_min": float(x[selected].min()), "raw_max": float(x[selected].max()), "raw_mean": float(x[selected].mean()), "mean_same_output_shap": float(y[selected].mean())})
    return pd.DataFrame(correlations), pd.DataFrame(bins)
