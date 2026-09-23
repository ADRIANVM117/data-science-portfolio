"""Frozen I008A utilities; data-loading free and inert until its runner is authorized."""

from __future__ import annotations

import hashlib
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from discovery.d007_raw_path_masks_ternary_xgboost import ENCODED_CLASS_ORDER, ternary_probabilities
from discovery.d008_raw_path_beyond_availability_intensity import C_COLUMNS, P_COLUMNS, assert_identical_d008_rows, validate_expected_fold_count
from src.exp000_validation import CLASS_ORDER


RECONSTRUCTION_LOG_LOSS_ATOL = 1e-8
CONSERVATION_ATOL = 1e-10
ARTIFACT_NAMES = (
    "I008A_reproduction_gate.csv", "I008A_probability_movement.csv",
    "I008A_realized_class_gain.csv", "I008A_prediction_transitions.csv",
    "I008A_metadata.json",
)
PROBABILITY_COLUMNS = ("p_minus", "p_zero", "p_plus")


def ordered_id_fingerprint(ids: pd.Series) -> str:
    """Return a deterministic, order-sensitive fingerprint for a nonempty ID sequence."""
    if ids.empty or ids.isna().any() or ids.duplicated().any():
        raise AssertionError("I008A ordered-ID fingerprint requires unique, nonmissing IDs.")
    payload = "\n".join(map(str, ids.to_list())).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def artifact_state(results_dir) -> tuple[str, list[str]]:
    """Classify lifecycle without changing any artifact."""
    existing = [name for name in ARTIFACT_NAMES if (results_dir / name).exists()]
    if not existing:
        return "fresh", existing
    if set(existing) == set(ARTIFACT_NAMES):
        return "complete", existing
    return "incomplete", existing


def assert_fresh_artifacts(results_dir) -> None:
    """Refuse partial failed runs and overwrite of completed interpretation outputs."""
    state, existing = artifact_state(results_dir)
    if state == "incomplete":
        raise FileExistsError(f"I008A found incomplete artifacts; explicit authorized cleanup is required: {existing}")
    if state == "complete":
        raise FileExistsError(f"I008A refuses to overwrite completed artifacts: {existing}")


def persisted_d008_log_losses(metrics: pd.DataFrame) -> dict[tuple[int, str], float]:
    """Extract the six exact D008 C/P Validation log-loss references."""
    required = {"fold", "representation", "multiclass_log_loss"}
    if not required.issubset(metrics.columns):
        raise AssertionError("I008A requires D008 metrics with fold, representation, and multiclass_log_loss.")
    rows = metrics.loc[metrics["representation"].isin(("C", "P")), ["fold", "representation", "multiclass_log_loss"]]
    if len(rows) != 6 or rows.duplicated(["fold", "representation"]).any() or set(rows["fold"]) != {1, 2, 3}:
        raise AssertionError("I008A requires exactly one persisted C/P loss for each D008 fold.")
    result = {(int(row.fold), str(row.representation)): float(row.multiclass_log_loss) for row in rows.itertuples(index=False)}
    if set(result) != {(fold, arm) for fold in (1, 2, 3) for arm in ("C", "P")} or not np.isfinite(list(result.values())).all():
        raise AssertionError("I008A persisted D008 losses are incomplete or non-finite.")
    return result


def validate_fit_preprocessing(fold: int, structure: pd.DataFrame, transformer, persisted: pd.DataFrame) -> None:
    """Require fold-only intensity parameters to reproduce persisted D008 values."""
    rows = persisted.loc[persisted["fold"].eq(fold)]
    if len(rows) != 1:
        raise AssertionError("I008A requires one persisted D008 preprocessing row per fold.")
    row = rows.iloc[0]
    observed = {
        "fit_intensity_median": float(transformer.fit_intensity_median_),
        "scaler_mean": float(transformer.scaler_.mean_[0]),
        "scaler_scale": float(transformer.scaler_.scale_[0]),
        "n_fit_defined_intensity": int(structure["N_obs_w"].gt(0).sum()),
        "n_fit_undefined_intensity": int(structure["N_obs_w"].eq(0).sum()),
    }
    for name, value in observed.items():
        expected = row[name]
        passed = value == int(expected) if name.startswith("n_") else np.isclose(value, float(expected), rtol=0.0, atol=1e-12)
        if not passed:
            raise AssertionError(f"I008A D008 fit-only preprocessing mismatch for fold {fold}, {name}.")


def probabilities_with_metadata(model, matrix: pd.DataFrame) -> tuple[pd.DataFrame, str, float]:
    """Use frozen D007 validation once while retaining raw dtype/tolerance metadata."""
    raw = np.asarray(model.predict_proba(matrix))
    if tuple(model.classes_) != ENCODED_CLASS_ORDER:
        raise AssertionError("I008A model class order differs from frozen encoded [0,1,2].")
    if not np.issubdtype(raw.dtype, np.floating):
        raise AssertionError("I008A probability output must have floating dtype.")

    class _Output:
        def __init__(self, values, classes):
            self._values, self.classes_ = values, classes
        def predict_proba(self, _matrix):
            return self._values

    probabilities = ternary_probabilities(_Output(raw, model.classes_), matrix)
    return probabilities, str(raw.dtype), float(max(1e-12, 2 * np.finfo(raw.dtype).eps))


def reconstruction_log_loss(target: pd.Series, probabilities: pd.DataFrame) -> float:
    """Compute only the frozen D008 reproduction-gate loss."""
    if not target.index.equals(probabilities.index) or set(target.unique()) != set(CLASS_ORDER):
        raise AssertionError("I008A requires aligned complete ternary Validation labels.")
    return float(log_loss(target.to_numpy(dtype=int), probabilities.to_numpy(), labels=list(CLASS_ORDER)))


def gate_record(*, fold: int, arm: str, target: pd.Series, features: pd.DataFrame, matrices: Mapping[str, pd.DataFrame], probabilities: pd.DataFrame, raw_probability_dtype: str, row_sum_tolerance: float, reconstructed_loss: float, persisted_loss: float) -> dict[str, object]:
    """Validate one reconstructed arm against frozen D008 evidence and make its audit record."""
    if arm not in ("C", "P") or arm not in matrices:
        raise AssertionError("I008A gate received an unknown D008 arm.")
    validate_expected_fold_count(fold, "validation", len(target))
    assert_identical_d008_rows(matrices, features, target)
    if tuple(matrices["C"].columns) != C_COLUMNS or tuple(matrices["P"].columns) != P_COLUMNS:
        raise AssertionError("I008A reconstructed D008 schemas differ from the frozen contract.")
    if not probabilities.index.equals(target.index) or tuple(probabilities.columns) != PROBABILITY_COLUMNS:
        raise AssertionError("I008A probability rows or label mapping are not aligned.")
    values = probabilities.to_numpy(dtype=float)
    if values.shape != (len(target), 3) or not np.isfinite(values).all() or (values < 0.0).any():
        raise AssertionError("I008A probabilities are not finite, nonnegative, or correctly shaped.")
    difference = float(reconstructed_loss - persisted_loss)
    passed = bool(np.isclose(reconstructed_loss, persisted_loss, rtol=0.0, atol=RECONSTRUCTION_LOG_LOSS_ATOL))
    if not passed:
        raise AssertionError(f"I008A D008 reconstruction loss mismatch in fold {fold}, arm {arm}: {difference:.16g}")
    return {
        "fold": fold, "arm": arm, "n_validation_rows": len(target),
        "ordered_id_fingerprint": ordered_id_fingerprint(features["ID"]),
        "model_encoded_classes": "[0,1,2]", "label_order": "[-1,0,+1]",
        "probability_dtype": raw_probability_dtype, "row_sum_tolerance": row_sum_tolerance,
        "persisted_log_loss": float(persisted_loss), "reconstructed_log_loss": float(reconstructed_loss),
        "absolute_loss_difference": abs(difference), "loss_tolerance_atol": RECONSTRUCTION_LOG_LOSS_ATOL,
        "passed": True,
    }


def probability_changes(control: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    """Compute only frozen p_D/q_plus changes, preserving undefined allocations."""
    if not control.index.equals(candidate.index) or tuple(control.columns) != PROBABILITY_COLUMNS or tuple(candidate.columns) != PROBABILITY_COLUMNS:
        raise AssertionError("I008A C/P probabilities must be identical in row identity and label order.")
    c, p = control.to_numpy(dtype=float), candidate.to_numpy(dtype=float)
    if not np.isfinite(c).all() or not np.isfinite(p).all() or (c < 0).any() or (p < 0).any():
        raise AssertionError("I008A probability decomposition requires finite nonnegative probabilities.")
    c_d, p_d = c[:, 0] + c[:, 2], p[:, 0] + p[:, 2]
    c_q = np.full(len(c), np.nan); p_q = np.full(len(p), np.nan)
    c_defined, p_defined = c_d > 0.0, p_d > 0.0
    c_q[c_defined] = c[c_defined, 2] / c_d[c_defined]
    p_q[p_defined] = p[p_defined, 2] / p_d[p_defined]
    both = c_defined & p_defined
    delta_q = np.full(len(c), np.nan); delta_q[both] = p_q[both] - c_q[both]
    return pd.DataFrame({"delta_p_D": p_d - c_d, "delta_q_plus": delta_q, "delta_q_defined": both}, index=control.index)


def per_row_log_loss(target: pd.Series, probabilities: pd.DataFrame) -> pd.Series:
    """Mirror sklearn log_loss clipping, without renormalizing probability rows."""
    if not target.index.equals(probabilities.index) or set(target.unique()) != set(CLASS_ORDER):
        raise AssertionError("I008A per-row loss requires aligned complete ternary labels.")
    values = probabilities.to_numpy(dtype=float)
    if values.shape != (len(target), 3) or not np.isfinite(values).all() or (values < 0.0).any():
        raise AssertionError("I008A per-row loss requires valid frozen probabilities.")
    eps = np.finfo(values.dtype).eps
    selected = values[np.arange(len(values)), np.searchsorted(np.asarray(CLASS_ORDER), target.to_numpy(dtype=int))]
    return pd.Series(-np.log(np.clip(selected, eps, 1.0 - eps)), index=target.index, name="per_row_log_loss")


def realized_gain(target: pd.Series, control: pd.DataFrame, candidate: pd.DataFrame) -> pd.Series:
    """Return g=loss(C)-loss(P), positive when P improves effective true-class probability."""
    return (per_row_log_loss(target, control) - per_row_log_loss(target, candidate)).rename("g_i")


def assert_gain_conservation(gain: pd.Series, *, persisted_delta: float, reconstructed_delta: float) -> None:
    """Fail closed unless per-row gains recover both frozen aggregate loss contrasts."""
    if not np.isfinite(gain.to_numpy(dtype=float)).all():
        raise AssertionError("I008A realized gains must be finite.")
    mean_gain = float(gain.mean())
    for value, context in ((persisted_delta, "persisted"), (reconstructed_delta, "reconstructed")):
        if not np.isclose(mean_gain, value, rtol=0.0, atol=CONSERVATION_ATOL):
            raise AssertionError(f"I008A realized-gain conservation fails against {context} D008 loss delta.")


def level1_summary(fold: int, changes: pd.DataFrame) -> dict[str, object]:
    """Produce exactly the frozen, target-free probability-movement summaries."""
    if tuple(changes.columns) != ("delta_p_D", "delta_q_plus", "delta_q_defined"):
        raise AssertionError("I008A Level 1 change schema is frozen.")
    dp, dq = changes["delta_p_D"], changes.loc[changes["delta_q_defined"].astype(bool), "delta_q_plus"]
    if not np.isfinite(dp.to_numpy(dtype=float)).all() or not np.isfinite(dq.to_numpy(dtype=float)).all():
        raise AssertionError("I008A Level 1 requires finite defined changes.")
    return {"fold": fold, "delta_p_D_mean": float(dp.mean()), "delta_p_D_median": float(dp.median()), "delta_p_D_mean_abs": float(dp.abs().mean()), "delta_p_D_median_abs": float(dp.abs().median()), "delta_q_plus_n_defined": int(len(dq)), "delta_q_plus_mean": float(dq.mean()) if len(dq) else np.nan, "delta_q_plus_median": float(dq.median()) if len(dq) else np.nan, "delta_q_plus_mean_abs": float(dq.abs().mean()) if len(dq) else np.nan, "delta_q_plus_median_abs": float(dq.abs().median()) if len(dq) else np.nan}


def level2_summary(fold: int, target: pd.Series, changes: pd.DataFrame, gain: pd.Series) -> pd.DataFrame:
    """Produce the only frozen true-class decomposition: exactly three classes."""
    if not target.index.equals(changes.index) or not target.index.equals(gain.index) or set(target.unique()) != set(CLASS_ORDER):
        raise AssertionError("I008A Level 2 requires aligned complete ternary rows.")
    rows: list[dict[str, object]] = []
    for label in CLASS_ORDER:
        selected = target.eq(label); q = changes.loc[selected & changes["delta_q_defined"].astype(bool), "delta_q_plus"]
        g = gain.loc[selected]
        rows.append({"fold": fold, "true_class": label, "n_rows": int(selected.sum()), "g_mean": float(g.mean()), "g_median": float(g.median()), "g_fraction_positive": float((g > 0.0).mean()), "delta_p_D_mean": float(changes.loc[selected, "delta_p_D"].mean()), "delta_q_plus_n_defined": int(len(q)), "delta_q_plus_mean": float(q.mean()) if len(q) else np.nan})
    return pd.DataFrame(rows)


def prediction_transitions(fold: int, target: pd.Series, control: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    """Persist exhaustive fixed C→P argmax transitions, unconditional and by true class."""
    if not target.index.equals(control.index) or not target.index.equals(candidate.index):
        raise AssertionError("I008A transitions require aligned target and C/P probabilities.")
    c_pred = np.asarray(CLASS_ORDER)[control.to_numpy(dtype=float).argmax(axis=1)]
    p_pred = np.asarray(CLASS_ORDER)[candidate.to_numpy(dtype=float).argmax(axis=1)]
    rows: list[dict[str, object]] = []
    for kind, subset, true_class in [("unconditional", np.ones(len(target), dtype=bool), np.nan), *[("by_true_class", target.to_numpy(dtype=int) == label, label) for label in CLASS_ORDER]]:
        for c_label in CLASS_ORDER:
            for p_label in CLASS_ORDER:
                rows.append({"fold": fold, "transition_type": kind, "true_class": true_class, "C_prediction": c_label, "P_prediction": p_label, "count": int((subset & (c_pred == c_label) & (p_pred == p_label)).sum())})
    return pd.DataFrame(rows)


__all__ = ["ARTIFACT_NAMES", "CONSERVATION_ATOL", "PROBABILITY_COLUMNS", "RECONSTRUCTION_LOG_LOSS_ATOL", "artifact_state", "assert_fresh_artifacts", "assert_gain_conservation", "gate_record", "level1_summary", "level2_summary", "ordered_id_fingerprint", "per_row_log_loss", "persisted_d008_log_losses", "prediction_transitions", "probabilities_with_metadata", "probability_changes", "reconstruction_log_loss", "realized_gain", "validate_fit_preprocessing"]
