"""Frozen D006 ternary tail interaction utilities; no data loading or execution."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, recall_score, roc_auc_score

from discovery.d003_incremental_cross_sectional_relative_path import DISCOVERY_FOLDS
from src.exp000_validation import CLASS_ORDER
from src.exp001_missingness import LOGISTIC_REGRESSION_PARAMS, RETURN_COLUMNS
from src.exp002_neutral_directional import build_missing_ratio
from src.exp008_recent_movement_intensity import RecentIntensityTransformer, build_recent_window_structure


D006_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "C0": ("missing_ratio", "q", "I_recent_z"),
    "C1": ("missing_ratio", "q", "I_recent_z", "O"),
    "C2": ("missing_ratio", "q", "I_recent_z", "O", "I_recent_z_x_O"),
}
ORIENTATION_COLUMNS = ("O", "I_recent_z_x_O")


def build_d006_structure(features: pd.DataFrame) -> pd.DataFrame:
    """Build target-free recent energy allocation while reusing EXP_008 I/q semantics."""
    base = build_recent_window_structure(features)
    window_columns = tuple(f"r{position}" for position in range(41, 53))
    values = features.loc[:, window_columns].to_numpy(dtype=float)
    observed = ~np.isnan(values)
    if not np.isfinite(values[observed]).all():
        raise AssertionError("D006 observed non-NaN W returns must be finite.")
    positive = observed & (values > 0.0)
    negative = observed & (values < 0.0)
    with np.errstate(over="raise", invalid="raise"):
        e_plus = np.square(np.where(positive, values, 0.0)).sum(axis=1)
        e_minus = np.square(np.where(negative, values, 0.0)).sum(axis=1)
    e_total = e_plus + e_minus
    if not np.isfinite(e_total).all() or (e_total < 0.0).any():
        raise AssertionError("D006 recent energy must be finite and non-negative.")
    orientation = np.zeros(len(features), dtype=float)
    defined = e_total > 0.0
    orientation[defined] = (e_plus[defined] - e_minus[defined]) / e_total[defined]
    structure = pd.concat(
        [base, pd.DataFrame({"E_plus": e_plus, "E_minus": e_minus, "E_total": e_total, "O": orientation}, index=features.index)],
        axis=1,
    )
    validate_d006_structure(structure, expected_index=features.index)
    return structure


def validate_d006_structure(structure: pd.DataFrame, *, expected_index: pd.Index | None = None) -> None:
    expected = ["N_obs_w", "q", "raw_I_recent", "E_plus", "E_minus", "E_total", "O"]
    if list(structure.columns) != expected:
        raise AssertionError("D006 structure has an unexpected schema.")
    if expected_index is not None and not structure.index.equals(expected_index):
        raise AssertionError("D006 structure changed row identity or order.")
    values = structure.loc[:, ["E_plus", "E_minus", "E_total", "O"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (structure["E_plus"] < 0).any() or (structure["E_minus"] < 0).any():
        raise AssertionError("D006 energy and orientation values must be finite.")
    if not np.allclose(structure["E_total"], structure["E_plus"] + structure["E_minus"], rtol=0.0, atol=1e-12):
        raise AssertionError("D006 E_total must equal E_plus plus E_minus.")
    if ((structure["O"] < -1.0) | (structure["O"] > 1.0)).any():
        raise AssertionError("D006 O must lie in [-1, 1].")
    zero_energy = structure["E_total"].eq(0.0)
    if not structure.loc[zero_energy, "O"].eq(0.0).all():
        raise AssertionError("D006 O must equal zero when E_total is zero.")


def build_d006_representations(features: pd.DataFrame, target: pd.Series, transformer: RecentIntensityTransformer) -> dict[str, pd.DataFrame]:
    """Build C0/C1/C2 with identical row identity and frozen predictor order."""
    if not features.index.equals(target.index):
        raise AssertionError("D006 features and ternary target must align.")
    if set(target.unique()) != set(CLASS_ORDER):
        raise AssertionError("D006 requires all three ternary classes in each modeled subset.")
    structure = build_d006_structure(features)
    intensity_z = transformer.transform(structure["raw_I_recent"])
    common = pd.concat([build_missing_ratio(features), structure.loc[:, ["q"]], intensity_z], axis=1)
    matrices = {
        "C0": common,
        "C1": pd.concat([common, structure.loc[:, ["O"]]], axis=1),
        "C2": pd.concat([common, structure.loc[:, ["O"]], pd.DataFrame({"I_recent_z_x_O": intensity_z["I_recent_z"] * structure["O"]}, index=features.index)], axis=1),
    }
    for name, matrix in matrices.items():
        validate_d006_matrix(name, matrix, expected_index=features.index)
    if not matrices["C0"].equals(matrices["C1"].loc[:, D006_COLUMNS["C0"]]):
        raise AssertionError("D006 C1 must add only O to C0.")
    if not matrices["C1"].equals(matrices["C2"].loc[:, D006_COLUMNS["C1"]]):
        raise AssertionError("D006 C2 must add only the frozen interaction to C1.")
    return matrices


def validate_d006_matrix(name: str, matrix: pd.DataFrame, *, expected_index: pd.Index | None = None) -> None:
    if name not in D006_COLUMNS or tuple(matrix.columns) != D006_COLUMNS[name]:
        raise AssertionError("D006 predictor schema differs from its frozen C0/C1/C2 definition.")
    if expected_index is not None and not matrix.index.equals(expected_index):
        raise AssertionError("D006 predictor matrix changed row identity or order.")
    if not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise AssertionError("D006 predictors must be finite after fit-only preprocessing.")
    if ((matrix["missing_ratio"] < 0.0) | (matrix["missing_ratio"] > 1.0)).any() or not np.isin(matrix["q"], [0, 1]).all():
        raise AssertionError("D006 must preserve missing_ratio and q semantics.")
    if "O" in matrix and ((matrix["O"] < -1.0) | (matrix["O"] > 1.0)).any():
        raise AssertionError("D006 O must remain naturally bounded.")


def make_d006_multinomial_logistic_regression() -> LogisticRegression:
    """Construct the fixed classifier while avoiding deprecated multi_class."""
    classifier = LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)
    for key, expected in LOGISTIC_REGRESSION_PARAMS.items():
        if classifier.get_params()[key] != expected:
            raise AssertionError(f"D006 LogisticRegression parameter {key!r} is not frozen.")
    if classifier.get_params()["multi_class"] != "deprecated":
        raise AssertionError("D006 requires the installed deprecated-default multinomial selection behavior.")
    return classifier


def ternary_probabilities(classifier: LogisticRegression, matrix: pd.DataFrame) -> pd.DataFrame:
    if tuple(classifier.classes_) != CLASS_ORDER:
        raise AssertionError("D006 multinomial classifier classes must be [-1, 0, 1].")
    values = classifier.predict_proba(matrix)
    if not np.isfinite(values).all() or (values < 0.0).any() or not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise AssertionError("D006 multinomial probabilities must be coherent and finite.")
    return pd.DataFrame(values, index=matrix.index, columns=("p_minus", "p_zero", "p_plus"))


def evaluate_d006_probabilities(target: pd.Series, probabilities: pd.DataFrame) -> tuple[dict[str, float], list[list[int]]]:
    if not target.index.equals(probabilities.index) or set(target.unique()) != set(CLASS_ORDER):
        raise AssertionError("D006 target and probabilities must align and contain all ternary classes.")
    truth = target.to_numpy(dtype=int)
    prediction = np.asarray(CLASS_ORDER)[probabilities.to_numpy().argmax(axis=1)]
    recalls = recall_score(truth, prediction, labels=CLASS_ORDER, average=None, zero_division=0)
    minus_auc = float(roc_auc_score(target.eq(-1).astype(int), probabilities["p_minus"]))
    plus_auc = float(roc_auc_score(target.eq(1).astype(int), probabilities["p_plus"]))
    metrics = {
        "tail_auc_minus": minus_auc, "tail_auc_plus": plus_auc, "macro_tail_auc": (minus_auc + plus_auc) / 2.0,
        "multiclass_log_loss": float(log_loss(truth, probabilities.to_numpy(), labels=list(CLASS_ORDER))),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "recall_negative": float(recalls[0]), "recall_neutral": float(recalls[1]), "recall_positive": float(recalls[2]),
    }
    return metrics, confusion_matrix(truth, prediction, labels=CLASS_ORDER).astype(int).tolist()


def orientation_shift(classifier: LogisticRegression, name: str, matrix: pd.DataFrame) -> float:
    """Return the frozen Validation sign-reversal probability contrast mean."""
    validate_d006_matrix(name, matrix)
    if name not in {"C1", "C2"}:
        raise AssertionError("D006 orientation diagnostic applies only to C1 and C2.")
    reversed_matrix = matrix.copy()
    reversed_matrix["O"] = -reversed_matrix["O"]
    if name == "C2":
        reversed_matrix["I_recent_z_x_O"] = -reversed_matrix["I_recent_z_x_O"]
    original = ternary_probabilities(classifier, matrix)
    reversed_probabilities = ternary_probabilities(classifier, reversed_matrix)
    shift = (original["p_plus"] - original["p_minus"]) - (reversed_probabilities["p_plus"] - reversed_probabilities["p_minus"])
    if not np.isfinite(shift.to_numpy()).all():
        raise AssertionError("D006 orientation shifts must be finite.")
    return float(shift.mean())


def orientation_consistency(shifts: Sequence[float]) -> str:
    if len(shifts) != 3 or not np.isfinite(np.asarray(shifts, dtype=float)).all():
        raise AssertionError("D006 orientation requires exactly three finite fold shifts.")
    if all(value > 0.0 for value in shifts):
        return "continuation"
    if all(value < 0.0 for value in shifts):
        return "opposite"
    return "unstable"


def candidate_screen(aucs: Sequence[float], deltas: Sequence[float]) -> dict[str, float | bool | int]:
    if len(aucs) != 3 or len(deltas) != 3:
        raise AssertionError("D006 screens require exactly three Discovery folds.")
    auc_values, delta_values = np.asarray(aucs, dtype=float), np.asarray(deltas, dtype=float)
    if not np.isfinite(auc_values).all() or not np.isfinite(delta_values).all():
        raise AssertionError("D006 screen values must be finite.")
    return {
        "n_discovery_folds": 3,
        "auc_above_one_half_all_three": bool((auc_values > 0.5).all()),
        "delta_positive_all_three": bool((delta_values > 0.0).all()),
        "mean_auc": float(auc_values.mean()), "mean_delta": float(delta_values.mean()),
        "mean_delta_positive": bool(delta_values.mean() > 0.0),
        "screen_met": bool((auc_values > 0.5).all() and (delta_values > 0.0).all() and delta_values.mean() > 0.0),
    }


__all__ = [
    "D006_COLUMNS", "DISCOVERY_FOLDS", "RecentIntensityTransformer", "build_d006_representations",
    "build_d006_structure", "candidate_screen", "evaluate_d006_probabilities", "make_d006_multinomial_logistic_regression",
    "orientation_consistency", "orientation_shift", "ternary_probabilities", "validate_d006_matrix", "validate_d006_structure",
]
