"""Frozen D007 raw-path versus masks-only ternary utilities; no data loading."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss, recall_score, roc_auc_score
from xgboost import XGBClassifier

from discovery.d003_incremental_cross_sectional_relative_path import DISCOVERY_FOLDS, validate_discovery_features
from src.exp000_validation import CLASS_ORDER, majority_class
from src.exp001_missingness import RETURN_COLUMNS

XGBOOST_VERSION = "3.4.1"
ENCODED_CLASS_ORDER = (0, 1, 2)
MASK_COLUMNS = tuple(f"m{i}" for i in range(53))
M_COLUMNS = MASK_COLUMNS
RM_COLUMNS = RETURN_COLUMNS + MASK_COLUMNS
XGBOOST_PARAMS: Mapping[str, object] = {
    "objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss", "n_estimators": 300,
    "learning_rate": 0.05, "max_depth": 3, "min_child_weight": 50, "subsample": 0.8,
    "colsample_bytree": 1.0, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0,
    "tree_method": "hist", "n_jobs": 1, "random_state": 20260908, "verbosity": 0,
}


def assert_ternary_target(target: pd.Series, *, expected_index: pd.Index, context: str) -> None:
    if not target.index.equals(expected_index) or set(target.unique()) != set(CLASS_ORDER):
        raise AssertionError(f"D007 requires aligned ternary target classes for {context}.")


def build_masks(features: pd.DataFrame) -> pd.DataFrame:
    validate_discovery_features(features)
    result = features.loc[:, RETURN_COLUMNS].isna().astype("int8")
    result.columns = MASK_COLUMNS
    validate_d007_matrix("M", result, expected_index=features.index)
    return result


def build_d007_representations(features: pd.DataFrame, target: pd.Series) -> dict[str, pd.DataFrame]:
    validate_discovery_features(features)
    assert_ternary_target(target, expected_index=features.index, context="representation construction")
    masks = build_masks(features)
    matrices = {"M": masks, "R+M": pd.concat([features.loc[:, RETURN_COLUMNS].copy(), masks], axis=1)}
    for name, matrix in matrices.items():
        validate_d007_matrix(name, matrix, expected_index=features.index, source_features=features)
    if not masks.equals(matrices["R+M"].loc[:, M_COLUMNS]):
        raise AssertionError("D007 R+M must append M unchanged.")
    return matrices


def validate_d007_matrix(name: str, matrix: pd.DataFrame, *, expected_index: pd.Index | None = None, source_features: pd.DataFrame | None = None) -> None:
    expected = M_COLUMNS if name == "M" else RM_COLUMNS if name == "R+M" else None
    if expected is None or tuple(matrix.columns) != expected:
        raise AssertionError("D007 predictor schema differs from frozen M/R+M definitions.")
    if expected_index is not None and not matrix.index.equals(expected_index):
        raise AssertionError("D007 matrix changed source row identity or order.")
    masks = matrix.loc[:, M_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(masks).all() or not np.isin(masks, [0.0, 1.0]).all():
        raise AssertionError("D007 masks must be finite binary indicators.")
    if name == "R+M":
        raw = matrix.loc[:, RETURN_COLUMNS].to_numpy(dtype=float)
        observed = ~np.isnan(raw)
        if not np.isfinite(raw[observed]).all():
            raise AssertionError("D007 observed raw returns must be finite.")
        if not np.array_equal(masks, np.isnan(raw).astype(float)):
            raise AssertionError("D007 masks must exactly encode raw-return NaNs.")
        if source_features is not None and not matrix.loc[:, RETURN_COLUMNS].equals(source_features.loc[:, RETURN_COLUMNS]):
            raise AssertionError("D007 must preserve raw returns exactly.")


def encode_ternary_target(target: pd.Series) -> pd.Series:
    assert_ternary_target(target, expected_index=target.index, context="encoding")
    result = target.map({-1: 0, 0: 1, 1: 2})
    if result.isna().any() or set(result.unique()) != set(ENCODED_CLASS_ORDER):
        raise AssertionError("D007 ternary target encoding failed.")
    return result.astype("int8")


def make_d007_classifier() -> XGBClassifier:
    if xgboost.__version__ != XGBOOST_VERSION:
        raise AssertionError(f"D007 requires xgboost {XGBOOST_VERSION}, found {xgboost.__version__}.")
    model = XGBClassifier(**XGBOOST_PARAMS)
    for name, expected in XGBOOST_PARAMS.items():
        if model.get_params().get(name) != expected:
            raise AssertionError(f"D007 XGBoost parameter {name!r} differs from frozen specification.")
    if model.get_params().get("early_stopping_rounds") is not None:
        raise AssertionError("D007 forbids early stopping.")
    return model


def ternary_probabilities(model: XGBClassifier, matrix: pd.DataFrame) -> pd.DataFrame:
    if tuple(model.classes_) != ENCODED_CLASS_ORDER:
        raise AssertionError("D007 XGBoost classes must be encoded [0,1,2].")
    raw_values = np.asarray(model.predict_proba(matrix))
    if not np.issubdtype(raw_values.dtype, np.floating):
        raise AssertionError("D007 probability output must use a floating dtype.")
    row_sum_tolerance = max(1e-12, 2 * np.finfo(raw_values.dtype).eps)
    values = np.asarray(raw_values, dtype=float)
    if values.shape != (len(matrix), 3) or not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=row_sum_tolerance):
        raise AssertionError("D007 probabilities must be finite and coherent.")
    return pd.DataFrame(values, index=matrix.index, columns=("p_minus", "p_zero", "p_plus"))


def evaluate_d007_probabilities(target: pd.Series, probabilities: pd.DataFrame) -> tuple[dict[str, float], list[list[int]]]:
    assert_ternary_target(target, expected_index=probabilities.index, context="evaluation")
    truth = target.to_numpy(dtype=int)
    prediction = np.asarray(CLASS_ORDER)[probabilities.to_numpy().argmax(axis=1)]
    recalls = recall_score(truth, prediction, labels=CLASS_ORDER, average=None, zero_division=0)
    negative_auc = float(roc_auc_score(target.eq(-1).astype(int), probabilities.p_minus))
    positive_auc = float(roc_auc_score(target.eq(1).astype(int), probabilities.p_plus))
    metrics = {
        "multiclass_log_loss": float(log_loss(truth, probabilities.to_numpy(), labels=list(CLASS_ORDER))),
        "tail_auc_minus": negative_auc, "tail_auc_plus": positive_auc, "macro_tail_auc": (negative_auc + positive_auc) / 2.0,
        "accuracy": float(accuracy_score(truth, prediction)), "macro_f1": float(f1_score(truth, prediction, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)), "recall_negative": float(recalls[0]),
        "recall_neutral": float(recalls[1]), "recall_positive": float(recalls[2]),
    }
    return metrics, confusion_matrix(truth, prediction, labels=CLASS_ORDER).astype(int).tolist()


def evaluate_majority_baseline(fit_target: pd.Series, validation_target: pd.Series) -> tuple[int, dict[str, float], list[list[int]]]:
    assert_ternary_target(fit_target, expected_index=fit_target.index, context="majority fit")
    assert_ternary_target(validation_target, expected_index=validation_target.index, context="majority validation")
    selected = majority_class(fit_target)
    prediction = np.full(len(validation_target), selected, dtype=int)
    truth = validation_target.to_numpy(dtype=int)
    recalls = recall_score(truth, prediction, labels=CLASS_ORDER, average=None, zero_division=0)
    metrics = {"accuracy": float(accuracy_score(truth, prediction)), "macro_f1": float(f1_score(truth, prediction, labels=CLASS_ORDER, average="macro", zero_division=0)), "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)), "recall_negative": float(recalls[0]), "recall_neutral": float(recalls[1]), "recall_positive": float(recalls[2])}
    return selected, metrics, confusion_matrix(truth, prediction, labels=CLASS_ORDER).astype(int).tolist()


def primary_screen(deltas: Sequence[float]) -> dict[str, float | bool | int]:
    values = np.asarray(deltas, dtype=float)
    if len(values) != 3 or not np.isfinite(values).all():
        raise AssertionError("D007 primary screen requires exactly three finite fold deltas.")
    mean = float(values.mean())
    return {"n_discovery_folds": 3, "delta_log_loss_positive_all_three": bool((values > 0).all()), "mean_delta_log_loss": mean, "mean_delta_log_loss_positive": bool(mean > 0), "screen_met": bool((values > 0).all() and mean > 0)}
