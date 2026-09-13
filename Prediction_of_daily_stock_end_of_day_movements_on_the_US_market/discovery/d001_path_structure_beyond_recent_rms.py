"""Discovery-only utilities for D001: path magnitude beyond recent RMS.

This module is restricted to the post-EXP_009 Discovery scope, days 0-352 x
E_dev.  It is not an Internal Confirmation experiment.

The C and P representations are a controlled model-representation comparison,
not a perfect conditional-information test: RMS is deterministic from P, but a
finite depth-limited tree ensemble may not reconstruct it as efficiently as a
direct scalar input.  Therefore P > C can show greater discrimination under
this frozen exploratory learner, not unique incremental information conditional
on RMS; P <= C cannot prove that path structure contains no information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier


DISCOVERY_DAY_START = 0
DISCOVERY_DAY_END = 352
RECENT_WINDOW_COLUMNS: tuple[str, ...] = tuple(f"r{position}" for position in range(41, 53))
RECENT_WINDOW_SIZE = 12
CONTROL_COLUMNS: tuple[str, ...] = ("I_recent",)
PATH_COLUMNS: tuple[str, ...] = tuple(f"abs_r{position}" for position in range(41, 53))
XGBOOST_PARAMS: Mapping[str, object] = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_child_weight": 50,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "n_jobs": 1,
    "random_state": 20260908,
    "verbosity": 0,
}
GLOBAL_SHAP_SAMPLE_MAX = 2_000
INTERACTION_SHAP_SAMPLE_MAX = 500
GLOBAL_SHAP_SEED = 20260908
INTERACTION_SHAP_SEED = 20261908


@dataclass(frozen=True)
class DiscoveryFold:
    """One inclusive expanding Discovery fit/validation split."""

    fold: int
    fit_start: int
    fit_end: int
    validation_start: int
    validation_end: int


DISCOVERY_FOLDS: tuple[DiscoveryFold, ...] = (
    DiscoveryFold(1, 0, 202, 203, 252),
    DiscoveryFold(2, 0, 252, 253, 302),
    DiscoveryFold(3, 0, 302, 303, 352),
)


def validate_discovery_scope(frame: pd.DataFrame, development_equities: Iterable[int]) -> None:
    """Fail if data outside the authorized Discovery boundary enter D001."""
    if not {"ID", "day", "equity"}.issubset(frame.columns):
        raise AssertionError("D001 rows require ID, day, and equity.")
    if not frame["day"].between(DISCOVERY_DAY_START, DISCOVERY_DAY_END).all():
        raise AssertionError("D001 rejects days outside 0-352 Discovery.")
    development = set(development_equities)
    if not set(frame["equity"]).issubset(development):
        raise AssertionError("D001 rejects E_holdout or unknown equities.")
    if frame["ID"].duplicated().any():
        raise AssertionError("D001 source rows contain duplicate IDs.")


def _window_values(frame: pd.DataFrame) -> np.ndarray:
    if RECENT_WINDOW_COLUMNS != tuple(f"r{position}" for position in range(41, 53)):
        raise AssertionError("D001 W must be exactly r41 through r52.")
    if len(RECENT_WINDOW_COLUMNS) != RECENT_WINDOW_SIZE:
        raise AssertionError("D001 W must contain exactly 12 positions.")
    missing = set(RECENT_WINDOW_COLUMNS).difference(frame.columns)
    if missing:
        raise AssertionError(f"D001 is missing W columns: {sorted(missing)}")
    try:
        return frame.loc[:, RECENT_WINDOW_COLUMNS].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AssertionError("D001 W values must be numeric.") from error


def prepare_complete_w_population(
    features: pd.DataFrame, target: pd.Series, development_equities: Iterable[int]
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Apply target-free complete-W eligibility, then construct C/P sources."""
    if not features.index.equals(target.index):
        raise AssertionError("D001 features and target must have identical indices.")
    validate_discovery_scope(features, development_equities)
    values = _window_values(features)
    n_obs = (~np.isnan(values)).sum(axis=1).astype("int16")
    complete = n_obs == RECENT_WINDOW_SIZE
    retained_features = features.loc[complete].copy()
    retained_target = target.loc[retained_features.index].copy()
    if retained_features.empty:
        raise AssertionError("D001 complete-W eligibility produced no retained rows.")
    retained_values = _window_values(retained_features)
    if np.isnan(retained_values).any():
        raise AssertionError("D001 retained complete-W rows cannot contain NaNs.")
    if not np.isfinite(retained_values).all():
        raise AssertionError("D001 retained observed W values must be finite.")
    structure = pd.DataFrame(
        {"N_obs_w": n_obs, "complete_w": complete}, index=features.index
    )
    if not np.array_equal(structure["complete_w"].to_numpy(), structure["N_obs_w"].eq(12).to_numpy()):
        raise AssertionError("D001 complete-W must equal N_obs,W == 12.")
    return retained_features, retained_target, structure


def build_d001_representations(complete_features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build exactly C_D001=[RMS] and P_D001=[12 absolute magnitudes]."""
    values = _window_values(complete_features)
    if np.isnan(values).any() or not np.isfinite(values).all():
        raise AssertionError("D001 representations require complete finite W values.")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            rms = np.sqrt(np.square(values).mean(axis=1))
            magnitudes = np.abs(values)
        except FloatingPointError as error:
            raise AssertionError("D001 magnitude construction was non-finite.") from error
    control = pd.DataFrame({"I_recent": rms}, index=complete_features.index)
    path = pd.DataFrame(magnitudes, columns=PATH_COLUMNS, index=complete_features.index)
    representations = {"C": control, "P": path}
    for name, matrix in representations.items():
        validate_d001_representation(name, matrix, expected_index=complete_features.index)
    return representations


def validate_d001_representation(
    name: str, matrix: pd.DataFrame, *, expected_index: pd.Index | None = None
) -> None:
    expected = CONTROL_COLUMNS if name == "C" else PATH_COLUMNS if name == "P" else None
    if expected is None:
        raise AssertionError(f"Unknown D001 representation: {name}")
    if tuple(matrix.columns) != expected:
        raise AssertionError(f"D001 {name} predictor schema is frozen.")
    if expected_index is not None and not matrix.index.equals(expected_index):
        raise AssertionError(f"D001 {name} changed row identity or order.")
    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError(f"D001 {name} predictors must be finite.")
    if (values < 0.0).any():
        raise AssertionError(f"D001 {name} magnitude predictors must be non-negative.")


def assert_identical_representation_rows(
    representations: Mapping[str, pd.DataFrame], features: pd.DataFrame, target: pd.Series
) -> None:
    """Require C/P to use exactly the same retained source rows and labels."""
    if not features.index.equals(target.index) or features["ID"].duplicated().any():
        raise AssertionError("D001 retained features and target are not aligned.")
    for name in ("C", "P"):
        if name not in representations:
            raise AssertionError(f"D001 representation {name} is missing.")
        validate_d001_representation(name, representations[name], expected_index=features.index)
        if len(representations[name]) != len(target):
            raise AssertionError("D001 C/P must retain identical source rows.")


def assert_both_binary_classes(target: pd.Series, *, context: str) -> None:
    if set(target.unique()) != {0, 1}:
        raise AssertionError(f"D001 requires both binary classes for {context}.")


def split_discovery_fold(
    features: pd.DataFrame,
    target: pd.Series,
    development_equities: Iterable[int],
    fold: DiscoveryFold,
) -> tuple[tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series]]:
    """Create one frozen Discovery-only expanding chronological split."""
    if fold not in DISCOVERY_FOLDS:
        raise AssertionError("D001 fold is not one of the three frozen Discovery folds.")
    if fold.fit_end >= fold.validation_start:
        raise AssertionError("D001 fit and validation days must not overlap.")
    validate_discovery_scope(features, development_equities)
    if not features.index.equals(target.index):
        raise AssertionError("D001 features and target must align before splitting.")
    fit_mask = features["day"].between(fold.fit_start, fold.fit_end)
    validation_mask = features["day"].between(fold.validation_start, fold.validation_end)
    fit = (features.loc[fit_mask].copy(), target.loc[fit_mask].copy())
    validation = (features.loc[validation_mask].copy(), target.loc[validation_mask].copy())
    if fit[0].empty or validation[0].empty:
        raise AssertionError("D001 Discovery fold cannot be empty.")
    if set(fit[0]["ID"]) & set(validation[0]["ID"]):
        raise AssertionError("D001 fit and validation rows must be disjoint.")
    return fit, validation


def make_d001_classifier() -> XGBClassifier:
    """Construct the one approved fixed exploratory XGBoost learner."""
    classifier = XGBClassifier(**XGBOOST_PARAMS)
    parameters = classifier.get_params()
    for name, expected in XGBOOST_PARAMS.items():
        if parameters.get(name) != expected:
            raise AssertionError(f"D001 XGBoost parameter {name!r} differs from the approved configuration.")
    if parameters.get("early_stopping_rounds") is not None:
        raise AssertionError("D001 must not use early stopping.")
    return classifier


def incremental_auc_records(fold_deltas: Mapping[int, float]) -> pd.DataFrame:
    """Create non-overlapping fold, mean, and sample-std AUC records."""
    if set(fold_deltas) != {1, 2, 3}:
        raise AssertionError("D001 incremental AUC records require exactly folds 1, 2, and 3.")
    values = np.asarray([fold_deltas[fold] for fold in (1, 2, 3)], dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError("D001 incremental AUC deltas must be finite.")
    records: list[dict[str, int | str | float]] = [
        {"record_type": "fold", "fold": fold, "delta_p_minus_c_validation": float(fold_deltas[fold])}
        for fold in (1, 2, 3)
    ]
    records.extend(
        [
            {"record_type": "aggregate_mean", "fold": "all", "delta_p_minus_c_validation": float(values.mean())},
            {"record_type": "aggregate_std_ddof_1", "fold": "all", "delta_p_minus_c_validation": float(values.std(ddof=1))},
        ]
    )
    return pd.DataFrame(records)


def deterministic_sample_indices(index: pd.Index, *, maximum: int, seed: int) -> pd.Index:
    """Sample without replacement deterministically while preserving source order."""
    if maximum <= 0 or not index.is_unique:
        raise AssertionError("D001 SHAP sampling requires a positive limit and unique source indices.")
    size = min(len(index), maximum)
    positions = np.sort(np.random.default_rng(seed).choice(len(index), size=size, replace=False))
    return index.take(positions)


def make_d001_tree_explainer(classifier: XGBClassifier) -> shap.TreeExplainer:
    """Build the audited SHAP 0.52 / XGBoost 3.4 tree-path explainer."""
    if not isinstance(classifier, XGBClassifier):
        raise AssertionError("D001 SHAP requires an XGBClassifier.")
    return shap.TreeExplainer(
        classifier,
        data=None,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
        feature_names=list(PATH_COLUMNS),
    )


def explain_d001_path_validation(
    classifier: XGBClassifier, path_matrix: pd.DataFrame, *, fold_number: int
) -> dict[str, object]:
    """Produce validation-only global and interaction SHAP objects for D001.

    Raw-margin additivity is checked against XGBoost ``output_margin``. SHAP
    is exploratory attribution under the tree-path-dependent convention, not
    causal or unique feature importance for dependent path magnitudes.
    """
    validate_d001_representation("P", path_matrix)
    if fold_number not in (1, 2, 3):
        raise AssertionError("D001 SHAP requires a frozen Discovery fold number.")
    global_index = deterministic_sample_indices(
        path_matrix.index, maximum=GLOBAL_SHAP_SAMPLE_MAX, seed=GLOBAL_SHAP_SEED + fold_number
    )
    global_matrix = path_matrix.loc[global_index]
    explainer = make_d001_tree_explainer(classifier)
    explanation = explainer(global_matrix)
    values = np.asarray(explanation.values, dtype=float)
    if values.shape != (len(global_matrix), len(PATH_COLUMNS)) or not np.isfinite(values).all():
        raise AssertionError("D001 global SHAP values have an unexpected non-finite shape.")
    margin = np.asarray(classifier.predict(global_matrix, output_margin=True), dtype=float)
    reconstruction = np.asarray(explanation.base_values, dtype=float) + values.sum(axis=1)
    if not np.allclose(reconstruction, margin, rtol=1e-5, atol=1e-5):
        raise AssertionError("D001 raw SHAP values failed the supported additivity check.")
    interaction_index = deterministic_sample_indices(
        global_index, maximum=INTERACTION_SHAP_SAMPLE_MAX, seed=INTERACTION_SHAP_SEED + fold_number
    )
    interaction_values = np.asarray(explainer.shap_interaction_values(global_matrix.loc[interaction_index]), dtype=float)
    expected_shape = (len(interaction_index), len(PATH_COLUMNS), len(PATH_COLUMNS))
    if interaction_values.shape != expected_shape or not np.isfinite(interaction_values).all():
        raise AssertionError("D001 SHAP interaction values have an unexpected non-finite shape.")
    mean_abs = pd.Series(np.abs(values).mean(axis=0), index=PATH_COLUMNS, name=f"fold_{fold_number}")
    return {
        "global_index": global_index,
        "global_explanation": explanation,
        "interaction_index": interaction_index,
        "interaction_values": interaction_values,
        "mean_abs_shap": mean_abs,
    }


def cross_fold_shap_summary(fold_mean_abs: Mapping[int, pd.Series]) -> pd.DataFrame:
    """Return a beeswarm-independent positional ranking/stability table."""
    if set(fold_mean_abs) != {1, 2, 3}:
        raise AssertionError("D001 SHAP summary requires all three Discovery folds.")
    table = pd.DataFrame({f"mean_abs_shap_fold_{fold}": fold_mean_abs[fold] for fold in (1, 2, 3)})
    if not table.index.equals(pd.Index(PATH_COLUMNS)) or not np.isfinite(table.to_numpy(dtype=float)).all():
        raise AssertionError("D001 SHAP summary must contain finite PATH_COLUMNS values.")
    for fold in (1, 2, 3):
        table[f"rank_fold_{fold}"] = table[f"mean_abs_shap_fold_{fold}"].rank(ascending=False, method="min")
    table["mean_abs_shap_across_folds"] = table[[f"mean_abs_shap_fold_{fold}" for fold in (1, 2, 3)]].mean(axis=1)
    table["mean_rank_across_folds"] = table[[f"rank_fold_{fold}" for fold in (1, 2, 3)]].mean(axis=1)
    return table
