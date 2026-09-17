"""Synthetic integrity tests for frozen D005; never loads physical Discovery data."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from discovery.d003_incremental_cross_sectional_relative_path import assert_e_dev_only, validate_discovery_features  # noqa: E402
from discovery.d004_hierarchical_recent_intensity_ternary_decision import (  # noqa: E402
    DirectionalPriors,
    build_d004_matrix,
    directional_probability,
    fit_directional_priors,
    fit_intensity_transformer,
)
from discovery.d005_gated_positional_path_conditional_sign import (  # noqa: E402
    DISCOVERY_FOLDS,
    SIGN_FORBIDDEN_FEATURES,
    PositionalPathTransformer,
    build_d005_path_matrix,
    descriptive_candidate_screen,
    evaluate_d005_sign_model,
    make_positional_logistic_regression,
    prepare_gated_directional_rows,
    sign_majority_class,
    strict_d004_gate,
)
from src.exp001_missingness import LOGISTIC_REGRESSION_PARAMS, RETURN_COLUMNS  # noqa: E402
from src.exp002_neutral_directional import build_binary_target, make_binary_logistic_regression  # noqa: E402
from src.exp003_conditional_directional_sign import evaluate_sign_majority_baseline, positive_probability  # noqa: E402
from src.exp005_positional_path_signal import PATH_MASK_COLUMNS, build_path_mask  # noqa: E402


def assert_raises(action, phrase: str) -> None:
    try:
        action()
    except AssertionError as error:
        assert phrase in str(error)
        return
    raise AssertionError("Expected integrity AssertionError.")


def synthetic_features(n: int = 12) -> pd.DataFrame:
    values = np.tile(np.arange(53, dtype=float), (n, 1)) + np.arange(n, dtype=float)[:, None]
    values[1, 2] = np.nan
    if n > 4:
        values[4, :] = np.nan  # Must be gated before EXP_005 evaluability excludes it.
    frame = pd.DataFrame(values, columns=RETURN_COLUMNS)
    frame.insert(0, "equity", np.arange(1, n + 1))
    frame.insert(0, "day", np.arange(n) % 3)
    frame.insert(0, "ID", np.arange(100, 100 + n))
    return frame


def priors() -> DirectionalPriors:
    return DirectionalPriors(pi_minus=0.6, pi_plus=0.4, pi_max=0.6, threshold=0.625)


def test_exact_gate_strictness_and_authoritative_probability_column() -> None:
    p = priors()
    values = np.array([p.threshold, np.nextafter(p.threshold, np.inf), np.nextafter(p.threshold, -np.inf)])
    gate = strict_d004_gate(values, p, index=pd.Index([11, 12, 13]))
    assert gate.tolist() == [False, True, False]

    class Stub:
        classes_ = np.array([1, 0])
        def predict_proba(self, matrix):
            return np.tile(np.array([0.7, 0.3]), (len(matrix), 1))
    assert np.allclose(directional_probability(Stub(), pd.DataFrame({"B": [0.0, 1.0]})), [0.7, 0.7])


def test_d004_gate_pipeline_is_fit_only_and_uses_authoritative_utilities() -> None:
    frame = synthetic_features(12)
    fit_features, validation_features = frame.iloc[:8], frame.iloc[8:]
    fit_reod = pd.Series([-1, 0, 1, 0, -1, 1, 0, -1], index=fit_features.index)
    validation_reod = pd.Series([-1, 1, 0, 1], index=validation_features.index)
    _, transformer = fit_intensity_transformer(fit_features)
    fit_matrix, _ = build_d004_matrix(fit_features, fit_reod, transformer)
    validation_matrix, _ = build_d004_matrix(validation_features, validation_reod, transformer)
    model = make_binary_logistic_regression().fit(fit_matrix, build_binary_target(fit_reod))
    frozen_priors = fit_directional_priors(fit_reod)
    probabilities = directional_probability(model, validation_matrix)
    observed = strict_d004_gate(probabilities, frozen_priors, index=validation_features.index)
    assert observed.equals(pd.Series(probabilities > frozen_priors.threshold, index=validation_features.index, name="G", dtype=bool))
    median, mean, scale = transformer.fit_intensity_median_, transformer.scaler_.mean_.copy(), transformer.scaler_.scale_.copy()
    altered_validation = validation_features.copy()
    altered_validation.loc[:, RETURN_COLUMNS] = 1e8
    _ = build_d004_matrix(altered_validation, validation_reod, transformer)
    assert transformer.fit_intensity_median_ == median
    assert np.array_equal(transformer.scaler_.mean_, mean) and np.array_equal(transformer.scaler_.scale_, scale)
    assert fit_directional_priors(fit_reod) == frozen_priors


def test_population_order_all_nan_and_gate_invariance() -> None:
    frame = synthetic_features(6)
    labels = pd.Series([-1, 1, 0, -1, 1, 0], index=frame.index)
    p = priors()
    probabilities = np.array([0.7, 0.7, 0.7, 0.1, 0.7, 0.7])
    rows = prepare_gated_directional_rows(frame, labels, probabilities, p)
    # index 4 is G=1 and directional, then excluded only because its own path is all NaN.
    assert rows.gate.loc[4]
    assert rows.manifest["n_gate_1"] == 5
    assert rows.manifest["n_directional_before_evaluability"] == 3
    assert rows.manifest["n_all_nan_directional_excluded"] == 1
    assert rows.raw_returns.index.tolist() == [0, 1]
    assert rows.sign_target.tolist() == [0, 1]
    original_gate = rows.gate.copy()
    changed = frame.drop(index=4)
    changed_rows = prepare_gated_directional_rows(changed, labels.drop(index=4), np.delete(probabilities, 4), p)
    assert changed_rows.gate.equals(original_gate.drop(index=4))


def test_fit_only_path_schema_masks_and_forbidden_gate_variables() -> None:
    frame = synthetic_features(10)
    labels = pd.Series([-1, 1, -1, 1, -1, 1, -1, 1, -1, 1], index=frame.index)
    probabilities = np.full(len(frame), 0.7)
    fit_rows = prepare_gated_directional_rows(frame.iloc[:6], labels.iloc[:6], probabilities[:6], priors())
    validation_rows = prepare_gated_directional_rows(frame.iloc[6:], labels.iloc[6:], probabilities[6:], priors())
    transformer = PositionalPathTransformer().fit(fit_rows.raw_returns)
    medians, mean, scale = transformer.medians_.copy(), transformer.scaler_.mean_.copy(), transformer.scaler_.scale_.copy()
    fit_matrix = build_d005_path_matrix(transformer, fit_rows)
    validation_matrix = build_d005_path_matrix(transformer, validation_rows)
    assert fit_matrix.shape[1] == validation_matrix.shape[1] == 106
    assert tuple(fit_matrix.columns) == PATH_MASK_COLUMNS
    assert fit_matrix.index.equals(fit_rows.sign_target.index) and validation_matrix.index.equals(validation_rows.sign_target.index)
    assert transformer.medians_.equals(medians) and np.array_equal(transformer.scaler_.mean_, mean) and np.array_equal(transformer.scaler_.scale_, scale)
    assert set(SIGN_FORBIDDEN_FEATURES).isdisjoint(fit_matrix.columns)
    assert np.isin(fit_matrix.iloc[:, 53:].to_numpy(), [0, 1]).all()
    assert fit_matrix.iloc[:, 53:].equals(build_path_mask(transformer, fit_rows.raw_returns).iloc[:, 53:])
    perturbed = validation_rows.raw_returns.copy(); perturbed.loc[:, RETURN_COLUMNS] = 1e9
    _ = transformer.transform_return_block(perturbed)
    assert transformer.medians_.equals(medians) and np.array_equal(transformer.scaler_.mean_, mean) and np.array_equal(transformer.scaler_.scale_, scale)


def test_both_classes_model_metrics_baseline_and_probability_auc() -> None:
    frame = synthetic_features(12)
    labels = pd.Series([-1, 1] * 6, index=frame.index)
    probabilities = np.full(len(frame), 0.7)
    fit_rows = prepare_gated_directional_rows(frame.iloc[:8], labels.iloc[:8], probabilities[:8], priors())
    validation_rows = prepare_gated_directional_rows(frame.iloc[8:], labels.iloc[8:], probabilities[8:], priors())
    transformer = PositionalPathTransformer().fit(fit_rows.raw_returns)
    fit_matrix, validation_matrix = build_d005_path_matrix(transformer, fit_rows), build_d005_path_matrix(transformer, validation_rows)
    model = make_positional_logistic_regression().fit(fit_matrix, fit_rows.sign_target)
    metrics, confusion = evaluate_d005_sign_model(validation_rows.sign_target, validation_matrix, model)
    probability = positive_probability(model, validation_matrix)
    assert np.isclose(metrics["roc_auc"], roc_auc_score(validation_rows.sign_target, probability))
    assert len(confusion) == len(confusion[0]) == 2
    assert {name: model.get_params()[name] for name in LOGISTIC_REGRESSION_PARAMS} == LOGISTIC_REGRESSION_PARAMS
    assert sign_majority_class(pd.Series([0, 1], dtype="int8")) == 0
    baseline, _ = evaluate_sign_majority_baseline(validation_rows.sign_target, 0)
    assert set(baseline) == {"accuracy", "balanced_accuracy", "recall_negative", "recall_positive"}
    assert_raises(
        lambda: evaluate_d005_sign_model(validation_rows.sign_target.iloc[:1], validation_matrix.iloc[:1], model),
        "Both sign classes",
    )


def test_screen_boundaries_and_protected_discovery_boundary() -> None:
    assert descriptive_candidate_screen([0.51, 0.52, 0.53])["descriptive_candidate_screen_met"]
    assert not descriptive_candidate_screen([0.51, 0.50, 0.53])["descriptive_candidate_screen_met"]
    assert [(f.fold, f.train_start, f.train_end, f.oos_start, f.oos_end) for f in DISCOVERY_FOLDS] == [
        (1, 0, 202, 203, 252), (2, 0, 252, 253, 302), (3, 0, 302, 303, 352)
    ]
    frame = synthetic_features(4)
    validate_discovery_features(frame)
    partition = pd.DataFrame({"equity": [1, 2, 3, 4, 99], "partition": ["E_dev"] * 4 + ["E_holdout"]})
    assert_e_dev_only(frame, partition)
    invalid_day = frame.copy(); invalid_day.loc[0, "day"] = 353
    assert_raises(lambda: validate_discovery_features(invalid_day), "days outside")
    invalid_equity = frame.copy(); invalid_equity.loc[0, "equity"] = 99
    assert_raises(lambda: assert_e_dev_only(invalid_equity, partition), "non-E_dev")
    source = (PROJECT_ROOT / "discovery" / "run_d005_gated_positional_path_conditional_sign.py").read_text(encoding="utf-8")
    assert "input_test" not in source and "output_test" not in source and "read_csv" not in source.split("load_physical_discovery_data", 1)[0]


if __name__ == "__main__":
    import unittest
    tests = (
        test_exact_gate_strictness_and_authoritative_probability_column,
        test_d004_gate_pipeline_is_fit_only_and_uses_authoritative_utilities,
        test_population_order_all_nan_and_gate_invariance,
        test_fit_only_path_schema_masks_and_forbidden_gate_variables,
        test_both_classes_model_metrics_baseline_and_probability_auc,
        test_screen_boundaries_and_protected_discovery_boundary,
    )
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(unittest.FunctionTestCase(test) for test in tests))
    raise SystemExit(not result.wasSuccessful())
