"""Synthetic integrity tests for frozen D006; never loads physical Discovery data."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from discovery.d003_incremental_cross_sectional_relative_path import assert_e_dev_only, validate_discovery_features  # noqa: E402
from discovery.d004_hierarchical_recent_intensity_ternary_decision import fit_intensity_transformer  # noqa: E402
from discovery.d006_ternary_tail_intensity_interaction import (  # noqa: E402
    D006_COLUMNS,
    DISCOVERY_FOLDS,
    build_d006_representations,
    build_d006_structure,
    candidate_screen,
    evaluate_d006_probabilities,
    make_d006_multinomial_logistic_regression,
    orientation_consistency,
    orientation_shift,
    ternary_probabilities,
)
from src.exp001_missingness import LOGISTIC_REGRESSION_PARAMS, RETURN_COLUMNS  # noqa: E402
from src.exp002_neutral_directional import build_missing_ratio  # noqa: E402


def assert_raises(action, phrase: str) -> None:
    try:
        action()
    except AssertionError as error:
        assert phrase in str(error)
        return
    raise AssertionError("Expected an integrity AssertionError.")


def synthetic_features(n: int = 18) -> pd.DataFrame:
    values = np.zeros((n, 53), dtype=float)
    for row in range(n):
        values[row, 41] = float((row % 5) - 2)
        values[row, 42] = float(((2 * row) % 7) - 3)
        values[row, 51] = float((row % 3) - 1)
    values[3, 41:53] = 0.0
    values[4, 41:53] = np.nan
    values[5, 41:53] *= 2.0
    frame = pd.DataFrame(values, columns=RETURN_COLUMNS)
    frame.insert(0, "equity", np.arange(1, n + 1))
    frame.insert(0, "day", np.arange(n) % 4)
    frame.insert(0, "ID", np.arange(100, 100 + n))
    return frame


def labels(index: pd.Index) -> pd.Series:
    return pd.Series(np.resize(np.array([-1, 0, 1], dtype=int), len(index)), index=index, name="reod")


def test_energy_orientation_algebra_and_zero_missingness_semantics() -> None:
    frame = synthetic_features(8)
    frame.loc[0, "r41":"r52"] = [2.0, -2.0] + [0.0] * 10
    frame.loc[1, "r41":"r52"] = [3.0] + [0.0] * 11
    frame.loc[2, "r41":"r52"] = [-3.0] + [0.0] * 11
    frame.loc[3, "r41":"r52"] = 0.0
    frame.loc[4, "r41":"r52"] = np.nan
    frame.loc[5, "r41":"r52"] = [6.0] + [0.0] * 11
    structure = build_d006_structure(frame)
    assert np.isclose(structure.loc[0, "O"], 0.0)
    assert np.isclose(structure.loc[1, "O"], 1.0) and np.isclose(structure.loc[2, "O"], -1.0)
    assert structure.loc[3, "N_obs_w"] == 12 and structure.loc[3, "q"] == 0 and structure.loc[3, "O"] == 0.0
    assert structure.loc[4, "N_obs_w"] == 0 and structure.loc[4, "q"] == 1 and np.isnan(structure.loc[4, "raw_I_recent"]) and structure.loc[4, "O"] == 0.0
    assert np.isclose(structure.loc[1, "O"], structure.loc[5, "O"])
    signed_energy = structure["E_plus"] - structure["E_minus"]
    defined = structure["E_total"] > 0.0
    assert np.allclose((structure.loc[defined, "raw_I_recent"] ** 2) * structure.loc[defined, "O"], signed_energy.loc[defined] / structure.loc[defined, "N_obs_w"])
    assert_raises(lambda: build_d006_structure(frame.assign(r41=np.inf)), "non-NaN")


def test_fit_only_intensity_and_exact_nested_matrix_schemas() -> None:
    frame = synthetic_features()
    target = labels(frame.index)
    fit, validation = frame.iloc[:12], frame.iloc[12:]
    y_fit, y_validation = target.iloc[:12], target.iloc[12:]
    fit_structure, transformer = fit_intensity_transformer(fit)
    median, mean, scale = transformer.fit_intensity_median_, transformer.scaler_.mean_.copy(), transformer.scaler_.scale_.copy()
    fit_matrices = build_d006_representations(fit, y_fit, transformer)
    validation_matrices = build_d006_representations(validation, y_validation, transformer)
    assert tuple(fit_matrices["C0"].columns) == D006_COLUMNS["C0"]
    assert tuple(fit_matrices["C1"].columns) == D006_COLUMNS["C1"]
    assert tuple(fit_matrices["C2"].columns) == D006_COLUMNS["C2"]
    assert [matrix.shape[1] for matrix in fit_matrices.values()] == [3, 4, 5]
    assert fit_matrices["C0"].equals(fit_matrices["C1"].loc[:, D006_COLUMNS["C0"]])
    assert fit_matrices["C1"].equals(fit_matrices["C2"].loc[:, D006_COLUMNS["C1"]])
    assert all(matrix.index.equals(y_fit.index) for matrix in fit_matrices.values())
    assert all(matrix.index.equals(y_validation.index) for matrix in validation_matrices.values())
    assert fit_matrices["C0"]["missing_ratio"].equals(build_missing_ratio(fit)["missing_ratio"])
    assert transformer.fit_intensity_median_ == median and np.array_equal(transformer.scaler_.mean_, mean) and np.array_equal(transformer.scaler_.scale_, scale)
    perturbed = validation.copy(); perturbed.loc[:, "r41":"r52"] = 1e9
    _ = build_d006_representations(perturbed, y_validation, transformer)
    assert transformer.fit_intensity_median_ == median and np.array_equal(transformer.scaler_.mean_, mean) and np.array_equal(transformer.scaler_.scale_, scale)
    assert fit_structure.index.equals(fit.index)


def test_multinomial_probability_metrics_orientation_and_screens() -> None:
    frame = synthetic_features()
    target = labels(frame.index)
    fit, validation = frame.iloc[:12], frame.iloc[12:]
    y_fit, y_validation = target.iloc[:12], target.iloc[12:]
    _, transformer = fit_intensity_transformer(fit)
    fit_matrices = build_d006_representations(fit, y_fit, transformer)
    validation_matrices = build_d006_representations(validation, y_validation, transformer)
    model = make_d006_multinomial_logistic_regression().fit(fit_matrices["C2"], y_fit)
    assert {name: model.get_params()[name] for name in LOGISTIC_REGRESSION_PARAMS} == LOGISTIC_REGRESSION_PARAMS
    probabilities = ternary_probabilities(model, validation_matrices["C2"])
    metrics, matrix = evaluate_d006_probabilities(y_validation, probabilities)
    assert set(metrics) == {"tail_auc_minus", "tail_auc_plus", "macro_tail_auc", "multiclass_log_loss", "accuracy", "macro_f1", "recall_negative", "recall_neutral", "recall_positive"}
    assert len(matrix) == len(matrix[0]) == 3 and np.allclose(probabilities.sum(axis=1), 1.0)

    class ContinuationStub:
        classes_ = np.array([-1, 0, 1])
        def predict_proba(self, matrix):
            o = matrix["O"].to_numpy(dtype=float)
            return np.column_stack([0.3 - 0.1 * o, np.full(len(o), 0.4), 0.3 + 0.1 * o])
    orientation_matrix = validation_matrices["C2"].copy(); orientation_matrix["O"] = 0.5; orientation_matrix["I_recent_z_x_O"] = 0.0
    assert orientation_shift(ContinuationStub(), "C2", orientation_matrix) > 0.0
    assert orientation_consistency([0.01, 0.02, 0.03]) == "continuation"
    assert orientation_consistency([-0.01, -0.02, -0.03]) == "opposite"
    assert orientation_consistency([0.01, -0.02, 0.03]) == "unstable"
    assert candidate_screen([0.51, 0.52, 0.53], [0.01, 0.02, 0.03])["screen_met"]
    assert not candidate_screen([0.51, 0.50, 0.53], [0.01, 0.02, 0.03])["screen_met"]


def test_discovery_boundaries_and_runner_has_no_competition_or_fallback() -> None:
    assert [(fold.fold, fold.train_start, fold.train_end, fold.oos_start, fold.oos_end) for fold in DISCOVERY_FOLDS] == [(1, 0, 202, 203, 252), (2, 0, 252, 253, 302), (3, 0, 302, 303, 352)]
    frame = synthetic_features(6)
    validate_discovery_features(frame)
    partition = pd.DataFrame({"equity": [1, 2, 3, 4, 5, 6, 99], "partition": ["E_dev"] * 6 + ["E_holdout"]})
    assert_e_dev_only(frame, partition)
    bad_day = frame.copy(); bad_day.loc[0, "day"] = 353
    assert_raises(lambda: validate_discovery_features(bad_day), "days outside")
    bad_equity = frame.copy(); bad_equity.loc[0, "equity"] = 99
    assert_raises(lambda: assert_e_dev_only(bad_equity, partition), "non-E_dev")
    source = (PROJECT_ROOT / "discovery" / "run_d006_ternary_tail_intensity_interaction.py").read_text(encoding="utf-8")
    assert "input_test" not in source and "output_test" not in source and "full-training" not in source


if __name__ == "__main__":
    import unittest
    tests = (
        test_energy_orientation_algebra_and_zero_missingness_semantics,
        test_fit_only_intensity_and_exact_nested_matrix_schemas,
        test_multinomial_probability_metrics_orientation_and_screens,
        test_discovery_boundaries_and_runner_has_no_competition_or_fallback,
    )
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(unittest.FunctionTestCase(test) for test in tests))
    raise SystemExit(not result.wasSuccessful())
