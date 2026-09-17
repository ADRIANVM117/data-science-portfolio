"""Synthetic integrity tests for frozen D003; no physical Discovery data."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from discovery.d003_incremental_cross_sectional_relative_path import (  # noqa: E402
    CANDIDATE_COLUMNS,
    CONTROL_COLUMNS,
    DISCOVERY_FOLDS,
    assert_e_dev_only,
    build_common_component,
    build_d003_matrices,
    build_relative_returns,
    descriptive_candidate_rule,
    evaluate_d003_model,
    prepare_directional_subset,
    split_discovery_fold,
    validate_discovery_features,
)
from src.exp001_missingness import LOGISTIC_REGRESSION_PARAMS, RETURN_COLUMNS  # noqa: E402
from src.exp005_positional_path_signal import PositionalPathTransformer, make_positional_logistic_regression  # noqa: E402


def assert_raises(action, expected: str) -> None:
    try:
        action()
    except AssertionError as error:
        assert expected in str(error)
        return
    raise AssertionError("Expected integrity assertion.")


def features() -> pd.DataFrame:
    values = np.zeros((6, 53), dtype=float)
    values[:, 0] = [1.0, 3.0, np.nan, 5.0, 10.0, 14.0]
    values[:, 1] = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    frame = pd.DataFrame(values, columns=RETURN_COLUMNS)
    frame.insert(0, "equity", [1, 2, 1, 2, 1, 2])
    frame.insert(0, "day", [0, 0, 1, 1, 2, 2])
    frame.insert(0, "ID", np.arange(10, 16))
    return frame


def test_exact_all_row_same_day_median_and_relative_missingness_identity() -> None:
    frame = features()
    common = build_common_component(frame)
    relative = build_relative_returns(frame)
    assert np.isclose(common.loc[0, "r0"], 2.0) and np.isclose(common.loc[1, "r0"], 2.0)
    assert np.isclose(common.loc[2, "r0"], 5.0) and np.isclose(common.loc[3, "r0"], 5.0)
    assert np.isclose(relative.loc[0, "r0"], -1.0)
    assert np.isclose(relative.loc[1, "r0"], 1.0)
    assert np.isnan(relative.loc[2, "r0"])
    assert relative.isna().equals(frame.loc[:, RETURN_COLUMNS].isna())
    # An observed residual equal to zero is not missing.
    assert relative.loc[3, "r0"] == 0.0 and not pd.isna(relative.loc[3, "r0"])


def test_frozen_discovery_boundaries_and_directional_evaluability() -> None:
    assert [(f.fold, f.train_start, f.train_end, f.oos_start, f.oos_end) for f in DISCOVERY_FOLDS] == [
        (1, 0, 202, 203, 252), (2, 0, 252, 253, 302), (3, 0, 302, 303, 352)
    ]
    frame = features()
    labels = pd.Series([-1, 1, 0, 1, -1, 1], index=frame.index)
    relative = build_relative_returns(frame)
    own, rel, target, manifest = prepare_directional_subset(frame, labels, relative)
    assert len(own) == len(rel) == len(target) == 5
    assert rel.isna().equals(own.isna())
    assert manifest["n_directional_before_evaluability"] == 5
    bad = frame.copy(); bad.loc[0, "day"] = 353
    assert_raises(lambda: validate_discovery_features(bad), "days outside")
    tiny = frame.copy(); tiny["day"] = [0, 0, 1, 1, 2, 2]
    subsets = split_discovery_fold(tiny, labels, type(DISCOVERY_FOLDS[0])(1, 0, 0, 1, 2))
    assert set(subsets.fit_features["ID"]).isdisjoint(set(subsets.validation_features["ID"]))
    partition = pd.DataFrame({"equity": [1, 2, 3], "partition": ["E_dev", "E_dev", "E_holdout"]})
    assert_e_dev_only(frame, partition)
    leaked = frame.copy(); leaked.loc[0, "equity"] = 3
    assert_raises(lambda: assert_e_dev_only(leaked, partition), "non-E_dev")


def test_separate_fit_only_transformers_and_exact_a_b_schema() -> None:
    frame = features()
    labels = pd.Series([-1, 1, -1, 1, -1, 1], index=frame.index)
    relative = build_relative_returns(frame)
    own, rel, target, _ = prepare_directional_subset(frame, labels, relative)
    own_transformer = PositionalPathTransformer().fit(own.iloc[:4])
    relative_transformer = PositionalPathTransformer().fit(rel.iloc[:4])
    medians_before = relative_transformer.medians_.copy()
    matrices = build_d003_matrices(own_transformer, relative_transformer, own, rel, target)
    assert tuple(matrices["A"].columns) == CONTROL_COLUMNS
    assert tuple(matrices["B"].columns) == CANDIDATE_COLUMNS
    assert matrices["A"].shape[1] == 106 and matrices["B"].shape[1] == 159
    assert matrices["A"].equals(matrices["B"].loc[:, CONTROL_COLUMNS])
    assert np.isfinite(matrices["B"].to_numpy(dtype=float)).all()
    assert relative_transformer.medians_.equals(medians_before)


def test_exact_model_and_descriptive_rule_and_runner_boundary() -> None:
    model = make_positional_logistic_regression()
    assert {name: model.get_params()[name] for name in LOGISTIC_REGRESSION_PARAMS} == LOGISTIC_REGRESSION_PARAMS
    assert descriptive_candidate_rule([0.001, 0.002, 0.003])["delta_auc_positive_all_three"]
    assert not descriptive_candidate_rule([0.001, 0.0, 0.003])["delta_auc_positive_all_three"]
    assert_raises(lambda: descriptive_candidate_rule([0.01, 0.02]), "exactly three")
    source = (PROJECT_ROOT / "discovery" / "run_d003_incremental_cross_sectional_relative_path.py").read_text(encoding="utf-8")
    assert "input_test" not in source and "output_test" not in source
    assert "data" not in source.split("DISCOVERY_DATA_DIR", 1)[0]


def test_end_to_end_synthetic_evaluation_uses_supported_probability_interface() -> None:
    """Exercise the exact runner failure path through fit/predict/evaluation."""
    frame = features()
    labels = pd.Series([-1, 1, -1, 1, -1, 1], index=frame.index)
    relative = build_relative_returns(frame)
    own, rel, target, _ = prepare_directional_subset(frame, labels, relative)
    own_transformer = PositionalPathTransformer().fit(own.iloc[:4])
    relative_transformer = PositionalPathTransformer().fit(rel.iloc[:4])
    fit_matrices = build_d003_matrices(
        own_transformer, relative_transformer, own.iloc[:4], rel.iloc[:4], target.iloc[:4]
    )
    validation_matrices = build_d003_matrices(
        own_transformer, relative_transformer, own.iloc[4:], rel.iloc[4:], target.iloc[4:]
    )
    metrics = {}
    for condition in ("A", "B"):
        model = make_positional_logistic_regression().fit(fit_matrices[condition], target.iloc[:4])
        observed, _ = evaluate_d003_model(target.iloc[4:], validation_matrices[condition], model)
        probability = model.predict_proba(validation_matrices[condition])[:, 1]
        assert np.isclose(observed["roc_auc"], roc_auc_score(target.iloc[4:], probability))
        metrics[condition] = observed
    delta = metrics["B"]["roc_auc"] - metrics["A"]["roc_auc"]
    screen = descriptive_candidate_rule([delta, delta, delta])
    assert screen["n_discovery_folds"] == 3


if __name__ == "__main__":
    import unittest

    suite = unittest.TestSuite(unittest.FunctionTestCase(test) for test in (
        test_exact_all_row_same_day_median_and_relative_missingness_identity,
        test_frozen_discovery_boundaries_and_directional_evaluability,
        test_separate_fit_only_transformers_and_exact_a_b_schema,
        test_exact_model_and_descriptive_rule_and_runner_boundary,
        test_end_to_end_synthetic_evaluation_uses_supported_probability_interface,
    ))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
