"""Contract tests for EXP_001 missingness-only representations and decisions."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.exp001_missingness import (  # noqa: E402
    LOGISTIC_REGRESSION_PARAMS,
    RETURN_COLUMNS,
    VARIANT_COLUMNS,
    build_missingness_representations,
    evaluate_predictions,
    joint_decision,
    make_logistic_regression,
    strictly_greater,
    validate_representation,
)


def synthetic_returns() -> pd.DataFrame:
    values = np.zeros((2, 53), dtype=float)
    values[0, [0, 16, 17, 34, 35, 52]] = np.nan
    values[1, [1, 18, 36]] = np.nan
    return pd.DataFrame(values, columns=RETURN_COLUMNS, index=pd.Index([11, 29], name="row_id"))


def assert_raises_integrity(action, expected_message: str) -> None:
    try:
        action()
    except AssertionError as error:
        assert expected_message in str(error)
        return
    raise AssertionError("Expected an integrity assertion to be raised.")


def test_representations_have_exact_contract_schemas_and_values() -> None:
    returns = synthetic_returns()
    representations = build_missingness_representations(returns)

    assert set(representations) == {"M1", "M2", "M3"}
    for variant, matrix in representations.items():
        assert list(matrix.columns) == list(VARIANT_COLUMNS[variant])
        assert matrix.index.equals(returns.index)

    assert representations["M1"].loc[11, "missing_ratio"] == 6 / 53
    assert representations["M2"].loc[11, "missing_early"] == 2 / 17
    assert representations["M2"].loc[11, "missing_mid"] == 2 / 18
    assert representations["M2"].loc[11, "missing_late"] == 2 / 18
    assert representations["M3"].loc[11, "m0"] == 1
    assert representations["M3"].loc[11, "m1"] == 0
    assert representations["M3"].loc[29, "m1"] == 1


def test_representation_validation_rejects_contract_deviations() -> None:
    returns = synthetic_returns()
    matrix = build_missingness_representations(returns)["M3"]
    malformed = matrix.rename(columns={"m0": "bad"})
    assert_raises_integrity(
        lambda: validate_representation("M3", malformed, expected_index=returns.index),
        "columns",
    )
    non_binary = matrix.astype(float).copy()
    non_binary.iloc[0, 0] = 0.5
    assert_raises_integrity(lambda: validate_representation("M3", non_binary), "binary")
    assert_raises_integrity(
        lambda: validate_representation("M3", matrix, expected_index=pd.Index([29, 11], name="row_id")),
        "row identity",
    )


def test_fixed_logistic_regression_configuration() -> None:
    classifier = make_logistic_regression()
    parameters = classifier.get_params()
    for name, expected in LOGISTIC_REGRESSION_PARAMS.items():
        assert parameters[name] == expected
    # In sklearn 1.6 the deprecated argument is left at its default rather
    # than explicitly forcing a now-deprecated value.
    assert parameters["multi_class"] == "deprecated"


def test_metrics_include_all_classes_and_decision_is_strict() -> None:
    target = pd.Series([-1, 0, 1, -1, 0, 1], name="reod")
    prediction = np.array([-1, 0, 1, 0, 0, 1], dtype=np.int8)
    metrics, matrix = evaluate_predictions(target, prediction)
    assert set(metrics) == {
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "recall_class_-1",
        "recall_class_0",
        "recall_class_1",
    }
    assert matrix == [[1, 1, 0], [0, 2, 0], [0, 0, 2]]
    assert joint_decision(metrics, baseline_accuracy=0.5)["all_fold_conditions_met"]
    assert not joint_decision(metrics, baseline_accuracy=metrics["accuracy"])["accuracy_above_baseline"]
    assert not strictly_greater(metrics["accuracy"] + 5e-17, metrics["accuracy"])
    flat_metrics = dict(metrics)
    flat_metrics["balanced_accuracy"] = 1 / 3
    assert not joint_decision(flat_metrics, baseline_accuracy=0.5)["balanced_accuracy_above_one_third"]


if __name__ == "__main__":
    import unittest

    suite = unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_representations_have_exact_contract_schemas_and_values,
            test_representation_validation_rejects_contract_deviations,
            test_fixed_logistic_regression_configuration,
            test_metrics_include_all_classes_and_decision_is_strict,
        )
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
