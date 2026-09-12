"""Synthetic contract tests for frozen EXP_005 positional-path signal."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.exp000_validation import FOLDS, FoldSpec, build_equity_partition, split_fold  # noqa: E402
from src.exp001_missingness import RETURN_COLUMNS  # noqa: E402
from src.exp005_positional_path_signal import (  # noqa: E402
    MASK_COLUMNS,
    PATH_MASK_COLUMNS,
    SCALED_RETURN_COLUMNS,
    PositionalPathTransformer,
    aggregate_primary_decision,
    aggregate_secondary_decision,
    assert_identical_representation_rows,
    build_original_mask,
    build_path_mask,
    delta_vs_baseline,
    evaluate_sign_predictions,
    make_positional_logistic_regression,
    positive_probability,
    prepare_evaluable_directional_rows,
    primary_fold_decision,
    secondary_fold_decision,
    sign_majority_class,
    threshold_predictions,
    validate_mask_matrix,
    validate_path_mask_matrix,
)


def assert_raises(action, message: str) -> None:
    try:
        action()
    except AssertionError as error:
        assert message in str(error)
        return
    raise AssertionError("Expected integrity assertion.")


def features_from_return_values(values: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(values, columns=RETURN_COLUMNS)
    frame.insert(0, "equity", np.arange(1, len(frame) + 1))
    frame.insert(0, "day", 0)
    frame.insert(0, "ID", np.arange(100, 100 + len(frame)))
    return frame


def full_return_values(n_rows: int) -> np.ndarray:
    return np.zeros((n_rows, len(RETURN_COLUMNS)), dtype=float)


def test_population_mask_and_frozen_split_boundaries() -> None:
    values = full_return_values(4)
    values[0, 2] = np.nan
    values[1, 0] = np.nan
    values[2, :] = np.nan
    features = features_from_return_values(values)
    reod = pd.Series([-1, 1, 1, 0], name="reod")
    prepared = prepare_evaluable_directional_rows(features, reod)
    assert prepared.sign_target.tolist() == [0, 1]
    assert prepared.raw_returns.index.tolist() == [0, 1]
    assert prepared.manifest["n_directional_before_evaluability"] == 3
    assert prepared.manifest["n_all_nan_directional_excluded"] == 1
    assert prepared.manifest["n_evaluable_directional_retained"] == 2
    mask = build_original_mask(prepared.raw_returns)
    assert list(mask.columns) == list(MASK_COLUMNS) and mask.shape == (2, 53)
    assert mask.loc[0, "m2"] == 1 and mask.loc[0, "m0"] == 0
    assert mask.loc[1, "m0"] == 1 and mask.loc[1, "m1"] == 0
    validate_mask_matrix(mask, expected_index=prepared.sign_target.index)

    rows, labels, identifier = [], [], 0
    for day in range(4):
        for equity in range(5):
            row = {"ID": identifier, "day": day, "equity": equity}
            row.update({column: 0.0 for column in RETURN_COLUMNS})
            rows.append(row)
            labels.append(-1 if identifier % 2 == 0 else 1)
            identifier += 1
    panel = pd.DataFrame(rows)
    partition = build_equity_partition(panel["equity"], n_holdout=1)
    subsets = split_fold(panel, pd.Series(labels, name="reod"), partition, FoldSpec(1, 0, 1, 2, 3))
    assert set(subsets.fit_features["equity"]).isdisjoint(
        set(partition.loc[partition["partition"].eq("E_holdout"), "equity"])
    )
    assert [(f.fold, f.train_start, f.train_end, f.oos_start, f.oos_end) for f in FOLDS] == [
        (1, 0, 302, 303, 352), (2, 0, 352, 353, 402),
        (3, 0, 402, 403, 452), (4, 0, 452, 453, 502),
    ]


def test_transformer_is_fit_only_and_rejects_undefined_median() -> None:
    fit_values = full_return_values(2)
    fit_values[:, 0] = [1.0, 3.0]
    fit_values[:, 1] = [np.nan, 5.0]
    fit_raw = pd.DataFrame(fit_values, columns=RETURN_COLUMNS, index=[10, 11])
    transformer = PositionalPathTransformer().fit(fit_raw)
    assert transformer.medians_.loc["r0"] == 2.0
    assert transformer.medians_.loc["r1"] == 5.0
    assert transformer.scaler_.mean_[0] == 2.0
    medians_before = transformer.medians_.copy()
    means_before = transformer.scaler_.mean_.copy()
    oos_values = full_return_values(1)
    oos_values[:, 0] = [1000.0]
    oos_values[:, 1] = [np.nan]
    oos_raw = pd.DataFrame(oos_values, columns=RETURN_COLUMNS, index=[99])
    transformed = transformer.transform_return_block(oos_raw)
    assert list(transformed.columns) == list(SCALED_RETURN_COLUMNS)
    assert np.isfinite(transformed.to_numpy()).all()
    assert transformer.medians_.equals(medians_before)
    assert np.array_equal(transformer.scaler_.mean_, means_before)

    undefined = fit_raw.copy()
    undefined["r52"] = np.nan
    assert_raises(lambda: PositionalPathTransformer().fit(undefined), "median must be finite")


def test_exact_path_mask_schema_preserves_raw_masks_and_row_identity() -> None:
    values = full_return_values(3)
    values[0, 0] = np.nan
    values[1, 1] = np.nan
    values[2, 2] = np.nan
    raw = pd.DataFrame(values, columns=RETURN_COLUMNS, index=[21, 34, 55])
    transformer = PositionalPathTransformer().fit(raw)
    a_mask = build_original_mask(raw)
    b_path_mask = build_path_mask(transformer, raw)
    target = pd.Series([0, 1, 0], index=raw.index, name="S", dtype="int8")
    assert b_path_mask.shape == (3, 106)
    assert list(b_path_mask.columns) == list(PATH_MASK_COLUMNS)
    assert b_path_mask.loc[21, "m0"] == 1 and b_path_mask.loc[21, "m1"] == 0
    assert b_path_mask.loc[34, "m1"] == 1 and b_path_mask.loc[34, "m0"] == 0
    assert set(b_path_mask.columns).isdisjoint({"R_obs", "P", "N_obs", "day", "equity", "ID"})
    assert_identical_representation_rows(a_mask, b_path_mask, target)
    validate_path_mask_matrix(b_path_mask, expected_index=raw.index)
    assert_raises(
        lambda: assert_identical_representation_rows(a_mask.iloc[::-1], b_path_mask, target),
        "identical row identities",
    )


def test_model_probability_threshold_metrics_and_fit_only_baseline() -> None:
    assert sign_majority_class(pd.Series([0, 1], name="S")) == 0
    classifier = make_positional_logistic_regression()
    matrix = pd.DataFrame(
        np.vstack([np.zeros(106), np.ones(106), np.zeros(106), np.ones(106)]),
        columns=PATH_MASK_COLUMNS,
    )
    target = pd.Series([0, 1, 0, 1], name="S")
    classifier.fit(matrix, target)
    probability = positive_probability(classifier, matrix)
    assert threshold_predictions(np.array([0.49, 0.5, 0.51])).tolist() == [0, 1, 1]
    assert_raises(lambda: threshold_predictions(probability, threshold=0.4), "exactly 0.5")
    metrics, confusion = evaluate_sign_predictions(
        pd.Series([0, 0, 1, 1]), np.array([0, 0, 1, 1]), probability=np.array([0.1, 0.4, 0.6, 0.8])
    )
    assert confusion == [[2, 0], [0, 2]] and metrics["roc_auc"] == 1.0


def test_decisions_deltas_and_no_historical_or_competition_dependency() -> None:
    primary = aggregate_primary_decision([0.51, 0.52, 0.53, 0.54], [0.01, 0.02, 0.03, 0.04])
    assert primary["incremental_positional_return_evidence"]
    assert not aggregate_primary_decision([0.51, 0.52, 0.53, 0.54], [0.01, 0.0, 0.03, 0.04])["incremental_positional_return_evidence"]
    assert not primary_fold_decision(0.5, 0.01)["path_roc_auc_above_one_half"]
    failed = {"balanced_accuracy": 0.5, "recall_negative": 1.0, "recall_positive": 0.0}
    assert not aggregate_secondary_decision([secondary_fold_decision(failed)] * 4, [0.01] * 4)[
        "fixed_threshold_path_sign_classification_evidence"
    ]
    assert np.isclose(delta_vs_baseline(0.55, 0.50), 0.05)
    runner = PROJECT_ROOT / "scripts" / "run_exp_005_positional_path_signal.py"
    assert runner.exists()
    runner_source = runner.read_text(encoding="utf-8")
    assert "input_test" not in runner_source
    assert "EXP_003_metrics" not in runner_source and "EXP_004_metrics" not in runner_source


if __name__ == "__main__":
    import unittest

    suite = unittest.TestSuite(unittest.FunctionTestCase(test) for test in (
        test_population_mask_and_frozen_split_boundaries,
        test_transformer_is_fit_only_and_rejects_undefined_median,
        test_exact_path_mask_schema_preserves_raw_masks_and_row_identity,
        test_model_probability_threshold_metrics_and_fit_only_baseline,
        test_decisions_deltas_and_no_historical_or_competition_dependency,
    ))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
