"""Synthetic integrity tests for Discovery-only D001."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import discovery.run_d001_path_structure_beyond_recent_rms as d001_runner  # noqa: E402

from discovery.d001_path_structure_beyond_recent_rms import (  # noqa: E402
    CONTROL_COLUMNS,
    DISCOVERY_FOLDS,
    PATH_COLUMNS,
    RECENT_WINDOW_COLUMNS,
    RECENT_WINDOW_SIZE,
    XGBOOST_PARAMS,
    assert_both_binary_classes,
    assert_identical_representation_rows,
    build_d001_representations,
    cross_fold_shap_summary,
    deterministic_sample_indices,
    explain_d001_path_validation,
    incremental_auc_records,
    make_d001_classifier,
    prepare_complete_w_population,
    split_discovery_fold,
    validate_discovery_scope,
)


def assert_raises(action, text: str) -> None:
    try:
        action()
    except AssertionError as error:
        assert text in str(error)
        return
    raise AssertionError("Expected assertion was not raised.")


def frame(rows: int, *, days: list[int] | None = None, equities: list[int] | None = None) -> pd.DataFrame:
    data = pd.DataFrame(np.zeros((rows, 53)), columns=[f"r{i}" for i in range(53)])
    data.insert(0, "equity", equities if equities is not None else [10] * rows)
    data.insert(0, "day", days if days is not None else list(range(rows)))
    data.insert(0, "ID", list(range(100, 100 + rows)))
    return data


def test_boundary_window_zero_and_infinity_semantics() -> None:
    assert RECENT_WINDOW_COLUMNS == tuple(f"r{i}" for i in range(41, 53))
    assert RECENT_WINDOW_SIZE == 12
    features = frame(3, days=[0, 352, 353], equities=[10, 10, 10])
    assert_raises(lambda: validate_discovery_scope(features, {10}), "outside 0-352")
    assert_raises(lambda: validate_discovery_scope(features.iloc[:2], {11}), "E_holdout")

    allowed = frame(3, days=[0, 1, 2], equities=[10, 10, 10])
    allowed.loc[0, list(RECENT_WINDOW_COLUMNS)] = np.nan
    allowed.loc[1, "r41"] = np.nan
    target = pd.Series([0, 1, 0], index=allowed.index, dtype="int8")
    retained, retained_target, structure = prepare_complete_w_population(allowed, target, {10})
    assert structure["N_obs_w"].tolist() == [0, 11, 12]
    assert retained.index.tolist() == [2] and retained_target.index.tolist() == [2]
    reps = build_d001_representations(retained)
    assert reps["C"].iloc[0, 0] == 0.0  # observed zeros remain observed.

    infinity = allowed.copy()
    infinity.loc[2, "r41"] = np.inf
    assert_raises(
        lambda: prepare_complete_w_population(infinity, target, {10}),
        "retained observed W values must be finite",
    )


def test_rms_absolute_path_alignment_and_dimensions() -> None:
    features = frame(2, days=[3, 4], equities=[10, 10])
    features.loc[0, list(RECENT_WINDOW_COLUMNS)] = np.arange(-6, 6, dtype=float)
    features.loc[1, list(RECENT_WINDOW_COLUMNS)] = -2.0
    target = pd.Series([0, 1], index=features.index, dtype="int8")
    retained, retained_target, _ = prepare_complete_w_population(features, target, {10})
    reps_a = build_d001_representations(retained)
    reps_b = build_d001_representations(retained)
    assert tuple(reps_a["C"].columns) == CONTROL_COLUMNS and reps_a["C"].shape[1] == 1
    assert tuple(reps_a["P"].columns) == PATH_COLUMNS and reps_a["P"].shape[1] == 12
    assert np.allclose(reps_a["P"].iloc[0].to_numpy(), np.abs(np.arange(-6, 6, dtype=float)))
    assert np.isclose(reps_a["C"].iloc[0, 0], np.sqrt(np.mean(np.arange(-6, 6, dtype=float) ** 2)))
    assert reps_a["C"].index.equals(reps_a["P"].index)
    assert reps_a["C"].equals(reps_b["C"]) and reps_a["P"].equals(reps_b["P"])
    assert_identical_representation_rows(reps_a, retained, retained_target)
    assert_raises(
        lambda: assert_identical_representation_rows({**reps_a, "P": reps_a["P"].iloc[::-1]}, retained, retained_target),
        "changed row identity or order",
    )


def test_folds_classes_and_exact_xgboost_configuration() -> None:
    assert [(fold.fit_start, fold.fit_end, fold.validation_start, fold.validation_end) for fold in DISCOVERY_FOLDS] == [
        (0, 202, 203, 252), (0, 252, 253, 302), (0, 302, 303, 352)
    ]
    rows = []
    labels = []
    for day in [0, 202, 203, 252, 253, 302, 303, 352]:
        for equity in [10, 11]:
            row = {"ID": len(rows), "day": day, "equity": equity}
            row.update({f"r{i}": 0.0 for i in range(53)})
            rows.append(row)
            labels.append(len(rows) % 2)
    features = pd.DataFrame(rows)
    target = pd.Series(labels, index=features.index, dtype="int8")
    for fold in DISCOVERY_FOLDS:
        (fit_features, fit_target), (validation_features, validation_target) = split_discovery_fold(features, target, {10, 11}, fold)
        assert fit_features["day"].max() < validation_features["day"].min()
        assert fit_features["day"].max() <= 302 and validation_features["day"].max() <= 352
        assert set(fit_features["equity"]).issubset({10, 11})
        assert_both_binary_classes(fit_target, context="synthetic fit")
        assert_both_binary_classes(validation_target, context="synthetic validation")
    model = make_d001_classifier()
    assert all(model.get_params()[key] == value for key, value in XGBOOST_PARAMS.items())
    assert model.get_params()["early_stopping_rounds"] is None
    runner_source = (PROJECT_ROOT / "discovery" / "run_d001_path_structure_beyond_recent_rms.py").read_text(encoding="utf-8")
    assert "input_test" not in runner_source and "output_test" not in runner_source


def test_incremental_records_and_synthetic_shap_support() -> None:
    records = incremental_auc_records({1: 0.1, 2: 0.2, 3: 0.3})
    assert records["record_type"].tolist() == ["fold", "fold", "fold", "aggregate_mean", "aggregate_std_ddof_1"]
    assert records["fold"].tolist() == [1, 2, 3, "all", "all"]
    assert np.isclose(records.iloc[3]["delta_p_minus_c_validation"], 0.2)
    assert np.isclose(records.iloc[4]["delta_p_minus_c_validation"], 0.1)

    large_index = pd.RangeIndex(2_101)
    global_a = deterministic_sample_indices(large_index, maximum=2_000, seed=20260909)
    global_b = deterministic_sample_indices(large_index, maximum=2_000, seed=20260909)
    interaction = deterministic_sample_indices(global_a, maximum=500, seed=20261909)
    assert global_a.equals(global_b) and len(global_a) == 2_000
    assert len(interaction) == 500 and interaction.isin(global_a).all()

    rng = np.random.default_rng(20260908)
    matrix = pd.DataFrame(np.abs(rng.normal(size=(64, 12))), columns=PATH_COLUMNS)
    signal = matrix["abs_r41"] + matrix["abs_r48"]
    target = (signal > signal.median()).astype("int8")
    temporary = XGBClassifier(
        objective="binary:logistic", eval_metric="auc", n_estimators=6,
        max_depth=2, min_child_weight=1, learning_rate=0.2, tree_method="hist",
        n_jobs=1, random_state=20260908, verbosity=0,
    ).fit(matrix, target)
    result = explain_d001_path_validation(temporary, matrix, fold_number=1)
    explanation = result["global_explanation"]
    assert explanation.values.shape == (64, 12) and np.isfinite(explanation.values).all()
    assert result["interaction_values"].shape == (64, 12, 12)
    assert result["interaction_index"].isin(result["global_index"]).all()
    summary = cross_fold_shap_summary({1: result["mean_abs_shap"], 2: result["mean_abs_shap"], 3: result["mean_abs_shap"]})
    assert summary.index.tolist() == list(PATH_COLUMNS) and np.isfinite(summary.to_numpy(dtype=float)).all()
    source = (PROJECT_ROOT / "discovery" / "d001_path_structure_beyond_recent_rms.py").read_text(encoding="utf-8")
    assert "input_training" not in source and "output_training" not in source


def test_physical_discovery_loader_fails_loud_without_partition() -> None:
    runner_source = (PROJECT_ROOT / "discovery" / "run_d001_path_structure_beyond_recent_rms.py").read_text(encoding="utf-8")
    assert '"input_training.csv"' not in runner_source
    assert '"output_training_*.csv"' not in runner_source
    assert "DISCOVERY_INPUT_PATH" in runner_source and "DISCOVERY_LABEL_PATH" in runner_source
    originals = (
        d001_runner.DISCOVERY_INPUT_PATH,
        d001_runner.DISCOVERY_LABEL_PATH,
        d001_runner.DISCOVERY_MANIFEST_PATH,
    )
    try:
        missing = PROJECT_ROOT / "discovery" / "tests" / "does_not_exist.csv"
        d001_runner.DISCOVERY_INPUT_PATH = missing
        d001_runner.DISCOVERY_LABEL_PATH = missing
        d001_runner.DISCOVERY_MANIFEST_PATH = missing
        try:
            d001_runner.load_discovery_rows()
        except FileNotFoundError as error:
            assert "physical Discovery partition" in str(error)
        else:
            raise AssertionError("D001 loader must fail without a physical Discovery partition.")
    finally:
        (
            d001_runner.DISCOVERY_INPUT_PATH,
            d001_runner.DISCOVERY_LABEL_PATH,
            d001_runner.DISCOVERY_MANIFEST_PATH,
        ) = originals


if __name__ == "__main__":
    import unittest

    suite = unittest.TestSuite(unittest.FunctionTestCase(test) for test in (
        test_boundary_window_zero_and_infinity_semantics,
        test_rms_absolute_path_alignment_and_dimensions,
        test_folds_classes_and_exact_xgboost_configuration,
        test_incremental_records_and_synthetic_shap_support,
        test_physical_discovery_loader_fails_loud_without_partition,
    ))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
