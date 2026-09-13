"""Synthetic contract tests for frozen EXP_007 sequential chronology signal."""

from inspect import signature
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.exp001_missingness import RETURN_COLUMNS  # noqa: E402
from src.exp003_conditional_directional_sign import evaluate_sign_predictions, threshold_predictions  # noqa: E402
from src.exp005_positional_path_signal import PositionalPathTransformer  # noqa: E402
from src.exp007_sequential_chronology_signal import (  # noqa: E402
    BATCH_SIZE,
    N_EPOCHS,
    N_TRAINABLE_PARAMETERS,
    PERMUTATION,
    GRUSignClassifier,
    aggregate_primary_decision,
    aggregate_secondary_decision,
    assert_frozen_permutation,
    assert_identical_initial_state,
    assert_paired_rows,
    assert_paired_sequences,
    assert_runtime_versions,
    build_paired_models,
    build_permuted_sequences,
    build_real_sequences,
    deterministic_epoch_order,
    epoch_seed,
    make_adam,
    permutation_diagnostics,
    predict_positive_probability,
    primary_fold_decision,
    secondary_fold_decision,
    train_paired_models,
)


def synthetic_raw_returns(n_rows: int = 6) -> pd.DataFrame:
    values = np.tile(np.arange(len(RETURN_COLUMNS), dtype=float), (n_rows, 1))
    values += np.arange(n_rows, dtype=float)[:, None]
    values[0, 0] = np.nan
    values[1, 7] = np.nan
    values[2, 52] = np.nan
    return pd.DataFrame(values, columns=RETURN_COLUMNS, index=np.arange(100, 100 + n_rows))


def synthetic_target(index: pd.Index) -> pd.Series:
    return pd.Series([0, 1] * (len(index) // 2) + ([0] if len(index) % 2 else []), index=index, name="S", dtype="int8")


def test_runtime_model_and_exact_permutation_contract() -> None:
    assert_runtime_versions()
    assert_frozen_permutation()
    diagnostics = permutation_diagnostics()
    assert diagnostics == {
        "length": 53,
        "n_unique": 53,
        "is_identity": False,
        "n_fixed_points": 1,
        "n_adjacent_original_pairs": 3,
        "longest_original_order_run": 2,
    }
    model = GRUSignClassifier()
    assert sum(parameter.numel() for parameter in model.parameters()) == N_TRAINABLE_PARAMETERS == 977
    assert model.gru.input_size == 2 and model.gru.hidden_size == 16
    assert model.gru.num_layers == 1 and model.gru.batch_first and not model.gru.bidirectional
    assert model.gru.dropout == 0.0 and model.head.in_features == 16 and model.head.out_features == 1


def test_real_permuted_pairing_preserves_rows_labels_and_pairs() -> None:
    raw = synthetic_raw_returns()
    target = synthetic_target(raw.index)
    transformer = PositionalPathTransformer().fit(raw)
    real, index = build_real_sequences(transformer, raw)
    permuted = build_permuted_sequences(real)
    assert real.shape == permuted.shape == (6, 53, 2)
    assert real.dtype == permuted.dtype == np.float32
    assert_paired_sequences(real, permuted)
    assert np.array_equal(permuted[:, 0, :], real[:, PERMUTATION[0], :])
    assert_paired_rows(index, index.copy(), target, target.copy())


def test_paired_initialization_and_independent_optimizers() -> None:
    real_model, permuted_model = build_paired_models()
    assert_identical_initial_state(real_model, permuted_model)
    assert real_model is not permuted_model
    for left, right in zip(real_model.parameters(), permuted_model.parameters()):
        assert left.data_ptr() != right.data_ptr()
        assert torch.equal(left, right)
    real_optimizer = make_adam(real_model)
    permuted_optimizer = make_adam(permuted_model)
    assert real_optimizer is not permuted_optimizer
    assert real_optimizer.defaults["foreach"] is False and real_optimizer.defaults["fused"] is False


def test_deterministic_epoch_orders_and_exact_ten_epoch_paired_training() -> None:
    target = pd.Series([0, 1, 0, 1, 0, 1], name="S", dtype="int8")
    # Every position within a row is equal, so applying the frozen permutation
    # leaves the tensor unchanged while still satisfying paired-sequence checks.
    real = np.zeros((6, 53, 2), dtype=np.float32)
    real[:, :, 0] = np.arange(6, dtype=np.float32)[:, None]
    # Equal paired inputs make equal terminal parameters a direct check that
    # initialization, targets, and minibatch indices were shared.
    permuted = build_permuted_sequences(real)
    first = deterministic_epoch_order(n_rows=len(target), fold_number=2, epoch_number=3)
    second = deterministic_epoch_order(n_rows=len(target), fold_number=2, epoch_number=3)
    assert np.array_equal(first, second)
    assert epoch_seed(fold_number=2, epoch_number=3) == 20262912
    real_model, permuted_model = build_paired_models()
    records = train_paired_models(real_model, permuted_model, real, permuted, target, fold_number=1)
    assert len(records) == 2 * N_EPOCHS
    assert {record["epoch"] for record in records} == set(range(1, 11))
    assert all(record["n_updates"] == 1 for record in records)
    assert all("optimization_loss" in record and "full_fit_bce" in record for record in records)
    for left, right in zip(real_model.parameters(), permuted_model.parameters()):
        assert torch.equal(left, right)


def test_training_api_has_no_oos_arguments_and_probability_metrics_are_oriented_to_s_one() -> None:
    parameters = set(signature(train_paired_models).parameters)
    assert not any("oos" in parameter or "temporal" in parameter or "joint" in parameter for parameter in parameters)
    model = GRUSignClassifier()
    sequence = np.zeros((4, 53, 2), dtype=np.float32)
    probability = predict_positive_probability(model, sequence)
    assert probability.shape == (4,) and np.isfinite(probability).all()
    assert threshold_predictions(np.array([0.49, 0.5, 0.51])).tolist() == [0, 1, 1]
    metrics, matrix = evaluate_sign_predictions(
        pd.Series([0, 0, 1, 1]),
        np.array([0, 0, 1, 1]),
        probability=np.array([0.1, 0.4, 0.6, 0.8]),
    )
    assert metrics["roc_auc"] == 1.0 and matrix == [[2, 0], [0, 2]]


def test_primary_secondary_decisions_and_runner_is_training_only() -> None:
    passed = aggregate_primary_decision([0.51, 0.52, 0.53, 0.54], [0.01, 0.02, 0.03, 0.04])
    assert passed["chronological_organization_evidence"]
    assert not aggregate_primary_decision([0.51, 0.52, 0.53, 0.54], [0.01, 0.0, 0.03, 0.04])["chronological_organization_evidence"]
    assert not primary_fold_decision(0.5, 0.01)["real_roc_auc_above_one_half"]
    failed = {"balanced_accuracy": 0.5, "recall_negative": 1.0, "recall_positive": 0.0}
    assert not aggregate_secondary_decision([secondary_fold_decision(failed)] * 4, [0.01] * 4)[
        "fixed_threshold_real_sign_classification_evidence"
    ]
    runner = PROJECT_ROOT / "scripts" / "run_exp_007_sequential_chronology_signal.py"
    source = runner.read_text(encoding="utf-8")
    assert "input_test" not in source and "EXP_006_metrics" not in source
    assert BATCH_SIZE == 512


if __name__ == "__main__":
    import unittest

    suite = unittest.TestSuite(unittest.FunctionTestCase(test) for test in (
        test_runtime_model_and_exact_permutation_contract,
        test_real_permuted_pairing_preserves_rows_labels_and_pairs,
        test_paired_initialization_and_independent_optimizers,
        test_deterministic_epoch_orders_and_exact_ten_epoch_paired_training,
        test_training_api_has_no_oos_arguments_and_probability_metrics_are_oriented_to_s_one,
        test_primary_secondary_decisions_and_runner_is_training_only,
    ))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
