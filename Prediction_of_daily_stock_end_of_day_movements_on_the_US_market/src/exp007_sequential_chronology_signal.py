"""Frozen utilities for EXP_007 sequential chronology signal.

This module deliberately accepts fit tensors only in its training API. OOS
tensors are evaluated by separate functions after the fixed ten epochs.
"""

from __future__ import annotations

import copy
import platform
import random
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch import nn

from src.exp001_missingness import strictly_greater
from src.exp003_conditional_directional_sign import (
    PREDICTION_THRESHOLD,
    SIGN_CLASS_ORDER,
    assert_both_sign_classes,
    threshold_predictions,
)
from src.exp005_positional_path_signal import (
    MASK_COLUMNS,
    SCALED_RETURN_COLUMNS,
    PositionalPathTransformer,
    build_original_mask,
)


REQUIRED_PYTHON_VERSION = "3.13.2"
REQUIRED_NUMPY_VERSION = "2.2.3"
REQUIRED_TORCH_VERSION = "2.12.0+cpu"
MODEL_SEED = 20260908
BATCH_ORDER_SEED = 20260909
PERMUTATION_SEED = 20260908
BATCH_SIZE = 512
N_EPOCHS = 10
SEQUENCE_LENGTH = 53
N_INPUT_FEATURES = 2
HIDDEN_SIZE = 16
N_TRAINABLE_PARAMETERS = 977

PERMUTATION: tuple[int, ...] = (
    13, 44, 24, 4, 37, 23, 17, 50, 12, 6, 46, 7, 32, 5, 2, 33,
    21, 40, 41, 51, 35, 11, 18, 3, 16, 49, 42, 30, 31, 14, 19, 20,
    15, 48, 45, 47, 27, 39, 29, 9, 43, 10, 1, 25, 22, 28, 26, 8,
    34, 38, 36, 0, 52,
)


def assert_runtime_versions() -> None:
    """Require the frozen deterministic runtime versions."""
    if platform.python_version() != REQUIRED_PYTHON_VERSION:
        raise AssertionError(
            f"EXP_007 requires Python {REQUIRED_PYTHON_VERSION}, found {platform.python_version()}."
        )
    if np.__version__ != REQUIRED_NUMPY_VERSION:
        raise AssertionError(f"EXP_007 requires NumPy {REQUIRED_NUMPY_VERSION}, found {np.__version__}.")
    if torch.__version__ != REQUIRED_TORCH_VERSION:
        raise AssertionError(f"EXP_007 requires PyTorch {REQUIRED_TORCH_VERSION}, found {torch.__version__}.")


def configure_deterministic_cpu() -> None:
    """Configure the frozen CPU-only deterministic execution environment."""
    assert_runtime_versions()
    if torch.cuda.is_available():
        raise AssertionError("EXP_007 is frozen for CPU execution only.")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise AssertionError("EXP_007 requires one CPU and one interop thread.")


def reset_model_rng() -> None:
    """Reset every relevant model-initialization RNG to the frozen seed."""
    random.seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)
    torch.manual_seed(MODEL_SEED)


def epoch_seed(*, fold_number: int, epoch_number: int) -> int:
    if fold_number not in (1, 2, 3, 4) or not 1 <= epoch_number <= N_EPOCHS:
        raise AssertionError("EXP_007 batch-order seed requires a frozen fold and epoch.")
    return BATCH_ORDER_SEED + 1000 * fold_number + epoch_number


def deterministic_epoch_order(*, n_rows: int, fold_number: int, epoch_number: int) -> np.ndarray:
    if n_rows <= 0:
        raise AssertionError("Cannot construct an EXP_007 batch order for zero fit rows.")
    return np.random.default_rng(epoch_seed(fold_number=fold_number, epoch_number=epoch_number)).permutation(n_rows)


def _longest_order_run(permutation: Sequence[int]) -> int:
    longest = current = 1
    for left, right in zip(permutation, permutation[1:]):
        if abs(right - left) == 1:
            current += 1
        else:
            current = 1
        longest = max(longest, current)
    return longest


def permutation_diagnostics() -> dict[str, int | bool]:
    values = np.asarray(PERMUTATION, dtype=np.int64)
    return {
        "length": int(len(values)),
        "n_unique": int(np.unique(values).size),
        "is_identity": bool(np.array_equal(values, np.arange(SEQUENCE_LENGTH))),
        "n_fixed_points": int(np.sum(values == np.arange(SEQUENCE_LENGTH))),
        "n_adjacent_original_pairs": int(np.sum(np.abs(np.diff(values)) == 1)),
        "longest_original_order_run": _longest_order_run(PERMUTATION),
    }


def assert_frozen_permutation() -> None:
    regenerated = tuple(np.random.default_rng(PERMUTATION_SEED).permutation(SEQUENCE_LENGTH).tolist())
    if PERMUTATION != regenerated:
        raise AssertionError("EXP_007 permutation differs from the frozen seed-generated vector.")
    diagnostics = permutation_diagnostics()
    expected = {
        "length": 53,
        "n_unique": 53,
        "is_identity": False,
        "n_fixed_points": 1,
        "n_adjacent_original_pairs": 3,
        "longest_original_order_run": 2,
    }
    if diagnostics != expected:
        raise AssertionError("EXP_007 permutation structural diagnostics differ from the frozen contract.")


def build_real_sequences(
    transformer: PositionalPathTransformer, raw_returns: pd.DataFrame
) -> tuple[np.ndarray, pd.Index]:
    """Create `N x 53 x 2` Real sequences from one fit-only transformer."""
    masks = build_original_mask(raw_returns)
    scaled = transformer.transform_return_block(raw_returns)
    if not masks.index.equals(scaled.index) or list(scaled.columns) != list(SCALED_RETURN_COLUMNS):
        raise AssertionError("EXP_007 return and mask blocks must preserve identical row identity.")
    if list(masks.columns) != list(MASK_COLUMNS):
        raise AssertionError("EXP_007 masks must be exactly m0 through m52.")
    values = scaled.to_numpy(dtype=np.float32)
    mask_values = masks.to_numpy(dtype=np.float32)
    sequence = np.stack([values, mask_values], axis=-1).astype(np.float32, copy=False)
    validate_real_sequences(sequence, expected_n_rows=len(raw_returns))
    return sequence, raw_returns.index


def validate_real_sequences(sequence: np.ndarray, *, expected_n_rows: int | None = None) -> None:
    if sequence.ndim != 3 or sequence.shape[1:] != (SEQUENCE_LENGTH, N_INPUT_FEATURES):
        raise AssertionError("EXP_007 Real sequence must have shape (N, 53, 2).")
    if expected_n_rows is not None and sequence.shape[0] != expected_n_rows:
        raise AssertionError("EXP_007 sequence changed retained row count.")
    if sequence.dtype != np.float32 or not np.isfinite(sequence).all():
        raise AssertionError("EXP_007 sequence must be finite float32.")
    masks = sequence[:, :, 1]
    if not np.isin(masks, [0.0, 1.0]).all():
        raise AssertionError("EXP_007 mask channel must remain original binary and unscaled.")


def build_permuted_sequences(real_sequence: np.ndarray) -> np.ndarray:
    assert_frozen_permutation()
    validate_real_sequences(real_sequence)
    permuted = real_sequence[:, np.asarray(PERMUTATION, dtype=np.int64), :].copy()
    assert_paired_sequences(real_sequence, permuted)
    return permuted


def assert_paired_sequences(real_sequence: np.ndarray, permuted_sequence: np.ndarray) -> None:
    validate_real_sequences(real_sequence)
    validate_real_sequences(permuted_sequence, expected_n_rows=real_sequence.shape[0])
    expected = real_sequence[:, np.asarray(PERMUTATION, dtype=np.int64), :]
    if not np.array_equal(permuted_sequence, expected):
        raise AssertionError("EXP_007 Permuted sequence must apply pi jointly to return/mask pairs.")
    # Sorting rows lexicographically is unnecessary: an indexed permutation already
    # proves preservation of the full 53 paired observations for each row.
    for row in range(real_sequence.shape[0]):
        real_pairs = real_sequence[row]
        permuted_pairs = permuted_sequence[row]
        if not np.array_equal(permuted_pairs, real_pairs[np.asarray(PERMUTATION)]):
            raise AssertionError("EXP_007 Permuted row did not preserve its paired observations.")


def assert_paired_rows(
    real_index: pd.Index, permuted_index: pd.Index, real_target: pd.Series, permuted_target: pd.Series
) -> None:
    if not real_index.equals(permuted_index) or not real_index.equals(real_target.index) or not real_index.equals(permuted_target.index):
        raise AssertionError("EXP_007 Real and Permuted rows must retain identical identities and order.")
    if not real_target.equals(permuted_target):
        raise AssertionError("EXP_007 Real and Permuted targets must be identical.")


class GRUSignClassifier(nn.Module):
    """The exact frozen one-layer unidirectional EXP_007 GRU."""

    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=N_INPUT_FEATURES,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.head = nn.Linear(HIDDEN_SIZE, 1)
        if sum(parameter.numel() for parameter in self.parameters()) != N_TRAINABLE_PARAMETERS:
            raise AssertionError("EXP_007 GRU parameter count must be exactly 977.")

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3 or tuple(sequence.shape[1:]) != (SEQUENCE_LENGTH, N_INPUT_FEATURES):
            raise AssertionError("EXP_007 GRU expects tensors of shape (N, 53, 2).")
        _, hidden = self.gru(sequence)
        return self.head(hidden[-1]).squeeze(-1)


def make_adam(model: nn.Module) -> torch.optim.Adam:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
        foreach=False,
        fused=False,
    )
    expected: Mapping[str, object] = {
        "lr": 0.001,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0,
        "amsgrad": False,
        "foreach": False,
        "fused": False,
    }
    for name, value in expected.items():
        if optimizer.defaults.get(name) != value:
            raise AssertionError(f"EXP_007 Adam parameter {name!r} differs from the frozen contract.")
    return optimizer


def build_paired_models() -> tuple[GRUSignClassifier, GRUSignClassifier]:
    reset_model_rng()
    template = GRUSignClassifier()
    state = copy.deepcopy(template.state_dict())
    real_model = GRUSignClassifier()
    permuted_model = GRUSignClassifier()
    real_model.load_state_dict(copy.deepcopy(state))
    permuted_model.load_state_dict(copy.deepcopy(state))
    assert_identical_initial_state(real_model, permuted_model)
    return real_model, permuted_model


def assert_identical_initial_state(real_model: nn.Module, permuted_model: nn.Module) -> None:
    if real_model is permuted_model:
        raise AssertionError("EXP_007 Real and Permuted models must be independent objects.")
    real_state = real_model.state_dict()
    permuted_state = permuted_model.state_dict()
    if real_state.keys() != permuted_state.keys():
        raise AssertionError("EXP_007 paired models do not share the same parameter schema.")
    for name in real_state:
        if not torch.equal(real_state[name], permuted_state[name]):
            raise AssertionError(f"EXP_007 initial state differs for {name}.")
    for real_parameter, permuted_parameter in zip(real_model.parameters(), permuted_model.parameters()):
        if real_parameter.data_ptr() == permuted_parameter.data_ptr():
            raise AssertionError("EXP_007 paired models share parameter storage.")


def _as_fit_tensors(sequence: np.ndarray, target: pd.Series) -> tuple[torch.Tensor, torch.Tensor]:
    validate_real_sequences(sequence, expected_n_rows=len(target))
    assert_both_sign_classes(target, context="EXP_007 fit")
    tensor_x = torch.from_numpy(sequence)
    tensor_y = torch.from_numpy(target.to_numpy(dtype=np.float32, copy=True))
    return tensor_x, tensor_y


def _full_fit_diagnostics(model: nn.Module, features: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(features)
        full_loss = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="mean")
        probability = torch.sigmoid(logits).cpu().numpy()
    truth = target.cpu().numpy().astype(np.int8)
    prediction = threshold_predictions(probability)
    return {
        "full_fit_bce": float(full_loss.item()),
        "fit_roc_auc": float(roc_auc_score(truth, probability)),
        "fit_balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
    }


def train_paired_models(
    real_model: GRUSignClassifier,
    permuted_model: GRUSignClassifier,
    real_sequence: np.ndarray,
    permuted_sequence: np.ndarray,
    fit_target: pd.Series,
    *,
    fold_number: int,
) -> list[dict[str, int | float | str]]:
    """Train exactly ten epochs using fit tensors only; no OOS argument exists."""
    assert_paired_sequences(real_sequence, permuted_sequence)
    real_x, target = _as_fit_tensors(real_sequence, fit_target)
    permuted_x, permuted_target = _as_fit_tensors(permuted_sequence, fit_target)
    if not torch.equal(target, permuted_target):
        raise AssertionError("EXP_007 paired training targets differ.")
    assert_identical_initial_state(real_model, permuted_model)
    real_optimizer = make_adam(real_model)
    permuted_optimizer = make_adam(permuted_model)
    if real_optimizer is permuted_optimizer:
        raise AssertionError("EXP_007 paired models require independent optimizers.")
    loss_function = nn.BCEWithLogitsLoss(reduction="mean")
    records: list[dict[str, int | float | str]] = []
    expected_updates = int(np.ceil(len(fit_target) / BATCH_SIZE))

    for epoch_number in range(1, N_EPOCHS + 1):
        order = deterministic_epoch_order(
            n_rows=len(fit_target), fold_number=fold_number, epoch_number=epoch_number
        )
        real_model.train()
        permuted_model.train()
        loss_sums = {"real": 0.0, "permuted": 0.0}
        update_count = 0
        for start in range(0, len(order), BATCH_SIZE):
            batch_indices = torch.from_numpy(order[start : start + BATCH_SIZE].astype(np.int64, copy=False))
            batch_target = target[batch_indices]
            for condition, model, optimizer, features in (
                ("real", real_model, real_optimizer, real_x),
                ("permuted", permuted_model, permuted_optimizer, permuted_x),
            ):
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(model(features[batch_indices]), batch_target)
                loss.backward()
                optimizer.step()
                loss_sums[condition] += float(loss.item()) * len(batch_indices)
            update_count += 1
        if update_count != expected_updates:
            raise AssertionError("EXP_007 update count differs from the fixed batch schedule.")

        # Both models have finished their epoch before these final-state diagnostics.
        for condition, model, features in (
            ("real", real_model, real_x),
            ("permuted", permuted_model, permuted_x),
        ):
            diagnostics = _full_fit_diagnostics(model, features, target)
            records.append(
                {
                    "fold": fold_number,
                    "condition": condition,
                    "epoch": epoch_number,
                    "epoch_seed": epoch_seed(fold_number=fold_number, epoch_number=epoch_number),
                    "n_fit_rows": int(len(fit_target)),
                    "n_updates": update_count,
                    "optimization_loss": loss_sums[condition] / len(fit_target),
                    **diagnostics,
                }
            )
    if len(records) != 2 * N_EPOCHS:
        raise AssertionError("EXP_007 must record both conditions for exactly ten epochs.")
    return records


def predict_positive_probability(model: nn.Module, sequence: np.ndarray) -> np.ndarray:
    """Generate P(S=1) after training; the caller controls OOS timing."""
    validate_real_sequences(sequence)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(sequence))
        probability = torch.sigmoid(logits).cpu().numpy()
    if not np.isfinite(probability).all() or ((probability < 0.0) | (probability > 1.0)).any():
        raise AssertionError("EXP_007 probabilities must be finite P(S=1) values in [0, 1].")
    return probability


def primary_fold_decision(real_auc: float, real_minus_permuted_auc: float) -> dict[str, bool]:
    return {
        "real_roc_auc_above_one_half": strictly_greater(real_auc, 0.5),
        "real_minus_permuted_auc_positive": strictly_greater(real_minus_permuted_auc, 0.0),
    }


def aggregate_primary_decision(
    real_aucs: Sequence[float], real_minus_permuted_aucs: Sequence[float]
) -> dict[str, int | float | bool]:
    if len(real_aucs) != 4 or len(real_minus_permuted_aucs) != 4:
        raise AssertionError("EXP_007 requires exactly four Joint OOS folds.")
    mean_auc = float(np.mean(real_aucs))
    mean_delta = float(np.mean(real_minus_permuted_aucs))
    all_auc = all(strictly_greater(value, 0.5) for value in real_aucs)
    all_delta = all(strictly_greater(value, 0.0) for value in real_minus_permuted_aucs)
    return {
        "n_joint_folds": 4,
        "all_four_real_roc_auc_above_one_half": all_auc,
        "mean_joint_real_roc_auc": mean_auc,
        "mean_joint_real_roc_auc_above_one_half": strictly_greater(mean_auc, 0.5),
        "all_four_real_minus_permuted_auc_positive": all_delta,
        "mean_joint_real_minus_permuted_auc": mean_delta,
        "mean_joint_real_minus_permuted_auc_positive": strictly_greater(mean_delta, 0.0),
        "chronological_organization_evidence": (
            all_auc and strictly_greater(mean_auc, 0.5) and all_delta and strictly_greater(mean_delta, 0.0)
        ),
    }


def secondary_fold_decision(real_metrics: Mapping[str, float]) -> dict[str, bool]:
    result = {
        "balanced_accuracy_above_one_half": strictly_greater(real_metrics["balanced_accuracy"], 0.5),
        "recall_negative_positive": real_metrics["recall_negative"] > 0.0,
        "recall_positive_positive": real_metrics["recall_positive"] > 0.0,
    }
    result["all_fold_conditions_met"] = all(result.values())
    return result


def aggregate_secondary_decision(
    fold_decisions: Sequence[Mapping[str, bool]], real_ba_deltas: Sequence[float]
) -> dict[str, int | float | bool]:
    if len(fold_decisions) != 4 or len(real_ba_deltas) != 4:
        raise AssertionError("EXP_007 requires exactly four Joint OOS folds.")
    mean_delta = float(np.mean(real_ba_deltas))
    all_conditions = all(bool(decision["all_fold_conditions_met"]) for decision in fold_decisions)
    return {
        "n_joint_folds": 4,
        "all_four_fixed_threshold_conditions_met": all_conditions,
        "mean_joint_real_balanced_accuracy_delta_vs_baseline": mean_delta,
        "mean_joint_real_balanced_accuracy_delta_positive": strictly_greater(mean_delta, 0.0),
        "fixed_threshold_real_sign_classification_evidence": (
            all_conditions and strictly_greater(mean_delta, 0.0)
        ),
    }


__all__ = [
    "BATCH_ORDER_SEED", "BATCH_SIZE", "GRUSignClassifier", "HIDDEN_SIZE", "MODEL_SEED",
    "N_EPOCHS", "N_TRAINABLE_PARAMETERS", "PERMUTATION", "PERMUTATION_SEED",
    "aggregate_primary_decision", "aggregate_secondary_decision", "assert_frozen_permutation",
    "assert_identical_initial_state", "assert_paired_rows", "assert_paired_sequences",
    "assert_runtime_versions", "build_paired_models", "build_permuted_sequences", "build_real_sequences",
    "configure_deterministic_cpu", "deterministic_epoch_order", "epoch_seed", "make_adam",
    "permutation_diagnostics", "predict_positive_probability", "primary_fold_decision",
    "secondary_fold_decision", "train_paired_models", "validate_real_sequences",
]
