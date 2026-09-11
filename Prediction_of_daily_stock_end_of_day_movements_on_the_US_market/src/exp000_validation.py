"""Protocol utilities for the EXP_000 validation-baseline contract.

This module intentionally contains no predictive model or preprocessing logic.
It defines the frozen entity partition, temporal folds, integrity checks, and
the majority-class baseline required by EXP_000.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score


CLASS_ORDER: tuple[int, int, int] = (-1, 0, 1)
PARTITION_SEED = 20260908
N_HOLDOUT_EQUITIES = 366
N_DEV_EQUITIES = 1463


@dataclass(frozen=True)
class FoldSpec:
    """Inclusive chronological boundaries for one expanding EXP_000 fold."""

    fold: int
    train_start: int
    train_end: int
    oos_start: int
    oos_end: int


FOLDS: tuple[FoldSpec, ...] = (
    FoldSpec(1, 0, 302, 303, 352),
    FoldSpec(2, 0, 352, 353, 402),
    FoldSpec(3, 0, 402, 403, 452),
    FoldSpec(4, 0, 452, 453, 502),
)


@dataclass
class FoldSubsets:
    """Feature and target subsets dictated by one EXP_000 fold."""

    fold: FoldSpec
    fit_features: pd.DataFrame
    fit_target: pd.Series
    temporal_features: pd.DataFrame
    temporal_target: pd.Series
    joint_features: pd.DataFrame
    joint_target: pd.Series


def build_equity_partition(
    equity_ids: Iterable[int],
    *,
    seed: int = PARTITION_SEED,
    n_holdout: int = N_HOLDOUT_EQUITIES,
) -> pd.DataFrame:
    """Create the fixed EXP_000 equity partition from IDs only."""
    unique_equities = np.asarray(sorted(set(equity_ids)))
    if not 0 < n_holdout < len(unique_equities):
        raise ValueError("n_holdout must be strictly between 0 and the number of equities.")

    permutation = np.random.default_rng(seed).permutation(unique_equities)
    holdout = set(permutation[:n_holdout])
    partition = pd.DataFrame(
        {
            "equity": unique_equities,
            "partition": np.where(np.isin(unique_equities, list(holdout)), "E_holdout", "E_dev"),
        }
    )
    validate_equity_partition(partition, unique_equities, n_holdout=n_holdout)
    return partition


def validate_equity_partition(
    partition: pd.DataFrame,
    expected_equities: Iterable[int],
    *,
    n_holdout: int = N_HOLDOUT_EQUITIES,
) -> None:
    """Assert that a stored partition exactly covers the training universe."""
    required_columns = {"equity", "partition"}
    if not required_columns.issubset(partition.columns):
        raise AssertionError(f"Partition requires columns {required_columns}.")
    if partition["equity"].duplicated().any():
        raise AssertionError("An equity appears more than once in the partition.")
    if not partition["partition"].isin(["E_dev", "E_holdout"]).all():
        raise AssertionError("Partition labels must be E_dev or E_holdout.")

    expected = set(expected_equities)
    observed = set(partition["equity"])
    if observed != expected:
        raise AssertionError("Partition equities do not exactly match the training universe.")
    if int((partition["partition"] == "E_holdout").sum()) != n_holdout:
        raise AssertionError(f"Expected exactly {n_holdout} holdout equities.")
    if int((partition["partition"] == "E_dev").sum()) != len(expected) - n_holdout:
        raise AssertionError("Unexpected number of development equities.")


def split_fold(
    features: pd.DataFrame,
    target: pd.Series,
    partition: pd.DataFrame,
    fold: FoldSpec,
) -> FoldSubsets:
    """Construct fit, Temporal OOS, and Joint OOS without any refitting."""
    if not features.index.equals(target.index):
        raise AssertionError("Features and target must have identical indices.")
    if not {"ID", "day", "equity"}.issubset(features.columns):
        raise AssertionError("Features must include ID, day, and equity.")
    observed_holdout_size = int((partition["partition"] == "E_holdout").sum())
    validate_equity_partition(
        partition,
        features["equity"].unique(),
        n_holdout=observed_holdout_size,
    )

    dev_equities = set(partition.loc[partition["partition"] == "E_dev", "equity"])
    holdout_equities = set(partition.loc[partition["partition"] == "E_holdout", "equity"])
    train_days = features["day"].between(fold.train_start, fold.train_end)
    oos_days = features["day"].between(fold.oos_start, fold.oos_end)
    dev_rows = features["equity"].isin(dev_equities)
    holdout_rows = features["equity"].isin(holdout_equities)

    subsets = FoldSubsets(
        fold=fold,
        fit_features=features.loc[train_days & dev_rows].copy(),
        fit_target=target.loc[train_days & dev_rows].copy(),
        temporal_features=features.loc[oos_days & dev_rows].copy(),
        temporal_target=target.loc[oos_days & dev_rows].copy(),
        joint_features=features.loc[oos_days & holdout_rows].copy(),
        joint_target=target.loc[oos_days & holdout_rows].copy(),
    )
    assert_fold_integrity(subsets, dev_equities, holdout_equities)
    return subsets


def assert_fold_integrity(
    subsets: FoldSubsets,
    dev_equities: set[int],
    holdout_equities: set[int],
) -> None:
    """Assert the information boundaries defined in EXP_000."""
    fold = subsets.fold
    if not dev_equities.isdisjoint(holdout_equities):
        raise AssertionError("E_dev and E_holdout must be disjoint.")
    if fold.train_end >= fold.oos_start:
        raise AssertionError("Training days must end before OOS days begin.")

    named_frames = {
        "fit": subsets.fit_features,
        "temporal_oos": subsets.temporal_features,
        "joint_oos": subsets.joint_features,
    }
    for name, frame in named_frames.items():
        if frame.empty:
            raise AssertionError(f"{name} is empty for fold {fold.fold}.")
        if frame["ID"].duplicated().any():
            raise AssertionError(f"{name} contains duplicate IDs.")

    if not set(subsets.fit_features["equity"]).issubset(dev_equities):
        raise AssertionError("Holdout equity found in fit subset.")
    if not set(subsets.temporal_features["equity"]).issubset(dev_equities):
        raise AssertionError("Non-development equity found in Temporal OOS.")
    if not set(subsets.joint_features["equity"]).issubset(holdout_equities):
        raise AssertionError("Non-holdout equity found in Joint OOS.")
    if not subsets.fit_features["day"].between(fold.train_start, fold.train_end).all():
        raise AssertionError("Fit contains an OOS day.")
    if not subsets.temporal_features["day"].between(fold.oos_start, fold.oos_end).all():
        raise AssertionError("Temporal OOS contains a non-OOS day.")
    if not subsets.joint_features["day"].between(fold.oos_start, fold.oos_end).all():
        raise AssertionError("Joint OOS contains a non-OOS day.")

    id_sets = {name: set(frame["ID"]) for name, frame in named_frames.items()}
    if id_sets["fit"] & id_sets["temporal_oos"]:
        raise AssertionError("fit and Temporal OOS share IDs.")
    if id_sets["fit"] & id_sets["joint_oos"]:
        raise AssertionError("fit and Joint OOS share IDs.")
    if id_sets["temporal_oos"] & id_sets["joint_oos"]:
        raise AssertionError("Temporal OOS and Joint OOS share IDs.")


def majority_class(fit_target: pd.Series) -> int:
    """Return the fit-only majority class, breaking ties with ``[-1, 0, 1]``."""
    if fit_target.empty:
        raise AssertionError("Cannot compute a majority class from an empty fit target.")
    counts = fit_target.value_counts().reindex(CLASS_ORDER, fill_value=0)
    maximum = counts.max()
    return next(label for label in CLASS_ORDER if counts.loc[label] == maximum)


def class_counts(target: pd.Series) -> dict[str, int]:
    """Return all class counts in the stable contract order."""
    counts = target.value_counts().reindex(CLASS_ORDER, fill_value=0)
    return {f"n_class_{label}": int(counts.loc[label]) for label in CLASS_ORDER}


def evaluate_constant_baseline(target: pd.Series, predicted_class: int) -> tuple[dict[str, float], list[list[int]]]:
    """Evaluate constant predictions using the metrics fixed in EXP_000."""
    if target.empty:
        raise AssertionError("Cannot evaluate an empty target subset.")
    prediction = np.full(len(target), predicted_class, dtype=np.int8)
    truth = target.to_numpy(dtype=np.int8)
    metrics = {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
    }
    matrix = confusion_matrix(truth, prediction, labels=CLASS_ORDER).astype(int).tolist()
    return metrics, matrix


def fold_manifest_rows(subsets: FoldSubsets) -> list[dict[str, int | str]]:
    """Create row-count and class-distribution records for contract checks."""
    records: list[dict[str, int | str]] = []
    for subset_name, target in (
        ("fit", subsets.fit_target),
        ("temporal_oos", subsets.temporal_target),
        ("joint_oos", subsets.joint_target),
    ):
        record: dict[str, int | str] = {"fold": subsets.fold.fold, "subset": subset_name, "n_rows": int(len(target))}
        record.update(class_counts(target))
        records.append(record)
    return records


def with_injected_fit_row(subsets: FoldSubsets, feature_row: pd.DataFrame, target_row: pd.Series) -> FoldSubsets:
    """Test helper: return subsets with an extra fit row for integrity testing."""
    return replace(
        subsets,
        fit_features=pd.concat([subsets.fit_features, feature_row]),
        fit_target=pd.concat([subsets.fit_target, target_row]),
    )
