"""Bounded, validation-only real SHAP interpretation for the frozen D001 P models.

This runner reads only the physically materialized Discovery partition.  It
reconstructs P_D001 models solely because the original fitted estimators were
not persisted, verifies their historical validation AUCs before any SHAP call,
and performs no new predictive comparison or feature engineering.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import shap
import sklearn
import xgboost


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from discovery.d001_path_structure_beyond_recent_rms import (  # noqa: E402
    DISCOVERY_FOLDS,
    GLOBAL_SHAP_SAMPLE_MAX,
    GLOBAL_SHAP_SEED,
    INTERACTION_SHAP_SAMPLE_MAX,
    INTERACTION_SHAP_SEED,
    PATH_COLUMNS,
    assert_both_binary_classes,
    assert_identical_representation_rows,
    build_d001_representations,
    cross_fold_shap_summary,
    explain_d001_path_validation,
    make_d001_classifier,
    prepare_complete_w_population,
    split_discovery_fold,
)
from discovery.run_d001_path_structure_beyond_recent_rms import (  # noqa: E402
    DISCOVERY_INPUT_PATH,
    DISCOVERY_LABEL_PATH,
    DISCOVERY_MANIFEST_PATH,
    PARTITION_PATH,
    RESULTS_DIR,
    load_discovery_rows,
)


METRICS_PATH = RESULTS_DIR / "D001_metrics.csv"
AUC_TOLERANCE = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_physical_discovery_boundary() -> dict[str, object]:
    """Verify only the physical partition, hashes, and frozen partition reference."""
    if not all(path.exists() for path in (DISCOVERY_INPUT_PATH, DISCOVERY_LABEL_PATH, DISCOVERY_MANIFEST_PATH, PARTITION_PATH)):
        raise FileNotFoundError("D001 SHAP requires the physical Discovery input, labels, manifest, and frozen partition.")
    manifest = json.loads(DISCOVERY_MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = {
        "row_count": 472_816,
        "unique_id_count": 472_816,
        "unique_day_count": 353,
        "unique_equity_count": 1_463,
        "authorized_day_min": 0,
        "authorized_day_max": 352,
        "e_dev_count": 1_463,
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise AssertionError(f"D001 physical manifest {name!r} differs from the frozen boundary.")
    hashes = {
        "input_sha256": sha256_file(DISCOVERY_INPUT_PATH),
        "label_sha256": sha256_file(DISCOVERY_LABEL_PATH),
        "equity_partition_sha256": sha256_file(PARTITION_PATH),
    }
    if any(manifest.get(name) != value for name, value in hashes.items()):
        raise AssertionError("D001 physical Discovery hash validation failed.")
    return manifest


def authoritative_p_validation_auc() -> dict[int, float]:
    """Read only historical D001 P validation AUCs for reconstruction verification."""
    if not METRICS_PATH.exists():
        raise FileNotFoundError("D001 SHAP reconstruction requires persisted D001_metrics.csv.")
    metrics = pd.read_csv(METRICS_PATH)
    rows = metrics.loc[metrics["condition"].eq("P"), ["fold", "auc_validation"]]
    if set(rows["fold"]) != {1, 2, 3} or rows["fold"].duplicated().any():
        raise AssertionError("D001 persisted P validation AUCs must contain exactly folds 1-3.")
    return {int(row.fold): float(row.auc_validation) for row in rows.itertuples(index=False)}


def interaction_summary(fold_interactions: dict[int, np.ndarray]) -> pd.DataFrame:
    """Summarize every unique off-diagonal positional interaction by fold and rank."""
    records: list[dict[str, int | str | float]] = []
    for fold in (1, 2, 3):
        values = np.asarray(fold_interactions[fold], dtype=float)
        if values.shape != (len(PATH_COLUMNS), len(PATH_COLUMNS)):
            raise AssertionError("D001 SHAP interaction summary has an unexpected shape.")
        for left in range(len(PATH_COLUMNS)):
            for right in range(left + 1, len(PATH_COLUMNS)):
                records.append(
                    {
                        "fold": fold,
                        "position_left": PATH_COLUMNS[left],
                        "position_right": PATH_COLUMNS[right],
                        "mean_abs_interaction": float(values[left, right]),
                    }
                )
    table = pd.DataFrame(records)
    table["rank_within_fold"] = table.groupby("fold")["mean_abs_interaction"].rank(ascending=False, method="min")
    cross = (
        table.groupby(["position_left", "position_right"], as_index=False)
        .agg(
            mean_abs_interaction_across_folds=("mean_abs_interaction", "mean"),
            mean_rank_across_folds=("rank_within_fold", "mean"),
            max_rank_across_folds=("rank_within_fold", "max"),
        )
        .sort_values(["mean_abs_interaction_across_folds", "mean_rank_across_folds"], ascending=[False, True])
    )
    cross["cross_fold_rank"] = cross["mean_abs_interaction_across_folds"].rank(ascending=False, method="min")
    return table.sort_values(["fold", "rank_within_fold", "position_left", "position_right"]), cross


def rank_stability(summary: pd.DataFrame) -> dict[str, float]:
    """Return descriptive rank correlations among the three fold rankings."""
    ranks = summary[["rank_fold_1", "rank_fold_2", "rank_fold_3"]]
    return {
        "spearman_rank_fold_1_vs_2": float(ranks.iloc[:, 0].corr(ranks.iloc[:, 1], method="spearman")),
        "spearman_rank_fold_1_vs_3": float(ranks.iloc[:, 0].corr(ranks.iloc[:, 2], method="spearman")),
        "spearman_rank_fold_2_vs_3": float(ranks.iloc[:, 1].corr(ranks.iloc[:, 2], method="spearman")),
    }


def main() -> None:
    started = time.perf_counter()
    physical_manifest = validate_physical_discovery_boundary()
    features, target, development_equities = load_discovery_rows()
    if len(features) != 472_816 or not features["day"].between(0, 352).all():
        raise AssertionError("D001 SHAP loaded data outside the physical Discovery boundary.")
    if not features["equity"].isin(development_equities).all():
        raise AssertionError("D001 SHAP loaded E_holdout observations.")

    authoritative_auc = authoritative_p_validation_auc()
    reconstructed: dict[int, tuple[object, pd.DataFrame, pd.DataFrame, float]] = {}
    reconstruction_records: list[dict[str, int | float]] = []

    # Reconstruction phase: all historical P AUCs must match before any SHAP call.
    for fold in DISCOVERY_FOLDS:
        (fit_features, fit_target), (validation_features, validation_target) = split_discovery_fold(
            features, target, development_equities, fold
        )
        fit_features, fit_target, _ = prepare_complete_w_population(fit_features, fit_target, development_equities)
        validation_features, validation_target, _ = prepare_complete_w_population(
            validation_features, validation_target, development_equities
        )
        assert_both_binary_classes(fit_target, context=f"D001 SHAP reconstruction fit fold {fold.fold}")
        assert_both_binary_classes(validation_target, context=f"D001 SHAP reconstruction validation fold {fold.fold}")
        fit_path = build_d001_representations(fit_features)["P"]
        validation_path = build_d001_representations(validation_features)["P"]
        assert_identical_representation_rows({"C": build_d001_representations(fit_features)["C"], "P": fit_path}, fit_features, fit_target)
        assert_identical_representation_rows({"C": build_d001_representations(validation_features)["C"], "P": validation_path}, validation_features, validation_target)
        classifier = make_d001_classifier()
        classifier.fit(fit_path, fit_target)
        if tuple(classifier.classes_) != (0, 1):
            raise AssertionError("D001 SHAP P reconstruction requires binary classes [0, 1].")
        reconstructed_auc = float(roc_auc_score(validation_target, classifier.predict_proba(validation_path)[:, 1]))
        difference = reconstructed_auc - authoritative_auc[fold.fold]
        reconstruction_records.append(
            {
                "fold": fold.fold,
                "authoritative_p_validation_auc": authoritative_auc[fold.fold],
                "reconstructed_p_validation_auc": reconstructed_auc,
                "difference": difference,
            }
        )
        if not np.isclose(reconstructed_auc, authoritative_auc[fold.fold], rtol=0.0, atol=AUC_TOLERANCE):
            raise AssertionError(f"D001 P reconstruction AUC mismatch in fold {fold.fold}: {difference:.16g}")
        reconstructed[fold.fold] = (classifier, validation_path, validation_features, reconstructed_auc)

    # SHAP phase: validation-only, after all reconstruction checks have passed.
    fold_mean_abs: dict[int, pd.Series] = {}
    fold_interactions: dict[int, np.ndarray] = {}
    global_records: list[pd.DataFrame] = []
    shap_manifest: list[dict[str, int | float]] = []
    for fold in DISCOVERY_FOLDS:
        classifier, validation_path, validation_features, _ = reconstructed[fold.fold]
        explained = explain_d001_path_validation(classifier, validation_path, fold_number=fold.fold)
        global_index = explained["global_index"]
        interaction_index = explained["interaction_index"]
        explanation = explained["global_explanation"]
        values = np.asarray(explanation.values, dtype=float)
        sample_path = validation_path.loc[global_index]
        sample = pd.DataFrame({"fold": fold.fold, "ID": validation_features.loc[global_index, "ID"].to_numpy()})
        for position, column in enumerate(PATH_COLUMNS):
            sample[f"value_{column}"] = sample_path[column].to_numpy()
            sample[f"shap_{column}"] = values[:, position]
        sample["base_value_raw_margin"] = np.asarray(explanation.base_values, dtype=float)
        sample["model_raw_margin"] = np.asarray(classifier.predict(sample_path, output_margin=True), dtype=float)
        global_records.append(sample)
        fold_mean_abs[fold.fold] = explained["mean_abs_shap"]
        fold_interactions[fold.fold] = np.abs(np.asarray(explained["interaction_values"], dtype=float)).mean(axis=0)
        shap_manifest.append(
            {
                "fold": fold.fold,
                "validation_rows": len(validation_path),
                "global_shap_rows": len(global_index),
                "global_shap_seed": GLOBAL_SHAP_SEED + fold.fold,
                "interaction_shap_rows": len(interaction_index),
                "interaction_shap_seed": INTERACTION_SHAP_SEED + fold.fold,
            }
        )

    summary = cross_fold_shap_summary(fold_mean_abs)
    fold_table = summary.reset_index(names="position").melt(
        id_vars="position", value_vars=[f"mean_abs_shap_fold_{fold}" for fold in (1, 2, 3)],
        var_name="measure", value_name="mean_abs_shap"
    )
    fold_table["fold"] = fold_table["measure"].str.extract(r"(\d+)$").astype(int)
    fold_table["rank_within_fold"] = fold_table.apply(
        lambda row: summary.iloc[row.name % len(PATH_COLUMNS)][f"rank_fold_{int(row['fold'])}"], axis=1
    )
    fold_table = fold_table[["fold", "position", "mean_abs_shap", "rank_within_fold"]].sort_values(["fold", "rank_within_fold", "position"])
    interaction_by_fold, interaction_cross_fold = interaction_summary(fold_interactions)
    stability = rank_stability(summary)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(reconstruction_records).to_csv(RESULTS_DIR / "D001_shap_reconstruction.csv", index=False)
    pd.DataFrame(shap_manifest).to_csv(RESULTS_DIR / "D001_shap_sample_manifest.csv", index=False)
    pd.concat(global_records, ignore_index=True).to_csv(RESULTS_DIR / "D001_shap_global_values.csv", index=False)
    fold_table.to_csv(RESULTS_DIR / "D001_shap_mean_abs_by_fold.csv", index=False)
    summary.reset_index(names="position").to_csv(RESULTS_DIR / "D001_shap_cross_fold_summary.csv", index=False)
    interaction_by_fold.to_csv(RESULTS_DIR / "D001_shap_interactions_by_fold.csv", index=False)
    interaction_cross_fold.to_csv(RESULTS_DIR / "D001_shap_interactions_cross_fold.csv", index=False)
    metadata = {
        "candidate": "D001",
        "model_provenance": "reconstructed_P_only_original_models_not_persisted",
        "reconstruction_auc_tolerance": AUC_TOLERANCE,
        "shap": {
            "version": shap.__version__,
            "explainer": "TreeExplainer",
            "data": None,
            "feature_perturbation": "tree_path_dependent",
            "model_output": "raw",
            "feature_names": list(PATH_COLUMNS),
            "global_sample_max": GLOBAL_SHAP_SAMPLE_MAX,
            "interaction_sample_max": INTERACTION_SHAP_SAMPLE_MAX,
        },
        "xgboost_version": xgboost.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "physical_partition_manifest": physical_manifest,
        "rank_stability": stability,
        "internal_confirmation_accessed": False,
        "e_holdout_accessed": False,
        "competition_test_accessed": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    (RESULTS_DIR / "D001_shap_metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    print(pd.DataFrame(reconstruction_records).to_string(index=False))
    print(summary.to_string())
    print(pd.DataFrame(shap_manifest).to_string(index=False))


if __name__ == "__main__":
    main()
