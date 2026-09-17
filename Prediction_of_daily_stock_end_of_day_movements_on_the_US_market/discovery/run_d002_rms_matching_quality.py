"""Run only the authorized D002 target-control matching-quality audit."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from discovery.d001_path_structure_beyond_recent_rms import prepare_complete_w_population  # noqa: E402
from discovery.run_d001_path_structure_beyond_recent_rms import (  # noqa: E402
    RESULTS_DIR,
    load_discovery_rows,
)
from discovery.run_d001_shap_interpretation import validate_physical_discovery_boundary  # noqa: E402
from discovery.d002_local_organization import (  # noqa: E402
    DISCOVERY_BLOCKS,
    matching_quality_summary,
    match_neutral_to_directional_by_rms,
    window_energy_and_rms,
)


def main() -> None:
    validate_physical_discovery_boundary()
    features, target, development_equities = load_discovery_rows()
    complete_features, complete_target, _ = prepare_complete_w_population(features, target, development_equities)
    energy, rms = window_energy_and_rms(complete_features)
    base = pd.DataFrame(
        {
            "ID": complete_features["ID"].to_numpy(),
            "day": complete_features["day"].to_numpy(),
            "Z": complete_target.to_numpy(),
            "I_recent": rms,
            "E_zero": energy == 0.0,
        },
        index=complete_features.index,
    )
    quality: list[dict[str, float | int | str]] = []
    pairs: list[pd.DataFrame] = []
    for block_name, start, end in DISCOVERY_BLOCKS:
        block = base.loc[base["day"].between(start, end)].copy()
        if block.empty:
            raise AssertionError(f"D002 block {block_name} is empty.")
        rms_iqr = float(block["I_recent"].quantile(0.75) - block["I_recent"].quantile(0.25))
        matches = match_neutral_to_directional_by_rms(block[["ID", "Z", "I_recent"]])
        quality.append(
            matching_quality_summary(
                matches,
                rms_scale_iqr=rms_iqr,
                block=block_name,
                total_rows=len(block),
                zero_energy_rows=int(block["E_zero"].sum()),
            )
        )
        block_pairs = matches.pairs.copy()
        block_pairs.insert(0, "block", block_name)
        pairs.append(block_pairs)
    pair_frame = pd.concat(pairs, ignore_index=True)
    pooled_iqr = float(base["I_recent"].quantile(0.75) - base["I_recent"].quantile(0.25))
    pooled_matches = type("PooledMatches", (), {"pairs": pair_frame, "neutral_available": int(base["Z"].eq(0).sum()), "directional_available": int(base["Z"].eq(1).sum())})()
    quality.append(
        matching_quality_summary(
            pooled_matches,
            rms_scale_iqr=pooled_iqr,
            block="pooled_discovery",
            total_rows=len(base),
            zero_energy_rows=int(base["E_zero"].sum()),
        )
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pair_frame.to_csv(RESULTS_DIR / "D002_rms_matched_pairs.csv", index=False)
    pd.DataFrame(quality).to_csv(RESULTS_DIR / "D002_rms_matching_quality.csv", index=False)
    print(pd.DataFrame(quality).to_string(index=False))


if __name__ == "__main__":
    main()
