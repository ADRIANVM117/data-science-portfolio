# src/preprocessing.py

import numpy as np
import pandas as pd


TRAIN_END_DAY = 401
VAL_START_DAY = 402
CLIP_LIMIT = 20.0


def temporal_split(df):
    """
    Split temporal definido en 02.

    Train: day 0-401
    Validation: day 402-502
    """

    train_dev = df.loc[
        df["day"] <= TRAIN_END_DAY
    ].copy()

    val_dev = df.loc[
        df["day"] >= VAL_START_DAY
    ].copy()

    return train_dev, val_dev


def add_missingness_features(df, return_cols):
    out = df.copy()

    out["missing_count"] = (
        out[return_cols].isna().sum(axis=1)
    )

    out["missing_share"] = (
        out["missing_count"] / len(return_cols)
    )

    return out


def fit_robust_params(train_df, return_cols):
    """
    Parámetros estimados SOLO con train.
    """

    params = pd.DataFrame({
        "median": train_df[return_cols].median(),
        "q25": train_df[return_cols].quantile(0.25),
        "q75": train_df[return_cols].quantile(0.75),
    })

    params["iqr"] = (
        params["q75"] - params["q25"]
    )

    return params


def transform_returns(
    df,
    return_cols,
    robust_params,
    clip_limit=CLIP_LIMIT
):
    """
    Robust scaling -> clipping -> imputación neutral.
    """

    out = add_missingness_features(
        df,
        return_cols
    )

    for col in return_cols:

        out[col] = (
            out[col] - robust_params.loc[col, "median"]
        ) / robust_params.loc[col, "iqr"]

    out[return_cols] = (
        out[return_cols]
        .clip(-clip_limit, clip_limit)
        .fillna(0.0)
    )

    return out