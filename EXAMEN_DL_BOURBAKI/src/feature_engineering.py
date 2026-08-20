import pandas as pd 
import numpy as  np 

quant_features = [
    # Path
    "path_return_bps",
    "abs_path_return_bps",
    "return_early_bps",
    "return_middle_bps",
    "return_late_bps",
    "early_late_change_bps",

    # Activity / volatility
    "realized_vol_bps",

    # Path structure
    "path_efficiency",
    "early_vol_share",
    "late_vol_share",
    "vol_concentration",
    "sign_persistence",
    "path_asymmetry",

    # Data availability
    "n_obs",
]

# Final Quant Feature Builder
def build_quant_features(
    train_df,
    val_df,
    return_cols,
    lower_q=0.001,
    upper_q=0.999
):
    """
    Construye las features cuantitativas finales.

    Los límites de winsorización se estiman exclusivamente
    utilizando train y después se aplican sin recalibración
    sobre validation.
    """

    train_out = train_df.copy()
    val_out = val_df.copy()

    # --------------------------------------------------------
    # 1. Winsorización por intervalo aprendida sólo en train
    # --------------------------------------------------------

    lower_bounds = train_df[return_cols].quantile(lower_q)
    upper_bounds = train_df[return_cols].quantile(upper_q)

    train_out[return_cols] = train_out[return_cols].clip(
        lower=lower_bounds,
        upper=upper_bounds,
        axis=1
    )

    val_out[return_cols] = val_out[return_cols].clip(
        lower=lower_bounds,
        upper=upper_bounds,
        axis=1
    )

    # Ventanas temporales
    early_cols = [f"r{i}" for i in range(0, 18)]
    middle_cols = [f"r{i}" for i in range(18, 36)]
    late_cols = [f"r{i}" for i in range(36, 53)]

    first_half_cols = return_cols[:len(return_cols) // 2]
    second_half_cols = return_cols[len(return_cols) // 2:]

    # --------------------------------------------------------
    # 2. Feature engineering
    # --------------------------------------------------------

    for df in [train_out, val_out]:

        # Disponibilidad
        df["n_obs"] = df[return_cols].notna().sum(axis=1)

        # ---- Path / displacement ----

        df["path_return_bps"] = (
            df[return_cols].sum(axis=1, skipna=True)
        )

        df["abs_path_return_bps"] = (
            df["path_return_bps"].abs()
        )

        df["return_early_bps"] = (
            df[early_cols].sum(axis=1, skipna=True)
        )

        df["return_middle_bps"] = (
            df[middle_cols].sum(axis=1, skipna=True)
        )

        df["return_late_bps"] = (
            df[late_cols].sum(axis=1, skipna=True)
        )

        df["early_late_change_bps"] = (
            df["return_late_bps"]
            - df["return_early_bps"]
        )

        # ---- Volatilidad ----

        squared_returns = df[return_cols] ** 2
        total_variation = squared_returns.sum(axis=1, skipna=True)

        df["realized_vol_bps"] = np.sqrt(total_variation)

        # ---- Path efficiency ----

        total_abs_move = (
            df[return_cols]
            .abs()
            .sum(axis=1, skipna=True)
        )

        df["path_efficiency"] = np.where(
            total_abs_move > 0,
            df["abs_path_return_bps"] / total_abs_move,
            0.0
        )

        # ---- Distribución temporal de volatilidad ----

        early_variation = (
            squared_returns[early_cols]
            .sum(axis=1, skipna=True)
        )

        late_variation = (
            squared_returns[late_cols]
            .sum(axis=1, skipna=True)
        )

        df["early_vol_share"] = np.where(
            total_variation > 0,
            early_variation / total_variation,
            0.0
        )

        df["late_vol_share"] = np.where(
            total_variation > 0,
            late_variation / total_variation,
            0.0
        )

        # ---- Concentración de volatilidad ----

        df["vol_concentration"] = np.where(
            total_variation > 0,
            squared_returns.max(axis=1) / total_variation,
            0.0
        )

        # ---- Persistencia de signo ----

        path_sign = np.sign(df["path_return_bps"])
        signs = np.sign(df[return_cols])

        observed = df[return_cols].notna()
        same_sign = signs.eq(path_sign, axis=0)

        df["sign_persistence"] = np.where(
            df["n_obs"] > 0,
            (same_sign & observed).sum(axis=1) / df["n_obs"],
            np.nan
        )

        # ---- Asimetría temporal ----

        first_activity = (
            df[first_half_cols]
            .abs()
            .sum(axis=1, skipna=True)
        )

        second_activity = (
            df[second_half_cols]
            .abs()
            .sum(axis=1, skipna=True)
        )

        total_activity = first_activity + second_activity

        df["path_asymmetry"] = np.where(
            total_activity > 0,
            (second_activity - first_activity) / total_activity,
            np.nan
        )

    return train_out, val_out, lower_bounds, upper_bounds