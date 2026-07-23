"""Feature engineering and models for day-ahead price forecasting.

Honest information window: prices for delivery day D are set at the D-1
auction. When forecasting D on the morning of D-1, every price up to the end
of D-1 is already public (published ~13:00 on D-2), so features may only use
lags of 24 hours or more. Everything below respects that constraint.

Modelling note: electricity price LEVELS are non-stationary and tree models
cannot extrapolate outside the training range, so the learned target is the
DELTA against the 24h-lagged price (the naive forecast). The model predicts
the correction to apply on top of "yesterday's price", and the other lag
features are expressed relative to that same anchor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

REL_LAGS = (48, 72, 168)

FEATURES = [f"dlag_{lag}" for lag in REL_LAGS] + [
    "droll_24",
    "droll_168",
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
]


def make_features(series: pd.DataFrame) -> pd.DataFrame:
    """Build the supervised frame. Anchor: lag_24 (yesterday, same hour)."""
    df = series.copy()
    df["lag_24"] = df["price_eur_mwh"].shift(24)
    for lag in REL_LAGS:
        df[f"dlag_{lag}"] = df["price_eur_mwh"].shift(lag) - df["lag_24"]
    df["droll_24"] = df["price_eur_mwh"].shift(24).rolling(24).mean() - df["lag_24"]
    df["droll_168"] = df["price_eur_mwh"].shift(24).rolling(168).mean() - df["lag_24"]
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    df["target_delta"] = df["price_eur_mwh"] - df["lag_24"]
    return df.dropna()


def evaluate(df: pd.DataFrame, test_days: int = 28) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on everything except the last ``test_days``, forecast those, score.

    Baselines: naive 24h (yesterday's price, the standard benchmark in this
    market) and naive 168h (same hour last week).
    """
    cutoff = df.index.max().normalize() - pd.Timedelta(days=test_days - 1)
    train, test = df[df.index < cutoff], df[df.index >= cutoff]

    preds = pd.DataFrame(index=test.index)
    preds["actual"] = test["price_eur_mwh"]
    preds["naive_24h"] = test["lag_24"]
    preds["naive_168h"] = test["lag_24"] + test["dlag_168"]

    model = HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.05, random_state=42
    )
    model.fit(train[FEATURES], train["target_delta"])
    preds["gradient_boosting"] = test["lag_24"] + model.predict(test[FEATURES])

    rows = []
    for name in ["naive_24h", "naive_168h", "gradient_boosting"]:
        err = preds[name] - preds["actual"]
        rows.append(
            {
                "model": name,
                "mae_eur_mwh": err.abs().mean(),
                "rmse_eur_mwh": float(np.sqrt((err**2).mean())),
            }
        )
    metrics = pd.DataFrame(rows)
    naive_mae = metrics.loc[metrics["model"] == "naive_24h", "mae_eur_mwh"].iloc[0]
    metrics["improvement_vs_naive_24h"] = 1 - metrics["mae_eur_mwh"] / naive_mae
    return metrics, preds
