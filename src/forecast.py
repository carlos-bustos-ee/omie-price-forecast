"""Feature engineering, point + probabilistic models, and a walk-forward backtest
for day-ahead electricity price forecasting.

Honest information window: prices for delivery day D are set at the D-1 auction.
When forecasting D on the morning of D-1, every price up to the end of D-1 is
already public (published ~13:00 on D-2), so features may only use lags of 24
hours or more. Everything below respects that constraint.

Modelling note: electricity price LEVELS are non-stationary and tree models
cannot extrapolate outside the training range, so the learned target is the
DELTA against the 24h-lagged price (the naive forecast). Every model predicts
the correction to apply on top of "yesterday's price"; the remaining lag
features are expressed relative to that same anchor.

Probabilistic forecasts come from gradient-boosting quantile regression, then a
split-conformal calibration step widens the P10-P90 band on held-out data so its
empirical coverage matches the nominal 80% (raw GBM quantiles are overconfident).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

REL_LAGS = (48, 72, 168)
QUANTILES = (0.1, 0.5, 0.9)
TARGET_COVERAGE = 0.80

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


# --------------------------------------------------------------------------- #
# Models                                                                      #
# --------------------------------------------------------------------------- #
def _fit_set(train: pd.DataFrame) -> tuple[HistGradientBoostingRegressor, dict]:
    """One point model (mean) + one gradient-boosting model per quantile."""
    point = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, early_stopping=False, random_state=42
    )
    point.fit(train[FEATURES], train["target_delta"])
    quantiles = {
        q: HistGradientBoostingRegressor(
            loss="quantile", quantile=q, max_iter=400, learning_rate=0.05,
            early_stopping=False, random_state=42,
        ).fit(train[FEATURES], train["target_delta"])
        for q in QUANTILES
    }
    return point, quantiles


def _quantile_prices(quantiles: dict, rows: pd.DataFrame) -> dict:
    anchor = rows["lag_24"].to_numpy()
    return {q: anchor + quantiles[q].predict(rows[FEATURES]) for q in QUANTILES}


def _calibrate_width(y: np.ndarray, qp: dict, target: float = TARGET_COVERAGE) -> float:
    """Smallest width multiplier k so [P50-k(P50-P10), P50+k(P90-P50)] covers `target`."""
    q10, q50, q90 = qp[0.1], qp[0.5], qp[0.9]
    for k in np.linspace(0.6, 3.5, 59):
        lo, hi = q50 - k * (q50 - q10), q50 + k * (q90 - q50)
        if np.mean((y >= lo) & (y <= hi)) >= target:
            return float(k)
    return 3.5


def _fit_calibrated(train: pd.DataFrame, calib_days: int = 21):
    """Fit models on full train; learn the conformal width on a held-out tail."""
    n_calib = calib_days * 24
    inner, calib = train.iloc[:-n_calib], train.iloc[-n_calib:]
    _, quant_inner = _fit_set(inner)
    k = _calibrate_width(calib["price_eur_mwh"].to_numpy(), _quantile_prices(quant_inner, calib))
    point, quantiles = _fit_set(train)   # deploy on all data
    return point, quantiles, k


# --------------------------------------------------------------------------- #
# Walk-forward backtest                                                       #
# --------------------------------------------------------------------------- #
def backtest(df: pd.DataFrame, test_days: int = 28, retrain_every: int = 7) -> pd.DataFrame:
    """Expanding-window walk-forward evaluation over the last ``test_days``.

    For each delivery day the model has only ever seen data strictly before that
    day (no leakage); it is refit every ``retrain_every`` days to mimic a desk
    that periodically retrains. Returns an hourly frame with the actual price,
    the naive-24h benchmark, the point forecast and the calibrated P10/P50/P90.
    """
    cutoff = df.index.max().normalize() - pd.Timedelta(days=test_days - 1)
    test_index = df.index[df.index >= cutoff]
    days = sorted({d.normalize() for d in test_index})

    preds = pd.DataFrame(index=test_index)
    preds["actual"] = df.loc[test_index, "price_eur_mwh"]
    preds["naive_24h"] = df.loc[test_index, "lag_24"]

    point = quantiles = k = None
    for i, day in enumerate(days):
        if i % retrain_every == 0:
            point, quantiles, k = _fit_calibrated(df[df.index < day])
        rows = df[(df.index >= day) & (df.index < day + pd.Timedelta(days=1))]
        qp = _quantile_prices(quantiles, rows)
        preds.loc[rows.index, "point"] = rows["lag_24"].to_numpy() + point.predict(rows[FEATURES])
        preds.loc[rows.index, "q50"] = qp[0.5]
        preds.loc[rows.index, "q10"] = qp[0.5] - k * (qp[0.5] - qp[0.1])   # conformal widen
        preds.loc[rows.index, "q90"] = qp[0.5] + k * (qp[0.9] - qp[0.5])

    preds[["q10", "q50", "q90"]] = np.sort(preds[["q10", "q50", "q90"]].to_numpy(), axis=1)
    return preds


# --------------------------------------------------------------------------- #
# Scoring                                                                      #
# --------------------------------------------------------------------------- #
def _pinball(y: pd.Series, yhat: pd.Series, q: float) -> float:
    d = y - yhat
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def score(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Point metrics (MAE/RMSE vs the naive-24h benchmark) plus probabilistic metrics."""
    y = preds["actual"]
    rows = []
    for name in ["naive_24h", "point"]:
        err = preds[name] - y
        rows.append(
            {
                "model": name,
                "mae_eur_mwh": float(err.abs().mean()),
                "rmse_eur_mwh": float(np.sqrt((err**2).mean())),
            }
        )
    metrics = pd.DataFrame(rows)
    naive_mae = metrics.loc[metrics["model"] == "naive_24h", "mae_eur_mwh"].iloc[0]
    metrics["improvement_vs_naive_24h"] = 1 - metrics["mae_eur_mwh"] / naive_mae

    pinball = float(np.mean([_pinball(y, preds[f"q{int(q * 100)}"], q) for q in QUANTILES]))
    coverage = float(((y >= preds["q10"]) & (y <= preds["q90"])).mean())
    prob = pd.DataFrame(
        [{"metric": "pinball_loss_eur_mwh (avg P10/P50/P90)", "value": round(pinball, 3)},
         {"metric": "coverage_P10_P90 (target 0.80)", "value": round(coverage, 3)}]
    )
    return metrics, prob
