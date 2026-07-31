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
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REL_LAGS = (48, 72, 168)
QUANTILES = (0.1, 0.5, 0.9)
TARGET_COVERAGE = 0.80

BASE_FEATURES = [f"dlag_{lag}" for lag in REL_LAGS] + [
    "droll_24",
    "droll_168",
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
]

# Domestic exogenous features from REE/ESIOS day-ahead (D+1) forecasts. These are
# available BEFORE the D-1 auction, so they carry no look-ahead: the net-load
# forecast (demand minus wind minus solar) is the single strongest driver of the
# marginal price. ``d_netload_24`` mirrors the delta target — how much
# tighter/looser the system is than the same hour yesterday.
EXOG_FEATURES = ["demand_fc", "wind_fc", "solar_fc", "netload_fc", "d_netload_24"]

# Cross-border feature: France's day-ahead LOAD forecast (from ENTSO-E). High
# French demand pulls power south over the interconnector and lifts the Spanish
# price. Only load is used, NOT French wind/solar: under EU Reg. 543/2013 the
# day-ahead load forecast is published >=2h before day-ahead gate closure (so it
# is available at the ~12:00 D-1 MIBEL auction), whereas the wind/solar forecast
# deadline is 18:00 D-1 — after the gate — so using it would be look-ahead.
FR_EXOG_FEATURES = ["fr_load_fc", "d_fr_load_24"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Base autoregressive features, plus any exogenous features present."""
    return BASE_FEATURES + [c for c in EXOG_FEATURES + FR_EXOG_FEATURES if c in df.columns]


def make_features(series: pd.DataFrame, exog: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the supervised frame. Anchor: lag_24 (yesterday, same hour).

    If ``exog`` (hourly demand_fc/wind_fc/solar_fc, MW) is given, its day-ahead
    forecast for each delivery hour is joined on the target timestamp — legitimate
    because those forecasts are published before the auction — and turned into a
    net-load level and its change versus the same hour yesterday.
    """
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
    if exog is not None:
        ex = exog.reindex(df.index)
        if {"demand_fc", "wind_fc", "solar_fc"}.issubset(ex.columns):  # Spain (ESIOS)
            df["demand_fc"] = ex["demand_fc"]
            df["wind_fc"] = ex["wind_fc"]
            df["solar_fc"] = ex["solar_fc"]
            df["netload_fc"] = ex["demand_fc"] - ex["wind_fc"] - ex["solar_fc"]
            df["d_netload_24"] = df["netload_fc"] - df["netload_fc"].shift(24)
        if "fr_load_fc" in ex.columns:  # France (ENTSO-E) — load only, see FR_EXOG_FEATURES
            df["fr_load_fc"] = ex["fr_load_fc"]
            df["d_fr_load_24"] = df["fr_load_fc"] - df["fr_load_fc"].shift(24)
    df["target_delta"] = df["price_eur_mwh"] - df["lag_24"]
    return df.dropna()


# --------------------------------------------------------------------------- #
# Models                                                                      #
# --------------------------------------------------------------------------- #
def _fit_point(train: pd.DataFrame, features: list[str]) -> HistGradientBoostingRegressor:
    """The point (mean) gradient-boosting model on the delta target."""
    point = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, early_stopping=False, random_state=42
    )
    point.fit(train[features], train["target_delta"])
    return point


def _fit_set(train: pd.DataFrame, features: list[str]) -> tuple[HistGradientBoostingRegressor, dict]:
    """One point model (mean) + one gradient-boosting model per quantile."""
    point = _fit_point(train, features)
    quantiles = {
        q: HistGradientBoostingRegressor(
            loss="quantile", quantile=q, max_iter=400, learning_rate=0.05,
            early_stopping=False, random_state=42,
        ).fit(train[features], train["target_delta"])
        for q in QUANTILES
    }
    return point, quantiles


def _quantile_prices(quantiles: dict, rows: pd.DataFrame, features: list[str]) -> dict:
    anchor = rows["lag_24"].to_numpy()
    return {q: anchor + quantiles[q].predict(rows[features]) for q in QUANTILES}


def _fit_lasso(train: pd.DataFrame, features: list[str]):
    """Regularized linear autoregressive baseline (standardized features + LASSO,
    alpha chosen by cross-validation). This is the workhorse benchmark of the
    electricity-price-forecasting literature (the linear half of Lago et al.'s
    LEAR); it predicts the same DELTA target as the gradient boosting model, so
    the two are directly comparable and both are beatable only by genuinely
    modelling the correction on top of the naive forecast."""
    model = make_pipeline(
        StandardScaler(),
        # TimeSeriesSplit (not plain K-fold) so alpha is chosen without letting
        # future folds inform the past — consistent with the walk-forward design.
        LassoCV(cv=TimeSeriesSplit(5), max_iter=10000, random_state=42),
    )
    model.fit(train[features], train["target_delta"])
    return model


def _calibrate_width(y: np.ndarray, qp: dict, target: float = TARGET_COVERAGE) -> float:
    """Smallest width multiplier k so [P50-k(P50-P10), P50+k(P90-P50)] covers `target`."""
    q10, q50, q90 = qp[0.1], qp[0.5], qp[0.9]
    for k in np.linspace(0.6, 3.5, 59):
        lo, hi = q50 - k * (q50 - q10), q50 + k * (q90 - q50)
        if np.mean((y >= lo) & (y <= hi)) >= target:
            return float(k)
    return 3.5


def _calibrate_width_by_hour(
    y: np.ndarray, qp: dict, hours: np.ndarray, target: float = TARGET_COVERAGE,
    min_points: int = 8,
) -> dict:
    """One conformal width multiplier per hour of day.

    The per-day error-by-hour analysis shows the interval is too narrow in the
    volatile midday hours and too wide overnight, so a single global multiplier
    can't hit 80% coverage everywhere. This learns k separately for each hour on
    the held-out calibration window. Hours with too few calibration points fall
    back to the global multiplier so a thin bucket can't produce a wild width.
    """
    global_k = _calibrate_width(y, qp, target)
    q10, q50, q90 = qp[0.1], qp[0.5], qp[0.9]
    ks = {}
    for h in range(24):
        mask = hours == h
        if mask.sum() < min_points:
            ks[h] = global_k
            continue
        ks[h] = _calibrate_width(
            y[mask], {0.1: q10[mask], 0.5: q50[mask], 0.9: q90[mask]}, target
        )
    return ks


def _fit_calibrated(train: pd.DataFrame, features: list[str], calib_days: int = 21,
                    by_hour: bool = False):
    """Fit models on full train; learn the conformal width on a held-out tail.

    Returns ``k`` as a single float (global calibration) or a ``{hour: k}`` dict
    (hour-conditional calibration).
    """
    n_calib = calib_days * 24
    inner, calib = train.iloc[:-n_calib], train.iloc[-n_calib:]
    _, quant_inner = _fit_set(inner, features)
    qp_calib = _quantile_prices(quant_inner, calib, features)
    y_calib = calib["price_eur_mwh"].to_numpy()
    if by_hour:
        k = _calibrate_width_by_hour(y_calib, qp_calib, calib.index.hour.to_numpy())
    else:
        k = _calibrate_width(y_calib, qp_calib)
    point, quantiles = _fit_set(train, features)   # deploy on all data
    return point, quantiles, k


# --------------------------------------------------------------------------- #
# Walk-forward backtest                                                       #
# --------------------------------------------------------------------------- #
def backtest(
    df: pd.DataFrame, test_days: int = 28, retrain_every: int = 7,
    calibration: str = "global", features: list[str] | None = None,
    fit_quantiles: bool = True, fit_lasso: bool = True,
) -> pd.DataFrame:
    """Expanding-window walk-forward evaluation over the last ``test_days``.

    For each delivery day the model has only ever seen data strictly before that
    day (no leakage); it is refit every ``retrain_every`` days to mimic a desk
    that periodically retrains. Returns an hourly frame with the actual price,
    the naive-24h benchmark, the point forecast and the calibrated P10/P50/P90.

    ``calibration`` is ``"global"`` (a single conformal width; the default) or
    ``"hourly"`` (a width per hour of day). Both learn the width only on a
    held-out tail the models never trained on. Global is the default because,
    on a 21-day calibration window, the per-hour version overfits — it flattens
    the hourly coverage spread slightly but *worsens* the pinball loss and drops
    overall coverage below target (see the README experiment). ``"hourly"`` is
    kept for when more calibration history is available.
    """
    by_hour = calibration == "hourly"
    features = features if features is not None else feature_columns(df)
    cutoff = df.index.max().normalize() - pd.Timedelta(days=test_days - 1)
    test_index = df.index[df.index >= cutoff]
    days = sorted({d.normalize() for d in test_index})

    preds = pd.DataFrame(index=test_index)
    preds["actual"] = df.loc[test_index, "price_eur_mwh"]
    preds["naive_24h"] = df.loc[test_index, "lag_24"]

    point = quantiles = k = lasso = None
    for i, day in enumerate(days):
        if i % retrain_every == 0:
            train = df[df.index < day]
            if fit_quantiles:
                point, quantiles, k = _fit_calibrated(train, features, by_hour=by_hour)
            else:
                point = _fit_point(train, features)   # point-only variant (ablation legs)
            if fit_lasso:
                lasso = _fit_lasso(train, features)
        rows = df[(df.index >= day) & (df.index < day + pd.Timedelta(days=1))]
        anchor = rows["lag_24"].to_numpy()
        preds.loc[rows.index, "point"] = anchor + point.predict(rows[features])
        if fit_lasso:
            preds.loc[rows.index, "lasso"] = anchor + lasso.predict(rows[features])
        if fit_quantiles:
            qp = _quantile_prices(quantiles, rows, features)
            kk = np.array([k[h] for h in rows.index.hour]) if by_hour else k
            preds.loc[rows.index, "q50"] = qp[0.5]
            preds.loc[rows.index, "q10"] = qp[0.5] - kk * (qp[0.5] - qp[0.1])   # conformal widen
            preds.loc[rows.index, "q90"] = qp[0.5] + kk * (qp[0.9] - qp[0.5])

    if fit_quantiles:
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
    names = [c for c in ["naive_24h", "lasso", "point_price_only", "point_es", "point"]
             if c in preds.columns]
    rows = []
    for name in names:
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


def score_by_hour(preds: pd.DataFrame) -> pd.DataFrame:
    """Break the backtest down by hour of day.

    Electricity price forecasting is not uniformly hard across the day: the
    solar-driven midday trough and the evening ramp behave very differently from
    the flat overnight hours. This groups the out-of-sample predictions by hour
    and reports, for each, the naive-24h MAE, the model MAE, the improvement the
    model buys, and the empirical P10-P90 coverage — i.e. *where* the model adds
    value and whether the uncertainty band stays honest across the day.
    """
    df = preds.copy()
    df["hour"] = df.index.hour
    rows = []
    for hour, g in df.groupby("hour"):
        y = g["actual"]
        naive_mae = float((g["naive_24h"] - y).abs().mean())
        point_mae = float((g["point"] - y).abs().mean())
        cov = float(((y >= g["q10"]) & (y <= g["q90"])).mean())
        rows.append(
            {
                "hour": int(hour),
                "n_days": int(len(g)),
                "naive_mae_eur_mwh": naive_mae,
                "point_mae_eur_mwh": point_mae,
                "improvement_pct": (1 - point_mae / naive_mae) * 100 if naive_mae else float("nan"),
                "coverage_P10_P90": cov,
            }
        )
    return pd.DataFrame(rows).set_index("hour")


# --------------------------------------------------------------------------- #
# Statistical significance                                                    #
# --------------------------------------------------------------------------- #
def diebold_mariano(actual, pred_a, pred_b, hac_lag: int = 24, loss: str = "abs"):
    """Diebold-Mariano test on the loss differential of forecast A vs B.

    A headline "-15% MAE" is worthless if it isn't distinguishable from noise.
    Hourly forecast errors are strongly autocorrelated, so the *effective* sample
    is far smaller than the raw hour count; using a naive variance would badly
    overstate significance. This uses a Newey-West (Bartlett-kernel, ``hac_lag``)
    long-run variance and a standard-normal reference. A **negative** statistic
    means A has the lower average loss (A is better). Returns ``(stat, p_value)``.
    """
    a = np.asarray(actual, float)
    d = np.abs(a - np.asarray(pred_a, float)) - np.abs(a - np.asarray(pred_b, float))
    if loss == "squared":
        d = (a - np.asarray(pred_a, float)) ** 2 - (a - np.asarray(pred_b, float)) ** 2
    n = len(d)
    dm0 = d - d.mean()
    lrv = np.mean(dm0 ** 2)
    for lag in range(1, hac_lag + 1):
        cov = np.mean(dm0[lag:] * dm0[:-lag])
        lrv += 2.0 * (1.0 - lag / (hac_lag + 1.0)) * cov   # Bartlett weight
    if lrv <= 0:   # identical/degenerate forecasts -> no measurable difference
        return 0.0, 1.0
    stat = float(d.mean() / np.sqrt(lrv / n))
    p = float(2.0 * stats.norm.cdf(-abs(stat)))
    return stat, p


def significance_tests(preds: pd.DataFrame, hac_lag: int = 24) -> pd.DataFrame:
    """Diebold-Mariano tests down the ablation ladder: does each layer add real
    skill? Absolute-error loss, HAC variance. Pairs adapt to which forecasts are
    present so the France (ENTSO-E) and Spain (ESIOS) contributions are isolated."""
    y = preds["actual"].to_numpy()
    cols = preds.columns
    pairs = [("point", "naive_24h")]                       # full model vs naive
    if "point_es" in cols:
        pairs.append(("point", "point_es"))               # what ENTSO-E (France) adds
        pairs.append(("point_es", "naive_24h"))           # ESIOS model vs naive (backs the headline)
        if "point_price_only" in cols:
            pairs.append(("point_es", "point_price_only"))  # what ESIOS (Spain) adds
    elif "point_price_only" in cols:
        pairs.append(("point", "point_price_only"))       # what the exogenous set adds
    if "point_price_only" in cols:
        pairs.append(("point_price_only", "naive_24h"))
    if "lasso" in cols:
        pairs.append(("lasso", "naive_24h"))
    rows = []
    for a, b in pairs:
        if a not in preds.columns or b not in preds.columns:
            continue
        stat, p = diebold_mariano(y, preds[a].to_numpy(), preds[b].to_numpy(), hac_lag=hac_lag)
        rows.append(
            {
                "comparison": f"{a} vs {b}",
                "mae_a": round(float((preds[a] - preds["actual"]).abs().mean()), 3),
                "mae_b": round(float((preds[b] - preds["actual"]).abs().mean()), 3),
                "dm_stat": round(stat, 3),
                "p_value": p,
                "a_better_at_5pct": bool(stat < 0 and p < 0.05),
            }
        )
    return pd.DataFrame(rows)
