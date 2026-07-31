"""Exogenous drivers of the day-ahead price: REE/ESIOS day-ahead (D+1) forecasts
of demand, wind and solar PV for the Spanish peninsula.

Why forecasts, not realised values (the leakage point): the price for delivery
day D is set at the D-1 auction (~12:00 on D-1). At that moment nobody knows what
demand or renewable output day D will actually be — but REE has already published
its **day-ahead (D+1) forecasts**, so those are legitimately available to a price
forecaster. Using realised generation would be look-ahead leakage; using the
day-ahead forecast is exactly the information a market participant has.

ESIOS indicators used (peninsular, hourly):
    1775  Previsión diaria D+1 demanda           -> demand_fc   (MW)
    1777  Previsión diaria D+1 eólica            -> wind_fc     (MW)
    1779  Previsión diaria D+1 fotovoltaica      -> solar_fc    (MW)

The token is read from the ESIOS_TOKEN environment variable (or a local .env
file) and is never committed. The downloaded series is cached to
``data/exog/esios_forecasts.csv`` so the study reproduces without a token and so
we never re-request data we already hold (ESIOS's responsible-use terms).
"""
from __future__ import annotations

import os
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "exog" / "esios_forecasts.csv"

INDICATORS = {"demand_fc": 1775, "wind_fc": 1777, "solar_fc": 1779}
# Realised (metered, real-time) counterparts, used only to AUDIT that the
# forecasts above are genuine day-ahead forecasts and not outturn in disguise.
REALISED_INDICATORS = {"demand": 1293, "wind": 551, "solar": 552}
_BASE = "https://api.esios.ree.es/indicators"


def _token() -> str | None:
    """ESIOS token from the environment, falling back to a local .env file."""
    tok = os.environ.get("ESIOS_TOKEN")
    if tok:
        return tok.strip()
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("ESIOS_TOKEN="):
                return line.split("=", 1)[1].strip()
    return None


def _headers(token: str) -> dict:
    return {
        "Accept": "application/json; application/vnd.esios-api-v2+json",
        "Content-Type": "application/json",
        "x-api-key": token,
    }


def _fetch_indicator(ind: int, start: date, end: date, headers: dict,
                     session: requests.Session, trunc: str | None = "hour") -> pd.Series:
    """Fetch one indicator as a tz-naive local (peninsula) series.

    ``trunc="hour"`` asks ESIOS for hourly means (right for the D+1 forecasts).
    ``trunc=None`` returns the raw series (needed for real-time realised series,
    which ESIOS would otherwise *sum* to ~12x under an hourly truncation — the
    caller resamples those to an hourly mean).
    """
    params = {
        "start_date": f"{start:%Y-%m-%d}T00:00",
        "end_date": f"{end:%Y-%m-%d}T23:59",
    }
    if trunc:
        params["time_trunc"] = trunc
    resp = session.get(f"{_BASE}/{ind}", headers=headers, params=params, timeout=90)
    resp.raise_for_status()
    values = resp.json()["indicator"]["values"]
    if not values:
        return pd.Series(dtype=float)
    frame = pd.DataFrame(values)
    # keep the peninsular series if several geographies are returned
    if "geo_id" in frame and frame["geo_id"].nunique() > 1:
        frame = frame[frame["geo_id"] == 8741]
    ts = pd.to_datetime(frame["datetime"], utc=True).dt.tz_convert(
        "Europe/Madrid").dt.tz_localize(None)
    series = pd.Series(frame["value"].to_numpy(float), index=ts).sort_index()
    return series[~series.index.duplicated(keep="first")]


def _download(start: date, end: date) -> pd.DataFrame:
    token = _token()
    if not token:
        raise RuntimeError(
            "ESIOS_TOKEN not set. Put it in a .env file or the environment. "
            "Request a free personal token at https://www.esios.ree.es/es/pagina/api"
        )
    headers = _headers(token)
    cols = {}
    with requests.Session() as sess:
        for name, ind in INDICATORS.items():
            cols[name] = _fetch_indicator(ind, start, end, headers, sess)
            time.sleep(0.3)  # be polite to the API
    df = pd.DataFrame(cols)
    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    # Forward-fill the few missing hours. This never uses a future value: a filled
    # hour only inherits an already-published value from the past. (Interpolation
    # was avoided precisely because it blends in the next, later observation.)
    return df.reindex(full).ffill()


def _covers(cached: pd.DataFrame, start: date, end: date) -> bool:
    """Does the cache span [start, end]? Compared on dates (the cache is hourly)."""
    return cached.index.min().date() <= start and cached.index.max().date() >= end


def forecast_vs_realised(start: date, end: date) -> pd.DataFrame:
    """Audit that the exogenous features are genuine D+1 forecasts, not outturn.

    The whole leakage-free claim rests on 1775/1777/1779 being the day-ahead
    forecast published before the auction — not a value silently revised toward
    the realised outturn. This compares each forecast against its realised
    counterpart (ESIOS real-time series) and returns the normalised MAE. A real
    day-ahead forecast has a clearly non-zero error (demand ~1-3%, solar ~5-15%,
    wind ~15-30%); an nMAE near 0 would mean the "forecast" is really outturn and
    the features leak. Needs a token (hits realised series not in the cache).
    """
    token = _token()
    if not token:
        raise RuntimeError("ESIOS_TOKEN needed to fetch realised series for the audit.")
    headers = _headers(token)
    rows = []
    with requests.Session() as sess:
        for name in ("demand", "wind", "solar"):
            fc = _fetch_indicator(INDICATORS[f"{name}_fc"], start, end, headers, sess)
            real = _fetch_indicator(REALISED_INDICATORS[name], start, end, headers, sess,
                                    trunc=None).resample("h").mean()
            both = pd.concat({"fc": fc, "real": real}, axis=1).dropna()
            nmae = 100 * (both["fc"] - both["real"]).abs().mean() / both["real"].mean()
            rows.append({"series": name, "n_hours": len(both),
                         "nmae_pct": round(float(nmae), 1),
                         "corr": round(float(both["fc"].corr(both["real"])), 3)})
    return pd.DataFrame(rows)


def load_exog(start: date, end: date, refresh: bool = False) -> pd.DataFrame:
    """Return hourly [demand_fc, wind_fc, solar_fc] (MW) for [start, end].

    Uses the committed on-disk cache whenever it already covers the range (so we
    never re-request held data — ESIOS's responsible-use terms). Only downloads
    when the cache is missing or short *and* a token is available; if a download
    is needed but no token exists, falls back to the cache so the study still runs
    from the committed data without a token.
    """
    cached = pd.read_csv(CACHE, index_col=0, parse_dates=True) if CACHE.exists() else None
    if cached is not None and not refresh and _covers(cached, start, end):
        return cached
    try:
        df = _download(start, end)
    except Exception:
        if cached is not None:
            return cached  # best-effort: run from the committed cache without a token
        raise
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE)
    return df
