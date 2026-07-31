"""Cross-border exogenous driver: France's day-ahead LOAD forecast, from ENTSO-E.

Spain and France are linked by the Pyrenees interconnector, so how tight the
French system is spills into the Spanish price: high French demand pulls power
south and lifts the Spanish price.

⚠️ Leakage-critical timing. Only the **load** forecast is used as a model feature.
Under EU Reg. 543/2013 the day-ahead total-load forecast is published at least two
hours before day-ahead gate closure, so it is available at the ~12:00 D-1 MIBEL
auction. The day-ahead **wind/solar** generation forecast, however, has an 18:00
D-1 publication deadline — *after* the gate — so a forecaster could not rely on it
at auction time; using it would be look-ahead. Wind/solar are still fetched and
cached (for the forecast-vs-realised audit and transparency) but are NOT fed to
the model. See ``FR_EXOG_FEATURES`` in ``forecast.py``.

Fetched via the ENTSO-E Transparency Platform (``entsoe-py``), day-ahead (A01):
    query_load_forecast('FR')            -> fr_load_fc   (MW)   [used]
    query_wind_and_solar_forecast('FR')  -> fr_wind_fc, fr_solar_fc (MW)  [audit only]

The token is read from ENTSOE_TOKEN (env or .env) and never committed; the series
is cached to ``data/exog/entsoe_fr_forecasts.csv`` and committed so the study
reproduces without a token.
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "exog" / "entsoe_fr_forecasts.csv"
_ZONE = "FR"
_CHUNK_DAYS = 90  # keep each request well under the API's 1-year limit


def _token() -> str | None:
    tok = os.environ.get("ENTSOE_TOKEN")
    if tok:
        return tok.strip()
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("ENTSOE_TOKEN="):
                return line.split("=", 1)[1].strip()
    return None


def _to_madrid_hourly(series_or_df):
    """Resample a 15-min tz-aware series/frame to an hourly, Madrid-naive index."""
    out = series_or_df.tz_convert("Europe/Madrid").resample("h").mean()
    out.index = out.index.tz_localize(None)
    return out


def _download(start: date, end: date) -> pd.DataFrame:
    from entsoe import EntsoePandasClient  # imported lazily; only needed to refresh

    token = _token()
    if not token:
        raise RuntimeError("ENTSOE_TOKEN not set (env or .env). Generate one under "
                           "'My Account -> Web API Access' at transparency.entsoe.eu.")
    client = EntsoePandasClient(api_key=token)
    parts = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=_CHUNK_DAYS), end + timedelta(days=1))
        s = pd.Timestamp(chunk_start, tz="Europe/Madrid")
        e = pd.Timestamp(chunk_end, tz="Europe/Madrid")
        # process_type="A01" = day-ahead explicitly (do not rely on the library default;
        # it is load-bearing for the leakage claim on the load forecast).
        load = _to_madrid_hourly(
            client.query_load_forecast(_ZONE, start=s, end=e, process_type="A01")["Forecasted Load"])
        ws = _to_madrid_hourly(
            client.query_wind_and_solar_forecast(_ZONE, start=s, end=e, process_type="A01"))
        wind = ws.filter(like="Wind").sum(axis=1)   # onshore + offshore
        solar = ws["Solar"] if "Solar" in ws else pd.Series(0.0, index=ws.index)
        parts.append(pd.DataFrame({"fr_load_fc": load, "fr_wind_fc": wind, "fr_solar_fc": solar}))
        chunk_start = chunk_end
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    return df.reindex(full).ffill()   # forward-fill only; never blends in a future value


def forecast_vs_realised(start: date, end: date) -> pd.DataFrame:
    """Audit that the France features are day-ahead forecasts, not outturn.

    ENTSO-E's document types are day-ahead *by construction* (Day-ahead Total Load
    Forecast, Day-ahead Generation Forecast for Wind and Solar), but this checks
    it empirically anyway, mirroring the ESIOS audit: the forecast should differ
    from the realised load/generation by a genuine day-ahead margin, not ~0.
    Needs a token (hits realised series).
    """
    from entsoe import EntsoePandasClient

    token = _token()
    if not token:
        raise RuntimeError("ENTSOE_TOKEN needed to fetch realised series for the audit.")
    client = EntsoePandasClient(api_key=token)
    s = pd.Timestamp(start, tz="Europe/Madrid")
    e = pd.Timestamp(end + timedelta(days=1), tz="Europe/Madrid")
    fc_load = _to_madrid_hourly(client.query_load_forecast(_ZONE, start=s, end=e)["Forecasted Load"])
    real_load = _to_madrid_hourly(client.query_load(_ZONE, start=s, end=e)["Actual Load"])
    fc_ws = _to_madrid_hourly(client.query_wind_and_solar_forecast(_ZONE, start=s, end=e))
    gen = _to_madrid_hourly(client.query_generation(_ZONE, start=s, end=e))
    gen.columns = [c[0] if isinstance(c, tuple) else c for c in gen.columns]
    real_solar = gen.filter(like="Solar").sum(axis=1)
    real_wind = gen.filter(like="Wind").sum(axis=1)
    checks = {
        "load": (fc_load, real_load),
        "solar": (fc_ws["Solar"], real_solar),
        "wind": (fc_ws.filter(like="Wind").sum(axis=1), real_wind),
    }
    rows = []
    for name, (fc, real) in checks.items():
        both = pd.concat({"fc": fc, "real": real}, axis=1).dropna()
        nmae = 100 * (both["fc"] - both["real"]).abs().mean() / both["real"].mean()
        rows.append({"series": f"FR_{name}", "n_hours": len(both),
                     "nmae_pct": round(float(nmae), 1),
                     "corr": round(float(both["fc"].corr(both["real"])), 3)})
    return pd.DataFrame(rows)


def _covers(cached: pd.DataFrame, start: date, end: date) -> bool:
    return cached.index.min().date() <= start and cached.index.max().date() >= end


def load_entsoe(start: date, end: date, refresh: bool = False) -> pd.DataFrame:
    """Return hourly [fr_load_fc, fr_wind_fc, fr_solar_fc] (MW) for [start, end].

    Uses the committed cache when it covers the range; otherwise downloads (needs
    a token) and, if no token is available, falls back to the cache so the study
    still runs from committed data.
    """
    cached = pd.read_csv(CACHE, index_col=0, parse_dates=True) if CACHE.exists() else None
    if cached is not None and not refresh and _covers(cached, start, end):
        return cached
    try:
        df = _download(start, end)
    except Exception:
        if cached is not None:
            return cached
        raise
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE)
    return df
