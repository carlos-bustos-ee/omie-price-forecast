"""Parse raw OMIE files into a tidy hourly price series for Spain."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


def parse_file(path: Path) -> pd.DataFrame:
    """Return a frame [timestamp, price_eur_mwh] for one delivery day.

    File rows look like ``2026;07;21;13;162.28;162.28;`` with columns
    year;month;day;period;price_pt;price_es. Files carry 24 hourly periods up
    to Sep-2025 and 96 quarter-hourly periods afterwards. Quarter-hours are
    averaged into hours so the whole history is comparable. On DST-change days
    the period-to-hour mapping is approximate (fine for forecasting research,
    not meant for settlement).
    """
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = [p for p in line.strip().split(";") if p != ""]
        if len(parts) < 6 or not parts[0].isdigit():
            continue
        y, m, d, period = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        rows.append((date(y, m, d), period, float(parts[5])))
    if not rows:
        return pd.DataFrame(columns=["timestamp", "price_eur_mwh"])
    frame = pd.DataFrame(rows, columns=["day", "period", "price"])
    quarter_hourly = len(frame) >= 90
    frame["hour"] = (frame["period"] - 1) // 4 if quarter_hourly else frame["period"] - 1
    hourly = frame.groupby(["day", "hour"], as_index=False)["price"].mean()
    hourly = hourly[hourly["hour"] < 24]
    hourly["timestamp"] = pd.to_datetime(hourly["day"]) + pd.to_timedelta(
        hourly["hour"], unit="h"
    )
    return hourly[["timestamp", "price"]].rename(columns={"price": "price_eur_mwh"})


def build_series(paths: list[Path]) -> pd.DataFrame:
    """Concatenate daily files into one continuous hourly series."""
    frames = [parse_file(p) for p in sorted(paths)]
    series = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    series = series.drop_duplicates("timestamp").set_index("timestamp")
    # Reindex to a complete hourly grid; forward-fill the few DST gaps so lag
    # features stay aligned.
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="h")
    series = series.reindex(full_index).ffill()
    series.index.name = "timestamp"
    return series
