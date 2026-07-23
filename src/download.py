"""Download raw day-ahead marginal price files from OMIE.

OMIE (the Iberian day-ahead market operator) publishes one public file per
delivery day (``marginalpdbc_YYYYMMDD.1``) with the marginal prices for
Portugal and Spain. Since October 2025 the market trading unit is 15 minutes,
so current files carry 96 periods per day (92/100 on DST-change days).
"""
from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path

import requests

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
URL = (
    "https://www.omie.es/es/file-download"
    "?parents%5B0%5D=marginalpdbc&filename=marginalpdbc_{stamp}.1"
)
HEADERS = {"User-Agent": "omie-price-forecast (academic project)"}


def fetch_day(day: date, session: requests.Session | None = None) -> Path | None:
    """Download one delivery day, cache it under data/raw/ and return the path."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / f"marginalpdbc_{day:%Y%m%d}.1"
    if target.exists() and target.stat().st_size > 0:
        return target
    sess = session or requests.Session()
    try:
        resp = sess.get(URL.format(stamp=f"{day:%Y%m%d}"), headers=HEADERS, timeout=30)
    except requests.RequestException:
        return None
    if resp.status_code != 200 or "MARGINALPDBC" not in resp.text[:40]:
        return None
    target.write_text(resp.text, encoding="utf-8")
    return target


def download_range(start: date, end: date, pause: float = 0.25) -> list[Path]:
    """Download every delivery day in [start, end]; skip days OMIE has not published."""
    paths: list[Path] = []
    with requests.Session() as sess:
        day = start
        while day <= end:
            cached = RAW_DIR / f"marginalpdbc_{day:%Y%m%d}.1"
            path = cached if cached.exists() else fetch_day(day, sess)
            if path is not None:
                paths.append(path)
                if not cached.exists():
                    time.sleep(pause)
            day += timedelta(days=1)
    return paths
