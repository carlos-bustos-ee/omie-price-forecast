"""End-to-end pipeline: download OMIE prices, build the dataset, evaluate models."""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .dataset import build_series
from .download import download_range
from .forecast import evaluate, make_features

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=220, help="days of history to download")
    parser.add_argument("--test-days", type=int, default=28, help="hold-out length in days")
    args = parser.parse_args()

    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=args.days - 1)
    print(f"Downloading OMIE day-ahead prices {start} -> {end} ...")
    paths = download_range(start, end)
    print(f"  {len(paths)} daily files available")

    series = build_series(paths)
    processed = ROOT / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    series.to_csv(processed / "prices_hourly_es.csv")
    print(f"  {len(series)} hourly prices -> data/processed/prices_hourly_es.csv")

    df = make_features(series)
    metrics, preds = evaluate(df, test_days=args.test_days)

    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    metrics.to_csv(results / "metrics.csv", index=False)
    print()
    print(metrics.round(3).to_string(index=False))

    window = preds.tail(14 * 24)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(window.index, window["actual"], label="Actual", linewidth=1.4)
    ax.plot(window.index, window["gradient_boosting"], label="Gradient boosting", linewidth=1.1)
    ax.plot(window.index, window["naive_24h"], label="Naive 24h", linewidth=0.9, alpha=0.6)
    ax.set_title("OMIE Spain day-ahead price — last 14 days of the hold-out")
    ax.set_ylabel("EUR/MWh")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(results / "forecast_last14d.png", dpi=150)
    print("\nPlot -> results/forecast_last14d.png")


if __name__ == "__main__":
    main()
