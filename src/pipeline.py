"""End-to-end pipeline: download OMIE prices, build the dataset, run a
walk-forward backtest of point + probabilistic forecasts, and plot the results.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .dataset import build_series
from .download import download_range
from .forecast import backtest, make_features, score

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=420, help="days of history to download")
    parser.add_argument("--test-days", type=int, default=28, help="walk-forward window (days)")
    parser.add_argument("--retrain-every", type=int, default=7, help="retrain cadence (days)")
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
    print(f"  walk-forward backtest: last {args.test_days} days, retrain every "
          f"{args.retrain_every} days ...")
    preds = backtest(df, test_days=args.test_days, retrain_every=args.retrain_every)
    metrics, prob = score(preds)

    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    preds.to_csv(results / "backtest_predictions.csv")
    metrics.to_csv(results / "metrics.csv", index=False)
    prob.to_csv(results / "metrics_probabilistic.csv", index=False)
    print()
    print(metrics.round(3).to_string(index=False))
    print()
    print(prob.to_string(index=False))

    window = preds.tail(14 * 24)

    # 1) point forecast vs actual vs naive
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(window.index, window["actual"], label="Actual", linewidth=1.4)
    ax.plot(window.index, window["point"], label="Gradient boosting (point)", linewidth=1.1)
    ax.plot(window.index, window["naive_24h"], label="Naive 24h", linewidth=0.9, alpha=0.55)
    ax.set_title("OMIE Spain day-ahead price — point forecast (last 14 days of backtest)")
    ax.set_ylabel("EUR/MWh")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(results / "forecast_point_last14d.png", dpi=150)

    # 2) probabilistic fan chart (P10-P90 band + P50 + actual)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(window.index, window["q10"], window["q90"], alpha=0.25,
                    label="P10-P90 interval", color="tab:orange")
    ax.plot(window.index, window["q50"], label="P50 (median)", linewidth=1.0, color="tab:orange")
    ax.plot(window.index, window["actual"], label="Actual", linewidth=1.3, color="tab:blue")
    ax.set_title("OMIE Spain day-ahead price — probabilistic forecast (last 14 days of backtest)")
    ax.set_ylabel("EUR/MWh")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(results / "forecast_fan_last14d.png", dpi=150)

    print("\nPlots -> results/forecast_point_last14d.png, results/forecast_fan_last14d.png")


if __name__ == "__main__":
    main()
