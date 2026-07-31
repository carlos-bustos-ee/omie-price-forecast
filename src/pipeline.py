"""End-to-end pipeline: download OMIE prices, build the dataset, run a
walk-forward backtest of point + probabilistic forecasts against a naive and a
LASSO benchmark, test the differences with Diebold-Mariano, and plot the results.
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
from .exog import load_exog
from .forecast import (
    BASE_FEATURES,
    backtest,
    feature_columns,
    make_features,
    score,
    score_by_hour,
    significance_tests,
)

ROOT = Path(__file__).resolve().parent.parent


def run(series, exog=None, test_days: int = 120, retrain_every: int = 7) -> None:
    """Backtest -> score -> significance -> save artifacts + plots. Takes a tidy
    hourly price series (and optional ESIOS exogenous forecasts) so it can be
    driven from a live download or a cached CSV.

    With ``exog`` present the headline gradient-boosting model uses the day-ahead
    demand/wind/solar forecasts; a second price-only backtest is run so the
    ablation (``point`` vs ``point_price_only``) isolates what the exogenous
    forecasts actually add, tested for significance with Diebold-Mariano.
    """
    df = make_features(series, exog=exog)
    print(f"  walk-forward backtest: last {test_days} days, retrain every "
          f"{retrain_every} days ...")
    preds = backtest(df, test_days=test_days, retrain_every=retrain_every,
                     features=feature_columns(df))
    if exog is not None:
        base = backtest(df, test_days=test_days, retrain_every=retrain_every,
                        features=BASE_FEATURES)
        preds["point_price_only"] = base["point"]
        preds["lasso"] = base["lasso"]  # report the standard price-only LEAR baseline
    metrics, prob = score(preds)
    by_hour = score_by_hour(preds)
    sig = significance_tests(preds)

    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    preds.to_csv(results / "backtest_predictions.csv")
    metrics.to_csv(results / "metrics.csv", index=False)
    prob.to_csv(results / "metrics_probabilistic.csv", index=False)
    by_hour.to_csv(results / "metrics_by_hour.csv")
    sig.to_csv(results / "significance_dm.csv", index=False)

    print()
    print(metrics.round(3).to_string(index=False))
    print()
    print(prob.to_string(index=False))
    print()
    print("Diebold-Mariano tests (abs-error loss, HAC variance):")
    print(sig.to_string(index=False))
    print()
    print("Per-hour breakdown (naive vs model MAE, EUR/MWh):")
    print(by_hour.round(2).to_string())

    window = preds.tail(14 * 24)

    # 1) point forecast vs actual vs naive (+ the exogenous-features gain if present)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(window.index, window["actual"], label="Actual", linewidth=1.4)
    gbm_label = ("Gradient boosting (+ ESIOS forecasts)" if "point_price_only" in preds.columns
                 else "Gradient boosting (point)")
    ax.plot(window.index, window["point"], label=gbm_label, linewidth=1.1)
    if "point_price_only" in preds.columns:
        ax.plot(window.index, window["point_price_only"], label="Gradient boosting (price-only)",
                linewidth=0.9, color="tab:green", alpha=0.7)
    ax.plot(window.index, window["naive_24h"], label="Naive 24h", linewidth=0.9, alpha=0.5)
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

    # 3) error-by-hour: where the model beats naive, and coverage across the day
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(by_hour.index, by_hour["naive_mae_eur_mwh"], "o-", color="tab:gray",
            alpha=0.7, label="Naive 24h MAE")
    ax.plot(by_hour.index, by_hour["point_mae_eur_mwh"], "o-", color="tab:blue",
            label="Gradient boosting MAE")
    ax.fill_between(by_hour.index, by_hour["point_mae_eur_mwh"],
                    by_hour["naive_mae_eur_mwh"], color="tab:green", alpha=0.15,
                    label="MAE reduction (model gain)")
    ax.set_title(f"OMIE Spain day-ahead price — error by hour of day ({test_days}-day backtest)")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("MAE (EUR/MWh)")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(loc="upper left")

    ax2 = ax.twinx()
    ax2.plot(by_hour.index, by_hour["coverage_P10_P90"] * 100, "s--", color="tab:orange",
             alpha=0.6, linewidth=1.0, label="P10-P90 coverage")
    ax2.axhline(80, color="tab:orange", linestyle=":", alpha=0.5)
    ax2.set_ylabel("P10-P90 coverage (%)", color="tab:orange")
    ax2.set_ylim(0, 105)
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(results / "error_by_hour.png", dpi=150)

    print("\nPlots -> results/forecast_point_last14d.png, results/forecast_fan_last14d.png, "
          "results/error_by_hour.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=420, help="days of history to download")
    parser.add_argument("--test-days", type=int, default=120, help="walk-forward window (days)")
    parser.add_argument("--retrain-every", type=int, default=7, help="retrain cadence (days)")
    parser.add_argument("--no-exog", action="store_true",
                        help="skip ESIOS exogenous forecasts (price-only model)")
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

    exog = None
    if not args.no_exog:
        try:
            exog = load_exog(series.index.min().date(), series.index.max().date())
            print(f"  {len(exog)} hourly ESIOS forecasts (demand/wind/solar) -> data/exog/")
        except Exception as err:  # no token and no cache -> fall back to price-only
            print(f"  ESIOS forecasts unavailable ({err}); running price-only.")

    run(series, exog=exog, test_days=args.test_days, retrain_every=args.retrain_every)


if __name__ == "__main__":
    main()
