# OMIE Day-Ahead Electricity Price Forecasting (Spain)

Point **and probabilistic** day-ahead forecasts for the Spanish zone of the Iberian
electricity market (MIBEL), using only public data from [OMIE](https://www.omie.es/),
the market operator. The pipeline downloads real market prices, engineers a
leakage-free feature set, and evaluates the models with a **walk-forward backtest**
that a trading desk would recognise — including a **conformally calibrated P10–P90
interval**, not just a point number.

**Headline (28-day walk-forward, weekly retrain, 418 days of history):**

| Model | MAE (€/MWh) | RMSE (€/MWh) | vs. naive-24h |
|---|---|---|---|
| Naive 24h (yesterday's price) | 19.14 | 30.17 | — |
| **Gradient boosting (point)** | **16.26** | **22.39** | **+15.1% MAE** |

Probabilistic: **pinball loss 5.29 €/MWh**, **P10–P90 coverage 81.4%** (nominal 80%) after conformal calibration.

| Point forecast | Probabilistic forecast |
|---|---|
| ![point](results/forecast_point_last14d.png) | ![fan](results/forecast_fan_last14d.png) |

---

## Why this is a fair evaluation (and not a leaky one)

Prices for delivery day **D** clear at the **D-1** auction (published ~13:00 CET the
day before). A real forecast for D, produced on the morning of D-1, already knows
every price up to the end of D-1 — and nothing after. Three design choices keep the
evaluation honest:

1. **Feature window ≥ 24 h.** Every predictor is a lag of 24 hours or more, so no
   feature ever uses information that wouldn't exist at forecast time. No leakage.
2. **Delta target.** Price *levels* are non-stationary and tree ensembles can't
   extrapolate beyond their training range, so the models learn the **correction
   against the 24h-lagged price** (the naive benchmark). This one detail is what makes
   the model beat the naive baseline across seasons instead of drifting with the level.
3. **Walk-forward backtest.** Instead of one lucky train/test split, the model is
   refit weekly and always scored on days it has *never seen*. The reported numbers
   are the average over a rolling 28-day out-of-sample window.

The naive-24h benchmark is genuinely hard to beat in this market (it *is* the standard
reference), so a double-digit MAE reduction using only price history and calendar
features is a solid, honest result.

## Probabilistic forecasting

A point forecast is not enough for a trading or risk decision — you need the
*distribution*. The pipeline trains gradient-boosting **quantile** models (P10/P50/P90)
and then applies a **split-conformal calibration**: the raw quantile band is
overconfident (empirical coverage ≈ 64% vs. a nominal 80%), so a width multiplier is
learned on held-out data and applied to the interval. Calibrated coverage lands at
**81.4%**, and the band visibly widens in volatile ramps and the solar-driven zero-price
troughs — where uncertainty really is higher.

## Where the model earns its keep (error by hour of day)

A single headline MAE hides *where* the gain comes from. Breaking the backtest down by
hour of day tells the real story:

![error by hour](results/error_by_hour.png)

- **Overnight (00–06h) it's basically a tie.** Prices are flat, so "yesterday, same hour"
  is already a strong forecast (naive MAE ≈ 11 €/MWh) and the model barely moves it — at a
  couple of hours it is even marginally worse. There is little structure left to capture.
- **The model earns its keep in the ramps.** In the morning solar build-up (08–10h) the
  naive error blows up to **32 €/MWh** while the model holds ~22 — a **~30% cut**. The same
  happens on the evening ramp: hour 18 goes 29.8 → 20.7 (**−31%**) and hour 20 goes
  16.4 → 10.7 (**−35%**). Exactly the hours a trader cares about are the hours the model
  helps most.
- **Coverage is honest but not flat.** The P10–P90 band averages ~80% but runs from **68%**
  in the most volatile midday hours to **>90%** in the quiet ones — i.e. the interval is a
  touch narrow precisely when volatility peaks. That is a concrete, measured motivation for
  the hour-conditional calibration on the roadmap, not a guess.

Numbers per hour are in `results/metrics_by_hour.csv`.

## Data

- **Source:** OMIE public daily files (`marginalpdbc_YYYYMMDD.1`), downloaded and cached
  by `src/download.py`. No API key required.
- **Granularity:** since the EU switch to 15-minute market time units (Oct 2025) the
  files carry 96 periods/day; earlier files are hourly. Quarter-hours are averaged into
  hours so the whole history is comparable (`src/dataset.py`).
- **History used:** 418 delivery days (~10,000 hourly prices).

## Run it

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Windows (use bin/ on Linux/Mac)
.venv/Scripts/python -m src.pipeline --days 420 --test-days 28 --retrain-every 7
```

Outputs (in `results/`): `metrics.csv`, `metrics_probabilistic.csv`,
`metrics_by_hour.csv`, `backtest_predictions.csv`, and the three plots above.

## Project layout

```
src/
  download.py    # fetch + cache OMIE daily price files
  dataset.py     # parse (hourly + 15-min formats) into a tidy hourly series
  forecast.py    # features, point + quantile models, conformal calibration, backtest, scoring
  pipeline.py    # end-to-end: download -> dataset -> backtest -> metrics + plots
```

## Roadmap

- **Exogenous drivers** — demand, wind and solar forecasts from ENTSO-E and ESIOS
  (REE); the biggest remaining lever on point error.
- **Benchmark vs. the literature** — compare against LEAR / DNN from
  [`epftoolbox`](https://github.com/jeslago/epftoolbox) with a Diebold–Mariano test.
- **Hour-conditional conformal calibration** — the coverage-by-hour analysis above shows
  the band is too narrow midday and too wide overnight; calibrating the width per hour
  should flatten coverage to the 80% target across the day.
- **Rolling-origin backtest across full seasons** (this window is 28 days of summer; a
  full-year rolling origin would test the model through the seasonal price regimes).

## Author

Carlos Bustos — electrical engineering student (UCLM, Spain), focused on power systems
and energy markets. Built with an AI-assisted workflow.
