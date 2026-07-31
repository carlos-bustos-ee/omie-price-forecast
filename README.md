# OMIE Day-Ahead Electricity Price Forecasting (Spain)

Point **and probabilistic** day-ahead forecasts for the Spanish zone of the Iberian
electricity market (MIBEL). The pipeline downloads real market prices from
[OMIE](https://www.omie.es/) and the day-ahead **demand / wind / solar forecasts**
from [ESIOS (REE)](https://www.esios.ree.es/), engineers a feature set that is
**leakage-free by construction and verified empirically** (below), and evaluates the models
with a **walk-forward backtest** a trading desk would
recognise — a naive and a LASSO benchmark, a **conformally calibrated P10–P90
interval**, and a **Diebold–Mariano significance test**, not just a single point number.

**Headline (120-day walk-forward, weekly retrain, ~14 months of history):**

| Model | MAE (€/MWh) | RMSE (€/MWh) | vs. naive-24h | Better than naive? (DM) |
|---|---|---|---|---|
| Naive 24h (yesterday's price) | 16.28 | 24.87 | — | — |
| LASSO autoregressive baseline | 15.39 | 21.48 | +5.5% | Barely (p = 0.03) |
| Gradient boosting, price-only | 15.37 | 21.25 | +5.6% | No (p = 0.18) |
| **Gradient boosting + ESIOS forecasts** | **9.67** | **13.54** | **+40.6%** | **Yes (p ≈ 2e-17)** |

Probabilistic (the +ESIOS model): **pinball loss 3.26 €/MWh**, **P10–P90 coverage 83.9%**
(nominal 80%) after conformal calibration.

| Point forecast | Probabilistic forecast |
|---|---|
| ![point](results/forecast_point_last14d.png) | ![fan](results/forecast_fan_last14d.png) |

The one lesson this project is built to show: **on ~14 months of data, autoregressive
price history alone cannot beat the naive benchmark by a statistically significant margin —
but the day-ahead demand/wind/solar forecast can, decisively.** The next section proves it
rather than asserting it.

---

## What actually moves the needle (and how we know it isn't luck)

A "−40% MAE" headline is worthless without asking whether it could be noise, and *which*
ingredient earned it. Hourly forecast errors are strongly autocorrelated, so the
**effective** sample is far smaller than the 2,880 test hours; the
[Diebold–Mariano](https://en.wikipedia.org/wiki/Diebold%E2%80%93Mariano_test) test with a
Newey–West (HAC) long-run variance is the honest check (`src/forecast.py: diebold_mariano`).
Reading the ablation bottom-up:

| Comparison | MAE (a) | MAE (b) | DM stat | p-value | a better at 5%? |
|---|---|---|---|---|---|
| LASSO vs naive | 15.39 | 16.28 | −2.14 | 0.03 | **Barely** |
| GBM price-only vs naive | 15.37 | 16.28 | −1.34 | 0.18 | **No** |
| **GBM +ESIOS vs GBM price-only** | 9.67 | 15.37 | −11.15 | 7e-29 | **Yes** |
| **GBM +ESIOS vs naive** | 9.67 | 16.28 | −8.51 | 2e-17 | **Yes** |

- **Autoregression alone barely moves the needle.** The linear LASSO baseline squeaks past
  naive (−5.5%, p = 0.03) and the price-only gradient boosting doesn't clear the bar at all
  (−5.6% but p = 0.18) — a marginal edge at best from price history and calendar alone. An
  earlier **28-day summer**
  window of this same pipeline showed a flashy **+15% MAE**; extending to 120 days and
  running DM revealed that was a seasonal artifact, not a durable edge. That is exactly why
  you backtest over a long, varied window and test significance instead of quoting one lucky
  month. (In much of the electricity-price-forecasting literature an autoregressive LEAR
  model *does* clearly beat naive; that it barely does here is itself informative — the 2025–26
  Spanish market is so dominated by swings between triple-digit evening peaks and zero/near-zero
  midday solar hours that yesterday's price is a weak guide, and the fundamentals do the work.)
- **The exogenous forecast is the edge.** Adding the ESIOS day-ahead demand/wind/solar
  forecasts cuts MAE by a further ~37% (15.37 → 9.67) with a DM statistic of −11, i.e.
  overwhelmingly significant. This is the market intuition made quantitative: in a
  renewables-dominated system the clearing price is set by the **net load** (demand minus
  wind minus solar), and the day-ahead forecast of that net load is public before the
  auction. Knowing tomorrow's solar tells you when the price will collapse to zero — the
  autoregressive model, however good, cannot.

## Why this is a fair evaluation (and not a leaky one)

Prices for delivery day **D** clear at the **D-1** auction (~12:00 CET the day before). A
real forecast for D, produced that morning, may only use information already public by then.
Four design choices keep the evaluation honest:

1. **Price features are lagged ≥ 24 h.** Every autoregressive predictor is a lag of 24 hours
   or more, so none uses a price that wouldn't exist at forecast time.
2. **Exogenous features are day-ahead *forecasts*, not realised values.** Using the realised
   demand/wind/solar of day D would be look-ahead leakage — nobody knows it at the auction.
   The pipeline uses ESIOS's **"Previsión diaria D+1"** series (indicators 1775/1777/1779):
   these are the forecasts REE publishes the day before, so they are genuinely available to a
   market participant, and they are exactly what a real price forecaster feeds in.
3. **Delta target.** Price *levels* are non-stationary and tree ensembles can't extrapolate
   beyond their training range, so every model learns the **correction against the 24h-lagged
   price**; all models are compared on identical footing.
4. **Walk-forward backtest.** Models are refit weekly and always scored on days they have
   *never seen*; the numbers are the average over a rolling 120-day out-of-sample window.

**The exogenous features are verified to be genuine forecasts, not outturn in disguise.**
The whole leakage-free claim on point 2 rests on the ESIOS series being the day-ahead
forecast and not a value silently revised toward the realised outturn. That is *tested*, not
assumed (`src/exog.py: forecast_vs_realised`): each forecast is compared against its realised
counterpart, and the errors are exactly those of a real day-ahead forecast — nowhere near the
~0 that outturn contamination would produce.

| Series | forecast vs realised, normalised MAE | typical day-ahead error |
|---|---|---|
| Demand | 1.5% | ~1–3% ✓ |
| Solar PV | 16.1% | ~5–15% ✓ |
| Wind | 22.6% | ~15–30% ✓ |

A 22.6% wind error is emphatically a *forecast*; if the feature were leaking outturn it would
be near zero and the whole −40% would be spurious. It isn't.

## Where the model earns its keep (error by hour of day)

![error by hour](results/error_by_hour.png)

Breaking the 120-day backtest down by hour shows the exogenous model gains everywhere, but
most in the **morning solar ramp** (hours 7–9, up to **−56%** vs naive) and the **evening
ramp** (hours 18–19, ~**−52%**) — precisely the volatile hours a trader cares about, and
precisely where knowing tomorrow's wind and solar matters. The P10–P90 band tracks the 80%
target across the day (per-hour coverage 75–91%). Numbers per hour: `results/metrics_by_hour.csv`.

## Probabilistic forecasting

A point forecast is not enough for a risk decision — you need the *distribution*. The
pipeline trains gradient-boosting **quantile** models (P10/P50/P90) and applies a
**split-conformal calibration**: the raw quantile band is overconfident, so a width
multiplier learned on held-out data widens it to the nominal 80%. Calibrated coverage lands
at **83.9%**, widening in the volatile ramps and solar-driven troughs where uncertainty is
genuinely higher.

### Experiment: does hour-conditional calibration help?

The obvious refinement is to calibrate the conformal width **per hour** rather than globally.
It's implemented (`backtest(..., calibration="hourly")`) and tested head-to-head over the
120-day window:

| Calibration | Pinball ↓ | Overall coverage | Hourly coverage spread (mean \|cov−80%\|) |
|---|---|---|---|
| **Global (shipped)** | 3.264 | 83.9% | 4.5 pp |
| Hourly | 3.260 | 81.8% | 3.2 pp |

With the stronger (+ESIOS) model the two are essentially tied on the proper scoring rule
(pinball), while hourly nudges overall coverage closer to 80% and flattens the per-hour
spread. The gain is marginal and buys 24 extra parameters estimated on a 21-day window, so
the shipped default stays **global** for robustness; `calibration="hourly"` is available when
more calibration history warrants it. (On the earlier price-only model hourly clearly
*overfit* — worse pinball; the better base model narrowed the gap.)

## Data

- **Prices** — OMIE public daily files (`marginalpdbc_YYYYMMDD.1`), downloaded and cached by
  `src/download.py`. No key required. Since the EU 15-minute market switch (Oct 2025) files
  carry 96 periods/day; quarter-hours are averaged to hours so the whole history is
  comparable (`src/dataset.py`).
- **Exogenous forecasts** — ESIOS (REE) day-ahead demand (1775), wind (1777) and solar PV
  (1779), peninsular, hourly, via `src/exog.py`. Needs a free personal token (below). The
  downloaded series is cached to `data/exog/esios_forecasts.csv` and **committed**, so the
  study reproduces without a token and no data is ever re-requested (ESIOS's responsible-use
  terms).
- **History used:** ~420 delivery days (~10,000 hourly prices).

## Run it

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Windows (use bin/ on Linux/Mac)
.venv/Scripts/python -m src.pipeline --days 420 --test-days 120 --retrain-every 7
```

The cached ESIOS data is committed, so it runs out of the box. To **refresh** the exogenous
data you need a free ESIOS token (request at <https://www.esios.ree.es/es/pagina/api>): put
it in a `.env` file as `ESIOS_TOKEN=...` (git-ignored). To run the **price-only** model
without any token, pass `--no-exog`.

Outputs (in `results/`): `metrics.csv`, `metrics_probabilistic.csv`, `metrics_by_hour.csv`,
`significance_dm.csv`, `backtest_predictions.csv`, and the three plots above.

## Project layout

```
src/
  download.py    # fetch + cache OMIE daily price files
  dataset.py     # parse (hourly + 15-min formats) into a tidy hourly series
  exog.py        # fetch + cache ESIOS day-ahead demand/wind/solar forecasts
  forecast.py    # features (autoregressive + exogenous), GB + quantile + LASSO models,
                 # conformal calibration, walk-forward backtest, scoring, Diebold-Mariano
  pipeline.py    # end-to-end: download -> dataset + exog -> backtest -> metrics + plots
```

## Limitations & honest caveats

- **Forecast, not actuals.** The exogenous features are ESIOS's archived day-ahead
  ("Previsión diaria D+1") forecasts. If those series were ever silently revised with
  outturn, some optimism could leak in; they are published as the day-ahead forecast and used
  as such here.
- **~14 months only.** OMIE exposes a limited public history; a longer record would tighten
  every estimate. The DST period-to-hour mapping is approximate (fine for research, not
  settlement).
- **Point + interval, not a full density**, and no trading/P&L layer — the target is forecast
  accuracy, honestly measured.

## Roadmap

- **ENTSO-E cross-border features** — interconnector flows and neighbouring-area load, to
  complement the domestic ESIOS drivers (API access granted and token in hand; integration next).
- **Stronger literature benchmark** — the full LEAR / DNN from
  [`epftoolbox`](https://github.com/jeslago/epftoolbox) alongside the current LASSO.
- **Longer history / rolling-origin backtest** — to shrink the confidence intervals further.

## Author

Carlos Bustos — electrical engineering student (UCLM, Spain), focused on power systems
and energy markets. Built with an AI-assisted workflow.
