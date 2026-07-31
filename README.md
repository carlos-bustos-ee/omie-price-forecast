# OMIE Day-Ahead Electricity Price Forecasting (Spain)

Point **and probabilistic** day-ahead forecasts for the Spanish zone of the Iberian
electricity market (MIBEL). The pipeline downloads real market prices from
[OMIE](https://www.omie.es/) and the day-ahead **demand / wind / solar forecasts** from
[ESIOS (REE)](https://www.esios.ree.es/), engineers a feature set that is **leakage-free by
construction and verified empirically**, and evaluates the models with a **walk-forward
backtest** a trading desk would recognise — a naive and a LASSO benchmark, a **conformally
calibrated P10–P90 interval**, and a **Diebold–Mariano significance test**, not just a single
point number. It also documents a **cross-border experiment (France, ENTSO-E) whose headline
gain turned out to be look-ahead leakage** — caught, killed, and written up below.

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

The one lesson this project is built to show: **on ~14 months of data, autoregressive price
history alone barely beats the naive benchmark — but the day-ahead demand/wind/solar forecast
beats it decisively.** The next section proves it rather than asserting it.

---

## What actually moves the needle (and how we know it isn't luck)

A "−40% MAE" headline is worthless without asking whether it could be noise, and *which*
ingredient earned it. Hourly forecast errors are strongly autocorrelated, so the **effective**
sample is far smaller than the 2,880 test hours; the
[Diebold–Mariano](https://en.wikipedia.org/wiki/Diebold%E2%80%93Mariano_test) test with a
Newey–West (HAC) long-run variance is the honest check (`src/forecast.py: diebold_mariano`).
Reading the ablation bottom-up:

| Comparison | MAE (a) | MAE (b) | DM stat | p-value | a better at 5%? |
|---|---|---|---|---|---|
| LASSO vs naive | 15.39 | 16.28 | −2.14 | 0.03 | **Barely** |
| GBM price-only vs naive | 15.37 | 16.28 | −1.34 | 0.18 | **No** |
| **GBM +ESIOS vs GBM price-only** | 9.67 | 15.37 | −11.15 | 7e-29 | **Yes** |
| **GBM +ESIOS vs naive** | 9.67 | 16.28 | −8.51 | 2e-17 | **Yes** |

- **Autoregression alone barely moves the needle.** The linear LASSO squeaks past naive
  (−5.5%, p = 0.03) and the price-only gradient boosting doesn't clear the bar at all
  (−5.6%, p = 0.18). An earlier **28-day summer** window of this pipeline showed a flashy
  **+15% MAE**; extending to 120 days and running DM revealed that was a seasonal artifact,
  not a durable edge — which is exactly why you backtest over a long, varied window and test
  significance. (In much of the price-forecasting literature an autoregressive LEAR model
  *does* clearly beat naive; that it barely does here is informative — the 2025–26 Spanish
  market is so dominated by swings between triple-digit evening peaks and zero/near-zero midday
  solar hours that yesterday's price is a weak guide, and the fundamentals do the work.)
- **The forecast of fundamentals is the edge.** Adding the ESIOS day-ahead demand/wind/solar
  forecasts cuts MAE by a further ~37% (15.37 → 9.67), DM statistic −11 — overwhelmingly
  significant. In a renewables-dominated system the clearing price is set by the **net load**
  (demand − wind − solar), whose day-ahead forecast is public before the auction. Knowing
  tomorrow's solar tells you when the price will collapse to zero; the autoregressive model,
  however good, cannot.

## Why this is a fair evaluation (and not a leaky one)

Prices for delivery day **D** clear at the **D-1** auction (~12:00 CET the day before). A real
forecast for D, produced that morning, may only use information already public by then. Four
design choices keep the evaluation honest:

1. **Price features are lagged ≥ 24 h.** Every autoregressive predictor is a lag of 24 hours or
   more, so none uses a price that wouldn't exist at forecast time.
2. **Exogenous features are day-ahead *forecasts*, not realised values.** Using the realised
   demand/wind/solar of day D would be look-ahead leakage. The pipeline uses ESIOS's
   **"Previsión diaria D+1"** demand/wind/solar (indicators 1775/1777/1779), published the
   morning of D-1, so they are genuinely available to a market participant at the auction.
3. **Delta target.** Price *levels* are non-stationary and tree ensembles can't extrapolate
   beyond their training range, so every model learns the **correction against the 24h-lagged
   price**; all models are compared on identical footing.
4. **Walk-forward backtest.** Models are refit weekly and always scored on days they have
   *never seen*; the numbers are the average over a rolling 120-day out-of-sample window.

**The exogenous features are verified to be genuine forecasts, not outturn in disguise.** The
leakage-free claim on point 2 rests on the ESIOS series being the day-ahead forecast, not a
value silently revised toward the realised outturn. That is *tested* (`src/exog.py:
forecast_vs_realised`): each forecast is compared against its realised counterpart, and the
errors are exactly those of a real day-ahead forecast — nowhere near the ~0 that outturn
contamination would produce.

| Series | forecast vs realised, normalised MAE | typical day-ahead error |
|---|---|---|
| Demand | 1.5% | ~1–3% ✓ |
| Solar PV | 16.1% | ~5–15% ✓ |
| Wind | 22.6% | ~15–30% ✓ |

## Cross-border features: a leakage cautionary tale

Spain and France share the Pyrenees interconnector, so French tightness should spill into the
Spanish price — a natural next feature. Adding France's day-ahead **net load** (demand − wind −
solar) from [ENTSO-E](https://transparency.entsoe.eu/) looked like a clean win: MAE **9.67 →
9.22 (−4.7%)**, Diebold–Mariano **p = 0.002**.

**It was look-ahead leakage, and here is the tell.** Under EU Reg. 543/2013 the day-ahead
**wind/solar** generation forecast is published at **18:00 D-1 — *after* the ~12:00 MIBEL
gate**, so a forecaster does not have it at auction time. Only the **load** forecast is
published early enough (≥2 h before gate). Rebuilding the France feature from the **load
forecast alone** — the only leakage-safe part — the gain collapses to **9.67 → 9.64,
p = 0.76: not significant.** The apparent edge was the unavailable-at-gate renewable forecast
leaking in.

Two things worth stating:

- The France series *are* genuine day-ahead forecasts (forecast-vs-realised nMAE 1.9% load /
  6.5% solar / 12.3% wind), so this is **not** outturn-in-disguise — it is a *publication-time*
  leak. A feature isn't fair just because it's a forecast; it also has to be published **before
  your decision**. This project checks both failure modes.
- A 21-day window couldn't even distinguish the (leaky) France gain from noise (p = 0.22);
  only the 120-day window flagged it as "significant" — a reminder that more power surfaces
  effects, real *and* spurious, so the leakage check has to be structural, not statistical.

So the **shipped model uses Spain (ESIOS) only.** The ENTSO-E integration is kept as a
documented negative result (`src/exog_entsoe.py`, run with `--with-entsoe`). A headline gain
that evaporates under a publication-time check was never skill.

## Where the model earns its keep (error by hour of day)

![error by hour](results/error_by_hour.png)

Breaking the 120-day backtest down by hour: the model gains most in the **morning solar ramp**
(hours 7–9, up to ~**−56%** vs naive) and the **evening ramp** (hours 18–19, ~**−52%**) —
precisely the volatile hours a trader cares about, and where knowing tomorrow's wind and solar
matters. The P10–P90 band tracks the 80% target across the day (per-hour coverage 75–91%).
Numbers per hour: `results/metrics_by_hour.csv`.

## Probabilistic forecasting

A point forecast is not enough for a risk decision — you need the *distribution*. The pipeline
trains gradient-boosting **quantile** models (P10/P50/P90) and applies a **split-conformal
calibration**: the raw quantile band is overconfident, so a width multiplier learned on
held-out data widens it to the nominal 80%. Calibrated coverage lands at **83.9%**, widening in
the volatile ramps and solar-driven troughs where uncertainty is genuinely higher.

### Experiment: does hour-conditional calibration help?

The obvious refinement is to calibrate the conformal width **per hour** rather than globally.
It's implemented (`backtest(..., calibration="hourly")`) and tested head-to-head over the
120-day window:

| Calibration | Pinball ↓ | Overall coverage | Hourly coverage spread (mean \|cov−80%\|) |
|---|---|---|---|
| **Global (shipped)** | 3.264 | 83.9% | 4.5 pp |
| Hourly | 3.260 | 81.8% | 3.2 pp |

The two are essentially tied on the proper scoring rule (pinball), while hourly nudges overall
coverage closer to 80% and flattens the per-hour spread — but it buys 24 extra parameters
estimated on only a 21-day window. The shipped default stays **global** for robustness;
`calibration="hourly"` is available when more calibration history warrants it.

## Data

- **Prices** — OMIE public daily files (`marginalpdbc_YYYYMMDD.1`), downloaded and cached by
  `src/download.py`. No key required. Since the EU 15-minute market switch (Oct 2025) files
  carry 96 periods/day; quarter-hours are averaged to hours so the whole history is comparable
  (`src/dataset.py`).
- **Domestic forecasts (Spain)** — ESIOS (REE) day-ahead demand (1775), wind (1777) and solar
  PV (1779), peninsular, hourly, via `src/exog.py`. Needs a free personal token.
- **Cross-border forecasts (France)** — ENTSO-E day-ahead load (used) and wind/solar (fetched
  for the audit only, excluded from the model — see the cautionary tale), via
  `src/exog_entsoe.py`. Needs a free ENTSO-E token.
- Exogenous series are cached to `data/exog/*.csv` and **committed**, so the study reproduces
  without any token and no data is ever re-requested (providers' responsible-use terms).
- **History used:** ~420 delivery days (~10,000 hourly prices).

## Run it

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Windows (use bin/ on Linux/Mac)
.venv/Scripts/python -m src.pipeline --days 420 --test-days 120 --retrain-every 7
```

The cached ESIOS data is committed, so it runs out of the box. To **refresh** it you need a
free ESIOS token (<https://www.esios.ree.es/es/pagina/api>) in a git-ignored `.env` as
`ESIOS_TOKEN=...`. Flags: `--no-exog` runs the **price-only** model with no token;
`--with-entsoe` re-runs the France cross-border ablation (needs `ENTSOE_TOKEN` too).

Outputs (in `results/`): `metrics.csv`, `metrics_probabilistic.csv`, `metrics_by_hour.csv`,
`significance_dm.csv`, `backtest_predictions.csv`, and the three plots above.

## Project layout

```
src/
  download.py     # fetch + cache OMIE daily price files
  dataset.py      # parse (hourly + 15-min formats) into a tidy hourly series
  exog.py         # fetch + cache ESIOS day-ahead demand/wind/solar forecasts (Spain)
  exog_entsoe.py  # fetch + cache ENTSO-E day-ahead forecasts (France) — cross-border ablation
  forecast.py     # features (autoregressive + exogenous), GB + quantile + LASSO models,
                  # conformal calibration, walk-forward backtest, scoring, Diebold-Mariano
  pipeline.py     # end-to-end: download -> dataset + exog -> backtest -> metrics + plots
```

## Limitations & honest caveats

- **Forecast, not actuals.** The exogenous features are archived day-ahead forecasts (ESIOS
  "Previsión diaria D+1"). The forecast-vs-realised audit confirms they carry a genuine
  day-ahead error, so no outturn is leaking in.
- **Publication timing matters.** As the France case shows, a day-ahead forecast is only fair
  if it is published before the auction gate; the pipeline uses only such series.
- **~14 months only.** OMIE exposes a limited public history; a longer record would tighten
  every estimate. The DST period-to-hour mapping is approximate (a known ~2-day/year artifact,
  shared across the price and forecast series so relative alignment holds).
- **Point + interval, not a full density**, and no trading/P&L layer — the target is forecast
  accuracy, honestly measured.

## Roadmap

- **Leakage-safe cross-border signals** — e.g. day-ahead net transfer capacity / scheduled
  commercial exchanges, which *are* published before gate, in place of the excluded France
  renewable forecast.
- **Stronger literature benchmark** — the full LEAR / DNN from
  [`epftoolbox`](https://github.com/jeslago/epftoolbox) alongside the current LASSO.
- **Longer history / rolling-origin backtest** — to shrink the confidence intervals further.

## Author

Carlos Bustos — electrical engineering student (UCLM, Spain), focused on power systems
and energy markets. Built with an AI-assisted workflow.
