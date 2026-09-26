# RV-RA conditional-value analysis

## Research question and role

This analysis asks whether the residual-asymmetry component has detectable
forecast value in particular volatility states. It is reported as descriptive
2017 evidence. The main model comparison and prequential combination set are
defined in `analysis.yaml`.

## Sample and outcome

The sample contains the 241 S&P 500 forecast origins from 3 January through
14 December 2017 that are common to horizons 1, 5, and 10. It contains 723
model-horizon observations. The last ten-day target matures on 29 December
2017, before the January 2018 through February 2022 evaluation sample begins.

For origin \(t\), define

\[
D_t=\frac{1}{3}\sum_{h\in\{1,5,10\}}
\frac{L_{RA,t,h}-L_{RV,t,h}}{\bar L_{RV,h}},
\]

where \(L_{RA,t,h}\) and \(L_{RV,t,h}\) are QLIKE losses and
\(\bar L_{RV,h}\) is the mean `RV-NN-SV` loss for horizon \(h\) on the same
241 origins. Negative values favor `RV-RA-NN-SV`. Giving each horizon equal
weight prevents the number of available model-horizon observations from
changing the estimand.

## Conditional estimates

The continuous-state regression is

\[
D_t=\delta_{r(t)}+\beta_A q_t^c+\beta_V v_t
+\beta_{AV}q_t^c v_t+u_t.
\]

Here \(q_t^c\) is the standardized asymmetry input, centered using the
estimation sample available at the most recent monthly re-estimation date;
\(v_t\) is standardized lagged log realized variance; and
\(\delta_{r(t)}\) denotes monthly re-estimation-date fixed effects. The slopes
therefore use variation within each monthly re-estimation period. Two additional
regressions compare high asymmetry and high log realized variance with their
complements. A state is high when its origin value exceeds the corresponding
estimation sample's 75th percentile. If an indicator does not vary within a
re-estimation period, the estimate is reported as unavailable with its support
counts; the sample and threshold remain unchanged.

Uncertainty uses 2,000 circular moving-block bootstrap replications over
ordered origins, block length 10, and seed 20260730. Each replication
recomputes the horizon normalizers and all reported estimates. Intervals are
the 2.5th and 97.5th percentiles.

## Counterfactual comparisons

At horizons 5 and 10, the calculation uses the saved parameter ancestry,
origin state, future log-realized-variance paths, state innovations, return
innovations, and return transformation under three specifications:

1. `predicted_asymmetry` uses the model's predictive future-asymmetry paths.
2. `realized_asymmetry` replaces only future asymmetry with its observed path.
   With lag-one timing, observed asymmetry at step \(j-1\) first affects step
   \(j\).
3. `no_asymmetry_term` sets \(\gamma_A=0\) in the same parameter draws.

The three contrasts compare predicted with observed asymmetry, predicted
asymmetry with no asymmetry term, and observed asymmetry with no asymmetry
term. Each contrast first averages paired QLIKE differences within horizon and
then gives horizons 5 and 10 equal weight. The observed-asymmetry calculation
measures the value of the future input, while the zero-coefficient calculation
isolates the fitted asymmetry term. Neither is a separately estimated forecast
model.

## Equal-weight comparison and outputs

The equal-weight comparison reports the QLIKE difference between the
three-member mean (`NN-SV`, `RV-NN-SV`, and `RV-RA-NN-SV`) and the two-member
mean (`NN-SV` and `RV-NN-SV`). The weights are fixed, and the contrast gives
horizons 1, 5, and 10 equal weight.

The public package contains the estimates table, monthly panel, two-panel
figure, and summary. The table reports no significance stars or model-selection
decision. Origin-level losses and predictive arrays derive from restricted
data and are not distributed.

## Default reconstruction from compact records

The default `reproduce.py` command uses the member forecasts, counterfactual
variance means and coefficient posterior draws in `inputs/evaluation/`. It
constructs observable inputs from lawful daily data and calls
`analyze_conditional_records.py`, which shares the normalization, regression and
bootstrap implementation with the full-component calculation. Neither of the
large component collections is required for this reconstruction.

## Optional: forecast generation and full path replay

Generate each of the three 2017 forecast panels from lawfully obtained daily
data by repeating the following command with these model and output-stem pairs:

| `MODEL` | `STEM` |
|---|---|
| `NN-SV` | `nn_sv_2017` |
| `RV-NN-SV` | `rv_nn_sv_2017` |
| `RV-RA-NN-SV` | `rv_ra_nn_sv_2017` |

```text
PYTHONPATH=code .venv/bin/python code/scripts/generate_forecasts.py \
  --data reproduced/daily.parquet \
  --market SP500 \
  --model MODEL \
  --origin-start 2017-01-03 \
  --origin-end 2017-12-14 \
  --refit-schedule design/rv_ra_refit_schedule_2017.csv \
  --output work/STEM.parquet \
  --work-dir work/sampling/STEM
```

Then prepare the observable conditional-analysis inputs with:

```text
PYTHONPATH=code .venv/bin/python code/scripts/prepare_rv_ra_conditional_inputs.py \
  --data reproduced/daily.parquet \
  --nn-forecasts work/nn_sv_2017.parquet \
  --rv-nn-forecasts work/rv_nn_sv_2017.parquet \
  --rv-ra-forecasts work/rv_ra_nn_sv_2017.parquet \
  --refit-schedule design/rv_ra_refit_schedule_2017.csv \
  --output-dir work/rv_ra_conditional_inputs
```

This produces the paired and member QLIKE inputs and the origin-level state
covariates. Replaying the complete counterfactual paths additionally requires
the separate conditional-component collection; ordinary scoring components
written by `generate_forecasts.py --components-dir` do not contain the necessary
parameter ancestry and innovations.

To extract the conditional components from retained monthly archives:

```sh
PYTHONPATH=code .venv/bin/python code/scripts/export_rv_ra_components.py \
  --source RETAINED_INPUTS --output CONDITIONAL_COMPONENTS --checks LOCAL_CHECKS
```

The source directory contains monthly `rv_ra/` directories and `tables/` with
paired losses, member losses and coefficient summaries. The exporter retains
twelve parameter sets, 8,000 coefficient posterior draws per month, and 482
five- and ten-day stochastic decompositions. The full coefficient posterior
is distinct from the parameter subset used for prediction. Source verification
records are written separately from the components.

With this collection and lawful daily data, use its three prediction panels
to prepare the observable inputs and then reconstruct all four reported outputs:

```sh
PYTHONPATH=code .venv/bin/python code/scripts/prepare_rv_ra_conditional_inputs.py \
  --data reproduced/daily.parquet \
  --nn-forecasts CONDITIONAL_COMPONENTS/nn_sv_forecasts.parquet \
  --rv-nn-forecasts CONDITIONAL_COMPONENTS/rv_nn_sv_forecasts.parquet \
  --rv-ra-forecasts CONDITIONAL_COMPONENTS/rv_ra_forecasts.parquet \
  --refit-schedule design/rv_ra_refit_schedule_2017.csv \
  --output-dir reproduced/rv_ra_inputs

PYTHONPATH=code .venv/bin/python code/scripts/analyze_rv_ra_conditional_value.py \
  --config design/rv_ra_conditional_value.json \
  --paired-losses reproduced/rv_ra_inputs/paired_losses.csv \
  --nn-losses reproduced/rv_ra_inputs/nn_sv_losses.csv \
  --rv-nn-losses reproduced/rv_ra_inputs/rv_nn_sv_losses.csv \
  --gamma-summary CONDITIONAL_COMPONENTS/gamma_summary.csv \
  --rv-ra-root CONDITIONAL_COMPONENTS/rv_ra \
  --daily reproduced/daily.parquet \
  --output-root reproduced/rv_ra_conditional_value
```

The daily input reconstructs origin histories and monthly feature transformations,
including the training-sample RV floor. Parameter ancestry and random innovations
are retained. The coefficient summaries use the mean, sample standard deviation
and 2.5%, 50%, 97.5% quantiles of the full monthly posterior draws. This route
performs no model estimation. The separate conditional-component collection
is required only for this optional path replay; the default reported-result
reconstruction uses the compact records included in the code archive.
