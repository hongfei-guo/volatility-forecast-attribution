# Computation code

`bnsv/` contains the statistical calculations, `models/` contains the Stan
models, `scripts/` contains the command-line programs, and `tests/` contains
synthetic tests.

`models/nn_sv.stan` is the shared specification for NN-SV and RV-NN-SV. It
uses the signed-return predictor when `D=1` and adds realised variance when
`D=2`. `models/rv_ra_nn_sv.stan` adds a centred residual-asymmetry term.
`models/sv.stan` is the uncorrected stochastic-volatility model, and
`models/rv_lin_sv.stan` replaces the neural correction with a centred linear
function.

The empirical forecast path is intentionally compact:

1. `forecast_features.py` constructs expanding-window return and lagged-RV
   features without using future observations.
2. `forecast_fit.py` supplies the Stan inputs, deterministic initial values,
   sampling sequence, and diagnostic criteria.
3. `forecast_filter.py` integrates the latent baseline state over fixed
   posterior parameter atoms.
4. `forecast_paths.py` draws future realised variance and stochastic-volatility
   paths; `forecast_calculation.py` contains their deterministic calculation.
5. `forecast_generation.py` applies the analysis dates in
   `design/refit_schedule.csv` and writes the forecast panel.

## Public calculations

`reproduce.py` defaults to the compact evaluation records. `build_record_losses.py`
recalculates outcome-based losses; `analyze_conditional_records.py` reuses the same
RV-RA normalization, regression and bootstrap as the full-component route.


| Program | Required input | Output |
|---|---|---|
| `build_daily_data.py` | Oxford-Man CSV described in `data/README.md` | daily Parquet file and JSON data report |
| `generate_forecasts.py` | daily data and `design/forecast_generation.yaml` | one NN-SV, RV-NN-SV, RV-RA-NN-SV, SV, or RV-LIN-SV panel, with optional predictive components |
| `generate_benchmarks.py` | daily data and `design/external_benchmarks.yaml` | HAR-RV, SHAR-RV, log-HAR-RV, and Gaussian Realized GARCH panels, with optional Gaussian components |
| `generate_garch_models.py` | daily data and the corresponding GARCH design | GARCH-t or one-day Student-t Realized GARCH panel, with optional components |
| `audit_calendars.py` | daily Parquet file | JSON calendar report |
| `build_variance_losses.py` | daily data, calendar report, and packaged forecasts | variance-loss Parquet file |
| `combine_forecasts.py` | candidate panels, daily data, losses or packaged weights, and `design/combination.yaml` | prequential weights and variance-mean combination panels |
| `prepare_evaluation_forecasts.py` | compact component index, variance-only panels, and `design/analysis.yaml` | evaluation forecast panel excluding combination warm-up observations |
| `build_predictive_losses.py` | lawful daily data, generated component-aware panels, and their component directory | variance, density, interval, and one-day analytic tail loss rows |
| `prepare_rv_ra_conditional_inputs.py` | lawful daily data, three 2017 forecast panels, and the 2017 estimation schedule | observable losses and state covariates for the RV-RA conditional analysis |
| `export_rv_ra_components.py` | retained monthly RV-RA archives and member records | prediction-only panels, full coefficient posterior draws and minimal counterfactual components |
| `summarize_evaluation.py --scope variance` | variance-loss file and `design/analysis.yaml` | `mcs_sets.csv`, `mcs_common_sample_mean_qlike.csv`, and `dm_qlike.csv` |
| `prepare_identification_simulation.py` | packaged identification design, seed table, and calibration curve | 400 synthetic datasets and their fit plan |
| `run_identification_simulation.py --dry-run-all` | prepared synthetic experiment | validation result printed to the terminal; no fit files |
| `run_identification_simulation.py --task-id N` | one prepared task and CmdStan | one task directory under the requested output path |
| `summarize_identification_simulation.py` | included per-replication records, or prepared synthetic data and compact fits | simulation summaries and diagnostics, including `simulation_summary.csv` |

The package README gives the executable forecast-generation, QLIKE, and
identification commands.

## Optional: full predictive-component calculations

The default `reproduce.py` workflow reconstructs all reported tables and figures
from `inputs/evaluation/` and lawful source data. The following commands instead
recalculate distribution scores or counterfactual paths from full components;
those separate collections are not required by the default workflow.

`build_predictive_losses.py` computes distribution scores from predictive arrays
and the documented daily data. The standard forecast, GARCH, comparison,
calibration and tail calculations can consume either route's compatible loss panels.

`results/garch_t/comparisons.csv` concatenates the three market-specific
outputs from `compare_garch_t.py`. The accompanying
`distribution_calibration.csv` applies the calibration calculation in
`summarize_evaluation.py` to the GARCH-t loss rows and retains the GARCH-t
results.

`analyze_rv_ra_conditional_value.py` requires four restricted tables and the
twelve monthly RV-RA forecast directories:

- paired RV-NN-SV and RV-RA-NN-SV QLIKE losses;
- NN-SV losses;
- RV-NN-SV losses;
- the monthly posterior summary for the asymmetry coefficient; and
- monthly directories containing `forecast_records.jsonl`, one compact refit,
  and the predictive arrays for horizons 1, 5, and 10.

With those inputs, the command is:

```bash
PYTHONPATH=code .venv/bin/python \
  code/scripts/analyze_rv_ra_conditional_value.py \
  --config design/rv_ra_conditional_value.json \
  --paired-losses RESTRICTED/paired_losses.csv \
  --nn-losses RESTRICTED/nn_sv_losses.csv \
  --rv-nn-losses RESTRICTED/rv_nn_sv_losses.csv \
  --gamma-summary RESTRICTED/gamma_a_summary.csv \
  --rv-ra-root RESTRICTED/rv_ra \
  --output-root reproduced/rv_ra_conditional_value
```

It writes `estimates.csv`, `monthly_panel.csv`, `figure.png`, and
`summary.json`. The public archive includes those four aggregate outputs, but
not the refits or stochastic decompositions used for full path replay. The default
conditional analysis uses the included counterfactual means and coefficient
posterior draws. For optional replay with the separate conditional components,
add `--daily` to reconstruct the observed histories from lawful daily data.
The [conditional-analysis design](../design/rv_ra_conditional_value.md) gives
the extraction, observable-input preparation and analysis commands.

Multiscale, outcome-reconstruction and plotting entrypoints are documented in
[the package README](../README.md), including each required input and output.
