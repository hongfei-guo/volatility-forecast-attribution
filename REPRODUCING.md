# Reproducing the results

[Research overview](README.md) · [Archived replication materials](https://doi.org/10.5281/zenodo.22923099)

Information, Not Flexibility: Attributing Forecast Gains in Neural Stochastic Volatility

## Scope

This replication package accompanies the comparison of realised information, neural
functional form, and prequential forecast combination for the S&P 500, FTSE
100, and DAX. The evaluation sample runs from January 2018 through February
2022, and the forecast horizons are 1, 5, and 10 trading sessions.

The package supports the following uses:

- generating the NN-SV, RV-NN-SV, RV-RA-NN-SV, SV, and RV-LIN-SV forecast
  panels from a lawfully obtained Oxford-Man source file;
- generating the HAR-RV, SHAR-RV, log-HAR-RV, and Gaussian Realized GARCH
  benchmark panels;
- generating return-only GARCH-t and one-day Student-t Realized GARCH panels;
- regenerating equal and QLIKE prequential weights and their variance-mean
  combination panels;
- reproducing the QLIKE evaluation from forecast panels and that source file;
- regenerating and fitting the synthetic identification experiment; and
- reconstructing all manuscript tables and figures from retained components
  and lawful source data without rerunning MCMC.

The empirical calculation uses expanding-window Bayesian estimation on the
market-specific dates in `design/refit_schedule.csv`, daily latent-state
filtering, and path simulation at horizons 1, 5, and 10. RV-LIN-SV is
evaluated at horizon 1. The included forecast panels provide a compact
starting point for the evaluation, while
`code/scripts/generate_forecasts.py` reconstructs the five state-space model
panels from the documented daily data.

Third-party source observations are excluded. The archive contains calculated
evaluation records, whose use remains subject to the applicable source-data terms. Full posterior and predictive-path collections are kept separately; the
small coefficient posterior and five-origin components are included. The forecast generator can
write the five predictive-component arrays needed for distributional and
tail-risk evaluation to a user-selected local or external-storage directory.

## Data availability

The files in `forecasts/` contain predictions and model availability only.
They contain no realised returns, realised variances, daily losses, or source
observations. `data/README.md` identifies the Oxford-Man snapshot needed for
the public QLIKE calculation and documents all units and transformations. The
archived provider terms permit unrestricted research use with attribution to
Heber, Lunde, Shephard and Sheppard (2009), *Oxford-Man Institute's realized
library*, Version 0.3. The exact source download and terms are linked in
`data/README.md`.

The compact archive retains the functionals and evaluation records used for
log scores, PIT, CRPS, VaR, ES and FZ0 in the reported comparisons. It does not
retain the full predictive arrays. Researchers with the documented Oxford-Man
input can generate those arrays and the corresponding loss rows for NN-SV,
RV-NN-SV, SV, and RV-LIN-SV. Equal and log-score whole-path combinations can
then be generated conditional on their weight paths. The package also supplies
the GARCH-t, Student-t Realized GARCH, and RV-RA forecast-generation paths.
The RV-RA result reconstruction uses retained counterfactual variance means and
full coefficient posterior draws. The optional distribution-level replay uses
the separate conditional-component collection described below.

## Contents

- `code/`: statistical calculations, command-line programs, Stan models, and
  synthetic tests.
- `design/`: sample, model, evaluation, and estimand definitions.
- `environment/`: Python dependencies and runtime information.
- `data/`: data documentation and the identification calibration curve.
- `forecasts/`: prediction-only forecast panels and prequential weights.
- `inputs/`: evaluation records, simulation records and five-origin components.
- `results/`: aggregate tables and figures.
- `LICENSE.txt`: terms for the original materials in this package.

## Environment and tests

Use Python 3.10 (the verified runtime is Python 3.10.5) with the pinned
dependencies below. Create the environment using your Python 3.10 executable:

```bash
python3.10 -m venv .venv
.venv/bin/pip install -r environment/requirements.txt
```

Run the tests:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=code \
  .venv/bin/python -m pytest -q -p no:cacheprovider code/tests
```

The tests use only synthetic inputs and complete without restricted data.

The design files fix the random seeds, and `environment/` records the software
used for the reported calculations. Posterior draws need not be bitwise
identical across operating systems; reproduced summaries should agree within
Monte Carlo uncertainty.

## Reproduce all tables and figures without MCMC

The compact archive includes unrounded prediction functionals, daily evaluation
records, simulation records, and the small five-origin decomposition components.
It reconstructs all 21 tables and four figures without either large component
collection or new MCMC estimation. Lawfully obtained OMI and FirstRate source
files remain required in the layout documented in `data/README.md` and
`data/source_identity.json`.

After installing the requirements, run from the extracted archive:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=code .venv/bin/python code/scripts/reproduce.py \
  --data-root DATA_ROOT \
  --output REPRODUCED
```

Use a new output directory. The entry point constructs the source data, applies
the common coverage screens and fixed-outcome reconstruction, recalculates QLIKE,
MSE, interval coverage, VaR exceedances, quantile loss and FZ0, and recomputes the
DM, MCS, Holm, GR and bootstrap summaries. It writes all tables as LaTeX, CSV and
JSON under `tables/`, with `tables/index.csv` giving the manuscript labels, and
four PDF and PNG figures under `figures/`.

The retained daily negative log score, CRPS and PIT are evaluated at the paper's
original outcomes. Separate records cover the distribution scores that change
under reconstructed outcomes. They are float64 values, without rounding or draw
thinning, and belong to each model or combination's own predictive distribution.
The reader verifies that its complete outcome panel matches the stored score
identity before using these values. These records do not support rescoring the
complete predictive distribution at arbitrary new outcomes.

The RV-RA calculation uses the retained counterfactual variance means, complete
monthly coefficient posterior draws and observation inputs reconstructed from
lawful daily data. Every bootstrap replication recomputes the horizon normalizers
from the separately calculated model losses. The OC-input HAR control is recalculated
by ordinary least squares. Simulation tables aggregate all included per-replication
scientific records, retaining unsuccessful fits in the stated denominators. No
CmdStan or MCMC command is invoked by this route.

For separate execution, `--stage inputs`, `scores`, `summaries`, and `exhibits`
run those stages in order in the same output directory. Each stage requires the
preceding outputs. `--records` selects another directory with the same evaluation
record schema. Logs identify failures; do not combine different data snapshots
or outcome definitions.

For full distribution-level rescoring, retain the original scoring and RV-RA
component collections and supply both `--scoring-components` and
`--conditional-components`. The entry point then uses those arrays instead of
stored evaluation records. These large archives are not required for the default
reconstruction of the paper's reported results.

The figures use Arial. Different fonts or software builds may change pixel layout;
the scientific comparisons should agree at the reported precision. Font files
and third-party source observations are not distributed.

## Evaluation record inputs

`inputs/evaluation/` contains:

| File | Role |
|---|---|
| `prediction_records.parquet` | Forecast keys, sample flags, exact variance means, NLS/CRPS/PIT, interval endpoints, VaR and ES |
| `reconstructed_scores.parquet` | NLS/CRPS/PIT only for the forecast rows whose values change under reconstructed outcomes |
| `outcome_identity.json` | Identities of the original and reconstructed outcome panels used by the stored scores |
| `counterfactual_variances.parquet` | Three exact variance means at each of the 241 origins and two counterfactual horizons |
| `gamma_draws.npz` | The 8,000 coefficient posterior draws for each of twelve monthly refits |
| `*_forecasts.parquet` | The three prediction-only 2017 member panels used to reconstruct conditional-analysis inputs |

The outcome identity is calculated from ordered forecast keys and unrounded
variance/return outcomes. It prevents a different target series from silently
reusing scores calculated for another data definition. The file contains hashes,
not the outcomes themselves.

To regenerate these records from the complete calculation and retained conditional
components, use:

```sh
PYTHONPATH=code .venv/bin/python code/scripts/export_evaluation_records.py \
  --reproduced FULL_REPRODUCED \
  --conditional-components CONDITIONAL_COMPONENTS \
  --output EVALUATION_RECORDS
```

The exporter copies numerical values without changing precision and computes the
counterfactual means with the original parameter ancestry and innovations. Model
estimation, source-data construction and full distribution-scoring code remain
available below. The default compact workflow does not need those large arrays.

## State-space forecast generation

After constructing `reproduced/daily.parquet`, generate one market-model panel
with:

```bash
PYTHONPATH=code .venv/bin/python code/scripts/generate_forecasts.py \
  --data reproduced/daily.parquet \
  --market DAX \
  --model NN-SV \
  --output reproduced/DAX_NN-SV.parquet \
  --work-dir reproduced/sampling/DAX_NN-SV \
  --components-dir reproduced/components \
  --show-progress
```

Use `NN-SV`, `RV-NN-SV`, `RV-RA-NN-SV`, `SV`, or `RV-LIN-SV` with each of
`SP500`, `FTSE100`, and `DAX`. Each call writes one prediction-only Parquet panel. The
sampling directory contains CmdStan output and may be placed on external
storage. Running all fifteen market-model calculations requires substantial
computation. For a bounded check beginning at the first forecast origin, add
`--origin-end 2018-01-02`. Omit `--components-dir` when only variance-mean
panels are required. When supplied, each panel row records the relative name
of a compressed component file containing returns, variances, mixture
locations, conditional standard deviations, and Student-t degrees of freedom.

The program follows the same order at each origin: it first generates the
forecast and then incorporates the next observed return into the state
filter. Future realised variance in multi-step RV-NN-SV forecasts is integrated
with a Bayesian log-HAR model. The fixed seeds identify the calculation;
floating-point and MCMC differences across systems can lead to Monte Carlo
variation.

## External benchmark generation

Generate the four external benchmark panels for one market with:

```bash
PYTHONPATH=code .venv/bin/python code/scripts/generate_benchmarks.py \
  --data reproduced/daily.parquet \
  --market DAX \
  --components-dir reproduced/components \
  --output reproduced/DAX_benchmarks.parquet
```

The command estimates HAR-RV, SHAR-RV, log-HAR-RV, and Gaussian Realized
GARCH on expanding windows and the dates in `design/refit_schedule.csv`. It
writes horizons 1, 5, and 10 in the same prediction-only panel format used in
`forecasts/`. The component option applies to Gaussian Realized GARCH; its
archives contain the four Gaussian arrays required for scoring. Recursive
HAR-RV and SHAR-RV level forecasts are floored at zero.

## Student-t GARCH generation

Generate the return-only GARCH-t panel with:

```bash
PYTHONPATH=code .venv/bin/python code/scripts/generate_garch_models.py \
  --data reproduced/daily.parquet \
  --model GARCH-t \
  --market DAX \
  --components-output reproduced/components \
  --output reproduced/DAX_GARCH-t.parquet
```

Replace the model with `Realized-GARCH-t-MLE` for the one-day Student-t
Realized GARCH panel. Both models use the dates in `design/refit_schedule.csv`.
If an estimation fails, forecasts remain unavailable until the next listed
estimation date; no previous estimate is carried forward. After constructing
their loss panels, `compare_garch_t.py` and `compare_realized_garch_t.py`
rebuild the packaged pairwise comparison tables.

## QLIKE evaluation

Place the Oxford-Man file described in `data/README.md` at
`OXFORD_MAN.csv`, then run:

```bash
mkdir -p reproduced

PYTHONPATH=code .venv/bin/python code/scripts/build_daily_data.py \
  --source OXFORD_MAN.csv \
  --output reproduced/daily.parquet \
  --report reproduced/daily_report.json

PYTHONPATH=code .venv/bin/python code/scripts/audit_calendars.py \
  --data reproduced/daily.parquet \
  --output reproduced/calendar_report.json

PYTHONPATH=code .venv/bin/python code/scripts/build_variance_losses.py \
  --data reproduced/daily.parquet \
  --forecasts forecasts/SP500.parquet forecasts/FTSE100.parquet forecasts/DAX.parquet \
  --calendar-report reproduced/calendar_report.json \
  --output reproduced/variance_losses.parquet

PYTHONPATH=code .venv/bin/python code/scripts/summarize_evaluation.py \
  --losses reproduced/variance_losses.parquet \
  --config design/analysis.yaml \
  --scope variance \
  --output reproduced/evaluation
```

The commands write:

| Command | Output |
|---|---|
| `build_daily_data.py` | `daily.parquet` and `daily_report.json` |
| `audit_calendars.py` | `calendar_report.json` |
| `build_variance_losses.py` | `variance_losses.parquet` |
| `summarize_evaluation.py --scope variance` | `mcs_sets.csv`, `mcs_common_sample_mean_qlike.csv`, and `dm_qlike.csv` |

`dm_qlike.csv` is the 39-row QLIKE subset of the 48-row packaged
`results/evaluation/dm_results.csv`. The calendar calculation records missing
exchange sessions and does not impute observations.

## Forecast combination

The package includes a prediction-only 2017 warm-up panel. It contains model
forecasts, dates, availability, and exchange-session indices, but no realised
outcomes or evaluated log densities. With the daily data and variance losses
constructed above, regenerate the equal and QLIKE weight paths and their
variance-mean panels with:

```bash
PYTHONPATH=code .venv/bin/python code/scripts/combine_forecasts.py \
  --data reproduced/daily.parquet \
  --forecasts forecasts/SP500.parquet forecasts/FTSE100.parquet forecasts/DAX.parquet \
  --losses reproduced/variance_losses.parquet \
  --warmup-forecasts forecasts/combination_warmup.parquet \
  --calendar-report reproduced/calendar_report.json \
  --modes equal qlike \
  --output reproduced/combination_forecasts.parquet \
  --weights-output reproduced/combination_weights.csv
```

The 2017 realised-variance targets are reconstructed from the lawfully obtained
daily data. The weight calculation uses only targets that have matured by each
monthly update and discounts them by exchange-session age with a 252-session
half-life.

The package also documents the log-score optimization. Exact regeneration of
its weights requires the 2017 candidate log densities, which can be recalculated
from the warm-up entries in the separate scoring collection and lawful daily data. The complete packaged monthly weight path can instead be
passed with `--packaged-weights forecasts/combination_weights.csv` to rebuild
the three modes' variance-mean panels conditional on the reported weights.

With candidate component files, add `--predictive-root` and
`--components-output` to the same command. Equal and log-score combinations
then select one candidate and one source draw for each complete forecast path;
the resulting component files can be evaluated with the loss command below.

## Distributional and tail-loss construction

Build density and tail losses directly from generated Parquet panels and their
component directory:

```bash
PYTHONPATH=code .venv/bin/python code/scripts/build_predictive_losses.py \
  --data reproduced/daily.parquet \
  --forecasts reproduced/DAX_NN-SV.parquet \
  --forecast-root reproduced/components \
  --calendar-audit reproduced/calendar_report.json \
  --output reproduced/DAX_NN-SV_losses.parquet
```

The loss builder uses the producer's variance mean, checks it against the
saved paths when it is a path mean, and computes CRPS, log score, PIT,
prediction intervals, and one-day analytic VaR, ES, and FZ0. Component files
are generated locally and are not part of this compact archive.

After generating the probabilistic, combination and variance-only loss panels
for the evaluation period, rebuild the distributional tables with:

```bash
PYTHONPATH=code .venv/bin/python code/scripts/summarize_evaluation.py \
  --losses reproduced/losses/*.parquet \
  --config design/analysis.yaml \
  --scope full \
  --output reproduced/evaluation
```

The loss panels must cover the models listed in `design/analysis.yaml`, without
duplicate forecast rows. Exclude the 2017 combination warm-up observations.

The tail calculation uses one panel per market, restricted to horizon 1 and the
models in `design/tail_evaluation.yaml`. Prepare those panels and summarise them:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import pandas as pd
import yaml

losses = pd.concat([pd.read_parquet(p) for p in Path('reproduced/losses').glob('*.parquet')])
models = yaml.safe_load(Path('design/tail_evaluation.yaml').read_text())['probabilistic_models']
tail = losses[losses.horizon.eq(1) & losses.model_id.isin(models)]
output = Path('reproduced/tail_losses')
output.mkdir(parents=True, exist_ok=True)
for market, panel in tail.groupby('market'):
    panel.to_parquet(output / f'{market}.parquet', index=False)
PY

PYTHONPATH=code .venv/bin/python code/scripts/summarize_tail.py \
  --losses reproduced/tail_losses/SP500.parquet reproduced/tail_losses/FTSE100.parquet reproduced/tail_losses/DAX.parquet \
  --config design/tail_evaluation.yaml \
  --output reproduced/tail
```

## Optional: rescoring from full predictive components

The default table-and-figure reconstruction uses `inputs/evaluation/` and the
documented source data. It does not require the separate scoring or RV-RA
component collections. The commands in this section recalculate distribution
scores from retained predictive arrays.

`bnsv/scoring_components.py` reads both producer path files and compact scoring
files. The compact representation retains float64 cumulative-return and
integrated-variance draws; one-day files also retain mixture locations,
conditional standard deviations and Student-t degrees of freedom where used.
Multiday PIT randomization is retained when reproducing a stored finite-ensemble
rank. No training matrix or observed-history tail is needed for scoring.

Combination files may contain only source-file references and the source-model
and source-draw selection indices. GARCH-t files may contain the original
omega/alpha/beta/nu parameters, forecast variance, return transformation, path
count and random seed. The reader reconstructs the same scoring vectors from
those representations. It does not estimate a model.

To build a compact collection from retained records:

```sh
PYTHONPATH=code python code/scripts/export_scoring_components.py \
  --records SOURCE_RECORDS.parquet --output COMPONENTS \
  --checks LOCAL_CHECKS --workers 2
```

The source table identifies forecasts and their original component paths/hashes,
representation (`draws`, `selection`, or `garch_parameters`), and the reference
outcomes/scores used for verification. It is a private preparation input, not
a file to redistribute. The exporter writes only prediction metadata to
`COMPONENTS/forecasts.parquet`; original outcomes and reference losses are not
copied. Source paths, check reports and training histories are excluded from
that collection. Select its evaluation records and add the variance-only
forecasts before constructing losses:

```sh
PYTHONPATH=code .venv/bin/python code/scripts/prepare_evaluation_forecasts.py \
  --component-index COMPONENTS/forecasts.parquet \
  --point-panels forecasts/SP500.parquet forecasts/FTSE100.parquet forecasts/DAX.parquet \
  --config design/analysis.yaml \
  --output reproduced/evaluation_forecasts.parquet

PYTHONPATH=code .venv/bin/python code/scripts/build_predictive_losses.py \
  --data reproduced/daily.parquet \
  --forecasts reproduced/evaluation_forecasts.parquet \
  --forecast-root COMPONENTS \
  --calendar-audit reproduced/calendar_report.json \
  --output reproduced/losses/evaluation.parquet
```

This selection excludes warm-up records and uses the evaluation dates in the
analysis design. The resulting loss panel can be used in the table commands
above; do not also supply overlapping model-level loss panels.
This optional route uses the separate `scoring_components.zip` collection and
lawful daily data. The default reconstruction of the reported tables and figures
uses the included evaluation records.

## Optional: RV-RA forecast generation and full path replay

The default RV-RA table reconstruction uses the counterfactual means and full
coefficient posterior draws in `inputs/evaluation/`, with observation inputs
constructed from lawful daily data. The separate conditional-component collection
is needed only to replay the full counterfactual paths described below.

`RV-RA-NN-SV` uses the same forecast command as the other state-space models.
Its three lag-one inputs are the standardized return, log realised variance,
and bounded semivariance imbalance. The model uses the common estimation dates
and withholds forecasts after an unsuccessful fit or inadequate filter until
the next listed date.

The observable 2017 inputs for the descriptive conditional-value analysis can
be reconstructed with `code/scripts/prepare_rv_ra_conditional_inputs.py`.
`design/rv_ra_conditional_value.md` gives the forecast and preparation
commands. Full path replay uses a separate conditional-component collection: twelve parameter sets, the full coefficient
posterior draws, 482 stochastic decompositions and prediction-only panels for
the three models. No training matrices, historical observations or evaluation
outcomes are included in that collection.

`export_rv_ra_components.py` extracts these objects from retained monthly
archives. With the components and lawful daily data,
`analyze_rv_ra_conditional_value.py --daily` reconstructs the origin histories
and training-sample feature transformations, including each month's RV floor.
The counterfactuals retain the original parameter ancestry and random
innovations. The design document gives the complete commands for this optional replay route.
The default compact-record calculation does not read these separate components.

## Synthetic identification experiment

To recompute the scalar and correction-function recovery summaries from the
included per-replication records, without fitting models:

```sh
PYTHONPATH=code python code/scripts/summarize_identification_simulation.py \
  --records inputs/identification --output reproduced/identification
```

The input files contain scalar and functional recovery statistics, DGP truths,
diagnostics and all 400 fit outcomes, including two unsuccessful fits. This
command recomputes aggregate errors, coverage, planned denominators and declaration
counts. It does not recompute each replication's posterior intervals or function
recovery from the underlying posterior draws. The posterior-level route below
requires the synthetic preparation and retained compact fits, or new fitting.


The full `reproduce.py` workflow includes the correction-index summaries and
their rows in the simulation table. They can also be reconstructed separately
from the retained posterior quantiles and probabilities:

```sh
python code/scripts/summarize_correction_index.py \
  --output reproduced/identification/correction_index_summary.csv
```

`inputs/identification/correction_index.csv` retains all 400 replication keys.
Posterior summaries use successful fits; the DGP-N interval coverage uses all
200 replications, with failed fits counted as noncoverage. The summary also
reports the direction of interval misses, the number of upper endpoints below
the threshold, and the maximum posterior threshold probability. The threshold uses
the unrounded value in `design/identification.yaml`. The script can also export
these records directly from saved index draws with `--fits` and `--tasks`.


Generate the 400 synthetic datasets and validate all fit inputs:

```bash
PYTHONPATH=code .venv/bin/python \
  code/scripts/prepare_identification_simulation.py \
  --output identification_preparation

PYTHONPATH=code .venv/bin/python \
  code/scripts/run_identification_simulation.py \
  --preparation identification_preparation \
  --output identification_fits \
  --dry-run-all
```

The first command writes the simulated datasets and fit plan under
`identification_preparation/`. The second validates all 400 tasks and reports
the task count without fitting models or writing fit results.

`design/identification_seeds.csv` records the random streams used to generate
the datasets, estimate the 400 models, and construct the predictive draws.

Fitting the experiment requires CmdStan and substantial computation. A single
task is run by replacing `--dry-run-all` with `--task-id 1`; its files are
written to `identification_fits/task_001/`. After running the 400 tasks, use:

```bash
PYTHONPATH=code .venv/bin/python \
  code/scripts/summarize_identification_simulation.py \
  --preparation identification_preparation \
  --fits identification_fits \
  --output reproduced/identification
```

The summary command writes the eight files in `results/identification/`.
`simulation_summary.csv` gives the main recovery quantities. The
replication-level tables permit independent summaries and retain all 400
planned replications: 398 fits meet the stated diagnostics and two DGP-N fits
do not. Coverage uses the planned denominator and counts these two fits as
uncovered; declaration rates count them as non-declarations. Continuous
recovery measures condition on the 398 successful fits.

## Multiscale models

`RV-NN-SV-HAR` and `RV-LIN-SV-HAR` are the code identifiers for the paper's
RV-NN-SV-MS and RV-LIN-SV-MS. Both use return feedback and daily, weekly and
monthly open-to-close realised history. The code fixes the grid-agreement
filter for this pair; the primary models retain their stated filter.

```sh
PYTHONPATH=code python code/scripts/generate_har_input_forecasts.py \
  --data reproduced/daily.parquet --market SP500 --output-dir reproduced/ms/SP500
```

Repeat for FTSE100 and DAX. Generation requires CmdStan and substantial compute.
Both members use four chains, 1,000 warm-up iterations and 2,000 retained draws
per chain, target acceptance 0.97 and tree depth 12. Divergence-only failures
allow a 0.99 attempt; ESS-only failures allow 4,000 retained draws. The agreement
tolerance is 0.001, the ESS fraction is 0.20, and grid refinement ends at level 6.
The FTSE linear fit of 1 February 2022 used a 4,000-draw extension after an
R-hat-only failure. Regenerating this fit requires its retained posterior cache; the automatic
retry rule does not give every R-hat failure an extension. The no-MCMC route
uses the stored MS forecasts and scoring components directly. Prediction-only MS panels are included in `forecasts/multiscale.parquet`.

## Forecast comparisons

After building the daily data and calendar report, evaluate the included panels:

```sh
PYTHONPATH=code python code/scripts/build_variance_losses.py \
  --data reproduced/daily.parquet \
  --forecasts forecasts/SP500.parquet forecasts/FTSE100.parquet \
    forecasts/DAX.parquet forecasts/multiscale.parquet \
  --calendar-report reproduced/calendar_report.json \
  --output reproduced/variance_losses.parquet
PYTHONPATH=code python code/scripts/evaluate_comparisons.py \
  --original reproduced/variance_losses.parquet \
  --output reproduced/comparisons --mcs --fluctuation
```

The primary nine-model and expanded eleven-model confidence sets are computed
separately. MCS uses 2,000 stationary-bootstrap draws, mean block length 10 and
seed 20260730. MS form contrasts use a three-test Holm family; daily–MS contrasts
use a six-test family. The three origin-based subperiods do not restart fitting
or combination histories. Top-five adverse origins are selected separately at
each horizon. Loss columns not present in the input cannot be evaluated: the
variance-only command does not produce density-score tables.

The GR statistic uses windows of floor(0.30 P), Bartlett/Newey-West long-run
variance and six lags in these samples. Reference bands use the 95th percentile
of each Gaussian-increment path's maximum absolute rolling statistic, 50,000
paths and seed 20260913. They are indicative for the expanding-window Bayesian
design. The null is zero expected loss difference at every origin, not constancy
at an unspecified nonzero mean.

## Outcome reconstruction

The original Oxford-Man outcomes define the main evaluation. The sensitivity
uses the same predictions, weights and eligible samples with reconstructed
variance and return outcomes. Obtain the licensed files in this layout:

```text
DATA_ROOT/raw/external/omi/oxfordmanrealizedvolatilityindices.csv
DATA_ROOT/raw/hf/firstrate/SPX_full_5min.txt
DATA_ROOT/raw/hf/firstrate/UKX_full_5min.txt
DATA_ROOT/raw/hf/firstrate/DAX_full_5min.txt
```

```sh
PYTHONPATH=code python code/scripts/audit_source_coverage.py \
  --data-root DATA_ROOT --output reproduced/coverage
PYTHONPATH=code python code/scripts/reconstruct_evaluation_outcomes.py \
  --daily reproduced/daily.parquet \
  --audit reproduced/coverage/source_hf_comparison.csv \
  --output reproduced/reconstruction
```

FirstRate records are timestamp/open/high/low/close. Bar-start timestamps use
New York local time and are converted to exchange time. Only continuous-session
bars enter the complete five-minute grid; auction bars are excluded and scheduled
half-days respected. The rule flags OMI span below 95% of scheduled duration or
observations per scheduled hour below 50% of the market/year median: 21 FTSE and
one DAX sessions. Source counts measure events, not five-minute bars. Changes to
closing prices propagate to adjacent returns, giving 40 FTSE and two DAX changed
daily outcomes. Complete-day ratios use both screens and complete independent grids.

For exact density and tail rescoring, supply the original score panel and
predictive components; a variance mean alone cannot recover a distribution:

```sh
PYTHONPATH=code python code/scripts/rescore_reconstructed_outcomes.py \
  --daily reproduced/daily.parquet \
  --reconstructed-daily reproduced/reconstruction/daily_reconstructed.parquet \
  --scores SCORES.parquet --components COMPONENTS --output reproduced/rescoring
PYTHONPATH=code python code/scripts/evaluate_comparisons.py \
  --original reproduced/rescoring/original_scores.parquet \
  --reconstructed reproduced/rescoring/reconstructed_scores.parquet \
  --output reproduced/comparisons --mcs --fluctuation
```

Score records are keyed by market, model_id, origin_date, mature_date and horizon.
They include target_dates_json, variance_forecast, eligibility and the evaluated
score fields. A probabilistic record whose return target changes additionally
requires return_distribution and predictive_file, relative to COMPONENTS;
predictive_sha256, when supplied, is checked. Component NPZ files contain
raw_returns, raw_variances, raw_return_locations and raw_conditional_sds, plus
degrees_of_freedom for Student-t mixtures. Missing components cause an explicit
error; the program does not estimate replacement forecasts.

## Baseline decomposition

```sh
PYTHONPATH=code python code/scripts/decompose_five_forecasts.py \
  --records inputs/five_forecasts/records.csv --components inputs/five_forecasts \
  --output reproduced/decomposition
```

The 15 records identify origin_date, model_id, variance_forecast, component_file
and component_sha256. Each component NPZ contains the aligned baseline_state,
correction and variance vectors, return_scale, and its model/origin identifiers.
It contains no training features, observed return/RV history or price targets.
The calculation verifies variance = return_scale² × exp(baseline_state + correction),
then computes B = mean(return_scale² × exp(baseline_state)) and A = mean(variance)/B.
It reconstructs the table from predictive components, not from table means. The
components and their records are included in `inputs/five_forecasts/`. This is a decomposition of the fitted forecast,
not a separately estimated no-correction model or a causal ablation.

## Figures and result map

```sh
PYTHONPATH=code python code/scripts/render_combination_weights.py
PYTHONPATH=code python code/scripts/render_information_cumulative.py
PYTHONPATH=code python code/scripts/render_sp500_information_diagnostics.py \
  --scores SCORES.parquet
PYTHONPATH=code python code/scripts/render_information_fluctuation.py
```

The first command uses included weights. The others use the original score panel
and outputs under `reproduced/comparisons/`. All write separate PDF and PNG files
to `reproduced/figures/`; `--output` selects a different directory. The cumulative
and GR commands also accept `--series`, and GR accepts `--summary`. Captions remain
in the manuscript. `results/figures/` contains reference vector figures.

| Paper result | Calculation | Input required beyond the ZIP |
|---|---|---|
| Primary QLIKE tables and MCS | build_variance_losses.py; summarize_evaluation.py | Lawful OMI data |
| MS and three-period QLIKE tables, expanded MCS | evaluate_comparisons.py | Lawful OMI data and included forecasts |
| Density, calibration and tail tables | build_record_losses.py; summarize_tail.py | Lawful OMI/FirstRate source data; evaluation records are included |
| Coverage and reconstructed-outcome table | audit_source_coverage.py; reconstruct_evaluation_outcomes.py; build_record_losses.py | Lawful OMI/FirstRate data |
| Five-origin decomposition | decompose_five_forecasts.py | None; the fifteen components and records are included |
| Synthetic identification table | summarize_identification_simulation.py --records | None for aggregation of included scientific records; compact posteriors and synthetic data for posterior-level recalculation |
| RV-RA conditional analysis | analyze_conditional_records.py | Lawful OMI data; means and coefficient draws are included |
| Four figures | render_* commands above | Included weights; lawful OMI for score-based figures |

Results under `results/comparisons/` contain the original and reconstructed
outcome definitions used in the paper. The repository omits source observations and large prediction-path collections; its
evaluation records retain the daily statistics needed for the reported results.
The corresponding author can help identify the required source and components
subject to access and licensing conditions. Full distribution-level rescoring
requires the predictive components described above. Software versions
and seeds are retained where needed to reproduce the calculations.

## License

The original materials in this package are licensed under CC BY 4.0. The
license does not cover third-party source data, which are not included.
