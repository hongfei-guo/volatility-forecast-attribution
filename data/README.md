# Data availability and units

## Oxford-Man realised measures

The public QLIKE calculation requires the Oxford-Man Realized Library source
used in the study. The data are third-party material and are not redistributed
in this archive. Researchers must obtain lawful access from the provider or an
authorised distributor under the applicable terms.

- Source: Heber, Gerd, Asger Lunde, Neil Shephard and Kevin K. Sheppard (2009),
  *Oxford-Man Institute's realized library*, Oxford-Man Institute, University of
  Oxford, Version 0.3.
- Archived source ZIP: [1 March 2022 snapshot](https://web.archive.org/web/20220301022212id_/https://realized.oxford-man.ox.ac.uk/images/oxfordmanrealizedvolatilityindices.zip).
- Source filename: `oxfordmanrealizedvolatilityindices.csv`.
- File size: 57,237,569 bytes.
- SHA-256: `94ba808f56301d2d463e3b111fdcce469808f098350a4281c09584457957f8aa`.
- Selected symbols: `.SPX`, `.GDAXI`, and `.FTSE`.

## Obtaining the source file

1. Download the [archived Oxford-Man source ZIP](https://web.archive.org/web/20220301022212id_/https://realized.oxford-man.ox.ac.uk/images/oxfordmanrealizedvolatilityindices.zip).
   This snapshot contains observations through 25 February 2022.
2. Extract `oxfordmanrealizedvolatilityindices.csv`. The ZIP is 19,275,486 bytes,
   with SHA-256 `f620ad530e35e4e28c667cfd53a9b68e8cac42133bdf3e734614fffc58f87be1`.
3. Check the extracted CSV against the size and SHA-256 above. A different
   snapshot may not reproduce the reported samples and evaluation records.
4. For `reproduce.py --data-root DATA_ROOT`, place the CSV in the layout below.
   For the standalone `build_daily_data.py --source` command only, it can instead
   be placed at any specified path, such as `OXFORD_MAN.csv` in the package root.

The complete workflow expects these four files under `DATA_ROOT`:

```text
DATA_ROOT/
  raw/external/omi/oxfordmanrealizedvolatilityindices.csv
  raw/hf/firstrate/SPX_full_5min.txt
  raw/hf/firstrate/UKX_full_5min.txt
  raw/hf/firstrate/DAX_full_5min.txt
```

The standalone `OXFORD_MAN.csv` example does not replace this layout. Source
identities and the FirstRate timestamp conventions are in `source_identity.json`.

## Research use and attribution

The [archived official Terms and Conditions](https://web.archive.org/web/20211205161411id_/https://realized.oxford-man.ox.ac.uk/terms-and-conditions)
permit researchers to use the library freely without restrictions, provided
that work using it cites Heber, Lunde, Shephard and Sheppard (2009) and identifies
the library version. The citation above supplies both. This package's calculated
evaluation records are outputs of the academic research using Version 0.3.

The source retains its copyright notice. The package licence does not relicense
the Oxford-Man database or the underlying Thomson Reuters tick data. Source
observations remain outside this package and are obtained from the archived
provider distribution. Please retain the required library citation in subsequent
research using these materials.

`code/scripts/build_daily_data.py` selects the three indices, uses the source
trading-date label, constructs close-to-close log returns in decimal units,
and expresses realised variances in squared decimal-return units. The resulting
archive contains 5,551 SP500 observations from 2000-01-04, 5,613 DAX
observations from 2000-01-04, and 5,585 FTSE100 observations from 2000-01-05.
All three series end on 2022-02-25, for 16,749 observations in total.

`code/scripts/audit_calendars.py` compares observed dates with the relevant
exchange calendars. Missing sessions are recorded and no observations are
imputed.

## Included forecast panels

Each file in `forecasts/` contains one row per market, model, forecast origin,
and horizon. `variance_forecast` is integrated variance in squared
decimal-return units, `cumulative_return_mean` is a decimal log return, and
`horizon` counts trading sessions. Combination fields report the members
available at the forecast origin and their weights.

The forecast panels contain no source observations, realised outcomes, daily
losses, posterior draws, or predictive arrays.

`forecasts/combination_warmup.parquet` contains the 2017 NN-SV and RV-NN-SV
variance forecasts used to initialize the QLIKE combination history. Its 4,434
rows form the complete two-model candidate grid; 66 SP500 NN-SV rows are marked
unavailable. The file contains no realised variance, returns, evaluated log
densities, source observations, or local paths. Researchers with the documented
Oxford-Man file can reconstruct the corresponding realised targets.

## Restricted empirical inputs

Source observations and reconstructed price/variance series are not distributed.
Unrounded daily evaluation records and predictive functionals are included in
`inputs/evaluation/` for the reported outcome definitions. These are calculated
statistics, not a replacement for the underlying source files. The package licence
does not grant rights in third-party data or override their terms. Full predictive
components are retained separately for distribution-level rescoring. Researchers with the documented Oxford-Man source can regenerate
the daily data, state-space predictive components, and their log-score, PIT,
CRPS, VaR, ES, and FZ0 loss rows. The same workflow generates the Gaussian and
Student-t GARCH components. Gaussian component files contain four scoring
arrays; Student-t files add path-specific degrees of freedom. The posterior and
counterfactual portions of the RV-RA conditional-value analysis depend on
additional compact inputs described in the package README. Nothing in the
archive changes the third-party data terms.

## Synthetic input

`identification_reference_curve.npz` contains the calibration curve used by
the synthetic identification experiment. It is included in this archive and
is read by `code/scripts/prepare_identification_simulation.py`.

## FirstRate reconstruction

The outcome reconstruction uses separately licensed five-minute SPX, UKX
and DAX data. Raw prices and reconstructed daily outcome series are not distributed. The
compact package includes the calculated distribution-score records needed for
the reported reconstructed-outcome comparisons. `source_identity.json` records relative filenames, hashes and
construction conventions without price observations. See `../README.md`
for acquisition layout, the common span/count screen and the executable
reconstruction commands. The additional MS forecast panel contains predictions
and eligibility information only.
