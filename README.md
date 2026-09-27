# Information, Not Flexibility

*Information, Not Flexibility: Attributing Forecast Gains in Neural Stochastic Volatility*  
Hongfei Guo, J. Miguel Marín and Helena Veiga.

When a neural volatility model improves a forecast, is the gain due to its inputs,
its functional form or forecast combination? This study separates those channels
within a Bayesian stochastic volatility framework, using three equity markets
and one-, five- and ten-day forecasts. It also studies identification and the
existence of finite multi-step predictive variance means.

[Replication archive](https://doi.org/10.5281/zenodo.22923099) · [Reproduction guide](REPRODUCING.md)

## Explore the code

- `code/bnsv/`: models, filtering, scoring and evaluation.
- `code/scripts/`: forecast generation and result reproduction.
- `code/tests/`: synthetic tests requiring no restricted market data.
- `design/`: model, sample and evaluation definitions.

## Start with the synthetic tests

Using Python 3.10, run from the repository root:

```bash
python3.10 -m venv .venv
.venv/bin/pip install -r environment/requirements.txt
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=code .venv/bin/python -m pytest -q -p no:cacheprovider code/tests
```

## Reproduce the empirical results

The default reproduction rebuilds the reported tables and figures without
rerunning MCMC. It requires the documented Oxford-Man and FirstRate source data,
which are not included. The repository supplies saved forecasts, evaluation
records and compact simulation components.

See [REPRODUCING.md](REPRODUCING.md) for data acquisition, run commands, exhibit
mapping and the additional components required for distribution-level rescoring.
Original replication materials are licensed under [CC BY 4.0](LICENSE.txt).
