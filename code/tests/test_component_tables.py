from pathlib import Path
import importlib.util
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'code/scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_baseline_weighted_multiplier_and_component_alignment():
    module = load_script('decompose_five_forecasts')
    b = np.log([1.0, 3.0])
    c = np.log([2.0, 0.5])
    v = np.array([2.0, 1.5])
    result = module.decompose(1.0, b, c, v)
    assert result['baseline_moment'] == pytest.approx(2.0)
    assert result['forecast'] == pytest.approx(1.75)
    assert result['correction_multiplier'] == pytest.approx(0.875)
    assert result['correction_multiplier'] != pytest.approx(np.exp(c.mean()))
    with pytest.raises(ValueError, match='aligned'):
        module.decompose(1.0, b[:1], c, v)
    with pytest.raises(AssertionError):
        module.decompose(1.0, b, c[::-1], v)


def test_summary_retains_unsuccessful_replications(tmp_path):
    records = tmp_path/'records'
    shutil.copytree(ROOT/'inputs/identification', records)
    command = [sys.executable, str(ROOT/'code/scripts/summarize_identification_simulation.py'),
               '--records', str(records), '--output', str(tmp_path/'summary')]
    subprocess.run(command, check=True, capture_output=True)
    summary = pd.read_csv(tmp_path/'summary/simulation_summary.csv').set_index('dgp_id')
    assert summary.loc['DGP-N', 'planned_replications'] == 200
    assert summary.loc['DGP-N', 'successful_fits'] == 198
    assert summary.loc['DGP-N', 'unsuccessful_fits'] == 2
    assert summary.loc['DGP-N', 'mean_function_pointwise_coverage_planned_denominator'] == pytest.approx(
        summary.loc['DGP-N', 'mean_function_pointwise_coverage_conditional_successful'] * 198 / 200)
    outcomes = pd.read_csv(records/'fit_outcomes.csv')
    outcomes[outcomes.fit_outcome == 'successful'].to_csv(records/'fit_outcomes.csv', index=False)
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'retain all planned replications' in result.stderr
