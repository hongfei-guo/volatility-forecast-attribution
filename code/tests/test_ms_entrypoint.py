from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

from bnsv.har_filter_policy import POLICY
from bnsv.har_input_generation import generate_har_input_pair


def test_multiscale_default_uses_reported_filter():
    assert POLICY == 'grid_agreement'
    assert inspect.signature(generate_har_input_pair).parameters['filter_policy'].default == POLICY


def test_multiscale_cli_exposes_only_reported_filter():
    script = Path(__file__).resolve().parents[1] / 'scripts/generate_har_input_forecasts.py'
    result = subprocess.run([sys.executable, str(script), '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert '--filter-policy {grid_agreement}' in result.stdout
    assert '--sampling-policy {matched_097_divergence_099_ess}' in result.stdout
