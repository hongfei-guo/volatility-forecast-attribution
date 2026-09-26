"""Reconstruct the five-origin baseline/correction table from aligned draws."""
from pathlib import Path
import argparse
import hashlib

import numpy as np
import pandas as pd


def decompose(return_scale, baseline_state, correction, variance):
    scale_array = np.asarray(return_scale)
    if scale_array.size != 1:
        raise ValueError("return scale must be scalar")
    scale = float(scale_array.reshape(-1)[0])
    b, c, v = (np.asarray(x, dtype=float) for x in (baseline_state, correction, variance))
    if not (b.ndim == c.ndim == v.ndim == 1 and b.shape == c.shape == v.shape and b.size):
        raise ValueError('predictive arrays must be nonempty aligned vectors')
    if not (np.isfinite(scale) and scale > 0 and np.isfinite(b).all() and np.isfinite(c).all() and np.isfinite(v).all() and (v > 0).all()):
        raise ValueError('invalid predictive component values')
    np.testing.assert_allclose(scale**2 * np.exp(b + c), v, rtol=1e-12, atol=1e-15)
    baseline = float(np.mean(scale**2 * np.exp(b)))
    forecast = float(np.mean(v))
    return dict(forecast=forecast, baseline_moment=baseline,
                correction_multiplier=forecast / baseline,
                mean_baseline=float(b.mean()), mean_correction=float(c.mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--components', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = pd.read_csv(args.records, dtype={'origin_date': str})
    models = ['NN-SV', 'RV-NN-SV', 'RV-LIN-SV']
    if len(source) != 15 or source.duplicated(['origin_date', 'model_id']).any():
        raise ValueError('expected fifteen unique model/origin records')
    if source.origin_date.nunique() != 5 or any(set(g.model_id) != set(models) for _, g in source.groupby('origin_date')):
        raise ValueError('each of five origins must contain the three compared models')
    rows = []
    for date, group in source.groupby('origin_date', sort=True):
        for model in models:
            record = group[group.model_id == model].iloc[0]
            path = args.components / record.component_file
            if hashlib.sha256(path.read_bytes()).hexdigest() != record.component_sha256:
                raise ValueError('component differs from its record')
            with np.load(path, allow_pickle=False) as z:
                if str(z['origin_date']) != date or str(z['model_id']) != model:
                    raise ValueError('component identity differs from its record')
                result = decompose(z['return_scale'], z['baseline_state'], z['correction'], z['variance'])
            np.testing.assert_allclose(result['forecast'], record.variance_forecast, rtol=1e-11, atol=1e-14)
            rows.append(dict(origin_date=date, model_id=model, **result))
    table = pd.DataFrame(rows)
    contrasts = []
    for date, group in table.groupby('origin_date', sort=True):
        reference = group[group.model_id == 'NN-SV'].iloc[0]
        for model in models[1:]:
            row = group[group.model_id == model].iloc[0]
            total = np.log(row.forecast / reference.forecast)
            baseline = np.log(row.baseline_moment / reference.baseline_moment)
            correction = np.log(row.correction_multiplier / reference.correction_multiplier)
            np.testing.assert_allclose(total, baseline + correction, rtol=1e-12, atol=1e-12)
            contrasts.append(dict(origin_date=date, model_id=model, log_forecast_ratio=total,
                                  log_baseline_ratio=baseline, log_correction_multiplier_ratio=correction))
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output/'five_dates_decomposition.csv', index=False)
    pd.DataFrame(contrasts).to_csv(args.output/'five_dates_log_decomposition.csv', index=False)
    print('Reconstructed fifteen forecasts and ten baseline/correction contrasts.')


if __name__ == '__main__':
    main()
