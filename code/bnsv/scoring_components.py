"""Lossless scoring inputs from predictive draws, selections or GARCH parameters."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .garch_models import simulate_garch_t_paths


def from_paths(arrays, horizon):
    returns = np.asarray(arrays['raw_returns'])
    variances = np.asarray(arrays['raw_variances'])
    if returns.ndim != 2 or returns.shape != variances.shape or returns.shape[1] != horizon:
        raise ValueError('predictive paths do not match the horizon')
    result = {'cumulative_returns': returns.sum(axis=1),
              'integrated_variances': variances.sum(axis=1)}
    if horizon == 1:
        locations = np.asarray(arrays['raw_return_locations'])
        sds = np.asarray(arrays['raw_conditional_sds'])
        if locations.shape != returns.shape or sds.shape != returns.shape:
            raise ValueError('one-day mixture arrays must match the predictive paths')
        result['locations'] = locations[:, 0]
        result['conditional_sds'] = sds[:, 0]
        if 'degrees_of_freedom' in arrays:
            result['degrees_of_freedom'] = np.asarray(arrays['degrees_of_freedom'])
    return result


def replay_garch(arrays):
    fit = SimpleNamespace(**{key: float(arrays[key]) for key in ['omega', 'alpha', 'beta', 'nu']})
    horizon, paths = int(arrays['horizon']), int(arrays['paths'])
    z, v = simulate_garch_t_paths(fit, forecast_variance=float(arrays['forecast_variance']),
                                horizon=horizon, paths=paths,
                                rng=np.random.default_rng(int(arrays['seed'])))
    mean, scale = float(arrays['return_mean']), float(arrays['return_scale'])
    result = {'cumulative_returns': (mean + scale*z).sum(axis=1),
              'integrated_variances': (scale**2*v).sum(axis=1)}
    if horizon == 1:
        result.update(locations=np.full(paths, mean),
                      conditional_sds=scale*np.sqrt(v[:, 0]),
                      degrees_of_freedom=np.full(paths, fit.nu))
    return result


def _inside(root, name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('component paths must be relative to their directory')
    return root/path


def load_components(root, filename, horizon, _parents=()):
    root = Path(root)
    path = _inside(root, filename)
    if filename in _parents:
        raise ValueError('cyclic component reference')
    with np.load(path, allow_pickle=False) as z:
        if 'source_files' in z:
            files = z['source_files'].tolist()
            models, draws = z['source_model_indices'], z['source_draw_indices']
            if models.ndim != 1 or draws.shape != models.shape:
                raise ValueError('unaligned combination indices')
            if models.dtype.kind not in 'iu' or draws.dtype.kind not in 'iu':
                raise ValueError('combination indices must be integers')
            if models.size == 0 or models.min() < 0 or models.max() >= len(files) or draws.min() < 0:
                raise ValueError('invalid combination indices')
            result = {}
            expected_fields = None
            for model in np.unique(models):
                member = load_components(root, files[int(model)], horizon, (*_parents, filename))
                fields = set(member) - {"pit_uniform"}
                if expected_fields is not None and fields != expected_fields:
                    raise ValueError("candidate scoring fields differ")
                expected_fields = fields
                chosen = models == model
                for key, values in member.items():
                    if key == 'pit_uniform':
                        continue
                    if draws[chosen].max() >= len(values):
                        raise ValueError('combination index exceeds candidate draws')
                    result.setdefault(key, np.empty(models.size, dtype=values.dtype))[chosen] = values[draws[chosen]]
        elif 'omega' in z:
            if int(z['horizon']) != horizon:
                raise ValueError('GARCH horizon differs from the record')
            result = replay_garch(z)
        elif 'cumulative_returns' in z:
            keys = ['cumulative_returns', 'integrated_variances']
            if horizon == 1:
                keys += ['locations', 'conditional_sds']
                if 'degrees_of_freedom' in z:
                    keys.append('degrees_of_freedom')
            result = {key: np.asarray(z[key]) for key in keys}
        else:
            result = from_paths(z, horizon)
        if 'pit_uniform' in z:
            result['pit_uniform'] = float(z['pit_uniform'])
    n = len(result['cumulative_returns'])
    for key, values in result.items():
        if key == 'pit_uniform':
            if not 0 <= values <= 1:
                raise ValueError('PIT randomization must be in [0,1]')
        elif values.shape != (n,) or n == 0 or not np.isfinite(values).all():
            raise ValueError(f'invalid scoring vector: {key}')
    if not (result['integrated_variances'] > 0).all():
        raise ValueError('predictive variances must be positive')
    if horizon == 1 and not (result['conditional_sds'] > 0).all():
        raise ValueError('conditional standard deviations must be positive')
    return result


def ensemble_pit(draws, target, uniform):
    values = np.asarray(draws)
    return float((np.sum(values < target) + uniform*(np.sum(values == target)+1))/(values.size+1))
