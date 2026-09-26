"""Transfer and reuse accepted single-origin HAR posteriors without new MCMC."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import zipfile
import tempfile
import shutil
import numpy as np
from .forecast_features import HAR_INPUT_MODELS
FILES = ('posterior.npz', 'run.json', 'fit.json', 'prior_calibration.json', 'stan_data.json')

def accepted_fit(fit):
    if fit['status'] != 'accepted' or not fit['attempts']:
        raise ValueError('posterior requires an accepted fit')
    diagnostic = fit['attempts'][-1]['diagnostics']
    if not diagnostic['passed'] or diagnostic['divergences'] != 0:
        raise ValueError('posterior does not pass the recorded diagnostics')

def load_posterior_caches(roots, *, market, data_sha256):
    result = {}
    for root in map(Path, roots):
        manifest = json.loads((root / 'POSTERIORS.json').read_text())
        if manifest['market'] != market or manifest['data_sha256'] != data_sha256:
            raise ValueError('posterior cache market or data identity differs')
        name = manifest['model_id']
        if name not in HAR_INPUT_MODELS:
            raise ValueError('posterior cache model is unsupported')
        for origin in manifest.get('unavailable_origins', []):
            key = (name, origin)
            if key in result:
                continue
            result[key] = {'unavailable': True, 'manifest': root / 'POSTERIORS.json'}
        for entry in manifest['refits']:
            key = (name, entry['origin'])
            paths = {}
            for (filename, identity) in entry['files'].items():
                path = (root / identity['path']).resolve()
                if root.resolve() not in path.parents:
                    raise ValueError('cache member is outside its root')
                data = path.read_bytes()
                if len(data) != identity['bytes'] or hashlib.sha256(data).hexdigest() != identity['sha256']:
                    raise ValueError('cached posterior member identity differs')
                paths[filename] = path
            if set(paths) != set(FILES):
                raise ValueError('cached posterior evidence is incomplete')
            if key in result and (not result[key].get('unavailable')):
                with np.load(result[key]['posterior.npz'], allow_pickle=False) as first, np.load(paths['posterior.npz'], allow_pickle=False) as second:
                    identical = set(first.files) == set(second.files) and all((np.array_equal(first[field], second[field]) for field in first.files))
                if not identical:
                    raise ValueError('conflicting accepted posteriors for the same model/date')
                continue
            result[key] = paths
    return result

def read_cached_refit(paths, *, market, model_id, origin, seed, data, transform, particle_count):
    run = json.loads(paths['run.json'].read_text())
    fit = json.loads(paths['fit.json'].read_text())
    accepted_fit(fit)
    for (key, expected) in {'market': market, 'model_id': model_id, 'origin_date': origin, 'seed': seed, 'horizon': 1, 'chains': 4, 'training_rows': int(data['N']), 'feature_names': list(transform.feature_names)}.items():
        if run[key] != expected:
            raise ValueError(f'cached posterior identity differs: {key}')
    stored_data = json.loads(paths['stan_data.json'].read_text())
    if set(stored_data) != set(data):
        raise ValueError('cached Stan input schema differs')
    for key in data:
        (a, b) = (np.asarray(stored_data[key]), np.asarray(data[key]))
        if a.shape != b.shape or not np.allclose(a, b, rtol=1e-12, atol=1e-14):
            raise ValueError(f'cached Stan inputs differ: {key}')
    for (key, expected) in {'rv_floor': transform.rv_floor, 'return_mean': transform.return_scaler.mean, 'return_scale': transform.return_scaler.scale, 'feature_mean': transform.feature_mean, 'feature_scale': transform.feature_scale}.items():
        if not np.allclose(run[key], expected, rtol=1e-12, atol=1e-14):
            raise ValueError(f'cached transformation differs: {key}')
    with np.load(paths['posterior.npz'], allow_pickle=False) as archive:
        compact = {key: archive[key] for key in archive.files}
    if compact['weights'].shape != (particle_count,) or compact['state'].shape != (particle_count,):
        raise ValueError('cached posterior particle count differs')
    for key in ('feature_mean', 'feature_scale', 'return_mean', 'return_scale'):
        if not np.allclose(compact[key], run[key], rtol=1e-12, atol=1e-14):
            raise ValueError(f'cached posterior transformation differs: {key}')
    if list(compact['feature_names']) != list(transform.feature_names):
        raise ValueError('cached posterior feature names differ')
    return (compact, fit['attempts'])
