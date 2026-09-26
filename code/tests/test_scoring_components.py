from pathlib import Path
import numpy as np
import pytest
from bnsv.scoring_components import load_components, from_paths, replay_garch, ensemble_pit
from bnsv.garch_models import GarchTFit, simulate_garch_t_paths


def test_summed_vectors_preserve_paths_and_pit(tmp_path):
    returns = np.array([[.1, .2], [-.3, .1], [.2, -.1]])
    variance = np.array([[1., 2.], [2., 4.], [3., 6.]])
    raw = {'raw_returns': returns, 'raw_variances': variance}
    np.savez_compressed(tmp_path/'paths.npz', **raw)
    expected = from_paths(raw, 2)
    np.savez_compressed(tmp_path/'vectors.npz', **expected, pit_uniform=.37)
    a, b = load_components(tmp_path, 'paths.npz', 2), load_components(tmp_path, 'vectors.npz', 2)
    for key in expected:
        np.testing.assert_array_equal(a[key], b[key])
    assert ensemble_pit(a['cumulative_returns'], .1, .37) == ensemble_pit(b['cumulative_returns'], .1, b['pit_uniform'])


def test_combination_indices_preserve_aligned_candidate_fields(tmp_path):
    a = {'cumulative_returns': np.array([1., 2., 3.]), 'integrated_variances': np.array([2., 3., 4.]),
         'locations': np.zeros(3), 'conditional_sds': np.ones(3), 'degrees_of_freedom': np.full(3, 8.)}
    b = {k: v+10 for k,v in a.items()}
    np.savez_compressed(tmp_path/'a.npz', **a)
    np.savez_compressed(tmp_path/'b.npz', **b)
    np.savez_compressed(tmp_path/'mix.npz', source_files=np.array(['a.npz','b.npz']),
                        source_model_indices=np.array([0,1,0],dtype=np.uint8),
                        source_draw_indices=np.array([2,0,1],dtype=np.uint16))
    result = load_components(tmp_path,'mix.npz',1)
    for key in a:
        np.testing.assert_array_equal(result[key], [a[key][2],b[key][0],a[key][1]])
    with pytest.raises(ValueError, match='horizon'):
        raw={'raw_returns':np.zeros((3,2)), 'raw_variances':np.ones((3,2))}
        from_paths(raw,1)


def test_garch_parameters_replay_exact_samples():
    fit=GarchTFit(variance_bar=1.,persistence=.95,arch_share=.05/.95,omega=.05,
                  alpha=.05,beta=.9,nu=8.,negative_log_likelihood=10.,optimizer_start=0)
    z,v=simulate_garch_t_paths(fit,forecast_variance=1.,horizon=5,paths=32,rng=np.random.default_rng(7))
    payload={k:np.asarray(getattr(fit,k)) for k in ['omega','alpha','beta','nu']}
    payload.update(horizon=np.asarray(5),paths=np.asarray(32),seed=np.asarray(7),
                   forecast_variance=np.asarray(1.),return_mean=np.asarray(.01),return_scale=np.asarray(.02))
    result=replay_garch(payload)
    np.testing.assert_array_equal(result['cumulative_returns'],(.01+.02*z).sum(axis=1))
    np.testing.assert_array_equal(result['integrated_variances'],(.02**2*v).sum(axis=1))
