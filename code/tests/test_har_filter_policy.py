from types import SimpleNamespace
import numpy as np
import pytest

from bnsv.forecast_filter import FixedParameterGridFilter
import bnsv.har_filter_policy as policy


def test_streaming_transition_matches_dense():
    args=dict(parameters={'mu':np.zeros(6),'phi':np.linspace(.8,.98,6),
        'sigma_eta':np.full(6,.1),'nu':np.full(6,8.)},initial_state=np.linspace(-.2,.2,6),
        weights=np.full(6,1/6),ancestry=np.arange(6),level=2,transition_batch_size=2)
    dense=FixedParameterGridFilter(**args)
    streamed=FixedParameterGridFilter(**args,cache_transition_kernel=False)
    for z in [.1,-.2,.05]:
        for bank in (dense,streamed):
            bank.step(x_t=np.zeros(4),observed_z=z,correction=lambda p,x:np.zeros(6))
        np.testing.assert_allclose(dense.state_probabilities,streamed.state_probabilities,rtol=1e-14,atol=1e-15)
        np.testing.assert_allclose(dense.weights,streamed.weights,rtol=1e-14,atol=1e-15)
    assert streamed._transition_kernel is None


def test_direct_agreement_separates_low_ess_and_numerical_error(monkeypatch):
    monkeypatch.setattr(policy,'variance_mean',lambda b,**kw:b.mean)
    a=SimpleNamespace(size=4000,weights=np.full(4000,1/4000),ess=3900.,mean=1.)
    b=SimpleNamespace(size=4000,weights=a.weights.copy(),ess=3900.01,mean=1.0001)
    kwargs=dict(model_id='RV-LIN-SV-HAR',next_x=np.zeros(4),return_scale=1.)
    assert policy.assess_agreement({3:a,4:b},**kwargs)['decision']=='continue'
    a.ess=700.;b.ess=700.01
    assert policy.assess_agreement({3:a,4:b},**kwargs)['decision']=='refit'
    b.mean=1.01
    assert policy.assess_agreement({3:a,4:b},**kwargs)['decision']=='refine'


def test_refinement_replays_all_history_and_exhaustion_is_not_mcmc(monkeypatch):
    class Bank:
        def __init__(self):self.seen=[]
        def step(self,**kw):self.seen.append(kw['observed_z'])
    original={2:Bank(),3:Bank(),4:Bank()};history=[(np.zeros(4),.1)]
    made=[]
    def make(*a,**kw):
        value=Bank();made.append(value);return value
    monkeypatch.setattr(policy,'new_bank',make)
    def replay(refit,level,history,model):
        bank=make()
        for x,z in history:bank.step(observed_z=z)
        return bank
    monkeypatch.setattr(policy,'replay_finer',replay)
    monkeypatch.setattr(policy,'filter_bank_ess_diagnostic',lambda *a,**kw:{'decision':'unresolved'})
    monkeypatch.setattr(policy,'assess_agreement',lambda b,**kw:{'decision':'continue' if max(b)==5 else 'refine'})
    ok,details=policy.update_and_assess(original,refit={'weights':np.full(4,.25)},history=history,model_id='RV-LIN-SV-HAR',
        x=np.zeros(4),observed_z=.2,next_x=np.zeros(4),return_scale=1.)
    assert ok and made[0].seen==[.1,.2] and set(original)=={3,4,5}
    monkeypatch.setattr(policy,'assess_agreement',lambda b,**kw:{'decision':'refine'})
    with pytest.raises(policy.FilterAccuracyError):
        policy.update_and_assess(original,refit={'weights':np.full(4,.25)},history=history,model_id='RV-LIN-SV-HAR',
            x=np.zeros(4),observed_z=.3,next_x=np.zeros(4),return_scale=1.)


def test_analytic_variance_mean_at_refit():
    n=4
    p={'mu':np.zeros(n),'phi':np.full(n,.9),'sigma_eta':np.full(n,.1),'nu':np.full(n,8.),
       'beta':np.zeros((n,4)),'x_center':np.zeros((n,4))}
    b=FixedParameterGridFilter(parameters=p,initial_state=np.full(n,.2),weights=np.full(n,.25),
        ancestry=np.arange(n),level=2,cache_transition_kernel=False)
    expected=.02**2*np.exp(.9*.2+.5*.1**2)
    assert np.isclose(policy.variance_mean(b,model_id='RV-LIN-SV-HAR',next_x=np.zeros(4),return_scale=.02),expected)


def test_batched_refinement_preserves_global_weights(monkeypatch):
    from bnsv.forecast_paths import filter_from_refit
    n=35
    refit={'weights':np.arange(1,n+1,dtype=float),'ancestry':np.arange(n),
           'state':np.linspace(-.2,.2,n),'parameter__mu':np.zeros(n),
           'parameter__phi':np.linspace(.8,.97,n),'parameter__sigma_eta':np.full(n,.1),
           'parameter__nu':np.full(n,8.),'parameter__beta':np.zeros((n,4)),
           'parameter__x_center':np.zeros((n,4))}
    refit['weights']/=refit['weights'].sum()
    history=[(np.zeros(4),z) for z in [.1,-.2,.05]]
    expected=filter_from_refit(refit,level=2)
    for x,z in history:
        expected.step(x_t=x,observed_z=z,correction=lambda p,x:np.zeros(n))
    monkeypatch.setattr(policy,'new_bank',lambda r,l:filter_from_refit(r,level=l,cache_transition_kernel=False))
    actual=policy.replay_finer(refit,2,history,'RV-LIN-SV-HAR')
    np.testing.assert_allclose(actual.weights,expected.weights,rtol=1e-13,atol=1e-15)
    np.testing.assert_allclose(actual.state_probabilities,expected.state_probabilities,rtol=1e-13,atol=1e-15)
    np.testing.assert_allclose(actual.cumulative_log_likelihood,expected.cumulative_log_likelihood,rtol=1e-13,atol=1e-15)


def test_month_blocks_match_one_continuous_run(synthetic_frame,tmp_path,monkeypatch):
    import pandas as pd
    import bnsv.har_input_generation as generation
    from bnsv.forecast_features import HAR_INPUT_MODELS
    from test_har_input_extension import MODELS,SyntheticFit
    frame=synthetic_frame.copy();frame['date']+=pd.DateOffset(years=1)
    monkeypatch.setattr(policy,'filter_bank_ess_diagnostic',lambda *a,**kw:{'decision':'continue'})
    def fit(**kw):return SyntheticFit(kw['data'],kw['model_id']),[]
    common=dict(daily=frame,market='SP500',stan_dir=MODELS,correction_scales=dict.fromkeys(HAR_INPUT_MODELS,.17),
        particle_count=4,path_count=8,fit_function=fit,filter_policy=policy.POLICY)
    full=generation.generate_har_input_pair(**common,output_dir=tmp_path/'full',
        forecast_start=frame.loc[59,'date'],forecast_end=frame.loc[81,'date'])
    parts=[]
    for month,dates in frame.loc[59:81].groupby(frame.loc[59:81,'date'].dt.to_period('M')):
        parts.append(generation.generate_har_input_pair(**common,output_dir=tmp_path/str(month),
            forecast_start=dates.date.iloc[0],forecast_end=dates.date.iloc[-1]))
    combined=pd.concat(parts,ignore_index=True)
    pd.testing.assert_frame_equal(full,combined)


def test_low_ess_interval_does_not_require_forecast_precision(monkeypatch):
    class Bank:
        ess=710.
        def step(self,**kw): pass
    monkeypatch.setattr(policy,'filter_bank_ess_diagnostic',
        lambda *a,**kw:{'decision':'refit','ess_lower':704.,'ess_upper':717.})
    def forbidden(*a,**kw): raise AssertionError('discarded posterior need not certify variance mean')
    monkeypatch.setattr(policy,'assess_agreement',forbidden)
    ok,details=policy.update_and_assess({2:Bank(),3:Bank(),4:Bank()},
        refit={'weights':np.full(4000,1/4000)},history=[],model_id='RV-NN-SV-HAR',
        x=np.zeros(4),observed_z=.1,next_x=np.zeros(4),return_scale=1.)
    assert not ok and details[-1]['acceptance_basis']=='ess_interval_below_threshold'


def test_shared_refit_does_not_update_other_discarded_bank(synthetic_frame,tmp_path,monkeypatch):
    import pandas as pd
    import bnsv.har_input_generation as generation
    from bnsv.forecast_features import HAR_INPUT_MODELS
    from test_har_input_extension import MODELS,SyntheticFit
    frame=synthetic_frame.copy();frame['date']+=pd.DateOffset(years=1)
    calls=[]
    def update(*a,**kw):
        calls.append(kw['model_id'])
        assert kw['model_id']==HAR_INPUT_MODELS[0]
        return False,[{'decision':'refit'}]
    monkeypatch.setattr(generation,'update_and_assess',update)
    def fit(**kw):return SyntheticFit(kw['data'],kw['model_id']),[]
    panel=generation.generate_har_input_pair(daily=frame,market='SP500',stan_dir=MODELS,
        output_dir=tmp_path/'run',correction_scales=dict.fromkeys(HAR_INPUT_MODELS,.17),
        forecast_start=frame.loc[59,'date'],forecast_end=frame.loc[61,'date'],
        particle_count=4,path_count=8,fit_function=fit,filter_policy=policy.POLICY)
    assert len(panel)==6 and calls==[HAR_INPUT_MODELS[0]]*2
