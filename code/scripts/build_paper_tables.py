#!/usr/bin/env python3
"""Build the 21 manuscript tables from reproduced calculations and fixed design constants."""
from pathlib import Path
import argparse,json,re
import numpy as np
import pandas as pd
from bnsv.forecast_fit import NN_OUTPUT_BASE_SCALE, LINEAR_BASE_SCALE
from bnsv.mcs_dm import diebold_mariano
from bnsv.sample_alignment import select_evaluation_sample

MARKETS=['SP500','FTSE100','DAX']
LABELS={'SP500':r'S\&P~500','FTSE100':r'FTSE~100','DAX':'DAX'}
G='Realized-GARCH-Gaussian-QMLE';T='Realized-GARCH-t-MLE';NN='NN-SV';RV='RV-NN-SV';LIN='RV-LIN-SV';RA='RV-RA-NN-SV';MS='RV-NN-SV-HAR';ML='RV-LIN-SV-HAR'
MODELS=['HAR-RV','SHAR-RV','log-HAR-RV',G,'SV',NN,RV,'NN-ENSEMBLE-EQUAL','NN-ENSEMBLE-QLIKE']
NAMES={G:'RG (G)',T:r'RG ($t$)',MS:'RV-NN-SV-MS',ML:'RV-LIN-SV-MS'}

def one(d,**keys):
    mask=np.ones(len(d),dtype=bool)
    for k,v in keys.items():mask &= d[k].eq(v).to_numpy()
    x=d.loc[mask]
    if len(x)!=1:raise ValueError(f'expected one row for {keys}; found {len(x)}')
    return x.iloc[0]

def f(x,n=4,sign=False):
    if not np.isfinite(x):return '--'
    if round(float(x),n)==0:x=0.
    return format(float(x),f'+.{n}f' if sign and x!=0 else f'.{n}f')

def pval(x,n=3):return '<0.001' if x<.001 else f(x,n)
def diff(r,scale=1,n=4):
    return f(r['mean_loss_difference']*scale,n,True)+' ['+pval(r.get('p_value',r.get('p_value_two_sided')))+']'
def holm(v):
    v=np.asarray(v);order=np.argsort(v);r=np.empty(len(v));r[order]=np.minimum(1,np.maximum.accumulate((len(v)-np.arange(len(v)))*v[order]));return r

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reproduced',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();base=a.reproduced;a.output.mkdir(parents=True,exist_ok=True)
    tables=[]
    def save(label,title,panels):
        data=dict(label=label,title=title,panels=[dict(title=t,columns=c,rows=r) for t,c,r in panels])
        stem=label.split(':',1)[1]
        (a.output/(stem+'.json')).write_text(json.dumps(data,indent=2)+'\n')
        csv=[]
        for title,columns,rows in panels:
            for row in rows:
                if len(row)!=len(columns):raise ValueError((label,row,columns))
                cells=list(map(str,row));csv.append(dict(panel=title,**dict(zip(columns,cells))))
        layout=Path(__file__).resolve().parents[2]/'design/table_layouts'/f'{stem}.tex'
        template=layout.read_text()
        all_rows=[row for _,_,rows_ in panels for row in rows_]
        if len(re.findall(r'@@ROW_\d+@@',template))!=len(all_rows):
            raise ValueError(f'table layout row count differs: {label}')
        for i,row in enumerate(all_rows):
            rendered=' & '.join(str(c).replace('<0.001', '$<0.001$') for c in row)+r' \\'
            template=template.replace(f'@@ROW_{i}@@',rendered)
        (a.output/(stem+'.tex')).write_text(template)
        pd.DataFrame(csv).to_csv(a.output/(stem+'.csv'),index=False)
        tables.append(dict(label=label,table=stem+'.tex',rows=sum(len(r) for _,_,r in panels)))
    daily=pd.read_parquet(base/'daily.parquet');scores=pd.read_parquet(base/'scores.parquet');scores=select_evaluation_sample(scores)
    for c in ['origin_date','mature_date']:scores[c]=pd.to_datetime(scores[c])
    cal=pd.read_csv(base/'evaluation/distribution_calibration.csv');dm=pd.read_csv(base/'evaluation/dm_results.csv');means=pd.read_csv(base/'evaluation/mcs_common_sample_mean_qlike.csv');mcs=pd.read_csv(base/'evaluation/mcs_sets.csv')
    tail=pd.read_csv(base/'tail/all_models.csv');tdm=pd.read_csv(base/'tail/pairwise_dm.csv');cmp=pd.read_csv(base/'comparisons/comparisons.csv');rg=pd.read_csv(base/'realized_garch_t/pairwise_dm.csv')
    def comparison(role,metric,h,m,period='full',variant='original'):
        return one(cmp,role=role,metric=metric,horizon=h,market=m,period=period,variant=variant)
    contrasts=[('Realised information','information','qlike',h) for h in (1,5,10)]+[('Neural form','neural_form','qlike',1)]+[('QLIKE combination','combination','qlike',h) for h in (1,5,10)]+[('Realised information','information','return_log_score',1),('Negative-log-score combination','combination','return_log_score',1)]
    # Model classification and prior scales are fixed scientific design information.
    rows=[['Gaussian Realized GARCH','Gaussian','Yes',r'Linear-quadratic; $1-2\gamma\tau_2>0$','Finite; computed analytically'],[r'Student-$t$ Realized GARCH',r'Student-$t$','Yes',r'Linear-quadratic; fitted $\gamma\tau_2>0$','Infinite; one-day only'],[LIN,r'Student-$t$','Yes',r'Unbounded linear return feedback; $\beta_z\neq0$','Infinite; one-day only'],['SV and NN-SV',r'Student-$t$','No','Zero or uniformly bounded correction','Finite conditional on parameters'],['RV-NN-SV and RV-RA-NN-SV',r'Student-$t$','Yes','Uniformly bounded correction','Finite conditional on parameters']]
    save('tab:triad','Return laws, observable feedback, and multi-step variance means.',[('', ['Specification','Return law','Uses realised information','Feedback into log variance',r'$\ell>1$ variance mean'],rows)])
    save('tab:oa-prior-scales','Prior scales for the observable correction functions.',[('', ['Market',NN,RV,LIN],[[LABELS[m],f(NN_OUTPUT_BASE_SCALE[m,NN],5),f(NN_OUTPUT_BASE_SCALE[m,RV],5),f(LINEAR_BASE_SCALE[m],5)] for m in MARKETS])])
    sim=pd.read_csv(base/'identification/simulation_summary.csv').set_index('dgp_id');rows=[]
    specs=[('Fits that pass the MCMC diagnostics',lambda r:f'{int(r.successful_fits)}/{int(r.planned_replications)}'),(r'Bias of the posterior mean',lambda r:f(r.phi_mean_bias_conditional_successful,5)),(r'RMSE of the posterior mean',lambda r:f(r.phi_rmse_conditional_successful,5)),(r'Coverage of the 95\% interval',lambda r:f(100*r.phi_coverage_planned_denominator,1)+r'\%'),('Median nRMSE',lambda r:f(r.median_function_nrmse_conditional_successful,3) if pd.notna(r.median_function_nrmse_conditional_successful) else 'n.a.'),('Mean pointwise coverage of the 95\% interval',lambda r:f(100*r.mean_function_pointwise_coverage_planned_denominator,1)+r'\%'),('RMSE of the posterior mean',lambda r:f(r.one_step_variance_rmse_conditional_successful,3)),(r'Coverage of the 95\% interval',lambda r:f(100*r.one_step_variance_coverage_planned_denominator,1)+r'\%'),(r'Replications with $S_{c,r}^\star\geq q_c$',lambda r:f'{int(r.declarations_at_or_above_qc)}/{int(r.replications_at_or_above_qc)}' if r.replications_at_or_above_qc else 'n.a.'),(r'Replications with $S_{c,r}^\star<q_c$',lambda r:f'{int(r.declarations_below_qc)}/{int(r.replications_below_qc)}')]
    for title,fn in specs:rows.append([title,*[fn(sim.loc[k]) for k in ('DGP-N','DGP-0')]])
    index_summary=pd.read_csv(base/'identification/correction_index_summary.csv')
    index_rows=[]
    for title,quantity,percent in [
        (r'Median of the true index $S_{c,r}^\star$','true_index_median',False),
        ('Median of the posterior medians','posterior_median_median',False),
        ('95th percentile of the posterior medians','posterior_median_q95',False),
        (r'Coverage of the 95\% interval for $S_{c,r}^\star$','coverage_all_replications',True),
        (r'Median posterior probability of reaching $q_c$','probability_at_least_qc_median',True),
    ]:
        values=[]
        for dgp in ('DGP-N','DGP-0'):
            if quantity=='coverage_all_replications' and dgp=='DGP-0':
                values.append('n.a.')
            else:
                value=one(index_summary,dgp_id=dgp,quantity=quantity)['value']
                values.append(f(100*value,1)+r'\%' if percent else ('0' if value==0 else f(value,5)))
        index_rows.append([title,*values])
    rows[8:8]=index_rows
    save('tab:simulation','Simulation results: recovery of the two parts of the log variance and declarations of a material correction.',[('', ['Quantity','DGP-N','DGP-0'],rows)])
    rows=[]
    for m in MARKETS:
        d=daily[daily.market.eq(m)]
        for col,title,scale in [('return_cc',r'Return (\%)',100),('rv_oc',r'$\mathrm{RV}^{OC}\times10^4$',1e4),('rv_cc',r'$\mathrm{RV}^{CC}\times10^4$',1e4),('asymmetry','Asymmetry',1)]:
            v=d[col]*scale;rows.append([LABELS[m],title,f'{len(v):,}',*[f(x,3) for x in [v.mean(),v.std(ddof=1),v.min(),v.max()]]])
    save('tab:data-summary','Descriptive statistics for the daily data.',[('', ['Market','Variable',r'$N$','Mean','SD','Minimum','Maximum'],rows)])
    panels=[]
    for title,loss in [('Mean QLIKE differences','qlike'),('Mean negative log score differences','return_log_score')]:
        rows=[[name,rf'$\ell={h}$',*[diff(comparison(role,loss,h,m)) for m in MARKETS]] for name,role,metric,h in contrasts if metric==loss]
        panels.append((title,['Comparison','Horizon',*[LABELS[m] for m in MARKETS]],rows))
    save('tab:matched-comparisons','Matched comparisons of information, functional form, and forecast combination.',panels)
    rows=[]
    for model in MODELS:
        values=[]
        for m in MARKETS:
            for h in (1,5,10):
                r=one(means,market=m,horizon=h,model_id=model);setrow=one(mcs,market=m,horizon=h);v=f(r.mean_qlike)
                cell=means[means.market.eq(m)&means.horizon.eq(h)]
                if r.mean_qlike==cell.mean_qlike.min():v=r'\underline{'+v+'}'
                if model in json.loads(setrow.retained_models_json):v=r'\textbf{'+v+'}'
                values.append(v)
        rows.append([{'NN-ENSEMBLE-EQUAL':'Equal-weight comb.','NN-ENSEMBLE-QLIKE':'QLIKE comb.',G:'Realized GARCH'}.get(model,model),*values])
    save('tab:primary-qlike-mcs','Mean QLIKE and 90\% model confidence sets.',[('', ['Model',*[rf'{LABELS[m]} $\ell={h}$' for m in MARKETS for h in (1,5,10)]],rows)])
    panels=[];rows=[]
    for m in MARKETS:
        d=cal[cal.market.eq(m)&cal.horizon.eq(1)&cal.model_id.isin(['SV',NN,RV,LIN,RA])];r=one(cal,market=m,horizon=1,model_id=G)
        rows.append([LABELS[m],f(d.coverage_95.min(),3)+'--'+f(d.coverage_95.max(),3),f'{int((d.pit_ks_p_value<.05).sum())}/5',f(r.coverage_95,3),pval(r.coverage_95_exact_binomial_p_value)])
    panels.append(('Central calibration',['Market','Student-t state-space coverage range','PIT rejects','RG (G) coverage','Binomial p'],rows))
    for level in ('01','05'):
        rows=[]
        for m in MARKETS:
            for model in (RV,G,T):
                r=one(tail,market=m,model_id=model);rows.append([LABELS[m],NAMES.get(model,model),f(100*r[f'var_{level}_exceedance_rate'],2),f(r[f'mean_var_es_{level}_fz0'])])
        panels.append((f'{int(level)}\% tail',['Market','Model',r'Exc. (\%)',r'$FZ_0$'],rows))
    rows=[]
    for ma,mb,alpha in [(RV,G,.01),(T,G,.01),(T,RV,.01),(T,RV,.05)]:
        rows.append([NAMES.get(ma,ma)+' $-$ '+NAMES.get(mb,mb),str(int(alpha*100))+r'\%',*[diff(one(tdm,market=m,model_a=ma,model_b=mb,alpha=alpha,loss=f'var_es_{int(alpha*100):02d}_fz0')) for m in MARKETS]])
    panels.append(('Selected FZ0 differences',['Comparison','Tail',*[LABELS[m] for m in MARKETS]],rows));save('tab:density-tail','One-day distributional calibration and tail comparisons.',panels)
    # MS pairwise means and Holm families are recomputed on each pair's common sample.
    msrows=[]
    for role,ma,mb in [('form',MS,ML),('NN',MS,RV),('LIN',ML,LIN)]:
        for m in MARKETS:
            z=scores[scores.market.eq(m)&scores.horizon.eq(1)&scores.model_id.isin([ma,mb])].pivot(index=['origin_date','mature_date'],columns='model_id',values='qlike')[[ma,mb]].dropna().sort_index()
            msrows.append(dict(role=role,market=m,n=len(z),mean_a=z[ma].mean(),mean_b=z[mb].mean(),**diebold_mariano(z[ma].to_numpy(),z[mb].to_numpy(),horizon=1)))
    ms=pd.DataFrame(msrows)
    for isform in [True,False]:
        ix=ms.index[ms.role.eq('form')==isform];ms.loc[ix,'holm']=holm(ms.loc[ix,'p_value'])
    ms.to_csv(a.output/'multiscale_calculations.csv',index=False)
    panels=[]
    for role,title in [('form','Neural versus linear correction with multiscale inputs'),('NN','Multiscale versus single lag: neural'),('LIN','Multiscale versus single lag: linear')]:
        rows=[]
        for m in MARKETS:
            r=one(ms,role=role,market=m);rows.append([LABELS[m],str(int(r.n)),f(r.mean_a),f(r.mean_b),f(r.mean_loss_difference,4,True),pval(r.p_value,4),pval(r.holm,4)])
        panels.append((title,['Market','N','First model QLIKE','Second model QLIKE','Difference','Raw DM p','Holm p'],rows))
    save('tab:ms-comparisons','Multiscale inputs: one-day mean QLIKE and paired comparisons.',panels)
    # Coverage, original/reconstructed outcomes, and distributional appendices.
    audit=pd.read_csv(base/'coverage/source_hf_comparison.csv',parse_dates=['date']);audit['year']=audit.date.dt.year;audit['rate']=audit.nobs/audit.expected_hours;audit['count_ratio']=audit.rate/audit.groupby(['market','year']).rate.transform('median');audit['span_flag']=audit.span_fraction<.95;audit['count_flag']=audit.count_ratio<.5
    rows=[]
    for m in MARKETS:
        d=audit[audit.market.eq(m)&audit.phase.eq('evaluation')];rows.append([LABELS[m],str(len(d)),str(int(d.span_flag.sum())),str(int(d.count_flag.sum())),str(int((d.span_flag|d.count_flag).sum()))])
    panels=[('Evaluation-period screening counts',['Market','Source dates','Span flags','Count flags','Either flag'],rows)]
    d=pd.read_csv(base/'reconstruction/coverage_flags.csv');rows=[]
    for m in ['FTSE100','DAX']:
        for r in d[d.market.eq(m)].sort_values('date').itertuples():rows.append([LABELS[m],r.date,f(100*r.span_fraction,1),str(int(r.nobs)),f(r.count_ratio,3),str(int(r.expected_bars))])
    panels.append(('All flagged evaluation-period dates',['Market','Date',r'Span (\%)','Observations','Count ratio','Five-minute bars'],rows));save('tab:coverage-screen','Common intraday-coverage screens and reconstructed sessions.',panels)
    panels=[]
    for m,period,title,hs,losses in [('FTSE100','full','FTSE 100, full evaluation period',(1,5,10),['qlike']),('FTSE100','2020','FTSE 100, 2020',(1,5,10),['qlike']),('DAX','full','DAX information comparison, full evaluation period',(1,5,10),['qlike'])]:
        pairs=[('Information','information',loss,h) for loss in losses for h in hs]
        if m=='FTSE100' and period=='full':
            pairs += [('Neural form','neural_form','qlike',1), *[('Combination','combination','qlike',h) for h in (1,5,10)]]
        pairs += [('Information','information','return_log_score',1)]
        if m=='FTSE100' and period=='full':pairs += [('Combination','combination','return_log_score',1)]
        rows=[]
        for title_,role,loss,h in pairs:
            old=comparison(role,loss,h,m,period);new=comparison(role,loss,h,m,period,'reconstructed');rows.append([title_,'QLIKE' if loss=='qlike' else 'Neg. log score',str(h),str(int(old.n)),diff(old),diff(new)])
        panels.append((title,['Comparison','Loss',r'$\ell$','N','Original','Reconstructed'],rows))
    save('tab:ftse-outcome-sensitivity','Matched comparisons under original and reconstructed evaluation outcomes.',panels)
    rows=[]
    for model,hs in [(NN,(1,5,10)),(RV,(1,5,10)),(LIN,(1,)),(G,(1,)),(T,(1,))]:
        for h in hs:rows.append([NAMES.get(model,model),rf'$\ell={h}$',*[f(one(cal,model_id=model,market=m,horizon=h).mean_return_crps,6) for m in MARKETS]])
    panels=[('Mean CRPS',['Model or comparison','Horizon or score',*[LABELS[m] for m in MARKETS]],rows)];rows=[]
    for loss,name,n in [('qlike','QLIKE',4),('return_log_score','Negative log score',4),('return_crps','CRPS',7)]:rows.append([name,r'$\ell=1$',*[diff(one(rg,market=m,model_a=T,model_b=G,loss=loss),n=n) for m in MARKETS]])
    panels.append(('RG (t) minus RG (G)',['Model or comparison','Horizon or score',*[LABELS[m] for m in MARKETS]],rows));save('tab:distributional-accuracy','Selected distributional-accuracy comparisons.',panels)
    rows=[]
    for m in MARKETS:
        for model in (NN,RV,G):
            for h in (5,10):
                r=one(cal,market=m,model_id=model,horizon=h);rows.append([LABELS[m],NAMES.get(model,model),str(h),str(int(r.observations)),f(r.pit_mean,3),f(r.pit_variance,3),f(r.coverage_90,3)+' ('+f(r.coverage_90_mean_width,4)+')'])
    save('tab:multistep-calibration','Descriptive multi-step calibration summaries.',[('', ['Market','Model',r'$\ell$','N','PIT mean','PIT variance',r'90\% coverage (width)'],rows)])
    rows=[]
    for m in MARKETS:
        for model in (RV,G,T):
            r=one(tail,market=m,model_id=model)
            for level in ('01','05'):rows.append([LABELS[m],NAMES.get(model,model),str(int(level))+r'\%',f(r[f'mean_var_{level}_quantile_loss']*1e4,3),f(100*r[f'var_{level}_exceedance_rate'],2),*[pval(r[f'var_{level}_{x}_p_value']) for x in ('kupiec','independence','conditional_coverage')]])
    save('tab:var-backtests','VaR quantile loss and exceedance backtests.',[('', ['Market','Model','Tail',r'QL $\times10^4$',r'Exc. (\%)',r'$p_{UC}$',r'$p_{IND}$',r'$p_{CC}$'],rows)])
    panels=[]
    for period in ['2018-2019','2020','2021-2022Feb']:
        rows=[]
        for name,role,loss,h in contrasts:
            label_ = {'information':RV+' $-$ '+NN,'neural_form':RV+' $-$ '+LIN,'combination':('QLIKE combination' if loss=='qlike' else 'Log-score combination')+' $-$ '+RV}[role]
            rows.append([label_,'QLIKE' if loss=='qlike' else 'Negative log score',str(h),*[diff(comparison(role,loss,h,m,period))+'; '+str(int(comparison(role,loss,h,m,period).n)) for m in MARKETS]])
        panels.append((period,['Comparison','Loss',r'$\ell$',*[LABELS[m] for m in MARKETS]],rows))
    save('tab:subsample-comparisons','Matched forecast comparisons by subperiod.',panels)
    rows=[]
    for loss,hs,title,scale in [('qlike',(1,5,10),'QLIKE',1),('return_log_score',(1,),'Negative log score',1),('return_crps',(1,5,10),r'CRPS $\times10^4$',1e4)]:
        for h in hs:rows.append([title,str(h),*[diff(comparison('information',loss,h,m,'2020'),scale=scale) for m in MARKETS]])
    save('tab:subsample-scores','Mean loss differences in the 2020 realised-information comparison.',[('', ['Loss','Horizon',*[LABELS[m] for m in MARKETS]],rows)])
    top=pd.read_csv(base/'comparisons/largest_origins.csv',parse_dates=['origin_date','mature_date']);top=top[top.horizon.eq(1)&top['rank'].le(5)].sort_values('origin_date');dates=top.origin_date.tolist()
    rows=[]
    for model in [NN,RV,'SV',LIN,*MODELS[:4],'NN-ENSEMBLE-EQUAL','NN-ENSEMBLE-QLIKE']:
        rows.append([{G:'Gaussian Realized GARCH','NN-ENSEMBLE-EQUAL':'Equal-weight combination','NN-ENSEMBLE-QLIKE':'QLIKE combination'}.get(model,model),*[f(one(scores,market='SP500',model_id=model,horizon=1,origin_date=day).variance_forecast*1e4) for day in dates]])
    rows.append(['Realised CC variance',*[f(one(scores,market='SP500',model_id=NN,horizon=1,origin_date=day).realized_variance*1e4) for day in dates]])
    save('tab:sp500-five-dates','One-day variance forecasts on the five largest adverse S\&P 500 origins in 2020.',[('', ['Model',*[day.strftime('%-d %b') for day in dates]],rows)])
    decomp=pd.read_csv(base/'decomposition/five_dates_decomposition.csv');rows=[]
    for day in dates:
        for model in (NN,RV,LIN):
            r=one(decomp,origin_date=str(day.date()),model_id=model);rows.append([day.strftime('%-d %b'),model,f(r.baseline_moment*1e4),f(r.correction_multiplier),f(r.forecast*1e4)])
    save('tab:sp500-baseline-decomposition','Baseline and correction components of the five S\&P 500 forecasts.',[('', ['Origin','Model',r'$B\times10^4$','$A$',r'$\widehat V_{t,1}\times10^4$'],rows)])
    mcsall=json.loads((base/'comparisons/mcs.json').read_text());expanded={r['market']:r for r in mcsall if r['variant']=='original' and r['universe']=='expanded_eleven'};rows=[]
    for model in [*MODELS,MS,ML]:
        vals=[]
        for m in MARKETS:
            r=expanded[m];means_=pd.Series(r['means']);rank=int(means_.rank(method='min')[model]);v=f(means_[model])+f' [{rank}]';v+=r'\textsuperscript{*}' if model in r['retained'] else '';vals.append(v)
        rows.append([{G:'Gaussian Realized GARCH','NN-ENSEMBLE-EQUAL':'Equal-weight combination','NN-ENSEMBLE-QLIKE':'QLIKE combination'}.get(model,NAMES.get(model,model)),*vals])
    rows.append(['Common sample size',*[str(expanded[m]['n']) for m in MARKETS]])
    save('tab:oa-expanded-ms','Expanded eleven-model comparison: one-day mean QLIKE, ranks, and 90\% model confidence sets.',[('', ['Model',*[LABELS[m] for m in MARKETS]],rows)])
    oc=select_evaluation_sample(pd.read_parquet(base/'har_oc/losses.parquet'));oc=oc[oc.model_id.eq('HAR-RV-OC-input')].copy()
    for c in ('origin_date','mature_date'):oc[c]=pd.to_datetime(oc[c])
    if 'qlike' not in oc:oc['qlike']=oc['qlike__rv_cc']
    mixed=pd.concat([scores,oc],ignore_index=True);rows=[]
    for m in MARKETS:
        names=[MS,ML,RV,'HAR-RV','HAR-RV-OC-input'];d=mixed[mixed.market.eq(m)&mixed.horizon.eq(1)&mixed.model_id.isin(names)].pivot(index=['origin_date','mature_date'],columns='model_id',values='qlike')[names].dropna().sort_index()
        for model,label in [('HAR-RV','CC'),('HAR-RV-OC-input','OC')]:
            dm_=diebold_mariano(d[ML].to_numpy(),d[model].to_numpy(),horizon=1);rows.append([LABELS[m],label,str(len(d)),f(d[ML].mean()),f(d[model].mean()),f(dm_['mean_loss_difference'],4,True),pval(dm_['p_value'],4)])
    save('tab:oa-ms-har','Descriptive one-day comparisons of RV-LIN-SV-MS with HAR benchmarks.',[('', ['Market','HAR inputs','N','LIN-MS QLIKE','HAR QLIKE','Difference','DM p'],rows)])
    rows=[]
    for h in (1,5,10):rows.append(['Mean QLIKE',rf'$\ell={h}$',*[diff(one(dm,market=m,horizon=h,model_a=RA,model_b=RV,loss_column='qlike')) for m in MARKETS]])
    panels=[('RV-RA-NN-SV minus RV-NN-SV',['Comparison','Horizon',*[LABELS[m] for m in MARKETS]],rows)];gd=pd.read_csv(base/'garch_t/comparisons.csv');rows=[]
    for loss,hs,title in [('qlike',(1,5,10),'Mean QLIKE'),('return_log_score',(1,),'Mean negative log score')]:
        for h in hs:rows.append([title,rf'$\ell={h}$',*[diff(one(gd,market=m,horizon=h,loss=loss)) for m in MARKETS]])
    panels.append(('NN-SV minus GARCH(1,1)-t',['Comparison','Horizon',*[LABELS[m] for m in MARKETS]],rows));save('tab:secondary-comparisons','Residual-asymmetry and return-only benchmark comparisons.',panels)
    estimates=pd.read_csv(base/'rv_ra/estimates.csv');rows=[]
    labels=['Asymmetry slope','Log-RV slope','Interaction slope','High-asymmetry contrast','High-log-RV contrast','Predicted $-$ realised asymmetry','Predicted $-$ no-asymmetry term','Realised $-$ no-asymmetry term','Equal weight: three $-$ two models']
    for title,r in zip(labels,estimates.itertuples()):
        if np.isfinite(r.estimate):
            rows.append([title,f(r.estimate,5,True),'['+f(r.ci_lower,5,True)+', '+f(r.ci_upper,5,True)+']',str(r.origin_count)])
        elif title=='High-log-RV contrast':
            rows.append([title,'not estimable','',str(int(estimates.origin_count.max()))])
        else:
            raise ValueError(f'unexpected unavailable conditional estimate: {title}')
    save('tab:rv-ra-conditional','Conditional and counterfactual realised-asymmetry comparisons, S\&P 500, 2017.',[('', ['Estimand','Estimate',r'95\% interval','Origins'],rows)])
    if len(tables)!=21:raise ValueError('expected 21 manuscript tables')
    pd.DataFrame(tables).to_csv(a.output/'index.csv',index=False)
    print('Generated all 21 manuscript tables as CSV, JSON and LaTeX.')


if __name__=='__main__':main()
