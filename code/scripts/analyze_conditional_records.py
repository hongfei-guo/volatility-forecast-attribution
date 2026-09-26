#!/usr/bin/env python3
"""Recompute RV-RA conditional comparisons from retained forecast means and daily inputs."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from analyze_rv_ra_conditional_value import (read_json,validate_config,load_paired_losses,
    load_member_losses,loss_matrices,qlike,summarize_conditional,write_outputs)
from export_rv_ra_components import gamma_row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['records','observable-inputs','config','output-root']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();config=read_json(a.config);validate_config(config)
    paired,origins=load_paired_losses(a.observable_inputs/'paired_losses.csv')
    rv=load_member_losses(a.observable_inputs/'rv_nn_sv_losses.csv','RV-NN-SV',origins)
    nn=load_member_losses(a.observable_inputs/'nn_sv_losses.csv','NN-SV',origins)
    ra=load_member_losses(a.observable_inputs/'rv_ra_nn_sv_losses.csv','RV-RA-NN-SV',origins)
    states=pd.read_csv(a.observable_inputs/'origin_states.csv').set_index('origin_date').loc[origins]
    means=pd.read_parquet(a.records/'counterfactual_variances.parquet').set_index(['origin_date','horizon'])
    expected=pd.MultiIndex.from_product([origins,[5,10]],names=['origin_date','horizon'])
    if means.index.has_duplicates or set(means.index)!=set(expected):raise ValueError('counterfactual mean grid differs')
    means=means.loc[expected]
    modes=['predicted_asymmetry','realized_asymmetry','no_asymmetry_term']
    if not np.isfinite(means[modes]).all().all() or not (means[modes]>0).all().all():raise ValueError('counterfactual variances must be positive and finite')
    comparison={mode:np.empty((241,2)) for mode in modes};two=np.empty((241,3));three=two.copy()
    for i,origin in enumerate(origins):
        for j,h in enumerate([1,5,10]):
            x=rv.loc[(origin,h)];n=nn.loc[(origin,h)];r=ra.loc[(origin,h)]
            for member in [n,r]:
                if json.loads(member.target_dates_json)!=json.loads(x.target_dates_json):raise ValueError('conditional targets differ')
                np.testing.assert_allclose(member.realized_variance,x.realized_variance,rtol=0,atol=0)
            two[i,j]=qlike(x.realized_variance,.5*(n.variance_forecast+x.variance_forecast))
            three[i,j]=qlike(x.realized_variance,(n.variance_forecast+x.variance_forecast+r.variance_forecast)/3)
        for j,h in enumerate([5,10]):
            for mode in modes:comparison[mode][i,j]=qlike(rv.loc[(origin,h),'realized_variance'],means.loc[(origin,h),mode])
    with np.load(a.records/'gamma_draws.npz',allow_pickle=False) as arrays:
        rows=[gamma_row(arrays[date],i+1,pd.Timestamp(date).strftime('%Y-%m-%d')) for i,date in enumerate(sorted(arrays.files))]
    gamma=pd.DataFrame(rows).set_index('origin').sort_index()
    if len(gamma)!=12:raise ValueError('expected twelve coefficient posteriors')
    rv_array,ra_array=loss_matrices(origins=origins,paired=paired)
    result=summarize_conditional(config,states,rv_array,ra_array,comparison,two,three,gamma)
    write_outputs(result,config,a.output_root)
    print('Conditional tables and bootstrap intervals reconstructed from unrounded records.')


if __name__=='__main__':main()
