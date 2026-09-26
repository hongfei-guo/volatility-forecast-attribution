#!/usr/bin/env python3
"""Export unrounded evaluation records for the paper's fixed outcome definitions."""
from pathlib import Path
import argparse,json,shutil
import numpy as np,pandas as pd
from bnsv.evaluation_records import KEYS,SCORES,METADATA,FUNCTIONALS,ordered,target_identity
from analyze_rv_ra_conditional_value import run_analysis,read_json


def export(reproduced,conditional,output):
    original=ordered(pd.read_parquet(reproduced/'scores.parquet'))
    rebuilt=ordered(pd.read_parquet(reproduced/'rescoring/reconstructed_scores.parquet'))
    pd.testing.assert_frame_equal(original[KEYS],rebuilt[KEYS],check_exact=True)
    output.mkdir(parents=True,exist_ok=False)
    original[METADATA+SCORES+FUNCTIONALS].to_parquet(output/'prediction_records.parquet',index=False,compression='zstd')
    changed=np.zeros(len(original),dtype=bool)
    for c in SCORES:changed |= ~(original[c].eq(rebuilt[c]) | (original[c].isna()&rebuilt[c].isna())).to_numpy()
    rebuilt.loc[changed,KEYS+SCORES].to_parquet(output/'reconstructed_scores.parquet',index=False,compression='zstd')
    identities={'original':target_identity(original),'reconstructed':target_identity(rebuilt)}
    (output/'outcome_identity.json').write_text(json.dumps(identities,indent=2)+'\n')
    for name in ['nn_sv_forecasts.parquet','rv_nn_sv_forecasts.parquet','rv_ra_forecasts.parquet']:
        shutil.copy2(conditional/name,output/name)
    gammas={}
    for p in sorted((conditional/'rv_ra').glob('*/refits/*.npz')):
        with np.load(p,allow_pickle=False) as d:gammas[p.stem.rsplit('_',1)[-1]]=d['posterior_gamma_A'].copy()
    np.savez_compressed(output/'gamma_draws.npz',**gammas)
    config=read_json(Path(__file__).resolve().parents[2]/'design/rv_ra_conditional_value.json')
    r=run_analysis(config=config,paired_losses=reproduced/'rv_ra_inputs/paired_losses.csv',
        nn_losses=reproduced/'rv_ra_inputs/nn_sv_losses.csv',rv_nn_losses=reproduced/'rv_ra_inputs/rv_nn_sv_losses.csv',
        gamma_summary_path=conditional/'gamma_summary.csv',rv_ra_root=conditional/'rv_ra',daily_path=reproduced/'daily.parquet')
    r['counterfactual_variances'].to_parquet(output/'counterfactual_variances.parquet',index=False,compression='zstd')
    print(json.dumps(dict(forecasts=len(original),changed_distribution_scores=int(changed.sum()),
                         counterfactual_rows=len(r['counterfactual_variances']),bytes=sum(p.stat().st_size for p in output.iterdir())),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reproduced',type=Path,required=True);p.add_argument('--conditional-components',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();export(a.reproduced,a.conditional_components,a.output)
