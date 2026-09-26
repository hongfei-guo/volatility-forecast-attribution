#!/usr/bin/env python3
"""Reconstruct the paper's tables and figures from retained components and lawful data."""
from pathlib import Path
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'code/scripts'
MARKETS = ('SP500', 'FTSE100', 'DAX')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--scoring-components', type=Path)
    p.add_argument('--conditional-components', type=Path)
    p.add_argument('--records', type=Path, default=ROOT/'inputs/evaluation')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=('all','inputs','scores','summaries','exhibits'), default='all',
                   help='separate stages allow scoring and table construction to be run independently')
    p.add_argument('--workers', type=int, choices=(1,2), default=2)
    a = p.parse_args()
    a.output = a.output.resolve()
    if bool(a.scoring_components) != bool(a.conditional_components):
        p.error('supply both component collections for the draw-based route')
    use_records = a.scoring_components is None
    for attr in ('data_root','scoring_components','conditional_components','records'):
        if getattr(a,attr) is not None:setattr(a,attr,getattr(a,attr).resolve())
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    logs=out/'logs';logs.mkdir(exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(ROOT/'code'),PYTHONDONTWRITEBYTECODE='1',
             OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',
             MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MPLCONFIGDIR=str(out/'plot_cache'))
    def run(script,*args,tag=None):
        command=[sys.executable,str(SCRIPTS/script),*map(str,args)]
        with (logs/((tag or script)+'.log')).open('w') as handle:
            process=subprocess.run(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
        if process.returncode:
            raise RuntimeError(f'{script} failed; see {logs/((tag or script)+".log")}')
        print(tag or script,flush=True)
    def stage(name):return a.stage in ('all',name)
    if stage('inputs'):
        run('build_daily_data.py','--source',a.data_root/'raw/external/omi/oxfordmanrealizedvolatilityindices.csv','--output',out/'daily.parquet','--report',out/'daily_report.json')
        run('audit_calendars.py','--data',out/'daily.parquet','--output',out/'calendar_report.json')
        run('audit_source_coverage.py','--data-root',a.data_root,'--output',out/'coverage')
        run('reconstruct_evaluation_outcomes.py','--daily',out/'daily.parquet','--audit',out/'coverage/source_hf_comparison.csv','--output',out/'reconstruction')
        if not use_records:
            run('prepare_evaluation_forecasts.py','--component-index',a.scoring_components/'forecasts.parquet','--point-panels',*[ROOT/'forecasts'/f'{m}.parquet' for m in MARKETS],'--output',out/'evaluation_forecasts.parquet')
            forecasts=pd.read_parquet(out/'evaluation_forecasts.parquet')
            (out/'forecast_panels').mkdir(exist_ok=True)
            for m in MARKETS:forecasts[forecasts.market.eq(m)].to_parquet(out/'forecast_panels'/f'{m}.parquet',index=False)
    if stage('scores'):
        if use_records:
            run('build_record_losses.py','--records',a.records,'--daily',out/'daily.parquet',
                '--reconstructed-daily',out/'reconstruction/daily_reconstructed.parquet',
                '--calendar-report',out/'calendar_report.json','--output',out/'rescoring')
            losses=pd.read_parquet(out/'rescoring/original_scores.parquet')
            losses.to_parquet(out/'scores.parquet',index=False)
            (out/'losses').mkdir(exist_ok=True)
            for m in MARKETS:losses[losses.market.eq(m)].to_parquet(out/'losses'/f'{m}.parquet',index=False)
        else:
            (out/'losses').mkdir(exist_ok=True)
            def market(m):
                run('build_predictive_losses.py','--data',out/'daily.parquet','--forecasts',out/'forecast_panels'/f'{m}.parquet','--forecast-root',a.scoring_components,'--calendar-audit',out/'calendar_report.json','--output',out/'losses'/f'{m}.parquet',tag=f'score_{m}')
            with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
                list(pool.map(market,MARKETS))
            losses=pd.concat([pd.read_parquet(out/'losses'/f'{m}.parquet') for m in MARKETS],ignore_index=True)
            losses.to_parquet(out/'scores.parquet',index=False)
            run('rescore_reconstructed_outcomes.py','--daily',out/'daily.parquet','--reconstructed-daily',out/'reconstruction/daily_reconstructed.parquet','--scores',out/'scores.parquet','--components',a.scoring_components,'--output',out/'rescoring')
    if stage('summaries'):
        run('summarize_evaluation.py','--losses',*[out/'losses'/f'{m}.parquet' for m in MARKETS],'--config',ROOT/'design/analysis.yaml','--scope','full','--output',out/'evaluation')
        run('evaluate_comparisons.py','--original',out/'scores.parquet','--reconstructed',out/'rescoring/reconstructed_scores.parquet','--output',out/'comparisons','--mcs','--fluctuation')
        models=set(yaml.safe_load((ROOT/'design/tail_evaluation.yaml').read_text())['probabilistic_models'])
        (out/'tail_panels').mkdir(exist_ok=True)
        (out/'benchmark_panels').mkdir(exist_ok=True)
        garch=[];rg=[]
        for m in MARKETS:
            d=pd.read_parquet(out/'losses'/f'{m}.parquet')
            d[d.horizon.eq(1)&d.model_id.isin(models)].to_parquet(out/'tail_panels'/f'{m}.parquet',index=False)
            for model,name in [('NN-SV','nn'),('GARCH-t','garch')]:
                d[d.model_id.eq(model)].to_parquet(out/'benchmark_panels'/f'{m}_{name}.parquet',index=False)
            run('compare_garch_t.py','--existing-losses',out/'benchmark_panels'/f'{m}_nn.parquet','--garch-losses',out/'benchmark_panels'/f'{m}_garch.parquet','--market',m,'--output',out/'garch_t'/f'{m}.json',tag=f'garch_{m}')
            for r in json.loads((out/'garch_t'/f'{m}.json').read_text())['comparisons']:garch.append(dict(r,market=m))
            run('compare_realized_garch_t.py','--existing-losses',out/'losses'/f'{m}.parquet','--student-t-losses',out/'losses'/f'{m}.parquet','--market',m,'--output-dir',out/'realized_garch_t'/m,tag=f'realized_garch_{m}')
        pd.DataFrame(garch).to_csv(out/'garch_t/comparisons.csv',index=False)
        for name in ['pairwise_dm.csv','model_metrics.csv']:
            pd.concat([pd.read_csv(out/'realized_garch_t'/m/name) for m in MARKETS]).to_csv(out/'realized_garch_t'/name,index=False)
        run('summarize_tail.py','--losses',*[out/'tail_panels'/f'{m}.parquet' for m in MARKETS],'--config',ROOT/'design/tail_evaluation.yaml','--output',out/'tail')
        run('run_har_oc_control.py','--data',out/'daily.parquet','--calendar-report',out/'calendar_report.json','--reference-dir',ROOT/'forecasts','--refit-schedule',ROOT/'design/refit_schedule.csv','--output-dir',out/'har_oc')
        run('summarize_identification_simulation.py','--records',ROOT/'inputs/identification','--output',out/'identification')
        run('summarize_correction_index.py','--records',ROOT/'inputs/identification/correction_index.csv','--output',out/'identification/correction_index_summary.csv')
        run('decompose_five_forecasts.py','--records',ROOT/'inputs/five_forecasts/records.csv','--components',ROOT/'inputs/five_forecasts','--output',out/'decomposition')
        c=a.records if use_records else a.conditional_components
        run('prepare_rv_ra_conditional_inputs.py','--data',out/'daily.parquet','--nn-forecasts',c/'nn_sv_forecasts.parquet','--rv-nn-forecasts',c/'rv_nn_sv_forecasts.parquet','--rv-ra-forecasts',c/'rv_ra_forecasts.parquet','--refit-schedule',ROOT/'design/rv_ra_refit_schedule_2017.csv','--output-dir',out/'rv_ra_inputs')
        if use_records:
            run('analyze_conditional_records.py','--records',a.records,'--observable-inputs',out/'rv_ra_inputs','--config',ROOT/'design/rv_ra_conditional_value.json','--output-root',out/'rv_ra')
        else:
            run('analyze_rv_ra_conditional_value.py','--config',ROOT/'design/rv_ra_conditional_value.json','--paired-losses',out/'rv_ra_inputs/paired_losses.csv','--nn-losses',out/'rv_ra_inputs/nn_sv_losses.csv','--rv-nn-losses',out/'rv_ra_inputs/rv_nn_sv_losses.csv','--gamma-summary',c/'gamma_summary.csv','--rv-ra-root',c/'rv_ra','--daily',out/'daily.parquet','--output-root',out/'rv_ra')
        run('combine_forecasts.py','--data',out/'daily.parquet','--forecasts',*[ROOT/'forecasts'/f'{m}.parquet' for m in MARKETS],'--losses',out/'scores.parquet','--warmup-forecasts',ROOT/'forecasts/combination_warmup.parquet','--calendar-report',out/'calendar_report.json','--modes','equal','qlike','--output',out/'combination_forecasts.parquet','--weights-output',out/'combination_weights.csv')
    if stage('exhibits'):
        run('build_paper_tables.py','--reproduced',out,'--output',out/'tables')
        run('render_combination_weights.py','--weights',out/'combination_weights.csv','--output',out/'figures')
        run('render_information_cumulative.py','--series',out/'comparisons/information_series.csv','--output',out/'figures')
        run('render_sp500_information_diagnostics.py','--scores',out/'scores.parquet','--output',out/'figures')
        run('render_information_fluctuation.py','--series',out/'comparisons/fluctuation_series.csv','--summary',out/'comparisons/fluctuation_summary.csv','--output',out/'figures')
    print(f'Completed {a.stage}: {out}',flush=True)


if __name__=='__main__':main()
