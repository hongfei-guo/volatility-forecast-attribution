#!/usr/bin/env python3
"""Recalculate outcome-based losses and attach exact stored distribution scores."""
from pathlib import Path
import argparse,json,subprocess,sys
import pandas as pd
from bnsv.evaluation_records import attach_functionals


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['records','daily','reconstructed-daily','calendar-report','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    records=pd.read_parquet(a.records/'prediction_records.parquet');identity=json.loads((a.records/'outcome_identity.json').read_text());patch=pd.read_parquet(a.records/'reconstructed_scores.parquet')
    temp=a.output/'variance_losses.parquet'
    subprocess.run([sys.executable,str(Path(__file__).with_name('build_variance_losses.py')),
        '--data',str(a.daily),'--forecasts',str(a.records/'prediction_records.parquet'),
        '--calendar-report',str(a.calendar_report),'--output',str(temp)],check=True)
    variance_losses=pd.read_parquet(temp)
    for variant,path in [('original',a.daily),('reconstructed',a.reconstructed_daily)]:
        daily=pd.read_parquet(path);daily.date=pd.to_datetime(daily.date)
        scores=attach_functionals(records,variance_losses,daily,identity[variant],patch if variant=='reconstructed' else None)
        scores['evaluation_variant']=variant
        scores.to_parquet(a.output/f'{variant}_scores.parquet',index=False,compression='zstd')

        print(variant,len(scores),'records evaluated',flush=True)

    temp.unlink()

if __name__=='__main__':main()
