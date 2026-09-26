"""Render the archived GR statistic and reference bands; no statistical rerun."""
from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.dates as mdates
ROOT=Path(__file__).resolve().parents[2]
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--series',type=Path,default=ROOT/'reproduced/comparisons/fluctuation_series.csv')
parser.add_argument('--summary',type=Path,default=ROOT/'reproduced/comparisons/fluctuation_summary.csv')
parser.add_argument('--output',type=Path,default=ROOT/'reproduced/figures')
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(args.series,parse_dates=['window_end_origin']);summary=pd.read_csv(args.summary)
labels={'SP500':'S&P 500','FTSE100':'FTSE 100','DAX':'DAX'}
fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True)
for ax,m in zip(axes,labels):
 g=d[d.market==m].sort_values('window_end_origin');r=summary[summary.market==m].iloc[0];cv=float(r.critical_95_mc)
 assert len(g)==int(r.n-r.window+1)
 np.testing.assert_allclose(g.critical95,cv,rtol=1e-12)
 np.testing.assert_allclose(abs(g.statistic).max(),r.max_abs_stat,rtol=1e-12)
 ax.plot(g.window_end_origin,g.statistic,color='#226d85',lw=1.5)
 ax.axhline(cv,color='#9a3737',ls='--',lw=1);ax.axhline(-cv,color='#9a3737',ls='--',lw=1);ax.axhline(0,color='gray',lw=.6)
 ax.axvspan(pd.Timestamp('2020-01-01'),pd.Timestamp('2021-01-01'),color='#f4eadb',zorder=0)
 ax.set_title(labels[m],loc='left',fontsize=10)
 ax.set_ylabel('Rolling statistic');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18)
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6));axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'));axes[-1].set_xlabel('Rolling-window endpoint (forecast origin)')
fig.tight_layout()
for ext in ['pdf','png']:fig.savefig(args.output/f'information_fluctuation.{ext}',dpi=200,bbox_inches='tight')
plt.close(fig)
print('Rendered archived series without recalculating statistics or reference bands.')
