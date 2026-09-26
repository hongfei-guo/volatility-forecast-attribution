"""Render original archived information losses; presentation only."""
from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.dates as mdates
import argparse
REPO=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--series',type=Path,default=REPO/'reproduced/comparisons/information_series.csv')
parser.add_argument('--output',type=Path,default=REPO/'reproduced/figures')
args=parser.parse_args();ROOT=args.output;ROOT.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(args.series,parse_dates=['origin_date','mature_date'])
assert not d.duplicated(['market','origin_date']).any()
series=[]
fig,axes=plt.subplots(3,1,figsize=(9,7.1),sharex=True)
periods=[('2018-2019','2018-01-01','2020-01-01'),('2020','2020-01-01','2021-01-01'),('2021-Feb 2022','2021-01-01','2022-02-28')]
for ax,market in zip(axes,['SP500','FTSE100','DAX']):
 g=d[d.market==market].sort_values('origin_date').copy();g['cumulative_qlike_difference']=g.loss_difference.cumsum();series.append(g)
 ax.axvspan(pd.Timestamp('2020-01-01'),pd.Timestamp('2021-01-01'),color='#F3E8D9',zorder=0)
 ax.plot(g.origin_date,g.cumulative_qlike_difference,lw=1.5,color='#215E75')
 ax.axhline(0,color='#777777',lw=.7)
 for boundary in ['2020-01-01','2021-01-01']:ax.axvline(pd.Timestamp(boundary),color='#777777',ls='--',lw=.7)
 ax.set_title({'SP500':'S&P 500','FTSE100':'FTSE 100','DAX':'DAX'}[market],loc='left',fontweight='bold',fontsize=11)
 ax.set_ylabel('Cumulative difference',fontsize=9);ax.grid(axis='y',alpha=.18)
 ax.spines[['top','right']].set_visible(False)
 for label,start,end in periods:
  mid=pd.Timestamp(start)+(pd.Timestamp(end)-pd.Timestamp(start))/2
  ax.text(mid,.96,label,transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=8,bbox={'facecolor':'white','alpha':.75,'edgecolor':'none','pad':1})
axes[-1].xaxis.set_major_locator(mdates.YearLocator());axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[-1].set_xlabel('Forecast origin',fontsize=10)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(ROOT/f'information_cumulative_subsamples.{ext}',dpi=200,bbox_inches='tight')
plt.close(fig)
