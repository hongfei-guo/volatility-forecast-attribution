"""Render the original-outcome S&P 500 forecast diagnostics."""
from pathlib import Path
import re,json
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.dates as mdates
import argparse
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scores',type=Path,required=True);p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[2]/'reproduced/figures');a=p.parse_args()
REPO=Path(__file__).resolve().parents[2];FIG=a.output;FIG.mkdir(parents=True,exist_ok=True)
d=pd.read_parquet(a.scores);d=d[d.main_evaluation_eligible&(d.horizon==1)&(d.market=='SP500')]
for col in ['origin_date','mature_date']:d[col]=pd.to_datetime(d[col])
k=['origin_date','mature_date'];nn=d[d.model_id=='NN-SV'];rv=d[d.model_id=='RV-NN-SV']
pairs=nn[k+['variance_forecast','realized_variance','qlike']].merge(rv[k+['variance_forecast','realized_variance','qlike']],on=k,validate='one_to_one',suffixes=('_nn','_rv'))
np.testing.assert_array_equal(pairs.realized_variance_nn,pairs.realized_variance_rv)
pairs=pairs.rename(columns={'variance_forecast_nn':'f_NN','variance_forecast_rv':'f_RV','realized_variance_nn':'y'});pairs['loss_difference']=pairs.qlike_rv-pairs.qlike_nn;pairs['market']='SP500';pairs['horizon']=1
top=pairs[pairs.origin_date.dt.year==2020].sort_values(['loss_difference','origin_date'],ascending=[False,True]).head(5).copy();top['rank']=np.arange(1,len(top)+1)
g=pairs[(pairs.market=='SP500')&(pairs.horizon==1)&(pairs.origin_date.dt.year==2020)].sort_values('origin_date').copy();g['cumulative']=g.loss_difference.cumsum()
z=g[(g.mature_date>='2020-02-01')&(g.mature_date<'2020-05-01')];t=top[(top.horizon==1)&(top['rank']<=5)]
fig,axes=plt.subplots(2,1,figsize=(9,7.5),gridspec_kw={'height_ratios':[1,1]})
a=axes[0]
a.plot(z.mature_date,1e4*z.y,color='#202020',lw=1.7,label='Realised CC variance')
a.plot(z.mature_date,1e4*z.f_NN,color='#b56a21',lw=1.5,label='NN-SV')
a.plot(z.mature_date,1e4*z.f_RV,color='#226d85',lw=1.5,label='RV-NN-SV')
a.set_yscale('log');a.set_ylabel('Variance × $10^4$ (log scale)');a.set_xlabel('Target date');a.set_title('A',loc='left',fontsize=11)
a.legend(frameon=False,ncol=3,fontsize=9,loc='upper right');a.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2));a.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
for r in t.itertuples():
 a.scatter(r.mature_date,r.y*1e4,color='#9a3737',s=22,zorder=4);a.annotate(str(r.rank),(r.mature_date,r.y*1e4),xytext=(4,6),textcoords='offset points',fontsize=9,color='#9a3737')
a=axes[1];a.plot(g.origin_date,g.cumulative,color='#226d85',lw=1.7);a.axhline(0,color='#666',lw=.8);a.set_xlabel('Forecast origin');a.set_ylabel('Cumulative QLIKE difference');a.set_title('B',loc='left',fontsize=11)
offsets={1:(20,18),2:(-32,28),3:(-30,10),4:(-40,10),5:(-12,36)}
for r in t.itertuples():
 y=float(g.loc[g.origin_date==r.origin_date,'cumulative'].iloc[0]);a.scatter(r.origin_date,y,color='#9a3737',s=23,zorder=4);a.annotate(str(r.rank),(r.origin_date,y),xytext=offsets[r.rank],textcoords='offset points',arrowprops={'arrowstyle':'-','color':'#9a3737','lw':.6},fontsize=9,color='#9a3737')
a.set_xlim(pd.Timestamp('2020-01-01'),pd.Timestamp('2020-12-31'));a.xaxis.set_major_locator(mdates.MonthLocator(interval=2));a.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
a.text(.98,.94,f'Year-end net: {g.loss_difference.sum():.3f}',transform=a.transAxes,ha='right',va='top',fontsize=9)
for a in axes:
 a.spines[['top','right']].set_visible(False);a.grid(axis='y',alpha=.18)
fig.tight_layout();
for ext in ['png','pdf']:fig.savefig(FIG/f'sp500_2020_information_diagnostics.{ext}',dpi=220,bbox_inches='tight')
plt.close(fig)
