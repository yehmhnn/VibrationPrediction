"""Read-only analysis of exactly nine authorized MAT recordings. Outputs stay here."""
import os
os.environ['MPLCONFIGDIR']='<TEMP_DIR>/studyII_mpl'
import csv,json,hashlib,platform
from pathlib import Path
import numpy as np, scipy, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import welch,find_peaks
from scipy.spatial.distance import jensenshannon
O=Path(__file__).resolve().parent; ROOT=O.parents[1]
E=ROOT/'VibrationPrediction_Report_TwoStudies_20261016'
manifest=E/'evidence/datasets/processed/cwru_multi_record_B/manifest.csv'
method=E/'data/spectral_method.json'; normpath=E/'evidence/datasets/processed/cwru_multi_record_B/normalization.json'
rows=[r for r in csv.DictReader(manifest.open()) if r['split'] in ('train','validation')]
assert {r['record_id'] for r in rows}=={'105_0','106_1','107_2','118_0','119_1','120_2','130@6_0','131@6_1','132@6_2'}
assert all(int(r['load_id'])<3 for r in rows)
params=json.loads(method.read_text())['welch']; norm=json.loads(normpath.read_text()); std=norm['train_std'];mu=norm['train_mean']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def savecsv(name,data):
 with (O/name).open('w',newline='') as h:
  w=csv.DictWriter(h,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
provenance={str(p.relative_to(ROOT)):sha(p) for p in [manifest,method,normpath]}
S={}; summary=[];peaks=[];bands=[];spectra=[]; arr={}
edges=np.arange(0,6001,1000)
for r in rows:
 rid=r['record_id'];p=O/'inputs'/Path(r['source_path']).name
 assert sha(p)==r['source_sha256'],rid
 provenance[str(p.relative_to(ROOT))]=sha(p)
 mat=loadmat(p);x=np.asarray(mat[r['source_key']],dtype=np.float64).ravel();assert len(x)==int(r['num_samples']) and np.isfinite(x).all()
 assert int(np.asarray(mat[next(k for k in mat if k.endswith('RPM'))]).ravel()[0])==int(r['rpm'])
 f,P=welch(x,fs=12000,**params);df=f[1]-f[0];power=P.sum()*df;shape=P/power
 S[rid]=(P,shape);arr[rid]=x
 pk,_=find_peaks(P,distance=int(np.ceil(50/df)));pk=sorted(pk,key=lambda i:P[i],reverse=True)[:5]
 for rank,i in enumerate(pk,1):peaks.append(dict(record_id=rid,rank=rank,frequency_hz=f[i],psd_original_units2_per_hz=P[i]))
 s=dict(record_id=rid,fault_type=r['fault_type'],split=r['split'],load_hp=int(r['load_id']),rpm=int(r['rpm']),samples=len(x),duration_s=len(x)/12000,welch_segments=(len(x)-8192)//4096+1,mean_original_units=x.mean(),rms_original_units=np.sqrt(np.mean(x*x)),ac_rms_original_units=x.std(),time_mean_square_original_units2=np.mean(x*x),welch_power_original_units2=power,welch_vs_variance_relative_difference=power/x.var()-1,welch_power_model_units2=power/std**2,dominant_peak_hz=f[pk[0]],centroid_hz=np.sum(f*P)/P.sum())
 summary.append(s)
 for lo,hi in zip(edges[:-1],edges[1:]):
  mask=(f>=lo)&((f<hi) if hi<6000 else (f<=hi));bp=P[mask].sum()*df
  bands.append(dict(record_id=rid,band_low_hz=lo,band_high_hz=hi,power_original_units2=bp,fraction_total=bp/power))
 for i,hz in enumerate(f):spectra.append(dict(record_id=rid,frequency_hz=hz,psd_original_units2_per_hz=P[i],psd_model_units2_per_hz=P[i]/std**2,unit_area_psd_per_hz=shape[i]))
 # Exact target grid used in the existing validation metric, no model inference.
 if r['split']=='validation':
  y=(x-mu)/std;t=np.concatenate([y[i+512:i+608] for i in range(0,len(y)-608+1,96)])
  s['zero_forecast_rmse_model_units']=np.sqrt(np.mean(t*t))
for s in summary:s.setdefault('zero_forecast_rmse_model_units','')
savecsv('record_summary.csv',summary);savecsv('dominant_peaks.csv',peaks);savecsv('band_power.csv',bands);savecsv('welch_spectra.csv',spectra)
faults=[('inner_race_fault','Inner race'),('ball_fault','Ball'),('outer_race_fault','Outer race (@6)')];colors=['#0072B2','#009E73','#D55E00'];styles=['-','--','-']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
def finish(fig,name):
 fig.savefig(O/(name+'.png'),bbox_inches='tight');fig.savefig(O/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
allP=np.concatenate([v[0][1:] for v in S.values()]);allQ=np.concatenate([v[1][1:] for v in S.values()])
limits=[(max(allP.min()*.7,1e-14),allP.max()*2),(max(allQ.min()*.7,1e-14),allQ.max()*2)]
for fault,title in faults:
 fig,axes=plt.subplots(2,1,figsize=(11,7.5),sharex=True,layout='constrained')
 for r in [r for r in rows if r['fault_type']==fault]:
  load=int(r['load_id']);P,Q=S[r['record_id']];label=f"{r['record_id']} | {load} hp | {r['split']} | {r['rpm']} RPM"
  for ax,y in zip(axes,[P,Q]):ax.semilogy(f[1:],y[1:],color=colors[load],ls=styles[load],lw=.8,label=label)
 for j,ax in enumerate(axes):ax.set(xlim=(0,6000),ylim=limits[j],ylabel=['PSD (source signal units²/Hz)','Unit-area PSD (1/Hz)'][j]);ax.grid(alpha=.2)
 axes[0].set_title(title+' — absolute spectral power');axes[1].set_title('Spectral shape — each PSD divided by its integrated power');axes[0].legend(fontsize=9);axes[1].set_xlabel('Frequency (Hz)')
 fig.suptitle('Study II · 0/1 hp training versus 2 hp validation · no smoothing',fontsize=13)
 finish(fig,f'psd_{fault}')
fig,axes=plt.subplots(3,2,figsize=(13,10),sharex=True,layout='constrained')
for i,(fault,title) in enumerate(faults):
 for r in [r for r in rows if r['fault_type']==fault]:
  load=int(r['load_id'])
  for j,y in enumerate(S[r['record_id']]):axes[i,j].semilogy(f[1:],y[1:],color=colors[load],ls=styles[load],lw=.7,label=f'{load} hp '+r['split'])
 for j in range(2):axes[i,j].set(xlim=(0,6000),ylim=limits[j],title=title+(' — absolute power' if j==0 else ' — unit-area shape'),ylabel='source units²/Hz' if j==0 else '1/Hz');axes[i,j].grid(alpha=.18)
axes[0,0].legend(fontsize=9);axes[-1,0].set_xlabel('Frequency (Hz)');axes[-1,1].set_xlabel('Frequency (Hz)');fig.suptitle('Study II spectral overview · common scales within each column',fontsize=15)
finish(fig,'spectral_overview')
comparisons=[]
for fault,title in faults:
 rr=[r for r in rows if r['fault_type']==fault];a,b,v=[r['record_id'] for r in rr];P0,Q0=S[a];P1,Q1=S[b];Pv,Qv=S[v];ref=(P0+P1)/2;qr=ref/(ref.sum()*df)
 for train in [a,b,'equal_record_training_reference']:
  pt,qt=S[train] if train in S else (ref,qr)
  mask=(f>0)&(pt>=pt.max()*1e-8)&(Pv>0)
  db=np.full_like(f,np.nan);db[mask]=10*np.log10(Pv[mask]/pt[mask])
  mass0=qt*df;mass1=Qv*df
  comparisons.append(dict(fault_type=fault,validation_record=v,training_reference=train,power_ratio=Pv.sum()/pt.sum(),shape_total_variation=.5*np.abs(mass0-mass1).sum(),shape_jensen_shannon_distance_base2=jensenshannon(mass0,mass1,base=2),masked_bins=int((~mask).sum()),fraction_valid_bins_above_6db=np.mean(db[mask]>6),fraction_valid_bins_below_minus6db=np.mean(db[mask]<-6)))
 savecsv(f'validation_vs_training_{fault}.csv',[dict(frequency_hz=f[i],validation_to_equal_train_psd_db=(10*np.log10(Pv[i]/ref[i]) if f[i]>0 and ref[i]>=ref.max()*1e-8 else ''),validation_to_equal_train_shape_db=(10*np.log10(Qv[i]/qr[i]) if f[i]>0 and ref[i]>=ref.max()*1e-8 else '')) for i in range(len(f))])
savecsv('spectral_comparisons.csv',comparisons)
rmse=[]
for model in ['tsflow','timesnet']:
 p=E/f'evidence/logs/convergence/multi_record_B_{model}_objective_v1/result.json';result=json.loads(p.read_text());provenance[str(p.relative_to(ROOT))]=sha(p)
 for rid,m in result['best_validation_record_metrics']['per_record'].items():
  z=next(s for s in summary if s['record_id']==rid)['zero_forecast_rmse_model_units']
  rmse.append(dict(model=model,checkpoint_epoch=result['best_epoch'],record_id=rid,selected_checkpoint_rmse_model_units=m['RMSE'],zero_forecast_rmse_model_units=z,rmse_divided_by_zero=m['RMSE']/z,num_windows=m['num_windows']))
savecsv('validation_rmse_context.csv',rmse)
checks=dict(authorized_record_count=len(rows),no_test_signal_opened=True,source_hashes_all_match=True,band_power_sum_matches_total=all(abs(sum(b['power_original_units2'] for b in bands if b['record_id']==s['record_id'])-s['welch_power_original_units2'])<1e-12 for s in summary),unit_area_integrals=[float(q.sum()*df) for p,q in S.values()],frequency_resolution_hz=df,welch=params,average='mean',return_onesided=True,smoothing='none',power_integration='sum of one-sided PSD bins * df, including DC and Nyquist; bands partition bins without overlap',units='original MAT numerical acceleration units; physical calibration not independently verified; model PSD = original PSD/train_std²',versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__),source_sha256=provenance)
(O/'provenance_and_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(summary,indent=2));print('COMPARISONS',json.dumps(comparisons,indent=2));print('RMSE',json.dumps(rmse,indent=2))
# Training-to-training reference is descriptive, not a null distribution.
traincmp=[]
for fault,title in faults:
 rr=[r['record_id'] for r in rows if r['fault_type']==fault and r['split']=='train'];a,b=rr
 p,q=S[a][1]*df,S[b][1]*df
 traincmp.append(dict(fault_type=fault,record_0hp=a,record_1hp=b,shape_total_variation=.5*np.abs(p-q).sum(),shape_jensen_shannon_distance_base2=jensenshannon(p,q,base=2)))
savecsv('training_shape_comparisons.csv',traincmp)
fig,axes=plt.subplots(1,3,figsize=(15,5.5),layout='constrained',gridspec_kw={'width_ratios':[1.8,1,1]})
M=np.array([[b['fraction_total']*100 for b in bands if b['record_id']==r['record_id']] for r in rows])
im=axes[0].imshow(M,cmap='Blues',vmin=0,vmax=100,aspect='auto')
labels=[r['record_id']+' | '+r['load_id']+' hp '+('VAL' if r['split']=='validation' else 'train') for r in rows]
axes[0].set(yticks=range(9),yticklabels=labels,xticks=range(6),xticklabels=['0–1','1–2','2–3','3–4','4–5','5–6'],xlabel='Frequency band (kHz)',title='Fraction of Welch power (%)')
for i in range(9):
 for j in range(6):axes[0].text(j,i,f'{M[i,j]:.1f}',ha='center',va='center',fontsize=9,color='white' if M[i,j]>55 else '#172533')
for ax,key,title in [(axes[1],'rms_original_units','RMS amplitude'),(axes[2],'welch_power_original_units2','Integrated Welch power')]:
 ax.barh(range(9),[s[key] for s in summary],color=[colors[int(r['load_id'])] for r in rows]);ax.set(yticks=range(9),yticklabels=[],ylim=(8.5,-.5),title=title,xlabel='Source signal units' if key.startswith('rms') else 'Source signal units²');ax.grid(axis='x',alpha=.2)
fig.suptitle('Study II · broad-band shape and absolute scale',fontsize=15);finish(fig,'band_power_and_amplitude')
# Track script and generated-output hashes separately from immutable input hashes.
checks['analysis_script_sha256']=sha(Path(__file__))
checks['training_shape_comparisons']=traincmp
checks['max_absolute_welch_vs_time_variance_relative_difference']=max(abs(s['welch_vs_variance_relative_difference']) for s in summary)
(O/'provenance_and_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
