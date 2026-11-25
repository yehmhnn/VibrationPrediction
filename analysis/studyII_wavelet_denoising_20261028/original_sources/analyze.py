import os
os.environ['MPLCONFIGDIR']='<TEMP_DIR>/studyII_mpl'
from pathlib import Path
import csv,json,inspect,hashlib,platform
import numpy as np,scipy,pywt,skimage,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import welch,find_peaks
from scipy.stats import kurtosis
from skimage.restoration import denoise_wavelet
import skimage.restoration._denoise as implementation
O=Path(__file__).resolve().parent;R=O.parents[1];OLD=R/'analysis/studyII_welch_20261028';E=R/'VibrationPrediction_Report_TwoStudies_20261016'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvout(name,rows):
 with (O/name).open('w',newline='') as h:
  w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
manifest=E/'evidence/datasets/processed/cwru_multi_record_B/manifest.csv'
rows=[r for r in csv.DictReader(manifest.open()) if r['split'] in ('train','validation')]
expected={'105_0','106_1','107_2','118_0','119_1','120_2','130@6_0','131@6_1','132@6_2'}
assert {r['record_id'] for r in rows}==expected and all(int(r['load_id'])<3 for r in rows)
params=dict(fs=12000,window='hann',nperseg=8192,noverlap=4096,nfft=8192,detrend='constant',scaling='density',average='mean',return_onesided=True)
settings=dict(sigma=None,wavelet='db4',wavelet_levels=3,convert2ycbcr=False,method='BayesShrink',rescale_sigma=True,channel_axis=None)
config=dict(versions={k:v for k,v in [('python',platform.python_version()),('numpy',np.__version__),('scipy',scipy.__version__),('PyWavelets',pywt.__version__),('scikit-image',skimage.__version__),('matplotlib',matplotlib.__version__)]},signature=str(inspect.signature(denoise_wavelet)),settings=settings,threshold_modes=['soft','hard'],boundary='symmetric, inherited PyWavelets default verified by inspection',welch=params,main_time_interval_seconds=[1,1.05],impulse_window='20 ms centered on original maximum absolute deviation from mean, clipped at endpoints',source_hashes={str(manifest.relative_to(R)):sha(manifest)},units='original MAT numerical signal units; physical calibration not independently verified',scope='offline full-record signal analysis; not leakage-safe forecasting preprocessing')
oldhash={str(p.relative_to(R)):sha(p) for p in OLD.rglob('*') if p.is_file()}
(O/'installed_api_source.txt').write_text(inspect.getsource(denoise_wavelet)+'\n'+inspect.getsource(implementation._wavelet_threshold)+'\n'+inspect.getsource(implementation._sigma_est_dwt)+'\n'+inspect.getsource(implementation._bayes_thresh))
assert "mode='soft'" in str(inspect.signature(denoise_wavelet))
summary=[];bands=[];peaks=[];thresholds=[];checks=[];allpsd=[]
plt.rcParams.update({'font.size':9,'savefig.dpi':160,'axes.spines.top':False,'axes.spines.right':False})
colors={'original':'#30343b','soft':'#D55E00','hard':'#0072B2'}
def rms(x):return float(np.sqrt(np.mean(x*x)))
def flat(p):return float(np.exp(np.mean(np.log(np.maximum(p,np.finfo(float).tiny))))/np.mean(p))
def save(fig,name):
 fig.savefig(O/(name+'.png'),bbox_inches='tight');fig.savefig(O/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
for row in rows:
 rid=row['record_id'];p=OLD/'inputs'/Path(row['source_path']).name;assert sha(p)==row['source_sha256'];config['source_hashes'][str(p.relative_to(R))]=sha(p)
 m=loadmat(p);x=np.asarray(m[row['source_key']],dtype=np.float64).ravel();assert len(x)==int(row['num_samples']) and int(row['sampling_rate_hz'])==12000 and np.isfinite(x).all()
 f,P=welch(x,**params);df=f[1]-f[0];power=P.sum()*df
 # Match original PSD against prior saved computation without rerunning it.
 prior=next(r for r in csv.DictReader((OLD/'record_summary.csv').open()) if r['record_id']==rid)
 assert np.isclose(power,float(prior['welch_power_original_units2']),rtol=1e-12)
 ids,_=find_peaks(P,distance=int(np.ceil(50/df)));ids=sorted(ids,key=lambda i:P[i],reverse=True)[:5]
 # Noise/threshold diagnostics reconstructed from the inspected implementation; not used to denoise.
 coeff=pywt.wavedecn(x,'db4',mode='symmetric',level=3);detail=coeff[-1]['d'];nz=detail[detail!=0];sigma=float(np.median(np.abs(nz))/scipy.stats.norm.ppf(.75))
 for j,c in enumerate(coeff[1:]):
  d=c['d'];v=np.mean(d*d);thr=sigma**2/np.sqrt(max(v-sigma**2,np.finfo(d.dtype).eps));thresholds.append(dict(record_id=rid,detail_level=3-j,sigma_estimate=sigma,threshold=thr,coefficient_count=len(d),fraction_abs_coefficients_below_or_equal_threshold=np.mean(np.abs(d)<=thr)))
 signals={'original':x};PS={'original':P};resids={};PR={}
 for mode in ['soft','hard']:
  y=denoise_wavelet(x,mode=mode,**settings);assert y.shape==x.shape and np.isfinite(y).all();z=x-y;_,Q=welch(y,**params);_,Z=welch(z,**params)
  signals[mode]=y;PS[mode]=Q;resids[mode]=z;PR[mode]=Z
  cross=2*np.mean(y*z);identity=np.mean(x*x)-np.mean(y*y)-np.mean(z*z)-cross
  b24=(f>=2000)&(f<4000)
  # Fixed peak-exclusion mask for an explicitly descriptive background proxy.
  pp,_=find_peaks(P,distance=int(np.ceil(20/df)));pp=sorted(pp,key=lambda i:P[i],reverse=True)[:20]
  bg=b24.copy()
  for i in pp:bg &= np.abs(f-f[i])>10
  i0=ids[0];newi=int(np.argmax(Q[1:])+1);zi=int(np.argmax(Z[1:])+1)
  s=dict(record_id=rid,fault_type=row['fault_type'],split=row['split'],load_hp=int(row['load_id']),method=mode,sigma_estimate=sigma,original_rms=rms(x),denoised_rms=rms(y),residual_rms=rms(z),original_welch_power=power,denoised_welch_power=Q.sum()*df,relative_power_change=Q.sum()/P.sum()-1,residual_welch_power=Z.sum()*df,residual_power_over_original=Z.sum()/P.sum(),original_2to4khz_power=P[b24].sum()*df,denoised_2to4khz_power=Q[b24].sum()*df,relative_2to4khz_power_change=Q[b24].sum()/P[b24].sum()-1,residual_2to4khz_fraction=Z[b24].sum()/Z.sum(),original_peak_hz=f[i0],denoised_peak_hz=f[newi],original_peak_same_bin_psd_ratio=Q[i0]/P[i0],residual_peak_hz=f[zi],residual_flatness_0to6khz=flat(Z[1:]),residual_flatness_2to4khz=flat(Z[b24]),background_proxy_original_median=P[bg].mean() if False else np.median(P[bg]),background_proxy_denoised_median=np.median(Q[bg]),background_proxy_ratio=np.median(Q[bg])/np.median(P[bg]),original_crest_factor=np.max(np.abs(x-x.mean()))/x.std(),denoised_crest_factor=np.max(np.abs(y-y.mean()))/y.std(),original_excess_kurtosis=kurtosis(x),denoised_excess_kurtosis=kurtosis(y),original_max_deviation_event_retention=abs(y[np.argmax(abs(x-x.mean()))]-y.mean())/np.max(abs(x-x.mean())),twice_mean_denoised_residual_product=cross,time_energy_identity_error=identity)
  summary.append(s)
  for lo in range(0,6000,1000):
   mask=(f>=lo)&((f<lo+1000) if lo<5000 else f<=6000)
   bands.append(dict(record_id=rid,method=mode,band_low_hz=lo,band_high_hz=lo+1000,original_power=P[mask].sum()*df,denoised_power=Q[mask].sum()*df,residual_power=Z[mask].sum()*df,original_fraction=P[mask].sum()/P.sum(),denoised_fraction=Q[mask].sum()/Q.sum(),residual_fraction=Z[mask].sum()/Z.sum()))
  for rank,i in enumerate(ids,1):
   sl=slice(max(1,i-2),min(len(f),i+3));local=max(1,i-2)+np.argmax(Q[sl])
   peaks.append(dict(record_id=rid,method=mode,original_peak_rank=rank,original_peak_hz=f[i],original_psd=P[i],denoised_same_bin_psd=Q[i],same_bin_ratio=Q[i]/P[i],local_max_hz=f[local],local_max_ratio=Q[local]/P[i],residual_same_bin_psd=Z[i]))
  checks.append(dict(record_id=rid,method=mode,length_preserved=len(y)==len(x),finite=True,energy_identity_absolute_error=abs(identity)))
 for i,hz in enumerate(f):allpsd.append(dict(record_id=rid,frequency_hz=hz,original=P[i],soft=PS['soft'][i],hard=PS['hard'][i],soft_residual=PR['soft'][i],hard_residual=PR['hard'][i]))
 np.savez_compressed(O/(rid+'_signals.npz'),original=x,soft=signals['soft'],hard=signals['hard'],soft_residual=resids['soft'],hard_residual=resids['hard'],sampling_rate_hz=12000)
 fig,ax=plt.subplots(3,2,figsize=(13,10),layout='constrained');t=np.arange(len(x))/12000;main=(t>=1)&(t<1.05);event=np.argmax(abs(x-x.mean()));start=max(0,min(event-120,len(x)-240));ext=slice(start,start+240)
 for key,y in signals.items():
  ax[0,0].plot(t[main],y[main],color=colors[key],lw=.8,label=key);ax[2,0].plot(t[ext],y[ext],color=colors[key],lw=.8,label=key)
  ax[0,1].semilogy(f[1:],PS[key][1:],color=colors[key],lw=.8,label=key);ax[2,1].semilogy(f,PS[key],color=colors[key],lw=.8,label=key)
 for mode in ['soft','hard']:
  ax[1,0].plot(t[main],resids[mode][main],color=colors[mode],lw=.8,label=mode+' residual');ax[1,1].semilogy(f[1:],PR[mode][1:],color=colors[mode],lw=.8,label=mode+' residual')
 lim=max(abs(x[main]).max(),*(abs(y[main]).max() for y in signals.values()))*1.08
 ax[0,0].set(ylim=(-lim,lim),title='Common interval: original and reconstructions');ax[1,0].set(ylim=(-lim,lim),title='Residuals: same amplitude scale as row above');ax[2,0].set_title('20 ms around largest original absolute deviation')
 for a in ax[:,0]:a.set(xlabel='Time (s)',ylabel='Source signal units')
 for a in ax[:,1]:a.set(xlabel='Frequency (Hz)',ylabel='PSD (source units²/Hz)')
 ax[0,1].set(xlim=(0,6000),title='Welch PSD: original and reconstructions');ax[1,1].set(xlim=(0,6000),title='Residual Welch PSD');ax[2,1].set(xlim=(2000,4000),title='Preservation of the 2–4 kHz structure')
 for a in [ax[0,1],ax[1,1]]:a.axvspan(2000,4000,color='#c7d6e3',alpha=.15)
 for a in ax.flat:a.grid(alpha=.18);a.legend(fontsize=8,loc='best')
 fig.suptitle(f"{rid} | {row['fault_type']} | {row['load_id']} hp {row['split']} | db4 / 3 levels / BayesShrink",fontsize=13);save(fig,rid+'_comparison')
 print(rid,'completed',flush=True)
csvout('metrics.csv',summary);csvout('band_power.csv',bands);csvout('peak_changes.csv',peaks);csvout('threshold_diagnostics.csv',thresholds);csvout('welch_spectra.csv',allpsd)
config['analysis_script_sha256']=sha(Path(__file__));config['checks']=checks;config['prior_analysis_unchanged']=all(sha(R/k)==v for k,v in oldhash.items());assert config['prior_analysis_unchanged'];assert max(c['energy_identity_absolute_error'] for c in checks)<1e-12
(O/'config_and_provenance.json').write_text(json.dumps(config,indent=2)+'\n')
fig,ax=plt.subplots(1,3,figsize=(15,5),layout='constrained');names=[r['record_id'] for r in rows];pos=np.arange(9)
for mode,shift in [('soft',-.18),('hard',.18)]:
 ss=[s for s in summary if s['method']==mode]
 for a,key,scale,title in [(ax[0],'relative_power_change',100,'Change in total Welch power (%)'),(ax[1],'relative_2to4khz_power_change',100,'Change in 2–4 kHz power (%)'),(ax[2],'original_peak_same_bin_psd_ratio',100,'Original strongest peak PSD retained (%)')]:
  a.barh(pos+shift,[s[key]*scale for s in ss],height=.34,color=colors[mode],label=mode);a.set(title=title,yticks=pos,yticklabels=names if a==ax[0] else [],ylim=(8.6,-.6));a.grid(axis='x',alpha=.2)
ax[0].legend();fig.suptitle('Fixed db4/BayesShrink denoising · offline full recordings',fontsize=14);save(fig,'denoising_overview')
