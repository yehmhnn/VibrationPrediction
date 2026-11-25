from pathlib import Path
import csv,json,hashlib,inspect,sys
import numpy as np
from scipy.io import loadmat
R=Path('.');O=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'scripts'))
import run_convergence_experiment as old
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
b=R/'datasets/processed/cwru_multi_record_B';m=json.loads((b/'metadata.json').read_text());n=json.loads((b/'normalization.json').read_text())
assert sha(b/'manifest.csv')==m['manifest_sha256'];assert sha(b/'normalization.json')==m['normalization_sha256']
rows=[x for x in csv.DictReader((b/'manifest.csv').open()) if x['split'] in ['train','validation']];tr=[x for x in rows if x['split']=='train'];assert len(tr)==6 and len(rows)==9;assert {x['record_id'] for x in tr}==set(n['training_record_ids']);assert n['uses_validation_or_test'] is False
raw={}
for x in rows:
 assert int(x['load_id'])<3
 p=b/x['relative_path'];assert sha(p)==m['processed_sha256'][x['relative_path']];a=np.load(p);assert len(a)==int(x['num_samples']) and np.isfinite(a).all()
 if x['split']=='train':
  p=Path(x['source_path']);assert sha(p)==n['source_sha256'][str(p)];raw[x['record_id']]=np.asarray(loadmat(p,variable_names=[x['source_key']])[x['source_key']],dtype=np.float64).reshape(-1)
a=np.concatenate(list(raw.values()));assert abs(a.mean()-n['train_mean'])<1e-14 and abs(a.std()-n['train_std'])<1e-14
for name in ['studyII','same_condition']:(O/'datasets'/name).mkdir(parents=True)
def manifest(folder,records):
 with (folder/'manifest.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
f=O/'datasets/studyII'
for x in rows:(f/x['relative_path']).symlink_to(b/x['relative_path'])
manifest(f,rows);(f/'normalization.json').write_text(json.dumps(n,indent=2))
regions=[]
for x in tr:
 count=len(raw[x['record_id']]);cut=int(.7*count)
 for split,start,end in [('train',0,cut),('validation',cut+1200,count)]:
  z=dict(x);z.update(record_id=x['record_id']+'_'+split,parent_record_id=x['record_id'],split=split,source_start=str(start),source_end_exclusive=str(end),num_samples=str(end-start),relative_path=x['record_id']+'_'+split+'.npy');regions.append(z)
a=np.concatenate([raw[x['parent_record_id']][int(x['source_start']):int(x['source_end_exclusive'])] for x in regions if x['split']=='train']);mu=float(a.mean());sd=float(a.std());assert sd>0
f=O/'datasets/same_condition'
for x in regions:
 z=raw[x['parent_record_id']][int(x['source_start']):int(x['source_end_exclusive'])];np.save(f/x['relative_path'],((z-mu)/sd).astype('float32'))
manifest(f,regions);(f/'normalization.json').write_text(json.dumps({'train_mean':mu,'train_std':sd,'ddof':0,'training_sample_count':len(a),'uses_validation_or_test':False,'gap_samples':1200,'split_fraction':.7,'training_regions':[x for x in regions if x['split']=='train']},indent=2))
from sequence_data import ManifestSequenceDataset
counts={name:{s:len(ManifestSequenceDataset(O/'datasets'/name,s,512,96,32 if s=='train' else 96)) for s in ['train','validation']} for name in ['studyII','same_condition']}
assert counts['studyII']=={'train':22754,'validation':3785}
for x in tr:
 z=[v for v in regions if v['parent_record_id']==x['record_id']];assert int(z[0]['source_end_exclusive'])+1200==int(z[1]['source_start'])
fit=inspect.getsource(old.fit).replace('lr=.001)','lr=.001,weight_decay=args.weight_decay)')
start=fit.index('            torch.save(');end=fit.index('\n',start)
fit=fit[:start]+"            save_checkpoint(output/'best.pt',model,optimizer,epoch,steps,forecasts)"+fit[end:]
fit=fit.replace("        if assessment['stop']:break","        save_checkpoint(output/f'epoch_{epoch:04d}.pt',model,optimizer,epoch,steps,forecasts)\n        save_checkpoint(output/'last.pt',model,optimizer,epoch,steps,forecasts)\n        if assessment['stop']:break")
(O/'fit_body.py').write_text(fit)
(O/'configs').mkdir();configs=[]
for id,model,ds,drop,wd,seed in [('E1','timesnet','same_condition',.1,.01,42),('E2','timesnet','studyII',.3,.01,42),('E3','timesnet','studyII',.1,.1,42),('E4','tsflow','studyII',0.,.1,42),('E5_seed7','timesnet','studyII',.1,.01,7),('E5_seed123','timesnet','studyII',.1,.01,123),('E6_seed7','timesnet','studyII',.3,.01,7),('E6_seed123','timesnet','studyII',.3,.01,123)]:
 c=dict(id=id,model=model,dataset=ds,dropout=drop,weight_decay=wd,seed=seed,max_epochs=150 if model=='tsflow' else 100,device='cuda',min_delta=.001,patience=10,stability_window=10,min_assessment_epochs=20,rmse_relative_range=.02,loss_relative_trend=.02,gap_relative_increase=.05,stop_policy='none',evaluation_seed=420001,fm_evaluation_draws=4,window_counts=counts[ds],L=512,H=96,batch_size=32,learning_rate=.001,train_stride=32,eval_stride=96)
 (O/'configs'/f'{id}.json').write_text(json.dumps(c,indent=2));configs.append(c)
(O/'plan.json').write_text(json.dumps(configs,indent=2));(O/'data_checks.json').write_text(json.dumps({'status':'PASS','counts':counts,'same_condition_mu':mu,'same_condition_std':sd,'same_condition_training_samples':len(a),'gap_samples':1200,'studyII_train_norm_recomputed':True,'reserved_payloads_accessed':False},indent=2))
(O/'source_hashes.json').write_text(json.dumps({str(p):sha(p) for p in (R/'scripts').glob('*.py')},indent=2));print(counts)
