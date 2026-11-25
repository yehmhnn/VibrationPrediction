import os,sys,json,time,random,hashlib,argparse,csv
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
ROOT=Path('.');BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from convergence_utils import *
from record_evaluation import IdentifiedDataset
from sequence_data import ManifestSequenceDataset
from tsflow_data import TSFlowManifestDataset
from run_flow_experiment import create_model,seed_everything
from run_experiments import TimesNetLightning
from run_convergence_experiment import sync,assessment_config

def save_checkpoint(path,model,optimizer,epoch,steps,forecasts):
 state={'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'epoch':epoch,'optimizer_step':steps,'validation_forecast_RMSE':forecasts['RMSE'],'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
 tmp=path.with_suffix('.tmp');torch.save(state,tmp);os.replace(tmp,path)
def fit(model,loaders,args,output):
    """Caller supplies fresh directory; synthetic tests supply synthetic loaders."""
    if any((output/name).exists() for name in ['history.csv','best.pt','result.json','validation_records.jsonl','assessment_history.jsonl']):raise FileExistsError('Refusing to overwrite existing experiment artifacts')
    is_flow=args.model=='tsflow';fields=FLOW_FIELDS if is_flow else TIMESNET_FIELDS
    train_key=fields[2];val_key=fields[3]
    monitor=StabilizationAssessment(assessment_config(args))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=args.weight_decay)
    rows=[];steps=0;best=float('inf');best_epoch=None;optimization_seconds=0.
    sync(args.device);started=time.perf_counter()
    for epoch in range(1,args.max_epochs+1):
        model.train();total,count=0.,0
        sync(args.device);tick=time.perf_counter()
        for batch in loaders['train']:
            batch=move(batch,args.device);optimizer.zero_grad(set_to_none=True)
            if is_flow:loss,n=flow_objective(model,batch)
            else:
                x,y=batch[:2];error=model(x.float())-y.float();loss=error.square().mean();n=y.numel()
            if not torch.isfinite(loss):raise ValueError('Nonfinite training objective')
            loss.backward()
            if is_flow:torch.nn.utils.clip_grad_norm_(model.parameters(),.5)
            optimizer.step();steps+=1;total+=loss.item()*n;count+=n
        if not count:raise ValueError('Empty training loader')
        sync(args.device);optimization_seconds+=time.perf_counter()-tick
        tick=time.perf_counter()
        fm=validation_flow_loss(model,loaders['validation'],args.device,args.evaluation_seed,args.fm_evaluation_draws) if is_flow else None
        forecasts=validation_forecast(model,loaders['validation'],args.device,10042,flow=is_flow)
        sync(args.device);validation_seconds=time.perf_counter()-tick
        train_loss=total/count;validation_loss=fm if is_flow else forecasts['MSE']
        assessment=monitor.update(forecasts['RMSE'],train_loss,validation_loss)
        if forecasts['RMSE']<best:
            best=forecasts['RMSE'];best_epoch=epoch;best_record_metrics=forecasts['record_metrics']
            save_checkpoint(output/'best.pt',model,optimizer,epoch,steps,forecasts)
        row={'epoch':epoch,'optimizer_step':steps,train_key:train_loss,val_key:validation_loss,'validation_forecast_RMSE':forecasts['RMSE'],'validation_forecast_MAE':forecasts['MAE'],'elapsed_training_seconds':time.perf_counter()-started,'optimization_seconds':optimization_seconds,'validation_seconds':validation_seconds,'assessment':assessment['status']}
        rows.append(row);write_history(output/'history.csv',rows,fields)
        with (output/'validation_records.jsonl').open('a') as f:f.write(json.dumps({'epoch':epoch,**forecasts['record_metrics']})+'\n')
        with (output/'assessment_history.jsonl').open('a') as f:f.write(json.dumps({'epoch':epoch,**assessment})+'\n')
        print(f"epoch={epoch} step={steps} train={train_loss:.6f} validation_objective={validation_loss:.6f} validation_RMSE={forecasts['RMSE']:.6f} assessment={assessment['status']}",flush=True)
        save_checkpoint(output/f'epoch_{epoch:04d}.pt',model,optimizer,epoch,steps,forecasts)
        save_checkpoint(output/'last.pt',model,optimizer,epoch,steps,forecasts)
        if assessment['stop']:break
    result={'epochs_completed':len(rows),'optimizer_steps':steps,'best_epoch':best_epoch,'best_validation_forecast_RMSE':best,'termination_reason':assessment['status'] if assessment['stop'] else 'maximum_epoch_budget','assessment':assessment,'criterion':asdict(monitor.config),'training_seconds':time.perf_counter()-started,'optimization_seconds':optimization_seconds,'test_evaluated':False,'in_condition_reference_evaluated':False,'best_validation_record_metrics':None}
    result['validation_seconds']=sum(r['validation_seconds'] for r in rows)
    result['best_validation_record_metrics']=best_record_metrics
    if args.evaluate_test or getattr(args,'in_condition_reference',False):
        checkpoint=torch.load(output/'best.pt',map_location=args.device,weights_only=False);model.load_state_dict(checkpoint['model_state_dict'])
    if args.evaluate_test:
        test=validation_forecast(model,loaders['test'],args.device,20042,flow=is_flow,arrays=True,prefixes=(24,48,72,96))
        np.save(output/'record_ids.npy',test.pop('record_ids'));np.save(output/'forecast_origins.npy',test.pop('forecast_origins'))
        np.save(output/'targets.npy',test.pop('targets'));np.save(output/'predictive_mean.npy',test.pop('predictions'));result.update(test_evaluated=True,test_metrics=test)
    if getattr(args,'in_condition_reference',False):
        reference=validation_forecast(model,loaders['in_condition_reference'],args.device,30042,flow=is_flow,prefixes=(24,48,72,96))
        result.update(in_condition_reference_evaluated=True,in_condition_reference={'label':'selected checkpoint on training-condition records; reused training data; not independent evidence; excluded from selection/stopping',**reference})
    result['total_elapsed_seconds']=time.perf_counter()-started
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result,rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args();cfg=json.loads(Path(a.config).read_text());args=argparse.Namespace(**cfg)
 for path,digest in json.loads((BASE/'source_hashes.json').read_text()).items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest, 'Source changed: '+path
 assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8' and torch.cuda.is_available()
 folder=BASE/'datasets'/cfg['dataset'];rows=list(csv.DictReader((folder/'manifest.csv').open()));assert all(int(x['load_id'])<3 and x['split'] in ['train','validation'] for x in rows)
 allowed={str((folder/x['relative_path']).resolve()) for x in rows}
 def guard(event,values):
  if event=='open' and isinstance(values[0],(str,bytes,os.PathLike)):
   p=Path(os.fsdecode(values[0])).resolve()
   if ROOT/'datasets' in p.parents and str(p) not in allowed:raise PermissionError('Blocked unapproved dataset access: '+str(p))
 sys.addaudithook(guard)
 output=BASE/('smoke' if a.smoke else 'runs')/cfg['id'];output.mkdir(parents=True,exist_ok=False)
 seed_everything(args.seed);torch.use_deterministic_algorithms(True);flow=args.model=='tsflow'
 data={s:IdentifiedDataset(TSFlowManifestDataset(folder,s,512,96,stride=32 if s=='train' else 96),flow=True) if flow else IdentifiedDataset(ManifestSequenceDataset(folder,s,512,96,stride=32 if s=='train' else 96)) for s in ['train','validation']}
 counts={s:len(d) for s,d in data.items()};assert counts==cfg['window_counts']
 if a.smoke:
  for s,d in list(data.items()):
   if s=='train':indices=list(range(32))
   else:
    indices=[];seen={}
    for i,(rec,start) in enumerate(d.windows.window_index):
     if seen.get(rec,0)<(16 if flow else 32):indices.append(i);seen[rec]=seen.get(rec,0)+1
   sub=Subset(d,indices);sub.record_metadata=d.record_metadata;data[s]=sub
  args.max_epochs=1
 generator=torch.Generator().manual_seed(args.seed)
 loaders={s:DataLoader(d,batch_size=16 if flow and s=='validation' else 32,shuffle=s=='train',num_workers=0,generator=generator if s=='train' else None) for s,d in data.items()}
 model=create_model(512,96,.001,8) if flow else TimesNetLightning(input_length=512,prediction_length=96)
 if not flow:model.model.dropout.p=args.dropout
 model=model.to('cuda');pvals=[m.p for m in model.modules() if isinstance(m,torch.nn.Dropout)]
 if not flow:assert pvals and all(p==args.dropout for p in pvals)
 args.evaluate_test=False;args.in_condition_reference=False
 (output/'config.json').write_text(json.dumps({**cfg,'smoke':a.smoke,'effective_dropout_modules':pvals,'evaluated_windows':{s:len(d) for s,d in data.items()},'deterministic':True},indent=2))
 torch.cuda.reset_peak_memory_stats();start=time.perf_counter();result,history=fit(model,loaders,args,output)
 assert len(history)==args.max_epochs and result['optimizer_steps']==len(loaders['train'])*args.max_epochs
 if a.smoke:
  weights={k:v.detach().clone() for k,v in model.state_dict().items()};grads=[p.grad.clone() if p.grad is not None else None for p in model.parameters()]
  validation_forecast(model,loaders['validation'],'cuda',10042,flow=flow)
  assert all(torch.equal(weights[k],v) for k,v in model.state_dict().items())
  assert all((g is None and p.grad is None) or (g is not None and p.grad is not None and torch.equal(g,p.grad)) for g,p in zip(grads,model.parameters()))
 cp=torch.load(output/'last.pt',map_location='cpu',weights_only=False);assert cp['epoch']==args.max_epochs;assert all(g['weight_decay']==args.weight_decay for g in cp['optimizer_state_dict']['param_groups']);assert len(cp['optimizer_state_dict']['state'])>0;model.load_state_dict(cp['model_state_dict'])
 report={'status':'PASS','wall_seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'checkpoint_bytes':(output/'last.pt').stat().st_size,'optimizer_steps':result['optimizer_steps'],'full_window_counts':counts,'smoke':a.smoke,'test_access':False}
 (output/'checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
