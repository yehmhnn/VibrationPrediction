def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args();cfg=json.loads(Path(a.config).read_text());args=argparse.Namespace(**cfg)
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
 cp=torch.load(output/'last.pt',map_location='cpu',weights_only=False);assert cp['epoch']==args.max_epochs;model.load_state_dict(cp['model_state_dict'])
 report={'status':'PASS','wall_seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'checkpoint_bytes':(output/'last.pt').stat().st_size,'optimizer_steps':result['optimizer_steps'],'full_window_counts':counts,'smoke':a.smoke,'test_access':False}
 (output/'checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
