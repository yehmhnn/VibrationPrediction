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
