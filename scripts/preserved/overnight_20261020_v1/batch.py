from pathlib import Path
import os,sys,json,subprocess,time,hashlib,fcntl,shutil
B=Path(__file__).resolve().parent;R=B.parents[1];P=str(R/'.pixi/envs/default/bin/python')
assert sys.argv[1:]==['--approved'],'Full training requires explicit approval and --approved'
lock=(B/'batch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (B/'BATCH_STARTED.json').exists(),'No reruns or overwriting'
assert all(v==0 for v in json.loads((B/'smoke_exitcodes.json').read_text()).values())
def check_integrity():
 for path,digest in json.loads((B/'sealed_hashes.json').read_text()).items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,'Changed: '+path
check_integrity();assert shutil.disk_usage(B).free>15*1024**3,'Need 15 GiB free'
plan=json.loads((B/'plan.json').read_text());assert len(plan)==8
assert all(not (B/'runs'/c['id']).exists() for c in plan)
(B/'BATCH_STARTED.json').write_text(json.dumps({'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'pid':os.getpid()},indent=2))
(B/'batch_logs').mkdir(exist_ok=False);records=[];env={**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
def summary():
 lines=['# OVERNIGHT SUMMARY','','Fixed plan; no adaptive configuration selection. No reserved test inference.','','| Run | Exit code | Status | Wall seconds |','|---|---:|---|---:|']
 for x in records:lines.append(f"| {x['id']} | {x['exitcode']} | {x['status']} | {x['seconds']:.1f} |")
 lines+=['','Configs: configs/*.json. Logs: batch_logs/<id>.log. Exit codes: batch_logs/<id>.exitcode.','Full histories, per-record metrics and every epoch checkpoint: runs/<id>/.','Failures do not change budgets. Successful checks establish execution only; scientific interpretation requires review.']
 (B/'OVERNIGHT_SUMMARY.md').write_text('\n'.join(lines)+'\n');(B/'batch_status.json').write_text(json.dumps(records,indent=2))
try:
 for cfg in plan:
  id=cfg['id'];start=time.perf_counter();code=125;status='preflight_failed'
  try:
   check_integrity();assert shutil.disk_usage(B).free>15*1024**3
   jobs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()
   assert not jobs,'GPU occupied: '+jobs
   cmd=[P,'-B',str(B/'train.py'),'--config',str(B/'configs'/f'{id}.json')]
   with (B/'batch_logs'/f'{id}.log').open('x') as f:
    subprocess.run(['nvidia-smi'],stdout=f,stderr=subprocess.STDOUT)
    code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=R).returncode
   status='completed' if code==0 else 'failed'
  except Exception as e:
   (B/'batch_logs'/f'{id}.preflight_error.txt').write_text(repr(e))
  (B/'batch_logs'/f'{id}.exitcode').write_text(str(code)+'\n');records.append(dict(id=id,exitcode=code,status=status,seconds=time.perf_counter()-start));summary()
  if status=='preflight_failed':break
finally:summary()

sys.exit(0 if len(records)==len(plan) and all(x["exitcode"]==0 for x in records) else 1)
