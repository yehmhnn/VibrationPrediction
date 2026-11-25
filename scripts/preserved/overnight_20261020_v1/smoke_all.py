from pathlib import Path
import json,subprocess,os,time
B=Path(__file__).resolve().parent;P='./.pixi/envs/default/bin/python';env={**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'};codes={}
for cfg in json.loads((B/'plan.json').read_text()):
 id=cfg['id']
 with (B/f'smoke_{id}.log').open('x') as f:
  result=subprocess.run([P,'-B',str(B/'train.py'),'--config',str(B/'configs'/f'{id}.json'),'--smoke'],stdout=f,stderr=subprocess.STDOUT,env=env)
 codes[id]=result.returncode;(B/f'smoke_{id}.exitcode').write_text(str(result.returncode)+'\n');print(id,'PASS' if result.returncode==0 else 'FAIL',flush=True)
(B/'smoke_exitcodes.json').write_text(json.dumps(codes,indent=2))
