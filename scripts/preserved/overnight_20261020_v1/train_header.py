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
