import torch
from torch.utils.tensorboard import SummaryWriter
import os.path as path
import shutil
import numpy as np
import json


SEP='-'*40

class Jimmy:
    def __init__(self,experiment_name,metric_keys,logdir='.',verbose=5):
        self.metric_keys=set(['train_'+k for k in metric_keys]+['val_'+k for k in metric_keys])
        self.verbose=verbose

        self.log={k:[] for k in self.metric_keys]}
        self.last_epoch={k:0 for k in self.metric_keys]}
        self.t={k:0 for k in self.metric_keys}
        self.model_t=0
        self.epoch=1
        self.logdir=logdir
        config_item={'exp_name':experiment_name}
        config_path=path.join(logdir,'logger_config.json')
        if path.exists(config_path):
            with open(config_path) as fd:
                config_dict=json.load(fd)
            exp_id=int(list(config_dict.keys())[-1])+1
        else:
           exp_id=1 
           config_dict=dict()
        config_dict[exp_id]=config_item
        with open(config_path,'w+') as fd:
            json.dump(config_dict,fd,indent=2)
        
        self.tag=str(exp_id)
        self.summ_path=path.join(logdir,self.tag)  
        self.sw=SummaryWriter(self.summ_path)
    
    def update(self, dic):
        assert all([k in self.metric_keys for k in dic.keys()])
        for k,v in dic.items():
        if  torch.isnan(v):
            print("NAN")#raise Exception('nan')
        v=float(v)
        self.t[k]+=1
        self.sw.add_scalar(k,v,self.t[k])
    
    def close_epoch(self):
        pri=False
        if self.verbose and self.epoch%self.verbose==0:
                print(f'Epoch {self.epoch}')
                pri=True
        for k in self.metric_keys:
            aux=self.log[k]
            if not aux: continue
            aux=aux[self.last_epoch[k]:]
            self.last_epoch[k]=self.t[k]
            if pri:  print(f'{k} average: {sum(aux)/len(aux)}')
        if pri: print("\n")
    def close(self,model):
        self.sw.close()
        torch.save({'state_dict':model.state_dict(),
                    'log':self.log}, path.join(self.summ_path,'mdl.pt'))
            