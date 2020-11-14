
import torch
from torch.utils.tensorboard import SummaryWriter
import os.path as path
import json

class Jimmy:
    def __init__(self,experiment_name,metric_keys,model,logdir='.',verbose=5,**kwargs):
        self.metric_keys=set(['train_'+k for k in metric_keys]+['val_'+k for k in metric_keys])
        self.verbose=verbose
        self.model=model
        self.save_every=kwargs.get('save_every',10)
        self.main_metric=kwargs.get('main_metric',None)
        self.log={k:[] for k in self.metric_keys}
        self.last_epoch={k:0 for k in self.metric_keys}
        self.t={k:0 for k in self.metric_keys}
        self.model_t=0
        self.epoch=1
        self.logdir=logdir
        self.best_metric_val=-1 ##TODO
        
        self.tag=experiment_name
        self.summ_path=path.join(logdir,self.tag)  
        self.sw=SummaryWriter(self.summ_path)
    
    def update(self, dic):
        assert all([k in self.metric_keys for k in dic.keys()])
        for k,v in dic.items():
            v=float(v)
            self.t[k]+=1
            self.sw.add_scalar(k,v,self.t[k])
            self.log[k].append(v)
    
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
        if self.main_metric and self.log[self.main_metric][-1]>self.best_metric_val:
            torch.save(self.model.state_dict(), path.join(self.summ_path,f'best_mdl.pt'))
            self.best_metric_val=self.log[self.main_metric][-1]
        if self.save_every and self.epoch%self.save_every==0:
        
            self.flush()
        self.epoch+=1
    def flush(self):
        torch.save(self.model.state_dict(), path.join(self.summ_path,f'mdl{self.epoch}.pt'))
        with open(path.join(self.summ_path,'log.json'),'w+') as fd:
            json.dump(self.log,fd)
        
    def close(self):
        self.sw.close()
        self.flush
     
            