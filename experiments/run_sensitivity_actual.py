from __future__ import annotations
import csv,gc,json,sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from asc_stream.config import ASCConfig
from asc_stream.model import AdaptiveSketchStreamClusterer
from asc_stream.optimized import OptimizedAdaptiveSketchClustererArray
from asc_stream.comparators import nearest_assign
from asc_stream.fast_metrics import ari_nmi
GRIDS={'original_dim':[64,128,256],'projection_dim':[16,32,48],'window_size':[500,1000,1500],'max_rank':[8,16,32],'max_microclusters':[100,200,300]}

def pstate(model):
    proj=model.omega.nbytes;sk=sum(r.nbytes for r in model.sketch.rows)+8*len(model.sketch.weights)+8*len(model.sketch.leverage_scores)+model.sketch.basis.nbytes+model.sketch.spectral_values.nbytes+64
    mc=model.microclusters;micro=mc.centers[:mc.m].nbytes+mc.weights[:mc.m].nbytes+mc.last[:mc.m].nbytes+mc.created[:mc.m].nbytes+mc.ids[:mc.m].nbytes
    return (proj+sk+micro)/2**20

def run(cfg,axis,val,z,y):
    x=z[:,:val] if axis=='original_dim' else z;mc=deepcopy(cfg['proposed_model'])
    if axis!='original_dim':mc[axis]=val
    if axis=='projection_dim':
        mc['max_rank']=min(mc['max_rank'],val);mc['initial_rank']=min(mc['initial_rank'],mc['max_rank']);mc['min_rank']=min(mc['min_rank'],mc['initial_rank'])
    if axis=='max_rank':mc['initial_rank']=min(mc['initial_rank'],val);mc['min_rank']=min(mc['min_rank'],mc['initial_rank'])
    mc.update(original_dim=x.shape[1],seed=7,projection_seed=7);gc.collect();m=OptimizedAdaptiveSketchClustererArray(AdaptiveSketchStreamClusterer(ASCConfig(**mc),standardize=False));online=macro=0.;ars=[];nms=[]
    for s in range(0,len(x),1000):
        e=min(s+1000,len(x));t=perf_counter();zz=m.process_batch(x[s:e]);online+=perf_counter()-t
        if s<1000:continue
        t=perf_counter();C=m.macro_centers(4);pred=nearest_assign(m.sketch.adapt_matrix(zz),C);macro+=perf_counter()-t;a,n=ari_nmi(y[s:e],pred);ars.append(a);nms.append(n)
    return {'axis':axis,'value':val,'seed':7,'ari':float(np.mean(ars)),'nmi':float(np.mean(nms)),'runtime_ms_per_1000':(online+macro)*1e6/len(x),'persistent_numeric_mb':pstate(m),'rank_mean':float(np.mean(m.rank_history)),'rank_max_observed':int(np.max(m.rank_history))}

def main():
    cfg=json.load(open(ROOT/'configs/controlled_representation_drift.json'));c=np.load(ROOT/'data/controlled/representation_drift_standardized.npz');z=c['z'].astype(float);y=c['y'].astype(int);rows=[]
    for axis,vals in GRIDS.items():
        for val in vals:
            r=run(cfg,axis,val,z,y);rows.append(r);print(json.dumps(r),flush=True)
    out=ROOT/'results/execution_runs/sensitivity';out.mkdir(parents=True,exist_ok=True)
    with (out/'seedwise.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
