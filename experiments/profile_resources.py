from __future__ import annotations
import csv,gc,json,sys,tracemalloc
from copy import deepcopy
from pathlib import Path
from time import perf_counter
import numpy as np, psutil
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from asc_stream.config import ASCConfig
from asc_stream.model import AdaptiveSketchStreamClusterer
from asc_stream.scaling import OnlineStandardizer
from asc_stream.optimized import OptimizedAdaptiveSketchClustererArray
from asc_stream.comparators import nearest_assign
SEEDS=[7,31,53]

def preprocess(x):
    st=OnlineStandardizer(x.shape[1]);z=np.empty_like(x,dtype=np.float64)
    for i,row in enumerate(x):z[i]=st.transform_then_update(row)
    return z

def pstate(model):
    proj=int(model.omega.nbytes)
    sk=sum(int(r.nbytes) for r in model.sketch.rows)+8*len(model.sketch.weights)+8*len(model.sketch.leverage_scores)+int(model.sketch.basis.nbytes)+int(model.sketch.spectral_values.nbytes)+64
    mc=model.microclusters;micro=int(mc.centers[:mc.m].nbytes+mc.weights[:mc.m].nbytes+mc.last[:mc.m].nbytes+mc.created[:mc.m].nbytes+mc.ids[:mc.m].nbytes)
    return (proj+sk+micro)/2**20

def run_method(cfg,z,y,method,seed):
    mc=deepcopy(cfg['proposed_model']);mc.update(original_dim=z.shape[1],seed=seed,projection_seed=seed)
    if method=='fixed_rank':mc.update(min_rank=8,initial_rank=8,max_rank=8)
    m=OptimizedAdaptiveSketchClustererArray(AdaptiveSketchStreamClusterer(ASCConfig(**mc),standardize=False))
    proc=psutil.Process();gc.collect();rss0=proc.memory_info().rss;peak=rss0
    tracemalloc.start();base_current,_=tracemalloc.get_traced_memory()
    online=macro=0.0;ars=[];nms=[]
    for s in range(0,len(z),1000):
        e=min(s+1000,len(z));t=perf_counter();zz=m.process_batch(z[s:e]);online+=perf_counter()-t;peak=max(peak,proc.memory_info().rss)
        if s<1000:continue
        t=perf_counter();C=m.macro_centers(4);pred=nearest_assign(m.sketch.adapt_matrix(zz),C);macro+=perf_counter()-t;peak=max(peak,proc.memory_info().rss)
        ars.append(adjusted_rand_score(y[s:e],pred));nms.append(normalized_mutual_info_score(y[s:e],pred))
    current,tmpeak=tracemalloc.get_traced_memory();tracemalloc.stop()
    total=online+macro;ms=total*1e6/len(z)
    return {'method':method,'seed':seed,'projection_seed':seed,'ari':float(np.mean(ars)),'nmi':float(np.mean(nms)),
            'online_ms_per_1000':online*1e6/len(z),'macro_ms_per_1000':macro*1e6/len(z),'combined_ms_per_1000':ms,
            'throughput_obs_per_s':1e6/ms,'persistent_numeric_mb':pstate(m),
            'python_tracemalloc_peak_mb':max(tmpeak-base_current,0)/2**20,'incremental_peak_rss_mb':max(peak-rss0,0)/2**20}

def main():
    cfg=json.loads((ROOT/'configs/controlled_representation_drift.json').read_text());raw=np.load(ROOT/'data/controlled/representation_drift_stream.npz');x=raw['x'].astype(float);y=raw['y'].astype(int)
    pre=[]
    for i in range(5):
        gc.collect();t=perf_counter();z=preprocess(x);pre.append((perf_counter()-t)*1e6/len(x))
    cache=np.load(ROOT/'data/controlled/representation_drift_standardized.npz');z=cache['z'].astype(float);y=cache['y'].astype(int)
    rows=[]
    for method in ['proposed','fixed_rank']:
        for seed in SEEDS:
            r=run_method(cfg,z,y,method,seed);rows.append(r);print(method,seed,round(r['combined_ms_per_1000'],2),round(r['python_tracemalloc_peak_mb'],3),flush=True)
    out=ROOT/'results/execution_runs/resources';out.mkdir(parents=True,exist_ok=True)
    with (out/'seedwise.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    summary=[]
    for method in ['proposed','fixed_rank']:
        rr=[r for r in rows if r['method']==method]
        d={'method':method,'resource_seeds':'7;31;53','n_resource_runs':len(rr),'common_preprocessing_median_ms_per_1000':float(np.median(pre))}
        for k in ['online_ms_per_1000','macro_ms_per_1000','combined_ms_per_1000','throughput_obs_per_s','persistent_numeric_mb','python_tracemalloc_peak_mb','incremental_peak_rss_mb']:
            d[k+'_median']=float(np.median([r[k] for r in rr]));d[k+'_max']=float(np.max([r[k] for r in rr]))
        summary.append(d)
    with (out/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=summary[0].keys());w.writeheader();w.writerows(summary)
    meta={'preprocessing_repetitions':5,'preprocessing_median_ms_per_1000':float(np.median(pre)),'resource_seeds':SEEDS,'note':'tracemalloc captures Python-tracked allocations; NumPy/native allocations are additionally reflected imperfectly through incremental RSS sampled at chunk boundaries. These are scoped engineering measurements, not inherent algorithmic bounds.'}
    (out/'README.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
