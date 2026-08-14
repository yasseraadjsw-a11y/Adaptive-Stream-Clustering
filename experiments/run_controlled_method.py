from __future__ import annotations
import argparse,json,os,sys,gc
from pathlib import Path
from time import perf_counter
import numpy as np, psutil
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from asc_stream.scaling import OnlineStandardizer
from asc_stream.config import ASCConfig
from asc_stream.model import AdaptiveSketchStreamClusterer
from asc_stream.optimized import OptimizedAdaptiveSketchClustererArray
from asc_stream.comparators import FRAARTComparator,TWStreamComparator,nearest_assign
from asc_stream.release_io import ensure_not_protected_output
SEEDS=[7,13,19,23,31,37,41,43,47,53]
METHODS=['proposed','fixed_rank','twstream','fra_art']

def preprocess(x):
    st=OnlineStandardizer(x.shape[1]); z=np.empty_like(x,dtype=np.float64)
    for i,row in enumerate(x):z[i]=st.transform_then_update(row)
    return z,st

def radius_from_warmup(z):
    nn=NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean').fit(z[:500]);d,_=nn.kneighbors(z[:500]);return float(np.quantile(d[:,-1],.80))

def pstate(model):
    proj=model.omega.nbytes
    sketch=sum(r.nbytes for r in model.sketch.rows)+8*len(model.sketch.weights)+8*len(model.sketch.leverage_scores)+model.sketch.basis.nbytes+model.sketch.spectral_values.nbytes+64
    mc=model.microclusters; micro=mc.centers[:mc.m].nbytes+mc.weights[:mc.m].nbytes+mc.last[:mc.m].nbytes+mc.created[:mc.m].nbytes+mc.ids[:mc.m].nbytes
    return {'projection_bytes':proj,'sketch_basis_probability_bytes':sketch,'microcluster_bytes':micro,'persistent_numeric_bytes':proj+sketch+micro,'allocated_numeric_bytes':proj+sketch+micro}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--method',choices=METHODS,required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assert a.seed in SEEDS;out_path=ensure_not_protected_output(ROOT,a.out)
    raw=np.load(ROOT/'data/controlled/representation_drift_stream.npz');x=raw['x'].astype(np.float64);y=raw['y'].astype(np.int64);cache=ROOT/'data/controlled/representation_drift_standardized.npz'; c=np.load(cache) if cache.exists() else None; z=c['z'].astype(np.float64) if c is not None else preprocess(x)[0]; radius=radius_from_warmup(z[50:1000]);d=z.shape[1]
    cfg=json.loads((ROOT/'configs/controlled_representation_drift.json').read_text())
    if a.method in {'proposed','fixed_rank'}:
        mc=dict(cfg['proposed_model']);mc.update(original_dim=d,seed=a.seed,projection_seed=a.seed)
        if a.method=='fixed_rank': mc.update(initial_rank=8,min_rank=8,max_rank=8)
        base=AdaptiveSketchStreamClusterer(ASCConfig(**mc),standardize=False);model=OptimizedAdaptiveSketchClustererArray(base);xin=z
    elif a.method=='fra_art':
        model=FRAARTComparator(d,a.seed,a=.5,vigilance=.8,choice=.001,beta=1.);lo=z[50:1000].min(0);hi=z[50:1000].max(0);xin=np.clip((z-lo)/(hi-lo+1e-12),0,1)
    else:
        model=TWStreamComparator(d,a.seed,max_clusters=200,max_outliers=200,radius=radius,k=8,lam=.0028);xin=z
    gc.collect();proc=psutil.Process();rss0=proc.memory_info().rss;peak=rss0;online=macro=0.;chunks=[]
    for s in range(0,len(x),1000):
        e=min(s+1000,len(x));t=perf_counter();zz=model.process_batch(xin[s:e]);online+=perf_counter()-t;peak=max(peak,proc.memory_info().rss)
        if s<1000:continue
        t=perf_counter();# Common offline summary evaluator used consistently across the controlled comparison.
        C=model.macro_centers(4); pred=nearest_assign(model.sketch.adapt_matrix(zz),C) if a.method in {'proposed','fixed_rank'} else nearest_assign(zz,C);macro+=perf_counter()-t
        chunks.append({'start':s,'end':e,'ari':float(adjusted_rand_score(y[s:e],pred)),'nmi':float(normalized_mutual_info_score(y[s:e],pred))})
    state=pstate(model) if a.method in {'proposed','fixed_rank'} else {k:v for k,v in model.diagnostics().items() if k.endswith('_bytes')}
    total=online+macro
    res={'method':a.method,'seed':a.seed,'ari':float(np.mean([c['ari'] for c in chunks])),'nmi':float(np.mean([c['nmi'] for c in chunks])),'runtime_ms_per_1000':total*1e6/len(x),'online_seconds':online,'macro_seconds':macro,'persistent_numeric_mb':state['persistent_numeric_bytes']/2**20,'allocated_numeric_mb':state.get('allocated_numeric_bytes',state['persistent_numeric_bytes'])/2**20,'incremental_peak_rss_mb':max(peak-rss0,0)/2**20,'chunks':chunks,'protocol':{'same_stream':True,'n':len(x),'d':d,'data_seed':2026,'model_seed':a.seed,'proposed_projection_seed':a.seed if a.method in {'proposed','fixed_rank'} else None,'projection_varies_across_proposed_repetitions':True,'labels_used_online':False,'common_causal_standardization_outside_timing':True,'shared_macro_evaluator_known_k':4,'implementation_scope':'study_implementation_common_protocol'}}
    out_path.parent.mkdir(parents=True,exist_ok=True);out_path.write_text(json.dumps(res,indent=2));print(json.dumps({k:res[k] for k in ['method','seed','ari','nmi','runtime_ms_per_1000','persistent_numeric_mb']},indent=2))
if __name__=='__main__':main()
