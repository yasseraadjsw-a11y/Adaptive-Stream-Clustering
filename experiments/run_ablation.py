from __future__ import annotations
import argparse,json,sys,gc
from copy import deepcopy
from pathlib import Path
from time import perf_counter
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from asc_stream.config import ASCConfig
from asc_stream.model import AdaptiveSketchStreamClusterer
from asc_stream.scaling import OnlineStandardizer
from asc_stream.optimized import OptimizedAdaptiveSketchClustererArray
from asc_stream.comparators import nearest_assign
from asc_stream.release_io import ensure_not_protected_output
V=['proposed','fixed_rank','uniform_sampling','keep_all','dense_projection','leverage_weighting']
SEEDS=[7,13,19,23,31,37,41,43,47,53]

def causal_standardize(x):
    st=OnlineStandardizer(x.shape[1]); z=np.empty_like(x,dtype=np.float64)
    for i,row in enumerate(x): z[i]=st.transform_then_update(row)
    return z

def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=V,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert a.seed in SEEDS;out_path=ensure_not_protected_output(ROOT,a.out)
    cfg=json.loads((ROOT/'configs/controlled_representation_drift.json').read_text());raw=np.load(ROOT/'data/controlled/representation_drift_stream.npz');x=raw['x'].astype(np.float64);y=raw['y'].astype(np.int64);cache=ROOT/'data/controlled/representation_drift_standardized.npz'
    if cache.exists():
        c=np.load(cache); z=c['z'].astype(np.float64); y=c['y'].astype(np.int64)
    else:
        z=causal_standardize(x)
    mc=deepcopy(cfg['proposed_model']);mc.update(original_dim=z.shape[1],seed=a.seed,projection_seed=a.seed)
    if a.variant=='fixed_rank':mc.update(min_rank=8,initial_rank=8,max_rank=8)
    elif a.variant=='uniform_sampling':mc.update(leverage_mode='uniform')
    elif a.variant=='keep_all':mc.update(leverage_mode='off')
    elif a.variant=='dense_projection':mc.update(projection_mode='dense')
    elif a.variant=='leverage_weighting':mc.update(leverage_mode='weight')
    model=OptimizedAdaptiveSketchClustererArray(AdaptiveSketchStreamClusterer(ASCConfig(**mc),standardize=False));gc.collect();online=macro=0.;chunks=[]
    for s in range(0,len(z),1000):
        e=min(s+1000,len(z));t=perf_counter();zz=model.process_batch(z[s:e]);online+=perf_counter()-t
        if s<1000:continue
        t=perf_counter();C=model.macro_centers(4);pred=nearest_assign(model.sketch.adapt_matrix(zz),C);macro+=perf_counter()-t
        chunks.append({'start':s,'end':e,'ari':float(adjusted_rand_score(y[s:e],pred)),'nmi':float(normalized_mutual_info_score(y[s:e],pred))})
    ranks=np.asarray(model.rank_history,dtype=float)
    out={'variant':a.variant,'seed':a.seed,'projection_seed':mc.get('projection_seed'),'ari':float(np.mean([c['ari'] for c in chunks])),'nmi':float(np.mean([c['nmi'] for c in chunks])),'runtime_ms_per_1000':(online+macro)*1e6/len(z),'rank_mean':float(ranks.mean()),'rank_change_count':int(np.sum(ranks[1:]!=ranks[:-1])),'final_rank':int(ranks[-1]),'chunks':chunks}
    out_path.parent.mkdir(parents=True,exist_ok=True);out_path.write_text(json.dumps(out,indent=2));print(json.dumps({k:out[k] for k in ['variant','seed','ari','nmi','runtime_ms_per_1000']},indent=2))
if __name__=='__main__': main()
