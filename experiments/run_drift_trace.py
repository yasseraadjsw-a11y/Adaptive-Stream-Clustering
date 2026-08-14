from __future__ import annotations

"""Rebuild time-resolved reconstruction-error/rank evidence for the final ASC controller."""

import csv, gc, json, sys
from copy import deepcopy
from pathlib import Path
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from asc_stream.config import ASCConfig
from asc_stream.model import AdaptiveSketchStreamClusterer
from asc_stream.optimized import OptimizedAdaptiveSketchClustererArray
from asc_stream.comparators import nearest_assign

SEEDS=[7,13,19,23,31,37,41,43,47,53]
DRIFTS=[3000,6000]
ASSOCIATION_WINDOW=500


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)


def main():
    cfg=json.loads((ROOT/'configs/controlled_representation_drift.json').read_text())
    c=np.load(ROOT/'data/controlled/representation_drift_standardized.npz');z=c['z'].astype(np.float64);y=c['y'].astype(np.int64)
    outdir=ROOT/'results/execution_runs/drift';outdir.mkdir(parents=True,exist_ok=True)
    delays=[];quality=[];runs=[];rankdist=[]
    for seed in SEEDS:
        mc=deepcopy(cfg['proposed_model']);mc.update(original_dim=z.shape[1],seed=seed,projection_seed=seed)
        gc.collect();model=OptimizedAdaptiveSketchClustererArray(AdaptiveSketchStreamClusterer(ASCConfig(**mc),standardize=False))
        chunks=[]
        for s in range(0,len(z),1000):
            e=min(s+1000,len(z));zz=model.process_batch(z[s:e])
            if s<1000:continue
            C=model.macro_centers(4);pred=nearest_assign(model.sketch.adapt_matrix(zz),C)
            chunks.append({'seed':seed,'start':s,'end':e,'ari':float(adjusted_rand_score(y[s:e],pred)),'nmi':float(normalized_mutual_info_score(y[s:e],pred))})
        quality.extend(chunks)
        ranks=np.asarray(model.rank_history,int); changes=np.flatnonzero(np.r_[False,ranks[1:]!=ranks[:-1]])+1  # 1-based observation time
        errs=np.asarray(model.error_history,float);thr=np.asarray(model.threshold_history,float)
        associated=set()
        for d in DRIFTS:
            candidates=changes[(changes>=d)&(changes<=d+ASSOCIATION_WINDOW)]
            delay=float(candidates[0]-d) if len(candidates) else float('nan')
            if len(candidates):associated.add(int(candidates[0]))
            pre=[q for q in chunks if q['end']==d]
            preari=pre[0]['ari'] if pre else float('nan'); prenmi=pre[0]['nmi'] if pre else float('nan')
            target=.95*preari
            after=[q for q in chunks if q['start']>=d and q['ari']>=target]
            recovery=float(after[0]['end']-d) if after else float('nan')
            delays.append({'seed':seed,'drift_point':d,'adaptation_delay_points':delay,'pre_ari':preari,'pre_nmi':prenmi,'recovery_target_ari':target,'recovery_time_points':recovery})
        # Every rank change outside the two post-drift association windows is a false update for this controlled schedule.
        false=[int(t) for t in changes if not any(d<=t<=d+ASSOCIATION_WINDOW for d in DRIFTS)]
        vals,counts=np.unique(ranks,return_counts=True)
        for v,n in zip(vals,counts):rankdist.append({'seed':seed,'rank':int(v),'count':int(n),'percent':float(100*n/len(ranks))})
        runs.append({'seed':seed,'rank_mean':float(ranks.mean()),'rank_min':int(ranks.min()),'rank_max':int(ranks.max()),'final_rank':int(ranks[-1]),'rank_change_count':int(len(changes)),'percent_rank32':float(100*np.mean(ranks==32)),'false_update_count':len(false),'false_update_rate_per_1000':float(len(false)*1000/len(ranks))})
        np.savez_compressed(outdir/f'trace_seed_{seed}.npz',time=np.arange(1,len(ranks)+1),rank=ranks,reconstruction_error=errs,threshold=thr,rank_changed=np.asarray(model.rank_change_history,bool),basis_updated=np.asarray(model.basis_update_history,bool))
        if seed==7:
            dec=[]
            keep=set(range(1,len(ranks)+1,25))|set(DRIFTS)|set(int(t) for t in changes)
            for t in sorted(keep):dec.append({'time':t,'rank':int(ranks[t-1]),'reconstruction_error':float(errs[t-1]),'threshold':float(thr[t-1]),'rank_changed':int(t in set(changes))})
            write_csv(outdir/'trace_seed7_decimated.csv',dec)
    write_csv(outdir/'adaptation_delay.csv',delays);write_csv(outdir/'drift_quality.csv',quality);write_csv(outdir/'run_summary.csv',runs);write_csv(outdir/'rank_distribution_seedwise.csv',rankdist)
    arr_delay=np.array([r['adaptation_delay_points'] for r in delays if np.isfinite(r['adaptation_delay_points'])])
    d3000=[r['adaptation_delay_points'] for r in delays if r['drift_point']==3000 and np.isfinite(r['adaptation_delay_points'])]
    d6000=[r['adaptation_delay_points'] for r in delays if r['drift_point']==6000 and np.isfinite(r['adaptation_delay_points'])]
    rec3000=[r['recovery_time_points'] for r in delays if r['drift_point']==3000 and np.isfinite(r['recovery_time_points'])]
    rec6000=[r['recovery_time_points'] for r in delays if r['drift_point']==6000 and np.isfinite(r['recovery_time_points'])]
    summary={'rank_mean':float(np.mean([r['rank_mean'] for r in runs])),'rank_min':min(r['rank_min'] for r in runs),'rank_max':max(r['rank_max'] for r in runs),'final_rank_min':min(r['final_rank'] for r in runs),'final_rank_max':max(r['final_rank'] for r in runs),'rank_changes_mean':float(np.mean([r['rank_change_count'] for r in runs])),'pct_rank32_mean':float(np.mean([r['percent_rank32'] for r in runs])),'delay_t3000_mean':float(np.mean(d3000)),'delay_t6000_mean':float(np.mean(d6000)),'recovery_t3000_mean':float(np.mean(rec3000)),'recovery_t6000_mean':float(np.mean(rec6000)),'false_update_rate_per_1000_mean':float(np.mean([r['false_update_rate_per_1000'] for r in runs])),'association_window_points':ASSOCIATION_WINDOW,'definition':'adaptation delay = first rank change in [drift, drift+500]; false update = rank change outside both declared 500-point post-drift windows; recovery = first 1000-point evaluation chunk at >=95% of the immediately pre-drift ARI'}
    (outdir/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
