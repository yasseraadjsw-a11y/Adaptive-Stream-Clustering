from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from time import perf_counter

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asc_stream.baselines import CluStreamBaseline, DenStreamBaseline, StreamKMPlusPlusBaseline
from asc_stream.comparators import FRAARTComparator, TWStreamComparator, nearest_assign
from asc_stream.fast_metrics import ari_nmi
from asc_stream.model import AdaptiveSketchStreamClusterer
from asc_stream.optimized import OptimizedAdaptiveSketchClustererArray, causal_standardize_dense
from asc_stream.paper_protocol import DISPLAY, N_CLUSTERS, asc_config_from_paper, load_full_paper_dataset, load_packaged_study_dataset, load_paper_config
from asc_stream.method_registry import method_metadata
from asc_stream.release_io import ensure_execution_output

METHODS=("proposed","fixed_rank","clustream","denstream","streamkmpp","twstream","fra_art")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _execution_fingerprint() -> str:
    digest = hashlib.sha256()
    paths = [
        Path(__file__),
        ROOT / "src" / "asc_stream" / "model.py",
        ROOT / "src" / "asc_stream" / "comparators.py",
        ROOT / "src" / "asc_stream" / "baselines.py",
        ROOT / "src" / "asc_stream" / "paper_protocol.py",
        ROOT / "configs" / "datasets" / "covertype.json",
        ROOT / "configs" / "datasets" / "electricity.json",
        ROOT / "configs" / "datasets" / "tweeteval.json",
        ROOT / "configs" / "datasets" / "synthetic_gmm.json",
        ROOT / "configs" / "modern_methods_all_datasets.json",
    ]
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(_file_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _modern_config() -> dict:
    path = ROOT / "configs" / "modern_methods_all_datasets.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("schema") != "modern_methods_all_datasets_v1":
        raise ValueError(f"Unexpected modern-method protocol schema in {path}")
    return cfg


def _dense_view(block) -> np.ndarray:
    if sparse.issparse(block):
        return np.asarray(block.toarray(), dtype=np.float64)
    return np.asarray(block, dtype=np.float64)


def _calibration_radius(calibration: np.ndarray) -> float:
    if len(calibration) < 3:
        return 1.0
    k = min(6, len(calibration))
    distances, _ = NearestNeighbors(n_neighbors=k, algorithm="brute").fit(calibration).kneighbors(calibration)
    return float(max(np.quantile(distances[:, -1], 0.80), 1e-6))


def build_method(
    name: str,
    dataset: str,
    d: int,
    seed: int,
    *,
    tweet_preprojected: bool,
    prestandardized: bool = False,
    calibration: np.ndarray | None = None,
    n_observations: int | None = None,
):
    p=load_paper_config(ROOT,dataset)
    standardize=(dataset != "tweeteval") and not prestandardized
    if name in {"proposed","fixed_rank"}:
        projection_mode="identity" if dataset=="tweeteval" and tweet_preprojected else None
        cfg=asc_config_from_paper(ROOT,dataset,d,seed,fixed_rank=(name=="fixed_rank"),projection_mode=projection_mode)
        if projection_mode=="identity": cfg.projection_dim=d
        base=AdaptiveSketchStreamClusterer(cfg,standardize=standardize)
        # Use the array-backed execution-equivalent engine for complete-stream
        # runs. It keeps the same projection/sketch/rank controller and the
        # same bounded micro-cluster update, while removing Python list costs.
        return OptimizedAdaptiveSketchClustererArray(base), cfg.to_dict()
    if name in {'clustream','denstream','streamkmpp'}:
        b=p['baselines'][name]
        if name=='clustream':
            return CluStreamBaseline(d,float(b['radius']),int(b['max_microclusters']),seed,standardize=standardize), dict(b)
        if name=='denstream':
            return DenStreamBaseline(d,float(b['epsilon']),int(b['max_microclusters']),seed,standardize=standardize,beta=float(b['beta']),mu=float(b['mu']),fading_lambda=float(b['lambda'])), dict(b)
        return StreamKMPlusPlusBaseline(d,int(b['coreset_size']),int(b['buffer_size']),seed,standardize=standardize), dict(b)
    modern = _modern_config()["methods"]
    if name == "twstream":
        if calibration is None:
            raise ValueError("TWStream requires the common preprocessed calibration prefix")
        settings = modern["twstream"]
        radius = _calibration_radius(calibration)
        params = {
            "k_neighbors": int(settings["k_neighbors"]),
            "lambda": float(settings["lambda"]),
            "radius_rule": "80th percentile of calibration-prefix 6-NN distance",
            "calibrated_radius": radius,
            "max_clusters": 200,
            "max_outliers": 200,
        }
        return TWStreamComparator(
            d,
            seed,
            max_clusters=params["max_clusters"],
            max_outliers=params["max_outliers"],
            radius=radius,
            k=params["k_neighbors"],
            lam=params["lambda"],
        ), params
    if name == "fra_art":
        settings = modern["fra_art"]
        params = {
            "a": float(settings["a"]),
            "vigilance": float(settings["vigilance"]),
            "choice": float(settings["choice"]),
            "beta": float(settings["beta"]),
            "input_transform": "calibration-prefix min-max after common dataset preprocessing",
        }
        return FRAARTComparator(
            d,
            seed,
            a=params["a"],
            vigilance=params["vigilance"],
            choice=params["choice"],
            beta=params["beta"],
            max_stream_points=max(int(n_observations or 1), 1),
        ), params
    raise ValueError(name)


def method_input_transform(name: str, calibration: np.ndarray):
    """Return the fixed method-required transform after common preprocessing.

    All seven methods receive the same stream order and common dataset
    preprocessing. FRA-ART additionally requires bounded [0, 1] inputs for its
    fractional/complement encoding; its bounds are fixed from the same declared
    calibration prefix and never use labels.
    """
    if name == "fra_art":
        lo = np.min(calibration, axis=0)
        hi = np.max(calibration, axis=0)

        def transform(block):
            dense = _dense_view(block)
            return np.clip((dense - lo) / (hi - lo + 1e-12), 0.0, 1.0)

        return transform
    if name == "twstream":
        return _dense_view
    return None


def _is_asc(model):
    return hasattr(model, "sketch") and hasattr(model, "config")

def _adapt_block(model, z):
    return model.sketch.adapt_matrix(z) if _is_asc(model) else z


def safe_weighted_kmeans(centers: np.ndarray, weights: np.ndarray, k: int, seed: int, n_init: int = 1, max_iter: int = 20) -> np.ndarray:
    """Deterministic weighted Lloyd evaluator used only by the paper multi-dataset runner.

    It avoids OpenMP-dependent KMeans stalls on degenerate micro-cluster states
    while implementing the manuscript's declared weighted K-means macro step.
    """
    x=np.asarray(centers,dtype=np.float64); w=np.maximum(np.asarray(weights,dtype=np.float64),1e-12)
    kk=min(int(k),len(x))
    if kk<=1: return np.average(x,axis=0,weights=w,keepdims=True)
    best=None; best_obj=np.inf
    for init_id in range(int(n_init)):
        rng=np.random.default_rng(int(seed)+104729*init_id)
        chosen=[]
        p0=w/w.sum(); first=int(rng.choice(len(x),p=p0)); chosen.append(first)
        c_list=[x[first].copy()]
        min_d2=np.sum((x-x[first])**2,axis=1)
        while len(c_list)<kk:
            score=w*np.maximum(min_d2,0.0); total=float(score.sum())
            if total<=1e-20 or not np.isfinite(total):
                remaining=np.asarray([i for i in range(len(x)) if i not in chosen],dtype=np.int64)
                j=int(remaining[0]) if len(remaining) else int(chosen[-1])
            else:
                j=int(rng.choice(len(x),p=score/total))
                if j in chosen:
                    remaining=np.asarray([i for i in range(len(x)) if i not in chosen],dtype=np.int64)
                    if len(remaining): j=int(remaining[np.argmax(score[remaining])])
            chosen.append(j); c_list.append(x[j].copy())
            min_d2=np.minimum(min_d2,np.sum((x-x[j])**2,axis=1))
        c=np.vstack(c_list)
        for _ in range(int(max_iter)):
            x2=np.sum(x*x,axis=1)[:,None]; c2=np.sum(c*c,axis=1)[None,:]
            d2=np.maximum(x2+c2-2.0*x@c.T,0.0); lab=np.argmin(d2,axis=1)
            new=c.copy()
            for j in range(kk):
                mask=(lab==j)
                if np.any(mask): new[j]=np.average(x[mask],axis=0,weights=w[mask])
            shift=float(np.sum((new-c)*(new-c)))
            c=new
            if shift <= 1e-10: break
        d2=np.maximum(np.sum(x*x,axis=1)[:,None]+np.sum(c*c,axis=1)[None,:]-2.0*x@c.T,0.0)
        obj=float(np.sum(w*np.min(d2,axis=1)))
        if obj < best_obj: best_obj=obj; best=c.copy()
    return best


def paper_macro_centers(model, k: int, seed: int) -> np.ndarray:
    if _is_asc(model):
        centers,weights=model.microclusters.centers_weights(model.clustering_basis)
        return safe_weighted_kmeans(centers,weights,k,seed)
    if isinstance(model,CluStreamBaseline):
        centers,weights=model.microclusters.adapted_centers_and_weights(model.basis)
        return safe_weighted_kmeans(centers,weights,k,seed)
    if isinstance(model,DenStreamBaseline):
        active=model.p_micro if model.p_micro else model.o_micro
        if not active: raise RuntimeError('DenStream has no active micro-clusters')
        for c in active: model._decay_to(c,model.time)
        centers=np.vstack([c.center for c in active]); weights=np.asarray([max(c.weight,1e-12) for c in active])
        return safe_weighted_kmeans(centers,weights,k,seed)
    if isinstance(model,StreamKMPlusPlusBaseline):
        model._compress(); centers=np.vstack(model.values); weights=np.asarray(model.weights,dtype=np.float64)
        return safe_weighted_kmeans(centers,weights,k,seed)
    return model.macro_centers(k)


def _snapshot_summary(model):
    """Copy the current online summary without changing future stream state."""
    if _is_asc(model):
        basis=model.clustering_basis.copy()
        centers,weights=model.microclusters.centers_weights(basis)
        return centers.copy(),weights.copy(),basis
    if isinstance(model,CluStreamBaseline):
        centers,weights=model.microclusters.adapted_centers_and_weights(model.basis)
        return centers.copy(),weights.copy(),None
    if isinstance(model,DenStreamBaseline):
        active=model.p_micro if model.p_micro else model.o_micro
        if not active:
            return np.zeros((1,model.original_dim)),np.ones(1),None
        # Bring the copied summary to the current stream time.  Do not call
        # macro_centers(), which mutates the live list during evaluation.
        centers=[];weights=[]
        for c in active:
            dt=max(model.time-c.last_time,0); fac=2.0**(-model.fading_lambda*dt)
            w=max(c.weight*fac,1e-12)
            centers.append((c.linear_sum*fac)/w);weights.append(w)
        return np.vstack(centers),np.asarray(weights,dtype=np.float64),None
    if isinstance(model,StreamKMPlusPlusBaseline):
        # Evaluation must not force an extra coreset compression.  The online
        # process already performs merge-reduce when buffer_size is reached.
        if not model.values:
            return np.zeros((1,model.original_dim)),np.ones(1),None
        return np.vstack(model.values),np.asarray(model.weights,dtype=np.float64),None
    if isinstance(model,TWStreamComparator):
        if model.m:
            age=np.maximum(model.t-model.last[:model.m],0)
            weights=model.weights[:model.m]*(2.0**(-model.lam*age))
            weights=weights*model._boundary_confidence()
            return model.centers[:model.m].copy(),np.maximum(weights,1e-12),None
        if model.mo:
            age=np.maximum(model.t-model.out_last[:model.mo],0)
            weights=model.out_weights[:model.mo]*(2.0**(-model.lam*age))
            return model.out_centers[:model.mo].copy(),np.maximum(weights,1e-12),None
        return np.zeros((1,model.d)),np.ones(1),None
    if isinstance(model,FRAARTComparator):
        if model.m:
            return model.category_centers().copy(),model.counts[:model.m].copy(),None
        return np.zeros((1,model.d)),np.ones(1),None
    raise TypeError(type(model))


def evaluate_full_stream(
    model,
    x,
    y: np.ndarray,
    k: int,
    *,
    chunk_size: int,
    work_dir: Path,
    preprocess_block=None,
) -> dict:
    n=int(x.shape[0])
    out_dim=model.config.projection_dim if _is_asc(model) else int(x.shape[1])
    seed=int(getattr(getattr(model,"config",None),"seed",getattr(model,"seed",7)))
    work_dir.mkdir(parents=True,exist_ok=True)
    mmap_path=work_dir/'stream_representation.f32'
    rep=np.memmap(mmap_path,dtype=np.float32,mode='w+',shape=(n,out_dim))
    snapshots=[]
    online_seconds=0.0; macro_seconds=0.0
    t_all=perf_counter()
    for start in range(0,n,chunk_size):
        end=min(start+chunk_size,n)
        block=x[start:end] if sparse.issparse(x) else np.asarray(x[start:end])
        if preprocess_block is not None:
            block=preprocess_block(block)
        t=perf_counter(); z=model.process_batch(block); online_seconds+=perf_counter()-t
        rep[start:end]=z.astype(np.float32,copy=False)
        if start>0:
            sc,sw,sb=_snapshot_summary(model)
            snapshots.append((start,end,sc,sw,sb))
        if os.environ.get("ASC_PROGRESS") == "1" and (end==n or end%(50*chunk_size)==0):
            print(f"online {end}/{n} wall={perf_counter()-t_all:.2f}s",file=sys.stderr,flush=True)
    rep.flush()

    # Deferred chunk evaluation: identical 1000-point boundaries, but the
    # evaluator cannot mutate or stall the live online state.
    chunk_ari=[];chunk_nmi=[]
    for idx,(start,end,sc,sw,sb) in enumerate(snapshots):
        t=perf_counter(); centers=safe_weighted_kmeans(sc,sw,k,seed); t_km=perf_counter()
        z=np.asarray(rep[start:end],dtype=np.float64)
        ze=(z@sb)@sb.T if sb is not None else z
        pred=nearest_assign(ze,centers); t_pred=perf_counter(); macro_seconds+=t_pred-t
        aa,nn=ari_nmi(y[start:end],pred);chunk_ari.append(float(aa));chunk_nmi.append(float(nn)); t_met=perf_counter()
        if os.environ.get("ASC_PROGRESS") == "1" and (idx<5 or (idx+1)%100==0 or idx+1==len(snapshots)):
            print(f"eval {idx+1}/{len(snapshots)} km={t_km-t:.4f} pred={t_pred-t_km:.4f} metric={t_met-t_pred:.4f} wall={t_met-t_all:.2f}s m={len(sc)}",file=sys.stderr,flush=True)

    # Complete-window partition with final online summary.
    sc,sw,sb=_snapshot_summary(model)
    t=perf_counter(); centers=safe_weighted_kmeans(sc,sw,k,seed)
    pred_all=np.empty(n,dtype=np.int32)
    for start in range(0,n,chunk_size):
        end=min(start+chunk_size,n)
        z=np.asarray(rep[start:end],dtype=np.float64)
        ze=(z@sb)@sb.T if sb is not None else z
        pred_all[start:end]=nearest_assign(ze,centers)
    macro_seconds+=perf_counter()-t
    global_ari,global_nmi=ari_nmi(y,pred_all)
    elapsed=perf_counter()-t_all
    diagnostics=model.telemetry() if _is_asc(model) else model.diagnostics()
    try:
        rep.flush()
        if getattr(rep,'_mmap',None) is not None: rep._mmap.close()
    finally:
        del rep
    try:mmap_path.unlink()
    except OSError:pass
    return {
        'online_chunk_mean_ari':float(np.mean(chunk_ari)) if chunk_ari else None,
        'online_chunk_mean_nmi':float(np.mean(chunk_nmi)) if chunk_nmi else None,
        'complete_window_final_ari':float(global_ari),'complete_window_final_nmi':float(global_nmi),
        'evaluated_observations':n,'observation_coverage':1.0,
        'online_seconds':online_seconds,'macro_seconds':macro_seconds,'elapsed_seconds':elapsed,
        'runtime_ms_per_1000':1_000_000.0*elapsed/max(n,1),
        'diagnostics':diagnostics,
        'evaluation_engine':'deferred_nonmutating_weighted_lloyd',
        'evaluation_chunk_size':int(chunk_size),
        'initialization_excluded_observations':int(min(chunk_size,n)),
    }


def main():
    ap=argparse.ArgumentParser(description='Direct full-stream execution runner. Outputs are isolated from protected manuscript results.')
    ap.add_argument('--dataset',required=True,choices=list(DISPLAY))
    ap.add_argument('--method',required=True,choices=METHODS)
    ap.add_argument('--seed',type=int,default=7)
    ap.add_argument('--chunk-size',type=int,default=1000)
    ap.add_argument('--source',choices=['study','canonical'],default='study',help='study uses packaged SHA-256-verified assets; canonical requires prepared public-source assets where applicable')
    ap.add_argument('--max-observations',type=int,default=None,help='Optional smoke-test limit; full study executions omit this option.')
    ap.add_argument('--out',type=Path,default=None)
    a=ap.parse_args()
    out=a.out or (ROOT/'results'/'execution_runs'/'primary'/a.dataset/a.method/f'seed_{a.seed}.json')
    out=ensure_execution_output(ROOT,out)
    loaded=(load_packaged_study_dataset(ROOT,a.dataset) if a.source=='study' else load_full_paper_dataset(ROOT,a.dataset,allow_packaged_tweeteval=False))
    x,y=loaded.x,loaded.y
    full_stream=True
    if a.max_observations is not None:
        x=x[:a.max_observations];y=y[:a.max_observations];full_stream=False
    prestandardized=False
    if a.dataset != "tweeteval" and not sparse.issparse(x):
        # The paper declares one common causal standardization pipeline.  Apply
        # it once, in exact Welford order, before the method-specific online
        # state.  Dataset loading/preprocessing remains outside algorithm timing.
        x=causal_standardize_dense(np.asarray(x,dtype=np.float64))
        prestandardized=True
    calibration_rows=min(1000,int(x.shape[0]))
    calibration=_dense_view(x[:calibration_rows])
    model,params=build_method(
        a.method,
        a.dataset,
        int(x.shape[1]),
        a.seed,
        tweet_preprojected=(a.dataset=='tweeteval' and a.source=='study'),
        prestandardized=prestandardized,
        calibration=calibration,
        n_observations=int(x.shape[0]),
    )
    preprocess_block=method_input_transform(a.method,calibration)
    tmp_root=ROOT/'results'/'execution_runs'/'_tmp'; tmp_root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='asc_primary_eval_',dir=str(tmp_root)) as td:
        metrics=evaluate_full_stream(
            model,
            x,
            y,
            N_CLUSTERS[a.dataset],
            chunk_size=a.chunk_size,
            work_dir=Path(td),
            preprocess_block=preprocess_block,
        )
    result={
        'schema':'unified_primary_run_v1','protocol_id':'paper_aligned_seven_method_full_stream_v1',
        'dataset':a.dataset,'display':DISPLAY[a.dataset],'method':a.method,'seed':a.seed,
        'full_stream':full_stream and int(x.shape[0])==int(loaded.y.shape[0]),
        'public_source_verified':bool(loaded.canonical) and a.max_observations is None,'source_mode':a.source,
        'data_source':loaded.source,'evidence_note':loaded.note,'parameters':params,'implementation':method_metadata(a.method),
        'common_protocol':{
            'stream_order':'unchanged',
            'dataset_preprocessing':('identity packaged representation' if a.dataset=='tweeteval' else 'causal Welford standardization'),
            'calibration_rows':calibration_rows,
            'evaluation':'complete-window final partition plus non-mutating online chunk summaries',
            'labels_used_online':False,
        },
        'execution_environment':{
            'python':sys.version,
            'platform':platform.platform(),
            'numpy':np.__version__,
            'thread_environment':{key:os.environ.get(key) for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')},
            'execution_fingerprint_sha256':_execution_fingerprint(),
        },**metrics,
    }
    if not result['full_stream']: result['evidence_note']+='; smoke-test subset requested; manuscript study results use full streams'
    print(json.dumps(result,indent=2))
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__':main()
