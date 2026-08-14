from __future__ import annotations

import numpy as np
from scipy import sparse
from .comparators import weighted_macro_centers

try:
    from numba import njit
except Exception:  # pragma: no cover - pure NumPy fallback remains available
    njit = None


def _causal_standardize_dense_python(x, eps=1e-8):
    x=np.asarray(x,dtype=np.float64)
    out=np.empty_like(x); mean=np.zeros(x.shape[1],np.float64); m2=np.zeros(x.shape[1],np.float64); count=0
    for i in range(x.shape[0]):
        row=x[i]
        if count < 2: out[i]=row-mean
        else: out[i]=(row-mean)/np.sqrt(m2/max(count-1,1)+eps)
        count+=1; delta=row-mean; mean+=delta/count; delta2=row-mean; m2+=delta*delta2
    return np.nan_to_num(out,copy=False)

if njit is not None:
    @njit(cache=True)
    def _causal_standardize_dense_numba(x, eps=1e-8):
        n,d=x.shape; out=np.empty((n,d),np.float64); mean=np.zeros(d,np.float64); m2=np.zeros(d,np.float64); count=0
        for i in range(n):
            if count < 2:
                for j in range(d): out[i,j]=x[i,j]-mean[j]
            else:
                den=count-1
                for j in range(d): out[i,j]=(x[i,j]-mean[j])/np.sqrt(m2[j]/den+eps)
            count += 1
            for j in range(d):
                delta=x[i,j]-mean[j]; mean[j]+=delta/count; delta2=x[i,j]-mean[j]; m2[j]+=delta*delta2
        return out
else:
    _causal_standardize_dense_numba=None

def causal_standardize_dense(x, eps=1e-8):
    """Exact-order Welford causal standardization for a complete dense stream."""
    a=np.ascontiguousarray(x,dtype=np.float64)
    if _causal_standardize_dense_numba is not None: return _causal_standardize_dense_numba(a,float(eps))
    return _causal_standardize_dense_python(a,float(eps))

class OptimizedAdaptiveSketchClusterer:
    """Execution-optimized wrapper around the ASC online state machine.
    It preserves the same projection, sketch, rank-adaptation, and micro-clustering logic but
    batches the fixed linear projection and removes per-point diagnostic timer calls.
    """
    def __init__(self, base_model):
        self.base=base_model
        self.omega=base_model.omega; self.sketch=base_model.sketch; self.microclusters=base_model.microclusters
        self.config=base_model.config; self.time=0; self.rank_history=[]; self.error_history=[]; self.threshold_history=[]; self.rank_change_history=[]; self.basis_update_history=[]
    @property
    def basis(self):return self.sketch.basis[:,:self.sketch.rank]
    @property
    def clustering_basis(self):
        if self.config.use_adapted_representation_for_clustering:return self.basis
        return np.eye(self.config.projection_dim,dtype=np.float64)
    def process_batch(self,x):
        if sparse.issparse(x): Z=np.asarray(x @ self.omega,dtype=np.float64)
        else: Z=np.asarray(x,dtype=np.float64)@self.omega
        for z in Z:
            self.time+=1
            u=self.sketch.update(z)
            za=self.sketch.adapted(z) if self.config.use_adapted_representation_for_clustering else z
            sw=u.effective_weight if self.config.leverage_mode=='weight' else 1.0
            rs=u.leverage_weight**(-self.config.leverage_radius_strength) if self.config.leverage_mode=='weight' else 1.0
            self.microclusters.update(z,za,self.clustering_basis,self.time,sample_weight=sw,radius_scale=rs)
            self.rank_history.append(self.sketch.rank); self.error_history.append(float(u.error)); self.threshold_history.append(float(u.threshold)); self.rank_change_history.append(bool(u.rank_changed)); self.basis_update_history.append(bool(u.updated))
        return Z
    def macro_centers(self,k):
        centers,weights=self.microclusters.adapted_centers_and_weights(self.clustering_basis)
        return weighted_macro_centers(centers,weights,k,self.config.seed)

    def telemetry(self):
        ranks=np.asarray(self.rank_history,dtype=np.int64)
        return {
            'observations':int(self.time),
            'rank_mean':float(ranks.mean()) if ranks.size else float('nan'),
            'rank_std_within_run':float(ranks.std(ddof=0)) if ranks.size else float('nan'),
            'rank_min_observed':int(ranks.min()) if ranks.size else int(self.sketch.rank),
            'rank_max_observed':int(ranks.max()) if ranks.size else int(self.sketch.rank),
            'rank_change_count':int(np.count_nonzero(np.diff(ranks))) if ranks.size>1 else 0,
            'final_rank':int(self.sketch.rank),
            'sketch_seen':int(self.sketch.seen),
            'sketch_accepted':int(self.sketch.accepted),
            'sampling_acceptance':float(self.sketch.accepted/max(self.sketch.seen,1)),
            'basis_updates':int(np.count_nonzero(self.basis_update_history)),
            'final_microclusters':int(len(self.microclusters.clusters)),
            'basis_update_interval_semantics':'accepted sketch rows',
            'execution_engine':'array_optimized_equivalent',
        }

class ProjectedMicroClusterArray:
    """Array-backed implementation equivalent to OnlineMicroClusterSet in the adaptive projected space.
    Distances are computed in orthonormal low-rank coordinates and cached between basis refreshes.
    """
    def __init__(self,dim:int,radius:float,max_clusters:int,decay:float,prune_policy:str='utility'):
        self.dim=dim;self.radius=float(radius);self.max_clusters=int(max_clusters);self.decay=float(decay);self.prune_policy=prune_policy
        self.centers=np.zeros((max_clusters,dim),np.float64);self.weights=np.zeros(max_clusters,np.float64);self.last=np.zeros(max_clusters,np.int64);self.created=np.zeros(max_clusters,np.int64);self.ids=np.zeros(max_clusters,np.int64)
        self.m=0;self.next_id=0;self.pruned=0;self.merged=0;self.coords=np.empty((0,0),np.float64);self._basis_cols=0
    def refresh_basis(self,basis):
        self.coords=self.centers[:self.m]@basis if self.m else np.empty((0,basis.shape[1]),np.float64);self._basis_cols=int(basis.shape[1])
    def _ensure(self,basis,basis_changed=False):
        # The optimized ASC wrapper knows exactly when AdaptiveProjectedSketch
        # refreshes its basis (BasisUpdate.updated).  Avoid hashing the complete
        # basis for every stream observation.
        if basis_changed or self.coords.shape!=(self.m,basis.shape[1]): self.refresh_basis(basis)
    def _delete(self,j,basis):
        if j<self.m-1:
            self.centers[j:self.m-1]=self.centers[j+1:self.m];self.weights[j:self.m-1]=self.weights[j+1:self.m];self.last[j:self.m-1]=self.last[j+1:self.m];self.created[j:self.m-1]=self.created[j+1:self.m];self.ids[j:self.m-1]=self.ids[j+1:self.m]
            self.coords[j:self.m-1]=self.coords[j+1:self.m]
        self.m-=1
        self.coords=self.coords[:self.m].copy()
    def update(self,z_raw,z_adapted,basis,now,sample_weight=1.,radius_scale=1.,basis_changed=False):
        sw=max(float(sample_weight),1e-12);self._ensure(basis,basis_changed=basis_changed)
        if self.m==0:
            j=0;self.m=1;self.centers[j]=z_raw;self.weights[j]=sw;self.last[j]=now;self.created[j]=now;self.ids[j]=self.next_id;self.next_id+=1;self.coords=np.asarray([z_raw@basis],dtype=np.float64);return int(self.ids[j])
        q=z_raw@basis;ds=np.linalg.norm(self.coords-q,axis=1);j=int(np.argmin(ds))
        if ds[j] <= self.radius*max(float(radius_scale),1e-6):
            ow=self.decay*self.weights[j];nw=ow+sw;self.centers[j]=(ow*self.centers[j]+sw*z_raw)/max(nw,1e-12);self.weights[j]=nw;self.last[j]=now;self.coords[j]=self.centers[j]@basis;return int(self.ids[j])
        if self.m>=self.max_clusters:
            if self.prune_policy=='merge':
                # exact closest pair in cached low-rank coordinates
                G=np.sum(self.coords*self.coords,axis=1);d2=np.maximum(G[:,None]+G[None,:]-2*self.coords@self.coords.T,0);np.fill_diagonal(d2,np.inf);a,b=np.unravel_index(np.argmin(d2),d2.shape);a,b=sorted((int(a),int(b)));tot=self.weights[a]+self.weights[b];self.centers[a]=(self.weights[a]*self.centers[a]+self.weights[b]*self.centers[b])/max(tot,1e-12);self.weights[a]=tot;self.last[a]=max(self.last[a],self.last[b],now);self.coords[a]=self.centers[a]@basis;self._delete(b,basis);self.merged+=1
            else:
                scores=self.weights[:self.m]/(1.+np.maximum(now-self.last[:self.m],0));self._delete(int(np.argmin(scores)),basis);self.pruned+=1
        j=self.m;self.m+=1;self.centers[j]=z_raw;self.weights[j]=sw;self.last[j]=now;self.created[j]=now;self.ids[j]=self.next_id;self.next_id+=1;new_coord=(z_raw@basis).reshape(1,-1);self.coords=np.vstack((self.coords,new_coord));return int(self.ids[j])
    def centers_weights(self,basis):
        self._ensure(basis,basis_changed=False);return self.coords@basis.T,self.weights[:self.m].copy()
    @property
    def clusters(self):
        # compatibility only for len(); do not allocate normal cluster objects.
        return range(self.m)

class OptimizedAdaptiveSketchClustererArray(OptimizedAdaptiveSketchClusterer):
    def __init__(self,base_model):
        super().__init__(base_model)
        self.microclusters=ProjectedMicroClusterArray(base_model.config.projection_dim,base_model.config.microcluster_radius,base_model.config.max_microclusters,base_model.config.decay,base_model.config.prune_policy)
        self.base.microclusters=self.microclusters
    def process_batch(self,x):
        if sparse.issparse(x): Z=np.asarray(x @ self.omega,dtype=np.float64)
        else: Z=np.asarray(x,dtype=np.float64)@self.omega
        for z in Z:
            self.time+=1;u=self.sketch.update(z);basis=self.clustering_basis
            za=(z@basis)@basis.T if self.config.use_adapted_representation_for_clustering else z
            sw=u.effective_weight if self.config.leverage_mode=='weight' else 1.;rs=u.leverage_weight**(-self.config.leverage_radius_strength) if self.config.leverage_mode=='weight' else 1.
            self.microclusters.update(z,za,basis,self.time,sw,rs,basis_changed=bool(u.updated));self.rank_history.append(self.sketch.rank); self.error_history.append(float(u.error)); self.threshold_history.append(float(u.threshold)); self.rank_change_history.append(bool(u.rank_changed)); self.basis_update_history.append(bool(u.updated))
        return Z
    def macro_centers(self,k):
        centers,weights=self.microclusters.centers_weights(self.clustering_basis);return weighted_macro_centers(centers,weights,k,self.config.seed)


