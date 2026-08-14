"""Recent comparative methods under the manuscript's common evaluator.

These are source-aligned study implementations, not claims of byte-for-byte
reproduction of the authors' native end-to-end software.  TWStream preserves
its online active/outlier summaries, augmented-kNN structure and boundary
confidence ingredients; FRA-ART uses the declared SIBF/Fuzzy-ART variant.
Both expose online summaries to the study's common macro evaluator, exactly as
declared in Section 5.2.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.cluster import KMeans
try:
    from numba import njit
except Exception:
    njit = None

if njit is not None:
    @njit(cache=True)
    def _fra_batch_numba(P, counts, raw_sums, m, X, a, rho, alpha, beta):
        d = X.shape[1]
        p = 4*d
        I = np.empty(p, np.float64)
        for rr in range(X.shape[0]):
            for q in range(d):
                v=X[rr,q]
                if v<0.0: v=0.0
                elif v>1.0: v=1.0
                f=v**a
                I[q]=v; I[d+q]=f; I[2*d+q]=1.0-v; I[3*d+q]=1.0-f
            if m==0:
                for q in range(p): P[0,q]=I[q]
                counts[0]=1.0
                for q in range(d): raw_sums[0,q]=X[rr,q]
                m=1
                continue
            normI=0.0
            for q in range(p): normI += I[q]
            if normI<1e-12: normI=1e-12
            chosen=-1; bestT=-1.0e300
            for j in range(m):
                numer=0.0; denom=0.0
                for q in range(p):
                    pv=P[j,q]; iv=I[q]
                    denom += pv
                    numer += pv if pv<iv else iv
                if numer/normI >= rho:
                    T=numer/(alpha+denom)
                    # Equivalent to argsort(T)[::-1] followed by first resonant category.
                    # For exact ties, prefer the larger index as reversed argsort does.
                    if T>bestT or (T==bestT and j>chosen):
                        bestT=T; chosen=j
            if chosen<0:
                j=m
                for q in range(p): P[j,q]=I[q]
                counts[j]=1.0
                for q in range(d): raw_sums[j,q]=X[rr,q]
                m += 1
            else:
                j=chosen
                for q in range(p):
                    mn=P[j,q] if P[j,q]<I[q] else I[q]
                    P[j,q]=(1.0-beta)*P[j,q]+beta*mn
                counts[j]+=1.0
                for q in range(d): raw_sums[j,q]+=X[rr,q]
        return m
else:
    _fra_batch_numba = None



def nearest_assign(x: np.ndarray, centers: np.ndarray, block: int = 2048) -> np.ndarray:
    if len(centers) == 0:
        return np.zeros(len(x), dtype=np.int64)
    out = np.empty(len(x), dtype=np.int64)
    c2 = np.sum(centers * centers, axis=1)[None, :]
    for s in range(0, len(x), block):
        xb = x[s:s+block]
        x2 = np.sum(xb * xb, axis=1)[:, None]
        d2 = np.maximum(x2 + c2 - 2.0 * xb @ centers.T, 0.0)
        out[s:s+len(xb)] = np.argmin(d2, axis=1)
    return out


def weighted_macro_centers(centers: np.ndarray, weights: np.ndarray, k: int, seed: int) -> np.ndarray:
    if len(centers) == 0:
        return np.zeros((1, 1), dtype=np.float64)
    kk = min(int(k), len(centers))
    if kk <= 1:
        return np.average(centers, axis=0, weights=np.maximum(weights, 1e-12), keepdims=True)
    km = KMeans(n_clusters=kk, random_state=seed, n_init=10, algorithm='lloyd')
    km.fit(centers, sample_weight=np.maximum(weights, 1e-12))
    return km.cluster_centers_


def arr_bytes(*arrays) -> int:
    n = 0
    for a in arrays:
        if isinstance(a, np.ndarray):
            n += int(a.nbytes)
    return n



class TWStreamComparator:
    """TWStream source-aligned online-stage reconstruction for the common protocol.

    The native paper includes a three-way offline clustering engine.  In this
    study implementation, online summaries and boundary confidence are retained
    while final macro extraction is intentionally replaced by the manuscript's
    common weighted evaluator; therefore this class must not be described as a
    native end-to-end TWStream reproduction.
    """
    def __init__(self,d:int,seed:int,max_clusters=200,max_outliers=200,radius=10.5,k=8,lam=.0028):
        self.d=d; self.seed=seed; self.max_clusters=max_clusters; self.max_outliers=max_outliers; self.radius=float(radius); self.k=min(k,max_clusters-1)
        self.lam=float(lam)
        self.centers=np.zeros((max_clusters,d),np.float64); self.weights=np.zeros(max_clusters,np.float64); self.last=np.zeros(max_clusters,np.int64); self.m=0
        self.out_centers=np.zeros((max_outliers,d),np.float64); self.out_weights=np.zeros(max_outliers,np.float64); self.out_last=np.zeros(max_outliers,np.int64); self.mo=0
        self.neigh=np.full((max_clusters,self.k),-1,np.int32); self.ndist=np.full((max_clusters,self.k),np.inf,np.float64)
        self.t=0; self.absorbed=0; self.created=0; self.promoted=0; self.replaced_outlier=0
    def _decay(self,w,last): return w*(2.0**(-self.lam*max(self.t-int(last),0)))
    def _refresh_row(self,i):
        if self.m<=1:
            self.neigh[i].fill(-1); self.ndist[i].fill(np.inf); return
        ds=np.linalg.norm(self.centers[:self.m]-self.centers[i],axis=1);ds[i]=np.inf
        kk=min(self.k,self.m-1); ids=np.argpartition(ds,kk-1)[:kk];ids=ids[np.argsort(ds[ids])]
        self.neigh[i].fill(-1);self.ndist[i].fill(np.inf);self.neigh[i,:kk]=ids;self.ndist[i,:kk]=ds[ids]
    def _update_graph(self,i):
        if self.m<=1:self._refresh_row(i);return
        ds=np.linalg.norm(self.centers[:self.m]-self.centers[i],axis=1);ds[i]=np.inf
        self._refresh_row(i)
        # Update only entries whose relation to i changed; periodic exact row refresh is unnecessary
        # because this augmented graph is maintained incrementally.
        for j in range(self.m):
            if j==i:continue
            row=self.neigh[j]; pos=np.where(row==i)[0]
            if len(pos):
                self.ndist[j,pos[0]]=ds[j]
                order=np.argsort(self.ndist[j]);self.ndist[j]=self.ndist[j,order];self.neigh[j]=self.neigh[j,order]
            elif ds[j] < self.ndist[j,-1]:
                self.ndist[j,-1]=ds[j];self.neigh[j,-1]=i
                order=np.argsort(self.ndist[j]);self.ndist[j]=self.ndist[j,order];self.neigh[j]=self.neigh[j,order]
    def _promote(self,z,w,last):
        if self.m>=self.max_clusters:
            # Keep bounded active graph: demote lowest current-density active MC to outlier pool.
            dec=self.weights[:self.m]*(2.0**(-self.lam*np.maximum(self.t-self.last[:self.m],0)));j=int(np.argmin(dec))
            self._put_outlier(self.centers[j].copy(),float(dec[j]),int(self.last[j]))
            self.centers[j]=z;self.weights[j]=w;self.last[j]=last
            # identity remains j, refresh its graph relation.
            self._update_graph(j);return
        j=self.m;self.m+=1;self.centers[j]=z;self.weights[j]=w;self.last[j]=last;self._update_graph(j)
    def _put_outlier(self,z,w=1.0,last=None):
        last=self.t if last is None else last
        if self.mo<self.max_outliers:
            j=self.mo;self.mo+=1
        else:
            dec=self.out_weights[:self.mo]*(2.0**(-self.lam*np.maximum(self.t-self.out_last[:self.mo],0)));j=int(np.argmin(dec));self.replaced_outlier+=1
        self.out_centers[j]=z;self.out_weights[j]=w;self.out_last[j]=last
    def process_one(self,z):
        self.t+=1
        if self.m:
            ds=np.linalg.norm(self.centers[:self.m]-z,axis=1);j=int(np.argmin(ds))
            if ds[j]<=self.radius:
                w=self._decay(self.weights[j],self.last[j]);nw=w+1.;self.centers[j]=(w*self.centers[j]+z)/nw;self.weights[j]=nw;self.last[j]=self.t;self.absorbed+=1;self._update_graph(j);return
        if self.mo:
            ds=np.linalg.norm(self.out_centers[:self.mo]-z,axis=1);j=int(np.argmin(ds))
            if ds[j]<=self.radius:
                w=self._decay(self.out_weights[j],self.out_last[j]);nw=w+1.;self.out_centers[j]=(w*self.out_centers[j]+z)/nw;self.out_weights[j]=nw;self.out_last[j]=self.t
                # W_min=2 promotion threshold.
                if nw>=2.0:
                    zz=self.out_centers[j].copy();ww=float(nw);ll=int(self.out_last[j]);
                    if j<self.mo-1:
                        self.out_centers[j:self.mo-1]=self.out_centers[j+1:self.mo];self.out_weights[j:self.mo-1]=self.out_weights[j+1:self.mo];self.out_last[j:self.mo-1]=self.out_last[j+1:self.mo]
                    self.mo-=1;self.promoted+=1;self._promote(zz,ww,ll)
                return
        self.created+=1;self._put_outlier(z.copy(),1.0,self.t)
    def process_batch(self,x):
        for z in x:self.process_one(z)
        return x
    def _boundary_confidence(self):
        conf=np.ones(self.m,np.float64)
        if self.m<=1:return conf
        dw=self.weights[:self.m]*(2.0**(-self.lam*np.maximum(self.t-self.last[:self.m],0)));medw=np.median(dw)
        for i in range(self.m):
            ids=self.neigh[i];ids=ids[ids>=0]
            if len(ids)==0:continue
            vec=self.centers[ids]-self.centers[i];pos=np.mean(vec>0,axis=0);skew=float(np.mean(np.abs(pos-.5))*2.)
            spars=float(np.mean(self.ndist[i,:len(ids)])/(self.radius+1e-12));evol=float(abs(dw[i]-medw)/(medw+1e-12))
            conf[i]=1.-(skew+min(spars,2.)/2.+min(evol,2.)/2.)/3.
        return np.clip(conf,.05,1.)
    def macro_centers(self,k):
        if self.m==0:
            if self.mo:return weighted_macro_centers(self.out_centers[:self.mo],self.out_weights[:self.mo],k,self.seed)
            return np.zeros((1,self.d))
        dw=self.weights[:self.m]*(2.0**(-self.lam*np.maximum(self.t-self.last[:self.m],0)));return weighted_macro_centers(self.centers[:self.m],dw*self._boundary_confidence(),k,self.seed)
    def state_components(self):
        active=self.centers[:self.m].nbytes+self.weights[:self.m].nbytes+self.last[:self.m].nbytes
        out=self.out_centers[:self.mo].nbytes+self.out_weights[:self.mo].nbytes+self.out_last[:self.mo].nbytes
        graph=self.neigh[:self.m].nbytes+self.ndist[:self.m].nbytes;total=active+out+graph
        return {'active_microcluster_bytes':active,'outlier_pool_bytes':out,'knn_graph_bytes':graph,'persistent_numeric_bytes':total,'allocated_numeric_bytes':arr_bytes(self.centers,self.weights,self.last,self.out_centers,self.out_weights,self.out_last,self.neigh,self.ndist)}
    def diagnostics(self):return {'active_microclusters':self.m,'outlier_microclusters':self.mo,'created':self.created,'absorbed':self.absorbed,'promoted':self.promoted,'replaced_outlier':self.replaced_outlier,**self.state_components()}



class FRAARTComparator:
    """FRA-ART SIBF/Fuzzy-ART study implementation for the common protocol.

    This is the declared SIBF variant (f(x)=x^a); it does not claim to exercise
    every FRA-ART basis-function variant from the source paper.
    x*=[x,f(x)] (2d), then complement coding -> 4d input. No random projection or category cap.
    Preallocated storage avoids Python reallocation overhead but active-state memory is reported separately.
    """
    def __init__(self,d:int,seed:int,a=.5,vigilance=.8,choice=.001,beta=1.0,max_stream_points=9000):
        self.d=d;self.seed=seed;self.a=float(a);self.rho=float(vigilance);self.alpha=float(choice);self.beta=float(beta);self.p=4*d
        # Storage policy only: grow active category arrays on demand instead of
        # reserving O(n*d) bytes for the maximum possible number of categories.
        # This does not change resonance, choice, vigilance, prototype updates,
        # category order, or any prediction/evaluation rule.
        self.max_capacity=int(max_stream_points)
        self.capacity=max(1,min(self.max_capacity,1024))
        self.prototypes=np.empty((self.capacity,self.p),np.float64)
        self.counts=np.zeros(self.capacity,np.float64)
        self.raw_sums=np.empty((self.capacity,d),np.float64)
        self.m=0
    def _ensure_capacity(self, needed:int):
        if needed <= self.capacity:
            return
        newcap=min(self.max_capacity,max(needed,self.capacity*2))
        if newcap < needed:
            raise RuntimeError("FRA-ART category storage exhausted")
        P=np.empty((newcap,self.p),np.float64);P[:self.m]=self.prototypes[:self.m]
        C=np.zeros(newcap,np.float64);C[:self.m]=self.counts[:self.m]
        R=np.empty((newcap,self.d),np.float64);R[:self.m]=self.raw_sums[:self.m]
        self.prototypes,self.counts,self.raw_sums=P,C,R
        self.capacity=newcap
    def _encode(self,x):
        xx=np.clip(x,0,1);f=np.power(xx,self.a);star=np.concatenate((xx,f));return np.concatenate((star,1.-star))
    def process_one(self,x):
        I=self._encode(x)
        if self.m==0:
            self._ensure_capacity(1)
            self.prototypes[0]=I;self.counts[0]=1.;self.raw_sums[0]=x;self.m=1;return 0
        P=self.prototypes[:self.m];mins=np.minimum(P,I);numer=mins.sum(axis=1);denom=P.sum(axis=1);T=numer/(self.alpha+denom);order=np.argsort(T)[::-1];normI=max(float(I.sum()),1e-12);chosen=-1
        for j in order:
            if numer[j]/normI>=self.rho:chosen=int(j);break
        if chosen<0:
            j=self.m;self._ensure_capacity(j+1);self.prototypes[j]=I;self.counts[j]=1.;self.raw_sums[j]=x;self.m+=1;return j
        j=chosen;self.prototypes[j]=(1-self.beta)*self.prototypes[j]+self.beta*mins[j];self.counts[j]+=1.;self.raw_sums[j]+=x;return j
    def process_batch(self,x):
        x=np.asarray(x,dtype=np.float64)
        if _fra_batch_numba is None:
            for row in x:self.process_one(row)
            return x
        self._ensure_capacity(self.m+len(x))
        self.m=int(_fra_batch_numba(self.prototypes,self.counts,self.raw_sums,self.m,x,self.a,self.rho,self.alpha,self.beta))
        return x
    def category_centers(self):return self.raw_sums[:self.m]/np.maximum(self.counts[:self.m,None],1e-12)
    def macro_centers(self,k):return weighted_macro_centers(self.category_centers(),self.counts[:self.m],k,self.seed)
    def state_components(self):
        proto=self.prototypes[:self.m].nbytes;raw=self.raw_sums[:self.m].nbytes+self.counts[:self.m].nbytes;total=proto+raw
        return {'prototype_bytes':proto,'category_raw_summary_bytes':raw,'persistent_numeric_bytes':total,'allocated_numeric_bytes':arr_bytes(self.prototypes,self.counts,self.raw_sums)}
    def diagnostics(self):return {'categories':self.m,**self.state_components()}


