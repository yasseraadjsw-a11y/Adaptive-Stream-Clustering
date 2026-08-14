from __future__ import annotations
import numpy as np


def _contingency(y_true, y_pred):
    yt=np.asarray(y_true).reshape(-1)
    yp=np.asarray(y_pred).reshape(-1)
    if yt.size != yp.size: raise ValueError('label length mismatch')
    if yt.size == 0: return np.zeros((0,0),np.int64)
    _, ti=np.unique(yt,return_inverse=True)
    _, pi=np.unique(yp,return_inverse=True)
    q=int(pi.max())+1; r=int(ti.max())+1
    return np.bincount(ti*q+pi,minlength=r*q).reshape(r,q).astype(np.int64,copy=False)


def adjusted_rand_from_contingency(c: np.ndarray) -> float:
    n=int(c.sum())
    if n < 2: return 1.0
    a=c.sum(axis=1,dtype=np.int64); b=c.sum(axis=0,dtype=np.int64)
    def comb2(v):
        v=np.asarray(v,dtype=np.float64)
        return float(np.sum(v*(v-1.0)*0.5))
    nij=comb2(c.ravel()); ai=comb2(a); bj=comb2(b); total=n*(n-1.0)*0.5
    expected=ai*bj/total
    max_index=0.5*(ai+bj)
    den=max_index-expected
    num=nij-expected
    if abs(den) <= np.finfo(float).eps:
        return 1.0
    return float(num/den)


def normalized_mutual_info_from_contingency(c: np.ndarray) -> float:
    n=float(c.sum())
    if n <= 0: return 1.0
    a=c.sum(axis=1,dtype=np.float64); b=c.sum(axis=0,dtype=np.float64)
    nz=np.nonzero(c)
    vals=c[nz].astype(np.float64)
    mi=float(np.sum((vals/n)*np.log((vals*n)/(a[nz[0]]*b[nz[1]]))))
    pa=a[a>0]/n; pb=b[b>0]/n
    ha=float(-np.sum(pa*np.log(pa))); hb=float(-np.sum(pb*np.log(pb)))
    den=0.5*(ha+hb)
    if den <= np.finfo(float).eps:
        return 1.0
    # sklearn clips tiny numerical excursions to [0,1]
    return float(min(max(mi/den,0.0),1.0))


def ari_nmi(y_true,y_pred):
    c=_contingency(y_true,y_pred)
    return adjusted_rand_from_contingency(c), normalized_mutual_info_from_contingency(c)
