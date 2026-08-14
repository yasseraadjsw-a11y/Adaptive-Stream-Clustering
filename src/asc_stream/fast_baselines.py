from __future__ import annotations
import numpy as np

class FastCluStreamBaseline:
    """Array-backed execution-equivalent of the study CluStream micro-summary path.
    Input must already have the common causal standardization applied.
    """
    def __init__(self,d,radius,max_microclusters,seed):
        self.original_dim=int(d);self.radius=float(radius);self.max_microclusters=int(max_microclusters);self.seed=int(seed)
        self.centers=np.zeros((self.max_microclusters,self.original_dim),np.float64)
        self.weights=np.zeros(self.max_microclusters,np.float64);self.last=np.zeros(self.max_microclusters,np.int64)
        self.created=np.zeros(self.max_microclusters,np.int64);self.m=0;self.time=0;self.pruned=0;self.merged=0
    def _delete(self,j):
        if j<self.m-1:
            self.centers[j:self.m-1]=self.centers[j+1:self.m];self.weights[j:self.m-1]=self.weights[j+1:self.m]
            self.last[j:self.m-1]=self.last[j+1:self.m];self.created[j:self.m-1]=self.created[j+1:self.m]
        self.m-=1
    def process_batch(self,x):
        a=np.asarray(x,dtype=np.float64)
        for z in a:
            self.time+=1
            if self.m==0:
                self.centers[0]=z;self.weights[0]=1.;self.last[0]=self.time;self.created[0]=self.time;self.m=1;continue
            ds=np.linalg.norm(self.centers[:self.m]-z,axis=1);j=int(np.argmin(ds))
            if ds[j]<=self.radius:
                ow=self.weights[j];nw=ow+1.;self.centers[j]=(ow*self.centers[j]+z)/nw;self.weights[j]=nw;self.last[j]=self.time
            else:
                if self.m>=self.max_microclusters:
                    score=self.weights[:self.m]/(1.+np.maximum(self.time-self.last[:self.m],0));self._delete(int(np.argmin(score)));self.pruned+=1
                j=self.m;self.centers[j]=z;self.weights[j]=1.;self.last[j]=self.time;self.created[j]=self.time;self.m+=1
        return a
    def summary(self):return self.centers[:self.m].copy(),self.weights[:self.m].copy()
    def diagnostics(self):return {'final_microclusters':self.m,'pruned_microclusters':self.pruned,'merged_microclusters':0,'model_state_mb':float((self.centers[:self.m].nbytes+self.weights[:self.m].nbytes+self.last[:self.m].nbytes+self.created[:self.m].nbytes)/2**20),'execution_engine':'array_clustream_equivalent'}

class FastDenStreamBaseline:
    """Vectorized-array DenStream study implementation for preprocessed dense streams."""
    def __init__(self,d,radius,max_microclusters,seed,beta=.2,mu=6.,fading_lambda=.01):
        self.original_dim=int(d);self.radius=float(radius);self.max_microclusters=int(max_microclusters);self.seed=int(seed);self.beta=float(beta);self.mu=float(mu);self.fading_lambda=float(fading_lambda)
        cap=self.max_microclusters+4
        self.p_ls=np.zeros((cap,d));self.p_ss=np.zeros((cap,d));self.p_w=np.zeros(cap);self.p_last=np.zeros(cap,np.int64);self.p_created=np.zeros(cap,np.int64);self.mp=0
        self.o_ls=np.zeros((cap,d));self.o_ss=np.zeros((cap,d));self.o_w=np.zeros(cap);self.o_last=np.zeros(cap,np.int64);self.o_created=np.zeros(cap,np.int64);self.mo=0
        self.time=0;self.pruned=0;self.promoted=0
        self.tp=max(1,int(np.ceil((1./self.fading_lambda)*np.log2((self.beta*self.mu)/(self.beta*self.mu-1.)))))
    def _decay(self,kind,now):
        if kind=='p': ls,ss,w,last,m=self.p_ls,self.p_ss,self.p_w,self.p_last,self.mp
        else: ls,ss,w,last,m=self.o_ls,self.o_ss,self.o_w,self.o_last,self.mo
        if m==0:return
        dt=np.maximum(now-last[:m],0);fac=np.exp2(-self.fading_lambda*dt)
        ls[:m]*=fac[:,None];ss[:m]*=fac[:,None];w[:m]*=fac;last[:m]=now
    def _delete(self,kind,j):
        if kind=='p': ls,ss,w,last,created,m=self.p_ls,self.p_ss,self.p_w,self.p_last,self.p_created,self.mp
        else: ls,ss,w,last,created,m=self.o_ls,self.o_ss,self.o_w,self.o_last,self.o_created,self.mo
        if j<m-1:
            ls[j:m-1]=ls[j+1:m];ss[j:m-1]=ss[j+1:m];w[j:m-1]=w[j+1:m];last[j:m-1]=last[j+1:m];created[j:m-1]=created[j+1:m]
        if kind=='p':self.mp-=1
        else:self.mo-=1
    def _append(self,kind,z,now):
        if kind=='p':j=self.mp;self.p_ls[j]=z;self.p_ss[j]=z*z;self.p_w[j]=1.;self.p_last[j]=now;self.p_created[j]=now;self.mp+=1
        else:j=self.mo;self.o_ls[j]=z;self.o_ss[j]=z*z;self.o_w[j]=1.;self.o_last[j]=now;self.o_created[j]=now;self.mo+=1
    def _try(self,kind,z,now):
        m=self.mp if kind=='p' else self.mo
        if m==0:return None
        self._decay(kind,now)
        if kind=='p':ls,ss,w=self.p_ls,self.p_ss,self.p_w
        else:ls,ss,w=self.o_ls,self.o_ss,self.o_w
        centers=ls[:m]/np.maximum(w[:m,None],1e-12);dist=np.linalg.norm(centers-z,axis=1);order=np.argsort(dist)
        nls=ls[:m]+z;nss=ss[:m]+z*z;nw=w[:m]+1.;mean=nls/nw[:,None];var=np.maximum(nss/nw[:,None]-mean*mean,0.);rad=np.sqrt(np.sum(var,axis=1))
        good=rad[order]<=self.radius
        if not np.any(good):return None
        j=int(order[int(np.argmax(good))])
        ls[j]=nls[j];ss[j]=nss[j];w[j]=nw[j]
        return j
    def _prune(self,now):
        self._decay('p',now);self._decay('o',now)
        if self.mp:
            keep=self.p_w[:self.mp]>=self.beta*self.mu
            old=self.mp; idx=np.flatnonzero(keep);n=len(idx)
            self.p_ls[:n]=self.p_ls[idx];self.p_ss[:n]=self.p_ss[idx];self.p_w[:n]=self.p_w[idx];self.p_last[:n]=self.p_last[idx];self.p_created[:n]=self.p_created[idx];self.mp=n;self.pruned+=old-n
        if self.mo:
            denom=np.exp2(-self.fading_lambda*self.tp)-1.;age=now-self.o_created[:self.mo];numer=np.exp2(-self.fading_lambda*(age+self.tp))-1.;xi=numer/denom if abs(denom)>1e-15 else np.zeros(self.mo)
            keep=self.o_w[:self.mo]>=xi;old=self.mo;idx=np.flatnonzero(keep);n=len(idx)
            self.o_ls[:n]=self.o_ls[idx];self.o_ss[:n]=self.o_ss[idx];self.o_w[:n]=self.o_w[idx];self.o_last[:n]=self.o_last[idx];self.o_created[:n]=self.o_created[idx];self.mo=n;self.pruned+=old-n
        while self.mp+self.mo>self.max_microclusters:
            if self.mo:self._delete('o',int(np.argmin(self.o_w[:self.mo])))
            else:self._delete('p',int(np.argmin(self.p_w[:self.mp])))
            self.pruned+=1
    def process_batch(self,x):
        a=np.asarray(x,dtype=np.float64)
        for z in a:
            self.time+=1;j=self._try('p',z,self.time)
            if j is None:
                oj=self._try('o',z,self.time)
                if oj is not None:
                    if self.o_w[oj]>=self.beta*self.mu:
                        j2=self.mp;self.p_ls[j2]=self.o_ls[oj];self.p_ss[j2]=self.o_ss[oj];self.p_w[j2]=self.o_w[oj];self.p_last[j2]=self.o_last[oj];self.p_created[j2]=self.o_created[oj];self.mp+=1;self._delete('o',oj);self.promoted+=1
                else:self._append('o',z,self.time)
            if self.time%self.tp==0 or self.mp+self.mo>self.max_microclusters:self._prune(self.time)
        return a
    def summary(self):
        if self.mp:self._decay('p',self.time);m=self.mp;return self.p_ls[:m]/np.maximum(self.p_w[:m,None],1e-12),np.maximum(self.p_w[:m],1e-12).copy()
        self._decay('o',self.time);m=self.mo;return self.o_ls[:m]/np.maximum(self.o_w[:m,None],1e-12),np.maximum(self.o_w[:m],1e-12).copy()
    def diagnostics(self):return {'potential_microclusters':self.mp,'outlier_microclusters':self.mo,'promoted_microclusters':self.promoted,'pruned_microclusters':self.pruned,'execution_engine':'array_denstream_equivalent'}

class FastStreamKMPlusPlusBaseline:
    """Array-buffer D2 non-uniform coreset reconstruction for preprocessed dense data."""
    def __init__(self,d,coreset_size=500,buffer_size=1000,seed=7):
        self.original_dim=int(d);self.coreset_size=int(coreset_size);self.buffer_size=int(buffer_size);self.seed=int(seed);self.values=np.empty((0,d),np.float64);self.weights=np.empty(0,np.float64);self.compressions=0
    def _pilot(self,x,w,k,rng):
        first=int(rng.choice(len(x),p=w/w.sum()));cs=[x[first]];md=np.sum((x-x[first])**2,axis=1)
        while len(cs)<k:
            score=np.maximum(w,1e-12)*(md+1e-15);tot=float(score.sum());j=int(rng.choice(len(x),p=score/tot)) if tot>0 else 0;cs.append(x[j]);md=np.minimum(md,np.sum((x-x[j])**2,axis=1))
        return np.vstack(cs)
    def _compress(self):
        if len(self.values)<=self.coreset_size:return
        x=self.values;w=self.weights;k=min(self.coreset_size,len(x));rng=np.random.default_rng(self.seed+self.compressions);pilot=self._pilot(x,np.maximum(w,1e-12),min(8,k,len(x)),rng)
        md=np.full(len(x),np.inf)
        for c in pilot:md=np.minimum(md,np.sum((x-c)**2,axis=1))
        score=np.maximum(w,1e-12)*(md+0.05*max(float(np.mean(md)),1e-12));
        if (not np.isfinite(score).all()) or float(score.sum())<=0:score=np.maximum(w,1e-12)
        prob=score/score.sum();chosen=rng.choice(len(x),size=k,replace=False,p=prob);incl=np.minimum(1.,k*prob[chosen]);self.values=x[chosen].copy();self.weights=w[chosen]/np.maximum(incl,1e-12);self.compressions+=1
    def process_batch(self,x):
        a=np.asarray(x,dtype=np.float64);pos=0
        while pos<len(a):
            room=max(self.buffer_size-len(self.values),1);take=min(room,len(a)-pos)
            self.values=np.vstack((self.values,a[pos:pos+take]));self.weights=np.concatenate((self.weights,np.ones(take)));pos+=take
            if len(self.values)>=self.buffer_size:self._compress()
        return a
    def summary(self):return self.values.copy(),self.weights.copy()
    def diagnostics(self):return {'final_coreset_points':len(self.values),'compressions':self.compressions,'execution_engine':'array_streamkmpp_d2'}
