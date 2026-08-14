from __future__ import annotations
import argparse,json,sys,os
from pathlib import Path
import numpy as np
from scipy import sparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from asc_stream.paper_protocol import load_full_paper_dataset,recent_method_settings,DISPLAY
from asc_stream.external.twstream_official import TWStreamOfficialAdapter
from asc_stream.fast_metrics import ari_nmi
from asc_stream.method_registry import method_metadata

def minmax(x):
    if sparse.issparse(x):
        x=x.tocsr().astype(np.float64,copy=False)
        mn=x.min(axis=0); mx=x.max(axis=0)
        if sparse.issparse(mn): mn=mn.toarray()
        if sparse.issparse(mx): mx=mx.toarray()
        mn=np.asarray(mn).reshape(-1); mx=np.asarray(mx).reshape(-1)
        if np.any(mn < -1e-12):
            # General sparse min-max with nonzero minima would introduce dense
            # offsets. Canonical TweetEval HashingVectorizer features are nonnegative.
            raise ValueError('sparse official TWStream adapter requires nonnegative sparse inputs')
        return x.multiply(1.0/np.maximum(mx,1e-12)).tocsr()
    a=np.asarray(x,float);lo=a.min(axis=0);hi=a.max(axis=0);return (a-lo)/np.maximum(hi-lo,1e-12)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True,choices=['covertype','electricity','tweeteval','synthetic_gmm']);ap.add_argument('--tweet-packaged-study-representation',action='store_true');a=ap.parse_args()
    D=load_full_paper_dataset(ROOT,a.dataset,allow_packaged_tweeteval=a.tweet_packaged_study_representation)
    x=D.x if sparse.issparse(D.x) else np.asarray(D.x,float)
    # Native adapter receives the same deterministic [0,1] scaling declared for recent-method evaluation.
    x=minmax(x);tw=recent_method_settings(ROOT,a.dataset,'twstream')
    # If dataset config does not carry recent settings, use Table-4 values explicitly.
    radius=float(tw.get('radius',10.4113));lam=float(tw.get('lambda',0.0028));k=int(tw.get('k',8));tau=float(tw.get('tau',0.65))
    adapter=TWStreamOfficialAdapter(ROOT/'external_sources'/'TWStream',ROOT/'results'/'external'/f'twstream_official_{a.dataset}')
    r=adapter.run(x,radius=radius,lambda_=lam,k=k,tau=tau)
    mask=r.labels>=0
    quality={'ari_assigned_only':None,'nmi_assigned_only':None}
    if np.any(mask):
        aa,nn=ari_nmi(np.asarray(D.y)[mask],r.labels[mask]);quality={'ari_assigned_only':float(aa),'nmi_assigned_only':float(nn)}
    out={'dataset':DISPLAY[a.dataset],'protocol':'study_configuration','implementation':method_metadata('twstream_official'),'canonical_data':bool(D.canonical),'coverage':r.coverage,**quality,**r.metadata()}
    outp=ROOT/'results'/'external'/f'twstream_official_{a.dataset}'/'result.json';outp.parent.mkdir(parents=True,exist_ok=True);outp.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
