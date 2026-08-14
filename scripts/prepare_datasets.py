from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import arff
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from asc_stream.datasets.provenance import (TWEETEVAL_COMMIT,TWEETEVAL_SENTIMENT_SPLITS,TWEETEVAL_VECTORIZER,sha256_file)


def electricity(raw: Path,out: Path) -> dict:
    if raw.suffix.lower()==".arff":
        data,_=arff.loadarff(raw);df=pd.DataFrame(data)
        for c in df.columns:
            if df[c].dtype==object:
                df[c]=df[c].map(lambda v:v.decode("utf-8") if isinstance(v,(bytes,bytearray)) else v)
    else:
        df=pd.read_csv(raw)
    expected=["date","day","period","nswprice","nswdemand","vicprice","vicdemand","transfer","class"]
    cols=[str(c).lower() for c in df.columns];df.columns=cols
    if cols!=expected:raise ValueError(f"unexpected Electricity columns: {df.columns.tolist()}")
    for c in expected[:-1]:
        df[c]=pd.to_numeric(df[c],errors="raise")
    x=df[expected[:-1]].to_numpy(np.float64)
    y=df["class"].astype(str).str.upper().map({"DOWN":0,"UP":1})
    if y.isna().any():raise ValueError("unknown Electricity label")
    np.savez_compressed(out,x=x,y=y.to_numpy(np.int64))
    return {"rows":len(x),"features":x.shape[1],"classes":2,"order":"OpenML dataset 151 file order retained","preprocessing":"all non-class fields kept numeric; causal standardization occurs online; labels are never used by representation/rank control"}


def covertype(raw: Path,out: Path) -> dict:
    with gzip.open(raw,"rt") as f:df=pd.read_csv(f,header=None)
    if df.shape[1]!=55:raise ValueError(f"expected 55 CoverType columns, got {df.shape[1]}")
    x=df.iloc[:,:54].to_numpy(np.float64);y=df.iloc[:,54].to_numpy(np.int64)-1
    np.savez_compressed(out,x=x,y=y)
    return {"rows":len(x),"features":54,"classes":7,"order":"UCI file order retained","preprocessing":"all 54 cartographic features retained; causal standardization inside model"}


def tweeteval(root: Path,out: Path,n_features: int, source_marker: Path | None = None) -> dict:
    if int(n_features) != int(TWEETEVAL_VECTORIZER["n_features"]):
        raise ValueError(f"Canonical TweetEval requires {TWEETEVAL_VECTORIZER['n_features']} hashed features, got {n_features}")
    if source_marker is None:
        source_marker=root.parents[1]/"TWEETEVAL_SOURCE.json"
    if not source_marker.exists():
        raise FileNotFoundError(f"Missing pinned TweetEval source marker: {source_marker}. Re-run acquire_datasets.py.")
    source=json.loads(source_marker.read_text(encoding="utf-8"))
    if source.get("commit") != TWEETEVAL_COMMIT:
        raise ValueError(f"TweetEval source revision {source.get('commit')} != pinned {TWEETEVAL_COMMIT}")
    texts=[];labels=[];split_sizes={};raw_sha={}
    for split in ["train","val","test"]:
        tp=root/f"{split}_text.txt";lp=root/f"{split}_labels.txt"
        t=tp.read_text(encoding="utf-8").splitlines();y=[int(v) for v in lp.read_text(encoding="utf-8").splitlines()]
        if len(t)!=len(y):raise ValueError(f"TweetEval {split} length mismatch")
        if len(t)!=TWEETEVAL_SENTIMENT_SPLITS[split]:
            raise ValueError(f"TweetEval {split} expected {TWEETEVAL_SENTIMENT_SPLITS[split]} rows at pinned revision, got {len(t)}")
        split_sizes[split]=len(t);texts.extend(t);labels.extend(y)
        raw_sha[tp.name]=sha256_file(tp);raw_sha[lp.name]=sha256_file(lp)
    vec=HashingVectorizer(
        n_features=TWEETEVAL_VECTORIZER["n_features"],
        alternate_sign=TWEETEVAL_VECTORIZER["alternate_sign"],
        norm=TWEETEVAL_VECTORIZER["norm"],
        lowercase=TWEETEVAL_VECTORIZER["lowercase"],
        ngram_range=tuple(TWEETEVAL_VECTORIZER["ngram_range"]),
        dtype=np.float32,
    )
    # Stateless text feature extraction; no labels or future samples are used to fit the transform.
    x=vec.transform(texts).tocsr().astype(np.float32,copy=False)
    feature_file=out.with_name("tweeteval_features.npz")
    sparse.save_npz(feature_file,x,compressed=True)
    np.savez_compressed(out,y=np.asarray(labels,np.int64),split_sizes=json.dumps(split_sizes),feature_file=feature_file.name,source_revision=TWEETEVAL_COMMIT)
    return {
        "schema_version":"canonical_tweeteval_v1",
        "rows":x.shape[0],"features":TWEETEVAL_VECTORIZER["n_features"],"classes":3,
        "order":"train then validation then test, preserving each official split order",
        "preprocessing":"stateless word 1-2 gram HashingVectorizer stored in CSR form; no labels or future samples are used to fit the transform",
        "split_sizes":split_sizes,"feature_storage":"CSR",
        "source_repository":"cardiffnlp/tweeteval","source_revision":TWEETEVAL_COMMIT,
        "vectorizer":TWEETEVAL_VECTORIZER,"raw_sha256":raw_sha,
        "feature_sha256":sha256_file(feature_file),
    }


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--dataset",choices=["electricity","covertype","tweeteval","all"],default="all")
    p.add_argument("--raw-root",type=Path,default=Path("data/public/raw"));p.add_argument("--out-root",type=Path,default=Path("data/public/processed"));p.add_argument("--tweet-features",type=int,default=2048)
    a=p.parse_args();a.out_root.mkdir(parents=True,exist_ok=True)
    manifest_path=a.out_root/"preprocessing_manifest.json"
    if manifest_path.exists():
        try:meta=json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:meta={}
    else:meta={}
    names=["electricity","covertype","tweeteval"] if a.dataset=="all" else [a.dataset]
    for n in names:
        out=a.out_root/f"{n}.npz"
        if n=="electricity":meta[n]=electricity(a.raw_root/n/"electricity.arff",out)
        elif n=="covertype":meta[n]=covertype(a.raw_root/n/"covtype.data.gz",out)
        else:meta[n]=tweeteval(a.raw_root/n/"tweeteval/sentiment",out,a.tweet_features,a.raw_root/n/"TWEETEVAL_SOURCE.json")
        print(n,meta[n])
    manifest_path.write_text(json.dumps(meta,indent=2),encoding="utf-8")
if __name__=="__main__":main()
