from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

USER_AGENT = "AdaptiveStreamClustering/2.0"
TWEETEVAL_COMMIT = "4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66"
SOURCES = {
    "covertype": {
        "landing": "https://archive.ics.uci.edu/dataset/31/covertype",
        "downloads": [
            "https://archive.ics.uci.edu/static/public/31/covertype.zip",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz",
            "https://kdd.ics.uci.edu/databases/covertype/covtype.data.gz",
        ],
    },
    "electricity": {
        "landing": "https://www.openml.org/d/151",
        "metadata": "https://www.openml.org/api/v1/json/data/151",
    },
    "tweeteval": {
        "landing": "https://github.com/cardiffnlp/tweeteval",
        "revision": TWEETEVAL_COMMIT,
        "download": f"https://github.com/cardiffnlp/tweeteval/archive/{TWEETEVAL_COMMIT}.zip",
        "raw_base": f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/{TWEETEVAL_COMMIT}/datasets/sentiment",
    },
}

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def md5(path: Path) -> str:
    h=hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def download(url: str, target: Path, retries: int=3) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    last=None
    for attempt in range(retries):
        tmp=target.with_suffix(target.suffix+".part")
        try:
            req=urllib.request.Request(url, headers={"User-Agent":USER_AGENT})
            with urllib.request.urlopen(req, timeout=180) as r, tmp.open("wb") as out:
                shutil.copyfileobj(r,out)
            tmp.replace(target)
            return
        except Exception as exc:
            last=exc; tmp.unlink(missing_ok=True); time.sleep(2**attempt)
    raise RuntimeError(f"Download failed for {url}: {last}")

def acquire_covertype(root: Path) -> list[Path]:
    root.mkdir(parents=True,exist_ok=True)
    # Prefer the current UCI ZIP; fall back to the legacy official gzip endpoints.
    zip_path=root/"covertype.zip"
    try:
        download(SOURCES["covertype"]["downloads"][0],zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            member=next((n for n in zf.namelist() if n.endswith("covtype.data.gz")),None)
            if member is None: raise RuntimeError("covtype.data.gz not found in UCI archive")
            out=root/"covtype.data.gz"
            with zf.open(member) as src, out.open("wb") as dst: shutil.copyfileobj(src,dst)
        return [out]
    except Exception:
        zip_path.unlink(missing_ok=True)
    out=root/"covtype.data.gz"
    last=None
    for url in SOURCES["covertype"]["downloads"][1:]:
        try: download(url,out); return [out]
        except Exception as exc: last=exc
    raise RuntimeError(f"All official CoverType download endpoints failed: {last}")

def acquire_electricity(root: Path) -> list[Path]:
    root.mkdir(parents=True,exist_ok=True)
    meta_path=root/"openml_151_metadata.json"
    download(SOURCES["electricity"]["metadata"],meta_path)
    meta=json.loads(meta_path.read_text(encoding="utf-8"))
    desc=meta.get("data_set_description",{})
    data_url=desc.get("url")
    if not data_url: raise RuntimeError("OpenML dataset 151 metadata did not provide a data URL")
    out=root/"electricity.arff"
    download(data_url,out)
    expected_md5=desc.get("md5_checksum")
    if expected_md5 and md5(out).lower()!=str(expected_md5).lower():
        out.unlink(missing_ok=True)
        raise RuntimeError("OpenML Electricity checksum verification failed")
    return [meta_path,out]

def acquire_tweeteval(root: Path) -> list[Path]:
    """Acquire the official TweetEval sentiment files at a pinned revision.

    The pinned commit prevents a later upstream edit from silently changing the
    canonical text stream. The archive route is attempted first; if it is not
    reachable, the six required files are downloaded directly from the same
    commit.
    """
    root.mkdir(parents=True,exist_ok=True)
    required=["train_text.txt","train_labels.txt","val_text.txt","val_labels.txt","test_text.txt","test_labels.txt"]
    out=root/"tweeteval"/"sentiment"
    if out.parent.exists(): shutil.rmtree(out.parent)
    out.mkdir(parents=True,exist_ok=True)
    archive=root/f"tweeteval-{TWEETEVAL_COMMIT}.zip"
    archive_error=None
    try:
        download(SOURCES["tweeteval"]["download"],archive)
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                marker="datasets/sentiment/"
                if marker in member and not member.endswith("/"):
                    rel=member.split(marker,1)[1]
                    if rel in required or rel == "mapping.txt":
                        target=out/rel; target.parent.mkdir(parents=True,exist_ok=True)
                        with zf.open(member) as src, target.open("wb") as dst: shutil.copyfileobj(src,dst)
    except Exception as exc:
        archive_error=exc
    missing=[name for name in required if not (out/name).exists()]
    if missing:
        for name in missing:
            download(f"{SOURCES['tweeteval']['raw_base']}/{name}",out/name)
        # mapping is useful metadata but not required by the clustering pipeline.
        try: download(f"{SOURCES['tweeteval']['raw_base']}/mapping.txt",out/'mapping.txt')
        except Exception: pass
    missing=[name for name in required if not (out/name).exists()]
    if missing:
        raise RuntimeError(f"TweetEval sentiment acquisition incomplete at {TWEETEVAL_COMMIT}: {missing}; archive_error={archive_error}")
    (root/'TWEETEVAL_SOURCE.json').write_text(json.dumps({
        'repository':'cardiffnlp/tweeteval','commit':TWEETEVAL_COMMIT,
        'task':'sentiment','order':['train','val','test'],
    },indent=2),encoding='utf-8')
    return sorted(p for p in out.glob('*') if p.is_file())+[root/'TWEETEVAL_SOURCE.json']

def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",choices=["covertype","electricity","tweeteval","all"],default="all")
    ap.add_argument("--root",type=Path,default=Path("data/public/raw"))
    args=ap.parse_args()
    names=["covertype","electricity","tweeteval"] if args.dataset=="all" else [args.dataset]
    funcs={"covertype":acquire_covertype,"electricity":acquire_electricity,"tweeteval":acquire_tweeteval}
    args.root.mkdir(parents=True,exist_ok=True)
    manifest_path=args.root/"acquisition_manifest.json"
    if manifest_path.exists():
        try: manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception: manifest=[]
    else:
        manifest=[]
    manifest=[row for row in manifest if row.get("dataset") not in names]
    for name in names:
        files=funcs[name](args.root/name)
        for path in files:
            if path.is_file(): manifest.append({"dataset":name,"path":str(path),"bytes":path.stat().st_size,"sha256":sha256(path),"source_revision":(TWEETEVAL_COMMIT if name=="tweeteval" else None)})
        print(f"{name}: OK")
    manifest.sort(key=lambda row:(row.get("dataset",""),row.get("path","")))
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")

if __name__=="__main__": main()
