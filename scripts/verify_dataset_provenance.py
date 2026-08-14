from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from asc_stream.datasets.provenance import dataset_status

DATASETS=['covertype','electricity','tweeteval','synthetic_gmm']

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',choices=DATASETS+['all'],default='all')
    ap.add_argument('--require-canonical',action='store_true',help='require canonical public/source provenance, not only the packaged study asset')
    a=ap.parse_args()
    names=DATASETS if a.dataset=='all' else [a.dataset]
    rows=[dataset_status(ROOT,n).to_dict() for n in names]
    errors=[]
    for r in rows:
        if a.require_canonical:
            if not r['canonical']: errors.append(f"{r['dataset']}: canonical source preparation is not available/verified")
        elif not (r['canonical'] or r.get('study_asset_verified',False)):
            errors.append(f"{r['dataset']}: neither canonical data nor the packaged study asset passed integrity verification")
    print(json.dumps({'pass':not errors,'strict_canonical':bool(a.require_canonical),'errors':errors,'datasets':rows},indent=2,ensure_ascii=False))
    if errors: raise SystemExit(1)
if __name__=='__main__':main()
