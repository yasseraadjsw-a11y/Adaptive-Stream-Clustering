from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from asc_stream.external.twstream_official import TWStreamOfficialAdapter,TWSTREAM_COMMIT

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--method',choices=['twstream'],default='twstream');ap.add_argument('--check-only',action='store_true');a=ap.parse_args()
    adapter=TWStreamOfficialAdapter(ROOT/'external_sources'/'TWStream',ROOT/'results'/'external'/'twstream_official')
    status=adapter.toolchain_status();out={'method':'twstream','pinned_commit':TWSTREAM_COMMIT,'toolchain':status}
    if not a.check_only:
        out['jar']=str(adapter.setup())
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
