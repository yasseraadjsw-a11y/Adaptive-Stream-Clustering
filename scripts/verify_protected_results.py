from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REF=ROOT/'docs/protected_main_results_sha256.txt'
PREFIX='results/main_results/'

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--update-reference',action='store_true',help='rewrite the registry after an intentional release assembly change')
    args=ap.parse_args()
    actual={p.relative_to(ROOT).as_posix():sha256(p) for p in (ROOT/'results/main_results').rglob('*') if p.is_file()}
    if args.update_reference:
        lines=[f"{actual[rel]}  {rel}" for rel in sorted(actual)]
        REF.write_text("\n".join(lines)+"\n",encoding='utf-8')
        print(json.dumps({'updated':True,'files':len(actual),'reference':str(REF.relative_to(ROOT))},indent=2))
        return
    errors=[]; expected={}
    for line in REF.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        digest,rel=line.split('  ',1); rel=rel.lstrip('./')
        if not rel.startswith(PREFIX): errors.append(f'invalid protected path: {rel}');continue
        expected[rel]=digest
    for rel,digest in expected.items():
        if rel not in actual: errors.append(f'missing protected result: {rel}')
        elif actual[rel]!=digest: errors.append(f'protected result hash changed: {rel}')
    for rel in sorted(set(actual)-set(expected)): errors.append(f'unregistered protected result: {rel}')
    out={'pass':not errors,'expected_files':len(expected),'actual_files':len(actual),'hash_mismatches':sum('hash changed' in e for e in errors),'errors':errors}
    print(json.dumps(out,indent=2))
    if errors: raise SystemExit(1)
if __name__=='__main__': main()
