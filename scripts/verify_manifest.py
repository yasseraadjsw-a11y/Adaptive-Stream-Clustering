from __future__ import annotations
import csv, hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'MANIFEST_SHA256.csv'


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def main() -> None:
    errors=[]; checked=0
    if not MANIFEST.exists():
        errors.append('MANIFEST_SHA256.csv is missing')
    else:
        with MANIFEST.open(newline='',encoding='utf-8-sig') as f:
            rows=list(csv.DictReader(f))
        seen=set()
        for r in rows:
            rel=r.get('path','')
            if not rel or rel in seen:
                errors.append(f'invalid/duplicate manifest path: {rel!r}'); continue
            seen.add(rel); p=ROOT/rel
            if not p.exists():
                errors.append(f'missing: {rel}'); continue
            if not p.is_file():
                errors.append(f'not a file: {rel}'); continue
            try: expected_size=int(r['size_bytes'])
            except Exception:
                errors.append(f'invalid size: {rel}'); continue
            if p.stat().st_size != expected_size:
                errors.append(f'size mismatch: {rel}'); continue
            if sha256(p) != r.get('sha256',''):
                errors.append(f'hash mismatch: {rel}'); continue
            checked += 1
        # Two-way inventory check: release files not listed in the manifest are
        # also detected, except explicitly mutable runtime/cache areas.
        allowed_prefixes=("results/execution_runs/",)
        # GitHub Actions checks out the repository with its internal .git
        # directory present. Repository metadata is not a release artifact.
        allowed_parts={"__pycache__",".pytest_cache",".git"}
        actual=set()
        for p in ROOT.rglob("*"):
            if not p.is_file(): continue
            rel=p.relative_to(ROOT).as_posix()
            if rel=="MANIFEST_SHA256.csv" or rel.startswith(allowed_prefixes) or any(part in allowed_parts for part in p.parts): continue
            actual.add(rel)
        manifest_inventory={x for x in seen if x!="MANIFEST_SHA256.csv"}
        for rel in sorted(actual-manifest_inventory): errors.append(f"unexpected unmanifested file: {rel}")
        for rel in sorted(manifest_inventory-actual):
            if not (ROOT/rel).exists(): continue

    out={'pass':not errors,'checked_files':checked,'errors':errors}
    print(json.dumps(out,indent=2,ensure_ascii=False))
    if errors: raise SystemExit(1)

if __name__=='__main__': main()
