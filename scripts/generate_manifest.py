from __future__ import annotations
import argparse, csv, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".git"}
EXCLUDED_PREFIXES = [
    Path("results/execution_runs"),
]
EXCLUDED_NAMES = {"MANIFEST_SHA256.csv"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".part"}


def included(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if path.name in EXCLUDED_NAMES or path.suffix in EXCLUDED_SUFFIXES:
        return False
    if any(part in EXCLUDED_PARTS for part in rel.parts):
        return False
    if any(rel == pref or pref in rel.parents for pref in EXCLUDED_PREFIXES):
        return False
    return path.is_file()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the immutable release-file SHA-256 manifest.")
    ap.add_argument("--out", type=Path, default=ROOT / "MANIFEST_SHA256.csv")
    args = ap.parse_args()
    rows=[]
    for p in sorted(ROOT.rglob("*")):
        if included(p):
            rows.append({"path": p.relative_to(ROOT).as_posix(), "size_bytes": p.stat().st_size, "sha256": sha256(p)})
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["path","size_bytes","sha256"]); w.writeheader(); w.writerows(rows)
    print(f"manifest_files={len(rows)} out={args.out}")

if __name__ == "__main__":
    main()
