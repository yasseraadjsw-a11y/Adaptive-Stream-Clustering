from __future__ import annotations
import ast, json, re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

REQUIRED_DECLARED={
    'numpy':'numpy','scipy':'scipy','sklearn':'scikit-learn','matplotlib':'matplotlib','pandas':'pandas',
    'psutil':'psutil','numba':'numba','pytest':'pytest',
}
ALLOWED_BACKUP_SUFFIXES=()
FORBIDDEN_PATTERNS=('*.pre_*','*.bak','*.orig','*~')


def declared_packages() -> set[str]:
    text=(ROOT/'requirements.txt').read_text(encoding='utf-8')
    names=set()
    for line in text.splitlines():
        line=line.strip()
        if not line or line.startswith('#'): continue
        names.add(re.split(r'[<>=!~\[]',line,maxsplit=1)[0].strip().lower())
    return names


def imported_top_level() -> set[str]:
    out=set()
    for p in list((ROOT/'src').rglob('*.py'))+list((ROOT/'scripts').rglob('*.py'))+list((ROOT/'experiments').rglob('*.py'))+list((ROOT/'tests').rglob('*.py')):
        try: tree=ast.parse(p.read_text(encoding='utf-8'))
        except Exception: continue
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):
                out.update(a.name.split('.')[0] for a in n.names)
            elif isinstance(n,ast.ImportFrom) and n.module:
                out.add(n.module.split('.')[0])
    return out


def main() -> None:
    errors=[]; warnings=[]
    req=declared_packages(); imports=imported_top_level()
    for mod,pkg in REQUIRED_DECLARED.items():
        if mod in imports and pkg.lower() not in req:
            errors.append(f'dependency not declared: import {mod} requires {pkg}')
    for pattern in FORBIDDEN_PATTERNS:
        for p in ROOT.rglob(pattern): errors.append(f'development backup file packaged: {p.relative_to(ROOT)}')
    cache_files=[p for p in ROOT.rglob('*') if p.is_file() and any(part in {'__pycache__','.pytest_cache'} for part in p.parts)]
    if cache_files:
        warnings.append(f'runtime cache files present in working tree (excluded from release manifest): {len(cache_files)}')
    # Protected manuscript results are never default execution outputs; main.py and
    # experiment runners route new runs to results/execution_runs/.
    # Documentation must not advertise absent main.py actions.
    main_text=(ROOT/'main.py').read_text(encoding='utf-8')
    advertised=set()
    for md in [ROOT/'README.md',ROOT/'README_AR.md',ROOT/'REPRODUCIBILITY.md',ROOT/'data/README.md']:
        if md.exists():
            advertised.update(re.findall(r'python\s+main\.py\s+([a-z0-9-]+)',md.read_text(encoding='utf-8')))
    for action in sorted(advertised):
        if f'"{action}"' not in main_text and f"'{action}'" not in main_text:
            errors.append(f'documented main.py action is absent: {action}')
    required_files=[
        'results/main_results/multidataset/dataset_quality_with_sd.csv',
        'data/study_assets_manifest.json','scripts/verify_manifest.py','scripts/generate_manifest.py',
        'experiments/run_primary.py','experiments/aggregate_primary_runs.py','experiments/run_rank.py','src/asc_stream/release_io.py',
        'run_verify.sh','RUN_VERIFY_WINDOWS.bat','setup_and_verify.sh','SETUP_AND_VERIFY_WINDOWS.bat',
        '.github/workflows/ci.yml','results/execution_runs/README.md',
    ]
    for rel in required_files:
        if not (ROOT/rel).exists(): errors.append(f'required release file missing: {rel}')
    # Direct runners must independently protect the immutable manuscript result tree.
    for rel in ['experiments/run_primary.py','experiments/run_rank.py','experiments/run_ablation.py','experiments/run_core_pair.py','experiments/run_controlled_method.py']:
        text=(ROOT/rel).read_text(encoding='utf-8')
        if 'ensure_not_protected_output' not in text and 'ensure_execution_output' not in text:
            errors.append(f'direct runner lacks protected-output guard: {rel}')
    primary_text=(ROOT/'experiments/run_primary.py').read_text(encoding='utf-8')
    aggregate_text=(ROOT/'experiments/aggregate_primary_runs.py').read_text(encoding='utf-8')
    if 'ensure_execution_output' not in primary_text or 'ensure_execution_output' not in aggregate_text:
        errors.append('unified primary execution path is not restricted to results/execution_runs')
    # A clean release contains no runtime artifacts under execution_runs.
    execution_root=ROOT/'results'/'execution_runs'
    unexpected=[p.relative_to(ROOT).as_posix() for p in execution_root.rglob('*') if p.is_file() and p.name!='README.md'] if execution_root.exists() else []
    if unexpected:
        warnings.append(f'fresh execution artifacts present (excluded from release manifest): {len(unexpected)}')
    out={'pass':not errors,'errors':errors,'warnings':warnings,'declared_dependencies':sorted(req),'advertised_actions':sorted(advertised)}
    print(json.dumps(out,indent=2,ensure_ascii=False))
    if errors: raise SystemExit(1)

if __name__=='__main__': main()
