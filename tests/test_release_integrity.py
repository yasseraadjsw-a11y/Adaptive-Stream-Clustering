from __future__ import annotations
import json
from pathlib import Path

from asc_stream.datasets.provenance import dataset_status, validate_study_asset
from asc_stream.paper_protocol import load_packaged_study_dataset

ROOT=Path(__file__).resolve().parents[1]


def test_required_runtime_dependencies_are_declared():
    req=(ROOT/'requirements.txt').read_text(encoding='utf-8').lower()
    assert 'numba' in req
    assert 'python-docx' not in req


def test_no_development_backup_files_in_source_tree():
    bad=[]
    for p in (ROOT/'src').rglob('*'):
        if p.is_file() and ('.pre_' in p.name or p.suffix in {'.bak','.orig'} or p.name.endswith('~')):
            bad.append(p)
    assert not bad


def test_packaged_study_assets_are_fingerprint_verified():
    assert dataset_status(ROOT,'covertype').study_asset_verified
    assert dataset_status(ROOT,'electricity').study_asset_verified
    assert dataset_status(ROOT,'synthetic_gmm').study_asset_verified
    assert dataset_status(ROOT,'synthetic_gmm').canonical
    assert validate_study_asset(ROOT,'tweeteval_sentiment_projected',(59899,256)).study_asset_verified


def test_packaged_study_loader_shapes():
    expected={'covertype':(581012,54),'electricity':(45312,8),'tweeteval':(59899,256),'synthetic_gmm':(9000,256)}
    # Avoid loading every large asset in the normal test suite; use two
    # representative assets and verify the rest through the fingerprint test.
    for ds in ('electricity','synthetic_gmm'):
        d=load_packaged_study_dataset(ROOT,ds)
        assert d.x.shape==expected[ds]
        assert len(d.y)==expected[ds][0]


def test_synthetic_benchmark_and_controlled_analysis_share_fixed_realization():
    import numpy as np
    study=np.load(ROOT/'data/rank_validation/processed/synthetic_gmm.npz',allow_pickle=False)
    controlled=np.load(ROOT/'data/controlled/representation_drift_stream.npz',allow_pickle=False)
    assert np.array_equal(study['x'],controlled['x'].astype(study['x'].dtype))
    assert np.array_equal(study['y'],controlled['y'])


def test_main_cli_has_complete_reproduction_actions():
    text=(ROOT/'main.py').read_text(encoding='utf-8')
    for action in ('run-primary','run-modern-all','run-rank','run-controlled','rebuild-controlled-data','verify-manifest','verify-package','verify-data','verify-canonical-data','run-study-suite'):
        assert f'"{action}"' in text
    assert 'CONTROLLED_METHODS = ["proposed", "fixed_rank", "twstream", "fra_art"]' in text


def test_study_assets_manifest_schema():
    m=json.loads((ROOT/'data/study_assets_manifest.json').read_text(encoding='utf-8'))
    assert m['schema_version']=='study_assets_v1'
    assert set(m['assets'])=={'covertype','electricity','tweeteval_sentiment_projected','synthetic_gmm'}


def test_protected_main_results_output_guard():
    from asc_stream.release_io import ensure_execution_output, ensure_not_protected_output
    import pytest
    with pytest.raises(ValueError):
        ensure_not_protected_output(ROOT, Path('results/main_results/should_not_write.json'))
    safe=ensure_not_protected_output(ROOT, Path('results/execution_runs/smoke.json'))
    assert 'execution_runs' in safe.parts
    safe=ensure_execution_output(ROOT, Path('results/execution_runs/smoke.json'))
    assert 'execution_runs' in safe.parts
    with pytest.raises(ValueError):
        ensure_execution_output(ROOT, Path('temporary/smoke.json'))


def test_cross_platform_verify_launchers_and_ci_exist():
    assert (ROOT/'run_verify.sh').exists()
    assert (ROOT/'RUN_VERIFY_WINDOWS.bat').exists()
    assert (ROOT/'setup_and_verify.sh').exists()
    assert (ROOT/'SETUP_AND_VERIFY_WINDOWS.bat').exists()
    ci=(ROOT/'.github/workflows/ci.yml').read_text(encoding='utf-8')
    assert 'ubuntu-latest' in ci and 'windows-latest' in ci
    assert 'compileall' in ci


def test_protected_main_results_reference_is_complete_and_current():
    import hashlib
    ref={}
    for line in (ROOT/'docs/protected_main_results_sha256.txt').read_text(encoding='utf-8').splitlines():
        if line.strip():
            digest,rel=line.split('  ',1);ref[rel.lstrip('./')]=digest
    files=[p for p in (ROOT/'results/main_results').rglob('*') if p.is_file()]
    # The registry is authoritative.  Do not hard-code a file count: adding a
    # protected result table must require a registry update, not a test edit.
    assert set(ref)=={p.relative_to(ROOT).as_posix() for p in files}
    for p in files:
        rel=p.relative_to(ROOT).as_posix()
        assert hashlib.sha256(p.read_bytes()).hexdigest()==ref[rel]
