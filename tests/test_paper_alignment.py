from pathlib import Path
import json
import sys
import numpy as np
from scipy import sparse as sp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from asc_stream.baselines import StreamKMPlusPlusBaseline
from asc_stream.config import ASCConfig
from asc_stream.model import AdaptiveSketchStreamClusterer
from asc_stream.paper_protocol import recent_method_settings, load_paper_config
from asc_stream.datasets.provenance import TWEETEVAL_COMMIT
from asc_stream.external.twstream_official import TWStreamOfficialAdapter, TWSTREAM_COMMIT
from asc_stream.method_registry import method_metadata


def test_study_protocol_core_constants():
    p=load_paper_config(ROOT,'electricity')['proposed']
    assert p['rank_tolerance']==0.04
    assert p['stable_intervals_before_shrink']==4
    assert p['leverage_sampling_rate']==0.65
    assert p['min_sampling_probability']==0.15


def test_main_model_telemetry_is_self_contained():
    cfg=ASCConfig(original_dim=4,projection_dim=4,projection_mode='identity',window_size=64,max_rank=4,initial_rank=4,min_rank=2,max_microclusters=8,seed=7)
    m=AdaptiveSketchStreamClusterer(cfg)
    m.process_batch(np.random.default_rng(1).normal(size=(100,4)))
    t=m.telemetry()
    assert t['observations']==100 and t['sketch_seen']==100
    assert 0 <= t['sampling_acceptance'] <= 1


def test_streamkm_reduction_uses_weighted_d2_sampling():
    m=StreamKMPlusPlusBaseline(3,coreset_size=10,buffer_size=20,seed=7,standardize=False)
    x=np.random.default_rng(2).normal(size=(25,3));m.process_batch(x)
    assert len(m.values) <= 15
    assert m.compressions >= 1
    assert m.diagnostics()['reduction']=='weighted_D2_nonuniform_representative_sampling'


def test_recent_method_parameters_have_one_config_source():
    tw=recent_method_settings(ROOT,'covertype','twstream')
    fra=recent_method_settings(ROOT,'covertype','fra_art')
    assert tw['radius'] == 10.411281070423867 and tw['k'] == 8
    assert tw['lambda'] == 0.0028 and tw['tau'] == 0.65
    assert fra['fractional_order'] == 0.5 and fra['vigilance'] == 0.8
    assert fra['choice'] == 0.001 and fra['beta'] == 1.0


def test_external_sources_are_pinned():
    assert len(TWEETEVAL_COMMIT)==40
    assert len(TWSTREAM_COMMIT)==40
    status=TWStreamOfficialAdapter.toolchain_status()
    assert set(status)=={'git','mvn','java','javac'}


def test_method_registry_separates_study_and_official_twstream():
    study=method_metadata('twstream'); official=method_metadata('twstream_official')
    assert study['native_output'] is False
    assert official['native_output'] is True
    assert study['implementation_id'] != official['implementation_id']
    assert official['source_commit'] == TWSTREAM_COMMIT


def test_official_twstream_csv_writer_accepts_sparse_input(tmp_path):
    x=sp.csr_matrix(np.array([[0.,1.,0.],[2.,0.,3.],[0.,0.,4.]],dtype=float))
    out=tmp_path/'input.csv'
    n,d=TWStreamOfficialAdapter._write_input_csv(out,x,block_rows=2)
    assert (n,d)==(3,3)
    assert np.allclose(np.loadtxt(out,delimiter=','),x.toarray())


def test_main_cli_uses_main_results_path():
    text=(ROOT/'main.py').read_text()
    assert 'results/main_results' in text
    assert 'results/final_execution' not in text
    assert 'results/rerun' not in text


def test_unified_primary_runner_covers_all_seven_methods_without_reading_results():
    text=(ROOT/'experiments'/'run_primary.py').read_text()
    for method in ('proposed','fixed_rank','clustream','denstream','streamkmpp','twstream','fra_art'):
        assert f'"{method}"' in text
    assert 'results/main_results' not in text


def test_modern_diagnostic_runner_does_not_merge_saved_proposed_results():
    text=(ROOT/'experiments'/'run_modern_all_datasets.py').read_text()
    assert 'results/main_results' not in text
    assert 'table6_direct_comparison.csv' not in text


def test_primary_aggregator_uses_fresh_execution_tree_only():
    text=(ROOT/'experiments'/'aggregate_primary_runs.py').read_text()
    assert 'results/main_results' not in text
    for output in ('execution_manifest.json','seedwise_results.csv','dataset_method_summary.csv','overall_equal_dataset_summary.csv'):
        assert output in text


def test_rank_runner_uses_the_main_proposed_engine_not_legacy_controller():
    text=(ROOT/'experiments'/'run_rank.py').read_text()
    assert 'from experiments.run_primary import' in text
    assert 'build_method' in text and 'evaluate_full_stream' in text
    assert 'rank_diagnostics' not in text
    assert '"engine": "main_proposed"' in text


def test_primary_baseline_settings_match_manuscript_table6():
    expected={
        'covertype':(1.50,1.50),
        'electricity':(1.00,1.00),
        'tweeteval':(0.85,0.85),
        'synthetic_gmm':(1.50,1.50),
    }
    for ds,(clu_r,den_eps) in expected.items():
        cfg=load_paper_config(ROOT,ds)
        b=cfg['baselines']
        assert float(b['clustream']['radius']) == clu_r
        assert float(b['denstream']['epsilon']) == den_eps
        assert float(b['denstream']['beta']) == 0.20
        assert int(b['denstream']['mu']) == 6
        assert float(b['denstream']['lambda']) == 0.01
        assert int(b['streamkmpp']['coreset_size']) == 500
        assert int(b['streamkmpp']['buffer_size']) == 1000


def test_config_has_no_separate_basis_decay_parameter():
    fields=set(ASCConfig.__dataclass_fields__)
    assert 'basis_decay' not in fields
    cfg=ASCConfig(original_dim=8,projection_dim=8,max_rank=8,initial_rank=8,min_rank=4)
    assert cfg.retained_energy == 0.95
    assert cfg.decay == 0.95


def test_streamkm_compression_preserves_total_summary_mass():
    m=StreamKMPlusPlusBaseline(2,coreset_size=10,buffer_size=20,seed=13,standardize=False)
    x=np.random.default_rng(9).normal(size=(45,2))
    m.process_batch(x)
    # Force the final pending coreset reduction as the macro stage does.
    m._compress()
    assert np.isclose(sum(m.weights),45.0,rtol=0,atol=1e-10)
    assert len(m.values) <= 10
