from pathlib import Path
import json, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from asc_stream.config import ASCConfig
from asc_stream.model import AdaptiveSketchStreamClusterer

def test_paper_configuration_defaults():
    c=ASCConfig(original_dim=256)
    assert c.leverage_mode=='sample'
    assert c.uniform_floor==0.05
    assert c.leverage_sampling_rate==0.65
    assert c.min_sampling_probability==0.15
    assert c.max_sampling_probability==1.0
    assert c.leverage_regularization==1e-3
    assert c.use_adapted_representation_for_clustering is True

def test_sampling_only_controls_sketch_admission():
    rng=np.random.default_rng(123);x=rng.normal(size=(180,16))
    c=ASCConfig(original_dim=16,projection_dim=8,initial_rank=4,min_rank=2,max_rank=8,window_size=128,max_microclusters=32,seed=7,projection_seed=7,leverage_mode='sample',min_sampling_probability=.15,max_sampling_probability=.25,leverage_sampling_rate=.2,basis_update_interval=20)
    m=AdaptiveSketchStreamClusterer(c,standardize=False)
    for row in x:m.process_one(row)
    assert m.time==len(x)
    assert m.sketch.accepted < len(x)
    assert len(m.rank_history)==len(x)
    assert len(m.microclusters.clusters)>0

def test_rank_adapted_representation_is_used_for_clustering():
    c=ASCConfig(original_dim=8,projection_dim=8,initial_rank=4,min_rank=2,max_rank=8,window_size=64,projection_mode='identity',use_adapted_representation_for_clustering=True)
    m=AdaptiveSketchStreamClusterer(c,standardize=False)
    z=np.arange(8,dtype=float)
    adapted=m.sketch.adapted(z)
    assert adapted.shape==z.shape
    assert np.allclose(adapted,(z@m.basis)@m.basis.T)

def test_controlled_data_shape():
    d=np.load(ROOT/'data/controlled/representation_drift_stream.npz')
    assert d['x'].shape==(9000,256)
    assert d['y'].shape==(9000,)

def test_main_controlled_raw_count():
    for m in ['proposed','fixed_rank','twstream','fra_art']:
        assert len(list((ROOT/f'results/main_results/controlled/raw/{m}').glob('seed_*.json')))==10

def test_sparse_text_projection_matches_dense_without_online_scaling():
    from scipy import sparse
    rng=np.random.default_rng(44)
    x=rng.normal(size=(12,20))
    x[np.abs(x)<0.8]=0.0
    c=ASCConfig(original_dim=20,projection_dim=8,initial_rank=4,min_rank=2,max_rank=8,window_size=64,seed=11,projection_seed=11,basis_update_interval=1000)
    dense=AdaptiveSketchStreamClusterer(c,standardize=False)
    sparse_model=AdaptiveSketchStreamClusterer(c,standardize=False)
    zd=dense.process_batch(x)
    zs=sparse_model.process_batch(sparse.csr_matrix(x))
    assert np.allclose(zd,zs)


def test_denstream_study_baseline_tracks_potential_or_outlier_microclusters():
    from asc_stream.baselines import DenStreamBaseline
    rng=np.random.default_rng(5)
    x=np.r_[rng.normal(0,.05,size=(80,2)),rng.normal(2,.05,size=(80,2))]
    model=DenStreamBaseline(2,radius=.35,max_microclusters=32,seed=7,standardize=False,beta=.2,mu=6.0,fading_lambda=.01)
    z=model.process_batch(x)
    centers=model.macro_centers(2)
    assert z.shape==x.shape
    assert centers.shape==(2,2)
    d=model.diagnostics()
    assert d['potential_microclusters']+d['outlier_microclusters']>0

def test_controlled_rebuild_matches_fixed_observations_and_labels():
    from asc_stream.datasets import rebuild_controlled_representation_stream
    fixed=np.load(ROOT/'data/controlled/representation_drift_stream.npz')
    rebuilt=rebuild_controlled_representation_stream()
    # BLAS/NumPy versions may differ at the last floating-point bit while
    # representing the same declared stream. Labels and drift locations remain
    # exact; numeric observations are compared at machine-precision tolerance.
    np.testing.assert_allclose(rebuilt.x,fixed['x'],rtol=0,atol=2e-15)
    assert np.array_equal(rebuilt.y,fixed['y'])
    assert tuple(rebuilt.change_points)==tuple(int(v) for v in fixed['drift_points'])
