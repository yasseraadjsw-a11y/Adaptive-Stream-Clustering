import numpy as np
from asc_stream.comparators import FRAARTComparator


def test_fra_numba_batch_matches_reference_pointwise():
    rng=np.random.default_rng(123)
    x=rng.random((120,12),dtype=np.float64)
    fast=FRAARTComparator(12,7,a=.5,vigilance=.8,choice=.001,beta=1.0,max_stream_points=len(x))
    ref=FRAARTComparator(12,7,a=.5,vigilance=.8,choice=.001,beta=1.0,max_stream_points=len(x))
    fast.process_batch(x)
    for row in x:
        ref.process_one(row)
    assert fast.m == ref.m
    np.testing.assert_array_equal(fast.counts[:fast.m],ref.counts[:ref.m])
    # The compiled scalar loop and NumPy's vectorized reference may differ by
    # a few binary rounding units while remaining numerically equivalent.
    tol=8*np.finfo(np.float64).eps
    np.testing.assert_allclose(fast.prototypes[:fast.m],ref.prototypes[:ref.m],rtol=tol,atol=tol)
    np.testing.assert_allclose(fast.raw_sums[:fast.m],ref.raw_sums[:ref.m],rtol=tol,atol=tol)
