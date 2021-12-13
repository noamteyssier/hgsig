"""
Testing suite for `hgsig`
"""

import pytest
import numpy as np
from hgsig import HGSig

SEED = 42
NUM = 10000
NUM_X = 3000
NUM_C = 7
NUM_G = 50
np.random.seed(SEED)


def build_clusters(size=NUM):
    """
    creates the clusters array
    """
    return np.array([
        f"c{i}" for i in np.random.choice(NUM_C, size=size)])


def build_groups(size=NUM):
    """
    creates the groups array
    """
    return np.array([
        f"g{i}" for i in np.random.choice(NUM_G, size=size)])


def test_init_multiple_ref():
    """
    tests the initialization of the object
    w. multiple references
    """
    reference = ["g0", "g1"]
    clusters = build_clusters()
    groups = build_groups()

    HGSig(clusters, groups, reference)
    HGSig(clusters, groups, reference, method="fishers")

    with pytest.raises(ValueError):
        HGSig(clusters, groups, reference, method="not_a_method")

    assert True

def test_init_single_ref():
    """
    tests the initialization of the object
    w. single reference
    """
    reference = "g0"
    clusters = build_clusters()
    groups = build_groups()

    # makes sure that there will be enough
    # of the reference to not hit the hypergeom
    # overdrawing condition
    groups[:NUM_X] = reference

    HGSig(clusters, groups, reference)
    HGSig(clusters, groups, reference, method="fishers")

    with pytest.raises(ValueError):
        HGSig(clusters, groups, reference, method="not_a_method")

    assert True

def test_init_overdrawn_reference():
    """
    tests the exit condition where the test draw
    has more of a category than the reference
    """
    reference = ["g0", "g1"]
    clusters = build_clusters()
    groups = build_groups()
    with pytest.raises(ValueError):
        # by definition the mean between two will cause the references
        # to be over or underrepresented
        HGSig(clusters, groups, reference, method="hypergeom", agg="mean")
    assert True

def test_init_agg():
    """
    tests the aggregating conditions for the references
    """
    reference = ["g0", "g1"]
    clusters = build_clusters()
    groups = build_groups()

    for agg in ["sum", "mean", "median"]:
        HGSig(
            clusters,
            groups,
            reference,
            method="fishers",
            agg=agg)

    with pytest.raises(ValueError):
        HGSig(clusters, groups, reference, method="fishers", agg="not_an_agg")

def test_run_single_reference():
    """
    runs the method through a range of conditions
    """
    reference = "g0"
    clusters = build_clusters()
    groups = build_groups()

    # makes sure that there will be enough
    # of the reference to not hit the hypergeom
    # overdrawing condition
    groups[:NUM_X] = reference

    for method in ["fishers", "hypergeom"]:
        hgs = HGSig(clusters, groups, reference, method=method)
        pval = hgs.run()
        assert isinstance(pval, np.ndarray)
        assert pval.shape == (np.unique(groups).size, np.unique(clusters).size)

def test_run_multi_reference():
    """
    runs the method through a range of conditions
    """
    for i in np.arange(2, 6):
        reference = [f"g{i}" for i in np.random.choice(NUM_G, size=i, replace=False)]
        clusters = build_clusters()
        groups = build_groups()

        # makes sure that there will be enough
        # of the reference to not hit the hypergeom
        # overdrawing condition
        groups[:NUM_X] = np.random.choice(reference, NUM_X)

        for method in ["fishers", "hypergeom"]:
            hgs = HGSig(clusters, groups, reference, method=method)
            pval = hgs.run()
            assert isinstance(pval, np.ndarray)
            assert pval.shape == (np.unique(groups).size, np.unique(clusters).size)

        # only run aggregation tests on fishers because it is not guaranteed to pass
        # all tests with hypergeometric testing
        for agg in ["sum", "mean", "median"]:
            hgs = HGSig(clusters, groups, reference, method="fishers", agg=agg)
            pval = hgs.run()
            assert isinstance(pval, np.ndarray)
            assert pval.shape == (np.unique(groups).size, np.unique(clusters).size)
