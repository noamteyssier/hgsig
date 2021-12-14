"""
Utility functions
"""
import numpy as np
from scipy.stats import hypergeom
from scipy.stats import fisher_exact


def hypergeom_test(r_draw, t_draw):
    """
    performs significance testing between cluster
    representations as a hypergeometric test.

    M : total number of observations in reference
    n : number of observations in a specific category in reference
    N : number of draws (aka total number of observations in test)
    k : number of observations in a specific category in test

    pval : survival P( k | HyperGeom( M, n, N ) )

    inputs:
        r_draw : np.ndarray
            the reference draw
        t_draw : np.ndarray
            the test draw
        overrep : bool
            Whether to specifically test for overrepresentation.
            Otherwise will test for underrepresentation.
    """
    assert r_draw.size == t_draw.size
    if not np.all(r_draw >= t_draw):
        raise ValueError(
                "Some values are larger in the test draw than in the reference")
    if np.all(r_draw == t_draw):
        return np.ones_like(r_draw)

    param_M = r_draw.sum()
    param_n = r_draw
    param_N = t_draw.sum()
    param_k = t_draw

    pval_high = hypergeom.sf(param_k, M=param_M, n=param_n, N=param_N)
    pval_low = hypergeom.cdf(param_k, M=param_M, n=param_n, N=param_N)

    return multidim_min(pval_high, pval_low)


def fishers_test(r_draw, t_draw):
    """
    performs significance testing between cluster
    representations as a fishers exact test.

    M : total number of observations in reference
    N : total number of observations in test
    """
    assert r_draw.size == t_draw.size
    if np.all(r_draw == t_draw):
        return np.ones_like(r_draw)

    num_obs = r_draw.size
    param_M = r_draw.sum()
    param_N = t_draw.sum()

    pval_high = np.zeros(num_obs)
    pval_low = np.zeros(num_obs)
    for i in np.arange(num_obs):
        table = np.array([
            [r_draw[i], param_M - r_draw[i]],
            [t_draw[i], param_N - t_draw[i]]])
        pval_high[i] = fisher_exact(table, alternative="greater")[1]
        pval_low[i] = fisher_exact(table, alternative="less")[1]

    return multidim_min(pval_high, pval_low)


def multidim_min(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    takes the minimum value between two arrays of equal size
    """
    assert x.size == y.size
    mat = np.vstack([x, y])
    return np.min(mat, axis=0)


def percent_change(r_draw, t_draw):
    """
    calculates the percent change between a reference group
    and a test group. Will first normalize the vectors so that
    their total will sum to 1
    """
    assert r_draw.size == t_draw.size
    r_norm = r_draw / r_draw.sum()
    t_norm = t_draw / t_draw.sum()
    return (t_norm - r_norm) / r_norm


def false_discovery_rate(pval):
    """
    converts the pvalues into false discovery rate q-values
    """
    dim = pval.shape
    qval = p_adjust_bh(pval.ravel())
    return qval.reshape(dim)


def p_adjust_bh(p):
    """
    Benjamini-Hochberg p-value correction for multiple hypothesis testing.
    https://stackoverflow.com/a/33532498
    """
    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]

