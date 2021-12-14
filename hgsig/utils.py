"""
Utility functions
"""
from typing import Union
import numpy as np
from scipy.stats import (
    hypergeom,
    fisher_exact,
    chi2_contingency)


def hypergeom_test(
        r_draw: np.ndarray,
        t_draw: np.ndarray) -> np.ndarray:
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


def exact_test(
        r_draw: np.ndarray,
        t_draw: np.ndarray,
        method: Union[fisher_exact, chi2_contingency]) -> np.ndarray:
    """
    perform a conditional test by creating a contigency table.

    inputs:
        r_draw: np.ndarray
            the numerical values representing the number of observations in each category
            for the reference distribution
        t_draw: np.ndarray
            the numerical values representing the number of observations in each category
            for the test distribution
        method: Union[fisher_exact, chi2_contingency]
            the statistical test to measure significance
    """
    assert r_draw.size == t_draw.size
    if np.all(r_draw == t_draw):
        return np.ones_like(r_draw)

    num_obs = r_draw.size
    param_M = r_draw.sum()
    param_N = t_draw.sum()

    pval = np.zeros(num_obs)
    for i in np.arange(num_obs):
        table = np.array([
            [r_draw[i], param_M - r_draw[i]],
            [t_draw[i], param_N - t_draw[i]]])

        pval[i] = method(table)[1]

    return pval 


def fishers_test(
        r_draw: np.ndarray,
        t_draw: np.ndarray) -> np.ndarray:
    """
    performs significance testing between cluster
    representations as a fishers exact test.

    M : total number of observations in reference
    N : total number of observations in test
    """
    return exact_test(r_draw, t_draw, fisher_exact) 


def chisquare_test(
        r_draw: np.ndarray,
        t_draw: np.ndarray) -> np.ndarray:
    """
    performs significance testing between cluster
    representations as a chi-squared test.

    M : total number of observations in reference
    N : total number of observations in test
    """
    return exact_test(r_draw, t_draw, chi2_contingency)


def multidim_min(
        x: np.ndarray,
        y: np.ndarray) -> np.ndarray:
    """
    takes the minimum value between two arrays of equal size
    """
    assert x.size == y.size
    mat = np.vstack([x, y])
    return np.min(mat, axis=0)


def percent_change(
        r_draw: np.ndarray,
        t_draw: np.ndarray) -> np.ndarray:
    """
    calculates the percent change between a reference group
    and a test group. Will first normalize the vectors so that
    their total will sum to 1
    """
    assert r_draw.size == t_draw.size
    r_norm = r_draw / r_draw.sum()
    t_norm = t_draw / t_draw.sum()
    return (t_norm - r_norm) / r_norm


def false_discovery_rate(
        pval: np.ndarray) -> np.ndarray:
    """
    converts the pvalues into false discovery rate q-values
    """
    dim = pval.shape
    qval = p_adjust_bh(pval.ravel())
    return qval.reshape(dim)


def p_adjust_bh(
        p: np.ndarray) -> np.ndarray:
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

