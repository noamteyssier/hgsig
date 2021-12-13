"""
Utility functions
"""
import numpy as np
from scipy.stats import hypergeom
from scipy.stats import fisher_exact


def hypergeom_test(r_draw, t_draw, overrep=True):
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

    param_M = r_draw.sum()
    param_n = r_draw
    param_N = t_draw.sum()
    param_k = t_draw

    if overrep:
        pval = hypergeom.sf(param_k, M=param_M, n=param_n, N=param_N)
    else:
        pval = hypergeom.cdf(param_k, M=param_M, n=param_n, N=param_N)
    return pval


def fishers_test(r_draw, t_draw, overrep=True):
    """
    performs significance testing between cluster
    representations as a fishers exact test.

    M : total number of observations in reference
    N : total number of observations in test
    """
    assert r_draw.size == t_draw.size
    if overrep:
        alt = "greater"
    else:
        alt = "less"

    num_obs = r_draw.size
    param_M = r_draw.sum()
    param_N = t_draw.sum()

    pval = np.zeros(num_obs)
    for i in np.arange(num_obs):
        table = np.array([
            [r_draw[i], param_M - r_draw[i]],
            [t_draw[i], param_N - t_draw[i]]])
        pval[i] = fisher_exact(table, alternative=alt)[1]
    return pval


