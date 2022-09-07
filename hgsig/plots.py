"""
Plotting functions for HGSig objects
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .hgsig import HGSig


def _filter_significant(
        mat: pd.DataFrame,
        hgs: HGSig,
        use_pval: bool = False,
        threshold: float = 0.05) -> pd.DataFrame:
    """
    filters the dataframe to only the significantly differentially
    expressed clusters/guides
    """
    values = hgs.get_pval() if use_pval else hgs.get_qval()
    min_sig = values.min(axis=1)
    mask = min_sig < threshold
    return mat.iloc[mask]


def plot_hgsig(
        hgs: HGSig,
        basis: str = "snlf",
        transpose: bool = False,
        filter_significant: bool = True,
        use_pval: bool = False,
        show: bool = True,
        threshold: float = 0.05,
        cmap="seismic",
        linewidths=0.5,
        linecolor="black",
        center=0,
        vmin=-2,
        vmax=2,
        **kwargs):
    """
    plotting clustermap for HGSig object
    """
    basis_map = {
            "snlf": hgs.get_snlf,
            "nlf": hgs.get_nlf,
            "pcc": hgs.get_pcc}
    if basis not in basis_map.keys():
        raise KeyError(
                "Provided Basis {basis} not in implemented bases [snlf, nlf, pcc]")

    mat = pd.DataFrame(
        basis_map[basis]().copy(),
        index=hgs.get_groups(),
        columns=hgs.get_clusters())

    if filter_significant:
        mat = _filter_significant(
                mat, hgs, 
                use_pval=use_pval, 
                threshold=threshold)

    if transpose:
        mat = mat.transpose()

    sns.clustermap(
        mat,
        cmap=cmap,
        linewidths=linewidths,
        linecolor=linecolor,
        center=center,
        vmin=vmin,
        vmax=vmax,
        **kwargs)

    if show:
        plt.show()
