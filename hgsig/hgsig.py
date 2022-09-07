"""
Differential Representation Testing
"""
from typing import List, Union
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from .utils import (
    chisquare_test,
    fishers_test,
    hypergeom_test,
    percent_change,
    false_discovery_rate)


class HGSig:
    """
    Differential Representation Testing

    Methods
    -------
    fit:
        performs the representation test
    get_clusters:
        returns the unique cluster names
    get_groups:
        returns the unique group names
    get_pval:
        returns the calculated pvalues of the significance test
    get_qval:
        returns the BH-adjusted pvalues of the significance test
    get_pcc:
        returns the percent change of each groups cluster representation
    get_nlf:
        returns the negative log adjusted pvalues (negative log fdr)
    get_snlf:
        returns the signed negative log adjusted pvalues (signed negative log fdr)

    Attributes
    ----------
    clusters: npt.NDArray[np.object_]
        The provided clusters
    groups: npt.NDArray[np.object_]
        The provided groups
    reference: npt.NDArray[np.object_]
        The provided references
    method: str
        The provided testing method
    agg: str
        The provided aggregation method
    """
    def __init__(
            self,
            clusters: npt.ArrayLike,
            groups: npt.ArrayLike,
            reference: Union[List[str], str, npt.ArrayLike],
            method: str = "hypergeom",
            agg: str = "sum"):
        """

        Parameters
        ----------
        clusters: npt.ArrayLike
            the array representing which cluster an observation belongs to
        groups: npt.ArrayLike
            the array representing which group an observation belongs to
        reference: Union[List[str], str, npt.ArrayLike]
            the value(s) representing which group(s) to use as reference.
            Will aggregate the values of the references if multiple are provided.
        method: str
            the method to calculate significance with (hypergeom, fishers, chisquare)
        agg: str
            the aggregation method to use for multiple reference values.
            known values : (sum[default], mean, median)
        """


        self.clusters = np.array(clusters)
        self.groups = np.array(groups)
        self.reference = np.array(reference)
        self.method = method
        self.agg = agg
        self._isfit = False

        self._build_unique()
        self._validate_inputs()
        self._validate_agg()
        self._set_reference()
        self._initialize_distributions()
        self._initialize_references()
        self._validate_method()

        self.pval_mat = np.zeros_like(self.distributions)
        self.pcc_mat = np.zeros_like(self.distributions)
        self.qval_mat = np.zeros_like(self.distributions)
        self.nlf_mat = np.zeros_like(self.distributions)
        self.snlf_mat = np.zeros_like(self.distributions)

    def _build_unique(self):
        """
        determines the unique group and cluster names and their respective
        counts
        """
        self.c_unique, self.c_counts = np.unique(
                self.clusters, return_counts=True)
        self.g_unique, self.g_counts = np.unique(
                self.groups, return_counts=True)

    def _validate_inputs(self):
        """
        validates that the inputs are in the expected format
        """
        c_total: int = self.c_counts.sum()
        g_total: int = self.g_counts.sum()

        if c_total != g_total:
            raise ValueError(
                    f"Provided inputs are different sizes: {c_total} != {g_total}")

        if len(self.clusters) <= 1:
            raise ValueError(
                    "Provided inputs must contain more than 2 observations")

        if not np.all(np.isin(self.reference, self.g_unique)):
            raise ValueError(
                    f"Provided reference ({self.reference}) not in provided groups")

        if self.g_unique.size <= 1:
            raise ValueError(
                    "Provided groups must have more than one value")

        if self.c_unique.size <= 1:
            raise ValueError(
                    "Provided clusters must have more than one value")

    def _validate_agg(self):
        """
        validates the aggregation method for the reference groups
        """
        self.agg_metric = {
            "sum": np.sum,
            "mean": np.mean,
            "median": np.median}
        if self.agg not in self.agg_metric.keys():
            raise ValueError(
                f"""
                Provided aggregation {self.agg} not in known metrics:
                {', '.join(self.agg_metric.keys())}
                """)

    def _set_reference(self):
        """
        sets the reference index
        """
        self.ref_idx = np.flatnonzero(np.isin(self.g_unique, self.reference))

    def _initialize_distributions(self):
        """
        calculates the cluster representation of each cluster~group
        """
        self.distributions = np.zeros((self.g_unique.size, self.c_unique.size))
        for idx, group in tqdm(enumerate(self.g_unique), desc="Calculating Distributions"):
            self.distributions[idx] = np.array([
                np.sum(
                    (self.clusters == cluster) &
                    (self.groups == group))
                for cluster in self.c_unique])

    def _initialize_references(self):
        """
        calculates the cluster represenation of each cluster~group
        for the reference groups
        """
        self.ref_dist = self.agg_metric[self.agg](
                self.distributions[self.ref_idx],
                axis=0)

    def _validate_method(self):
        """
        confirms that the provided method is known
        and that is applicable for the dataset
        """
        self.methods = {
                "fishers": fishers_test,
                "chisquare": chisquare_test,
                "hypergeom": hypergeom_test}

        if self.method not in self.methods.keys():
            raise ValueError(
                f"""
                Provided method {self.method} not in known methods:
                {', '.join(self.methods.keys())}
                """)

        if self.method == "hypergeom":
            for dist in self.distributions:
                if np.any(dist > self.ref_dist):
                    raise ValueError(
                        """
                        Cannot perform hypergeometric testing as one or more test
                        distributions contain values higher than in the reference
                        distributions. Please increase the size of the reference
                        dataset or rerun the tool with `method=fishers`
                        """)

    def fit(self):
        """
        Performs the differential representation testing
        """
        self.pval_mat = np.zeros_like(self.distributions)
        self.pcc_mat = np.zeros_like(self.distributions)

        for idx, dist in tqdm(enumerate(self.distributions), desc="Calculating Significance"):

            # calculate the significance
            self.pval_mat[idx] = self.methods[self.method](
                    r_draw=self.ref_dist,
                    t_draw=dist)

            # calculate the percent change
            self.pcc_mat[idx] = percent_change(
                    self.ref_dist,
                    dist)

        self.qval_mat = self._calculate_fdr()
        self.nlf_mat = self._calculate_nlf()
        self.snlf_mat = self._calculate_snlf()
        self._isfit = True

    def _calculate_fdr(self) -> npt.NDArray[np.float64]:
        """
        calculates the false discovery rate
        """
        return false_discovery_rate(self.pval_mat)

    def _calculate_nlf(self) -> npt.NDArray[np.float64]:
        """
        calculates the negative log false discovery rate
        """
        return -np.log10(self.qval_mat)

    def _calculate_snlf(self) -> npt.NDArray[np.float64]:
        """
        calculates the signed negative log false discovery rate
        """
        return np.sign(self.pcc_mat) * self.nlf_mat

    def get_pval(self) -> npt.NDArray[np.float64]:
        """
        retrieve the pval matrix
        """
        if not self._isfit:
            raise AttributeError(
                "Please run the .fit() method first")
        return self.pval_mat

    def get_qval(self) -> npt.NDArray[np.float64]:
        """
        retrieve the q-value matrix
        """
        if not self._isfit:
            raise AttributeError(
                "Please run the .fit() method first")
        return self.qval_mat

    def get_nlf(self) -> npt.NDArray[np.float64]:
        """
        retrieve the -log10 transformed q-value matrix
        """
        if not self._isfit:
            raise AttributeError(
                "Please run the .fit() method first")
        return self.nlf_mat

    def get_snlf(self) -> npt.NDArray[np.float64]:
        """
        retrieve the percent change signed -log10 transformed q-value matrix
        """
        if not self._isfit:
            raise AttributeError(
                "Please run the .fit() method first")
        return self.snlf_mat

    def get_pcc(self) -> npt.NDArray[np.float64]:
        """
        retrieve the percent change matrix
        """
        if not self._isfit:
            raise AttributeError(
                "Please run the .fit() method first")
        return self.pcc_mat

    def get_groups(self) -> npt.NDArray[np.object_]:
        """
        retrieve the group names
        """
        return self.g_unique

    def get_clusters(self) -> npt.NDArray[np.object_]:
        """
        retrieve the cluster names
        """
        return self.c_unique

    def __repr__(self) -> str:
        """
        string representation of object
        """
        name = "HGSig"
        num_g = f"n_groups: {self.g_unique.size}"
        num_c = f"n_groups: {self.c_unique.size}"
        method = f"method: {self.method}"
        reference = f"reference: {self.reference}"
        fit = f"is fit: {self._isfit}"
        attr = [name, num_g, num_c, method, reference, fit]
        return "\n  ".join(attr)
