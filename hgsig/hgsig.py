"""
Differential Representation Testing
"""
from typing import List, Union
import numpy as np
from tqdm import tqdm

from .utils import hypergeom_test
from .utils import fishers_test


class HGSig:
    def __init__(
            self,
            clusters: np.ndarray,
            groups: np.ndarray,
            reference: Union[List[str], str],
            method: str = "hypergeom",
            agg: str = "sum"):

        """
        Differential Representation Testing

        Inputs:
            clusters: np.ndarray
                the array representing which cluster an observation belongs to
            groups: np.ndarray
                the array representing which group an observation belongs to
            reference: Union[List[str], str]
                the value(s) representing which group(s) to use as reference.
                Will aggregate the values of the references if multiple are provided.
            method: str
                the method to calculate significance with (hypergeom, fishers)
            agg: str
                the aggregation method to use for multiple reference values.
                known values : (sum[default], mean, median)
        """

        self.clusters = np.array(clusters)
        self.groups = np.array(groups)
        self.reference = np.array(reference)
        self.method = method
        self.agg = agg

        self.c_unique, self.c_counts = np.unique(self.clusters, return_counts=True)
        self.g_unique, self.g_counts = np.unique(self.groups, return_counts=True)

        self._validate_inputs()
        self._validate_agg()
        self._set_reference()
        self._initialize_distributions()
        self._initialize_references()
        self._validate_method()


    def _validate_inputs(self):
        """
        validates that the inputs are in the expected format
        """
        c_total = self.c_counts.sum()
        g_total = self.g_counts.sum()
        
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
        for idx, group in tqdm(enumerate(self.g_unique)):
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

    def run(self):
        """
        Performs the differential representation testing
        """
        pval_mat = np.zeros_like(self.distributions)
        for idx, dist in tqdm(enumerate(self.distributions)):
            for overrep in [True, False]:
                pval_mat[idx] = self.methods[self.method](
                        r_draw=self.ref_dist,
                        t_draw=dist,
                        overrep=overrep)
        return pval_mat

    def __repr__(self) -> str:
        """
        string representation of object
        """
        name = "HGSig"
        num_g = f"n_groups: {self.g_unique.size}"
        num_c = f"n_groups: {self.c_unique.size}"
        method = f"method: {self.method}"
        reference = f"reference: {self.reference}"
        attr = [name, num_g, num_c, method, reference]
        return "\n  ".join(attr)
