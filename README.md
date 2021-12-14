# HGSig
This tool is used to measure the differential clustered representation of grouped objects. The original motivation was in CRISPRi single-cell sequencing data and measuring the differential representation of individual knockdowns in each of the leiden clusters. This was used to guide whether a knockdown had a significant representation from the non-targeting controls and so provide a hint at the potential function of that knockdown. This tool is a means of generalizing the code to any sort of clusters and groups with references and provide an API for testing different differential representation strategies in a reproducible way. 

# Installation
## pip 
```bash
pip install hgsig
```

## github
```bash
git clone https://github.com/noamteyssier/hgsig
cd hgsig
pip install .
pytest -v
```

# Usage
This tool is intended to be used as a python module. 

## Differential Representation with Multiple References
```python
from hgsig import HGSig

# Number of observations
size = 10000

# Number of Groups
n_groups = 50

# Number of Clusters
n_clusters = 8

# randomly assign clusters
clusters = np.array([
    f"c{i}" for i in np.random.choice(n_clusters, size=size)
])

# randomly assign groups
groups = np.array([
    f"g{i}" for i in np.random.choice(n_groups, size=size)
])

# initialize object
hgs = HGSig(
    clusters,
    groups,
    reference=["g0", "g3"]
)

# run testing
pval = hgs.run()

```

## Differential Representation using a fisher's exact test
```python
from hgsig import HGSig

# Number of observations
size = 10000

# Number of Groups
n_groups = 50

# Number of Clusters
n_clusters = 8

# randomly assign clusters
clusters = np.array([
    f"c{i}" for i in np.random.choice(n_clusters, size=size)
])

# randomly assign groups
groups = np.array([
    f"g{i}" for i in np.random.choice(n_groups, size=size)
])

# initialize object
hgs = HGSig(
    clusters,
    groups,
    reference=["g0", "g3"],
    method="fishers"
)

# run testing
pval = hgs.run()
```

## Differential Representation using a single reference group
It is highly recommended here to use a fisher's exact test because the hypergeometric testing conditions will generally not be satisfied using only a single group. This is because if the groups are of equal sizes it is likely you will have more than the original number of observations in the reference group and thus fail the prerequirements for the hypergeometric test. This condition is not required for a fisher's exact test and so it should be used in this case.
```python
from hgsig import HGSig

# Number of observations
size = 10000

# Number of Groups
n_groups = 50

# Number of Clusters
n_clusters = 8

# randomly assign clusters
clusters = np.array([
    f"c{i}" for i in np.random.choice(n_clusters, size=size)
])

# randomly assign groups
groups = np.array([
    f"g{i}" for i in np.random.choice(n_groups, size=size)
])

# initialize object
hgs = HGSig(
    clusters,
    groups,
    reference="g0",
    method="fishers"
)

# run testing
pval = hgs.run()
```

## Differential Representation of multiple groups with an alternative aggregation function
The default aggregation function for the references is to sum the values across each of the conditions, but it is also possible to use alternative aggregation strategies if it is of interest.
```python
from hgsig import HGSig

# Number of observations
size = 10000

# Number of Groups
n_groups = 50

# Number of Clusters
n_clusters = 8

# randomly assign clusters
clusters = np.array([
    f"c{i}" for i in np.random.choice(n_clusters, size=size)
])

# randomly assign groups
groups = np.array([
    f"g{i}" for i in np.random.choice(n_groups, size=size)
])

# initialize object
hgs = HGSig(
    clusters,
    groups,
    reference=["g0", "g1", "g2"],
    method="fishers",
    agg="mean"
)

# run testing
pval = hgs.run()
```
