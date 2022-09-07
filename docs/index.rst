hgsig
#####

Background
----------
This tool is used to measure the differential clustered representation of grouped objects. 
The original motivation was in CRISPRi single-cell sequencing data and measuring the differential representation of individual knockdowns in each of the leiden clusters.

This was used to guide whether a knockdown had a significant representation from the non-targeting controls and so provide a hint at the potential function of that knockdown.
This tool is a means of generalizing the code to any sort of clusters and groups with references and provide an API for testing different differential representation strategies in a reproducible way.

Implementation
--------------
This tool performs under/overrepresentation tests either with a hypergeometric test, fisher's exact test, or a chi-square test.
You supply which of your groups should be considered references, and this will handle the rest!

Usage
-----
There is only one class exposed in this module: :func:`HGSig <hgsig.HGSig>` and a single function: :func:`plot_hgsig <hgsig.plot_hgsig>` which takes an ``HGSig`` object as input.

Toy Dataset
^^^^^^^^^^^
Let's first generate a toy dataset to show what kind of data this is expecting.
The ``clusters`` and ``groups`` below would be equivalent to taking the respective columns from a dataframe.

.. code-block:: python3

   import numpy as np

   size = 10000   # number of observations (akin to number of cells)
   n_groups = 50  # number of groups (akin to number of sgRNAs)
   n_clusters = 8 # number of clusters (aking to number of leiden clusters)
 
   # randomly assign clusters
   clusters = np.array([
       f"c{i}" for i in np.random.choice(n_clusters, size=size)
   ])
 
   # randomly assign groups
   groups = np.array([
       f"g{i}" for i in np.random.choice(n_groups, size=size)
   ])

   # the `clusters` object's size will be equal to the number of observations
   # you can consider this to be the `cluster` label on a dataframe for a set
   # of observations
   assert(clusters.size == size)

   # the `groups` object's size will be equal to the number of observations as well
   assert(groups.size == clusters.size)
   

Performing a Representation Test
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Now that we have our data ready we can perform a representation test with the ``HGSig`` class.
By default the representation test used will be a hypergeometric test.

.. code-block:: python3
   
   from hgsig import HGSig

   # initialize object
   hgs = HGSig(
       clusters,
       groups,
       reference=["g0", "g3"] # let's select `g0` and `g3` as our reference groups
   )

   # perform the test
   hgs.fit()

   # recover results
   pval = hgs.get_pval() # get individual pvalues
   pcc = hgs.get_pcc() # get percent changes

Customizing the Representation Test
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A hypergeometric test may not be applicable in all cases - and so I've included other significance tests.
These can be supplied with ``method`` flag on the ``HGSig`` constructor. 

The available tests are:  ``[ 'hypergeom', 'fishers', 'chisquare' ]``.

.. code-block:: python3
   
   # initialize object
   hgs = HGSig(
       clusters,
       groups,
       reference=["g0", "g3"],
       method='fishers'
   )

   # perform the test
   hgs.fit()

Visualizing the Results
^^^^^^^^^^^^^^^^^^^^^^^
Once the representation test is complete you can visualize the results on a clustermap with the :func:`plot_hgsig <hgsig.plot_hgsig>` function.

You can choose which basis (percent change, signed negative log fdr, etc.) you'd like to view the clustering with.

This function acts as a lightweight wrapper over the `seaborn clustermap <https://seaborn.pydata.org/generated/seaborn.clustermap.html>`_ function - so you can pass in all keyword arguments you would expect.

>>> plot_hgsig(hgs, basis='snlf', filter_significant=True)


API
---
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api



Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
