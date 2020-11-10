Data
====

scCODA uses a standard regression data structure for the models.
We have a data matrix with count data of dimension NxK, and a covariate matrix of dimension NxD, often with binary covariates.

Note that the count data must be of compositional nature, i.e. the sum of counts for each sample is fixed.
However, the data does not need to be normalized. scCODA works on the integer count data.

The models in scCODA use their own data structure, that is based on the anndata package.
Hereby, ``data.X`` stores the cell counts, and ``data.obs`` stores the information about the covariates.

Data generation methods
~~~~~~~~~~~~~~~~~~~~~~~

To test new modeling approaches, ``sccoda.util.data_generation`` contains methods to generate compositional data with different properties (``sccoda.util.data_generation``).


Data import methods
~~~~~~~~~~~~~~~~~~~

``sccoda.util.cell_composition_data`` contains methods to import count data from various sources into the data structure used by scCODA.
You can either import data directly from a pandas DataFrame via ``from_pandas()``, or get the count data from single-cell expression data used in scanpy.
Considering you have one anndata object with the single-cell expression data for each sample, ``from_scanpy_list`` (for in-memory data) and ``from_scanpy_dir`` (for data stored on disk) can transform the information from these files drectly into a compositional analysis dataset.


