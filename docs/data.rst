Data
====

compositionalDiff uses a standard regression data structure for the models.
We have a data matrix with count data of dimension NxK, and a covariate matrix of dimension NxD, often with binary covariates.
Note that the count data must be of compositional nature, i.e. the sum of counts for each sample is fixed.

Data generation methods
~~~~~~~~~~~~~~~~~~~~~~~

To test new modeling approaches, compositionalDiff contains methods to generate compositional data with different properties (`cd.util.compositional_analysis_generation_toolbox`).

