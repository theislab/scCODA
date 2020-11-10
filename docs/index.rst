.. compositionalDiff documentation master file, created by
   sphinx-quickstart on Tue Jan  7 09:13:28 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to scCODA's documentation!
=============================================

This package contains statistical models to analyze changes in compositional data, especially from single-cell RNA-seq experiments.

The package is available on `github <https://github.com/johannesostner/SCDCdm_public>`_.

Please also check out the `tutorial <https://github.com/johannesostner/SCDCdm_public/blob/master/tutorials/Tutorial.ipynb>`_ that goes over the most important functionalities.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   data
   models

Modules
=======

.. autosummary::
   :toctree: modules

   sccoda.model.dirichlet_models
   sccoda.model.other_models
   sccoda.util.cell_composition_data
   sccoda.util.comp_ana
   sccoda.util.data_generation
   sccoda.util.multi_parameter_analysis_functions
   sccoda.util.multi_parameter_sampling
   sccoda.util.result_classes

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
