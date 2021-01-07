.. automodule:: sccoda

API
===

We advise to import scCODA in a python session via::

    import sccoda
    dat = sccoda.util.cell_composition_data
    ana = sccoda.util.compositional_analysis
    viz = sccoda.util.data_visualization

The workflow in scCODA starts with reading in cell count data (``dat``) and visualizing them (``viz``)
or synthetically generating cell count data (``util.data_generation``).

Data acquisition
----------------

**Integrating data sources (dat)** (scanpy or pandas)

.. autosummary::
    :toctree: .

    sccoda.util.cell_composition_data.from_pandas
    sccoda.util.cell_composition_data.from_scanpy
    sccoda.util.cell_composition_data.from_scanpy_dir
    sccoda.util.cell_composition_data.from_scanpy_list
    sccoda.util.cell_composition_data.read_anndata_one_sample


**Synthetic data generation**

.. autosummary::
    :toctree: .

    sccoda.util.data_generation.generate_case_control
    sccoda.util.data_generation.b_w_from_abs_change
    sccoda.util.data_generation.counts_from_first
    sccoda.util.data_generation.sparse_effect_matrix

**Compositional data visualization**

Compositional datasets can be plotted via the methods in ``util.data_visualization``.

.. autosummary::
    :toctree: .

    sccoda.util.data_visualization.stacked_barplot
    sccoda.util.data_visualization.boxplots
    sccoda.util.data_visualization.stackbar

Model setup and inference
-------------------------

Using the scCODA model is easiest by generating an instance of ``ana.CompositionalAnalysis``.
By specifying the formula via the `patsy <https://patsy.readthedocs.io/en/latest/>`_ syntax, many combinations and
transformations of the covariates can be performed without redefining the covariate matrix. Also, the reference cell
type needs to be specified in this step.

**The scCODA model**

.. autosummary::
    :toctree: .

    sccoda.util.comp_ana.CompositionalAnalysis
    sccoda.model.dirichlet_models.CompositionalModel
    sccoda.model.dirichlet_models.ReferenceModel

**Utility functions**

.. autosummary::
    :toctree: .

    sccoda.util.helper_functions.sample_size_estimate

Result evaluation
-----------------

Executing an inference method on a compositional model produces a ``sccoda.util.result_classes.CAResult`` object. This
class extends the ``InferenceData`` class of `arviz <https://arviz-devs.github.io/arviz/>`_ and supports all its
diagnostic and plotting functionality.

.. autosummary::
    :toctree: .

    sccoda.util.result_classes.CAResult


Model comparison
----------------

``sccoda.models.other_models`` contains implementations of several compositional methods frm microbiome analysis and
non-compositional tests that can be used for comparison.

.. autosummary::
    :toctree: .

    sccoda.model.other_models.SimpleModel
    sccoda.model.other_models.scdney_model
    sccoda.model.other_models.NonBayesianModel
    sccoda.model.other_models.HaberModel
    sccoda.model.other_models.CLRModel
    sccoda.model.other_models.TTest
    sccoda.model.other_models.CLRModel_ttest
    sccoda.model.other_models.ALDEx2Model
    sccoda.model.other_models.ALRModel_ttest
    sccoda.model.other_models.ALRModel_wilcoxon
    sccoda.model.other_models.AncomModel
    sccoda.model.other_models.DirichRegModel
