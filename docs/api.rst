.. automodule:: sccoda

API
===

We advise to import scCODA in a python session via::

    import sccoda
    dat = sccoda.util.cell_composition_data
    ana = sccoda.util.compositional_analysis
    viz = sccoda.util.data_visualization

The workflow in scCODA starts with reading in cell count data (``dat``) and visualizing them (``viz``)
or synthetically generating cell count data (``sccoda.util.data_generation``).
Then, compositional modeling is performed via ``ana.CompositionalAnalysis``.

Data acquisition
----------------

**Integrating data sources (dat)** (scanpy or pandas)

.. autosummary::
    :toctree: .

    sccoda.util.cell_composition_data.from_pandas
    sccoda.util.cell_composition_data.read_anndata_one_sample
    sccoda.util.cell_composition_data.from_scanpy_dir
    sccoda.util.cell_composition_data.from_scanpy_list
