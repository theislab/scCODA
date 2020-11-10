"""
Initialization of scCODA models

:authors: Johannes Ostner
"""
import numpy as np
import patsy as pt
import importlib

from sccoda.model import dirichlet_models as dm
from sccoda.model import other_models as om
from sccoda.model import dirichlet_time_models as tm


class CompositionalAnalysis:
    """
    Initializer class for compositional models. Please refer to the tutorial for using this class.
    """

    def __new__(cls, data, formula, baseline_index=None, time_column=None):
        """
        Builds count and covariate matrix, returns a CompositionalModel object

        Usage: model = CompositionalAnalysis(data, formula="covariate1 + covariate2", baseline_index="CellTypeA")

        Parameters
        ----------
        data -- anndata object
            anndata object with cell counts as data.X and covariates saved in data.obs
        formula -- string
            R-style formula for building the covariate matrix.
            Categorical covariates are handled automatically, with the covariate value of the first sample being used as the reference category.
            To set a different level as the reference category, use "C(<CovariateName>, Treatment('<ReferenceLevelName>'))"
        baseline_index -- string or int
            Column index that sets the baseline cell type. Can either reference the name of a column or the n-th column (starting at 0)

        Returns
        -------
        model
            A scCODA.models.dirichlet_models.CompositionalModel object
        """

        importlib.reload(dm)

        cell_types = data.var.index.to_list()

        # Get count data
        data_matrix = data.X.astype("float32")

        # Build covariate matrix from R-like formula
        covariate_matrix = pt.dmatrix(formula, data.obs)
        covariate_names = covariate_matrix.design_info.column_names[1:]
        covariate_matrix = covariate_matrix[:, 1:]

        # Invoke instance of the correct model depending on baseline index
        # If baseline index is "simple": Invokes a model for model comparison
        if baseline_index == "simple":
            return om.SimpleModel(covariate_matrix=np.array(covariate_matrix), data_matrix=data_matrix,
                                  cell_types=cell_types, covariate_names=covariate_names, formula=formula)
        # No baseline index
        elif baseline_index is None:
            if time_column is None:
                return dm.NoBaselineModel(covariate_matrix=np.array(covariate_matrix), data_matrix=data_matrix,
                                          cell_types=cell_types, covariate_names=covariate_names, formula=formula)
            else:
                return tm.NoBaselineModelTime(covariate_matrix=np.array(covariate_matrix), data_matrix=data_matrix,
                                              cell_types=cell_types, covariate_names=covariate_names, formula=formula,
                                              time_matrix=data.obs[time_column].to_numpy())
        # Column name as baseline index
        elif baseline_index in cell_types:
            num_index = cell_types.index(baseline_index)
            return dm.BaselineModel(covariate_matrix=np.array(covariate_matrix), data_matrix=data_matrix,
                                    cell_types=cell_types, covariate_names=covariate_names,
                                    baseline_index=num_index, formula=formula)
        # Numeric baseline index
        elif isinstance(baseline_index, int) & (baseline_index < len(cell_types)):
            return dm.BaselineModel(covariate_matrix=np.array(covariate_matrix), data_matrix=data_matrix,
                                    cell_types=cell_types, covariate_names=covariate_names,
                                    baseline_index=baseline_index, formula=formula)
        # None of the above: Throw error
        else:
            raise NameError("Baseline index is not a valid cell type!")
