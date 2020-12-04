"""
Initialization of scCODA models

:authors: Johannes Ostner
"""
import numpy as np
import patsy as pt
import importlib

from sccoda.model import dirichlet_models as dm

class CompositionalAnalysis:
    """
    Initializer class for compositional models. Please refer to the tutorial for using this class.
    """

    def __new__(cls, data, formula, reference_cell_type=None):
        """
        Builds count and covariate matrix, returns a CompositionalModel object

        Usage: model = CompositionalAnalysis(data, formula="covariate1 + covariate2", reference_cell_type="CellTypeA")

        Parameters
        ----------
        data -- anndata object
            anndata object with cell counts as data.X and covariates saved in data.obs
        formula -- string
            R-style formula for building the covariate matrix.
            Categorical covariates are handled automatically, with the covariate value of the first sample being used as the reference category.
            To set a different level as the reference category, use "C(<CovariateName>, Treatment('<ReferenceLevelName>'))"
        reference_cell_type -- string or int
            Column index that sets the reference cell type. Can either reference the name of a column or the n-th column (starting at 0)

        Returns
        -------
        model
            A scCODA.models.dirichlet_models.CompositionalModel object
        """

        cell_types = data.var.index.to_list()

        # Get count data
        data_matrix = data.X.astype("float64")

        # Build covariate matrix from R-like formula
        covariate_matrix = pt.dmatrix(formula, data.obs)
        covariate_names = covariate_matrix.design_info.column_names[1:]
        covariate_matrix = covariate_matrix[:, 1:]

        # Invoke instance of the correct model depending on reference cell type
        # No reference cell type
        if reference_cell_type is None:
            return dm.NoReferenceModel(covariate_matrix=np.array(covariate_matrix), data_matrix=data_matrix,
                                      cell_types=cell_types, covariate_names=covariate_names, formula=formula)

        # Column name as reference cell type
        elif reference_cell_type in cell_types:
            num_index = cell_types.index(reference_cell_type)
            return dm.ReferenceModel(covariate_matrix=np.array(covariate_matrix), data_matrix=data_matrix,
                                    cell_types=cell_types, covariate_names=covariate_names,
                                    reference_cell_type=num_index, formula=formula)

        # Numeric reference cell type
        elif isinstance(reference_cell_type, int) & (reference_cell_type < len(cell_types)):
            return dm.ReferenceModel(covariate_matrix=np.array(covariate_matrix), data_matrix=data_matrix,
                                    cell_types=cell_types, covariate_names=covariate_names,
                                    reference_cell_type=reference_cell_type, formula=formula)

        # None of the above: Throw cell type
        else:
            raise NameError("Reference index is not a valid cell type!")
