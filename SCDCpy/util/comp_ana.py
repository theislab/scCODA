import numpy as np
import patsy as pt
import importlib

from SCDCpy.model import dirichlet_models as dm


#%%


class CompositionalAnalysis:

    def __new__(cls, data, formula, baseline_index=None):
        """
        Builds count and covariate matrix, returns a CompositionalModel object
        :param data: anndata object with cell counts as data.X and covariates saved in data.obs
        :param formula: string - R-style formula for building the covariate matrix
        :param baseline_index: int - baseline index
        :return: A CompositionalModel object
        """
        importlib.reload(dm)

        cell_types = data.var.index.to_list()

        # Get count data
        data_matrix = data.X

        # Build covariate matrix from R-like formula
        covariate_matrix = pt.dmatrix(formula, data.obs)
        covariate_names = covariate_matrix.design_info.column_names[1:]
        covariate_matrix = covariate_matrix[:, 1:]

        # Invoke instance of the correct model depending on baseline index
        if baseline_index is None:
            return dm.NoBaselineModel(covariate_matrix=np.array(covariate_matrix), data_matrix=data_matrix,
                                      cell_types=cell_types, covariate_names=covariate_names)
        else:
            return dm.BaselineModel(covariate_matrix=np.array(covariate_matrix), data_matrix=data_matrix,
                                    cell_types=cell_types, covariate_names=covariate_names,
                                    baseline_index=baseline_index)
