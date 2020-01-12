import numpy as np
import patsy as pt
import pandas as pd

from model import dirichlet_models as dm
#%%


class CompositionalAnalysis:

    def __new__(cls, data, formula, baseline_index=0):
        """
        Builds count and covariate matrix, returns a CompositionalModel object
        :param data: anndata object with cell counts as data.X and covariats saved in data.obs
        :param formula: string - R-style formula for building the covariate matrix
        :param baseline_index: int - baseline index
        :return: A CompositionalModel object
        """

        # Get count data
        data_matrix = data.X

        # Build covariate matrix from R-like formula
        covariate_matrix = pt.dmatrix(formula, data.obs)
        covariate_matrix = np.array(covariate_matrix[:, 1:])

        # Invoke instance of the correct model depending on baseline index
        if baseline_index is None:
            return dm.NoBaselineModel(covariate_matrix, data_matrix)
        else:
            return dm.BaselineModel(covariate_matrix, data_matrix, baseline_index)
