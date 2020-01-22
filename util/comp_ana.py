import numpy as np
import patsy as pt
import pandas as pd

from model import dirichlet_models as dm
from util import result_classes as res

#%%


class CompositionalAnalysis:

    def __init__(self, data, formula, baseline_index=0):
        """
        Builds count and covariate matrix, returns a CompositionalModel object
        :param data: anndata object with cell counts as data.X and covariats saved in data.obs
        :param formula: string - R-style formula for building the covariate matrix
        :param baseline_index: int - baseline index
        :return: A CompositionalModel object
        """

        self.data = data
        self.cell_types = data.var.index.to_list()

        # Get count data
        data_matrix = data.X

        # Build covariate matrix from R-like formula
        covariate_matrix = pt.dmatrix(formula+"-1", data.obs)
        self.covariate_names = covariate_matrix.design_info.column_names

        # Invoke instance of the correct model depending on baseline index
        if baseline_index is None:
            self.model = dm.NoBaselineModel(np.array(covariate_matrix), data_matrix)
            self.baseline = False
        else:
            self.model = dm.BaselineModel(np.array(covariate_matrix), data_matrix, baseline_index)
            self.baseline = True

    def sample(self, method="HMC", *args, **kwargs):

        if method == "HMC":
            params, y_hat = self.model.sample_hmc(*args, **kwargs)
            return res.CompAnaResult(params=params, y_hat=y_hat, y=self.data.X, baseline=self.baseline,
                                     cell_types=self.cell_types, covariate_names=self.covariate_names)

        elif method == "NUTS":
            params, y_hat = self.model.sample_nuts(*args, **kwargs)
            return res.CompAnaResult(params=params, y_hat=y_hat, y=self.data.X, baseline=self.baseline,
                                     cell_types=self.cell_types, covariate_names=self.covariate_names)

        else:
            print("Not a valid sampling method. Use HMC or NUTS!")
            return
