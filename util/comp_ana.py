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

        data_matrix = data.X

        # Build covariate matrix
        covariate_matrix = pt.dmatrix(formula, data.obs)
        covariate_matrix = np.array(covariate_matrix[:, 1:])

        # Invoke instance of the correct model depending on baseline index
        if baseline_index is None:
            return dm.NoBaselineModel(covariate_matrix, data_matrix)
        else:
            return dm.BaselineModel(covariate_matrix, data_matrix, baseline_index)


#%%
# Testing...
from util import compositional_analysis_generation_toolbox as gen
import anndata as ad


n = 3

cases = 1
K = 5
n_samples = [n, n]
n_total = np.full(shape=[2*n], fill_value=1000)



data = gen.generate_case_control(cases, K, n_total[0], n_samples,
                                       w_true=np.array([[1, 0, 0, 0, 0]]),
                                       b_true=np.log(np.repeat(0.2, K)).tolist())

print(data.X)
print(data.obs)

#%%

model = CompositionalAnalysis(data, "x_0", 4)

#%%
res = model.sample()
print(res)
#%%
res.plot()

#%%

# test out anndata
anndat = ad.AnnData(X=y, obs=cov)
print(anndat.obs)