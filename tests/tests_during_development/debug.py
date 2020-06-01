import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
import arviz as az

from scdcdm.util import data_generation as gen
from scdcdm.util import comp_ana as mod
from scdcdm.util import result_classes as res
from scdcdm.util import multi_parameter_sampling as mult
from scdcdm.util import cell_composition_data as dat

pd.options.display.float_format = '{:10,.3f}'.format
pd.set_option('display.max_columns', None)

# Artificial data
np.random.seed(1234)

n = 3

cases = 1
K = 5
n_samples = [n, n]
n_total = np.full(shape=[2*n], fill_value=1000)

data = gen.generate_case_control(cases, K, n_total[0], n_samples,
                                 w_true=np.array([[1, 0, 0, 0, 0]]),
                                 b_true=np.log(np.repeat(0.2, K)).tolist())

print(data.uns["w_true"])
print(data.uns["b_true"])

print(data.X)
print(data.obs)

#%%
importlib.reload(mod)

ana = mod.CompositionalAnalysis(data, "x_0", baseline_index=None)
print(ana.x)
print(ana.y)
print(ana.covariate_names)

params_mcmc = ana.sample_hmc(num_results=int(1000), n_burnin=500)
print(params_mcmc)

#%%

params_mcmc.summary(hdi_prob=0.9)