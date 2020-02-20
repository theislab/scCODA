import arviz as az
import pandas as pd
import numpy as np
import importlib
from util import result_classes as res
from util import comp_ana as mod
from util import compositional_analysis_generation_toolbox as gen
from util import multi_parameter_sampling as mult
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
#%%

cases = [1]
K = [5]
n_samples = [[i+6, i+6] for i in range(5)]
#n_samples = [[9, 9]]
n_total = [1000]
b_true = [np.array([0.2, 0.2, 0.2, 0.2, 0.2]).tolist()]
w_true = []
for x in [1]:
    w_true.append(np.array([[x, 0, 0, 0, 0]]).tolist())
num_results = [2e4]

#%%
importlib.reload(mult)

p = mult.MultiParamSimulation(cases, K, n_total, n_samples, b_true, w_true, num_results,
                              baseline_index=4, formula="x_0")

#%%

p.simulate()

#%%

p.get_discovery_rates()


#%%
print(p.mcmc_results)

print(p.parameters)

#%%
p.save(path="./data/", filename="mult_test")
