import arviz as az
import pandas as pd
import numpy as np
import importlib
from util import result_classes as res
from util import comp_ana as mod
from util import compositional_analysis_generation_toolbox as gen

arviz_data = az.load_arviz_data('centered_eight')


#%%
# Artificial data

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
importlib.reload(res)

ana = mod.CompositionalAnalysis(data, "x_0", baseline_index=None)

#%%
params_mcmc = ana.sample(method="HMC", num_results=int(1000), n_burnin=500)

params_mcmc.summary()

#%%

hmc_res = params_mcmc.raw_params
y_hat = params_mcmc.y_hat
baseline = False
cell_types = params_mcmc.cell_types
covariate_names = params_mcmc.covariate_names

#%%

pred_test = {"prediction": [hmc_res["prediction"]]}

arviz_test = az.from_dict(
    posterior={var_name: [var] for var_name, var in hmc_res.items() if
               "prediction" not in var_name},
    posterior_predictive=pred_test,
    observed_data={"y": data.X}
)

#%%

print(arviz_test.observed_data)
