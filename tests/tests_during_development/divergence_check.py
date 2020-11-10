import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import importlib
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt

from sccoda.util import result_classes as res
from sccoda.model import dirichlet_models as mod
from sccoda.util import data_visualization as viz
from sccoda.util import comp_ana as ca
from sccoda.model import dirichlet_time_models as tm

tfd = tfp.distributions
tfb = tfp.bijectors

pd.set_option('display.max_columns', 500)
#%%
# Testing
from sccoda.util import data_generation as gen

n = 5

cases = 1
K = 5
n_samples = [n, n]
n_total = np.full(shape=[2*n], fill_value=1000)

data = gen.generate_case_control(cases, K, n_total[0], n_samples,
                                 w_true=np.array([[1, 0, 0, 0, 0]]),
                                 b_true=np.log(np.repeat(0.2, K)).tolist())

x = data.obs.values
y = data.X
print(x)
print(y)

#%%
importlib.reload(mod)
importlib.reload(res)
import patsy as pt

cell_types = data.var.index.to_list()

formula = "x_0"

# Get count data
data_matrix = data.X.astype("float32")

# Build covariate matrix from R-like formula
covariate_matrix = pt.dmatrix(formula, data.obs)
covariate_names = covariate_matrix.design_info.column_names[1:]
covariate_matrix = covariate_matrix[:, 1:]

dtype = tf.float32

N, K = data_matrix.shape
D = covariate_matrix.shape[1]

beta_size = [D, K]
alpha_size = [1, K]

#%%

viz.plot_feature_stackbars(data, ["x_0"])

#%%


importlib.reload(mod)
importlib.reload(res)

formula = "x_0"

model = mod.BaselineModel(covariate_matrix=np.array(covariate_matrix), data_matrix=data_matrix,
                                    cell_types=cell_types, covariate_names=covariate_names, formula=formula, baseline_index=4)

#%%
result = model.sample_nuts(num_results=int(5000), n_burnin=0, num_adapt_steps=1000)

result.summary_extended()
#%%

print(result.sample_stats["target_log_prob"])
#%%
az.plot_trace(result, combined=True, compact=True,
              divergences=None,
              #coords={"draw": range(5000, 10000)}
              )
plt.show()

#%%
plt.plot(result.sample_stats['step_size'][0])
plt.xlabel("sample")
plt.ylabel("step size")
plt.show()

#%%

div_sample_ids = np.where(result.sample_stats["log_acc_ratio"] == -np.inf)[1]
print(len(div_sample_ids))

# exactly 1 divergence that is not -inf after burnin
print(set(np.where(result.sample_stats["diverging"])[1].tolist()) - set(div_sample_ids.tolist()))

#%%
az.style.use("arviz-darkgrid")

ax = az.plot_parallel(result, norm_method="normal",
                      var_names=["mu_b", "sigma_b", "b_offset", "ind_raw", "alpha"],
                      #coords={"draw": range(5000, 10000)}
                      )
ax.set_xticklabels(ax.get_xticklabels(), rotation=70)
plt.show()

#%%

for x in [str(y) for y in range(1)]:

    ax = az.plot_pair(result,
                      divergences=False,
                      var_names=["mu_b", "sigma_b", "b_offset", "ind_raw", "alpha"],
                      coords={"cell_type": [x], "cell_type_nb": [x]},
                      )
    #ax.set_xticklabels(ax.get_xticklabels(), rotation=70)
    plt.show()

#%%

result.save("C:/Users/Johannes/Documents/PhD/scCODA/data/nuts_wide_230920")

#%%
import pickle as pkl

with open("C:/Users/Johannes/Documents/PhD/scCODA/data/nuts_standard_230920", "rb") as file:
    result_ = pkl.load(file)