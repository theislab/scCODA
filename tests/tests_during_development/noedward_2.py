from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
import importlib
import pandas as pd
from tensorflow_probability.python.experimental import edward2 as ed

from scdcdm.util import result_classes as res
from scdcdm.model import dirichlet_models as mod
from scdcdm.util import comp_ana as ca
from scdcdm.model import dirichlet_time_models as tm

tfd = tfp.distributions
tfb = tfp.bijectors

pd.set_option('display.max_columns', 500)
#%%
# Testing
from scdcdm.util import data_generation as gen

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


params = dict(mu_b = tf.zeros(1, name="init_mu_b", dtype=dtype),
                       sigma_b = tf.ones(1, name="init_sigma_b", dtype=dtype),
                       b_offset = tf.zeros(beta_size, name='init_b_offset', dtype=dtype),
                       ind_raw = tf.zeros(beta_size, name='init_ind_raw', dtype=dtype),
                       alpha = tf.zeros(alpha_size, name='init_alpha', dtype=dtype),
                       predictions = data_matrix
                           )

params["ind"] = 1 / (1 + tf.exp(-1*params["ind_raw"]))
params["b_raw"] = params["mu_b"] + params["sigma_b"] * params["b_offset"]
params["beta"] = params["b_raw"] * params["ind"]
params["concentrations"] = tf.exp(params["alpha"]
                           + tf.matmul(tf.cast(covariate_matrix, dtype), params["beta"]))

print(params)

#%%


importlib.reload(mod)
importlib.reload(res)
import patsy as pt

formula = "x_0"

model = mod.NoBaselineModelNoEdward(covariate_matrix=np.array(covariate_matrix), data_matrix=data_matrix,
                                    cell_types=cell_types, covariate_names=covariate_names, formula=formula)
#print(model.target_log_prob_fn(*(params.values())))

#%%
result = model.sample_hmc(num_results=int(1000), n_burnin=500)

result.summary()

#%%
model_2 = ca.CompositionalAnalysis(data, "x_0", baseline_index=None)
print(model_2.target_log_prob_fn(model_2.params[0], model_2.params[1], model_2.params[2], model_2.params[3], model_2.params[4]))

#%%
res_2 = model_2.sample_hmc(num_results=int(20000), n_burnin=5000)
res_2.summary()



#%%

time = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype="float32")

phi = np.random.normal(0, 1, size=(D, K))

phi_ = np.repeat(phi[np.newaxis,:], N, axis=0)

print(phi_.shape)

test = np.power(phi_, time[:, np.newaxis, np.newaxis])

print(test.shape)
print(test)

beta = np.random.normal(0, 1, size=(D, K))

b_ = beta[np.newaxis, :, :] * test

print(b_.shape)

c_ = covariate_matrix[:, :, np.newaxis] * b_

print(c_.shape)
print(np.sum(c_, axis=1))

#%%

importlib.reload(mod)
importlib.reload(tm)
importlib.reload(res)

model_t = tm.NoBaselineModelTime(covariate_matrix=np.array(covariate_matrix), data_matrix=data_matrix,
                                 cell_types=cell_types, covariate_names=covariate_names, formula=formula,
                                 time_matrix=time)

result_t = model_t.sample_hmc(num_results=int(1000), n_burnin=500)

result_t.summary()

#%%

data.obs["time"] = time
print(data.obs)

#%%

model_t_2 = ca.CompositionalAnalysis(data, "x_0", baseline_index=None, time_column="time")

result_t_2 = model_t_2.sample_hmc(num_results=int(1000), n_burnin=500)

#%%

result_t_2.summary()

#%%

print(result_t_2.posterior["phi"])



