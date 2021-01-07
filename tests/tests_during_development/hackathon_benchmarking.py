from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
import importlib
import pandas as pd
from tensorflow_probability.python.experimental import edward2 as ed
import arviz as az
import matplotlib.pyplot as plt
import scanpy as sp

from scdcdm.util import result_classes as res
from scdcdm.model import dirichlet_models as mod
from scdcdm.util import comp_ana as ca
from scdcdm.model import dirichlet_time_models as tm
from scdcdm.util import data_visualization as viz

tfd = tfp.distributions
tfb = tfp.bijectors

pd.set_option('display.max_columns', 500)

#%%

# get Lisa's dataset

data_path = "C:/Users/Johannes/Documents/PhD/single-cell/hackathon_sep2020/thymus_data.h5ad"

data = sp.read(data_path)

#%%

# pseudo-covariate of 1 on all samples
data.obs["c"] = 1

print(data.X)

#%%

viz.plot_feature_stackbars(data, ["day"])

#%%
importlib.reload(ca)
importlib.reload(mod)
importlib.reload(tm)

model = ca.CompositionalAnalysis(data, formula="c", baseline_index=None, time_column="day")

result = model.sample_hmc(num_results=int(20000), n_burnin=0)

result.summary()

#%%

print(result.posterior["phi"][-1])

#%%

az.plot_trace(result, var_names=["beta", "phi"], compact=True)
plt.show()


