import numpy as np
import arviz as az
import seaborn as sns
import pandas as pd
import pickle as pkl
import importlib
import anndata as ad

import matplotlib.pyplot as plt
from util import result_classes as res
from util import comp_ana as mod
from util import compositional_analysis_generation_toolbox as gen

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
#%%
# Import data

celltypes_path = "C:/Users/Johannes/Documents/Uni/Master's_Thesis/data/20200112_SLE_FreqTable.csv"

celltypes = pd.read_csv(celltypes_path, sep=";", header=[0], index_col=[0])

#%%

# Groups

group_df = pd.DataFrame(celltypes["Group"])
print(group_df)

#%%
# Get cell counts (rounded???)

cell_freq = celltypes.iloc[:, 3:]
cell_counts = cell_freq.multiply(celltypes["Unfiltered|count_CD45"], axis=0).round(0).values
print(cell_counts)

#%%
# Get names of celltypes

cluster_names = np.arange(60)+1
print(cluster_names)

cell_types = pd.DataFrame(index=cluster_names)
print(cell_types)

#%%
# Put all together

sle_freq_data = ad.AnnData(X=cell_counts, var=cell_types, obs=group_df)
print(sle_freq_data.obs)

#%%
# Modeling without baseline

ana = mod.CompositionalAnalysis(sle_freq_data, "Group", baseline_index=None)

#%%
ca_result = ana.sample_hmc(num_results=int(20000), n_burnin=5000)

ca_result.summary(credible_interval=0.95)

#%%
az.plot_trace(ca_result)
plt.show()

#%%

# Modeling with baseline
ana_2 = mod.CompositionalAnalysis(sle_freq_data, "Group", baseline_index=None)

#%%
ca_result_2 = ana_2.sample_hmc(num_results=int(2e4), n_burnin=5e3)

ca_result_2.summary(credible_interval=0.95)

