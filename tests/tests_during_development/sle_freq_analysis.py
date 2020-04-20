import numpy as np
import arviz as az
import seaborn as sns
import pandas as pd
import pickle as pkl
import importlib
import anndata as ad

import matplotlib.pyplot as plt
from scdcdm.util import result_classes as res
from scdcdm.util import comp_ana as mod
from scdcdm.util import data_generation as gen

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
sns.set_style("ticks")
#%%
# Import data

celltypes_path = "C:/Users/Johannes/Documents/Uni/Master's_Thesis/data/20200112_SLE_FreqTable.csv"

celltypes = pd.read_csv(celltypes_path, sep=";", header=[0], index_col=[0])

#%%

# Groups

group_df = pd.DataFrame(celltypes["Group"])
group_df["int_Group"] = [0, 0, 0, 0, 0, 1, 1, 1, 1]
print(group_df)

#%%
# Get cell counts (rounded???)

cell_freq = celltypes.iloc[:, 3:]
cell_counts_df = cell_freq.multiply(celltypes["Unfiltered|count_CD45"], axis=0).round(0)
cell_counts = cell_counts_df.values

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


#%%
# Use only celltypes with more than 3% in at least one sample (17 types)

cells_rep = cell_counts_df.loc[:, ~(cell_freq < 0.03).all()]
cells_rep_rel = cell_freq.loc[:, ~(cell_freq < 0.03).all()]
cell_types_rep = pd.DataFrame(index=cells_rep.columns)

sle_freq_data = ad.AnnData(X=cells_rep.values, var=cell_types_rep, obs=group_df)
print(sle_freq_data.X.shape)

#%%
cells_plot = pd.melt(cells_rep_rel.reset_index(), id_vars="file", var_name="Cell_Type", value_name="Count")

cells_plot_2 = pd.melt(cells_rep_rel.drop(columns="Clust_36|percent_total").reset_index(), id_vars="file", var_name="Cell_Type", value_name="Count")

#%%
fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(data=cells_plot_2, x="Cell_Type", y="Count", hue="file", ax=ax)
plt.xticks(rotation=90)
plt.show()

#%%
# Modeling without baseline

print(cells_rep.iloc[:, 14:])
ana = mod.CompositionalAnalysis(sle_freq_data, "int_Group", baseline_index=3)

#%%
ca_result = ana.sample_hmc(num_results=int(20000), n_burnin=5000)

ca_result.summary(credible_interval=0.95)

#%%
az.plot_trace(ca_result)
plt.show()

#%%
freq_plot = pd.melt(cell_freq.reset_index(), id_vars="file", var_name="Cell_Type", value_name="Count")
freq_plot = pd.merge(freq_plot, group_df.reset_index())
freq_plot["Cell_Type"] = [x[0] for x in freq_plot["Cell_Type"].str.split("|")]
freq_plot = freq_plot.rename(columns={"Count": "Freq"})

print(freq_plot)
plt.figure(figsize=(20, 6))
sns.boxplot(data=freq_plot, x="Cell_Type", y="Freq", hue="Group")
plt.xticks(rotation=90)
plt.savefig("C:/Users/Johannes/Documents/Uni/Master's_Thesis/stuff/cyTOF_shares_all.png")
plt.show()

#%%
freq_plot_rep = pd.melt(cells_rep_rel.drop(columns="Clust_36|percent_total").reset_index(), id_vars="file", var_name="Cell_Type", value_name="Count")
freq_plot_rep = pd.merge(freq_plot_rep, group_df.reset_index())
freq_plot_rep["Cell_Type"] = [x[0] for x in freq_plot_rep["Cell_Type"].str.split("|")]
freq_plot_rep = freq_plot_rep.rename(columns={"Count": "Freq"})
print(freq_plot_rep)
plt.figure(figsize=(20, 6))

sns.boxplot(data=freq_plot_rep, x="Cell_Type", y="Freq", hue="Group")
plt.xticks(rotation=90)
plt.savefig("C:/Users/Johannes/Documents/Uni/Master's_Thesis/stuff/cyTOF_shares_relevant_no36.png")
plt.show()
