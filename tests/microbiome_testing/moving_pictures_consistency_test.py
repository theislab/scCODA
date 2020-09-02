"""
Testing scdcdm for consistency on the "Moving pictures" dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
import arviz as az
import pickle as pkl
from matplotlib.pyplot import cm
import matplotlib

from scdcdm.util import data_generation as gen
from scdcdm.util import comp_ana as mod
from scdcdm.util import result_classes as res
from scdcdm.util import multi_parameter_sampling as mult
from scdcdm.util import cell_composition_data as dat
from scdcdm.util import data_visualization as viz

pd.options.display.float_format = '{:10,.3f}'.format
pd.set_option('display.max_columns', None)

write_path = "C:/Users/Johannes/Documents/Uni/Master's_Thesis/compositionalDiff-johannes_tests_2/tests/microbiome_testing"

#%%

# read phylum-level data from biom file as tsv
data_path = "C:/Users/Johannes/AppData/Local/Packages/CanonicalGroupLimited.Ubuntu18.04onWindows_79rhkp1fndgsc/LocalState/rootfs/home/johannes/qiime2_projects/moving-pictures-tutorial"

with open(data_path+"/exported_data/feature-table-l2.tsv", "rb") as f:
    biom_data = pd.read_csv(f, sep="\t", header=1, index_col="#OTU ID")

biom_data = biom_data.transpose()

# remove rare groups (<10 in all samples)

# read metadata
with open(data_path+"/sample-metadata.tsv", "rb") as f:
    metadata = pd.read_csv(f, sep="\t", index_col="sample-id").iloc[1:, :]

metadata_columns = ["subject", "reported-antibiotic-usage", "days-since-experiment-start", "body-site"]

# add subject to count data
biom_data = pd.merge(biom_data, metadata[metadata_columns], left_index=True, right_index=True)

data = dat.from_pandas(biom_data, metadata_columns)
data.obs = data.obs.rename(columns={"reported-antibiotic-usage": "antibiotic", "body-site": "site",
                                    "days-since-experiment-start": "days_since_start"})

print(data.obs)

#%%
importlib.reload(viz)

sns.set(style="ticks", font_scale=2)
args_swarmplot={"hue": "subject", "size": 10, "palette": "Reds"}
viz.boxplot_facets(data, feature="site")
plt.show()

#%%

# Model that differentiates both palms
model_palms = mod.CompositionalAnalysis(data[data.obs["site"].isin(["left palm", "right palm"])], "site", baseline_index=None)

result_palms = model_palms.sample_hmc(num_results=int(20000), n_burnin=5000)

result_palms.summary_extended(hdi_prob=0.95)

#%%

with az.rc_context(rc={'plot.max_subplots': None}):
    az.plot_trace(result_palms, compact=True)
    plt.show()

#%%

# less samples, less burnin
result_palms_5000 = model_palms.sample_hmc(num_results=int(5000), n_burnin=1000)

result_palms_5000.summary_extended(hdi_prob=0.95)

#%%

with az.rc_context(rc={'plot.max_subplots': None}):
    az.plot_trace(result_palms_5000, var_names="beta")
    plt.show()



#%%

with az.rc_context(rc={'plot.max_subplots': None}):
    az.plot_trace(result_palms, var_names=["beta", "b_raw"])
    plt.show()

#%%

az.rhat(result_palms, method="folded")


#%%
# run palms comparison multiple times

results = []
n_chains = 50

model_palms = mod.CompositionalAnalysis(data[data.obs["site"].isin(["left palm", "right palm"])], "site", baseline_index=None)

for n in range(n_chains):
    result_temp = model_palms.sample_hmc(num_results=int(20000), n_burnin=5000)

    results.append(result_temp)

#%%
res_all = az.concat(results, dim="chain")

print(res_all.posterior)

#%%
az.to_netcdf(res_all, write_path + "/multi_chain_50_len20000_all")

#%%

acc_probs = pd.DataFrame(pd.concat([r.effect_df.loc[:, "Inclusion probability"] for r in results]))

acc_probs["chain_no"] = np.concatenate([np.repeat(i+1, 21) for i in range(n_chains)])

acc_probs.index = acc_probs.index.droplevel(0)

acc_probs = acc_probs.reset_index()

print(acc_probs)

#%%
with open(write_path + "/multi_chain_50_len20000_acc", "wb") as file:
    pkl.dump(acc_probs, file)


#%%
with open(write_path + "/multi_chain_50_len20000_acc", "rb") as file:
    acc_probs = pkl.load(file)

res_all = az.from_netcdf(write_path + "/multi_chain_50_len20000_all")

#%%
coords = {"cell_type": "k__Bacteria;p__Proteobacteria"}
az.plot_trace(res_all, var_names="beta", coords=coords)
plt.show()


#%%
sns.set(style="ticks", font_scale=1)

n_chains = 50
col = [cm.tab20(i % 20) for i in range(n_chains)]

g = sns.FacetGrid(data=acc_probs.loc[acc_probs["Cell Type"].isin(["k__Bacteria;p__Fusobacteria", "k__Bacteria;p__Firmicutes", "k__Bacteria;p__Tenericutes"])], col="Cell Type", col_wrap=3)
g.map(sns.kdeplot, "Inclusion probability")
rug = g.map(sns.rugplot, "Inclusion probability", height=0.3, color="black")
for ax in g.axes:
    ax.axvline(0.81, color="red", linewidth=0.5)

# There is no labels, need to define the labels
legend_labels = [i+1 for i in range(n_chains)]

# Create the legend patches
legend_patches = [matplotlib.patches.Patch(color=C, label=L) for
                  C, L in zip(col, legend_labels)]

# Plot the legend
# plt.legend(handles=legend_patches, loc="lower right")

plt.show()


#%%
# Only inconsistencies for Firmicutes

acc_probs["is_significant"] = np.where(acc_probs["Inclusion probability"] > 0.81, True, False)

sig_table = acc_probs.groupby(["Cell Type"]).agg({"Inclusion probability": "mean", "is_significant": "sum"})

print(sig_table)

#%%

types = ["k__Bacteria;p__Proteobacteria", "k__Bacteria;p__FBP"]

ax = az.plot_pair(
    res_all,
    kind=["scatter", "kde"],
    kde_kwargs={"fill_last": False},
    marginals=True,
    coords={"chain": [33], "cell_type": types, "cell_type_nb": types},
    point_estimate="median",
)

plt.show()

#%%

az.rhat(res_all)

#%%

az.summary(res_all)

#%%


