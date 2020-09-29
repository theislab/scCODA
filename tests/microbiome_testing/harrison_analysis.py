import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import importlib


from scdcdm.util import data_generation as gen
from scdcdm.util import comp_ana as mod
from scdcdm.util import cell_composition_data as dat
from scdcdm.util import data_visualization as viz

pd.options.display.float_format = '{:10,.3f}'.format
pd.set_option('display.max_columns', 10)

#%%

data_path = "C:/Users/Johannes/Documents/PhD/data/rosen_mincount10_maxee2_trim200_results_forpaper/"

otu_file = "rosen_mincount10_maxee2_trim200.otu_table.99.denovo.rdp_assigned.paper_samples.txt"

with open(data_path + otu_file, "rb") as file:
    otus = pd.read_csv(file, sep="\t", index_col=0)

otus = otus[~otus.index.str.endswith('TI')]
otus = otus[~otus.index.str.endswith('F1')]
otus = otus[~otus.index.str.endswith('F1T')]
otus = otus[~otus.index.str.endswith('SI')]

split = otus.index.str.extract(r"^([0-9\-]*)([A-Z]+)$")

split.index = otus.index

otus.loc[:, "location"] = split.loc[:, 1]

otus.index = split.loc[:, 0]

print(pd.unique(otus.index))

print(pd.unique(otus["location"]))


#%%
meta_file = "patient_clinical_metadata.csv"

with open(data_path + meta_file, "rb") as file:
    metadata = pd.read_csv(file, sep=",", index_col=0)

print(metadata)

#%%

meta_rel = metadata[(~pd.isna(metadata["mbs_consolidated"])) & (~pd.isna(metadata["bal"]))]

print(meta_rel)

#%%

data = pd.merge(otus, meta_rel, right_index=True, left_index=True)

#%%

data_bal = data.loc[data["location"] == "B"]

print(np.sum(data_bal, axis=0))

#%%

col = metadata.columns[metadata.columns.isin(data_bal.columns)].tolist() + ["location"]

print(col)

# Remove otus with low counts (<100 total) --> Leaves 250 OTUs
# Leaving in low expression OTUs leads to nonconvergence for shorter chains (1000 samples),
# longer chains cant be done on my computer

counts_bal = data_bal.iloc[:, :-33]

counts_bal = counts_bal.loc[:, np.sum(counts_bal, axis=0) > 0]

#%%

seq_depths = []
for n in range(counts_bal.shape[1]):
    seq_depths.append(np.sum(counts_bal.iloc[:, n] > 0))

print(seq_depths)

plt.hist(seq_depths, bins=np.max(seq_depths))
plt.xlabel("sequencing depth")
plt.ylabel("count")
plt.show()

#%%
cum_seq = [np.sum([y > x for y in seq_depths]) for x in range(np.max(seq_depths))]
for i in range(len(cum_seq)):
    print(f"{i}: {cum_seq[i]}")
# --> Take seq. depth >= 10; 239 OTUs

counts_bal_depth10 = counts_bal.iloc[:, np.where(np.array(seq_depths) >= 10)[0]]
print(counts_bal_depth10.shape)

#%%

data_bal_expr = pd.merge(counts_bal_depth10, data_bal.loc[:, col], right_index=True, left_index=True)

print(data_bal_expr)

#%%

data_scdcdm = dat.from_pandas(data_bal_expr, col)

print(data_scdcdm.X.shape)

#%%

# Free up some memory

del([counts_bal, data, data_bal, metadata, meta_rel, file, otus, split, counts_bal_depth10])

#%%
importlib.reload(mod)

model_mbs = mod.CompositionalAnalysis(data_scdcdm, "mbs_consolidated", baseline_index=None)

result_mbs = model_mbs.sample_hmc(num_results=int(10000), n_burnin=0)

result_mbs.summary_extended(hdi_prob=0.95)


#%%

tax_interesting = result_mbs.effect_df[~result_mbs.effect_df["Inclusion probability"].isin([0, 1])]

print([x[1] for x in tax_interesting.index])

#%%

names = result_mbs.posterior.coords["cell_type"].values

#%%

coords = {"cell_type": names[:10], "cell_type_nb": names[:10]}
vn = ["alpha", "mu_b", "sigma_b", "b_offset", "ind_raw", "ind", "b_raw", "beta"]

az.plot_trace(result_mbs, coords=coords, var_names=vn, compact=True)

plt.show()

#%%

viz.plot_feature_stackbars(data_scdcdm, ["mbs_consolidated"])
plt.show()

#%%

# group means of data
gms = data_bal_expr.groupby("mbs_consolidated").mean().iloc[:, :250]

gms = gms.transpose()
gms["diff"] = gms["Aspiration/Penetration"] - gms["Normal"]
gms["logfold"] = np.log2((gms["Aspiration/Penetration"]+0.5)/(gms["Normal"]+0.5))

#%%
plt.hist(gms["logfold"], bins=25)
plt.show()

plt.hist(result_mbs.effect_df["Final Parameter"], bins=25)
plt.show()

#%%

plt.scatter(gms["logfold"], result_mbs.effect_df["Final Parameter"])
plt.xlabel("logfold change")
plt.ylabel("model parameter")
plt.show()

#%%

plt.scatter(gms["Normal"], np.exp(result_mbs.intercept_df["Final Parameter"]))
plt.xlabel("logfold change")
plt.ylabel("model parameter")
plt.show()

#%%

coords = {"cell_type": names[:10], "cell_type_nb": names[:10], "draw": [x for x in range(500)]}
vn = ["alpha", "mu_b", "sigma_b", "b_offset", "ind_raw", "ind", "b_raw", "beta"]

az.plot_trace(result_mbs, coords=coords, var_names=vn, compact=True)

plt.show()







#%%

# Try primal-dual averaging step size adaptation

importlib.reload(mod)

model_mbs = mod.CompositionalAnalysis(data_scdcdm, "mbs_consolidated", baseline_index=None)

#%%

result_mbs = model_mbs.sample_hmc_da(num_results=int(10000), n_burnin=0, num_adapt_steps=4000)

result_mbs.summary_extended(hdi_prob=0.95)

#%%

tax_interesting = result_mbs.effect_df[~result_mbs.effect_df["Inclusion probability"].isin([0, 1])]

print(len([x[1] for x in tax_interesting.index]))

names_int = tax_interesting.index.get_level_values("Cell Type")

#%%

print(result_mbs.effect_df.loc[result_mbs.effect_df.index.get_level_values("Cell Type").isin(tax_interesting)])

#%%

coords = {"cell_type": names_int, "cell_type_nb": names_int, "draw": [x for x in range(200)]}
vn = ["alpha", "mu_b", "sigma_b", "b_offset", "ind_raw", "ind", "b_raw", "beta"]

az.plot_trace(result_mbs, coords=coords, var_names=vn, compact=True)

plt.show()

#%%

coords = {"cell_type": names_int[[3,5]], "cell_type_nb": names_int[[3,5]], "draw": [x for x in range(200)]}
vn = ["alpha", "mu_b", "sigma_b", "b_offset", "ind_raw", "ind", "b_raw", "beta"]

az.plot_trace(result_mbs, coords=coords, var_names=vn, compact=True)

plt.show()

#%%

print(result_mbs.posterior["beta"].sel(cell_type=names_int[[3,5]]))






#%%

# trying out nuts sampling

importlib.reload(mod)

model_mbs = mod.CompositionalAnalysis(data_scdcdm, "mbs_consolidated", baseline_index=None)

result_mbs_nuts = model_mbs.sample_nuts(num_results=int(1000), n_burnin=0, num_adapt_steps=400)


#%%
result_mbs_nuts.summary_extended(hdi_prob=0.95)

#%%

tax_interesting = result_mbs_nuts.effect_df[~result_mbs_nuts.effect_df["Inclusion probability"].isin([0, 1])]

print(len([x[1] for x in tax_interesting.index]))

names_int = tax_interesting.index.get_level_values("Cell Type")

#%%

print(result_mbs_nuts.effect_df.loc[result_mbs_nuts.effect_df.index.get_level_values("Cell Type").isin(tax_interesting)])

#%%

coords = {"cell_type": names_int, "cell_type_nb": names_int, "draw": [x for x in range(1000)]}
vn = ["alpha", "mu_b", "sigma_b", "b_offset", "ind_raw", "ind", "b_raw", "beta"]

az.plot_trace(result_mbs_nuts, coords=coords, var_names=vn, compact=True)

plt.show()

#%%

coords = {"cell_type": names_int[[2, 3, 5]], "cell_type_nb": names_int[[2, 3, 5]], "draw": [x for x in range(200)]}
vn = ["alpha", "mu_b", "sigma_b", "b_offset", "ind_raw", "ind", "b_raw", "beta"]

az.plot_trace(result_mbs_nuts, coords=coords, var_names=vn, compact=True)

plt.show()

#%%

print(result_mbs_nuts.posterior["beta"].sel(cell_type=names_int[[2, 3]]))

#%%
result_mbs_nuts.save("C:/Users/Johannes/Documents/PhD/scdcdm/data/harrison_nuts_230920")

#%%

plt.plot(result_mbs_nuts.sample_stats['step_size'][0])
plt.xlabel("sample")
plt.ylabel("step size")
plt.show()

#%%

plt.plot(result_mbs_nuts.sample_stats['leapfrogs_taken'][0])
plt.xlabel("sample")
plt.ylabel("No. of leapfrog steps")
plt.show()



#%%
depths = [36, 35, 34, 33, 32, 31, 30, 29, 28]

results = []

for i in depths:
    counts_bal_ = counts_bal.iloc[:, np.where(np.array(seq_depths) > i)[0]]
    print(f"sequencing depth: {i+1}")
    print(f"{counts_bal_.shape[1]} OTUs")

    data_bal_expr = pd.merge(counts_bal_, data_bal.loc[:, col], right_index=True, left_index=True)
    data_scdcdm = dat.from_pandas(data_bal_expr, col)

    model_mbs = mod.CompositionalAnalysis(data_scdcdm, "mbs_consolidated", baseline_index=None)
    result_mbs_nuts = model_mbs.sample_nuts(num_results=int(50), n_burnin=0, num_adapt_steps=45)

    results.append(result_mbs_nuts)

#%%

fig, ax = plt.subplots(ncols=9, sharey=True, figsize=(20, 5))
for i in range(9):
    ax[i].plot(results[i].sample_stats['step_size'][0])
    ax[i].set_title(f"{cum_seq[depths[i]]} OTUs")

    ax[i].set_xlabel("sample")
ax[0].set_ylabel("step size")
plt.show()

