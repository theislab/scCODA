import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
import arviz as az
import pickle as pkl
from matplotlib.pyplot import cm

from sccoda.util import data_generation as gen
from sccoda.util import comp_ana as mod
from sccoda.util import result_classes as res
from sccoda.util import multi_parameter_sampling as mult
from sccoda.util import cell_composition_data as dat

pd.options.display.float_format = '{:10,.3f}'.format
pd.set_option('display.max_columns', None)

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


def plot_one_stackbar(y, type_names, title, level_names):

    plt.figure(figsize=(20, 10))
    n_samples, n_types = y.shape
    r = np.array(range(n_samples))
    sample_sums = np.sum(y, axis=1)
    barwidth = 0.85
    cum_bars = np.zeros(n_samples)
    colors = cm.tab20

    for n in range(n_types):
        bars = [i / j * 100 for i, j in zip([y[k][n] for k in range(n_samples)], sample_sums)]
        plt.bar(r, bars, bottom=cum_bars, color=colors(n % 20), width=barwidth, label=type_names[n])
        cum_bars += bars

    plt.title(title)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
    plt.xticks(r, level_names, rotation=45)

    plt.show()


plot_one_stackbar(data.X, data.var.index, "samples", data.obs.index)

#%%


def plot_feature_stackbars(data, features):

    type_names = data.var.index
    for f in features:
        levels = pd.unique(data.obs[f])
        n_levels = len(levels)
        feature_totals = np.zeros([n_levels, data.X.shape[1]])

        for level in range(n_levels):
            l_indices = np.where(data.obs[f] == levels[level])
            feature_totals[level] = np.sum(data.X[l_indices], axis=0)

        plot_one_stackbar(feature_totals, type_names=type_names, level_names=levels, title=f)


plot_feature_stackbars(data, ["site", "antibiotic", "subject"])


#%%
# remove rare groups (<10 in all samples)
biom_data_nonrare = df = biom_data.drop([x for x in biom_data.columns[:-4] if all(biom_data[x] < 10)], 1)

data_nonrare = dat.from_pandas(biom_data_nonrare, metadata_columns)
data_nonrare.obs = data_nonrare.obs.rename(columns={"reported-antibiotic-usage": "antibiotic", "body-site": "site",
                                                    "days-since-experiment-start": "days_since_start"})
print(data_nonrare.X)
print(data_nonrare.obs)


#%%
# No significances
# Model with subject as covariate
model_subject = mod.CompositionalAnalysis(data, "subject", baseline_index=None)

result_subject = model_subject.sample_hmc(num_results=int(20000), n_burnin=5000)

result_subject.summary_extended(hdi_prob=0.95)

#%%
az.plot_trace(result_subject, var_names=["beta"])
plt.show()

#%%
# Model with antibiotic use as covariate
model_anti = mod.CompositionalAnalysis(data, "antibiotic", baseline_index=None)

result_anti = model_anti.sample_hmc(num_results=int(20000), n_burnin=5000)

result_anti.summary_extended(hdi_prob=0.95)
# Significances: _ and Actinobacteria; both negative

#%%
# Model with subject as covariate, less groups
model_subject_nonrare = mod.CompositionalAnalysis(data_nonrare, "subject", baseline_index=None)

result_subject_nonrare = model_subject_nonrare.sample_hmc(num_results=int(20000), n_burnin=5000)

result_subject_nonrare.summary_extended(hdi_prob=0.95)
# No significances

#%%
# Model with antibiotic use as covariate, less groups
model_anti_nonrare = mod.CompositionalAnalysis(data_nonrare, "antibiotic", baseline_index=None)

result_anti_nonrare = model_anti_nonrare.sample_hmc(num_results=int(20000), n_burnin=5000)

result_anti_nonrare.summary_extended(hdi_prob=0.95)
# Significances: _ and Actinobacteria; both negative

#%%
# Model with site as covariate, less groups
model_site_nonrare = mod.CompositionalAnalysis(data_nonrare, "site", baseline_index=None)

result_site_nonrare = model_site_nonrare.sample_hmc(num_results=int(20000), n_burnin=5000)

result_site_nonrare.summary_extended(hdi_prob=0.95)

# Many significances, all positive

#%%
az.plot_trace(result_site_nonrare, var_names=["beta"])
plt.show()
# Looks really nice for once

#%%

# Now with level 6 data

#%%

# read phylum-level data from biom file as tsv
data_path = "C:/Users/Johannes/AppData/Local/Packages/CanonicalGroupLimited.Ubuntu18.04onWindows_79rhkp1fndgsc/LocalState/rootfs/home/johannes/qiime2_projects/moving-pictures-tutorial"

with open(data_path+"/exported_data/gut-table.tsv", "rb") as f:
    biom_data_l6 = pd.read_csv(f, sep="\t", header=1, index_col="#OTU ID")

biom_data_l6 = biom_data_l6.transpose()

# remove rare groups (<10 in all samples)

# read metadata
with open(data_path+"/sample-metadata.tsv", "rb") as f:
    metadata = pd.read_csv(f, sep="\t", index_col="sample-id").iloc[1:, :]

metadata_columns = ["subject", "reported-antibiotic-usage", "days-since-experiment-start", "body-site"]

# add subject to count data
biom_data_l6 = pd.merge(biom_data_l6, metadata[metadata_columns], left_index=True, right_index=True)

data_l6 = dat.from_pandas(biom_data_l6, metadata_columns)
data_l6.obs = data_l6.obs.rename(columns={"reported-antibiotic-usage": "antibiotic", "body-site": "site",
                                    "days-since-experiment-start": "days_since_start"})

print(data_l6.obs)

#%%

# Model with subject as covariate
model_subject_l6 = mod.CompositionalAnalysis(data_l6, "subject", baseline_index=None)

result_subject_l6 = model_subject_l6.sample_hmc(num_results=int(20000), n_burnin=5000)

result_subject_l6.summary_extended(hdi_prob=0.95)
# Looks pretty decent - some species were always included, none were never included
# ~25% significances, in both directions

#%%

pd.set_option('display.max_rows', None)
result_subject_l6.summary_extended(hdi_prob=0.95)

#%%
# Model with site as covariate, less groups
model_site_nonrare = mod.CompositionalAnalysis(data_nonrare, "site", baseline_index="k__Bacteria;p__Bacteroidetes")

result_site_nonrare = model_site_nonrare.sample_hmc(num_results=int(20000), n_burnin=5000)

result_site_nonrare.summary_extended(hdi_prob=0.95)

# Many significances, all positive
