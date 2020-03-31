"""
Power analysis for the Dirichlet-multinomial model
"""

import numpy as np
import arviz as az
import seaborn as sns
import pandas as pd
import pickle as pkl
import importlib
import anndata as ad
import ast
from itertools import product

import statsmodels.api as sm
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
from scdcdm.util import result_classes as res
from scdcdm.util import multi_parameter_sampling as mult
from scdcdm.util import multi_parameter_analysis_functions as ana
from scdcdm.util import compositional_analysis_generation_toolbox as gen

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
sns.set_style("ticks")

#%%
# Load data
result_path = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\compositionalDiff-johannes_tests_2\\data\\benchmark_results"

with open(result_path + "\\results_aggregated.pkl", "rb") as f:
    all_study_params_agg_2 = pkl.load(file=f)


#%%
all_study_params_agg_2["log_increase"] = np.log(all_study_params_agg_2["num_increase"])
all_study_params_agg_2["total_samples"] = all_study_params_agg_2["n_controls"] + all_study_params_agg_2["n_cases"]

#%%
# Final model

model_samples = smf.ols(formula='mcc ~ total_samples + log_fold_increase * log_increase', data=all_study_params_agg_2).fit()
print(model_samples.summary())


#%%
# Data points for power analysis plot
num_increases = np.arange(10, 1010, 10)

base_counts = np.arange(10, 1010, 10)


params = pd.DataFrame(list(product(num_increases, base_counts)), columns=['num_increase', 'base_count'])
params.loc[:, "log_fold_increase"] = np.log2((params.loc[:, "num_increase"] + params.loc[:, "base_count"]) / params.loc[:, "base_count"])
params.loc[:, "log_increase"] = np.log(params["num_increase"])

#%%
# How many samples do we need for a certain MCC?
def counts_for_mcc(mcc_desired, data):
    increase_effects = 0.4940 * data["log_fold_increase"] + 0.2453 * data["log_increase"] - 0.0647 * data["log_fold_increase"] * data["log_increase"]

    return (mcc_desired + 1.0757 - increase_effects) / 0.0157

params.loc[:, "mcc_08"] = counts_for_mcc(0.8, params)

print(params)


#%%
# Prepare data for heatmap
heatmap_data = params[["num_increase", "base_count", "mcc_08"]].\
    rename(columns={"num_increase": "Increase", "base_count": "Base", "mcc_08": "Total samples needed"}).\
    pivot_table("Total samples needed", "Increase", "Base")

# note: the df.index is a series of elevation values
tick_step = 100
tick_min = 100
tick_max = 1100
ticklabels = range(tick_min, tick_max, tick_step)

#%%
result_path = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\compositionalDiff-johannes_tests_2\\data\\benchmark_results"

sns.set_context("paper", font_scale=1.4)

yticks = []
xticks = []

for label in ticklabels:
    idx_pos = heatmap_data.index.get_loc(label)
    yticks.append(idx_pos)

    idx_pos_2 = heatmap_data.columns.get_loc(label)
    xticks.append(idx_pos_2)

ax1 = sns.heatmap(data=heatmap_data, robust=True, cbar_kws={"label": "Total samples needed"}, annot=False,
                  yticklabels=ticklabels, xticklabels=ticklabels)
ax1.set_yticks(yticks)
ax1.set_xticks(xticks)

# plt.savefig(result_path + "\\mcc_08_concept_fig.svg", format="svg", bbox_inches="tight")
# plt.savefig(result_path + "\\mcc_08_concept_fig.png", format="png", bbox_inches="tight")

plt.show()


#%%

num_increases = np.arange(10, 1010, 10)
base_counts = np.arange(10, 1010, 10)


def one_des_mcc(mcc_desired, base, inc):
    log_inc = np.log(inc)
    lf_inc = np.log2((inc + base) / base)
    increase_effects = 0.4940 * lf_inc + 0.2453 * log_inc - 0.0647 * lf_inc * log_inc

    return (mcc_desired + 1.0757 - increase_effects) / 0.0157


X, Y = np.meshgrid(num_increases, base_counts)
Z = one_des_mcc(0.8, X, Y)

#%%
contours = plt.contour(base_counts, num_increases, Z, levels=[5, 10, 20, 30, 40, 50, 60], colors='black', origin="upper")
plt.clabel(contours, inline=True, fontsize=8, fmt="%1.0f")

plt.imshow(Z, extent=[0, 1000, 0, 1000], origin='lower',
           cmap=sns.cm.rocket_r, alpha=0.9)
plt.colorbar(label="Total samples needed")
plt.xlabel("Base")
plt.ylabel("Increase")

plt.savefig(result_path + "\\mcc_08_concept_fig.svg", format="svg", bbox_inches="tight")
plt.savefig(result_path + "\\mcc_08_concept_fig.png", format="png", bbox_inches="tight")

plt.show()

