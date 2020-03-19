import numpy as np
import arviz as az
import seaborn as sns
import pandas as pd
import pickle as pkl
import importlib
import anndata as ad
import ast

import matplotlib.pyplot as plt
from SCDCpy.util import result_classes as res
from SCDCpy.util import multi_parameter_sampling as mult
from SCDCpy.util import multi_parameter_analysis_functions as ana
from SCDCpy.util import compositional_analysis_generation_toolbox as gen

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
sns.set_style("ticks")

#%%

result_path = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\compositionalDiff-johannes_tests_2\\data\\benchmark_results"

with open(result_path + "\\results_aggregated.pkl", "rb") as f:
    all_study_params_agg_2 = pkl.load(file=f)

#%%
g = sns.PairGrid(data=all_study_params_agg_2[["mcc", "n_controls", "n_cases", "b_count", "num_increase", "log_fold_increase"]])
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, n_levels=6)
plt.show()


#%%
all_study_params_agg_2["log_increase"] = np.log(all_study_params_agg_2["num_increase"])
all_study_params_agg_2["total_samples"] = all_study_params_agg_2["n_controls"] + all_study_params_agg_2["n_cases"]


#%%
sns.pairplot(data=all_study_params_agg_2[["mcc", "n_controls", "n_cases","b_count", "num_increase",
                                          "log_fold_increase", "log_increase", "total_samples"]],
             plot_kws={"alpha": 0.2})
plt.show()

#%%

import statsmodels.api as sm
import statsmodels.formula.api as smf

est = smf.ols(formula='mcc ~ n_controls * n_cases + log_increase*b_count* log_fold_increase', data=all_study_params_agg_2).fit()
print(est.summary())

#%%

model_increase = smf.ols(formula='mcc ~ log_increase', data=all_study_params_agg_2).fit()
print(model_increase.summary())

#%%

model_samples = smf.ols(formula='mcc ~ total_samples + log_fold_increase * log_increase', data=all_study_params_agg_2).fit()
print(model_samples.summary())

#%%
norm_df = all_study_params_agg_2[["mcc", "n_controls", "n_cases","b_count", "num_increase",
                                          "log_fold_increase", "log_increase", "total_samples"]]
norm_df = (norm_df-norm_df.mean())/norm_df.std()

sns.pairplot(data=norm_df,plot_kws={"alpha": 0.2})
plt.show()

#%%

norm_model = smf.ols(formula='mcc ~ log_increase+b_count+log_fold_increase-1', data=norm_df).fit()
print(norm_model.summary())

#%%
stan_df = all_study_params_agg_2[["n_controls", "n_cases","b_count", "num_increase",
                                          "log_fold_increase", "log_increase", "total_samples"]]
stan_df = (stan_df-stan_df.mean())
stan_df["mcc"] = all_study_params_agg_2["mcc"]

sns.pairplot(data=stan_df,plot_kws={"alpha": 0.2})
plt.show()

#%%

stan_model = smf.ols(formula='mcc ~ total_samples + log_fold_increase * log_increase', data=stan_df).fit()
print(stan_model.summary())