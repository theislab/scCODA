import numpy as np
import arviz as az
import seaborn as sns
import pandas as pd
import pickle as pkl
import importlib
import anndata as ad
import ast

import matplotlib.pyplot as plt
from scdcdm.util import result_classes as res
from scdcdm.util import multi_parameter_sampling as mult
from scdcdm.util import multi_parameter_analysis_functions as ana
from scdcdm.util import compositional_analysis_generation_toolbox as gen

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.4)

result_path = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\compositionalDiff-johannes_tests_2\\data\\benchmark_results"

#%%
importlib.reload(ana)
# Get data

path = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\compositionalDiff-johannes_tests_2\\data\\model_comparison"


results, all_study_params, all_study_params_agg = ana.multi_run_study_analysis_prepare(path)

all_study_params_agg = ana.get_scores(all_study_params_agg, models=4)
all_study_params = ana.get_scores(all_study_params, models=4)

#%%
all_study_params.loc[all_study_params["n_samples"] == "[1, 1]", "mcc_0"] = 0
all_study_params_agg.loc[all_study_params_agg["n_samples"] == "[1, 1]", "mcc_0"] = 0


#%%

print(all_study_params)

#%%
b = []
for y1_0 in [115, 280, 1000]:
    b_i = np.round(gen.counts_from_first(y1_0, 5000, 5), 3)
    b.append(np.round(np.log(b_i / 5000), 2))
print(b)

b_counts = dict(zip([b_i[0] for b_i in b], [115, 280, 1000]))

#%%
b2 = []
for y1_0 in [115, 280, 1000]:
    b_i = np.round(gen.counts_from_first(y1_0, 5000, 5), 3)
    b2.append(b_i)

b_w_dict = {}
i = 0
w_all = []
for b_i in b2:
    b_t = np.round(np.log(b_i / 5000), 3)
    print(b_t)
    w_d = {}
    for change in [10, 50, 100]:
        _, w = gen.b_w_from_abs_change(b_i, change, 5000)
        w_0 = np.round(w[0], 3)
        w_d[w_0] = change
    b_w_dict[b_t[0]] = w_d
    i += 1
print(b_w_dict)

#%%

all_study_params_agg["n_controls"] = [ast.literal_eval(x)[0] for x in all_study_params_agg["n_samples"].tolist()]
all_study_params_agg["n_cases"] = [ast.literal_eval(x)[1] for x in all_study_params_agg["n_samples"].tolist()]
all_study_params_agg["n_total"] = all_study_params_agg["n_total"].astype("float")
all_study_params_agg["w"] = [ast.literal_eval(x)[0][0] for x in all_study_params_agg["w_true"]]
all_study_params_agg["b_0"] = [ast.literal_eval(x)[0] for x in all_study_params_agg["b_true"]]
all_study_params_agg["b_count"] = [b_counts[np.round(x, 2)] for x in all_study_params_agg["b_0"]]

bs = all_study_params_agg["b_0"].tolist()
ws = all_study_params_agg["w"].tolist()
increases = []
for i in range(len(bs)):
    increases.append(b_w_dict[bs[i]][ws[i]])
all_study_params_agg["num_increase"] = increases
all_study_params_agg["log_fold_increase"] = np.log2((all_study_params_agg["num_increase"] +
                                                       all_study_params_agg["b_count"]) /
                                                      all_study_params_agg["b_count"])

print(all_study_params_agg)

all_study_params["n_controls"] = [ast.literal_eval(x)[0] for x in all_study_params["n_samples"].tolist()]
all_study_params["n_cases"] = [ast.literal_eval(x)[1] for x in all_study_params["n_samples"].tolist()]
all_study_params["n_total"] = all_study_params["n_total"].astype("float")
all_study_params["w"] = [ast.literal_eval(x)[0][0] for x in all_study_params["w_true"]]
all_study_params["b_0"] = [ast.literal_eval(x)[0] for x in all_study_params["b_true"]]
all_study_params["b_count"] = [b_counts[np.round(x, 2)] for x in all_study_params["b_0"]]

bs = all_study_params["b_0"].tolist()
ws = all_study_params["w"].tolist()
increases = []
for i in range(len(bs)):
    increases.append(b_w_dict[bs[i]][ws[i]])
all_study_params["num_increase"] = increases
all_study_params["log_fold_increase"] = np.log2((all_study_params["num_increase"] +
                                                       all_study_params["b_count"]) /
                                                      all_study_params["b_count"])

print(all_study_params)

#%%

models = ["Poisson (Haber et al.)", "Simple DM", "scdcdm", "scDC (SydneyBioX)"]
param_cols = ["b_count", "num_increase", "n_controls", "n_cases", "log_fold_increase"]
metrics = ["tpr", "tnr", "precision", "accuracy", "youden", "f1_score", "mcc"]

final_df_agg = pd.DataFrame()
final_df = pd.DataFrame()
for m in range(len(models)):
    curr_model = models[m]
    m_str = str(m)

    cols = ["b_count", "num_increase", "n_controls", "n_cases", "log_fold_increase"]
    for x in metrics:
        cols.append(x + "_" + m_str)
    temp_df = all_study_params[cols]
    temp_df.loc[:, "model"] = curr_model

    temp_df_agg = all_study_params_agg[cols]
    temp_df_agg.loc[:, "model"] = curr_model

    rename_dict = dict(zip([x + "_" + m_str for x in metrics], metrics))
    temp_df = temp_df.rename(columns=rename_dict)
    temp_df_agg = temp_df_agg.rename(columns=rename_dict)

    final_df = final_df.append(temp_df)
    final_df_agg = final_df_agg.append(temp_df_agg)


#%%
plot_df_agg = final_df_agg.rename(columns={"b_count": "Base", "num_increase": "Increase", "model": "Model"})

fg = sns.FacetGrid(data=plot_df_agg, row="Base", col="Increase",
                   gridspec_kws={"right": 0.88, "left": 0.05, "top": 0.95, "bottom": 0.05, "wspace": 0.07, "hspace": 0.15})
fg.map_dataframe(sns.lineplot, x='n_controls', y='mcc', hue='Model', palette="colorblind")

fg.fig.set_size_inches(17, 12)
fg.axes[2, 1].set_xlabel('Replicates per group')
fg.axes[1, 0].set_ylabel('MCC')
fg.add_legend()
plt.savefig(result_path + "\\model_comparison.svg", format="svg", bbox_inces="tight")
plt.savefig(result_path + "\\model_comparison.png", format="png", bbox_inces="tight")

plt.show()

#%%
plot_df = final_df.rename(columns={"b_count": "Base", "num_increase": "Increase", "model": "Model"})

fg = sns.FacetGrid(data=plot_df, row="Base", col="Increase",
                   gridspec_kws={"right": 0.88, "left": 0.05, "top": 0.95, "bottom": 0.05, "wspace": 0.07, "hspace": 0.15})
fg.map_dataframe(sns.lineplot, x='n_controls', y='mcc', hue='Model', palette="colorblind")

fg.fig.set_size_inches(17, 12)
fg.axes[2, 1].set_xlabel('Replicates per group')
fg.axes[1, 0].set_ylabel('MCC')
fg.add_legend()
plt.savefig(result_path + "\\model_comparison_confint.svg", format="svg", bbox_inces="tight")
plt.savefig(result_path + "\\model_comparison_confint.png", format="png", bbox_inces="tight")

plt.show()

#%%
sns.palplot(sns.cm.rocket(15))
plt.show()

#%%

print(results)