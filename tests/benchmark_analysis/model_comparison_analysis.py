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
# Get data:
# all_study_params: One line per data point
# all_study_params_agg: Aggregate identical simulation parameters

path = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\compositionalDiff-johannes_tests_2\\data\\model_comparison"


results, all_study_params, all_study_params_agg = ana.multi_run_study_analysis_prepare(path)

all_study_params_agg = ana.get_scores(all_study_params_agg, models=4)
all_study_params = ana.get_scores(all_study_params, models=4)

#%%
# Set deprecated cases for poisson model (1 sample per group) to mcc=0
all_study_params.loc[all_study_params["n_samples"] == "[1, 1]", "mcc_0"] = 0
all_study_params_agg.loc[all_study_params_agg["n_samples"] == "[1, 1]", "mcc_0"] = 0


#%%
# Find relations between numerical values and compsition/inrease vectors
b = []
for y1_0 in [200, 400, 600, 800, 1000]:
    b_i = np.round(gen.counts_from_first(y1_0, 5000, 5), 3)
    b.append(np.round(np.log(b_i / 5000), 2))
print(b)

b_counts = dict(zip([b_i[0] for b_i in b], [200, 400, 600, 800, 1000]))

b2 = []
for y1_0 in [200, 400, 600, 800, 1000]:
    b_i = np.round(gen.counts_from_first(y1_0, 5000, 5), 3)
    b2.append(b_i)

b_w_dict = {}
w_rel_dict = {}
i = 0
w_all = []
for b_i in b2:
    b_t = np.round(np.log(b_i / 5000), 3)
    print(b_t)
    w_d = {}
    for change in [b_i[0]/3, b_i[0]/2, b_i[0], b_i[0]*2, b_i[0]*3]:
        _, w = gen.b_w_from_abs_change(b_i, change, 5000)
        w_0 = np.round(w[0], 3)
        w_d[w_0] = change
        rel_change = np.round(change/b_i[0], 2)
        w_rel_dict[w_0] = rel_change
    b_w_dict[b_t[0]] = w_d
    i += 1
print(b_w_dict)

#%%
# Get some new metrics that make plotting more convenient

all_study_params_agg["n_controls"] = [ast.literal_eval(x)[0] for x in all_study_params_agg["n_samples"].tolist()]
all_study_params_agg["n_cases"] = [ast.literal_eval(x)[1] for x in all_study_params_agg["n_samples"].tolist()]
all_study_params_agg["n_total"] = all_study_params_agg["n_total"].astype("float")
all_study_params_agg["w"] = [ast.literal_eval(x)[0][0] for x in all_study_params_agg["w_true"]]
all_study_params_agg["b_0"] = [ast.literal_eval(x)[0] for x in all_study_params_agg["b_true"]]
all_study_params_agg["b_count"] = [b_counts[np.round(x, 2)] for x in all_study_params_agg["b_0"]]

bs = all_study_params_agg["b_0"].tolist()
ws = all_study_params_agg["w"].tolist()
rels = [0.33, 0.5, 1, 2, 3]
increases = []
rel_changes = []
for i in range(len(bs)):
    increases.append(b_w_dict[bs[i]][ws[i]])
    rel_changes.append(w_rel_dict[ws[i]])
all_study_params_agg["num_increase"] = increases
all_study_params_agg["rel_increase"] = rel_changes
all_study_params_agg["log_fold_increase"] = np.round(np.log2((all_study_params_agg["num_increase"] +
                                                              all_study_params_agg["b_count"]) /
                                                              all_study_params_agg["b_count"]), 2)

print(all_study_params_agg)

all_study_params["n_controls"] = [ast.literal_eval(x)[0] for x in all_study_params["n_samples"].tolist()]
all_study_params["n_cases"] = [ast.literal_eval(x)[1] for x in all_study_params["n_samples"].tolist()]
all_study_params["n_total"] = all_study_params["n_total"].astype("float")
all_study_params["w"] = [ast.literal_eval(x)[0][0] for x in all_study_params["w_true"]]
all_study_params["b_0"] = [ast.literal_eval(x)[0] for x in all_study_params["b_true"]]
all_study_params["b_count"] = [b_counts[np.round(x, 2)] for x in all_study_params["b_0"]]

bs = all_study_params["b_0"].tolist()
ws = all_study_params["w"].tolist()
rels = [0.33, 0.5, 1, 2, 3]
increases = []
rel_changes = []
for i in range(len(bs)):
    increases.append(b_w_dict[bs[i]][ws[i]])
    rel_changes.append(w_rel_dict[ws[i]])
all_study_params["num_increase"] = increases
all_study_params["rel_increase"] = rel_changes
all_study_params["log_fold_increase"] = np.round(np.log2((all_study_params["num_increase"] +
                                                          all_study_params["b_count"]) /
                                                          all_study_params["b_count"]), 2)

print(all_study_params)

#%%

result_path = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\compositionalDiff-johannes_tests_2\\data\\benchmark_results"

with open(result_path + "\\model_comparison_results_aggregated.pkl", "wb") as f:
    pkl.dump(all_study_params_agg, file=f)
with open(result_path + "\\model_comparison_results.pkl", "wb") as f:
    pkl.dump(all_study_params, file=f)

#%%
# Convert data from wide to long

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
# Plot grid of base count/log-fold increase
plot_df_agg = final_df_agg.rename(columns={"b_count": "Base", "num_increase": "Increase", "model": "Model",
                                           "log_fold_increase": "log-fold increase"})

fg = sns.FacetGrid(data=plot_df_agg, row="Base", col="log-fold increase",
                   gridspec_kws={"right": 0.88, "left": 0.05, "top": 0.95, "bottom": 0.05, "wspace": 0.07, "hspace": 0.15})
fg.map_dataframe(sns.lineplot, x='n_controls', y='mcc', hue='Model', palette="colorblind")

fg.fig.set_size_inches(17, 12)
fg.axes[-1, 2].set_xlabel('Replicates per group')
fg.axes[2, 0].set_ylabel('MCC')
fg.add_legend()
plt.savefig(result_path + "\\model_comparison_grouped.svg", format="svg", bbox_inces="tight")
plt.savefig(result_path + "\\model_comparison_grouped.png", format="png", bbox_inces="tight")

plt.show()

#%%
# Plot grid of base count/log-fold increase, with confints

plot_df = final_df.rename(columns={"b_count": "Base", "num_increase": "Increase", "model": "Model",
                                   "log_fold_increase": "log-fold increase"})

fg = sns.FacetGrid(data=plot_df, row="Base", col="log-fold increase",
                   gridspec_kws={"right": 0.88, "left": 0.05, "top": 0.95, "bottom": 0.05, "wspace": 0.07, "hspace": 0.15})
fg.map_dataframe(sns.lineplot, x='n_controls', y='mcc', hue='Model', palette="colorblind")

fg.fig.set_size_inches(17, 12)
fg.axes[-1, 2].set_xlabel('Replicates per group')
fg.axes[2, 0].set_ylabel('MCC')
fg.add_legend()
plt.savefig(result_path + "\\model_comparison_grouped_confint.svg", format="svg", bbox_inces="tight")
plt.savefig(result_path + "\\model_comparison_grouped_confint.png", format="png", bbox_inces="tight")

plt.show()

#%%

print(all_study_params.loc[(all_study_params["b_count"]==1000) & (all_study_params["log_fold_increase"]==2),
      ["tpr_3", "tnr_3", "mcc_3", "n_controls"]])

print(all_study_params.loc[(all_study_params["b_count"]==200) & (all_study_params["num_increase"]==600),
      ["tpr_1", "tnr_1", "mcc_1", "n_controls"]])

#%%
# Plot for concept fig
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=plot_df, x="n_controls", y="mcc", hue="Model", palette="colorblind", ax = ax)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set(xlabel="Replicates per group", ylabel="MCC")

plt.savefig(result_path + "\\model_comparison_total_confint.svg", format="svg", bbox_inches="tight")
plt.savefig(result_path + "\\model_comparison_total_confint.png", format="png", bbox_inches="tight")

plt.show()

#%%
fg = sns.FacetGrid(data=plot_df_agg, row="Base",
                   gridspec_kws={"right": 0.88, "left": 0.05, "top": 0.95, "bottom": 0.05, "wspace": 0.07, "hspace": 0.15})
fg.map_dataframe(sns.lineplot, x='n_controls', y='mcc', hue='Model', palette="colorblind")

fg.add_legend()
# plt.savefig(result_path + "\\model_comparison.svg", format="svg", bbox_inces="tight")
# plt.savefig(result_path + "\\model_comparison.png", format="png", bbox_inces="tight")

plt.show()

#%%

fg = sns.FacetGrid(data=plot_df_agg, row="log-fold increase",
                   gridspec_kws={"right": 0.88, "left": 0.05, "top": 0.95, "bottom": 0.05, "wspace": 0.07, "hspace": 0.15})
fg.map_dataframe(sns.lineplot, x='n_controls', y='mcc', hue='Model', palette="colorblind")


fg.add_legend()
# plt.savefig(result_path + "\\model_comparison.svg", format="svg", bbox_inces="tight")
# plt.savefig(result_path + "\\model_comparison.png", format="png", bbox_inces="tight")

plt.show()


#%%
# Find data wth high sample size where something goes wrong in SCDC, re-check in R
# -> It barely does not register the one significant effect (p<0.1)

i = 0
for r in results:
    if r.parameters.loc[0, "b_true"] == [-1.609, -1.609, -1.609, -1.609, -1.609]:

        print(i)
        print(r.parameters.loc[9, :])
        i +=1

print(results[96].results[3][9])

data = results[96].data[9]

ns = [10, 10]
k = data.X.shape[1]
x_vec = data.X.flatten()
cell_types = ["cell_" + x for x in data.var.index.tolist()]
cell_types[0] = "cell_" + str(k)
conditions = ["Cond_0", "Cond_1"]

subjects = []
for n in range(ns[0]):
    subjects.append("Cond_0_sub_" + str(n))
for n in range(ns[1]):
    subjects.append("Cond_1_sub_" + str(n))

# produce lists to use in scdney
scdc_cellTypes = []
scdc_subject = []
scdc_cond = []
scdc_sample_cond = []

for i in range(len(x_vec)):
    current_count = x_vec[i]
    current_type = cell_types[i % k]
    current_subject = subjects[i // k]
    current_condition = conditions[i // (k * ns[0])]

    scdc_sample_cond.append(current_condition)

    for j in range(int(current_count)):
        scdc_cellTypes.append(current_type)
        scdc_subject.append(current_subject)
        scdc_cond.append(current_condition)

# save lists as csv
# path = "/home/icb/johannes.ostner/compositional_diff/compositionalDiff-johannes_tests_2/"
path = ""

#%%

with open(path + "paper_simulation_scripts/scdc_r_data/scdc_cellTypes.txt", "w") as f:
    for c in scdc_cellTypes:
        f.write(str(c) + "\n")
with open(path + "paper_simulation_scripts/scdc_r_data/scdc_subject.txt", "w") as f:
    for c in scdc_subject:
        f.write(str(c) + "\n")
with open(path + "paper_simulation_scripts/scdc_r_data/scdc_condition.txt", "w") as f:
    for c in scdc_cond:
        f.write(str(c) + "\n")
with open(path + "paper_simulation_scripts/scdc_r_data/scdc_short_conditions.txt", "w") as f:
    for c in scdc_sample_cond:
        f.write(str(c) + "\n")
