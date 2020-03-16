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
importlib.reload(ana)
# Get data

path = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\compositionalDiff-johannes_tests_2\\data\\overall_benchmark"

#%%
b = []
for y1_0 in [20, 30, 50, 75, 115, 180, 280, 430, 667, 1000]:
    b_i = np.round(gen.counts_from_first(y1_0, 5000, 5), 3)
    b.append(np.round(np.log(b_i / 5000), 2))
print(b)

b_counts = dict(zip([b_i[0] for b_i in b], [20, 30, 50, 75, 115, 180, 280, 430, 667, 1000]))

#%%
b2 = []
for y1_0 in [20, 30, 50, 75, 115, 180, 280, 430, 667, 1000]:
    b_i = np.round(gen.counts_from_first(y1_0, 5000, 5), 3)
    b2.append(b_i)

b_w_dict = {}
i = 0
w_all = []
for b_i in b2:
    b_t = np.round(np.log(b_i / 5000), 3)
    print(b_t)
    w_d = {}
    for change in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 400, 600, 800, 1000]:
        _, w = gen.b_w_from_abs_change(b_i, change, 5000)
        w_0 = np.round(w[0], 3)
        w_d[w_0] = change
    b_w_dict[b_t[0]] = w_d
    i += 1
print(b_w_dict)

#%%
params = []
for b_i in b:
    print(b_i[0])
    _, _, all_study_params_agg = ana.multi_run_study_analysis_prepare(path,
                                                                      file_identifier=str(b_i[0]))

    all_study_params_agg = ana.get_scores(all_study_params_agg)
    params.append(all_study_params_agg)

#%%

print(params)

#%%

all_study_params_agg_2 = pd.concat(params)
all_study_params_agg_2["n_controls"] = [ast.literal_eval(x)[0] for x in all_study_params_agg_2["n_samples"].tolist()]
all_study_params_agg_2["n_cases"] = [ast.literal_eval(x)[1] for x in all_study_params_agg_2["n_samples"].tolist()]
all_study_params_agg_2["n_total"] = all_study_params_agg_2["n_total"].astype("float")
all_study_params_agg_2["w"] = [ast.literal_eval(x)[0][0] for x in all_study_params_agg_2["w_true"]]
all_study_params_agg_2["b_0"] = [ast.literal_eval(x)[0] for x in all_study_params_agg_2["b_true"]]
all_study_params_agg_2["b_count"] = [b_counts[np.round(x, 2)] for x in all_study_params_agg_2["b_0"]]

bs = all_study_params_agg_2["b_0"].tolist()
ws = all_study_params_agg_2["w"].tolist()
increases = []
for i in range(len(bs)):
    increases.append(b_w_dict[bs[i]][ws[i]])
all_study_params_agg_2["num_increase"] = increases
all_study_params_agg_2["log_fold_increase"] = np.log2((all_study_params_agg_2["num_increase"] +
                                                       all_study_params_agg_2["b_count"]) /
                                                      all_study_params_agg_2["b_count"])

print(all_study_params_agg_2)

#%%

result_path = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\compositionalDiff-johannes_tests_2\\data\\benchmark_results"

with open(result_path + "\\results_aggregated.pkl", "wb") as f:
    pkl.dump(all_study_params_agg_2, file=f)

#%%

sns.heatmap(data=all_study_params_agg_2[["b_count", "num_increase", "mcc"]].pivot_table("mcc", "b_count", "num_increase"), vmin=-1, vmax=1)
plt.show()

#%%
ws = pd.unique(all_study_params_agg_2["w"])
print(len(ws))

#%%

sns.heatmap(data=all_study_params_agg_2[["n_cases", "n_controls", "mcc"]].pivot_table("mcc", "n_controls", "n_cases"), vmin=-1, vmax=1)
plt.savefig(result_path + "\\replicate_heatmap.png")
plt.show()

#%%

print(all_study_params_agg_2[all_study_params_agg_2["n_samples"]=="[10, 10]"])

#%%

def draw_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    d = data.pivot(index=args[1], columns=args[0], values=args[2])
    sns.heatmap(d, **kwargs, vmin=-1, vmax=1)

fg = sns.FacetGrid(all_study_params_agg_2, col='num_increase', row="b_count")
fg.map_dataframe(draw_heatmap, 'n_controls', 'n_cases', 'mcc', cbar=False)
plt.savefig(result_path + "\\150_heatmaps.png")
plt.show()

#%%
print(all_study_params_agg_2[(all_study_params_agg_2["b_count"]==667) & (all_study_params_agg_2["num_increase"]==1000)])

#%%

sns.lineplot(data=all_study_params_agg_2, x="b_count", y="mcc")
plt.show()

#%%
sns.lineplot(data=all_study_params_agg_2, x="num_increase", y="mcc", hue="b_count")
plt.show()

#%%
sns.lineplot(data=all_study_params_agg_2, x="log_fold_increase", y="mcc", hue="b_count")
plt.show()

#%%
# Inclusion probability



