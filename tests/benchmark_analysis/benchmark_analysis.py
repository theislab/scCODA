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

#%%
importlib.reload(ana)
# Get data

path = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\compositionalDiff-johannes_tests_2\\data\\overall_benchmark"

#%%
# Find relations between numerical values and composition/increase vectors

b = []
for y1_0 in [20, 30, 50, 75, 115, 180, 280, 430, 667, 1000]:
    b_i = np.round(gen.counts_from_first(y1_0, 5000, 5), 3)
    b.append(np.round(np.log(b_i / 5000), 2))
print(b)

b_counts = dict(zip([b_i[0] for b_i in b], [20, 30, 50, 75, 115, 180, 280, 430, 667, 1000]))

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
# all_study_params: One line per data point
# all_study_params_agg: Aggregate identical simulation parameters
p = []
params = []
for b_i in b:
    print(b_i[0])
    _, all_study_params, all_study_params_agg = ana.multi_run_study_analysis_prepare(path,
                                                                      file_identifier=str(b_i[0]))

    all_study_params_agg = ana.get_scores(all_study_params_agg)
    all_study_params = ana.get_scores(all_study_params)
    params.append(all_study_params_agg)
    p.append(all_study_params)

#%%
# Get some new metrics that make plotting more convenient

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
# Get some new metrics that make plotting more convenient

all_study_params_2 = pd.concat(p)
all_study_params_2["n_controls"] = [ast.literal_eval(x)[0] for x in all_study_params_2["n_samples"].tolist()]
all_study_params_2["n_cases"] = [ast.literal_eval(x)[1] for x in all_study_params_2["n_samples"].tolist()]
all_study_params_2["n_total"] = all_study_params_2["n_total"].astype("float")
all_study_params_2["w"] = [ast.literal_eval(x)[0][0] for x in all_study_params_2["w_true"]]
all_study_params_2["b_0"] = [ast.literal_eval(x)[0] for x in all_study_params_2["b_true"]]
all_study_params_2["b_count"] = [b_counts[np.round(x, 2)] for x in all_study_params_2["b_0"]]

bs = all_study_params_2["b_0"].tolist()
ws = all_study_params_2["w"].tolist()
increases = []
for i in range(len(bs)):
    increases.append(b_w_dict[bs[i]][ws[i]])
all_study_params_2["num_increase"] = increases
all_study_params_2["log_fold_increase"] = np.log2((all_study_params_2["num_increase"] +
                                                       all_study_params_2["b_count"]) /
                                                      all_study_params_2["b_count"])

print(all_study_params_2)

#%%
# Save that stuff
result_path = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\compositionalDiff-johannes_tests_2\\data\\benchmark_results"

with open(result_path + "\\normal_results_aggregated.pkl", "wb") as f:
    pkl.dump(all_study_params_agg_2, file=f)
with open(result_path + "\\normal_results.pkl", "wb") as f:
    pkl.dump(all_study_params_2, file=f)

#%%
# Exploration...
sns.heatmap(data=all_study_params_agg_2[["b_count", "num_increase", "mcc"]].pivot_table("mcc", "b_count", "num_increase"), vmin=-1, vmax=1)
plt.show()

#%%

sns.heatmap(data=all_study_params_agg_2[["n_cases", "n_controls", "mcc"]].pivot_table("mcc", "n_controls", "n_cases"), vmin=-1, vmax=1)
plt.show()

#%%

print(all_study_params_agg_2[all_study_params_agg_2["n_samples"]=="[10, 10]"])

#%%
# One heatmap for every combination of b, w
def draw_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    d = data.pivot(index=args[1], columns=args[0], values=args[2])
    sns.heatmap(d, **kwargs, vmin=-1, vmax=1)

fg = sns.FacetGrid(all_study_params_agg_2, col='num_increase', row="b_count")
fg.map_dataframe(draw_heatmap, 'n_controls', 'n_cases', 'mcc', cbar=False)
plt.show()

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
# Inclusion probability - No results here!

def recalculate_inclusion_probability(results, threshold):

    for r in results:
        for res in r.mcmc_results.values():
            res[1]["final_parameter"] = np.where(res[1]["inclusion_prob"] > threshold,
                                                 res[1]["mean_nonzero"],
                                                 0)
        r.get_discovery_rates()

    # Generate all_study_params
    all_study_params = pd.concat([r.parameters for r in results])
    simulation_parameters = ["cases", "K", "n_total", "n_samples", "b_true", "w_true", "num_results"]
    all_study_params[simulation_parameters] = all_study_params[simulation_parameters].astype(str)

    # Aggregate over identical parameter sets
    all_study_params_agg = all_study_params.groupby(simulation_parameters).sum()
    all_study_params_agg = ana.get_scores(all_study_params_agg.reset_index())

    return all_study_params_agg


def multiple_threshold_compare(results, thresholds, results_df):

    for t in thresholds:
        print("threshold: " + str(t))
        t_results = recalculate_inclusion_probability(results, t)

        t_results["threshold"] = t

        results_df = results_df.append(t_results)

    return results_df


import os


def multiple_thresholds_with_load(path, thresholds, file_identifier="result_"):
    files = os.listdir(path)

    results = []

    print("Calculating discovery rates...")
    i = 0

    # For all files:
    for f in files:
        i += 1

        print("Preparing: ", i / len(files))
        if file_identifier in f:
            # Load file
            r = ana.renamed_load(open(path + "/" + f, "rb"))

            results.append(r)

    results_df = pd.DataFrame()

    results_df = multiple_threshold_compare(results, thresholds, results_df)

    return results_df


id = "b_[[-2.01 -1.53 -1.53 -1.53 -1.53]]_w_[[[0.034, 0.0, 0.0, 0.0, 0.0]]]"
results_df = multiple_thresholds_with_load(path, [0.5, 0.7], file_identifier=id)

#%%
# Calculate metrics for different threshold values

thresholds = np.round(np.arange(0.05, 1, 0.05), 2)

total_df = pd.DataFrame()
for b_i in b:
    print(b_i[0])
    results_df = multiple_thresholds_with_load(path, thresholds, file_identifier=str(b_i[0]))

    total_df = total_df.append(results_df)

#%%

sns.lineplot(data=total_df, x="threshold", y="mcc", hue="")
plt.show()

#%%
# extra metrics

total_df_2 = total_df
total_df_2["n_controls"] = [ast.literal_eval(x)[0] for x in total_df_2["n_samples"].tolist()]
total_df_2["n_cases"] = [ast.literal_eval(x)[1] for x in total_df_2["n_samples"].tolist()]
total_df_2["n_total"] = total_df_2["n_total"].astype("float")
total_df_2["w"] = [ast.literal_eval(x)[0][0] for x in total_df_2["w_true"]]
total_df_2["b_0"] = [ast.literal_eval(x)[0] for x in total_df_2["b_true"]]
total_df_2["b_count"] = [b_counts[np.round(x, 2)] for x in total_df_2["b_0"]]

bs = total_df_2["b_0"].tolist()
ws = total_df_2["w"].tolist()
increases = []
for i in range(len(bs)):
    increases.append(b_w_dict[bs[i]][ws[i]])
total_df_2["num_increase"] = increases
total_df_2["log_fold_increase"] = np.log2((total_df_2["num_increase"] +
                                           total_df_2["b_count"]) /
                                          total_df_2["b_count"])

print(total_df_2)

#%%
# Saving
result_path = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\compositionalDiff-johannes_tests_2\\data\\benchmark_results"

with open(result_path + "\\thresholds.pkl", "wb") as f:
    pkl.dump(total_df_2, file=f)

#%%
# more plots
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
sns.lineplot(data=total_df_2, x="threshold", y="tpr", hue="num_increase", ax=ax[0])
plt.axvline(x=0.56)
sns.lineplot(data=total_df_2, x="threshold", y="tnr", hue="num_increase", ax=ax[1])
plt.axvline(x=0.56)

plt.show()

sns.lineplot(data=total_df_2.loc[total_df_2["num_increase"]>50], x="threshold", y="mcc")
plt.axvline(x=0.56)
plt.show()

#%%

# Negative effect testing
# Same setup as before
importlib.reload(ana)

path = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\compositionalDiff-johannes_tests_2\\data\\negative_benchmark"


_, all_study_params_neg, all_study_params_neg_agg = ana.multi_run_study_analysis_prepare(path)

all_study_params_neg_agg = ana.get_scores(all_study_params_neg_agg)
all_study_params_neg = ana.get_scores(all_study_params_neg)


b = []
for y1_0 in [115, 280, 1000]:
    b_i = np.round(gen.counts_from_first(y1_0, 5000, 5), 3)
    b.append(np.round(np.log(b_i / 5000), 2))
print(b)

b_counts = dict(zip([b_i[0] for b_i in b], [115, 280, 1000]))

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
    for change in [-10, -50, -100]:
        _, w = gen.b_w_from_abs_change(b_i, change, 5000)
        w_0 = np.round(w[0], 3)
        w_d[w_0] = change
    b_w_dict[b_t[0]] = w_d
    i += 1
print(b_w_dict)

all_study_params_neg = all_study_params_neg
all_study_params_neg["n_controls"] = [ast.literal_eval(x)[0] for x in all_study_params_neg["n_samples"].tolist()]
all_study_params_neg["n_cases"] = [ast.literal_eval(x)[1] for x in all_study_params_neg["n_samples"].tolist()]
all_study_params_neg["n_total"] = all_study_params_neg["n_total"].astype("float")
all_study_params_neg["w"] = [ast.literal_eval(x)[0][0] for x in all_study_params_neg["w_true"]]
all_study_params_neg["b_0"] = [ast.literal_eval(x)[0] for x in all_study_params_neg["b_true"]]
all_study_params_neg["b_count"] = [b_counts[np.round(x, 2)] for x in all_study_params_neg["b_0"]]

bs = all_study_params_neg["b_0"].tolist()
ws = all_study_params_neg["w"].tolist()
increases = []
for i in range(len(bs)):
    increases.append(b_w_dict[bs[i]][ws[i]])
all_study_params_neg["num_increase"] = increases
all_study_params_neg["log_fold_increase"] = np.log2((all_study_params_neg["num_increase"] +
                                                       all_study_params_neg["b_count"]) /
                                                      all_study_params_neg["b_count"])

print(all_study_params_neg)

all_study_params_neg_agg["n_controls"] = [ast.literal_eval(x)[0] for x in all_study_params_neg_agg["n_samples"].tolist()]
all_study_params_neg_agg["n_cases"] = [ast.literal_eval(x)[1] for x in all_study_params_neg_agg["n_samples"].tolist()]
all_study_params_neg_agg["n_total"] = all_study_params_neg_agg["n_total"].astype("float")
all_study_params_neg_agg["w"] = [ast.literal_eval(x)[0][0] for x in all_study_params_neg_agg["w_true"]]
all_study_params_neg_agg["b_0"] = [ast.literal_eval(x)[0] for x in all_study_params_neg_agg["b_true"]]
all_study_params_neg_agg["b_count"] = [b_counts[np.round(x, 2)] for x in all_study_params_neg_agg["b_0"]]

bs = all_study_params_neg_agg["b_0"].tolist()
ws = all_study_params_neg_agg["w"].tolist()
increases = []
for i in range(len(bs)):
    increases.append(b_w_dict[bs[i]][ws[i]])
all_study_params_neg_agg["num_increase"] = increases
all_study_params_neg_agg["log_fold_increase"] = np.log2((all_study_params_neg_agg["num_increase"] +
                                                       all_study_params_neg_agg["b_count"]) /
                                                      all_study_params_neg_agg["b_count"])

print(all_study_params_neg_agg)

#%%
# Save that stuff
result_path = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\compositionalDiff-johannes_tests_2\\data\\benchmark_results"

with open(result_path + "\\results_negative.pkl", "wb") as f:
    pkl.dump(all_study_params_neg, file=f)
with open(result_path + "\\results_negative_aggregated.pkl", "wb") as f:
    pkl.dump(all_study_params_neg_agg, file=f)
#%%
# Plots
result_path = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\compositionalDiff-johannes_tests_2\\data\\benchmark_results"


def draw_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    d = data.pivot(index=args[1], columns=args[0], values=args[2])
    sns.heatmap(d, **kwargs, vmin=-1, vmax=1)

fg = sns.FacetGrid(all_study_params_neg_agg, col='num_increase', row="b_count")
fg.map_dataframe(draw_heatmap, 'n_controls', 'n_cases', 'mcc', cbar=False)
plt.savefig(result_path + "\\negative_heatmaps.png")
plt.show()

#%%

# For comparison: Positive heatmaps
result_path = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\compositionalDiff-johannes_tests_2\\data\\benchmark_results"

with open(result_path + "\\results_aggregated.pkl", "rb") as f:
    all_study_params_agg_pos = pkl.load(file=f)

#%%

fg = sns.FacetGrid(all_study_params_agg_pos.loc[(all_study_params_agg_pos["b_count"].isin([115, 280, 1000])) &
                                              (all_study_params_agg_pos["num_increase"].isin([10, 50, 100]))],
                   col='num_increase', row="b_count")
fg.map_dataframe(draw_heatmap, 'n_controls', 'n_cases', 'mcc', cbar=False)
plt.savefig(result_path + "\\pos_heatmaps_for_negative.png")
plt.show()

# -> For same absolute cell count change, negative effects are detected even a little better!

#%%

def draw_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    d = data.pivot(index=args[1], columns=args[0], values=args[2])
    sns.heatmap(d, **kwargs, vmin=-1, vmax=1)

fg = sns.FacetGrid(all_study_params_agg_pos.loc[(all_study_params_agg_pos["b_count"].isin([50, 180, 430])) &
                                              (all_study_params_agg_pos["num_increase"].isin([40, 80, 200]))],
                   col='num_increase', row="b_count")
fg.map_dataframe(draw_heatmap, 'n_controls', 'n_cases', 'mcc', cbar=False)
plt.savefig(result_path + "\\negative_heatmaps.png")
plt.show()

