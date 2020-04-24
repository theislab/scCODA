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
from scdcdm.util import data_generation as gen

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.4)

#%%

result_path = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\compositionalDiff-johannes_tests_2\\data\\benchmark_results"

with open(result_path + "\\results_aggregated.pkl", "rb") as f:
    all_study_params_agg_2 = pkl.load(file=f)

result_path = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\compositionalDiff-johannes_tests_2\\data\\benchmark_results"

with open(result_path + "\\results_negative.pkl", "rb") as f:
    all_study_params_neg = pkl.load(file=f)

#%%


def draw_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    d = data.pivot(index=args[1], columns=args[0], values=args[2])
    sns.heatmap(d, **kwargs, vmin=-1, vmax=1)


fg = sns.FacetGrid(all_study_params_neg,
                   col='num_increase', row="b_count")
fg.map_dataframe(draw_heatmap, 'n_controls', 'n_cases', 'mcc', cbar=False)
# plt.savefig(result_path + "\\negative_heatmaps.svg", format="svg")
plt.show()

#%%

fg = sns.FacetGrid(all_study_params_agg_2.loc[(all_study_params_agg_2["b_count"].isin([115, 280, 1000])) &
                                              (all_study_params_agg_2["num_increase"].isin([10, 50, 100]))],
                   col='num_increase', row="b_count")
fg.map_dataframe(draw_heatmap, 'n_controls', 'n_cases', 'mcc', cbar=False)
#plt.savefig(result_path + "\\9_heatmaps_concept_fig.png")
plt.show()


#%%

base_counts = [50, 180, 430]
increases = [40, 80, 200]

paper_heatmaps_plot_df = all_study_params_agg_2.loc[(all_study_params_agg_2["b_count"].isin(base_counts)) &
                                                    (all_study_params_agg_2["num_increase"].isin(increases))].\
    rename(columns={"b_count": "Base", "num_increase": "Increase",
                    "n_controls": "Control replicates", "n_cases": "Condition replicates", "mcc": "MCC"})

sns.set_style("ticks")
sns.set_context("paper", font_scale=1.4)


def draw_heatmap(**kwargs):
    data = kwargs.pop('data')
    columns = kwargs.pop('columns')
    index = kwargs.pop('index')
    values = kwargs.pop('values')
    d = data.pivot(index=index, columns=columns, values=values)
    sns.heatmap(d, **kwargs)


fg = sns.FacetGrid(paper_heatmaps_plot_df, col='Increase', row="Base", height=3, aspect=1,
                   gridspec_kws={"right": 0.9, "left": 0.05, "top": 0.95, "bottom": 0.05, "wspace": 0.07, "hspace": 0.15},
                   )
cbar_ax = fg.fig.add_axes([.92, .3, .02, .4])
fg.map_dataframe(func=draw_heatmap, columns='Control replicates', index='Condition replicates', values='MCC',
                 cbar_ax=cbar_ax, vmin=-1, vmax=1, square=True)

fg.fig.set_size_inches(13, 12)
for ax in fg.axes[2, :]:
    ax.set_xlabel('Control replicates')

for ax in fg.axes[:, 0]:
 ax.set_ylabel('Condition replicates')

plt.savefig(result_path + "\\9_heatmaps_concept_fig.svg", format="svg")
plt.show()


#%%

fig, axn = plt.subplots(3, 3)
cbar_ax = fig.add_axes([.91, .3, .03, .4])

for i in range(3):
    for j in range(3):
        ax = axn[i, j]
        first_plot = ((i == 0) & (j == 0))
        print(first_plot)
        data = paper_heatmaps_plot_df.loc[(paper_heatmaps_plot_df["Base"] == base_counts[i]) &
                                          (paper_heatmaps_plot_df["Increase"] == base_counts[j])].\
            pivot(columns='Control replicates', index='Condition replicates', values='MCC')
        sns.heatmap(data=data, ax=ax, vmin=-1, vmax=1,
                    cbar=first_plot,
                    cbar_ax=None if not first_plot else cbar_ax
                    )

#fig.tight_layout(rect=[0, 0, .9, 1])
plt.show()

#%%
lineplot_data = all_study_params_agg_2.rename(
    columns={"b_count": "Base", "num_increase": "Absolute Increase", "log_fold_increase": "Log-fold increase",
             "n_controls": "Control replicates", "n_cases": "Condition replicates", "mcc": "MCC"}
)

sns.lineplot(data=lineplot_data, x="Base", y="MCC")
plt.show()

#%%
sns.lineplot(data=lineplot_data, x="Absolute Increase", y="MCC", hue="Base", legend="full", palette=sns.cm.rocket_r)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig(result_path + "\\absolute_increase_lines_concept_fig.svg", bbox_inches="tight", format="svg")

plt.show()

#%%
sns.lineplot(data=lineplot_data, x="Log-fold increase", y="MCC", hue="Base", legend="full", palette=sns.cm.rocket_r)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig(result_path + "\\log_fold_increase_lines_concept_fig.svg", bbox_inches="tight", format="svg")

plt.show()

#%%
sns.palplot(sns.cm.rocket(15))
plt.show()

#%%


# threshold plot from thesis


path_no_baseline = "D:/Uni/Master's_Thesis/simulation_results/results/final_testing/threshold_test/no_baseline_new"
path_baseline = "D:/Uni/Master's_Thesis/simulation_results/results/final_testing/threshold_test/baseline_new"

#%%
threshold_dict = {}

for t in np.round(np.arange(0.05, 1, 0.05), 2):
    results, all_study_params, all_study_params_agg = ana.multi_run_study_analysis_prepare(path_no_baseline, custom_threshold=t)
    all_study_params_agg["threshold"] = t
    threshold_dict[t] = all_study_params_agg

#%%
for t, d in threshold_dict.items():
    threshold_dict[t] = ana.get_scores(d)

#%%
for k, d in threshold_dict.items():
    fig, ax = plt.subplots(2, 2, figsize=(12, 16))

    d_2 = pd.DataFrame({"n": [ast.literal_eval(x)[0] for x in d["n_samples"]],
                        "w_true": d["w_true"],
                        "tpr": d["tpr"],
                        "tnr": d["tnr"],
                        "mcc": d["mcc"],
                        "f1_score": d["f1_score"]})
    sns.heatmap(d_2.pivot("n", "w_true", "tpr"), ax=ax[0, 0], vmin=0, vmax=1).set_title("MCMC TPR, t="+str(k))
    sns.heatmap(d_2.pivot("n", "w_true", "tnr"), ax=ax[0, 1], vmin=0, vmax=1).set_title("MCMC TNR, t="+str(k))
    sns.heatmap(d_2.pivot("n", "w_true", "mcc"), ax=ax[1, 1], vmin=-1, vmax=1).set_title("MCMC Matthews Correlation coefficient, t="+str(k))
    sns.heatmap(d_2.pivot("n", "w_true", "f1_score"), ax=ax[1, 0], vmin=0, vmax=1).set_title("MCMC F1-score, t="+str(k))
    plt.savefig("C:/Users/Johannes/Documents/Uni/Master's_Thesis/simulation_results/final_benchmark/threshold_test/new_no_baseline_t_"+str(k)+".png")

    plt.show()


#%%

threshold_df = pd.concat(threshold_dict.values())

#%%
sns.set(style="ticks")
fig, ax = plt.subplots(6, 4, figsize=(15, 25), sharey=True)

ws = pd.unique(threshold_df["w_true"])

for t in range(6):
    sns.lineplot(data=threshold_df[threshold_df["w_true"]==ws[t]], x="threshold", y="tpr", hue="n_samples", ax=ax[t,0])
    sns.lineplot(data=threshold_df[threshold_df["w_true"]==ws[t]], x="threshold", y="tnr", hue="n_samples", ax=ax[t,1])
    sns.lineplot(data=threshold_df[threshold_df["w_true"]==ws[t]], x="threshold", y="mcc", hue="n_samples", ax=ax[t,2])
    sns.lineplot(data=threshold_df[threshold_df["w_true"]==ws[t]], x="threshold", y="f1_score", hue="n_samples", ax=ax[t,3])

for a, row in zip(ax[:,0], ws):
    a.annotate(row, xy=(0, 0.5), xytext=(-a.yaxis.labelpad - 5, 0),
                xycoords=a.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')

for a, col in zip(ax[0], ["TPR", "TNR", "Matthews Correlation coefficient", "F1-score"]):
    a.annotate(col, xy=(0.5, 1), xytext=(0, 5),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')


fig.tight_layout()
plt.savefig("C:/Users/Johannes/Documents/Uni/Master's_Thesis/simulation_results/final_benchmark/threshold_test/new_rates_plot_no_baseline")
plt.show()


#%%
threshold_mean = threshold_df.groupby(["threshold"]).mean()

#print(threshold_mean.reset_index().loc[threshold_mean.reset_index().loc[:,"w_true"]=="[[1, 0, 0, 0, 0]]"])
print(threshold_mean)

# Optimal thresholds: baseline:0.6 or 0.65, no baseline:0.8 --> threshold: 0.7
# Optimal thresholds NEW: baseline:0.5 or 0.4, no baseline:0.5 or 0.45--> threshold: 0.5


#%%
results, all_study_params, all_study_params_agg = ana.multi_run_study_analysis_prepare(path_baseline, keep_results=True)

#%%
for r in results:
    if r.parameters["w_true"][0] == [[1, 0, 0, 0, 0]]:
        r.mcmc_results[4].plot()

#%%
for r in results:
    if r.parameters["w_true"][0] == [[1, 0, 0 ,0, 0]]:
        print(r.mcmc_results[4])