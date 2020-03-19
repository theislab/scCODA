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
plt.savefig(result_path + "\\negative_heatmaps.png")
plt.show()

#%%

fg = sns.FacetGrid(all_study_params_agg_2.loc[(all_study_params_agg_2["b_count"].isin([115, 280, 1000])) &
                                              (all_study_params_agg_2["num_increase"].isin([10, 50, 100]))],
                   col='num_increase', row="b_count")
fg.map_dataframe(draw_heatmap, 'n_controls', 'n_cases', 'mcc', cbar=False)
plt.savefig(result_path + "\\9_heatmaps_concept_fig.png")
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

plt.savefig(result_path + "\\9_heatmaps_concept_fig.png")
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
plt.savefig(result_path + "\\absolute_increase_lines_concept_fig.png", bbox_inches="tight")

plt.show()

#%%
sns.lineplot(data=lineplot_data, x="Log-fold increase", y="MCC", hue="Base", legend="full", palette=sns.cm.rocket_r)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig(result_path + "\\log_fold_increase_lines_concept_fig.png", bbox_inches="tight")

plt.show()

#%%
sns.palplot(sns.cm.rocket(15))
plt.show()
