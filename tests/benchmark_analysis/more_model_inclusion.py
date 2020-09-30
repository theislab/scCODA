import numpy as np
import arviz as az
import seaborn as sns
import pandas as pd
import pickle as pkl
import importlib
import os
import anndata as ad
import ast

import matplotlib.pyplot as plt
from scdcdm.util import result_classes as res
from scdcdm.util import multi_parameter_sampling as mult
from scdcdm.util import multi_parameter_analysis_functions as ana
from scdcdm.util import data_generation as gen
from scdcdm.model import other_models as om
from paper_simulation_scripts import model_comparison_addition as add

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.4)

#%%
importlib.reload(ana)
# Get data:
# all_study_params: One line per data point
# all_study_params_agg: Combine identical simulation parameters

path = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\SCDCdm\\data\\model_comparison"

results, all_study_params, all_study_params_agg = ana.multi_run_study_analysis_prepare(path)


def extract_all_generated_data(results, save_path):
    simulation_parameters = ["cases", "K", "n_total", "n_samples", "b_true", "w_true", "num_results"]

    l = len(results)
    k = 1

    generated_data = {}

    for r in results:
        print(f"{k}/{l}")

        generated_data["datasets"] = []

        generated_data["parameters"] = r.parameters.loc[:, simulation_parameters]

        for j in range(len(r.data)):
            generated_data["datasets"].append(r.data[j])

        file_name = f"model_comp_data_{k}"

        with open(save_path + file_name, "wb") as file:
            pkl.dump(generated_data, file)

        k += 1

#%%


dataset_path = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\SCDCdm\\data\\model_comparison\\generated_datasets\\"

#%%
extract_all_generated_data(results, dataset_path)

#%%




importlib.reload(om)

dataset_path = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\SCDCdm\\data\\model_comparison\\generated_datasets\\"

results = add.model_all_datasets(dataset_path, "ALDEx2", fit_args={"method": "we.eBH", "mc_samples": 128})

#%%

results = add.get_scores(results)

results = add.complete_results(results)

#%%


result_path = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\SCDCdm\\data\\extra_comparisons\\results\\"

results = []

for f in os.listdir(result_path):
    with open(result_path + f, "rb") as file:
        r = pkl.load(file)
        results.append(r)

#%%

all_res = pd.concat(results)
print(all_res)

#%%

all_res = add.complete_results(all_res)

#%%

leg_labels = ["clr_ttest", "Haber", "ttest"]

# Plot for concept fig
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=all_res, x="n_controls", y="mcc", hue="Model", palette="colorblind", ax=ax)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., labels=leg_labels)
ax.set(xlabel="Replicates per group", ylabel="MCC", ylim=[0,1])

#plt.savefig(result_path + "\\model_comparison_replicates_confint_extended.svg", format="svg", bbox_inches="tight")
#plt.savefig(result_path + "\\model_comparison_replicates_confint_extended.png", format="png", bbox_inches="tight")

plt.show()