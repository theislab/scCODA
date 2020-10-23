# Run model comparison benchmark for SCDCdm paper
import numpy as np
import pandas as pd
import pickle as pkl
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt

import importlib

# only need if on server:
# sys.path.insert(0, '/home/icb/johannes.ostner/compositional_diff/SCDCdm/')

# if running on server, delete "paper_simulation_scripts."
import paper_simulation_scripts.benchmark_utils as util

#%%

importlib.reload(util)

dataset_path = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\SCDCdm\\data\\model_comparison\\generated_datasets\\"
dataset_path_new = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\SCDCdm\\data\\model_comparison\\generated_datasets_test\\"

save_path = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\SCDCdm\\data\\model_comparison\\results\\"
save_path_new = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\SCDCdm\\data\\model_comparison\\results_test\\"

# Use all 10 models for comparison
models = ["simple_dm"]

#util.benchmark(dataset_path, save_path, models, "comp", server=False)
util.benchmark(dataset_path_new, save_path_new, models, "comp", server=False)




#%%
results = []

for f in os.listdir(save_path):
    print(f)
    with open(save_path + f, "rb") as file:
        r = pkl.load(file)
        if f == "ALDEx2_alr_results.pkl":
            r.loc[:, "model"] = "ALDEx2_alr"
            results.append(r)
        elif f == "old_results":
            old_results = r
        else:
            results.append(r)

results_new = []

for f in os.listdir(save_path_new):
    print(f)
    with open(save_path_new + f, "rb") as file:
        r = pkl.load(file)
        if f == "ALDEx2_alr_results.pkl":
            r.loc[:, "model"] = "ALDEx2_alr"
            results_new.append(r)
        elif f == "old_results":
            old_results = r
        else:
            results_new.append(r)

#%%

all_res = pd.concat(results)
all_res_new = pd.concat(results_new)

print(all_res_new)

#%%

all_res = util.complete_results(all_res)

#%%
all_res = pd.concat([all_res, old_results])

#%%

# Get only relevant models for plots
models_rel = ["scdcdm", "simple_dm", "scDC (SydneyBioX)", "ancom", "ALDEx2_alr", "alr_ttest", "alr_wilcoxon", "dirichreg", "Haber", "ttest"]
plot_df_new = all_res_new.loc[all_res_new["model"].isin(models_rel)]


#%%
plot_df_new.loc[:, 'model'] = pd.Categorical(plot_df_new['model'], models_rel)
plot_df = plot_df_new.sort_values("model")

linestyles = dict(zip(models_rel, [0, 1, 0, 1, 3, 0, 1, 3, 0, 1]))
colors = dict(zip(models_rel, [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]))

plot_df.loc[:, "linestyle"] = [linestyles[x] for x in plot_df["model"]]
plot_df.loc[:, "color"] = [colors[x] for x in plot_df["model"]]

#%%

leg_labels = ["SCDCdm", "Simple DM", "scDC (SydneyBioX)", "ancom", "ALDEx2",
              "ALR + t", "ALR + Wilcoxon", "Dirichlet regression", "Poisson regression", "t-test"]
print(leg_labels)

#%%

plot_path = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\SCDCdm\\data\\benchmark_results"
#palette = sns.color_palette(['#e41a1c','#377eb8','#4daf4a','#984ea3'])
#palette = sns.color_palette(['#e41a1c'])

# Plot for concept fig
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=plot_df, x="n_controls", y="mcc",
             hue="color",
             # palette=palette,
             style="linestyle",
             # dashes=[(1,0), (4, 4), (7, 2, 2, 2)],
             ax=ax,
             )
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., labels=leg_labels)
ax.set(xlabel="Replicates per group", ylabel="MCC", ylim=[0,1])

# plt.savefig(plot_path + "\\model_comparison_replicates_confint_extended.svg", format="svg", bbox_inches="tight")
# plt.savefig(plot_path + "\\model_comparison_replicates_confint_extended.png", format="png", bbox_inches="tight")

plt.show()

#%%

# Plot for concept fig
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=plot_df, x="log-fold increase", y="mcc",
             hue="color",
             style="linestyle",
             # dashes=[(1,0), (4, 4), (7, 2, 2, 2)],
             ax=ax,
             )
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., labels=leg_labels)
ax.set(xlabel="Replicates per group", ylabel="MCC", ylim=[0,1])

# plt.savefig(plot_path + "\\model_comparison_replicates_confint_extended.svg", format="svg", bbox_inches="tight")
# plt.savefig(plot_path + "\\model_comparison_replicates_confint_extended.png", format="png", bbox_inches="tight")

plt.show()