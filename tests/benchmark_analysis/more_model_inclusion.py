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
from sccoda.util import multi_parameter_analysis_functions as ana
from sccoda.util import data_generation as gen
from sccoda.model import other_models as om
from sccoda.model import dirichlet_models as mod
from paper_simulation_scripts import benchmark_utils as add
from paper_simulation_scripts.generate_data import generate_compositional_datasets
import patsy as pt

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.4)

#%%
importlib.reload(ana)
# Get data:
# all_study_params: One line per data point
# all_study_params_agg: Combine identical simulation parameters

path = "../../data/overall_benchmark"

results, all_study_params, all_study_params_agg = ana.multi_run_study_analysis_prepare(path)

#%%
def extract_all_generated_data(results, save_path):
    simulation_parameters = ["cases", "K", "n_total", "n_samples", "b_true", "w_true", "num_results"]

    l = len(results)
    k = 1

    generated_data = {}

    for r in results:
        print(f"{k}/{l}")

        generated_data["datasets"] = []

        generated_data["parameters"] = r.parameters.loc[:, simulation_parameters]

        print(r)
        for j in range(len(r.data)):
            generated_data["datasets"].append(r.data[j])

        file_name = f"model_comp_data_{k}"

        with open(save_path + file_name, "wb") as file:
            pkl.dump(generated_data, file)

        k += 1

#%%


dataset_path = "../../data/overall_benchmark/generated_datasets/"

#%%
extract_all_generated_data(results, dataset_path)

#%%




importlib.reload(om)
importlib.reload(add)

model_name = "dirichreg"

dataset_path = "../../data/model_comparison/generated_datasets/"

# results = add.model_all_datasets(dataset_path, "ALDEx2", fit_args={"method": "we.eBH", "mc_samples": 128})
results = add.model_all_datasets(dataset_path, model_name)
#%%

results = add.get_scores(results)

result_path = "../../data/extra_comparisons/results/"

with open(result_path + model_name + "_results.pkl", "wb") as f:
    pkl.dump(results, f)
#%%

results = add.complete_results(results)

#%%




#%%
result_path = "../../data/extra_comparisons/results/"

results = []

for f in os.listdir(result_path):
    print(f)
    with open(result_path + f, "rb") as file:
        r = pkl.load(file)
        if f == "ALDEx2_alr_results.pkl":
            r.loc[:, "model"] = "ALDEx2_alr"
            results.append(r)
        elif f == "old_results":
            old_results = r
        else:
            results.append(r)

#%%

all_res = pd.concat(results)
print(all_res)

#%%

all_res = add.complete_results(all_res)

#%%
all_res = pd.concat([all_res, old_results])

#%%

# Get only relevant models for plots
models_rel = ["scCODA", "simple_dm", "scDC (SydneyBioX)", "ancom", "ALDEx2_alr", "alr_ttest", "alr_wilcoxon", "dirichreg", "Haber", "ttest"]
plot_df = all_res.loc[all_res["Model"].isin(models_rel)]

#%%
plot_df.loc[:, 'Model'] = pd.Categorical(plot_df['Model'], models_rel)
plot_df = plot_df.sort_values("Model")

linestyles = dict(zip(models_rel, [0, 1, 0, 1, 3, 0, 1, 3, 0, 1]))
colors = dict(zip(models_rel, [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]))

plot_df.loc[:, "linestyle"] = [linestyles[x] for x in plot_df["Model"]]
plot_df.loc[:, "color"] = [colors[x] for x in plot_df["Model"]]

#%%

leg_labels = ["scCODA", "Simple DM", "scDC (SydneyBioX)", "ancom", "ALDEx2",
              "ALR + t", "ALR + Wilcoxon", "Dirichlet regression", "Poisson regression", "t-test"]
print(leg_labels)


#%%
plot_path = "../../data/benchmark_results"
palette = sns.color_palette(['#e41a1c','#377eb8','#4daf4a','#984ea3'])

# Plot for concept fig
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=plot_df, x="n_controls", y="mcc",
             hue="color", palette=palette,
             style="linestyle",
             # dashes=[(1, 0), (2, 2, 2, 2, 2, 5, 5, 2, 5, 2, 5, 5, 2, 2, 2, 2, 2, 10), (5, 2, 2, 2)],
             dashes=[(1,0), (4, 4), (7, 2, 2, 2)],
             ax=ax,
             )
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., labels=leg_labels)
ax.set(xlabel="Replicates per group", ylabel="MCC", ylim=[0,1])

plt.savefig(plot_path + "/model_comparison_replicates_confint_extended.svg", format="svg", bbox_inches="tight")
plt.savefig(plot_path + "/model_comparison_replicates_confint_extended.png", format="png", bbox_inches="tight")

plt.show()

#%%

# Plot for concept fig
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=plot_df, x="log-fold increase", y="mcc",
             hue="color", palette=palette,style="linestyle",
             # dashes=[(1, 0), (2, 2, 2, 2, 2, 5, 5, 2, 5, 2, 5, 5, 2, 2, 2, 2, 2, 10), (5, 2, 2, 2)],
             dashes=[(1,0), (4, 4), (7, 2, 2, 2)],
             ax=ax,
             )
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., labels=leg_labels)
ax.set(xlabel="Log-fold increase", ylabel="MCC", ylim=[0,1])

plt.savefig(plot_path + "/model_comparison_logfold_confint_extended.svg", format="svg", bbox_inches="tight")
plt.savefig(plot_path + "/model_comparison_logfold_confint_extended.png", format="png", bbox_inches="tight")

plt.show()


#%%
data_hm = all_res.loc[all_res["Model"]=="ALDEx2"]
data_hm = data_hm.groupby(["log-fold increase", "n_controls"]).agg({"mcc": "mean"}).reset_index()
data_hm = data_hm.pivot("log-fold increase", "n_controls", "mcc")
print(data_hm)

#%%
sns.heatmap(data=data_hm, vmin=-1, vmax=1)
plt.show()




#%%
import skbio

importlib.reload(om)
dataset_path = "../../data/model_comparison/generated_datasets/"

file_names = os.listdir(dataset_path)

results = []

simulation_parameters = ["cases", "K", "n_total", "n_samples", "b_true", "w_true", "num_results"]
params = pd.DataFrame(columns=simulation_parameters)

for name in file_names[23:24]:
    with open(dataset_path + name, "rb") as f:
        data = pkl.load(f)

        params = params.append(data["parameters"])

        for d in range(len(data["datasets"])):
            # mod = om.ALRModel_wilcoxon(data["datasets"][d])
            # mod.fit_model(reference_index=4)
            print(data["datasets"][d].X)
            mod = om.DirichRegModel(data["datasets"][d])
            mod.fit_model()
            print(mod.p_val)
            res = mod.eval_model()
            print(res)
            results.append(res)

#%%

pd.set_option('display.max_columns', 500)


for r in range(len(results)):
    print(params.iloc[r,:])
    print(results[r])


#%%

# generate some data

np.random.seed(1234)

n = 3

cases = 1
K = 5
n_samples = [n, n]
n_total = np.full(shape=[2*n], fill_value=1000)
data = gen.generate_case_control(cases, K, n_total[0], n_samples,
                                 w_true=np.array([[1, 0, 0, 0, 0]]),
                                 b_true=np.log(np.repeat(0.2, K)).tolist())

print(data.uns["w_true"])
print(data.uns["b_true"])

print(data.X)
print(data.obs)

#%%

# play around with ancom

df = pd.DataFrame(data.X, index=data.obs.index)


import skbio
importlib.reload(skbio)

ancom_out = skbio.stats.composition.ancom(df, data.obs["x_0"])

print(ancom_out[0]["Reject null hypothesis"])

#%%

#result_path = "../../scCODA/data/extra_comparisons/results/"
result_path = "../../data/model_comparison/model_comparison_new_005/"

results = []

for f in os.listdir(result_path):
    print(f)
    with open(result_path + f, "rb") as file:
        r = pkl.load(file)
        if f == "ALDEx2_alr_results.pkl":
            r.loc[:, "model"] = "ALDEx2_alr"
            results.append(r)
        elif f == "old_results":
            old_results = r
        else:
            results.append(r)

#%%

all_res = pd.concat(results)

all_res = add.complete_results(all_res)

all_res = pd.concat([all_res, old_results])

# Get only relevant models for plots
models_rel = ["scCODA", "simple_dm", "scDC (SydneyBioX)", "ancom", "ALDEx2_alr", "alr_ttest", "alr_wilcoxon", "dirichreg", "Haber", "ttest"]
plot_df = all_res.loc[all_res["Model"].isin(models_rel)]

#%%

print(plot_df.groupby(["Base", "Increase"]).count())

#%%
n_cell_types = [5]
n_cells = [5000]
n_samples = [[i+1, i+1] for i in range(10)]

write_path = "../../data/model_comparison/generated_datasets_test/"
# write_path = "/home/icb/johannes.ostner/compositional_diff/benchmark_results/model_comparison_data/"

base_change_n = [[200, 1/3, 4], [200, 1/2, 2], [200, 1, 1], [200, 2, 3],
                 [400, 1/3, 1], [400, 1/2, 1],
                 [600, 2, 1],
                 [800, 1/3, 3], [800, 1/2, 1], [800, 2, 2]]

i = 1
for b, c, n in base_change_n:
    file_name = f"comp_add_data_{i}"

    comp_data = generate_compositional_datasets(n_cell_types=n_cell_types, n_cells=n_cells,
                                                n_samples=n_samples, fct_base=[b], fct_change=[c],
                                                n_repetitions=n, mode="relative",
                                                write_path=write_path, file_name=file_name)

    i +=1



#%%
def extract_all_generated_data(save_path):
    files = os.listdir(save_path)

    l = len(files)
    k = 1

    generated_data = []

    for f in files:
        print(f"{k}/{l}")

        with open(save_path+f, "rb") as temp:
            data = pkl.load(temp)

        for d in data["datasets"]:
            generated_data.append(d)

        k += 1

    return generated_data

#%%
dataset_path = "../../data/model_comparison/generated_datasets_test_2/"

data_new = extract_all_generated_data(dataset_path)

#%%
vars_cont = []
vars_case = []

for d in data_new:

    x = d.X
    n = int(x.shape[0]/2)

    vars_cont = vars_cont + np.var(x[:n, :], axis=0).tolist()
    vars_case = vars_case + np.var(x[n:, :], axis=0).tolist()
print(np.mean(vars_cont))
print(np.mean(vars_case))



#%%

dataset_path = "../../data/model_comparison/generated_datasets_new_005/"

file_names = os.listdir(dataset_path)

results = []

simulation_parameters = ["cases", "K", "n_total", "n_samples", "b_true", "w_true", "num_results"]
params = pd.DataFrame(columns=simulation_parameters)

# for name in file_names[23]:
name = file_names[28]
with open(dataset_path + name, "rb") as f:
    data = pkl.load(f)
    print(data["parameters"])

    params = params.append(data["parameters"])

    for d in range(3):
        dat = data["datasets"][d]
        K = dat.X.shape[1]
        # Only one covariate
        formula = "x_0"

        cell_types = dat.var.index.to_list()

        # Get count data
        data_matrix = dat.X.astype("float32")

        # Build covariate matrix from R-like formula
        covariate_matrix = pt.dmatrix(formula, dat.obs)
        covariate_names = covariate_matrix.design_info.column_names[1:]
        covariate_matrix = covariate_matrix[:, 1:]

        # Init model. Baseline index is always the last cell type
        m = mod.BaselineModel(covariate_matrix=np.array(covariate_matrix), data_matrix=data_matrix,
                                  cell_types=cell_types, covariate_names=covariate_names, formula=formula,
                                  baseline_index=K - 1)

        # Run HMC sampling, get results
        result_temp = m.sample_hmc(num_results=20000, n_burnin=5000, num_adapt_steps=4000)
        results.append(result_temp.effect_df)

#%%

print(results[0])
