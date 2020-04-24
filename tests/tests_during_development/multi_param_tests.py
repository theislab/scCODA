import arviz as az
import pandas as pd
import numpy as np
import importlib
from scdcdm.util import result_classes as res
from scdcdm.util import comp_ana as mod
from scdcdm.util import data_generation as gen
from scdcdm.util import multi_parameter_sampling as mult
from scdcdm.model import dirichlet_models as dm
from scdcdm.model import other_models as om

import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
#%%

cases = [1]
K = [5]
n_samples = [[i+6, i+6] for i in range(5)]
#n_samples = [[9, 9]]
n_total = [1000]
b_true = [np.array([0.2, 0.2, 0.2, 0.2, 0.2]).tolist()]
w_true = []
for x in [1]:
    w_true.append(np.array([[x, 0, 0, 0, 0]]).tolist())
num_results = [2e4]

#%%
importlib.reload(mult)

p = mult.MultiParamSimulation(cases, K, n_total, n_samples, b_true, w_true, num_results,
                              baseline_index=4, formula="x_0")

#%%

p.simulate()

#%%

p.get_discovery_rates()


#%%
print(p.mcmc_results)

print(p.parameters)

#%%
p.save(path="./data/", filename="mult_test")


#%%

# Test multi model sampling
cases = [1]
K = [5]
n_samples = [[i+6, i+6] for i in range(1)]
#n_samples = [[9, 9]]
n_total = [1000]
b_true = [np.array([0.2, 0.2, 0.2, 0.2, 0.2]).tolist()]
w_true = []
for x in [1]:
    w_true.append(np.array([[x, 0, 0, 0, 0]]).tolist())
num_results = [2e4]
#models = ["Poisson", "Simple", "Baseline", "SCDC"]
models = ["SCDC"]


#%%
importlib.reload(mult)
#importlib.reload(om)


p = mult.MultiParamSimulationMultiModel(cases, K, n_total, n_samples, b_true, w_true, num_results,
                                        models)

#%%
p.simulate()

#%%

p.get_discovery_rates()

#%%
print(p.results)

print(p.parameters)


#%%

temp_data = p.data[0]

k = K[0]
x_vec = temp_data.X.flatten()
cell_types = ["cell_" + x for x in temp_data.var.index.tolist()]
cell_types[0] = "cell_" + str(k)
conditions = ["Cond_0", "Cond_1"]
ns = n_samples[0]

subjects = []
for n in range(ns[0]):
    subjects.append("Cond_0_sub_" + str(n))
for n in range(ns[1]):
    subjects.append("Cond_1_sub_" + str(n))
print(subjects)


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

#%%

with open("paper_simulation_scripts/scdc_r_data/scdc_cellTypes.txt", "w") as f:
    for c in scdc_cellTypes:
        f.write(str(c) +"\n")
with open("paper_simulation_scripts/scdc_r_data/scdc_subject.txt", "w") as f:
    for c in scdc_subject:
        f.write(str(c) +"\n")
with open("paper_simulation_scripts/scdc_r_data/scdc_condition.txt", "w") as f:
    for c in scdc_cond:
        f.write(str(c) +"\n")
with open("paper_simulation_scripts/scdc_r_data/scdc_short_conditions.txt", "w") as f:
    for c in scdc_sample_cond:
        f.write(str(c) +"\n")


#%%

with open("paper_simulation_scripts/scdc_r_data/scdc_summary.csv", "r") as f:
    r_summary = pd.read_csv(f, header=0, index_col=1)

print(r_summary)

#%%

p_values = r_summary.loc[r_summary.index.str.contains("condCond_1"), "p.value"].values

print(p_values)

#%%
tp = np.sum(p_values[-1] < 0.05)
fn = np.sum(p_values[-1] >= 0.05)
tn = np.sum(p_values[:-1] >= 0.05)
fp = np.sum(p_values[:-1] < 0.05)
print("tp: ", tp)
print("tn: ", tn)
print("fp: ", fp)
print("fn: ", fn)

#%%

import subprocess as sp

r_summary_2 = sp.call(['C:/Program Files/R/R-3.6.3/bin/Rscript', 'paper_simulation_scripts/scdc_r_data/scdney_server_script.r'])

#%%
print(r_summary_2)