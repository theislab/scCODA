import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pickle as pkl
import os
from pathlib import Path

import paper_simulation_scripts.benchmark_utils as util
#%%

# Read data
save_path = str(Path().absolute()) + "/data/overall_benchmark/overall_benchmark_005/"

sccoda_results = []
sccoda_files = []

for f in os.listdir(save_path):
    with open(save_path + f, "rb") as file:
        r = pkl.load(file)
        if f.startswith("scdcdm"):
            sccoda_results.append(r)
            sccoda_files.append(str.replace(f, "scdcdm", "sccoda"))

# scdc_results = pd.concat(scdc_results, ignore_index=True)

print(len(sccoda_files))



#%%


def threshold_function(k):
    return 1-0.77/(k**0.29)

#%%

save_path = str(Path().absolute()) + "/data/overall_benchmark/overall_benchmark_005/"
new_path = str(Path().absolute()) + "/data/overall_benchmark/overall_benchmark_005_reeval/"
file_names = os.listdir(save_path)

for j in range(len(file_names)):

    with open(save_path + file_names[j], "rb") as file:
        r = pkl.load(file)

    if j % 10 == 0:
        print(f"file {j}/{len(file_names)}")

    new_name = str.replace(file_names[j], "scdcdm", "sccoda")

    out_new = {}
    res = r["results"]
    out_new["effects"] = r["effects"]

    tps = []
    fps = []
    tns = []
    fns = []

    for i in range(len(r["effects"])):

        k = res.loc[i, "n_cell_types"]
        thresh = threshold_function(k)

        eff = r["effects"][i]
        reeval = (eff.loc[:, "Inclusion probability"] > thresh)

        tp = sum([reeval[0]])
        fp = sum(reeval[1:])
        fn = 1 - tp
        tn = k - 1 - fp

        tps.append(tp)
        fps.append(fp)
        tns.append(tn)
        fns.append(fn)

    res.loc[:, "tp"] = tps
    res.loc[:, "fp"] = fps
    res.loc[:, "tn"] = tns
    res.loc[:, "fn"] = fns

    res = util.get_scores(res)
    res["model"] = "sccoda"
    out_new["results"] = res

    with open(new_path + new_name, "wb") as f:
        pkl.dump(out_new, f)


#%%

with open(new_path + "sccoda_results_66.pkl", "rb") as f:
    test = pkl.load(f)
    print(test)


#%%

save_path = str(Path().absolute()) + "/data/model_comparison/model_comparison_new_005/"

for f in os.listdir(save_path):
    if f.startswith("sccoda"):
        with open(save_path + f, "rb") as file:
            temp = pkl.load(file)
            temp["results"] = util.get_scores(temp["results"])
        with open(save_path + f, "wb") as file:
            pkl.dump(temp, file)

#%%

all_res = []

save_path = str(Path().absolute()) + "/data/overall_benchmark/overall_benchmark_005_reeval/"
for f in os.listdir(save_path):
    with open(save_path + f, "rb") as file:
        temp = pkl.load(file)
        all_res.append(temp["results"])

res = pd.concat(all_res, ignore_index=True)

#%%

res.to_csv(str(Path().absolute()) + "/data/overall_benchmark/all_results.csv", encoding='utf-8', index=False)

#%%

print(res.columns)