import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pickle as pkl
import os

import paper_simulation_scripts.benchmark_utils as util
#%%

# Read data
save_path = os.path.expanduser("//data/model_comparison/model_comparison_new_005/")

sccoda_results = []
sccoda_files = []

for f in os.listdir(save_path):
    with open(save_path + f, "rb") as file:
        r = pkl.load(file)
        if f.startswith("scCODA"):
            sccoda_results.append(r)
            sccoda_files.append(f)

# scdc_results = pd.concat(scdc_results, ignore_index=True)

print(len(sccoda_files))

#%%


def threshold_function(k):
    return 1-0.77/(k**0.29)

#%%

new_path = os.path.expanduser("//data/model_comparison/model_comparison_reevaluated/")

for j in range(len(sccoda_files)):

    if j % 10 == 0:
        print(f"file {j}/{len(sccoda_files)}")

    dat = sccoda_results[j]
    out_new = {}
    res = dat["results"]
    out_new["effects"] = []

    for i in range(len(dat["effects"])):

        k = dat["results"].loc[i, "n_cell_types"]
        thresh = threshold_function(k)

        eff = dat["effects"][i].copy()
        reeval = (eff.loc[:, "Inclusion probability"] > thresh)
        eff.loc[:, "Final Parameter Reeval"] = reeval

        tp = sum([reeval[0]])
        fp = sum(reeval[1:])
        fn = 1 - tp
        tn = k - 1 - fp

        res.loc[i, "tp"] = tp
        res.loc[i, "fp"] = fp
        res.loc[i, "tn"] = tn
        res.loc[i, "fn"] = fn

        res = util.get_scores(res)

        out_new["effects"].append(eff)
        out_new["results"] = res

        with open(new_path + sccoda_files[j], "wb") as f:
            pkl.dump(out_new, f)


#%%

with open(new_path + sccoda_files[j], "rb") as f:
    test = pkl.load(f)
    print(test)