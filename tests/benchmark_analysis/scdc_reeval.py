import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pickle as pkl
import os

from pathlib import Path

#%%

print(Path().absolute())
#%%

# Read data
save_path = str(Path().absolute()) + "/data/model_comparison/model_comparison_new_005/"

scdc_results = []
scdc_files = []

for f in os.listdir(save_path):
    with open(save_path + f, "rb") as file:
        r = pkl.load(file)
        if f.startswith("scdc_r"):
            scdc_results.append(r)
            scdc_files.append(f)

# scdc_results = pd.concat(scdc_results, ignore_index=True)

print(len(scdc_files))

#%%
import re

completed_numbers = [int(re.findall(r"\d+", x)[0]) for x in scdc_files]
print(len(completed_numbers))

missing_numbers = set(np.arange(1, 251, 1).tolist()) - set(completed_numbers)


print(len(missing_numbers))

#%%
import shutil

data_path = os.path.expanduser("//data/model_comparison/generated_datasets_new_005/")
miss_path = os.path.expanduser("//data/model_comparison/scdc_missing_datasets/")

for n in missing_numbers:
    file_name = f"model_comp_data_{n}"
    shutil.copyfile(data_path + file_name, miss_path + file_name)

#%%

with open(miss_path + file_name, "rb") as path:
    test = pkl.load(path)

print(test)

#%%

# Rename all "scdcdm" to "sccoda"

# Read data
save_path = str(Path().absolute()) + "/data/threshold_determination/threshold_determination_005_balanced/"

scdcdm_results = []
scdcdm_files = []

for f in os.listdir(save_path):
    if f.startswith("scdcdm"):
        with open(save_path + f, "rb") as file:
            r = pkl.load(file)

            scdcdm_results.append(r)
            scdcdm_files.append(f)

sccoda_files = [str.replace(name, "scdcdm", "sccoda") for name in scdcdm_files]
print(sccoda_files)

sccoda_results = []

for res in scdcdm_results:
    res["results"].loc[:, "model"] = "sccoda"
    sccoda_results.append(res)

print(len(sccoda_results))
print(len(sccoda_files))

for i in range(len(sccoda_files)):
    with open(save_path + sccoda_files[i], "wb") as f:
        pkl.dump(sccoda_results[i], f)

#%%

with open(save_path+sccoda_files[23], "rb") as f:
    r = pkl.load(f)
    print(r["results"])

#%%



import shutil

data_path = str(Path().absolute()) + "/data/model_comparison/model_comparison_new_005/"
reeval_path = str(Path().absolute()) + "/data/model_comparison/model_comparison_reevaluated/"

for file_name in os.listdir(reeval_path):
    shutil.copyfile(reeval_path + file_name, data_path + file_name)