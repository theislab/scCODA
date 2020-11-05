import pickle as pkl
import sys
import os

import benchmark_utils as add

dataset_path = sys.argv[1]
save_path = sys.argv[2]
model_name = sys.argv[3]
count = int(sys.argv[4])
if sys.argv[5] == "True":
    keep_scdcdm_results = True
else:
    keep_scdcdm_results = False
print("model name:", model_name)

file_name = os.listdir(dataset_path)[count]

if model_name == "ALDEx2_alr":
    kwargs = {"server": True,
              "method": "we.eBH",
              "mc_samples": 128,
              "denom": [5],
              "alpha": 0.05,
              "fdr_correct": False}

elif model_name == "ALDEx2":
    kwargs = {"server": True,
              "method": "we.eBH",
              "mc_samples": 128,
              "alpha": 0.05,
              "fdr_correct": False}

elif model_name in ["simple_dm", "scdcdm"]:
    kwargs = {"num_results": 20000,
              "n_burnin": 5000,
              "num_adapt_steps": 4000,
              "keep_scdcdm_results": keep_scdcdm_results}

elif model_name in ["alr_ttest", "alr_wilcoxon"]:
    kwargs = {"reference_index": 4,
              "alpha": 0.05,
              "fdr_correct": True}
elif model_name in ["Haber", "ttest", "clr_ttest", "dirichreg"]:
    kwargs = {"alpha": 0.05,
              "fdr_correct": True}
elif model_name == "scdc":
    kwargs = {"server": True}
else:
    kwargs = {}

if keep_scdcdm_results:
    results, effects = add.model_on_one_datafile(dataset_path+file_name, model_name, **kwargs)
    results = add.get_scores(results)
    save = {"results": results, "effects": effects}
else:
    results = add.model_on_one_datafile(dataset_path+file_name, model_name, **kwargs)
    results = add.get_scores(results)
    save = results

with open(save_path + model_name + "_results_" + str(count) + ".pkl", "wb") as f:
    pkl.dump(save, f)
