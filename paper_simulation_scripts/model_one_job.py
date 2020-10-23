import pickle as pkl
import sys

import benchmark_utils as add

dataset_path = sys.argv[1]
save_path = sys.argv[2]
model_name = sys.argv[3]
print("model name:", model_name)

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
              "num_adapt_steps": 4000}

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

results = add.model_all_datasets(dataset_path, model_name, **kwargs)

results = add.get_scores(results)

with open(save_path + model_name + "_results.pkl", "wb") as f:
    pkl.dump(results, f)
