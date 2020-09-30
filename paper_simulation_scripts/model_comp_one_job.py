import pickle as pkl
import ast
import sys

import model_comparison_addition as add

dataset_path = "/home/icb/johannes.ostner/compositional_diff/benchmark_results/generated_datasets/"

save_path = "/home/icb/johannes.ostner/compositional_diff/benchmark_results/results/"

model_name = sys.argv[1]
print("model name:", model_name)

if model_name == "ALDEx2":
    results = add.model_all_datasets(dataset_path, "ALDEx2",
                                     fit_args={"server": True, "method": "we.eBH", "mc_samples": 128},
                                     alpha=0.05, fdr_correct=True)

else:
    results = add.model_all_datasets(dataset_path, model_name, alpha=0.05, fdr_correct=True)

results = add.get_scores(results)

with open(save_path + model_name + "_results.pkl", "wb") as f:
    pkl.dump(results, f)
