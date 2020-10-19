import pickle as pkl
import sys
import os

import model_comparison_addition as add

dataset_path = "/home/icb/johannes.ostner/compositional_diff/benchmark_results/generated_datasets/"

save_path = "/home/icb/johannes.ostner/compositional_diff/benchmark_results/results/"

model_name = sys.argv[1]
count = int(sys.argv[2])
print("model name:", model_name)

file_name = os.listdir(dataset_path)[count]

if model_name == "ALDEx2_alr":
    results = add.model_on_one_datafile(dataset_path+file_name, "ALDEx2",
                                        fit_args={"server": True, "method": "we.eBH", "mc_samples": 128, "denom": [5]},
                                        alpha=0.05, fdr_correct=False)

elif model_name == "ALDEx2":
    results = add.model_on_one_datafile(dataset_path+file_name, "ALDEx2",
                                        fit_args={"server": True, "method": "we.eBH", "mc_samples": 128},
                                        alpha=0.05, fdr_correct=False)

elif model_name == "simple_dm":
    results = add.model_on_one_datafile(dataset_path+file_name, model_name,
                                        num_results=20000, n_burnin=5000, num_adapt_steps=4000)

else:
    results = add.model_on_one_datafile(dataset_path+file_name, model_name)

results = add.get_scores(results)

with open(save_path + model_name + "_results_" + str(count) + ".pkl", "wb") as f:
    pkl.dump(results, f)