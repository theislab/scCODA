# Run model comparison benchmark for scCODA paper
import sys

# only need if on server:
sys.path.insert(0, '/home/icb/johannes.ostner/compositional_diff/scCODA/')

# if running on server:
import benchmark_utils as util
# else:
# import paper_simulation_scripts.benchmark_utils as util


dataset_path = "../data/threshold_determination/generated_datasets_005_balanced/"
dataset_path_server = '/home/icb/johannes.ostner/compositional_diff/benchmark_data/threshold_determination/generated_datasets_005_balanced/'

save_path = "../data/threshold_determination/threshold_determination_005_balanced/"
save_path_server = '/home/icb/johannes.ostner/compositional_diff/benchmark_results/threshold_determination_005_balanced/'

models = ["scCODA"]


# util.benchmark(dataset_path, save_path, models, "threshold", server=False, keep_sccoda_results=True)
util.benchmark(dataset_path_server, save_path_server, models, "threshold", server=True, keep_sccoda_results=True)
