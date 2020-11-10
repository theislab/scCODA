# Run model comparison benchmark for scCODA paper
import sys

# only need if on server:
sys.path.insert(0, '/home/icb/johannes.ostner/compositional_diff/scCODA/')

# if running on server:
# import benchmark_utils as util
# else:
import paper_simulation_scripts.benchmark_utils as util


dataset_path = "../data/overall_benchmark/generated_datasets_005/"
dataset_path_server = '/home/icb/johannes.ostner/compositional_diff/benchmark_data/overall_benchmark/generated_datasets_005/'

save_path = "../data/benchmark_results/overall_benchmark_005/"
save_path_server = '/home/icb/johannes.ostner/compositional_diff/benchmark_results/overall_benchmark_005/'

models = ["scCODA"]

# Dont try running this at home! There are 150.000 datasets in this benchmark!
util.benchmark(dataset_path_server, save_path_server, models, "overall", server=True, keep_sccoda_results=True)
