# Run model comparison benchmark for SCDCdm paper
import sys

# only need if on server:
sys.path.insert(0, '/home/icb/johannes.ostner/compositional_diff/SCDCdm/')

# if running on server:
# import benchmark_utils as util
# else:
import paper_simulation_scripts.benchmark_utils as util


dataset_path = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\SCDCdm\\data\\overall_benchmark\\generated_datasets_new_001\\"
dataset_path_server = '/home/icb/johannes.ostner/compositional_diff/benchmark_data/overall_benchmark/generated_datasets_new_001/'

save_path = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\SCDCdm\\data\\benchmark_results\\overall_benchmark_new_001\\"
save_path_server = '/home/icb/johannes.ostner/compositional_diff/benchmark_results/overall_benchmark_new_001/'

models = ["scdcdm"]

# Dont try running this at home! There are 150.000 datasets in this benchmark!
util.benchmark(dataset_path_server, save_path_server, models, "overall", server=True)
