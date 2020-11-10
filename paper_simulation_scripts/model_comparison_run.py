# Run model comparison benchmark for scCODA paper
import sys

# only need if on server:
sys.path.insert(0, '/home/icb/johannes.ostner/compositional_diff/scCODA/')

# if running on server:
import benchmark_utils as util
# else:
# import paper_simulation_scripts.benchmark_utils as util


dataset_path = "../data/model_comparison/generated_datasets_new_005/"
dataset_path_server = '/home/icb/johannes.ostner/compositional_diff/benchmark_data/model_comparison/scdc_missing_datasets/'

save_path = "../data/model_comparison/model_comparison_new_005/"
save_path_server = '/home/icb/johannes.ostner/compositional_diff/benchmark_results/scdc_completion/'

# Use all 10 models for comparison
# models = ["scCODA"]
# models = ["simple_dm", "alr_ttest", "ALDEx2_alr", "alr_wilcoxon", "Haber", "ttest"]
# models = ["ancom"]
models = ["scdc"]
# models = ["dirichreg"]

# util.benchmark(dataset_path, save_path, models, "comp", server=False)
util.benchmark(dataset_path_server, save_path_server, models, "comp", server=True)
