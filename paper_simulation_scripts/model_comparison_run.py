# Run model comparison benchmark for SCDCdm paper
import sys

# only need if on server:
sys.path.insert(0, '/home/icb/johannes.ostner/compositional_diff/SCDCdm/')

# if running on server:
# import benchmark_utils as util
# else:
import paper_simulation_scripts.benchmark_utils as util


dataset_path = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\SCDCdm\\data\\model_comparison\\generated_datasets_new_001\\"
dataset_path_server = '/home/icb/johannes.ostner/compositional_diff/benchmark_data/model_comparison/generated_datasets_new_001/'

save_path = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\SCDCdm\\data\\model_comparison\\model_comparison_new_001\\"
save_path_server = '/home/icb/johannes.ostner/compositional_diff/benchmark_results/model_comparison_new_001/'

# Use all 10 models for comparison
# models = ["scdcdm"]
# models = ["simple_dm", "alr_ttest", "ALDEx2_alr", "alr_wilcoxon", "dirichreg", "Haber", "ttest"]
models = ["ancom"]

util.benchmark(dataset_path, save_path, models, "comp", server=False)
# util.benchmark(dataset_path_server, save_path_server, models, "comp", server=True)
