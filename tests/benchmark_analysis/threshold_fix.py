import numpy as np
import arviz as az
import seaborn as sns
import pandas as pd
import pickle as pkl
import importlib
import anndata as ad
import ast

# import matplotlib.pyplot as plt
# from scCODA.util import result_classes as res
# from scCODA.util import multi_parameter_sampling as mult
# from scCODA.util import multi_parameter_analysis_functions as ana
# from scCODA.util import data_generation as gen
# from scCODA.model import other_models as om

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.4)

import paper_simulation_scripts.benchmark_utils as util

#%%
importlib.reload(util)

dataset_path = "../../data/threshold_determination/generated_datasets_005/"
dataset_path_server = '/home/icb/johannes.ostner/compositional_diff/benchmark_data/threshold_determination/generated_datasets_005/'

save_path = "../../data/threshold_determination/threshold_determination_005/"
save_path_server = '/home/icb/johannes.ostner/compositional_diff/benchmark_results/threshold_determination_005/'

models = ["scCODA"]


util.benchmark(dataset_path, save_path, models, "threshold", server=False, keep_sccoda_results=True)

