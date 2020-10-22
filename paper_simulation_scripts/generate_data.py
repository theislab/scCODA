import numpy as np
import os
import pickle as pkl
import pandas as pd
import itertools
import sys

sys.path.insert(0, '/home/icb/johannes.ostner/compositional_diff/SCDCdm/')

from scdcdm.util import data_generation as gen


def generate_compositional_datasets(n_cell_types, n_cells, n_samples,
                                    fct_base, fct_change,
                                    n_repetitions, mode="absolute",
                                    write_path = None, file_name = None):

    if mode == "absolute":
        simulation_params = list(itertools.product(n_cell_types, n_cells, n_samples, fct_base, fct_change))
    elif mode == "relative":
        temp_params = list(itertools.product(n_cell_types, n_cells, n_samples, fct_base))
        simulation_params = []
        for p in temp_params:
            for c in fct_change:
                p = list(p)
                base = p[3]
                change = base * c
                p_ = p + [change]
                simulation_params.append(p_)

    else:
        raise ValueError("Wrong mode specified!")

    out_parameters = pd.DataFrame(columns=['n_cell_types', 'n_cells',
                                           'n_controls', 'n_cases',
                                           'Base', 'Increase', 'log-fold increase',
                                           'b_true', 'w_true'])
    out_datasets = []

    i = 1

    # iterate over all combinations of parameters
    for n_cell_types_, n_cells_, n_samples_, fct_base_, fct_change_ in simulation_params:

        # initialize parameter df
        parameters = pd.DataFrame(columns=['n_cell_types', 'n_cells',
                                           'n_controls', 'n_cases',
                                           'Base', 'Increase', 'log-fold increase',
                                           'b_true', 'w_true'])
        datasets = []

        print(f"parameters {i}/{len(simulation_params)}")
        # calculate b for use in gen.generate_case_control, normalize
        b = np.round(gen.counts_from_first(fct_base_, n_cells_, n_cell_types_), 3)
        b_t = np.round(np.log(b / n_cells_), 3)
        # calculate w for use in gen.generate_case_control
        _, w = np.round(gen.b_w_from_abs_change(b, fct_change_, n_cells_), 3)

        lf_change = np.round(np.log2((fct_change_ + fct_base_) / fct_base_), 2)

        # Generate n_repetitions datasets, add to parameter df and dataset list
        for j in range(n_repetitions):
            temp_data = gen.generate_case_control(cases=1, K=n_cell_types_,
                                                  n_total=n_cells_, n_samples=n_samples_,
                                                  b_true=b_t, w_true=[w])
            datasets.append(temp_data)

            params = [n_cell_types_, n_cells_,
                      n_samples_[0], n_samples_[1],
                      fct_base_, fct_change_, lf_change,
                      b_t, [w]]
            parameters = parameters.append(dict(zip(parameters.columns, params)), ignore_index=True)

            if write_path is not None and file_name is not None:
                write = {"datasets": datasets, "parameters": parameters}

                with open(write_path + file_name + "_" + str(i), "wb") as f:
                    pkl.dump(write, f)
            else:
                out_datasets = out_datasets + datasets
                out_parameters = out_parameters.append(parameters)

        i += 1

    out = {"datasets": out_datasets, "parameters": out_parameters}

    return out


#%%

# generate data for overall benchamark

n_cell_types = [5]
n_cells = [5000]
n_samples = [[i+1, j+1] for i in range(10) for j in range(10)]
fct_base = [20, 30, 50, 75, 115, 180, 280, 430, 667, 1000]
fct_change = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 400, 600, 800, 1000]
n_repetitions = 10

# write_path = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\SCDCdm\\data\\overall_benchmark\\generated_data\\"
write_path = "/home/icb/johannes.ostner/compositional_diff/benchmark_results/overall_benchmark_data/"
file_name = "overall_data"

overall_data = generate_compositional_datasets(n_cell_types=n_cell_types, n_cells=n_cells,
                                               n_samples=n_samples, fct_base=fct_base, fct_change=fct_change,
                                               n_repetitions=n_repetitions, mode="absolute",
                                               write_path=write_path, file_name=file_name)


#%%

# generate data for model comparison benchamark

np.random.seed(1234)

n_cell_types = [5]
n_cells = [5000]
n_samples = [[i+1, i+1] for i in range(10)]
fct_base = [200, 400, 600, 800, 1000]
fct_change = [1/3, 1/2, 1, 2, 3]
n_repetitions = 20

write_path = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\SCDCdm\\data\\model_comparison\\generated_datasets_test\\"
# write_path = "/home/icb/johannes.ostner/compositional_diff/benchmark_results/model_comparison_data/"
file_name = "model_comp_data"

comp_data = generate_compositional_datasets(n_cell_types=n_cell_types, n_cells=n_cells,
                                               n_samples=n_samples, fct_base=fct_base, fct_change=fct_change,
                                               n_repetitions=n_repetitions, mode="relative",
                                               write_path=write_path, file_name=file_name)

