# Generates compositional datasets for benchmarks.

import numpy as np
import os
import pickle as pkl
import pandas as pd
import itertools
import sys

# Insert path to SCDCdm package for running on server
sys.path.insert(0, '/home/icb/johannes.ostner/compositional_diff/SCDCdm/')

from scdcdm.util import data_generation as gen


def generate_compositional_datasets(n_cell_types, n_cells, n_samples,
                                    fct_base, fct_change,
                                    n_repetitions, mode="absolute",
                                    write_path=None, file_name=None):
    """
    Generate compositional case-control data for all combinations of n_cell_types, n_cells, n_samples,
    fct_base, fct_change and save them to disk or return them.
    Datasets are always modeled such that there is an effect on the first cell type, while all other types are unaffected
    For each parameter combination, n_repetitions datasets are generated.

    Parameters
    ----------
    n_cell_types: list
        Number of cell types
    n_cells: list
        total number of cells per sample
    n_samples: list
        Number of samples per group. Each list enement must be of the type [n_controls, n_cases]
    fct_base: list
        Mean abundance for first cell type
    fct_change: list
        Change in first cell type between groups. See mode for details
    n_repetitions: int
        Number of repeated data generations for each parameter combination
    mode: str
        If "absolute", fct_change is interpreted as an absolute change in cell counts
        If "relative", fct_change is interpreted as a change relative to fct_base.
        fct_change(absolute) = fct_change(relative)*fct_base
    write_path: str
        Path to folder where files are written. If None, data is returned instead of written to disk
    file_name
        prefix for files. If None, data is returned instead of written to disk

    Returns
    -------
    If writing to disk is chosen, writes one pickled file per combination of parameters to disk.
    Each file contains n_repetitions datasets.
    They are structured a dict with "parameters" being a DataFrame that contains the generation parameters
    and "datasets" being a list of scdcdm datasets (see the scdcdm documentation for details)

    Otherwise, all generated datasets are returned in one dict with the same structure as described above
    """

    # Cenerate list of all parameter combinations
    if mode == "absolute":
        simulation_params = list(itertools.product(n_cell_types, n_cells, n_samples, fct_base, fct_change))
    elif mode == "relative":
        # For relative mode, we need to calculate the change for each parameter set separately
        temp_params = list(itertools.product(n_cell_types, n_cells, n_samples, fct_base))
        simulation_params = []
        for p in temp_params:
            p = list(p)

            # Balanced base generation
            if p[3] == "balanced":
                p[3] = p[1] * (1 / p[0])
            for c in fct_change:
                base = p[3]
                change = base * c
                # bugfix if ct1 makes up all the cells
                if base + change == p[1]:
                    change = change - 1
                p_ = p + [change]
                p_ = tuple(p_)
                simulation_params.append(p_)

    else:
        raise ValueError("Wrong mode specified!")

    # Initialize output components
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
            sigma = np.identity(n_cell_types_) * 0.05
            temp_data = gen.generate_case_control(cases=1, K=n_cell_types_,
                                                  n_total=n_cells_, n_samples=n_samples_,
                                                  b_true=b_t, w_true=[w],
                                                  sigma=sigma)
            datasets.append(temp_data)

            params = [n_cell_types_, n_cells_,
                      n_samples_[0], n_samples_[1],
                      fct_base_, fct_change_, lf_change,
                      b_t, [w]]
            parameters = parameters.append(dict(zip(parameters.columns, params)), ignore_index=True)

        # If writing to disk, do this now
        if write_path is not None and file_name is not None:
            write = {"datasets": datasets, "parameters": parameters}

            with open(write_path + file_name + "_" + str(i), "wb") as f:
                pkl.dump(write, f)
        # Else, add generated data and parameters to output objects
        else:
            out_datasets = out_datasets + datasets
            out_parameters = out_parameters.append(parameters)

        i += 1

    out = {"datasets": out_datasets, "parameters": out_parameters}

    return out


#%%
if __name__ == "main":
    # generate data for overall benchamark

    np.random.seed(1234)

    n_cell_types = [5]
    n_cells = [5000]
    n_samples = [[i+1, j+1] for i in range(10) for j in range(10)]
    fct_base = [20, 30, 50, 75, 115, 180, 280, 430, 667, 1000]
    fct_change = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 400, 600, 800, 1000]
    n_repetitions = 10

    write_path = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\SCDCdm\\data\\overall_benchmark\\generated_data\\"
    # write_path = "/home/icb/johannes.ostner/compositional_diff/benchmark_results/overall_benchmark_data/"
    file_name = "overall_data"

    overall_data = generate_compositional_datasets(n_cell_types=n_cell_types, n_cells=n_cells,
                                                   n_samples=n_samples, fct_base=fct_base, fct_change=fct_change,
                                                   n_repetitions=n_repetitions, mode="absolute",
                                                   write_path=write_path, file_name=file_name)


    # generate data for model comparison benchamark

    np.random.seed(1234)

    n_cell_types = [5]
    n_cells = [5000]
    n_samples = [[i+1, i+1] for i in range(10)]
    fct_base = [200, 400, 600, 800, 1000]
    fct_change = [1/3, 1/2, 1, 2, 3]
    n_repetitions = 20

    write_path = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\SCDCdm\\data\\model_comparison\\generated_datasets_new_001\\"
    # write_path = "/home/icb/johannes.ostner/compositional_diff/benchmark_results/model_comparison_data/"
    file_name = "model_comp_data"

    comp_data = generate_compositional_datasets(n_cell_types=n_cell_types, n_cells=n_cells,
                                                n_samples=n_samples, fct_base=fct_base, fct_change=fct_change,
                                                n_repetitions=n_repetitions, mode="relative",
                                                write_path=write_path, file_name=file_name)


    # generate data for threshold determination benchamark

    np.random.seed(1234)

    n_cell_types = np.arange(2, 16, 1).tolist()
    n_cells = [5000]
    n_samples = [[i+1, i+1] for i in range(10)]
    fct_base = ["balanced"]
    fct_change = [0.25, 0.5, 1]
    n_repetitions = 20

    write_path = "C:\\Users\\Johannes\\Documents\\Uni\\Master's_Thesis\\SCDCdm\\data\\threshold_determination\\generated_datasets_005_balanced\\"
    # write_path = "/home/icb/johannes.ostner/compositional_diff/benchmark_results/model_comparison_data/"
    file_name = "threshold_data"

    treshold_data = generate_compositional_datasets(n_cell_types=n_cell_types, n_cells=n_cells,
                                                    n_samples=n_samples, fct_base=fct_base, fct_change=fct_change,
                                                    n_repetitions=n_repetitions, mode="relative",
                                                    write_path=write_path, file_name=file_name)
