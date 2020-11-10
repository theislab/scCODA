"""
Functions for plotting and analyzing the results from running models over multiple parameter sets

:authors: Johannes Ostner
"""


import numpy as np
import seaborn as sns
import pandas as pd
import pickle as pkl
import os
import ast
import matplotlib.pyplot as plt

# Helpers for loading result classes from old environment
import io


class RenameUnpickler(pkl.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "multi_parameter_sampling" or module == "model.multi_parameter_sampling":
            renamed_module = "scCODA.util.multi_parameter_sampling"
        if module == "compositional_analysis_generation_toolbox" or module == "model.compositional_analysis_generation_toolbox":
            renamed_module = "scCODA.util.data_generation"
        if module == "result_classes" or module == "model.result_classes":
            renamed_module = "scCODA.util.result_classes"
        if module == "final_models" or module == "model.final_models":
            renamed_module = "scCODA.model.dirichlet_models"
        if module == "SCDCpy.util":
            renamed_module = "scCODA.util"

        renamed_name = name
        if name == 'Multi_param_simulation':
            renamed_name = "MultiParamSimulation"
        if name == 'MCMCResult':
            renamed_name = "CAResult"

        return super(RenameUnpickler, self).find_class(renamed_module, renamed_name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


def renamed_loads(pickled_bytes):
    file_obj = io.BytesIO(pickled_bytes)
    return renamed_load(file_obj)


def multi_run_study_analysis_prepare(path, file_identifier="result_", custom_threshold=None):

    """
    Function to read in and calculate discovery rates, ... for an entire directory of multi_parameter_sampling files

    Parameters
    ----------
    path -- str
        path to directory
    file_identifier -- str (optional)
        identifier that is part of all files we want to analyze - only these files are loaded
    custom_threshold -- float (optional)
        custom spike-and-slab threshold

    Returns
    -------
    results -- list
        List of raw result files
    all_study_params -- pandas DataFrame
        Parameters and result data for all files
    all_study_params_agg -- pandas DataFrame
        Parameters and result data, aggregated over all files with identical parameters

    """

    files = os.listdir(path)

    results = []

    print("Calculating discovery rates...")
    i = 0

    # For all files:
    for f in files:
        i += 1

        print("Preparing: ", i / len(files))
        if file_identifier in f:
            # Load file
            r = renamed_load(open(path + "/" + f, "rb"))

            if custom_threshold is not None:
                for r_k, r_i in r.mcmc_results.items():
                    r.mcmc_results[r_k].params["final_parameter"] = np.where(np.isnan(r_i.params["mean_nonzero"]),
                                                                             r_i.params["mean"],
                                                                             np.where(r_i.params[
                                                                                          "inclusion_prob"] > custom_threshold,
                                                                                      r_i.params["mean_nonzero"],
                                                                                      0))

            # Discovery rates for beta
            r.get_discovery_rates()

            results.append(r)

    # Generate all_study_params
    all_study_params = pd.concat([r.parameters for r in results])
    simulation_parameters = ["cases", "K", "n_total", "n_samples", "b_true", "w_true", "num_results"]
    all_study_params[simulation_parameters] = all_study_params[simulation_parameters].astype(str)

    # Aggregate over identical parameter sets
    all_study_params_agg = all_study_params.groupby(simulation_parameters).sum()

    return results, all_study_params, all_study_params_agg.reset_index()


def get_scores(agg_df, models=1):
    """
    Calculates extended summary statistics, such as TPR, TNR, youden index, f1-score, MCC

    Parameters
    ----------
    agg_df -- pandas DataFrame
    models -- int
        number of different models used for the analysis (relevant if data comes from MultiParameterSamplingMultiModel)

    Returns
    -------
    agg_df -- pandas DataFrame
        Same as input, with added columns for summary statistics

    """

    if models == 1:
        tp = agg_df["tp"]
        tn = agg_df["tn"]
        fp = agg_df["fp"]
        fn = agg_df["fn"]

        tpr = (tp / (tp + fn)).fillna(0)
        agg_df["tpr"] = tpr
        tnr = (tn / (tn + fp)).fillna(0)
        agg_df["tnr"] = tnr
        precision = (tp / (tp + fp)).fillna(0)
        agg_df["precision"] = precision
        acc = (tp + tn) / (tp + tn + fp + fn).fillna(0)
        agg_df["accuracy"] = acc

        agg_df["youden"] = tpr + tnr - 1
        agg_df["f1_score"] = 2 * (tpr * precision / (tpr + precision)).fillna(0)

        agg_df["mcc"] = (((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))).fillna(0)

    else:
        for m in range(models):
            m_str = str(m)
            tp = agg_df["tp_" + m_str]
            tn = agg_df["tn_" + m_str]
            fp = agg_df["fp_" + m_str]
            fn = agg_df["fn_" + m_str]

            tpr = (tp / (tp + fn)).fillna(0)
            agg_df["tpr_" + m_str] = tpr
            tnr = (tn / (tn + fp)).fillna(0)
            agg_df["tnr_" + m_str] = tnr
            precision = (tp / (tp + fp)).fillna(0)
            agg_df["precision_" + m_str] = precision
            acc = (tp + tn) / (tp + tn + fp + fn).fillna(0)
            agg_df["accuracy_" + m_str] = acc

            agg_df["youden_" + m_str] = tpr + tnr - 1
            agg_df["f1_score_" + m_str] = 2 * (tpr * precision / (tpr + precision)).fillna(0)

            agg_df["mcc_" + m_str] = (((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))).fillna(0)

    return agg_df


def plot_discovery_rates_agg(rates_df, dim_1='w_true', dim_2=None, path=None):
    """
    Plot heatmap of TPR and TNR for one parameter series vs. another


    Parameters
    ----------
    rates_df -- pandas DataFrame
        format as all_study_params_agg from multi_run_study_analysis_prepare
    dim_1 -- str
        parameter on x-axis
    dim_2 -- str
        parameter on y-axis
    path -- str
        directory to save plot to

    Returns
    -------

    """

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    # If no second dimension specified, make a 1d-heatmap
    if dim_2 is None:
        rates_df = rates_df.groupby(dim_1).mean().reset_index()
        dim_2 = "x"
        # Generate dataframe for plotting
        plot_data = pd.DataFrame({dim_1: rates_df[dim_1],
                                  dim_2: [1 for i in range(rates_df.shape[0])],
                                  "tpr": rates_df["tpr"],
                                  "tnr": rates_df["tnr"]
                                  })
        # plot Heatmaps
        sns.heatmap(plot_data.pivot(dim_1, dim_2, 'tpr'), ax=ax[0], vmin=0, vmax=1).set_title("MCMC TPR")
        sns.heatmap(plot_data.pivot(dim_1, dim_2, 'tnr'), ax=ax[1], vmin=0, vmax=1).set_title("MCMC TNR")
    else:
        rates_df = rates_df.groupby([dim_1, dim_2]).mean().reset_index()
        # Generate dataframe for plotting
        plot_data = pd.DataFrame({dim_1: rates_df[dim_1],
                                  dim_2: rates_df[dim_2],
                                  "tpr": rates_df["tpr"],
                                  "tnr": rates_df["tnr"]
                                  })
        # plot Heatmaps
        sns.heatmap(plot_data.pivot(dim_1, dim_2, 'tpr'), ax=ax[0], vmin=0, vmax=1).set_title("MCMC TPR")
        sns.heatmap(plot_data.pivot(dim_1, dim_2, 'tnr'), ax=ax[1], vmin=0, vmax=1).set_title("MCMC TNR")

    # Save
    if path is not None:
        plt.savefig(path)

    plt.show()


def plot_cases_vs_controls(rates_df, results, identifier_w="", path=None, suptitle=None):
    """
    Plot heatmaps of TPR and TNR for number of case vs. number of control samples.


    Parameters
    ----------
    rates_df -- pandas DataFrame
        Same as all_study_params_agg from multi_run_study_analysis_prepare
    results -- pandas DataFrame
        same format as results from multi_run_study_analysis_prepare
    identifier_w -- str (optional)
        if plotting only a subset of all ground truth effects
    path -- str
        directory to save plot to
    suptitle -- str
        Header for entirety of plots

    Returns
    -------

    """

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    # DataFrame for heatmaps
    rates_df = rates_df.loc[(rates_df["w_true"] == str(identifier_w))].groupby("n_samples").mean().reset_index()
    plot_data = pd.DataFrame({"controls": [ast.literal_eval(x)[0] for x in rates_df["n_samples"].tolist()],
                              "cases": [ast.literal_eval(x)[1] for x in rates_df["n_samples"].tolist()],
                              "tpr": rates_df["tpr"],
                              "tnr": rates_df["tnr"]
                              })

    # Plot heatmaps
    sns.heatmap(plot_data.pivot("controls", "cases", 'tpr'), ax=ax[0, 0], vmin=0, vmax=1).set_title("MCMC TPR")
    sns.heatmap(plot_data.pivot("controls", "cases", 'tnr'), ax=ax[0, 1], vmin=0, vmax=1).set_title("MCMC TNR")

    # Case vs. control count boxplots
    cases_y = []
    controls_y = []

    # Get count data
    for r in results:
        if r.parameters.loc[0, "w_true"] == identifier_w:
            for i in r.mcmc_results:
                n_cases = r.parameters.loc[i, "n_samples"][0]

                controls_y.extend(r.mcmc_results[i].y[:n_cases].tolist())
                cases_y.extend(r.mcmc_results[i].y[n_cases:].tolist())

    cases_y = pd.DataFrame(cases_y)
    controls_y = pd.DataFrame(controls_y)

    # Generate boxplots
    sns.boxplot(data=controls_y, ax=ax[1, 0]).set_title("control group cell counts")
    sns.boxplot(data=cases_y, ax=ax[1, 1]).set_title("case group cell counts")

    # Add title
    if suptitle is not None:
        plt.suptitle(suptitle)
    plt.tight_layout()

    # Save plot
    if path is not None:
        plt.savefig(path)

    plt.show()


def multi_run_study_analysis_prepare_per_param(path, file_identifier="result_"):
    """
    Function to calculate discovery rates, ... for an entire directory of multi_parameter_sampling files.
    Effect Discovery rates are calculated separately for each cell type.

    Parameters
    ----------
    path -- str
        path to directory
    file_identifier -- str (optional)
        identifier that is part of all files we want to analyze

    Returns
    -------
    results -- list
        List of raw result files
    all_study_params -- pandas DataFrame
        Parameters and result data for all files
    all_study_params_agg -- pandas DataFrame
        Parameters and result data, aggregated over all sets of identical parameters
    """

    files = os.listdir(path)

    results = []

    print("Calculating discovery rates...")
    i = 0

    for f in files:
        i += 1

        print("Preparing: ", i / len(files))
        if file_identifier in f:
            # Load file
            r = renamed_load(open(path + "/" + f, "rb"))

            # Discovery rates for beta per parameter
            r.get_discovery_rates_per_param()

            results.append(r)

    # Generate all_study_params
    all_study_params = pd.concat([r.parameters for r in results])
    simulation_parameters = ["cases", "K", "n_total", "n_samples", "b_true", "w_true", "num_results"]
    all_study_params[simulation_parameters] = all_study_params[simulation_parameters].astype(str)

    # Aggregate over identical parameter sets
    all_study_params_agg = all_study_params.groupby(simulation_parameters).sum()

    return results, all_study_params, all_study_params_agg.reset_index()


def plot_cases_vs_controls_per_param(K, rates_df, results, identifier_w, path=None, suptitle=None):
    """
    Plot heatmaps of discovery rate for number of case vs. number of control samples, for each cell type.
    Also plots counts of cases vs. controls for all cell types

    Parameters
    ----------
    K -- int

    rates_df -- pandas DataFrame
        same format as all_study_params_agg from multi_run_study_analysis_prepare
    results -- pandas DataFrame
        same format as results from multi_run_study_analysis_prepare
    identifier_w -- str
        if plotting only a subset of all ground truth effects
    path -- str
        directory to save plot to
    suptitle -- str
        Header for entirety of plots

    Returns
    -------

    """

    # plot initialization
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(2, K, figsize=(K*6, 10))
    for a in ax[1, :]:
        a.set_ylim(0, 1000)

    # Get heatmap data relevant for plotting
    rates_df = rates_df.loc[(rates_df["w_true"] == str(identifier_w))].groupby("n_samples").mean().reset_index()
    rates_df["controls"] = [ast.literal_eval(x)[0] for x in rates_df["n_samples"].tolist()]
    rates_df["cases"] = [ast.literal_eval(x)[1] for x in rates_df["n_samples"].tolist()]

    # For each cell type:
    for i in range(K):
        plot_df = rates_df.loc[:, ["controls", "cases", "correct_"+str(i), "false_"+str(i)]]
        plot_df["disc_rate"] = plot_df["correct_"+str(i)]/(plot_df["correct_"+str(i)] + plot_df["false_"+str(i)])
        # If Effect on cell type ==0: plot in blue, else plot in red
        if identifier_w[0][i] == 0:
            cmap = "Blues_r"
        else:
            cmap = "Reds_r"
        # Plot heatmap
        sns.heatmap(plot_df.pivot("controls", "cases", 'disc_rate'), ax=ax[0, i], vmin=0, vmax=1, cmap=cmap).\
            set_title("Cell type "+str(i+1)+" accuracy - " + "Effect: " + str(identifier_w[0][i]))

    # Get cell count data for each cell type
    cases_y = []
    controls_y = []
    for r in results:
        if r.parameters.loc[0, "w_true"] == identifier_w:
            for i in r.mcmc_results:
                n_cases = r.parameters.loc[i, "n_samples"][0]

                controls_y.extend(r.mcmc_results[i].y[:n_cases].tolist())
                cases_y.extend(r.mcmc_results[i].y[n_cases:].tolist())
    cases_y = cases_y
    controls_y = controls_y

    # Plot boxplots
    for i in range(K):
        box_df = pd.DataFrame({"controls": [y[i] for y in controls_y],
                               "cases": [y[i] for y in cases_y]})
        lf_change = np.round(np.log2(box_df["cases"].mean() / box_df["controls"].mean()), 2)
        sns.boxplot(data=box_df.loc[:, ["cases", "controls"]], ax=ax[1, i], order=["controls", "cases"]).\
            set_title("Log-fold change: "+str(lf_change))

    # Add title
    if suptitle is not None:
        plt.suptitle(suptitle)

    # Save plot
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)

    plt.show()


def multi_run_study_analysis_multi_model_prepare(path, file_identifier="result_"):
    """
    Function to calculate discovery rates, ... for an entire directory of multi_parameter_sampling_multi_model files

    Parameters
    ----------
    path -- str
        path to directory
    file_identifier -- str (optional)
        identifier that is part of all files we want to analyze

    Returns
    -------
    results -- list
        List of raw result files
    all_study_params -- pandas DataFrame
        Parameters and result data for all files
    all_study_params_agg -- pandas DataFrame
        Parameters and result data, aggregated over all sets of identical parameters
    """

    files = os.listdir(path)
    results = []
    print("Calculating discovery rates...")
    i = 0

    # For all files:
    for f in files:
        i += 1

        print("Preparing: ", i / len(files))
        if file_identifier in f:
            # Load file
            r = renamed_load(open(path + "/" + f, "rb"))

            # Discovery rates for beta
            r.get_discovery_rates()

            results.append(r)

    # Generate all_study_params
    all_study_params = pd.concat([r.parameters for r in results])
    simulation_parameters = ["cases", "K", "n_total", "n_samples", "b_true", "w_true", "num_results"]
    all_study_params[simulation_parameters] = all_study_params[simulation_parameters].astype(str)

    # Aggregate over identical parameter sets
    all_study_params_agg = all_study_params.groupby(simulation_parameters).sum()

    return results, all_study_params, all_study_params_agg.reset_index()


def plot_cases_vs_controls_per_param_2(K, rates_df, results, identifier_w, path=None, suptitle=None):
    """
    Optimized version of plot_cases_vs_controls_per_param

    Parameters
    ----------
    K -- int
        number of cell types
    rates_df -- pandas DataFrame
        same format as all_study_params_agg from multi_run_study_analysis_prepare
    results -- pandas DataFrame
        same format as results from multi_run_study_analysis_prepare
    identifier_w -- str (optional)
        if plotting only a subset of all ground truth effects
    path -- str
        directory to save plot to
    suptitle -- str
        Header for entirety of plots

    Returns
    -------

    """


    sns.set_style("whitegrid")

    # Get heatmap data relevant for plotting
    rates_df = rates_df.loc[(rates_df["w_true"] == str(identifier_w))].groupby("n_samples").mean().reset_index()
    rates_df["controls"] = [ast.literal_eval(x)[0] for x in rates_df["n_samples"].tolist()]
    rates_df["cases"] = [ast.literal_eval(x)[1] for x in rates_df["n_samples"].tolist()]

    # Get cell count data for each cell type
    cases_y = []
    controls_y = []
    for r in results:
        if r.parameters.loc[0, "w_true"] == identifier_w:
            for i in r.mcmc_results:
                n_cases = r.parameters.loc[i, "n_samples"][0]

                controls_y.extend(r.mcmc_results[i].y[:n_cases].tolist())
                cases_y.extend(r.mcmc_results[i].y[n_cases:].tolist())
    cases_y = cases_y
    controls_y = controls_y

    # For each cell type:
    for i in range(K):
        # Initialize plot
        fig, ax = plt.subplots(2, 1, figsize=(6, 10))
        ax[1].set_ylim(0, 1000)

        # DataFrame for heatmap
        plot_df = rates_df.loc[:, ["controls", "cases", "correct_"+str(i), "false_"+str(i)]]
        plot_df["disc_rate"] = plot_df["correct_"+str(i)]/(plot_df["correct_"+str(i)] + plot_df["false_"+str(i)])
        # If Effect on cell type ==0: plot in blue, else plot in red
        if identifier_w[0][i] == 0:
            cmap = "Blues_r"
        else:
            cmap = "Reds_r"
        # Plot heatmap
        sns.heatmap(plot_df.pivot("controls", "cases", 'disc_rate'), ax=ax[0], vmin=0, vmax=1, cmap=cmap).\
            set_title("Accuracy")

        # DataFrame for boxplot
        box_df = pd.DataFrame({"controls": [y[i] for y in controls_y],
                               "cases": [y[i] for y in cases_y]})
        change = np.round(box_df["cases"].mean() - box_df["controls"].mean(), 2)
        sns.boxplot(data=box_df.loc[:, ["cases", "controls"]], ax=ax[1], order=["controls", "cases"]).\
            set_title("Average change: "+str(change)+" cells")

        # Add title
        if suptitle is not None:
            plt.suptitle(suptitle)

        # Save plot
        plt.tight_layout()
        if path is not None:
            plt.savefig(path + "_type_" + str(i+1).replace(".", ""), bbox_inches="tight")

    plt.show()
