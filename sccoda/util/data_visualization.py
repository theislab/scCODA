"""
This script contains methods to visualize compositional data that was imported into scCODA's data format

:authors: Johannes Ostner
"""

# Setup

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import cm

sns.set_style("ticks")


def plot_one_stackbar(y, type_names, title, level_names):

    """
    Plots a stacked barplot for one (discrete) covariate
    Typical use: plot_one_stackbar(data.X, data.var.index, "xyz", data.obs.index)

    Parameters
    ----------
    y -- numpy array
        The count data, collapsed onto the level of interest. i.e. a binary covariate has two rows containing the count
        mean of all samples of each cell type
    type_names -- list-like
        The names of all cell types
    title -- string
        Plot title, usually the covariate's name
    level_names -- list-like
        names of the covariate's levels

    Returns
    -------
    a plot
    """

    plt.figure(figsize=(20, 10))
    n_samples, n_types = y.shape
    r = np.array(range(n_samples))
    sample_sums = np.sum(y, axis=1)
    barwidth = 0.85
    cum_bars = np.zeros(n_samples)
    colors = cm.tab20

    for n in range(n_types):
        bars = [i / j * 100 for i, j in zip([y[k][n] for k in range(n_samples)], sample_sums)]
        plt.bar(r, bars, bottom=cum_bars, color=colors(n % 20), width=barwidth, label=type_names[n])
        cum_bars += bars

    plt.title(title)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
    plt.xticks(r, level_names, rotation=45)

    plt.show()


def plot_feature_stackbars(data, features):

    """
    Plots stackbars for all listed covariates
    Usage: plot_feature_stackbars(data, ["cov1", "cov2", "cov3"])

    Parameters
    ----------
    data -- AnnData object
        A scCODA-compatible data object
    features -- list
        List of all covariates to plot. Specifying "samples" as an element will plot a stackbar with all samples

    Returns
    -------
    One plot for every category

    """

    type_names = data.var.index
    for f in features:

        if f == "samples":
            plot_one_stackbar(data.X, data.var.index, "samples", data.obs.index)
        else:
            levels = pd.unique(data.obs[f])
            n_levels = len(levels)
            feature_totals = np.zeros([n_levels, data.X.shape[1]])

            for level in range(n_levels):
                l_indices = np.where(data.obs[f] == levels[level])
                feature_totals[level] = np.sum(data.X[l_indices], axis=0)

            plot_one_stackbar(feature_totals, type_names=type_names, level_names=levels, title=f)


def grouped_boxplot(data, feature, log_scale=False, *args, **kwargs):
    """
    Grouped boxplot -cell types on x-Axis; one boxplot per feature level for each cell type
    Parameters
    ----------
    data -- AnnData object
        A scCODA-compatible data object
    feature -- string
        The name of the feature in data.obs to plot
    log_scale -- bool
        If true, use log(data.X + 1) (pseudocount 1 to avoid log(0)-issues)
    *args, **kwargs -- Passed to sns.boxplot

    Returns
    -------
    Plot!
    """

    plt.figure(figsize=(20, 10))

    # add pseudocount 1 if using log scale (needs to be improved)
    if log_scale:
        X = np.log(data.X + 1)
        value_name = "log(count)"
    else:
        X = data.X
        value_name = "count"

    count_df = pd.DataFrame(X, columns=data.var.index, index=data.obs.index).\
        merge(data.obs[feature], left_index=True, right_index=True)
    plot_df = pd.melt(count_df, id_vars=feature, var_name="Cell type", value_name=value_name)

    d = sns.boxplot(x="Cell type", y=value_name, hue=feature, data=plot_df, fliersize=1,
                    palette='Blues', *args, **kwargs)

    loc, labels = plt.xticks()
    d.set_xticklabels(labels, rotation=90)

    return d


def boxplot_facets(data, feature, log_scale=False, args_boxplot={}, args_swarmplot={}):
    """
    Grouped boxplot -cell types on x-Axis; one boxplot per feature level for each cell type
    Parameters
    ----------
    data -- AnnData object
        A scCODA-compatible data object
    feature -- string
        The name of the feature in data.obs to plot
    log_scale -- bool
        If true, use log(data.X + 1) (pseudocount 1 to avoid log(0)-issues)
    args_boxplot -- dict
        Arguments passed to sns.boxplot
    args_swarmplot -- dict
        Arguments passed to sns.swarmplot

    Returns
    -------
    Plot!
    """

    # add pseudocount 1 if using log scale (needs to be improved)
    if log_scale:
        X = np.log(data.X + 1)
        value_name = "log(count)"
    else:
        X = data.X
        value_name = "count"

    K = X.shape[1]

    count_df = pd.DataFrame(X, columns=data.var.index, index=data.obs.index).merge(data.obs, left_index=True,
                                                                                   right_index=True)
    plot_df = pd.melt(count_df, id_vars=data.obs.columns, var_name="Cell type", value_name=value_name)

    if "hue" in args_swarmplot:
        hue = args_swarmplot.pop("hue")
    else:
        hue = None

    g = sns.FacetGrid(plot_df, col="Cell type", sharey=False, col_wrap=np.floor(np.sqrt(K)), height=5, aspect=2)
    g.map(sns.boxplot, feature, value_name, palette="Blues", **args_boxplot)
    if hue is None:
        g.map(sns.swarmplot, feature, value_name, color="black", **args_swarmplot).set_titles("{col_name}")
    else:
        g.map(sns.swarmplot, feature, value_name, hue, **args_swarmplot).set_titles("{col_name}")
    return g
