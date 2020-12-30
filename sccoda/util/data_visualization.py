"""
This document contains methods to visualize compositional data that was imported into scCODA's data format.

:authors: Johannes Ostner
"""

# Setup

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm, rcParams

sns.set_style("ticks")


def stackbar(
        y,
        type_names,
        title,
        level_names,
        figsize=None,
        dpi=100,
        cmap=cm.tab20,
        plot_legend=True,
):
    """
    Plots a stacked barplot for one (discrete) covariate
    Typical use (only inside stacked_barplot): plot_one_stackbar(data.X, data.var.index, "xyz", data.obs.index)

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
    figsize -- tuple
        figure size
    dpi -- int
        dpi setting
    cmap -- matplotlib.cm colormap
        The color map for the barplot
    plot_legend -- bool
        If True, adds a legend

    Returns
    -------
    ax -- plot

    """
    n_bars, n_types = y.shape

    figsize = rcParams["figure.figsize"] if figsize is None else figsize

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    r = np.array(range(n_bars))
    sample_sums = np.sum(y, axis=1)

    barwidth = 0.85
    cum_bars = np.zeros(n_bars)

    for n in range(n_types):
        bars = [i / j * 100 for i, j in zip([y[k][n] for k in range(n_bars)], sample_sums)]
        plt.bar(r, bars, bottom=cum_bars, color=cmap(n % cmap.N), width=barwidth, label=type_names[n], linewidth=0)
        cum_bars += bars

    ax.set_title(title)
    if plot_legend:
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
    ax.set_xticks(r)
    ax.set_xticklabels(level_names, rotation=45)
    ax.set_ylabel("Proportion")

    return ax


def stacked_barplot(
        data,
        feature_name,
        figsize=None,
        dpi=100,
        cmap=cm.tab20,
        plot_legend=True,
):

    """
    Plots a stacked barplot for all levels of a covariate or all samples (if feature_name=="samples").
    Usage: plot_feature_stackbars(data, ["cov1", "cov2", "cov3"])

    Parameters
    ----------
    data -- AnnData object
        A scCODA compositional data object
    feature_name -- string
        The name of the covariate to plot. If feature_name=="samples", one bar for every sample will be plotted
    figsize -- tuple
        figure size
    dpi -- int
        dpi setting
    cmap -- matplotlib.cm colormap
        The color map for the barplot
    plot_legend -- bool
        If True, adds a legend

    Returns
    -------
    g -- plot

    """

    # cell type names
    type_names = data.var.index

    # option to plot one stacked barplot per sample
    if feature_name == "samples":
        g = stackbar(
            data.X,
            type_names=data.var.index,
            title="samples",
            level_names=data.obs.index,
            figsize=figsize,
            dpi=dpi,
            cmap=cmap,
            plot_legend=plot_legend,
            )
    else:
        levels = pd.unique(data.obs[feature_name])
        n_levels = len(levels)
        feature_totals = np.zeros([n_levels, data.X.shape[1]])

        for level in range(n_levels):
            l_indices = np.where(data.obs[feature_name] == levels[level])
            feature_totals[level] = np.sum(data.X[l_indices], axis=0)

        g = stackbar(
            feature_totals,
            type_names=type_names,
            title=feature_name,
            level_names=levels,
            figsize=figsize,
            dpi=dpi,
            cmap=cmap,
            plot_legend=plot_legend,
            )

    return g


def boxplots(
        data,
        feature_name,
        log_scale=False,
        plot_facets=False,
        add_dots=False,
        args_boxplot={},
        args_swarmplot={},
        figsize=None,
        dpi=100,
        cmap="Blues",
        plot_legend=True,
):
    """
    Grouped boxplot visualization. The cell counts for each cell type are shown as a group of boxplots,
    with intra--group separation by a covariate from data.obs.

    The cell type groups can either be ordered along the x-axis of a single plot (plot_facets=False) or as plot facets (plot_facets=True).

    Parameters
    ----------
    data -- AnnData object
        A scCODA-compatible data object
    feature_name -- string
        The name of the feature in data.obs to plot
    log_scale -- bool
        Transformation to log-scale. If true, use log(data.X + 1) instead of data.X (pseudocount 1 to avoid log(0)-issues)
    plot_facets -- bool
        If False, plot cell types on the x-axis. If True, plot as facets
    add_dots -- bool
        If True, overlay a scatterplot with one dot for each data point
    args_boxplot -- dict
        Arguments passed to sns.boxplot
    args_swarmplot -- dict
            Arguments passed to sns.swarmplot
    dpi -- int
        dpi setting
    cmap -- string
        The seaborn color map for the barplot
    plot_legend -- bool
        If True, adds a legend

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

    count_df = pd.DataFrame(X, columns=data.var.index, index=data.obs.index).\
        merge(data.obs[feature_name], left_index=True, right_index=True)
    plot_df = pd.melt(count_df, id_vars=feature_name, var_name="Cell type", value_name=value_name)

    if plot_facets:

        K = X.shape[1]

        g = sns.FacetGrid(
            plot_df,
            col="Cell type",
            sharey=False,
            col_wrap=np.floor(np.sqrt(K)),
            height=5,
            aspect=2,
        )
        g.map(
            sns.boxplot,
            feature_name,
            value_name,
            palette=cmap,
            order=pd.unique(plot_df[feature_name]),
            **args_boxplot
        )

        if add_dots:

            if "hue" in args_swarmplot:
                hue = args_swarmplot.pop("hue")
            else:
                hue = None

            if hue is None:
                g.map(
                    sns.swarmplot,
                    feature_name,
                    value_name,
                    color="black",
                    order=pd.unique(plot_df[feature_name]),
                    **args_swarmplot
                ).set_titles("{col_name}")
            else:
                g.map(
                    sns.swarmplot,
                    feature_name,
                    value_name,
                    hue,
                    order=pd.unique(plot_df[feature_name]),
                    **args_swarmplot
                ).set_titles("{col_name}")

        return g

    else:

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        sns.boxplot(x="Cell type", y=value_name, hue=feature_name, data=plot_df, fliersize=1,
                    palette=cmap, ax=ax, **args_boxplot)

        if add_dots:
            sns.swarmplot(
                x="Cell type",
                y=value_name,
                data=plot_df,
                hue=feature_name,
                ax=ax,
                dodge=True,
                color="black",
                **args_swarmplot
            )

        cell_types = pd.unique(plot_df["Cell type"])
        ax.set_xticklabels(cell_types, rotation=90)

        if plot_legend:
            handles, labels = ax.get_legend_handles_labels()
            handout = []
            labelout = []
            for h, l in zip(handles, labels):
                if l not in labelout:
                    labelout.append(l)
                    handout.append(h)
            ax.legend(handout, labelout, loc='upper left', bbox_to_anchor=(1, 1), ncol=1, title=feature_name)

        plt.tight_layout()

        return ax
