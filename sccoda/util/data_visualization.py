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
from matplotlib.colors import ListedColormap

from anndata import AnnData
from typing import Optional, Tuple, Collection, Union, List

sns.set_style("ticks")


def stackbar(
        y: np.ndarray,
        type_names: List[str],
        title: str,
        level_names: List[str],
        figsize: Optional[Tuple[int, int]] = None,
        dpi: Optional[int] = 100,
        cmap: Optional[ListedColormap] = cm.tab20,
        plot_legend: Optional[bool] = True,
) -> plt.Subplot:
    """
    Plots a stacked barplot for one (discrete) covariate
    Typical use (only inside stacked_barplot): plot_one_stackbar(data.X, data.var.index, "xyz", data.obs.index)

    Parameters
    ----------
    y
        The count data, collapsed onto the level of interest. i.e. a binary covariate has two rows, one for each group, containing the count
        mean of each cell type
    type_names
        The names of all cell types
    title
        Plot title, usually the covariate's name
    level_names
        names of the covariate's levels
    figsize
        figure size
    dpi
        dpi setting
    cmap
        The color map for the barplot
    plot_legend
        If True, adds a legend

    Returns
    -------
    Returns a plot

    ax
        a plot

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
        data: AnnData,
        feature_name: str,
        figsize: Optional[Tuple[int, int]] = None,
        dpi: Optional[int] = 100,
        cmap: Optional[ListedColormap] = cm.tab20,
        plot_legend: Optional[bool] = True,
        level_order: List[str] = None
) -> plt.Subplot:

    """
    Plots a stacked barplot for all levels of a covariate or all samples (if feature_name=="samples").
    Usage: plot_feature_stackbars(data, ["cov1", "cov2", "cov3"])

    Parameters
    ----------
    data
        A scCODA compositional data object
    feature_name
        The name of the covariate to plot. If feature_name=="samples", one bar for every sample will be plotted
    figsize
        figure size
    dpi
        dpi setting
    cmap
        The color map for the barplot
    plot_legend
        If True, adds a legend
    level_order
        Custom ordering of bars on the x-axis

    Returns
    -------
    Returns a plot

    g:
        a plot

    """

    # cell type names
    type_names = data.var.index

    # option to plot one stacked barplot per sample
    if feature_name == "samples":
        if level_order:
            assert set(level_order) == set(data.obs.index), "level order is inconsistent with levels"
            data = data[level_order]
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
        # Order levels
        if level_order:
            assert set(level_order) == set(data.obs[feature_name]), "level order is inconsistent with levels"
            levels = level_order
        elif hasattr(data.obs[feature_name], 'cat'):
            levels = data.obs[feature_name].cat.categories.to_list()
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
        data: AnnData,
        feature_name: str,
        y_scale: str = "relative",
        plot_facets: bool = False,
        add_dots: bool = False,
        cell_types: Optional[list] = None,
        args_boxplot: Optional[dict] = {},
        args_swarmplot: Optional[dict] = {},
        figsize: Optional[Tuple[int, int]] = None,
        dpi: Optional[int] = 100,
        cmap: Optional[str] = "Blues",
        plot_legend: Optional[bool] = True,
        level_order: List[str] = None
) -> Optional[Tuple[plt.Subplot, sns.axisgrid.FacetGrid]]:
    """\
    Grouped boxplot visualization. The cell counts for each cell type are shown as a group of boxplots,
    with intra--group separation by a covariate from data.obs.

    The cell type groups can either be ordered along the x-axis of a single plot (plot_facets=False) or as plot facets (plot_facets=True).

    Parameters
    ----------
    data
        A scCODA-compatible data object
    feature_name
        The name of the feature in data.obs to plot
    y_scale
        Transformation to of cell counts. Options: "relative" - Relative abundance, "log" - log(count), "count" - absolute abundance (cell counts)
    plot_facets
        If False, plot cell types on the x-axis. If True, plot as facets
    add_dots
        If True, overlay a scatterplot with one dot for each data point
    cell_types
        Subset of cell types that should be plotted
    args_boxplot
        Arguments passed to sns.boxplot
    args_swarmplot
            Arguments passed to sns.swarmplot
    figsize
        figure size
    dpi
        dpi setting
    cmap
        The seaborn color map for the barplot
    plot_legend
        If True, adds a legend
    level_order
        Custom ordering of bars on the x-axis

    Returns
    -------
    Depending on `plot_facets`, returns a :class:`~plt.AxesSubplot` (`plot_facets = False`) or :class:`~sns.axisgrid.FacetGrid` (`plot_facets = True`) object

    ax
        if `plot_facets = False`
    g
        if `plot_facets = True`
    """

    # y scale transformations
    if y_scale == "relative":
        sample_sums = np.sum(data.X, axis=1, keepdims=True)
        X = data.X/sample_sums
        value_name = "Proportion"
    # add pseudocount 1 if using log scale (needs to be improved)
    elif y_scale == "log":
        X = np.log(data.X + 1)
        value_name = "log(count)"
    elif y_scale == "count":
        X = data.X
        value_name = "count"
    else:
        raise ValueError("Invalid y_scale transformation")

    count_df = pd.DataFrame(X, columns=data.var.index, index=data.obs.index).\
        merge(data.obs[feature_name], left_index=True, right_index=True)
    plot_df = pd.melt(count_df, id_vars=feature_name, var_name="Cell type", value_name=value_name)
    if cell_types is not None:
        plot_df = plot_df[plot_df["Cell type"].isin(cell_types)]

    if plot_facets:

        if level_order is None:
            level_order = pd.unique(plot_df[feature_name])

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
            order=level_order,
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
                    order=level_order,
                    **args_swarmplot
                ).set_titles("{col_name}")
            else:
                g.map(
                    sns.swarmplot,
                    feature_name,
                    value_name,
                    hue,
                    order=level_order,
                    **args_swarmplot
                ).set_titles("{col_name}")

        return g

    else:

        if level_order:
            args_boxplot["hue_order"] = level_order
            args_swarmplot["hue_order"] = level_order

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


def rel_abundance_dispersion_plot(
        data: AnnData,
        abundant_threshold: Optional[float] = 0.9,
        default_color: Optional[str] = "Grey",
        abundant_color: Optional[str] = "Red",
        label_cell_types: bool = "True",
        figsize: Optional[Tuple[int, int]] = None,
        dpi: Optional[int] = 100,

) -> plt.Subplot:
    """
    Plots total variance of relative abundance versus minimum relative abundance of all cell types for determination of a reference cell type.
    If the count of the cell type is larger than 0 in more than abundant_threshold percent of all samples,
    the cell type will be marked in a different color.

    Parameters
    ----------
    data
        A scCODA compositional data object
    abundant_threshold
        Presence threshold for abundant cell types.
    default_color
        bar color for all non-minimal cell types, default: "Grey"
    abundant_color
        bar color for cell types with abundant percentage larger than abundant_threshold, default: "Red"
    label_cell_types
        boolean - label dots with cell type names
    figsize
        figure size
    dpi
        dpi setting

    Returns
    -------
    Returns a plot

    ax
        a plot
    """

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    rel_abun = data.X / np.sum(data.X, axis=1, keepdims=True)

    percent_zero = np.sum(data.X == 0, axis=0) / data.X.shape[0]
    nonrare_ct = np.where(percent_zero < 1-abundant_threshold)[0]

    # select reference
    cell_type_disp = np.var(rel_abun, axis=0) / np.mean(rel_abun, axis=0)

    is_abundant = [x in nonrare_ct for x in range(data.X.shape[1])]

    # Scatterplot
    plot_df = pd.DataFrame({
        "Total dispersion": cell_type_disp,
        "Cell type": data.var.index,
        "Presence": 1-percent_zero,
        "Is abundant": is_abundant
    })

    if len(np.unique(plot_df["Is abundant"])) > 1:
        palette = [default_color, abundant_color]
    elif np.unique(plot_df["Is abundant"]) == [False]:
        palette = [default_color]
    else:
        palette = [abundant_color]

    sns.scatterplot(
        data=plot_df,
        x="Presence",
        y="Total dispersion",
        hue="Is abundant",
        palette=palette
    )

    # Text labels for abundant cell types

    abundant_df = plot_df.loc[plot_df["Is abundant"] == True, :]

    def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            ax.text(point['x'] + .02*ax.get_xlim()[1], point['y'], str(point['val']))

    if label_cell_types:
        label_point(
            abundant_df["Presence"],
            abundant_df["Total dispersion"],
            abundant_df["Cell type"],
            plt.gca()
        )

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1, title="Is abundant")

    plt.tight_layout()
    return ax
