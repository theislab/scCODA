"""
Helper functions to convert single-cell data to scCODA compositional analysis data

:authors: Johannes Ostner
"""
import pandas as pd
import anndata as ad
import os
import numpy as np

from anndata import AnnData
from typing import Optional, Tuple, Collection, Union, List


def read_anndata_one_sample(
        adata: AnnData,
        cell_type_identifier: str,
        covariate_key: Optional[str] = None
) -> Tuple[np.ndarray, dict]:
    """
    Converts a single scRNA-seq data set from scanpy (anndata format) to a row of a cell count matrix.

    It is assumed that a column of adata.obs (e.g. Louvain clustering results) contains the cell type assignment.
    Additionally, covariates (control/disease group, ...) can be specified as a subdict in adata.uns

    Usage:

    ``cell_counts, covs = from_scanpy(adata, cell_type_identifier="Louvain", covariate_key="covariates")``

    Parameters
    ----------
    adata
        single-cell data object from scanpy
    cell_type_identifier
        column name in adata.obs that specifies the cell types
    covariate_key
        key for adata.uns, where the covariate values are stored

    Returns
    -------
    A numpy array for the cell counts and a dict for the covariates

    cell_counts
        cell count vector
    covs
        covariate dictionary
    """

    # Calculate cell counts for the sample
    cell_counts = adata.obs[cell_type_identifier].value_counts()

    # extracting covariates from uns
    if covariate_key is not None:
        covs = adata.uns[covariate_key]
        return cell_counts, covs

    else:
        return cell_counts


def from_scanpy_list(
        samples: List[AnnData],
        cell_type_identifier: str,
        covariate_key: Optional[str] = None,
        covariate_df: Optional[pd.DataFrame] = None
) -> AnnData:
    """
    Creates a compositional analysis data set from a list of scanpy data sets.

    To use this function, all data sets need to have one identically named column in adata.obs that contains the cell type assignment.
    Covariates can either be specified via a key in adata.uns, or as a separate DataFrame

    Usage:

    ``data = from_scanpy_list([adata1, adata2, adata3], cell_type_identifier="Louvain", covariate_df="covariates")``

    Parameters
    ----------
    samples
        list of scanpy data sets
    cell_type_identifier
        column name in adata.obs that specifies the cell types
    covariate_key
        key for adata.uns, where covariate values are stored
    covariate_df
        DataFrame with covariates

    Returns
    -------
    A compositional analysis data set

    data
        A compositional analysis data set
    """

    count_data = pd.DataFrame()
    covariate_data = pd.DataFrame()

    # iterate over anndata objects for each sample
    if covariate_key is not None:
        for s in samples:

            cell_counts, covs = read_anndata_one_sample(s, cell_type_identifier, covariate_key)
            cell_counts = pd.DataFrame(cell_counts).T
            count_data = pd.concat([count_data, cell_counts])
            covariate_data = pd.concat([covariate_data, pd.Series(covs).to_frame().T], ignore_index=True)
    elif covariate_df is not None:
        for s in samples:
            cell_counts = read_anndata_one_sample(s, cell_type_identifier)
            cell_counts = pd.DataFrame(cell_counts).T
            count_data = pd.concat([count_data, cell_counts])
            covariate_data = covariate_df
    else:
        print("No covariate information specified!")
        return

    # Replace NaNs
    count_data = count_data.fillna(0)
    covariate_data.index = covariate_data.index.astype(str)

    var_dat = count_data.sum(axis=0).rename("n_cells").to_frame()
    var_dat.index = var_dat.index.astype(str)

    return ad.AnnData(X=count_data.values,
                      var=var_dat,
                      obs=covariate_data)


def from_scanpy_dir(
        path: str,
        cell_type_identifier: str,
        covariate_key: Optional[str] = None,
        covariate_df: Optional[pd.DataFrame] = None
) -> AnnData:
    """
    Creates a compositional analysis data set from all scanpy data sets in a directory.

    To use this function, all data sets need to have one identically named column in adata.obs that contains the cell type assignment.
    Covariates can either be specified via a key in adata.uns, or as a separate DataFrame

    Usage:
    ``data = from_scanpy_dir("./path/to/directory", cell_type_identifier="Louvain", covariate_key="covariates")``

    Parameters
    ----------
    path
        path to directory
    cell_type_identifier
        column name in adata.obs that specifies the cell types
    covariate_key
        key for adata.uns, where covariate values are stored
    covariate_df
        DataFrame with covariates

    Returns
    -------
    A compositional analysis data set

    data
        A compositional analysis data set
    """

    count_data = pd.DataFrame()
    covariate_data = pd.DataFrame()

    filenames = os.listdir(path)
    if covariate_key is not None:
        for f in filenames:
            adata = ad.read_h5ad(f)

            cell_counts, covs = read_anndata_one_sample(adata, cell_type_identifier, covariate_key)
            cell_counts = pd.DataFrame(cell_counts).T
            count_data = pd.concat([count_data, cell_counts])
            covariate_data = pd.concat([covariate_data, pd.Series(covs).to_frame().T], ignore_index=True)
    elif covariate_df is not None:
        for f in filenames:
            adata = ad.read_h5ad(f)

            cell_counts = read_anndata_one_sample(adata, cell_type_identifier)
            cell_counts = pd.DataFrame(cell_counts).T
            count_data = pd.concat([count_data, cell_counts])
            covariate_data = covariate_df
    else:
        print("No covariate information specified!")
        return

    # Replace NaNs
    count_data = count_data.fillna(0)
    covariate_data.index = covariate_data.index.astype(str)

    var_dat = count_data.sum(axis=0).rename("n_cells").to_frame()
    var_dat.index = var_dat.index.astype(str)

    return ad.AnnData(X=count_data.values,
                      var=var_dat,
                      obs=covariate_data)


def from_scanpy(
        adata: AnnData,
        cell_type_identifier: str,
        sample_identifier: str,
        covariate_key: Optional[str] = None,
        covariate_df: Optional[pd.DataFrame] = None
) -> AnnData:

    """
    Creates a compositional analysis dataset from a single anndata object, as it is produced by e.g. scanpy.

    The anndata object needs to have a column in adata.obs that contains the cell type assignment,
    and one column that specifies the grouping into samples.
    Covariates can either be specified via a key in adata.uns, or as a separate DataFrame.

    NOTE: The order of samples in the returned dataset is determined by the first occurence of cells from each sample in `adata`

    Parameters
    ----------
    adata
        list of scanpy data sets
    cell_type_identifier
        column name in adata.obs that specifies the cell types
    sample_identifier
        column name in adata.obs that specifies the sample
    covariate_key
        key for adata.uns, where covariate values are stored
    covariate_df
        DataFrame with covariates

    Returns
    -------
    A compositional analysis data set

    data
        A compositional analysis data set

    """

    groups = adata.obs.value_counts([sample_identifier, cell_type_identifier])
    count_data = groups.unstack(level=cell_type_identifier)
    count_data = count_data.fillna(0)

    if covariate_key is not None:
        covariate_df = pd.DataFrame(adata.uns[covariate_key])
    elif covariate_df is None:
        print("No covariate information specified!")
        covariate_df = pd.DataFrame(index=count_data.index)

    if set(covariate_df.index) != set(count_data.index):
        raise ValueError("anndata sample names and covariate_df index do not have the same elements!")
    covs_ord = covariate_df.reindex(count_data.index)
    covs_ord.index = covs_ord.index.astype(str)

    var_dat = count_data.sum(axis=0).rename("n_cells").to_frame()
    var_dat.index = var_dat.index.astype(str)

    return ad.AnnData(X=count_data.values,
                      var=var_dat,
                      obs=covs_ord)


def from_pandas(
        df: pd.DataFrame,
        covariate_columns: List[str]
) -> AnnData:
    """
    Converts a Pandas DataFrame into a compositional analysis data set.
    The DataFrame must contain one row per sample, columns can be cell types or covariates

    Note that all columns that are not specified as covariates are assumed to be cell types.

    Usage:
    ``data = from_pandas(df, covariate_columns=["cov1", "cov2"])``

    Parameters
    ----------
    df
        A pandas DataFrame with each row representing a sample; the columns can be cell counts or covariates
    covariate_columns
        List of column names that are interpreted as covariates; all other columns will be seen as cell types

    Returns
    -------
    A compositional analysis data set

    data
        A compositional analysis data set
    """

    covariate_data = df.loc[:, covariate_columns]
    covariate_data.index = covariate_data.index.astype(str)
    count_data = df.loc[:, ~df.columns.isin(covariate_data)]
    celltypes = pd.DataFrame(index=count_data.columns)

    return ad.AnnData(X=count_data.values,
                      var=celltypes,
                      obs=covariate_data)
