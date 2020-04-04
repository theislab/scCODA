"""
Helper functions to convert single-cell data to scdcdm compositional analysis data
"""
import pandas as pd
import anndata as ad
import numpy as np
import os


def from_scanpy(adata, cell_type_identifier, covariate_key):
    """
    Converts a single scanpy file to a row of a cell count matrix
    Parameters
    ----------
    adata -- anndata object
        single-cell data object from scanpy
    cell_type_identifier -- str
        column name in adata.obs that specifies the cell types
    covariate_key -- str
        key for adata.uns, where the covariate values are stored

    Returns
    -------
    cell_counts
        cell count vector
    covs
        covariate vector
    """

    # Calculating cell counts for the sample
    cell_counts = adata.obs[cell_type_identifier].value_counts()

    # extracting covariates from uns
    covs = adata.uns[covariate_key]

    return cell_counts, covs


def from_scanpy_list(samples, cell_type_identifier, covariate_key):
    """
    Creates a compositional analysis data set from a list of scanpy data sets

    Parameters
    ----------
    samples -- list
        list of scanpy data sets
    cell_type_identifier -- str
        column name in adata.obs that specifies the cell types
    covariate_key -- str
        key for adata.uns, where the covariate values are stored

    Returns
    -------
    data
        A compositional analysis data set
    """

    count_data = pd.DataFrame()
    covariate_data = pd.DataFrame()

    # iterate over anndata objects for each sample
    for s in samples:

        cell_counts, covs = from_scanpy(s, cell_type_identifier, covariate_key)
        count_data = count_data.append(cell_counts, ignore_index=True)
        covariate_data = covariate_data.append(pd.Series(covs), ignore_index=True)

    # Replace NaNs
    count_data = count_data.fillna(0)
    covariate_data = covariate_data.fillna(0)

    return ad.AnnData(X=count_data.values,
                      var=count_data.sum(axis=0).rename("n_cells").to_frame(),
                      obs=covariate_data)


def from_scanpy_dir(path, cell_type_identifier, covariate_key):
    """
    Creates a compositional analysis data set from all scanpy data sets in a directory

    Parameters
    ----------
    path -- str
        path to directory
    cell_type_identifier -- str
        column name in adata.obs that specifies the cell types
    covariate_key -- str
        key for adata.uns, where the covariate values are stored

    Returns
    -------
    data
        A compositional analysis data set
    """

    count_data = pd.DataFrame()
    covariate_data = pd.DataFrame()

    filenames = os.listdir(path)
    for f in filenames:
        adata = ad.read_h5ad(f)

        cell_counts, covs = from_scanpy(adata, cell_type_identifier, covariate_key)
        count_data = count_data.append(cell_counts, ignore_index=True)
        covariate_data = covariate_data.append(pd.Series(covs), ignore_index=True)

    # Replace NaNs
    count_data = count_data.fillna(0)
    covariate_data = covariate_data.fillna(0)

    return ad.AnnData(X=count_data.values,
                      var=count_data.sum(axis=0).rename("n_cells").to_frame(),
                      obs=covariate_data)


def from_pandas(df, covariate_columns):
    """
    Converts a Pandas DataFrame into a compositional analysis data set.

    Parameters
    ----------
    df -- DataFrame
        A pandas DataFrame with each row representing a sample; the columns can be cell counts or covariates
    covariate_columns -- List
        List of column names that are interpreted as covariates; all other columns will be seen as cell types

    Returns
    -------
    data
        A compositional analysis data set
    """

    covariate_data = df.loc[:, covariate_columns]
    count_data = df.loc[:, ~df.columns.isin(covariate_data)]
    celltypes = pd.DataFrame(index=count_data.columns)

    return ad.AnnData(X=count_data.values,
                      var=celltypes,
                      obs=covariate_data)

