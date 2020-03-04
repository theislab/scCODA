import pandas as pd
import anndata as ad
import numpy as np
import os


def from_scanpy(adata, cell_type_identifier, covariate_key):

    # Calculating cell counts for the sample
    cell_counts = adata.obs[cell_type_identifier].value_counts()

    # extracting covariates from uns
    covs = adata.uns[covariate_key]

    return cell_counts, covs


def from_scanpy_list(samples, cell_type_identifier, covariate_key):

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

