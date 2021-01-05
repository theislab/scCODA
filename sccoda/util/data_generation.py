"""
Toolbox for simulating compositional data from ScRNA-seq

This toolbox provides data generation and modelling solutions for compositional data with different specifications.
This data might e.g. come from scRNA-seq experiments.
The covariates are represented by ``X``, the cell count matrix is denoted ``Y``.

To start, we set the dimensions of the data: Number of cell types (``K``), number of covariates (``D``),
number of samples (``N``), and number of cells per sample (``n_total``).

We now generate composition parameters (``b_true``, ``w_true``) and a covariance matrix (``Sigma``)
from some input specifications.
``b_true`` represents the base composition with all covariates set to 0. Adding ``X * w_true`` to this
gives the corresponding parameter for each sample.

After adding a gaussian noise (``Sigma``), we can build a concentration vector for each sample that sums up to 1
via the softmax function.
From there, we can calculate each row of the cell count matrix (``Y``) via a multinomial distribution.

:authors: Johannes Ostner
"""

import numpy as np
import anndata as ad
import pandas as pd
from scipy.special import softmax

from anndata import AnnData
from typing import Optional, Tuple, Collection, Union, List


def generate_case_control(
        cases: int = 1,
        K: int = 5,
        n_total: int = 1000,
        n_samples: List[any] = [5, 5],
        sigma: Optional[np.ndarray] = None,
        b_true: Optional[np.ndarray] = None,
        w_true: Optional[np.ndarray] = None
) -> AnnData:
    """
    Generates compositional data with binary covariates.

    Parameters
    ----------
    cases
        number of covariates.
        This will lead to D=2**cases columns in X, one for each combination of active/inactive covariates.
    K
        Number of cell types
    n_total
        number of cells per sample
    n_samples
        Number of samples per case combination. len(n_samples)=[2**cases]
    sigma
        correlation matrix for cell types,size KxK
    b_true
        bias coefficients, size K
    w_true
        Effect matrix, size DxK

    Returns
    -------
    compositional data

    data
        Anndata object
    """
    D = cases**2

    # Uniform intercepts if none are specifed
    if b_true is None:
        b_true = np.random.uniform(-3, 3, size=K).astype(np.float64)  # bias (alpha)

    # Randomly select covariates that should correlate if none are specified
    if w_true is None:
        n_d = np.random.choice(range(D), size=1)
        n_k = np.random.choice(range(K), size=1)
        w_true = sparse_effect_matrix(D, K, n_d, n_k)

    # Sigma is identity if not specified else
    if sigma is None:
        sigma = np.identity(K) * 0.05

    # noise = noise_std_true * np.random.randn(N, 1).astype(np.float64)

    # Initialize x, y
    x = np.zeros((sum(n_samples), cases))
    y = np.zeros((sum(n_samples), K)) 
    c = 0

    # Binary representation of a number x as list of fixed length
    def binary(num, length):
        return [int(x_n) for x_n in bin(num)[2:].zfill(length)]

    # For all combinations of cases
    for i in range(2**cases):
        # For each sample with this combination
        for j in range(n_samples[i]):
            # row of x is binary representation
            x[c+j] = binary(i, cases)

            # Generate y
            alpha = np.random.multivariate_normal(mean=x[c+j, :].T @ w_true + b_true, cov=sigma).astype(
                np.float64)

            concentration = softmax(alpha).astype(np.float64)
            z = np.random.multinomial(n_total, concentration)
            y[c+j] = z
        c = c+n_samples[i]

    x = x.astype(np.float64)
    y = y.astype(np.float64)

    x_names = ["x_" + str(n) for n in range(x.shape[1])]
    x_df = pd.DataFrame(x, columns=x_names)

    data = ad.AnnData(X=y, obs=x_df, uns={"b_true": b_true, "w_true": w_true})

    return data


def b_w_from_abs_change(
        counts_before: np.ndarray = np.array([200, 200, 200, 200, 200]),
        abs_change: int = 50,
        n_total: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates intercepts and slopes from a starting count and an absolute change for the first cell type

    Parameters
    ----------
    counts_before
        cell counts for control samples
    abs_change
        change of first cell type in terms of cell counts
    n_total
        number of cells per sample. This stays constant over all samples!!!

    Returns
    -------
    Returns an intercept and an effect array

    intercepts
        intercept parameters
    slopes
        slope parameters
    """

    K = counts_before.shape[0]

    # calculate intercepts for control samples
    b = np.log(counts_before / n_total)

    # count vector after applying the effect.
    # sum(counts_after) = n_total;
    # counts_after[0] = counts_before[0] + abs_change
    count_0_after = counts_before[0] + abs_change
    count_other_after = (n_total - count_0_after) / (K - 1)
    counts_after = np.repeat(count_other_after, K)
    counts_after[0] = count_0_after

    # Get parameter vector with effect
    b_after = np.log(counts_after / n_total)

    # w is the difference of b before and after
    w = b_after - b
    # Transform w such that only first entry is nonzero
    w = w - w[K - 1]

    return b, w


def counts_from_first(
        b_0: int = 200,
        n_total: int = 1000,
        K: int = 5
) -> np.ndarray:
    """
    Calculates a count vector from a given first entry, length and sum. The entries 2...K will get the same value.

    Parameters
    ----------
    b_0
        size of first entry
    n_total
        total sum of all entries
    K
        length of output vector (number of cell types)

    Returns
    -------
    An intercept array

    b
        count vector (not necessarily integer), size K

    """
    b = np.repeat((n_total-b_0)/(K-1), K)
    b[0] = b_0
    return b


def sparse_effect_matrix(
        D: int,
        K: int,
        n_d: int,
        n_k: int
) -> np.ndarray:
    """
    Generates a sparse effect matrix

    Parameters
    ----------
    D
        Number of covariates
    K
        Number of cell types
    n_d
        Number of covariates that effect each cell type
    n_k
        Number of cell types that are affected by each covariate

    Returns
    -------
    An effect matrix

    w_true
        Effect matrix
    """

    # Choose indices of affected cell types and covariates randomly
    d_eff = np.random.choice(range(D), size=n_d, replace=False)
    k_eff = np.random.choice(range(K), size=n_k, replace=False)

    # Possible entries of w_true
    w_choice = [0.3, 0.5, 1]

    w_true = np.zeros((D, K))
    # Fill in w_true
    for i in d_eff:
        for j in k_eff:
            c = np.random.choice(3, 1)
            w_true[i, j] = w_choice[c]

    return w_true
