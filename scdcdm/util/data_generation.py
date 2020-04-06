"""
Toolbox for simulating compositional data from ScRNA-seq

This toolbox provides data generation and modelling solutions for compositional data with different specifications.
This data might e.g. come from scRNA-seq experiments.
For scenarios 1-4, we first generate composition parameters (b_true, w_true) and a covariance matrix (x) from some input specifications.
We then build a concentration vector for each sample (row of x) that sums up to 1. From there, we can calculate each row of the cell count matrix (y) via a multinomial distribution

:authors: Johannes Ostner
"""

import numpy as np
import anndata as ad
import pandas as pd
from scipy.special import softmax


def generate_normal_uncorrelated(N, D, K, n_total, noise_std_true=1):
    """
    Scenario 1: Normally distributed, independent covariates

    Parameters
    ----------
    N -- int
        Number of samples
    D -- int
        Number of covariates
    K -- int
        Number of cell types
    n_total -- list
        Number of individual cells per sample
    noise_std_true -- float
        noise level. 0: No noise

    Returns
    -------
    data
        Anndata object
    """

    # Generate random composition parameters
    b_true = np.random.normal(0, 1, size=K).astype(np.float32)  # bias (alpha)
    w_true = np.random.normal(0, 1, size=(D, K)).astype(np.float32)  # weights (beta)

    # Generate random covariate matrix
    x = np.random.normal(0, 1, size=(N, D)).astype(np.float32)
    noise = noise_std_true * np.random.randn(N, 1).astype(np.float32)

    # Generate y
    y = np.zeros([N, K], dtype=np.float32)
    for i in range(N):
        # Concentration should sum to 1 for each sample
        concentration = softmax(x[i, :].T@w_true + b_true + noise[i, :]).astype(np.float32)
        y[i, :] = np.random.multinomial(n_total[i], concentration).astype(np.float32)

    x_names = ["x_" + str(n) for n in range(x.shape[1])]
    x_df = pd.DataFrame(x, columns=x_names)

    data = ad.AnnData(X=y, obs=x_df, uns={"b_true": b_true, "w_true": w_true})

    return data


def generate_normal_correlated(N, D, K, n_total, noise_std_true, covariate_mean=None, covariate_var=None):
    """
    Scenario 2: Correlated covariates

    Parameters
    ----------
    N -- int
        Number of samples
    D -- int
        Number of covariates
    K -- int
        Number of cell types
    n_total -- list
        Number of individual cells per sample
    noise_std_true -- float
        noise level. 0: No noise
    covariate_mean -- numpy array [D]
        Mean of each covariate
    covariate_var -- numpy array [DxD]
        Covariance matrix for covariates

    Returns
    -------
    data
        Anndata object
    """

    if covariate_mean is None:
        covariate_mean = np.zeros(shape=D)

    # Generate randomized covariate covariance matrix if none is specified
    if covariate_var is None:
        # Covariates drawn from MvNormal(0, Cov), Cov_ij = p ^|i-j| , p=0.4
        # Tibshirani for correlated covariates: Tibshirani (1996)
        p = 0.4
        covariate_var = np.zeros((D, D))
        for i in range(D):
            for j in range(D):
                covariate_var[i, j] = p**np.abs(i-j)

    # Generate random composition parameters
    b_true = np.random.normal(0, 1, size=K).astype(np.float32)  # bias (alpha)
    w_true = np.random.normal(0, 1, size=(D, K)).astype(np.float32)  # weights (beta)

    # Generate random covariate matrix
    x = np.random.multivariate_normal(size=N, mean=covariate_mean, cov=covariate_var).astype(np.float32)
    noise = noise_std_true * np.random.randn(N, 1).astype(np.float32)

    # Generate y
    y = np.zeros([N, K], dtype=np.float32)
    for i in range(N):
        # Concentration should sum to 1 for each sample
        concentration = softmax(x[i, :].T @ w_true + b_true + noise[i, :]).astype(np.float32)
        y[i, :] = np.random.multinomial(n_total[i], concentration).astype(np.float32)

    x_names = ["x_" + str(n) for n in range(x.shape[1])]
    x_df = pd.DataFrame(x, columns=x_names)

    data = ad.AnnData(X=y, obs=x_df, uns={"b_true": b_true, "w_true": w_true})

    return data


def generate_normal_xy_correlated(N, D, K, n_total, noise_std_true=1,
                                  covariate_mean=None, covariate_var=None, sigma=None):
    """
    Scenario 3: Correlated cell types and covariates

    Parameters
    ----------
    N -- int
        Number of samples
    D -- int
        Number of covariates
    K -- int
        Number of cell types
    n_total -- list
        Number of individual cells per sample
    noise_std_true -- float
        noise level. 0: No noise
    covariate_mean -- numpy array [D]
        Mean of each covariate
    covariate_var -- numpy array [DxD]
        Covariance matrix for all covaraiates
    sigma -- numpy array [KxK]
        correlation matrix for cell types

    Returns
    -------
    data
        Anndata object
    """

    if covariate_mean is None:
        covariate_mean = np.zeros(shape=D)

    if sigma is None:
        sigma = np.identity(K)

    # Generate randomized covariate covariance matrix if none is specified
    if covariate_var is None:
        # Covaraits drawn from MvNormal(0, Cov) Cov_ij = p ^|i-j| , p=0.4
        # Tibshirani for correlated covariates: Tibshirani (1996)
        p = 0.4
        covariate_var = np.zeros((D, D))
        for i in range(D):
            for j in range(D):
                covariate_var[i, j] = p**np.abs(i-j)

    # Generate random composition parameters
    b_true = np.random.normal(0, 1, size=K).astype(np.float32)  # bias (alpha)
    w_true = np.random.normal(0, 1, size=(D, K)).astype(np.float32)  # weights (beta)

    # Generate random covariate matrix
    x = np.random.multivariate_normal(size=N, mean=covariate_mean, cov=covariate_var).astype(np.float32)
    noise = noise_std_true * np.random.randn(N, 1).astype(np.float32)

    # Generate y
    y = np.zeros([N, K], dtype=np.float32)
    for i in range(N):
        # Each row of y is now influenced by sigma
        alpha = np.random.multivariate_normal(mean=x[i, :].T@w_true + b_true, cov=sigma*noise[i, :]).astype(np.float32)
        concentration = softmax(alpha).astype(np.float32)
        y[i, :] = np.random.multinomial(n_total[i], concentration).astype(np.float32)

    x_names = ["x_" + str(n) for n in range(x.shape[1])]
    x_df = pd.DataFrame(x, columns=x_names)

    data = ad.AnnData(X=y, obs=x_df, uns={"b_true": b_true, "w_true": w_true})

    return data


def sparse_effect_matrix(D, K, n_d, n_k):
    """
    Generates a sparse effect matrix

    Parameters
    ----------
    D -- int
        Number of covariates
    K -- int
        Number of cell types
    n_d -- int
        Number of covariates that effect a cell type
    n_k -- int
        Number of cell types that are affected by any covariate

    Returns
    -------
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
    

def generate_sparse_xy_correlated(N, D, K, n_total, noise_std_true=1,
                                  covariate_mean=None, covariate_var=None,
                                  sigma=None,
                                  b_true=None, w_true=None):
    """
    Scenario 4: Sparse true parameters

    Parameters
    ----------
    N -- int
        Number of samples
    D -- int
        Number of covariates
    K -- int
        Number of cell types
    n_total -- list
        Number of individual cells per sample
    noise_std_true -- float
        noise level. 0: No noise
    covariate_mean -- numpy array [D]
        Mean of each covariate
    covariate_var -- numpy array [DxD]
        Covariance matrix for all covaraiates
    sigma -- numpy array [KxK]
        correlation matrix for cell types
    b_true -- numpy array [K]
        bias coefficients
    w_true -- numpy array [DxK]
        Effect matrix

    Returns
    -------
    data
        Anndata object
    """

    if covariate_mean is None:
        covariate_mean = np.zeros(shape=D)

    if sigma is None:
        sigma = np.identity(K)

    # Generate randomized covariate covariance matrix if none is specified
    if covariate_var is None:
        # Covaraits drawn from MvNormal(0, Cov) Cov_ij = p ^|i-j| , p=0.4
        # Tibshirani for correlated covariates: Tibshirani (1996)
        p = 0.4
        covariate_var = np.zeros((D, D))
        for i in range(D):
            for j in range(D):
                covariate_var[i, j] = p ** np.abs(i - j)

    # Uniform intercepts if none are specifed
    if b_true is None:
        b_true = np.random.uniform(-3,3, size=K).astype(np.float32)  # bias (alpha)

    # Randomly select covariates that should correlate if none are specified
    if w_true is None:
        n_d = np.random.choice(range(D), size=1)
        n_k = np.random.choice(range(K), size=1)
        w_true = sparse_effect_matrix(D, K, n_d, n_k)

    # Generate random covariate matrix
    x = np.random.multivariate_normal(size=N, mean=covariate_mean, cov=covariate_var).astype(np.float32)
    noise = noise_std_true * np.random.randn(N, 1).astype(np.float32)

    # Generate y
    y = np.zeros([N, K], dtype=np.float32)
    for i in range(N):
        # Each row of y is now influenced by sigma
        alpha = np.random.multivariate_normal(mean=x[i, :].T @ w_true + b_true, cov=sigma * noise[i, :]).astype(
            np.float32)
        concentration = softmax(alpha).astype(np.float32)
        y[i, :] = np.random.multinomial(n_total[i], concentration).astype(np.float32)

    x_names = ["x_" + str(n) for n in range(x.shape[1])]
    x_df = pd.DataFrame(x, columns=x_names)

    data = ad.AnnData(X=y, obs=x_df, uns={"b_true": b_true, "w_true": w_true})

    return data


def generate_case_control(cases=1, K=5, n_total=1000, n_samples=[5,5], noise_std_true=0,
                          sigma=None, b_true=None, w_true=None):
    """
    Generates compositional data with binary covariates

    Parameters
    ----------
    cases -- int
        number of covariates
    K -- int
        Number of cell types
    n_total -- int
        number of cells per sample
    n_samples -- list
        Number of samples per case combination as array[2**cases]
    noise_std_true -- float
        noise level. 0: No noise - Not in use atm!!!
    sigma -- numpy array [KxK]
        correlation matrix for cell types
    b_true -- numpy array [K]
        bias coefficients
    w_true -- numpy array [DxK]
        Effect matrix

    Returns
    -------
    Anndata object
    """
    D = cases**2

    # Uniform intercepts if none are specifed
    if b_true is None:
        b_true = np.random.uniform(-3, 3, size=K).astype(np.float32)  # bias (alpha)

    # Randomly select covariates that should correlate if none are specified
    if w_true is None:
        n_d = np.random.choice(range(D), size=1)
        n_k = np.random.choice(range(K), size=1)
        w_true = sparse_effect_matrix(D, K, n_d, n_k)

    # Sigma is identity if not specified else
    if sigma is None:
        sigma = np.identity(K) * 0.05

    # noise = noise_std_true * np.random.randn(N, 1).astype(np.float32)

    # Initialize x, y
    x = np.zeros((sum(n_samples), cases))
    y = np.zeros((sum(n_samples), K)) 
    c = 0

    # Binary representation of x as list of fixed length
    def binary(x, length):
        return [int(x_n) for x_n in bin(x)[2:].zfill(length)]

    # For all combinations of cases
    for i in range(2**cases):
        # For each sample with this combination
        for j in range(n_samples[i]):
            # row of x is binary representation
            x[c+j] = binary(i, cases)

            # Generate y
            alpha = np.random.multivariate_normal(mean=x[c+j, :].T @ w_true + b_true, cov=sigma).astype(
                np.float32)

            concentration = softmax(alpha).astype(np.float32)
            z = np.random.multinomial(n_total, concentration)
            y[c+j] = z
        c = c+n_samples[i]

    x = x.astype(np.float32)
    y = y.astype(np.float32)

    x_names = ["x_" + str(n) for n in range(x.shape[1])]
    x_df = pd.DataFrame(x, columns=x_names)

    data = ad.AnnData(X=y, obs=x_df, uns={"b_true": b_true, "w_true": w_true})

    return data


def b_w_from_abs_change(counts_before=np.array([200, 200, 200, 200, 200]), abs_change=50, n_total=1000):
    """
    Calculates intercepts and slopes from a starting count and an absolute change for the first cell type

    Parameters
    ----------
    counts_before -- numpy array
        cell counts for control samples
    abs_change -- int
        change of first cell type in terms of cell counts
    n_total -- int
        number of cells per sample. This stays constant over all samples!!!

    Returns
    -------
    intercepts -- numpy array
        intercept parameters
    slopes -- numpy array
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


def counts_from_first(b_0=200, n_total=1000, K=5):
    b = np.repeat((n_total-b_0)/(K-1), K)
    b[0] = b_0
    return b
