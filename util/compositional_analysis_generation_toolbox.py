""""
Toolbox for simulating compositional data from ScRNA-seq

This toolbox provides data generation and modelling solutions for compositional data with different specifications.
This data might e.g. come from scRNA-seq experiments.
For scenarios 1-4, we first generate composition parameters (b_true, w_true) and a covariance matrix (x) from some input specifications.
We then build a concentration vector for each sample (row of x) that sums up to 1. From there, we can calculate each row of the cell count matrix (y) via a multinomial distribution

:authors: Johannes Ostner
"""

# TODO: Extensive introduction into data format

# Setup

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import anndata as ad
import pandas as pd

import tensorflow_probability as tfp
import warnings
from scipy.special import softmax

plt.style.use("ggplot")
warnings.filterwarnings('ignore')

tfd = tfp.distributions
tfb = tfp.bijectors
plt.style.use('ggplot')


# **Requirements:**
# * Python == 3.7
# * Tensorflow == 2.0
# * Tensorflow probability == 0.8.0

# ## Scenario 1: Normally distributed, independent covariates
# Default Settings
D = 4  # number of dimensions
N = 100  # number of datapoints to generate
K = 5 # number of cell types
n_total = [1000]*N
noise_std_true = 1.0

def generate_normal_uncorrelated (N, D, K, n_total, noise_std_true):
    """
    Scenario 1: Normally distributed, independent covariates
    :param N: Number of samples
    :param D: Number of covariates
    :param K: Number of cell types
    :param n_total: Number of individual cells per sample
    :param noise_std_true: noise level. 0: No noise

    :return: Anndata object
    """

    # Generate random composition parameters
    b_true = np.random.normal(0,1, size=K).astype(np.float32)  # bias (alpha)
    w_true = np.random.normal(0, 1, size=(D, K)).astype(np.float32)  # weights (beta)

    # Generate random covariate matrix
    x = np.random.normal(0, 1, size=(N,D)).astype(np.float32)
    noise = noise_std_true * np.random.randn(N, 1).astype(np.float32)

    # Generate y
    y = np.zeros([N, K], dtype=np.float32)
    for i in range(N):
        # Concentration should sum to 1 for each sample
        concentration = softmax(x[i,:].T@w_true + b_true + noise[i,:]).astype(np.float32)
        y[i, :] = np.random.multinomial(n_total[i], concentration).astype(np.float32)

    x_names = ["x_" + str(n) for n in range(x.shape[1])]
    x_df = pd.DataFrame(x, columns=x_names)

    data = ad.AnnData(X=y, obs=x_df, uns={"b_true": b_true, "w_true": w_true})

    return data


# ## Scenario 2: Correlated covariates
# Default Settings
covariate_mean = np.zeros(shape=(D))
covariate_var = np.identity(D)

def generate_normal_correlated (N, D, K, n_total, noise_std_true, covariate_mean = np.zeros(shape=(D)), covariate_var=None):
    """
    Scenario 2: Correlated covariates
    :param N: Number of samples
    :param D: Number of covariates
    :param K: Number of cell types
    :param n_total: Number of individual cells per sample
    :param noise_std_true: noise level. 0: No noise
    :param covariate_mean: Mean of each covariate
    :param covariate_var: Variance matrix for all covaraiates
    :return: Anndata object
    """

    # Generate randomized covariate covariance matrix if none is specified
    if covariate_var is None:
        # Covariates drawn from MvNormal(0, Cov), Cov_ij = p ^|i-j| , p=0.4
        # Tibshirani for correlated covariates: Tibshirani (1996)
        p = 0.4
        covariate_var = np.zeros((D,D))
        for i in range(D):
            for j in range(D):
                covariate_var[i,j] = p**np.abs(i-j)

    # Generate random composition parameters
    b_true = np.random.normal(0,1, size=K).astype(np.float32)  # bias (alpha)
    w_true = np.random.normal(0, 1, size=(D, K)).astype(np.float32)  # weights (beta)

    # Generate random covariate matrix
    x = np.random.multivariate_normal(size = N, mean=covariate_mean, cov=covariate_var).astype(np.float32)
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


# ## Scenario 3: Correlated cell types

def generate_normal_xy_correlated (N, D, K, n_total, noise_std_true=1, covariate_mean = np.zeros(shape=(D)), covariate_var=None, sigma=np.identity(K)):
    """
        Scenario 3: Correlated cell types and covariates
        :param N: Number of samples
        :param D: Number of covariates
        :param K: Number of cell types
        :param n_total: Number of individual cells per sample
        :param noise_std_true: noise level. 0: No noise
        :param covariate_mean: Mean of each covariate
        :param covariate_var: Variance matrix for all covaraiates
        :param sigma: correlation matrix for cell types - array[K,K]
        :return: Anndata object
        """

    # Generate randomized covariate covariance matrix if none is specified
    if covariate_var is None:
        # Covaraits drawn from MvNormal(0, Cov) Cov_ij = p ^|i-j| , p=0.4
        # Tibshirani for correlated covariates: Tibshirani (1996)
        p = 0.4
        covariate_var = np.zeros((D,D))
        for i in range(D):
            for j in range(D):
                covariate_var[i,j] = p**np.abs(i-j)

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
        alpha = np.random.multivariate_normal(mean = x[i,:].T@w_true + b_true, cov = sigma*noise[i,:]).astype(np.float32)
        concentration = softmax(alpha).astype(np.float32)
        y[i, :] = np.random.multinomial(n_total[i], concentration).astype(np.float32)

    x_names = ["x_" + str(n) for n in range(x.shape[1])]
    x_df = pd.DataFrame(x, columns=x_names)

    data = ad.AnnData(X=y, obs=x_df, uns={"b_true": b_true, "w_true": w_true})

    return data


# ## Scenario 4: Sparse true parameters
# Now we tend to effects that have most entries at 0 (only few interactions between covariates and cell types)

# Generates a sparse effect matrix w_true
def sparse_effect_matrix (D,K, n_d, n_k):
    """

    :param D: Number of covariates
    :param K: Number of cell types
    :param n_d: Number of covariates that effect a cell type
    :param n_k: Number of cell types that are affected by any covariate
    :return: w_true: Effect matrix
    """

    # Choose indices of affected cell types and covariates randomly
    d_eff = np.random.choice(range(D), size=n_d, replace=False)
    k_eff = np.random.choice(range(K), size=n_k, replace=False)

    # Possible entries of w_true
    w_choice = [0.3, 0.5, 1]
    
    w_true = np.zeros((D,K))
    # Fill in w_true
    for i in d_eff:
        for j in k_eff:
            c = np.random.choice(3, 1)
            w_true[i,j] = w_choice[c]
            
    return w_true
    

def generate_sparse_xy_correlated (N, D, K, n_total, noise_std_true=1,
                                    covariate_mean = np.zeros(shape=(D)), covariate_var=None,
                                    sigma=np.identity(K),
                                    b_true=None, w_true=None):

    """
    Scenario 4: Sparse true parameters
    :param N: Number of samples
    :param D: Number of covariates
    :param K: Number of cell types
    :param n_total: Number of individual cells per sample
    :param noise_std_true: noise level. 0: No noise
    :param covariate_mean: Mean of each covariate
    :param covariate_var: Variance matrix for all covaraiates
    :param sigma: correlation matrix for cell types - array[K,K]
    :param b_true: bias coefficients
    :param w_true: Effect matrix
    :return: Anndata object
    """

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


# ## Scenario 5: Uniform/Skewed cell composition
# 
# We switch from regression to case control:
# Take up to 4 cases, and corresponding weight vectors for each case combination

# Calculates the binary representation of an int
def binary(x, cases):
    return [int(i) for i in bin(x)[2:].zfill(cases)]

def generate_case_control (cases = 1, K = 5, n_total = 1000, n_samples = [5,5], noise_std_true=0,
                                    sigma=None,
                                    b_true=None, w_true=None):
    """
    Generates compositional data with b inary covariates
    :param cases: Number of cases/covariates
    :param K: Number of cell types
    :param n_total: Number of individual cells per sample
    :param n_samples: array[2**cases] - Number of samples per case combination
    :param noise_std_true: noise_std_true: noise level. 0: No noise - Not in use atm!!!
    :param sigma: correlation matrix for cell types - array[K,K]
    :param b_true: bias coefficients
    :param w_true: Effect matrix
    :return: Anndata object
    """

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


    noise = noise_std_true * np.random.randn(N, 1).astype(np.float32)

    # Initialize x, y
    x = np.zeros((sum(n_samples), cases))
    y = np.zeros((sum(n_samples), K)) 
    c=0

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
            z=np.random.multinomial(n_total, concentration)
            y[c+j] = z
        c=c+n_samples[i]

    x=x.astype(np.float32)
    y=y.astype(np.float32)

    x_names = ["x_" + str(n) for n in range(x.shape[1])]
    x_df = pd.DataFrame(x, columns=x_names)

    data = ad.AnnData(X=y, obs=x_df, uns={"b_true": b_true, "w_true": w_true})

    return data

#%%

def b_w_from_abs_change(counts_before=np.array([200, 200, 200, 200, 200]), abs_change=50, n_total=1000):
    """
    Calculates intercepts and slopes from a starting count and an absolute change for the first cell type
    :param counts_before: array[K] - cell counts for control samples
    :param abs_change: int - change of first cell type in terms of cell counts
    :param n_total: number of cells per sample. This stays constant over all samples!!!
    :return: b: intercepts
        w: slopes
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

# Generate a K-dim count vector b with b[0] ans sum(b) specifed, all other entries are the same
def counts_from_first(b_0 = 200, n_total = 1000, K = 5):
    b = np.repeat((n_total-b_0)/(K-1), K)
    b[0] = b_0
    return b
