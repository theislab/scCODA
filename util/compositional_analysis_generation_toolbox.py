# # Toolbox for simulating compositional data from ScRNA-seq
# This toolbox provides data generation and modelling solutions for compositional data with different specifications. This data might e.g. come from scRNA-seq experiments.

# Setup

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

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
# * Tensorflow 2.0
# * Tensorflow probability

# ## Scenario 1: Normally distributed, independent covariates
# Default Settings
D = 4  # number of dimensions
N = 100  # number of datapoints to generate
K = 5
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

    :return: b_true: bias coefficients
    w_true: weight cefficients
    x: Data set that can be used for simulation
    y: Corresponding compositional data
    """
    
    b_true = np.random.normal(0,1, size=K).astype(np.float32)  # bias (alpha)
    w_true = np.random.normal(0, 1, size=(D, K)).astype(np.float32)  # weights (beta)
    x = np.random.normal(0, 1, size=(N,D)).astype(np.float32)
    noise = noise_std_true * np.random.randn(N, 1).astype(np.float32)
    # concentration should sum to one
    y = np.zeros([N, K], dtype=np.float32)
    for i in range(N):
        concentration = softmax(x[i,:].T@w_true + b_true + noise[i,:]).astype(np.float32)
        y[i, :] = np.random.multinomial(n_total[i], concentration).astype(np.float32)
        
    return b_true, w_true, x, y


# ## Scenario 2: Correlated covariates
# Settings
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
    :return: b_true: bias coefficients
    w_true: weight cefficients
    x: Data set that can be used for simulation
    y: Corresponding compositional data
    """

    if covariate_var is None:
        # Covariates drawn from MvNormal(0, Cov), Cov_ij = p ^|i-j| , p=0.4
        # Tibshirani for correlated covariats Tibshirani (1996)
        p = 0.4
        covariate_var = np.zeros((D,D))
        for i in range(D):
            for j in range(D):
                covariate_var[i,j] = p**np.abs(i-j)
                
    b_true = np.random.normal(0,1, size=K).astype(np.float32)  # bias (alpha)
    w_true = np.random.normal(0, 1, size=(D, K)).astype(np.float32)  # weights (beta)
    x = np.random.multivariate_normal(size = N, mean=covariate_mean, cov=covariate_var).astype(np.float32)
    noise = noise_std_true * np.random.randn(N, 1).astype(np.float32)
    # concentration should sum to one
    y = np.zeros([N, K], dtype=np.float32)
    for i in range(N):
        concentration = softmax(x[i,:].T@w_true + b_true + noise[i,:]).astype(np.float32)
        y[i, :] = np.random.multinomial(n_total[i], concentration).astype(np.float32)
        
    return b_true, w_true, x, y


# ## Scenario 3: Correlated cell types

def generate_normal_xy_correlated (N, D, K, n_total, noise_std_true=1, covariate_mean = np.zeros(shape=(D)), covariate_var=None, sigma=np.identity(K)):
    
    '''Parameters:
    N: Number of samples
    D: Number of covariates
    K: Number of cell types
    n_total: Number of individual cells per sample
    noise_std_true: noise level. 0: No noise
    covariate_mean: means of covariates - vector[D]
    covariate_var: Variance of covariates - array[D,D]
    sigma: correlation matrix for cell types - array[K,K]
    
    Returns:
    b_true: bias coefficients
    w_true: weight cefficients
    x: Data set that can be used for simulation
    y: Corresponding compositional data'''
    
    if covariate_var is None:
        # Covaraits drawn from MvNormal(0, Cov) Cov_ij = p ^|i-j| , p=0.4
        # Tibshirani for correlated covariats Tibshirani (1996)
        p = 0.4
        covariate_var = np.zeros((D,D))
        for i in range(D):
            for j in range(D):
                covariate_var[i,j] = p**np.abs(i-j)
                
    b_true = np.random.normal(0,1, size=K).astype(np.float32)  # bias (alpha)
    w_true = np.random.normal(0, 1, size=(D, K)).astype(np.float32)  # weights (beta)
    x = np.random.multivariate_normal(size = N, mean=covariate_mean, cov=covariate_var).astype(np.float32)
    noise = noise_std_true * np.random.randn(N, 1).astype(np.float32)
    # concentration should sum to one
    y = np.zeros([N, K], dtype=np.float32)
    for i in range(N):
        alpha = np.random.multivariate_normal(mean = x[i,:].T@w_true + b_true, cov = sigma*noise[i,:]).astype(np.float32)
        concentration = softmax(alpha).astype(np.float32)
        y[i, :] = np.random.multinomial(n_total[i], concentration).astype(np.float32)
        
    return b_true, w_true, x, y


# ## Scenario 4: Sparse true parameters

def sparse_effect_matrix (D,K, n_d, n_k):
    d_eff = np.random.choice(range(D), size=n_d, replace=False)
    k_eff = np.random.choice(range(K), size=n_k, replace=False)
    w_choice = [0.3, 0.5, 1]
    
    w_true = np.zeros((D,K))
    for i in d_eff:
        for j in k_eff:
            c = np.random.choice(3, 1)
            w_true[i,j] = w_choice[c]
            
    return w_true
    

def generate_sparse_xy_correlated (N, D, K, n_total, noise_std_true=1,
                                    covariate_mean = np.zeros(shape=(D)), covariate_var=None,
                                    sigma=np.identity(K),
                                    b_true=None, w_true=None):
    
    '''Parameters:
    N: Number of samples
    D: Number of covariates
    K: Number of cell types
    n_total: Number of individual cells per sample
    noise_std_true: noise level. 0: No noise
    covariate_mean: means of covariates - vector[D]
    covariate_var: Variance of covariates - array[D,D]
    sigma: correlation matrix for cell types - array[K,K]
    b_true: Vector of true intercepts - array[D]
    w_true: True matrix of effects - array[D,K]
    
    Returns:
    b_true: bias coefficients
    w_true: weight cefficients
    x: Data set that can be used for simulation
    y: Corresponding compositional data'''
    
    if covariate_var is None:
        # Covaraits drawn from MvNormal(0, Cov) Cov_ij = p ^|i-j| , p=0.4
        # Tibshirani for correlated covariats Tibshirani (1996)
        p = 0.4
        covariate_var = np.zeros((D,D))
        for i in range(D):
            for j in range(D):
                covariate_var[i,j] = p**np.abs(i-j)
                
    if b_true is None:
        # Uniform intercepts
        b_true = np.random.uniform(-3,3, size=K).astype(np.float32)  # bias (alpha)
        
        
    if w_true is None:
        # Randomly select covariates that should correlate
        n_d = np.random.choice(range(D), size=1)
        n_k = np.random.choice(range(K), size=1)
        w_true = sparse_effect_matrix(D, K, n_d, n_k)  
    
    x = np.random.multivariate_normal(size = N, mean=covariate_mean, cov=covariate_var).astype(np.float32)
    noise = noise_std_true * np.random.randn(N, 1).astype(np.float32)
    # concentration should sum to one
    y = np.zeros([N, K], dtype=np.float32)
    for i in range(N):
        alpha = np.random.multivariate_normal(mean = x[i,:].T@w_true + b_true, cov = sigma*noise[i,:]).astype(np.float32)
        concentration = softmax(alpha).astype(np.float32)
        y[i, :] = np.random.multinomial(n_total[i], concentration).astype(np.float32)
        
    return b_true, w_true, x, y


# ## Scenario 5: Uniform/Skewed cell composition
# 
# We switch from regression to case control:
# Take up to 4 cases, and corresponding weight vectors for each case combination
#cases = 1
#K = 5
#n_samples = [5,5]
#p_samples = [[0.2, 0.2, 0.2, 0.2, 0.2], [0.7,0.3, 0, 0, 0]]
#n_total=500
#%%

def binary(x, cases):
    return [int(i) for i in bin(x)[2:].zfill(cases)]

def generate_case_control (cases, K, n_total, n_samples, noise_std_true=0,
                                    sigma=None,
                                    b_true=None, w_true=None):
    
    '''Parameters:
    cases: Number of cases/covariates
    K: Number of cell types
    n_total: Number of individual cells per sample
    n_samples: Number of samples per case combination - array[2**cases]
    p_samples: probability distribtions for all case combinations - array[2**cases, K]
    
    The order of combinations in n_samples and p_samples is binary: 000, 001, 010, 011, 100, ...
    
    Returns:
    x: Data set that can be used for simulation
    y: Corresponding compositional data'''

    if b_true is None:
        # Uniform intercepts
        b_true = np.random.uniform(-3, 3, size=K).astype(np.float32)  # bias (alpha)


    if w_true is None:
        # Randomly select covariates that should correlate
        n_d = np.random.choice(range(cases+1), size=1)
        n_k = np.random.choice(range(K+1), size=1)
        w_true = sparse_effect_matrix(cases, K, n_d, n_k)

    if sigma is None:
        sigma = np.identity(K) * 0.1


    noise = noise_std_true * np.random.randn(N, 1).astype(np.float32)
            
    x = np.zeros((sum(n_samples), cases))
    y = np.zeros((sum(n_samples), K)) 
    c=0

    for i in range(2**cases):
        for j in range(n_samples[i]):
            x[c+j] = binary(i, cases)

            alpha = np.random.multivariate_normal(mean=x[c+j, :].T @ w_true + b_true, cov=sigma).astype(
                np.float32)

            concentration = softmax(alpha).astype(np.float32)
            z=np.random.multinomial(n_total, concentration)
            y[c+j] = z
        c=c+n_samples[i]

    x=x.astype(np.float32)
    y=y.astype(np.float32)

    return x, y, b_true, w_true

#%%

def b_w_from_abs_change(counts_before=np.array([200, 200, 200, 200, 200]), abs_change=50, n_total=1000):
    K = counts_before.shape[0]
    b = np.log(counts_before / n_total)

    count_0_after = counts_before[0] + abs_change
    count_other_after = (n_total - count_0_after) / (K - 1)
    counts_after = np.repeat(count_other_after, K)
    counts_after[0] = count_0_after

    b_after = np.log(counts_after / n_total)

    w = b_after - b
    w = w - w[K - 1]

    return b, w

#%%
def counts_from_first(b_0 = 200, n_total = 1000, K = 5):
    b = np.repeat((n_total-b_0)/(K-1), K)
    b[0] = b_0
    return b

#%%
'''
cases = 1
K = 5
n_samples = [5,5]
n_total=1000

x, y, b_true, w_true = generate_case_control(cases, K, n_total, n_samples)

print(w_true)

print(y)

'''