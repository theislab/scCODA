
""""
This file defines multiple Dirichlet-multinomial models
for statistical analysis of compositional changes
For further reference, see:
Johannes Ostner: Development of a statistical framework for compositional analysis of single-cell data

:authors: Johannes Ostner
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import matplotlib.pyplot as plt
import numpy as np
import time

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.experimental import edward2 as ed
import seaborn as sns

tfd = tfp.distributions
tfb = tfp.bijectors

from util import compositional_analysis_generation_toolbox as gen
from util import result_classes as res


class dir_mult_logit_normal_noncentered_model_new:
    """"
    implements statistical model and
    test statistics for compositional differential change analysis
    without specification of a baseline cell type
    """

    def __init__(self, x, y, n_total, dtype=tf.float32):
        """
        Constructor of model class
        :param x: numpy array [NxD] - covariate matrix
        :param y: numpy array [NxK] - cell count matrix
        :param n_total: numpy array [N] - number of cells per sample
        :param dtype: data type for all numbers (for tensorflow)
        """
        self.x = tf.cast(x, dtype)
        self.y = tf.cast(y, dtype)
        self.n_total = tf.cast(n_total, dtype)
        self.dtype = dtype

        # Get dimensions of data
        N, D = x.shape
        K = y.shape[1]

        # Check input data
        if N != y.shape[0]:
            raise ValueError("Wrong input dimensions X[{},:] != y[{},:]".format(x.shape[0], y.shape[0]))

        if N != len(n_total):
            raise ValueError("Wrong input dimensions X[{},:] != n_total[{}]".format(x.shape[0], len(n_total)))

        def define_model(x,n_total, K):
            """
            Model definition in Edward2
            :param x: numpy array [NxD] - covariate matrix
            :param n_total: numpy array [N] - number of cells per sample
            :param K: Number of cell types
            :return: none
            """
            N,D = x.shape

            # normal prior on bias
            alpha = ed.Normal(loc=tf.zeros([K]), scale=tf.ones([K])*5, name="alpha")

            # Noncentered parametrization for raw intercepts (before spike-and-slab)
            mu_b = ed.Normal(loc=tf.zeros(1, dtype=dtype), scale=tf.ones(1, dtype=dtype), name="mu_b")
            sigma_b = ed.HalfCauchy(tf.zeros(1, dtype=dtype), tf.ones(1, dtype=dtype), name="sigma_b")
            b_offset = ed.Normal(loc=tf.zeros([D, K], dtype=dtype), scale=tf.ones([D, K], dtype=dtype), name="b_offset")

            b_raw = mu_b + sigma_b * b_offset

            # Spike-and-slab priors
            sigma_ind_raw = ed.Normal(
                loc=tf.zeros(shape=[D, K], dtype=dtype),
                scale=tf.ones(shape=[D, K], dtype=dtype),
                name='sigma_ind_raw')
            ind_t = sigma_ind_raw*50
            ind = tf.exp(ind_t) / (1 + tf.exp(ind_t))

            # Calculate betas
            beta = ind * b_raw

            # Concentration vector from intercepts, slopes
            concentration_ = tf.exp(alpha + tf.matmul(x, beta))
            # Cell count prediction via DirMult
            predictions = ed.DirichletMultinomial(n_total, concentration=concentration_, name="predictions")
            return predictions

        # Joint posterior distribution
        self.log_joint = ed.make_log_joint_fn(define_model)

        # Function to compute log posterior probability
        self.target_log_prob_fn = lambda alpha_, mu_b_,\
                                         sigma_b_, b_offset_, sigma_ind_raw_: self.log_joint(x=self.x,
                                                                            n_total=self.n_total,
                                                                            K=K,
                                                                            predictions=self.y,
                                                                            alpha=alpha_,
                                                                            mu_b=mu_b_,
                                                                            sigma_b=sigma_b_,
                                                                            b_offset=b_offset_,
                                                                            sigma_ind_raw=sigma_ind_raw_,
#                                                                            mu_ind_raw=mu_ind_raw_,
                                                                            )

        alpha_size = [K]
        beta_size = [D, K]

        # MCMC starting values
        self.params = [tf.random.normal(alpha_size, 0, 1, name='init_alpha'),
                       tf.zeros(1, name="init_mu_b", dtype = dtype),
                       tf.ones(1, name="init_sigma_b", dtype=dtype),
                       tf.random.normal(beta_size, 0, 1, name='init_b_offset'),
                       tf.zeros(beta_size, name='init_sigma_ind_raw', dtype=dtype),
                       ]

        self.vars = [tf.Variable(v, trainable=True) for v in self.params]



    def sample(self, num_results=int(10e3), n_burnin=int(5e3), num_leapfrog_steps=10, step_size = 0.01):
        """
        HMC sampling of the model

        :param n_iterations: number of HMC iterations
        :param n_burnin: number of burn-in iterations
        :param num_leapfrog_steps: number of leap-frog steps per iteration
        :param step_size: Initial step size
        :return: dict of parameters
        """
        N,D = self.x.shape
        K = self.y.shape[1]

        alpha_size = [K]
        beta_size = [D, K]

        # All parameters that are returned for analysis
        param_names = ["alpha", "mu_b", "sigma_b", "b_offset", "sigma_ind_raw", "beta"]
        init_state = self.params

        # constraints (not in use atm)
        constraining_bijectors = [
            tfb.Identity(),
            tfb.Identity(),
            tfb.Identity(),
            tfb.Identity(),
            tfb.Identity(),
        ]

        # HMC transition kernel
        kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self.target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps)
        kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=kernel, bijector=constraining_bijectors)
        kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=kernel, num_adaptation_steps=int(4000), target_accept_prob=0.9)

        # HMC sampling function
        @tf.function
        def sample_mcmc(num_results, n_burnin, kernel, current_state):
            return tfp.mcmc.sample_chain(
                num_results=num_results,
                num_burnin_steps=n_burnin,
                kernel=kernel,
                current_state=current_state,
                trace_fn=lambda _, pkr: [pkr.inner_results.inner_results.is_accepted,
                                         pkr.inner_results.inner_results.accepted_results.step_size])

        # HMC sampling process
        start = time.time()
        states, kernel_results = sample_mcmc(num_results, n_burnin, kernel, init_state)
        duration = time.time() - start
        print("MCMC sampling finished. ({:.3f} sec)".format(duration))

        # Re-calculation of some values (beta) and application of burnin
        def get_chains_after_burnin(samples, accept, n_burnin):
            # Samples after burn-in
            states_burnin = []
            acceptances = accept[0].numpy()
            accepted = acceptances[acceptances == True]

            for s in samples:
                states_burnin.append(s[n_burnin:])

            # acceptance rate
            p_accept = accepted.shape[0] / acceptances.shape[0]
            print('Acceptance rate: %0.1f%%' % (100 * p_accept))

            return states_burnin

        states_burnin = get_chains_after_burnin(states, kernel_results, n_burnin)

        # Calculate predicted cell counts (for analysis purposes)
        def get_y_hat(states_burnin):
            alphas_final = states_burnin[0].numpy().mean(axis=0)

            ind_raw = states_burnin[4].numpy()*50
            ind = np.exp(ind_raw) / (1 + np.exp(ind_raw))

            b_raw = np.array([states_burnin[1].numpy()[i] + (states_burnin[2].numpy()[i]*states_burnin[3].numpy()[i])
                              for i in range(num_results - n_burnin)])

            betas = ind * b_raw
            betas_final = betas.mean(axis=0)

            states_burnin.append(betas)

            return ed.DirichletMultinomial(self.n_total,
                                           concentration=tf.exp(tf.matmul(self.x, betas_final) + alphas_final)).numpy()

        y_hat = get_y_hat(states_burnin)

        return res.MCMCResult(int(self.x.shape[0]), dict(zip(param_names, states_burnin)), y_hat, self.y.numpy(), spike_slab=True)

#%%

class dir_mult_logit_normal_noncentered_model_baseline_new:
    """"
    implements statistical model and
    test statistics for compositional differential change analysis

    """

    def __init__(self, x, y, n_total, baseline_index, dtype=tf.float32):
        """

        :param X: Numpy Design NxD matrix of independent variables of interest
        :param Z: Numpy Design NxB matrix of confounders
        :param y: Numpy NxK Matrix of dependent variables
        :param n_total: Numpy Nx1 Vector of total observed counts
        """
        self.x = tf.cast(x, dtype)
        self.y = tf.cast(y, dtype)
        self.n_total = tf.cast(n_total, dtype)
        self.dtype = dtype
        self.baseline_index = baseline_index

        N, D = x.shape
        K = y.shape[1]

        # Check input data
        if N != y.shape[0]:
            raise ValueError("Wrong input dimensions X[{},:] != y[{},:]".format(x.shape[0], y.shape[0]))

        if N != len(n_total):
            raise ValueError("Wrong input dimensions X[{},:] != n_total[{}]".format(x.shape[0], len(n_total)))

        # Model definition
        def define_model(x,n_total, K):
            N,D = x.shape

            alpha = ed.Normal(  # normal prior on bias
                loc=tf.zeros([K]),
                scale=tf.ones([K])*5,
                name="alpha")

            mu_b = ed.Normal(loc=tf.zeros(1, dtype=dtype), scale=tf.ones(1, dtype=dtype), name="mu_b")
            sigma_b = ed.HalfCauchy(tf.zeros(1, dtype=dtype), tf.ones(1, dtype=dtype), name="sigma_b")
            b_offset = ed.Normal(loc=tf.zeros([D, K-1], dtype=dtype), scale=tf.ones([D, K-1], dtype=dtype), name="b_offset")

            #b_offset = b_offset+b_baseline

            b_raw = mu_b + sigma_b * b_offset
            b_raw = tf.concat(axis=1, values=[b_raw[:, :baseline_index],
                                                   tf.fill(value=0., dims=[D, 1]),
                                                   b_raw[:, baseline_index:]])
            #b_raw[:, baseline_index] = 0
            # b_raw = mu_b + (sigma_b*b_offset)

            sigma_ind_raw = ed.Normal(
                loc=tf.zeros(shape=[D, K], dtype=dtype),
                scale=tf.ones(shape=[D, K], dtype=dtype),
                name='sigma_ind_raw')

            ind_t = sigma_ind_raw * 50

            ind = tf.exp(ind_t) / (1 + tf.exp(ind_t))

            beta = ind * b_raw

            #beta = beta - beta[:, baseline_index]

            concentration_ = tf.exp(alpha + tf.matmul(x, beta))

            predictions = ed.DirichletMultinomial(n_total, concentration=concentration_, name="predictions")
            return predictions

        # Joint posterior distribution
        self.log_joint = ed.make_log_joint_fn(define_model)
        # Function to compute log posterior probability

        self.target_log_prob_fn = lambda alpha_, mu_b_,\
                                         sigma_b_, b_offset_, sigma_ind_raw_: self.log_joint(x=self.x,
                                                                            n_total=self.n_total,
                                                                            K=K,
                                                                            predictions=self.y,
                                                                            alpha=alpha_,
                                                                            mu_b=mu_b_,
                                                                            sigma_b = sigma_b_,
                                                                            b_offset = b_offset_,
                                                                            sigma_ind_raw = sigma_ind_raw_,
                                                                            )

        alpha_size = [K]
        beta_size = [D, K]

        self.params = [tf.random.normal(alpha_size, 0, 1, name='init_alpha'),
                       tf.zeros(1, name="init_mu_b", dtype = dtype),
                       tf.ones(1, name="init_sigma_b", dtype=dtype),
                       tf.random.normal([D, K-1], 0, 1, name='init_b_offset'),
                       #tf.random.normal(beta_size, 0,1, name='init_ind_raw'),
                       tf.zeros(beta_size, name='init_sigma_ind_raw', dtype=dtype),
                       ]

        self.vars = [tf.Variable(v, trainable=True) for v in self.params]



    def sample(self, num_results=int(10e3), n_burnin=int(5e3), num_leapfrog_steps=10, step_size = 0.01):
        """
        HMC sampling of the model

        :param n_iterations: number of HMC iterations
        :param n_burning: number of burn-in iterations
        :param n_leapfrog: number of leap-frog steps per iteration
        :param n_chains: number of MCMC chains (current supports only sampling from one chain)
        :return: dict of parameters
        """
        # TODO: Add support for multi-chain sampling (currently Jan 30th 2019 not supported for Edward2 models)
        N,D = self.x.shape
        K = self.y.shape[1]

        alpha_size = [K]
        beta_size = [D, K]

        param_names = ["alpha", "mu_b", "sigma_b", "b_offset", "ind_raw", "beta"]
        init_state = self.params

        constraining_bijectors = [
            tfb.Identity(),
            tfb.Identity(),
            tfb.Identity(),
            tfb.Identity(),
            tfb.Identity(),
        ]

        # HMC transition kernel
        kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self.target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps)
        kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=kernel, bijector=constraining_bijectors)
        kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=kernel, num_adaptation_steps=int(4000), target_accept_prob=0.9)

        @tf.function
        def sample_mcmc(num_results, n_burnin, kernel, current_state):
            return tfp.mcmc.sample_chain(
                num_results=num_results,
                num_burnin_steps=n_burnin,
                kernel=kernel,
                current_state=current_state,
                trace_fn=lambda _, pkr: [pkr.inner_results.inner_results.is_accepted,
                                         pkr.inner_results.inner_results.accepted_results.step_size])

        start = time.time()
        states, kernel_results = sample_mcmc(num_results, n_burnin, kernel, init_state)
        duration = time.time() - start
        print("MCMC sampling finished. ({:.3f} sec)".format(duration))

        def get_chains_after_burnin(samples, accept, n_burnin):
            # Samples after burn-in
            states_burnin = []
            acceptances = accept[0].numpy()
            accepted = acceptances[acceptances == True]

            for s in samples:
                states_burnin.append(s[n_burnin:])
            p_accept = accepted.shape[0] / acceptances.shape[0]

            print('Acceptance rate: %0.1f%%' % (100 * p_accept))


            return states_burnin

        states_burnin = get_chains_after_burnin(states, kernel_results, n_burnin)

        def get_y_hat(states_burnin):
            alphas_final = states_burnin[0].numpy().mean(axis=0)

            ind_raw = states_burnin[4].numpy() * 50
            ind = np.exp(ind_raw) / (1 + np.exp(ind_raw))

            b_raw_o = np.array([states_burnin[1].numpy()[i] + (states_burnin[2].numpy()[i]*states_burnin[3].numpy()[i])
                              for i in range(num_results - n_burnin)])

            b_raw = []

            for i in range(b_raw_o.shape[0]):
                b = b_raw_o[i, :, :]
                b_o = np.concatenate([b[:, :self.baseline_index],
                                      np.zeros(shape=[b.shape[0], 1]),
                                      b[:, self.baseline_index:]], axis=1)
                b_raw.append(b_o)
            b_raw = np.array(b_raw).astype("float32")

            betas = ind * b_raw
            #betas = np.array([betas[i] - betas[i, :, self.baseline_index] for i in range(betas.shape[0])])
            betas_final = betas.mean(axis=0)

            states_burnin.append(betas)

            return ed.DirichletMultinomial(self.n_total,
                                           concentration=tf.exp(tf.matmul(self.x, betas_final) + alphas_final)).numpy()

        y_hat = get_y_hat(states_burnin)

        return res.MCMCResult(int(self.x.shape[0]), dict(zip(param_names, states_burnin)), y_hat, self.y.numpy(), spike_slab=True)


