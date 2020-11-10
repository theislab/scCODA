import numpy as np
import time

import tensorflow as tf
import tensorflow_probability as tfp
from sccoda.model.dirichlet_models import CompositionalModel

tfd = tfp.distributions
tfb = tfp.bijectors

#%%


class NoBaselineModelTime(CompositionalModel):

    """
    WIP! DO NOT USE!!!

    implements statistical model and
    test statistics for compositional differential change analysis
    without specification of a baseline cell type
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor of model class

        :param covariate_matrix: numpy array [NxD] - covariate matrix
        :param data_matrix: numpy array [NxK] - cell count matrix
        :param sample_counts: numpy array [N] - number of cells per sample
        :param dtype: data type for all numbers (for tensorflow)

        :return
        NotImplementedError
        """

        super(self.__class__, self).__init__(*args, **kwargs)

        self.baseline_index = None
        dtype = tf.float32

        # All parameters that are returned for analysis
        self.param_names = ["mu_b", "sigma_b", "b_offset", "ind_raw", "alpha", "phi",
                            "ind", "b_raw", "beta"]

        alpha_size = [self.K]
        beta_size = [self.D, self.K]

        Root = tfd.JointDistributionCoroutine.Root

        def model():
            mu_b = yield Root(tfd.Independent(
                tfd.Normal(loc=tf.zeros(1, dtype=dtype),
                           scale=tf.ones(1, dtype=dtype),
                           name="mu_b"),
                reinterpreted_batch_ndims=1))

            sigma_b = yield Root(tfd.Independent(
                tfd.HalfCauchy(tf.zeros(1, dtype=dtype),
                               tf.ones(1, dtype=dtype),
                               name="sigma_b"),
                reinterpreted_batch_ndims=1))

            b_offset = yield Root(tfd.Independent(
                tfd.Normal(
                    loc=tf.zeros(beta_size, dtype=dtype),
                    scale=tf.ones(beta_size, dtype=dtype),
                    name="b_offset"),
                reinterpreted_batch_ndims=2))

            # Spike-and-slab
            ind_raw = yield Root(tfd.Independent(
                tfd.Normal(
                    loc=tf.zeros(shape=beta_size, dtype=dtype),
                    scale=tf.ones(shape=beta_size, dtype=dtype),
                    name='ind_raw'),
                reinterpreted_batch_ndims=2))

            ind = tf.exp(ind_raw * 50) / (1 + tf.exp(ind_raw * 50))
            b_raw = mu_b + sigma_b * b_offset
            beta = ind * b_raw

            alpha = yield Root(tfd.Independent(
                tfd.Normal(
                    loc=tf.zeros(alpha_size),
                    scale=tf.ones(alpha_size) * 5,
                    name="alpha"),
                reinterpreted_batch_ndims=1))

            phi = yield Root(tfd.Independent(
                tfd.Normal(tf.ones(beta_size, dtype=dtype),
                           tf.ones(beta_size, dtype=dtype) * 0.1,
                           name="phi"),
                reinterpreted_batch_ndims=2))

            phi_ = tf.repeat(phi[tf.newaxis, :], self.N, axis=0)
            b_time = tf.pow(phi_, self.time_matrix[:, tf.newaxis, tf.newaxis])
            b_ = beta[tf.newaxis, :, :] * b_time

            c_ = self.x[:, :, tf.newaxis] * b_

            concentrations = tf.exp(alpha + tf.reduce_sum(c_, axis=1))

            # Cell count prediction via DirMult
            predictions = yield Root(tfd.Independent(
                tfd.DirichletMultinomial(
                    total_count=tf.cast(self.n_total, dtype),
                    concentration=concentrations,
                    name="predictions"),
                reinterpreted_batch_ndims=1))

        self.model_struct = tfd.JointDistributionCoroutine(model)

        # Joint posterior distribution
        self.target_log_prob_fn = lambda *args:\
            self.model_struct.log_prob(list(args) + [tf.cast(self.y, dtype)])

        # MCMC starting values
        self.params = [tf.zeros(1, name="init_mu_b", dtype=dtype),
                       tf.ones(1, name="init_sigma_b", dtype=dtype),
                       tf.zeros(beta_size, name='init_b_offset', dtype=dtype),
                       tf.zeros(beta_size, name='init_ind_raw', dtype=dtype),
                       tf.zeros(alpha_size, name='init_alpha', dtype=dtype),
                       tf.ones(beta_size, name="init_phi", dtype=dtype),
                       ]

        # self.vars = [tf.Variable(v, trainable=True) for v in self.params]

    # Calculate predicted cell counts (for analysis purposes)
    def get_y_hat(self, states_burnin, num_results, n_burnin):
        chain_size_y = [num_results - n_burnin, self.N, self.K]

        alphas = states_burnin[4]
        alphas_final = alphas.mean(axis=0)

        ind_raw = states_burnin[3] * 50
        mu_b = states_burnin[0]
        sigma_b = states_burnin[1]
        b_offset = states_burnin[2]
        phi = states_burnin[5]

        ind_ = np.exp(ind_raw) / (1 + np.exp(ind_raw))

        b_raw_ = mu_b.reshape((num_results - n_burnin, 1, 1)) + np.einsum("...jk, ...j->...jk", b_offset, sigma_b)

        # phi_ = np.repeat(phi[:, np.newaxis, :, :], self.N, axis=1)

        # print(phi_.shape)

        beta_ = np.einsum("..., ...", ind_, b_raw_)

        # conc_ = np.exp(np.einsum("jk, ...kl->...jl", self.x, beta_)
        #                + alphas.reshape((num_results - n_burnin, 1, self.K)))

        # predictions_ = np.zeros(chain_size_y)
        # for i in range(num_results - n_burnin):
        #     pred = tfd.DirichletMultinomial(self.n_total, conc_[i, :, :]).mean().numpy()
        #     predictions_[i, :, :] = pred

        betas_final = beta_.mean(axis=0)
        states_burnin.append(ind_)
        states_burnin.append(b_raw_)
        # states_burnin.append(b_time_)
        states_burnin.append(beta_)
        # states_burnin.append(conc_)
        # states_burnin.append(predictions_)

        # concentration = np.exp(np.matmul(self.x, betas_final) + alphas_final).astype(np.float32)

        #  y_mean = concentration / np.sum(concentration, axis=1, keepdims=True) * self.n_total.numpy()[:, np.newaxis]
        # print(y_mean)

        return None