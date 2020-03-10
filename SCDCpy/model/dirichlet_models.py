
""""
This file defines multiple Dirichlet-multinomial models
for statistical analysis of compositional changes
For further reference, see:
Johannes Ostner: Development of a statistical framework for compositional analysis of single-cell data

:authors: Johannes Ostner
"""
import numpy as np
import time

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.experimental import edward2 as ed

from SCDCpy.util import result_classes as res

tfd = tfp.distributions
tfb = tfp.bijectors


class CompositionalModel:
    """
    Implements class framework for compositional data models
    """

    def __init__(self, covariate_matrix, data_matrix, cell_types, covariate_names, *args, **kwargs):
        """
        Generalized Constructor of model class

        Parameters
        ----------
        covariate_matrix -- numpy array [NxD] - covariate matrix
        data_matrix -- numpy array [NxK] - cell count matrix
        cell_types -- list of cell type names
        covariate_names -- List of covariate names
        """

        dtype = tf.float32
        self.x = tf.convert_to_tensor(covariate_matrix, dtype)
        self.y = tf.convert_to_tensor(data_matrix, dtype)
        sample_counts = np.sum(data_matrix, axis=1)
        self.n_total = tf.cast(sample_counts, dtype)
        self.cell_types = cell_types
        self.covariate_names = covariate_names

        # Get dimensions of data
        self.N, self.D = self.x.shape
        self.K = self.y.shape[1]

        # Check input data
        if self.N != self.y.shape[0]:
            raise ValueError("Wrong input dimensions X[{},:] != y[{},:]".format(self.x.shape[0], self.y.shape[0]))
        if self.N != len(self.n_total):
            raise ValueError("Wrong input dimensions X[{},:] != n_total[{}]".format(self.x.shape[0], len(self.n_total)))


    def sampling(self, num_results, n_burnin, kernel, init_state):
        """

        Parameters
        ----------
        num_results -- MCMC chain length (default 20000)
        n_burnin -- Number of burnin iterations (default 5000)
        kernel -- MCMC kernel object
        init_state -- Starting parameters

        Returns
        -------
        states -- States of MCMC chain
        kernel_results -- sampling meta-information
        """

        # HMC sampling function
        @tf.function
        def sample_mcmc(num_results_, n_burnin_, kernel_, current_state_):
            return tfp.mcmc.sample_chain(
                num_results=num_results_,
                num_burnin_steps=n_burnin_,
                kernel=kernel_,
                current_state=current_state_,
                # tace_fn=lambda _, pkr: pkr
                trace_fn=lambda _, pkr: [pkr.inner_results.inner_results.is_accepted,
                                         pkr.inner_results.inner_results.accepted_results.step_size]
                )

        # The actual sampling process
        start = time.time()
        states, kernel_results = sample_mcmc(num_results, n_burnin, kernel, init_state)
        duration = time.time() - start
        print("MCMC sampling finished. ({:.3f} sec)".format(duration))

        return states, kernel_results

    def get_chains_after_burnin(self, samples, accept, n_burnin):
        """
        Application of burnin after sampling
        Parameters
        ----------
        samples -- all kernel states
        accept -- Kernel meta-information
        n_burnin -- number of burnin iterations

        Returns
        -------
        states_burnin -- Kernel states without burnin samples
        """
        # Samples after burn-in
        states_burnin = []
        acceptances = accept[0].numpy()

        for s in samples:
            states_burnin.append(s[n_burnin:].numpy())

        # Calculate acceptance rate
        p_accept = sum(acceptances) / acceptances.shape[0]
        print('Acceptance rate: %0.1f%%' % (100 * p_accept))

        return states_burnin

    def sample_hmc(self, num_results=int(20e3), n_burnin=int(5e3), num_leapfrog_steps=10, step_size=0.01):
        """
        HMC sampling
        Parameters
        ----------
        num_results -- MCMC chain length (default 20000)
        n_burnin -- Number of burnin iterations (default 5000)
        num_leapfrog_steps -- HMC leapfrog steps (default 10)
        step_size -- Initial step size (default 0.01)

        Returns
        -------
        SCDCpy.util.result_data object
        """

        # (not in use atm)
        constraining_bijectors = [
            tfb.Identity(),
            tfb.Identity(),
            tfb.Identity(),
            tfb.Identity(),
            tfb.Identity(),
        ]

        # HMC transition kernel
        hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self.target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps)
        hmc_kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=hmc_kernel, bijector=constraining_bijectors)
        hmc_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=hmc_kernel, num_adaptation_steps=int(4000), target_accept_prob=0.8)

        # HMC sampling
        states, kernel_results = self.sampling(num_results, n_burnin, hmc_kernel, self.params)
        states_burnin = self.get_chains_after_burnin(states, kernel_results, n_burnin)

        y_hat = self.get_y_hat(states_burnin, num_results, n_burnin)

        params = dict(zip(self.param_names, states_burnin))

        # Result object generation setup
        if self.baseline_index is not None:
            cell_types_nb = self.cell_types[:self.baseline_index] + self.cell_types[self.baseline_index+1:]
        else:
            cell_types_nb = self.cell_types

        posterior = {var_name: [var] for var_name, var in params.items() if
                     "prediction" not in var_name}
        posterior_predictive = {"prediction": [params["prediction"]]}
        observed_data = {"y": self.y}
        dims = {"alpha": ["cell_type"],
                "mu_b": ["1"],
                "sigma_b": ["1"],
                "b_offset": ["covariate", "cell_type_nb"],
                "ind_raw": ["covariate", "cell_type_nb"],
                "ind": ["covariate", "cell_type_nb"],
                "b_raw": ["covariate", "cell_type_nb"],
                "beta": ["covariate", "cell_type"],
                "concentration": ["sample", "cell_type"],
                "prediction": ["sample", "cell_type"]
                }
        coords = {"cell_type": self.cell_types,
                  "cell_type_nb": cell_types_nb,
                  "covariate": self.covariate_names,
                  "sample": range(self.y.shape[0])
                  }

        return res.CAResultConverter(posterior=posterior,
                               posterior_predictive=posterior_predictive,
                               observed_data=observed_data,
                               dims=dims,
                               coords=coords).to_result_data(y_hat, baseline=False)

    def sample_nuts(self, num_results=int(10e3), n_burnin=int(5e3), max_tree_depth=10, step_size=0.01):
        """
        NUTS sampling - WIP, DO NOT USE!!!
        Parameters
        ----------
        num_results
        n_burnin
        max_tree_depth
        step_size

        Returns
        -------

        """
        # (not in use atm)
        constraining_bijectors = [
            tfb.Identity(),
            tfb.Identity(),
            tfb.Identity(),
            tfb.Identity(),
            tfb.Identity(),
        ]

        # NUTS transition kernel
        nuts_kernel = tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=self.target_log_prob_fn,
            step_size=step_size,
            max_tree_depth=max_tree_depth)
        nuts_kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=nuts_kernel, bijector=constraining_bijectors)
        nuts_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=nuts_kernel, num_adaptation_steps=int(4000), target_accept_prob=0.9,
            step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
                inner_results=pkr.inner_results._replace(step_size=new_step_size)
            ),
            step_size_getter_fn=lambda pkr: pkr.inner_results.step_size,
            log_accept_prob_getter_fn=lambda pkr: pkr.inner_results.log_accept_ratio,
        )

        states, kernel_results = self.sampling(num_results, n_burnin, nuts_kernel, self.params)
        states_burnin = self.get_chains_after_burnin(states, kernel_results, n_burnin)
        y_hat = self.get_y_hat(states_burnin, num_results, n_burnin)

        params = dict(zip(self.param_names, states_burnin))

        # Arviz setup
        posterior = {var_name: [var] for var_name, var in params.items() if
                     "prediction" not in var_name}
        posterior_predictive = {"prediction": [params["prediction"]]}
        observed_data = {"y": self.y}
        dims = {"alpha": ["cell_type"],
                "mu_b": ["1"],
                "sigma_b": ["1"],
                "b_offset": ["covariate", "cell_type"],
                "ind_raw": ["covariate", "cell_type"],
                "ind": ["covariate", "cell_type"],
                "b_raw": ["covariate", "cell_type"],
                "beta": ["cov", "cell_type"],
                "concentration": ["sample", "cell_type"],
                "prediction": ["sample", "cell_type"]
                }
        coords = {"cell_type": self.cell_types,
                  "covariate": self.covariate_names,
                  "sample": range(self.y.shape[0])
                  }

        return res.CAResultConverter(posterior=posterior,
                                     posterior_predictive=posterior_predictive,
                                     observed_data=observed_data,
                                     dims=dims,
                                     coords=coords).to_result_data(y_hat, baseline=False)


class NoBaselineModel(CompositionalModel):

    """"
    implements statistical model and
    test statistics for compositional differential change analysis
    without specification of a baseline cell type
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor of model class
        Parameters
        ----------
        args -- arguments passed to top-level class
        kwargs -- arguments passed to top-level class
        """
        super(self.__class__, self).__init__(*args, **kwargs)

        self.baseline_index = None
        dtype = tf.float32

        # All parameters that are returned for analysis
        self.param_names = ["alpha", "mu_b", "sigma_b", "b_offset", "ind_raw",
                            "ind", "b_raw", "beta", "concentration", "prediction"]

        def define_model(x, n_total, K):
            """
            Model definition in Edward2
            Parameters
            ----------
            x -- numpy array [NxD] - covariate matrix
            n_total -- numpy array [N] - number of cells per sample
            K -- Number of cell types
            """

            N, D = x.shape

            # normal prior on bias
            alpha = ed.Normal(loc=tf.zeros([K]), scale=tf.ones([K])*5, name="alpha")

            # Noncentered parametrization for raw slopes (before spike-and-slab)
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
        self.target_log_prob_fn = lambda alpha_, mu_b_, sigma_b_, b_offset_, sigma_ind_raw_:\
            self.log_joint(x=self.x,
                           n_total=self.n_total,
                           K=self.K,
                           predictions=self.y,
                           alpha=alpha_,
                           mu_b=mu_b_,
                           sigma_b=sigma_b_,
                           b_offset=b_offset_,
                           sigma_ind_raw=sigma_ind_raw_,
                           )

        alpha_size = [self.K]
        beta_size = [self.D, self.K]

        # MCMC starting values
        self.params = [tf.random.normal(alpha_size, 0, 1, name='init_alpha'),
                       tf.zeros(1, name="init_mu_b", dtype=dtype),
                       tf.ones(1, name="init_sigma_b", dtype=dtype),
                       tf.random.normal(beta_size, 0, 1, name='init_b_offset'),
                       tf.zeros(beta_size, name='init_sigma_ind_raw', dtype=dtype),
                       ]

    # Calculate predicted cell counts (for analysis purposes)
    def get_y_hat(self, states_burnin, num_results, n_burnin):
        """
        Calculate predicted cell counts (for analysis purposes) and add intermediate parameters to MCMC results
        Parameters
        ----------
        states_burnin -- MCMC chain without burnin samples
        num_results -- Chain length (with burnin)
        n_burnin -- Number of burnin samples

        Returns
        -------
        predicted cell counts
        """

        start = time.time()

        chain_size_y = [num_results - n_burnin, self.N, self.K]

        alphas = states_burnin[0]
        alphas_final = alphas.mean(axis=0)

        ind_raw = states_burnin[4] * 50
        mu_b = states_burnin[1]
        sigma_b = states_burnin[2]
        b_offset = states_burnin[3]

        ind_ = np.exp(ind_raw) / (1 + np.exp(ind_raw))

        b_raw_ = mu_b.reshape((num_results - n_burnin, 1, 1)) + np.einsum("...jk, ...j->...jk", b_offset, sigma_b)

        beta_ = np.einsum("..., ...", ind_, b_raw_)

        conc_ = np.exp(np.einsum("jk, ...kl->...jl", self.x, beta_)
                       + alphas.reshape((num_results - n_burnin, 1, self.K)))

        predictions_ = np.zeros(chain_size_y)
        for i in range(num_results-n_burnin):
            pred = tfd.DirichletMultinomial(self.n_total, conc_[i, :, :]).mean().numpy()
            predictions_[i, :, :] = pred

        betas_final = beta_.mean(axis=0)
        states_burnin.append(ind_)
        states_burnin.append(b_raw_)
        states_burnin.append(beta_)
        states_burnin.append(conc_)
        states_burnin.append(predictions_)

        duration = time.time() - start
        print("get_y_hat ({:.3f} sec)".format(duration))

        concentration = np.exp(np.matmul(self.x, betas_final) + alphas_final).astype(np.float32)
        y_mean = concentration / np.sum(concentration, axis=1, keepdims=True) * self.n_total.numpy()[:, np.newaxis]
        return y_mean


class BaselineModel(CompositionalModel):
    """
    implements statistical model and
    test statistics for compositional differential change analysis
    with specification of a baseline cell type
    """

    def __init__(self, baseline_index, *args, **kwargs):

        """
        Constructor of model class
        Parameters
        ----------
        baseline_index -- Index of reference cell type (column in count data matrix)
        args -- arguments passed to top-level class
        kwargs -- arguments passed to top-level class
        """
        super(self.__class__, self).__init__(*args, **kwargs)

        dtype = tf.float32
        self.baseline_index = baseline_index

        # All parameters that are returned for analysis
        self.param_names = ["alpha", "mu_b", "sigma_b", "b_offset", "ind_raw",
                            "ind", "b_raw", "beta", "concentration", "prediction"]

        def define_model(x, n_total, K):
            """
            Model definition in Edward2
            Parameters
            ----------
            x -- numpy array [NxD] - covariate matrix
            n_total -- numpy array [N] - number of cells per sample
            K -- Number of cell types
            """
            N, D = x.shape

            # normal prior on bias
            alpha = ed.Normal(loc=tf.zeros([K]), scale=tf.ones([K]) * 5, name="alpha")

            # Noncentered parametrization for raw slopes of all cell types except baseline type (before spike-and-slab)
            mu_b = ed.Normal(loc=tf.zeros(1, dtype=dtype), scale=tf.ones(1, dtype=dtype), name="mu_b")
            sigma_b = ed.HalfCauchy(tf.zeros(1, dtype=dtype), tf.ones(1, dtype=dtype), name="sigma_b")
            b_offset = ed.Normal(loc=tf.zeros([D, K-1], dtype=dtype), scale=tf.ones([D, K-1], dtype=dtype),
                                 name="b_offset")

            b_raw = mu_b + sigma_b * b_offset

            # Spike-and-slab priors
            sigma_ind_raw = ed.Normal(
                loc=tf.zeros(shape=[D, K-1], dtype=dtype),
                scale=tf.ones(shape=[D, K-1], dtype=dtype),
                name='sigma_ind_raw')
            ind_t = sigma_ind_raw * 50
            ind = tf.exp(ind_t) / (1 + tf.exp(ind_t))

            # Calculate betas
            beta = ind * b_raw

            # Include slope 0 for baseline cell type
            beta = tf.concat(axis=1, values=[beta[:, :baseline_index],
                                             tf.fill(value=0., dims=[D, 1]),
                                             beta[:, baseline_index:]])

            # Concentration vector from intercepts, slopes
            concentration_ = tf.exp(alpha + tf.matmul(x, beta))

            # Cell count prediction via DirMult
            predictions = ed.DirichletMultinomial(n_total, concentration=concentration_, name="predictions")
            return predictions

        # Joint posterior distribution
        self.log_joint = ed.make_log_joint_fn(define_model)
        # Function to compute log posterior probability

        self.target_log_prob_fn = lambda alpha_, mu_b_, sigma_b_, b_offset_, sigma_ind_raw_:\
            self.log_joint(x=self.x,
                           n_total=self.n_total,
                           K=self.K,
                           predictions=self.y,
                           alpha=alpha_,
                           mu_b=mu_b_,
                           sigma_b=sigma_b_,
                           b_offset=b_offset_,
                           sigma_ind_raw=sigma_ind_raw_,
                           )

        alpha_size = [self.K]
        beta_size = [self.D, self.K-1]

        # MCMC starting values
        self.params = [tf.random.normal(alpha_size, 0, 1, name='init_alpha'),
                       tf.zeros(1, name="init_mu_b", dtype=dtype),
                       tf.ones(1, name="init_sigma_b", dtype=dtype),
                       tf.random.normal(beta_size, 0, 1, name='init_b_offset'),
                       tf.zeros(beta_size, name='init_sigma_ind_raw', dtype=dtype),
                       ]

    # Calculate predicted cell counts (for analysis purposes)
    def get_y_hat(self, states_burnin, num_results, n_burnin):
        """
        Calculate predicted cell counts (for analysis purposes) and add intermediate parameters to MCMC results
        Parameters
        ----------
        states_burnin -- MCMC chain without burnin samples
        num_results -- Chain length (with burnin)
        n_burnin -- Number of burnin samples

        Returns
        -------
        predicted cell counts
        """

        chain_size_beta = [num_results - n_burnin, self.D, self.K]
        chain_size_beta_raw = [num_results - n_burnin, self.D, self.K-1]
        chain_size_y = [num_results - n_burnin, self.N, self.K]

        alphas = states_burnin[0]
        alphas_final = alphas.mean(axis=0)

        ind_raw = states_burnin[4] * 50
        mu_b = states_burnin[1]
        sigma_b = states_burnin[2]
        b_offset = states_burnin[3]

        ind_ = np.exp(ind_raw) / (1 + np.exp(ind_raw))

        b_raw_ = mu_b.reshape((num_results - n_burnin, 1, 1)) + np.einsum("...jk, ...j->...jk", b_offset, sigma_b)

        beta_temp = np.einsum("..., ...", ind_, b_raw_)

        beta_ = np.zeros(chain_size_beta)
        for i in range(num_results - n_burnin):
            beta_[i] = np.concatenate([beta_temp[i, :, :self.baseline_index],
                                       np.zeros(shape=[self.D, 1], dtype=np.float32),
                                       beta_temp[i, :, self.baseline_index:]], axis=1)

        conc_ = np.exp(np.einsum("jk, ...kl->...jl", self.x, beta_)
                       + alphas.reshape((num_results - n_burnin, 1, self.K))).astype(np.float32)

        predictions_ = np.zeros(chain_size_y)
        for i in range(num_results - n_burnin):
            pred = tfd.DirichletMultinomial(self.n_total, conc_[i, :, :]).mean().numpy()
            predictions_[i, :, :] = pred

        betas_final = beta_.mean(axis=0)
        states_burnin.append(ind_)
        states_burnin.append(b_raw_)
        states_burnin.append(beta_)
        states_burnin.append(conc_)
        states_burnin.append(predictions_)

        concentration = np.exp(np.matmul(self.x, betas_final) + alphas_final).astype(np.float32)
        y_mean = concentration / np.sum(concentration, axis=1, keepdims=True) * self.n_total.numpy()[:, np.newaxis]
        return y_mean


class NoBaselineModelNoEdward(CompositionalModel):

    """
    WIP! DO NOT USE!!!

    implements statistical model and
    test statistics for compositional differential change analysis
    without specification of a baseline cell type
    """

    def __init__(self, covariate_matrix, data_matrix):
        """
        Constructor of model class
        :param covariate_matrix: numpy array [NxD] - covariate matrix
        :param data_matrix: numpy array [NxK] - cell count matrix
        :param sample_counts: numpy array [N] - number of cells per sample
        :param dtype: data type for all numbers (for tensorflow)
        """

        dtype = tf.float32
        self.x = tf.cast(covariate_matrix, dtype)
        self.y = tf.cast(data_matrix, dtype)
        sample_counts = np.sum(data_matrix, axis=1)
        self.n_total = tf.cast(sample_counts, dtype)
        self.baseline_index = None

        # Get dimensions of data
        N, D = self.x.shape
        K = self.y.shape[1]

        # Check input data
        if N != self.y.shape[0]:
            raise ValueError("Wrong input dimensions X[{},:] != y[{},:]".format(self.x.shape[0], self.y.shape[0]))
        if N != len(self.n_total):
            raise ValueError("Wrong input dimensions X[{},:] != n_total[{}]".format(self.x.shape[0], len(self.n_total)))

        # All parameters that are returned for analysis
        self.param_names = ["mu_b", "sigma_b", "b_offset", "ind_raw", "alpha", "beta"]

        alpha_size = [1, K]
        beta_size = [D, K]

        self.model_struct = tfd.JointDistributionSequential([
            tfd.Independent(
                tfd.Normal(loc=tf.zeros(1, dtype=dtype),
                           scale=tf.ones(1, dtype=dtype),
                           name="mu_b"),
                reinterpreted_batch_ndims=1),

            tfd.Independent(
                tfd.HalfCauchy(tf.zeros(1, dtype=dtype),
                               tf.ones(1, dtype=dtype),
                               name="sigma_b"),
                reinterpreted_batch_ndims=1),

            tfd.Independent(
                tfd.Normal(
                    loc=tf.zeros([D, K], dtype=dtype),
                    scale=tf.ones([D, K], dtype=dtype),
                    name="b_offset"),
                reinterpreted_batch_ndims=2),

            # Spike-and-slab
            tfd.Independent(
                tfd.Normal(
                    loc=tf.zeros(shape=[D, K], dtype=dtype),
                    scale=tf.ones(shape=[D, K], dtype=dtype)*50,
                    name='ind_raw'),
                reinterpreted_batch_ndims=2),

            tfd.Independent(
                tfd.Normal(
                    loc=tf.zeros(alpha_size),
                    scale=tf.ones(alpha_size) * 5,
                    name="alpha"),
                reinterpreted_batch_ndims=2),

            # Cell count prediction via DirMult
            lambda alpha, ind_raw, b_offset, sigma_b, mu_b: tfd.Independent(
                tfd.DirichletMultinomial(
                    total_count=tf.cast(self.n_total, dtype),
                    concentration=tf.exp(alpha
                                         + tf.matmul(tf.cast(self.x, dtype),
                                                     (1 / (1 + tf.exp(-ind_raw)))
                                                     * (mu_b + sigma_b * b_offset)
                                                     )),
                    name="predictions"),
                reinterpreted_batch_ndims=1),
        ])

        # Joint posterior distribution
        self.target_log_prob_fn = lambda mu_b_, sigma_b_, b_offset_, ind_, alpha_:\
            self.model_struct.log_prob((mu_b_, sigma_b_, b_offset_, ind_, alpha_, tf.cast(self.y, dtype)))

        # MCMC starting values
        self.params = [tf.zeros(1, name="init_mu_b", dtype=dtype),
                       tf.ones(1, name="init_sigma_b", dtype=dtype),
                       tf.zeros(beta_size, name='init_b_offset', dtype=dtype),
                       tf.zeros(beta_size, name='init_ind_raw', dtype=dtype),
                       tf.zeros(alpha_size, name='init_alpha', dtype=dtype),
                       ]

        self.vars = [tf.Variable(v, trainable=True) for v in self.params]

    # Calculate predicted cell counts (for analysis purposes)
    def get_y_hat(self, states_burnin, num_results, n_burnin):
        alphas_final = states_burnin[4].numpy().mean(axis=0)

        ind_raw = states_burnin[3].numpy() * 50
        ind = np.exp(ind_raw) / (1 + np.exp(ind_raw))

        b_raw = np.array([states_burnin[0].numpy()[i] + (states_burnin[1].numpy()[i] * states_burnin[2].numpy()[i])
                          for i in range(num_results - n_burnin)])

        betas = ind * b_raw
        betas_final = betas.mean(axis=0)

        states_burnin.append(betas)

        return tfd.DirichletMultinomial(self.n_total,
                                        concentration=tf.exp(tf.matmul(self.x, betas_final) + alphas_final)).mean()
