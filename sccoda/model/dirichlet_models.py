"""
Dirichlet-multinomial models for statistical analysis of compositional changes

For further reference, see:
Johannes Ostner: Development of a statistical framework for compositional analysis of single-cell data

:authors: Johannes Ostner
"""
import numpy as np
import time

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.experimental import edward2 as ed

from sccoda.util import result_classes as res

tfd = tfp.distributions
tfb = tfp.bijectors


class CompositionalModel:
    """
    Implements class framework for compositional data models
    """

    def __init__(self, covariate_matrix, data_matrix, cell_types, covariate_names, formula, time_matrix=None, *args, **kwargs):
        """
        Generalized Constructor of model class

        Parameters
        ----------
        covariate_matrix -- numpy array [NxD]
            covariate matrix
        data_matrix -- numpy array [NxK]
            cell count matrix
        cell_types -- list
            Cell type names
        covariate_names -- List
            Covariate names
        """

        dtype = tf.float32
        self.x = tf.convert_to_tensor(covariate_matrix, dtype)

        # Add pseudocount if needed.
        if np.count_nonzero(data_matrix) != np.size(data_matrix):
            print("Zero counts encountered in data! Added a pseudocount of 0.5.")
            data_matrix += 0.5
        self.y = tf.convert_to_tensor(data_matrix, dtype)

        sample_counts = np.sum(data_matrix, axis=1)
        self.n_total = tf.cast(sample_counts, dtype)
        self.cell_types = cell_types
        self.covariate_names = covariate_names
        self.formula = formula

        if time_matrix is not None:
            self.time_matrix = tf.convert_to_tensor(time_matrix, dtype)

        # Get dimensions of data
        self.N, self.D = self.x.shape
        self.K = self.y.shape[1]

        # Check input data
        if self.N != self.y.shape[0]:
            raise ValueError("Wrong input dimensions X[{},:] != y[{},:]".format(self.x.shape[0], self.y.shape[0]))
        if self.N != len(self.n_total):
            raise ValueError("Wrong input dimensions X[{},:] != n_total[{}]".format(self.x.shape[0], len(self.n_total)))

    def sampling(self, num_results, n_burnin, kernel, init_state, trace_fn):
        """
        MCMC sampling process

        Parameters
        ----------
        num_results -- int
            MCMC chain length (default 20000)
        n_burnin -- int
            Number of burnin iterations (default 5000)
        kernel --
            tensorflow MCMC kernel object
        init_state -- dict
            Starting parameters

        Returns
        -------
        states -- list
            States of MCMC chain
        kernel_results -- list
            sampling meta-information
        duration -- float
            Duration of MCMC sampling process
        """

        # HMC sampling function
        @tf.function
        def sample_mcmc(num_results_, n_burnin_, kernel_, current_state_, trace_fn):

            return tfp.mcmc.sample_chain(
                num_results=num_results_,
                num_burnin_steps=n_burnin_,
                kernel=kernel_,
                current_state=current_state_,
                trace_fn=trace_fn
            )

        # The actual sampling process
        start = time.time()
        states, kernel_results = sample_mcmc(num_results, n_burnin, kernel, init_state, trace_fn)
        duration = time.time() - start
        print("MCMC sampling finished. ({:.3f} sec)".format(duration))

        return states, kernel_results, duration

    def get_chains_after_burnin(self, samples, kernel_results, n_burnin, is_nuts=False):
        """
        Application of burnin after sampling

        Parameters
        ----------
        samples -- list
            all kernel states
        kernel_results  -- list
            Kernel meta-information
        n_burnin -- int
            number of burnin iterations

        Returns
        -------
        states_burnin -- list
            Kernel states without burnin samples
        p_accept -- float
            acceptance rate of MCMC process
        """
        # Samples after burn-in
        states_burnin = []
        stats = {}

        for s in samples:
            states_burnin.append(s[n_burnin:].numpy())

        for k, v in kernel_results.items():
            stats[k] = v[n_burnin:].numpy()

        if is_nuts:
            p_accept = np.mean(np.exp(kernel_results["log_accept_ratio"].numpy()))
        else:
            acceptances = kernel_results["is_accepted"].numpy()

            # Calculate acceptance rate
            p_accept = sum(acceptances) / acceptances.shape[0]
        print('Acceptance rate: %0.1f%%' % (100 * p_accept))

        return states_burnin, stats, p_accept

    def sample_hmc(self, num_results=int(20e3), n_burnin=int(5e3), num_leapfrog_steps=10, step_size=0.01, num_adapt_steps=None):
        """
        HMC sampling

        Parameters
        ----------
        num_results -- int
            MCMC chain length (default 20000)
        n_burnin -- int
            Number of burnin iterations (default 5000)
        num_leapfrog_steps --  int
            HMC leapfrog steps (default 10)
        step_size -- float
            Initial step size (default 0.01)

        Returns
        -------
        result -- scCODA.util.result_data.CAResult object
            Compositional analysis result
        """

        # bijectors (not in use atm, therefore identity)
        constraining_bijectors = [tfb.Identity() for x in range(len(self.params))]

        # HMC transition kernel
        hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self.target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps)
        hmc_kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=hmc_kernel, bijector=constraining_bijectors)

        # Set default value for adaptation steps
        if num_adapt_steps is None:
            num_adapt_steps = int(0.8 * n_burnin)

        # Add step size adaptation
        hmc_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=hmc_kernel, num_adaptation_steps=num_adapt_steps, target_accept_prob=0.8)

        # tracing function
        def trace_fn(_, pkr):
            return {
                'target_log_prob': pkr.inner_results.inner_results.accepted_results.target_log_prob,
                'diverging': (pkr.inner_results.inner_results.log_accept_ratio < -1000.),
                'is_accepted': pkr.inner_results.inner_results.is_accepted,
                'step_size': pkr.inner_results.inner_results.accepted_results.step_size,
            }

        # HMC sampling
        states, kernel_results, duration = self.sampling(num_results, n_burnin, hmc_kernel, self.params, trace_fn)

        # apply burnin
        states_burnin, sample_stats, acc_rate = self.get_chains_after_burnin(states, kernel_results, n_burnin)

        # Calculate posterior predictive
        y_hat = self.get_y_hat(states_burnin, num_results, n_burnin)

        params = dict(zip(self.param_names, states_burnin))

        # Result object generation setup
        # Get names of cell types that are not the baseline
        if self.baseline_index is not None:
            cell_types_nb = self.cell_types[:self.baseline_index] + self.cell_types[self.baseline_index+1:]
        else:
            cell_types_nb = self.cell_types

        posterior = {var_name: [var] for var_name, var in params.items() if
                     "prediction" not in var_name}

        if "prediction" in self.param_names:
            posterior_predictive = {"prediction": [params["prediction"]]}
        else:
            posterior_predictive = {}

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

        sampling_stats = {"chain_length": num_results, "n_burnin": n_burnin,
                        "acc_rate": acc_rate, "duration": duration, "y_hat": y_hat}

        model_specs = {"baseline": self.baseline_index, "formula": self.formula}

        return res.CAResultConverter(posterior=posterior,
                                     posterior_predictive=posterior_predictive,
                                     observed_data=observed_data,
                                     dims=dims,
                                     sample_stats=sample_stats,
                                     coords=coords).to_result_data(sampling_stats=sampling_stats,
                                                                   model_specs=model_specs)

    def sample_hmc_da(self, num_results=int(20e3), n_burnin=int(5e3), num_leapfrog_steps=10, step_size=0.01, num_adapt_steps=None):
        """
        HMC sampling

        Parameters
        ----------
        num_results -- int
            MCMC chain length (default 20000)
        n_burnin -- int
            Number of burnin iterations (default 5000)
        num_leapfrog_steps --  int
            HMC leapfrog steps (default 10)
        step_size -- float
            Initial step size (default 0.01)

        Returns
        -------
        result -- scCODA.util.result_data.CAResult object
            Compositional analysis result
        """

        # (not in use atm)
        constraining_bijectors = [tfb.Identity() for x in range(len(self.params))]

        # HMC transition kernel
        hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self.target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps)
        hmc_kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=hmc_kernel, bijector=constraining_bijectors)

        # Set default value for adaptation steps
        if num_adapt_steps is None:
            num_adapt_steps = int(0.8*n_burnin)

        # Add step size adaptation
        hmc_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=hmc_kernel, num_adaptation_steps=num_adapt_steps, target_accept_prob=0.8, decay_rate=0.5)

        # tracing function
        def trace_fn(_, pkr):
            return {
                'target_log_prob': pkr.inner_results.inner_results.accepted_results.target_log_prob,
                'diverging': (pkr.inner_results.inner_results.log_accept_ratio < -1000.),
                "log_acc_ratio": pkr.inner_results.inner_results.log_accept_ratio,
                'is_accepted': pkr.inner_results.inner_results.is_accepted,
                'step_size': tf.exp(pkr.log_averaging_step[0]),
            }

        # HMC sampling
        states, kernel_results, duration = self.sampling(num_results, n_burnin, hmc_kernel, self.params, trace_fn)
        states_burnin, sample_stats, acc_rate = self.get_chains_after_burnin(states, kernel_results, n_burnin)

        y_hat = self.get_y_hat(states_burnin, num_results, n_burnin)

        params = dict(zip(self.param_names, states_burnin))

        # Result object generation setup
        if self.baseline_index is not None:
            cell_types_nb = self.cell_types[:self.baseline_index] + self.cell_types[self.baseline_index+1:]
        else:
            cell_types_nb = self.cell_types

        posterior = {var_name: [var] for var_name, var in params.items() if
                     "prediction" not in var_name}

        if "prediction" in self.param_names:
            posterior_predictive = {"prediction": [params["prediction"]]}
        else:
            posterior_predictive = {}

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

        # build dicionary with sampling statistics
        sampling_stats = {"chain_length": num_results, "n_burnin": n_burnin,
                        "acc_rate": acc_rate, "duration": duration, "y_hat": y_hat}

        model_specs = {"baseline": self.baseline_index, "formula": self.formula}

        return res.CAResultConverter(posterior=posterior,
                                     posterior_predictive=posterior_predictive,
                                     observed_data=observed_data,
                                     dims=dims,
                                     sample_stats=sample_stats,
                                     coords=coords).to_result_data(sampling_stats=sampling_stats,
                                                                   model_specs=model_specs)

    def sample_nuts(self, num_results=int(10e3), n_burnin=int(5e3), max_tree_depth=10, step_size=0.01, num_adapt_steps=None):
        """
        NUTS sampling - WIP, DO NOT USE!!!

        Parameters
        ----------
        num_results -- int
            MCMC chain length (default 20000)
        n_burnin -- int
            Number of burnin iterations (default 5000)
        max_tre_depth --  int
            Maximum tree depth (default 10)
        step_size -- float
            Initial step size (default 0.01)

        Returns
        -------
        error
            NotImplementedError
        """

        # bijectors (not in use atm, therefore identity)
        constraining_bijectors = [tfb.Identity() for x in range(len(self.params))]

        # NUTS transition kernel
        nuts_kernel = tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=self.target_log_prob_fn,
            step_size=step_size,
            max_tree_depth=max_tree_depth)
        nuts_kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=nuts_kernel,
            bijector=constraining_bijectors
        )

        # Set default value for adaptation steps
        if num_adapt_steps is None:
            num_adapt_steps = int(0.8 * n_burnin)

        # Step size adaptation
        nuts_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=nuts_kernel, num_adaptation_steps=num_adapt_steps, target_accept_prob=0.8,
            step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
                inner_results=pkr.inner_results._replace(step_size=new_step_size)
            ),
            step_size_getter_fn=lambda pkr: pkr.inner_results.step_size,
            log_accept_prob_getter_fn=lambda pkr: pkr.inner_results.log_accept_ratio,
        )

        # trace function
        def trace_fn(_, pkr):
            return {
                "target_log_prob": pkr.inner_results.inner_results.target_log_prob,
                "leapfrogs_taken": pkr.inner_results.inner_results.leapfrogs_taken,
                "diverging": pkr.inner_results.inner_results.has_divergence,
                "energy": pkr.inner_results.inner_results.energy,
                "log_accept_ratio": pkr.inner_results.inner_results.log_accept_ratio,
                'step_size': pkr.inner_results.inner_results.step_size[0],
                "reach_max_depth": pkr.inner_results.inner_results.reach_max_depth,
                "is_accepted": pkr.inner_results.inner_results.is_accepted,
            }

        # HMC sampling
        states, kernel_results, duration = self.sampling(num_results, n_burnin, nuts_kernel, self.params, trace_fn)
        states_burnin, sample_stats, acc_rate = self.get_chains_after_burnin(states, kernel_results, n_burnin, is_nuts=True)

        y_hat = self.get_y_hat(states_burnin, num_results, n_burnin)

        params = dict(zip(self.param_names, states_burnin))

        # Result object generation setup
        # Get names of cell types that are not the baseline
        if self.baseline_index is not None:
            cell_types_nb = self.cell_types[:self.baseline_index] + self.cell_types[self.baseline_index + 1:]
        else:
            cell_types_nb = self.cell_types

        posterior = {var_name: [var] for var_name, var in params.items() if
                     "prediction" not in var_name}

        if "prediction" in self.param_names:
            posterior_predictive = {"prediction": [params["prediction"]]}
        else:
            posterior_predictive = {}

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

        sampling_stats = {"chain_length": num_results, "n_burnin": n_burnin,
                          "acc_rate": acc_rate, "duration": duration, "y_hat": y_hat}

        model_specs = {"baseline": self.baseline_index, "formula": self.formula}

        return res.CAResultConverter(posterior=posterior,
                                     posterior_predictive=posterior_predictive,
                                     observed_data=observed_data,
                                     dims=dims,
                                     sample_stats=sample_stats,
                                     coords=coords).to_result_data(sampling_stats=sampling_stats,
                                                                   model_specs=model_specs)


class NoBaselineModel(CompositionalModel):

    """"
    implements statistical model for compositional differential change analysis without specification of a baseline cell type
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
            x -- numpy array [NxD]
                covariate matrix
            n_total -- numpy array [N]
                number of cells per sample
            K -- int
                Number of cell types
            """

            N, D = x.shape
            dtype = tf.float32

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
        states_burnin -- List
            MCMC chain without burnin samples
        num_results -- int
            Chain length (with burnin)
        n_burnin -- int
            Number of burnin samples

        Returns
        -------
        y_mean
            predicted cell counts
        """

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

        concentration = np.exp(np.matmul(self.x, betas_final) + alphas_final).astype(np.float32)
        y_mean = concentration / np.sum(concentration, axis=1, keepdims=True) * self.n_total.numpy()[:, np.newaxis]
        return y_mean


class BaselineModel(CompositionalModel):
    """
    implements statistical model for compositional differential change analysis with specification of a baseline cell type
    """

    def __init__(self, baseline_index, *args, **kwargs):

        """
        Constructor of model class

        Parameters
        ----------
        baseline_index -- string or int
            Index of reference cell type (column in count data matrix)
        args -- arguments passed to top-level class
        kwargs -- arguments passed to top-level class
        """
        super(self.__class__, self).__init__(*args, **kwargs)

        self.baseline_index = baseline_index
        dtype = tf.float32

        # All parameters that are returned for analysis
        self.param_names = ["alpha", "mu_b", "sigma_b", "b_offset", "ind_raw",
                            "ind", "b_raw", "beta", "concentration", "prediction"]

        def define_model(x, n_total, K):
            """
            Model definition in Edward2

            Parameters
            ----------
            x -- numpy array [NxD]
                covariate matrix
            n_total -- numpy array [N]
                number of cells per sample
            K -- int
                Number of cell types
            """
            dtype = tf.float32
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
        states_burnin -- List
            MCMC chain without burnin samples
        num_results -- int
            Chain length (with burnin)
        n_burnin -- int
            Number of burnin samples

        Returns
        -------
        y_mean
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
        self.param_names = ["mu_b", "sigma_b", "b_offset", "ind_raw", "alpha",
                            "ind", "b_raw", "beta", "concentration", "prediction"]

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

            ind_scaled = ind_raw * 50
            ind = tf.exp(ind_scaled) / (1 + tf.exp(ind_scaled))

            b_raw = mu_b + sigma_b * b_offset

            beta = ind * b_raw

            alpha = yield Root(tfd.Independent(
                tfd.Normal(
                    loc=tf.zeros(alpha_size),
                    scale=tf.ones(alpha_size) * 5,
                    name="alpha"),
                reinterpreted_batch_ndims=1))

            concentrations = tf.exp(alpha + tf.matmul(tf.cast(self.x, dtype), beta))

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
                       tf.zeros(alpha_size, name='init_alpha', dtype=dtype)
                       ]

    # Calculate predicted cell counts (for analysis purposes)
    def get_y_hat(self, states_burnin, num_results, n_burnin):
        chain_size_y = [num_results - n_burnin, self.N, self.K]

        alphas = states_burnin[4]
        alphas_final = alphas.mean(axis=0)

        ind_raw = states_burnin[3] * 50
        mu_b = states_burnin[0]
        sigma_b = states_burnin[1]
        b_offset = states_burnin[2]

        ind_ = np.exp(ind_raw) / (1 + np.exp(ind_raw))

        b_raw_ = mu_b.reshape((num_results - n_burnin, 1, 1)) + np.einsum("...jk, ...j->...jk", b_offset, sigma_b)

        beta_ = np.einsum("..., ...", ind_, b_raw_)

        conc_ = np.exp(np.einsum("jk, ...kl->...jl", self.x, beta_)
                       + alphas.reshape((num_results - n_burnin, 1, self.K)))

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


class BaselineModelNoEdward(CompositionalModel):

    """
    WIP! DO NOT USE!!!

    implements statistical model and
    test statistics for compositional differential change analysis
    without specification of a baseline cell type
    """

    def __init__(self, baseline_index, *args, **kwargs):
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

        self.baseline_index = baseline_index
        dtype = tf.float32

        # All parameters that are returned for analysis
        self.param_names = ["mu_b", "sigma_b", "b_offset", "ind_raw", "alpha",
                            "ind", "b_raw", "beta", "concentration", "prediction"]

        alpha_size = [self.K]
        beta_size = [self.D, self.K]
        beta_nobl_size = [self.D, self.K-1]

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
                    loc=tf.zeros(beta_nobl_size, dtype=dtype),
                    scale=tf.ones(beta_nobl_size, dtype=dtype),
                    name="b_offset"),
                reinterpreted_batch_ndims=2))

            # Spike-and-slab
            ind_raw = yield Root(tfd.Independent(
                tfd.Normal(
                    loc=tf.zeros(shape=beta_nobl_size, dtype=dtype),
                    scale=tf.ones(shape=beta_nobl_size, dtype=dtype),
                    name='ind_raw'),
                reinterpreted_batch_ndims=2))

            ind_scaled = ind_raw * 50
            ind = tf.exp(ind_scaled) / (1 + tf.exp(ind_scaled))

            b_raw = mu_b + sigma_b * b_offset

            beta = ind * b_raw

            # Include slope 0 for baseline cell type
            beta = tf.concat(axis=1, values=[beta[:, :baseline_index],
                                             tf.fill(value=0., dims=[self.D, 1]),
                                             beta[:, baseline_index:]])

            alpha = yield Root(tfd.Independent(
                tfd.Normal(
                    loc=tf.zeros(alpha_size),
                    scale=tf.ones(alpha_size) * 5,
                    name="alpha"),
                reinterpreted_batch_ndims=1))

            concentrations = tf.exp(alpha + tf.matmul(self.x, beta))

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
                       tf.random.normal(beta_nobl_size, 0, 1, name='init_b_offset', dtype=dtype),
                       tf.zeros(beta_nobl_size, name='init_ind_raw', dtype=dtype),
                       tf.random.normal(alpha_size, 0, 1, name='init_alpha', dtype=dtype)
                       ]

    # Calculate predicted cell counts (for analysis purposes)
    def get_y_hat(self, states_burnin, num_results, n_burnin):
        chain_size_y = [num_results - n_burnin, self.N, self.K]

        alphas = states_burnin[4]
        alphas_final = alphas.mean(axis=0)

        ind_raw = states_burnin[3] * 50
        mu_b = states_burnin[0]
        sigma_b = states_burnin[1]
        b_offset = states_burnin[2]

        ind_ = np.exp(ind_raw) / (1 + np.exp(ind_raw))

        b_raw_ = mu_b.reshape((num_results - n_burnin, 1, 1)) + np.einsum("...jk, ...j->...jk", b_offset, sigma_b)

        beta_ = np.einsum("..., ...", ind_, b_raw_)

        conc_ = np.exp(np.einsum("jk, ...kl->...jl", self.x, beta_)
                       + alphas.reshape((num_results - n_burnin, 1, self.K)))

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
