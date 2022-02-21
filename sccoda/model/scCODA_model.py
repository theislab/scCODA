"""
Dirichlet-multinomial models for statistical analysis of compositional changes in single-cell data.

For further reference, see:
Büttner, Ostner et al.: scCODA: A Bayesian model for compositional single-cell data analysis

:authors: Johannes Ostner
"""
import numpy as np
import time
import warnings

import tensorflow as tf
import tensorflow_probability as tfp

from sccoda.util import result_classes as res
from typing import Optional, Tuple, Collection, Union, List

tfd = tfp.distributions
tfb = tfp.bijectors


class CompositionalModel:
    """
    Dynamical framework for formulation and inference of Bayesian models for compositional data analysis.
    This framework is used to implement scCODA's model as a subclass of this class.
    Tensorflow probability then allows to run a multitude of inference algorithms on these models
    without the need to specify them every time.

    A `CompositionalModel` consists of the following parameters:

    - `covariate_matrix`: Numpy array that specifies the independent variables (X). Generated equivalently to the covariate matrix of a linear regression.

    - `data_matrix`: Dependent variable (Y). Includes the raw cell type counts for every sample.

    - `cell_types`, covariate_names: Names of cell types and covariates

    - `formula`: String that represents which covariates to include and how to transform them. Used analogously to the formula in R's lm function

    A specific compositional model is then implemented as a child class, with the following additional parameters
    specified in the constructor:

    - `target_log_prob_fn`: Log-probability function of the model. For more specific information, please refer to (tensorflow probability's API)[https://www.tensorflow.org/probability/api_docs/python/tfp]

    - `param_names`: Names of prior and intermediate parameters that are included in the model output. The order has to be the same as in the states_burnin output of `self.get_y_hat`

    - `init_params`: Initial values for the inference method

    Methods implemented by this class:

    - `sampling`: General MCMC sampling that uses a transition kernel

    - `get_chains_after_burnin`: Application of burn-in to MCMC sampling results

    - MCMC sampling methods (`sample_hmc`, `sample_hmc_da`, `sample_nuts`)

    Methods implemented by a child class:

    - `get_y_hat`: Calculation of intermediate parameters for all MCMC chain states and posterior mode of the cell count matrix

    """

    def __init__(
            self,
            covariate_matrix: np.ndarray,
            data_matrix: np.ndarray,
            cell_types: List[str],
            covariate_names: List[str],
            formula: str,
            *args,
            **kwargs
    ):
        """
        Generalized Constructor of Bayesian compositional model class.

        Parameters
        ----------
        covariate_matrix
            covariate matrix, size NxD
        data_matrix
            cell count matrix, size NxK
        cell_types
            Cell type names
        covariate_names
            Covariate names
        formula
            Formula (R-style)
        """

        dtype = tf.float64
        self.x = tf.convert_to_tensor(covariate_matrix, dtype)

        # Add pseudocount if zeroes are present.
        if np.count_nonzero(data_matrix) != np.size(data_matrix):
            print("Zero counts encountered in data! Added a pseudocount of 0.5.")
            data_matrix[data_matrix == 0] = 0.5
        self.y = tf.convert_to_tensor(data_matrix, dtype)

        sample_counts = np.sum(data_matrix, axis=1)
        self.n_total = tf.cast(sample_counts, dtype)
        self.cell_types = cell_types
        self.covariate_names = covariate_names
        self.formula = formula

        # Get dimensions of data
        self.N, self.D = self.x.shape
        self.K = self.y.shape[1]

        # Check input data
        if self.N != self.y.shape[0]:
            raise ValueError("Wrong input dimensions X[{},:] != y[{},:]".format(self.x.shape[0], self.y.shape[0]))
        if self.N != len(self.n_total):
            raise ValueError("Wrong input dimensions X[{},:] != n_total[{}]".format(self.x.shape[0], len(self.n_total)))

    def sampling(
            self,
            num_results: int,
            num_burnin: int,
            kernel,
            init_state: dict,
            trace_fn,
    ) -> Tuple[List[any], List[any], float]:
        """
        MCMC sampling process (tensorflow 2)

        Parameters
        ----------
        num_results
            MCMC chain length (default 20000)
        num_burnin
            Number of burnin iterations (default 5000)
        kernel
            tensorflow MCMC kernel object
        init_state
            Starting parameters
        trace_fn
            tracing function

        Returns
        -------
        MCMC chain states and results

        states
            States of MCMC chain
        kernel_results
            sampling meta-information
        duration
            Duration of MCMC sampling process
        """

        # HMC sampling function
        @tf.function(autograph=False)
        def sample_mcmc(num_results_, num_burnin_, kernel_, current_state_, trace_fn):

            return tfp.mcmc.sample_chain(
                num_results=num_results_,
                num_burnin_steps=num_burnin_,
                kernel=kernel_,
                current_state=current_state_,
                trace_fn=trace_fn
            )

        # The actual sampling process
        start = time.time()
        states, kernel_results = sample_mcmc(num_results, num_burnin, kernel, init_state, trace_fn)
        duration = time.time() - start
        print("MCMC sampling finished. ({:.3f} sec)".format(duration))

        return states, kernel_results, duration

    def get_chains_after_burnin(
            self,
            samples: List[any],
            kernel_results: List[any],
            num_burnin: int,
            is_nuts: bool = False
    ) -> Tuple[List[any], dict, float]:
        """
        Application of burn-in after MCMC sampling.
        Cuts the first `num_burnin` samples from all inferred variables and diagnostic statistics.

        Parameters
        ----------
        samples
            all kernel states
        kernel_results
            Kernel meta-information. The tracked statistics depend on the sampling method.
        num_burnin
            number of burn-in iterations
        is_nuts
            Specifies whether NUTS sampling was used

        Returns
        -------
        MCMC chain without burn-in, sampling statistics, acceptance rate

        states_burnin
            Kernel states without burn-in samples
        stats
            sampling statistics
        p_accept
            acceptance rate of MCMC process
        """

        # Samples after burn-in
        states_burnin = []
        stats = {}

        # Apply burn-in to MCMC results
        for s in samples:
            states_burnin.append(s[num_burnin:].numpy())

        # Apply burn-in to sampling statistics
        for k, v in kernel_results.items():
            stats[k] = v[num_burnin:].numpy()

        # Calculation of acceptance rate (different for NUTS sampling)
        if is_nuts:
            p_accept = np.mean(np.exp(kernel_results["log_accept_ratio"].numpy()))
        else:
            acceptances = kernel_results["is_accepted"].numpy()

            # Calculate acceptance rate
            p_accept = sum(acceptances) / acceptances.shape[0]
        print('Acceptance rate: %0.1f%%' % (100 * p_accept))

        return states_burnin, stats, p_accept

    def sample_hmc(
            self,
            num_results: int = int(20e3),
            num_burnin: int = int(5e3),
            num_adapt_steps: Optional[int] = None,
            num_leapfrog_steps: Optional[int] = 10,
            step_size: float = 0.01
    ) -> res.CAResult:

        """
        Hamiltonian Monte Carlo (HMC) sampling in tensorflow 2.

        Tracked diagnostic statistics:

        - `target_log_prob`: Value of the model's log-probability

        - `diverging`: Marks samples as diverging (NOTE: Handle with care, the spike-and-slab prior of scCODA usually leads to many samples being flagged as diverging)

        - `is_accepted`: Whether the proposed sample was accepted in the algorithm's acceptance step

        - `step_size`: The step size used by the algorithm in each step

        Parameters
        ----------
        num_results
            MCMC chain length (default 20000)
        num_burnin
            Number of burnin iterations (default 5000)
        num_adapt_steps
            Length of step size adaptation procedure
        num_leapfrog_steps
            HMC leapfrog steps (default 10)
        step_size
            Initial step size (default 0.01)

        Returns
        -------
        results object

        result
            Compositional analysis result
        """

        # HMC transition kernel
        hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self.target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps)
        hmc_kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=hmc_kernel, bijector=self.constraining_bijectors)

        # Set default value for adaptation steps if none given
        if num_adapt_steps is None:
            num_adapt_steps = int(0.8 * num_burnin)

        # Add step size adaptation (Andrieu, Thomas - 2008)
        hmc_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=hmc_kernel, num_adaptation_steps=num_adapt_steps, target_accept_prob=0.75)

        pbar = tfp.experimental.mcmc.ProgressBarReducer(num_results)
        hmc_kernel = tfp.experimental.mcmc.WithReductions(hmc_kernel, pbar)

        # diagnostics tracing function
        def trace_fn(_, pkr):
            return {
                'target_log_prob': pkr.inner_results.inner_results.inner_results.accepted_results.target_log_prob,
                'diverging': (pkr.inner_results.inner_results.inner_results.log_accept_ratio < -1000.),
                'is_accepted': pkr.inner_results.inner_results.inner_results.is_accepted,
                'step_size': pkr.inner_results.inner_results.inner_results.accepted_results.step_size,
            }

        # The actual HMC sampling process
        states, kernel_results, duration = self.sampling(num_results, num_burnin,
                                                         hmc_kernel, self.init_params, trace_fn)
        pbar.bar.close()
        # apply burn-in
        states_burnin, sample_stats, acc_rate = self.get_chains_after_burnin(states, kernel_results, num_burnin,
                                                                             is_nuts=False)

        # Calculate posterior predictive
        y_hat = self.get_y_hat(states_burnin, num_results, num_burnin)

        sampling_stats = {"chain_length": num_results, "num_burnin": num_burnin,
                          "acc_rate": acc_rate, "duration": duration, "y_hat": y_hat}

        result = self.make_result(states_burnin, sample_stats, sampling_stats)

        return result

    def sample_hmc_da(
            self,
            num_results: int = int(20e3),
            num_burnin: int = int(5e3),
            num_adapt_steps: Optional[int] = None,
            num_leapfrog_steps: Optional[int] = 10,
            step_size: float = 0.01
    ) -> res.CAResult:
        """
        HMC sampling with dual-averaging step size adaptation (Nesterov, 2009)

        Tracked diagnostic statistics:

        - `target_log_prob`: Value of the model's log-probability

        - `diverging`: Marks samples as diverging (NOTE: Handle with care, the spike-and-slab prior of scCODA usually leads to many samples being flagged as diverging)

        - `log_acc_ratio`: log-acceptance ratio

        - `is_accepted`: Whether the proposed sample was accepted in the algorithm's acceptance step

        - `step_size`: The step size used by the algorithm in each step

        Parameters
        ----------
        num_results
            MCMC chain length (default 20000)
        num_burnin
            Number of burnin iterations (default 5000)
        num_adapt_steps
            Length of step size adaptation procedure
        num_leapfrog_steps
            HMC leapfrog steps (default 10)
        step_size
            Initial step size (default 0.01)

        Returns
        -------
        result object

        result
            Compositional analysis result
        """

        # HMC transition kernel
        hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self.target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps)
        hmc_kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=hmc_kernel, bijector=self.constraining_bijectors)

        # Set default value for adaptation steps if none given
        if num_adapt_steps is None:
            num_adapt_steps = int(0.8 * num_burnin)

        # Add step size adaptation
        hmc_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=hmc_kernel, num_adaptation_steps=num_adapt_steps, target_accept_prob=0.75, decay_rate=0.75)

        pbar = tfp.experimental.mcmc.ProgressBarReducer(num_results)
        hmc_kernel = tfp.experimental.mcmc.WithReductions(hmc_kernel, pbar)

        # tracing function
        def trace_fn(_, pkr):
            return {
                'target_log_prob': pkr.inner_results.inner_results.inner_results.accepted_results.target_log_prob,
                'diverging': (pkr.inner_results.inner_results.inner_results.log_accept_ratio < -1000.),
                "log_acc_ratio": pkr.inner_results.inner_results.inner_results.log_accept_ratio,
                'is_accepted': pkr.inner_results.inner_results.inner_results.is_accepted,
                'step_size': tf.exp(pkr.inner_results.log_averaging_step[0]),
            }

        # HMC sampling
        states, kernel_results, duration = self.sampling(num_results, num_burnin, hmc_kernel, self.init_params,
                                                         trace_fn)
        states_burnin, sample_stats, acc_rate = self.get_chains_after_burnin(states, kernel_results, num_burnin,
                                                                             is_nuts=False)
        pbar.bar.close()

        y_hat = self.get_y_hat(states_burnin, num_results, num_burnin)

        sampling_stats = {"chain_length": num_results, "num_burnin": num_burnin,
                          "acc_rate": acc_rate, "duration": duration, "y_hat": y_hat}

        result = self.make_result(states_burnin, sample_stats, sampling_stats)

        return result

    def sample_nuts(
            self,
            num_results: int = int(10e3),
            num_burnin: int = int(5e3),
            num_adapt_steps: Optional[int] = None,
            max_tree_depth: int = 10,
            step_size: float = 0.01
    ) -> res.CAResult:
        """
        HMC with No-U-turn (NUTS) sampling.
        This method is untested and might yield different results than expected.

        Tracked diagnostic statistics:

        - `target_log_prob`: Value of the model's log-probability

        - `leapfrogs_taken`: Number of leapfrog steps taken by the integrator

        - `diverging`: Marks samples as diverging (NOTE: Handle with care, the spike-and-slab prior of scCODA usually leads to many samples being flagged as diverging)

        - `energy`: HMC "Energy" value for each step

        - `log_accept_ratio`: log-acceptance ratio

        - `step_size`: The step size used by the algorithm in each step

        - `reached_max_depth`: Whether the NUTS algorithm reached the maximum sampling depth in each step

        - `is_accepted`: Whether the proposed sample was accepted in the algorithm's acceptance step

        Parameters
        ----------
        num_results
            MCMC chain length (default 10000)
        num_burnin
            Number of burnin iterations (default 5000)
        num_adapt_steps
            Length of step size adaptation procedure
        max_tree_depth
            Maximum tree depth (default 10)
        step_size
            Initial step size (default 0.01)

        Returns
        -------
        result object

        result
            Compositional analysis result
        """

        # NUTS transition kernel
        nuts_kernel = tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=self.target_log_prob_fn,
            step_size=tf.cast(step_size, tf.float64),
            max_tree_depth=max_tree_depth)
        nuts_kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=nuts_kernel,
            bijector=self.constraining_bijectors
        )

        # Set default value for adaptation steps
        if num_adapt_steps is None:
            num_adapt_steps = int(0.8 * num_burnin)

        # Step size adaptation (Nesterov, 2009)
        nuts_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=nuts_kernel,
            num_adaptation_steps=num_adapt_steps,
            target_accept_prob=tf.cast(0.75, tf.float64),
            decay_rate=0.75,
            step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
                inner_results=pkr.inner_results._replace(step_size=new_step_size)
            ),
            step_size_getter_fn=lambda pkr: pkr.inner_results.step_size,
            log_accept_prob_getter_fn=lambda pkr: pkr.inner_results.log_accept_ratio,
        )

        pbar = tfp.experimental.mcmc.ProgressBarReducer(num_results)
        nuts_kernel = tfp.experimental.mcmc.WithReductions(nuts_kernel, pbar)

        # trace function
        def trace_fn(_, pkr):
            return {
                "target_log_prob": pkr.inner_results.inner_results.inner_results.target_log_prob,
                "leapfrogs_taken": pkr.inner_results.inner_results.inner_results.leapfrogs_taken,
                "diverging": pkr.inner_results.inner_results.inner_results.has_divergence,
                "energy": pkr.inner_results.inner_results.inner_results.energy,
                "log_accept_ratio": pkr.inner_results.inner_results.inner_results.log_accept_ratio,
                "step_size": pkr.inner_results.inner_results.inner_results.step_size,
                "reach_max_depth": pkr.inner_results.inner_results.inner_results.reach_max_depth,
                "is_accepted": pkr.inner_results.inner_results.inner_results.is_accepted,
            }

        # HMC sampling
        states, kernel_results, duration = self.sampling(num_results, num_burnin, nuts_kernel, self.init_params, trace_fn)
        states_burnin, sample_stats, acc_rate = self.get_chains_after_burnin(states, kernel_results, num_burnin,
                                                                             is_nuts=True)
        pbar.bar.close()


        y_hat = self.get_y_hat(states_burnin, num_results, num_burnin)

        sampling_stats = {"chain_length": num_results, "num_burnin": num_burnin,
                          "acc_rate": acc_rate, "duration": duration, "y_hat": y_hat}

        result = self.make_result(states_burnin, sample_stats, sampling_stats)

        return result

    def make_result(
            self,
            states_burnin,
            sample_stats: dict,
            sampling_stats: dict
    ):

        """
        Result object generating function. Transforms chain states to result object.

        Parameters
        ----------
        states_burnin
            MCMC chain states after burn-in removal
        sample_stats
            Dict with information about the MCMC samples
        sampling_stats
            Dict with information about the sampling process

        Returns
        -------
        result object

        result
            Compositional analysis result
        """

        params = dict(zip(self.param_names, states_burnin))

        # Result object generation process. Uses arviz's data structure.
        # Get names of cell types that are not the reference
        if self.reference_cell_type is not None:
            cell_types_nb = self.cell_types[:self.reference_cell_type] + self.cell_types[self.reference_cell_type + 1:]
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
                "sigma_d": ["covariate"],
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

        model_specs = {"reference": self.reference_cell_type, "formula": self.formula}

        return res.CAResultConverter(posterior=posterior,
                                     posterior_predictive=posterior_predictive,
                                     observed_data=observed_data,
                                     dims=dims,
                                     sample_stats=sample_stats,
                                     coords=coords).to_result_data(sampling_stats=sampling_stats,
                                                                   model_specs=model_specs)


class scCODAModel(CompositionalModel):
    """
    Statistical model for single-cell differential composition analysis with specification of a reference cell type.
    This is the standard scCODA model and recommenced for all uses.

    The hierarchical formulation of the model for one sample is:

    .. math::
         y|x &\\sim DirMult(\\phi, \\bar{y}) \\\\
         \\log(\\phi) &= \\alpha + x \\beta \\\\
         \\alpha_k &\\sim N(0, 5) \\quad &\\forall k \\in [K] \\\\
         \\beta_{m, \\hat{k}} &= 0 &\\forall m \\in [M]\\\\
         \\beta_{m, k} &= \\tau_{m, k} \\tilde{\\beta}_{m, k} \\quad &\\forall m \\in [M], k \\in \\{[K] \\smallsetminus \\hat{k}\\} \\\\
         \\tau_{m, k} &= \\frac{\\exp(t_{m, k})}{1+ \\exp(t_{m, k})} \\quad &\\forall m \\in [M], k \\in \\{[K] \\smallsetminus \\hat{k}\\} \\\\
         \\frac{t_{m, k}}{50} &\\sim N(0, 1) \\quad &\\forall m \\in [M], k \\in \\{[K] \\smallsetminus \\hat{k}\\} \\\\
         \\tilde{\\beta}_{m, k} &= \\sigma_m^2 \\cdot \\gamma_{m, k} \\quad &\\forall m \\in [M], k \\in \\{[K] \\smallsetminus \\hat{k}\\} \\\\
         \\sigma_m^2 &\\sim HC(0, 1) \\quad &\\forall m \\in [M] \\\\
         \\gamma_{m, k} &\\sim N(0,1) \\quad &\\forall m \\in [M], k \\in \\{[K] \\smallsetminus \\hat{k}\\} \\\\

    with y being the cell counts and x the covariates.

    For further information, see `scCODA: A Bayesian model for compositional single-cell data analysis`
    (Büttner, Ostner et al., 2020)

    """

    def __init__(
            self,
            reference_cell_type: int,
            *args,
            **kwargs):
        """
        Constructor of model class. Defines model structure, log-probability function, parameter names,
        and MCMC starting values.

        Parameters
        ----------
        reference_cell_type
            Index of reference cell type (column in count data matrix)
        args
            arguments passed to top-level class
        kwargs
            arguments passed to top-level class
        """

        super(self.__class__, self).__init__(*args, **kwargs)

        self.reference_cell_type = reference_cell_type
        dtype = tf.float64

        # All parameters that are returned for analysis
        self.param_names = ["sigma_d", "b_offset", "ind_raw", "alpha",
                            "ind", "b_raw", "beta", "concentration", "prediction"]

        alpha_size = [self.K]
        beta_size = [self.D, self.K]
        sigma_size = [self.D, 1]
        beta_nobl_size = [self.D, self.K-1]

        Root = tfd.JointDistributionCoroutine.Root

        def model():
            sigma_d = yield Root(tfd.Independent(
                tfd.HalfCauchy(tf.zeros(sigma_size, dtype=dtype),
                               tf.ones(sigma_size, dtype=dtype),
                               name="sigma_d"),
                reinterpreted_batch_ndims=2))

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
                    name="ind_raw"),
                reinterpreted_batch_ndims=2))

            ind_scaled = ind_raw * 50
            ind = tf.exp(ind_scaled) / (1 + tf.exp(ind_scaled))

            b_raw = sigma_d * b_offset

            beta = ind * b_raw

            # Include slope 0 for reference cell type
            beta = tf.concat(axis=1, values=[beta[:, :reference_cell_type],
                                             tf.zeros(shape=[self.D, 1], dtype=dtype),
                                             beta[:, reference_cell_type:]])

            alpha = yield Root(tfd.Independent(
                tfd.Normal(
                    loc=tf.zeros(alpha_size, dtype=dtype),
                    scale=tf.ones(alpha_size, dtype=dtype) * 5,
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
        @tf.function(experimental_compile=True)
        def target_log_prob_fn(*argsl):
            return self.model_struct.log_prob(list(argsl) + [tf.cast(self.y, dtype)])

        self.target_log_prob_fn = target_log_prob_fn

        # MCMC starting values
        self.init_params = [
            tf.ones(sigma_size, name="init_sigma_d", dtype=dtype),
            tf.random.normal(beta_nobl_size, 0, 1, name='init_b_offset', dtype=dtype),
            tf.zeros(beta_nobl_size, name='init_ind_raw', dtype=dtype),
            tf.random.normal(alpha_size, 0, 1, name='init_alpha', dtype=dtype)
        ]

        self.constraining_bijectors = [tfb.Identity() for x in range(len(self.init_params))]

    # Calculate predicted cell counts (for analysis purposes)
    def get_y_hat(
            self,
            states_burnin: List[any],
            num_results: int,
            num_burnin: int
    ) -> np.ndarray:
        """
        Calculate posterior mode of cell counts (for analysis purposes) and add intermediate parameters
        that are no priors to MCMC results.

        Parameters
        ----------
        states_burnin
            MCMC chain without burn-in samples
        num_results
            Chain length (with burn-in)
        num_burnin
            Number of burn-in samples

        Returns
        -------
        posterior mode

        y_mean
            posterior mode of cell counts
        """

        chain_size_y = [num_results - num_burnin, self.N, self.K]
        chain_size_beta = [num_results - num_burnin, self.D, self.K]

        alphas = states_burnin[3]
        alphas_final = alphas.mean(axis=0)

        ind_raw = states_burnin[2] * 50
        sigma_d = states_burnin[0]
        b_offset = states_burnin[1]

        ind_ = np.exp(ind_raw) / (1 + np.exp(ind_raw))

        b_raw_ = np.einsum("...jk, ...jl->...jk", b_offset, sigma_d)

        beta_temp = np.einsum("..., ...", ind_, b_raw_)

        beta_ = np.zeros(chain_size_beta)
        for i in range(num_results - num_burnin):
            beta_[i] = np.concatenate([beta_temp[i, :, :self.reference_cell_type],
                                       np.zeros(shape=[self.D, 1], dtype=np.float64),
                                       beta_temp[i, :, self.reference_cell_type:]], axis=1)
        conc_ = np.exp(np.einsum("jk, ...kl->...jl", self.x, beta_)
                       + alphas.reshape((num_results - num_burnin, 1, self.K)))

        predictions_ = np.zeros(chain_size_y)
        for i in range(num_results - num_burnin):
            pred = tfd.DirichletMultinomial(self.n_total, conc_[i, :, :]).mean().numpy()
            predictions_[i, :, :] = pred

        betas_final = beta_.mean(axis=0)
        states_burnin.append(ind_)
        states_burnin.append(b_raw_)
        states_burnin.append(beta_)
        states_burnin.append(conc_)
        states_burnin.append(predictions_)

        concentration = np.exp(np.matmul(self.x, betas_final) + alphas_final).astype(np.float64)

        y_mean = concentration / np.sum(concentration, axis=1, keepdims=True) * self.n_total.numpy()[:, np.newaxis]

        return y_mean
