"""
Models for the model comparison benchmark in `scCODA: A Bayesian model for compositional single-cell data analysis`
(Büttner, Ostner et al., 2020).

These models are otherwise not part of scCODA, but make a nice addition for comparison purposes
and are thus part of the main package.

:authors: Johannes Ostner, Maren Büttner
"""
import numpy as np
import pandas as pd
import os

import tensorflow as tf
import tensorflow_probability as tfp
from skbio.stats.composition import ancom
from anndata import AnnData

import statsmodels as sm
from statsmodels.formula.api import glm
from scipy import stats

from sccoda.util import result_classes as res
from sccoda.model import scCODA_model as dm
from typing import Optional, Tuple, Collection, Union, List

tfd = tfp.distributions
tfb = tfp.bijectors


class SimpleModel(dm.CompositionalModel):
    """
    Simple Dirichlet-Multinomial model with normal priors. Structure equivalent to scCODA's other models.

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
        self.param_names = ["b", "alpha", "beta", "concentration", "prediction"]

        alpha_size = [self.K]
        beta_size = [self.D, self.K]
        beta_nobl_size = [self.D, self.K-1]

        Root = tfd.JointDistributionCoroutine.Root

        def model():

            beta = yield Root(tfd.Independent(
                tfd.Normal(
                    loc=tf.zeros(beta_nobl_size, dtype=dtype),
                    scale=tf.ones(beta_nobl_size, dtype=dtype),
                    name="b"),
                reinterpreted_batch_ndims=2))

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
        self.target_log_prob_fn = lambda *args:\
            self.model_struct.log_prob(list(args) + [tf.cast(self.y, dtype)])

        self.init_params = [
            tf.random.normal(beta_nobl_size, 0, 1, name="init_b", dtype=dtype),
            tf.random.normal(alpha_size, 0, 1, name="init_alpha", dtype=dtype),
        ]

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

        # bijectors (not in use atm, therefore identity)
        constraining_bijectors = [tfb.Identity() for x in range(len(self.init_params))]

        # HMC transition kernel
        hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self.target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps)
        hmc_kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=hmc_kernel, bijector=constraining_bijectors)

        # Set default value for adaptation steps if none given
        if num_adapt_steps is None:
            num_adapt_steps = int(0.8 * num_burnin)

        # Add step size adaptation (Andrieu, Thomas - 2008)
        hmc_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=hmc_kernel, num_adaptation_steps=num_adapt_steps, target_accept_prob=0.8)

        # diagnostics tracing function
        def trace_fn(_, pkr):
            return {
                'target_log_prob': pkr.inner_results.inner_results.accepted_results.target_log_prob,
                'diverging': (pkr.inner_results.inner_results.log_accept_ratio < -1000.),
                'is_accepted': pkr.inner_results.inner_results.is_accepted,
                'step_size': pkr.inner_results.inner_results.accepted_results.step_size,
            }

        # The actual HMC sampling process
        states, kernel_results, duration = self.sampling(num_results, num_burnin,
                                                         hmc_kernel, self.init_params, trace_fn)

        # apply burn-in
        states_burnin, sample_stats, acc_rate = self.get_chains_after_burnin(states, kernel_results, num_burnin,
                                                                             is_nuts=False)

        # Calculate posterior predictive
        y_hat = self.get_y_hat(states_burnin, num_results, num_burnin)

        params = dict(zip(self.param_names, states_burnin))

        cell_types_nb = self.cell_types[:self.reference_cell_type] + self.cell_types[self.reference_cell_type + 1:]

        posterior = {var_name: [var] for var_name, var in params.items() if
                     "prediction" not in var_name}
        posterior_predictive = {"prediction": [params["prediction"]]}
        observed_data = {"y": self.y}
        dims = {"alpha": ["cell_type"],
                "b": ["covariate", "cell_type_nb"],
                "beta": ["covariate", "cell_type"],
                "concentration": ["sample", "cell_type"],
                "prediction": ["sample", "cell_type"]
                }
        coords = {"cell_type": self.cell_types,
                  "cell_type_nb": cell_types_nb,
                  "covariate": self.covariate_names,
                  "sample": range(self.y.shape[0])
                  }

        sampling_stats = {"chain_length": num_results, "num_burnin": num_burnin,
                          "acc_rate": acc_rate, "duration": duration, "y_hat": y_hat}

        model_specs = {"reference": self.reference_cell_type, "formula": self.formula}

        return res.CAResultConverter(posterior=posterior,
                                     posterior_predictive=posterior_predictive,
                                     observed_data=observed_data,
                                     dims=dims,
                                     sample_stats=sample_stats,
                                     coords=coords).to_result_data(sampling_stats=sampling_stats,
                                                                   model_specs=model_specs)

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
        chain_size_beta = [num_results - num_burnin, self.D, self.K]
        chain_size_y = [num_results - num_burnin, self.N, self.K]

        alphas = states_burnin[1]
        alphas_final = alphas.mean(axis=0)

        b = states_burnin[0]
        beta_ = np.zeros(chain_size_beta)
        for i in range(num_results - num_burnin):
            beta_[i] = np.concatenate([b[i, :, :self.reference_cell_type],
                                      np.zeros(shape=[self.D, 1], dtype=np.float64),
                                      b[i, :, self.reference_cell_type:]], axis=1)

        betas_final = beta_.mean(axis=0)

        conc_ = np.exp(np.einsum("jk, ...kl->...jl", self.x, beta_)
                       + alphas.reshape((num_results - num_burnin, 1, self.K))).astype(np.float64)

        predictions_ = np.zeros(chain_size_y)
        for i in range(num_results - num_burnin):
            pred = tfd.DirichletMultinomial(self.n_total, conc_[i, :, :]).mean().numpy()
            predictions_[i, :, :] = pred

        states_burnin.append(beta_)
        states_burnin.append(conc_)
        states_burnin.append(predictions_)

        concentration = np.exp(np.matmul(self.x, betas_final) + alphas_final).astype(np.float64)
        y_mean = concentration / np.sum(concentration, axis=1, keepdims=True) * self.n_total.numpy()[:, np.newaxis]
        return y_mean


class scdney_model:
    """
    wrapper for using the scdney package for R (Cao et al., 2019) with scCODA data
    """

    def __init__(
            self,
            data: AnnData,
            covariate_column: str = "x_0",
    ):
        """
        Prepares R sampling

        Parameters
        ----------
        data
            scCODA data object
        covariate_column: str
            Name of the covariate column in `data.obs`
        """

        # prepare list generation
        n, k = data.X.shape
        self.k = k
        x_vec = data.X.flatten()
        cell_types = ["cell_" + x for x in data.var.index.tolist()]
        # cell_types[0] = "cell_" + str(k)
        conditions = ["Cond_0", "Cond_1"]

        # get number of samples for both conditions
        ns_0 = int(sum(pd.factorize(data.obs[covariate_column])[0] == 0))
        ns = [ns_0, n-ns_0]

        subjects = []
        for n in range(ns[0]):
            subjects.append("Cond_0_sub_" + str(n))
        for n in range(ns[1]):
            subjects.append("Cond_1_sub_" + str(n))

        # produce lists to use in scdney
        self.scdc_celltypes = []
        self.scdc_subject = []
        self.scdc_cond = []
        self.scdc_sample_cond = []

        for i in range(len(x_vec)):
            current_count = x_vec[i]
            current_type = cell_types[i % k]
            current_subject = subjects[i // k]
            current_condition = conditions[i // (k * ns[0])]

            self.scdc_sample_cond.append(current_condition)

            for j in range(int(current_count)):
                self.scdc_celltypes.append(current_type)
                self.scdc_subject.append(current_subject)
                self.scdc_cond.append(current_condition)

    def analyze(
            self,
            ground_truth: np.array = None,
            r_home: str = "",
            r_path: str = r"",
            alpha: float = 0.05,
    ) -> Tuple[pd.DataFrame, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Analyzes results from R script for SCDC from scdney packege.
        It is assumed that the effect on the first cell type is significant, all others are not.

        Parameters
        ----------
        ground_truth
            binary array for comparison to ground truth
        r_home
            path to R installation on your machine, e.g. "C:/Program Files/R/R-4.0.3"
        r_path
            path to R executable on your machine, e.g. "C:/Program Files/R/R-4.0.3/bin/x64"
        alpha
            p-value cutoff


        Returns
        -------
        summary and classification results

        Tuple
            Tuple(raw summary from R, True positive...)
        """

        os.environ["R_HOME"] = r_home
        os.environ["PATH"] = r_path + ";" + os.environ["PATH"]

        if ground_truth is None:
            ground_truth = np.zeros(self.k)

        import rpy2.robjects as rp
        from rpy2.robjects import numpy2ri, pandas2ri
        numpy2ri.activate()
        pandas2ri.activate()

        r_summary = rp.r(f"""
            library(scdney)
            library(tidyverse)
            library(broom.mixed)
            clust = scDC_noClustering({rp.vectors.StrVector(self.scdc_celltypes).r_repr()}, 
                                      {rp.vectors.StrVector(self.scdc_subject).r_repr()},
                                             calCI=TRUE,
                                             calCI_method=c("BCa"),
                                             nboot=100)

            glm = fitGLM(clust, {rp.vectors.StrVector(self.scdc_sample_cond).r_repr()}, pairwise=FALSE, subject_effect=FALSE)
            sum = summary(glm$pool_res_fixed)
            sum
            """)

        r_summary = pd.DataFrame(r_summary)

        p_values = r_summary.loc[r_summary["term"].str.contains("condCond_1"), "p.value"].values

        true_indices = np.where(ground_truth == True)[0]
        false_indices = np.where(ground_truth == False)[0]

        pval = np.nan_to_num(np.array(p_values), nan=1)
        tp = sum(pval[true_indices] < alpha)
        fn = sum(pval[true_indices] >= alpha)
        tn = sum(pval[false_indices] >= alpha)
        fp = sum(pval[false_indices] < alpha)

        return r_summary, (tp, tn, fp, fn)


class NonBaysesianModel:
    """
    Superclass for making non-Bayesian models from scCODA data.
    """

    def __init__(
            self,
            data: AnnData,
            covariate_column: Optional[str] = "x_0",
    ):
        """
        Model initialization.

        Parameters
        ----------
        data
            CompositionalData object
        covariate_column
            Name of the covariate column in `data.obs`
        """

        x = data.obs.loc[:, covariate_column].to_numpy()
        y = data.X
        y[y == 0] = 1
        self.var = data.var

        self.x = x
        self.y = y
        self.n_total = np.sum(y, axis=1)
        self.covariate_column = covariate_column

        self.p_val = {}

        # Get dimensions of data
        N = y.shape[0]

        # Check input data
        if N != x.shape[0]:
            raise ValueError("Wrong input dimensions X[{},:] != y[{},:]".format(y.shape[0], x.shape[0]))
        if N != len(self.n_total):
            raise ValueError("Wrong input dimensions X[{},:] != n_total[{}]".format(y.shape[0], len(self.n_total)))

    def eval_model(
            self,
            ground_truth: List,
            alpha: float = 0.05,
            fdr_correct: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluates array of p-values compared to a ground truth via binary classification.

        Parameters
        ----------
        ground_truth
            List (boolean, length same as number of cell types) indicating differential abundance for each cell type
        alpha
            p-value (or q-value if using FDR correction) threshold
        fdr_correct
            Whether to use Benjamini-Hochberg FDR correction for multiple testing

        Returns
        -------
        classification results

        tp, tn, fp, fn
            Number of True positive, ... effects
        """

        true_indices = np.where(ground_truth == True)[0]
        false_indices = np.where(ground_truth == False)[0]

        if fdr_correct:
            pval = np.nan_to_num(np.array(self.p_val), nan=1)
            reject, pvals, _, _ = sm.stats.multitest.multipletests(pval, alpha, method="fdr_bh")
            tp = sum(reject[true_indices] == True)
            fn = sum(reject[true_indices] == False)
            tn = sum(reject[false_indices] == False)
            fp = sum(reject[false_indices] == True)
        else:
            pval = np.nan_to_num(np.array(self.p_val), nan=1)
            tp = sum(pval[true_indices] < alpha)
            fn = sum(pval[true_indices] >= alpha)
            tn = sum(pval[false_indices] >= alpha)
            fp = sum(pval[false_indices] < alpha)

        return tp, tn, fp, fn


class HaberModel(NonBaysesianModel):
    """
    Implements the Poisson regression model from Haber et al.
    """

    def fit_model(self):
        """
        Fits Poisson model

        Returns
        -------
        p_val
            p-values for differential abundance test of all cell types
        """

        p_val = []
        K = self.y.shape[1]

        if self.y.shape[0] == 2:
            p_val = [0 for _ in range(K)]
        else:
            for k in range(K):
                if len(self.x.shape) == 1:
                    x_ = self.x
                else:
                    x_ = self.x[:, 0]
                data_ct = pd.DataFrame({"x": x_,
                                        "y": self.y[:, k]})

                model_ct = glm('y ~ x', data=data_ct,
                               family=sm.genmod.families.Poisson(), offset=np.log(self.n_total)).fit()
                p_val.append(model_ct.pvalues[1])

        self.p_val = p_val


class CLRModel(NonBaysesianModel):
    """
    Implements a CLR transform and subsequent linear model on each cell type.
    """

    def fit_model(self):
        """
        Fits CLR model with linear model

        Returns
        -------
        p_val
            p-values for differential abundance test of all cell types
        """

        p_val = []
        K = self.y.shape[1]

        if self.y.shape[0] == 2:
            p_val = [0 for _ in range(K)]
        else:
            # computes clr-transformed data matrix as a pandas DataFrame
            geom_mean = np.prod(self.y, axis=1, keepdims=True) ** (1 / K)
            y_clr = np.log(self.y / geom_mean)

            for k in range(K):
                data_ct = pd.DataFrame({"x": self.x[:, 0],
                                        "y": y_clr[:, k]})

                model_ct = glm('y ~ x', data=data_ct).fit()
                p_val.append(model_ct.pvalues[1])

        self.p_val = p_val


class TTest(NonBaysesianModel):
    """
        Implements a t-test on each cell type.
    """

    def fit_model(self):
        """
        Fits t-test model

        Returns
        -------
        p_val
            p-values for differential abundance test of all cell types
        """

        p_val = []
        N, K = self.y.shape

        n_group = int(N/2)

        if self.y.shape[0] == 2:
            p_val = [0 for _ in range(K)]
        else:
            for k in range(K):

                test = stats.ttest_ind(self.y[0:n_group, k], self.y[n_group:, k])
                p_val.append(test[1])

        self.p_val = p_val


class CLRModel_ttest(NonBaysesianModel):
    """
    Implements a CLR transform and subsequent t-test on each cell type.
    """

    def fit_model(self):
        """
        Fits CLR model with t-test

        Returns
        -------
        p_val
            p-values for differential abundance test of all cell types
        """

        p_val = []
        N, K = self.y.shape

        n_group = int(N/2)

        if self.y.shape[0] == 2:
            p_val = [0 for _ in range(K)]
        else:
            # computes clr-transformed data matrix as a pandas DataFrame
            geom_mean = np.prod(self.y, axis=1, keepdims=True) ** (1 / K)
            y_clr = np.log(self.y / geom_mean)

            for k in range(K):
                test = stats.ttest_ind(y_clr[0:n_group, k], y_clr[n_group:, k])
                p_val.append(test[1])

        self.p_val = p_val


class ALDEx2Model(NonBaysesianModel):
    """
    Wrapper for using the ALDEx2 package for R (Fernandes et al., 2014)
    """

    def fit_model(
            self,
            method: str = "we.eBH",
            r_home: str = "",
            r_path: str = r"",
            *args,
            **kwargs
    ):
        """
        Fits ALDEx2 model.

        Parameters
        ----------
        method
            method that is used to calculate p-values (column name in ALDEx2's output)
        r_home
            path to R installation on your machine, e.g. "C:/Program Files/R/R-4.0.3"
        r_path
            path to R executable on your machine, e.g. "C:/Program Files/R/R-4.0.3/bin/x64"
        args
            passed to `ALDEx2.clr`
        kwargs
            passed to `ALDEx2.clr`

        Returns
        -------

        """

        os.environ["R_HOME"] = r_home
        os.environ["PATH"] = r_path + ";" + os.environ["PATH"]

        K = self.y.shape[1]

        if self.y.shape[0] == 2:
            p_val = [0 for _ in range(K)]
            self.result = None
        else:

            import rpy2.robjects as rp
            from rpy2.robjects import numpy2ri, pandas2ri
            numpy2ri.activate()
            pandas2ri.activate()
            import rpy2.robjects.packages as rpackages
            aldex2 = rpackages.importr("ALDEx2")

            x_fact = pd.factorize(self.x)[0]

            cond = rp.vectors.FloatVector(x_fact.astype("str").flatten().tolist())

            X_t = self.y.T
            nr, nc = X_t.shape
            X_r = rp.r.matrix(X_t, nrow=nr, ncol=nc)

            if "denom" in kwargs.keys():
                kwargs["denom"] = rp.vectors.FloatVector(kwargs["denom"])

            aldex_out = aldex2.aldex_clr(X_r, cond, *args, **kwargs)
            aldex_out = aldex2.aldex_ttest(aldex_out)
            aldex_out = pd.DataFrame(aldex_out)

            p_val = aldex_out.loc[:, method]

            self.result = aldex_out

        self.p_val = p_val


class ALRModel_ttest(NonBaysesianModel):
    """
    Implements a ALR transform and subsequent t-test on each cell type.
    """

    def fit_model(
            self,
            reference_cell_type: int
    ):
        """
        Fits ALR model with t-test

        Parameters
        ----------
        reference_cell_type
            index of reference cell type

        Returns
        -------
        p-values

        p_val
            p-values for differential abundance test of all cell types
        """

        p_val = []
        N, K = self.y.shape

        n_group = int(N/2)

        if self.y.shape[0] == 2:
            p_val = [0 for _ in range(K)]
        else:
            # computes alr-transformed data matrix as a pandas DataFrame
            y_alr = np.log(self.y / self.y[:, reference_cell_type][:, np.newaxis])

            for k in range(K):
                test = stats.ttest_ind(y_alr[0:n_group, k], y_alr[n_group:, k])
                p = test[1]
                if np.isnan(p):
                    p = 1
                p_val.append(p)

        self.p_val = p_val


class ALRModel_wilcoxon(NonBaysesianModel):
    """
    Implements a ALR transform and subsequent Wilcoxon rank-sum test on each cell type.
    """

    def fit_model(
            self,
            reference_cell_type: int
    ):
        """
        Fits ALR model with Wilcoxon rank-sum test

        Parameters
        ----------
        reference_cell_type
            index of reference cell type

        Returns
        -------
        p-values

        p_val
            p-values for differential abundance test of all cell types
        """

        p_val = []
        N, K = self.y.shape

        n_group = int(N/2)

        if self.y.shape[0] == 2:
            p_val = [0 for _ in range(K)]
        else:
            # computes alr-transformed data matrix as a pandas DataFrame
            y_alr = np.log(self.y / self.y[:, reference_cell_type][:, np.newaxis])

            for k in range(K):
                test = stats.ranksums(y_alr[0:n_group, k], y_alr[n_group:, k])
                p = test[1]
                if np.isnan(p):
                    p = 1
                p_val.append(p)

        self.p_val = p_val


class AncomModel():
    """
        Wrapper for the ancom model for compositional differentiation analysis (Mandal et al., 2015)
    """

    def __init__(
            self,
            data: AnnData,
            covariate_column: Optional[str] = "x_0",
    ):
        """
        Model initialization.

        Parameters
        ----------
        data
            CompositionalData object
        covariate_column
            Column with the (binary) trait
        """
        x = data.obs.loc[:, covariate_column]
        y = data.X
        y[y == 0] = 0.5
        y = pd.DataFrame(y, index=data.obs.index, columns=data.var.index)
        self.x = x
        self.y = y
        self.n_total = np.sum(y, axis=1)

        self.ancom_out = []

        # Get dimensions of data
        N = y.shape[0]

        # Check input data
        if N != x.shape[0]:
            raise ValueError("Wrong input dimensions X[{},:] != y[{},:]".format(y.shape[0], x.shape[0]))
        if N != len(self.n_total):
            raise ValueError("Wrong input dimensions X[{},:] != n_total[{}]".format(y.shape[0], len(self.n_total)))

    def fit_model(
            self,
            alpha: float = 0.05,
            tau: float = 0.02,
            *args,
            **kwargs,
    ):
        """

        Parameters
        ----------
        alpha
            FDR level for multiplicity correction
        tau
            cutoff parameter
        args
            passed to skbio.stats.composition.ancom
        kwargs
            passed to skbio.stats.composition.ancom

        Returns
        -------

        """

        K = self.y.shape[1]

        if self.y.shape[0] == 2:
            ancom_out = [False for _ in range(K)]
        else:
            ancom_out = ancom(self.y, self.x, alpha=alpha, tau=tau, *args, **kwargs)

        self.ancom_out = ancom_out

    def eval_model(
            self,
            ground_truth: List
    ) -> Tuple[int, int, int, int]:
        """
        Evaluates array of results for ancom compared to a ground tuth via binary classification.

        Parameters
        ----------

        ground_truth
            List (boolean, length same as number of cell types) indicating differential abundance for each cell type

        Returns
        -------
        classification results

        tp, tn, fp, fn
            Number of True positive, ... effects
        """
        K = self.y.shape[1]

        if self.y.shape[0] == 2:
            accept = [False for _ in range(K)]
        else:
            accept = self.ancom_out[0]["Reject null hypothesis"].tolist()

        true_indices = np.where(ground_truth == True)[0]
        false_indices = np.where(ground_truth == False)[0]

        accept = np.array(accept)

        tp = sum(accept[true_indices] == True)
        fn = sum(accept[true_indices] == False)
        tn = sum(accept[false_indices] == False)
        fp = sum(accept[false_indices] == True)

        return tp, tn, fp, fn


class DirichRegModel(NonBaysesianModel):

    """
    Wrapper for using the DirichReg package in R (Maier, 2014) with scCODA's infrastructure
    """

    def fit_model(
            self,
            r_home: str = "",
            r_path: str = r"",
    ):

        """
        fits the DirichReg model.

        Parameters
        ----------
        r_home
            path to R installation on your machine, e.g. "C:/Program Files/R/R-4.0.3"
        r_path
            path to R executable on your machine, e.g. "C:/Program Files/R/R-4.0.3/bin/x64"

        Returns
        -------

        """

        os.environ["R_HOME"] = r_home
        os.environ["PATH"] = r_path + ";" + os.environ["PATH"]

        K = self.y.shape[1]

        if self.y.shape[0] == 2:
            p_val = [0 for _ in range(K)]
            self.result = None
        else:

            import rpy2.robjects as rp
            from rpy2.robjects import numpy2ri, pandas2ri
            numpy2ri.activate()
            pandas2ri.activate()

            p_val = rp.r(f"""
            library(DirichletReg)

            counts = {pandas2ri.py2rpy_pandasdataframe(pd.DataFrame(self.y, columns=self.var.index)).r_repr()}
            counts$counts = DR_data(counts)
            data = cbind(counts, {pandas2ri.py2rpy_pandasdataframe(pd.DataFrame(self.x, columns=[self.covariate_column])).r_repr()})

            fit = DirichReg(counts ~ {self.covariate_column}, data)
            if(fit$optimization$convergence > 2L) {{
            pvals = matrix(rep(0, {K}),nrow = 1)
            }} else {{
            u = summary(fit)
            pvals = u$coef.mat[grep('Intercept', rownames(u$coef.mat), invert=T), 4]
            v = names(pvals)
            pvals = matrix(pvals, ncol=length(u$varnames))
            rownames(pvals) = gsub('condition', '', v[1:nrow(pvals)])
            colnames(pvals) = u$varnames
            }}
            pvals
            """)
            p_val = p_val[0]

        self.p_val = p_val


class BetaBinomialModel(NonBaysesianModel):
    """
    Wrapper for using the corncob package for R (Martin et al., 2020)
    """

    def fit_model(
            self,
            r_home: str = "",
            r_path: str = r"",
    ):
        """
        Fits Beta-Binomial model.

        Parameters
        ----------
        method
            method that is used to calculate p-values 
        r_home
            path to R installation on your machine, e.g. "C:/Program Files/R/R-4.0.3"
        r_path
            path to R executable on your machine, e.g. "C:/Program Files/R/R-4.0.3/bin/x64"
        Returns
        -------
        """

        os.environ["R_HOME"] = r_home
        os.environ["PATH"] = r_path + ";" + os.environ["PATH"]

        K = self.y.shape[1]

        if self.y.shape[0] == 2:
            p_val = [0 for _ in range(K)]
            self.result = None
        else:

            import rpy2.robjects as rp
            from rpy2.robjects import numpy2ri, pandas2ri
            numpy2ri.activate()
            pandas2ri.activate()

            if self.y.shape[0] == 4:
                phi = 1
            else:
                phi = self.covariate_column
            
            p_val = rp.r(f"""
            library(corncob)
            library(phyloseq)
            
            
            #prepare phyloseq data format
            
            counts = {pandas2ri.py2rpy_pandasdataframe(pd.DataFrame(self.y, columns=self.var.index)).r_repr()}
            
            sample = {pandas2ri.py2rpy_pandasdataframe(pd.DataFrame(self.x, columns=[self.covariate_column])).r_repr()}
            
            cell_types = colnames(counts)
            
            OTU = otu_table(counts, taxa_are_rows = FALSE)
            
            #create phyloseq data object
            data = phyloseq(OTU, sample_data(sample))
            
            corncob_out = differentialTest(formula = ~ {self.covariate_column},
                                  phi.formula = ~ {phi},
                                  formula_null = ~ 1,
                                  phi.formula_null = ~ {phi},
                                  test = "LRT",
                                  boot = FALSE,
                                  data = data,
                                  fdr_cutoff = 0.05
                                  )
            
             p_vals = corncob_out$p_fdr 
            
             p_vals
            """)

        self.p_val = p_val


class ANCOMBCModel(NonBaysesianModel):
    """
    Wrapper for using the ANCOMBC package for R (Lin and Peddada, 2020)
    """

    def fit_model(
            self,
            method: str = "fdr",
            lib_cut: int = 0,
            r_home: str = "",
            r_path: str = r"",
            alpha: float = 0.05,
            zero_cut: float = 0.9,
    ):
        """
        Fits ANCOM with bias correction model.

        Parameters
        ----------
        method
            method that is used to calculate p-values 
        lib_cut
            threshold to filter out classes
        r_home
            path to R installation on your machine, e.g. "C:/Program Files/R/R-4.0.3"
        r_path
            path to R executable on your machine, e.g. "C:/Program Files/R/R-4.0.3/bin/x64"
        alpha
            Nominal FDR value
        zero_cut
            Prevalence cutoff for cell types (cell types with higher percentage of zero entries are dropped)

        Returns
        -------
        """

        os.environ["R_HOME"] = r_home
        os.environ["PATH"] = r_path + ";" + os.environ["PATH"]

        K = self.y.shape[1]

        if self.y.shape[0] == 2:
            p_val = [0 for _ in range(K)]
            self.result = None
        else:

            import rpy2.robjects as rp
            from rpy2.robjects import numpy2ri, pandas2ri
            numpy2ri.activate()
            pandas2ri.activate()
            
            p_val = rp.r(f"""
            library(ANCOMBC)
            library(phyloseq)
            
            #prepare phyloseq data format
            
            counts = {pandas2ri.py2rpy_pandasdataframe(pd.DataFrame(self.y, columns=self.var.index)).r_repr()}
            
            sample = {pandas2ri.py2rpy_pandasdataframe(pd.DataFrame(self.x, 
                                   columns=[self.covariate_column])).r_repr()}
           
            cell_types = colnames(counts)
           
            OTU = otu_table(t(counts), taxa_are_rows = TRUE)
            
            #create phyloseq data object
            data = phyloseq(OTU, sample_data(sample))
           
            ancombc_out = ancombc(phyloseq = data,            
                                  formula = "{self.covariate_column}",
                                  p_adj_method = "{method}", 
                                  zero_cut = {zero_cut}, 
                                  lib_cut = {lib_cut}, 
                                  group = "{self.covariate_column}", 
                                  struc_zero = TRUE, 
                                  neg_lb = TRUE, tol = 1e-5, 
                                  max_iter = 100, 
                                  conserve = TRUE, 
                                  alpha = {alpha}, 
                                  global = FALSE
                                  )
            
            out = ancombc_out$res
            #return adjusted p-values
            p_vals = out$q[,1] 
            
            p_vals
            """)

        self.p_val = p_val
