"""
Models for the model comparison benchmark in `scCODA: A Bayesian model for compositional single-cell data analysis`
(Büttner, Ostner et al., 2020).

These models are otherwise not part of scCODA, but make a nice addition for comparison purposes
and are thus part of the main package.

:authors: Johannes Ostner
"""
import numpy as np
import pandas as pd
import os

import tensorflow as tf
import tensorflow_probability as tfp
import skbio
from tensorflow_probability.python.experimental import edward2 as ed
from anndata import AnnData

import statsmodels as sm
from statsmodels.formula.api import glm
from scipy import stats

from sccoda.util import result_classes as res
from sccoda.model import dirichlet_models as dm
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
        self.param_names = ["alpha", "b", "beta", "concentration", "prediction"]

        # Model definition
        def define_model(x, n_total, K):
            N, D = x.shape
            dtype = tf.float64

            alpha = ed.Normal(loc=tf.zeros([K], dtype=dtype), scale=tf.ones([K], dtype=dtype), name="alpha")
            b = ed.Normal(loc=tf.zeros([D, K-1], dtype=dtype), scale=tf.ones([D, K-1], dtype=dtype), name="b")

            beta = tf.concat(axis=1, values=[b[:, :reference_cell_type],
                                             tf.fill(value=0., dims=[D, 1]),
                                             b[:, reference_cell_type:]])

            concentration_ = tf.exp(alpha + tf.matmul(x, beta))

            # Likelihood
            predictions = ed.DirichletMultinomial(n_total, concentration=concentration_, name="predictions")
            return predictions

        # Joint posterior distribution
        self.log_joint = ed.make_log_joint_fn(define_model)
        # Function to compute log posterior probability

        self.target_log_prob_fn = lambda alpha_, b_: \
            self.log_joint(x=self.x,
                           n_total=self.n_total,
                           K=self.K,
                           predictions=self.y,
                           alpha=alpha_,
                           b=b_,
                           )

        alpha_size = [self.K]
        beta_size = [self.D, self.K-1]

        self.params = [tf.random.normal(mean=0, stddev=1, name="init_alpha", shape=alpha_size, dtype=dtype),
                       tf.random.normal(mean=0, stddev=1, name="init_b", shape=beta_size, dtype=dtype),
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

        # identity bijectors
        constraining_bijectors = [
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

        # Set default value for adaptation steps
        if num_adapt_steps is None:
            num_adapt_steps = int(0.8 * num_burnin)

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
        states, kernel_results, duration = self.sampling(num_results, num_burnin, hmc_kernel, self.params, trace_fn)

        # apply burnin
        states_burnin, sample_stats, acc_rate = self.get_chains_after_burnin(states, kernel_results, num_burnin)

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

        alphas = states_burnin[0]
        alphas_final = alphas.mean(axis=0)

        b = states_burnin[1]
        beta = np.zeros(chain_size_beta)
        for i in range(num_results - num_burnin):
            beta[i] = np.concatenate([b[i, :, :self.reference_cell_type],
                                      np.zeros(shape=[self.D, 1], dtype=np.float64),
                                      b[i, :, self.reference_cell_type:]], axis=1)

        betas_final = beta.mean(axis=0)

        conc_ = np.exp(np.einsum("jk, ...kl->...jl", self.x, beta)
                       + alphas.reshape((num_results - num_burnin, 1, self.K))).astype(np.float64)

        predictions_ = np.zeros(chain_size_y)
        for i in range(num_results - num_burnin):
            pred = tfd.DirichletMultinomial(self.n_total, conc_[i, :, :]).mean().numpy()
            predictions_[i, :, :] = pred

        states_burnin.append(beta)
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
            data: AnnData
    ):
        """
        Prepares R sampling

        Parameters
        ----------
        data
            scCODA data object
        """

        # prepare list generation
        n, k = data.X.shape
        x_vec = data.X.flatten()
        cell_types = ["cell_" + x for x in data.var.index.tolist()]
        cell_types[0] = "cell_" + str(k)
        conditions = ["Cond_0", "Cond_1"]

        # get number of samples for both conditionas
        ns_0 = int(sum(data.obs["x_0"]))
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
            server: bool = False
    ) -> Tuple[pd.DataFrame, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Analyzes results from R script for SCDC from scdney packege.
        It is assumed that the effect on the first cell type is significant, all others are not.

        Parameters
        ----------
        server
            Execution on server

        Returns
        -------
        summary and classification results

        Tuple
            Tuple(raw summary from R, True positive...)
        """
        if server:
            os.environ["R_HOME"] = "/home/icb/johannes.ostner/anaconda3/lib/R"
            os.environ["PATH"] = r"/home/icb/johannes.ostner/anaconda3/lib/R/bin" + ";" + os.environ["PATH"]
        else:
            os.environ["R_HOME"] = "C:/Program Files/R/R-4.0.3"
            if "C:/Program Files/R/R-4.0.3/bin/x64" not in os.environ["PATH"]:
                os.environ["PATH"] = r"C:/Program Files/R/R-4.0.3/bin/x64" + ";" + os.environ["PATH"]

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

            glm = fitGLM(clust, {rp.vectors.StrVector(self.scdc_sample_cond).r_repr()}, pairwise=FALSE)
            sum = summary(glm$pool_res_random)
            print(sum)
            sum
            """)

        r_summary = pd.DataFrame(r_summary)

        p_values = r_summary.loc[r_summary["term"].str.contains("condCond_1"), "p.value"].values

        tp = np.sum(p_values[-1] < 0.05)
        fn = np.sum(p_values[-1] >= 0.05)
        tn = np.sum(p_values[:-1] >= 0.05)
        fp = np.sum(p_values[:-1] < 0.05)

        return r_summary, (tp, tn, fp, fn)


class NonBaysesianModel:
    """
    Superclass for making non-Bayesian models from scCODA data.
    """

    def __init__(
            self,
            data: AnnData
    ):
        """
        Model initialization.

        Parameters
        ----------
        data
            CompositionalData object
        """

        x = data.obs.to_numpy()
        y = data.X
        self.var = data.var

        self.x = x
        self.y = y
        self.n_total = np.sum(y, axis=1)

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
            alpha: float = 0.05,
            fdr_correct: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluates array of p-values.
        It is assumed that the effect on the first cell type is significant, all others are not.

        Parameters
        ----------
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

        K = self.y.shape[1]
        ks = list(range(K))[1:]

        if fdr_correct:
            reject, pvals, _, _ = sm.stats.multitest.multipletests(self.p_val, alpha, method="fdr_bh")
            tp = sum([reject[0] == True])
            fn = sum([reject[0] == False])
            tn = sum([reject[k] == False for k in ks])
            fp = sum([reject[k] == True for k in ks])
        else:
            tp = sum([self.p_val[0] < alpha])
            fn = sum([self.p_val[0] >= alpha])
            tn = sum([self.p_val[k] >= alpha for k in ks])
            fp = sum([self.p_val[k] < alpha for k in ks])

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
                data_ct = pd.DataFrame({"x": self.x[:, 0],
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
            server: bool = False,
            *args,
            **kwargs
    ):
        """
        Fits ALDEx2 model.

        Parameters
        ----------
        method
            method that is used to calculate p-values (column name in ALDEx2's output)
        server
            indicator for remote execution
        args
            passed to `ALDEx2.clr`
        kwargs
            passed to `ALDEx2.clr`

        Returns
        -------

        """

        if server:
            os.environ["R_HOME"] = "/home/icb/johannes.ostner/anaconda3/lib/R"
            os.environ["PATH"] = r"/home/icb/johannes.ostner/anaconda3/lib/R/bin" + ";" + os.environ["PATH"]
        else:
            os.environ["R_HOME"] = "C:/Program Files/R/R-4.0.3"
            os.environ["PATH"] = r"C:/Program Files/R/R-4.0.3/bin/x64" + ";" + os.environ["PATH"]

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

            cond = rp.vectors.FloatVector(self.x.astype("str").flatten().tolist())

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
            data: AnnData
    ):
        """
        Model initialization.

        Parameters
        ----------
        data
            CompositionalData object
        """
        x = data.obs
        y = pd.DataFrame(data.X, index=data.obs.index)
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

    def fit_model(self):
        """
        Fits ancom model

        Returns
        -------
        p-values

        p_val
            p-values for differential abundance test of all cell types
        """

        K = self.y.shape[1]

        if self.y.shape[0] == 2:
            ancom_out = [False for _ in range(K)]
        else:
            ancom_out = skbio.stats.composition.ancom(self.y, self.x["x_0"])

        self.ancom_out = ancom_out

    def eval_model(self) -> Tuple[int, int, int, int]:
        """
        Evaluates array of p-values.
        It is assumed that the effect on the first cell type is significant, all others are not.

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

        ks = list(range(K))[1:]

        tp = sum([accept[0]])
        fn = sum([accept[0] is False])
        tn = sum([accept[k] is False for k in ks])
        fp = sum([accept[k] for k in ks])

        return tp, tn, fp, fn


class DirichRegModel(NonBaysesianModel):

    """
    Wrapper for using the DirichReg package in R (Maier, 2014) with scCODA's infrastructure
    """

    def fit_model(
            self,
            server: bool = False
    ):

        """
        fits the DirichReg model.

        Parameters
        ----------
        server
            indicator for remote execution

        Returns
        -------

        """

        if server:
            os.environ["R_HOME"] = "/home/icb/johannes.ostner/anaconda3/lib/R"
            os.environ["PATH"] = r"/home/icb/johannes.ostner/anaconda3/lib/R/bin" + ";" + os.environ["PATH"]
        else:
            os.environ["R_HOME"] = "C:/Program Files/R/R-4.0.3"
            if "C:/Program Files/R/R-4.0.3/bin/x64" not in os.environ["PATH"]:
                os.environ["PATH"] = r"C:/Program Files/R/R-4.0.3/bin/x64" + ";" + os.environ["PATH"]

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
            data = cbind(counts, {pandas2ri.py2rpy_pandasdataframe(pd.DataFrame(self.x, columns=["x_0"])).r_repr()})

            fit = DirichReg(counts ~ x_0, data)
            if(fit$optimization$convergence > 2L) {{
            pvals = matrix(c(0,0,0,0,0),nrow = 1)
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
