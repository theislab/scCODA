"""
Results class that summarizes the results of scCODA and calculates test statistics.
This class extends the ´´InferenceData`` class in the ``arviz`` package and can use all plotting and diacgnostic
functionalities of it.

Additionally, this class can produce nicely readable outputs for scCODA.

:authors: Johannes Ostner
"""
import numpy as np
import arviz as az
import pandas as pd
import pickle as pkl

from typing import Optional, Tuple, Collection, Union, List


class CAResultConverter(az.data.io_dict.DictConverter):
    """
    Helper class for result conversion into arviz's format
    """

    def to_result_data(self, sampling_stats, model_specs):

        post = self.posterior_to_xarray()
        ss = self.sample_stats_to_xarray()
        postp = self.posterior_predictive_to_xarray()
        prior = self.prior_to_xarray()
        ssp = self.sample_stats_prior_to_xarray()
        prip = self.prior_predictive_to_xarray()
        obs = self.observed_data_to_xarray()

        return CAResult(
            sampling_stats, model_specs,
            **{
                "posterior": post,
                "sample_stats": ss,
                "posterior_predictive": postp,
                "prior": prior,
                "sample_stats_prior": ssp,
                "prior_predictive": prip,
                "observed_data": obs,
            }
        )


class CAResult(az.InferenceData):
    """
    Result class for scCODA, extends the arviz framework for inference data.

    The CAResult class is an extension of az.InferenceData, that adds some information about the compositional model
    and is able to print humanly readable results.
    It supports all functionality from az.InferenceData.
    """

    def __init__(
            self,
            sampling_stats: dict,
            model_specs: dict,
            **kwargs
    ):
        """
        Gathers sampling information from a compositional model and converts it to a ``az.InferenceData`` object.
        The following attributes are added during class initialization:

        ``self.sampling_stats``: dict - see below
        ``self.model_specs``: dict - see below

        ``self.intercept_df``: Intercept dataframe from ``CAResult.summary_prepare``
        ``self.effect_df``: Effect dataframe from ``CAResult.summary_prepare``

        Parameters
        ----------
        sampling_stats
            Information and statistics about the MCMC sampling procedure.
            Default keys:
            - "chain_length": Length of MCMC chain (with burnin samples)
            - "num_burnin": Number of burnin samples
            - "acc_rate": MCMC Acceptance rate
            - "duration": Duration of MCMC sampling

        model_specs
            All information and statistics about the model specifications.
            Default keys:
            - "formula": Formula string
            - "reference": int - identifier of reference cell type

            Added during class initialization:
            - "threshold_prob": Threshold for inclusion probability that separates significant from non-significant effects
        kwargs
            passed to az.InferenceData. This includes the MCMC chain states and statistics for eachs MCMC sample.
        """
        super(self.__class__, self).__init__(**kwargs)

        self.sampling_stats = sampling_stats
        self.model_specs = model_specs

        if "ind" in list(self.posterior.data_vars):
            self.is_sccoda = True
        else:
            self.is_sccoda = False

        intercept_df, effect_df = self.summary_prepare()

        self.intercept_df = intercept_df
        self.effect_df = effect_df

    def summary_prepare(
            self,
            est_fdr: float = 0.05,
            *args,
            **kwargs
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates summary dataframes for intercepts and slopes.
        This function builds on and supports all functionalities from ``az.summary``.

        Parameters
        ----------
        est_fdr
            Desired FDR value
        args
            Passed to ``az.summary``
        kwargs
            Passed to ``az.summary``

        Returns
        -------
        Intercept and effect DataFrames

        intercept_df -- pandas df
            Summary of intercept parameters. Contains one row per cell type.

            Columns:
            - Final Parameter: Final intercept model parameter
            - HDI X%: Upper and lower boundaries of confidence interval (width specified via hdi_prob=)
            - SD: Standard deviation of MCMC samples
            - Expected sample: Expected cell counts for a sample with no present covariates. See the tutorial for more explanation

        effect_df -- pandas df
            Summary of effect (slope) parameters. Contains one row per covariate/cell type combination.

            Columns:
            - Final Parameter: Final effect model parameter. If this parameter is 0, the effect is not significant, else it is.
            - HDI X%: Upper and lower boundaries of confidence interval (width specified via hdi_prob=)
            - SD: Standard deviation of MCMC samples
            - Expected sample: Expected cell counts for a sample with only the current covariate set to 1. See the tutorial for more explanation
            - log2-fold change: Log2-fold change between expected cell counts with no covariates and with only the current covariate
            - Inclusion probability: Share of MCMC samples, for which this effect was not set to 0 by the spike-and-slab prior.
        """

        # initialize summary df from arviz and separate into intercepts and effects.
        summ = az.summary(self, *args, **kwargs, kind="stats", var_names=["alpha", "beta"])
        effect_df = summ.loc[summ.index.str.match("|".join(["beta"]))].copy()
        intercept_df = summ.loc[summ.index.str.match("|".join(["alpha"]))].copy()

        # Build neat index
        cell_types = self.posterior.coords["cell_type"].values
        covariates = self.posterior.coords["covariate"].values

        intercept_df.index = pd.Index(cell_types, name="Cell Type")
        effect_df.index = pd.MultiIndex.from_product([covariates, cell_types],
                                                     names=["Covariate", "Cell Type"])

        # Calculation of columns that are not from az.summary
        intercept_df = self.complete_alpha_df(intercept_df)
        effect_df = self.complete_beta_df(intercept_df, effect_df, est_fdr)

        # Give nice column names, remove unnecessary columns
        hdis = intercept_df.columns[intercept_df.columns.str.contains("hdi")]
        hdis_new = hdis.str.replace("hdi_", "HDI ")


        # Credible interval
        if self.is_sccoda is True:
            ind_post = self.posterior["ind"]

            b_raw_sel = self.posterior["b_raw"] * ind_post.where(ind_post >= 1e-3)

            res = az.convert_to_inference_data(b_raw_sel)

            summary_sel = az.summary(res, kind="stats", var_names=["x"], skipna=True, *args, **kwargs)

            ref_index = self.model_specs["reference"]
            n_conditions = len(self.posterior.coords["covariate"])
            n_cell_types = len(self.posterior.coords["cell_type"])

            def insert_row(idx, df, df_insert):
                return pd.concat([df.iloc[:idx, ], df_insert, df.iloc[idx:, ]]).reset_index(drop=True)

            for i in range(n_conditions):
                summary_sel = insert_row((i*n_cell_types) + ref_index, summary_sel,
                                         pd.DataFrame.from_dict(data={"mean": [0], "sd": [0], hdis[0]: [0], hdis[1]: [0]}))

            effect_df.loc[:, hdis[0]] = list(summary_sel[hdis[0]])
            effect_df.loc[:, hdis[1]] = list(summary_sel.loc[:, hdis[1]])

        intercept_df = intercept_df.loc[:, ["final_parameter", hdis[0], hdis[1], "sd", "expected_sample"]].copy()
        intercept_df = intercept_df.rename(columns=dict(zip(
            intercept_df.columns,
            ["Final Parameter", hdis_new[0], hdis_new[1], "SD", "Expected Sample"]
        )))

        effect_df = effect_df.loc[:, ["final_parameter", hdis[0], hdis[1], "sd", "inclusion_prob",
                                       "expected_sample", "log_fold"]].copy()
        effect_df = effect_df.rename(columns=dict(zip(
            effect_df.columns,
            ["Final Parameter", hdis_new[0], hdis_new[1], "SD", "Inclusion probability",
             "Expected Sample", "log2-fold change"]
        )))

        return intercept_df, effect_df

    def complete_beta_df(
            self,
            intercept_df: pd.DataFrame,
            effect_df: pd.DataFrame,
            target_fdr: float=0.05,
    ) -> pd.DataFrame:
        """
        Evaluation of MCMC results for effect parameters. This function is only used within self.summary_prepare.
        This function also calculates the posterior inclusion probability for each effect and decides whether effects are significant.

        Parameters
        ----------
        intercept_df
            Intercept summary, see ``self.summary_prepare``
        effect_df
            Effect summary, see ``self.summary_prepare``
        target_fdr
            Desired FDR value

        Returns
        -------
        effect DataFrame

        effect_df
            DataFrame with inclusion probability, final parameters, expected sample
        """
        beta_inc_prob = []
        beta_nonzero_mean = []

        beta_raw = np.array(self.posterior["beta"])[0]

        # Calculate inclusion prob, nonzero mean for every effect
        for j in range(beta_raw.shape[1]):
            for i in range(beta_raw.shape[2]):
                beta_i_raw = beta_raw[:, j, i]
                beta_i_raw_nonzero = np.where(np.abs(beta_i_raw) > 1e-3)[0]
                prob = beta_i_raw_nonzero.shape[0] / beta_i_raw.shape[0]
                beta_inc_prob.append(prob)
                if len(beta_i_raw[beta_i_raw_nonzero]) > 0:
                    beta_nonzero_mean.append(beta_i_raw[beta_i_raw_nonzero].mean())
                else:
                    beta_nonzero_mean.append(0)

        effect_df.loc[:, "inclusion_prob"] = beta_inc_prob
        effect_df.loc[:, "mean_nonzero"] = beta_nonzero_mean

        # Inclusion prob threshold value. Direct posterior probability approach cf. Newton et al. (2004)
        if self.is_sccoda is True:
            def opt_thresh(result, alpha):

                incs = np.array(result.loc[result["inclusion_prob"] > 0, "inclusion_prob"])
                incs[::-1].sort()

                for c in np.unique(incs):
                    fdr = np.mean(1 - incs[incs >= c])

                    if fdr < alpha:
                        # ceiling with 3 decimals precision
                        c = np.floor(c * 10 ** 3) / 10 ** 3
                        return c, fdr
                return 1., 0

            threshold, fdr_ = opt_thresh(effect_df, target_fdr)

            self.model_specs["threshold_prob"] = threshold

            # Decide whether betas are significant or not, set non-significant ones to 0
            effect_df.loc[:, "final_parameter"] = np.where(effect_df.loc[:, "inclusion_prob"] >= threshold,
                                                           effect_df.loc[:, "mean_nonzero"],
                                                           0)
        else:
            effect_df.loc[:, "final_parameter"] = effect_df.loc[:, "mean_nonzero"]

        # Get expected sample, log-fold change
        D = len(effect_df.index.levels[0])
        K = len(effect_df.index.levels[1])

        y_bar = np.mean(np.sum(np.array(self.observed_data.y), axis=1))
        alpha_par = intercept_df.loc[:, "final_parameter"]
        alphas_exp = np.exp(alpha_par)
        alpha_sample = (alphas_exp / np.sum(alphas_exp) * y_bar).values

        beta_mean = alpha_par
        beta_sample = []
        log_sample = []

        for d in range(D):
            beta_d = effect_df.loc[:, "final_parameter"].values[(d*K):((d+1)*K)]
            beta_d = (beta_mean + beta_d)
            beta_d = np.exp(beta_d)
            beta_d = beta_d / np.sum(beta_d) * y_bar

            beta_sample = np.append(beta_sample, beta_d)
            log_sample = np.append(log_sample, np.log2(beta_d/alpha_sample))

        effect_df.loc[:, "expected_sample"] = beta_sample
        effect_df.loc[:, "log_fold"] = log_sample

        return effect_df

    def complete_alpha_df(
            self,
            intercept_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Evaluation of MCMC results for intercepts. This function is only used within self.summary_prepare.

        Parameters
        ----------
        intercept_df
            Intercept summary, see self.summary_prepare

        Returns
        -------
        intercept DataFrame

        intercept_df
            Summary DataFrame with expected sample, final parameters
        """

        intercept_df = intercept_df.rename(columns={"mean": "final_parameter"})

        # Get expected sample
        y_bar = np.mean(np.sum(np.array(self.observed_data.y), axis=1))
        alphas_exp = np.exp(intercept_df.loc[:, "final_parameter"])
        alpha_sample = (alphas_exp / np.sum(alphas_exp) * y_bar).values
        intercept_df.loc[:, "expected_sample"] = alpha_sample

        return intercept_df

    def summary(
            self,
            *args,
            **kwargs
    ):
        """
        Printing method for scCODA's summary.

        Usage: ``result.summary()``

        Parameters
        ----------
        args
            Passed to az.summary
        kwargs
            Passed to az.summary

        Returns
        -------
        prints to console

        """

        # If other than default values for e.g. confidence interval are specified,
        # recalculate them for intercept and effect DataFrames
        if args or kwargs:
            intercept_df, effect_df = self.summary_prepare(*args, **kwargs)
        else:
            intercept_df = self.intercept_df
            effect_df = self.effect_df

        # Get number of samples, cell types
        if self.sampling_stats["y_hat"] is not None:
            data_dims = self.sampling_stats["y_hat"].shape
        else:
            data_dims = (10, 5)

        # Cut down DataFrames to relevant info
        alphas_print = intercept_df.loc[:, ["Final Parameter", "Expected Sample"]]
        betas_print = effect_df.loc[:, ["Final Parameter", "Expected Sample", "log2-fold change"]]

        # Print everything neatly
        print("Compositional Analysis summary:")
        print("")
        print("Data: %d samples, %d cell types" % data_dims)
        print("Reference index: %s" % str(self.model_specs["reference"]))
        print("Formula: %s" % self.model_specs["formula"])
        print("")
        print("Intercepts:")
        print(alphas_print)
        print("")
        print("")
        print("Effects:")
        print(betas_print)

    def summary_extended(
            self,
            *args,
            **kwargs
    ):

        """
        Extended (diagnostic) printing function that shows more info about the sampling result

        Parameters
        ----------
        args
            Passed to az.summary
        kwargs
            Passed to az.summary

        Returns
        -------
        Prints to console

        """

        # If other than default values for e.g. confidence interval are specified,
        # recalculate them for intercept and effect DataFrames
        if args or kwargs:
            intercept_df, effect_df = self.summary_prepare(*args, **kwargs)
        else:
            intercept_df = self.intercept_df
            effect_df = self.effect_df

        # Get number of samples, cell types
        data_dims = self.sampling_stats["y_hat"].shape

        # Print everything
        print("Compositional Analysis summary (extended):")
        print("")
        print("Data: %d samples, %d cell types" % data_dims)
        print("Reference index: %s" % str(self.model_specs["reference"]))
        print("Formula: %s" % self.model_specs["formula"])
        if self.is_sccoda:
            print("Spike-and-slab threshold: {threshold:.3f}".format(threshold=self.model_specs["threshold_prob"]))
        print("")
        print("MCMC Sampling: Sampled {num_results} chain states ({num_burnin} burnin samples) in {duration:.3f} sec. "
              "Acceptance rate: {ar:.1f}%".format(num_results=self.sampling_stats["chain_length"],
                                                  num_burnin=self.sampling_stats["num_burnin"],
                                                  duration=self.sampling_stats["duration"],
                                                  ar=(100*self.sampling_stats["acc_rate"])))
        print("")
        print("Intercepts:")
        print(intercept_df)
        print("")
        print("")
        print("Effects:")
        print(effect_df)

    def compare_parameters_to_truth(
            self,
            b_true: pd.Series,
            w_true: pd.Series,
            *args,
            **kwargs
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extends data frames from summary_prepare by a comparison to some ground truth slope and intercept values that are
        assumed to be from the same generative model (e.g. in data_generation)

        Parameters
        ----------
        b_true
            Ground truth slope values. Length must be same as number of cell types
        w_true
            Ground truth intercept values. Length must be same as number of cell types*number of covariates
        args
            Passed to az.summary
        kwargs
            Passed to az.summary

        Returns
        -------
        Extends intercept and effect DataFrames

        intercept_df
            Summary DataFrame for intercepts
        effect_df
            Summary DataFrame for effects
        """

        intercept_df, effect_df = self.summary_prepare(*args, **kwargs)

        intercept_df.columns = intercept_df.columns.str.replace('final_parameter', 'predicted')
        effect_df.columns = effect_df.columns.str.replace('final_parameter', 'predicted')

        # Get true params, join to calculated parameters
        b_true = b_true.rename("truth")
        intercept_df = intercept_df.join(b_true)
        w_true = w_true.rename("truth")
        effect_df = effect_df.join(w_true)

        # decide whether effects are found correctly
        intercept_df['dist_to_truth'] = intercept_df['truth'] - intercept_df['predicted']
        intercept_df['effect_correct'] = ((intercept_df['truth'] == 0) == (intercept_df['predicted'] == 0))
        effect_df['dist_to_truth'] = effect_df['truth'] - effect_df['predicted']
        effect_df['effect_correct'] = ((effect_df['truth'] == 0) == (effect_df['predicted'] == 0))

        return intercept_df, effect_df

    def distance_to_truth(self) -> pd.DataFrame:
        """
        Compares real cell count matrix to the posterior mode cell count matrix that arises from the calculated parameters

        Returns
        -------
        DataFrame with distances

        ret
            DataFrame
        """

        # Get absolute (counts) and relative error matrices
        y = np.array(self.observed_data.y)
        y_hat = self.sampling_stats["y_hat"]
        err = np.abs(y_hat - y)

        err_rel = err / y
        err_rel[np.isinf(err_rel)] = 1.
        err_rel[np.isnan(err_rel)] = 0.

        # Calculate mean errors for each cell type and overall
        avg_abs_cell_type_error = np.mean(err, axis=0, dtype=np.float64)
        avg_rel_cell_type_error = np.mean(err_rel, axis=0, dtype=np.float64)
        avg_abs_total_error = np.mean(err, dtype=np.float64)
        avg_rel_total_error = np.mean(err_rel, dtype=np.float64)

        ret = pd.DataFrame({'Cell Type': np.arange(y.shape[1] + 1),
                            'Absolute Error': np.append(avg_abs_total_error, avg_abs_cell_type_error),
                            'Relative Error': np.append(avg_rel_total_error, avg_rel_cell_type_error),
                            'Actual Means': np.append(np.mean(y, axis=(0, 1)), np.mean(y, axis=0)),
                            'Predicted Means': np.append(np.mean(y_hat, axis=(0, 1)), np.mean(y_hat, axis=0))})

        ret['Cell Type'][0] = 'Total'
        return ret

    def credible_effects(
            self,
            est_fdr=None
    ) -> pd.Series:

        """
        Decides which effects of the scCODA model are credible based on an adjustable inclusion probability threshold.

        Parameters
        ----------
        est_fdr
            Estimated false discovery rate. Must be between 0 and 1

        Returns
        -------
        Credible effect decision series

        out
            Boolean values whether effects are credible under inc_prob_threshold
        """

        if type(est_fdr) == float:
            if est_fdr < 0 or est_fdr > 1:
                raise ValueError("est_fdr must be between 0 and 1!")
            else:
                _, eff_df = self.summary_prepare(est_fdr=est_fdr)
        else:
            eff_df = self.effect_df

        out = eff_df["Final Parameter"] != 0
        out.rename("credible change")

        return out

    def save(
            self,
            path_to_file: str
    ):
        """
        Function to save scCODA results to disk via pickle. Caution: Files can quickly become very large!

        Parameters
        ----------
        path_to_file
            saving location on disk

        Returns
        -------

        """
        with open(path_to_file, "wb") as f:
            pkl.dump(self, file=f, protocol=4)

    def set_fdr(
            self,
            est_fdr: float,
            *args,
            **kwargs):
        """
        Direct posterior probability approach to calculate credible effects while keeping the expected FDR at a certain level

        Parameters
        ----------
        est_fdr
            Desired FDR value
        args
            passed to self.summary_prepare
        kwargs
            passed to self.summary_prepare

        Returns
        -------
        Adjusts self.intercept_df and self.effect_df
        """

        intercept_df, effect_df = self.summary_prepare(est_fdr=est_fdr, *args, **kwargs)

        self.intercept_df = intercept_df
        self.effect_df = effect_df
