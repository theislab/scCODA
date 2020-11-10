"""
Results class that summarizes the results of the different inference methods and calculates test statistics

:authors: Johannes Ostner
"""
import numpy as np
import arviz as az
import pandas as pd
import pickle as pkl


class CAResultConverter(az.data.io_dict.DictConverter):
    """
    Helper class for result conversion in arviz's format
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
    Result class, extends the arviz framework for inference data.

    The CAResult class is an extension of az.InferenceData, that adds some information about the compositional model.
    It supports all functionality from az.InferenceData.
    """

    def __init__(self, sampling_stats, model_specs, **kwargs):
        """
        The following attributes are added during class initialization:

        self.sampling_stats: dict - see below
        self.model_specs: dict - see below

        self.intercept_df: Summary dataframe from CAResult.summary_prepare
        self.effect_df: Summary dataframe from CAResult.summary_prepare

        Parameters
        ----------
        sampling_stats -- dict
            Information and statistics about the MCMC sampling procedure.
            Default keys:
            "chain_length": Length of MCMC chain (with burnin samples)
            "n_burnin": Number of burnin samples
            "acc_rate": MCMC Acceptance rate
            "duration": Duration of MCMC sampling
        model_specs -- dict
            All information and statistics about the model specifications.
            Default keys:
            "formula": Formula string
            "baseline": int - identifier of baseline cell type
            Added during class initialization: "threshold_prob": Threshold for inclusion probability that separates significant from non-significant effects
        kwargs -- passed to az.InferenceData
        """
        super(self.__class__, self).__init__(**kwargs)

        self.sampling_stats = sampling_stats
        self.model_specs = model_specs

        intercept_df, effect_df = self.summary_prepare()

        self.intercept_df = intercept_df
        self.effect_df = effect_df

    def summary_prepare(self, *args, **kwargs):
        """
        Generates summary dataframes for intercepts and slopes.
        This function supports all functionalities from az.summary.

        Parameters
        ----------
        args -- Passed to az.summary
        kwargs -- Passed to az.summary

        Returns
        -------
        intercept_df -- pandas df
            Summary of intercept parameters. Contains one row per cell type. Columns:
            Final Parameter: Final intercept model parameter
            hdi X%: Upper and lower boundaries of confidence interval (width specified via hdi_prob=)
            SD: Standard deviation of MCMC samples
            Expected sample: Expected cell counts for a sample with no present covariates. See the tutorial for more explanation

        effect_df -- pandas df
            Summary of effect (slope) parameters. Contains one row per covariate/cell type combination. Columns:
            Final Parameter: Final effect model parameter. If this parameter is 0, the effect is not significant, else it is.
            HDI X%: Upper and lower boundaries of confidence interval (width specified via hdi_prob=)
            SD: Standard deviation of MCMC samples
            Expected sample: Expected cell counts for a sample with only the current covariate set to 1. See the tutorial for more explanation
            log2-fold change: Log2-fold change between expected cell counts with no covariates and with only the current covariate
            Inclusion probability: Share of MCMC samples, for which this effect was not set to 0 by the spike-and-slab prior.
        """
        # initialize summary df
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
        effect_df = self.complete_beta_df(intercept_df, effect_df)

        # Give nice column names, remove unnecessary columns
        hdis = intercept_df.columns[intercept_df.columns.str.contains("hdi")]
        hdis_new = hdis.str.replace("hdi_", "HDI ")

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

    def complete_beta_df(self, intercept_df, effect_df):
        """
        Evaluation of MCMC results for effect parameters. This function is only used within self.summary_prepare.

        Parameters
        ----------
        intercept_df -- Data frame
            Intercept summary, see self.summary_prepare
        effect_df -- Data frame
            Effect summary, see self.summary_prepare

        Returns
        -------
        effect_df -- DataFrame
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
                beta_nonzero_mean.append(beta_i_raw[beta_i_raw_nonzero].mean())

        effect_df.loc[:, "inclusion_prob"] = beta_inc_prob
        effect_df.loc[:, "mean_nonzero"] = beta_nonzero_mean

        # Inclusion prob threshold value
        threshold = 1-(0.77/(beta_raw.shape[2]**0.29))
        self.model_specs["threshold_prob"] = threshold

        # Decide whether betas are significant or not
        effect_df.loc[:, "final_parameter"] = np.where(effect_df.loc[:, "inclusion_prob"] > threshold,
                                                      effect_df.loc[:, "mean_nonzero"],
                                                      0)

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

    def complete_alpha_df(self, intercept_df):
        """
        Evaluation of MCMC results for intercepts. This function is only used within self.summary_prepare.

        Parameters
        ----------
        intercept_df -- Data frame
            Intercept summary, see self.summary_prepare

        Returns
        -------
        intercept_df -- DataFrame
            Summary DataFrame with expected sample, final parameters
        """

        intercept_df = intercept_df.rename(columns={"mean": "final_parameter"})

        # Get expected sample
        y_bar = np.mean(np.sum(np.array(self.observed_data.y), axis=1))
        alphas_exp = np.exp(intercept_df.loc[:, "final_parameter"])
        alpha_sample = (alphas_exp / np.sum(alphas_exp) * y_bar).values
        intercept_df.loc[:, "expected_sample"] = alpha_sample

        return intercept_df

    def summary(self, *args, **kwargs):
        """
        Printing method for summary data

        Usage: result.summary()

        Parameters
        ----------
        args -- Passed to az.summary
        kwargs -- Passed to az.summary

        Returns
        -------

        """

        # If other than default values for e.g. confidence interval are specified, recalculate the intercept and effect DataFrames
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

        # Print everything
        print("Compositional Analysis summary:")
        print("")
        print("Data: %d samples, %d cell types" % data_dims)
        print("Baseline index: %s" % str(self.model_specs["baseline"]))
        print("Formula: %s" % self.model_specs["formula"])
        print("")
        print("Intercepts:")
        print(alphas_print)
        print("")
        print("")
        print("Effects:")
        print(betas_print)

    def summary_extended(self, *args, **kwargs):

        # If other than default values for e.g. confidence interval are specified, recalculate the intercept and effect DataFrames
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
        print("Baseline index: %s" % str(self.model_specs["baseline"]))
        print("Formula: %s" % self.model_specs["formula"])
        print("Spike-and-slab threshold: {threshold:.3f}".format(threshold=self.model_specs["threshold_prob"]))
        print("")
        print("MCMC Sampling: Sampled {num_results} chain states ({n_burnin} burnin samples) in {duration:.3f} sec. "
              "Acceptance rate: {ar:.1f}%".format(num_results=self.sampling_stats["chain_length"],
                                                  n_burnin=self.sampling_stats["n_burnin"],
                                                  duration=self.sampling_stats["duration"],
                                                  ar=(100*self.sampling_stats["acc_rate"])))
        print("")
        print("Intercepts:")
        print(intercept_df)
        print("")
        print("")
        print("Effects:")
        print(effect_df)

    def compare_to_truth(self, b_true, w_true, *args, **kwargs):
        """
        Extends data frames from summary_prepare by a comparison to some ground truth slope and intercept values

        Parameters
        ----------
        b_true -- pandas Series
            Ground truth slope values. Length must be same as number of cell types
        w_true -- pandas Series
            Ground truth intercept values. Length must be same as number of cell types*number of covariates
        args -- Passed to az.summary
        kwargs -- Passed to az.summary

        Returns
        -------
        intercept_df -- DataFrame
            Summary DataFrame for intercepts
        effect_df -- DataFrame
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

    def distances(self):
        """
        Compares real cell count matrix to the cell count matrix that arises from the calculated parameters

        Returns
        -------
        ret: DataFrame
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

    def save(self, path_to_file):
        with open(path_to_file, "wb") as f:
            pkl.dump(self, file=f, protocol=4)

