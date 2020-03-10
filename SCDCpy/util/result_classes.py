"""
This file contains
Results objects that summarize the results of the different
inference methods and calculates test statistics


:authors: Benjamin Schubert, Johannes Ostner
"""
import numpy as np
import arviz as az
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import time
from abc import ABCMeta, abstractmethod


class CAResultConverter(az.data.io_dict.DictConverter):
    """
    Helper class for result conversion
    """

    def to_result_data(self, y_hat, baseline):

        post = self.posterior_to_xarray()
        ss = self.sample_stats_to_xarray()
        postp = self.posterior_predictive_to_xarray()
        prior = self.prior_to_xarray()
        ssp = self.sample_stats_prior_to_xarray()
        prip = self.prior_predictive_to_xarray()
        obs = self.observed_data_to_xarray()

        return CAResult(
            y_hat=y_hat, baseline=baseline,
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
    Result class, extends the arviz framework fot inference data
    """

    def __init__(self, y_hat, baseline, **kwargs):
        super(self.__class__, self).__init__(**kwargs)

        self.baseline = baseline
        self.y_hat = y_hat

    def summary_prepare(self, *args, **kwargs):
        """
        Preparation of summary method
        Parameters
        ----------
        args -- Passed to az.summary
        kwargs -- Passed to az.summary

        Returns
        -------
        alphas_df and betas_df for summary method
        """
        # initialize summary df
        summ = az.summary(self, *args, **kwargs, kind="stats", var_names=["alpha", "beta"])
        betas_df = summ.loc[summ.index.str.match("|".join(["beta"]))]
        alphas_df = summ.loc[summ.index.str.match("|".join(["alpha"]))]

        cell_types = self.posterior.coords["cell_type"]
        covariates = self.posterior.coords["covariate"]

        alphas_df.index = pd.Index(cell_types, name="Cell Type")
        betas_df.index = pd.MultiIndex.from_product([covariates, cell_types],
                                                    names=["Covariate", "Cell Type"])

        alphas_df = self.complete_alpha_df(alphas_df)
        betas_df = self.complete_beta_df(alphas_df, betas_df)

        return alphas_df, betas_df

    def complete_beta_df(self, alphas_df, betas_df):
        """
        Evaluation of MCMC results for slopes
        Parameters
        ----------
        alphas_df -- Data frame with intercept summary from az.summary
        betas_df -- Data frame with slope summary from az.summary

        Returns
        -------
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

        betas_df["inclusion_prob"] = beta_inc_prob
        betas_df["mean_nonzero"] = beta_nonzero_mean

        # Inclusion prob threshold value
        if self.baseline is None:
            threshold_factor = 0.87
        else:
            threshold_factor = 0.98
        threshold = 1 - threshold_factor / np.sqrt(beta_raw.shape[2])

        # Decide whether betas are significant or not
        betas_df["final_parameter"] = np.where(betas_df["inclusion_prob"] > threshold,
                                               betas_df["mean_nonzero"],
                                               0)

        # Get expected sample, log-fold change
        D = len(betas_df.index.levels[0])
        K = len(betas_df.index.levels[1])

        y_bar = np.mean(np.sum(np.array(self.observed_data.y), axis=1))
        alpha_par = alphas_df["final_parameter"]
        alphas_exp = np.exp(alpha_par)
        alpha_sample = (alphas_exp / np.sum(alphas_exp) * y_bar).values

        beta_mean = alpha_par
        beta_sample = []
        log_sample = []

        for d in range(D):
            beta_d = betas_df["final_parameter"].values[(d*K):((d+1)*K)]
            beta_d = (beta_mean + beta_d)
            beta_d = np.exp(beta_d)
            beta_d = beta_d / np.sum(beta_d) * y_bar

            beta_sample = np.append(beta_sample, beta_d)
            log_sample = np.append(log_sample, np.log2(beta_d/alpha_sample))

        betas_df["expected_sample"] = beta_sample
        betas_df["log_fold"] = log_sample

        return betas_df

    def complete_alpha_df(self, alphas_df):
        """
        Evaluation of MCMC results for intercepts
        Parameters
        ----------
        alphas_df -- Data frame with intercept summary from az.summary

        Returns
        -------
        Summary DataFrame with expected sample, final parameters
        """

        alphas_df["final_parameter"] = alphas_df["mean"]

        # Get expected sample
        y_bar = np.mean(np.sum(np.array(self.observed_data.y), axis=1))
        alphas_exp = np.exp(alphas_df["final_parameter"])
        alpha_sample = (alphas_exp / np.sum(alphas_exp) * y_bar).values
        alphas_df["expected_sample"] = alpha_sample

        return alphas_df

    def summary(self, *args, **kwargs):
        """
        Printing method for summary data
        Parameters
        ----------
        args -- Passed to az.summary
        kwargs -- Passed to az.summary

        Returns
        -------

        """
        alphas_df, betas_df = self.summary_prepare(*args, **kwargs)

        hpds = alphas_df.columns[alphas_df.columns.str.contains("hpd")]
        hpds_new = hpds.str.replace("hpd_", "HPD ")

        alphas_print = alphas_df.loc[:, ["final_parameter", hpds[0], hpds[1], "sd", "expected_sample"]]
        alphas_print = alphas_print.rename(columns=dict(zip(
            alphas_print.columns,
            ["Final Parameter", hpds_new[0], hpds_new[1], "SD", "Expected Sample"]
        )))

        betas_print = betas_df.loc[:, ["final_parameter", hpds[0], hpds[1], "sd", "inclusion_prob",
                                       "expected_sample", "log_fold"]]
        betas_print = betas_print.rename(columns=dict(zip(
            betas_print.columns,
            ["Final Parameter", hpds_new[0], hpds_new[1], "SD", "Inclusion probability",
             "Expected Sample", "log2-fold change"]
        )))

        print("Compositional Analysis summary:")
        print("Intercepts:")
        print(alphas_print)
        print("")
        print("")
        print("Effects:")
        print(betas_print)

    def compare_to_truth(self, b_true, w_true, *args, **kwargs):
        """
        Extends data frames from summary_prepare by a comparison to some ground truth slope and intercept values
        Parameters
        ----------
        b_true -- Ground truth slope values
        w_true -- Ground truth intercept values
        args -- Passed to az.summary
        kwargs -- Passed to az.summary

        Returns
        -------
        alphas_df, betas_df
        """

        alphas_df, betas_df = self.summary_prepare(*args, **kwargs)

        alphas_df.columns = alphas_df.columns.str.replace('final_parameter', 'predicted')
        betas_df.columns = betas_df.columns.str.replace('final_parameter', 'predicted')

        # Get true params, join to calculated parameters
        b_true = b_true.rename("truth")
        alphas_df = alphas_df.join(b_true)
        w_true = w_true.rename("truth")
        betas_df = betas_df.join(w_true)

        # decide whether effects are found correctly
        alphas_df['dist_to_truth'] = alphas_df['truth'] - alphas_df['predicted']
        alphas_df['effect_correct'] = ((alphas_df['truth'] == 0) == (alphas_df['predicted'] == 0))
        betas_df['dist_to_truth'] = betas_df['truth'] - betas_df['predicted']
        betas_df['effect_correct'] = ((betas_df['truth'] == 0) == (betas_df['predicted'] == 0))

        return alphas_df, betas_df

    def distances(self):
        """
        Compares real cell count matrix to the cell count matrix that arises from the calculated parameters
        Returns
        -------
        DataFrame
        """

        # Get absolute (counts) and relative error matrices
        y = np.array(self.observed_data.y)
        err = np.abs(self.y_hat - y)

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
                            'Predicted Means': np.append(np.mean(self.y_hat, axis=(0, 1)), np.mean(self.y_hat, axis=0))})

        ret['Cell Type'][0] = 'Total'
        return ret
