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
from abc import ABCMeta, abstractmethod


class CompAnaResult:
    """
    Abstract Result class defining the result interface
    """

    def __init__(self, params, y_hat, y, x, baseline, cell_types, covariate_names):
        """
        Init function
        :param params: the trace of the parameters
        :param y_hat: cell count matrix calculated by the model
        :param y: true (observed) cell count matrix
        """

        self.y_hat = y_hat
        self.y = y
        self.cell_types = cell_types
        self.covariate_names = covariate_names
        self.raw_params = params
        self.X = x

        # Setup arviz plot compatibility
        self.arviz_params = self.__transform_data_to_inference_data()
        df = az.summary(self.arviz_params, kind="stats")

        # Select relevant params
        self.params = df.loc[df.index.str.match("|".join(["alpha", "beta"]))]

        # For sipke-and-slab prior: Select significant effects via inclusion probability
        self.params["inclusion_prob"] = np.NaN
        self.params["mean_nonzero"] = np.NaN

        beta_raw = params["beta"]
        beta_inc_prob = []
        beta_nonzero_mean = []

        for j in range(beta_raw.shape[1]):
            for i in range(beta_raw.shape[2]):
                beta_i_raw = beta_raw[:, j, i]
                beta_i_raw_nonzero = np.where(np.abs(beta_i_raw) > 1e-3)[0]
                prob = beta_i_raw_nonzero.shape[0]/beta_i_raw.shape[0]
                beta_inc_prob.append(prob)
                beta_nonzero_mean.append(beta_i_raw[beta_i_raw_nonzero].mean())

        self.params.loc[self.params.index.str.match(r"beta\["), "inclusion_prob"] = beta_inc_prob
        self.params.loc[self.params.index.str.match(r"beta\["), "mean_nonzero"] = beta_nonzero_mean

        # Inclusion prob threshold value
        if baseline is None:
            threshold_factor = 0.87
        else:
            threshold_factor = 0.98
        threshold = 1-threshold_factor/np.sqrt(beta_raw.shape[2])

        self.params["final_parameter"] = np.where(np.isnan(self.params["mean_nonzero"]),
                                                  self.params["mean"],
                                                  np.where(self.params["inclusion_prob"] > threshold,
                                                           self.params["mean_nonzero"],
                                                           0))

    def compare_to_truth(self, true_params, varnames=None):
        """
        Compares the calculated parameters to a ground truth reference
        :param true_params: !!!!
        :param varnames: a list of string specifying the variables to plot
        :return: a pandas DataFrame
        """

        if varnames is None:
            comp = self.params['final_parameter']
        else:
            comp = self.params[self.params.index.str.contains("|".join(varnames)), 'final_parameter']

        comp = pd.DataFrame(comp)
        comp.columns = comp.columns.str.replace('final_parameter', 'predicted')

        # Get true params, join to calculated parameters
        t = pd.DataFrame.from_dict(true_params, columns=["truth"], orient='index')
        comp = comp.join(t)

        # decide whether effects are found correctly
        comp['dist_to_truth'] = comp['truth'] - comp['predicted']
        comp['effect_correct'] = ((comp['truth'] == 0) == (comp['predicted'] == 0))

        return comp

    def distances(self):
        """
        Comnpares real cell count matrix to the cell count matrix that arises from the calculated parameters
        :return: a pandas DataFrame
        """

        # Get absolute (counts) and relative error matrices
        err = np.abs(self.y_hat - self.y)

        err_rel = err / self.y
        err_rel[np.isinf(err_rel)] = 1.
        err_rel[np.isnan(err_rel)] = 0.

        # Calculate mean errors for each cell type and total
        avg_abs_cell_type_error = np.mean(err, axis=0, dtype=np.float64)
        avg_rel_cell_type_error = np.mean(err_rel, axis=0, dtype=np.float64)
        avg_abs_total_error = np.mean(err, dtype=np.float64)
        avg_rel_total_error = np.mean(err_rel, dtype=np.float64)

        ret = pd.DataFrame({'Cell Type': np.arange(self.y.shape[1] + 1),
                            'Absolute Error': np.append(avg_abs_total_error, avg_abs_cell_type_error),
                            'Relative Error': np.append(avg_rel_total_error, avg_rel_cell_type_error),
                            'Actual Means': np.append(np.mean(self.y, axis=(0, 1)), np.mean(self.y, axis=0)),
                            'Predicted Means': np.append(np.mean(self.y_hat, axis=(0, 1)), np.mean(self.y_hat, axis=0))})

        ret['Cell Type'][0] = 'Total'
        return ret

    def summary_prepare(self, varnames=None, credible_interval=0.94):

        # names of confidence interval columns
        hpd_lower = np.round((1-credible_interval)/2, 3)
        hpd_higher = 1-hpd_lower
        hpd_lower_str = "HPD "+str(hpd_lower*100)+"%"
        hpd_higher_str = "HPD "+str(hpd_higher*100)+"%"

        # custom confidence intervals
        intervals = az.summary(self.arviz_params, credible_interval=credible_interval, kind="stats")
        intervals = intervals.loc[intervals.index.str.match("|".join(["alpha", "beta"])),
                                  intervals.columns.str.contains("hpd_")]
        par = pd.concat([self.params.drop(columns=["hpd_3%", "hpd_97%"]),
                         intervals],
                        axis=1, join="inner")

        # Complete DataFrame for summary
        if varnames is None:
            summ_df = par
        else:
            summ_df = par[par.index.str.contains("|".join(varnames))]

        summ_df = summ_df.rename(columns=dict(zip(
            summ_df.columns,
            ["Mean", "SD", "Inclusion Probability", "Mean (Non-zero)",
             "Final Parameter", hpd_lower_str, hpd_higher_str]
        )))

        # Get intercept data frame, add expected sample
        alphas_df = summ_df.loc[summ_df.index.str.contains("alpha"),
                                ["Final Parameter", hpd_lower_str, hpd_higher_str, "SD"]]
        alphas_exp = np.exp(alphas_df)

        y_bar = np.mean(np.sum(self.y, axis=1))
        alpha_sample = (alphas_exp["Final Parameter"]
                        / np.sum(alphas_exp["Final Parameter"])
                        * y_bar).values
        alphas_df["Expected Sample"] = alpha_sample

        # Effect data frame
        betas_df = summ_df.loc[summ_df.index.str.contains("beta"),
                               ["Final Parameter", hpd_lower_str, hpd_higher_str, "SD", "Inclusion Probability"]]

        K = alphas_df.shape[0]
        D = int(betas_df.shape[0]/K)

        # add expected sample, log-fold change for effects
        beta_mean = alphas_df["Final Parameter"].values
        beta_sample = []
        log_sample = []
        for d in range(D):
            beta_d = betas_df["Final Parameter"].values[(d*K):((d+1)*K)]
            beta_d = beta_mean + beta_d
            beta_d = np.exp(beta_d)
            beta_d = beta_d / np.sum(beta_d) * y_bar

            beta_sample = np.append(beta_sample, beta_d)
            log_sample = np.append(log_sample, np.log2(beta_d/alpha_sample))

        betas_df["Expected Sample"] = beta_sample
        betas_df["log2-fold change"] = log_sample

        # Make nice indices
        alphas_df.index = pd.Index(self.cell_types, name="Cell Type")
        betas_df.index = pd.MultiIndex.from_product([self.covariate_names, self.cell_types],
                                                    names=["Covariate", "Cell Type"])

        return alphas_df, betas_df

    def summary(self, *args, **kwargs):
        alphas, betas = self.summary_prepare(*args, **kwargs)

        print("Compositional Analysis summary:")
        print("Intercepts:")
        print(alphas)
        print("")
        print("")
        print("Effects:")
        print(betas)

    def traceplots(self, varnames=None):
        """
        Traceplots
        :param varnames: a list of string specifying the variables to plot
        :return: traceplots of all model parameters
        """
        if varnames is None:
            plot_df = self.params
        else:
            plot_df = self.params[self.params.index.str.contains("|".join(varnames))]

        print(self.arviz_params)

        az.plot_trace(self.arviz_params)
        plt.show()

    def posterior_plots(self, varnames=None):

        if varnames is None:
            plot_df = self.params
        else:
            plot_df = self.params[self.params.index.str.contains("|".join(varnames))]

        az.plot_posterior(self.arviz_params, ref_val=0, color='#87ceeb')
        plt.show()

    def plot_autocorr(self, varnames=None):

        if varnames is None:
            plot_df = self.params
        else:
            plot_df = self.params[self.params.index.str.contains("|".join(varnames))]

        az.plot_autocorr(self.arviz_params, max_lag=self.arviz_params.posterior.sizes['draw'])
        plt.show()

    def __transform_data_to_inference_data(self):
        """
        transforms the sampled data to InferenceData object used by arviz
        :return: arviz.InferenceData
        """
        return az.convert_to_inference_data(
            {var_name: var[np.newaxis] for var_name, var in self.raw_params.items() if
             "concentration" not in var_name
             })


class CAResultConverter(az.data.io_dict.DictConverter):

    def to_result_data(self, y_hat, baseline):
        return CAResult(
            y_hat, baseline=baseline,
            **{
                "posterior": self.posterior_to_xarray(),
                "sample_stats": self.sample_stats_to_xarray(),
                "posterior_predictive": self.posterior_predictive_to_xarray(),
                "prior": self.prior_to_xarray(),
                "sample_stats_prior": self.sample_stats_prior_to_xarray(),
                "prior_predictive": self.prior_predictive_to_xarray(),
                "observed_data": self.observed_data_to_xarray(),
            }
        )


class CAResult(az.InferenceData):

    def __init__(self, y_hat, baseline, **kwargs):
        super(self.__class__, self).__init__(**kwargs)

        self.baseline = baseline
        self.y_hat = y_hat

    def summary_prepare(self, *args, **kwargs):
        # initialize summary df
        summ = az.summary(self, *args, **kwargs, kind="stats", var_names=["alpha", "beta"])
        betas_df = summ.loc[summ.index.str.match("|".join(["beta"]))]
        alphas_df = summ.loc[summ.index.str.match("|".join(["alpha"]))]

        cell_types = self.posterior.coords["cell_type"]
        covariates = self.posterior.coords["cov"]

        alphas_df.index = pd.Index(cell_types, name="Cell Type")
        betas_df.index = pd.MultiIndex.from_product([covariates, cell_types],
                                                    names=["Covariate", "Cell Type"])

        alphas_df = self.complete_alpha_df(alphas_df)
        betas_df = self.complete_beta_df(alphas_df, betas_df)

        return alphas_df, betas_df

    def complete_beta_df(self, alphas_df, betas_df):
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

        alphas_df["final_parameter"] = alphas_df["mean"]

        # Get expected sample
        y_bar = np.mean(np.sum(np.array(self.observed_data.y), axis=1))
        alphas_exp = np.exp(alphas_df["final_parameter"])
        alpha_sample = (alphas_exp / np.sum(alphas_exp) * y_bar).values
        alphas_df["expected_sample"] = alpha_sample

        return alphas_df

    def summary(self, *args, **kwargs):
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
        :return: a pandas DataFrame
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
                            'Predicted Means': np.append(np.mean(y_hat, axis=(0, 1)), np.mean(y_hat, axis=0))})

        ret['Cell Type'][0] = 'Total'
        return ret
