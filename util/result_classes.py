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

class AResult(metaclass=ABCMeta):
    """
    Abstract Result class defining the result interface
    """

    @abstractmethod
    def summary(self, varnames=None):
        """
        Summarizes the results returning
        estimates, CI/HDI, and test statistics
        as pandas dataframe
        :param varnames: a list of string specifying the variables to summarize
        :return: A Pandas dataframe
        """
        raise NotImplementedError

    def __str__(self):
        return str(self.summary())

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    def plot(self, varnames=None):
        """
        Plots a summary of all/specified inferred parameters
        :param varnames: a list of string specifying the variables to plot
        :return: a matplotlib derived figure
        """
        raise NotImplementedError

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


class MCMCResult(AResult):
    """
    Result class for MCMC samples
    """

    def __init__(self, N, params, y_hat, y, spike_slab=False):
        """
        Init function
        :param N: The sample size
        :param params: the trace of the parameters
        :param y_hat: cell count matrix calculated by the model
        :param y: true (observed) cell count matrix
        :param spike_slab: boolean - indicates whether the model uss a spike-and-slab prior
        """
        self.N = N
        self.y_hat = y_hat
        self.y = y

        self.__raw_params = params

        # Setup arviz plot compatibility
        self.arviz_params = self.__transform_data_to_inference_data()
        df = az.summary(self.arviz_params)

        # Calculate confidence intervals
        self.params = df[df.index.str.match("|".join(["alpha", "beta", "gamma"]))]
        self.params["z"] = (np.abs(np.divide(self.params["mean"], self.params["sd"]))).where(self.params["sd"]!=0, 1.000)
        self.params["Pr(>|z|)"] = 2 * (1 - st.norm(loc=0, scale=1).cdf(self.params["z"]))

        # For sipke-and-slab prior: Select significant effects via inclusion probability
        if spike_slab == True:
            #self.params.loc[self.params.index.str.match("beta"), ["z", "Pr(>|z|)"]] = np.NaN

            beta_raw = self.__raw_params["beta"]
            beta_inc_prob = []
            beta_nonzero_mean = []

            for j in range(beta_raw.shape[1]):
                for i in range(beta_raw.shape[2]):
                    beta_i_raw = beta_raw[:,j,i]
                    beta_i_raw_nonzero = np.where(np.abs(beta_i_raw) > 1e-3)[0]
                    prob = beta_i_raw_nonzero.shape[0]/beta_i_raw.shape[0]
                    beta_inc_prob.append(prob)
                    beta_nonzero_mean.append(beta_i_raw[beta_i_raw_nonzero].mean())

            self.params["inclusion_prob"] = np.NaN
            self.params.loc[self.params.index.str.match("beta\["), "inclusion_prob"] = beta_inc_prob
            self.params["mean_nonzero"] = np.NaN
            self.params.loc[self.params.index.str.match("beta\["), "mean_nonzero"] = beta_nonzero_mean

            self.params["final_parameter"] = np.where(np.isnan(self.params["mean_nonzero"]),
                                                      self.params["mean"],
                                                      np.where(self.params["inclusion_prob"] > 1-1/np.sqrt(beta_raw.shape[2]),
                                                               self.params["mean_nonzero"],
                                                               0))
        else:
            self.params["final_parameter"] = np.where(self.params["Pr(>|z|)"]>0.05, 0, self.params["mean"])

    def summary(self, varnames=None):
        """
        Get model summary statistics
        :param varnames: a list of string specifying the variables to plot
        :return: a pandas DataFrame
        """
        if varnames is None:
            return self.params
        else:
            return self.params[self.params.index.str.contains("|".join(varnames))]

    def plot(self, varnames=None):
        """
        Traceplots
        :param varnames: a list of string specifying the variables to plot
        :return: traceplots of all model parameters
        """
        az.plot_trace(self.arviz_params)
        plt.show()
        #az.plot_posterior(self.arviz_params, ref_val=0, color='#87ceeb')
        #plt.show()
        #az.plot_autocorr(self.arviz_params, max_lag=self.arviz_params.posterior.sizes['draw'])
        #plt.show()

    def __transform_data_to_inference_data(self):
        """
        transforms the sampled data to InferenceData object used by arviz
        :return: arvis.InferenceData
        """
        return az.convert_to_inference_data(
            {var_name: var[np.newaxis] for var_name, var in self.__raw_params.items() if "concentration" not in var_name})

    @property
    def raw_params(self):
        return self.__raw_params


