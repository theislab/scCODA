"""
This file contains
Results objects that summarize the results of the different
inference methods and calculates test statistics


:authors: Benjamin Schubert
"""
import numpy as np
import arviz as az
import pandas as pd
import scipy.stats as st
from abc import ABCMeta, abstractmethod
#import warnings
#warnings.filterwarnings("ignore")


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


class MCMCResult(AResult):
    """
    Result class for MCMC samples
    """

    def __init__(self, N, params):
        """
        Init function
        :param N: The sample size
        :param params: the trace of the parameters
        """
        self.N = N
        self.__raw_params = params
        self.arvis_params = self.__transform_data_to_inference_data()
        df = az.summary(self.arvis_params)
        self.params = df[df.index.str.match("|".join(["alpha", "beta", "gamma"]))]
        self.params["z"] = np.abs(np.divide(self.params["mean"], self.params["sd"]))
        self.params["Pr(>|z|)"] = 2 * (1 - st.norm(loc=0, scale=1).cdf(self.params["z"]))

    def summary(self, varnames=None):
        if varnames is None:
            return self.params
        else:
            return self.params[self.params.index.str.contains("|".join(varnames))]

    def plot(self, varnames=None):
        return az.plot_trace(self.arvis_params)

    def __transform_data_to_inference_data(self):
        """
        transforms the sampled data to InferenceData object used by arviz

        :TODO: For now it will add an additional dimension representing the number of chains
        :return: arvis.InferenceData
        """
        return az.convert_to_inference_data(
            {var_name: var[np.newaxis] for var_name, var in self.__raw_params.items()})

    @property
    def raw_params(self):
        return self.__raw_params


class MAPResult(AResult):
    """
    Result class for MAP estimates
    """
    def __init__(self, N, params):
        """
        Init function
        :param N: the sample size
        :param params: the MAP estimates of the parameters
        """
        self.N = N
        self.__raw_params = params
        # is the already finalized pandas dataframe
        self.params = self.__calculate_statistcs()

    def summary(self, varnames=None):
        if varnames is None:
            return self.params
        else:
            return self.params[self.params.index.str.contains("|".join(varnames))]

    def plot(self, varnames=None):
        pass

    def __calculate_statistcs(self):
        tmp = {"var":[], "mean":[], "sd":[], "CI_low":[], "CI_high":[], "z":[], "Pr(>|z|)":[]}
        params = np.concatenate([v.flatten() for n,v in self.__raw_params["params"].items()])
        sd = np.concatenate([v.flatten() for n, v in self.__raw_params["sds"].items()])
        ci_low, ci_high = st.t.interval(0.95, self.N - 1, loc=params, scale=sd)
        z = np.abs(np.divide(params, sd))
        p = 2 * (1 - st.norm(loc=0, scale=1).cdf(z))

        for var_name, v in self.__raw_params["params"].items():
            if len(v.shape) <= 1:
                tmp["var"].extend("{}[{}]".format(var_name, c) for c in range(len(v)))
            else:
                shape = v.shape
                tmp["var"].extend("{}[{},{}]".format(var_name, *np.unravel_index(c, shape))
                                  for c in range(shape[0]*shape[1]))

        tmp["mean"] = params
        tmp["sd"] = sd
        tmp["CI_low"] = ci_low
        tmp["CI_high"] = ci_high
        tmp["z"] = z
        tmp["Pr(>|z|)"] = p

        return pd.DataFrame.from_dict(tmp).set_index("var")

    @property
    def raw_params(self):
        return self.__raw_params






