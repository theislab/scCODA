""""
This file contains a framework to simulate multiple sets of parameters, and aggregate the results.
The functions to evaluate the results via plots, ... can be found in 'multi_parameter_analysis_functions'

:authors: Johannes Ostner
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import pickle as pkl
import datetime
from sklearn.metrics import confusion_matrix

import tensorflow_probability as tfp
import itertools

tfd = tfp.distributions
tfb = tfp.bijectors

from util import compositional_analysis_generation_toolbox as gen
from model import dirichlet_models as mod

#%%

class Multi_param_simulation:

    """
    Implements subsequent generation and simulation of datasets with a multitude of parameters, such as data dimensions, effect combinations or MCMC chain length.
    Parameters are passed to the
    """


    def __init__(self, cases=[1], K=[5], n_total=[1000], n_samples=[[5,5]],
                 b_true=[None], w_true=[None], num_results=[10e3], model=mod.compositional_model_no_baseline):

        """
        constructor. Simulated Parameters are passed to the constructor as lists, except the type of model, which is fixed for all simulations.
        The simulation is carried out over all possible combinations of specified parameters.
        See the default values for examples

        :param cases: list of int - Number of (binary) covariates
        :param K: list of int - Number of cell types
        :param n_total: list of int - number of cells per sample
        :param n_samples: list of lists - number of samples. Each sublist specifies the number of samples for each covariate combination, length 2**cases
        :param b_true: list of lists - Base composition. Each sublist has dimension K.
        :param w_true: list of lists - Effect composition. Each sublist is a nested list that represents a DxK effect matrix
        :param num_results: list of int - MCMC chain length
        :param model: Model used
        """

        # HMC Settings
        self.n_burnin = int(5e3)  # number of burn-in steps
        self.step_size = 0.01
        self.num_leapfrog_steps = 10

        # All parameter combinations
        self.l = list(itertools.product(cases, K, n_total, n_samples, b_true, w_true, num_results))

        # Setup result objects
        self.mcmc_results = {}
        self.parameters = pd.DataFrame(
            {'cases': [], 'K': [], 'n_total': [], 'n_samples': [], 'b_true': [], 'w_true': [], 'num_results': []})

        self.model = model

    def simulate(self, keep_raw_params=True):

        """
        Generation and modeling of single-cell-like data
        :param keep_raw_params: boolean - if True, all MCMC values are saved. Caution! Eats a lot of memory
        :return: None. Fills up self.mcmc_results
        """

        i = 0

        # iterate over all parameter combinations

        for c, k, nt, ns, b, w, nr in self.l:
            # generate data set
            x_temp, y_temp, b_temp, w_temp = gen.generate_case_control(cases=c, K=k, n_total=nt, n_samples=ns,
                                                                       b_true=b, w_true=w)

            # Save parameter set
            s = [c, k, nt, ns, b, w, nr]
            print('Simulating:', s)
            self.parameters.loc[i] = s

            # if baseline model: Simulate with baseline, else: without. The baseline index is always the last one
            if self.model == mod.compositional_model_baseline:
                model_temp = self.model(x=x_temp, y=y_temp, n_total=np.repeat(nt, x_temp.shape[0]), baseline_index=k-1)
            else:
                model_temp = self.model(x=x_temp, y=y_temp, n_total=np.repeat(nt, x_temp.shape[0]))

            # HMC sampling, save results
            result_temp = model_temp.sample(int(nr), self.n_burnin, self.num_leapfrog_steps, self.step_size)
            if keep_raw_params == False:
                result_temp._MCMCResult__raw_params = {}
            self.mcmc_results[i] = result_temp

            i += 1

        return None

    def get_discovery_rates(self):

        """
        Calculates discovery rates and other statistics for each effect, summarized over all entries of beta
        :return: None; extends self.parameters
        """

        def get_discovery_rate(res):

            tp = []
            tn = []
            fp = []
            fn = []
            ws = self.parameters.loc[:, "w_true"]

            # For all parameter sets:
            for i in range(len(self.parameters)):

                # Locate modelled slopes
                par = res[i].params
                betas = par.loc[par.index.str.contains('beta')]["final_parameter"].tolist()
                betas = [0 if b == 0 else 1 for b in betas]

                # Locate ground truth slopes
                wt = [item for sublist in ws[i] for item in sublist]
                wt = [0 if w == 0 else 1 for w in wt]

                # Calculate confusion matrix: beta[d,k]==0 vs. beta[d,k] !=0
                tn_, fp_, fn_, tp_ = confusion_matrix(wt, betas).ravel()

                tp.append(tp_)
                tn.append(tn_)
                fp.append(fp_)
                fn.append(fn_)

            return tp, tn, fp, fn

        # add results to self.parameters
        rates = get_discovery_rate(self.mcmc_results)
        self.parameters['tp'] = rates[0]
        self.parameters['tn'] = rates[1]
        self.parameters['fp'] = rates[2]
        self.parameters['fn'] = rates[3]

        return None

    def get_discovery_rates_per_param(self):

        """
        Discovery rates and other statistics for each entry of beta separately. This only works for cases==[1]
        :return: None, extends self.parameters
        """


        def get_discovery_rate_per_param(res):

            correct = []
            false = []
            ws = self.parameters.loc[:, "w_true"]

            # For each parameter set:
            for i in range(len(self.parameters)):

                # Locate modelled slopes
                par = res[i].params
                betas = par.loc[par.index.str.contains('beta')]["final_parameter"].tolist()
                betas = [0 if b == 0 else 1 for b in betas]

                # Locate ground truth slopes
                wt = ws[i][0]
                wt = [0 if w == 0 else 1 for w in wt]

                K = len(wt)

                correct_ = np.zeros(K)
                false_ = np.zeros(K)
                # Count how often each beta[k] is correctly/falsely identified
                for i in range(K):
                    if wt[i] == betas[i]:
                        correct_[i] += 1
                    else:
                        false_[i] += 1

                correct.append(correct_)
                false.append(false_)

            return correct, false

        correct, false = get_discovery_rate_per_param(self.mcmc_results)
        K = len(correct[0])
        # Add results to self.paramerters
        for i in range(K):
            self.parameters['correct_'+str(i)] = [correct[n][i] for n in range(len(correct))]
            self.parameters['false_'+str(i)] = [false[n][i] for n in range(len(false))]

        return None


    def plot_discovery_rates(self, dim_1='w_true', dim_2='n_samples'):
        """
        plots TPR and TNR for two parameter series specified in the constructor (e.g. w_true vs. n_samples)
        :param dim_1: string - parameter on x-axis
        :param dim_2: string - parameter on y-axis
        :return: None - plot!!!
        """

        plot_data = self.parameters[[dim_1, dim_2, "tpr_mcmc", "tnr_mcmc"]]
        plot_data[[dim_1, dim_2]] = plot_data[[dim_1, dim_2]].astype(str)

        fig, ax = plt.subplots(1, 2)
        sns.heatmap(plot_data.pivot(dim_1, dim_2, 'tpr_mcmc'), ax=ax[0]).set_title("MCMC TPR")
        sns.heatmap(plot_data.pivot(dim_1, dim_2, 'tnr_mcmc'), ax=ax[1]).set_title("MCMC TNR")
        plt.show()


    def save(self, path='''/Users/Johannes/Documents/Uni/Master's Thesis/simulation_results/tests/''',
             filename=str(datetime.datetime.now())):
        """
        saves results to a pickle file
        :param path: string - directory
        :param filename: string - file name
        :return:
        """
        with open(path + filename + '.pkl', 'wb') as f:
            pkl.dump(self, f)


#%%
class Multi_param_simulation_multi_model:

    """
    Implements subsequent simulation of parameter sets with multiple models
    """

    def __init__(self, cases=[1], K=[5], n_total=[1000], n_samples=[[5,5]],
                 b_true=[None], w_true=[None], num_results=[10e3], models=[mod.compositional_model_no_baseline]):

        """
        Constructor - This class provides a framework for running an instance of multi_param_simulation with more than one model.
        Each generated dataset is evaluated by each model
        :param cases: See multi_param_simulation
        :param K: See multi_param_simulation
        :param n_total: See multi_param_simulation
        :param n_samples: See multi_param_simulation
        :param b_true: See multi_param_simulation
        :param w_true: See multi_param_simulation
        :param num_results: See multi_param_simulation
        :param models: List of models to evaluate
        """

        # HMC Settings
        self.n_burnin = int(5e3)  # number of burn-in steps
        self.step_size = 0.01
        self.num_leapfrog_steps = 10

        # All parameter combinations
        self.l = list(itertools.product(cases, K, n_total, n_samples, b_true, w_true, num_results))

        # Setup result objects
        self.results = {}
        self.parameters = pd.DataFrame(
            {'cases': [], 'K': [], 'n_total': [], 'n_samples': [], 'b_true': [], 'w_true': [], 'num_results': []})

        self.models = models

    def simulate(self, keep_raw_params=True):
        """
        Generation and modeling of single-cell-like data
        :param keep_raw_params: boolean - if True, all MCMC values are saved. Caution! Eats a lot of memory
        :return: None. Fills up self.mcmc_results
        """

        for j in range(len(self.models)):
            self.results[j] = {}

        i = 0

        # For each parameter combination:
        for c, k, nt, ns, b, w, nr in self.l:
            # Generate dataset
            x_temp, y_temp, b_temp, w_temp = gen.generate_case_control(cases=c, K=k, n_total=nt, n_samples=ns,
                                                                       b_true=b, w_true=w, sigma=np.identity(k) * 0.01)

            # Write parameter combination
            s = [c, k, nt, ns, b, w, nr]
            print('Simulating:', s)
            self.parameters.loc[i] = s

            j=0

            # For each model:
            for model in self.models:

                # if baseline model: Simulate with baseline, else: without. The baseline index is always the last one
                if model == model == mod.compositional_model_baseline:
                    model_temp = model(x=x_temp, y=y_temp, n_total=np.repeat(nt, x_temp.shape[0]), baseline_index=k-1)
                else:
                    model_temp = model(x=x_temp, y=y_temp, n_total=np.repeat(nt, x_temp.shape[0]))

                # HMC sampling, save results
                result_temp = model_temp.sample(int(nr), self.n_burnin, self.num_leapfrog_steps, self.step_size)
                if keep_raw_params == False:
                    result_temp._MCMCResult__raw_params = {}

                self.results[j][i] = result_temp
                j += 1

            i += 1

        return None

    def get_discovery_rates(self):
        """
        Calculates discovery rates and other statistics for each effect, summarized over all entries of beta
        :return: None; extends self.parameters
        """

        def get_discovery_rate(res):
            tp = []
            tn = []
            fp = []
            fn = []
            ws = self.parameters.loc[:, "w_true"]

            # For all parameter sets:
            for i in range(len(self.parameters)):
                # Locate modelled slopes
                par = res[i].params
                betas = par.loc[par.index.str.contains('beta')]["final_parameter"].tolist()
                betas = [0 if b == 0 else 1 for b in betas]

                # Locate ground truth slopes
                wt = [item for sublist in ws[i] for item in sublist]
                wt = [0 if w == 0 else 1 for w in wt]

                # Calculate confusion matrix: beta[d,k]==0 vs. beta[d,k] !=0
                tn_, fp_, fn_, tp_ = confusion_matrix(wt, betas).ravel()

                tp.append(tp_)
                tn.append(tn_)
                fp.append(fp_)
                fn.append(fn_)

            return tp, tn, fp, fn

        # add results to self.parameters
        for j in range(len(self.models)):
            rates = get_discovery_rate(self.results[j])
            self.parameters['tp_'+str(j)] = rates[0]
            self.parameters['tn_'+str(j)] = rates[1]
            self.parameters['fp_'+str(j)] = rates[2]
            self.parameters['fn_'+str(j)] = rates[3]

        return None

    def save(self, path='''/Users/Johannes/Documents/Uni/Master's Thesis/simulation_results/tests/''',
             filename=str(datetime.datetime.now())):
        """
        saves results to a pickle file
        :param path: string - directory
        :param filename: string - file name
        :return:
        """
        with open(path + filename + '.pkl', 'wb') as f:
            pkl.dump(self, f)
