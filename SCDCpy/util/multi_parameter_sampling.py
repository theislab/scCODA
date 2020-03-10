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

from SCDCpy.util import compositional_analysis_generation_toolbox as gen
from SCDCpy.model import dirichlet_models as mod
from SCDCpy.util import comp_ana as ca

tfd = tfp.distributions
tfb = tfp.bijectors

#%%


class MultiParamSimulation:

    """
    Implements subsequent generation and simulation of datasets with a multitude of parameters,
     such as data dimensions, effect combinations or MCMC chain length.
    """

    def __init__(self, cases=[1], K=[5], n_total=[1000], n_samples=[[5,5]],
                 b_true=[None], w_true=[None], num_results=[10e3], baseline_index=None, formula="x_0"):
        """
        constructor. Simulated Parameters are passed to the constructor as lists, except the type of model, which is fixed for all simulations.
        The simulation is carried out over all possible combinations of specified parameters.
        See the default values for examples
        Parameters
        ----------
        cases -- Number of (binary) covariates
        K -- Number of cell types
        n_total -- number of cells per sample
        n_samples -- number of samples. Each sublist specifies the number of samples for each covariate combination, length 2**cases
        b_true -- Base composition. Each sublist has dimension K.
        w_true -- Effect composition. Each sublist is a nested list that represents a DxK effect matrix
        num_results -- MCMC chain length
        baseline_index -- Index of reference cellltype (None for no baseline)
        formula -- R-style formula used in model specification
        """

        # HMC Settings
        self.n_burnin = int(5e3)  # number of burn-in steps
        self.step_size = 0.01
        self.num_leapfrog_steps = 10

        # All parameter combinations
        self.simulation_params = list(itertools.product(cases, K, n_total, n_samples, b_true, w_true, num_results))

        # Setup result objects
        self.mcmc_results = {}
        self.parameters = pd.DataFrame(
            {'cases': [], 'K': [], 'n_total': [], 'n_samples': [], 'b_true': [], 'w_true': [], 'num_results': []})

        self.baseline_index = baseline_index
        self.formula = formula

    def simulate(self):
        """
        Generation and modeling of single-cell-like data
        Returns
        -------
        None. Fills up self.mcmc_results
        """

        i = 0
        # iterate over all parameter combinations

        for c, k, nt, ns, b, w, nr in self.simulation_params:
            # generate data set
            temp_data = gen.generate_case_control(cases=c, K=k, n_total=nt, n_samples=ns, b_true=b, w_true=w)

            # Save parameter set
            s = [c, k, nt, ns, b, w, nr]
            print('Simulating:', s)
            self.parameters.loc[i] = s

            # if baseline model: Simulate with baseline, else: without. The baseline index is always the last one
            ana = ca.CompositionalAnalysis(temp_data, self.formula, baseline_index=self.baseline_index)

            result_temp = ana.sample_hmc(num_results=int(nr), n_burnin=self.n_burnin,
                                         step_size=self.step_size, num_leapfrog_steps=self.num_leapfrog_steps)

            self.mcmc_results[i] = result_temp.summary_prepare()

            i += 1

        return None

    def get_discovery_rates(self):
        """
        Calculates discovery rates and other statistics for each effect, summarized over all entries of beta
        Returns
        -------
        None; extends self.parameters
        """

        tp = []
        tn = []
        fp = []
        fn = []
        ws = self.parameters.loc[:, "w_true"]

        # For all parameter sets:
        for i in range(len(self.parameters)):

            # Locate modelled slopes
            betas = self.mcmc_results[i][1]["final_parameter"].tolist()
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

        # add results to self.parameters
        self.parameters['tp'] = tp
        self.parameters['tn'] = tn
        self.parameters['fp'] = fp
        self.parameters['fn'] = fn

        return None

    def get_discovery_rates_per_param(self):
        """
        Discovery rates and other statistics for each entry of beta separately. This only works for cases==[1]
        Returns
        -------
        None, extends self.parameters
        """

        correct = []
        false = []
        ws = self.parameters.loc[:, "w_true"]

        K = len(ws[0][0])

        # For each parameter set:
        for i in range(len(self.parameters)):

            # Locate modelled slopes
            betas = self.mcmc_results[i][1]["final_parameter"].tolist()
            betas = [0 if b == 0 else 1 for b in betas]

            # Locate ground truth slopes
            wt = ws[i][0]
            wt = [0 if w == 0 else 1 for w in wt]

            correct_ = np.zeros(K)
            false_ = np.zeros(K)
            # Count how often each beta[k] is correctly/falsely identified
            for k in range(K):
                if wt[k] == betas[k]:
                    correct_[k] += 1
                else:
                    false_[k] += 1

            correct.append(correct_)
            false.append(false_)

        # Add results to self.paramerters
        for i in range(K):
            self.parameters['correct_'+str(i)] = [correct[n][i] for n in range(len(correct))]
            self.parameters['false_'+str(i)] = [false[n][i] for n in range(len(false))]

        return None

    def plot_discovery_rates(self, dim_1='w_true', dim_2='n_samples'):
        """
        plots TPR and TNR for two parameter series specified in the constructor (e.g. w_true vs. n_samples)
        Parameters
        ----------
        dim_1 -- parameter on x-axis
        dim_2 -- parameter on y-axis

        Returns
        -------
        a plot
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
        Parameters
        ----------
        path -- directory
        filename -- name of new file

        Returns
        -------

        """

        with open(path + filename + '.pkl', 'wb') as f:
            pkl.dump(self, f)


class Multi_param_simulation_multi_model:

    """
    Implements subsequent simulation of parameter sets with multiple models.
    This class provides a framework for running an instance of multi_param_simulation with more than one model.
    Each generated dataset is evaluated by each model
    """

    def __init__(self, cases=[1], K=[5], n_total=[1000], n_samples=[[5,5]],
                 b_true=[None], w_true=[None], num_results=[10e3], models=[mod.NoBaselineModel]):
        """

        Parameters
        ----------
        cases -- Number of (binary) covariates
        K -- Number of cell types
        n_total -- number of cells per sample
        n_samples -- number of samples. Each sublist specifies the number of samples for each covariate combination, length 2**cases
        b_true -- Base composition. Each sublist has dimension K.
        w_true -- Effect composition. Each sublist is a nested list that represents a DxK effect matrix
        num_results -- MCMC chain length
        models -- List of models to evaluate
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
            temp_data = gen.generate_case_control(cases=c, K=k, n_total=nt, n_samples=ns,
                                                                       b_true=b, w_true=w, sigma=np.identity(k) * 0.01)

            x_temp = temp_data.obs.values
            y_temp = temp_data.X

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
