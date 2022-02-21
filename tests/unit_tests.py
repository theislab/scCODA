"""
Unit tests for scCODA
"""

import unittest
import numpy as np
import scanpy as sc
import tensorflow as tf
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))

from sccoda.util import cell_composition_data as dat
from sccoda.util import comp_ana as mod
from sccoda.util import data_generation as gen

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


class TestDataGeneration(unittest.TestCase):
    """
    Testing whether the data generation functions from data_generation work as intended
    Returns
    -------
    boolean -- all tests were passed or not
    """

    def setUp(self):

        self.N = 3
        self.D = 1
        self.K = 2
        self.n_total = [1000] * self.N
        self.noise_std_true = 1
        self.covariate_mean = None
        self.covariate_var = None
        self.sigma = None
        self.b_true = None
        self.w_true = None

    def test_case_control_gen(self):
        """
        Tests data generation for case/control scenarios
        Returns
        -------
        boolean -- all tests were passed or not
        """
        np.random.seed(1234)

        cases = 1
        K = 2
        n_total = 1000
        n_samples = [2, 2]
        noise_std_true = 0
        sigma = None
        b_true = None
        w_true = None

        data = gen.generate_case_control(cases, K, n_total, n_samples, sigma, b_true, w_true)

        test = True
        if any(np.abs(data.obs["x_0"] - [0, 0, 1, 1]) > 1e-5):
            print("obs is not correct!")
            test = False
        if not np.array_equal(data.X, np.array([[74., 926.], [58., 942.], [32., 968.], [53., 947.]])):
            print("X is not correct!")
            test = False
        if any(data.uns["b_true"] - np.array([-1.8508832,  0.7326526], dtype=np.float64) > 1e-5) or \
           not np.array_equal(data.uns["w_true"], np.array([[0., 0.]])):
            print("uns is not correct!")
            test = False

        self.assertTrue(test)

    def test_change_functions(self):
        """
        Tests gen.b_w_from_abs_change and gen.counts_from_first
        Returns
        -------
        boolean -- all tests were passed or not
        """
        np.random.seed(1234)
        correct = True

        counts_before = np.array([600, 400])
        abs_change = 100
        n_total = 1000
        K = 2
        b_0 = 600

        b, w = gen.b_w_from_abs_change(counts_before, abs_change, n_total)

        if any(np.abs(b - [-0.51082562, -0.91629073]) > 1e-5):
            print("gen.b_w_from_abs_change: b not correct!")
            correct = False

        if any(np.abs(w - [0.44183275, 0.]) > 1e-5):
            print("gen.b_w_from_abs_change: b not correct!")
            correct = False

        b_2 = gen.counts_from_first(b_0, n_total, K)
        if not np.array_equal(b_2, [600., 400.]):
            print("gen.counts_from_first not correct!")
            correct = False

        self.assertTrue(correct)


class TestDataImport(unittest.TestCase):

    def test_from_pandas(self):
        # Get Haber Salmonella data
        data_raw = pd.read_csv(os.path.abspath("sccoda/datasets/haber_counts.csv"))

        salm_indices = [0, 1, 2, 3, 8, 9]
        salm_df = data_raw.iloc[salm_indices, :]

        data_salm = dat.from_pandas(salm_df, covariate_columns=["Mouse"])
        data_salm.obs["Condition"] = data_salm.obs["Mouse"].str.replace(r"_[0-9]", "")

        # Only check size of x, obs
        x_shape = (data_salm.X.shape == (6, 8))
        obs_shape = (data_salm.obs.shape == (6, 2))

        self.assertTrue(x_shape & obs_shape)

    def test_from_scanpy(self):
        # Get scanpy example data, add covariates, read in three times
        adata_ref = sc.datasets.pbmc3k_processed()
        adata_ref.uns["cov"] = {"x_0": 0, "x_1": 1}
        adata_ref_1 = adata_ref.copy()
        adata_ref_1.uns["cov"] = {"x_0": 1, "x_1": 1}

        data = dat.from_scanpy_list([adata_ref, adata_ref, adata_ref_1],
                                    cell_type_identifier="louvain",
                                    covariate_key="cov")

        # Only check size of x, obs
        x_shape = (data.X.shape == (3, 8))
        obs_shape = (data.obs.shape == (3, 2))
        var_names = (data.var.index.tolist() == ['CD4 T cells', 'CD14+ Monocytes', 'B cells', 'CD8 T cells',
                                                 'NK cells', 'FCGR3A+ Monocytes', 'Dendritic cells', 'Megakaryocytes'])

        self.assertTrue(x_shape & obs_shape & var_names)


class TestModels(unittest.TestCase):

    def setUp(self):

        # Get Haber count data
        data_raw = pd.read_csv(os.path.abspath("sccoda/datasets/haber_counts.csv"))

        salm_indices = [0, 1, 2, 3, 8, 9]
        salm_df = data_raw.iloc[salm_indices, :]

        data_salm = dat.from_pandas(salm_df, covariate_columns=["Mouse"])
        data_salm.obs["Condition"] = data_salm.obs["Mouse"].str.replace(r"_[0-9]", "")
        self.data = data_salm

    def test_hmc(self):
        np.random.seed(1234)
        tf.random.set_seed(5678)

        model_salm = mod.CompositionalAnalysis(self.data, formula="Condition", reference_cell_type=5)

        # Run MCMC
        sim_results = model_salm.sample_hmc(num_results=20000, num_burnin=5000)
        self.sim_results = sim_results
        alpha_df, beta_df = sim_results.summary_prepare()

        # Mean cell counts for both groups
        alphas_true = np.round(np.mean(self.data.X[:4], 0), 0)
        betas_true = np.round(np.mean(self.data.X[4:], 0), 0)

        # Mean cell counts for simulated data
        final_alphas = np.round(alpha_df.loc[:, "Expected Sample"].tolist(), 0)
        final_betas = np.round(beta_df.loc[:, "Expected Sample"].tolist(), 0)

        # Check if model approximately predicts ground truth
        differing_alphas = any(np.abs(alphas_true - final_alphas) > 30)
        differing_betas = any(np.abs(betas_true - final_betas) > 30)

        self.assertTrue((not differing_alphas) & (not differing_betas))

    def test_hmc_da(self):
        np.random.seed(1234)
        tf.random.set_seed(5678)

        model_salm = mod.CompositionalAnalysis(self.data, formula="Condition", reference_cell_type=5)

        # Run MCMC
        sim_results = model_salm.sample_hmc_da(num_results=20000, num_burnin=5000)
        self.sim_results = sim_results
        alpha_df, beta_df = sim_results.summary_prepare()

        # Mean cell counts for both groups
        alphas_true = np.round(np.mean(self.data.X[:4], 0), 0)
        betas_true = np.round(np.mean(self.data.X[4:], 0), 0)

        # Mean cell counts for simulated data
        final_alphas = np.round(alpha_df.loc[:, "Expected Sample"].tolist(), 0)
        final_betas = np.round(beta_df.loc[:, "Expected Sample"].tolist(), 0)

        # Check if model approximately predicts ground truth
        differing_alphas = any(np.abs(alphas_true - final_alphas) > 30)
        differing_betas = any(np.abs(betas_true - final_betas) > 30)

        self.assertTrue((not differing_alphas) & (not differing_betas))

    def test_nuts(self):
        np.random.seed(1234)
        tf.random.set_seed(5678)

        model_salm = mod.CompositionalAnalysis(self.data, formula="Condition", reference_cell_type=5)

        # Run MCMC
        sim_results = model_salm.sample_nuts(num_results=2000, num_burnin=500)
        self.sim_results = sim_results
        alpha_df, beta_df = sim_results.summary_prepare()

        # Mean cell counts for both groups
        alphas_true = np.round(np.mean(self.data.X[:4], 0), 0)
        betas_true = np.round(np.mean(self.data.X[4:], 0), 0)

        # Mean cell counts for simulated data
        final_alphas = np.round(alpha_df.loc[:, "Expected Sample"].tolist(), 0)
        final_betas = np.round(beta_df.loc[:, "Expected Sample"].tolist(), 0)

        # Check if model approximately predicts ground truth
        differing_alphas = any(np.abs(alphas_true - final_alphas) > 30)
        differing_betas = any(np.abs(betas_true - final_betas) > 30)

        self.assertTrue((not differing_alphas) & (not differing_betas))

    def test_multi_cond(self):
        np.random.seed(1234)
        tf.random.set_seed(5678)

        self.data.obs["Condition2"] = np.random.randint(0, 2, len(self.data.obs))

        model_salm = mod.CompositionalAnalysis(self.data, formula="Condition+Condition2", reference_cell_type=5)

        # Run MCMC
        sim_results = model_salm.sample_hmc(num_results=20000, num_burnin=5000)
        self.sim_results = sim_results
        alpha_df, beta_df = sim_results.summary_prepare()

        # Mean cell counts for both groups
        alphas_true = np.round(np.mean(self.data.X[:4], 0), 0)
        betas_true = np.round(np.mean(self.data.X[4:], 0), 0)

        # Mean cell counts for simulated data
        final_alphas = np.round(alpha_df.loc[:, "Expected Sample"].tolist(), 0)
        final_betas = np.round(beta_df.loc[("Condition[T.Salm]",), "Expected Sample"].tolist(), 0)

        # Check if model approximately predicts ground truth
        differing_alphas = any(np.abs(alphas_true - final_alphas) > 30)
        differing_betas = any(np.abs(betas_true - final_betas) > 30)
        differing_rand = any(beta_df.loc[("Condition2",), "Final Parameter"] != 0)

        self.assertTrue((not differing_alphas) & (not differing_betas) & (not differing_rand))


if __name__ == '__main__':
    unittest.main()
