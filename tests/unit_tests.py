"""
Unit tests for SCDC_dm
"""

import unittest
import numpy as np
import arviz as az
import pandas as pd
import pickle as pkl
import importlib
import anndata as ad
import ast

from scdcdm.util import result_classes as res
from scdcdm.util import multi_parameter_sampling as mult
from scdcdm.util import multi_parameter_analysis_functions as ana
from scdcdm.util import data_generation as gen

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

#%%


class TestDataGeneration(unittest.TestCase):

    def test_data_generation(self):
        """
        Testing whether the data generation functions from data_generation work as intended
        Returns
        -------
        boolean -- all tests were passed or not
        """
        np.random.seed(1234)

        N = 3
        D = 1
        K = 2
        n_total = [1000]*N
        noise_std_true = 1
        covariate_mean = None
        covariate_var = None
        sigma = None
        b_true = None
        w_true = None

        test_1 = True
        data_sc_1 = gen.generate_normal_uncorrelated(N, D, K, n_total, noise_std_true)
        if any(np.abs(data_sc_1.obs["x_0"] - [-0.720589, 0.887163, 0.859588]) > 1e-5):
            print("scenario1.obs is not correct!")
            test_1 = False
        if not np.array_equal(data_sc_1.X, np.array([[591., 409.], [959., 41.], [965.,  35.]])):
            print("scenario1.X is not correct!")
            test_1 = False

        test_2 = True
        data_sc_2 = gen.generate_normal_correlated(N, D, K, n_total, noise_std_true, covariate_mean, covariate_var)
        if any(np.abs(data_sc_2.obs["x_0"] - [-0.202646, -0.655969, 0.193421]) > 1e-5):
            print("scenario2.obs is not correct!")
            test_2 = False
        if not np.array_equal(data_sc_2.X, np.array([[383., 617.], [162., 838.], [680., 320.]])):
            print("scenario2.X is not correct!")
            test_2 = False

        test_3 = True
        data_sc_3 = gen.generate_normal_xy_correlated(N, D, K, n_total, noise_std_true, covariate_mean, covariate_var, sigma)
        if any(np.abs(data_sc_3.obs["x_0"] - [0.841675, 2.390960, 0.076200]) > 1e-5):
            print("scenario3.obs is not correct!")
            test_3 = False
        if not np.array_equal(data_sc_3.X, np.array([[598., 402.], [878., 122.], [104., 896.]])):
            print("scenario3.X is not correct!")
            test_3 = False

        test_4 = True
        data_sc_4 = gen.generate_sparse_xy_correlated(N, D, K, n_total, noise_std_true, covariate_mean, covariate_var, sigma,
                                                      b_true, w_true)
        if any(np.abs(data_sc_4.obs["x_0"] - [0.34769, -0.55482, -0.804660]) > 1e-5):
            print("scenario4.obs is not correct!")
            test_4 = False
        if not np.array_equal(data_sc_4.X, np.array([[550., 450.], [796., 204.], [848., 152.]])):
            print("scenario4.X is not correct!")
            test_4 = False

        assert all([test_1, test_2, test_3, test_4])

        return True

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

        data = gen.generate_case_control(cases, K, n_total, n_samples, noise_std_true, sigma, b_true, w_true)

        test = True
        if any(np.abs(data.obs["x_0"] - [0, 0, 1, 1]) > 1e-5):
            print("obs is not correct!")
            test = False
        if not np.array_equal(data.X, np.array([[74., 926.], [58., 942.], [32., 968.], [53., 947.]])):
            print("X is not correct!")
            test = False
        if not np.array_equal(data.uns["b_true"], np.array([-1.8508832,  0.7326526], dtype=np.float32)) & \
            np.array_equal(data.uns["w_true"], np.array([[0., 0.]])):
            print("uns is not correct!")
            test = False

        assert test

        return True

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

        assert correct

        return True


if __name__ == '__main__':
    unittest.main()
