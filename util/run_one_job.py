import sys
import ast
import numpy as np

from util import multi_parameter_sampling as mult

cases = ast.literal_eval(sys.argv[1])
print("cases:", cases)
K = ast.literal_eval(sys.argv[2])
print("K:", K)
# n_samples = [[i+1,j+1] for i in range(10) for j in range(10)]
n_total = ast.literal_eval(sys.argv[3])
print("n_total:", n_total)
n_samples = ast.literal_eval(sys.argv[4])
print("n_samples:", n_samples)
b_true = ast.literal_eval(sys.argv[5])
print("b_true:", b_true)
w_true = ast.literal_eval(sys.argv[6])
print("w_true:", w_true)
num_results = ast.literal_eval(sys.argv[7])
print("num_results:", num_results)
n = ast.literal_eval(sys.argv[8])
print("n:", n)

p = mult.MultiParamSimulation(cases, K, n_total, n_samples, b_true, w_true, num_results,
                              baseline_index=4, formula="x_0")

p.simulate()

p.save(path="/home/icb/johannes.ostner/compositional_diff/results/final_testing/multi_cov_test/no_baseline_new_005/",
       filename="result_b_" + str(np.round(b_true, 2)).replace("  ", " ") + "_w_" + str(w_true) + "_round_" + str(n))
