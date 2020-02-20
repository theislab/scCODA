import sys
import numpy as np
import os
from util import compositional_analysis_generation_toolbox as gen

cases = [1]
K = [5]
#n_samples = [[i+1,j+1] for i in range(10) for j in range(10)]
n_samples = [[i+1, i+1] for i in range(10)]
n_total = [5000]
num_results = [2e4]

#%%
b = []
for y1_0 in [20, 30, 50, 75, 115, 180, 280, 430, 667, 1000]:
    b.append(gen.counts_from_first(y1_0, 5000, 5))
print(b)

b_w_dict = {}
i=0
for b_i in b:
    b_t = np.log(b_i / 5000)
    w_t = []
    for change in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 400, 600, 800, 1000]:
        _, w = gen.b_w_from_abs_change(b_i, change, 5000)
        w_t.append(w)
    b_w_dict[i] = (b_t, w_t)
    i+=1

print(b_w_dict[0])
#%%

for i in range(10):
    b = b_w_dict[i][0]
    for w in b_w_dict[i][1]:
        print(b)
        print(w)
        for n in range(10):
            with open("/storage/groups/imm01/workspace/singleCellComposition/model/schedule_script.sh", "w") as fh:
                fh.writelines("#!/bin/bash\n")
                fh.writelines("#SBATCH -o /storage/groups/imm01/workspace/singleCellComposition/first_benchmark_1107/out.o\n")
                fh.writelines("#SBATCH -e /storage/groups/imm01/workspace/singleCellComposition/first_benchmark_1107/error.e\n")
                fh.writelines("#SBATCH -p serial_fed28\n")
                fh.writelines("#SBATCH -c 1\n")
                fh.writelines("#SBATCH --mem-per-cpu=5000\n")
                fh.writelines("#SBATCH --nice=100\n")
                fh.writelines("#SBATCH -t 0-15:00:00\n")
                fh.writelines("/home/icb/johannes.ostner/anaconda3/bin/python /storage/groups/imm01/workspace/singleCellComposition/model/run_one_job_3.py " +
                        str(cases).replace(" ", "") + " " +
                        str(K).replace(" ", "") + " " +
                        str(n_total).replace(" ", "") + " " +
                        str(n_samples).replace(" ", "") + " " +
                        str(b).replace(" ", "") + " " +
                        str(w).replace(" ", "") + " " +
                        str(num_results).replace(" ", "") + " " +
                        str(n).replace(" ", ""))

            os.system("sbatch schedule_script.sh")

