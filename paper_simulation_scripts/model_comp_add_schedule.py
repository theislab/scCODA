# Create bash script to execute model_comp_one_job.py
import os

models = ["Haber", "ttest", "clr_ttest"]

count = 0

for m in models:
    with open("/home/icb/johannes.ostner/compositional_diff/benchmark_scripts/paper_simulation_scripts/comp_add_script_" + str(count) + ".sh", "w") as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH -o /home/icb/johannes.ostner/compositional_diff/benchmark_results/out_add_" + str(count) + ".o\n")
        fh.writelines("#SBATCH -e /home/icb/johannes.ostner/compositional_diff/benchmark_results/error_add_" + str(count) + ".e\n")
        fh.writelines("#SBATCH -p icb_cpu\n")
        fh.writelines("#SBATCH --exclude=ibis-ceph-[002-006,008-019],ibis216-010-[011-012,020-037,051,064],icb-rsrv[05-06,08],ibis216-224-[010-011]\n")
        fh.writelines("#SBATCH --constraint='opteron_6378'")
        fh.writelines("#SBATCH -c 1\n")
        fh.writelines("#SBATCH --mem=5000\n")
        fh.writelines("#SBATCH --nice=100\n")
        fh.writelines("#SBATCH -t 2-00:00:00\n")
        fh.writelines("/home/icb/johannes.ostner/anaconda3/bin/python /home/icb/johannes.ostner/compositional_diff/benchmark_scripts/paper_simulation_scripts/model_comp_one_job.py " +
                str(m).replace(" ", "")
                      )

    os.system("sbatch /home/icb/johannes.ostner/compositional_diff/benchmark_scripts/paper_simulation_scripts/comp_add_script_" + str(count) + ".sh")
    count += 1
