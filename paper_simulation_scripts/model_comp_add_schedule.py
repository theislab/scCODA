# Create bash script to execute model_comp_one_job.py
import os

models = ["simple_dm"]

for m in models:
    if m == "simple_dm":
        dataset_path = "/home/icb/johannes.ostner/compositional_diff/benchmark_results/generated_datasets/"
        num_files = len(os.listdir(dataset_path))

        for count in range(num_files):

            with open(
                    "/home/icb/johannes.ostner/compositional_diff/benchmark_scripts/paper_simulation_scripts/comp_add_script_" + str(m) + str(count) + ".sh", "w") as fh:
                fh.writelines("#!/bin/bash\n")
                fh.writelines("#SBATCH -o /home/icb/johannes.ostner/compositional_diff/benchmark_results/out_add_" + str(m) + str(count) + ".o\n")
                fh.writelines("#SBATCH -e /home/icb/johannes.ostner/compositional_diff/benchmark_results/error_add_" + str(m) + str(count) + ".e\n")
                fh.writelines("#SBATCH -p icb_cpu\n")
                fh.writelines("#SBATCH --exclude=ibis-ceph-[002-006,008-019],ibis216-010-[011-012,020-037,051,064],icb-rsrv[05-06,08],ibis216-224-[010-011]\n")
                fh.writelines("#SBATCH --constraint='opteron_6378'")
                fh.writelines("#SBATCH -c 1\n")
                fh.writelines("#SBATCH --mem=5000\n")
                fh.writelines("#SBATCH --nice=100\n")
                fh.writelines("#SBATCH -t 2-00:00:00\n")
                fh.writelines(
                    "/home/icb/johannes.ostner/anaconda3/bin/python /home/icb/johannes.ostner/compositional_diff/benchmark_scripts/paper_simulation_scripts/model_comp_one_job_batched.py " +
                    str(m).replace(" ", "") + " " +
                    str(count).replace(" ", "")
                    )

            os.system(
                "sbatch /home/icb/johannes.ostner/compositional_diff/benchmark_scripts/paper_simulation_scripts/comp_add_script_" + str(m) + str(count) + ".sh")

    else:
        with open("/home/icb/johannes.ostner/compositional_diff/benchmark_scripts/paper_simulation_scripts/comp_add_script_" + str(m) + ".sh", "w") as fh:
            fh.writelines("#!/bin/bash\n")
            fh.writelines("#SBATCH -o /home/icb/johannes.ostner/compositional_diff/benchmark_results/out_add_" + str(m) + ".o\n")
            fh.writelines("#SBATCH -e /home/icb/johannes.ostner/compositional_diff/benchmark_results/error_add_" + str(m) + ".e\n")
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

        os.system("sbatch /home/icb/johannes.ostner/compositional_diff/benchmark_scripts/paper_simulation_scripts/comp_add_script_" + str(m) + ".sh")
