#!/bin/bash
#SBATCH -o /home/icb/johannes.ostner/compositional_diff/benchmark_results/data_gen_out.o
#SBATCH -e /home/icb/johannes.ostner/compositional_diff/benchmark_results/data_gen_error.e
#SBATCH -p icb_cpu
#SBATCH -c 1
#SBATCH -w ibis216-010-036
#SBATCH --mem-per-cpu=5000
#SBATCH --nice=100
#SBATCH -t 0-15:00:00

/home/icb/johannes.ostner/anaconda3/bin/python /home/icb/johannes.ostner/compositional_diff/benchmark_scripts/paper_simulation_scripts/generate_data.py



