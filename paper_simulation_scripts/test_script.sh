#!/bin/bash
#SBATCH -o /home/icb/johannes.ostner/compositional_diff/compositionalDiff-johannes_tests_2/benchmark_results/out.o
#SBATCH -e /home/icb/johannes.ostner/compositional_diff/compositionalDiff-johannes_tests_2/benchmark_results/error.e
#SBATCH -p icb_cpu
#SBATCH -c 1
#SBATCH -w ibis216-010-036
#SBATCH --mem-per-cpu=5000
#SBATCH --nice=100
#SBATCH -t 0-15:00:00

/home/icb/johannes.ostner/anaconda3/bin/python /home/icb/johannes.ostner/compositional_diff/compositionalDiff-johannes_tests_2/server_scripts/test_python.py



