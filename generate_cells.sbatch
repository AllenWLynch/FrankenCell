#!/bin/bash

#SBATCH --array=1-100
#SBATCH --output=generate_log_%A_%a.out
#SBATCH --job-name=frankencells
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1gb
#SBATCH --cpus-per-task=5

conda run -p /liulab/alynch/projects/cuda \
    python generate-frankencells @data/read_depth_tests/rd_test_trial_${SLURM_ARRAY_TASK_ID}.txt -j 5