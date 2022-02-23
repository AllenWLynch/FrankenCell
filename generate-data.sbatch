#!/bin/bash

#SBATCH --array=1-75%10
#SBATCH --output=data/read_depth_tests/logs/generate_log_%A_%a.out
#SBATCH --job-name=frankencells
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2gb
#SBATCH --cpus-per-task=5

source /liulab/alynch/miniconda3/bin/activate
conda activate /liulab/alynch/projects/cuda

python generate-frankencells @data/read_depth_tests/trials/rd_test_trial_${SLURM_ARRAY_TASK_ID}.txt -j 5
