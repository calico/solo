#!/bin/sh
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --mem 120000
#SBATCH --time 8:00:00
#SBATCH -J solo_permformace_test
#SBATCH --array=1-6
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/solo_performance_%A_%a.out
#SBATCH -e logs/solo_performance_%A_%a.err

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo 'pbmc'
source activate solo-sc
solo -p -a -g -r 2 --set-reproducible-seed "$SLURM_ARRAY_TASK_ID" -o results_pbmc_"$SLURM_ARRAY_TASK_ID" ../solo_params_example.json 2c.h5ad
            