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
echo 'kidney'
source activate solo-sc
solo -g -r 2 -d 2 -t sum -o results_kidney_"$SLURM_ARRAY_TASK_ID" ../solo_params_example.json gene_ad_filtered_PoolB4FACs_L4_Rep1.h5ad
            
