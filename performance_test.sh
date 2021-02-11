#!/bin/sh
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --mem 120000
#SBATCH --time 8:00:00
#SBATCH -J solo_permformace_test_%j


source activate solo-sc
solo -g -r 2 -d 2 -t sum -o testdata/results solo_params_example.json testdata/2c.h5ad
            
python calculate_performancy.py