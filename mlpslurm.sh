#!/bin/bash -l
#SBATCH --parsable
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --mem-per-gpu=12G
#SBATCH --time=24:00:00
##SBATCH --exclude=gpu113,gpu114,gpu118,gpu119,gpu123,gpu127,gpu136,gpu137,gpu138,gpu139,gpu238
#SBATCH --exclude=gpu150,gpu123,gpu124,gpu125,gpu126,gpu127,gpu113,gpu114,gpu118,gpu119

source /nfs/scistore14/chenggrp/ptuo/pkgs/deepmd-kit/sourceme.sh
conda activate seq

python mlp_train_simplex.py 128 MLP
