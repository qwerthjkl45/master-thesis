#!/bin/bash

#SBATCH --time=28:60:60
#SBATCH -N 1
#SBATCH --gres=gpu:1

nvidia-smi
srun python -u train_sid_sa.py 
