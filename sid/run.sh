#!/bin/bash

#SBATCH --time=29:60:60
#SBATCH -N 1
#SBATCH --gres=gpu:1

nvidia-smi
srun python -u train.py 
