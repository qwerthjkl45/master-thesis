#!/bin/bash

#SBATCH --time=18:60:60
#SBATCH -N 1
#SBATCH --gres=gpu:1

nvidia-smi
srun python -u train_model_with_sa.py model=iphone batch_size=30 dped_dir=/var/scratch/ycyang/DPED/dped/DIV_2K  eval_step=100 num_train_iters=10
