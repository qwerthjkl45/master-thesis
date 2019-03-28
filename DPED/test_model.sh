#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

nvidia-smi 
srun -u python3 test_model.py model=iphone dped_dir=/var/scratch/ycyang/DPED/input/ test_subset=full resolution=orig use_gpu=true iteration=200 dir_model=models/
