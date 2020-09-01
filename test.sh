#!/bin/bash
# Set job requirements
#SBATCH --job-name=R_CR2
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_shared

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=kondilidisn9@gmail.com

#Loading modules
module purge
module load pre2019
module load 2019
module load eb

module load python/3.5.0
module load cuDNN/7.0.5-CUDA-9.0.176
module load NCCL/2.0.5-CUDA-9.0.176
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/hpc/eb/Debian9/cuDNN/7.1-CUDA-8.0.44-GCCcore-5.4.0/lib64:$LD_LIBRARY_PATHs

# python3 train.py --model cat_aware --dataset redial --batch_size 16 --task ratings --hidden_layers 1000

python3 train.py --model auto --dataset redial --batch_size 8 --task ratings --hidden_layers 10
