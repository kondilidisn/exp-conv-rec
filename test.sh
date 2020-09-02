#!/bin/bash
# Set job requirements
#SBATCH --job-name=Thesis
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --partition=gpu_short

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=kondilidisn9@gmail.com

#Loading modules
module purge
module load pre2019
module load 2019
module load eb
module load Python/3.6.6-foss-2019b
#module load Python/3.7.5-foss-2019b
module load cuDNN/7.0.5-CUDA-9.0.176
module load NCCL/2.0.5-CUDA-9.0.176 

export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/hpc/eb/Debian9/cuDNN/7.1-CUDA-8.0.44-GCCcore-5.4.0/lib64:$LD_LIBRARY_PATHs

# python3 train.py --model cat_aware --dataset redial --batch_size 16 --task ratings --hidden_layers 1000

# python3 train.py --model auto --dataset redial --batch_size 8 --task ratings --hidden_layers 10

python3 train_bert.py --CLS_mode 1_CLS --cat_sa_alpha 1.0 --use_cuda True --conversations_per_batch 1 --task semantic --use_pretrained True

# python3 train_bert.py --CLS_mode 1_CLS --cat_sa_alpha 0.0 --use_cuda True --conversations_per_batch 1 --task semantic --use_pretrained True

# python3 train_bert.py --CLS_mode 1_CLS --cat_sa_alpha 0.5 --use_cuda True --conversations_per_batch 1 --task semantic --use_pretrained True