
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


pip3 install jsonlines --user