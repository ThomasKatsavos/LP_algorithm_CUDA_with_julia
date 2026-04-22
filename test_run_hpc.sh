#!/bin/bash

#SBATCH --job-name=lp_cuda

#SBATCH --partition=gpu

#SBATCH --gres=gpu:1

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4

#SBATCH --mem=16G

#SBATCH --time=00:20:00

#SBATCH --output=lp_results2_%j.log

module purge

module load gcc/14.2.0 julia/1.11.3 cuda/12.8.1


export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

export JULIA_PKG_PRECOMPILE_AUTO=0

cd ~/project2/julia_cuda/LP_algorithm_CUDA_with_julia/

julia --project=. benchmarks_run.jl
