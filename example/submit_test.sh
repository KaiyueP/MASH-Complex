#!/bin/bash
#SBATCH -J fmo3_test
#SBATCH --qos=debug
#SBATCH --time=00:30:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --constraint=cpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kaiyue_peng@berkeley.edu

export OMP_NUM_THREADS=128
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

conda activate mash

python mash.py @fmo3.in > run.dat

