#!/bin/bash
#SBATCH -J 3p0_CdSe_ext_30states
#SBATCH --qos=regular
#SBATCH --time=23:50:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --constraint=cpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bkhou@berkeley.edu

export OMP_NUM_THREADS=128
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

module load python
conda activate mash_env

python mash.py @lvc_ext.in > run.dat



