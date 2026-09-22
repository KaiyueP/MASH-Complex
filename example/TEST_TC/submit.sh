#!/bin/bash
#SBATCH -J mash-complex-test 
#SBATCH --qos=debug
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --constraint=cpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kaiyue_peng@berkeley.edu

export OMP_NUM_THREADS=128
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

HomeDir=$(pwd)
ScrDir=/pscratch/sd/k/kaiyuep/mash_test/TC_complex_test
mkdir -p "$ScrDir"

cp mash.py tc_ext.in tc_params_AU.npz "$ScrDir/" 
cd "$ScrDir/"

module load python
module load conda
conda activate mash

python mash.py @tc_ext.in -ckpt -ckptfile test_complex_ckpt.npz -ckptfrac 0.01 > run.dat

cp run.dat pop.out "$HomeDir/"


