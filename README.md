## Code for Multi-state MASH
Please note that this is a code package of MASH developped upon xxx
## Overview
This directory contains
- A main Python script `mash.py` 
- A script `model.py` with functions for setting up the model, handling input/output, sampling initial variables, and running a debug trajectory
- A Fortran source code in `src/` that runs the trajectories
These allow you to run MASH for various potentials, initial conditions and observables.

## How to compile
Prerequisites:
- gfortran (it also compiles and runs with ifort, but you will need to manually update the makefile)
- lapack
- numpy.f2py for Python3 

Compile by typing `make [option]` and one of the following options
- `clean` Remove compiled files and restart from clean directory
- `fast` Produces fast parallelized code
- `debug` Uses plenty of warning flags and should tell at which line the code breaks.

Also check that you have the required Python packages installed. 
One way to ensure this is to (preferably in a virtual environment) run `pip install -r requirements.txt`.

## How to run
```mash.py +[args].in```
where `[args].in` is an argument file with one line per argument (allowing commented lines). See `examples` for example input files.
There are plenty of available option flags and you can add more to suit your system.

Hints: 
- Make sure `mash.py` is executable, otherwise `chmod u+x mash.py`
- Create a symbolic link to `mash.py` in some place in your PATH, so that you can call it from anywhere.
- You may need to add the directory to your PYTHONPATH.

## What comes in
Run `mash.py -h` to see the available options. In general you should know about the following:
- `model` String specifying your model system.
- Model-specific parameters like `beta`, `Delta` etc.
- `init` Integer that specifies initial state (index starts at 0)
- `dt` Timestep
- `nt` Number of timesteps
- `ntraj` Number of trajectories
- `nucsamp` Nuclear sampling option (see "sampling" in model.py)
- `elsamp` Electronic sampling option (see "sampling" in model.py)
- `obstyp` Specify what kind of observables you want to measure (e.g. `pop` for populations)

## What comes out
Depends on what `obstyp` you specified, but for `pop` you should see a file `pop.out` which contains time in the first column and then the state populations in the following columns.

## Add a new potential
The Fortran code contains a few potentials, e.g. `linvib` and `tully`. If you want to create a different potential, add another file with the subroutines `pot` and `grad` specifying the diabatic potential and gradient. Copy the initialization routine from one of the existing potentials, and add a potential-specific init subroutine to `f2py.f90`.


# Parallelism on NERSC Perlmutter (CPU)

This repository runs MASH dynamics with a Python driver (`mash.py`) calling a Fortran backend (`mashf90`).  
It supports:
- **Hybrid parallelism**: MPI across ranks + OpenMP within each rank
- **Restartable checkpointing**: save intermediate `.npz` checkpoints and continue later, including extending `-ntraj`

This README focuses on **how to submit jobs correctly on NERSC** and how to use **checkpoint/restart** safely.

---

## 1) Parallelization model (MPI + OpenMP)

### What gets parallelized?
- **OpenMP threads**: used inside a single MPI rank to run `npar` trajectories per batch (one `runpar(...)` call).
- **MPI ranks**: run independent batches in parallel (each rank samples and advances its own trajectories). Global sums are reduced at checkpoints and at the end.

### Key rule: keep `-npar` consistent with OpenMP threads
In typical runs, you should set:
- `OMP_NUM_THREADS = SLURM_CPUS_PER_TASK`
- `-npar = SLURM_CPUS_PER_TASK`

This makes each MPI rank run exactly one batch of `npar` trajectories using `npar` OpenMP threads (1 thread per trajectory).

### Choose a node layout
Perlmutter CPU nodes have **128 physical cores** (2 sockets).

Two common choices:

**A) 2 MPI ranks per node, 64 threads per rank (recommended baseline)**
- `--ntasks-per-node=2`
- `--cpus-per-task=64`
- Total threads per node = 2 × 64 = 128

**B) 1 MPI rank per node, 128 threads per rank**
- `--ntasks-per-node=1`
- `--cpus-per-task=128`
- Useful if you prefer fewer ranks (less MPI overhead) or if per-rank memory is large.

---

## 2) Threading environment (avoid oversubscription)

Always make BLAS/LAPACK single-threaded, otherwise you get nested threading (very slow):

```bash
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

Recommended OpenMP binding on Perlmutter:

```bash
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
```

Important: `OMP_NUM_THREADS` must be a valid integer. In Slurm jobs, set it explicitly:

```bash
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
```

## 3) Checkpointing and restart

### What is saved?
When checkpointing is enabled, `mash.py` writes a restartable NumPy archive (default: `mash_ckpt.npz`) containing:
- `Bt_sum` (and optional `Ead_sum`): **sums** over all successful trajectories so far
- `success`: number of successful trajectories accumulated
- `attempted`: number attempted trajectories (multiple of `npar`)
- `discarded`: `attempted - success`
- `meta`: metadata (nf, ns, nt, dt, npar, basis, model, etc.) to reject incompatible restarts

### Flags
- `-ckpt`  
  Enable checkpointing.
- `-ckptfile mash_ckpt.npz`  
  Set checkpoint filename.
- `-ckptfrac 0.1`  
  Save checkpoint every fraction of the **final target** (default 0.1 → 10%, 20%, …, 90%).
- `-restart`  
  Load `-ckptfile` (if present) and continue. You can increase `-ntraj` to average more trajectories.

### Important rule: checkpoints are batch-aligned
The Fortran kernel runs exactly `npar` trajectories per call. Therefore:
- Effective trajectories are truncated to multiples of `npar`
- Checkpoints are written only at full-batch boundaries (`attempted % npar == 0`)

---

## 4) Typical workflow

### A) Start a run with checkpointing
```bash
python -u mash.py @lvc_ext.in \
  -ckpt -ckptfile mash_ckpt.npz -ckptfrac 0.1 \
  -ntraj 25600 -npar 64
```

### B) Restart and extend to more trajectories
Increase `-ntraj` to accumulate more statistics:

```bash
python -u mash.py @lvc_ext.in \
  -restart -ckpt -ckptfile mash_ckpt.npz -ckptfrac 0.1 \
  -ntraj 51200 -npar 64
```

The code will:
1. Load `mash_ckpt.npz`
2. Compute remaining batches to reach the new `-ntraj` target
3. Run only the remaining batches
4. Save updated averages and an updated checkpoint

---

## 5) Example Slurm submission script (Perlmutter CPU)

Example: **4 nodes**, **2 MPI ranks per node**, **64 OpenMP threads per rank**.

```bash
#!/bin/bash
#SBATCH -J CdSe_ext_states_110_init100
#SBATCH --qos=debug
#SBATCH --time=00:20:00
#SBATCH --constraint=cpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bkhou@berkeley.edu

# --- OpenMP binding ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# --- Make BLAS/LAPACK single-threaded ---
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

module load python
conda activate mash_env

# IMPORTANT:
# - Use srun so all MPI ranks launch.
# - Set -npar equal to cpus-per-task for best utilization.
# - -u makes python output unbuffered (useful in logs).
srun -n $SLURM_NTASKS --cpu-bind=cores \
  python -u mash.py @lvc_ext.in \
    -restart -ckpt -ckptfile mash_ckpt.npz -ckptfrac 0.1 \
    -ntraj 25600 -npar $SLURM_CPUS_PER_TASK \
  > run.dat 2>&1
```

---

## 6) Common pitfalls

### Pitfall 1: `OMP_NUM_THREADS` is empty (libgomp error)
Symptom:
```
libgomp: Invalid value for environment variable OMP_NUM_THREADS:
```
Fix:
```bash
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
```

### Pitfall 2: `-npar` does not match threads
Best practice:
```bash
-npar $SLURM_CPUS_PER_TASK
```

### Pitfall 3: `ntraj` not divisible by `npar`
The code effectively runs:
```text
ntraj_effective = (ntraj // npar) * npar
```
Choose `ntraj` as a multiple of `npar` for exact counts.

### Pitfall 4: Restarting with inconsistent parameters
Checkpoint metadata enforces consistency. Restart will fail if any of these differ:
- `nf, ns, nt, dt, npar, basis, model, obstyp, ead`

---

## 7) Outputs

Typical outputs include:
- Population (and optional Ead) output files written by `model.savedata(...)`
- `mash_ckpt.npz` (checkpoint file, if enabled)
- `log.out` (trajectory counters; format depends on your `mash.py`)




***

