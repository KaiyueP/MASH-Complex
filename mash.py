#!/usr/bin/env python3
import numpy as np
import os
import sys
sys.path.insert(0,'/global/homes/k/kaiyuep/multimash')
import time
import json

import model
from src import mashf90

# ============================================================
# Optional MPI support
# ============================================================
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    mpi_enabled = (size > 1)
    if mpi_enabled:
        print(f"[MPI] Rank {rank}/{size} initialized.")
except Exception:
    MPI = None
    comm = None
    rank = 0
    size = 1
    mpi_enabled = False


def fmt_seconds(sec: float) -> str:
    h = int(sec // 3600)
    sec -= 3600 * h
    m = int(sec // 60)
    sec -= 60 * m
    return f"{h:d}:{m:02d}:{sec:06.3f}"


def r0_print(*args, **kwargs):
    if rank == 0:
        print(*args, **kwargs, flush=True)

# ---------------- checkpoint helpers ----------------

def _ckpt_meta(args, nf, ns):
    """Small metadata blob to reject incompatible restarts."""
    return dict(
        model=str(args.model),
        obstyp=str(args.obstyp),
        basis=str(args.basis),
        units=str(getattr(args, "units", "")),
        beta=float(args.beta),
        nf=int(nf),
        ns=int(ns),
        nt=int(args.nt),
        dt=float(args.dt),
        npar=int(args.npar),
        ead=bool(getattr(args, "ead", False)),
    )


def save_checkpoint(fname, meta, Bt_sum, success, attempted, discarded, Ead_sum=None):
    """Atomic save: write tmp then rename."""
    tmp = fname + ".tmp.npz"
    payload = dict(
        meta=json.dumps(meta),
        Bt_sum=Bt_sum,
        success=np.int64(success),
        attempted=np.int64(attempted),
        discarded=np.int64(discarded),
    )
    if Ead_sum is not None:
        payload["Ead_sum"] = Ead_sum
    np.savez(tmp, **payload)
    os.replace(tmp, fname)


def load_checkpoint(fname):
    z = np.load(fname, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    Bt_sum = z["Bt_sum"]
    success = int(z["success"])
    attempted = int(z["attempted"])
    discarded = int(z["discarded"])
    Ead_sum = z["Ead_sum"] if "Ead_sum" in z.files else None
    return meta, Bt_sum, success, attempted, discarded, Ead_sum


# ---------------- timing ----------------
t_total0 = time.perf_counter()
t_fortran = 0.0
t_loop0 = None


# ============================================================
# Read input parameters
# ============================================================
args = model.read_args()

def announce_tc_setup(args):
    if args.model == "tc":
        print("[TC] QD baths are loaded from tc_params_AU.npz; cavity states are phonon-free.")

announce_tc_setup(args)

# RNG seeding per rank (avoid identical streams across MPI ranks)
base_seed = getattr(args, "seed", None)
if base_seed is None:
    base_seed = 12345
seed_rank = (int(base_seed) + 1000003 * rank) % (2**32)
np.random.seed(seed_rank)

# commonly used args
beta = args.beta
nf = args.nf
dt = args.dt
nt = args.nt
ntraj_requested = args.ntraj
npar = args.npar
obstyp = args.obstyp

# MPI only supported for population observable in this script
if mpi_enabled and obstyp != "pop":
    if rank == 0:
        print("MPI mode currently implemented only for obstyp='pop'.")
        print("Run with a single rank for obstyp='nuc', or extend reductions for nuc observables.")
    comm.Barrier()
    sys.exit(1)

# OpenMP / BLAS threading control
if npar > 1 and ("OMP_NUM_THREADS" not in os.environ):
    os.environ["OMP_NUM_THREADS"] = str(npar)

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# time grid
t = np.arange(nt + 1) * dt

# ============================================================
# Initialize model / Fortran module
# ============================================================
mass, omega, nf, ns = model.setup_model(args)
mashf90.init_mash(beta)

# Debug (rank 0 only)
if getattr(args, "debug", False):
    if rank == 0:
        model.debug(args, mass, omega, nf, ns)
    if mpi_enabled:
        comm.Barrier()
    sys.exit(0)

# ============================================================
# Allocate observables (sums over successful trajectories)
# ============================================================
if obstyp == "pop":
    Bt_local = np.zeros((nt + 1, ns), dtype=float)
    # Ead_local[:,0] = <Ead>(t)
    # Ead_local[:,1] = <V0>(t)
    Ead_local = np.zeros((nt + 1, 2), dtype=float) if getattr(args, "ead", False) else None

# ============================================================
# Checkpoint / restart state (offsets from previous runs)
# ============================================================
ckpt_enabled = bool(getattr(args, "ckpt", False))
ckptfile = str(getattr(args, "ckptfile", "mash_ckpt.npz"))
ckptfrac = float(getattr(args, "ckptfrac", 0.1))
do_restart = bool(getattr(args, "restart", False))

Bt_offset = np.zeros((nt + 1, ns), dtype=float)
Ead_offset = np.zeros((nt + 1, 2), dtype=float) if getattr(args, "ead", False) else None
attempted_offset = 0
success_offset = 0
discarded_offset = 0

meta_now = _ckpt_meta(args, nf, ns)

if do_restart and os.path.exists(ckptfile):
    if mpi_enabled:
        if rank == 0:
            meta_old, Bt_offset, success_offset, attempted_offset, discarded_offset, Ead_offset = load_checkpoint(ckptfile)
        else:
            meta_old = None

        meta_old = comm.bcast(meta_old, root=0)

        # sanity check meta on all ranks
        for k in ["model", "obstyp", "basis", "units", "beta", "nf", "ns", "nt", "dt", "npar", "ead"]:
            oldv = meta_old.get(k, None)
            newv = meta_now.get(k, None)
            if oldv != newv:
                if rank == 0:
                    print(f"Checkpoint meta mismatch on key '{k}': {oldv} != {newv}")
                comm.Barrier()
                sys.exit(1)

        attempted_offset = comm.bcast(attempted_offset if rank == 0 else None, root=0)
        success_offset   = comm.bcast(success_offset   if rank == 0 else None, root=0)
        discarded_offset = comm.bcast(discarded_offset if rank == 0 else None, root=0)

        if rank != 0:
            Bt_offset = np.zeros((nt + 1, ns), dtype=float)
            if getattr(args, "ead", False):
                Ead_offset = np.zeros(nt + 1, dtype=float)

        comm.Bcast(Bt_offset, root=0)
        if getattr(args, "ead", False):
            comm.Bcast(Ead_offset, root=0)

        if rank == 0:
            r0_print("")
            r0_print("========== Checkpoint restart ==========")
            r0_print(f"File       : {ckptfile}")
            r0_print(f"Attempted  : {attempted_offset}")
            r0_print(f"Success    : {success_offset}")
            r0_print(f"Discarded  : {discarded_offset}")
            r0_print("========================================")
            r0_print("")

    else:
        meta_old, Bt_offset, success_offset, attempted_offset, discarded_offset, Ead_offset = load_checkpoint(ckptfile)
        for k in ["model", "obstyp", "basis", "units", "beta", "nf", "ns", "nt", "dt", "npar", "ead"]:
            oldv = meta_old.get(k, None)
            newv = meta_now.get(k, None)
            if oldv != newv:
                print(f"Checkpoint meta mismatch on key '{k}': {oldv} != {newv}")
                sys.exit(1)

        print(f"[restart] loaded {ckptfile}: attempted={attempted_offset}, success={success_offset}, discarded={discarded_offset}")

# enforce batch consistency
if attempted_offset % max(1, npar) != 0:
    if rank == 0:
        print(f"ERROR: checkpoint attempted={attempted_offset} is not divisible by npar={npar}.")
        print("This script assumes checkpoints are saved only at full-batch boundaries.")
    if mpi_enabled:
        comm.Barrier()
    sys.exit(1)

# ============================================================
# Decide how many batches to run (supports extending -ntraj)
# ============================================================
nbatch_target = ntraj_requested // max(1, npar)
ntraj_target_effective = nbatch_target * npar

if rank == 0 and ntraj_target_effective != ntraj_requested:
    print(f"Warning: ntraj ({ntraj_requested}) not divisible by npar ({npar}).")
    print(f"         Target will be {ntraj_target_effective} trajectories (drop remainder).")

nbatch_done_offset = attempted_offset // max(1, npar)
nbatch_remaining = max(0, nbatch_target - nbatch_done_offset)

if nbatch_remaining == 0:
    if rank == 0:
        if nbatch_target < nbatch_done_offset:
            print(f"[restart] Note: checkpoint already has {attempted_offset} attempted trajectories, which exceeds requested target {ntraj_target_effective}.")
        Bt_avg = Bt_offset / max(1, success_offset)
        model.savedata(Bt_avg, t, args, obstyp)
        if getattr(args, "ead", False):
            Ead_avg = Ead_offset / max(1, success_offset)
            model.savedata(Ead_avg, t, args, "Ead")
        print("[restart] nothing remaining to run; wrote averages from checkpoint and exiting.")
    if mpi_enabled:
        comm.Barrier()
    sys.exit(0)

nbatch_total = nbatch_remaining
ntraj_effective_run = nbatch_total * npar

if nbatch_total == 0:
    if rank == 0:
        print("Error: remaining work is zero batches (ntraj too small vs npar).")
    if mpi_enabled:
        comm.Barrier()
    sys.exit(1)

if rank == 0 and do_restart:
    print(f"[restart] remaining batches = {nbatch_total} (remaining traj = {ntraj_effective_run}) "
          f"to reach target {ntraj_target_effective}.")

# ============================================================
# Split remaining batches across ranks (synchronized step loop)
# ============================================================
if mpi_enabled:
    nb_base = nbatch_total // size
    nb_rem = nbatch_total % size
    nbatch_local = nb_base + (1 if rank < nb_rem else 0)

    batch_start = rank * nb_base + min(rank, nb_rem)
    batch_end = batch_start + nbatch_local
    local_batches = np.arange(batch_start, batch_end, dtype=int)

    max_steps = comm.allreduce(nbatch_local, op=MPI.MAX)
else:
    nbatch_local = nbatch_total
    local_batches = np.arange(0, nbatch_total, dtype=int)
    max_steps = nbatch_total

# ============================================================
# Checkpoint targets: based on FINAL target, skip already-done
# ============================================================
do_checkpoints = ckpt_enabled and (ckptfrac > 0.0) and (nbatch_target > 0)

if do_checkpoints:
    fracs = np.arange(ckptfrac, 1.0, ckptfrac)  # e.g. 0.1..0.9
    checkpoint_targets = [int(np.ceil(f * nbatch_target)) for f in fracs]
    # ensure strictly increasing and within [1, nbatch_target-1]
    checkpoint_targets = sorted(set([x for x in checkpoint_targets if 1 <= x <= max(0, nbatch_target - 1)]))
else:
    checkpoint_targets = []

next_ckpt_idx = 0
while next_ckpt_idx < len(checkpoint_targets) and checkpoint_targets[next_ckpt_idx] <= nbatch_done_offset:
    next_ckpt_idx += 1

# ============================================================
# Main loop
# ============================================================
attempted_local = 0
success_local = 0

t_loop0 = time.perf_counter()

reps = {'site': 'd', 'exc': 'e', 'adia': 'a', 'dia': 'd'}
rep = reps[args.basis]

for step in range(max_steps):
    if step < nbatch_local:
        # Sample a full batch of npar trajectories
        q0, p0, qe0, pe0 = model.sample(args, mass, omega, nf, ns)
        q = np.array(q0.copy(), order='F')
        p = np.array(p0.copy(), order='F')
        qe = np.array(qe0.copy(), order='F')
        pe = np.array(pe0.copy(), order='F')

        t_call0 = time.perf_counter()
        if getattr(args, "ead", False):
            bt, Et, ead_batch, v0_batch, ierr = mashf90.runpar_ead(q, p, qe, pe, rep, dt, nt, nf, ns, npar)
        else:
            bt, Et, ierr = mashf90.runpar(q, p, qe, pe, rep, dt, nt, nf, ns, npar)
        t_fortran += (time.perf_counter() - t_call0)

        ###
        # # optional: save the final q, p, qe, pe
        # ok = (ierr == 0)  # shape (npar,)

        # # q has shape (nf, npar); transpose to (n_ok, nf) for easy later use
        # q_ok  = q[:, ok].T
        # p_ok  = p[:, ok].T
        # qe_ok = qe[:, ok].T
        # pe_ok = pe[:, ok].T
        # global_batch = int(local_batches[step])
        # np.savez(
        #     f"final_bach{global_batch}.npz",
        #     q=q_ok, p=p_ok, qe=qe_ok, pe=pe_ok,
        #     ierr=ierr)
        ###

        # Counters
        attempted_local += npar
        nbad = int(np.sum(ierr > 0))
        success_batch = npar - nbad
        success_local += success_batch

        # Accumulate sums (Fortran returns sum over trajectories; failed trajs are zeros)
        Bt_local += bt
        if getattr(args, "ead", False):
            Ead_local[:, 0] += ead_batch
            Ead_local[:, 1] += v0_batch

    # ---------------- checkpoint check ----------------
    if do_checkpoints and (next_ckpt_idx < len(checkpoint_targets)):
        done_batches_local = min(step + 1, nbatch_local)

        if mpi_enabled:
            done_batches_global = comm.allreduce(done_batches_local, op=MPI.SUM)
        else:
            done_batches_global = done_batches_local

        done_batches_total = nbatch_done_offset + done_batches_global  # absolute toward nbatch_target

        if done_batches_total >= checkpoint_targets[next_ckpt_idx]:
            # Reduce "new-run" sums across ranks ONLY at checkpoint time
            if mpi_enabled:
                Bt_sum_new = np.zeros_like(Bt_local)
                comm.Allreduce(Bt_local, Bt_sum_new, op=MPI.SUM)
                success_new = comm.allreduce(success_local, op=MPI.SUM)

                if getattr(args, "ead", False):
                    Ead_sum_new = np.zeros_like(Ead_local)
                    comm.Allreduce(Ead_local, Ead_sum_new, op=MPI.SUM)
            else:
                Bt_sum_new = Bt_local
                success_new = success_local
                if getattr(args, "ead", False):
                    Ead_sum_new = Ead_local

            # Add offsets (IMPORTANT)
            Bt_total = Bt_offset + Bt_sum_new
            success_total_ckpt = success_offset + success_new

            attempted_total = attempted_offset + done_batches_global * npar
            discarded_total = attempted_total - success_total_ckpt

            if getattr(args, "ead", False):
                Ead_total = Ead_offset + Ead_sum_new
            else:
                Ead_total = None

            if rank == 0:
                save_checkpoint(
                    ckptfile, meta_now,
                    Bt_total, success_total_ckpt,
                    attempted_total, discarded_total,
                    Ead_sum=Ead_total
                )

                r0_print("")
                r0_print("============= Checkpoint saved ==========")
                r0_print(f"File      : {ckptfile}")
                r0_print(f"Progress  : batches {done_batches_total}/{nbatch_target}")
                r0_print(f"Attempted : {attempted_total}/{ntraj_target_effective}")
                r0_print(f"Success   : {success_total_ckpt}")
                r0_print(f"Discarded : {discarded_total}")
                r0_print("=========================================")
                r0_print("")

                # optional: overwrite current averages for monitoring
                Bt_avg_now = Bt_total / max(1, success_total_ckpt)
                model.savedata(Bt_avg_now, t, args, obstyp)
                if getattr(args, "ead", False):
                    Ead_avg_now = Ead_total / max(1, success_total_ckpt)
                    model.savedata(Ead_avg_now, t, args, "Ead")

                np.savetxt("log.out",
                           np.array([success_total_ckpt, attempted_total, discarded_total], dtype=np.int64),
                           fmt="%i")

                print(f"[checkpoint] batches {done_batches_total}/{nbatch_target} "
                      f"attempted={attempted_total}/{ntraj_target_effective} "
                      f"success={success_total_ckpt} discarded={discarded_total}")

            next_ckpt_idx += 1

t_loop1 = time.perf_counter()

# ============================================================
# Final reduction and output
# ============================================================
if mpi_enabled:
    Bt_sum_new = np.zeros_like(Bt_local)
    comm.Allreduce(Bt_local, Bt_sum_new, op=MPI.SUM)
    success_new = comm.allreduce(success_local, op=MPI.SUM)

    if getattr(args, "ead", False):
        Ead_sum_new = np.zeros_like(Ead_local)
        comm.Allreduce(Ead_local, Ead_sum_new, op=MPI.SUM)
else:
    Bt_sum_new = Bt_local
    success_new = success_local
    if getattr(args, "ead", False):
        Ead_sum_new = Ead_local

if rank == 0:
    Bt_total = Bt_offset + Bt_sum_new
    success_total = success_offset + success_new

    attempted_total = attempted_offset + nbatch_total * npar
    # sanity: this should match target effective trajectories
    # (unless user requested smaller target than checkpoint already had, which we exit earlier)
    discarded_total = attempted_total - success_total

    Bt_avg = Bt_total / max(1, success_total)
    model.savedata(Bt_avg, t, args, obstyp)

    if getattr(args, "ead", False):
        Ead_total = Ead_offset + Ead_sum_new
        Ead_avg = Ead_total / max(1, success_total)
        model.savedata(Ead_avg, t, args, "Ead")
    else:
        Ead_total = None

    # Always write a final checkpoint too
    if ckpt_enabled:
        save_checkpoint(
            ckptfile, meta_now,
            Bt_total, success_total,
            attempted_total, discarded_total,
            Ead_sum=Ead_total
        )

    np.savetxt("log.out",
               np.array([success_total, attempted_total, discarded_total], dtype=np.int64),
               fmt="%i")

# ============================================================
# Timing summary (report MAX across ranks)
# ============================================================
t_total1 = time.perf_counter()
total_wall_local = t_total1 - t_total0
loop_wall_local  = t_loop1  - t_loop0

if mpi_enabled:
    total_wall = comm.allreduce(total_wall_local, op=MPI.MAX)
    loop_wall  = comm.allreduce(loop_wall_local,  op=MPI.MAX)
    t_fortran_report = comm.allreduce(t_fortran, op=MPI.MAX)
else:
    total_wall = total_wall_local
    loop_wall = loop_wall_local
    t_fortran_report = t_fortran

if rank == 0:
    print("\n================ Timing ================")
    print(f"MPI ranks             : {size}")
    print(f"Total wall time (max) : {fmt_seconds(total_wall)}")
    print(f"Loop wall time  (max) : {fmt_seconds(loop_wall)}")
    print(f"Fortran time   (max)  : {fmt_seconds(t_fortran_report)}  ({t_fortran_report/max(loop_wall,1e-12):.1%} of loop)")
    print(f"Requested trajectories: {ntraj_requested}")
    print(f"Target effective traj : {ntraj_target_effective}  (npar={npar})")
    print("========================================\n")

if mpi_enabled:
    comm.Barrier()
