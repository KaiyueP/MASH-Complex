import numpy as np

# ---------------------------
# Define the original quantities
# ---------------------------
nstate = 9

# The system Hamiltonian in Ha
ExcEn = np.array([line.strip().split()[1] for line in open("exciton.dat", 'r')]).astype(float) # in Ha
ExcEn = ExcEn[0:0+nstate]
ExcEn -= ExcEn[0]
ham_sys = np.diag(ExcEn)

# Load vibrational frequencies from w.dat:
# The file is assumed to have (at least) two columns;
# we take the second column and convert THz -> angular frequency (rad/s):
w_SI = np.loadtxt("w.dat")[:, 1][6:] * 2 * np.pi * 1e12  # in s^-1

# Load Vklq from Vklq-diabatic.dat:
# The file is assumed to have the desired data in the last column.
# We then reshape it into (n_modes, nstate, nstate) and convert units.
Vklq_SI = np.loadtxt("Vklq-diabatic.dat")[:, -1].reshape(-1, 40, 40)  # in sqrt(J)/s
Vklq_SI = Vklq_SI[6:,:nstate,:nstate]

# save the reorganization energies
JtoeV = 6.241509e18  # J/eV
lmd = np.einsum("kij->ij", Vklq_SI**2 / w_SI[:, None, None]**2 / 2) * JtoeV  # in eV
np.savetxt("lmd_nm.dat", lmd, fmt="%.16f")


# ---------------------------
# Define conversion factors
# ---------------------------

# (1) Convert energies from eV to Hartree:
# 1 eV = 1/27.211386245988 Hartree.
eV_to_Hartree = 1 / 27.211386245988

# (2) Convert angular frequencies from s^-1 to atomic units:
# Atomic unit of time: t0 = 2.4188843265857e-17 s.
t0 = 2.4188843265857e-17  # s
w_AU = w_SI * t0

# (3) Convert Vklq from sqrt(J)/s to atomic units:
# The atomic unit of energy is 1 Hartree = 4.3597447222071e-18 J.
Eh = 4.3597447222071e-18  # J
# Thus, the conversion factor is:
conv_factor_V = t0 / np.sqrt(Eh)
Vklq_AU = Vklq_SI * conv_factor_V

# ---------------------------
# Save the results
# ---------------------------
np.savez("lvc_params_AU.npz", ham_sys_AU=ham_sys, w_AU=w_AU, Vklq_AU=Vklq_AU)
