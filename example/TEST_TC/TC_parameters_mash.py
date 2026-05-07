import numpy as np

# ============================================================
# User settings for TC model
# ============================================================
# system setup
nstate_per_qd = 4
nqd_per_layer = 5
nlayers = 3 #need to be fixed here
nmode_per_qd = 30
N_cavity = 20
L=40.0*1e-6 #um to m cavity length
QD_1="3.9nm_4ML" #low energy
QD_2="3.9nm_3ML" #medium energy	
QD_3="3.0nm_4ML" #high energy
QD_spacing = 20 *1e-9 #nm to m
N_QD = nlayers * nqd_per_layer

# Fundamental constants and unit conversions
kb=1.380649e-23
c=3e8 #m/s
nmtoau=18.8972598858 #1nm in Bohr
autoev= 27.2114079527 # 1 hartree in ev
evtoau=1/autoev
autoj = 4.3597447222071e-18 # 1 Hartree in J
jtoev=6.2415091e18 #1j in eV
jtoau = jtoev * evtoau # 1 J in au
evtoj=1.602176565e-19 #1eV in J
hbar= 1.054571817e-34 #J⋅s
hbar_eV=6.582119569e-16 #eV⋅s
C_light = 3e8  # m/s
C_au = C_light/2.18769126364e6  # AU
autos = 2.4188843265857e-17
conv_factor_V = autos / np.sqrt(autoj)


def load_sorted_mode_block(qd_name):
	"""Load and rank vibrational modes by coupling strength.

	The ranking score is the total mode coupling weight,
	sum(V_nm^2 / w^2), computed after skipping the first `skip` rows
	from the raw files.
	sort using first 10 states
	"""
 	# in SI
	w_SI = np.loadtxt("/pscratch/sd/k/kaiyuep/QD_data/%s/w.dat" % qd_name)[:, 1][6:] * 2 * np.pi * 1e12
	Vklq_SI = np.load("/pscratch/sd/k/kaiyuep/QD_data/%s/Vklq.npy" % qd_name)[6:, :10, :10] 

	mode_strength = np.einsum("kij->k", Vklq_SI**2 / w_SI[:, None, None]**2)
	sort_idx = np.argsort(mode_strength)[::-1]
	select_idx = sort_idx

	w_SI_block = w_SI[select_idx]
	Vklq_SI_block = Vklq_SI[select_idx]
	return w_SI_block, Vklq_SI_block, mode_strength[select_idx]


# read date from files
#all in AU
ExcEn1 = np.array([line.strip().split()[1] for line in open("/pscratch/sd/k/kaiyuep/QD_data/%s/exciton.dat" % QD_1, 'r')]).astype(float) # in Ha
ExcEn1 = ExcEn1[:nstate_per_qd]
filename='/pscratch/sd/k/kaiyuep/QD_data/%s/intElecDipoleMoments.dat'%QD_1
mu1 = np.sqrt(np.loadtxt(filename,dtype='float',comments='#')[:nstate_per_qd,2])
w_SI_block1, Vklq_SI_block1, strength1 = load_sorted_mode_block(QD_1)
w_SI_block1=w_SI_block1[:nmode_per_qd]
Vklq_SI_block1 = Vklq_SI_block1[:nmode_per_qd, :nstate_per_qd, :nstate_per_qd]
w_AU_block1 = w_SI_block1 * autos #convert to au
Vklq_AU_block1 = Vklq_SI_block1 * conv_factor_V

#all in AU
ExcEn2 = np.array([line.strip().split()[1] for line in open("/pscratch/sd/k/kaiyuep/QD_data/%s/exciton.dat" % QD_2, 'r')]).astype(float) # in Ha
ExcEn2 = ExcEn2[:nstate_per_qd]
filename='/pscratch/sd/k/kaiyuep/QD_data/%s/intElecDipoleMoments.dat'%QD_2
mu2 = np.sqrt(np.loadtxt(filename,dtype='float',comments='#')[:nstate_per_qd,2])
w_SI_block2, Vklq_SI_block2, strength2 = load_sorted_mode_block(QD_2)
w_SI_block2=w_SI_block2[:nmode_per_qd]
Vklq_SI_block2 = Vklq_SI_block2[:nmode_per_qd, :nstate_per_qd, :nstate_per_qd]
w_AU_block2 = w_SI_block2 * autos #convert to au
Vklq_AU_block2 = Vklq_SI_block2 * conv_factor_V

#all in AU
ExcEn3 = np.array([line.strip().split()[1] for line in open("/pscratch/sd/k/kaiyuep/QD_data/%s/exciton.dat" % QD_3, 'r')]).astype(float) # in Ha
ExcEn3 = ExcEn3[:nstate_per_qd]
filename='/pscratch/sd/k/kaiyuep/QD_data/%s/intElecDipoleMoments.dat'%QD_3
mu3 = np.sqrt(np.loadtxt(filename,dtype='float',comments='#')[:nstate_per_qd,2])
w_SI_block3, Vklq_SI_block3, strength3 = load_sorted_mode_block(QD_3)
w_SI_block3=w_SI_block3[:nmode_per_qd]
Vklq_SI_block3 = Vklq_SI_block3[:nmode_per_qd, :nstate_per_qd, :nstate_per_qd]
w_AU_block3 = w_SI_block3 * autos #convert to au
Vklq_AU_block3 = Vklq_SI_block3 * conv_factor_V

mu_max=np.max([mu1, mu2, mu3])
E_c=ExcEn3[0]*autoj #in Ha to j
lambda_c = hbar*2*np.pi*c/E_c #m
L_cavity = lambda_c/2 #m
k_x=2*np.pi*np.arange(0, N_cavity)/L
k_z=E_c*evtoj/(hbar*c)
E_cavity = hbar*c*np.sqrt(k_x**2 + k_z**2)*jtoau #j to au 

x_QD=np.arange(nqd_per_layer)*QD_spacing
d_v = 10e-9 #nm to m, spacing between layers
z_QD = np.array([L_cavity/2 - d_v, L_cavity/2, L_cavity/2 + d_v ]) #m
g_s_ref=1e-3*evtoau #eV to au #coupling strength


#reorg in eV
lmd1 = np.einsum("kij->ij", Vklq_SI_block1**2 / w_SI_block1[:, None, None]**2 / 2) *jtoev  # in eV
np.savetxt("lmd1_nm.dat", lmd1, fmt="%.16f")
lmd2 = np.einsum("kij->ij", Vklq_SI_block2**2 / w_SI_block2[:, None, None]**2 / 2) *jtoev  # in eV
np.savetxt("lmd2_nm.dat", lmd2, fmt="%.16f")
lmd3 = np.einsum("kij->ij", Vklq_SI_block3**2 / w_SI_block3[:, None, None]**2 / 2) *jtoev  # in eV
np.savetxt("lmd3_nm.dat", lmd3, fmt="%.16f")

# ============================================================
# Build TC single-excitation Hamiltonian in atomic units
# State ordering: [QD0 states, QD1 states, ..., QD(N_QD-1) states, CAV states]
# ============================================================
ns = N_QD * nstate_per_qd + N_cavity
ham_sys_AU = np.zeros((ns, ns), dtype=float)

# QD diagonal energies: build sequence of QD blocks per layer
# QD types for each layer (assume layers correspond to QD_1, QD_2, QD_3)
ExcEn1_layer = np.tile(ExcEn1, nqd_per_layer)
ExcEn2_layer = np.tile(ExcEn2, nqd_per_layer)
ExcEn3_layer = np.tile(ExcEn3, nqd_per_layer)
ExcEn = np.concatenate([ExcEn1_layer, ExcEn2_layer, ExcEn3_layer]) 
ham_sys_AU[:N_QD * nstate_per_qd, :N_QD * nstate_per_qd] = np.diag(ExcEn)
if N_QD*nstate_per_qd != ns - N_cavity:
    	raise ValueError("Mismatch in state counts: QD states = %d, expected %d" % (N_QD*nstate_per_qd, ns - N_cavity))	
ham_sys_AU[N_QD * nstate_per_qd:, N_QD * nstate_per_qd:] = np.diag(E_cavity)

# off diagonal cavity-QD coupling

qd_states_per_layer = nqd_per_layer * nstate_per_qd
for layer_idx, (mu_block, z_qd) in enumerate([(mu1, z_QD[0]), (mu2, z_QD[1]), (mu3, z_QD[2])]):
	layer_state_offset = layer_idx * qd_states_per_layer
	for i in range(nqd_per_layer):
		x_qd = x_QD[i]
		for j in range(nstate_per_qd):
			qd_state = layer_state_offset + i * nstate_per_qd + j
			for l in range(N_cavity):
				g_s = g_s_ref / mu_max * mu_block[j] / np.sqrt(E_cavity[0]) * np.sqrt(E_cavity[l]) * \
					np.sin(k_z * z_qd) * \
					np.exp(-1j * k_x[l] * x_qd)

				cav_state = N_QD * nstate_per_qd + l
				ham_sys_AU[cav_state, qd_state] = g_s
				ham_sys_AU[qd_state, cav_state] = g_s.conjugate()
    
# ============================================================
# Build QD-local baths: each QD gets an identical bath block.
# Cavity states remain phonon-free.
# ============================================================
ns_qd = N_QD * nstate_per_qd #total QD state
nw = N_QD * nmode_per_qd # total QD phonon mode

# Build per-QD lists for vibrational modes and coupling blocks by repeating
# Tile the sorted blocks across QDs in each layer
w_AU = np.concatenate([
    np.tile(w_AU_block1, nqd_per_layer),
    np.tile(w_AU_block2, nqd_per_layer),
    np.tile(w_AU_block3, nqd_per_layer)
])

Vklq_blocks = [Vklq_AU_block1, Vklq_AU_block2, Vklq_AU_block3]

Vklq_AU = np.zeros((nw, ns, ns), dtype=float)
mode_owner = np.zeros(nw, dtype=np.int32)

# Assign modes and couplings for each QD
for i_qd in range(N_QD):
	# Determine which block (QD type) this QD belongs to by layer
	layer_idx = i_qd // nqd_per_layer
	V_block = Vklq_blocks[layer_idx]

	# Mode and state ranges for this QD
	m0 = i_qd * nmode_per_qd
	m1 = m0 + nmode_per_qd
	s0 = i_qd * nstate_per_qd
	s1 = s0 + nstate_per_qd

	Vklq_AU[m0:m1, s0:s1, s0:s1] = V_block
	# 1-based QD id for local modes owned by this QD block.
	mode_owner[m0:m1] = i_qd + 1



np.savez(
	"tc_params_AU.npz",
	ham_sys_AU=ham_sys_AU,
	w_AU=w_AU,
	Vklq_AU=Vklq_AU,
	mode_owner=mode_owner,
	N_QD=np.int64(N_QD),
	N_cavity=np.int64(N_cavity),
	nmode_per_qd=np.int64(nmode_per_qd),
	nstate_per_qd=np.int64(nstate_per_qd),
	ns=np.int64(ns),
	nw=np.int64(nw)
)

print("Saved tc_params_AU.npz")
print(f"states = {ns} (QD states = {ns_qd}, N_QD={N_QD}, nstate_per_qd={nstate_per_qd}, N_cavity={N_cavity})")
