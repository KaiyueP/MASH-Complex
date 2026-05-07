import numpy as np
import argparse
import sys
from src import mashf90
import os 

script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = script_dir + '/input'

class CustomArgumentParser(argparse.ArgumentParser):
    """ Allow commented lines in input file """
    def convert_arg_line_to_args(self, line):
        if line.strip().startswith('#'):
            return []
        return line.split()

""" Unit conversions to atomic units """
cmm1 = 4.556335e-6
fs = 41.34136e0
kB = 3.166829e-6
eV = 1./27.2113961

""" Define global variables """
Vconst = 0.; nf_site = 0; ns = 0
w_low = 0.; c_low = 0.; w_intra = 0.; c_intra = 0.

def read_args():
    """ Run file with 'mash.py +input.in' """
    parser = CustomArgumentParser(fromfile_prefix_chars=["@","+"])
    # General
    parser.add_argument("-model", type=str, default="spinboson", help="Model system", 
                        choices=["spinboson","fmo3","fmo7","fmo8","tully1","tully2","lh2","lhc2","lvc","qvc","tc"])
    parser.add_argument("-basis", type=str, default="site", help="Diabatic basis for certain model systems", 
                        choices=["exc","site","adia","dia"])
    parser.add_argument("-units", type=str, default="au",help="""Choose unit system. 
                        cmm1: input in cmm1, output in fs. 
                        fs: input in fs, output in fs. """,
                        choices=["au","cmm1","fs"])
    parser.add_argument("-obstyp", type=str, default="pop", help="Observable types", 
                        choices=["pop","nuc"])
    parser.add_argument("-init", type=int,help="Initial state (in Python indexing)")
    parser.add_argument("-initbasis",type=str,default='dia',choices=["dia","adia","site","exc"],help='Basis for initial state')
    parser.add_argument("-nucsamp",type=str,default='cl', 
                        help="Nuclear sampling. classical/cl=classical, wigner/wig=thermal wigner, GS=ground state Wigner, clzero=classical T=0, WP=wavepacket",
                        choices=['classical','cl','wigner','wig','GS','clzero','WP'])
    parser.add_argument("-elsamp",type=str,default='focused',choices=["focused","theta"],help='Choice of initial distribution')
    parser.add_argument("-beta",type=float,default=1.,help="Reciprocal temperature [in a.u.]")
    parser.add_argument("-T",type=float,default=0,help="Temperature [in kelvin]")
    # Debugging
    parser.add_argument("-debug",action="store_true",help='Run one debug trajectory and plot energy conservation etc.')
    parser.add_argument("-seed",type=float,default=None,help="If set, initialize random seed.")
    parser.add_argument("-ead", action="store_true", help="Also compute and output <Ead>(t)=<sum_a Phi_a^ad(t) * Vad_a(Q(t))>.")
    # Convergence
    parser.add_argument("-dt",default=41,type=float,help="Time step")
    parser.add_argument("-nt","-TS",type=int,help="Number of time steps")
    parser.add_argument("-ntraj","-traj",type=int,help="Number of trajectories")
    parser.add_argument("-nf",type=int,default=1,help="Number of bath modes.")
    parser.add_argument("-npar","-n",default=1,type=int,help="Number of parallel processors")
    # Spin-boson model
    parser.add_argument("-Delta",type=float,default=1.,help="Diabatic coupling")
    parser.add_argument("-epsilon","-eps",type=float,default=0.,help="(Half) energy bias")
    parser.add_argument("-lamda",type=float,default=0,help="System-bath reorganization energy.")
    parser.add_argument("-omegac",type=float,default=0.,help="Cutoff frequency.")
    # Frenkel-exciton models
    parser.add_argument("-disorder",type=str,default='none',choices=["none","fmo","lhc2"],help='Choice of static disorder')
    parser.add_argument("-bath",type=str,default='debye',choices=["debye","coursegrain","B777"],help='Choice of low-frequency bath for fmo and lhc2')
    parser.add_argument("-wmax",type=float,default=None,help="Max omega in discreziation")
    parser.add_argument("-polaron",type=str,choices=["vpt","langfirsov"],help='Type of polaron transformation to remove high-frequency modes')
    # Tully model
    parser.add_argument("-WPenergy",type=float,default=0.,help="Wavepacket energy")
    parser.add_argument("-gamma",type=float,default=0.,help="Wavepacket width parameter")
    parser.add_argument("-pinit",type=float,default=0.,help="Initial momentum")
    parser.add_argument("-ckpt", action="store_true", help="Enable checkpointing (rank 0 writes a global .npz).")
    parser.add_argument("-restart", action="store_true", help="Restart from existing checkpoint file if present.")
    parser.add_argument("-ckptfile", type=str, default="mash_ckpt.npz", help="Checkpoint filename.")
    parser.add_argument("-ckptfrac", type=float, default=0.1, help="Checkpoint fraction (e.g. 0.1 -> every 10%).")
    parser.add_argument("--complex_version", action="store_true", help="Use complex Hermitian system Hamiltonian support when available.")
    args = parser.parse_args()

    # print input arguments:
    print('Input arguments:')
    print("Model: ",args.model)
    print("Units: ",args.units)
    print("Observable type: ",args.obstyp)
    print("Initial state: ",args.init)
    print("Initial basis: ",args.initbasis)
    print("Temperature: ",args.T)
    print("Reciprocal temperature: ",args.beta)
    print("Number of bath modes: ",args.nf)
    print("Number of time steps: ",args.nt)
    print("Time step: ",args.dt)
    print("Number of trajectories: ",args.ntraj)
    print("Number of parallel processors: ",args.npar)
    print("Diabatic coupling: ",args.Delta)
    print("Energy bias: ",args.epsilon)
    print("Reorganization energy: ",args.lamda)
    print("Cutoff frequency: ",args.omegac)
    print("Disorder: ",args.disorder)
    print("Bath: ",args.bath)
    print("Polaron transformation: ",args.polaron)
    print("Wavepacket energy: ",args.WPenergy)
    print("Wavepacket width: ",args.gamma)
    print("Initial momentum: ",args.pinit)
    print("Nuclear sampling: ",args.nucsamp)
    print("Electronic sampling: ",args.elsamp)
    print("Debug: ",args.debug)
    print("Ead computation: ",args.ead)
    print("Complex version: ",args.complex_version)
    print("Seed: ",args.seed)

    """ ======= Convert input arguments to atomic units ======="""
    # print input arguments:
    if args.units=="cmm1":
        args.epsilon = args.epsilon * cmm1
        args.Delta = args.Delta * cmm1
        args.lamda = args.lamda * cmm1
        args.omegac = args.omegac * cmm1
        args.dt = args.dt * fs
    if args.units=='fs':
        args.dt = args.dt * fs
    if args.T:
        args.beta = 1./(kB*args.T)

    if not args.seed is None:
        np.random.seed(args.seed)
    
    return args


"""======== Initialize system ========="""
def setup_model(args):
    global Vconst, nf_site, w_low, c_low, w_intra, c_intra, ns
    model = args.model
    epsilon = args.epsilon
    Delta = args.Delta
    nf = args.nf
    omega = np.zeros(nf,dtype=np.float64)
    if args.complex_version and model != 'tc':
        print("Warning: complex_version is currently implemented for model='tc' only. Using real backend for this model.")

    if model=='spinboson':
        # debye spectral density: J(w) = lmd/2 * wwc/(w^2 + wc^2)
        print("Using Ohmic Spectral Density...")
        ns = 2
        Vconst = np.array([[epsilon,Delta],[Delta,-epsilon]])
        Vlin = np.zeros((nf,ns,ns),dtype=np.float64)
        fac = np.sqrt(0.5*args.lamda/nf)
        c = np.zeros(nf)
        for i in range(nf):
            omega[i] = args.omegac*np.tan(0.5*np.pi*(i+0.5)/nf)
            c[i] = fac*omega[i]
            Vlin[i,0,0] = c[i]
            Vlin[i,1,1] = -c[i]
        
        # ohmic spectral density: J(w) = pi * lmd * (w/wc) * exp(-w/wc)
        # print("Using Ohmic Spectral Density...")
        # ns = 2
        # Vconst = np.array([[epsilon,Delta],[Delta,-epsilon]])
        # Vlin = np.zeros((nf,ns,ns),dtype=np.float64)
        # fac = np.sqrt(2*args.lamda/nf)
        # c = np.zeros(nf)

        # for i in range(nf):
        #     omega[i] = args.omegac*(-np.log((nf-i-0.5)/nf))
        #     c[i] = fac*omega[i]
        #     Vlin[i,0,0] = c[i]
        #     Vlin[i,1,1] = -c[i]

        print("Reorganization energy check: ", np.sum(c**2/omega**2/2))

    elif model == 'lvc':
        print("Using Linear Vibronic Model (LVC)")
        params = np.load("lvc_params_AU.npz")
        Vconst = np.array(params["ham_sys_AU"].real, dtype=np.float64)
        omega = params["w_AU"]
        Vlin = params["Vklq_AU"]
        ns = Vconst.shape[0]
        nf = omega.shape[0]
        print(f"Generating lvc systems of dimensions {ns}, with bath modes {nf}")
        lmd = np.einsum('aij->ij', Vlin**2/omega[:,None,None]**2/2)
        print("reorganization energy arr: ", lmd)

    elif model == 'qvc':
        print("Using Quadratic Vibronic Model (QVC)")
        params = np.load("qvc_params_AU.npz")
        Vconst = np.array(params["ham_sys_AU"].real, dtype=np.float64)
        omega = params["w_AU"]
        Vlin = params["Vklq_AU"]
        Wqud = params["Wklq_AU"]
        ns = Vconst.shape[0]
        nf = omega.shape[0]
        print(f"Generating qvc systems of dimensions {ns}, with bath modes {nf}")
    
    elif model == 'tc':
        Vconst, omega, Vlin, mode_owner, n_qd, nstate_per_qd, n_cavity, ns, nf = setup_tc_model(args.complex_version)

    elif model=='fmo3':
        ns = 3
        Vlin = np.zeros((nf,ns,ns),dtype=np.float64)
        omega = np.zeros(nf,dtype=np.float64)
        Vconst = np.diag([12410.,12530,12210])
        Vconst[0,1] = -87.7
        Vconst[0,2] =   5.5
        Vconst[1,2] =  30.8
        Vconst = Vconst + Vconst.T - np.diag(np.diag(Vconst))
        Vconst -= np.eye(ns)*np.trace(Vconst)/ns

        Vconst = Vconst*cmm1
        nf_site = nf//ns
        fac = np.sqrt(2*args.lamda/nf_site)
        c = np.zeros(nf_site)
        for i in range(nf_site):
            omega[i] = args.omegac*np.tan(0.5*np.pi*(i+0.5)/nf_site)
            c[i] = fac*omega[i]
        for i in range(ns):
            omega[i*nf_site:(i+1)*nf_site] = omega[:nf_site]
            Vlin[i*nf_site:(i+1)*nf_site,i,i] = c

    elif model=='fmo7':
        ns = 7
        omega = np.zeros(nf,dtype=np.float64)
        Vconst = np.diag([12410.,12530,12210,12320,12480,12630,12440])
        Vconst[0,1] = -87.7
        Vconst[0,2] =   5.5
        Vconst[0,3] = - 5.9
        Vconst[0,4] =   6.7
        Vconst[0,5] = -13.7
        Vconst[0,6] = - 9.9
        Vconst[1,2] =  30.8
        Vconst[1,3] =   8.2
        Vconst[1,4] =   0.7
        Vconst[1,5] =  11.8
        Vconst[1,6] =   4.3
        Vconst[2,3] = -53.5
        Vconst[2,4] = - 2.2
        Vconst[2,5] =  -9.6
        Vconst[2,6] =   6.0
        Vconst[3,4] = -70.7
        Vconst[3,5] = -17.0
        Vconst[3,6] = -63.3
        Vconst[4,5] =  81.1
        Vconst[4,6] = - 1.3
        Vconst[5,6] =  39.7
        Vconst = Vconst + Vconst.T - np.diag(np.diag(Vconst))
        Vconst -= np.eye(ns)*np.min(np.diag(Vconst))
        Vconst = Vconst*cmm1

    elif model=="fmo8":
        ns = 8
        """ Vconst from Schmidt am Busch, Müh, Madjet and Renger JPCL 2, 93 (2011) """
        Vconst = np.loadtxt(input_dir+'/FMO-HS.dat') * cmm1
        omega = np.zeros(nf,dtype=np.float64)

    elif model=='lh2':
        ns = 24
        """ Ordering as follows: 
        --------B850---------   ---B800----
        1A 1B 2A 2B ... 8A 8B   1 2 3 ... 8
        """
        """ Load system Hamiltonian: Tretiak, Middleton, Chernyak, Mukamel JPCB 104, 9540 (2000) """
        Vconst = np.loadtxt(input_dir+'/LH2-HS.dat') * cmm1

        """ Load discrete bath: Rätsep, Cai, Reimers, Freiberg JCP 134, 024506 (2011) """
        specden = np.loadtxt(input_dir+'/LH2-specden.dat')
        w_intra = specden[:,0] * cmm1
        S_HR = specden[:,1]/1000.
        c_intra = np.sqrt(2.*w_intra**3*S_HR)
        Jw = np.pi/2. * c_intra**2/w_intra
        lamda_intra = np.sum(Jw/w_intra)/np.pi
        print('lambda (intra): ', lamda_intra/cmm1)
        
        """ Ohmic part: J(w) = eta*w*exp(-w/wc)"""
        xi = 0.4
        eta = 2*np.pi*xi 
        omegac = 200.*cmm1
        nf_site = args.nf
        w_low = np.zeros(nf_site)
        c_low = np.zeros(nf_site)
        fac = np.sqrt(2*eta*omegac/(nf_site*np.pi))
        for i in range(nf_site):
            w_low[i] = -omegac*np.log((i+0.5)/nf_site)
            c_low[i] = fac*w_low[i]
        Jw_Ohm = np.pi/2.*c_low**2/w_low
        lamda_Ohm = np.sum(Jw_Ohm/w_low)/np.pi
        print('lambda (Ohmic): ', lamda_Ohm/cmm1)

    elif model=='lhc2':
        ns = 14
        """ Load system Hamiltonian: couplings from Müh, Madjet and Renger JPCB 114, 13517 (2010).
         Site energies from Müh and Renger Biochim. Biophys. Acta 1817, 1446 (2012). """
        Vconst = np.loadtxt(input_dir+'/LHCII-HS.dat') * cmm1
        E,U = np.linalg.eigh(Vconst)
        print(U[args.init,:]**2)

        """ Load discrete bath: Renger, Madjet, Knorr and Müh J. Plant Physiol. 168, 1497 (2011) """
        specden = np.loadtxt(input_dir+'/LHCII-specden.dat')
        w_intra = specden[:,0] * cmm1
        S_HR = specden[:,1]
        c_intra = np.sqrt(2.*w_intra**3*S_HR)

        """ Low-frequency bath: Renger et al. J. Plant Physiol. 168, 1497 (2011) """
        S0 = 0.5
        w1 = 0.56 * cmm1
        w2 = 1.94 * cmm1
        s1 = 0.8
        s2 = 0.5
        def Jw_term(w,wi,si):
            return si/(2*np.math.factorial(7)*wi**4)*w**3*np.exp(-np.sqrt(w/wi))
        def Jw_l(w):
            return S0/(s1+s2)*(Jw_term(w,w1,s1) + Jw_term(w,w2,s2))

        maxw = args.wmax * cmm1
        nf_site = nf
        w_low = maxw * np.arange(1,nf+1)/nf
        Jw_disc = Jw_l(w_low)
        dw = w_low[1]-w_low[0]
        g_Ohm = np.sqrt(Jw_disc*dw)
        c_low = g_Ohm * np.sqrt(2*w_low**3)

        lamda = 0.5*np.sum(c_low**2/w_low**2)
        print(lamda/cmm1)
        lamda_intra = 0.5*np.sum(c_intra**2/w_intra**2)
        print(lamda_intra/cmm1)

    if 'fmo' in model:
        if args.bath=='debye':
            nf_site = nf//ns
            fac = np.sqrt(2*args.lamda/nf_site)
            c = np.zeros(nf_site)
            for i in range(nf_site):
                omega[i] = args.omegac*np.tan(0.5*np.pi*(i+0.5)/nf_site)
                c[i] = fac*omega[i]
            Vlin = np.zeros((nf,ns,ns))
            for i in range(ns):
                omega[i*nf_site:(i+1)*nf_site] = omega[:nf_site]
                Vlin[i*nf_site:(i+1)*nf_site,i,i] = c
        elif args.bath=='coursegrain':
            """ Course-grained Huang-Rhys factors for low-frequency modes.
             Based on raw data from Klinger, Lindorfer, Müh and Renger JCP 153, 215103 (2020) """
            specden = np.loadtxt(input_dir+'/FMO-specden_coursegrain.dat')
            w_low = specden[:,0] * cmm1
            S_HR = specden[:,1]
            c_low = np.sqrt(2.*w_low**3*S_HR)
            nf_site = len(w_low)
        elif args.bath=='B777':
            """ Adolphs and Renger, Biophys J. 91, 2778 (2006) """
            S0 = 0.5
            w1 = 0.56 * cmm1
            w2 = 1.94 * cmm1
            s1 = 0.8
            s2 = 0.5
            def Jw_term(w,wi,si):
                return si/(2*np.math.factorial(7)*wi**4)*w**3*np.exp(-np.sqrt(w/wi))
            def Jw_l(w):
                return S0/(s1+s2)*(Jw_term(w,w1,s1) + Jw_term(w,w2,s2))
            maxw = args.wmax * cmm1
            nf_site = nf
            w_low = maxw * np.arange(1,nf+1)/nf
            Jw_disc = Jw_l(w_low)
            dw = w_low[1]-w_low[0]
            g_Ohm = np.sqrt(Jw_disc*dw)
            c_low = g_Ohm * np.sqrt(2*w_low**3)

    if model=='fmo8':
        """ Load experimental spectral density for high-frequency modes:
            Wendling et al. JPCB 2000, 104, 5825 for Prostechochloris aestuarii """
        specden = np.loadtxt(input_dir+'/FMO-specden.dat')
        w_intra = specden[:,0] * cmm1
        S_HR = specden[:,1]
        c_intra = np.sqrt(2.*w_intra**3*S_HR)

        lamda = 0.5*np.sum(c_low**2/w_low**2)
        print(lamda/cmm1)
        lamda_intra = 0.5*np.sum(c_intra**2/w_intra**2)
        print(lamda_intra/cmm1)


    if model in ['fmo8','lh2','lhc2']:
        w,c,Vconst_renorm = polaron_transform(w_low,w_intra,c_low,c_intra,Vconst,args)
            
        """ Total vibronic part of Hamiltonian """
        nf_site = len(w)
        nf = ns*nf_site
        omega = np.zeros(nf)
        kappa = np.zeros((nf_site,ns))
        for i in range(ns):
            omega[i*nf_site:(i+1)*nf_site] = w
            kappa[:,i] = c.copy()

    """ ======= Initialize mass ======= """
    mass = np.ones(nf)

    """ ====== Initialize fortran potential ===== """
    if 'tully' in model:
        nf = 1
        ns = 2
        mass = np.ones(1)*2000. 
        mashf90.init_tully(model[-1],mass)
    elif model in ['fmo3','fmo7','fmo8','lh2','lhc2']:
        if args.polaron is None:        
            mashf90.init_frexc(mass,omega,Vconst,kappa,nf,nf_site,ns)
        else:
            mashf90.init_frexc(mass,omega,Vconst_renorm,kappa,nf,nf_site,ns)
    elif model in ['spinboson','lvc']:
        mashf90.init_linvib(mass,omega,Vconst,Vlin,nf,ns)
    elif model == 'tc':
        # TC hybrid backend uses mode_owner to know whether each mode is
        # shared across many states (owner=0) or local to one QD block (owner=i).
        if args.complex_version:
            mashf90.init_tchybrid_complex(mass,omega,np.asarray(Vconst,dtype=np.complex128),Vlin,mode_owner,nf,ns,n_qd,nstate_per_qd,n_cavity)
            print(f"Complex TC mode enabled: complex Vconst only for TC with ns={ns}")
        else:
            mashf90.init_tchybrid(mass,omega,Vconst,Vlin,mode_owner,nf,ns,n_qd,nstate_per_qd,n_cavity)
    elif model in ['qvc']:
        mashf90.init_qudvib(mass,omega,Vconst,Vlin,Wqud,nf,ns)
    else:
        sys.exit('Model not recognized!')
    return mass,omega,nf,ns


def setup_tc_model(complex_version=False):
    """Load TC parameters with QD-local baths and cavity states without phonon coupling."""
    print("Using Tavis-Cummings-like Model (TC)")
    params = np.load("tc_params_AU.npz")
    if complex_version:
        Vconst = np.array(params["ham_sys_AU"], dtype=np.complex128)
        if np.any(np.abs(np.imag(np.diag(Vconst))) > 1e-12):
            raise ValueError("In complex_version, ham_sys_AU must have real diagonal.")
        if not np.allclose(Vconst, Vconst.conjugate().T, atol=1e-10, rtol=1e-10):
            raise ValueError("In complex_version, ham_sys_AU must be Hermitian.")
    else:
        Vconst = np.array(params["ham_sys_AU"].real, dtype=np.float64)
    omega = params["w_AU"]
    Vlin = params["Vklq_AU"]
    if "mode_owner" not in params.files:
        raise ValueError("tc_params_AU.npz is missing mode_owner. Regenerate with example/TEST_TC/TC_parameters_mash.py")
    mode_owner = params["mode_owner"].astype(np.int32)
    # Read optional metadata (N_QD, N_cavity, nstate_per_qd)
    n_qd = int(params["N_QD"]) if "N_QD" in params.files else 0
    n_cavity = int(params["N_cavity"]) if "N_cavity" in params.files else 0
    nstate_per_qd = int(params["nstate_per_qd"]) if "nstate_per_qd" in params.files else 0
    
    # Infer dimensions from arrays in the file
    ns = int(Vconst.shape[0])  # total electronic diabatic states (includes cavity states)
    nw = int(omega.shape[0])  # total phonon modes
    # Basic shape checks
    if Vlin.shape[0] != nw or Vlin.shape[1] != ns or Vlin.shape[2] != ns:
        raise ValueError(f"Vklq_AU has incompatible shape {Vlin.shape}, expected (nw,ns,ns)=({nw},{ns},{ns})")
    if mode_owner.shape[0] != nw:
        raise ValueError(f"mode_owner has length {mode_owner.shape[0]}, expected {nw}")

    ns_qd = n_qd * nstate_per_qd if (n_qd > 0 and nstate_per_qd > 0) else None
    if ns_qd is not None:
        expected_ns = ns_qd + n_cavity
        if expected_ns != ns:
            raise ValueError(f"TC parameter file is inconsistent: N_QD * nstate_per_qd + N_cavity = {expected_ns}, but ham_sys_AU implies ns = {ns}")

    # If cavities are present, ensure their indices are after QD blocks
    if ns_qd is not None and n_cavity > 0:
        cavity_start = ns_qd
        if cavity_start < 0 or cavity_start > ns:
            raise ValueError("Inconsistent TC metadata for cavity indexing")
        cavity_slice = slice(cavity_start, ns)
        if np.any(Vlin[:, cavity_slice, :]) or np.any(Vlin[:, :, cavity_slice]):
            raise ValueError("TC cavity states must have zero phonon coupling in Vklq_AU")

    print(f"Generating tc systems of dimensions {ns}, with bath modes {nw}")
    if n_qd > 0:
        print(f"TC layout: N_QD={n_qd}, nstate_per_qd={nstate_per_qd}, N_cavity={n_cavity} (cavity electronic states; cavity phonon coupling must be zero)")

    # mode_owner is provided by tc_params_AU.npz (generated in TC_parameters_mash.py):
    #   mode_owner[a] = 0  : shared/mixed mode
    #   mode_owner[a] = i>0: mode local to QD i (1-based index)

    lmd = np.einsum('aij->ij', Vlin**2/omega[:,None,None]**2/2)
    print("reorganization energy arr: ", lmd)
    n_shared = int(np.sum(mode_owner == 0))
    print(f"TC mode partition: shared/mixed={n_shared}, qd-local={nw - n_shared}")
    return Vconst, omega, Vlin, mode_owner, n_qd, nstate_per_qd, n_cavity, ns, nw


def polaron_transform(w_Ohm,w_intra,c_Ohm,c_intra,Vconst,args):
    """ Calculate renormalized Vconst from selected set of modes through the polaron transformation """
    w = np.concatenate([w_Ohm,w_intra])
    c = np.concatenate([c_Ohm,c_intra])

    if args.polaron is None:
        return w,c,Vconst

    """ Select intra part for PT """
    toPT = np.ones(len(w),dtype=bool)
    toPT[:len(w_Ohm)] = False

    """ Only transform modes with w > kT """
    toPT[w < 1./args.beta] = False

    w_high = w[toPT]
    c_high = c[toPT]
    w = w[~toPT]
    c = c[~toPT]

    nf_high = len(w_high)

    g = c_high / np.sqrt(2*w_high**3)
    nBE = 1./(np.exp(args.beta*w_high)-1)
    coth = 2*nBE + 1
     
    if args.polaron=='langfirsov':
        # Scale all off-diagonal potential by the same band narrowing factor
        # based on a full Lang-Firsov polaron transformation of the high-
        # frequency intramolecular modes
        S_HR = g**2
        f = np.exp(-np.sum(S_HR*coth))
        Hel = Vconst.copy()
        Hel[~np.eye(ns,dtype=bool)] *= f
        print('f',f)
    elif args.polaron=='vpt':
        """ Variational polaron transformation (site-dependent) """
        f = np.zeros((ns,nf_high))
        for n in range(ns):
            f[n,:] = g.copy() 
        niter = 20
        bandnarrow = np.zeros((ns,ns))
        for iter in range(niter):
            oldf = f.copy()
            Hel = np.zeros((ns,ns))
            for n in range(ns):
                Hel[n,n] = Vconst[n,n] - np.sum(w_high * (2*g*f[n]-f[n]**2))
                for m in range(n+1,ns):
                    bandnarrow[n,m] = np.exp(-0.5*np.sum((f[n,:]**2+f[m,:]**2)*coth))
                    Hel[n,m] = Vconst[n,m] * bandnarrow[n,m]
                    Hel[m,n] = Vconst[m,n] * bandnarrow[n,m]
            eps,U = np.linalg.eigh(Hel)
            Pk = np.exp(-args.beta*eps)
            Pk /= np.sum(Pk)
            for n in range(ns):
                tmp = 0
                for m in range(ns):
                    if m!=n: 
                        um = np.sum(U[n,:]*U[m,:]*Pk)/np.sum(U[n,:]**2*Pk)
                        tmp += um*Hel[n,m]
                f[n,:] = w_high * g / (w_high - coth*tmp)
            change = np.linalg.norm(f-oldf)/np.linalg.norm(oldf)
            print(iter,change)
            if change < 1e-6:
                break
            if iter==niter-1:
                print('Convergence not reached')
                sys.exit()
        print('Resulting band narrow factors:')
        avg = 0
        min = 1
        for n in range(ns):
            for m in range(n+1,ns):
                avg += bandnarrow[n,m]
                min = np.min([min,bandnarrow[n,m]])
        avg /= ns*(ns-1)/2
        print('bandnarrow avg',avg)
        print('bandnarrow min',min)
    else:
        sys.exit('args.polaron='+str(args.polaron)+' is not a valid option')
    return w,c,Hel

""" Debugging section to check energy conservation, plot adiabatic populations etc. """
def debug(args,mass,omega,nf,ns):
    np.random.seed(0)
    beta = args.beta
    dt = args.dt
    nt = args.nt
    """Initialize phase-space variables"""
    q0,p0,qe0,pe0 = sample(args,mass,omega,nf,ns)

    import matplotlib.pyplot as plt
    """ Energy conservation """
    q,p,qe,pe,at,Et,ierr=mashf90.runtrj(q0[:,0], p0[:,0], qe0[:,0], pe0[:,0], dt, nt, nf, ns)
    t = dt*np.arange(nt+1)
    Eref = nf/beta
    plt.plot(t,Et/Eref,'o',alpha=0.5)
    plt.savefig("debug_energy.pdf")
    plt.close()
    np.savetxt("debug_Et.dat", Et)
    np.savetxt("debug_q.dat", q)
    np.savetxt("debug_p.dat", p)

    """ Adiabatic populations """
    plt.figure()
    qa = np.zeros_like(qe)
    pa = np.zeros_like(pe)
    popa = np.zeros_like(qe)
    Va = np.zeros((nt+1,ns))
    for it in range(nt+1):
        qa[it],pa[it] = mashf90.dia2ad(q[it],qe[it],pe[it])
        popa[it] = qa[it]**2 + pa[it]**2
        Va[it],U = mashf90.get_vad(q[it],ns)
    for n in range(ns):
        label=str(n) 
        plt.plot(t*0.0241888,popa[:,n],'-',color='C%i'%n,alpha=0.5,label=label)
    popat = [popa[it,at[it]-1] for it in range(nt+1)]
    plt.plot(t*0.0241888,popat,'k:',alpha=0.5,label='Active')
    plt.legend()
    plt.savefig("debug_pop_ad.png")
    plt.close()

    plt.figure()
    V0 = np.sum(mass*omega**2*q**2/2.,1)
    for n in range(ns):
        plt.plot(t*0.0241888,(Va[:,n]-V0)*27.2114079527,label='V'+str(n))
    plt.xlabel('Time (fs)')
    plt.ylabel('Adiabatic energies Va-V0 (eV)')
    plt.legend()
    plt.savefig("debug_V_ad.png")

    plt.show()

def sample(args,mass,omega,nf,ns):
    """ Sample npar sets of initial phase-space variables.
        qe and pe are real and imaginary parts of c in the diabatic representation """
    npar = args.npar
    beta = args.beta
    q = np.empty((nf,npar),order="F")
    p = np.empty((nf,npar),order="F")
    qe = np.empty((ns,npar),order="F")
    pe = np.empty((ns,npar),order="F")

    if args.disorder!='none':
        Vconst_dis = np.copy(Vconst)
        if args.disorder=='fmo':
            E_sig = np.ones(ns) * 100 * cmm1
        elif args.disorder=='lhc2':
            E_sig = np.ones(ns) * 120 * cmm1
        newE = np.random.normal(np.diag(Vconst), E_sig)
        np.fill_diagonal(Vconst_dis, newE)
        if args.model in ['fmo8','lh2','lhc2']:
            w,c,Vconst_dis = polaron_transform(w_low,w_intra,c_low,c_intra,Vconst_dis,args)
        kappa = np.zeros((nf_site,ns))
        for n in range(ns):
            omega[n*nf_site:(n+1)*nf_site] = w
            kappa[:,n] = c.copy()
        mashf90.init_frexc(mass,omega,Vconst_dis,kappa,nf,nf_site,ns)    

    for j in range(npar):
        """ Nuclear sampling """
        qmin = 0
        if args.nucsamp in ['classical','cl']:
            """ Classical """
            q[:,j] = np.random.normal(qmin,1./np.sqrt(beta*mass*omega**2),nf)
            p[:,j] = np.random.normal(0,np.sqrt(mass/beta),nf)
        elif args.nucsamp in ['wigner','wig']:
            """ Wigner """
            qsig = np.sqrt(1./(2*mass*omega*np.tanh(beta*omega/2)))
            psig = np.sqrt(mass*omega/(2*np.tanh(beta*omega/2)))
            q[:,j] = np.random.normal(qmin,qsig,nf)
            p[:,j] = np.random.normal(0,psig,nf)
        elif args.nucsamp=='GS':
            """ Ground state Wigner """
            qsig = np.sqrt(1./(2*mass*omega))
            psig = np.sqrt(mass*omega/2.)
            q[:,j] = np.random.normal(qmin,qsig,nf)
            p[:,j] = np.random.normal(0,psig,nf)
        elif args.nucsamp=='clzero':
            """ Ground state classical """
            qsig = 0.
            psig = 0.
            q[:,j] = np.random.normal(qmin,qsig,nf)
            p[:,j] = np.random.normal(0,psig,nf)
        elif args.nucsamp=='WP':
            """ Wave packet (for Tully models) """
            if args.gamma>0:
                qsig = 1./np.sqrt(2.*args.gamma)
                psig = np.sqrt(args.gamma/2.)
            else:
                qsig = psig = 0
            qmean = -15.
            if args.WPenergy:
                pmean = np.sqrt(2.*mass*args.WPenergy)
            else:
                pmean = args.pinit
            q[:,j] = np.random.normal(qmean,qsig,nf)
            p[:,j] = np.random.normal(pmean,psig,nf)

        """ Electronic sampling """
        if args.elsamp in ['theta']:
            """ Theta sampling: sample from region where Theta_init(c) = 1 """
            while 1:
                normals = np.random.randn(2*ns)
                normals = normals/np.linalg.norm(normals)
                qej=normals[:ns]
                pej=normals[ns:]
                pop = qej**2+pej**2
                l=0
                for k in range(ns):
                    if pop[k]>pop[l]: l=k # Find index of state with largest population
                if l==args.init:
                    qe[:,j] = qej; pe[:,j] = pej
                    break
        elif args.elsamp=='focused':
            """ Focussed sampling: sample from points where Phi_n = \delta_{n,init} """
            """ If there are multiple indices in initlist, set initial state to linear combination of them """
            Hn = np.sum(1./np.arange(1,ns+1))
            alpha = (ns-1.)/(Hn-1.)
            Pn = np.ones(ns)*(alpha-1.)/(alpha*ns)
            Pn[args.init] = 1./ns + (ns-1.)/(alpha*ns)
            phi = np.random.uniform(0,2*np.pi,ns)
            qe[:,j] = np.cos(phi)*np.sqrt(Pn)
            pe[:,j] = np.sin(phi)*np.sqrt(Pn)

        """ Perform basis transformation (if requested) """
        if args.initbasis=='adia':
            """ Convert from adia to dia """
            vad,U = mashf90.get_vad(q[:,j],ns)
            qe[:,j] = np.dot(U,qe[:,j])
            pe[:,j] = np.dot(U,pe[:,j])
        elif args.initbasis=='exc':
            """ Convert from exc to dia: transform given by q=0 """
            vad,U = mashf90.get_vad(0.*q[:,j],ns)
            qe[:,j] = np.dot(U,qe[:,j])
            pe[:,j] = np.dot(U,pe[:,j])

    return q,p,qe,pe

def savedata(B,t,args,name):
    tout=t/fs if args.units in ['cmm1','fs'] else t
    out = np.column_stack([tout,B])
    if name is None: name=args.obstyp
    np.savetxt('%s.out'%name,out)

