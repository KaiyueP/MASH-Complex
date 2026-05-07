import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
plt.rcParams['font.size'] = 15

autoev = 27.2114079527  # 1 Hartree in eV

def plot_mash_population_and_avg_energy(
    pop_file,
    lmd_file,
    exciton_file,
    n_start=0,
    time_conversion_factor=0.0241888,  # a.u. -> fs; set to None to skip
    save_pop_fig=None,
    save_avgE_fig="avgE_compare.pdf",
):
    """
    Plot (1) state populations vs time from a MASH population file,
    and (2) the corresponding average energy vs time.

    Parameters
    ----------
    pop_file : str
        Path to the MASH population file. First column = time,
        remaining columns = populations of each state.
    lmd_file : str
        Path to the lambda matrix file (lmd_nm.dat).
    exciton_file : str
        Path to the exciton energy file (e.g. ../bse/exciton.dat).
        Second column is assumed to be exciton energies in Hartree.
    n_start : int, optional
        Starting index (0-based) in the exciton.dat file for the states.
    time_conversion_factor : float or None, optional
        Factor to multiply the time column by (e.g. a.u. -> fs).
        If None, no conversion is done.
    save_pop_fig : str or None, optional
        Filename for saving the population figure. If None, not saved.
    save_avgE_fig : str or None, optional
        Filename for saving the average energy figure. If None, not saved.

    Returns
    -------
    fig_pop : matplotlib.figure.Figure
        Figure for populations vs time.
    fig_avgE : matplotlib.figure.Figure
        Figure for average energy vs time.
    """

    # ---- Helper: read exciton energies in eV ----
    def read_excitonE(excit_file, n_state, n_start):
        # Second column in Hartree, convert to eV
        excit_en = np.loadtxt(excit_file, usecols=1) * autoev
        excit_en = excit_en[n_start:n_start + n_state]
        return excit_en

    # ---- Load population data ----
    mash = np.loadtxt(pop_file)

    # Convert time if requested
    if time_conversion_factor is not None:
        mash[:, 0] *= time_conversion_factor

    # Determine the number of population columns actually present
    num_lines = mash.shape[1] - 1  # first column is time

    # We'll only use up to min(num_lines, n_state) states consistently
    n_state = num_lines

    # ---- Plot 1: populations vs time ----
    cmap = plt.get_cmap('viridis')
    norm = Normalize(vmin=0, vmax=n_state - 1)

    fig_pop, ax_pop = plt.subplots(figsize=(10, 6))

    for n in range(n_state):
        color = cmap(norm(n))
        ax_pop.plot(
            mash[:, 0],
            mash[:, n + 1],
            label=f"mash{n + 1}",
            color=color,
            linestyle='-'
        )

    ax_pop.set_xlabel("Time (fs)" if time_conversion_factor is not None else "Time (a.u.)")
    ax_pop.set_ylabel("Population")
    ax_pop.set_title("Population Over Time")
    ax_pop.legend(loc='best', fontsize='small')
    fig_pop.tight_layout()

    if save_pop_fig is not None:
        fig_pop.savefig(save_pop_fig, dpi=500)

    plt.close()


    # ---- Plot 2: average energy vs time ----
    lmd = np.loadtxt(lmd_file)
    lmd_diag = np.diag(lmd)[:n_state]

    exciton = read_excitonE(exciton_file, n_state, n_start)[:n_state]

    avgE_mash = np.zeros(mash.shape[0])

    for n in range(n_state):
        avgE_mash += (exciton[n] - lmd_diag[n]) * mash[:, n + 1]

    fig_avgE, ax_avgE = plt.subplots(figsize=(10, 6))
    ax_avgE.plot(mash[:, 0], avgE_mash, label="mash")

    ax_avgE.set_xlabel("Time (fs)" if time_conversion_factor is not None else "Time (a.u.)")
    ax_avgE.set_ylabel("Average Energy (eV)")
    ax_avgE.set_title("Average Energy Over Time")
    ax_avgE.legend(loc='best', fontsize='small')
    ax_avgE.set_xscale("log")
    fig_avgE.tight_layout()

    if save_avgE_fig is not None:
        fig_avgE.savefig(save_avgE_fig, dpi=500)

    # Show both figures
    plt.show()
    plt.close()

    plt.plot(np.arange(len(lmd_diag)), lmd_diag, "-o")
    plt.savefig("lmd.png")

    return fig_pop, fig_avgE


fig_pop, fig_avgE = plot_mash_population_and_avg_energy(
    pop_file="./pop.out",
    lmd_file="./lmd_nm.dat",
    exciton_file="./exciton.dat",
    n_start=0,
    save_pop_fig="pop_mash.png",
    save_avgE_fig="avgE_compare.png",
)


