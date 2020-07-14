from __future__ import division

import numpy as np
import pandas as pd
import sys
import os
import copy
import h5py
import csv
from scipy import interpolate
import scipy.optimize as opt # for least square method

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

from matplotlib.colors import LogNorm

from data import ADD_METHODS_ALL_PAR
from comparison import TWO_SIMS, THREE_SIMS
from settings import resolutions

from model_sets.models import simulations_nonblacklisted
simulations = simulations_nonblacklisted

__outplotdir__ = "../figs/all3/rns_sequences/"

#

class Options(object):
    NJ = 64  # Number of J=const sequences to construct
    NM0 = 64  # Number of M0=const sequences to construct
    dJ = 0.01  # Step size used to rootfind the viscous ejecta
    Jmin = 3.5  # Minum J to search while rootfinding for the viscous ejecta mass
    rej = 300.0  # Estimate the ejecta assuming that everything becomes unbound once it reaches a given radius
    Jfac_fidu = lambda _, rc: (Options.rej / (np.minimum(rc, Options.rej))) ** (1. / 2.)
    list_merger_outcomes = ["PC", "BH", "HMNS", "SMNS", "MNS"]

options = Options()

class Paths:
    to_rns_sequences = "../Data/RNS/RNS.dat.gz"

# rns = globals()
seq = {}# globals() # seq[EOS] cotaining shit
mass_shedding = {}
''' ------------------| READ RNS SEUQENCES |---------------- '''

# def load_rns_table():
names = ['EOS', 'ratio', 'rho_c', 'M', 'M_0', 'r_star', 'Omega', 'Omega_K', 'I_45', 'a']
rns = pd.read_table(Paths.to_rns_sequences, names=names, sep=' ', skiprows=1)
rns = rns.sort_values(['EOS', 'rho_c', 'M'])
rns['diff_Omega'] = np.abs(rns['Omega_K'] - rns['Omega'])
rns["J"] = rns["a"] * rns["M"] ** 2
rns["P"] = 2 * np.pi / rns["Omega"] * 1e3
rns_eos = rns.EOS.unique()
print("RNS sequences loaded")
rns = rns.dropna()
# return rns

''' ----| BOUNDING SEQUENCES in J-M0 palne |----'''
# now exactly sure if it is the same
def compute_constant_J_sequences(eos):
    seq[eos] = {}
    sel = rns[rns['EOS'] == eos]

    seq[eos]= {}
    seq[eos]["J"] = np.arange(0, sel["J"].max(), sel["J"].max()/Options.NJ)
    seq[eos]["rho_c"] = np.array(sel["rho_c"].unique())
    Nrho = seq[eos]["rho_c"].shape[0]

    # make sequence
    seq[eos]["M"]   = np.empty((Options.NJ, Nrho)) # 2D
    seq[eos]["M_0"] = np.empty((Options.NJ, Nrho))
    seq[eos]["Omega"]=np.empty((Options.NJ, Nrho))
    seq[eos]["r_star"]=np.empty((Options.NJ, Nrho))

    for irho in range(Nrho):
        subsel = sel[ rns["rho_c"] == seq[eos]["rho_c"][irho] ]
        seq[eos]["M"][:,irho] =np.interp(seq[eos]["J"], subsel["J"], subsel["M"], right=np.nan)
        seq[eos]["M_0"][:,irho]=np.interp(seq[eos]["J"], subsel["J"], subsel["M_0"], right=np.nan)
        seq[eos]["Omega"][:,irho]=np.interp(seq[eos]["J"], subsel["J"], subsel["Omega"], right=np.nan)
        seq[eos]["r_star"][:, irho] = np.interp(seq[eos]["J"], subsel["J"], subsel["r_star"], right=np.nan)

    # compute properties of marginally stable stars
    seq[eos]["rho_c_max"]   = np.empty(Options.NJ)
    seq[eos]["M_max"]       = np.empty(Options.NJ)
    seq[eos]["M_0_max"]     = np.empty(Options.NJ)
    seq[eos]["Omega_max"]   = np.empty(Options.NJ)
    seq[eos]["r_star_max"]  = np.empty(Options.NJ)
    seq[eos]["stable"]      = np.zeros((Options.NJ, Nrho), dtype=np.bool)
    for ij in range(Options.NJ):
        imax = np.ma.masked_invalid(seq[eos]["M"][ij, :]).argmax()
        seq[eos]["rho_c_max"][ij]   = seq[eos]["rho_c"][imax]
        seq[eos]["M_max"][ij]       = seq[eos]["M"][ij, imax]
        seq[eos]["M_0_max"][ij]     = seq[eos]["M_0"][ij, imax]
        seq[eos]["Omega_max"][ij]   = seq[eos]["Omega"][ij, imax]
        seq[eos]["r_star_max"][ij]  = seq[eos]["r_star"][ij, imax]
        seq[eos]["stable"][ij, :imax + 1] = True
        seq[eos]["stable"][ij, imax + 1:] = False

    # create interpolator
    J_grid, rho_c_grid = np.meshgrid(seq[eos]["J"], seq[eos]["rho_c"], indexing='ij')
    xi = np.column_stack((J_grid.flatten(), rho_c_grid.flatten()))

    mask = np.zeros_like(seq[eos]["M"].flatten(), dtype=np.bool)
    for p in [seq[eos]["M"], seq[eos]["M_0"], seq[eos]["Omega"], seq[eos]["r_star"]]:
        mask = (mask) | (np.isnan(p.flatten()))

    vals = []
    for p in [seq[eos]["M"], seq[eos]["M_0"], seq[eos]["Omega"], seq[eos]["r_star"]]:
        vals.append(np.array(p.flatten()[~mask]))
    vals = np.column_stack(vals)

    seq[eos]["interp"] = interpolate.LinearNDInterpolator(np.array(xi[~mask]), vals)

def compute_bounding_sequences_in_J_M0_for_EOS(eos):

    # if not "EOS" in rns.keys(): load_rns_table()
    # print(rns)
    # if not "EOS" in rns.keys(): raise NameError()
    #
    seq[eos] = {}
    sel = rns[rns['EOS'] == eos]
    print("{} {} of models in the RNS sequence".format(eos, len(sel)))
    #
    dJ = sel["J"].max() / (Options.NJ + 1)  # steps for the J sequence
    Jmax = sel["J"].max()  # maximum J from RNS sequences for a given EOS (~100.000 for DD2(
    seq[eos]["Jf"] = np.arange(0, Jmax + dJ, dJ)  # grid of Jf
    seq[eos]["Jc"] = 0.5 * (seq[eos]["Jf"][1:] + seq[eos]["Jf"][:-1])  # center of bins for Jf
    #
    seq[eos]["M0_min"] = np.zeros_like(seq[eos]["Jc"]) # lower baryonic mass boundary
    seq[eos]["M0_max"] = np.zeros_like(seq[eos]["Jc"]) # upper baryonic mass boundary
    # Mass shedding sequence
    seq[eos]["rho_c"] = np.zeros_like(seq[eos]["Jc"]) #
    seq[eos]["ratio"] = np.zeros_like(seq[eos]["Jc"]) #
    seq[eos]["M"] = np.zeros_like(seq[eos]["Jc"]) #
    seq[eos]["J"] = np.zeros_like(seq[eos]["Jc"]) #
    seq[eos]["P0"] = np.zeros_like(seq[eos]["Jc"]) #
    # for every J_i in the bins of the sequence for EOS
    for ij in range(seq[eos]["Jc"].shape[0]):

        Jmin = seq[eos]["Jf"][ij]       # assume that the min value is the current one
        Jmax = seq[eos]["Jf"][ij + 1]   # and that the maximum value is the next
        #
        sub_sel = sel[(sel.J >= Jmin) & (sel.J < Jmax)]
        #
        imin = sub_sel["M_0"].idxmin()
        imax = sub_sel["M_0"].idxmax()
        #
        seq[eos]["M0_min"][ij] = sub_sel["M_0"][imin]
        seq[eos]["M0_max"][ij] = sub_sel["M_0"][imax]
        #
        seq[eos]["rho_c"][ij] = sub_sel["rho_c"][imin]
        seq[eos]["ratio"][ij] = sub_sel["ratio"][imin]
        seq[eos]["M"][ij] = sub_sel["M"][imin]
        seq[eos]["J"][ij] = sub_sel["J"][imin]
        seq[eos]["P0"][ij] = sub_sel["P"][imin]
    seq[eos]["M0"] = seq[eos]["M0_min"]
    #
    sel = rns[(rns["EOS"] == eos) & (rns["J"] == 0)]
    seq[eos]["M0_TOV"] = sel["M_0"].max()
    print("--- done--- ")

''' Fitting [ Jc - M0_max ] '''

def M0_max_fitting_function(x, J):
    a,b,c,d = x
    return a*J**3 + b*J**2 + c*J + d

def M0_max_residuals(x, J, M0_max):
    return M0_max - M0_max_fitting_function(x, J)

def M0_max_init_guess():
    a = 1
    b = 1
    c = 1
    d = 1
    return np.array([a,b,c,d])

def M0_max_closure(x):
    return lambda J: M0_max_fitting_function(x, J)

def get_rns_M0_max_fit(eos):
    #
    if len(seq.keys()) == 0: compute_bounding_sequences_in_J_M0_for_EOS(eos)
    #
    iopt = np.ones_like(seq[eos]["Jc"], dtype=np.bool)
    iopt[1] = False
    iopt = seq[eos]["Jc"] > 2.0
    seq[eos]["M0_max_fit_res"] = opt.least_squares(M0_max_residuals, M0_max_init_guess(),
                                                   args=(seq[eos]["Jc"][iopt], seq[eos]["M0_max"][iopt]))
    seq[eos]["M0_max_fit"] = M0_max_closure(seq[eos]["M0_max_fit_res"].x)
    print("M0_max_fit: {} {} {}".format(eos, len(seq[eos]["Jc"][iopt]),
        100 * np.max(np.abs((seq[eos]["M0_max_fit"](seq[eos]["Jc"][iopt]) - seq[eos]["M0_max"][iopt]) /
                                       seq[eos]["M0_max"][iopt]))))
    return seq[eos]

''' Fitting [ Jc - M0_min ] '''

def M0_min_fitting_function(x, J):
    a,b,c,d = x
    return a*J**2 + b*J + c*J**(0.5) + d

def M0_min_residuals(x, J, M0_min):
    return M0_min - M0_min_fitting_function(x, J)

def M0_min_init_guess():
    a = 1
    b = 1
    c = 1
    d = 1
    return np.array([a,b,c,d])

def M0_min_closure(x):
    return lambda J: M0_min_fitting_function(x, J)

def get_rns_M0_min_fit(eos):
    #
    if len(seq.keys()) == 0: compute_bounding_sequences_in_J_M0_for_EOS(eos)
    #
    #iopt = np.ones_like(seq[eos]["Jc"], dtype=np.bool)
    print(seq[eos].keys())
    iopt = seq[eos]["Jc"] > 2.0
    seq[eos]["M0_min_fit_res"] = opt.least_squares(M0_min_residuals, M0_min_init_guess(),
                                            args=(seq[eos]["Jc"][iopt], seq[eos]["M0_min"][iopt]))
    seq[eos]["M0_min_fit"] = M0_min_closure(seq[eos]["M0_min_fit_res"].x)
    print("M0_min_fit: {} {} {}".format(eos, len(seq[eos]["Jc"][iopt]),
        100*np.max(np.abs((seq[eos]["M0_min_fit"](seq[eos]["Jc"][iopt]) - seq[eos]["M0_min"][iopt])/
                     seq[eos]["M0_min"][iopt]))))

    return seq[eos]

''' Fitting [ Jc - M_min ] '''

def M_min_fitting_function(x, M0):
    a,b,c = x
    return a*M0**2 + b*M0 + c

def M_min_residuals(x, M0, M):
    return M - M_min_fitting_function(x, M0)

def M_min_init_guess():
    a = 1
    b = 1
    c = 1
    return np.array([a,b,c])

def M_min_closure(x):
    return lambda M0: M_min_fitting_function(x, M0)

def get_rns_M_min_fit(eos):#rns_eos: #options.EOS:
    #
    if len(seq.keys()) == 0: compute_bounding_sequences_in_J_M0_for_EOS(eos)
    #
    iopt = np.ones_like(seq[eos]["Jc"], dtype=np.bool)
    #iopt = seq[eos].Jc > 2.0
    seq[eos]["M_min_fit_res"] = opt.least_squares(M_min_residuals, M_min_init_guess(),
                                            args=(seq[eos]["M0_min"][iopt], seq[eos]["M"][iopt]))
    seq[eos]["M_min_fit"] = M_min_closure(seq[eos]["M_min_fit_res"].x)

    # print("M_min_fit: {} {} {}".format(eos, len(seq[eos]["Jc"][iopt]),
    #     100*np.max(np.abs((seq[eos]["M_min_fit"](seq[eos]["Jc"][iopt]) - seq[eos]["M_min"][iopt])/
    #                  seq[eos]["M0_min"][iopt]))))
    return seq

''' Fitting [ M0 - P0 ] '''

def P0_fitting_function(x, M0):
    a,b = x
    return a*(M0 - 2.5) + b

def P0_residuals(x, M0, P0):
    return P0 - P0_fitting_function(x, M0)

def P0_init_guess():
    a = 1
    b = 1
    return np.array([a,b])

def P0_closure(x):
    return lambda M0: P0_fitting_function(x, M0)

def get_rns_P0_fit(eos):#rns_eos: #options.EOS:
    #
    if len(seq.keys()) == 0: compute_bounding_sequences_in_J_M0_for_EOS(eos)
    #
    iopt = np.ones_like(seq[eos]["Jc"], dtype=np.bool)
    iopt = (seq[eos]["M0_min"] > 2.4) & (seq[eos]["M0_min"] < 2.6)
    seq[eos]["P0_fit_res"] = opt.least_squares(P0_residuals, P0_init_guess(),
                                    args=(seq[eos]["M0_min"][iopt], seq[eos]["P0"][iopt]))
    seq[eos]["P0_fit"] = P0_closure(seq[eos]["P0_fit_res"].x)
    iopt = (seq[eos]["M0_min"] > 2.0)
    seq[eos]["P0_fit_err"] = np.max(np.abs((seq[eos]["P0_fit"](seq[eos]["M0_min"][iopt]) -
                                         seq[eos]["P0"][iopt])))

def get_Jmax_and_M0_sup(eos):
    #
    if len(seq.keys()) == 0: compute_constant_J_sequences(eos)
    if not "M0_min_fit" in seq.keys(): get_rns_M0_min_fit(eos)
    if not "M0_max_fit" in seq.keys(): get_rns_M0_max_fit(eos)
    #
    Jmax = seq[eos]["Jf"][-1]
    f = lambda J: (seq[eos]["M0_max_fit"](J) - seq[eos]["M0_min_fit"](J))**2
    res = opt.minimize_scalar(f, bracket=(0.75*Jmax, 1.25*Jmax))
    seq[eos]["Jmax"] = res.x
    seq[eos]["M0_sup"] = seq[eos]["M0_max_fit"](res.x)

    return seq

''' --- Mass shedding sequence --- '''

def compute_mass_shedding_seq(eos):
    #
    L = []
    sel_1 = rns[rns.EOS == eos]
    #
    for rho in sel_1["rho_c"].unique():
        sel_2 = sel_1[rns["rho_c"] == rho]
        idx = np.abs(np.array(sel_2["diff_Omega"])).argmin()
        L.append(sel_2.iloc[idx])
    #
    mass_shedding = pd.DataFrame(L).sort_values(["rho_c", "M"])
    #

    rho_max = {}
    sel = mass_shedding#[mass_shedding.EOS == eos]
    imax = np.ma.masked_invalid(sel["M"]).argmax()
    rho_max[eos] = sel["rho_c"].iloc[imax]

    L = []
    for i, m in mass_shedding.iterrows():
        if m.rho_c > rho_max[m.EOS]:
            L.append(False)
        else:
            L.append(True)
    mass_shedding["stable"] = L

''' --- Merger Outcome --- '''

def get_merger_outcome(tcoll_gw, mb, eos):

    if len(seq.keys()) == 0: compute_bounding_sequences_in_J_M0_for_EOS(eos)
    if not "Jmax" in seq.keys(): get_Jmax_and_M0_sup(eos)

    if tcoll_gw * 1.e3 < 1.5:  # ms
        return Options.list_merger_outcomes[0]
    elif tcoll_gw * 1.e3 > 1.5 and not np.isinf(tcoll_gw):  # ms
        return Options.list_merger_outcomes[1]
    elif mb < seq[eos]["M0_TOV"]:
        return Options.list_merger_outcomes[2]
    elif mb < seq[eos]["M0_sup"]:
        return Options.list_merger_outcomes[3]
    else:
        return Options.list_merger_outcomes[4]


''' --------- PLOTTING ---------- '''
# have not finished
def plot_sequences(eos):
    #
    if len(seq.keys()) == 0: compute_bounding_sequences_in_J_M0_for_EOS(eos)
    if len(mass_shedding.keys()) == 0: compute_mass_shedding_seq(eos)
    if not "Jmax" in seq.keys(): get_Jmax_and_M0_sup(eos)
    #

    # if not "J" in seq[eos].keys():
    #     raise NameError()

    norm = mpl.colors.Normalize(vmin=0, vmax=seq[eos]["J"].max())

    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.12, 0.82-0.15, 0.92-0.12])
    cax = fig.add_axes([0.84, 0.12, 0.02, 0.92-0.12])

    cmap = plt.get_cmap("viridis")
    for ij in range(len(seq[eos]["J"])):
        color = cmap(norm(seq[eos]["J"][ij]))
        stable = seq[eos]["stable"][ij]
        ax.plot(seq[eos]["rho_c"][stable], seq[eos]["M"][ij, stable], color=color)
        ax.plot(seq[eos]["rho_c_max"][ij], seq[eos]["M_max"][ij], 'o', color='k')
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
    cbar.set_label(r"$J$")

    ms_sel = mass_shedding[(mass_shedding["stable"])]
    ax.plot(ms_sel["rho_c"], ms_sel["M"], 'k-')

    ax.set_xlabel(r"$\rho_c$")
    ax.set_ylabel(r"$M$")

    ax.set_title(eos)

    print("plotted: {}".format(__outplotdir__ + "Jseq_{}.png".format(eos)))
    plt.savefig(__outplotdir__+"Jseq_{}.png".format(eos))
    plt.close()

def plot_rns_J_Mb_sequeces(eos):
    #
    if len(seq.keys()) == 0: compute_bounding_sequences_in_J_M0_for_EOS(eos)
    if not "M0_min_fit" in seq.keys(): get_rns_M0_min_fit(eos)
    if not "M0_max_fit" in seq.keys(): get_rns_M0_max_fit(eos)
    if not "Jmax" in seq.keys(): get_Jmax_and_M0_sup(eos)
    #

    fig = plt.figure(figsize=[4, 2.5])
    ax = fig.add_axes([0.15, 0.15, 0.95 - 0.15, 0.95 - 0.15])

    ax.scatter(seq[eos]["Jc"], seq[eos]["M0_min"], color="blue")
    ax.scatter(seq[eos]["Jc"], seq[eos]["M0_max"], color="red")

    J = np.linspace(0, seq[eos]["Jmax"], 100)
    ax.plot(J, seq[eos]["M0_min_fit"](J), color="blue")
    ax.plot(J, seq[eos]["M0_max_fit"](J), color="red")

    ax.set_xlim(xmin=2)
    ax.set_ylim(ymin=2.5)

    ax.set_xlabel(r"$J\ [G\, c^{-1} M_\odot^2]$")
    ax.set_ylabel(r"$M_b\ [M_\odot]$")

    ax.text(0.075, 0.925, eos, fontsize='large',
            va='top', transform=ax.transAxes)

    print("plotted: {}".format(__outplotdir__ + "rns_seq_{}.png".format(eos.lower())))
    plt.savefig(__outplotdir__ + "rns_seq_{}.png".format(eos.lower()), dpi=256)
    plt.close()

def plot_rns_Mb_M_massshedding_sequeces(eos):
    #
    if len(seq.keys()) == 0: compute_bounding_sequences_in_J_M0_for_EOS(eos)
    if not "M0_min_fit" in seq.keys(): get_rns_M0_min_fit(eos)
    if not "M0_max_fit" in seq.keys(): get_rns_M0_max_fit(eos)
    if not "M_min_fit" in seq.keys(): get_rns_M_min_fit(eos)
    #

    fig = plt.figure(figsize=[4, 2.5])
    ax = fig.add_axes([0.15, 0.15, 0.95 - 0.15, 0.95 - 0.15])

    ax.scatter(seq[eos]["M0_min"], seq[eos]["M"], color="red")

    M0 = np.linspace(seq[eos]["M0_min"].min(), seq[eos]["M0_min"].max(), 100)
    ax.plot(M0, seq[eos]["M_min_fit"](M0), color="red")

    ax.set_xlim(xmin=2)
    ax.set_ylim(ymin=2)

    ax.set_xlabel(r"$M_b\ [M_\odot]$")
    ax.set_ylabel(r"$M\ [M_\odot]$")

    ax.text(0.075, 0.925, eos, fontsize='large', va='top', transform=ax.transAxes)

    print(__outplotdir__ + "mass_shedding_mass_{}.png".format(eos.lower()))
    plt.savefig(__outplotdir__ + "mass_shedding_mass_{}.png".format(eos.lower()), dpi=256)
    plt.close()

def plot_P0_Mb_sequences(eos):

    if len(seq.keys()) == 0: compute_bounding_sequences_in_J_M0_for_EOS(eos)
    if not "M0_min" in seq.keys(): get_rns_M0_min_fit(eos)
    if not "P0" in seq.keys(): get_rns_P0_fit(eos)
    #

    fig = plt.figure(figsize=[4, 2.5])
    ax = fig.add_axes([0.15, 0.15, 0.95 - 0.15, 0.95 - 0.15])

    ax.plot(seq[eos]["M0_min"], seq[eos]["P0"], color="blue", label=eos)

    ax.set_xlim(xmin=1.5, xmax=3.5)
    ax.set_ylim(ymin=0.5, ymax=1.25)

    ax.set_ylabel(r"$P_0\ [{\rm ms}]$")
    ax.set_xlabel(r"$M_b\ [M_\odot]$")

    ax.legend(loc="upper right", ncol=2)

    print(__outplotdir__ + "mass_shedding_spin_{}.png".format(eos))
    plt.savefig(__outplotdir__ + "mass_shedding_spin_{}.png".format(eos), dpi=256)
    plt.close()

def plot_P0_Mb_sequences_with_fit(eos):

    if len(seq.keys()) == 0: compute_bounding_sequences_in_J_M0_for_EOS(eos)
    if not "M0_min" in seq.keys(): get_rns_M0_min_fit(eos)
    if not "P0" in seq.keys(): get_rns_P0_fit(eos)
    #

    fig = plt.figure(figsize=[4, 2.5])
    ax = fig.add_axes([0.15, 0.15, 0.95 - 0.15, 0.95 - 0.15])

    ax.plot(seq[eos]["M0_min"], seq[eos]["P0"], "o")
    ax.plot(seq[eos]["M0_min"], seq[eos]["P0_fit"](seq[eos]["M0_min"]), "-")

    ax.set_xlim(xmin=2.0, xmax=3.)
    ax.set_ylim(ymin=0.5, ymax=2.0)

    ax.set_ylabel(r"$P_0\ [{\rm ms}]$")
    ax.set_xlabel(r"$M_b\ [M_\odot]$")

    print(__outplotdir__ + "mass_shedding_spin_{}.png".format(eos.lower()))
    plt.savefig(__outplotdir__ + "mass_shedding_spin_{}.png".format(eos.lower()), dpi=256)
    plt.close()

if __name__ == '__main__':

    plot_sequences("BLh")
    exit(1)

    plot_rns_J_Mb_sequeces("BLh")
    plot_rns_Mb_M_massshedding_sequeces("BLh")
    plot_P0_Mb_sequences("BLh")
    plot_P0_Mb_sequences_with_fit("BLh")
