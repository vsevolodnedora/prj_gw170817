# ---
#
# Attempt to reconstract a way to compute the Viscouse ejecta masses
# from how much barionic mass and angular momentum is there to be
# removed for rigidly rotating configuration
#
# ---
from __future__ import division
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import numpy as np
import re
import sys
from scivis import units as ut
import copy
import csv
from scipy import interpolate
import scipy.optimize as opt # for least square method
sys.path.append('/data01/numrel/vsevolod.nedora/bns_ppr_tools/')
from preanalysis import LOAD_INIT_DATA
from outflowed import EJECTA_PARS
from preanalysis import LOAD_ITTIME
from plotting_methods import PLOT_MANY_TASKS
from tables import *
# from settings import simulations, old_simulations, resolutions
from legacy import _models_old
from legacy._models_old import Paths
from data import ADD_METHODS_ALL_PAR

__outplotdir__ = "/data01/numrel/vsevolod.nedora/figs/all3/prj_viscous_ejecta/"

''' ----------------------------------------------| SETTINGS |-------------------------------------------------------'''

class Struct(object):
    pass

options = Struct()

# ---
sim = "BLh_M13641364_M0_LK_SR"#"DD2_M13641364_M0_LK_SR_R04"
options.EOS = ["BLh", "DD2", "LS220", "SFHo", "SLy4"]
options.NJ  = 64    # Number of J=const sequences to construct
options.NM0 = 64    # Number of M0=const sequences to construct
options.dJ = 0.01   # Step size used to rootfind the viscous ejecta
options.Jmin = 3.5  # Minum J to search while rootfinding for the viscous ejecta mass
options.rej = 300.0 # Estimate the ejecta assuming that everything becomes unbound once it reaches a given radius
options.Jfac_fidu = lambda rc: (options.rej/(np.minimum(rc, options.rej)))**(1./2.)
# ---

sims = _models_old.simulations[_models_old.correct_init_data]
#
sims["M"] = sims["M1"] + sims["M2"]
sims["nu"] = sims["M1"]*sims["M2"]/(sims["M"]**2)
sims["q"] = sims["M2"]/sims["M1"]

sims["Eb"] = (sims["M"] - sims["MADM"] + sims["EGW"])/(sims["M"]*sims["nu"])
sims["j"] = (sims["JADM"] - sims["JGW"])/(sims["M"]**2*sims["nu"])

sims["Mfinal"] = sims["MADM"] - sims["EGW"]
sims["Jfinal"] = sims["JADM"] - sims["JGW"]
sims["afinal"] = sims["Jfinal"]/(sims["Mfinal"]**2)

# print(sims.loc["BLh_M13641364_M0_LK_SR"]["JADM"], sims.loc["BLh_M13641364_M0_LK_SR"]["JGW"]); exit(1)

''' ------------------| READ RNS SEUQENCES |---------------- '''

names = ['EOS', 'ratio', 'rho_c', 'M', 'M_0', 'r_star', 'Omega', 'Omega_K', 'I_45', 'a']
rns = pd.read_table(Paths.to_rns_sequences, names=names, sep=' ', skiprows=1)
rns = rns.sort_values(['EOS', 'rho_c', 'M'])
rns['diff_Omega'] = np.abs(rns['Omega_K'] - rns['Omega'])
rns["J"] = rns["a"]*rns["M"]**2
rns["P"] = 2*np.pi/rns["Omega"]*1e3
rns_eos = rns.EOS.unique()

rns = rns.dropna()

''' ------------------| READ TOV data |----------------------'''

M_of_M0 = {} # function
for eos in options.EOS:
    rhoc, M, M0 = np.loadtxt(Paths.to_tovs+"{}_sequence.txt".format(eos),
                             usecols=(0,1,2), unpack=True)
    imax = rhoc.argmax()
    M = M[:imax]
    M0 = M0[:imax]

    M_of_M0[eos] = interpolate.interp1d(M0, M, bounds_error=False)

''' ----| BOUNDING SEQUENCES in J-M0 palne |----'''

seq = {} # seq[EOS] cotaining shit
for eos in options.EOS: #rns_eos: #
    seq[eos] = {}               # to append shit
    sel = rns[rns.EOS == eos]   # RNS sequences for a given
    # print(sel.J) ***EQVIVALENT***
    # print(sel["J"]); exit(1)
    print("{} {} of models in the sequence".format(eos, len(sel)))
    #
    dJ = sel["J"].max() / (options.NJ + 1) # steps for the J sequence
    Jmax = sel["J"].max() # maximum J from RNS sequences for a given EOS (~100.000 for DD2(
    seq[eos]["Jf"] = np.arange(0, Jmax + dJ, dJ) # grid of Jf
    seq[eos]["Jc"] = 0.5 * (seq[eos]["Jf"][1:] + seq[eos]["Jf"][:-1]) # center of bins for Jf
    #
    seq[eos]["M0_min"] = np.zeros_like(seq[eos]["Jc"]) # lower baryonic mass boundary
    seq[eos]["M0_max"] = np.zeros_like(seq[eos]["Jc"]) # upper baryonic mass boundary
    # Mass shedding sequence
    seq[eos]["rho_c"] = np.zeros_like(seq[eos]["Jc"]) #
    seq[eos]["ratio"] = np.zeros_like(seq[eos]["Jc"]) #
    seq[eos]["M"] = np.zeros_like(seq[eos]["Jc"]) #
    seq[eos]["J"] = np.zeros_like(seq[eos]["Jc"]) #
    seq[eos]["P0"] = np.zeros_like(seq[eos]["Jc"]) #
    #
    for ij in range(seq[eos]["Jc"].shape[0]):
        # for every J_i in the bins of the sequence for EOS
        Jmin = seq[eos]["Jf"][ij] # assume that the min value is the current one
        Jmax = seq[eos]["Jf"][ij + 1] # and that the maximum value is the next
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

for eos in options.EOS: # rns_eos: #
    sel = rns[(rns.EOS == eos) & (rns.J == 0)]
    seq[eos]["M0_TOV"] = sel["M_0"].max()

''' Fitting Functions VIA least squares [ Jc - M0_max ] '''

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

for eos in options.EOS: # rns_eos:
    iopt = np.ones_like(seq[eos]["Jc"], dtype=np.bool)
    iopt[1] = False
    iopt = seq[eos]["Jc"] > 2.0
    seq[eos]["M0_max_fit_res"] = opt.least_squares(M0_max_residuals, M0_max_init_guess(),
                                            args=(seq[eos]["Jc"][iopt], seq[eos]["M0_max"][iopt]))
    seq[eos]["M0_max_fit"] = M0_max_closure(seq[eos]["M0_max_fit_res"].x)
    print("{} {} {}".format(eos, len(seq[eos]["Jc"][iopt]),
    100*np.max(np.abs((seq[eos]["M0_max_fit"](seq[eos]["Jc"][iopt]) - seq[eos]["M0_max"][iopt])/
                 seq[eos]["M0_max"][iopt]))))

''' Fitting Functions VIA least squares [ Jc - M0_min ] '''

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

for eos in options.EOS:#rns_eos: #options.EOS:
    iopt = np.ones_like(seq[eos]["Jc"], dtype=np.bool)
    iopt = seq[eos]["Jc"] > 2.0
    seq[eos]["M0_min_fit_res"] = opt.least_squares(M0_min_residuals, M0_min_init_guess(),
                                            args=(seq[eos]["Jc"][iopt], seq[eos]["M0_min"][iopt]))
    seq[eos]["M0_min_fit"] = M0_min_closure(seq[eos]["M0_min_fit_res"].x)
    print("{} {} {}".format(eos, len(seq[eos]["Jc"][iopt]),
        100*np.max(np.abs((seq[eos]["M0_min_fit"](seq[eos]["Jc"][iopt]) - seq[eos]["M0_min"][iopt])/
                     seq[eos]["M0_min"][iopt]))))

''' Fitting Functions VIA least squares [ Jc - M_min ] '''

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

for eos in options.EOS:#rns_eos: #options.EOS:
    iopt = np.ones_like(seq[eos]["Jc"], dtype=np.bool)
    #iopt = seq[eos].Jc > 2.0
    seq[eos]["M_min_fit_res"] = opt.least_squares(M_min_residuals, M_min_init_guess(),
                                            args=(seq[eos]["M0_min"][iopt], seq[eos]["M"][iopt]))
    seq[eos]["M_min_fit"] = M_min_closure(seq[eos]["M_min_fit_res"].x)

''' Fitting Functions VIA least squares [ M0 - P0 ] '''

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

for eos in options.EOS:#rns_eos: #options.EOS:
    iopt = np.ones_like(seq[eos]["Jc"], dtype=np.bool)
    iopt = (seq[eos]["M0_min"] > 2.4) & (seq[eos]["M0_min"] < 2.6)
    seq[eos]["P0_fit_res"] = opt.least_squares(P0_residuals, P0_init_guess(),
                                    args=(seq[eos]["M0_min"][iopt], seq[eos]["P0"][iopt]))
    seq[eos]["P0_fit"] = P0_closure(seq[eos]["P0_fit_res"].x)
    iopt = (seq[eos]["M0_min"] > 2.0)
    seq[eos]["P0_fit_err"] = np.max(np.abs((seq[eos]["P0_fit"](seq[eos]["M0_min"][iopt]) -
                                         seq[eos]["P0"][iopt])))

for eos in options.EOS: # rns_eos: #options.EOS:
    Jmax = seq[eos]["Jf"][-1]
    f = lambda J: (seq[eos]["M0_max_fit"](J) - seq[eos]["M0_min_fit"](J))**2
    res = opt.minimize_scalar(f, bracket=(0.75*Jmax, 1.25*Jmax))
    seq[eos]["Jmax"] = res.x
    seq[eos]["M0_sup"] = seq[eos]["M0_max_fit"](res.x)

''' -------| Merger outcome | ------- '''

MergerOutcomes = ["PC", "BH", "HMNS", "SMNS", "MNS"]

class MergerOutcome:
    PC   = MergerOutcomes[0]
    BH   = MergerOutcomes[1]
    HMNS = MergerOutcomes[2]
    SMNS = MergerOutcomes[3]
    MNS  = MergerOutcomes[4]

L = []
for _, m in sims.iterrows():
    if m.tcoll_gw*1.e3 < 1.5: # ms
        L.append(MergerOutcome.PC)
    elif m.tcoll_gw*1.e3 > 1.5 and not np.isinf(m.tcoll_gw): # ms
        L.append(MergerOutcome.BH)
    elif m.Mb < seq[m.EOS]["M0_TOV"]:
        L.append(MergerOutcome.MNS)
    elif m.Mb < seq[m.EOS]["M0_sup"]:
        L.append(MergerOutcome.SMNS)
    else:
        L.append(MergerOutcome.HMNS)
sims["outcome"] = L

# print(len(sims[sims.resolution == 0.083333]))
# print(len(sims[models.fiducial]))
# print(len(sims[models.with_disk_mass]))
# print(len(sims[models.with_disk_mass &
#               ((sims.outcome == MergerOutcome.SMNS) |
#                (sims.outcome == MergerOutcome.MNS))]))
# print(len(sims[models.fiducial &
#               ((sims.outcome == MergerOutcome.SMNS) |
#                (sims.outcome == MergerOutcome.MNS))]))
# sims["q"].min()

L = []
for _, m in sims.iterrows():
    if np.isfinite(m.tcoll):
        L.append(0)
    else:
        o_data = ADD_METHODS_ALL_PAR(sim)
        table = o_data.get_gw_data("EJ.dat").T
        #
        t, Jdot, JGW = table[:, 0], table[:, 3], table[:, 5]
        #
        t = ut.conv_time(ut.cactus, ut.cgs, t)*1e3
        tGW = ut.conv_time(ut.cactus, ut.cgs, (m.JADM - JGW)/Jdot)*1e3
        idx = (t > t.max() - 1.5) & (t < t.max() - 0.5) # over the last 1 ms
        L.append(np.mean(tGW[idx]))
sims["tGW"] = L
# print sims[models.fiducial]["tGW"].argmin()

''' -------| Viscouse Ejecta |------- '''
# Root-finder
def step_to_zero(fun, xstart=1., xmin=0., dx=0.1):
    xp = xstart
    while fun(xp) >= 0 and xp >= xmin:
        xp -= dx
    return xp

long_sims = ["BLh_M13641364_M0_LK_SR"]

dM_max, dJ_max = {}, {}
dM_fidu, dJ_fidu = {}, {}
for name, model in sims.iterrows():
    if name in long_sims:# == MergerOutcome.SMNS or model.outcome == MergerOutcome.MNS:

        o_data = ADD_METHODS_ALL_PAR(name)
        mjenclosed = o_data.get_3d_data("MJ_encl.txt")
        if not len(mjenclosed) > 0: raise ValueError("No disk data for: {}".format(name))
        iters, times, mjenclosed = mjenclosed[0], mjenclosed[1], mjenclosed[2] # 1D, 1D, list(2D arrays)
        #
        mj = mjenclosed[-1]  # list 0f 2D array [1:rcyl 2:drcyl 3:M 4:J 5:I]
        rc, drc, M, J = mj[0, :], mj[1, :], mj[2, :], mj[3, :]
        #
        Jf_max = model.Jfinal - np.cumsum((J * rc * drc)[::-1]) # array of shells (excluding one, after another, from outermost)
        Mf = model.Mb - np.cumsum((M * rc * drc)[::-1]) # masses of the consequeteve shells
        Mf_fun = interpolate.interp1d(Jf_max[::-1], Mf[::-1], kind="linear", assume_sorted=False)
        #
        fun = lambda Jf: seq[model.EOS]["M0_min_fit"](Jf) - Mf_fun(Jf)
        x = step_to_zero(fun, xstart=Jf_max.max(), dx=options.dJ, xmin=options.Jmin)
        #
        dM_max[name] = model.Mb - Mf_fun(x) # how much Mb to remove to reach RNS sequence
        dJ_max[name] = model.Jfinal - x     # how much J  to remove to reach RNS sequence
        #
        # --- Fedu ---
        #
        Jf_fidu = model.Jfinal - np.cumsum((J * rc * options.Jfac_fidu(rc) * drc)[::-1])
        Mf_fun = interpolate.interp1d(Jf_fidu[::-1], Mf[::-1], kind="linear", assume_sorted=False)
        fun = lambda Jf: seq[model.EOS]["M0_min_fit"](Jf) - Mf_fun(Jf)
        x = step_to_zero(fun, xstart=Jf_fidu.max(), dx=options.dJ, xmin=options.Jmin)
        #
        dM_fidu[name] = model.Mb - Mf_fun(x)
        dJ_fidu[name] = model.Jfinal - x
    else:
        dM_max[name] = np.nan
        dJ_max[name] = np.nan
        dM_fidu[name] = np.nan
        dJ_fidu[name] = np.nan

sims["dM_0_max"] = pd.Series(dM_max)
sims["dJ_rns_max"] = pd.Series(dJ_max)
sims["dM_0_fidu"] = pd.Series(dM_fidu)
sims["dJ_rns_fidu"] = pd.Series(dJ_fidu)

''' ---| Error Estimation |--- '''

# sel = (not sims.tcoll_gw) | (not np.isnan(sims.Mdisk3D))
#
# hr = []
# for name, m in sims[sel].iterrows():
#     if re.match(".*_HR", name):
#         c = m.copy()
#         c.name = name.replace("_HR", "")
#         hr.append(c)
# hr = pd.DataFrame(hr, columns=sims.columns)
#
# for n, m in hr.iterrows():
#     rel_err = np.abs(sims.loc[n]["dM_0_max"] - m["dM_0_max"]) / sims.loc[n]["dM_0_max"]
#     print((n, rel_err))
#
# print(("-" * 60))
#
# for n, m in hr.iterrows():
#     rel_err = np.abs(sims.loc[n]["dM_0_fidu"] - m["dM_0_fidu"]) / sims.loc[n]["dM_0_fidu"]
#     print((n, rel_err))

''' ---| PLOT/SAVE RNS DATA |---'''

plot_rns_sequnces = True
save_rns_sequences = True

# Plot Jc - M0
if plot_rns_sequnces:
    for eos in options.EOS:
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

        plt.savefig(__outplotdir__ + "rns_seq_{}.png".format(eos.lower()), dpi=256)
        plt.close()

# PLOT M0 - M
if plot_rns_sequnces:
    for eos in options.EOS:
        fig = plt.figure(figsize=[4, 2.5])
        ax = fig.add_axes([0.15, 0.15, 0.95 - 0.15, 0.95 - 0.15])

        ax.scatter(seq[eos]["M0_min"], seq[eos]["M"], color="red")

        M0 = np.linspace(seq[eos]["M0_min"].min(), seq[eos]["M0_min"].max(), 100)
        ax.plot(M0, seq[eos]["M_min_fit"](M0), color="red")

        ax.set_xlim(xmin=2)
        ax.set_ylim(ymin=2)

        ax.set_xlabel(r"$M_b\ [M_\odot]$")
        ax.set_ylabel(r"$M\ [M_\odot]$")

        ax.text(0.075, 0.925, eos, fontsize='large',
                va='top', transform=ax.transAxes)

        plt.savefig(__outplotdir__ + "mass_shedding_mass_{}.png".format(eos.lower()), dpi=256)
        plt.close()

#
if plot_rns_sequnces:

    fig = plt.figure(figsize=[4, 2.5])
    ax = fig.add_axes([0.15, 0.15, 0.95 - 0.15, 0.95 - 0.15])

    for idx, eos in enumerate(options.EOS):
        ax.plot(seq[eos]["M0_min"], seq[eos]["P0"], color="blue",
                label=eos)

    ax.set_xlim(xmin=1.5, xmax=3.5)
    ax.set_ylim(ymin=0.5, ymax=1.25)

    ax.set_ylabel(r"$P_0\ [{\rm ms}]$")
    ax.set_xlabel(r"$M_b\ [M_\odot]$")

    ax.legend(loc="upper right", ncol=2)

    plt.savefig(__outplotdir__ + "mass_shedding_spin.png", dpi=256)
    plt.close()

if plot_rns_sequnces:

    for eos in options.EOS:
        fig = plt.figure(figsize=[4, 2.5])
        ax = fig.add_axes([0.15, 0.15, 0.95 - 0.15, 0.95 - 0.15])

        ax.plot(seq[eos]["M0_min"], seq[eos]["P0"], "o")
        ax.plot(seq[eos]["M0_min"], seq[eos]["P0_fit"](seq[eos]["M0_min"]), "-")

        ax.set_xlim(xmin=2.0, xmax=3.)
        ax.set_ylim(ymin=0.5, ymax=2.0)

        ax.set_ylabel(r"$P_0\ [{\rm ms}]$")
        ax.set_xlabel(r"$M_b\ [M_\odot]$")

        plt.savefig(__outplotdir__ + "mass_shedding_spin_{}.png".format(eos.lower()), dpi=256)
        plt.close()

if save_rns_sequences:
    ofile = open(__outplotdir__ + "mass_shedding_spin.txt", "w")
    ofile.write("EOS a b e\n")
    for eos in options.EOS:
        ofile.write("{} {} {} {}\n".format(eos, seq[eos]["P0_fit_res"].x[0], seq[eos]["P0_fit_res"].x[1],
                                           seq[eos]["P0_fit_err"]))
    del ofile

if save_rns_sequences:
    for eos in options.EOS:
        ofile = open(__outplotdir__ + "mass_shedding_{}.txt".format(eos.lower()), "w")
        ofile.write("# 1:rho_c 2:axis_ratio 3:M 4:J 5:P0 6:M0\n")
        for ij in range(seq[eos]["Jc"].shape[0]):
            ofile.write("{} {} {} {} {} {}\n".format(
                seq[eos]["rho_c"][ij],
                seq[eos]["ratio"][ij],
                seq[eos]["M"][ij],
                seq[eos]["J"][ij],
                seq[eos]["P0"][ij],
                seq[eos]["M0"][ij]))

''' ---| PLOT RNS + MODELS |--- '''

plot_merger_outcome = True
save_merger_outcome = True


if plot_merger_outcome:

    # limits = {"SLy4":[(3., 7.), (2.5, 3.5)],
    #           "SFho":[(3., 7.), (2.5, 3.5)],
    #           "LS220":[(3., 7.), (2.5, 3.5)],
    #           "BLh":[(3., 7.), (2.5, 3.5)],
    #           "DD2":[(3., 7.), (2.5, 3.5)]}

    for eos in options.EOS:
        fig = plt.figure(figsize=[4, 2.5])
        ax = fig.add_axes([0.14, 0.15, 1.0 - 0.14 * 2, 0.95 - 0.15])

        J = np.linspace(0, seq[eos]["Jmax"], 100)

        ax.fill_between(J, seq[eos]["M0_min_fit"](J), seq[eos]["M0_max_fit"](J),
                        edgecolor='black', facecolor='lightgrey',
                        label="RNS")

        for idx, outcome in enumerate(MergerOutcomes):
            sel1 = sims[(sims.EOS == eos) & (sims.outcome == outcome) & _models_old.fiducial]
            if len(sel1.Jfinal) > 0:
                ax.plot(sel1.Jfinal, sel1.Mb, _models_old.marker_list[idx],
                        color=_models_old.color_list[idx], markeredgecolor='black', label=str(outcome))
            # for Jfinal, Mb, vis in zip(sel1.Jfinal, sel1.Mb, sel1.viscosity):
            #     if vis == "LK":
            #         ax.plot(sel1.Jfinal, sel1.Mb, models.marker_list[idx],
            #             color=models.color_list[idx], markeredgecolor='black')
            #     else:
            #         ax.plot(sel1.Jfinal, sel1.Mb, models.marker_list[idx],
            #                 color=models.color_list[idx], label=str(outcome))



            # print(eos, models.color_list[idx], str(outcome), len(sel1.Jfinal))
            # if len(sel1.Jfinal) > 0:
            #     ax.plot(sel1.Jfinal, sel1.Mb, models.marker_list[idx],
            #             color=models.color_list[idx], label=str(outcome))

        # for idx, outcome in enumerate(MergerOutcomes):
        #     #
        #     sel1 = sims[(sims.EOS == eos) & (sims.outcome == outcome) & models.fiducial & models.viscous]
        #     sel2 = sims[(sims.EOS == eos) & (sims.outcome == outcome) & models.fiducial & models.non_viscous]
        #
        #     print(eos, models.color_list[idx], str(outcome), len(sel1.Jfinal), len(sel2.Jfinal))
        #     if len(sel1.Jfinal) > 0:
        #         if len(sel2.Jfinal) == 0:
        #             ax.plot(sel1.Jfinal, sel1.Mb, models.marker_list[idx],
        #                     color=models.color_list[idx], label=str(outcome), markeredgecolor='black')
        #         else:
        #             ax.plot(sel1.Jfinal, sel1.Mb, models.marker_list[idx],
        #                     color=models.color_list[idx], markeredgecolor='black')
        #     #
        #     if len(sel2.Jfinal) > 0:
        #         if len(sel1.Jfinal) == 0:
        #             ax.plot(sel2.Jfinal, sel2.Mb, models.marker_list[idx],
        #                     color=models.color_list[idx], label=str(outcome))
        #         else:
        #             ax.plot(sel2.Jfinal, sel2.Mb, models.marker_list[idx],
        #                     color=models.color_list[idx])

        # if eos == "BLh":
            # mod = sims.loc["BLh_M13641364_M0_LK_SR"]
            # ax.plot(mod.Jfinal, mod.Mb, "*", color="gold", label=r"$(1.364+1.364)M_\odot$ -- M0 -- LK")

        # ax.axhline(y=seq[eos].M0_TOV, color="black", linestyle=":", label=r"$M_{\rm TOV}$")
        # ax.axhline(y=seq[eos].M0_sup, color="black", linestyle="--", label=r"$M_{\rm SMNS}$")


        ax.set_xlim(3, 7)
        M0_min = 2.5
        M0_max = 3.5
        ax.set_ylim(M0_min, M0_max)

        ax.set_xlabel(r"$J\ [G\, c^{-1} M_\odot^2]$")
        ax.set_ylabel(r"$M_b\ [M_\odot]$")

        han, lab = ax.get_legend_handles_labels()
        # if eos == "DD2":
        ax.add_artist(ax.legend(han[:-1], lab[:-1], loc="upper left"))
        ax.add_artist(ax.legend([han[-1]], [lab[-1]], loc="lower right"))
        if eos == "BLh":
            ax.add_artist(ax.legend([han[-2]], [lab[-2]], loc="lower right",
                                    handletextpad=0.05))
        ax.text(0.5, 0.95, eos, ha="center", va="top", transform=ax.transAxes, fontsize="large")

        ax2 = ax.twinx()
        ax2.set_ylim(M0_min, M0_max)
        ax2.set_ylabel(r"$M\ [M_\odot]$")
        # ax2.axhline(y=2.5, color="black", linestyle="--")

        M0_ticks = np.linspace(M0_min, M0_max, 7)
        # print(M0_ticks, )
        M_ticks = 2 * M_of_M0[eos](M0_ticks / 2.)
        M_labels = ["{:.2f}".format(m) for m in M_ticks]
        ax2.set_yticks(M0_ticks)
        ax2.set_yticklabels(M_labels)

        plt.savefig(__outplotdir__ + "outcome_{}.png".format(eos.lower()), dpi=256)
        plt.close()

if save_merger_outcome:
    ofile = open(__outplotdir__ + "outcome.txt", "w")
    for name, m in sims.iterrows():
        ofile.write("{} {}\n".format(name, m["outcome"]))
    ofile.close()


''' ---| PLOT VISCOUS EJECTA |--- '''

plot_models = [
    ("BLh_M13641364_M0_LK_SR", (3., 5.), (2.6, 3.1)),    # J -- Mb
    ("BLh_M11461635_M0_LK_SR", (3., 6.5), (2.6, 3.1)),
    ("DD2_M13641364_M0_LK_SR_R04", (3., 6.5), (2.6, 3.1)),
    ("DD2_M13641364_M0_SR_R04", (3., 6.5), (2.6, 3.1)),
    ("DD2_M15091235_M0_LK_SR", (3., 6.5), (2.6, 3.1)),
    # ("DD2_M13641364_M0_LK_SR_R04", (3., 8), (2.2, 3.2)) #
]
#

def __plot_total_J_Mb_evol(ax, o_data, times, color='red', ls='-', lw=1.):
    """

    :param ax:
    :param o_data:
    :param times:  list of postmerger times in [ms]
    :param color:
    :param ls:
    :param lw:
    :return:
    """
    # collecting data
    mbs = []
    jcs = []

    tmerg = o_data.get_par("tmerg_r")
    _, avaltimes, _ = o_data.get_3d_data("MJ_encl.txt")
    avaltimes = avaltimes * 1.e3
    for t in times:
        t = t + tmerg * 1.e3
        print("{} {:.1f}, {:.1f}, {:.1f}".format(o_data.sim, t, avaltimes.min(), avaltimes.max()))
        if t > avaltimes.min() and t < avaltimes.max():
            # mj = o_data.get_3d_data("MJ_encl.txt", t=t / 1.e3)
            # t = t + tmerg * 1.e3 # postmerger time is of interest
            mj = o_data.get_3d_data("MJ_encl.txt", t=t/1.e3)  # , it=2211840)
            mj = mj.T
            #
            rc, drc, Mb, Jb = mj[:, 0], mj[:, 1], mj[:, 2], mj[:, 3]
            #
            Jout_max = np.cumsum((Jb * rc * drc)[::-1])
            Mout = np.cumsum((Mb * rc * drc)[::-1])
            #
            if len(Jout_max)>0 and not np.isnan(Mout[-1]) and not np.isnan(Jout_max[-1]):
                mbs.append(Mout[-1])
                jcs.append(Jout_max[-1])
            #
            # print("t:{}".format(t), len(rc), len(drc), len(Mb), len(Jb), Mout[-1], Jout_max[-1])
        else:
            pass
            # print("t:{}".format(t))

    #
    return jcs, mbs




for model, xlim, ylim in plot_models:
    Mb_0 = sims.loc[model]["Mb"]
    J_0 = sims.loc[model]["Jfinal"]
    eos = sims.loc[model]["EOS"]
    #
    fig = plt.figure(figsize=[4, 2.5])
    ax = fig.add_axes([0.14, 0.15, 1.0 - 0.14 * 2, 0.95 - 0.15])
    # fig = plt.figure() # figsize=[4., 2.5] figsize=[4.2, 3.6]
    # ax = fig.add_axes([0.15, 0.15, 0.95 - 0.15, 0.95 - 0.15])#fig.add_axes() # [0.15, 0.15, 0.95 - 0.15, 0.95 - 0.15]
    #
    o_data = ADD_METHODS_ALL_PAR(model)
    mj = o_data.get_3d_data("MJ_encl.txt", t=-1)#, it=2211840)
    mj = mj.T
    #
    rc, drc, Mb, Jb = mj[:, 0], mj[:, 1], mj[:, 2], mj[:, 3]
    #
    Jout_max = np.cumsum((Jb * rc * drc)[::-1])
    Jout_fidu = np.cumsum((Jb * rc * options.Jfac_fidu(rc) * drc)[::-1])
    Mout = np.cumsum((Mb * rc * drc)[::-1])

    Mout_fidu = np.interp(Jout_max, Jout_fidu, Mout)

    color = _models_old.color_list[1]

    if model == "BLh_M13641364_M0_LK_SR":
        ax.plot(J_0, Mb_0, '*', mfc="gold", mec=color, markersize=10)
    else:
        ax.plot(J_0, Mb_0, 'o', color=color)

    #  ---------- evolution ----------------
    jevo, mbevo, =__plot_total_J_Mb_evol(ax, o_data, times=[10, 20, 30, 40, 50, 60, 70, 80, 90])
    ax.plot(jevo, mbevo, color="red", ls='-', lw=0.7, label='Evol. 3D data')
    ax.plot(jevo[0], mbevo[0], marker='x')
    # --------------------------------------

    ax.fill_between(J_0 - Jout_fidu, Mb_0 - Mout, Mb_0,
                    # np.minimum(Mb_0, seq[eos].M0_min_fit(J_0 - Jout)),
                    # where=Mb_0 - Mout <= 1.05*seq[eos].M0_min_fit(J_0 - Jout),
                    color=color, alpha=0.3, zorder=-2,
                    label="Disk ejecta")
    ax.fill_between(J_0 - Jout_max, Mb_0 - Mout, Mb_0 - Mout_fidu,
                    # np.minimum(Mb_0, seq[eos].M0_min_fit(J_0 - Jout)),
                    # where=Mb_0 - Mout <= 1.05*seq[eos].M0_min_fit(J_0 - Jout),
                    color=color, alpha=0.6, zorder=-1,
                    label="Remnant ejecta")

    ax.plot(J_0 - Jout_fidu, Mb_0 - Mout, color=color, zorder=-1)

    J = np.linspace(0, seq[eos]["Jmax"], 200)
    ax.fill_between(J, seq[eos]["M0_min_fit"](J), seq[eos]["M0_max_fit"](J),
                    edgecolor='black', facecolor='lightgrey',
                    label="RNS", zorder=0)

    ax.set_xlabel(r"$J\ [G\, c^{-1} M_\odot^2]$")
    ax.set_ylabel(r"$M_b\ [M_\odot]$")

    if sims.loc[model]["viscosity"] == "LK":
        ax.text(0.95, 0.05, r"{} -- $({} + {})\, M_\odot$ -- M0 -- LK".format(
            sims.loc[model]["EOS"], sims.loc[model]["M1"], sims.loc[model]["M2"]),
                ha="right", va="bottom", transform=ax.transAxes)
    else:
        ax.text(0.95, 0.05, r"{} -- $({} + {})\, M_\odot$ -- M0".format(
            sims.loc[model]["EOS"], sims.loc[model]["M1"], sims.loc[model]["M2"]),
                ha="right", va="bottom", transform=ax.transAxes)

    ax.legend(loc="lower right", bbox_to_anchor=(0.95, 0.15))
    ax.minorticks_on()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.tick_params(
        axis='both', which='both', labelleft=True,
        labelright=False, tick1On=True, tick2On=False,
        labelsize=int(12),
        direction='in',
        bottom=True, top=True, left=True, right=False
    )

    # -- for Grav. Mass axis

    ax2 = ax.twinx()
    ax2.set_ylim(ylim[0], ylim[-1])
    ax2.set_ylabel(r"$M\ [M_\odot]$")
    # ax2.axhline(y=2.5, color="black", linestyle="--")

    M0_ticks = np.linspace(ylim[0], ylim[-1], 6)
    # print(M0_ticks, )
    M_ticks = 2 * M_of_M0[eos](M0_ticks / 2.)
    M_labels = ["{:.2f}".format(m) for m in M_ticks]
    ax2.set_yticks(M0_ticks)
    ax2.set_yticklabels(M_labels)
    ax2.minorticks_on()
    ax2.tick_params(
        axis='both', which='both', labelleft=False,
        labelright=True, tick1On=False, tick2On=True,
        labelsize=int(12),
        direction='in',
        bottom=True, top=True, left=False, right=True
    )

    # plt.savefig(__outplotdir__ + "outcome_{}.pdf".format(model.lower()))
    plt.tight_layout()
    plt.savefig(__outplotdir__ + "outcome_{}.png".format(model.lower()), dpi=256)
    plt.close()

