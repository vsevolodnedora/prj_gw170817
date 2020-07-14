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

import sys
from scivis import units as ut
import os
import copy
import h5py
import csv
from scipy import interpolate
import scipy.optimize as opt # for least square method
sys.path.append('/data01/numrel/vsevolod.nedora/bns_ppr_tools/')
from preanalysis import LOAD_INIT_DATA
from outflowed import EJECTA_PARS
from preanalysis import LOAD_ITTIME
from plotting_methods import PLOT_MANY_TASKS
from profile import LOAD_PROFILE_XYXZ, LOAD_RES_CORR, LOAD_DENSITY_MODES
from utils import Paths, Lists, Labels, Constants, Printcolor, UTILS, Files, PHYSICS
from data import *
from tables import *
# from settings import simulations, old_simulations, resolutions

__outplotdir__ = "/data01/numrel/vsevolod.nedora/figs/all3/prj_viscous_ejecta/"

class Struct(object):
    pass

options = Struct()

# ---
sim = "BLh_M13641364_M0_LK_SR"#"DD2_M13641364_M0_LK_SR_R04"
list_of_eoss = ["BLh", "DD2"]
options.NJ  = 64    # Number of J=const sequences to construct
options.NM0 = 64    # Number of M0=const sequences to construct
options.dJ = 0.01   # Step size used to rootfind the viscous ejecta
options.Jmin = 3.5  # Minum J to search while rootfinding for the viscous ejecta mass
options.rej = 300.0 # Estimate the ejecta assuming that everything becomes unbound once it reaches a given radius
options.Jfac_fidu = lambda rc: (options.rej/(np.minimum(rc, options.rej)))**(1./2.)
# ---

tbl = GET_PAR_FROM_TABLE()
tbl.set_intable = Paths.output + "models3.csv"
tbl.load_table()

# --- read gw data ---

dic = tbl.get_simdic(sim)
dic["M"] = float(dic["M1"]) + float(dic["M2"])
dic["nu"] = float(dic["M1"]) * float(dic["M2"]) / (float(dic["M"]) ** 2)
#
dic["Eb"] = (float(dic["M"]) - float(dic["MADM"]) + float(dic["EGW"])) / (float(dic["M"]) * float(dic["nu"]))
dic["j"] = (float(dic["JADM"]) - float(dic["JGW"])) / (float(dic["M"]) ** 2 * float(dic["nu"]))
#
dic["Mfinal"] = float(dic["MADM"]) - float(dic["EGW"])  # EGW is the last point from the E_GW.dat file
dic["Jfinal"] = float(dic["JADM"]) - float(dic["JGW"])
dic["afinal"] = float(dic["Jfinal"]) / (float(dic["Mfinal"]) ** 2)


# --- read RNS data ---
names = ['EOS', 'ratio', 'rho_c', 'M', 'M_0', 'r_star', 'Omega', 'Omega_K', 'I_45', 'a']
rns = pd.read_table(Paths.rns + 'RNS.dat.gz', names=names, sep=' ', skiprows=1)
rns = rns.sort_values(['EOS', 'rho_c', 'M'])
rns['diff_Omega'] = np.abs(rns['Omega_K'] - rns['Omega'])
rns["J"] = rns["a"]*rns["M"]**2 # a - acceleration
rns["P"] = 2*np.pi/rns["Omega"]*1e3 # linear momentum
rns_eos = rns.EOS.unique()
# exit(1)
rns = rns.dropna() # drops all the entris of the dataframe that has NaN in any of the columns
# --- read TOVs ---
M_of_M0 = {}
for eos in list_of_eoss:
    # mind the units! New tables have cgs! Also! Mind the 0th column in old tables being N -- not rho
    rhoc, M, M0 = np.loadtxt(Paths.TOVs+"/{}_sequence.txt".format(eos), usecols=(0,1,2), unpack=True) # usecols=(1,2,3)
    imax = rhoc.argmax() # end of stability branch?
    M = M[:imax]
    M0 = M0[:imax]
    M_of_M0[eos] = interpolate.interp1d(M0, M)

# Bounding sequences in J-M0 plane
'''
   Here we create the mass shedding sequence, which binds the stable configurations from below, 
   and the maximum mass for a fixed J, which binds the stable configurations from above.
'''
seq = {}
for eos in rns_eos: #options.EOS:
    seq[eos] = {} # to append shit
    sel = rns[rns.EOS == eos] # RNS sequences for a given
    # print(sel.J) ***EQVIVALENT***
    # print(sel["J"]); exit(1)
    print("{} {}".format(eos, len(sel)))
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
    #
for eos in rns_eos:  # options.EOS:
    # for every EOS set selected data where J = 0,
    sel = rns[(rns.EOS == eos) & (rns.J == 0)] # sel are the models with J = 0
    seq[eos]["M0_TOV"] = sel["M_0"].max() # set M0_TOV to the maximum M of a J = 0 models

'''
    Unknown piece of code | M0 MAX FIT
'''


# fitting function
def M0_max_fitting_function(x, J):
    a,b,c,d = x
    return a*J**3 + b*J**2 + c*J + d

# fitting function
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

for eos in rns_eos: # options.EOS:
    iopt = np.ones_like(seq[eos]["Jc"], dtype=np.bool)
    iopt[1] = False
    iopt = seq[eos]["Jc"] > 2.0
    seq[eos]["M0_max_fit_res"] = opt.least_squares(M0_max_residuals, M0_max_init_guess(),
                                            args=(seq[eos]["Jc"][iopt], seq[eos]["M0_max"][iopt]))
    seq[eos]["M0_max_fit"] = M0_max_closure(seq[eos]["M0_max_fit_res"].x)
    print("{} {} {}".format(eos, len(seq[eos]["Jc"][iopt]),
    100*np.max(np.abs((seq[eos]["M0_max_fit"](seq[eos]["Jc"][iopt]) - seq[eos]["M0_max"][iopt])/
                 seq[eos]["M0_max"][iopt]))))

'''
    Unknown piece of code | M0 MIN FIT
'''

# fitting function
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

for eos in rns_eos: #options.EOS:
    iopt = np.ones_like(seq[eos]["Jc"], dtype=np.bool)
    iopt = seq[eos]["Jc"] > 2.0
    seq[eos]["M0_min_fit_res"] = opt.least_squares(M0_min_residuals, M0_min_init_guess(),
                                            args=(seq[eos]["Jc"][iopt], seq[eos]["M0_min"][iopt]))
    seq[eos]["M0_min_fit"] = M0_min_closure(seq[eos]["M0_min_fit_res"].x)
    print("{} {} {}".format(eos, len(seq[eos]["Jc"][iopt]),
        100*np.max(np.abs((seq[eos]["M0_min_fit"](seq[eos]["Jc"][iopt]) - seq[eos]["M0_min"][iopt])/
                     seq[eos]["M0_min"][iopt]))))

'''
    Unknown piece of code | M MIN FIT
'''

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

for eos in rns_eos: #options.EOS:
    iopt = np.ones_like(seq[eos]["Jc"], dtype=np.bool)
    #iopt = seq[eos].Jc > 2.0
    seq[eos]["M_min_fit_res"] = opt.least_squares(M_min_residuals, M_min_init_guess(),
                                            args=(seq[eos]["M0_min"][iopt], seq[eos]["M"][iopt]))
    seq[eos]["M_min_fit"] = M_min_closure(seq[eos]["M_min_fit_res"].x)

'''
    Unknown piece of code | P0 FIT
'''

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

for eos in rns_eos: #options.EOS:
    iopt = np.ones_like(seq[eos]["Jc"], dtype=np.bool)
    iopt = (seq[eos]["M0_min"] > 2.4) & (seq[eos]["M0_min"] < 2.6)
    seq[eos]["P0_fit_res"] = opt.least_squares(P0_residuals, P0_init_guess(),
                                    args=(seq[eos]["M0_min"][iopt], seq[eos]["P0"][iopt]))
    seq[eos]["P0_fit"] = P0_closure(seq[eos]["P0_fit_res"].x)
    iopt = (seq[eos]["M0_min"] > 2.0)
    seq[eos]["P0_fit_err"] = np.max(np.abs((seq[eos]["P0_fit"](seq[eos]["M0_min"][iopt]) -
                                         seq[eos]["P0"][iopt])))

for eos in rns_eos: #options.EOS:
    Jmax = seq[eos]["Jf"][-1]
    f = lambda J: (seq[eos]["M0_max_fit"](J) - seq[eos]["M0_min_fit"](J))**2
    res = opt.minimize_scalar(f, bracket=(0.75*Jmax, 1.25*Jmax))
    seq[eos]["Jmax"] = res.x
    seq[eos]["M0_sup"] = seq[eos]["M0_max_fit"](res.x)

'''
    MERGER OUTCOME
'''
# SKIPPED

'''
    GW timescale
'''
L = []
o_data = ADD_METHODS_ALL_PAR(sim)
tcoll = tbl.get_par(sim, "tcoll_gw")
if np.isinf(tcoll):
    L.append(0)
    l = 0.
else:
    table = o_data.get_gw_data("EJ.dat").T
    t, Jdot, JGW = table[:, 0], table[:, 3], table[:, 5]
    # t, Jdot, JGW = np.loadtxt(models.get_path(m.name) + "/waveforms/EJ.dat",
    #                           usecols=(0, 3, 4), unpack=True)
    t = ut.conv_time(ut.cactus, ut.cgs, t) * 1e3
    tGW = ut.conv_time(ut.cactus, ut.cgs, (float(dic["JADM"]) - JGW) / Jdot) * 1e3
    idx = (t > t.max() - 1.5) & (t < t.max() - 0.5)
    l = np.mean(tGW[idx])
    L.append(l)
dic["tGW"] = l

    # print("sim: {} tGW:{}".format(sim, l.argmin()))
'''
    Viscous ejecta
'''
def step_to_zero(fun, xstart=1., xmin=0., dx=0.1):
    xp = xstart
    while fun(xp) >= 0 and xp >= xmin:
        xp -= dx
    return xp



# extract data
o_data = ADD_METHODS_ALL_PAR(sim)
Jfinal = dic["Jfinal"]  # after the GW were taken away from Initial JADM
Mb = float(dic["Mb"])
eos = dic["EOS"]
mjenclosed = o_data.get_3d_data("MJ_encl.txt")
iters, times, mjenclosed = mjenclosed[0], mjenclosed[1], mjenclosed[2]
mj = mjenclosed[-1]  # 2D array [1:rcyl 2:drcyl 3:M 4:J 5:I]
rc, drc, M, J = mj[0, :], mj[1, :], mj[2, :], mj[3, :]

# Summ the data rc, drc, M, J = np.loadtxt(fname, unpack=True)
Jf_max = Jfinal - np.cumsum((J * rc * drc)[::-1])
Jf_fidu = Jfinal - np.cumsum((J * rc * options.Jfac_fidu(rc) * drc)[::-1])
Mf = Mb - np.cumsum((M * rc * drc)[::-1])

# print(name, model.Jfinal, abs(Jf_max[-1])/model.Jfinal)

Mf_fun = interpolate.interp1d(Jf_max[::-1], Mf[::-1], kind="linear", assume_sorted=False)
fun = lambda Jf: seq[eos]["M0_min_fit"](Jf) - Mf_fun(Jf)
x = step_to_zero(fun, xstart=Jf_max.max(), dx=options.dJ, xmin=options.Jmin)
dM_max = Mb - Mf_fun(x)
dJ_max = Jfinal - x

Mf_fun = interpolate.interp1d(Jf_fidu[::-1], Mf[::-1], kind="linear", assume_sorted=False)
fun = lambda Jf: seq[eos]["M0_min_fit"](Jf) - Mf_fun(Jf)
x = step_to_zero(fun, xstart=Jf_fidu.max(), dx=options.dJ, xmin=options.Jmin)
dM_fidu = Mb - Mf_fun(x)
dJ_fidu = Jfinal - x

dic = tbl.get_simdic(sim)
dic["dM_0_max"] = dM_max
dic["dJ_rns_max"] = dJ_max
dic["dM_0_fidu"] = dM_fidu
dic["dJ_rns_fidu"] = dJ_fidu

'''
    Error Estimation
'''

# sel = (sims.outcome == MergerOutcome.MNS) | (sims.outcome == MergerOutcome.SMNS)
# sel = sel & sims.MdiskPP.notnull()

# hr = []
# for name, m in sims[sel].iterrows():
#     if re.match(".*_HR", name):
#         c = m.copy()
#         c.name = name.replace("_HR", "")
#         hr.append(c)
# hr = pd.DataFrame(hr, columns=sims.columns)

# for n, m in hr.iterrows():
#     rel_err = np.abs(sims.loc[n]["dM_0_max"] - m["dM_0_max"]) / sims.loc[n]["dM_0_max"]
#     print((n, rel_err))
#
# print(("-" * 60))

# for n, m in hr.iterrows():
#     rel_err = np.abs(sims.loc[n]["dM_0_fidu"] - m["dM_0_fidu"]) / sims.loc[n]["dM_0_fidu"]
#     print((n, rel_err))

# print sims["dM_0_max"][(models.fiducial &
#         ((sims.outcome == MergerOutcome.MNS) | (sims.outcome == MergerOutcome.SMNS))) |
#         (sims.label == "DD2_M135135_M0")]

# print sims["dM_0_fidu"][(models.fiducial &
#         ((sims.outcome == MergerOutcome.MNS) | (sims.outcome == MergerOutcome.SMNS))) |
#         (sims.label == "DD2_M135135_M0")]

# print sims.loc["DD2_M135135_M0"]["dM_0_max"], sims.loc["DD2_M135135_M0"]["dM_0_fidu"]

'''
    PLOTS [SINGLE STARS]
'''

for eos in list_of_eoss:
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

    plt.savefig(__outplotdir__ + "rns_seq_{}.pdf".format(eos.lower()))
    plt.close()

for eos in list_of_eoss:
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

    plt.savefig(__outplotdir__ + "mass_shedding_mass_{}.pdf".format(eos.lower()))
    plt.close()

fig = plt.figure(figsize=[4, 2.5])
ax = fig.add_axes([0.15, 0.15, 0.95 - 0.15, 0.95 - 0.15])

for idx, eos in enumerate(list_of_eoss):
    ax.plot(seq[eos]["M0_min"], seq[eos]["P0"], color="blue",
            label=eos)

ax.set_xlim(xmin=1.5, xmax=3.5)
ax.set_ylim(ymin=0.5, ymax=1.25)

ax.set_ylabel(r"$P_0\ [{\rm ms}]$")
ax.set_xlabel(r"$M_b\ [M_\odot]$")

ax.legend(loc="upper right", ncol=2)

plt.savefig(__outplotdir__ + "mass_shedding_spin.pdf")
plt.close()

for eos in rns_eos:
    fig = plt.figure(figsize=[4, 2.5])
    ax = fig.add_axes([0.15, 0.15, 0.95 - 0.15, 0.95 - 0.15])

    ax.plot(seq[eos]["M0_min"], seq[eos]["P0"], "o")
    ax.plot(seq[eos]["M0_min"], seq[eos]["P0_fit"](seq[eos]["M0_min"]), "-")

    ax.set_xlim(xmin=2.0, xmax=3.)
    ax.set_ylim(ymin=0.5, ymax=2.0)

    ax.set_ylabel(r"$P_0\ [{\rm ms}]$")
    ax.set_xlabel(r"$M_b\ [M_\odot]$")

    plt.savefig(__outplotdir__ + "mass_shedding_spin_{}.pdf".format(eos.lower()))
    plt.close()

ofile = open(__outplotdir__ + "mass_shedding_spin.txt", "w")
ofile.write("EOS a b e\n")
for eos in rns_eos:

    ofile.write("{} {} {} {}\n".format(eos, seq[eos]["P0_fit_res"].x[0], seq[eos]["P0_fit_res"].x[1],
                                       seq[eos]["P0_fit_err"]))
del ofile

for eos in rns_eos:
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

'''
    PLOTS [BINARIES]
'''

for eos in list_of_eoss:
    fig = plt.figure(figsize=[4, 2.5])
    ax = fig.add_axes([0.14, 0.15, 1.0 - 0.14 * 2, 0.95 - 0.15])

    J = np.linspace(0, seq[eos]["Jmax"], 100)

    ax.fill_between(J, seq[eos]["M0_min_fit"](J), seq[eos]["M0_max_fit"](J),
                    edgecolor='black', facecolor='lightgrey',
                    label="RNS")

    # for idx, outcome in enumerate(MergerOutcomes):
    #     sel = sims[(sims.EOS == eos) & (sims.outcome == outcome) & models.fiducial]
    #     ax.plot(sel.Jfinal, sel.Mb, models.marker_list[idx],
    #             color=models.color_list[idx], label=outcome)
    # if eos == "DD2":
    #     mod = sims.loc["DD2_M135135_M0"]
    ax.plot(float(dic["Jfinal"]), float(dic["Mb"]), "*", color="gold", label=sim.replace("_","\_")+r" -- M0")

    # ax.axhline(y=seq[eos].M0_TOV, color="black", linestyle=":", label=r"$M_{\rm TOV}$")
    # ax.axhline(y=seq[eos].M0_sup, color="black", linestyle="--", label=r"$M_{\rm SMNS}$")

    ax.set_xlim(3, 9)
    M0_min = 2.5
    M0_max = 4.0
    ax.set_ylim(M0_min, M0_max)

    ax.set_xlabel(r"$J\ [G\, c^{-1} M_\odot^2]$")
    ax.set_ylabel(r"$M_b\ [M_\odot]$")

    han, lab = ax.get_legend_handles_labels()
    if eos == "BHBlp":
        ax.add_artist(ax.legend(han[:-1], lab[:-1], loc="upper left"))
        ax.add_artist(ax.legend([han[-1]], [lab[-1]], loc="lower right"))
    elif eos == "DD2":
        ax.add_artist(ax.legend([han[-2]], [lab[-2]], loc="lower right",
                                handletextpad=0.05))
    ax.text(0.5, 0.95, eos, ha="center", va="top", transform=ax.transAxes, fontsize="large")

    ax2 = ax.twinx()
    ax2.set_ylim(M0_min, M0_max)
    ax2.set_ylabel(r"$M\ [M_\odot]$")
    # ax2.axhline(y=2.5, color="black", linestyle="--")

    M0_ticks = np.linspace(M0_min, M0_max, 7)
    M_ticks = 2 * M_of_M0[eos](M0_ticks / 2.)
    M_labels = ["{:.2f}".format(m) for m in M_ticks]
    ax2.set_yticks(M0_ticks)
    ax2.set_yticklabels(M_labels)

    plt.savefig(__outplotdir__ + "outcome_{}.pdf".format(eos.lower()))
    plt.close()

# ofile = open("outcome.txt", "w")
# for name, m in sims.iterrows():
#     ofile.write("{} {}\n".format(name, m["outcome"]))
# ofile.close()

# fig = plt.figure(figsize=[4,2.5])
# ax = fig.add_axes([0.15, 0.15, 0.95-0.15, 0.95-0.15])
#
# for ieos, eos in enumerate(models.EOS):
#     sel = sims[(sims.EOS == eos) & models.fiducial & (
#         (sims.outcome == MergerOutcome.SMNS) |
#         (sims.outcome == MergerOutcome.MNS))]
#     dM = sel.Mfinal - seq[eos].M_min_fit(sel.Mb)
#     norm_Mb = np.array(sel["Mb"])/float(seq[eos].M0_sup)
#     ax.scatter(norm_Mb, dM,
#                color=models.color_list[ieos],
#                marker=models.marker_list[ieos],
#                label=models.print_eos(eos))
# mod = sims.loc["DD2_M135135_M0"]
# ax.plot(mod.Mb/seq["DD2"].M0_sup, mod.Mfinal - seq["DD2"].M_min_fit(mod.Mb),
#         "*", color="gold", label=r"DD2 -- $(1.35+1.35)M_\odot$ -- M0")
#
# han, lab = ax.get_legend_handles_labels()
# ax.add_artist(ax.legend(han[1:], lab[1:], loc="upper left", ncol=2))
# ax.add_artist(ax.legend([han[0]], [lab[0]], loc="lower left"))
#
# #ax.set_ylim(0.1, 0.2)
#
# ax.set_xlabel(r"$M_b/M_{\rm RNS}$")
# ax.set_ylabel(r"$\Delta M\ [M_\odot]$")
#
# plt.savefig(options.prefix + "outcome_mass.pdf")
# plt.close()


'''
    skipped
'''

plot_models = [
    ("DD2_M135135_LK", (3.5, 6), (2.7, 3)),
    ("DD2_M135135_M0", (3.5, 6), (2.7, 3)),
    ("SFHo_M1251365_LK", (3.5, 5), (2.7, 3)),
    ("SFHo_M140120_M0", (3.5, 5), (2.7, 3)),
    ("SFHo_M140120_LK", (3.5, 5), (2.7, 3)),
]

for model, xlim, ylim in plot_models:
    Mb_0 = sims.loc[model]["Mb"]
    J_0 = sims.loc[model]["Jfinal"]
    eos = sims.loc[model]["EOS"]

    fig = plt.figure(figsize=[4, 2.5])
    ax = fig.add_axes([0.15, 0.15, 0.95 - 0.15, 0.95 - 0.15])

    rc, drc, Mb, Jb = np.loadtxt("../../../data/MJ_encl/{}.txt".format(model), unpack=True)

    Jout_max = np.cumsum((Jb * rc * drc)[::-1])
    Jout_fidu = np.cumsum((Jb * rc * options.Jfac_fidu(rc) * drc)[::-1])
    Mout = np.cumsum((Mb * rc * drc)[::-1])

    Mout_fidu = np.interp(Jout_max, Jout_fidu, Mout)

    color = models.color_list[1]
    if model == "DD2_M135135_M0":
        ax.plot(J_0, Mb_0, '*', mfc="gold", mec=color, markersize=10)
    else:
        ax.plot(J_0, Mb_0, 'o', color=color)
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

    J = np.linspace(0, seq[eos].Jmax, 200)
    ax.fill_between(J, seq[eos].M0_min_fit(J), seq[eos].M0_max_fit(J),
                    edgecolor='black', facecolor='lightgrey',
                    label="RNS", zorder=0)

    ax.set_xlabel(r"$J\ [G\, c^{-1} M_\odot^2]$")
    ax.set_ylabel(r"$M_b\ [M_\odot]$")

    if model == "DD2_M135135_M0":
        ax.text(0.95, 0.05, r"{} -- $({} + {})\, M_\odot$ -- M0".format(
            sims.loc[model]["EOS"], sims.loc[model]["M1"], sims.loc[model]["M2"]),
                ha="right", va="bottom", transform=ax.transAxes)
    else:
        ax.text(0.95, 0.05, r"{} -- $({} + {})\, M_\odot$".format(
            sims.loc[model]["EOS"], sims.loc[model]["M1"], sims.loc[model]["M2"]),
                ha="right", va="bottom", transform=ax.transAxes)

    ax.legend(loc="lower right", bbox_to_anchor=(0.95, 0.15))

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plt.savefig(options.prefix + "outcome_{}.pdf".format(model.lower()))
    plt.close()

fig, axes = plt.subplots(2, 1, figsize=[4,4], sharex=True)
fig.subplots_adjust(left=0.15, bottom=0.1, top=0.98, right=0.98, hspace=0.1)

# for ieos, eos in enumerate(list_of_eoss):
#     sel = sims[(sims.EOS == eos) & models.with_disk_mass]
#     axes[0].scatter(sel["Mb"]/seq[eos].M0_sup, sel["dM_0_max"],
#            color=models.color_list[ieos], marker=models.marker_list[ieos],
#            label=models.print_eos(eos))
#     axes[1].scatter(sel["Mb"]/seq[eos].M0_sup, sel["dM_0_fidu"],
#                color=models.color_list[ieos], marker=models.marker_list[ieos],
#                label=models.print_eos(eos))
# mod = sims.loc["DD2_M135135_M0"]

print(dic["dM_0_fidu"], dic["dM_0_max"])

axes[1].plot(float(dic["Mb"])/seq["DD2"]["M0_sup"], dic["dM_0_fidu"], "*",
        color="gold", label=r"DD2 -- $(1.35+1.35)M_\odot$ -- M0")
axes[0].plot(float(dic["Mb"])/seq["DD2"]["M0_sup"], dic["dM_0_max"], "*",
        color="gold", label=r"DD2 -- $(1.35+1.35)M_\odot$ -- M0")

han, lab = axes[0].get_legend_handles_labels()
axes[0].add_artist(axes[0].legend(han[1:], lab[1:], loc="lower left", ncol=2))
axes[0].add_artist(axes[0].legend([han[0]], [lab[0]], loc="upper left"))

axes[0].set_ylabel(r"$M_{\rm ej}^{\max}\ [M_\odot]$")

axes[1].set_xlabel(r"$M_b/M_{\rm RNS}$")
axes[1].set_ylabel(r"$M_{\rm ej}^{\rm disk}\ [M_\odot]$")

axes[0].set_ylim(0, 0.50)
axes[1].set_ylim(0, 1.10)

plt.savefig(__outplotdir__ + "Mej_visc.pdf")
plt.close()