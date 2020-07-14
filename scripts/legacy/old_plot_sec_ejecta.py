from __future__ import division

import numpy as np
import pandas as pd
import sys
import os
import copy
import h5py
import csv
from scipy import interpolate

import matplotlib

import uutils

matplotlib.use('agg')
import matplotlib.pyplot as plt
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

from matplotlib.colors import LogNorm, Normalize

from data import ADD_METHODS_ALL_PAR
from comparison import TWO_SIMS, THREE_SIMS
from settings import resolutions

from model_sets import models as md
import rns as rns
import tov as tov
simulations = md.simulations_nonblacklisted

__outplotdir__ = "/data01/numrel/vsevolod.nedora/figs/all3/plot_sec_ej/"

#




def load_J_Mb_evol(o_data, times, options_Jfac_fidu, cumsum=True):
    #
    ts, rc, mbs, mbsfedu, jcs, jcsfedu, jfcs = [], [], [], [], [], [], []
    #
    tmerg = o_data.get_par("tmerg_r")
    _, avaltimes, _ = o_data.get_3d_data("MJ_encl.txt")
    avaltimes = avaltimes * 1.e3
    for t in times:
        t = t + tmerg * 1.e3
        print("{} {:.1f}, {:.1f}, {:.1f}".format(o_data.sim, t, avaltimes.min(), avaltimes.max()))
        if t > avaltimes.min() and t <= avaltimes.max():
            # mj = o_data.get_3d_data("MJ_encl.txt", t=t / 1.e3)
            # t = t + tmerg * 1.e3 # postmerger time is of interest
            mj = o_data.get_3d_data("MJ_encl.txt", t=t / 1.e3)  # , it=2211840)
            mj = mj.T
            #
            rc, drc, Mb, Jb, Jfb, I = mj[:, 0], mj[:, 1], mj[:, 2], mj[:, 3], mj[:, 4], mj[:, 5]
            #

            # print("rc:")
            # print(rc)
            # print("drc")
            # print(drc)
            # exit(1)

            if cumsum:
                Jout_max = np.cumsum((Jb * rc * drc)[::-1])
                # Jfbout_max = np.cumsum((Jfb * rc * drc)[::-1])
                Jfbout_max = np.array(Jfb * drc)[::-1] #* 2*np.pi rc * 512. # \int dz = 512.
                Jout_fidu = np.cumsum((Jb * rc * options_Jfac_fidu(rc) * drc)[::-1])
                Mout = np.cumsum((Mb * rc * drc)[::-1])
                Mout_fidu = np.interp(Jout_max, Jout_fidu, Mout)
            else:
                Jout_max = np.array((Jb * rc * drc)[::-1])
                # Jfbout_max = np.cumsum((Jfb * rc * drc)[::-1])
                Jfbout_max = np.array(Jfb * drc)[::-1]  # * 2*np.pi rc * 512. # \int dz = 512.
                Jout_fidu = np.array((Jb * rc * options_Jfac_fidu(rc) * drc)[::-1])
                Mout = np.array((Mb * rc * drc)[::-1])
                Mout_fidu = np.interp(Jout_max, Jout_fidu, Mout)
            #
            if len(Jout_max) > 0 and not np.isnan(Mout[-1]) and not np.isnan(Jout_max[-1]):
                ts.append(t)
                mbs.append(Mout)
                mbsfedu.append(Mout_fidu)
                jcs.append(Jout_max)
                jfcs.append(Jfbout_max)
                jcsfedu.append(Jout_fidu)
                # print("\t {}, {}, {}, {}".format(t, len(mbs), len(mbsfedu), len(jcs), len(jcsfedu)))
        else:
            pass
    ts = np.array(ts)
    assert len(rc) > 0
    assert len(ts) > 0
    # assert len(ts) == len(rc)
    assert len(ts) == len(mbs)
    assert len(ts) == len(mbsfedu)
    assert len(ts) == len(jcs)
    assert len(ts) == len(jcsfedu)
    assert len(ts) == len(jfcs)

    return ts, rc[::-1], mbs, mbsfedu, jcs, jcsfedu, jfcs

def step_to_zero(fun, xstart=1, xmin=0, dx=0.1):
    xp = xstart
    while fun(xp) >= 0 and xp >= xmin:
        xp -= dx
    return xp

def add_dm_dj(o_data, model, seq):

    mj = o_data.get_3d_data("MJ_encl.txt", t=-1)  # , it=2211840)
    mj = mj.T

    rc, drc, Mb, Jb = mj[:, 0], mj[:, 1], mj[:, 2], mj[:, 3]
    Jf_max = model.Jfinal - np.cumsum((Jb * rc * drc)[::-1])
    Jf_fidu = model.Jfinal - np.cumsum((Jb * rc * rns.options.Jfac_fidu(rc) * drc)[::-1])
    Mf = model.Mb - np.cumsum((Mb * rc * drc)[::-1])

    #

    Mf_fun = interpolate.interp1d(Jf_max[::-1], Mf[::-1], kind="linear", assume_sorted=False)
    fun = lambda Jf: seq[model.EOS]["M0_min_fit"](Jf) - Mf_fun(Jf)
    x = step_to_zero(fun, xstart=Jf_max.max(), dx=rns.options.dJ, xmin=rns.options.Jmin)
    model["dM_0_max"] = model.Mb - Mf_fun(x)
    model["dJ_rns_max"] = model.Jfinal - x

    #

    Mf_fun = interpolate.interp1d(Jf_fidu[::-1], Mf[::-1], kind="linear", assume_sorted=False)
    fun = lambda Jf: seq[model.EOS]["M0_min_fit"](Jf) - Mf_fun(Jf)
    x = step_to_zero(fun, xstart=Jf_fidu.max(), dx=rns.options.dJ, xmin=rns.options.Jmin)
    model["dM_0_fidu"] = model.Mb - Mf_fun(x)
    model["dJ_rns_fidu"] = model.Jfinal - x

    return model



# def load_J_Mb_evol(o_data, times, options_Jfac_fidu):
#     """
#
#     :param ax:
#     :param o_data:
#     :param times:  list of postmerger times in [ms]
#     :param color:
#     :param ls:
#     :param lw:
#     :return:
#     """
#     # collecting data
#     mbs = []
#     jcs = []
#     jcsfedu = []
#     #
#     mb_shells_end = []
#     jmax_shells_end = []
#     jfedu_shells_end = []
#     #
#     tmerg = o_data.get_par("tmerg_r")
#     _, avaltimes, _ = o_data.get_3d_data("MJ_encl.txt")
#     avaltimes = avaltimes * 1.e3
#     for t in times:
#         t = t + tmerg * 1.e3
#         print("{} {:.1f}, {:.1f}, {:.1f}".format(o_data.sim, t, avaltimes.min(), avaltimes.max()))
#         if t > avaltimes.min() and t < avaltimes.max():
#             # mj = o_data.get_3d_data("MJ_encl.txt", t=t / 1.e3)
#             # t = t + tmerg * 1.e3 # postmerger time is of interest
#             mj = o_data.get_3d_data("MJ_encl.txt", t=t/1.e3)  # , it=2211840)
#             mj = mj.T
#             #
#             rc, drc, Mb, Jb = mj[:, 0], mj[:, 1], mj[:, 2], mj[:, 3]
#             #
#             Jout_max = np.cumsum((Jb * rc * drc)[::-1])
#             Jout_fidu = np.cumsum((Jb * rc * options_Jfac_fidu(rc) * drc)[::-1])
#             Mout = np.cumsum((Mb * rc * drc)[::-1])
#             #
#             if len(Jout_max)>0 and not np.isnan(Mout[-1]) and not np.isnan(Jout_max[-1]):
#                 mbs.append(Mout[-1])
#                 jcs.append(Jout_max[-1])
#                 jcsfedu.append(Jout_fidu[-1])
#             #
#
#             #
#             # print("t:{}".format(t), len(rc), len(drc), len(Mb), len(Jb), Mout[-1], Jout_max[-1])
#         else:
#             pass
#     #
#     mj = o_data.get_3d_data("MJ_encl.txt", t=-1) # end time
#     mj = mj.T
#     rc, drc, Mb, Jb = mj[:, 0], mj[:, 1], mj[:, 2], mj[:, 3]
#     Jout_max = np.cumsum((Jb * rc * drc)[::-1])
#     Jout_fidu = np.cumsum((Jb * rc * options_Jfac_fidu(rc) * drc)[::-1])
#     Mout = np.cumsum((Mb * rc * drc)[::-1])
#     Mout_fidu = np.interp(Jout_max, Jout_fidu, Mout) # = f(Mb)
#     #
#
#
#     #
#     return jcs, mbs, Jout_fidu, Mout

def plot_rns_mb_j_and_evold():

    sim = "BLh_M13641364_M0_LK_SR"
    plotdic = {
        "xmin":3.5, "xmax":5.5,
        "ymin":2.7, "ymax":3.1
    }

    #
    Mb_0 = simulations.loc[sim]["Mb"]
    J_0 = simulations.loc[sim]["Jfinal"]
    eos = simulations.loc[sim]["EOS"]
    #
    # sim = "BLh_M13641364_M0_LK_SR"  # "DD2_M13641364_M0_LK_SR_R04"
    # options_EOS = ["BLh", "DD2", "LS220", "SFHo", "SLy4"]
    # options_NJ = 64  # Number of J=const sequences to construct
    # options_NM0 = 64  # Number of M0=const sequences to construct
    # options_dJ = 0.01  # Step size used to rootfind the viscous ejecta
    # options_Jmin = 3.5  # Minum J to search while rootfinding for the viscous ejecta mass
    options_rej = 300.0  # Estimate the ejecta assuming that everything becomes unbound once it reaches a given radius
    options_Jfac_fidu = lambda rc: (options_rej / (np.minimum(rc, options_rej))) ** (1. / 2.)
    # #
    o_data = ADD_METHODS_ALL_PAR(sim)
    # mj = o_data.get_3d_data("MJ_encl.txt", t=-1)  # , it=2211840)
    # mj = mj.T
    # rc, drc, Mb, Jb = mj[:, 0], mj[:, 1], mj[:, 2], mj[:, 3]
    # Jout_max = np.cumsum((Jb * rc * drc)[::-1])
    # Jout_fidu = np.cumsum((Jb * rc * options_Jfac_fidu(rc) * drc)[::-1])
    # Mout = np.cumsum((Mb * rc * drc)[::-1])
    # Mout_fidu = np.interp(Jout_max, Jout_fidu, Mout) # = f(Mb)
    #

    # //// PLOT
    # ax.plot(J_0, Mb_0, '*', mfc="gold", mec=color, markersize=10) # End of the Simulation Position

    fig = plt.figure(figsize=[4, 2.5])
    ax = fig.add_axes([0.14, 0.15, 1.0 - 0.14 * 2, 0.95 - 0.15])

    mk = md.get_marker_for_outcome(rns.get_merger_outcome(o_data.get_par("tcoll"),
                                                          o_data.get_initial_data_par("Mb"),
                                                          o_data.get_initial_data_par("EOS")))
    ax.plot(J_0, Mb_0, mk, mfc="gold", mec='black', markersize=8)

    #  ---------- 3D data ----------------
    ts, rc, mbs, mbsfedu, jcs, jcsfedu, jfcs = load_J_Mb_evol(o_data, times=[10, 20, 30, 40, 50, 60, 70, 80, 90],
                                                  options_Jfac_fidu=options_Jfac_fidu)

    #  ---------- GW - 3D = Evolution ----------------

    print("")

    print("J_0 - jcsfedu[-1]: {}".format(J_0 - jcsfedu[-1]))
    print("J_0 - jcs[-1]    : {}".format(J_0 - jcs[-1]))
    print("Mb_0 - mbs[-1]     {}".format(Mb_0 - mbs[-1]))
    print("Mb_0 - mbsfedu[-1] {}".format(Mb_0 - mbsfedu[-1]))

    ax.fill_between(J_0 - jcsfedu[-1], Mb_0 - mbs[-1], Mb_0,
                    # np.minimum(Mb_0, seq[eos].M0_min_fit(J_0 - Jout)),
                    # where=Mb_0 - Mout <= 1.05*seq[eos].M0_min_fit(J_0 - Jout),
                    color="deepskyblue", alpha=0.3, zorder=-2,
                    label="Disk ejecta")
    ax.fill_between(J_0 - jcs[-1], Mb_0 - mbs[-1], Mb_0 - mbsfedu[-1],
                    # np.minimum(Mb_0, seq[eos].M0_min_fit(J_0 - Jout)),
                    # where=Mb_0 - Mout <= 1.05*seq[eos].M0_min_fit(J_0 - Jout),
                    color="cornflowerblue", alpha=0.6, zorder=-1,
                    label="Remnant ejecta")

    #  ---------- evolution in 3D ----------------

    jevo = [j[-1] for j in jcs]
    mbevo = [mb[-1] for mb in mbs]
    jfevo = [jfcs[-1] for jf in jfcs]
    #
    ax.plot(jevo, mbevo, color="black", marker='.', markersize=3, lw=0, label='Evol. 3D data')
    func = interpolate.interp1d(jevo, mbevo, kind="linear", fill_value="extrapolate")
    tmp_j = np.array([4.1, 4.2, 4.3, 4.4, 4.5])
    tmp_m = func(tmp_j)
    ax.plot(tmp_j, tmp_m, color="red", marker='x', markersize=3, lw=0, label='Ext. Evolution')
    # ax.annotate('t:{:.1f}'.format(ts[-1]), xy=(jevo[-1], mbevo[-1]), xycoords='data')
    ax.annotate('t:{:.1f}'.format(ts[-1]),
                xy=(jevo[-1], mbevo[-1]), xycoords='data',
                xytext=(-15, 25), textcoords='offset points',
                arrowprops=dict(facecolor='black', shrink=0.01),
                horizontalalignment='right', verticalalignment='bottom')
    # ax.plot(jevo[0], mbevo[0], marker='x')




    # --------------------------------------------


    # Plot RNS Sequence
    rns.compute_bounding_sequences_in_J_M0_for_EOS(eos)
    rns.get_rns_M0_min_fit(eos)
    rns.get_rns_M0_max_fit(eos)
    rns.get_Jmax_and_M0_sup(eos)
    J = np.linspace(0, rns.seq[eos]["Jmax"], 200)
    ax.fill_between(J, rns.seq[eos]["M0_min_fit"](J), rns.seq[eos]["M0_max_fit"](J),
                    edgecolor='black', facecolor='lightgrey',
                    label="RNS", zorder=0)

    # label
    ax.set_xlabel(r"$J\ [G\, c^{-1} M_\odot^2]$")
    ax.set_ylabel(r"$M_b\ [M_\odot]$")

    # limits
    ax.set_xlim(plotdic["xmin"], plotdic["xmax"])
    ax.set_ylim(plotdic["ymin"], plotdic["ymax"])

    # legend
    # ax.legend(loc="lower right", bbox_to_anchor=(0.95, 0.15))
    ax.legend(fancybox=True, loc='lower right',
               # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
               shadow=False, ncol=1, fontsize=9,
               framealpha=0., borderaxespad=0.)
    # ticks
    ax.minorticks_on()
    ax.tick_params(
        axis='both', which='both', labelleft=True,
        labelright=False, tick1On=True, tick2On=False,
        labelsize=int(12),
        direction='in',
        bottom=True, top=True, left=True, right=False
    )

    # for Mg
    ax2 = ax.twinx()
    ax2.set_ylim(plotdic["ymin"], plotdic["ymax"])
    ax2.set_ylabel(r"$M\ [M_\odot]$")
    # ax2.axhline(y=2.5, color="black", linestyle="--")

    M0_ticks = np.linspace(plotdic["ymin"], plotdic["ymax"], 6)

    # convert Mb -> M using TOV
    tov.load_tov_get_M_of_M0(eos)
    M_ticks = 2 * tov.M_of_M0[eos](M0_ticks / 2.)
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
    print(__outplotdir__ + "outcome_{}.png".format(sim.lower()))
    plt.savefig(__outplotdir__ + "outcome_{}.png".format(sim.lower()), dpi=256)
    plt.close()

def plot_max_sec_ej_mass():

    sim = "BLh_M13641364_M0_LK_SR"

    #

    fig = plt.figure(figsize=[4, 2.5])
    ax = fig.add_axes([0.15, 0.15, 0.95 - 0.15, 0.95 - 0.15])

    # for ieos, eos in enumerate(models.EOS):
    #     sel = sims[(sims.EOS == eos) & models.with_disk_mass]
    #     ax.scatter(sel["Mb"] / seq[eos].M0_sup, sel["dM_0_max"],
    #                color=models.color_list[ieos], marker=models.marker_list[ieos],
    #                label=models.print_eos(eos))

    mod = simulations.loc[sim]
    o_data = ADD_METHODS_ALL_PAR(sim)
    rns.compute_bounding_sequences_in_J_M0_for_EOS(mod["EOS"])
    rns.get_rns_M0_min_fit(mod["EOS"])
    rns.get_rns_M0_max_fit(mod["EOS"])
    rns.get_Jmax_and_M0_sup(mod["EOS"])
    mod = add_dm_dj(o_data, mod, rns.seq)

    # sec ejecta
    ax.plot(mod.Mb / rns.seq[mod["EOS"]]["M0_sup"], mod["dM_0_max"], "*",
            color="gold", label="secular ejecta")

    # sww
    msww = o_data.get_outflow_data(0, "bern_geoend", "total_flux.dat")
    print(msww.shape)
    ax.plot(mod.Mb / rns.seq[mod["EOS"]]["M0_sup"], msww[-1,2], "P",
            color="gold", label=r"$M_{\rm sww}$ after "+"{:.1f}ms".format(1e3*msww[-1,0]))

    M_of_t = interpolate.interp1d(msww[:,2], msww[:,0], kind="linear",fill_value="extrapolate")
    t_to_rns = M_of_t(mod["dM_0_max"])
    print("time_for_the_wind_to_reach_secualr_ejecta {:.1f} [ms]".format(1e3*t_to_rns))

    #  ---------- 3D data ----------------

    # ts, mbs, mbsfedu, jcs, jcsfedu = load_J_Mb_evol(o_data, times=[10, 20, 30, 40, 50, 60, 70, 80, 90],
    #                                                 options_Jfac_fidu=rns.options.Jfac_fidu)
    # jevo = [j[-1] for j in jcs]
    # mbevo = [mb[-1] for mb in mbs]


    han, lab = ax.get_legend_handles_labels() # list of labels from all ax.plot()
    ax.add_artist(ax.legend(han[:2], lab[:2], loc="upper left"))
    ax.add_artist(ax.legend(han[2:], lab[2:], loc="lower left", ncol=2))

    ax.set_title(sim.replace('_', '\_'))

    ax.set_xlabel(r"$M_b/M_{\rm RNS}$")
    ax.set_ylabel(r"$M_{\rm ej}^{\max}\ [M_\odot]$")

    # ax.set_xlim(0.9, 1.2)
    ax.set_ylim(0, 0.25)
    print(__outplotdir__ + "Mej_wind_max_{}.png".format(sim.lower()))
    plt.savefig(__outplotdir__ + "Mej_wind_max_{}.png".format(sim.lower()), dpi=256)
    plt.close()

def test_plot_mj_encl_evol():

    sim = "BLh_M13641364_M0_LK_SR"
    o_data = ADD_METHODS_ALL_PAR(sim)

    times = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    ts, rc, mbs, mbsfedu, jcs, jcsfedu, jfcs = load_J_Mb_evol(o_data, times=times, options_Jfac_fidu=rns.options.Jfac_fidu)
    jevo = np.array([j[-1] for j in jcs])
    mbevo = np.array([mb[-1] for mb in mbs])
    jfevo = jfcs#np.array([jf[-1] for jf in jfcs])
    print(jfevo)

    func_mb_of_t = interpolate.interp1d(mbevo, times, kind="linear", fill_value="extrapolate")

    # fig = plt.figure(figsize=[4, 2.5])
    # ax = fig.add_axes([0.14, 0.15, 1.0 - 0.14 * 2, 0.95 - 0.15])

    fig = plt.figure(figsize=(5.2,5.6))
    ax = fig.add_subplot(211)

    # barionic mass evolution
    ax.plot(ts, mbevo, color="red", marker='.', ls='-', label='Evolution $M_{b}(t)$')

    ax.set_ylabel(r"$M_b\ [M_\odot]$")
    # ax.set_xlabel(r"$t-t_{\rm merg}$ [ms]")

    ax.minorticks_on()
    ax.tick_params(
        axis='both', which='both', labelleft=True,
        labelright=False, tick1On=True, tick2On=False,
        labelsize=int(12),
        direction='in',
        bottom=True, top=True, left=True, right=False
    )
    #

    # ejecta
    msww = o_data.get_outflow_data(0, "bern_geoend", "total_flux.dat")
    # print(msww.shape)
    tmerg = o_data.get_par("tmerg")
    ax.plot(np.array(msww[:, 0]-tmerg)*1e3, mbevo[0] - msww[:, 2], ls='-',color="green", label=r"$M_b[0] - M_{\rm ej}^{wind}$")

    ax.legend(fancybox=True, loc='lower left',
               # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
               shadow=False, ncol=1, fontsize=9,
               framealpha=0., borderaxespad=0.)

    ''' --- '''

    ax2 = fig.add_subplot(212)

    ax2.set_ylabel(r"$J\ [G\, c^{-1} M_\odot^2]$")

    ax2.plot(ts, jevo, color="blue", marker='.', ls='-', label=r"Evolution $\int J(r)rdr$ ")

    ax2.minorticks_on()
    ax2.tick_params(
        axis='both', which='both', labelleft=False,
        labelright=True, tick1On=False, tick2On=True,
        labelsize=int(12),
        direction='in',
        bottom=True, top=True, left=False, right=True
    )

    ax2.set_xlabel(r"$t-t_{\rm merg}$ [ms]")

    ax22 = ax2.twinx()

    ax22.set_ylabel(r"$Jf\ [GEO]]$")

    ax22.plot(ts, jfevo, color="magenta", marker='.', ls='-', label=r"Time evolution $\int J_{f;-1}r_{-1}dr_{-1} z dz$ ")


    ax22.minorticks_on()
    ax22.tick_params(
        axis='both', which='both', labelleft=False,
        labelright=True, tick1On=False, tick2On=True,
        labelsize=int(12),
        direction='in',
        bottom=True, top=True, left=False, right=True
    )

    ax2.legend(fancybox=True, loc='lower left',
               # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
               shadow=False, ncol=1, fontsize=9,
               framealpha=0., borderaxespad=0.)
    ax22.legend(fancybox=True, loc='upper right',
               # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
               shadow=False, ncol=1, fontsize=9,
               framealpha=0., borderaxespad=0.)

    plt.tight_layout()
    print(__outplotdir__ + "mj_evol_{}.png".format(sim.lower()))
    plt.savefig(__outplotdir__ + "mj_evol_{}.png".format(sim.lower()), dpi=128)
    plt.close()

# does not work for LS220
def get_tgw(sim):

    o_data = ADD_METHODS_ALL_PAR(sim)
    tcoll = o_data.get_par("tcoll")
    JADM = o_data.get_initial_data_par("JADM")

    import scidata.units as ut

    L = []
    if np.isfinite(tcoll):
        L.append(0)
    else:
        o_data = ADD_METHODS_ALL_PAR(sim)
        table = o_data.get_gw_data("EJ.dat").T
        #
        t, Jdot, JGW = table[:, 0], table[:, 3], table[:, 5]
        #
        t = ut.conv_time(ut.cactus, ut.cgs, t) * 1e3
        tGW = ut.conv_time(ut.cactus, ut.cgs, (JADM - JGW) / Jdot) * 1e3
        idx = (t > t.max() - 1.5) & (t < t.max() - 0.5)  # over the last 1 ms
        L.append(np.mean(tGW[idx]))
    return L[0]

def plot_gw_timescale():

    sim = "LS220_M14691268_M0_LK_SR"

    fig = plt.figure(figsize=[4, 2.5])
    ax = fig.add_axes([0.14, 0.15, 0.95 - 0.15, 0.95 - 0.15])

    # for ieos, eos in enumerate(options.EOS):
    #     sel = sims[(sims.EOS == eos) & models.fiducial &
    #                ((sims.outcome == MergerOutcome.MNS) | (sims.outcome == MergerOutcome.SMNS))]
    #     ax.plot(sel["Jfinal"], 1e-3 * sel["tGW"],
    #             models.marker_list[ieos], color=models.color_list[ieos],
    #             label=models.print_eos(eos))


    tgw = get_tgw(sim) # plot_gw_timescale
    jfinal = simulations.loc[sim]["Jfinal"]

    print(sim, jfinal, tgw)
    ax.plot(jfinal, 1e-3 * tgw, "*", color="gold",  label=sim.replace('_', '\_'))

    ax.set_xlabel(r"$J\ [G\, c^{-1} M_\odot^2]$")
    ax.set_ylabel(r"$\tau_{\rm GW}\ [{\rm s}]$")

    han, lab = ax.get_legend_handles_labels()
    ax.add_artist(ax.legend(han[:-1], lab[:-1], loc="lower left", ncol=2))
    ax.add_artist(ax.legend([han[-1]], [lab[-1]], loc="upper left"))

    ax.set_xlim(4, 6.5)

    ax.set_ylim(ymin=1e-2, ymax=1e3)
    ax.set_yscale("log")

    print(__outplotdir__ + "GW_timescale_{}.png".format(sim.lower()))
    plt.savefig(__outplotdir__ + "GW_timescale_{}.png".format(sim.lower()), dpi=256)
    plt.close()



""" --------- METHODS --------- """

def plot_Jtot_2d(plot_dic):
    """

    plot_dic = {
        "sim":"BLh_M13641364_M0_LK_SR",
        "times: "np.arange(start=0, stop=95, step=5),
        "xmin": 0., "xmax": 100.,
        "ymin": 0, "ymax": 15.,
        "vmin": 1e-3, "vmax": 1e-0,
        "norm": "log", "plot_dic": "jet",
        "v_n_x": r"$t-t_{\rm merg}$ [ms]",
        "v_n_y": r"cylindrical $R$",
        "v_n":   r"$J_{r} [GEO]$",
        "cmap":  "jet",
        "titel": r"\texttt{" + sim.replace('_', '\_') + "}",
        "figname": __outplotdir__ + "evol_j_2d_{}_R1.png".format(sim.lower())
    }

    :param plot_dic:
    :return:
    """

    #
    # sim = "BLh_M13641364_M0_LK_SR"
    # times = np.arange(start=0, stop=95, step=5)
    o_data = ADD_METHODS_ALL_PAR(plot_dic["sim"])

    #

    # collect data
    ts, rc, mbs, mbsfedu, jcs, jcsfedu, jfcs = \
        load_J_Mb_evol(o_data, times=plot_dic["times"], options_Jfac_fidu=rns.options.Jfac_fidu, cumsum=False)

    plot_t_arr = []
    plot_r_arr = []
    plot_2d_arr = np.zeros(len(rc))
    # jcs = np.zeros(len(ts))
    #
    for i in range(len(ts)):
        print("\tt:{:.1f}".format(ts[i]))
        plot_t_arr.append(ts[i])
        plot_r_arr = rc
        # print(rc)
        # print(jcs[i])
        # print(len(rc), len(jcs[i]))
        plot_2d_arr = np.vstack((plot_2d_arr, jcs[i]))
        #
    plot_2d_arr = np.delete(plot_2d_arr, 0, 0)
    # print(plot_2d_arr[:, (rc>14.)&(rc<16.)])
    print("\tdata is collected")

    # Plotting Dic
    x_arr, y_arr, z_arr = plot_t_arr, plot_r_arr, plot_2d_arr.T
    vmin = plot_dic["vmin"]
    vmax = plot_dic["vmax"]
    if plot_dic["norm"] == "norm" or plot_dic["norm"] == "linear" or plot_dic["norm"] == None:
        norm = Normalize(vmin=vmin, vmax=vmax)
    elif plot_dic["norm"] == "log":
        assert vmin > 0
        assert vmax > 0
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else: raise NameError("wrong norm")
    #
    fig = plt.figure(figsize=(4., 2.5))
    ax = fig.add_subplot(111)
    #
    im = ax.pcolormesh(x_arr, y_arr, z_arr, norm=norm, cmap=plot_dic["cmap"])  # , vmin=dic["vmin"], vmax=dic["vmax"])
    im.set_rasterized(True)
    #
    ax.minorticks_on()
    ax.tick_params(
        axis='both', which='both', labelleft=True,
        labelright=False, tick1On=True, tick2On=True,
        labelsize=int(12),
        direction='in',
        bottom=True, top=True, left=True, right=True
    )
    #

    clb = fig.colorbar(im, ax=ax)
    plt.tight_layout()

    clb.ax.set_title(plot_dic["v_n"], fontsize=11)
    clb.ax.tick_params(labelsize=11)

    ax.set_xlim(plot_dic["xmin"], plot_dic["xmax"])
    ax.set_ylim(plot_dic["ymin"], plot_dic["ymax"])

    ax.set_xscale(plot_dic["xscale"])
    ax.set_yscale(plot_dic["yscale"])

    ax.set_xlabel(plot_dic["v_n_x"])
    ax.set_ylabel(plot_dic["v_n_y"])

    ax.set_title(plot_dic["titel"])

    plt.tight_layout()
    print(plot_dic["figname"])
    plt.savefig(plot_dic["figname"], dpi=128)
    plt.close()
    print("done")
    # exit(1)
    #

def plot_jflux_2d(plot_dic):
    """

        plot_dic = {
            "sim":"BLh_M13641364_M0_LK_SR",
            "times: "np.arange(start=0, stop=95, step=5),
            "xmin": 0., "xmax": 100.,
            "ymin": 0, "ymax": 15.,
            "vmin": 1e-3, "vmax": 1e-0,
            "norm": "log", "plot_dic": "jet",
            "v_n_x": r"$t-t_{\rm merg}$ [ms]",
            "v_n_y": r"cylindrical $R$",
            "v_n":   r"$J_{r} [GEO]$",
            "cmap":  "jet",
            "titel": r"\texttt{" + sim.replace('_', '\_') + "}",
            "figname": __outplotdir__ + "evol_j_2d_{}_R1.png".format(sim.lower())
        }

        :param plot_dic:
        :return:
        """

    o_data = ADD_METHODS_ALL_PAR(plot_dic["sim"])

    # collect data
    ts, rc, mbs, mbsfedu, jcs, jcsfedu, jfcs = \
        load_J_Mb_evol(o_data, times=plot_dic["times"], options_Jfac_fidu=rns.options.Jfac_fidu, cumsum=False)

    plot_t_arr = []
    plot_r_arr = []
    plot_2d_arr = np.zeros(len(rc))
    #
    for i in range(len(ts)):
        print("t:{:.1f}".format(ts[i]))
        plot_t_arr.append(ts[i])
        plot_r_arr = rc

        print(jfcs[i])

        plot_2d_arr = np.vstack((plot_2d_arr, jfcs[i]))
        #
    plot_2d_arr = np.delete(plot_2d_arr, 0, 0)
    print(plot_2d_arr[:, (rc > 14.) & (rc < 16.)])
    print("data is collected")

    # Plotting Dic
    x_arr, y_arr, z_arr = plot_t_arr, plot_r_arr, plot_2d_arr.T
    vmin = plot_dic["vmin"]
    vmax = plot_dic["vmax"]
    if plot_dic["norm"] == "norm" or plot_dic["norm"] == "linear" or plot_dic["norm"] == None:
        norm = Normalize(vmin=vmin, vmax=vmax)
    elif plot_dic["norm"] == "log":
        assert vmin > 0
        assert vmax > 0
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        raise NameError("wrong norm")
    #
    fig = plt.figure(figsize=(4., 2.5))
    ax = fig.add_subplot(111)
    #
    im = ax.pcolormesh(x_arr, y_arr, z_arr, norm=norm, cmap=plot_dic["cmap"])  # , vmin=dic["vmin"], vmax=dic["vmax"])
    im.set_rasterized(True)
    #

    clb = fig.colorbar(im, ax=ax)
    clb.ax.set_title(plot_dic["v_n"], fontsize=11)
    clb.ax.tick_params(labelsize=11)

    ax.set_xlim(plot_dic["xmin"], plot_dic["xmax"])
    ax.set_ylim(plot_dic["ymin"], plot_dic["ymax"])

    ax.set_xscale(plot_dic["xscale"])
    ax.set_yscale(plot_dic["yscale"])

    ax.set_xlabel(plot_dic["v_n_x"])
    ax.set_ylabel(plot_dic["v_n_y"])

    ax.set_title(plot_dic["titel"])

    ax.minorticks_on()
    ax.tick_params(
        axis='both', which='both', labelleft=True,
        labelright=False, tick1On=True, tick2On=True,
        labelsize=int(12),
        direction='in',
        bottom=True, top=True, left=True, right=True
    )

    plt.tight_layout()
    print(plot_dic["figname"])
    plt.savefig(plot_dic["figname"], dpi=128)
    plt.close()
    print("done")

def plot_total_angular_momentum(tasks, plot_dic):


    fig = plt.figure(figsize=plot_dic["figsize"])
    ax = fig.add_subplot(111)
    #

    for task in tasks:
        o_data = ADD_METHODS_ALL_PAR(task["sim"])
        ts, rc, mbs, mbsfedu, jcs, jcsfedu, jfcs = \
            load_J_Mb_evol(o_data, times=task["times"], options_Jfac_fidu=rns.options.Jfac_fidu, cumsum=True)
        jevo = np.array([j[-1] for j in jcs])
        mbevo = np.array([mb[-1] for mb in mbs])
        ax.plot(ts, jevo, color=task["color"], marker=task["marker"], ls=task["ls"], alpha=task["alpha"],
                label=task["label"])

    ax.set_xlim(plot_dic["xmin"], plot_dic["xmax"])
    ax.set_ylim(plot_dic["ymin"], plot_dic["ymax"])

    ax.set_xscale(plot_dic["xscale"])
    ax.set_yscale(plot_dic["yscale"])

    ax.set_xlabel(plot_dic["v_n_x"])
    ax.set_ylabel(plot_dic["v_n_y"])

    ax.set_title(plot_dic["title"])

    ax.legend(**plot_dic["legend"])

    ax.minorticks_on()
    ax.tick_params(
        axis='both', which='both', labelleft=True,
        labelright=False, tick1On=True, tick2On=True,
        labelsize=int(12),
        direction='in',
        bottom=True, top=True, left=True, right=True
    )

    plt.tight_layout()
    print(plot_dic["figname"])
    plt.savefig(plot_dic["figname"], dpi=128)
    plt.close()
    print("done")

def plot_total_angular_momentum_flux(tasks, plot_dic):


    fig = plt.figure(figsize=plot_dic["figsize"])
    ax = fig.add_subplot(111)
    #

    for task in tasks:
        o_data = ADD_METHODS_ALL_PAR(task["sim"])
        ts, rc, mbs, mbsfedu, jcs, jcsfedu, jfcs = \
            load_J_Mb_evol(o_data, times=task["times"], options_Jfac_fidu=rns.options.Jfac_fidu, cumsum=True)
        i = uutils.find_nearest_index(rc, task["rext"])
        jfevo = np.array([j[i] for j in jfcs])
        mbevo = np.array([mb[i] for mb in mbs])
        ax.plot(ts, jfevo, color=task["color"], marker=task["marker"], ls=task["ls"], alpha=task["alpha"],
                label=task["label"])

    ax.set_xlim(plot_dic["xmin"], plot_dic["xmax"])
    ax.set_ylim(plot_dic["ymin"], plot_dic["ymax"])

    ax.set_xscale(plot_dic["xscale"])
    ax.set_yscale(plot_dic["yscale"])

    ax.set_xlabel(plot_dic["v_n_x"])
    ax.set_ylabel(plot_dic["v_n_y"])

    ax.set_title(plot_dic["title"])

    ax.legend(**plot_dic["legend"])

    ax.minorticks_on()
    ax.tick_params(
        axis='both', which='both', labelleft=True,
        labelright=False, tick1On=True, tick2On=True,
        labelsize=int(12),
        direction='in',
        bottom=True, top=True, left=True, right=True
    )

    plt.tight_layout()
    print(plot_dic["figname"])
    plt.savefig(plot_dic["figname"], dpi=128)
    plt.close()
    print("done")

def __d1order__(arr, dx):
    idx = 1. / dx
    df = []
    for i in range(1, len(arr)-1):
        df.append(idx * idx * (arr[i+1] + arr[i-1] - 2 * arr[i]))
    return np.array(df)
def __df__(arr_x, arr_y, reinterpolate=True):
    #
    arr_x = np.array(arr_x)
    arr_y = np.array(arr_y)
    #
    if reinterpolate:
        _arr_x = np.linspace(arr_x[0],arr_x[-1],len(arr_x))
        _arr_y = interpolate.interp1d(arr_x, arr_y, kind="linear")(_arr_x)
        arr_x = _arr_x
        arr_y = _arr_y
    #
    dx = (arr_x[-1] - arr_x[0]) / (1. * len(arr_x))
    x0 = arr_x[0] - dx
    xN = arr_x[-1] + dx
    #
    _f_ = interpolate.interp1d(arr_x, arr_y, kind="linear", fill_value="extrapolate")
    #
    arr_y = np.insert(arr_y, 0, _f_(x0))
    arr_y = np.append(arr_y, _f_(xN))
    #
    arr_x = np.insert(arr_x, 0, x0)
    arr_x = np.append(arr_x, xN)
    #
    d_arr_y = __d1order__(arr_y, dx)
    arr_x = arr_x[1:-1]
    #
    assert len(d_arr_y) == len(arr_x)
    #
    return arr_x, d_arr_y
def plot_d_dt_total_angular_momentum(tasks, plot_dic):


    fig = plt.figure(figsize=plot_dic["figsize"])
    ax = fig.add_subplot(111)
    #

    ax.plot(-1, -1, color="gray", marker=tasks[0]["marker"], alpha=tasks[0]["alpha"], label="Positive")
    ax.plot(-1, -1, color="gray", marker="P", alpha=tasks[0]["alpha"], label="Negative")

    for task in tasks:
        o_data = ADD_METHODS_ALL_PAR(task["sim"])
        ts, rc, mbs, mbsfedu, jcs, jcsfedu, jfcs = \
            load_J_Mb_evol(o_data, times=task["times"], options_Jfac_fidu=rns.options.Jfac_fidu, cumsum=True)
        jevo = np.array([j[-1] for j in jcs])
        mbevo = np.array([mb[-1] for mb in mbs])
        #
        ts, djevo = __df__(ts / 0.004925794970773136, jevo)
        ts = ts * 0.004925794970773136

        if "mult" in plot_dic.keys():
            djevo = djevo * float(plot_dic["mult"])

        for t, j in zip(ts, djevo):
            if j > 0:
                marker = task["marker"]
            else:
                j = -1. * j
                marker = 'P'
            ax.plot(t, j, color=task["color"], marker=marker, alpha=task["alpha"])

        ax.plot(ts, np.abs(djevo), color=task["color"], ls=task["ls"],  alpha=task["alpha"], label=task["label"])


    han, lab = ax.get_legend_handles_labels()

    ax.add_artist(ax.legend([han[0],han[1]], [lab[0],lab[1]], loc="lower left", ncol = 1, fontsize = 8))
    ax.add_artist(ax.legend(han[2:], lab[2:], **plot_dic["legend"]))

    #
    # ax.minorticks_on()
    #
    ax.set_xlim(plot_dic["xmin"], plot_dic["xmax"])
    ax.set_ylim(plot_dic["ymin"], plot_dic["ymax"])

    ax.set_xscale(plot_dic["xscale"])
    ax.set_yscale(plot_dic["yscale"])

    ax.set_xlabel(plot_dic["v_n_x"])
    ax.set_ylabel(plot_dic["v_n_y"])

    ax.set_title(plot_dic["title"])

    # plt.legend(**plot_dic["legend"])
    ax.minorticks_on()
    ax.tick_params(
        axis='both', which='both', labelleft=True,
        labelright=False, tick1On=True, tick2On=True,
        labelsize=int(12),
        direction='in',
        bottom=True, top=True, left=True, right=True
    )
    plt.tight_layout()
    print(plot_dic["figname"])
    plt.savefig(plot_dic["figname"], dpi=128)
    plt.close()
    print("done")

def plot_difference_d_dt_Jtot_and_Jflux(tasks, plot_dic):

    fig = plt.figure(figsize=plot_dic["figsize"])
    ax = fig.add_subplot(111)
    #

    ax.plot(-1, -1, color="gray", marker=tasks[0]["marker"], alpha=tasks[0]["alpha"], label="Positive")
    ax.plot(-1, -1, color="gray", marker="P", alpha=tasks[0]["alpha"], label="Negative")

    for task in tasks:
        o_data = ADD_METHODS_ALL_PAR(task["sim"])
        ts, rc, mbs, mbsfedu, jcs, jcsfedu, jfcs = \
            load_J_Mb_evol(o_data, times=task["times"], options_Jfac_fidu=rns.options.Jfac_fidu, cumsum=True)

        i = uutils.find_nearest_index(rc, task["rext"])
        jfevo = np.array([j[i] for j in jfcs])
        #
        jevo = np.array([j[-1] for j in jcs])
        mbevo = np.array([mb[-1] for mb in mbs])
        #
        ts, djevo = __df__(ts / 0.004925794970773136, jevo)
        ts = ts * 0.004925794970773136

        if "mult" in plot_dic.keys():
            djevo = djevo * float(plot_dic["mult"])

        # for t, j, jf in zip(ts, djevo, jfevo):
        #     if j > 0:
        #         marker = task["marker"]
        #     else:
        #         j = -1. * j
        #         marker = 'P'
        #     ax.plot(t, j-jf, color=task["color"], marker=marker, alpha=task["alpha"])

        ax.plot(ts, -1 * (-1 * djevo - jfevo), color=task["color"], marker=task["marker"],
                ls=task["ls"],  alpha=task["alpha"], label=task["label"], markersize=task["ms"])


    han, lab = ax.get_legend_handles_labels()

    # ax.add_artist(ax.legend([han[0],han[1]], [lab[0],lab[1]], loc="lower left", ncol = 1, fontsize = 8))
    ax.add_artist(ax.legend(han[2:], lab[2:], **plot_dic["legend"]))

    #
    # ax.minorticks_on()
    #
    ax.set_xlim(plot_dic["xmin"], plot_dic["xmax"])
    ax.set_ylim(plot_dic["ymin"], plot_dic["ymax"])

    ax.set_xscale(plot_dic["xscale"])
    ax.set_yscale(plot_dic["yscale"])

    ax.set_xlabel(plot_dic["v_n_x"])
    ax.set_ylabel(plot_dic["v_n_y"])

    ax.set_title(plot_dic["title"])

    # plt.legend(**plot_dic["legend"])
    ax.minorticks_on()
    ax.tick_params(
        axis='both', which='both', labelleft=True,
        labelright=False, tick1On=True, tick2On=True,
        labelsize=int(12),
        direction='in',
        bottom=True, top=True, left=True, right=True
    )
    plt.tight_layout()
    print(plot_dic["figname"])
    plt.savefig(plot_dic["figname"], dpi=128)
    plt.close()
    print("done")

""" ----------- TASKS ----------- """

def task_plot_2D_ang_mom():

    simlist = ["BLh_M13641364_M0_LK_SR"]#, "DD2_M13641364_M0_LK_SR_R04"]#,
               # "LS220_M14691268_M0_LK_SR", "BLh_M11461635_M0_LK_SR",
               # "DD2_M15091235_M0_LK_SR", "SFHo_M11461635_M0_LK_SR",
               # "SLy4_M11461635_M0_LK_SR", "LS220_M14691268_M0_LK_SR",
               # "DD2_M13641364_M0_SR", "DD2_M14971245_M0_SR"]

    # Flux as a function of radius and time
    def_plot_dic = {
        "sim": None,
        "times": np.arange(start=0, stop=95, step=1),
        "xmin": 0., "xmax": 110.,
        "ymin": 0, "ymax": 15.,
        "vmin": 1e-3, "vmax": 1e-0,
        "xscale": "linear", "yscale": "linear",
        "norm": "log", "plot_dic": "jet",
        "v_n_x": r"$t-t_{\rm merg}$ [ms]",
        "v_n_y": r"cylindrical $R$",
        "v_n": r"$J_{r} [GEO]$",
        "cmap": "jet",
        "titel": None,
        "figname": __outplotdir__ + "tmp.png"
    }
    for sim in simlist:
        plot_dic = copy.deepcopy(def_plot_dic)
        plot_dic["sim"] = sim
        plot_dic["titel"] = r"\texttt{" + sim.replace('_', '\_') + "}"
        plot_dic["figname"] = __outplotdir__ + r"evol_j_2d_{}_R1.png".format(sim.lower())
        plot_Jtot_2d(plot_dic)
        #
        plot_dic["ymin"] = 15.
        plot_dic["ymax"] = 450.
        plot_dic["yscale"] = "log"
        plot_dic["figname"] = __outplotdir__ + "evol_j_2d_{}_R2.png".format(plot_dic["sim"].lower())
        plot_Jtot_2d(plot_dic)

    print("done")

def task_plot_2D_ang_mom_flux():

    simlist = ["BLh_M13641364_M0_LK_SR", "DD2_M13641364_M0_LK_SR_R04",
               "LS220_M14691268_M0_LK_SR", "BLh_M11461635_M0_LK_SR",
               "DD2_M15091235_M0_LK_SR", "SFHo_M11461635_M0_LK_SR",
               "SLy4_M11461635_M0_LK_SR", "LS220_M14691268_M0_LK_SR",
               "DD2_M13641364_M0_SR", "DD2_M14971245_M0_SR"]

    # Flux as a function of radius and time
    def_plot_dic = {
        "sim": None,
        "times": np.arange(start=0, stop=95, step=5),
        "xmin": 0., "xmax": 110.,
        "ymin": 0, "ymax": 15.,
        "vmin": 1e-8, "vmax": 1e-4,
        "xscale": "linear", "yscale": "linear",
        "norm": "log", "plot_dic": "jet",
        "v_n_x": r"$t-t_{\rm merg}$ [ms]",
        "v_n_y": r"cylindrical $R$",
        "v_n": r"$J_{r} [GEO]$",
        "cmap": "jet",
        "titel": None,
        "figname": __outplotdir__ + "tmp.png"
    }
    for sim in simlist:
        plot_dic = copy.deepcopy(def_plot_dic)
        plot_dic["sim"] = sim
        plot_dic["titel"] = r"\texttt{" + sim.replace('_', '\_') + "}"
        plot_dic["figname"] = __outplotdir__ + r"evol_jflux_2d_{}_R1.png".format(sim.lower())
        plot_jflux_2d(plot_dic)
        #
        plot_dic["ymin"] = 15.
        plot_dic["ymax"] = 450.
        plot_dic["yscale"] = "log"
        plot_dic["figname"] = __outplotdir__ + "evol_jflux_2d_{}_R2.png".format(plot_dic["sim"].lower())
        plot_jflux_2d(plot_dic)

    print("done")

def task_plot_total_ang_mom():

    task = [
        {"sim": "BLh_M13641364_M0_LK_SR", "color": "black", "lw": 1., "ls": '-', "marker": '.', "alpha": 1.,
         "times": np.arange(start=0, stop=95, step=5),
         "label": r"\texttt{" + "BLh_M13641364_M0_LK_SR".replace('_', '\_') + "}"},

        {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "navy", "lw": 1., "ls": '-', "marker": '.', "alpha": 1.,
         "times": np.arange(start=0, stop=95, step=5),
         "label": r"\texttt{" + "DD2_M13641364_M0_LK_SR_R04".replace('_', '\_') + "}"},

        {"sim": "LS220_M14691268_M0_LK_SR", "color": "red", "lw": 1., "ls": '-', "marker": '.', "alpha": 1.,
         "times": np.arange(start=0, stop=95, step=5),
         "label": r"\texttt{" + "LS220_M14691268_M0_LK_SR".replace('_', '\_') + "}"},

        {"sim": "BLh_M11461635_M0_LK_SR", "color": "gray", "lw": 1., "ls": '-', "marker": '.', "alpha": 1.,
         "times": np.arange(start=0, stop=95, step=5),
         "label": r"\texttt{" + "BLh_M11461635_M0_LK_SR".replace('_', '\_') + "}"},

        {"sim": "DD2_M15091235_M0_LK_SR", "color": "deepskyblue", "lw": 1., "ls": '-', "marker": '.', "alpha": 1.,
         "times": np.arange(start=0, stop=95, step=5),
         "label": r"\texttt{" + "DD2_M15091235_M0_LK_SR".replace('_', '\_') + "}"},

        {"sim": "SFHo_M11461635_M0_LK_SR", "color": "olive", "lw": 1., "ls": '-', "marker": '.', "alpha": 1.,
         "times": np.arange(start=0, stop=95, step=5),
         "label": r"\texttt{" + "SFHo_M11461635_M0_LK_SR".replace('_', '\_') + "}"},

        {"sim": "SLy4_M11461635_M0_LK_SR", "color": "magenta", "lw": 1., "ls": '-', "marker": '.', "alpha": 1.,
         "times": np.arange(start=0, stop=95, step=5),
         "label": r"\texttt{" + "SLy4_M11461635_M0_LK_SR".replace('_', '\_') + "}"},

        {"sim": "LS220_M11461635_M0_LK_SR", "color": "coral", "lw": 1., "ls": '-', "marker": '.', "alpha": 1.,
         "times": np.arange(start=0, stop=95, step=5),
         "label": r"\texttt{" + "LS220_M11461635_M0_LK_SR".replace('_', '\_') + "}"},

        {"sim": "DD2_M13641364_M0_SR", "color": "blue", "lw": 1., "ls": '--', "marker": '.', "alpha": 1.,
         "times": np.arange(start=0, stop=95, step=5),
         "label": r"\texttt{" + "DD2_M13641364_M0_SR".replace('_', '\_') + "}"},

        {"sim": "DD2_M14971245_M0_SR", "color": "deepskyblue", "lw": 1., "ls": '--', "marker": '.', "alpha": 1.,
         "times": np.arange(start=0, stop=95, step=5),
         "label": r"\texttt{" + "DD2_M14971245_M0_SR".replace('_', '\_') + "}"},
    ]

    # Total flux f(t)
    plot_dic = {
        "figsize": (6., 2.5),
        "xmin": 0., "xmax": 110.,
        "ymin": 4, "ymax": 6.5,
        "xscale": "linear", "yscale": "linear",
        "v_n_x": r"$t-t_{\rm merg}$ [ms]",
        "v_n_y": r"$J\ [G\, c^{-1} M_\odot^2]$",
        "legend": {"fancybox": False, "loc": 'center left',
                   "bbox_to_anchor":(1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0.,"frameon":False},
        "title":r"$Total Angular Momentum$",
        "figname": __outplotdir__ + "total_angular_momentum.png"
    }
    plot_total_angular_momentum(task, plot_dic)

    print("done")

def task_plot_total_ang_mom_flux():

    rext = 500

    task = [
        {"sim": "BLh_M13641364_M0_LK_SR", "color": "black", "lw": 1., "ls": '-', "marker": '.', "alpha": 1.,
         "times": np.arange(start=0, stop=95, step=5), "rext":rext,
         "label": r"\texttt{" + "BLh_M13641364_M0_LK_SR".replace('_', '\_') + "}"},

        {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "navy", "lw": 1., "ls": '-', "marker": '.', "alpha": 1.,
         "times": np.arange(start=0, stop=95, step=5),"rext":rext,
         "label": r"\texttt{" + "DD2_M13641364_M0_LK_SR_R04".replace('_', '\_') + "}"},

        {"sim": "LS220_M14691268_M0_LK_SR", "color": "red", "lw": 1., "ls": '-', "marker": '.', "alpha": 1.,
         "times": np.arange(start=0, stop=95, step=5),"rext":rext,
         "label": r"\texttt{" + "LS220_M14691268_M0_LK_SR".replace('_', '\_') + "}"},

        {"sim": "BLh_M11461635_M0_LK_SR", "color": "gray", "lw": 1., "ls": '-', "marker": '.', "alpha": 1.,
         "times": np.arange(start=0, stop=95, step=5),"rext":rext,
         "label": r"\texttt{" + "BLh_M11461635_M0_LK_SR".replace('_', '\_') + "}"},

        {"sim": "DD2_M15091235_M0_LK_SR", "color": "deepskyblue", "lw": 1., "ls": '-', "marker": '.', "alpha": 1.,
         "times": np.arange(start=0, stop=95, step=5),"rext":rext,
         "label": r"\texttt{" + "DD2_M15091235_M0_LK_SR".replace('_', '\_') + "}"},

        {"sim": "SFHo_M11461635_M0_LK_SR", "color": "olive", "lw": 1., "ls": '-', "marker": '.', "alpha": 1.,
         "times": np.arange(start=0, stop=95, step=5),"rext":rext,
         "label": r"\texttt{" + "SFHo_M11461635_M0_LK_SR".replace('_', '\_') + "}"},

        {"sim": "SLy4_M11461635_M0_LK_SR", "color": "magenta", "lw": 1., "ls": '-', "marker": '.', "alpha": 1.,
         "times": np.arange(start=0, stop=95, step=5),"rext":rext,
         "label": r"\texttt{" + "SLy4_M11461635_M0_LK_SR".replace('_', '\_') + "}"},

        {"sim": "LS220_M14691268_M0_LK_SR", "color": "coral", "lw": 1., "ls": '-', "marker": '.', "alpha": 1.,
         "times": np.arange(start=0, stop=95, step=5),"rext":rext,
         "label": r"\texttt{" + "LS220_M14691268_M0_LK_SR".replace('_', '\_') + "}"},

        {"sim": "DD2_M13641364_M0_SR", "color": "blue", "lw": 1., "ls": '--', "marker": '.', "alpha": 1.,
         "times": np.arange(start=0, stop=95, step=5),"rext":rext,
         "label": r"\texttt{" + "DD2_M13641364_M0_SR".replace('_', '\_') + "}"},

        {"sim": "DD2_M14971245_M0_SR", "color": "deepskyblue", "lw": 1., "ls": '--', "marker": '.', "alpha": 1.,
         "times": np.arange(start=0, stop=95, step=5),"rext":rext,
         "label": r"\texttt{" + "DD2_M14971245_M0_SR".replace('_', '\_') + "}"},
    ]

    # Total flux f(t)
    plot_dic = {
        "figsize": (6., 2.5),
        "xmin": 0., "xmax": 110.,
        "ymin": 1e-8, "ymax": 2e-6,
        "xscale": "linear", "yscale": "log",
        "v_n_x": r"$t-t_{\rm merg}$ [ms]",
        "v_n_y": r"$J_f\ [G\, c^{-1} M_\odot]$",
        "legend": {"fancybox": False, "loc": 'center left',
                   "bbox_to_anchor":(1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon":False},
        "title":r"Total Angular Momentum Flux $R_{\rm ext} = $"+str(rext),
        "figname": __outplotdir__ + "total_angular_momentum_flux.png"
    }
    plot_total_angular_momentum_flux(task, plot_dic)

    print("done")

def task_plot_d_dt_total_ang_mom():
    task = [
        {"sim": "BLh_M13641364_M0_LK_SR", "color": "black", "lw": 1., "ls": '-', "marker": 'o', "alpha": 1.,
         "times": np.arange(start=0, stop=95, step=5),
         "label": r"\texttt{" + "BLh_M13641364_M0_LK_SR".replace('_', '\_') + "}"},

        {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "navy", "lw": 1., "ls": '-', "marker": 'o', "alpha": 1.,
         "times": np.arange(start=0, stop=95, step=5),
         "label": r"\texttt{" + "DD2_M13641364_M0_LK_SR_R04".replace('_', '\_') + "}"},

        {"sim": "LS220_M14691268_M0_LK_SR", "color": "red", "lw": 1., "ls": '-', "marker": 'o', "alpha": 1.,
         "times": np.arange(start=0, stop=95, step=5),
         "label": r"\texttt{" + "LS220_M14691268_M0_LK_SR".replace('_', '\_') + "}"},

        {"sim": "BLh_M11461635_M0_LK_SR", "color": "gray", "lw": 1., "ls": '-', "marker": 'o', "alpha": 1.,
         "times": np.arange(start=0, stop=95, step=5),
         "label": r"\texttt{" + "BLh_M11461635_M0_LK_SR".replace('_', '\_') + "}"},

        {"sim": "DD2_M15091235_M0_LK_SR", "color": "deepskyblue", "lw": 1., "ls": '-', "marker": 'o', "alpha": 1.,
         "times": np.arange(start=0, stop=95, step=5),
         "label": r"\texttt{" + "DD2_M15091235_M0_LK_SR".replace('_', '\_') + "}"},

        {"sim": "SFHo_M11461635_M0_LK_SR", "color": "olive", "lw": 1., "ls": '-', "marker": 'o', "alpha": 1.,
         "times": np.arange(start=0, stop=95, step=5),
         "label": r"\texttt{" + "SFHo_M11461635_M0_LK_SR".replace('_', '\_') + "}"},

        {"sim": "SLy4_M11461635_M0_LK_SR", "color": "magenta", "lw": 1., "ls": '-', "marker": 'o', "alpha": 1.,
         "times": np.arange(start=0, stop=95, step=5),
         "label": r"\texttt{" + "SLy4_M11461635_M0_LK_SR".replace('_', '\_') + "}"},

        {"sim": "LS220_M14691268_M0_LK_SR", "color": "coral", "lw": 1., "ls": '-', "marker": 'o', "alpha": 1.,
         "times": np.arange(start=0, stop=95, step=5),
         "label": r"\texttt{" + "LS220_M14691268_M0_LK_SR".replace('_', '\_') + "}"},

        {"sim": "DD2_M13641364_M0_SR", "color": "blue", "lw": 1., "ls": '--', "marker": 'o', "alpha": 1.,
         "times": np.arange(start=0, stop=95, step=5),
         "label": r"\texttt{" + "DD2_M13641364_M0_SR".replace('_', '\_') + "}"},

        {"sim": "DD2_M14971245_M0_SR", "color": "deepskyblue", "lw": 1., "ls": '--', "marker": 'o', "alpha": 1.,
         "times": np.arange(start=0, stop=95, step=5),
         "label": r"\texttt{" + "DD2_M14971245_M0_SR".replace('_', '\_') + "}"},
    ]

    # Total flux f(t)
    plot_dic = {
        "figsize": (6., 2.5),
        "xmin": 0., "xmax": 110.,
        "ymin": 1e-10, "ymax": 1e-6,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "log",
        "v_n_x": r"$t-t_{\rm merg}$ [ms]",
        "v_n_y": r"$d/dt J\ [G\, c^{-1} M_\odot^2]$",
        "legend": {"fancybox": False, "loc": 'center left',
                   "bbox_to_anchor":(1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon":False},
        "title": r"$dJ_{\rm tot}/dt$",
        "figname": __outplotdir__ + "d_dt_total_angular_momentum.png"
    }
    plot_d_dt_total_angular_momentum(task, plot_dic)


    print("done")

def task_plot_d_dt_ang_mom_minus_ang_mom_flux():

    ms = 4

    task = [
        {"sim": "BLh_M13641364_M0_LK_SR", "color": "black", "lw": 1., "ls": '-', "marker": 'o', "alpha": 1., "ms":ms,
         "times": np.arange(start=0, stop=95, step=5), "rext":500,
         "label": r"\texttt{" + "BLh_M13641364_M0_LK_SR".replace('_', '\_') + "}"},

        {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "navy", "lw": 1., "ls": '-', "marker": 'o', "alpha": 1.,"ms":ms,
         "times": np.arange(start=0, stop=95, step=5),"rext":500,
         "label": r"\texttt{" + "DD2_M13641364_M0_LK_SR_R04".replace('_', '\_') + "}"},

        {"sim": "LS220_M14691268_M0_LK_SR", "color": "red", "lw": 1., "ls": '-', "marker": 'o', "alpha": 1.,"ms":ms,
         "times": np.arange(start=0, stop=95, step=5),"rext":500,
         "label": r"\texttt{" + "LS220_M14691268_M0_LK_SR".replace('_', '\_') + "}"},

        {"sim": "BLh_M11461635_M0_LK_SR", "color": "gray", "lw": 1., "ls": '-', "marker": 'o', "alpha": 1.,"ms":ms,
         "times": np.arange(start=0, stop=95, step=5),"rext":500,
         "label": r"\texttt{" + "BLh_M11461635_M0_LK_SR".replace('_', '\_') + "}"},

        {"sim": "DD2_M15091235_M0_LK_SR", "color": "deepskyblue", "lw": 1., "ls": '-', "marker": 'o', "alpha": 1.,"ms":ms,
         "times": np.arange(start=0, stop=95, step=5),"rext":500,
         "label": r"\texttt{" + "DD2_M15091235_M0_LK_SR".replace('_', '\_') + "}"},

        {"sim": "SFHo_M11461635_M0_LK_SR", "color": "olive", "lw": 1., "ls": '-', "marker": 'o', "alpha": 1.,"ms":ms,
         "times": np.arange(start=0, stop=95, step=5),"rext":500,
         "label": r"\texttt{" + "SFHo_M11461635_M0_LK_SR".replace('_', '\_') + "}"},

        {"sim": "SLy4_M11461635_M0_LK_SR", "color": "magenta", "lw": 1., "ls": '-', "marker": 'o', "alpha": 1.,"ms":ms,
         "times": np.arange(start=0, stop=95, step=5),"rext":500,
         "label": r"\texttt{" + "SLy4_M11461635_M0_LK_SR".replace('_', '\_') + "}"},

        {"sim": "LS220_M14691268_M0_LK_SR", "color": "coral", "lw": 1., "ls": '-', "marker": 'o', "alpha": 1.,"ms":ms,
         "times": np.arange(start=0, stop=95, step=5),"rext":500,
         "label": r"\texttt{" + "LS220_M14691268_M0_LK_SR".replace('_', '\_') + "}"},

        {"sim": "DD2_M13641364_M0_SR", "color": "blue", "lw": 1., "ls": '--', "marker": 'o', "alpha": 1.,"ms":ms,
         "times": np.arange(start=0, stop=95, step=5),"rext":500,
         "label": r"\texttt{" + "DD2_M13641364_M0_SR".replace('_', '\_') + "}"},

        {"sim": "DD2_M14971245_M0_SR", "color": "deepskyblue", "lw": 1., "ls": '--', "marker": 'o', "alpha": 1.,"ms":ms,
         "times": np.arange(start=0, stop=95, step=5),"rext":500,
         "label": r"\texttt{" + "DD2_M14971245_M0_SR".replace('_', '\_') + "}"},
    ]

    # Total flux f(t)
    plot_dic = {
        "figsize": (6., 2.5),
        "xmin": 0., "xmax": 110.,
        "ymin": 5e-8, "ymax": 2e-6,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "log",
        "v_n_x": r"$t-t_{\rm merg}$ [ms]",
        "v_n_y": r"$-1 \times (-dJ_{\rm tot}/dt - Jflux)$",# $[G\, c^{-1} M_\odot]$",
        "legend": {"fancybox": False, "loc": 'center left',
                   "bbox_to_anchor":(1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon":False},
        "title": r"$\nabla T^{\mu\nu}$",
        "figname": __outplotdir__ + "d_dt_total_angular_momentum_minus_ang_mom_flux.png"
    }

    plot_difference_d_dt_Jtot_and_Jflux(task, plot_dic)

if __name__ == '__main__':

    ''' test'''
    plot_rns_mb_j_and_evold()
    exit(1)

    """ J = f(t,r) """
    # task_plot_2D_ang_mom()

    """ Jflux = f(t,r) """
    task_plot_2D_ang_mom_flux()

    """ J = f(t)  """
    # task_plot_total_ang_mom()

    """ Jflux = f(t)  """
    # task_plot_total_ang_mom_flux()

    """ dJ/dt = f(t) """
    # task_plot_d_dt_total_ang_mom()

    """ DJ/dt - Jflux = f(t) """
    # task_plot_d_dt_ang_mom_minus_ang_mom_flux()


    exit(1)




    test_plot_mj_encl_evol()
    # plot_max_sec_ej_mass()
    # plot_gw_timescale()
    plot_rns_mb_j_and_evold()