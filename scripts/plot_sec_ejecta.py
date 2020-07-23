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
matplotlib.use('agg')
import matplotlib.pyplot as plt
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')


from matplotlib.colors import LogNorm, Normalize

sys.path.append('/data01/numrel/vsevolod.nedora/bns_ppr_tools/')

from data import ADD_METHODS_ALL_PAR
from comparison import TWO_SIMS, THREE_SIMS
from settings import resolutions

import rns
import tov
from model_sets import models as md

__outplotdir__ = "../figs/all3/plot_sec_ejecta/"
if not os.path.isdir(__outplotdir__):
    os.mkdir(__outplotdir__)

''' --- --- --- --- --- --- --- --- --- --- '''

''' --- additional data (TOV, RNS) --- '''
def apply_dissipation_to_j(j_arr, rc_arr):
    options_rej = 300.0  # Estimate the ejecta assuming that everything becomes unbound once it reaches a given radius
    options_Jfac_fidu = lambda rc: (options_rej / (np.minimum(rc, options_rej))) ** (1. / 2.)
    return j_arr * options_Jfac_fidu(rc_arr)

def step_to_zero(fun, xstart=1., xmin=0., dx=0.1):
    xp = xstart
    while fun(xp) >= 0 and xp >= xmin:
        xp -= dx
    return xp

def get_dM_max_dJ_max(o_data, rns):

    print("{}".format(o_data.sim))

    options_rej = 300.0
    options_dJ = 0.01
    options_Jmin = 3.5
    options_Jfac_fidu = lambda rc: (options_rej / (np.minimum(rc, options_rej))) ** (1. / 2.)
    #
    all_iterations, all_times, rc, all_j, all_jf, all_m, all_i = \
        o_data.get_enclosed_mj()
    #
    assert len(all_iterations) > 1
    j3d_last = all_j[-1]
    m3d_last = all_m[-1]
    #
    Jfinal = md.simulations.loc[o_data.sim]["Jfinal"]
    Mb = md.simulations.loc[o_data.sim]["Mb"]
    eos = md.simulations.loc[o_data.sim]["EOS"]

    # compute RNS quantities
    rns.compute_bounding_sequences_in_J_M0_for_EOS(eos)
    rns.get_Jmax_and_M0_sup(eos)

    Jf_max = Jfinal - np.cumsum(np.array(j3d_last)[::-1])
    Jf_fidu = Jfinal - np.cumsum(np.array(j3d_last * options_Jfac_fidu(rc))[::-1])
    Mf = Mb - np.cumsum(np.array(m3d_last)[::-1])
    #
    Mf_fun = interpolate.interp1d(Jf_max[::-1], Mf[::-1], kind="linear", assume_sorted=False)
    fun = lambda Jf: rns.seq[eos]["M0_min_fit"](Jf) - Mf_fun(Jf)
    x = step_to_zero(fun, xstart=Jf_max.max(), dx=options_dJ, xmin=options_Jmin)
    dM_max = Mb - Mf_fun(x)
    dJ_max = Jfinal - x
    #
    Mf_fun = interpolate.interp1d(Jf_fidu[::-1], Mf[::-1], kind="linear", assume_sorted=False)
    fun = lambda Jf: rns.seq[eos]["M0_min_fit"](Jf) - Mf_fun(Jf)
    x = step_to_zero(fun, xstart=Jf_fidu.max(), dx=options_dJ, xmin=options_Jmin)
    dM_fidu = Mb - Mf_fun(x)
    dJ_fidu = Jfinal - x
    #
    print("\t| {} {} {} {}".format(dM_max, dJ_max, dM_fidu, dJ_fidu))
    #
    return dM_max, dJ_max, dM_fidu, dJ_fidu

''' --- modules --- '''

def plot_j0_mb(tasks, plotdic):
    #
    fig = plt.figure(figsize=plotdic["figsize"])
    ax = fig.add_subplot(111)
    #
    # labels

    for task in tasks:
        if task["type"] == "all" or task["type"] == plotdic["type"]:

            Mb_0 = md.simulations.loc[task["sim"]][task["v_n_x"]]
            J_0 = md.simulations.loc[task["sim"]][task["v_n_y"]]
            eos = md.simulations.loc[task["sim"]]["EOS"]

            ax.plot(J_0, Mb_0, task["marker"], mfc=task["color"], mec='black',
                    markersize=task["ms"], label=task["label"])


    ax.set_yscale(plotdic["yscale"])
    ax.set_xscale(plotdic["xscale"])

    ax.set_xlabel(plotdic["xlabel"])  # , fontsize=11)
    ax.set_ylabel(plotdic["ylabel"])  # , fontsize=11)

    ax.set_xlim(plotdic["xmin"], plotdic["xmax"])
    ax.set_ylim(plotdic["ymin"], plotdic["ymax"])
    #
    ax.tick_params(axis='both', which='both', labelleft=True,
                   labelright=False, tick1On=True, tick2On=True,
                   labelsize=12,
                   direction='in',
                   bottom=True, top=True, left=True, right=True)
    ax.minorticks_on()
    # #
    ax.set_title(plotdic["title"])
    # #

    #
    ax.legend(**plotdic["legend"])

    plt.tight_layout()
    #

    print("plotted: \n")
    print(plotdic["figname"])
    plt.savefig(plotdic["figname"], dpi=128)
    plt.close()

def plot_j0_mb_with_RNS(tasks, plotdic):
    #
    fig = plt.figure(figsize=plotdic["figsize"])
    ax = fig.add_subplot(111)
    #
    # labels
    #
    eos = plotdic["EOS"]

    for task in tasks:
        model_eos = md.simulations.loc[task["sim"]]["EOS"]
        if eos == model_eos:# or task["type"] == plotdic["type"]:
            Mb_0 = md.simulations.loc[task["sim"]][task["v_n_x"]]
            J_0 = md.simulations.loc[task["sim"]][task["v_n_y"]]
            ax.plot(J_0, Mb_0, task["marker"], mfc=task["color"], mec='black',
                    markersize=task["ms"], label=task["label"])

    if plotdic["plot_rns"]:
        # Plot RNS Sequence
        rns.compute_bounding_sequences_in_J_M0_for_EOS(eos)
        rns.get_rns_M0_min_fit(eos)
        rns.get_rns_M0_max_fit(eos)
        rns.get_Jmax_and_M0_sup(eos)
        J = np.linspace(0, rns.seq[eos]["Jmax"], 200)
        ax.fill_between(J, rns.seq[eos]["M0_min_fit"](J), rns.seq[eos]["M0_max_fit"](J),
                        edgecolor='black', facecolor='lightgrey',
                        label="RNS", zorder=0)

    ax.set_yscale(plotdic["yscale"])
    ax.set_xscale(plotdic["xscale"])

    ax.set_xlabel(plotdic["xlabel"])  # , fontsize=11)
    ax.set_ylabel(plotdic["ylabel"])  # , fontsize=11)

    ax.set_xlim(plotdic["xmin"], plotdic["xmax"])
    ax.set_ylim(plotdic["ymin"], plotdic["ymax"])
    #
    if plotdic["use_tov_for_ax2"]:
        ax.minorticks_on()
        ax.tick_params(
            axis='both', which='both', labelleft=True,
            labelright=False, tick1On=True, tick2On=False,
            labelsize=12,
            direction='in',
            bottom=True, top=True, left=True, right=False
        )
        #
        ax2 = ax.twinx()
        ax2.set_ylim(plotdic["ymin"], plotdic["ymax"])
        ax2.set_ylabel(r"$M\ [M_\odot]$")
        #
        M0_ticks = np.linspace(plotdic["ymin"], plotdic["ymax"], 6)
        tov.load_tov_get_M_of_M0(plotdic["EOS"])
        M_ticks = 2 * tov.M_of_M0[eos](M0_ticks / 2.)
        M_labels = ["{:.2f}".format(m) for m in M_ticks]
        ax2.set_yticks(M0_ticks)
        ax2.set_yticklabels(M_labels)
        ax2.minorticks_on()
        ax2.tick_params(
            axis='both', which='both', labelleft=False,
            labelright=True, tick1On=False, tick2On=True,
            labelsize=12,
            direction='in',
            bottom=True, top=True, left=False, right=True
        )
    else:
        ax.tick_params(axis='both', which='both', labelleft=True,
                       labelright=False, tick1On=True, tick2On=True,
                       labelsize=12,
                       direction='in',
                       bottom=True, top=True, left=True, right=True)
        ax.minorticks_on()
    # #
    ax.set_title(plotdic["title"])
    # #

    #
    ax.legend(**plotdic["legend"])

    if plotdic["tight_layout"]: plt.tight_layout()
    #

    print("plotted: \n")
    print(plotdic["figname"])
    plt.savefig(plotdic["figname"], dpi=128)
    plt.close()

def plot_j0_mb_RNS_j3D_M3D_final(tasks, plotdic):
    #
    fig = plt.figure(figsize=plotdic["figsize"])
    ax = fig.add_subplot(111)
    #
    # labels
    #
    eos = plotdic["EOS"]

    plots = 0
    for task in tasks:
        model_eos = md.simulations.loc[task["sim"]]["EOS"]
        if eos == model_eos:  # or task["type"] == plotdic["type"]:
            # --- GETTING --- GW data ---
            Mb_0 = md.simulations.loc[task["sim"]][task["v_n_x"]]
            J_0 = md.simulations.loc[task["sim"]][task["v_n_y"]]
            # --- GETTING --- 3D data ---
            o_data=ADD_METHODS_ALL_PAR(task["sim"])
            all_iterations, all_times, rc, all_j, all_jf, all_m, all_i = \
                o_data.get_enclosed_mj()
            j_end_arr = np.array(all_j[-1])
            mb_end_arr = np.array(all_m[-1])
            jcs = np.cumsum(j_end_arr[::-1])
            mbs = np.cumsum(mb_end_arr[::-1])
            #
            mb_evo = np.array([np.sum(m) for m in all_m])
            j_evo = np.array([np.sum(j) for j in all_j])
            #
            Mb_0 = mb_evo[0]
            #
            options_rej = 300.0  # Estimate the ejecta assuming that everything becomes unbound once it reaches a given radius
            options_Jfac_fidu = lambda rc: (options_rej / (np.minimum(rc, options_rej))) ** (1. / 2.)
            #
            jcsfedu = np.cumsum(np.array(j_end_arr * options_Jfac_fidu(rc))[::-1])
            mbsfedu = np.interp(jcs, jcsfedu, mbs)
            #
            ax.plot(J_0, Mb_0, task["marker"], mfc=task["color"], mec='black',
                    markersize=task["ms"], label=task["label"])
            #
            ax.fill_between(J_0 - jcsfedu, Mb_0 - mbs, Mb_0,
                            color="deepskyblue", alpha=0.3, zorder=-2)
            ax.fill_between(J_0 - jcs, Mb_0 - mbs, Mb_0 - mbsfedu,
                            color="cornflowerblue", alpha=0.6, zorder=-1)
            # --- PLOTTING 3D EVOLUTION ---
            # mb_evo = np.array([np.sum(m) for m in all_m])
            # j_evo = np.array([np.sum(j) for j in all_j])
            #
            iterations, times, tot_j, tot_jf, tot_mb = o_data.get_total_enclosed_j_jf_mb()

            if task["use_rl_mass"]:
                iterations_rl, times_rl, masses_rl = o_data.get_summed_disk_remn_mass()
                #
                __tot_j = []
                __tot_mb = []
                for i in range(len(times)):
                    if iterations[i] in iterations_rl:
                        idx = list(iterations_rl).index(iterations[i])
                        __tot_j.append(tot_j[i])
                        __tot_mb.append(masses_rl[idx])
                #
                tot_j = np.array(__tot_j)
                tot_mb = np.array(__tot_mb)

            #
            tmerg = o_data.get_par("tmerg")
            #
            for t, j, mb in zip(times, tot_j, tot_mb):
                print("{:.1f}  {:.1f}  {:.2f}".format(t*1e3, j, mb))
            #
            if task["plot_evo"]:
                ax.plot(tot_j[times>tmerg], tot_mb[times>tmerg], color=task["color"], ls='-', markersize=8)
            #
            if task["ext"]:
                ext_times, ext_markers = [200, 300, 400, 500], ['x', 'x', 'x', 'x']
                for t, mark in zip(ext_times, ext_markers):
                    ext_j = interpolate.interp1d(times, tot_j, kind="linear", fill_value="extrapolate")(t/1.e3)
                    ext_mb =interpolate.interp1d(times, tot_mb, kind="linear", fill_value="extrapolate")(t/1.e3)
                    ax.plot(ext_j, ext_mb, color=task["color"], marker=mark, markersize=5)

            plots += 1
            if plots == 1 and eos == "BLh":
                ax.annotate("$[J_{ADM}-J_{GW}]$",
                            xy=(J_0, Mb_0), xycoords='data',
                            xytext=(-15, 25), textcoords='offset points',
                            arrowprops=dict(facecolor='gray', shrink=1, width=.5,headwidth=5),
                            horizontalalignment='right', verticalalignment='bottom')

            print("done")

    ax.plot([-1, -1], [-1, -1], color="gray", ls='-', label="evolution 3D")
    ax.scatter(-1, -1, color="gray", marker='x', label="extrapolation (every 100ms)")
    ax.fill_between(np.zeros(2,), np.ones(2, ), np.ones(2, ),
                    color="deepskyblue", alpha=0.3, zorder=-2, label="Disk ejecta")
    ax.fill_between(np.zeros(2,), np.ones(2, ), np.ones(2, ),
                    color="cornflowerblue", alpha=0.3, zorder=-2, label="Remnant ejecta")
    # r" $[J_{ADM}-J_{GW}]$"



    if plotdic["plot_rns"]:
        # Plot RNS Sequence
        rns.compute_bounding_sequences_in_J_M0_for_EOS(eos)
        rns.get_rns_M0_min_fit(eos)
        rns.get_rns_M0_max_fit(eos)
        rns.get_Jmax_and_M0_sup(eos)
        J = np.linspace(0, rns.seq[eos]["Jmax"], 200)
        ax.fill_between(J, rns.seq[eos]["M0_min_fit"](J), rns.seq[eos]["M0_max_fit"](J),
                        edgecolor='black', facecolor='lightgrey',
                        label="RNS", zorder=0)

    ax.set_yscale(plotdic["yscale"])
    ax.set_xscale(plotdic["xscale"])

    ax.set_xlabel(plotdic["xlabel"])  # , fontsize=11)
    ax.set_ylabel(plotdic["ylabel"])  # , fontsize=11)

    ax.set_xlim(plotdic["xmin"], plotdic["xmax"])
    ax.set_ylim(plotdic["ymin"], plotdic["ymax"])
    #
    if plotdic["use_tov_for_ax2"]:
        ax.minorticks_on()
        ax.tick_params(
            axis='both', which='both', labelleft=True,
            labelright=False, tick1On=True, tick2On=False,
            labelsize=12,
            direction='in',
            bottom=True, top=True, left=True, right=False
        )
        #
        ax2 = ax.twinx()
        ax2.set_ylim(plotdic["ymin"], plotdic["ymax"])
        ax2.set_ylabel(r"$M\ [M_\odot]$")
        #
        M0_ticks = np.linspace(plotdic["ymin"], plotdic["ymax"], 6)
        tov.load_tov_get_M_of_M0(plotdic["EOS"])
        M_ticks = 2 * tov.M_of_M0[eos](M0_ticks / 2.)
        M_labels = ["{:.2f}".format(m) for m in M_ticks]
        ax2.set_yticks(M0_ticks)
        ax2.set_yticklabels(M_labels)
        ax2.minorticks_on()
        ax2.tick_params(
            axis='both', which='both', labelleft=False,
            labelright=True, tick1On=False, tick2On=True,
            labelsize=12,
            direction='in',
            bottom=True, top=True, left=False, right=True
        )
    else:
        ax.tick_params(axis='both', which='both', labelleft=True,
                       labelright=False, tick1On=True, tick2On=True,
                       labelsize=12,
                       direction='in',
                       bottom=True, top=True, left=True, right=True)
        ax.minorticks_on()
    # #
    ax.set_title(plotdic["title"])

    # --- legends

    han, lab = ax.get_legend_handles_labels()
    ax.add_artist(ax.legend(han[:-3], lab[:-3], **plotdic["legend"]))
    tmp = copy.deepcopy(plotdic["legend"])
    tmp["loc"] = "upper left"
    tmp["ncol"] = 2
    ax.add_artist(ax.legend(han[-3:], lab[-3:], **tmp))

    # ax.legend(**plotdic["legend"])

    if plotdic["tight_layout"]: plt.tight_layout()
    #

    print("plotted: \n")
    print(plotdic["figname"])
    plt.savefig(plotdic["figname"], dpi=128)
    plt.close()

def plot_j0_mb_RNS_j3D_M3D_mej_final(tasks, plotdic):
    #
    fig = plt.figure(figsize=plotdic["figsize"])
    ax = fig.add_subplot(111)
    #
    # labels
    #
    eos = plotdic["EOS"]

    plots = 0
    for task in tasks:
        model_eos = md.simulations.loc[task["sim"]]["EOS"]
        if eos == model_eos:  # or task["type"] == plotdic["type"]:
            # --- GETTING --- GW data ---
            Mb_0 = md.simulations.loc[task["sim"]][task["v_n_x"]]
            J_0 = md.simulations.loc[task["sim"]][task["v_n_y"]]
            # --- GETTING --- 3D data ---
            o_data=ADD_METHODS_ALL_PAR(task["sim"])
            all_iterations, all_times, rc, all_j, all_jf, all_m, all_i = \
                o_data.get_enclosed_mj()
            j_end_arr = np.array(all_j[-1])
            mb_end_arr = np.array(all_m[-1])
            jcs = np.cumsum(j_end_arr[::-1])
            mbs = np.cumsum(mb_end_arr[::-1])
            #
            mb_evo = np.array([np.sum(m) for m in all_m])
            j_evo = np.array([np.sum(j) for j in all_j])
            #
            Mb_0 = mb_evo[0]
            #
            options_rej = 300.0  # Estimate the ejecta assuming that everything becomes unbound once it reaches a given radius
            options_Jfac_fidu = lambda rc: (options_rej / (np.minimum(rc, options_rej))) ** (1. / 2.)
            #
            jcsfedu = np.cumsum(np.array(j_end_arr * options_Jfac_fidu(rc))[::-1])
            mbsfedu = np.interp(jcs, jcsfedu, mbs)
            #
            ax.plot(J_0, Mb_0, task["marker"], mfc=task["color"], mec='black',
                    markersize=task["ms"], label=task["label"])
            #
            ax.fill_between(J_0 - jcsfedu, Mb_0 - mbs, Mb_0,
                            color="deepskyblue", alpha=0.3, zorder=-2)
            ax.fill_between(J_0 - jcs, Mb_0 - mbs, Mb_0 - mbsfedu,
                            color="cornflowerblue", alpha=0.6, zorder=-1)
            # --- PLOTTING 3D EVOLUTION ---
            # mb_evo = np.array([np.sum(m) for m in all_m])
            # j_evo = np.array([np.sum(j) for j in all_j])
            #
            iterations, times, tot_j, tot_jf, tot_mb = o_data.get_total_enclosed_j_jf_mb()

            if task["use_rl_mass"]:
                iterations_rl, times_rl, masses_rl = o_data.get_summed_disk_remn_mass()
                #
                __tot_j = []
                __tot_mb = []
                for i in range(len(times)):
                    if iterations[i] in iterations_rl:
                        idx = list(iterations_rl).index(iterations[i])
                        __tot_j.append(tot_j[i])
                        __tot_mb.append(masses_rl[idx])
                #
                tot_j = np.array(__tot_j)
                tot_mb = np.array(__tot_mb)

            #
            tmerg = o_data.get_par("tmerg")
            #
            for t, j, mb in zip(times, tot_j, tot_mb):
                print("{:.1f}  {:.1f}  {:.2f}".format(t*1e3, j, mb))
            #
            if task["plot_evo"]:
                ax.plot(tot_j[times>tmerg], tot_mb[times>tmerg], color=task["color"], ls='-', markersize=8)
            #
            if task["ext"] == "self":
                ext_times, ext_markers = [200, 300, 400, 500], ['x', 'x', 'x', 'x']
                for t, mark in zip(ext_times, ext_markers):
                    ext_j = interpolate.interp1d(times, tot_j, kind="linear", fill_value="extrapolate")(t/1.e3)
                    ext_mb =interpolate.interp1d(times, tot_mb, kind="linear", fill_value="extrapolate")(t/1.e3)
                    ax.plot(ext_j, ext_mb, color=task["color"], marker=mark, markersize=5)
            elif task["ext"] == "ejecta":
                ext_times, ext_markers = [200, 300, 400, 500], ['x', 'x', 'x', 'x']
                ej_times, ej_masses = o_data.get_time_data_arrs("Mej", 0, "bern_geoend")
                ext_ej_masses = interpolate.interp1d(ej_times, ej_masses, kind="linear", fill_value="extrapolate")(np.array(ext_times)/1.e3)
                ext_mb_system = tot_mb[-1] - (ext_ej_masses-ej_masses[-1]) #
                ext_j_system = interpolate.interp1d(tot_mb, tot_j, kind="linear", fill_value="extrapolate")(ext_mb_system)
                #
                print(task['sim'], ext_mb_system, ext_j_system)
                ax.plot(ext_j_system, ext_mb_system, color=task["color"], marker='x', markersize=5, linestyle = 'None')

            plots += 1
            if plots == 1 and eos == "BLh":
                ax.annotate("$[J_{ADM}-J_{GW}]$",
                            xy=(J_0, Mb_0), xycoords='data',
                            xytext=(-15, 25), textcoords='offset points',
                            arrowprops=dict(facecolor='gray', shrink=1, width=.5,headwidth=5),
                            horizontalalignment='right', verticalalignment='bottom')

            print("done")

    ax.plot([-1, -1], [-1, -1], color="gray", ls='-', label="evolution 3D")
    ax.scatter(-1, -1, color="gray", marker='x', label="extrapolation (every 100ms)")
    ax.fill_between(np.zeros(2,), np.ones(2, ), np.ones(2, ),
                    color="deepskyblue", alpha=0.3, zorder=-2, label="Disk ejecta")
    ax.fill_between(np.zeros(2,), np.ones(2, ), np.ones(2, ),
                    color="cornflowerblue", alpha=0.3, zorder=-2, label="Remnant ejecta")
    # r" $[J_{ADM}-J_{GW}]$"



    if plotdic["plot_rns"]:
        # Plot RNS Sequence
        rns.compute_bounding_sequences_in_J_M0_for_EOS(eos)
        rns.get_rns_M0_min_fit(eos)
        rns.get_rns_M0_max_fit(eos)
        rns.get_Jmax_and_M0_sup(eos)
        J = np.linspace(0, rns.seq[eos]["Jmax"], 200)
        ax.fill_between(J, rns.seq[eos]["M0_min_fit"](J), rns.seq[eos]["M0_max_fit"](J),
                        edgecolor='black', facecolor='lightgrey',
                        label="RNS", zorder=0)

    ax.set_yscale(plotdic["yscale"])
    ax.set_xscale(plotdic["xscale"])

    ax.set_xlabel(plotdic["xlabel"])  # , fontsize=11)
    ax.set_ylabel(plotdic["ylabel"])  # , fontsize=11)

    ax.set_xlim(plotdic["xmin"], plotdic["xmax"])
    ax.set_ylim(plotdic["ymin"], plotdic["ymax"])
    #
    if plotdic["use_tov_for_ax2"]:
        ax.minorticks_on()
        ax.tick_params(
            axis='both', which='both', labelleft=True,
            labelright=False, tick1On=True, tick2On=False,
            labelsize=12,
            direction='in',
            bottom=True, top=True, left=True, right=False
        )
        #
        ax2 = ax.twinx()
        ax2.set_ylim(plotdic["ymin"], plotdic["ymax"])
        ax2.set_ylabel(r"$M\ [M_\odot]$")
        #
        M0_ticks = np.linspace(plotdic["ymin"], plotdic["ymax"], 6)
        tov.load_tov_get_M_of_M0(plotdic["EOS"])
        M_ticks = 2 * tov.M_of_M0[eos](M0_ticks / 2.)
        M_labels = ["{:.2f}".format(m) for m in M_ticks]
        ax2.set_yticks(M0_ticks)
        ax2.set_yticklabels(M_labels)
        ax2.minorticks_on()
        ax2.tick_params(
            axis='both', which='both', labelleft=False,
            labelright=True, tick1On=False, tick2On=True,
            labelsize=12,
            direction='in',
            bottom=True, top=True, left=False, right=True
        )
    else:
        ax.tick_params(axis='both', which='both', labelleft=True,
                       labelright=False, tick1On=True, tick2On=True,
                       labelsize=12,
                       direction='in',
                       bottom=True, top=True, left=True, right=True)
        ax.minorticks_on()
    # #
    ax.set_title(plotdic["title"])

    # --- legends

    han, lab = ax.get_legend_handles_labels()
    ax.add_artist(ax.legend(han[:-3], lab[:-3], **plotdic["legend"]))
    tmp = copy.deepcopy(plotdic["legend"])
    tmp["loc"] = "upper left"
    tmp["ncol"] = 2
    ax.add_artist(ax.legend(han[-3:], lab[-3:], **tmp))

    # ax.legend(**plotdic["legend"])

    if plotdic["tight_layout"]: plt.tight_layout()
    #

    print("plotted: \n")
    print(plotdic["figname"])
    plt.savefig(plotdic["figname"], dpi=128)
    plt.close()

def plot_j0_mb_RNS_j3D_m3D_evo(tasks, plotdic):
    fig = plt.figure(figsize=plotdic["figsize"])
    ax = fig.add_subplot(111)
    #
    # labels
    #
    eos = plotdic["EOS"]

    plots = 0
    for task in tasks:
        model_eos = md.simulations.loc[task["sim"]]["EOS"]
        if eos == model_eos:  # or task["type"] == plotdic["type"]:

            # --- GETTING --- GW data ---
            Mb_0 = md.simulations.loc[task["sim"]][task["v_n_x"]]
            J_0 = md.simulations.loc[task["sim"]][task["v_n_y"]]

            # --- GETTING --- 3D data SHELL ---
            o_data = ADD_METHODS_ALL_PAR(task["sim"])
            all_iterations, all_times, rc, all_j, all_jf, all_m, all_i = \
                o_data.get_enclosed_mj()
            j_end_arr = np.array(all_j[-1])
            mb_end_arr = np.array(all_m[-1])
            jcs = np.cumsum(j_end_arr[::-1])
            mbs = np.cumsum(mb_end_arr[::-1])
            #
            mb_evo = np.array([np.sum(m) for m in all_m])
            j_evo = np.array([np.sum(j) for j in all_j])
            #
            Mb_0 = mb_evo[0] # Use t=0 EVOLUTION data for mass instead of INITAL MASS
            #
            options_rej = 300.0  # Estimate the ejecta assuming that everything becomes unbound once it reaches a given radius
            options_Jfac_fidu = lambda rc: (options_rej / (np.minimum(rc, options_rej))) ** (1. / 2.)
            #
            jcsfedu = np.cumsum(np.array(j_end_arr * options_Jfac_fidu(rc))[::-1])
            mbsfedu = np.interp(jcs, jcsfedu, mbs)

            # --- plot final state of the GW evolution
            if len(task["plot"].keys()) > 0:
                ax.plot(J_0, Mb_0, **task["plot"])

            # --- plot MODEL-RNS line (assuming not all J is left) -> disk ejecta
            if "plot_fill_disk" in task.keys() and len(task["plot_fill_disk"].keys())>0:
                ax.fill_between(J_0 - jcsfedu, Mb_0 - mbs, Mb_0, **task["plot_fill_disk"])

            # --- plot MODEL-RNS line (assuming ALL J and Mb are taken) -> remnant ejecta
            if "plot_fill_remnant" in task.keys() and len(task["plot_fill_remnant"].keys()) > 0:
                ax.fill_between(J_0 - jcs, Mb_0 - mbs, Mb_0 - mbsfedu, **task["plot_fill_remnant"])

            if "plot_expected_evo" in task.keys() and len(task["plot_expected_evo"].keys()) > 0:
                ax.plot(J_0 - jcsfedu, Mb_0 - mbs, **task["plot_expected_evo"])

            ''' --- plotting evolution & extrapolate --- '''
            evodic = task["evo"]
            if len(evodic.keys())>0:
                iterations, times, tot_j, tot_jf, tot_mb = o_data.get_total_enclosed_j_jf_mb()
                # overwrite mass evolution for more precise 3D analysis
                if evodic["use_rl_Mb"]:
                    iterations_rl, times_rl, masses_rl = o_data.get_summed_disk_remn_mass()
                    #
                    __tot_j = []
                    __tot_mb = []
                    for i in range(len(times)):
                        if iterations[i] in iterations_rl:
                            idx = list(iterations_rl).index(iterations[i])
                            __tot_j.append(tot_j[i])
                            __tot_mb.append(masses_rl[idx])
                    #
                    tot_j = np.array(__tot_j)
                    tot_mb = np.array(__tot_mb)
                #
                assert len(tot_j) == len(tot_mb)
                #
                tmerg = o_data.get_par("tmerg")
                times = times - tmerg
                if evodic["t1"]!=None:
                    tot_mb = tot_mb[times>evodic["t1"]/1.e3]
                    tot_j = tot_j[times>evodic["t1"]/1.e3]
                    times = times[times>evodic["t1"]/1.e3]
                else:
                    tot_mb = tot_mb[times >0.] # plot only postmerger
                    tot_j = tot_j[times > 0.]
                    times = times[times > 0.]
                #
                if evodic["t2"]!=None:
                    tot_mb = tot_mb[times<=evodic["t2"]/1.e3]
                    tot_j = tot_j[times<=evodic["t2"]/1.e3]
                    times = times[times<=evodic["t2"]/1.e3]
                print(task["sim"], times*1.e3)

                # --- PLOT EVOLUTION
                if len(evodic["plot"].keys())>0:
                    assert len(tot_j) == len(tot_mb)
                    ax.plot(tot_j, tot_mb, **evodic["plot"])

                # --- EXTRAPOLATE EVOLUTION
                if len(task["extevo"].keys())>0:
                    extdic = task["extevo"]
                    #
                    if extdic["data"] == "self":
                        ext_times = np.array(extdic["times"]) / 1.e3
                        ext_j = interpolate.interp1d(times, tot_j, kind="linear", fill_value="extrapolate")(ext_times)
                        ext_mb = interpolate.interp1d(times, tot_mb, kind="linear", fill_value="extrapolate")(ext_times)
                        if len(extdic["plot"].keys())>0:
                            assert len(ext_j) == len(ext_mb)
                            ax.plot(ext_j, ext_mb, **extdic["plot"])
                    elif extdic["data"] == "ejecta":
                        ext_times = np.array(extdic["times"]) / 1.e3
                        ej_times, ej_masses = o_data.get_time_data_arrs("Mej", 0, "bern_geoend")
                        ext_ej_m = interpolate.interp1d(ej_times, ej_masses, kind="linear", fill_value="extrapolate")(ext_times)
                        #
                        ext_mb_system = tot_mb[-1] - (ext_ej_m - ej_masses[-1])  # Post-END evilution of Mb due to EJECTA
                        ext_j_system = interpolate.interp1d(tot_mb, tot_j, kind="linear", fill_value="extrapolate")(ext_mb_system)
                        #
                        if len(extdic["plot"].keys()):
                            assert len(ext_j_system) == len(ext_mb_system)
                            ax.plot(ext_j_system, ext_mb_system, **extdic["plot"])

            # ANNOTATE THE END OF GW EVOLUTION WITH ARROW ANNOTATION
            plots += 1
            if plotdic["annotate"]:
                if plots == 1 and eos == "BLh":
                    ax.annotate("$[J_{ADM}-J_{GW}]$",
                                xy=(J_0, Mb_0), xycoords='data',
                                xytext=(-15, 25), textcoords='offset points',
                                arrowprops=dict(facecolor='gray', shrink=1, width=.5, headwidth=5),
                                horizontalalignment='right', verticalalignment='bottom')

            print("done")

    ax.plot([-1, -1], [-1, -1], color="gray", ls='-', label="$M_b$, $J$ evolution (3D data)")
    ax.scatter(-1, -1, color="gray", marker='x', label="extrapolation (every 50 ms)")
    if "plot_fill_disk" in plotdic.keys() and len(plotdic["plot_fill_disk"].keys())>0:
        ax.fill_between(np.zeros(2, ), np.ones(2, ), np.ones(2, ), **plotdic["plot_fill_disk"])
    if "plot_fill_remnant" in plotdic.keys() and len(plotdic["plot_fill_remnant"].keys()) > 0:
        ax.fill_between(np.zeros(2, ), np.ones(2, ), np.ones(2, ), **plotdic["plot_fill_remnant"])
    # r" $[J_{ADM}-J_{GW}]$"

    if plotdic["plot_rns"]:
        # Plot RNS Sequence
        rns.compute_bounding_sequences_in_J_M0_for_EOS(eos)
        rns.get_rns_M0_min_fit(eos)
        rns.get_rns_M0_max_fit(eos)
        rns.get_Jmax_and_M0_sup(eos)
        J = np.linspace(0, rns.seq[eos]["Jmax"], 200)
        ax.fill_between(J, rns.seq[eos]["M0_min_fit"](J), rns.seq[eos]["M0_max_fit"](J),
                        edgecolor='black', facecolor='lightgrey',
                        label="RNS", zorder=0)

    ax.set_yscale(plotdic["yscale"])
    ax.set_xscale(plotdic["xscale"])

    ax.set_xlabel(plotdic["xlabel"], fontsize=plotdic["fontsize"])  # , fontsize=11)
    ax.set_ylabel(plotdic["ylabel"], fontsize=plotdic["fontsize"])  # , fontsize=11)

    ax.set_xlim(plotdic["xmin"], plotdic["xmax"])
    ax.set_ylim(plotdic["ymin"], plotdic["ymax"])
    #
    if plotdic["use_tov_for_ax2"]:
        ax.minorticks_on()
        ax.tick_params(
            axis='both', which='both', labelleft=True,
            labelright=False, tick1On=True, tick2On=False,
            labelsize=plotdic["fontsize"],
            direction='in',
            bottom=True, top=True, left=True, right=False
        )
        #
        ax2 = ax.twinx()
        ax2.set_ylim(plotdic["ymin"], plotdic["ymax"])
        ax2.set_ylabel(r"$M\ [M_\odot]$", fontsize=plotdic["fontsize"])
        #
        M0_ticks = np.linspace(plotdic["ymin"], plotdic["ymax"], 6)
        tov.load_tov_get_M_of_M0(plotdic["EOS"])
        M_ticks = 2 * tov.M_of_M0[eos](M0_ticks / 2.)
        M_labels = ["{:.2f}".format(m) for m in M_ticks]
        ax2.set_yticks(M0_ticks)
        ax2.set_yticklabels(M_labels)
        ax2.minorticks_on()
        ax2.tick_params(
            axis='both', which='both', labelleft=False,
            labelright=True, tick1On=False, tick2On=True,
            labelsize=plotdic["fontsize"],
            direction='in',
            bottom=True, top=True, left=False, right=True
        )
    else:
        ax.tick_params(axis='both', which='both', labelleft=True,
                       labelright=False, tick1On=True, tick2On=True,
                       labelsize=plotdic["fontsize"],
                       direction='in',
                       bottom=True, top=True, left=True, right=True)
        ax.minorticks_on()
    # #
    ax.set_title(plotdic["title"], fontsize=plotdic["fontsize"])

    # --- legends

    if plotdic["split_legends"]:
        han, lab = ax.get_legend_handles_labels()
        ax.add_artist(ax.legend(han[:-3], lab[:-3], **plotdic["legend"]))
        tmp = copy.deepcopy(plotdic["legend"])
        tmp["loc"] = "upper left"
        tmp["ncol"] = 2
        ax.add_artist(ax.legend(han[-3:], lab[-3:], **tmp))
    else:
        ax.legend(**plotdic["legend"])

    # ax.legend(**plotdic["legend"])

    if plotdic["tight_layout"]: plt.tight_layout()
    #

    print("plotted: \n")
    print(plotdic["figname"])
    plt.savefig(plotdic["figname"], dpi=128)
    if plotdic["savepdf"]: plt.savefig(plotdic["figname"].replace(".png", ".pdf"))
    plt.close()

def plot_mbsupramss_sec_ev_mass(tasks, plotdic):

    fig = plt.figure(figsize=plotdic["figsize"])
    ax = fig.add_subplot(111)
    #
    # labels

    for task in tasks:
        o_data = ADD_METHODS_ALL_PAR(task["sim"])
        # times, masses = o_data.get_time_data_arrs(task["v_n"], det=task["det"], mask=task["mask"])
        Mb_0 = md.simulations.loc[task["sim"]]["Mb"]
        J_0 = md.simulations.loc[task["sim"]]["Jfinal"]
        eos = md.simulations.loc[task["sim"]]["EOS"]

        # compute RNS quantities
        rns.compute_bounding_sequences_in_J_M0_for_EOS(eos)
        rns.get_Jmax_and_M0_sup(eos)
        M0_sup = rns.seq[eos]["M0_sup"]
        Jmax = rns.seq[eos]["Jmax"]
        #
        dM_max, dJ_max, dM_fidu, dJ_fidu = get_dM_max_dJ_max(o_data, rns)

        # --- SELECT ---
        print("processing {} : {} {}".format(task["sim"], task["v_n_x"], task["v_n_y"]))

        if task["v_n_x"] == "Mb_M0_sup" and task["v_n_y"] == "dM_0_fidu":
            x = float(Mb_0 / M0_sup)
            y = float(dM_fidu)
            print("{} | {}:{} | {}:{}".format(o_data.sim, task["v_n_x"], x, task["v_n_y"], y))
            ax.plot(x, y, **task["plot"])
        elif task["v_n_x"] == "Mb_M0_sup" and task["v_n_y"] == "dM_0_max":
            x = float(Mb_0 / M0_sup)
            y = float(dM_max)
            print("{} | {}:{} | {}:{}".format(o_data.sim, task["v_n_x"], x, task["v_n_y"], y))
            ax.plot(x, y, **task["plot"])

        elif task["v_n_x"] == "Mb_M0_sup" and task["v_n_y"] == "dM_0_fidu_and_dM_0_max":
            x = float(Mb_0 / M0_sup)
            y1 = float(dM_max)
            y2 = float(dM_fidu)
            print("{} | {}:{} | {}:{}".format(o_data.sim, task["v_n_x"], x, task["v_n_y"], (y1,y2)))
            ax.plot([x, x], [y1, y2], **task["plot"])

        elif task["v_n_x"] == "Jfinal_Jmax" and task["v_n_y"] == "dM_0_fidu":
            x = float(J_0 / Jmax)
            y = float(dM_fidu)
            print("{} | {}:{} | {}:{}".format(o_data.sim, task["v_n_x"], x, task["v_n_y"], y))
            ax.plot(x, y, **task["plot"])
        elif task["v_n_x"] == "Jfinal_Jmax" and task["v_n_y"] == "dM_0_max":
            x = float(J_0 / Jmax)
            y = float(dM_max)
            print("{} | {}:{} | {}:{}".format(o_data.sim, task["v_n_x"], x, task["v_n_y"], y))
            ax.plot(x, y, **task["plot"])

        elif task["v_n_x"] == "Jfinal_Jmax" and task["v_n_y"] == "t_to_dM_0_fidu": # t_to_dM_0_fidu
            assert "tej1" in task.keys() and "tej2" in task.keys()
            times_ej, mass_ej = o_data.get_time_data_arrs("Mej", task["det"], task["mask"])
            if task["tej1"] != None:
                mass_ej = mass_ej[times_ej>task["tej1"]/1.e3]
                times_ej = times_ej[times_ej>task["tej1"]/1.e3]
            if task["tej2"] != None:
                mass_ej = mass_ej[times_ej<=task["tej2"]/1.e3]
                times_ej = times_ej[times_ej<=task["tej2"]/1.e3]
                #
            if task["method"] == "int1d":
                tsec_disk = interpolate.interp1d(mass_ej, times_ej, kind="linear", fill_value="extrapolate")(dM_fidu)
                # tsec_remn = interpolate.interp1d(mass_ej, times_ej, kind="linear", fill_value="extrapolate")(dM_max)
            elif task["method"] == "last20ms":
                tsec_disk = interpolate.interp1d(mass_ej, times_ej, kind="linear", fill_value="extrapolate")(dM_fidu)
                # tsec_remn = interpolate.interp1d(mass_ej, times_ej, kind="linear", fill_value="extrapolate")(dM_max)
            else: raise NameError("method is not recognized.")
            #
            x = float(J_0 / Jmax)

            ax.plot(x, tsec_disk*1e3, **task["plot"])
        elif task["v_n_x"] == "Jfinal_Jmax" and task["v_n_y"] == "t_to_dM_0_max": # t_to_dM_0_fidu
            assert "tej1" in task.keys() and "tej2" in task.keys()
            times_ej, mass_ej = o_data.get_time_data_arrs("Mej", task["det"], task["mask"])
            if task["tej1"] != None:
                mass_ej = mass_ej[times_ej>task["tej1"]/1.e3]
                times_ej = times_ej[times_ej>task["tej1"]/1.e3]
            if task["tej2"] != None:
                mass_ej = mass_ej[times_ej<=task["tej2"]/1.e3]
                times_ej = times_ej[times_ej<=task["tej2"]/1.e3]
                #
            if task["method"] == "int1d":
                # tsec_disk = interpolate.interp1d(mass_ej, times_ej, kind="linear", fill_value="extrapolate")(dM_fidu)
                tsec_remn = interpolate.interp1d(mass_ej, times_ej, kind="linear", fill_value="extrapolate")(dM_max)
            elif task["method"] == "last20ms":
                # tsec_disk = interpolate.interp1d(mass_ej, times_ej, kind="linear", fill_value="extrapolate")(dM_fidu)
                tsec_remn = interpolate.interp1d(mass_ej, times_ej, kind="linear", fill_value="extrapolate")(dM_max)
            else: raise NameError("method is not recognized.")
            #
            x = float(J_0 / Jmax)

            ax.plot(x, tsec_remn*1e3, **task["plot"])

        elif task["v_n_x"] == "Mb_M0_sup" and task["v_n_y"] == "t_to_dM_0_fidu": # t_to_dM_0_fidu
            assert "tej1" in task.keys() and "tej2" in task.keys()
            times_ej, mass_ej = o_data.get_time_data_arrs("Mej", task["det"], task["mask"])
            if task["tej1"] != None:
                mass_ej = mass_ej[times_ej>task["tej1"]/1.e3]
                times_ej = times_ej[times_ej>task["tej1"]/1.e3]
            if task["tej2"] != None:
                mass_ej = mass_ej[times_ej<=task["tej2"]/1.e3]
                times_ej = times_ej[times_ej<=task["tej2"]/1.e3]
                #
            if task["method"] == "int1d":
                tsec_disk = interpolate.interp1d(mass_ej, times_ej, kind="linear", fill_value="extrapolate")(dM_fidu)
                # tsec_remn = interpolate.interp1d(mass_ej, times_ej, kind="linear", fill_value="extrapolate")(dM_max)
            elif task["method"] == "last20ms":
                tsec_disk = interpolate.interp1d(mass_ej, times_ej, kind="linear", fill_value="extrapolate")(dM_fidu)
                # tsec_remn = interpolate.interp1d(mass_ej, times_ej, kind="linear", fill_value="extrapolate")(dM_max)
            else: raise NameError("method is not recognized.")
            #
            x = float(Mb_0 / M0_sup)

            ax.plot(x, tsec_disk*1e3, **task["plot"])
        elif task["v_n_x"] == "Mb_M0_sup" and task["v_n_y"] == "t_to_dM_0_max": # t_to_dM_0_fidu
            assert "tej1" in task.keys() and "tej2" in task.keys()
            times_ej, mass_ej = o_data.get_time_data_arrs("Mej", task["det"], task["mask"])
            if task["tej1"] != None:
                mass_ej = mass_ej[times_ej>task["tej1"]/1.e3]
                times_ej = times_ej[times_ej>task["tej1"]/1.e3]
            if task["tej2"] != None:
                mass_ej = mass_ej[times_ej<=task["tej2"]/1.e3]
                times_ej = times_ej[times_ej<=task["tej2"]/1.e3]
                #
            if task["method"] == "int1d":
                # tsec_disk = interpolate.interp1d(mass_ej, times_ej, kind="linear", fill_value="extrapolate")(dM_fidu)
                tsec_remn = interpolate.interp1d(mass_ej, times_ej, kind="linear", fill_value="extrapolate")(dM_max)
            elif task["method"] == "last20ms":
                # tsec_disk = interpolate.interp1d(mass_ej, times_ej, kind="linear", fill_value="extrapolate")(dM_fidu)
                tsec_remn = interpolate.interp1d(mass_ej, times_ej, kind="linear", fill_value="extrapolate")(dM_max)
            else: raise NameError("method is not recognized.")
            #
            x = float(Mb_0 / M0_sup)

            ax.plot(x, tsec_remn*1e3, **task["plot"])

        else:
            raise NameError("unrecognized v_n_x:{} ".format(task["v_n_x"]))



    # tmp = {"color": "gray", "marker": "x", "markersize": 5, "linestyle": ':', "linewidth": 0.5,
    #        "label": "Linear extrapolation"}
    # ax.plot([-1, -1], [-2., -2], **tmp)

    if "add_legend" in plotdic.keys() and len(plotdic["add_legend"].keys())>0:
        ax.plot(-1, -1, **plotdic["add_legend"]["1"])
        ax.plot(-1, -1, **plotdic["add_legend"]["2"])


    ax.set_yscale(plotdic["yscale"])
    ax.set_xscale(plotdic["xscale"])

    ax.set_xlabel(plotdic["xlabel"])  # , fontsize=11)
    ax.set_ylabel(plotdic["ylabel"])  # , fontsize=11)

    ax.set_xlim(plotdic["xmin"], plotdic["xmax"])
    ax.set_ylim(plotdic["ymin"], plotdic["ymax"])
    #
    ax.tick_params(axis='both', which='both', labelleft=True,
                   labelright=False, tick1On=True, tick2On=True,
                   labelsize=12,
                   direction='in',
                   bottom=True, top=True, left=True, right=True)
    ax.minorticks_on()
    #
    ax.set_title(plotdic["title"])

    # LEGENDS
    # han, lab = ax.get_legend_handles_labels()
    # ax.add_artist(ax.legend(han[:-1], lab[:-1], **plotdic["legend"]))  # default
    # tmp = copy.deepcopy(plotdic["legend"])
    # tmp["loc"] = "upper left"
    # tmp["bbox_to_anchor"] = (0., 1.)
    # ax.add_artist(ax.legend([han[-1]], [lab[-1]], **tmp))  # for extapolation
    if "add_legend" in plotdic.keys() and len(plotdic["add_legend"].keys())>0:
        han, lab = ax.get_legend_handles_labels()
        ax.add_artist(ax.legend(han[:-2], lab[:-2], **plotdic["legend"]))  # default
        tmp = copy.deepcopy(plotdic["legend"])
        tmp["loc"] = "upper right"
        tmp["bbox_to_anchor"] = (1., 1.)
        ax.add_artist(ax.legend([han[-2], han[-1]], [lab[-2], lab[-1]], **tmp))  # for extapolation
    else:
        ax.legend(**plotdic["legend"])
    plt.tight_layout()
    #

    print("plotted: \n")
    print(plotdic["figname"])
    plt.savefig(plotdic["figname"], dpi=128)
    plt.close()

''' --- tasks --- '''

def task_plot_j0_mb_init_data():

    task = [
        {"sim": "BLh_M13641364_M0_LK_SR", "color": "black", "marker": "o", "ms": 10, "alpha": 1., "type": "long", "t": -1},
        # {"sim": "BLh_M11841581_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"NS", "v_n": v_n, "t": -1},
        {"sim": "BLh_M11461635_M0_LK_SR", "color": "black", "marker": "d", "ms": 10, "alpha": 1., "type": "long", "t": -1},
        {"sim": "BLh_M10651772_M0_LK_SR", "color": "black", "marker": "s", "ms": 10, "alpha": 1., "type": "long",  "t": -1},
        # {"sim": "BLh_M10201856_M0_SR", "color": "black", "ls": "-", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
        # {"sim": "BLh_M10201856_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
        #
        {"sim": "DD2_M13641364_M0_SR", "color": "blue", "marker": "o", "ms": 10, "alpha": 1., "type": "long","t": -1},
        # {"sim": "DD2_M13641364_M0_SR_R04", "color": "blue", "ls": "--", "lw": 0.7, "alpha": 1., "t1":60, "v_n": v_n, "t": -1},
        {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "blue", "marker": "^", "ms": 10, "alpha": 1., "type": "long","t": 60 / 1.e3},
        {"sim": "DD2_M14971245_M0_SR", "color": "blue", "marker": "d", "ms": 10, "alpha": 1., "type": "long",  "t": -1},
        {"sim": "DD2_M15091235_M0_LK_SR", "color": "blue", "marker": "s", "ms": 10, "alpha": 1., "type": "long",   "t": -1},
        #
        # {"sim": "LS220_M13641364_M0_SR", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "t": -1},
        {"sim": "LS220_M13641364_M0_LK_SR_restart", "color": "red", "marker": "o", "ms": 10, "alpha": 1., "type": "short", "t": 25},
        {"sim": "LS220_M14691268_M0_LK_SR", "color": "red", "marker": "d", "ms": 10, "alpha": 1., "type": "long", "t": 60},
        {"sim": "LS220_M11461635_M0_LK_SR", "color": "red", "marker": "s", "ms": 10, "alpha": 1., "type": "short", "t": -1},
        {"sim": "LS220_M10651772_M0_LK_SR", "color": "red", "marker": "v", "ms": 10, "alpha": 1., "type": "short", "t": -1},
        #
        {"sim": "SFHo_M13641364_M0_SR", "color": "green", "marker": "o", "ms": 10, "alpha": 1., "type": "short", "t": -1},
        {"sim": "SFHo_M14521283_M0_SR", "color": "green", "marker": "d", "ms": 10, "alpha": 1., "type": "short", "t": -1},
        {"sim": "SFHo_M11461635_M0_LK_SR", "color": "green", "marker": "s", "ms": 10, "alpha": 1., "type": "long", "t": -1},
        #
        {"sim": "SLy4_M13641364_M0_SR", "color": "magenta", "marker": "o", "ms": 10, "alpha": 1., "type": "short", "t": -1},
        {"sim": "SLy4_M14521283_M0_SR", "color": "magenta", "marker": "d", "ms": 10, "alpha": 1., "type": "short", "t": -1},
        # {"sim": "SLy4_M10201856_M0_LK_SR", "color": "orange", "ls": "-.", "lw": 0.7, "alpha": 1., "v_n": v_n, "t": -1},
        {"sim": "SLy4_M11461635_M0_LK_SR", "color": "magenta", "marker": "s", "ms": 10, "alpha": 1., "type": "long",  "t": -1}
    ]

    for t in task: t["v_n_x"] = "Mb"
    for t in task: t["v_n_y"] = "Jfinal"
    for t in task: t["label"] = r"\texttt{" + t["sim"].replace('_', '\_') + "}"
    for t in task:
        if t["sim"] == "LS220_M13641364_M0_LK_SR_restart":
            t["label"] = t["label"].replace("\_restart", "")

    # --- LONG ---
    plot_dic = {
        "figsize": (6., 2.5),
        "type": "long",
        "xmin":4.5, "xmax":6.5,
        "ymin":2.7, "ymax":3.2,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "linear",
        "xlabel": r"$J\ [G\, c^{-1} M_\odot^2]$",
        "ylabel": r"$M_b\ [M_\odot]$",
        "legend": {"fancybox": False, "loc": 'center left',
                   "bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "total_j_mb_init_data.png"
    }
    plot_j0_mb(task, plot_dic)

def task_plot_j0_mb_with_rns():

    task = [
        {"sim": "BLh_M13641364_M0_LK_SR", "color": "black", "marker": "o", "ms": 8, "alpha": 1., "type": "long", "t": -1},
        # {"sim": "BLh_M11841581_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"NS", "v_n": v_n, "t": -1},
        {"sim": "BLh_M11461635_M0_LK_SR", "color": "black", "marker": "d", "ms": 8, "alpha": 1., "type": "long", "t": -1},
        {"sim": "BLh_M10651772_M0_LK_SR", "color": "black", "marker": "s", "ms": 8, "alpha": 1., "type": "long", "t": -1},

        {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "blue", "marker": "^", "ms": 8, "alpha": 1., "type": "long", "t": 60 / 1.e3},
        {"sim": "DD2_M14971245_M0_SR", "color": "blue", "marker": "d", "ms": 8, "alpha": 1., "type": "long", "t": -1},
        {"sim": "DD2_M15091235_M0_LK_SR", "color": "blue", "marker": "s", "ms": 8, "alpha": 1., "type": "long", "t": -1},
    ]

    for t in task: t["v_n_x"] = "Mb"
    for t in task: t["v_n_y"] = "Jfinal"
    for t in task: t["label"] = r"\texttt{" + t["sim"].replace('_', '\_') + "}"
    for t in task:
        if t["sim"] == "LS220_M13641364_M0_LK_SR_restart":
            t["label"] = t["label"].replace("\_restart", "")

    plot_dic = {
        "figsize": (4.2, 3.6),#(3.6, 4.2),
        "type": "long",
        "EOS": "DD2",
        "plot_rns": True,
        "xmin": 4.5, "xmax": 6.5,
        "ymin": 2.6, "ymax": 3.6,
        "use_tov_for_ax2": True,
        "xscale": "linear", "yscale": "linear",
        "xlabel": r"$J\ [G\, c^{-1} M_\odot^2]$",
        "ylabel": r"$M_b\ [M_\odot]$",
        "legend": {"fancybox": False, "loc": 'lower right',
                   # "bbox_to_anchor": (0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "total_j_mb_RNS_{}.png".format("DD2"),
        "tight_layout": True
    }
    plot_j0_mb_with_RNS(task, plot_dic)

    # ---- BLh ----

    plot_dic_blh = copy.deepcopy(plot_dic)
    plot_dic_blh["EOS"] = "BLh"
    plot_dic_blh["xmin"], plot_dic_blh["xmax"] = 3.0, 7.0
    plot_dic_blh["ymin"], plot_dic_blh["ymax"] = 2.6, 3.2
    plot_dic_blh["figname"] = __outplotdir__ + "total_j_mb_RNS_{}.png".format("BLh")

    plot_j0_mb_with_RNS(task, plot_dic_blh)

def task_plot_j0_mb_RNS_j3D_M3D_final():

    task = [
        {"sim": "BLh_M13641364_M0_LK_SR", "color": "black", "marker": "o", "ms": 6, "alpha": 1., "type": "long",
         "t": -1, "plot_evo":True, "ext":True, "use_rl_mass":False},
        # {"sim": "BLh_M11841581_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"NS", "v_n": v_n, "t": -1},
        {"sim": "BLh_M11461635_M0_LK_SR", "color": "blue", "marker": "d", "ms": 6, "alpha": 1., "type": "long",
         "t": -1, "plot_evo":True, "ext":True, "use_rl_mass":False},
        # {"sim": "BLh_M10651772_M0_LK_SR", "color": "black", "marker": "s", "ms": 10, "alpha": 1., "type": "long", "t": -1},

        {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "blue", "marker": "^", "ms": 6, "alpha": 1., "type": "long",
         "t": 60 / 1.e3, "plot_evo": True, "ext": True, "use_rl_mass": False},
        {"sim": "DD2_M13641364_M0_SR", "color": "orange", "marker": "v", "ms": 6, "alpha": 1., "type": "long",
         "t": 60 / 1.e3, "plot_evo":True, "ext":True, "use_rl_mass":False},
        {"sim": "DD2_M14971245_M0_SR", "color": "green", "marker": "d", "ms": 6, "alpha": 1., "type": "long",
         "t": -1, "plot_evo":False, "ext":False, "use_rl_mass":True},
        {"sim": "DD2_M15091235_M0_LK_SR", "color": "magenta", "marker": "s", "ms": 6, "alpha": 1., "type": "long",
         "t": -1, "plot_evo":False, "ext":False, "use_rl_mass":True},
    ]

    for t in task: t["v_n_x"] = "Mb"
    for t in task: t["v_n_y"] = "Jfinal"
    for t in task: t["label"] = r"\texttt{" + t["sim"].replace('_', '\_') + "}"
    for t in task:
        if t["sim"] == "LS220_M13641364_M0_LK_SR_restart":
            t["label"] = t["label"].replace("\_restart", "")

    plot_dic = {
        "figsize": (4.2, 3.6),#(3.6, 4.2),
        "type": "long",
        "EOS": "DD2",
        "plot_rns": True,
        "xmin": 4.5, "xmax": 6.0,#"xmin": 5.5, "xmax": 6.0,#"xmin": 4.5, "xmax": 6.0,
        "ymin": 2.8, "ymax": 3.1,#"ymin": 2.9, "ymax": 3.2,#"ymin": 2.8, "ymax": 3.2,
        "use_tov_for_ax2": True,
        "xscale": "linear", "yscale": "linear",
        "xlabel": r"$J\ [G\, c^{-1} M_\odot^2]$",
        "ylabel": r"$M_b\ [M_\odot]$",
        "legend": {"fancybox": False, "loc": 'lower right',
                   # "bbox_to_anchor": (0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "total_j_mb_RNS_{}.png".format("DD2"),
        "tight_layout": True
    }
    plot_j0_mb_RNS_j3D_M3D_final(task, plot_dic)

    # ---- BLh ----

    plot_dic_blh = copy.deepcopy(plot_dic)
    plot_dic_blh["EOS"] = "BLh"
    plot_dic_blh["xmin"], plot_dic_blh["xmax"] = 3.5, 6.5
    plot_dic_blh["ymin"], plot_dic_blh["ymax"] = 2.8, 3.1
    plot_dic_blh["figname"] = __outplotdir__ + "total_j_mb_RNS_{}.png".format("BLh")

    plot_j0_mb_RNS_j3D_M3D_final(task, plot_dic_blh)

def task_plot_j0_mb_RNS_j3D_M3D_mej_final():

    task = [
        {"sim": "BLh_M13641364_M0_LK_SR", "color": "black", "marker": "o", "ms": 6, "alpha": 1., "type": "long",
         "t": -1, "plot_evo":True, "ext":"ejecta", "use_rl_mass":False},
        # {"sim": "BLh_M11841581_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"NS", "v_n": v_n, "t": -1},
        {"sim": "BLh_M11461635_M0_LK_SR", "color": "blue", "marker": "d", "ms": 6, "alpha": 1., "type": "long",
         "t": -1, "plot_evo":True, "ext":"ejecta", "use_rl_mass":False},
        # {"sim": "BLh_M10651772_M0_LK_SR", "color": "black", "marker": "s", "ms": 10, "alpha": 1., "type": "long", "t": -1},

        {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "blue", "marker": "^", "ms": 6, "alpha": 1., "type": "long",
         "t": 60 / 1.e3, "plot_evo": True, "ext": "ejecta", "use_rl_mass": False},
        {"sim": "DD2_M13641364_M0_SR", "color": "orange", "marker": "v", "ms": 6, "alpha": 1., "type": "long",
         "t": 60 / 1.e3, "plot_evo":True, "ext":"ejecta", "use_rl_mass":False},
        {"sim": "DD2_M14971245_M0_SR", "color": "green", "marker": "d", "ms": 6, "alpha": 1., "type": "long",
         "t": -1, "plot_evo":False, "ext":"None", "use_rl_mass":True},
        {"sim": "DD2_M15091235_M0_LK_SR", "color": "magenta", "marker": "s", "ms": 6, "alpha": 1., "type": "long",
         "t": -1, "plot_evo":False, "ext":"None", "use_rl_mass":True},
    ]

    for t in task: t["v_n_x"] = "Mb"
    for t in task: t["v_n_y"] = "Jfinal"
    for t in task: t["label"] = r"\texttt{" + t["sim"].replace('_', '\_') + "}"
    for t in task:
        if t["sim"] == "LS220_M13641364_M0_LK_SR_restart":
            t["label"] = t["label"].replace("\_restart", "")

    plot_dic = {
        "figsize": (4.2, 3.6),#(3.6, 4.2),
        "type": "long",
        "EOS": "DD2",
        "plot_rns": True,
        "xmin": 4.5, "xmax": 6.0,#"xmin": 5.5, "xmax": 6.0,#"xmin": 4.5, "xmax": 6.0,
        "ymin": 2.8, "ymax": 3.1,#"ymin": 2.9, "ymax": 3.2,#"ymin": 2.8, "ymax": 3.2,
        "use_tov_for_ax2": True,
        "xscale": "linear", "yscale": "linear",
        "xlabel": r"$J\ [G\, c^{-1} M_\odot^2]$",
        "ylabel": r"$M_b\ [M_\odot]$",
        "legend": {"fancybox": False, "loc": 'lower right',
                   # "bbox_to_anchor": (0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "total_j_mb_RNS_{}.png".format("DD2"),
        "tight_layout": True
    }
    plot_j0_mb_RNS_j3D_M3D_mej_final(task, plot_dic)

    # ---- BLh ----

    plot_dic_blh = copy.deepcopy(plot_dic)
    plot_dic_blh["EOS"] = "BLh"
    plot_dic_blh["xmin"], plot_dic_blh["xmax"] = 3.5, 6.5
    plot_dic_blh["ymin"], plot_dic_blh["ymax"] = 2.8, 3.1
    plot_dic_blh["figname"] = __outplotdir__ + "total_j_mb_RNS_{}.png".format("BLh")

    plot_j0_mb_RNS_j3D_M3D_mej_final(task, plot_dic_blh)

def task_plot_j0_mb_RNS_j3D_m3D_evo():

    task = [
        # --- BLh ---
        {"sim": "BLh_M13641364_M0_LK_SR", "v_n_x": "Mb", "v_n_y": "Jfinal",
         "plot":{"marker":"d", "mfc":"gold", "mec":'black', "markersize": 10, "linestyle":'None', "label":None},
         "evo":{"t1":None, "t2":None, "use_rl_Mb": False,
                "plot": {"color":"black", "marker":"None", "markersize":5, "linestyle": '-', "linewidth":1.0}},
         "extevo": {"data":"self", "t1":None, "t2":None, "method": "intd1", "times":[100, 150, 200, 250, 300, 350, 400, 450, 500],
                    "plot":{"color":"black", "marker":"x", "markersize":5, "linestyle": ':', "linewidth":0.5}  } },
        #
        {"sim": "BLh_M11461635_M0_LK_SR", "v_n_x": "Mb", "v_n_y": "Jfinal",
         "plot": {"marker": "o", "mfc": "gold", "mec": 'black', "markersize": 10, "linestyle": 'None', "label": None},
         "evo": {"t1": None, "t2": 50, "use_rl_Mb": False,
                 "plot": {"color": "black", "marker": "None", "markersize": 5, "linestyle": '-', "linewidth": 1.0}},
         "extevo": {}# {"data": "self", "t1": None, "t2": None, "method": "intd1", "times": [100, 150, 200, 250, 300, 350, 400, 450, 500],
                    #"plot": {"color": "black", "marker": "x", "markersize": 5, "linestyle": ':', "linewidth": 0.5}}
         },

        # --- DD2 ---
        # {"sim": "DD2_M13641364_M0_SR", "v_n_x": "Mb", "v_n_y": "Jfinal",
        #  "plot": {"marker": "d", "mfc": "cyan", "mec": 'black', "markersize": 8, "linestyle": 'None', "label": None},
        #  "evo": {"t1": None, "t2": None, "use_rl_Mb": False,
        #          "plot": {"color": "black", "marker": "None", "markersize": 5, "linestyle": '-', "linewidth": 1.0}},
        #  "extevo": {"data": "self", "t1": None, "t2": None, "method": "intd1",
        #             "times": [100, 150, 200, 250, 300, 350, 400, 450, 500],
        #             "plot": {"color": "black", "marker": "x", "markersize": 5, "linestyle": ':', "linewidth": 0.5}}},

        {"sim": "DD2_M13641364_M0_SR_R04", "v_n_x": "Mb", "v_n_y": "Jfinal",
         "plot": {"marker": "d", "mfc": "cyan", "mec": 'black', "markersize": 8, "linestyle": 'None', "label": None},
         "evo": {},#{"t1": None, "t2": None, "use_rl_Mb": False, "plot": {"color": "black", "marker": "None", "markersize": 5, "linestyle": '-', "linewidth": 1.0}},
         "extevo": {}#{"data": "self", "t1": None, "t2": None, "method": "intd1", "times": [100, 150, 200, 250, 300, 350, 400, 450, 500],
                    #"plot": {"color": "black", "marker": "x", "markersize": 5, "linestyle": ':', "linewidth": 0.5}},
         },

        {"sim": "DD2_M13641364_M0_LK_SR_R04", "v_n_x": "Mb", "v_n_y": "Jfinal",
         "plot": {"marker": "s", "mfc": "royalblue", "mec": 'black', "markersize": 8, "linestyle": 'None', "label": None},
         "evo": {"t1": None, "t2": None, "use_rl_Mb": False,
                 "plot": {"color": "black", "marker": "None", "markersize": 5, "linestyle": '-', "linewidth": 1.0}},
         "extevo": {"data": "self", "t1": None, "t2": None, "method": "intd1", "times": [100, 150, 200, 250, 300, 350, 400, 450, 500],
                    "plot": {"color": "black", "marker": "x", "markersize": 5, "linestyle": ':', "linewidth": 0.5}}
         },

        {"sim": "DD2_M14971245_M0_SR", "v_n_x": "Mb", "v_n_y": "Jfinal",
         "plot": {"marker": "o", "mfc": "slateblue", "mec": 'black', "markersize": 8, "linestyle": 'None', "label": None},
         "evo": {},#{"t1": None, "t2": None, "use_rl_Mb": False,
                 #"plot": {"color": "black", "marker": "None", "markersize": 5, "linestyle": '-', "linewidth": 1.0}},
         "extevo": {}#{"data": "self", "t1": None, "t2": None, "method": "intd1", "times": [100, 150, 200, 250, 300, 350, 400, 450, 500],
                    #"plot": {"color": "black", "marker": "x", "markersize": 5, "linestyle": ':', "linewidth": 0.5}}
         },

        {"sim": "DD2_M15091235_M0_LK_SR", "v_n_x": "Mb", "v_n_y": "Jfinal",
         "plot": {"marker": "v", "mfc": "blueviolet", "mec": 'black', "markersize": 8, "linestyle": 'None', "label": None},
         "evo": {},#{"t1": None, "t2": 90, "use_rl_Mb": False,
                 #"plot": {"color": "black", "marker": "None", "markersize": 5, "linestyle": '-', "linewidth": 1.0}},
         "extevo":{} #{"data": "self", "t1": None, "t2": None, "method": "intd1", "times": [100, 150, 200, 250, 300, 350, 400, 450, 500],
                    #"plot": {"color": "black", "marker": "x", "markersize": 5, "linestyle": ':', "linewidth": 0.5}}
         },

        # --- LS220
        {"sim": "LS220_M14691268_M0_LK_SR", "v_n_x": "Mb", "v_n_y": "Jfinal",
         "plot": {"marker": "v", "mfc": "red", "mec": 'black', "markersize": 8, "linestyle": 'None', "label": None},
         "evo": {"t1": None, "t2": 90, "use_rl_Mb": False,
                 "plot": {"color": "black", "marker": "None", "markersize": 5, "linestyle": '-', "linewidth": 1.0}},
         "extevo": {}
         # {"data": "self", "t1": None, "t2": None, "method": "intd1", "times": [100, 150, 200, 250, 300, 350, 400, 450, 500],
         # "plot": {"color": "black", "marker": "x", "markersize": 5, "linestyle": ':', "linewidth": 0.5}}
         },

        # --- SFHo
        {"sim": "SFHo_M11461635_M0_LK_SR", "v_n_x": "Mb", "v_n_y": "Jfinal",
         "plot": {"marker": "v", "mfc": "red", "mec": 'black', "markersize": 8, "linestyle": 'None', "label": None},
         "evo": {"t1": None, "t2": 50, "use_rl_Mb": False,
                 "plot": {"color": "black", "marker": "None", "markersize": 5, "linestyle": '-', "linewidth": 1.0}},
         "extevo": {"data": "self", "t1": None, "t2": None, "method": "intd1", "times": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
            "plot": {"color": "black", "marker": "x", "markersize": 5, "linestyle": ':', "linewidth": 0.5}}
         },

        # --- SLy4
        {"sim": "SLy4_M11461635_M0_LK_SR", "v_n_x": "Mb", "v_n_y": "Jfinal",
         "plot": {"marker": "v", "mfc": "magenta", "mec": 'black', "markersize": 8, "linestyle": 'None', "label": None},
         "evo": {"t1": None, "t2": 50, "use_rl_Mb": False, "plot": {"color": "black", "marker": "None", "markersize": 5, "linestyle": '-', "linewidth": 1.0}},
         "extevo": {"data": "self", "t1": None, "t2": None, "method": "intd1",
                    "times": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
                    "plot": {"color": "black", "marker": "x", "markersize": 5, "linestyle": ':', "linewidth": 0.5}}
         }

    ]

    for t in task: t["plot"]["label"] = r"\texttt{" + t["sim"].replace('_', '\_') + "}"
    for t in task:
        if t["sim"] == "LS220_M13641364_M0_LK_SR_restart":
            t["label"] = t["label"].replace("\_restart", "")

    plot_dic = {
        "figsize": (4.2, 3.6),#(3.6, 4.2),
        "type": "long",
        "EOS": "DD2",
        "plot_rns": True,
        "xmin": 4.5, "xmax": 6.0,#"xmin": 5.5, "xmax": 6.0,#"xmin": 4.5, "xmax": 6.0,
        "ymin": 2.8, "ymax": 3.1,#"ymin": 2.9, "ymax": 3.2,#"ymin": 2.8, "ymax": 3.2,
        "use_tov_for_ax2": True,
        "xscale": "linear", "yscale": "linear",
        "xlabel": r"$J\ [G\, c^{-1} M_\odot^2]$",
        "ylabel": r"$M_b\ [M_\odot]$",
        "legend": {"fancybox": False, "loc": 'lower right',
                   # "bbox_to_anchor": (0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "total_j_mb_RNS_{}.png".format("DD2"),
        "tight_layout": True
    }
    plot_j0_mb_RNS_j3D_m3D_evo(task, plot_dic)

    # ---- BLh ----

    plot_dic_blh = copy.deepcopy(plot_dic)
    plot_dic_blh["EOS"] = "BLh"
    plot_dic_blh["xmin"], plot_dic_blh["xmax"] = 3.5, 6.5
    plot_dic_blh["ymin"], plot_dic_blh["ymax"] = 2.8, 3.1
    plot_dic_blh["figname"] = __outplotdir__ + "total_j_mb_RNS_{}.png".format("BLh")

    plot_j0_mb_RNS_j3D_m3D_evo(task, plot_dic_blh)

    # ---- LS220 ----

    plot_dic_blh = copy.deepcopy(plot_dic)
    plot_dic_blh["EOS"] = "LS220"
    plot_dic_blh["xmin"], plot_dic_blh["xmax"] = 3.5, 6.5
    plot_dic_blh["ymin"], plot_dic_blh["ymax"] = 2.8, 3.1
    plot_dic_blh["figname"] = __outplotdir__ + "total_j_mb_RNS_{}.png".format("LS220")

    plot_j0_mb_RNS_j3D_m3D_evo(task, plot_dic_blh)

    # ---- SFHo ----

    plot_dic_blh = copy.deepcopy(plot_dic)
    plot_dic_blh["EOS"] = "SFHo"
    plot_dic_blh["xmin"], plot_dic_blh["xmax"] = 3.5, 6.5
    plot_dic_blh["ymin"], plot_dic_blh["ymax"] = 2.7, 2.9
    plot_dic_blh["figname"] = __outplotdir__ + "total_j_mb_RNS_{}.png".format("SFHo")

    plot_j0_mb_RNS_j3D_m3D_evo(task, plot_dic_blh)

    # ---- SLy4 ----

    plot_dic_blh = copy.deepcopy(plot_dic)
    plot_dic_blh["EOS"] = "SLy4"
    plot_dic_blh["xmin"], plot_dic_blh["xmax"] = 3.5, 6.5
    plot_dic_blh["ymin"], plot_dic_blh["ymax"] = 2.7, 2.9
    plot_dic_blh["figname"] = __outplotdir__ + "total_j_mb_RNS_{}.png".format("SLy4")

    plot_j0_mb_RNS_j3D_m3D_evo(task, plot_dic_blh)

# single disk or remnant
def task_plot_mbsupramss_sec_ev_mass():


    v_n_x="Mb_M0_sup"
    v_n_y="dM_0_fidu"
    task = [
        # --- BLh ---
        {"sim": "BLh_M13641364_M0_LK_SR", "v_n_x":v_n_x, "v_n_y":v_n_y,
         "plot":{"marker":"d", "mfc":"gold", "mec":'black', "markersize": 8, "linestyle":'None', "label":None}},
        {"sim": "BLh_M11461635_M0_LK_SR", "v_n_x": v_n_x, "v_n_y": v_n_y,
         "plot": {"marker": "o", "mfc": "gold", "mec": 'black', "markersize": 8, "linestyle": 'None', "label": None}},
        # --- DD2 ---
        {"sim": "DD2_M13641364_M0_SR_R04", "v_n_x":v_n_x, "v_n_y":v_n_y,
         "plot": {"marker": "d", "mfc": "cyan", "mec": 'black', "markersize": 8, "linestyle": 'None', "label": None}},
        {"sim": "DD2_M13641364_M0_LK_SR_R04", "v_n_x":v_n_x, "v_n_y":v_n_y,
         "plot": {"marker": "s", "mfc": "royalblue", "mec": 'black', "markersize": 8, "linestyle": 'None', "label": None}},
        {"sim": "DD2_M14971245_M0_SR", "v_n_x":v_n_x, "v_n_y":v_n_y,
         "plot": {"marker": "o", "mfc": "slateblue", "mec": 'black', "markersize": 8, "linestyle": 'None',"label": None}},
        {"sim": "DD2_M15091235_M0_LK_SR", "v_n_x":v_n_x, "v_n_y":v_n_y,
         "plot": {"marker": "v", "mfc": "blueviolet", "mec": 'black', "markersize": 8, "linestyle": 'None',"label": None}},
        # --- SFHo
        {"sim": "SFHo_M11461635_M0_LK_SR", "v_n_x":v_n_x, "v_n_y":v_n_y,
         "plot": {"marker": "v", "mfc": "red", "mec": 'black', "markersize": 8, "linestyle": 'None', "label": None}},
        # --- SLy4
        {"sim": "SLy4_M11461635_M0_LK_SR", "v_n_x":v_n_x, "v_n_y":v_n_y,
         "plot": {"marker": "v", "mfc": "magenta", "mec": 'black', "markersize": 8, "linestyle": 'None', "label": None}},
    ]
    #
    for t in task: t["plot"]["label"] = r"\texttt{" + t["sim"].replace('_', '\_') + "}"

    plot_dic = {
        "figsize": (6., 2.5), #"figsize": (4.2, 3.6),#(3.6, 4.2),
        "type": "long",
        "xmin": 0.8, "xmax": 1.1,#"xmin": 5.5, "xmax": 6.0,#"xmin": 4.5, "xmax": 6.0,
        "ymin": 0, "ymax": 0.40,#"ymin": 2.9, "ymax": 3.2,#"ymin": 2.8, "ymax": 3.2,
        "xscale": "linear", "yscale": "linear",
        "xlabel": r"$M_b/M_{\rm RNS}$",
        "ylabel": r"$M_{\rm ej}^{\max}\ [M_\odot]$",
        "legend": {"fancybox": False, "loc": 'center left',
                   "bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        # "legend": {"fancybox": False, "loc": 'lower right',
        #            # "bbox_to_anchor": (0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #            "shadow": "False", "ncol": 1, "fontsize": 10,
        #            "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "final_{}__{}.png".format(v_n_x, v_n_y),
        "tight_layout": True
    }
    # plot_mbsupramss_sec_ev_mass(task, plot_dic)

    # --- disk ejecta ---

    v_n_y = "dM_0_fidu"
    for t in task: t["v_n_y"] = v_n_y
    disk_plot_dic = copy.deepcopy(plot_dic)
    disk_plot_dic["ylabel"] = r"$M_{\rm ej}^{\rm disk}\ [M_\odot]$"
    disk_plot_dic["figname"] = __outplotdir__ + "final_{}__{}.png".format(v_n_x, v_n_y)
    disk_plot_dic["ymin"], disk_plot_dic["ymax"] = 0., 0.15

    # plot_mbsupramss_sec_ev_mass(task, disk_plot_dic)

    # --- remnant ejecta ---

    v_n_y = "dM_0_max"
    for t in task: t["v_n_y"] = v_n_y
    remnant_plot_dic = copy.deepcopy(plot_dic)
    remnant_plot_dic["ylabel"] = r"$M_{\rm ej}^{\rm max}\ [M_\odot]$"
    remnant_plot_dic["figname"] = __outplotdir__ + "final_{}__{}.png".format(v_n_x, v_n_y)
    remnant_plot_dic["ymin"], remnant_plot_dic["ymax"] = 0., 0.30

    # plot_mbsupramss_sec_ev_mass(task, remnant_plot_dic)

    # --- disk and remnant ---

    v_n_y = "dM_0_fidu_and_dM_0_max"
    for t in task: t["v_n_y"] = v_n_y
    remnant_plot_dic = copy.deepcopy(plot_dic)
    remnant_plot_dic["ylabel"] = r"$M_{\rm ej}^{\rm sec}\ [M_\odot]$"
    remnant_plot_dic["figname"] = __outplotdir__ + "final_{}__{}.png".format(v_n_x, v_n_y)
    remnant_plot_dic["ymin"], remnant_plot_dic["ymax"] = 0., 0.30

    # plot_mbsupramss_sec_ev_mass(task, remnant_plot_dic)

    # --- time for wind to eject "disk"

    v_n_y = "t_to_dM_0_fidu"
    for t in task: t["v_n_y"] = v_n_y
    for t in task: t["tej1"], t["tej2"], t["method"], t["det"], t["mask"] = \
        None, None, "int1d", 0, "bern_geoend"
    for t in task:
        if t["sim"] == "BLh_M11461635_M0_LK_SR": t["tej1"], t["tej2"] = 20., 50.
        if t["sim"] == "DD2_M15091235_M0_LK_SR": t["tej1"], t["tej2"] = 20., 90.
    remnant_plot_dic = copy.deepcopy(plot_dic)
    remnant_plot_dic["ylabel"] = r"$t_{\rm ej}^{\rm disk}\ [ms]$"
    remnant_plot_dic["figname"] = __outplotdir__ + "final_{}__{}.png".format(v_n_x, v_n_y)
    remnant_plot_dic["ymin"], remnant_plot_dic["ymax"],remnant_plot_dic["yscale"]  = 0., 2000, "linear"

    # plot_mbsupramss_sec_ev_mass(task, remnant_plot_dic)

    # --- time for wind to eject "disk"
    v_n_x = "Jfinal_Jmax"
    v_n_y = "t_to_dM_0_fidu"
    for t in task: t["v_n_x"], t["v_n_y"] = v_n_x, v_n_y
    for t in task: t["tej1"], t["tej2"], t["method"], t["det"], t["mask"] = None, None, "int1d", 0, "bern_geoend"
    for t in task:
        del t["plot"]["mfc"]
        del t["plot"]["mec"]
        t["plot"]["color"] = ["red", "blue"]
    for t in task:
        if t["sim"] == "BLh_M11461635_M0_LK_SR": t["tej1"], t["tej2"] = 20., 50.
        if t["sim"] == "DD2_M15091235_M0_LK_SR": t["tej1"], t["tej2"] = 20., 90.
    remnant_plot_dic = copy.deepcopy(plot_dic)
    remnant_plot_dic["xlabel"] = r"$J_{\rm final}/J_{\rm RNS;max}$"
    remnant_plot_dic["ylabel"] = r"$t_{\rm ej}^{\rm disk}\ [ms]$"
    remnant_plot_dic["figname"] = __outplotdir__ + "final_{}__{}.png".format(v_n_x, v_n_y)
    remnant_plot_dic["ymin"], remnant_plot_dic["ymax"], remnant_plot_dic["yscale"] = 0., 2000, "linear"
    remnant_plot_dic["xmin"], remnant_plot_dic["xmax"], remnant_plot_dic["xscale"] = 0.8, 1.5, "linear"

    plot_mbsupramss_sec_ev_mass(task, remnant_plot_dic)

# both disk and remnant
def task_plot_mbsupramss_sec_ev_time():

    v_n_x="Mb_M0_sup"
    v_n_y2="t_to_dM_0_fidu"
    v_n_y1 = "t_to_dM_0_max"
    task = [
        # --- BLh ---
        {"sim": "BLh_M13641364_M0_LK_SR", "v_n_x":v_n_x, "v_n_y":v_n_y1,
         "plot":{"marker":"v", "mfc": "black", "mec":'black', "markersize": 8, "linestyle":'None', "label":None}},
        {"sim": "BLh_M13641364_M0_LK_SR", "v_n_x": v_n_x, "v_n_y": v_n_y2,
         "plot": {"marker": "^", "mfc": "black", "mec": 'black', "markersize": 8, "linestyle": 'None'}},
        #
        {"sim": "BLh_M11461635_M0_LK_SR", "v_n_x": v_n_x, "v_n_y": v_n_y1,
         "plot": {"marker": "v", "mfc": "gray", "mec": 'black', "markersize": 8, "linestyle": 'None', "label": None}},
        {"sim": "BLh_M11461635_M0_LK_SR", "v_n_x": v_n_x, "v_n_y": v_n_y2,
         "plot": {"marker": "^", "mfc": "gray", "mec": 'black', "markersize": 8, "linestyle": 'None'}},
        #

        # --- DD2 ---
        {"sim": "DD2_M13641364_M0_SR_R04", "v_n_x":v_n_x, "v_n_y":v_n_y1,
         "plot": {"marker": "v", "mfc": "cyan", "mec": 'black', "markersize": 8, "linestyle": 'None', "label": None}},
        {"sim": "DD2_M13641364_M0_SR_R04", "v_n_x": v_n_x, "v_n_y": v_n_y2,
         "plot": {"marker": "^", "mfc": "cyan", "mec": 'black', "markersize": 8, "linestyle": 'None'}},
        #
        {"sim": "DD2_M13641364_M0_LK_SR_R04", "v_n_x":v_n_x, "v_n_y":v_n_y1,
         "plot": {"marker": "v", "mfc": "royalblue", "mec": 'black', "markersize": 8, "linestyle": 'None', "label": None}},
        {"sim": "DD2_M13641364_M0_LK_SR_R04", "v_n_x": v_n_x, "v_n_y": v_n_y2,
         "plot": {"marker": "^", "mfc": "royalblue", "mec": 'black', "markersize": 8, "linestyle": 'None'}},
        #
        {"sim": "DD2_M14971245_M0_SR", "v_n_x":v_n_x, "v_n_y":v_n_y1,
         "plot": {"marker": "v", "mfc": "slateblue", "mec": 'black', "markersize": 8, "linestyle": 'None', "label": None}},
        {"sim": "DD2_M14971245_M0_SR", "v_n_x": v_n_x, "v_n_y": v_n_y2,
         "plot": {"marker": "^", "mfc": "slateblue", "mec": 'black', "markersize": 8, "linestyle": 'None'}},
        #
        {"sim": "DD2_M15091235_M0_LK_SR", "v_n_x":v_n_x, "v_n_y":v_n_y1,
         "plot": {"marker": "v", "mfc": "blueviolet", "mec": 'black', "markersize": 8, "linestyle": 'None', "label": None}},
        {"sim": "DD2_M15091235_M0_LK_SR", "v_n_x": v_n_x, "v_n_y": v_n_y2,
         "plot": {"marker": "^", "mfc": "blueviolet", "mec": 'black', "markersize": 8, "linestyle": 'None'}},
        # --- SFHo
        {"sim": "SFHo_M11461635_M0_LK_SR", "v_n_x":v_n_x, "v_n_y":v_n_y1,
         "plot": {"marker": "v", "mfc": "red", "mec": 'black', "markersize": 8, "linestyle": 'None', "label": None}},
        {"sim": "SFHo_M11461635_M0_LK_SR", "v_n_x": v_n_x, "v_n_y": v_n_y2,
         "plot": {"marker": "^", "mfc": "red", "mec": 'black', "markersize": 8, "linestyle": 'None'}},
        # --- SLy4
        {"sim": "SLy4_M11461635_M0_LK_SR", "v_n_x":v_n_x, "v_n_y":v_n_y1,
         "plot": {"marker": "v", "mfc": "magenta", "mec": 'black', "markersize": 8, "linestyle": 'None', "label": None}},
        {"sim": "SLy4_M11461635_M0_LK_SR", "v_n_x": v_n_x, "v_n_y": v_n_y2,
         "plot": {"marker": "^", "mfc": "magenta", "mec": 'black', "markersize": 8, "linestyle": 'None'}},
    ]
    #
    for t in task:
        if "label" in t["plot"].keys():
            t["plot"]["label"] = r"\texttt{" + t["sim"].replace('_', '\_') + "}"

    plot_dic = {
        "figsize": (6., 2.5), #"figsize": (4.2, 3.6),#(3.6, 4.2),
        "type": "long",
        "xmin": 0.8, "xmax": 1.1,#"xmin": 5.5, "xmax": 6.0,#"xmin": 4.5, "xmax": 6.0,
        "ymin": 0, "ymax": 0.40,#"ymin": 2.9, "ymax": 3.2,#"ymin": 2.8, "ymax": 3.2,
        "xscale": "linear", "yscale": "linear",
        "xlabel": r"$M_b/M_{\rm RNS}$",
        "ylabel": r"$M_{\rm ej}^{\max}\ [M_\odot]$",
        "legend": {"fancybox": False, "loc": 'center left',
                   "bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        # "legend": {"fancybox": False, "loc": 'lower right',
        #            # "bbox_to_anchor": (0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #            "shadow": "False", "ncol": 1, "fontsize": 10,
        #            "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "final_{}__{}.png".format(v_n_x, v_n_y1),
        "tight_layout": True,
        "add_legend":{"1":{"marker": "^", "mfc": "gray", "mec": 'black', "markersize": 8, "linestyle": 'None', "label": "Disk ejecta"},
                      "2":{"marker": "v", "mfc": "gray", "mec": 'black', "markersize": 8, "linestyle": 'None', "label": "Remnant ejecta"}}
    }

    # --- mass for sec ej vs Mmax
    v_n_y2 = "dM_0_fidu"
    v_n_y1 = "dM_0_max"
    for t in task:
        if "label" in t["plot"].keys(): t["v_n_y"] = v_n_y1
        else: t["v_n_y"] = v_n_y2

    remnant_plot_dic = copy.deepcopy(plot_dic)
    remnant_plot_dic["ymin"], remnant_plot_dic["ymax"] = 0, 0.40
    remnant_plot_dic["ylabel"] = r"$M_{\rm ej}^{\rm sec}\ [M_\odot]$"
    remnant_plot_dic["figname"] = __outplotdir__ + "final_{}__{}.png".format(v_n_x, v_n_y1)

    plot_mbsupramss_sec_ev_mass(task, remnant_plot_dic)

    # --- mass for sec ej vs J
    v_n_y2 = "dM_0_fidu"
    v_n_y1 = "dM_0_max"
    v_n_x = "Jfinal_Jmax"
    for t in task: t["v_n_x"] = v_n_x
    for t in task:
        if "label" in t["plot"].keys(): t["v_n_y"] = v_n_y1
        else: t["v_n_y"] = v_n_y2

    remnant_plot_dic = copy.deepcopy(plot_dic)
    remnant_plot_dic["ymin"], remnant_plot_dic["ymax"] = 0, 0.40
    remnant_plot_dic["ylabel"] = r"$M_{\rm ej}^{\rm sec}\ [M_\odot]$"
    remnant_plot_dic["xlabel"] = r"$J_{\rm final}/J_{\rm RNS;max}$"
    remnant_plot_dic["xmin"], remnant_plot_dic["xmax"], remnant_plot_dic["xscale"] = 0.8, 1.5, "linear"
    remnant_plot_dic["figname"] = __outplotdir__ + "final_{}__{}.png".format(v_n_x, v_n_y1)

    plot_mbsupramss_sec_ev_mass(task, remnant_plot_dic)

    # --- time for wind vs Mmax
    v_n_x = "Mb_M0_sup"
    for t in task: t["v_n_x"] = v_n_x
    v_n_y2="t_to_dM_0_fidu"
    v_n_y1 = "t_to_dM_0_max"
    for t in task:
        if "label" in t["plot"].keys():
            t["v_n_y"] = v_n_y1
        else:
            t["v_n_y"] = v_n_y2
    for t in task: t["tej1"], t["tej2"], t["method"], t["det"], t["mask"] = \
        None, None, "int1d", 0, "bern_geoend"
    for t in task:
        if t["sim"] == "BLh_M11461635_M0_LK_SR": t["tej1"], t["tej2"] = 20., 50.
        if t["sim"] == "DD2_M15091235_M0_LK_SR": t["tej1"], t["tej2"] = 20., 90.
    remnant_plot_dic = copy.deepcopy(plot_dic)
    remnant_plot_dic["ylabel"] = r"$t_{\rm ej}^{\rm sec}$ [ms]"
    remnant_plot_dic["figname"] = __outplotdir__ + "final_{}__{}.png".format(v_n_x, v_n_y1)
    remnant_plot_dic["ymin"], remnant_plot_dic["ymax"],remnant_plot_dic["yscale"]  = 0., 2000, "linear"

    plot_mbsupramss_sec_ev_mass(task, remnant_plot_dic)

    # --- time for wind vs Jmax
    for t in task:
        if "label" in t["plot"].keys():
            t["v_n_y"] = v_n_y1
        else:
            t["v_n_y"] = v_n_y2
    v_n_x = "Jfinal_Jmax"
    for t in task: t["v_n_x"] = v_n_x
    for t in task: t["tej1"], t["tej2"], t["method"], t["det"], t["mask"] = None, None, "int1d", 0, "bern_geoend"
    for t in task:
        if t["sim"] == "BLh_M11461635_M0_LK_SR": t["tej1"], t["tej2"] = 20., 50.
        if t["sim"] == "DD2_M15091235_M0_LK_SR": t["tej1"], t["tej2"] = 20., 90.
    remnant_plot_dic = copy.deepcopy(plot_dic)
    remnant_plot_dic["xlabel"] = r"$J_{\rm final}/J_{\rm RNS;max}$"
    remnant_plot_dic["ylabel"] = r"$t_{\rm ej}^{\rm sec}$ [ms]"
    remnant_plot_dic["figname"] = __outplotdir__ + "final_{}__{}.png".format(v_n_x, v_n_y1)
    remnant_plot_dic["ymin"], remnant_plot_dic["ymax"], remnant_plot_dic["yscale"] = 0., 2000, "linear"
    remnant_plot_dic["xmin"], remnant_plot_dic["xmax"], remnant_plot_dic["xscale"] = 0.8, 1.5, "linear"

    plot_mbsupramss_sec_ev_mass(task, remnant_plot_dic)


''' --------- iteration 2 --------- '''

def task_plot_j0_mb_RNS_j3D_m3D_evo_2():

    task = [
        # --- BLh ---
        {"sim": "BLh_M13641364_M0_LK_SR", "v_n_x": "Mb", "v_n_y": "Jfinal",
         "plot":{"marker":"d", "mfc":"gold", "mec":'black', "markersize": 7, "linestyle":'None', "label":"$J_{ADM}-J_{GW}$"},
         "evo":{"t1":None, "t2":None, "use_rl_Mb": False,
                "plot": {"color":"black", "marker":"None", "markersize":5, "linestyle": '-', "linewidth":1.0}},
         "extevo": {"data":"self", "t1":None, "t2":None, "method": "intd1", "times":[100, 150, 200, 250, 300],
                    "plot":{"color":"black", "marker":"x", "markersize":5, "linestyle": ':', "linewidth":0.5}  },
         "plot_fill_disk":      {},#{"color":"deepskyblue", "alpha":0.3, "zorder":-2},
         "plot_fill_remnant":   {},#{"color":"cornflowerblue", "alpha":0.6, "zorder":-1},
         "plot_expected_evo":   {"color":"green", "linewidth":0.8, "linestyle":"--",  "zorder":-1, "label":"Upper bound"}
         },
        #
        # {"sim": "BLh_M11461635_M0_LK_SR", "v_n_x": "Mb", "v_n_y": "Jfinal",
        #  "plot": {"marker": "o", "mfc": "gold", "mec": 'black', "markersize": 10, "linestyle": 'None', "label": None},
        #  "evo": {"t1": None, "t2": 50, "use_rl_Mb": False,
        #          "plot": {"color": "black", "marker": "None", "markersize": 5, "linestyle": '-', "linewidth": 1.0}},
        #  "extevo": {}# {"data": "self", "t1": None, "t2": None, "method": "intd1", "times": [100, 150, 200, 250, 300, 350, 400, 450, 500],
        #             #"plot": {"color": "black", "marker": "x", "markersize": 5, "linestyle": ':', "linewidth": 0.5}}
        #  },
        #
        # # --- DD2 ---
        # # {"sim": "DD2_M13641364_M0_SR", "v_n_x": "Mb", "v_n_y": "Jfinal",
        # #  "plot": {"marker": "d", "mfc": "cyan", "mec": 'black', "markersize": 8, "linestyle": 'None', "label": None},
        # #  "evo": {"t1": None, "t2": None, "use_rl_Mb": False,
        # #          "plot": {"color": "black", "marker": "None", "markersize": 5, "linestyle": '-', "linewidth": 1.0}},
        # #  "extevo": {"data": "self", "t1": None, "t2": None, "method": "intd1",
        # #             "times": [100, 150, 200, 250, 300, 350, 400, 450, 500],
        # #             "plot": {"color": "black", "marker": "x", "markersize": 5, "linestyle": ':', "linewidth": 0.5}}},
        #
        # {"sim": "DD2_M13641364_M0_SR_R04", "v_n_x": "Mb", "v_n_y": "Jfinal",
        #  "plot": {"marker": "d", "mfc": "cyan", "mec": 'black', "markersize": 8, "linestyle": 'None', "label": None},
        #  "evo": {},#{"t1": None, "t2": None, "use_rl_Mb": False, "plot": {"color": "black", "marker": "None", "markersize": 5, "linestyle": '-', "linewidth": 1.0}},
        #  "extevo": {}#{"data": "self", "t1": None, "t2": None, "method": "intd1", "times": [100, 150, 200, 250, 300, 350, 400, 450, 500],
        #             #"plot": {"color": "black", "marker": "x", "markersize": 5, "linestyle": ':', "linewidth": 0.5}},
        #  },
        #
        # {"sim": "DD2_M13641364_M0_LK_SR_R04", "v_n_x": "Mb", "v_n_y": "Jfinal",
        #  "plot": {"marker": "s", "mfc": "royalblue", "mec": 'black', "markersize": 8, "linestyle": 'None', "label": None},
        #  "evo": {"t1": None, "t2": None, "use_rl_Mb": False,
        #          "plot": {"color": "black", "marker": "None", "markersize": 5, "linestyle": '-', "linewidth": 1.0}},
        #  "extevo": {"data": "self", "t1": None, "t2": None, "method": "intd1", "times": [100, 150, 200, 250, 300, 350, 400, 450, 500],
        #             "plot": {"color": "black", "marker": "x", "markersize": 5, "linestyle": ':', "linewidth": 0.5}}
        #  },
        #
        # {"sim": "DD2_M14971245_M0_SR", "v_n_x": "Mb", "v_n_y": "Jfinal",
        #  "plot": {"marker": "o", "mfc": "slateblue", "mec": 'black', "markersize": 8, "linestyle": 'None', "label": None},
        #  "evo": {},#{"t1": None, "t2": None, "use_rl_Mb": False,
        #          #"plot": {"color": "black", "marker": "None", "markersize": 5, "linestyle": '-', "linewidth": 1.0}},
        #  "extevo": {}#{"data": "self", "t1": None, "t2": None, "method": "intd1", "times": [100, 150, 200, 250, 300, 350, 400, 450, 500],
        #             #"plot": {"color": "black", "marker": "x", "markersize": 5, "linestyle": ':', "linewidth": 0.5}}
        #  },
        #
        # {"sim": "DD2_M15091235_M0_LK_SR", "v_n_x": "Mb", "v_n_y": "Jfinal",
        #  "plot": {"marker": "v", "mfc": "blueviolet", "mec": 'black', "markersize": 8, "linestyle": 'None', "label": None},
        #  "evo": {},#{"t1": None, "t2": 90, "use_rl_Mb": False,
        #          #"plot": {"color": "black", "marker": "None", "markersize": 5, "linestyle": '-', "linewidth": 1.0}},
        #  "extevo":{} #{"data": "self", "t1": None, "t2": None, "method": "intd1", "times": [100, 150, 200, 250, 300, 350, 400, 450, 500],
        #             #"plot": {"color": "black", "marker": "x", "markersize": 5, "linestyle": ':', "linewidth": 0.5}}
        #  },
        #
        # # --- LS220
        # {"sim": "LS220_M14691268_M0_LK_SR", "v_n_x": "Mb", "v_n_y": "Jfinal",
        #  "plot": {"marker": "v", "mfc": "red", "mec": 'black', "markersize": 8, "linestyle": 'None', "label": None},
        #  "evo": {"t1": None, "t2": 90, "use_rl_Mb": False,
        #          "plot": {"color": "black", "marker": "None", "markersize": 5, "linestyle": '-', "linewidth": 1.0}},
        #  "extevo": {}
        #  # {"data": "self", "t1": None, "t2": None, "method": "intd1", "times": [100, 150, 200, 250, 300, 350, 400, 450, 500],
        #  # "plot": {"color": "black", "marker": "x", "markersize": 5, "linestyle": ':', "linewidth": 0.5}}
        #  },
        #
        # # --- SFHo
        # {"sim": "SFHo_M11461635_M0_LK_SR", "v_n_x": "Mb", "v_n_y": "Jfinal",
        #  "plot": {"marker": "v", "mfc": "red", "mec": 'black', "markersize": 8, "linestyle": 'None', "label": None},
        #  "evo": {"t1": None, "t2": 50, "use_rl_Mb": False,
        #          "plot": {"color": "black", "marker": "None", "markersize": 5, "linestyle": '-', "linewidth": 1.0}},
        #  "extevo": {"data": "self", "t1": None, "t2": None, "method": "intd1", "times": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        #     "plot": {"color": "black", "marker": "x", "markersize": 5, "linestyle": ':', "linewidth": 0.5}}
        #  },
        #
        # # --- SLy4
        # {"sim": "SLy4_M11461635_M0_LK_SR", "v_n_x": "Mb", "v_n_y": "Jfinal",
        #  "plot": {"marker": "v", "mfc": "magenta", "mec": 'black', "markersize": 8, "linestyle": 'None', "label": None},
        #  "evo": {"t1": None, "t2": 50, "use_rl_Mb": False, "plot": {"color": "black", "marker": "None", "markersize": 5, "linestyle": '-', "linewidth": 1.0}},
        #  "extevo": {"data": "self", "t1": None, "t2": None, "method": "intd1",
        #             "times": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        #             "plot": {"color": "black", "marker": "x", "markersize": 5, "linestyle": ':', "linewidth": 0.5}}
        #  }

    ]

    # for t in task: t["plot"]["label"] = r"\texttt{" + t["sim"].replace('_', '\_') + "}"
    # for t in task:
    #     if t["sim"] == "LS220_M13641364_M0_LK_SR_restart":
    #         t["label"] = t["label"].replace("\_restart", "")

    plot_dic = {
        "figsize": (4.2, 3.6),#(3.6, 4.2),
        "type": "long",
        "EOS": "BLh",
        "plot_rns": True,
        "xmin": 3.5, "xmax": 6.2,#"xmin": 5.5, "xmax": 6.0,#"xmin": 4.5, "xmax": 6.0,
        "ymin": 2.85, "ymax": 3.00,#"ymin": 2.9, "ymax": 3.2,#"ymin": 2.8, "ymax": 3.2,
        "use_tov_for_ax2": True,
        "xscale": "linear", "yscale": "linear",
        "xlabel": r"$J\ [G\, c^{-1} M_\odot^2]$",
        "ylabel": r"$M_b\ [M_\odot]$",
        "legend": {"fancybox": False, "loc": 'lower right',
                   # "bbox_to_anchor": (0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "title": "BLh q=1.00 (SR)",
        "figname": __outplotdir__ + "secular_j_mb_RNS_{}.png".format("blh"),
        "plot_fill_disk":   {},#{"color":"deepskyblue", "alpha":0.3, "zorder":-2, "label":"Disk ejecta"},
        "plot_fill_remnant":{},#{"color":"cornflowerblue", "alpha":0.3, "zorder":-2, "label":"Remnant ejecta"},
        "split_legends":False,
        "tight_layout": True,
        "annotate":False,
        "savepdf":True,
        "fontsize":12
    }
    plot_j0_mb_RNS_j3D_m3D_evo(task, plot_dic)

    # ---- BLh ----

    plot_dic_blh = copy.deepcopy(plot_dic)
    plot_dic_blh["EOS"] = "BLh"
    plot_dic_blh["xmin"], plot_dic_blh["xmax"] = 3.5, 6.5
    plot_dic_blh["ymin"], plot_dic_blh["ymax"] = 2.8, 3.1
    plot_dic_blh["figname"] = __outplotdir__ + "total_j_mb_RNS_{}.png".format("BLh")

    #plot_j0_mb_RNS_j3D_m3D_evo(task, plot_dic_blh)

    # ---- LS220 ----

    plot_dic_blh = copy.deepcopy(plot_dic)
    plot_dic_blh["EOS"] = "LS220"
    plot_dic_blh["xmin"], plot_dic_blh["xmax"] = 3.5, 6.5
    plot_dic_blh["ymin"], plot_dic_blh["ymax"] = 2.8, 3.1
    plot_dic_blh["figname"] = __outplotdir__ + "total_j_mb_RNS_{}.png".format("LS220")

    #plot_j0_mb_RNS_j3D_m3D_evo(task, plot_dic_blh)

    # ---- SFHo ----

    plot_dic_blh = copy.deepcopy(plot_dic)
    plot_dic_blh["EOS"] = "SFHo"
    plot_dic_blh["xmin"], plot_dic_blh["xmax"] = 3.5, 6.5
    plot_dic_blh["ymin"], plot_dic_blh["ymax"] = 2.7, 2.9
    plot_dic_blh["figname"] = __outplotdir__ + "total_j_mb_RNS_{}.png".format("SFHo")

    #plot_j0_mb_RNS_j3D_m3D_evo(task, plot_dic_blh)

    # ---- SLy4 ----

    plot_dic_blh = copy.deepcopy(plot_dic)
    plot_dic_blh["EOS"] = "SLy4"
    plot_dic_blh["xmin"], plot_dic_blh["xmax"] = 3.5, 6.5
    plot_dic_blh["ymin"], plot_dic_blh["ymax"] = 2.7, 2.9
    plot_dic_blh["figname"] = __outplotdir__ + "total_j_mb_RNS_{}.png".format("SLy4")

    #plot_j0_mb_RNS_j3D_m3D_evo(task, plot_dic_blh)




''' --- iteration 2 ---'''

if __name__ == "__main__":

    ''' --- --- J0 Mb from init data with RNS --- --- '''
    #task_plot_j0_mb_RNS_j3D_m3D_evo_2()

    #exit(0)

if __name__ == "__main__":

    ''' --- --- J0 Mb from init data --- --- '''
    # task_plot_j0_mb_init_data()

    ''' --- --- J0 Mb from init data with RNS --- --- '''
    # task_plot_j0_mb_with_rns()

    ''' --- --- J0 Mb from init data with RNS --- --- '''
    # task_plot_j0_mb_RNS_j3D_M3D_final()

    ''' --- --- J0 Mb from init data with RNS --- --- '''
    # task_plot_j0_mb_RNS_j3D_M3D_mej_final()

    ''' --- --- J0 Mb from init data with RNS --- --- '''
    #task_plot_j0_mb_RNS_j3D_m3D_evo()

    ''' --- Mb/M_rns vs Mej_sec --- '''
    # task_plot_mbsupramss_sec_ev_mass()
    # task_plot_mbsupramss_sec_ev_time()