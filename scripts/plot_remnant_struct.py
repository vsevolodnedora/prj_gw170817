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
from preanalysis import LOAD_INIT_DATA
from outflowed import EJECTA_PARS
from preanalysis import LOAD_ITTIME
from plotting_methods import PLOT_MANY_TASKS
from profile import LOAD_PROFILE_XYXZ, LOAD_RES_CORR, LOAD_DENSITY_MODES, MAINMETHODS_STORE, MAINMETHODS_STORE_XYXZ
from utils import Paths, Lists, Labels, Constants, Printcolor, UTILS, Files, PHYSICS

from data import ADD_METHODS_ALL_PAR
from comparison import TWO_SIMS, THREE_SIMS
from settings import resolutions
from scidata import units # for rc -> km

from uutils import *

import model_sets.models as md

__outplotdir__ = "../figs/all3/plot_remnant_struct/"
if not os.path.isdir(__outplotdir__):
    os.mkdir(__outplotdir__)


''' ----------------------- MODULES ------------------------- '''

def plot_rho_max(tasks, plotdic):
    fig = plt.figure(figsize=plotdic["figsize"])
    ax = fig.add_subplot(111)
    #
    # labels

    for task in tasks:
        if task["type"] == "all" or task["type"] == plotdic["type"]:
            o_data = ADD_METHODS_ALL_PAR(task["sim"])
            table = o_data.get_collated_data(v_n=task["v_n"] + ".asc")
            it, times, data = table[:,0], table[:,1], table[:,2]
            #
            if task["v_n"] == "dens.norm1":
                data = data * Constants.volume_constant ** 3
                t0, m0 = 0., o_data.get_initial_data_par("Mb")
                times = np.insert(times, 0, t0)
                data = np.insert(data, 0, m0)

            #
            times = times * Constants.time_constant / 1000
            tmerg = o_data.get_par("tmerg")

            times = (times - tmerg) * 1e3  # ms
            ax.plot(times, data, color=task["color"], ls=task["ls"], lw=task["lw"], alpha=task["alpha"],
                    label=task["label"])

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

def plot_dens_modes_2D(tasks, plotdic):

    fig = plt.figure(figsize=plotdic["figsize"])
    ax = fig.add_subplot(111)
    #
    # labels

    for task in tasks:
        if task["type"] == "all" or task["type"] == plotdic["type"]:

            fpath = Paths.ppr_sims + task["sim"] + "/" + task["fpath"]
            if not os.path.isfile(fpath):
                print("python /data01/numrel/vsevolod.nedora/bns_ppr_tools/slices.py -s {} -t dm --it all --rl 3 --plane xy xz".format(task["sim"]))
                os.system("python /data01/numrel/vsevolod.nedora/bns_ppr_tools/slices.py -s {} -t dm --it all --rl 3 --plane xy xz".format(task["sim"]))


            o_dm = LOAD_DENSITY_MODES(task["sim"])
            o_dm.gen_set['fname'] = fpath
            #
            times = o_dm.get_grid("times")
            mags = o_dm.get_data(task["m"], "int_phi_r")
            mags = np.abs(mags)
            if task["norm_to_m"] != None:
                # print('Normalizing')
                norm_int_phi_r1d = o_dm.get_data(task["norm_to_m"], 'int_phi_r')
                # print(norm_int_phi_r1d); exit(1)
                mags = mags / abs(norm_int_phi_r1d)[0]
            #
            o_par = ADD_METHODS_ALL_PAR(task["sim"])
            tmerg = o_par.get_par("tmerg")
            times = (times - tmerg) * 1e3  # ms
            #

            ax.plot(times, mags, color=task["color"], ls=task["ls"], lw=task["lw"], alpha=task["alpha"],
                    label=task["label"])

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
def plot_dens_modes_2D_2(tasks, plotdic):

    fig = plt.figure(figsize=plotdic["figsize"])
    ax = fig.add_subplot(111)
    #
    # labels

    for task in tasks:
        if plotdic["type"] == "all" or task["type"] == plotdic["type"]:

            fpath = Paths.ppr_sims + task["sim"] + "/" + task["fpath"]
            if not os.path.isfile(fpath):
                print("python /data01/numrel/vsevolod.nedora/bns_ppr_tools/slices.py -s {} -t dm --it all --rl 3 --plane xy xz".format(task["sim"]))
                os.system("python /data01/numrel/vsevolod.nedora/bns_ppr_tools/slices.py -s {} -t dm --it all --rl 3 --plane xy xz".format(task["sim"]))


            o_dm = LOAD_DENSITY_MODES(task["sim"])
            o_dm.gen_set['fname'] = fpath
            #
            times = o_dm.get_grid("times")
            mags = o_dm.get_data(task["m"], "int_phi_r")
            mags = np.abs(mags)
            if task["norm_to_m"] != None:
                # print('Normalizing')
                norm_int_phi_r1d = o_dm.get_data(task["norm_to_m"], 'int_phi_r')
                # print(norm_int_phi_r1d); exit(1)
                mags = mags / abs(norm_int_phi_r1d)[0]
            #
            o_par = ADD_METHODS_ALL_PAR(task["sim"])
            tmerg = o_par.get_par("tmerg")
            times = (times - tmerg) * 1e3  # ms
            #
            # print("Tmerg:{}".format(tmerg))
            # print("times", times)
            # print("limit: {} ".format((task["t2pm"]*1.e-3+tmerg)))
            #
            if "t1" in task.keys():
                mags = mags[times >= task["t1"]*1.e-3]
                times = times[times >= task["t1"]*1.e-3]
            if "t2" in task.keys():
                mags = mags[times < task["t2"]*1.e-3]
                times = times[times < task["t2"]*1.e-3]
            if "t2pm" in task.keys():
                mags = mags[times < task["t2pm"]]
                times = times[times < task["t2pm"]]
            # if "t1pm" in task.keys():
            #     mags = mags[times > (task["t2pm"]*1.e-3+tmerg)]
            #     times = times[times > (task["t2pm"]*1.e-3+tmerg)]
            #
            if "mmean" in task.keys():
                # moving mean smoothing
                N = task["mmean"]

                cumsum, moving_aves = [0], []
                for i, x in enumerate(mags, 1):
                    cumsum.append(cumsum[i - 1] + x)
                    if i >= N:
                        moving_ave = (cumsum[i] - cumsum[i - N]) / N
                        # can do stuff with moving_ave here
                        moving_aves.append(moving_ave)
                mags = np.array(moving_aves)

                cumsum, moving_aves = [0], []
                for i, x in enumerate(times, 1):
                    cumsum.append(cumsum[i - 1] + x)
                    if i >= N:
                        moving_ave = (cumsum[i] - cumsum[i - N]) / N
                        # can do stuff with moving_ave here
                        moving_aves.append(moving_ave)
                times = np.array(moving_aves)

            if "int" in task.keys():

                n_segments = int(task["int"][0])
                methods = task["int"][1]
                assert n_segments == len(methods)

                t_segments = np.array_split(times, n_segments)
                m_segments = np.array_split(mags, n_segments)

                if len(t_segments) < 3: raise ValueError("segment is too small for n_segment:{}".format(n_segments))

                all_times, all_mags = [], []
                for i in range(n_segments):
                    i_grid_times = np.mgrid[t_segments[i][0]:t_segments[i][-1]:100j]
                    if methods[i] == "unispline":
                        int_mags = interpolate.InterpolatedUnivariateSpline(t_segments[i], m_segments[i], k=2)(i_grid_times)
                        all_times.append(i_grid_times)
                        all_mags.append(int_mags)
                    elif methods[i] == "linear":
                        int_mags = interpolate.InterpolatedUnivariateSpline(t_segments[i], m_segments[i], k=1)(i_grid_times)
                        all_times.append(i_grid_times)
                        all_mags.append(int_mags)
                times = np.concatenate(all_times)
                mags = np.concatenate(all_mags)

                # # intepolation
                # grid_times, grid_mags = np.mgrid[times[0]:times[-1]:100j], np.mgrid[mags.min():mags.max():100j]
                # int_mags = interpolate.InterpolatedUnivariateSpline(times, mags,k=3)(grid_times)
                # #int_mags = interpolate.interp1d(times, mags, kind="cubic")(grid_times)
                # times = grid_times
                # mags = int_mags

            # print (task["sim"], times)
            # print (task["sim"], mags); exit(1)
            ax.plot(times, mags, **task["plot"])

    ax.set_yscale(plotdic["yscale"])
    ax.set_xscale(plotdic["xscale"])

    ax.set_xlabel(plotdic["xlabel"], fontsize=plotdic["fontsize"])  # , fontsize=11)
    ax.set_ylabel(plotdic["ylabel"], fontsize=plotdic["fontsize"])  # , fontsize=11)

    ax.set_xlim(plotdic["xmin"], plotdic["xmax"])
    ax.set_ylim(plotdic["ymin"], plotdic["ymax"])
    #
    ax.tick_params(axis='both', which='both', labelleft=True,
                   labelright=False, tick1On=True, tick2On=True,
                   labelsize=plotdic["fontsize"],
                   direction='in',
                   bottom=True, top=True, left=True, right=True)
    ax.minorticks_on()
    # #
    ax.set_title(plotdic["title"], fontsize=plotdic["fontsize"])
    # #
    # #
    if len(plotdic["modelegend"].keys())>0:
        lines = plotdic["modelegend"]["modes"]
        for line in lines:
            ax.plot([-10, -10], [-20, -20], **line)
        #
        han, lab = ax.get_legend_handles_labels()
        ax.add_artist(ax.legend(han[:-1 * len(lines)], lab[:-1 * len(lines)], **plotdic["legend"]))
        #
        ax.add_artist(ax.legend(han[len(han)-len(lines):], lab[len(lab)-len(lines):], **plotdic["modelegend"]["legend"]))
    else:
        ax.legend(**plotdic["legend"])

    plt.tight_layout()
    #
    print("plotted: \n")
    print(plotdic["figname"])
    plt.savefig(plotdic["figname"], dpi=128)
    if plotdic["savepdf"]: plt.savefig(plotdic["figname"].replace(".png", ".pdf"))
    plt.close()

def plot_center_of_mass_r(tasks, plotdic):
    fig = plt.figure(figsize=plotdic["figsize"])
    ax = fig.add_subplot(111)
    #
    # labels

    for task in tasks:
        if task["type"] == "all" or task["type"] == plotdic["type"]:

            fpath = Paths.ppr_sims + task["sim"] + "/" + task["fpath"]
            if not os.path.isfile(fpath):
                print(
                    "python /data01/numrel/vsevolod.nedora/bns_ppr_tools/slices.py -s {} -t dm --it all --rl 3 --plane xy xz".format(
                        task["sim"]))
                os.system(
                    "python /data01/numrel/vsevolod.nedora/bns_ppr_tools/slices.py -s {} -t dm --it all --rl 3 --plane xy xz".format(
                        task["sim"]))

            o_dm = LOAD_DENSITY_MODES(task["sim"])
            o_dm.gen_set['fname'] = fpath
            #
            times = o_dm.get_grid("times")
            xc = o_dm.get_grid("xc")
            yc = o_dm.get_grid("yc")
            rc = np.sqrt(xc**2 + yc**2)
            #
            o_par = ADD_METHODS_ALL_PAR(task["sim"])
            tmerg = o_par.get_par("tmerg")
            times = (times - tmerg) * 1e3  # ms
            #

            ax.plot(times, rc, color=task["color"], ls=task["ls"], lw=task["lw"], alpha=task["alpha"],
                    label=task["label"])

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

def plot_tot_remnant_mass_evo(tasks, plotdic):

    fig = plt.figure(figsize=plotdic["figsize"])
    ax = fig.add_subplot(111)
    #
    # labels

    for task in tasks:
        if task["type"] == "all" or task["type"] == plotdic["type"]:
            o_data = ADD_METHODS_ALL_PAR(task["sim"])
            it, times, data = o_data.get_remnant_mass()
            print(data)
            # if np.isnan(data).any():
            #     print("python /data01/numrel/vsevolod.nedora/bns_ppr_tools/profile.py -s {} -t mass --it all".format(task["sim"]))
            #     os.system("python /data01/numrel/vsevolod.nedora/bns_ppr_tools/profile.py -s {} -t mass --it all".format(task["sim"]))
            # if len(table) < 2: raise ValueError()
            # print(table.shape)
            # it, times, data = table[:, 0], table[:, 1], table[:, 2]

            #
            # times = times * Constants.time_constant / 1000
            tmerg = o_data.get_par("tmerg")

            times = (times - tmerg) * 1e3  # ms
            if "t1" in task.keys():
                data = data[times >= task["t1"]]
                times = times[times >= task["t1"]]
            print("{} min:{} max:{} len:{}".format(task["sim"], data.min(), data.max(), len(data)))
            ax.plot(times, data, color=task["color"], ls=task["ls"], lw=task["lw"], alpha=task["alpha"],
                    label=task["label"])

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

def plot_mass_ave_val_evo(tasks, plotdic):

    fig = plt.figure(figsize=plotdic["figsize"])
    ax = fig.add_subplot(111)
    #
    # labels

    for task in tasks:
        if task["type"] == "all" or task["type"] == plotdic["type"]:
            o_data = ADD_METHODS_ALL_PAR(task["sim"])
            table = o_data.get_disk_mass_ave_par_evo(v_n=task["v_n"], mask=task["mask"])
            it, times, data = table[:, 0], table[:, 1], table[:, 2]
            #

            #
            # times = times * Constants.time_constant / 1000
            tmerg = o_data.get_par("tmerg")
            # print(data)
            # if task["v_n"] == "theta": data = 90 - (data * 180 / np.pi)
            print(data[-1])
            times = (times - tmerg) * 1e3  # ms
            ax.plot(times, data, color=task["color"], ls=task["ls"], lw=task["lw"], alpha=task["alpha"],
                    label=task["label"])

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

def plot_remnant_timecorr(task, plotdic):
    sim = task["sim"]  # "BLh_M13641364_M0_LK_SR"
    v_n = task["v_n"]
    mask = task["mask"]

    if not os.path.isdir(plotdic["outdir"]):
        os.mkdir(plotdic["outdir"])


    o_data = ADD_METHODS_ALL_PAR(sim)
    times, bins, masses = o_data.get_disk_timecorr(v_n, mask=mask)
    tmerg = o_data.get_par("tmerg")
    times = (times - tmerg) * 1.e3 # ms
    #
    if v_n == "theta":
        bins = 90 - (bins * 180 / np.pi)
    if v_n == "hu_0":
        bins = bins * -1.

    #
    print("x: {} -> {}".format(times.min(), times.max()))
    print("y: {} -> {}".format(bins.min(), bins.max()))
    #

    print("mass", np.sum(masses))
    if task["normalize"]:
        masses = masses / np.sum(masses)
        masses = np.maximum(masses, 1e-10)
    #
    print(times.shape)
    print(bins.shape)
    print(masses.shape)

    # -------------------------------------- PLOTTING
    fig = plt.figure(figsize=(4., 2.5))
    ax = fig.add_subplot(111)
    #
    norm = LogNorm(plotdic["vmin"], plotdic["vmax"])
    im = ax.pcolormesh(times, bins, masses, norm=norm,
                       cmap=plotdic["cmap"])  # , vmin=dic["vmin"], vmax=dic["vmax"])
    im.set_rasterized(True)
    im.cmap.set_over(plotdic['set_under'])
    im.cmap.set_over(plotdic['set_over'])
    #
    ax.set_yscale(plotdic["yscale"])
    ax.set_xscale(plotdic["xscale"])
    #
    ax.set_xlabel(plotdic["xlabel"], fontsize=11)
    ax.set_ylabel(plotdic["ylabel"], fontsize=11)
    #
    ax.set_xlim(plotdic["xmin"], plotdic["xmax"])
    ax.set_ylim(plotdic["ymin"], plotdic["ymax"])
    #
    if "text" in plotdic.keys():
        plotdic["text"]["transform"] = ax.transAxes
        ax.text(**plotdic["text"])
    #
    ax.tick_params(axis='both', which='both', labelleft=True,
                   labelright=False, tick1On=True, tick2On=True,
                   labelsize=12,
                   direction='in',
                   bottom=True, top=True, left=True, right=True)
    ax.minorticks_on()
    #
    if "title" in plotdic.keys(): ax.set_title(plotdic["title"])
    #
    clb = fig.colorbar(im, ax=ax)
    plt.tight_layout()
    clb.ax.set_title(plotdic["clabel"], fontsize=11)
    clb.ax.tick_params(labelsize=11)
    #
    print("plotted: \n")
    print(plotdic["outdir"] + plotdic["figname"])
    plt.savefig(plotdic["outdir"] + plotdic["figname"], dpi=128)
    plt.close()

def plot_total_angular_momentum_colormesh(task, plotdic):

    sim = task["sim"]  # "BLh_M13641364_M0_LK_SR"
    v_n = task["v_n"]

    if not os.path.isdir(plotdic["outdir"]):
        os.mkdir(plotdic["outdir"])

    o_data = ADD_METHODS_ALL_PAR(sim)
    iterations, times, rc, all_j, all_jf, all_m, all_i = o_data.get_enclosed_mj(reshape=True)
    tmerg = o_data.get_par("tmerg")
    times = (times - tmerg) * 1.e3  # ms
    #

    print("x: {} -> {}".format(times.min(), times.max()))
    print("y: {} -> {}".format(rc.min(), rc.max()))
    #
    if v_n == "J": arr = all_j
    elif v_n == "Jflux": arr = all_jf
    else: raise NameError("v_n:{} not recognized".format(v_n))

    #
    arr = np.maximum(arr, 1e-10)
    #
    print(times.shape)
    print(rc.shape)
    print(all_j.shape)

    # -------------------------------------- PLOTTING
    fig = plt.figure(figsize=plotdic["figsize"])
    ax = fig.add_subplot(111)
    #
    norm = LogNorm(plotdic["vmin"], plotdic["vmax"])
    # # print(rc)
    # rc = units.conv_length(units.cactus, units.cgs, 1.) # cm
    # rc = rc * 1.e-5 # cm -> km
    # # rc = rc * 1.e-3 # m -> rm
    # print(rc)
    # exit(1)

    im = ax.pcolormesh(times, rc * constant_length, arr.T, norm=norm, cmap=plotdic["cmap"])  # , vmin=dic["vmin"], vmax=dic["vmax"])
    im.set_rasterized(True)
    im.cmap.set_over(plotdic['set_under'])
    im.cmap.set_over(plotdic['set_over'])
    #
    ax.set_yscale(plotdic["yscale"])
    ax.set_xscale(plotdic["xscale"])
    #
    ax.set_xlabel(plotdic["xlabel"], fontsize=plotdic["fontsize"])
    ax.set_ylabel(plotdic["ylabel"], fontsize=plotdic["fontsize"])
    #
    ax.set_xlim(plotdic["xmin"], plotdic["xmax"])
    ax.set_ylim(plotdic["ymin"], plotdic["ymax"])
    #
    if "text" in plotdic.keys():
        plotdic["text"]["transform"] = ax.transAxes
        ax.text(**plotdic["text"])
    #
    ax.tick_params(axis='both', which='both', labelleft=True,
                   labelright=False, tick1On=True, tick2On=True,
                   labelsize=plotdic["fontsize"],
                   direction='in',
                   bottom=True, top=True, left=True, right=True)
    ax.minorticks_on()
    #
    if "title" in plotdic.keys(): ax.set_title(plotdic["title"], fontsize=plotdic["fontsize"])
    #
    clb = fig.colorbar(im, ax=ax)
    plt.tight_layout()
    clb.ax.set_title(plotdic["clabel"], fontsize=plotdic["fontsize"])
    clb.ax.tick_params(labelsize=plotdic["fontsize"])
    clb.ax.minorticks_off()
    #
    print("plotted: \n")
    figname = plotdic["outdir"] + plotdic["figname"]
    print(figname)
    plt.savefig(figname, dpi=128)
    if plotdic["savepdf"]: plt.savefig(figname.replace(".png", ".pdf"))
    plt.close()

def plot_total_angular_momentum(tasks, plotdic):
    fig = plt.figure(figsize=plotdic["figsize"])
    ax = fig.add_subplot(111)

    for task in tasks:
        if plotdic["type"] == "all" or task["type"] == plotdic["type"]:

            o_data = ADD_METHODS_ALL_PAR(task["sim"])
            if task["v_n"] == "J":
                iterations, times, array, tot_jf, tot_mb = \
                    o_data.get_total_enclosed_j_jf_mb(task["rext"])

            elif task["v_n"] == "Jflux":
                iterations, times, tot_j, array, tot_mb = \
                    o_data.get_total_enclosed_j_jf_mb(task["rext"])
            elif task["v_n"] == "dJ/dt":
                times, array = o_data.get_total_enclosed_djdt(task["intmethod"], task["rext"])
                times = times[array<0]
                array = -1.*array[array<0]
                # array = np.abs(array)
                #
                # if len(array) % 3 == 0:
                #     array = np.mean(array.reshape(-1, 3), axis=1).flatten()
                #     times = np.mean(times.reshape(-1, 3), axis=1).flatten()
                # elif len(array) % 4 == 0:
                #     array = np.mean(array.reshape(-1, 4), axis=1).flatten()
                #     times = np.mean(times.reshape(-1, 4), axis=1).flatten()
                # elif len(array) % 5 == 0:
                #     array = np.mean(array.reshape(-1, 5), axis=1).flatten()
                #     times = np.mean(times.reshape(-1, 5), axis=1).flatten()
                # elif len(array) % 6 == 0:
                #     array = np.mean(array.reshape(-1, 6), axis=1).flatten()
                #     times = np.mean(times.reshape(-1, 6), axis=1).flatten()
            elif task["v_n"] == "dJ/dt-Jflux":
                times, djdt = o_data.get_total_enclosed_djdt()
                iterations, times, tot_j, tot_jflux, tot_mb = \
                    o_data.get_total_enclosed_j_jf_mb()
                #
                times = times
                array = djdt - tot_jflux

                times = times[array>0]
                array = array[array>0]
            elif task["v_n"] == "-(dJ/dt-Jflux)":
                times, djdt = o_data.get_total_enclosed_djdt()
                iterations, times, tot_j, tot_jflux, tot_mb = \
                    o_data.get_total_enclosed_j_jf_mb()
                #
                times = times
                array = djdt - tot_jflux

                times = times[array < 0]
                array = -1. * array[array < 0]
            elif task["v_n"] == "-(dJ/dt+Jflux)":
                times, djdt = o_data.get_total_enclosed_djdt()
                iterations, times, tot_j, tot_jflux, tot_mb = \
                    o_data.get_total_enclosed_j_jf_mb()
                #
                times = times
                array = -1. * (djdt + tot_jflux)
                print("djdt:")
                print(djdt)
                print('\n')
                print("jflux")
                print(tot_jflux)
                # exit(1)
                times = times[array > 0]
                array = array[array > 0]
            elif task["v_n"] == "(dJ/dt-Jflux)/(dJ/dt)":
                times, djdt = o_data.get_total_enclosed_djdt()
                iterations, times, tot_j, tot_jflux, tot_mb = \
                    o_data.get_total_enclosed_j_jf_mb()
                #
                times = times
                array = (djdt + tot_jflux)/djdt
                #
                times = times[array>0]
                array = array[array>0]
            elif task["v_n"] == "(dJ/dt+Jflux)/(dJ/dt)":
                times, djdt = o_data.get_total_enclosed_djdt()
                iterations, times, tot_j, tot_jflux, tot_mb = \
                    o_data.get_total_enclosed_j_jf_mb()
                #
                times = times
                array = (djdt + tot_jflux) / djdt
                #
                times = times[array > 0]
                array = array[array > 0]
            else:
                raise NameError("v_n {} is not recognized".format(task["v_n"]))

            tmerg = o_data.get_par("tmerg")
            times = (times - tmerg) * 1.e3  # ms
            #
            if "tmin" in task.keys():
                array = array[times > task["tmin"]]
                times = times[times > task["tmin"]]
            if "tmax" in task.keys():
                array = array[times <= task["tmax"]]
                times = times[times <= task["tmax"]]


            print(array)
            if "plot" in task.keys():
                ax.plot(times, array, **task["plot"])
            else:
                ax.plot(times, array, color=task["color"], ls=task["ls"], lw=task["lw"], alpha=task["alpha"],
                        label=task["label"])

    if not "fontsize" in plotdic.keys(): plotdic["fontsize"] = 12

    if len(plotdic["multilegend"].keys())>0:
        lines = plotdic["multilegend"]["lines"]
        for line in lines:
            ax.plot([-10, -10], [-20, -20], **line)
        #
        han, lab = ax.get_legend_handles_labels()
        ax.add_artist(ax.legend(han[:-1 * len(lines)], lab[:-1 * len(lines)], **plotdic["legend"]))
        #
        ax.add_artist(ax.legend(han[len(han)-len(lines):], lab[len(lab)-len(lines):], **plotdic["multilegend"]["legend"]))
    else:
        ax.legend(**plotdic["legend"])

    ax.set_yscale(plotdic["yscale"])
    ax.set_xscale(plotdic["xscale"])

    ax.set_xlabel(plotdic["xlabel"], fontsize=plotdic["fontsize"])  # , fontsize=11)
    ax.set_ylabel(plotdic["ylabel"], fontsize=plotdic["fontsize"])  # , fontsize=11)

    ax.set_xlim(plotdic["xmin"], plotdic["xmax"])
    ax.set_ylim(plotdic["ymin"], plotdic["ymax"])
    #
    ax.tick_params(axis='both', which='both', labelleft=True,
                   labelright=False, tick1On=True, tick2On=True,
                   labelsize=plotdic["fontsize"],
                   direction='in',
                   bottom=True, top=True, left=True, right=True)
    ax.minorticks_on()
    # #
    ax.set_title(plotdic["title"], fontsize=plotdic["fontsize"])
    # #

    #
    #ax.legend(**plotdic["legend"])

    plt.tight_layout()
    #

    print("plotted: \n")
    print(plotdic["figname"])
    plt.savefig(plotdic["figname"], dpi=128)
    if "savepdf" in plotdic.keys() and plotdic["savepdf"]: plt.savefig(plotdic["figname"].replace(".png", ".pdf"))
    plt.close()

""" ----------- TASKS ----------- """

def task_plot_rho_max():

    v_n = "rho.maximum"

    task = [
        {"sim": "BLh_M13641364_M0_LK_SR", "color": "black", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        # {"sim": "BLh_M11841581_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"NS", "v_n": v_n, "t": -1},
        {"sim": "BLh_M11461635_M0_LK_SR", "color": "black", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        {"sim": "BLh_M10651772_M0_LK_SR", "color": "black", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        # {"sim": "BLh_M10201856_M0_SR", "color": "black", "ls": "-", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
        # {"sim": "BLh_M10201856_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
        #
        {"sim": "DD2_M13641364_M0_SR", "color": "blue", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        # {"sim": "DD2_M13641364_M0_SR_R04", "color": "blue", "ls": "--", "lw": 0.7, "alpha": 1., "t1":60, "v_n": v_n, "t": -1},
        {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "blue", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        {"sim": "DD2_M14971245_M0_SR", "color": "blue", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        {"sim": "DD2_M15091235_M0_LK_SR", "color": "blue", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        #
        # {"sim": "LS220_M13641364_M0_SR", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "t": -1},
        {"sim": "LS220_M13641364_M0_LK_SR_restart", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "type": "short",
         "v_n": v_n},
        {"sim": "LS220_M14691268_M0_LK_SR", "color": "red", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        {"sim": "LS220_M11461635_M0_LK_SR", "color": "red", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n},
        {"sim": "LS220_M10651772_M0_LK_SR", "color": "red", "ls": ":", "lw": 0.7, "alpha": 1., "type": "short",
         "v_n": v_n},
        #
        {"sim": "SFHo_M13641364_M0_SR", "color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n},
        {"sim": "SFHo_M14521283_M0_SR", "color": "green", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n},
        {"sim": "SFHo_M11461635_M0_LK_SR", "color": "green", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        #
        {"sim": "SLy4_M13641364_M0_SR", "color": "magenta", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n},
        {"sim": "SLy4_M14521283_M0_SR", "color": "magenta", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n},
        # {"sim": "SLy4_M10201856_M0_LK_SR", "color": "orange", "ls": "-.", "lw": 0.7, "alpha": 1., "v_n": v_n, "t": -1},
        {"sim": "SLy4_M11461635_M0_LK_SR", "color": "magenta", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n}
    ]

    for t in task: t["label"] = r"\texttt{" + t["sim"].replace('_', '\_') + "}"
    for t in task:
        if t["sim"] == "LS220_M13641364_M0_LK_SR_restart":
            t["label"] = t["label"].replace("\_restart", "")

    plot_dic = {
        "type": "long",
        "figsize": (6., 2.5),
        "xmin": -20, "xmax": 110.,
        "ymin": 8e-4, "ymax": 4e-3,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "log",
        "xlabel": r"$t-t_{\rm merg}$ [ms]",
        "ylabel": r"$\rho_{\rm max}$",
        "legend": {"fancybox": False, "loc": 'center left',
                   "bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "rho_max_long.png"
    }

    plot_rho_max(task, plot_dic)

    plot_dic["type"] = "short"
    plot_dic["xmax"] = 40
    # plot_dic["ymin"], plot_dic["ymax"] = 2, 3.
    plot_dic["title"] = "Short-lived remnants"
    plot_dic["figname"] = __outplotdir__ + "rho_max_short.png"

    plot_rho_max(task, plot_dic)

    # --- dens_norm1 ---- #

    for t in task: t["v_n"] = "dens.norm1"

    plot_dic = {
        "type": "long",
        "figsize": (6., 2.5),
        "xmin": -25, "xmax": 100.,
        "ymin": 2., "ymax": 3.5,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "linear",
        "xlabel": r"$t-t_{\rm merg}$ [ms]",
        "ylabel": r"$M_b$",
        "legend": {"fancybox": False, "loc": 'center left',
                   "bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "title": r"Baryoinc mass from \texttt{dens.norm1}",
        "figname": __outplotdir__ + "dens_norm1_long.png"
    }

    plot_rho_max(task, plot_dic)

    plot_dic["type"] = "short"
    plot_dic["xmax"] = 40
    # plot_dic["ymin"], plot_dic["ymax"] = 2, 3.
    plot_dic["title"] = "Short-lived remnants"
    plot_dic["figname"] = __outplotdir__ + "dens_norm1_short.png"

    plot_rho_max(task, plot_dic)

def task_plot_dens_modes_2D():
    """
    none
    """
    ''' ---------------- rho modes ------------------- '''

    v_n = "rho_modes.h5"
    fpath = "slices/" + "rho_modes.h5"
    m = 1
    norm_to_m = 0
    task = [
        {"sim": "BLh_M13641364_M0_LK_SR", "color": "black", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
        # {"sim": "BLh_M11841581_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"NS", "v_n": v_n, "t": -1},
        {"sim": "BLh_M11461635_M0_LK_SR", "color": "black", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
        {"sim": "BLh_M10651772_M0_LK_SR", "color": "black", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
        # {"sim": "BLh_M10201856_M0_SR", "color": "black", "ls": "-", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
        # {"sim": "BLh_M10201856_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
        #
        {"sim": "DD2_M13641364_M0_SR", "color": "blue", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
        # {"sim": "DD2_M13641364_M0_SR_R04", "color": "blue", "ls": "--", "lw": 0.7, "alpha": 1., "t1":60, "v_n": v_n, "t": -1},
        {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "blue", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
        {"sim": "DD2_M14971245_M0_SR", "color": "blue", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
        {"sim": "DD2_M15091235_M0_LK_SR", "color": "blue", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
        #
        # {"sim": "LS220_M13641364_M0_SR", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "t": -1},
        {"sim": "LS220_M13641364_M0_LK_SR_restart", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "type": "short",
         "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
        {"sim": "LS220_M14691268_M0_LK_SR", "color": "red", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
        {"sim": "LS220_M11461635_M0_LK_SR", "color": "red", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
        {"sim": "LS220_M10651772_M0_LK_SR", "color": "red", "ls": ":", "lw": 0.7, "alpha": 1., "type": "short",
         "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
        #
        {"sim": "SFHo_M13641364_M0_SR", "color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
        {"sim": "SFHo_M14521283_M0_SR", "color": "green", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
        {"sim": "SFHo_M11461635_M0_LK_SR", "color": "green", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
        #
        {"sim": "SLy4_M13641364_M0_SR", "color": "magenta", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
        {"sim": "SLy4_M14521283_M0_SR", "color": "magenta", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
        # {"sim": "SLy4_M10201856_M0_LK_SR", "color": "orange", "ls": "-.", "lw": 0.7, "alpha": 1., "v_n": v_n, "t": -1},
        {"sim": "SLy4_M11461635_M0_LK_SR", "color": "magenta", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m}
    ]

    for t in task: t["label"] = r"\texttt{" + t["sim"].replace('_', '\_') + "}"
    for t in task:
        if t["sim"] == "LS220_M13641364_M0_LK_SR_restart":
            t["label"] = t["label"].replace("\_restart", "")

    def_plot_dic = {
        "figsize": (6., 2.5),
        "type": "long",
        "xmin": 0, "xmax": 100,
        "ymin": 1e-4, "ymax": 5e-1,
        "normalize": True,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "log",
        "xlabel": r"$t-t_{\rm merg}$ [ms]",
        "ylabel": r'$C_1(\rho)/C_0(\rho)$',
        "legend": {"fancybox": False, "loc": 'center left',
                   "bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "total_rho_mode_{}_long.png".format(m)
    }

    plot_dens_modes_2D(task, def_plot_dic)

    ''' ------------- dens modes ------------------- '''

    # "profiles/" + "density_modes_lap15.h5"
    v_n = "profiles/" + "density_modes_lap15.h5"
    for t in task: t["fpath"] = v_n

    def_plot_dic["ymin"], def_plot_dic["ymax"] = 1e-3, 1e-1
    def_plot_dic["figname"] = __outplotdir__ + "total_dens_mode_{}_long.png".format(m)
    def_plot_dic["ylabel"] = r"$C_1(D)/C_0(D)$"
    plot_dens_modes_2D(task, def_plot_dic)

    #
    m = 2
    for t in task: t["m"] = 2
    v_n = "profiles/" + "density_modes_lap15.h5"
    for t in task: t["fpath"] = v_n

    def_plot_dic["figname"] = __outplotdir__ + "total_dens_mode_{}_long.png".format(m)
    def_plot_dic["ylabel"] = r"$C_2(D)/C_0(D)$"
    plot_dens_modes_2D(task, def_plot_dic)

def task_plot_center_mass_r_3D():

    # v_n = "rho_modes.h5"
    # fpath = "slices/" + "rho_modes.h5"

    v_n = "density_modes_lap15.h5"
    fpath = "profiles/" + "density_modes_lap15.h5"

    m = 1
    norm_to_m = 0
    task = [
        {"sim": "BLh_M13641364_M0_LK_SR", "color": "black", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
        # {"sim": "BLh_M11841581_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"NS", "v_n": v_n, "t": -1},
        {"sim": "BLh_M11461635_M0_LK_SR", "color": "black", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
        {"sim": "BLh_M10651772_M0_LK_SR", "color": "black", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
        # {"sim": "BLh_M10201856_M0_SR", "color": "black", "ls": "-", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
        # {"sim": "BLh_M10201856_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
        #
        {"sim": "DD2_M13641364_M0_SR", "color": "blue", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
        # {"sim": "DD2_M13641364_M0_SR_R04", "color": "blue", "ls": "--", "lw": 0.7, "alpha": 1., "t1":60, "v_n": v_n, "t": -1},
        {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "blue", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
        {"sim": "DD2_M14971245_M0_SR", "color": "blue", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
        {"sim": "DD2_M15091235_M0_LK_SR", "color": "blue", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
        #
        # {"sim": "LS220_M13641364_M0_SR", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "t": -1},
        {"sim": "LS220_M13641364_M0_LK_SR_restart", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "type": "short",
         "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
        {"sim": "LS220_M14691268_M0_LK_SR", "color": "red", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
        {"sim": "LS220_M11461635_M0_LK_SR", "color": "red", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
        {"sim": "LS220_M10651772_M0_LK_SR", "color": "red", "ls": ":", "lw": 0.7, "alpha": 1., "type": "short",
         "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
        #
        {"sim": "SFHo_M13641364_M0_SR", "color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
        {"sim": "SFHo_M14521283_M0_SR", "color": "green", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
        {"sim": "SFHo_M11461635_M0_LK_SR", "color": "green", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
        #
        {"sim": "SLy4_M13641364_M0_SR", "color": "magenta", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
        {"sim": "SLy4_M14521283_M0_SR", "color": "magenta", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
        # {"sim": "SLy4_M10201856_M0_LK_SR", "color": "orange", "ls": "-.", "lw": 0.7, "alpha": 1., "v_n": v_n, "t": -1},
        {"sim": "SLy4_M11461635_M0_LK_SR", "color": "magenta", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m}
    ]

    for t in task: t["label"] = r"\texttt{" + t["sim"].replace('_', '\_') + "}"
    for t in task:
        if t["sim"] == "LS220_M13641364_M0_LK_SR_restart":
            t["label"] = t["label"].replace("\_restart", "")

    def_plot_dic = {
        "figsize": (6., 2.5),
        "type": "long",
        "xmin": 0, "xmax": 100,
        "ymin": 0, "ymax": 10,
        "normalize": True,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "linear",
        "xlabel": r"$t-t_{\rm merg}$ [ms]",
        "ylabel": r'$R_c$ [GEO]',
        "legend": {"fancybox": False, "loc": 'center left',
                   "bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "total_center_of_mass_evo_long.png"
    }

    plot_center_of_mass_r(task, def_plot_dic)

    #

    def_plot_dic["type"] = "short"
    def_plot_dic["xmax"] = 40
    def_plot_dic["ymin"], def_plot_dic["ymax"] = 0., 5.
    # plot_dic["ymin"], plot_dic["ymax"] = 2, 3.
    def_plot_dic["title"] = "Short-lived remnants"
    def_plot_dic["figname"] = __outplotdir__ + "total_center_of_mass_evo_short.png"

    plot_center_of_mass_r(task, def_plot_dic)

def task_plot_remnant_mass_evo():

    task = [
        {"sim": "BLh_M13641364_M0_LK_SR", "color": "black", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long"},
        # {"sim": "BLh_M11841581_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"NS"},
        {"sim": "BLh_M11461635_M0_LK_SR", "color": "black", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long"},
        {"sim": "BLh_M10651772_M0_LK_SR", "color": "black", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long"},
        # {"sim": "BLh_M10201856_M0_SR", "color": "black", "ls": "-", "lw": 0.7, "alpha": 1., "outcome":"PC"},
        # {"sim": "BLh_M10201856_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"PC"},
        #
        {"sim": "DD2_M13641364_M0_SR", "color": "blue", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long"},
        # {"sim": "DD2_M13641364_M0_SR_R04", "color": "blue", "ls": "--", "lw": 0.7, "alpha": 1., "t1":60},
        {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "blue", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long", "t1": 40},
        {"sim": "DD2_M14971245_M0_SR", "color": "blue", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long"},
        {"sim": "DD2_M15091235_M0_LK_SR", "color": "blue", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long"},
        #
        # {"sim": "LS220_M13641364_M0_SR", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1.},
        {"sim": "LS220_M13641364_M0_LK_SR_restart", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "type": "short"},
        {"sim": "LS220_M14691268_M0_LK_SR", "color": "red", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long"},
        {"sim": "LS220_M11461635_M0_LK_SR", "color": "red", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "short"},
        {"sim": "LS220_M10651772_M0_LK_SR", "color": "red", "ls": ":", "lw": 0.7, "alpha": 1., "type": "short"},
        #
        {"sim": "SFHo_M13641364_M0_SR", "color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short"},
        {"sim": "SFHo_M14521283_M0_SR", "color": "green", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short"},
        {"sim": "SFHo_M11461635_M0_LK_SR", "color": "green", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long"},
        #
        {"sim": "SLy4_M13641364_M0_SR", "color": "magenta", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short"},
        {"sim": "SLy4_M14521283_M0_SR", "color": "magenta", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short"},
        # {"sim": "SLy4_M10201856_M0_LK_SR", "color": "orange", "ls": "-.", "lw": 0.7, "alpha": 1.},
        {"sim": "SLy4_M11461635_M0_LK_SR", "color": "magenta", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long"}
    ]

    for t in task: t["label"] = r"\texttt{" + t["sim"].replace('_', '\_') + "}"
    for t in task:
        if t["sim"] == "LS220_M13641364_M0_LK_SR_restart":
            t["label"] = t["label"].replace("\_restart", "")

    # --- LONG ---
    plot_dic = {
        "figsize": (6., 2.5),
        "type": "long",
        "xmin": 0, "xmax": 1e2,
        "ymin": 2.6, "ymax": 3.,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "linear",
        "xlabel": r"$t-t_{\rm merg}$ [ms]",
        "ylabel": r"$M_{\rm remnant}$ $[M_{\odot}]$",
        "legend": {"fancybox": False, "loc": 'center left',
                   "bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "total_remnant_mass_evo_long.png"
    }
    plot_tot_remnant_mass_evo(task, plot_dic)

    # --- SHORT ---
    plot_dic["type"] = "short"
    plot_dic["xmax"] = 20
    plot_dic["ymin"], plot_dic["ymax"] = 2, 3.
    plot_dic["title"] = "Short-lived remnants"
    plot_dic["figname"] = __outplotdir__ + "total_remnant_mass_evo_short.png"
    #
    plot_tot_remnant_mass_evo(task, plot_dic)

def task_plot_mass_ave_val_evo():

    # python profile.py -t hist plothist --v_n rho temp Ye press r entr --mask remnant --it all -s None

    v_n = "Ye"
    mask = "remnant"
    task = [
        {"sim": "BLh_M13641364_M0_LK_SR", "color": "black", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "mask": mask, "t": -1},
        # {"sim": "BLh_M11841581_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"NS", "v_n": v_n, "t": -1},
        {"sim": "BLh_M11461635_M0_LK_SR", "color": "black", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "mask": mask, "t": -1},
        {"sim": "BLh_M10651772_M0_LK_SR", "color": "black", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "mask": mask, "t": -1},
        # {"sim": "BLh_M10201856_M0_SR", "color": "black", "ls": "-", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
        # {"sim": "BLh_M10201856_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
        #
        {"sim": "DD2_M13641364_M0_SR", "color": "blue", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n,
         "mask": mask, "t": -1},
        # {"sim": "DD2_M13641364_M0_SR_R04", "color": "blue", "ls": "--", "lw": 0.7, "alpha": 1., "t1":60, "v_n": v_n, "t": -1},
        {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "blue", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "mask": mask, "t": 60 / 1.e3},
        {"sim": "DD2_M14971245_M0_SR", "color": "blue", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n,
         "mask": mask, "t": -1},
        {"sim": "DD2_M15091235_M0_LK_SR", "color": "blue", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "mask": mask, "t": -1},
        #
        # {"sim": "LS220_M13641364_M0_SR", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "t": -1},
        {"sim": "LS220_M13641364_M0_LK_SR_restart", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "type": "short",
         "v_n": v_n, "mask": mask, "t": 25},
        {"sim": "LS220_M14691268_M0_LK_SR", "color": "red", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "mask": mask, "t": 60},
        {"sim": "LS220_M11461635_M0_LK_SR", "color": "red", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n, "mask": mask, "t": -1},
        {"sim": "LS220_M10651772_M0_LK_SR", "color": "red", "ls": ":", "lw": 0.7, "alpha": 1., "type": "short",
         "v_n": v_n, "mask": mask, "t": -1},
        #
        {"sim": "SFHo_M13641364_M0_SR", "color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n, "mask": mask, "t": -1},
        {"sim": "SFHo_M14521283_M0_SR", "color": "green", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n, "mask": mask, "t": -1},
        {"sim": "SFHo_M11461635_M0_LK_SR", "color": "green", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "mask": mask, "t": -1},
        #
        {"sim": "SLy4_M13641364_M0_SR", "color": "magenta", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n, "mask": mask, "t": -1},
        {"sim": "SLy4_M14521283_M0_SR", "color": "magenta", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n, "mask": mask, "t": -1},
        # {"sim": "SLy4_M10201856_M0_LK_SR", "color": "orange", "ls": "-.", "lw": 0.7, "alpha": 1., "v_n": v_n, "t": -1},
        {"sim": "SLy4_M11461635_M0_LK_SR", "color": "magenta", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "mask": mask, "t": -1}
    ]
    #
    for t in task: t["label"] = r"\texttt{" + t["sim"].replace('_', '\_') + "}"
    for t in task:
        if t["sim"] == "LS220_M13641364_M0_LK_SR_restart":
            t["label"] = t["label"].replace("\_restart", "")
    #
    def_plot_dic = {
        "figsize": (6., 2.5),
        "type": "long",
        "xmin": -10, "xmax": 100,
        "ymin": 0, "ymax": 0.15,
        "normalize": True,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "linear",
        "xlabel": r"$t-t_{\rm merg}$ [ms]",
        "ylabel": r"$\langle Y_e \rangle$",
        "legend": {"fancybox": False, "loc": 'center left',
                   "bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "total_massave_{}_long.png".format(v_n)
    }

    plot_dic = copy.deepcopy(def_plot_dic)

    plot_mass_ave_val_evo(task, plot_dic)

    plot_dic["xmin"], plot_dic["xmax"] = -10, 20
    # plot_dic["ymin"], plot_dic["ymax"] = 0., 0.3
    plot_dic["type"] = "short"
    plot_dic["title"] = "Short-lived remnants"
    plot_dic["figname"] = __outplotdir__ + "total_massave_{}_short.png".format(v_n)

    plot_mass_ave_val_evo(task, plot_dic)

    # --- temp
    plot_dic2 = copy.deepcopy(def_plot_dic)
    v_n = "temp"
    for t in task: t["v_n"] = v_n
    plot_dic2["title"] = "Long-lived remnants"
    plot_dic2["type"] = "long"
    plot_dic2["yscale"] = "linear"
    plot_dic2["ymin"], plot_dic2["ymax"] = 0, 50
    plot_dic2["ylabel"] = r"$\langle T \rangle$ [GEO]"
    plot_dic2["figname"] = __outplotdir__ + "total_massave_{}_long.png".format(v_n)

    plot_mass_ave_val_evo(task, plot_dic2)

    plot_dic2["ymin"], plot_dic2["ymax"] = 0, 60
    plot_dic2["xmin"], plot_dic2["xmax"] = -10, 20
    plot_dic2["type"] = "short"
    plot_dic2["title"] = "Short-lived remnants"
    plot_dic2["figname"] = __outplotdir__ + "total_massave_{}_short.png".format(v_n)

    plot_mass_ave_val_evo(task, plot_dic2)

    # --- entropy
    plot_dic3 = copy.deepcopy(def_plot_dic)
    v_n = "entr"
    for t in task: t["v_n"] = v_n
    plot_dic3["title"] = "Long-lived remnants"
    plot_dic3["type"] = "long"
    plot_dic3["yscale"] = "linear"
    plot_dic3["ymin"], plot_dic3["ymax"] = 0, 2
    plot_dic3["ylabel"] = r"$\langle s \rangle$ [GEO]"
    plot_dic3["figname"] = __outplotdir__ + "total_massave_{}_long.png".format(v_n)

    plot_mass_ave_val_evo(task, plot_dic3)

    plot_dic3["xmin"], plot_dic3["xmax"] = -10, 20
    plot_dic3["ymin"], plot_dic3["ymax"] = 0, 5
    plot_dic3["type"] = "short"
    plot_dic3["title"] = "Short-lived remnants"
    plot_dic3["figname"] = __outplotdir__ + "total_massave_{}_short.png".format(v_n)

    plot_mass_ave_val_evo(task, plot_dic3)

    # --- pressure
    plot_dic3 = copy.deepcopy(def_plot_dic)
    v_n = "press"
    for t in task: t["v_n"] = v_n
    plot_dic3["title"] = "Long-lived remnants"
    plot_dic3["type"] = "long"
    plot_dic3["yscale"] = "log"
    plot_dic3["ymin"], plot_dic3["ymax"] = 1e-5, 1e-3
    plot_dic3["ylabel"] = r"$\langle P \rangle$ [GEO]"
    plot_dic3["figname"] = __outplotdir__ + "total_massave_{}_long.png".format(v_n)

    plot_mass_ave_val_evo(task, plot_dic3)

    plot_dic3["xmin"], plot_dic3["xmax"] = -10, 20
    plot_dic3["ymin"], plot_dic3["ymax"] = 1e-5, 1e-3
    plot_dic3["type"] = "short"
    plot_dic3["title"] = "Short-lived remnants"
    plot_dic3["figname"] = __outplotdir__ + "total_massave_{}_short.png".format(v_n)

    plot_mass_ave_val_evo(task, plot_dic3)

    # --- r
    plot_dic5 = copy.deepcopy(def_plot_dic)
    v_n = "r"
    for t in task: t["v_n"] = v_n
    plot_dic5["title"] = "Long-lived remnants"
    plot_dic5["type"] = "long"
    plot_dic5["yscale"] = "linear"
    plot_dic5["xmin"], plot_dic5["xmax"] = -10, 100
    plot_dic5["ymin"], plot_dic5["ymax"] = 0, 20.
    plot_dic5["ylabel"] = r"$\langle R \rangle$ [GEO]"
    plot_dic5["figname"] = __outplotdir__ + "total_massave_{}_long.png".format(v_n)

    plot_mass_ave_val_evo(task, plot_dic5)

    plot_dic5["xmin"], plot_dic5["xmax"] = -10, 20
    plot_dic5["ymin"], plot_dic5["ymax"] = 0., 20.
    plot_dic5["type"] = "short"
    plot_dic5["title"] = "Short-lived remnants"
    plot_dic5["figname"] = __outplotdir__ + "total_massave_{}_short.png".format(v_n)

    plot_mass_ave_val_evo(task, plot_dic5)

def task_plot_remnant_timecorr():

    task = {
        "sim": "BLh_M13641364_M0_LK_SR",
        "v_n": "Ye",
        "normalize": True,
        "mask": "remnant"
    }

    def_plotdic = {"vmin": 1e-6, "vmax": 1e-3,
                   "xmin": 0, "xmax": 90,
                   "ymin": 0.05, "ymax": 0.15,
                   "cmap": "jet",
                   "set_under": "black",
                   "set_over": "red",
                   "yscale": "linear",
                   "xscale": "linear",
                   "xlabel": r"$t-t_{\rm merg}$ [ms]",
                   "ylabel": r"$\langle Y_e \rangle$",
                   "title": r"\texttt{" + task["sim"].replace("_", "\_") + "}",# + "[{}ms]".format(task["t1"]),
                   "clabel": r"$M_{\rm disk}/M$",
                   # "text": {"x": 0.3, "y": 0.95, "s": r"$\theta>60\deg$", "ha": "center", "va": "top", "fontsize": 11,
                   #          "color": "white",
                   #          "transform": None},
                   "outdir": __outplotdir__ ,
                   "figname": "final_disk_timecorr_{}_{}.png".format(task["v_n"], task["sim"])
                   }

    plot_remnant_timecorr(task, def_plotdic)

    #
    plotdic1 = copy.deepcopy(def_plotdic)
    task1 = copy.deepcopy(task)
    task1["sim"] = "BLh_M11461635_M0_LK_SR"
    plotdic1["title"] = r"\texttt{" + task1["sim"].replace("_", "\_") + "}" # + "[{}ms]".format(task["t1"]),
    plotdic1["figname"] = "final_disk_timecorr_{}_{}.png".format(task1["v_n"], task1["sim"])
    plot_remnant_timecorr(task1, plotdic1)

def task_plot_total_angular_momentum_colormesh():

    task = {
        "sim": "BLh_M13641364_M0_LK_SR",
        "v_n": "J"
    }

    def_plotdic = {"vmin": 1e-3, "vmax": 1e0,
                   "xmin": 0, "xmax": 90,
                   "ymin": 0, "ymax": 15,
                   "cmap": "jet",
                   "set_under": "black",
                   "set_over": "red",
                   "yscale": "linear",
                   "xscale": "linear",
                   "xlabel": r"$t-t_{\rm merg}$ [ms]",
                   "ylabel": r"$R_{\rm cyl}$",
                   "title": r"\texttt{" + task["sim"].replace("_", "\_") + "}",  # + "[{}ms]".format(task["t1"]),
                   "clabel": r"$J_{r}$",
                   # "text": {"x": 0.3, "y": 0.95, "s": r"$\theta>60\deg$", "ha": "center", "va": "top", "fontsize": 11,
                   #          "color": "white",
                   #          "transform": None},
                   "outdir": __outplotdir__,
                   "figname": "evol_j_2d_{}_R1.png".format(task["sim"])
                   }

    plot_total_angular_momentum_colormesh(task, def_plotdic)
    #
    plot_dic = copy.deepcopy(def_plotdic)
    plot_dic["ymin"], plot_dic["ymax"] = 15., 500.
    plot_dic["vmin"], plot_dic["vmax"] = 1e-1, 1e-4
    plot_dic["yscale"] = "log"
    plot_dic["figname"] = "evol_j_2d_{}_R2.png".format(task["sim"])
    plot_total_angular_momentum_colormesh(task, plot_dic)

def task_plot_total_angular_momentum_flux_colormesh():

    task = {
        "sim": "BLh_M13641364_M0_LK_SR",
        "v_n": "Jflux"
    }

    def_plotdic = {"vmin": 1e-6, "vmax": 1e-4,
                   "xmin": 0, "xmax": 90,
                   "ymin": 100, "ymax": 500,
                   "cmap": "jet",#"Spectral_r",
                   "set_under": "black",
                   "set_over": "red",
                   "yscale": "linear",
                   "xscale": "linear",
                   "xlabel": r"$t-t_{\rm merg}$ [ms]",
                   "ylabel": r"$R_{\rm cyl}$",
                   "title": r"\texttt{" + task["sim"].replace("_", "\_") + "}",  # + "[{}ms]".format(task["t1"]),
                   "clabel": r"$Jflux_{r}$",
                   # "text": {"x": 0.3, "y": 0.95, "s": r"$\theta>60\deg$", "ha": "center", "va": "top", "fontsize": 11,
                   #          "color": "white",
                   #          "transform": None},
                   "outdir": __outplotdir__,
                   "figname": "evol_jflux_2d_{}_R1.png".format(task["sim"])
                   }

    plot_total_angular_momentum_colormesh(task, def_plotdic)
    #

    plot_dic = copy.deepcopy(def_plotdic)
    task["sim"] = "BLh_M11461635_M0_LK_SR"
    plot_dic["title"] = r"\texttt{" + task["sim"].replace("_", "\_") + "}"  # + "[{}ms]".format(task["t1"]),
    plot_dic["figname"] = "evol_jflux_2d_{}_R1.png".format(task["sim"])
    plot_total_angular_momentum_colormesh(task, plot_dic)
    #
    plot_dic = copy.deepcopy(def_plotdic)
    plot_dic["title"] = r"\texttt{" + task["sim"].replace("_", "\_") + "}"  # + "[{}ms]".format(task["t1"]),
    plot_dic["figname"] = "evol_jflux_2d_{}_R1.png".format(task["sim"])
    plot_total_angular_momentum_colormesh(task, plot_dic)
    #

    task["sim"] = "DD2_M13641364_M0_LK_SR_R04"
    plot_dic["title"] = r"\texttt{" + task["sim"].replace("_", "\_") + "}"  # + "[{}ms]".format(task["t1"]),
    plot_dic["figname"] = "evol_jflux_2d_{}_R1.png".format(task["sim"])
    plot_total_angular_momentum_colormesh(task, plot_dic)
    #
    task["sim"] = "DD2_M13641364_M0_SR"
    plot_dic["title"] = r"\texttt{" + task["sim"].replace("_", "\_") + "}"  # + "[{}ms]".format(task["t1"]),
    plot_dic["figname"] = "evol_jflux_2d_{}_R1.png".format(task["sim"])
    plot_total_angular_momentum_colormesh(task, plot_dic)

def task_plot_total_angular_momentum():

    v_n = "J"
    rext = "<15"

    task = [
        {"sim": "BLh_M13641364_M0_LK_SR", "color": "black", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        # {"sim": "BLh_M11841581_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"NS", "v_n": v_n, "t": -1},
        {"sim": "BLh_M11461635_M0_LK_SR", "color": "black", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        {"sim": "BLh_M10651772_M0_LK_SR", "color": "black", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        # {"sim": "BLh_M10201856_M0_SR", "color": "black", "ls": "-", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
        # {"sim": "BLh_M10201856_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
        #
        {"sim": "DD2_M13641364_M0_SR_R04", "color": "blue", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "tmin":70},
        # {"sim": "DD2_M13641364_M0_SR_R04", "color": "blue", "ls": "--", "lw": 0.7, "alpha": 1., "t1":60, "v_n": v_n, "t": -1},
        {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "blue", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "tmin":40},
        {"sim": "DD2_M14971245_M0_SR", "color": "blue", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        {"sim": "DD2_M15091235_M0_LK_SR", "color": "blue", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        #
        # {"sim": "LS220_M13641364_M0_SR", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "t": -1},
        {"sim": "LS220_M13641364_M0_LK_SR_restart", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "type": "short",
         "v_n": v_n},
        {"sim": "LS220_M14691268_M0_LK_SR", "color": "red", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "tmax":70},
        {"sim": "LS220_M11461635_M0_LK_SR", "color": "red", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n},
        {"sim": "LS220_M10651772_M0_LK_SR", "color": "red", "ls": ":", "lw": 0.7, "alpha": 1., "type": "short",
         "v_n": v_n},
        #
        {"sim": "SFHo_M13641364_M0_SR", "color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n},
        {"sim": "SFHo_M14521283_M0_SR", "color": "green", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n},
        {"sim": "SFHo_M11461635_M0_LK_SR", "color": "green", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        #
        {"sim": "SLy4_M13641364_M0_SR", "color": "magenta", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n},
        {"sim": "SLy4_M14521283_M0_SR", "color": "magenta", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n},
        # {"sim": "SLy4_M10201856_M0_LK_SR", "color": "orange", "ls": "-.", "lw": 0.7, "alpha": 1., "v_n": v_n, "t": -1},
        {"sim": "SLy4_M11461635_M0_LK_SR", "color": "magenta", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n}
    ]

    for t in task: t["label"] = r"\texttt{" + t["sim"].replace('_', '\_') + "}"
    for t in task:
        if t["sim"] == "LS220_M13641364_M0_LK_SR_restart":
            t["label"] = t["label"].replace("\_restart", "")

    rext = "<500"
    for t in task: t["rext"] = rext
    plot_dic = {
        "figsize": (6., 2.5),
        "type": "long",
        "xmin": 0, "xmax": 1e2,
        "ymin": 4.5, "ymax": 6.5,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "linear",
        "xlabel": r"$t-t_{\rm merg}$ [ms]",
        "ylabel": r"$J_{\rm tot}\ [G\, c^{-1} M_\odot^2]$",
        "legend": {"fancybox": False, "loc": 'center left',
                   "bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "total_ang_mom_evo_long.png"
    }
    plot_total_angular_momentum(task, plot_dic)

    ''' ---- r < 15 ---- '''
    rext = "<8.2"
    for t in task: t["rext"] = rext

    # --- LONG ---
    plot_dic = {
        "figsize": (6., 2.5),
        "type": "long",
        "xmin": 0, "xmax": 1e2,
        "ymin": 3, "ymax": 6.5,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "linear",
        "xlabel": r"$t-t_{\rm merg}$ [ms]",
        "ylabel": r"$J_{r<12}\ [G\, c^{-1} M_\odot^2]$",
        "legend": {"fancybox": False, "loc": 'center left',
                   "bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "total_ang_mom_evo_r0_15_long.png"
    }
    plot_total_angular_momentum(task, plot_dic)

    # --- SHORT ---
    plot_dic["type"] = "short"
    plot_dic["xmax"] = 20
    plot_dic["ymin"], plot_dic["ymax"] = 3, 7.
    plot_dic["title"] = "Short-lived remnants"
    plot_dic["figname"] = __outplotdir__ + "total_ang_mom_evo_r0_15_short.png"
    #
    #plot_total_angular_momentum(task, plot_dic)

    ''' --- r > 15 ---- '''
    rext = ">15"
    for t in task: t["rext"] = rext
    # --- LONG ---
    plot_dic["type"] = "long"
    plot_dic["xmin"],plot_dic["xmax"] = 0, 1e2
    plot_dic["ymin"],plot_dic["ymax"] = 0, 3
    plot_dic["title"] = "Long-lived remnants"
    plot_dic["ylabel"] = r"$J_{r>15}\ [G\, c^{-1} M_\odot^2]$"
    plot_dic["figname"] = __outplotdir__ + "total_ang_mom_evo_r15_500_long.png"

    # plot_total_angular_momentum(task, plot_dic)

    # --- SHORT ---
    plot_dic["type"] = "short"
    plot_dic["xmax"] = 20
    plot_dic["ymin"], plot_dic["ymax"] = 0, 2
    plot_dic["title"] = "Short-lived remnants"
    plot_dic["figname"] = __outplotdir__ + "total_ang_mom_evo_r15_short.png"
    #
    # plot_total_angular_momentum(task, plot_dic)

def task_plot_total_angular_momentum_flux():

    extraction_radius = None
    v_n = "Jflux"

    task = [
        {"sim": "BLh_M13641364_M0_LK_SR", "color": "black", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        # {"sim": "BLh_M11841581_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"NS", "v_n": v_n, "t": -1},
        {"sim": "BLh_M11461635_M0_LK_SR", "color": "black", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        {"sim": "BLh_M10651772_M0_LK_SR", "color": "black", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        # {"sim": "BLh_M10201856_M0_SR", "color": "black", "ls": "-", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
        # {"sim": "BLh_M10201856_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
        #
        {"sim": "DD2_M13641364_M0_SR", "color": "blue", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        # {"sim": "DD2_M13641364_M0_SR_R04", "color": "blue", "ls": "--", "lw": 0.7, "alpha": 1., "t1":60, "v_n": v_n, "t": -1},
        {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "blue", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        {"sim": "DD2_M14971245_M0_SR", "color": "blue", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        {"sim": "DD2_M15091235_M0_LK_SR", "color": "blue", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        #
        # {"sim": "LS220_M13641364_M0_SR", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "t": -1},
        {"sim": "LS220_M13641364_M0_LK_SR_restart", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "type": "short",
         "v_n": v_n},
        {"sim": "LS220_M14691268_M0_LK_SR", "color": "red", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        {"sim": "LS220_M11461635_M0_LK_SR", "color": "red", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n},
        {"sim": "LS220_M10651772_M0_LK_SR", "color": "red", "ls": ":", "lw": 0.7, "alpha": 1., "type": "short",
         "v_n": v_n},
        #
        {"sim": "SFHo_M13641364_M0_SR", "color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n},
        {"sim": "SFHo_M14521283_M0_SR", "color": "green", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n},
        {"sim": "SFHo_M11461635_M0_LK_SR", "color": "green", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        #
        {"sim": "SLy4_M13641364_M0_SR", "color": "magenta", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n},
        {"sim": "SLy4_M14521283_M0_SR", "color": "magenta", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n},
        # {"sim": "SLy4_M10201856_M0_LK_SR", "color": "orange", "ls": "-.", "lw": 0.7, "alpha": 1., "v_n": v_n, "t": -1},
        {"sim": "SLy4_M11461635_M0_LK_SR", "color": "magenta", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n}
    ]

    for t in task: t["rext"] = extraction_radius
    for t in task: t["label"] = r"\texttt{" + t["sim"].replace('_', '\_') + "}"
    for t in task:
        if t["sim"] == "LS220_M13641364_M0_LK_SR_restart":
            t["label"] = t["label"].replace("\_restart", "")

    ''' --------- rext = 500 ------------ '''

    # --- LONG ---
    plot_dic = {
        "figsize": (6., 2.5),
        "type": "long",
        "xmin": 0, "xmax": 1e2,
        "ymin": 1e-6, "ymax": 1e-4,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "log",
        "xlabel": r"$t-t_{\rm merg}$ [ms]",
        "ylabel": r"$J_f|_{r=500}\ [G\, c^{-1} M_\odot]$",
        "legend": {"fancybox": False, "loc": 'center left',
                   "bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "total_ang_mom_flux_evo_long.png"
    }
    plot_total_angular_momentum(task, plot_dic)

    # --- SHORT ---
    plot_dic["type"] = "short"
    plot_dic["xmax"] = 20
    plot_dic["ymin"], plot_dic["ymax"] = 2, 3.
    plot_dic["title"] = "Short-lived remnants"
    plot_dic["figname"] = __outplotdir__ + "total_ang_mom_flux_evo_short.png"
    #
    # plot_total_angular_momentum(task, plot_dic)

    ''' --------- rext = 15 ------------ '''
    rlist = [100, 200, 300, 400, 500]
    ymins = [1e-7, 1e-7, 1e-7, 1e-8, 1e-8]
    ymaxs = [1e-4, 1e-4, 8e-5, 2e-5, 1e-5]
    for r, ymin, ymax in zip(rlist, ymins, ymaxs):
        print("\t\t{}".format(r))
        extraction_radius = r
        for t in task: t["rext"] = r
        plot_dic["type"] = "long"
        plot_dic["xmin"], plot_dic["xmax"] = 0, 1e2
        plot_dic["ymin"], plot_dic["ymax"] = 3e-8, 1e-6
        plot_dic["ylabel"] = r"$J_f|_{r=" + str(r) + "}\ [G\, c^{-1} M_\odot]$"
        plot_dic["title"] = "Long-lived remnants"
        plot_dic["figname"] = __outplotdir__ + "total_ang_mom_flux_evo_r{}_long.png".format(r)
        #
        # plot_total_angular_momentum(task, plot_dic)

def task_plot_total_djdt():
    v_n = "dJ/dt"
    extraction_radius = None
    interpolate_method = "None"

    task = [
        {"sim": "BLh_M13641364_M0_LK_SR", "color": "black", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        # {"sim": "BLh_M11841581_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"NS", "v_n": v_n, "t": -1},
        {"sim": "BLh_M11461635_M0_LK_SR", "color": "black", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        {"sim": "BLh_M10651772_M0_LK_SR", "color": "black", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        # {"sim": "BLh_M10201856_M0_SR", "color": "black", "ls": "-", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
        # {"sim": "BLh_M10201856_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
        #
        {"sim": "DD2_M13641364_M0_SR", "color": "blue", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        # {"sim": "DD2_M13641364_M0_SR_R04", "color": "blue", "ls": "--", "lw": 0.7, "alpha": 1., "t1":60, "v_n": v_n, "t": -1},
        {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "blue", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "tmin":40},
        {"sim": "DD2_M14971245_M0_SR", "color": "blue", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        {"sim": "DD2_M15091235_M0_LK_SR", "color": "blue", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        #
        # {"sim": "LS220_M13641364_M0_SR", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "t": -1},
        {"sim": "LS220_M13641364_M0_LK_SR_restart", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "type": "short",
         "v_n": v_n},
        {"sim": "LS220_M14691268_M0_LK_SR", "color": "red", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "tmax":70},
        {"sim": "LS220_M11461635_M0_LK_SR", "color": "red", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n},
        {"sim": "LS220_M10651772_M0_LK_SR", "color": "red", "ls": ":", "lw": 0.7, "alpha": 1., "type": "short",
         "v_n": v_n},
        #
        {"sim": "SFHo_M13641364_M0_SR", "color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n},
        {"sim": "SFHo_M14521283_M0_SR", "color": "green", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n},
        {"sim": "SFHo_M11461635_M0_LK_SR", "color": "green", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        #
        {"sim": "SLy4_M13641364_M0_SR", "color": "magenta", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n},
        {"sim": "SLy4_M14521283_M0_SR", "color": "magenta", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n},
        # {"sim": "SLy4_M10201856_M0_LK_SR", "color": "orange", "ls": "-.", "lw": 0.7, "alpha": 1., "v_n": v_n, "t": -1},
        {"sim": "SLy4_M11461635_M0_LK_SR", "color": "magenta", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n}
    ]

    for t in task: t["rext"] = extraction_radius
    for t in task: t["intmethod"] = interpolate_method
    for t in task: t["label"] = r"\texttt{" + t["sim"].replace('_', '\_') + "}"
    for t in task:
        if t["sim"] == "LS220_M13641364_M0_LK_SR_restart":
            t["label"] = t["label"].replace("\_restart", "")

    # --- LONG ---
    plot_dic = {
        "figsize": (6., 2.5),
        "type": "long",
        "xmin": 0, "xmax": 1e2,
        "ymin": 1e-6, "ymax": 1e-4,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "log",
        "xlabel": r"$t-t_{\rm merg}$ [ms]",
        "ylabel": r"$-dJ/dt\ [G\, c^{-1} M_\odot]$",
        "legend": {"fancybox": False, "loc": 'center left',
                   "bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "total_djdt_long.png"
    }
    plot_total_angular_momentum(task, plot_dic)

    # --- SHORT ---
    plot_dic["type"] = "short"
    plot_dic["xmax"] = 20
    plot_dic["ymin"], plot_dic["ymax"] = 2, 3.
    plot_dic["title"] = "Short-lived remnants"
    plot_dic["figname"] = __outplotdir__ + "total_djdt_short.png"

def task_plot_total_djdt_minus_Jflux():
    v_n = "dJ/dt-Jflux"
    extraction_radius = None
    interpolate_method = "None"

    task = [
        {"sim": "BLh_M13641364_M0_LK_SR", "color": "black", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        # {"sim": "BLh_M11841581_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"NS", "v_n": v_n, "t": -1},
        {"sim": "BLh_M11461635_M0_LK_SR", "color": "black", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        {"sim": "BLh_M10651772_M0_LK_SR", "color": "black", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        # {"sim": "BLh_M10201856_M0_SR", "color": "black", "ls": "-", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
        # {"sim": "BLh_M10201856_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
        #
        {"sim": "DD2_M13641364_M0_SR", "color": "blue", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        # {"sim": "DD2_M13641364_M0_SR_R04", "color": "blue", "ls": "--", "lw": 0.7, "alpha": 1., "t1":60, "v_n": v_n, "t": -1},
        {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "blue", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "tmin": 40},
        {"sim": "DD2_M14971245_M0_SR", "color": "blue", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        {"sim": "DD2_M15091235_M0_LK_SR", "color": "blue", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        #
        # {"sim": "LS220_M13641364_M0_SR", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "t": -1},
        {"sim": "LS220_M13641364_M0_LK_SR_restart", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "type": "short",
         "v_n": v_n},
        {"sim": "LS220_M14691268_M0_LK_SR", "color": "red", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "tmax": 70},
        {"sim": "LS220_M11461635_M0_LK_SR", "color": "red", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n},
        {"sim": "LS220_M10651772_M0_LK_SR", "color": "red", "ls": ":", "lw": 0.7, "alpha": 1., "type": "short",
         "v_n": v_n},
        #
        {"sim": "SFHo_M13641364_M0_SR", "color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n},
        {"sim": "SFHo_M14521283_M0_SR", "color": "green", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n},
        {"sim": "SFHo_M11461635_M0_LK_SR", "color": "green", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n},
        #
        {"sim": "SLy4_M13641364_M0_SR", "color": "magenta", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n},
        {"sim": "SLy4_M14521283_M0_SR", "color": "magenta", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n},
        # {"sim": "SLy4_M10201856_M0_LK_SR", "color": "orange", "ls": "-.", "lw": 0.7, "alpha": 1., "v_n": v_n, "t": -1},
        {"sim": "SLy4_M11461635_M0_LK_SR", "color": "magenta", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n}
    ]

    for t in task: t["rext"] = extraction_radius
    for t in task: t["intmethod"] = interpolate_method
    for t in task: t["label"] = r"\texttt{" + t["sim"].replace('_', '\_') + "}"
    for t in task:
        if t["sim"] == "LS220_M13641364_M0_LK_SR_restart":
            t["label"] = t["label"].replace("\_restart", "")

    # --- LONG --- #    \   djdt - tot_jflux      /
    plot_dic = {
        "figsize": (6., 2.5),
        "type": "long",
        "xmin": 0, "xmax": 1e2,
        "ymin": -1e-6, "ymax": -1e-4,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "log",
        "xlabel": r"$t-t_{\rm merg}$ [ms]",
        "ylabel": r"$(dJ/dt - J_{\rm flux}) > 0\ [G\, c^{-1} M_\odot]$",
        "legend": {"fancybox": False, "loc": 'center left',
                   "bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "total_djdt_minus_jflux_long.png"
    }
    # plot_total_angular_momentum(task, plot_dic)

    # ----------------- #  -(djdt - tot_jflux)  #

    v_n = "-(dJ/dt-Jflux)"
    for t in task: t["v_n"] = v_n
    plot_dic["ylabel"] = r"$-(dJ/dt - J_{\rm flux}) > 0\ [G\, c^{-1} M_\odot]$"
    plot_dic["figname"] = __outplotdir__+ "total_minus_djdt_minus_jflux_long.png"

    # plot_total_angular_momentum(task, plot_dic)

    # --------------- # -(dJ/dt-Jflux)

    v_n = "(dJ/dt-Jflux)/(dJ/dt)"
    for t in task: t["v_n"] = v_n
    plot_dic["ylabel"] = r"$(dJ/dt + J_{\rm flux})/(dJ/dt)$"
    plot_dic["ymin"], plot_dic["ymax"] = 2e-3, 2e0
    plot_dic["figname"] = __outplotdir__+ "total_minus_djdt_minus_jflux_dev_djdt_long.png"

    # plot_total_angular_momentum(task, plot_dic)

    # --------------- # (dJ/dt+Jflux)

    v_n = "-(dJ/dt+Jflux)"
    for t in task: t["v_n"] = v_n
    plot_dic["ylabel"] = r"$-(dJ/dt + J_{\rm flux})>0$"
    plot_dic["ymin"], plot_dic["ymax"] = 1e-9, 1e-3
    plot_dic["figname"] = __outplotdir__+ "total_minus_djdt_plus_jflux_long.png"

    # plot_total_angular_momentum(task, plot_dic)

    v_n = "(dJ/dt+Jflux)/(dJ/dt)"
    for t in task: t["v_n"] = v_n
    # plot_dic["ylabel"] = r"$(dJ_{\rm tot}/dt + J_{\rm flux})/(dJ_{\rm tot}/dt)>0$"
    plot_dic["ylabel"] = r"$( \dot{J_{\rm tot}} + J_{\rm flux})/ \dot{J_{\rm tot}}>0$"
    plot_dic["ymin"], plot_dic["ymax"] = 1e-2, 1e1
    plot_dic["figname"] = __outplotdir__+ "total_djdt_plus_jflux_dev_djdt_long.png"

    plot_total_angular_momentum(task, plot_dic)

    # --- SHORT ---
    # plot_dic["type"] = "short"
    # plot_dic["xmax"] = 20
    # plot_dic["ymin"], plot_dic["ymax"] = 2, 3.
    # plot_dic["title"] = "Short-lived remnants"
    # plot_dic["figname"] = __outplotdir__ + "total_djdt_minus_jflux_short.png"

''' ------------------------------- RESOLUTION TASKS ---------------------------- '''

def task_resolution_plot_total_angular_momentum_flux():

    extraction_radius = None
    v_n = "Jflux"

    task = [
        {"sim": "BLh_M13641364_M0_LK_SR", "color": "blue", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long","v_n": v_n},
        {"sim": "BLh_M13641364_M0_LK_LR", "color": "blue", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long","v_n": v_n},
        {"sim": "BLh_M13641364_M0_LK_HR", "color": "blue", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n},

        {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "red", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "tmin":40},
        {"sim": "DD2_M13641364_M0_LK_LR_R04", "color": "red", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long","v_n": v_n},
        {"sim": "DD2_M13641364_M0_LK_HR_R04", "color": "red", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n},
    ]

    for t in task: t["rext"] = extraction_radius
    for t in task: t["label"] = r"\texttt{" + t["sim"].replace('_', '\_') + "}"
    for t in task:
        if t["sim"] == "LS220_M13641364_M0_LK_SR_restart":
            t["label"] = t["label"].replace("\_restart", "")

    ''' --------- rext = 500 ------------ '''

    # --- LONG ---
    plot_dic = {
        "figsize": (6., 2.5),
        "type": "long",
        "xmin": 0, "xmax": 1e2,
        "ymin": 1e-6, "ymax": 1e-4,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "log",
        "xlabel": r"$t-t_{\rm merg}$ [ms]",
        "ylabel": r"$J_f|_{r=500}\ [G\, c^{-1} M_\odot]$",
        "legend": {"fancybox": False, "loc": 'center left',
                   "bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "resolution_ang_mom_flux_evo_long.png"
    }
    plot_total_angular_momentum(task, plot_dic)

    # --- SHORT ---
    plot_dic["type"] = "short"
    plot_dic["xmax"] = 20
    plot_dic["ymin"], plot_dic["ymax"] = 2, 3.
    plot_dic["title"] = "Short-lived remnants"
    plot_dic["figname"] = __outplotdir__ + "resolution_ang_mom_flux_evo_short.png"
    #
    # plot_total_angular_momentum(task, plot_dic)

    ''' --------- rext = 15 ------------ '''
    rlist = [100, 200, 300, 400, 500]
    ymins = [1e-7, 1e-7, 1e-7, 1e-8, 1e-8]
    ymaxs = [1e-4, 1e-4, 8e-5, 2e-5, 1e-5]
    for r, ymin, ymax in zip(rlist, ymins, ymaxs):
        print("\t\t{}".format(r))
        extraction_radius = r
        for t in task: t["rext"] = r
        plot_dic["type"] = "long"
        plot_dic["xmin"], plot_dic["xmax"] = 0, 1e2
        plot_dic["ymin"], plot_dic["ymax"] = 3e-8, 1e-6
        plot_dic["ylabel"] = r"$J_f|_{r=" + str(r) + "}\ [G\, c^{-1} M_\odot]$"
        plot_dic["title"] = "Long-lived remnants"
        plot_dic["figname"] = __outplotdir__ + "resolution_ang_mom_flux_evo_r{}_long.png".format(r)
        #
        # plot_total_angular_momentum(task, plot_dic)

def task_resolution_plot_total_angular_momentum():

    v_n = "J"
    rext = "<15"

    task = [
        {"sim": "BLh_M13641364_M0_LK_SR", "color": "blue", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long","v_n": v_n},
        {"sim": "BLh_M13641364_M0_LK_LR", "color": "blue", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long","v_n": v_n},
        {"sim": "BLh_M13641364_M0_LK_HR", "color": "blue", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n},

        {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "red", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "tmin":40},
        {"sim": "DD2_M13641364_M0_LK_LR_R04", "color": "red", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long","v_n": v_n},
        {"sim": "DD2_M13641364_M0_LK_HR_R04", "color": "red", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n},

        {"sim": "DD2_M15091235_M0_LK_SR", "color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n},
        {"sim": "DD2_M15091235_M0_LK_LR", "color": "green", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long","v_n": v_n},
        {"sim": "DD2_M15091235_M0_LK_HR", "color": "green", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n},

    ]

    # for k in task:
    #     os.system("python /data01/numrel/vsevolod.nedora/bns_ppr_tools/profile.py -s {} -t mjenclosed --it all".format(k["sim"]))


    for t in task: t["label"] = r"\texttt{" + t["sim"].replace('_', '\_') + "}"
    for t in task:
        if t["sim"] == "LS220_M13641364_M0_LK_SR_restart":
            t["label"] = t["label"].replace("\_restart", "")

    rext = "<500"
    for t in task: t["rext"] = rext
    plot_dic = {
        "figsize": (6., 2.5),
        "type": "long",
        "xmin": 0, "xmax": 1e2,
        "ymin": 4.5, "ymax": 6.5,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "linear",
        "xlabel": r"$t-t_{\rm merg}$ [ms]",
        "ylabel": r"$J_{\rm tot}\ [G\, c^{-1} M_\odot^2]$",
        "legend": {"fancybox": False, "loc": 'center left',
                   "bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "resolution_ang_mom_evo_long.png"
    }
    plot_total_angular_momentum(task, plot_dic)

    ''' ---- r < 15 ---- '''

    for t in task: t["rext"] = rext

    # --- LONG ---
    plot_dic = {
        "figsize": (6., 2.5),
        "type": "long",
        "xmin": 0, "xmax": 1e2,
        "ymin": 3, "ymax": 6.5,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "linear",
        "xlabel": r"$t-t_{\rm merg}$ [ms]",
        "ylabel": r"$J_{r<15}\ [G\, c^{-1} M_\odot^2]$",
        "legend": {"fancybox": False, "loc": 'center left',
                   "bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "total_ang_mom_evo_r0_15_long.png"
    }
    # plot_total_angular_momentum(task, plot_dic)

    # --- SHORT ---
    plot_dic["type"] = "short"
    plot_dic["xmax"] = 20
    plot_dic["ymin"], plot_dic["ymax"] = 3, 7.
    plot_dic["title"] = "Short-lived remnants"
    plot_dic["figname"] = __outplotdir__ + "resolution_ang_mom_evo_r0_15_short.png"
    #
    # plot_total_angular_momentum(task, plot_dic)

    ''' --- r > 15 ---- '''
    rext = ">15"
    for t in task: t["rext"] = rext
    # --- LONG ---
    plot_dic["type"] = "long"
    plot_dic["xmin"],plot_dic["xmax"] = 0, 1e2
    plot_dic["ymin"],plot_dic["ymax"] = 0, 3
    plot_dic["title"] = "Long-lived remnants"
    plot_dic["ylabel"] = r"$J_{r>15}\ [G\, c^{-1} M_\odot^2]$"
    plot_dic["figname"] = __outplotdir__ + "resolution_ang_mom_evo_r15_500_long.png"

    # plot_total_angular_momentum(task, plot_dic)

    # --- SHORT ---
    plot_dic["type"] = "short"
    plot_dic["xmax"] = 20
    plot_dic["ymin"], plot_dic["ymax"] = 0, 2
    plot_dic["title"] = "Short-lived remnants"
    plot_dic["figname"] = __outplotdir__ + "resolution_ang_mom_evo_r15_short.png"
    #
    # plot_total_angular_momentum(task, plot_dic)

def task_resolution_plot_total_djdt_minus_Jflux():
    v_n = "dJ/dt-Jflux"
    extraction_radius = None
    interpolate_method = "None"

    task = [
        {"sim": "BLh_M13641364_M0_LK_SR", "color": "blue", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n},
        {"sim": "BLh_M13641364_M0_LK_LR", "color": "blue", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n},
        {"sim": "BLh_M13641364_M0_LK_HR", "color": "blue", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n},

        {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "red", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "tmin":40},
        {"sim": "DD2_M13641364_M0_LK_LR_R04", "color": "red", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n},
        {"sim": "DD2_M13641364_M0_LK_HR_R04", "color": "red", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n},
    ]

    for t in task: t["rext"] = extraction_radius
    for t in task: t["intmethod"] = interpolate_method
    for t in task: t["label"] = r"\texttt{" + t["sim"].replace('_', '\_') + "}"
    for t in task:
        if t["sim"] == "LS220_M13641364_M0_LK_SR_restart":
            t["label"] = t["label"].replace("\_restart", "")

    # --- LONG --- #    \   djdt - tot_jflux      /
    plot_dic = {
        "figsize": (6., 2.5),
        "type": "long",
        "xmin": 0, "xmax": 1e2,
        "ymin": -1e-6, "ymax": -1e-4,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "log",
        "xlabel": r"$t-t_{\rm merg}$ [ms]",
        "ylabel": r"$(dJ/dt - J_{\rm flux}) > 0\ [G\, c^{-1} M_\odot]$",
        "legend": {"fancybox": False, "loc": 'center left',
                   "bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "resolution_djdt_minus_jflux_long.png"
    }
    # plot_total_angular_momentum(task, plot_dic)

    # ----------------- #  -(djdt - tot_jflux)  #

    v_n = "-(dJ/dt-Jflux)"
    for t in task: t["v_n"] = v_n
    plot_dic["ylabel"] = r"$-(dJ/dt - J_{\rm flux}) > 0\ [G\, c^{-1} M_\odot]$"
    plot_dic["figname"] = __outplotdir__+ "resolution_minus_djdt_minus_jflux_long.png"

    # plot_total_angular_momentum(task, plot_dic)

    # --------------- # -(dJ/dt-Jflux)

    v_n = "(dJ/dt-Jflux)/(dJ/dt)"
    for t in task: t["v_n"] = v_n
    plot_dic["ylabel"] = r"$(dJ/dt + J_{\rm flux})/(dJ/dt)$"
    plot_dic["ymin"], plot_dic["ymax"] = 2e-3, 2e0
    plot_dic["figname"] = __outplotdir__+ "resolution_minus_djdt_minus_jflux_dev_djdt_long.png"

    # plot_total_angular_momentum(task, plot_dic)

    # --------------- # (dJ/dt+Jflux)

    v_n = "-(dJ/dt+Jflux)"
    for t in task: t["v_n"] = v_n
    plot_dic["ylabel"] = r"$-(dJ/dt + J_{\rm flux})>0$"
    plot_dic["ymin"], plot_dic["ymax"] = 1e-7, 1e-3
    plot_dic["figname"] = __outplotdir__+ "resolution_minus_djdt_plus_jflux_long.png"

    plot_total_angular_momentum(task, plot_dic)

    # --- SHORT ---
    # plot_dic["type"] = "short"
    # plot_dic["xmax"] = 20
    # plot_dic["ymin"], plot_dic["ymax"] = 2, 3.
    # plot_dic["title"] = "Short-lived remnants"
    # plot_dic["figname"] = __outplotdir__ + "total_djdt_minus_jflux_short.png"

''' --- iteration 3 | tasks --- '''

def task_plot_rho_modes_2D_3():
    """
    none
    """

    v_n = "rho_modes.h5"
    fpath = "slices/" + "rho_modes.h5"

    norm_to_m = 0
    task = [
        # BLh q = 1
        {"sim": "BLh_M13641364_M0_LK_SR",    "m":1, "plot": {"color": "black", "ls": ":", "lw": 0.8, "alpha": 1.}, "type": "long"}, # 0
        {"sim": "BLh_M13641364_M0_LK_SR",    "m":2, "plot": {"color": "black", "ls": "-", "lw": 0.8, "alpha": 1., "label": "BLh q=1.00 (SR)"}, "type": "long"}, # 1
        # BLh q = 1.66
        {"sim": "BLh_M11461635_M0_LK_SR",    "m":1, "plot": {"color": "gray", "ls": ":", "lw": 0.8, "alpha": 1.}, "type": "long"}, # 2
        {"sim": "BLh_M11461635_M0_LK_SR",    "m":2, "plot": {"color": "gray", "ls": "-", "lw": 0.8, "alpha": 1., "label": "BLh q=1.42 (SR)"}, "type": "long"}, # 3

        # DD2 q = 1 noLK
        {"sim": "DD2_M13641364_M0_SR_R04",  "m": 1, "plot": {"color": "green", "ls": ":", "lw": 0.6, "alpha": 1.}, "type": "long", "t1": 40, "mmean":10},  # 4
        {"sim": "DD2_M13641364_M0_SR_R04",  "m": 2, "plot":  {"color": "green", "ls": "-", "lw": 0.9, "alpha": 1., "label": "DD2* q=1.00 (SR)"}, "type": "long", "mmean":10},  # 5
        # DD2 q = 1.0 LK
        {"sim": "DD2_M13641364_M0_LK_SR_R04","m":1, "plot": {"color": "blue", "ls": ":", "lw": 0.6, "alpha": 1.}, "type": "long", "t1": 40, "mmean":10}, # 6
        {"sim": "DD2_M13641364_M0_LK_SR_R04","m":2, "plot": {"color": "blue", "ls": "-", "lw": 0.9, "alpha": 1., "label": "DD2 q=1.00 (SR)"}, "type": "long", "t1": 40, "mmean":10},# 7
        # DD2 q = 1.22
        {"sim": "DD2_M15091235_M0_LK_SR",    "m":1, "plot": {"color": "green", "ls": ":", "lw": 0.6, "alpha": 1.}, "type": "long"}, # 8
        {"sim": "DD2_M15091235_M0_LK_SR",    "m":2, "plot": {"color": "green", "ls": "-", "lw": 0.9, "alpha": 1., "label": "DD2 q=1.22 (SR)"}, "type": "long"}, # 9

        # LS220 q=1 noLK
        {"sim": "LS220_M13641364_M0_SR", "m":1, "plot": {"color": "orange", "ls": ":", "lw": 0.8, "alpha": 1.}, "type": "short", "t2pm":16, "mmean":3}, # 10 "mmean":3,
        {"sim": "LS220_M13641364_M0_SR", "m":2, "plot": {"color": "orange", "ls": "-", "lw": 1.0, "alpha": 1., "label": "LS220* q=1.00 (SR)"}, "type": "short", "t2pm":16, "mmean":3}, # 11
        # LS220 q=1 LK
        {"sim": "LS220_M13641364_M0_LK_SR_restart", "m": 1, "plot": {"color": "red", "ls": ":", "lw": 1.0, "alpha": 1.}, "type": "short", "t2pm":14, "mmean":3},  # 12
        {"sim": "LS220_M13641364_M0_LK_SR_restart", "m": 2, "plot": {"color": "red", "ls": "-", "lw": 1.0, "alpha": 1., "label": "LS220 q=1.00 (SR)"}, "type": "short", "t2pm":14, "mmean":3},  # 13

        # LS220 q=1.43
        {"sim": "LS220_M11461635_M0_LK_SR",  "m":1, "plot": {"color": "orange", "ls": ":", "lw": 0.8, "alpha": 1.}, "type": "short"}, # 14
        {"sim": "LS220_M11461635_M0_LK_SR",  "m":2, "plot": {"color": "orange", "ls": "-", "lw": 0.8, "alpha": 1., "label": "LS220 q=1.43 (SR)"}, "type": "short"}, # 15

        # SLy4  q=1
        {"sim": "SLy4_M13641364_M0_SR",      "m":1, "plot": {"color": "magenta", "ls": ":", "lw": 0.8, "alpha": 1.}, "type": "short"}, # 16
        {"sim": "SLy4_M13641364_M0_SR",      "m":2, "plot": {"color": "magenta", "ls": "-", "lw": 0.8, "alpha": 1., "label": "SLy4* q=1.00 (SR)"}, "type": "short"}, # 17
        # SLy4  q=1.13
        {"sim": "SLy4_M14521283_M0_SR",      "m":1, "plot": {"color": "purple", "ls": ":", "lw": 0.8, "alpha": 1.}, "type": "short"}, # 18
        {"sim": "SLy4_M14521283_M0_SR",      "m":2, "plot": {"color": "purple", "ls": "-", "lw": 0.8, "alpha": 1., "label": "SLy4* q=1.13 (SR)"}, "type": "short"}, # 19
    ]

    for t in task:
        t["v_n"] = v_n
        t["fpath"] = fpath
        t["norm_to_m"] = norm_to_m
    for t in task:
        t["plot"]["color"] = md.sim_dic_color[t["sim"]]
        #t["ls"] = md.sim_dic_ls[t["sim"]]
        t["plot"]["lw"] = md.sim_dic_lw[t["sim"]]
    ##

    def_plot_dic = {
        "figsize": (6., 5.5),
        "type": "all",
        "xmin": 0, "xmax": 100,
        "ymin": 1e-3, "ymax": 1e-1,
        "normalize": True,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "log",
        "xlabel": r"$t-t_{\rm merg}$ [ms]",
        "ylabel": r'$|C_m|/C_0$',#r'$C_m(\rho)/C_0(\rho)$',
        "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "total_rho_mode.png",
        "savepdf": True,
        "fontsize": 14,
        #
        "legend": {"fancybox": False, "loc": 'upper right',
                   #"bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "modelegend":{
            "modes": [
                {"label":"m=1", "ls":":", "lw":0.6, "alpha":1., "color":"gray"},
                {"label":"m=2", "ls":"-", "lw":0.9, "alpha":1., "color":"gray"}
            ],
            "legend": {"fancybox": False, "loc": 'lower right',
                      "shadow": "False", "ncol": 1, "fontsize": 14,
                      "framealpha": 0., "borderaxespad": 0., "frameon": False}
        },
    }

    # # Long with q = 1
    def_plot_dic["title"] = r"Long-lived remnants with q=1.00"
    def_plot_dic["figname"] = __outplotdir__ + "dens_modes/" + "modes_rho_q1_long.png"
    tasks=[task[i] for i in [0,1,4,5]]
    # plot_dens_modes_2D_2(tasks, def_plot_dic)

    # # Long with EOS = BLh
    def_plot_dic["title"] = r"Long-lived BLh"
    def_plot_dic["figname"] = __outplotdir__ + "dens_modes/" + "modes_rho_blh_long.png"
    tasks=[task[i] for i in [0,1,2,3]]
    # plot_dens_modes_2D_2(tasks, def_plot_dic)

    # # Long with EOS = DD2 q = 1.00
    def_plot_dic["title"] = r"Long-lived DD2 q=1.00"
    def_plot_dic["figname"] = __outplotdir__ + "dens_modes/" + "modes_rho_dd2_long.png"
    tasks=[task[i] for i in [4,5,6,7]]
    plot_dens_modes_2D_2(tasks, def_plot_dic)

    ''' ----------+--------- '''

    # # Short with q = 1
    def_plot_dic["title"] = r"Short-lived remnants with q=1.00"
    def_plot_dic["figname"] = __outplotdir__ + "dens_modes/" + "modes_rho_q1_short.png"
    def_plot_dic["xmax"] = 30
    def_plot_dic["ymax"] = 1e-0
    tasks=[task[i] for i in [8, 9, 12, 13]]
    # plot_dens_modes_2D_2(tasks, def_plot_dic)

    # # Short with ls220
    def_plot_dic["title"] = r"Short-lived LS220 q=1.00"
    def_plot_dic["figname"] = __outplotdir__ + "dens_modes/" + "modes_rho_ls220_short.png"
    def_plot_dic["xmax"] = 20
    tasks=[task[i] for i in [10,11,12,13]]
    plot_dens_modes_2D_2(tasks, def_plot_dic)

    # # Short with sly4
    def_plot_dic["title"] = r"Short-lived SLy4 q=1.00"
    def_plot_dic["figname"] = __outplotdir__ + "dens_modes/" + "modes_rho_sly4_short.png"
    def_plot_dic["xmax"] = 30
    tasks=[task[i] for i in [12, 13, 14, 15]]
    # plot_dens_modes_2D_2(tasks, def_plot_dic)




    # v_n = "profiles/" + "density_modes_lap15.h5"
    # for t in task: t["fpath"] = v_n
    #
    #
    # ''' ---------------- rho modes ------------------- '''
    #
    # v_n = "rho_modes.h5"
    # fpath = "slices/" + "rho_modes.h5"
    # m = 1
    # norm_to_m = 0
    # task = [
    #     {"sim": "BLh_M13641364_M0_LK_SR", "color": "black", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long",
    #      "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
    #     # {"sim": "BLh_M11841581_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"NS", "v_n": v_n, "t": -1},
    #     {"sim": "BLh_M11461635_M0_LK_SR", "color": "black", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long",
    #      "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
    #     {"sim": "BLh_M10651772_M0_LK_SR", "color": "black", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
    #      "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
    #     # {"sim": "BLh_M10201856_M0_SR", "color": "black", "ls": "-", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
    #     # {"sim": "BLh_M10201856_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
    #     #
    #     {"sim": "DD2_M13641364_M0_SR", "color": "blue", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long",
    #      "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
    #     # {"sim": "DD2_M13641364_M0_SR_R04", "color": "blue", "ls": "--", "lw": 0.7, "alpha": 1., "t1":60, "v_n": v_n, "t": -1},
    #     {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "blue", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long",
    #      "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
    #     {"sim": "DD2_M14971245_M0_SR", "color": "blue", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long",
    #      "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
    #     {"sim": "DD2_M15091235_M0_LK_SR", "color": "blue", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
    #      "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
    #     #
    #     # {"sim": "LS220_M13641364_M0_SR", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "t": -1},
    #     {"sim": "LS220_M13641364_M0_LK_SR_restart", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "type": "short",
    #      "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
    #     {"sim": "LS220_M14691268_M0_LK_SR", "color": "red", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long",
    #      "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
    #     {"sim": "LS220_M11461635_M0_LK_SR", "color": "red", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "short",
    #      "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
    #     {"sim": "LS220_M10651772_M0_LK_SR", "color": "red", "ls": ":", "lw": 0.7, "alpha": 1., "type": "short",
    #      "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
    #     #
    #     {"sim": "SFHo_M13641364_M0_SR", "color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short",
    #      "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
    #     {"sim": "SFHo_M14521283_M0_SR", "color": "green", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short",
    #      "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
    #     {"sim": "SFHo_M11461635_M0_LK_SR", "color": "green", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long",
    #      "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
    #     #
    #     {"sim": "SLy4_M13641364_M0_SR", "color": "magenta", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short",
    #      "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
    #     {"sim": "SLy4_M14521283_M0_SR", "color": "magenta", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short",
    #      "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
    #     # {"sim": "SLy4_M10201856_M0_LK_SR", "color": "orange", "ls": "-.", "lw": 0.7, "alpha": 1., "v_n": v_n, "t": -1},
    #     {"sim": "SLy4_M11461635_M0_LK_SR", "color": "magenta", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
    #      "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m}
    # ]
    #
    # for t in task: t["label"] = r"\texttt{" + t["sim"].replace('_', '\_') + "}"
    # for t in task:
    #     if t["sim"] == "LS220_M13641364_M0_LK_SR_restart":
    #         t["label"] = t["label"].replace("\_restart", "")
    #
    # def_plot_dic = {
    #     "figsize": (6., 2.5),
    #     "type": "long",
    #     "xmin": 0, "xmax": 100,
    #     "ymin": 1e-4, "ymax": 5e-1,
    #     "normalize": True,
    #     # "mask_below": 1e-15,
    #     "xscale": "linear", "yscale": "log",
    #     "xlabel": r"$t-t_{\rm merg}$ [ms]",
    #     "ylabel": r'$C_1(\rho)/C_0(\rho)$',
    #     "legend": {"fancybox": False, "loc": 'center left',
    #                "bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
    #                "shadow": "False", "ncol": 1, "fontsize": 10,
    #                "framealpha": 0., "borderaxespad": 0., "frameon": False},
    #     "title": r"Long-lived remnants",
    #     "figname": __outplotdir__ + "total_rho_mode_{}_long.png".format(m)
    # }
    #
    # plot_dens_modes_2D(task, def_plot_dic)
    #
    # ''' ------------- dens modes ------------------- '''
    #
    # # "profiles/" + "density_modes_lap15.h5"
    # v_n = "profiles/" + "density_modes_lap15.h5"
    # for t in task: t["fpath"] = v_n
    #
    # def_plot_dic["ymin"], def_plot_dic["ymax"] = 1e-3, 1e-1
    # def_plot_dic["figname"] = __outplotdir__ + "total_dens_mode_{}_long.png".format(m)
    # def_plot_dic["ylabel"] = r"$C_1(D)/C_0(D)$"
    # plot_dens_modes_2D(task, def_plot_dic)
    #
    # #
    # m = 2
    # for t in task: t["m"] = 2
    # v_n = "profiles/" + "density_modes_lap15.h5"
    # for t in task: t["fpath"] = v_n
    #
    # def_plot_dic["figname"] = __outplotdir__ + "total_dens_mode_{}_long.png".format(m)
    # def_plot_dic["ylabel"] = r"$C_2(D)/C_0(D)$"
    # plot_dens_modes_2D(task, def_plot_dic)
def task_plot_dens_modes_2D_3():
    """
    none
    """

    # __outplotdir__ = __outplotdir__ + "dens_modes/"
    v_n = "density_modes_lap15.h5"
    fpath = "profiles/" + "density_modes_lap15.h5"

    norm_to_m = 0
    task = [
        # BLh q = 1
        {"sim": "BLh_M13641364_M0_LK_SR", "m": 1, "plot": {"color": "black", "ls": "--", "lw": 0.8, "alpha": 1.}, "type": "long"},  # 0
        {"sim": "BLh_M13641364_M0_LK_SR", "m": 2, "plot": {"color": "black", "ls": "-", "lw": 0.8, "alpha": 1., "label": "BLh* q=1.00 (SR)"}, "type": "long"},# 1
        # BLh q = 1.66
        {"sim": "BLh_M11461635_M0_LK_SR", "m": 1, "plot": {"color": "gray", "ls": "--", "lw": 0.8, "alpha": 1.},  "type": "long"},  # 2
        {"sim": "BLh_M11461635_M0_LK_SR", "m": 2,  "plot": {"color": "gray", "ls": "-", "lw": 0.8, "alpha": 1., "label": "BLh* q=1.42 (SR)"}, "type": "long"},# 3

        # DD2 q = 1 noLK
        {"sim": "DD2_M13641364_M0_SR_R04", "m": 1, "plot": {"color": "green", "ls": "--", "lw": 0.8, "alpha": 1.},  "type": "long", "t1": 40},  # 4
        {"sim": "DD2_M13641364_M0_SR_R04", "m": 2, "plot": {"color": "green", "ls": "-", "lw": 1.0, "alpha": 1., "label": "DD2 q=1.00 (SR)"}, "type": "long"},# 5
        # DD2 q = 1.0 LK
        {"sim": "DD2_M13641364_M0_LK_SR_R04", "m": 1, "plot": {"color": "blue", "ls": "--", "lw": 0.8, "alpha": 1.}, "type": "long", "t1": 40},  # 6
        {"sim": "DD2_M13641364_M0_LK_SR_R04", "m": 2, "plot": {"color": "blue", "ls": "-", "lw": 1.0, "alpha": 1., "label": "DD2* q=1.00 (SR)"}, "type": "long", "t1": 40},  # 7
        # DD2 q = 1.22
        {"sim": "DD2_M15091235_M0_LK_SR", "m": 1, "plot": {"color": "green", "ls": "--", "lw": 0.8, "alpha": 1.}, "type": "long"},  # 8
        {"sim": "DD2_M15091235_M0_LK_SR", "m": 2, "plot": {"color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "label": "DD2* q=1.22 (SR)"}, "type": "long"}, # 9

        # LS220 q=1 noLK
        {"sim": "LS220_M13641364_M0_SR", "m": 1, "plot": {"color": "orange", "ls": ":", "lw": 0.6, "alpha": 1.}, "type": "short", "int":[3, ["unispline", "linear", "unispline"]]},  # 10
        {"sim": "LS220_M13641364_M0_SR", "m": 2, "plot": {"color": "orange", "ls": "-", "lw": 0.9, "alpha": 1., "label": "LS220 q=1.00 (SR)"}, "type": "short", "int":[3, ["unispline", "linear", "unispline"]]},# 11
        # LS220 q=1 LK
        {"sim": "LS220_M13641364_M0_LK_SR_restart", "m": 1, "plot": {"color": "red", "ls": ":", "lw": 0.6, "alpha": 1.}, "type": "short", "int":[4, ["unispline", "linear", "linear", "unispline"]]},  # 12
        {"sim": "LS220_M13641364_M0_LK_SR_restart", "m": 2, "plot": {"color": "red", "ls": "-", "lw": 0.9, "alpha": 1., "label": "LS220* q=1.00 (SR)"}, "type": "short", "int":[4, ["unispline", "linear", "linear", "unispline"]]}, # 13

        # LS220 q=1.43
        {"sim": "LS220_M11461635_M0_LK_SR", "m": 1, "plot": {"color": "orange", "ls": ":", "lw": 0.6, "alpha": 1.}, "type": "short"},  # 14
        {"sim": "LS220_M11461635_M0_LK_SR", "m": 2, "plot": {"color": "orange", "ls": "-", "lw": 0.9, "alpha": 1., "label": "LS220* q=1.43 (SR)"}, "type": "short"},  # 15

        # SLy4  q=1
        {"sim": "SLy4_M13641364_M0_SR", "m": 1, "plot": {"color": "magenta", "ls": "--", "lw": 0.8, "alpha": 1.}, "type": "short"},  # 16
        {"sim": "SLy4_M13641364_M0_SR", "m": 2, "plot": {"color": "magenta", "ls": "-", "lw": 0.8, "alpha": 1., "label": "SLy4 q=1.00 (SR)"}, "type": "short"},  # 17
        # SLy4  q=1.13
        {"sim": "SLy4_M14521283_M0_SR", "m": 1, "plot": {"color": "purple", "ls": "--", "lw": 0.8, "alpha": 1.}, "type": "short"},  # 18
        {"sim": "SLy4_M14521283_M0_SR", "m": 2, "plot": {"color": "purple", "ls": "-", "lw": 0.8, "alpha": 1., "label": "SLy4 q=1.13 (SR)"}, "type": "short"}, # 19
    ]

    for t in task:
        t["v_n"] = v_n
        t["fpath"] = fpath
        t["norm_to_m"] = norm_to_m
    for t in task:
        t["plot"]["color"] = md.sim_dic_color[t["sim"]]
        #t["ls"] = md.sim_dic_ls[t["sim"]]
        # t["plot"]["lw"] = md.sim_dic_lw[t["sim"]]
    ##

    def_plot_dic = {
        "figsize": (6., 5.5),
        "type": "all",
        "xmin": 0, "xmax": 100,
        "ymin": 1e-5, "ymax": 1e-1,
        "normalize": True,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "log",
        "xlabel": r"$t-t_{\rm merg}$ [ms]",
        "ylabel": r'$|C_m|/C_0$', #r'$C_m/C_0$',
        "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "modes_dens.png",
        "savepdf": True,
        "fontsize": 14,
        #
        "legend": {"fancybox": False, "loc": 'upper right',
                   #"bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "modelegend":{
            "modes": [
                {"label":"m=1", "ls":":", "lw":0.6, "alpha":1., "color":"gray"},
                {"label":"m=2", "ls":"-", "lw":0.9, "alpha":1., "color":"gray"}
            ],
            "legend": {"fancybox": False, "loc": 'lower left',
                      "shadow": "False", "ncol": 1, "fontsize": 14,
                      "framealpha": 0., "borderaxespad": 0., "frameon": False}
        },
    }

    # # Long with q = 1
    def_plot_dic["title"] = r"Long-lived remnants with q=1.00"
    def_plot_dic["figname"] = __outplotdir__ + "dens_modes/" + "modes_dens_q1_long.png"
    tasks=[task[i] for i in [0,1,4,5]]
    # plot_dens_modes_2D_2(tasks, def_plot_dic)

    # # Long with EOS = BLh
    def_plot_dic["title"] = r"Long-lived BLh"
    def_plot_dic["figname"] = __outplotdir__ + "dens_modes/" + "modes_dens_blh_long.png"
    tasks=[task[i] for i in [0,1,2,3]]
    # plot_dens_modes_2D_2(tasks, def_plot_dic)

    # # Long with EOS = DD2
    def_plot_dic["title"] = r"Long-lived DD2"
    def_plot_dic["figname"] = __outplotdir__ + "dens_modes/" + "modes_dens_dd2_long.png"
    tasks=[task[i] for i in [4,5,6,7]]
    plot_dens_modes_2D_2(tasks, def_plot_dic)

    ''' ----------+--------- '''

    # # Short with q = 1
    def_plot_dic["title"] = r"Short-lived remnants with q=1.00"
    def_plot_dic["figname"] = __outplotdir__ + "dens_modes/" + "modes_dens_q1_short.png"
    def_plot_dic["xmax"] = 30
    def_plot_dic["ymax"] = 1e-0
    tasks=[task[i] for i in [8, 9, 12, 13]]
    # plot_dens_modes_2D_2(tasks, def_plot_dic)

    # # Short with ls220
    def_plot_dic["title"] = r"Short-lived LS220 q=1.00"
    def_plot_dic["figname"] = __outplotdir__ + "dens_modes/" + "modes_dens_ls220_short.png"
    def_plot_dic["xmax"] = 30
    tasks=[task[i] for i in [10,11,12,13]]
    plot_dens_modes_2D_2(tasks, def_plot_dic)

    # # Short with sly4
    def_plot_dic["title"] = r"Short-lived SLy4 q=1.00"
    def_plot_dic["figname"] = __outplotdir__ + "dens_modes/" + "modes_dens_sly4_short.png"
    def_plot_dic["xmax"] = 30
    tasks=[task[i] for i in [12, 13, 14, 15]]
    # plot_dens_modes_2D_2(tasks, def_plot_dic)

''' --- iteration 2 | tasks --- '''

def task_plot_rho_modes_2D_2():
    """
    none
    """


    v_n = "rho_modes.h5"
    fpath = "slices/" + "rho_modes.h5"

    norm_to_m = 0
    task = [
        # BLh q = 1
        {"sim": "BLh_M13641364_M0_LK_SR", "m":1, "plot": {"color": "black", "ls": "--", "lw": 0.8, "alpha": 1.}, "type": "long"}, # 0
        {"sim": "BLh_M13641364_M0_LK_SR", "m":2, "plot": {"color": "black", "ls": "-", "lw": 0.8, "alpha": 1., "label": "BLh* q=1.00 (SR)"}, "type": "long"}, # 1
        # BLh q = 1.66
        {"sim": "BLh_M11461635_M0_LK_SR", "m":1, "plot":{"color": "gray", "ls": "--", "lw": 0.8, "alpha": 1.}, "type": "long"}, # 2
        {"sim": "BLh_M11461635_M0_LK_SR", "m":2, "plot":{"color": "gray", "ls": "-", "lw": 0.8, "alpha": 1., "label": "BLh* q=1.42 (SR)"}, "type": "long", }, # 3

        # DD2 q = 1
        {"sim": "DD2_M13641364_M0_LK_SR_R04", "m":1, "plot": {"color": "blue", "ls": "--", "lw": 0.8, "alpha": 1.}, "type": "long", "t1": 40}, # 4
        {"sim": "DD2_M13641364_M0_LK_SR_R04", "m":2, "plot": {"color": "blue", "ls": "-", "lw": 0.8, "alpha": 1., "label": "DD2* q=1.00 (SR)"}, "type": "long", "t1": 40},# 5
        # DD2 q = 1.22
        {"sim": "DD2_M15091235_M0_LK_SR", "m":1, "plot": {"color": "green", "ls": "--", "lw": 0.8, "alpha": 1.}, "type": "long"}, # 6
        {"sim": "DD2_M15091235_M0_LK_SR", "m":2, "plot": {"color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "label": "DD2* q=1.22 (SR)"}, "type": "long"}, # 7

        # LS220 q=1
        {"sim": "LS220_M13641364_M0_LK_SR_restart", "m":1, "plot": {"color": "red", "ls": "--", "lw": 0.8, "alpha": 1.}, "type": "short"}, # 8
        {"sim": "LS220_M13641364_M0_LK_SR_restart", "m":2, "plot": {"color": "red", "ls": "-", "lw": 0.8, "alpha": 1., "label": "LS220* q=1.00 (SR)"}, "type": "short"}, # 9
        # LS220 q=1.43
        {"sim": "LS220_M11461635_M0_LK_SR", "m":1, "plot": {"color": "orange", "ls": "--", "lw": 0.8, "alpha": 1.}, "type": "short"}, # 10
        {"sim": "LS220_M11461635_M0_LK_SR", "m":2, "plot": {"color": "orange", "ls": "-", "lw": 0.8, "alpha": 1., "label": "LS220* q=1.43 (SR)"}, "type": "short"}, # 11

        # SLy4  q=1
        {"sim": "SLy4_M13641364_M0_SR", "m":1, "plot": {"color": "magenta", "ls": "--", "lw": 0.8, "alpha": 1.}, "type": "short"}, # 12
        {"sim": "SLy4_M13641364_M0_SR", "m":2, "plot": {"color": "magenta", "ls": "-", "lw": 0.8, "alpha": 1., "label": "SLy4 q=1.00 (SR)"}, "type": "short"}, # 13
        # SLy4  q=1.13
        {"sim": "SLy4_M14521283_M0_SR", "m":1, "plot": {"color": "purple", "ls": "--", "lw": 0.8, "alpha": 1.}, "type": "short"}, # 14
        {"sim": "SLy4_M14521283_M0_SR", "m":2, "plot": {"color": "purple", "ls": "-", "lw": 0.8, "alpha": 1., "label": "SLy4 q=1.13 (SR)"}, "type": "short"}, # 15
    ]

    for t in task:
        t["v_n"] = v_n
        t["fpath"] = fpath
        t["norm_to_m"] = norm_to_m

    ##

    def_plot_dic = {
        "figsize": (12., 3.5),
        "type": "all",
        "xmin": 0, "xmax": 100,
        "ymin": 1e-4, "ymax": 1e-1,
        "normalize": True,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "log",
        "xlabel": r"$t-t_{\rm merg}$ [ms]",
        "ylabel": r'$C_m(\rho)/C_0(\rho)$',
        "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "total_rho_mode.png",
        "savepdf": True,
        "fontsize": 14,
        #
        "legend": {"fancybox": False, "loc": 'upper right',
                   #"bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 12,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "modelegend":{
            "modes": [
                {"label":"m=1", "ls":"--", "lw":0.8, "alpha":1., "color":"gray"},
                {"label":"m=2", "ls":"-", "lw":0.8, "alpha":1., "color":"gray"}
            ],
            "legend": {"fancybox": False, "loc": 'lower right',
                      "shadow": "False", "ncol": 1, "fontsize": 12,
                      "framealpha": 0., "borderaxespad": 0., "frameon": False}
        },
    }

    # # Long with q = 1
    def_plot_dic["title"] = r"Long-lived remnants with q=1.00"
    def_plot_dic["figname"] = __outplotdir__ + "dens_modes/" + "total_rho_modes_q1_long.png"
    tasks=[task[i] for i in [0,1,4,5]]
    plot_dens_modes_2D_2(tasks, def_plot_dic)

    # # Long with EOS = BLh
    def_plot_dic["title"] = r"Long-lived BLh"
    def_plot_dic["figname"] = __outplotdir__ + "dens_modes/" + "total_rho_modes_blh_long.png"
    tasks=[task[i] for i in [0,1,2,3]]
    plot_dens_modes_2D_2(tasks, def_plot_dic)

    # # Long with EOS = DD2
    def_plot_dic["title"] = r"Long-lived DD2"
    def_plot_dic["figname"] = __outplotdir__ + "dens_modes/" + "total_rho_modes_dd2_long.png"
    tasks=[task[i] for i in [4,5,6,7]]
    plot_dens_modes_2D_2(tasks, def_plot_dic)

    ''' ----------+--------- '''

    # # Short with q = 1
    def_plot_dic["title"] = r"Short-lived remnants with q=1.00"
    def_plot_dic["figname"] = __outplotdir__ + "dens_modes/" + "total_rho_modes_q1_short.png"
    def_plot_dic["xmax"] = 30
    def_plot_dic["ymax"] = 1e-0
    tasks=[task[i] for i in [8, 9, 12, 13]]
    plot_dens_modes_2D_2(tasks, def_plot_dic)

    # # Short with ls220
    def_plot_dic["title"] = r"Short-lived remnants with q=1.00"
    def_plot_dic["figname"] = __outplotdir__ + "total_rho_modes_ls220_short.png"
    def_plot_dic["xmax"] = 30
    tasks=[task[i] for i in [8, 9, 10, 11]]
    plot_dens_modes_2D_2(tasks, def_plot_dic)

    # # Short with sly4
    def_plot_dic["title"] = r"Short-lived remnants with q=1.00"
    def_plot_dic["figname"] = __outplotdir__ + "total_rho_modes_sly4_short.png"
    def_plot_dic["xmax"] = 30
    tasks=[task[i] for i in [12, 13, 14, 15]]
    plot_dens_modes_2D_2(tasks, def_plot_dic)




    # v_n = "profiles/" + "density_modes_lap15.h5"
    # for t in task: t["fpath"] = v_n
    #
    #
    # ''' ---------------- rho modes ------------------- '''
    #
    # v_n = "rho_modes.h5"
    # fpath = "slices/" + "rho_modes.h5"
    # m = 1
    # norm_to_m = 0
    # task = [
    #     {"sim": "BLh_M13641364_M0_LK_SR", "color": "black", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long",
    #      "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
    #     # {"sim": "BLh_M11841581_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"NS", "v_n": v_n, "t": -1},
    #     {"sim": "BLh_M11461635_M0_LK_SR", "color": "black", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long",
    #      "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
    #     {"sim": "BLh_M10651772_M0_LK_SR", "color": "black", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
    #      "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
    #     # {"sim": "BLh_M10201856_M0_SR", "color": "black", "ls": "-", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
    #     # {"sim": "BLh_M10201856_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
    #     #
    #     {"sim": "DD2_M13641364_M0_SR", "color": "blue", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long",
    #      "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
    #     # {"sim": "DD2_M13641364_M0_SR_R04", "color": "blue", "ls": "--", "lw": 0.7, "alpha": 1., "t1":60, "v_n": v_n, "t": -1},
    #     {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "blue", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long",
    #      "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
    #     {"sim": "DD2_M14971245_M0_SR", "color": "blue", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long",
    #      "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
    #     {"sim": "DD2_M15091235_M0_LK_SR", "color": "blue", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
    #      "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
    #     #
    #     # {"sim": "LS220_M13641364_M0_SR", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "t": -1},
    #     {"sim": "LS220_M13641364_M0_LK_SR_restart", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "type": "short",
    #      "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
    #     {"sim": "LS220_M14691268_M0_LK_SR", "color": "red", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long",
    #      "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
    #     {"sim": "LS220_M11461635_M0_LK_SR", "color": "red", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "short",
    #      "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
    #     {"sim": "LS220_M10651772_M0_LK_SR", "color": "red", "ls": ":", "lw": 0.7, "alpha": 1., "type": "short",
    #      "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
    #     #
    #     {"sim": "SFHo_M13641364_M0_SR", "color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short",
    #      "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
    #     {"sim": "SFHo_M14521283_M0_SR", "color": "green", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short",
    #      "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
    #     {"sim": "SFHo_M11461635_M0_LK_SR", "color": "green", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long",
    #      "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
    #     #
    #     {"sim": "SLy4_M13641364_M0_SR", "color": "magenta", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short",
    #      "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
    #     {"sim": "SLy4_M14521283_M0_SR", "color": "magenta", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short",
    #      "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m},
    #     # {"sim": "SLy4_M10201856_M0_LK_SR", "color": "orange", "ls": "-.", "lw": 0.7, "alpha": 1., "v_n": v_n, "t": -1},
    #     {"sim": "SLy4_M11461635_M0_LK_SR", "color": "magenta", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
    #      "v_n": v_n, "fpath":fpath, "norm_to_m":norm_to_m, "m":m}
    # ]
    #
    # for t in task: t["label"] = r"\texttt{" + t["sim"].replace('_', '\_') + "}"
    # for t in task:
    #     if t["sim"] == "LS220_M13641364_M0_LK_SR_restart":
    #         t["label"] = t["label"].replace("\_restart", "")
    #
    # def_plot_dic = {
    #     "figsize": (6., 2.5),
    #     "type": "long",
    #     "xmin": 0, "xmax": 100,
    #     "ymin": 1e-4, "ymax": 5e-1,
    #     "normalize": True,
    #     # "mask_below": 1e-15,
    #     "xscale": "linear", "yscale": "log",
    #     "xlabel": r"$t-t_{\rm merg}$ [ms]",
    #     "ylabel": r'$C_1(\rho)/C_0(\rho)$',
    #     "legend": {"fancybox": False, "loc": 'center left',
    #                "bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
    #                "shadow": "False", "ncol": 1, "fontsize": 10,
    #                "framealpha": 0., "borderaxespad": 0., "frameon": False},
    #     "title": r"Long-lived remnants",
    #     "figname": __outplotdir__ + "total_rho_mode_{}_long.png".format(m)
    # }
    #
    # plot_dens_modes_2D(task, def_plot_dic)
    #
    # ''' ------------- dens modes ------------------- '''
    #
    # # "profiles/" + "density_modes_lap15.h5"
    # v_n = "profiles/" + "density_modes_lap15.h5"
    # for t in task: t["fpath"] = v_n
    #
    # def_plot_dic["ymin"], def_plot_dic["ymax"] = 1e-3, 1e-1
    # def_plot_dic["figname"] = __outplotdir__ + "total_dens_mode_{}_long.png".format(m)
    # def_plot_dic["ylabel"] = r"$C_1(D)/C_0(D)$"
    # plot_dens_modes_2D(task, def_plot_dic)
    #
    # #
    # m = 2
    # for t in task: t["m"] = 2
    # v_n = "profiles/" + "density_modes_lap15.h5"
    # for t in task: t["fpath"] = v_n
    #
    # def_plot_dic["figname"] = __outplotdir__ + "total_dens_mode_{}_long.png".format(m)
    # def_plot_dic["ylabel"] = r"$C_2(D)/C_0(D)$"
    # plot_dens_modes_2D(task, def_plot_dic)
def task_plot_dens_modes_2D_2():
    """
    none
    """

    # __outplotdir__ = __outplotdir__ + "dens_modes/"
    v_n = "density_modes_lap15.h5"
    fpath = "profiles/" + "density_modes_lap15.h5"

    norm_to_m = 0
    task = [
        # BLh q = 1
        {"sim": "BLh_M13641364_M0_LK_SR", "m":1, "plot": {"color": "black", "ls": "--", "lw": 0.8, "alpha": 1.}, "type": "long"}, # 0
        {"sim": "BLh_M13641364_M0_LK_SR", "m":2, "plot": {"color": "black", "ls": "-", "lw": 0.8, "alpha": 1., "label": "BLh* q=1.00 (SR)"}, "type": "long"}, # 1
        # BLh q = 1.66
        {"sim": "BLh_M11461635_M0_LK_SR", "m":1, "plot":{"color": "gray", "ls": "--", "lw": 0.8, "alpha": 1.}, "type": "long"}, # 2
        {"sim": "BLh_M11461635_M0_LK_SR", "m":2, "plot":{"color": "gray", "ls": "-", "lw": 0.8, "alpha": 1., "label": "BLh* q=1.42 (SR)"}, "type": "long"}, # 3

        # DD2 q = 1
        {"sim": "DD2_M13641364_M0_LK_SR_R04", "m":1, "plot": {"color": "blue", "ls": "--", "lw": 0.8, "alpha": 1.}, "type": "long", "t1": 40}, # 4
        {"sim": "DD2_M13641364_M0_LK_SR_R04", "m":2, "plot": {"color": "blue", "ls": "-", "lw": 0.8, "alpha": 1., "label": "DD2* q=1.00 (SR)"}, "type": "long", "t1": 40},# 5
        # DD2 q = 1.22
        {"sim": "DD2_M15091235_M0_LK_SR", "m":1, "plot": {"color": "green", "ls": "--", "lw": 0.8, "alpha": 1.}, "type": "long"}, # 6
        {"sim": "DD2_M15091235_M0_LK_SR", "m":2, "plot": {"color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "label": "DD2* q=1.22 (SR)"}, "type": "long"}, # 7

        # LS220 q=1
        {"sim": "LS220_M13641364_M0_LK_SR_restart", "m":1, "plot": {"color": "red", "ls": "--", "lw": 0.8, "alpha": 1.}, "type": "short"}, # 8
        {"sim": "LS220_M13641364_M0_LK_SR_restart", "m":2, "plot": {"color": "red", "ls": "-", "lw": 0.8, "alpha": 1., "label": "LS220* q=1.00 (SR)"}, "type": "short"}, # 9
        # LS220 q=1.43
        {"sim": "LS220_M11461635_M0_LK_SR", "m":1, "plot": {"color": "orange", "ls": "--", "lw": 0.8, "alpha": 1.}, "type": "short"}, # 10
        {"sim": "LS220_M11461635_M0_LK_SR", "m":2, "plot": {"color": "orange", "ls": "-", "lw": 0.8, "alpha": 1., "label": "LS220* q=1.43 (SR)"}, "type": "short"}, # 11

        # SLy4  q=1
        {"sim": "SLy4_M13641364_M0_SR", "m":1, "plot": {"color": "magenta", "ls": "--", "lw": 0.8, "alpha": 1.}, "type": "short"}, # 12
        {"sim": "SLy4_M13641364_M0_SR", "m":2, "plot": {"color": "magenta", "ls": "-", "lw": 0.8, "alpha": 1., "label": "SLy4 q=1.00 (SR)"}, "type": "short"}, # 13
        # SLy4  q=1.13
        {"sim": "SLy4_M14521283_M0_SR", "m":1, "plot": {"color": "purple", "ls": "--", "lw": 0.8, "alpha": 1.}, "type": "short"}, # 14
        {"sim": "SLy4_M14521283_M0_SR", "m":2, "plot": {"color": "purple", "ls": "-", "lw": 0.8, "alpha": 1., "label": "SLy4 q=1.13 (SR)"}, "type": "short"}, # 15
    ]

    for t in task:
        t["v_n"] = v_n
        t["fpath"] = fpath
        t["norm_to_m"] = norm_to_m

    ##

    def_plot_dic = {
        "figsize": (12., 3.5),
        "type": "all",
        "xmin": 0, "xmax": 100,
        "ymin": 1e-4, "ymax": 1e-1,
        "normalize": True,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "log",
        "xlabel": r"$t-t_{\rm merg}$ [ms]",
        "ylabel": r'$C_m/C_0$',
        "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "total_dens_mode.png",
        "savepdf": True,
        "fontsize": 14,
        #
        "legend": {"fancybox": False, "loc": 'upper right',
                   #"bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 12,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "modelegend":{
            "modes": [
                {"label":"m=1", "ls":"--", "lw":0.8, "alpha":1., "color":"gray"},
                {"label":"m=2", "ls":"-", "lw":0.8, "alpha":1., "color":"gray"}
            ],
            "legend": {"fancybox": False, "loc": 'lower right',
                      "shadow": "False", "ncol": 1, "fontsize": 12,
                      "framealpha": 0., "borderaxespad": 0., "frameon": False}
        },
    }

    # # Long with q = 1
    def_plot_dic["title"] = r"Long-lived remnants with q=1.00"
    def_plot_dic["figname"] = __outplotdir__ + "dens_modes/" + "total_dens_modes_q1_long.png"
    tasks=[task[i] for i in [0,1,4,5]]
    plot_dens_modes_2D_2(tasks, def_plot_dic)

    # # Long with EOS = BLh
    def_plot_dic["title"] = r"Long-lived BLh"
    def_plot_dic["figname"] = __outplotdir__ + "dens_modes/" + "total_dens_modes_blh_long.png"
    tasks=[task[i] for i in [0,1,2,3]]
    plot_dens_modes_2D_2(tasks, def_plot_dic)

    # # Long with EOS = DD2
    def_plot_dic["title"] = r"Long-lived DD2"
    def_plot_dic["figname"] = __outplotdir__ + "dens_modes/" + "total_dens_modes_dd2_long.png"
    tasks=[task[i] for i in [4,5,6,7]]
    plot_dens_modes_2D_2(tasks, def_plot_dic)

    ''' ----------+--------- '''

    # # Short with q = 1
    def_plot_dic["title"] = r"Short-lived remnants with q=1.00"
    def_plot_dic["figname"] = __outplotdir__ + "dens_modes/" + "total_dens_modes_q1_short.png"
    def_plot_dic["xmax"] = 30
    def_plot_dic["ymax"] = 1e-0
    tasks=[task[i] for i in [8, 9, 12, 13]]
    plot_dens_modes_2D_2(tasks, def_plot_dic)

    # # Short with ls220
    def_plot_dic["title"] = r"Short-lived remnants with q=1.00"
    def_plot_dic["figname"] = __outplotdir__ + "dens_modes/" + "total_dens_modes_ls220_short.png"
    def_plot_dic["xmax"] = 30
    tasks=[task[i] for i in [8, 9, 10, 11]]
    plot_dens_modes_2D_2(tasks, def_plot_dic)

    # # Short with sly4
    def_plot_dic["title"] = r"Short-lived remnants with q=1.00"
    def_plot_dic["figname"] = __outplotdir__ + "dens_modes/" + "total_dens_modes_sly4_short.png"
    def_plot_dic["xmax"] = 30
    tasks=[task[i] for i in [12, 13, 14, 15]]
    plot_dens_modes_2D_2(tasks, def_plot_dic)

def task_plot_total_angular_momentum_flux_colormesh_2():

    task = {
        "sim": "BLh_M13641364_M0_LK_SR",
        "v_n": "Jflux"
    }

    def_plotdic = {
                   "figsize": (6., 5.0),
                   "vmin": 1e-6, "vmax": 1e-4,
                   "xmin": 0, "xmax": 90,
                   "ymin": 100, "ymax": 700,
                   "cmap": "jet",#"Spectral_r",
                   "set_under": "black",
                   "set_over": "red",
                   "yscale": "linear",
                   "xscale": "linear",
                   "xlabel": r"$t-t_{\rm merg}$ [ms]",
                   "ylabel": r"$R_{\rm cyl}$ [km]",
                   "title": "BLh q=1.00 (SR)",  # + "[{}ms]".format(task["t1"]),
                   "clabel": r"$Jf_{r}$ [$G\, c^{-1} M_\odot$]",
                   # "text": {"x": 0.3, "y": 0.95, "s": r"$\theta>60\deg$", "ha": "center", "va": "top", "fontsize": 11,
                   #          "color": "white",
                   #          "transform": None},
                   "outdir": __outplotdir__,
                   "figname": "evol_jflux_2d_{}_R1.png".format(task["sim"]),
                   "savepdf":True,
                   "fontsize":14,
                   }

    plot_total_angular_momentum_colormesh(task, def_plotdic)
    #

    plot_dic = copy.deepcopy(def_plotdic)
    task["sim"] = "BLh_M11461635_M0_LK_SR"
    plot_dic["title"] = r"BLh q=1.43 (SR)"  # + "[{}ms]".format(task["t1"]),
    plot_dic["figname"] = "evol_jflux_2d_{}_R1.png".format(task["sim"])
    plot_total_angular_momentum_colormesh(task, plot_dic)
    #
    plot_dic = copy.deepcopy(def_plotdic)
    plot_dic["title"] = r"\texttt{" + task["sim"].replace("_", "\_") + "}"  # + "[{}ms]".format(task["t1"]),
    plot_dic["figname"] = "evol_jflux_2d_{}_R1.png".format(task["sim"])
    #plot_total_angular_momentum_colormesh(task, plot_dic)
    #

    task["sim"] = "DD2_M13641364_M0_LK_SR_R04"
    plot_dic["title"] = r"DD2 q=1.00 (SR)"  # + "[{}ms]".format(task["t1"]),
    plot_dic["figname"] = "evol_jflux_2d_{}_R1.png".format(task["sim"])
    plot_total_angular_momentum_colormesh(task, plot_dic)
    #
    task["sim"] = "DD2_M13641364_M0_SR"
    plot_dic["title"] = r"DD2* q=1.00 (SR)"  # + "[{}ms]".format(task["t1"]),
    plot_dic["figname"] = "evol_jflux_2d_{}_R1.png".format(task["sim"])
    plot_total_angular_momentum_colormesh(task, plot_dic)

def task_plot_total_angular_momentum_2():

    task = [

        {"sim":"BLh_M13641364_M0_LK_SR", "v_n": "J", "rext":">8.2","plot":{"color":"black","ls":"-","lw":0.8,"label":r"BLh* q=1.00 (SR)"}},
        {"sim": "BLh_M13641364_M0_LK_SR", "v_n": "J", "rext": "<8.2", "plot": {"color": "gray", "ls": "--", "lw": 0.8}},

        {"sim": "DD2_M13641364_M0_SR", "v_n": "J", "rext": ">8.2", "plot": {"color": "blue", "ls": "-", "lw": 0.8, "label": r"DD2 q=1.00 (SR)"}},
        {"sim": "DD2_M13641364_M0_SR", "v_n": "J", "rext": "<8.2", "plot": {"color": "cyan", "ls": "--", "lw": 0.8}}
    ]

    for t in task:
        t["plot"]["color"] = md.sim_dic_color[t["sim"]]
        #t["plot"]["ls"] = md.sim_dic_ls[t["sim"]]
        t["plot"]["lw"] = md.sim_dic_lw[t["sim"]]

    # v_n = "J"
    #
    #
    #
    # v_n = "J"
    # rext = "<15"
    #
    # task = [
    #     {"sim": "BLh_M13641364_M0_LK_SR", "color": "black", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long",
    #      "v_n": v_n},
    #     # {"sim": "BLh_M11841581_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"NS", "v_n": v_n, "t": -1},
    #     {"sim": "BLh_M11461635_M0_LK_SR", "color": "black", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long",
    #      "v_n": v_n},
    #     {"sim": "BLh_M10651772_M0_LK_SR", "color": "black", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
    #      "v_n": v_n},
    #     # {"sim": "BLh_M10201856_M0_SR", "color": "black", "ls": "-", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
    #     # {"sim": "BLh_M10201856_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
    #     #
    #     {"sim": "DD2_M13641364_M0_SR_R04", "color": "blue", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long",
    #      "v_n": v_n, "tmin":70},
    #     # {"sim": "DD2_M13641364_M0_SR_R04", "color": "blue", "ls": "--", "lw": 0.7, "alpha": 1., "t1":60, "v_n": v_n, "t": -1},
    #     {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "blue", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long",
    #      "v_n": v_n, "tmin":40},
    #     {"sim": "DD2_M14971245_M0_SR", "color": "blue", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long",
    #      "v_n": v_n},
    #     {"sim": "DD2_M15091235_M0_LK_SR", "color": "blue", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
    #      "v_n": v_n},
    #     #
    #     # {"sim": "LS220_M13641364_M0_SR", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "t": -1},
    #     {"sim": "LS220_M13641364_M0_LK_SR_restart", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "type": "short",
    #      "v_n": v_n},
    #     {"sim": "LS220_M14691268_M0_LK_SR", "color": "red", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long",
    #      "v_n": v_n, "tmax":70},
    #     {"sim": "LS220_M11461635_M0_LK_SR", "color": "red", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "short",
    #      "v_n": v_n},
    #     {"sim": "LS220_M10651772_M0_LK_SR", "color": "red", "ls": ":", "lw": 0.7, "alpha": 1., "type": "short",
    #      "v_n": v_n},
    #     #
    #     {"sim": "SFHo_M13641364_M0_SR", "color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short",
    #      "v_n": v_n},
    #     {"sim": "SFHo_M14521283_M0_SR", "color": "green", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short",
    #      "v_n": v_n},
    #     {"sim": "SFHo_M11461635_M0_LK_SR", "color": "green", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long",
    #      "v_n": v_n},
    #     #
    #     {"sim": "SLy4_M13641364_M0_SR", "color": "magenta", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short",
    #      "v_n": v_n},
    #     {"sim": "SLy4_M14521283_M0_SR", "color": "magenta", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short",
    #      "v_n": v_n},
    #     # {"sim": "SLy4_M10201856_M0_LK_SR", "color": "orange", "ls": "-.", "lw": 0.7, "alpha": 1., "v_n": v_n, "t": -1},
    #     {"sim": "SLy4_M11461635_M0_LK_SR", "color": "magenta", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
    #      "v_n": v_n}
    # ]

    # for t in task: t["label"] = r"\texttt{" + t["sim"].replace('_', '\_') + "}"
    # for t in task:
    #     if t["sim"] == "LS220_M13641364_M0_LK_SR_restart":
    #         t["label"] = t["label"].replace("\_restart", "")

    # rext = "<500"
    # for t in task: t["rext"] = rext
    plot_dic = {
        "figsize": (6., 5.5),
        "type": "all",
        "xmin": 0, "xmax": 1e2,
        "ymin": 0.0, "ymax": 6.2,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "linear",
        "xlabel": r"$t-t_{\rm merg}$ [ms]",
        "ylabel": r"$J_{\rm tot}\ [G\, c^{-1} M_\odot^2]$",
        "legend": {"fancybox": False, "loc": 'upper right',
                   #"bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "multilegend": {
            "lines": [
                {"label": r"$R_{\rm ext}>12$km", "ls": "-", "lw": 0.8, "alpha": 1., "color": "gray"},
                {"label": r"$R_{\rm ext}<12$km", "ls": "--", "lw": 0.8, "alpha": 1., "color": "gray"}
            ],
            "legend": {"fancybox": False, "loc": 'upper left',
                       "shadow": "False", "ncol": 1, "fontsize": 14,
                       "framealpha": 0., "borderaxespad": 0., "frameon": False}
        },
        "title": r"BLh* q=1.00 (SR)",
        "figname": __outplotdir__ + "ang_mom_evo.png",
        "savepdf":True,
        "fontsize":14
    }
    plot_total_angular_momentum(task, plot_dic)

    # ''' ---- r < 15 ---- '''
    # rext = "<8.2"
    # for t in task: t["rext"] = rext
    #
    # # --- LONG ---
    # plot_dic = {
    #     "figsize": (6., 2.5),
    #     "type": "long",
    #     "xmin": 0, "xmax": 1e2,
    #     "ymin": 3, "ymax": 6.5,
    #     # "mask_below": 1e-15,
    #     "xscale": "linear", "yscale": "linear",
    #     "xlabel": r"$t-t_{\rm merg}$ [ms]",
    #     "ylabel": r"$J_{r<12}\ [G\, c^{-1} M_\odot^2]$",
    #     "legend": {"fancybox": False, "loc": 'center left',
    #                "bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
    #                "shadow": "False", "ncol": 1, "fontsize": 10,
    #                "framealpha": 0., "borderaxespad": 0., "frameon": False},
    #     "title": r"Long-lived remnants",
    #     "figname": __outplotdir__ + "total_ang_mom_evo_r0_15_long.png"
    # }
    # plot_total_angular_momentum(task, plot_dic)

    # # --- SHORT ---
    # plot_dic["type"] = "short"
    # plot_dic["xmax"] = 20
    # plot_dic["ymin"], plot_dic["ymax"] = 3, 7.
    # plot_dic["title"] = "Short-lived remnants"
    # plot_dic["figname"] = __outplotdir__ + "total_ang_mom_evo_r0_15_short.png"
    # #
    # #plot_total_angular_momentum(task, plot_dic)
    #
    # ''' --- r > 15 ---- '''
    # rext = ">15"
    # for t in task: t["rext"] = rext
    # # --- LONG ---
    # plot_dic["type"] = "long"
    # plot_dic["xmin"],plot_dic["xmax"] = 0, 1e2
    # plot_dic["ymin"],plot_dic["ymax"] = 0, 3
    # plot_dic["title"] = "Long-lived remnants"
    # plot_dic["ylabel"] = r"$J_{r>15}\ [G\, c^{-1} M_\odot^2]$"
    # plot_dic["figname"] = __outplotdir__ + "total_ang_mom_evo_r15_500_long.png"
    #
    # # plot_total_angular_momentum(task, plot_dic)
    #
    # # --- SHORT ---
    # plot_dic["type"] = "short"
    # plot_dic["xmax"] = 20
    # plot_dic["ymin"], plot_dic["ymax"] = 0, 2
    # plot_dic["title"] = "Short-lived remnants"
    # plot_dic["figname"] = __outplotdir__ + "total_ang_mom_evo_r15_short.png"
    #
    # plot_total_angular_momentum(task, plot_dic)

''' ===================== | PAPER | ========================'''

if __name__ == '__main__':

    ''' --- density modes --- '''

    task_plot_rho_modes_2D_3()
    task_plot_dens_modes_2D_3()

    ''' --- total angular momentum evolution --- '''

    # task_plot_total_angular_momentum_2()

    ''' --- angular momentum flux colormesh --- '''
    # task_plot_total_angular_momentum_flux_colormesh_2()

''' --- iteration 3 --- '''

if __name__ == '__main__':

    #task_plot_rho_modes_2D_3()
    #task_plot_dens_modes_2D_3()

    #exit(0)
    pass

''' --- iteration 2 --- '''

if __name__ == '__main__':

    ''' --- density modes --- '''
    # task_plot_rho_modes_2D_2()
    # task_plot_dens_modes_2D_2()

    ''' --- angular momentum colormesh --- '''
    #task_plot_total_angular_momentum_flux_colormesh_2()

    ''' --- total angular momentum --- '''

    #task_plot_total_angular_momentum_2()

    ''' end '''
    #exit(1)
    #pass

''' --- iteration 0 --- '''

if __name__ == '__main__':


    """ --- central density --- """
    # task_plot_rho_max()

    """ --- density modes 2D & 3D --- """
    # task_plot_dens_modes_2D()

    """ --- center of mass 3D --- """
    # task_plot_center_mass_r_3D()

    """ --- remnant mass --- """
    # task_plot_remnant_mass_evo()

    """ --- timecorr --- """
    # task_plot_remnant_timecorr()

    """ --- massave quantities --- """
    # task_plot_mass_ave_val_evo()

    """ --- angular momentum [colormesh] --- """
    # task_plot_total_angular_momentum_colormesh()

    """ --- angular momentum flux [colormesh] --- """
    # task_plot_total_angular_momentum_flux_colormesh()

    """ --- total angular momentum --- """
    #task_plot_total_angular_momentum()

    """ --- total angular momentum flux --- """
    # task_plot_total_angular_momentum_flux()

    """ --- total angular momentum flux --- """
    # task_plot_total_djdt()

    """ --- total -dJ/dt - Jflux --- """
    # task_plot_total_djdt_minus_Jflux()

    """ -------------------------- RESOLUTION ------------------------ """
    # task_resolution_plot_total_angular_momentum_flux()

    # task_resolution_plot_total_angular_momentum()

    # task_resolution_plot_total_djdt_minus_Jflux()