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

import model_sets.models as md

from uutils import *

__outplotdir__ = "../figs/all3/plot_disk_struct/"
if not os.path.isdir(__outplotdir__):
    os.mkdir(__outplotdir__)

''' ----------------------- MODULES ------------------------- '''

def plot_rho_max(tasks, plotdic):
    fig = plt.figure(figsize=plotdic["figsize"])
    ax = fig.add_subplot(111)
    #
    # labels

    for task in tasks:
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

def plot_tot_disk_mass_evo(tasks, plotdic):

    fig = plt.figure(figsize=plotdic["figsize"])
    ax = fig.add_subplot(111)
    #
    # labels

    for task in tasks:
        if plotdic["type"] == "all" or task["type"] == plotdic["type"]:
            o_data = ADD_METHODS_ALL_PAR(task["sim"])
            it, times, data = o_data.get_disk_mass()
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

    ax.set_yscale(plotdic["yscale"], fontsize=plotdic["fontsize"])
    ax.set_xscale(plotdic["xscale"], fontsize=plotdic["fontsize"])

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

    if "aspect" in plotdic.keys(): ax.set_aspect(**plotdic["aspect"])

    ax.set_title(plotdic["title"], fontsize=plotdic["fontsize"])
    # #

    #
    ax.legend(**plotdic["legend"])

    plt.tight_layout()
    #

    print("plotted: \n")
    print(plotdic["figname"])
    plt.savefig(plotdic["figname"], dpi=128)
    if plotdic["savepdf"]: plt.savefig(plotdic["figname"].replace(".png", ".pdf"))
    plt.close()

def plot_final_disk_histogram(tasks, plotdic):


    fig = plt.figure(figsize=plotdic["figsize"])
    ax = fig.add_subplot(111)
    #
    # labels

    averages = []
    for task in tasks:
        if plotdic["type"] == "all" or task["type"] == plotdic["type"]:
            o_data = ADD_METHODS_ALL_PAR(task["sim"])
            data = o_data.get_disk_hist(task["v_n"], t=task["t"])
            print(data.shape)
            # print(data.shape)
            bins, mass = data[0,:], data[1,:]
            # if len(table) < 2: raise ValueError()
            # print(table.shape)
            # it, times, data = table[:, 0], table[:, 1], table[:, 2]



            ave = np.sum(bins * mass) / np.sum(mass)
            averages.append(ave)
            # print("Average: | {} | {}".format(task["sim"], ave))
            if plotdic["normalize"]:  mass = mass / np.sum(mass)
            #
            # times = times * Constants.time_constant / 1000

            if task["v_n"] == "theta": bins = 90. - (180 * bins / np.pi)
            if task["v_n"] == "r": bins = bins * constant_length

            print("{} bin_min:{} bin_max:{} len:{}".format(task["sim"], bins.min(), bins.max(), len(bins)))
            ax.plot(bins, mass, color=task["color"], ls=task["ls"], lw=task["lw"], alpha=task["alpha"],
                    label=task["label"], drawstyle="steps")

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
    if "title" in plotdic.keys(): ax.set_title(plotdic["title"], fontsize=plotdic["fontsize"])
    # #

    #

    if len(plotdic["legend"].keys()) >0 : ax.legend(**plotdic["legend"])

    plt.tight_layout()
    #

    for task, val in zip(tasks, averages):
        print("\t{} \t {}".format(task["sim"], val))

    print("plotted: \n")

    print(plotdic["figname"])
    plt.savefig(plotdic["figname"], dpi=128)
    if plotdic["savepdf"]: plt.savefig(plotdic["figname"].replace(".png", ".pdf"))
    plt.close()

def custom_plot_final_disk_histogram(tasks, plotdics):



    fig, axes = plt.subplots(figsize=plotdics[0]["figsize"], ncols=len(plotdics), nrows=1, sharey=True)
    # ax = fig.add_subplot(111)
    i = 0
    for ax, plotdic in zip(axes, plotdics):
        #
        # labels

        for task in tasks:
            if True:#task["type"] == "all" or task["type"] == plotdic["type"]:

                o_data = ADD_METHODS_ALL_PAR(task["sim"])
                data = o_data.get_disk_hist(plotdic["task_v_n"], t=task["t"])
                print(data.shape)
                # print(data.shape)
                bins, mass = data[0, :], data[1, :]
                # if len(table) < 2: raise ValueError()
                # print(table.shape)
                # it, times, data = table[:, 0], table[:, 1], table[:, 2]

                ave = np.sum(bins * mass) / np.sum(mass)
                #averages.append(ave)
                # print("Average: | {} | {}".format(task["sim"], ave))
                if plotdic["normalize"]:  mass = mass / np.sum(mass)

                # times = times * Constants.time_constant / 1000

                if plotdic["task_v_n"] == "theta": bins = 90. - (180 * bins / np.pi)
                if plotdic["task_v_n"] == "r": bins = bins * constant_length

                print("{} bin_min:{} bin_max:{} len:{}".format(task["sim"], bins.min(), bins.max(), len(bins)))
                if not "plot" in task.keys():
                    ax.plot(bins, mass, color=task["color"], ls=task["ls"], lw=task["lw"], alpha=task["alpha"],
                        label=task["label"], drawstyle="steps")
                else:
                    ax.plot(bins, mass, **task["plot"])
                #

                # o_data = ADD_METHODS_ALL_PAR(task["sim"])
                # hist = o_data.get_outflow_hist(det=task["det"], mask=task["mask"], v_n = plotdic["task_v_n"])
                # dataarr = hist[0, :]
                # massarr = hist[1, :]
                # #
                # if plotdic["task_v_n"] == "theta": dataarr = 90 - (dataarr * 180 / np.pi)
                # if plotdic["task_v_n"] == "phi": dataarr = dataarr / np.pi * 180.
                # #
                # if task['normalize']: massarr /= np.sum(massarr)
                #
                # print(dataarr); exit(1)
                # if not "plot" in task.keys():
                #     ax.plot(dataarr, massarr, color=task["color"], ls=task["ls"], lw=task["lw"], alpha=task["alpha"],
                #             label=task["label"], drawstyle="steps")
                # else:
                #     ax.plot(dataarr, massarr, **task["plot"])

        ax.set_yscale(plotdic["yscale"])
        ax.set_xscale(plotdic["xscale"])

        ax.set_xlabel(plotdic["xlabel"], fontsize = plotdic["fontsize"])  # , fontsize=11)
        if i == 0: ax.set_ylabel(plotdic["ylabel"], fontsize = plotdic["fontsize"])  # , fontsize=11)

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
        if "title" in plotdic.keys(): ax.set_title(plotdic["title"])
        # #
        if i > 0:
            ax.tick_params(labelleft=False)
            #ax.get_yaxis().set_ticks([])
            # ax.get_yaxis().set_visible(False)
        if len(plotdic["legend"].keys())>0 : ax.legend(**plotdic["legend"])
        i = i + 1
    #
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.)
    #
    print("plotted: \n")
    print(plotdics[0]["figname"])
    plt.savefig(plotdics[0]["figname"], dpi=128)
    if "savepdf" in plotdics[0].keys() and plotdics[0]["savepdf"]:
        plt.savefig(plotdics[0]["figname"].replace(".png", ".pdf"))
    plt.close()

def plot_mass_ave_val_evo(tasks, plotdic):

    fig = plt.figure(figsize=plotdic["figsize"])
    ax = fig.add_subplot(111)
    #
    # labels

    for task in tasks:
        if task["type"] == "all" or task["type"] == plotdic["type"]:
            o_data = ADD_METHODS_ALL_PAR(task["sim"])
            table = o_data.get_disk_mass_ave_par_evo(v_n=task["v_n"])
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

def plot_disk_timecorr(task, plotdic):
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
    fig = plt.figure(figsize=plotdic["figsize"])
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
    ax.set_xlabel(plotdic["xlabel"], fontsize=plotdic['fontsize'])
    ax.set_ylabel(plotdic["ylabel"], fontsize=plotdic['fontsize'])
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
                   labelsize=plotdic['fontsize'],
                   direction='in',
                   bottom=True, top=True, left=True, right=True)
    ax.minorticks_on()
    #
    if "title" in plotdic.keys(): ax.set_title(plotdic["title"],fontsize=plotdic['fontsize'])
    #
    clb = fig.colorbar(im, ax=ax)
    plt.tight_layout()
    clb.ax.set_title(plotdic["clabel"], fontsize=plotdic['fontsize'])
    clb.ax.tick_params(labelsize=plotdic['fontsize'])
    clb.ax.minorticks_off() # use this if you are using David's file
    #
    print("plotted: \n")
    fname = plotdic["outdir"] + plotdic["figname"]
    print(fname)
    plt.savefig(fname, dpi=128)
    if plotdic["savepdf"]: plt.savefig(fname.replace(".png",".pdf"))
    plt.close()

def plot_final_disk_corr(task, plotdic):

    sim = task["sim"] # "BLh_M13641364_M0_LK_SR"
    t1, t2 = task["t1"], task["t2"]# 60, 80

    v_n_x = task["v_n_x"] # "Q_eff_nua"  # optd_0_nua optd_1_nua
    v_n_y = task["v_n_y"] # "dens_unb_bern"
    # plane = task["plane"]# "xz"
    # mask = task["mask"]

    if not os.path.isdir(plotdic["outdir"]):
        os.mkdir(plotdic["outdir"])

    o_methods = LOAD_RES_CORR(sim)
    d1class = ADD_METHODS_ALL_PAR(sim)

    all_x_arr = []
    all_y_arr = []
    all_z_arr = []

    _, iterations, alltimes = o_methods.get_ittime("profiles", "prof")
    if t1 == t2:
        # if the required times are -- just timesteps or >time
        print("Mode: {} = {} -> collecting the data".format(task["t1"], task["t2"]))
        times = task["t1"]
        if isinstance(times, str):
            val = float(str(times[1:]))
            # iterations = np.array(o_methods.list_iterations, dtype=int)
            alltimes = (alltimes - d1class.get_par("tmerg")) * 1e3
            iterations = iterations[alltimes >= val]
            alltimes = alltimes[alltimes >= val]
        elif isinstance(times, int) or isinstance(times, float):
            alltimes = [times]
            it = o_methods.get_it_for_time(t1 / 1.e3, output="profiles", d1d2d3="prof")
            iterations = [it]
        else:
            raise NameError("no method set for times:{}".format(times))
        #
        for it, t in zip(iterations, alltimes):
            print("loading data for it:{} time:{:.1f}".format(it, t))
            #it = o_methods.get_it_for_time(t1 / 1.e3, output="profiles", d1d2d3="prof")
            table = o_methods.get_res_corr(it, v_n_x, v_n_y)
            all_x_arr.append(np.array(table[0, 1:]))  # * 6.176269145886162e+17
            all_y_arr.append(np.array(table[1:, 0]))  # * -1.
            all_z_arr.append(np.array(table[1:, 1:]))
            #print(np.sum(all_z_arr))

    else:
        print("Mode: {} != {} -> collecting the data".format(task["t1"], task["t2"]))
        # if the requtired is the cumulative over the many timestesp
        iterations = iterations[(alltimes >= t1 * 1.e-3) & (alltimes < t2 * 1.e-3)]
        alltimes = alltimes[(alltimes >= t1 * 1.e-3) & (alltimes < t2 * 1.e-3)]
        assert len(iterations) == len(alltimes)
        all_x_arr = np.zeros(0, )
        all_y_arr = np.zeros(0, )
        all_z_arr = []
        #
        for it, t in zip(iterations, alltimes):
            print("loading: {:d} {:.1f}".format(it, t * 1.e3)),
            table = o_methods.get_res_corr(it, v_n_x, v_n_y)
            x_arr = np.array(table[0, 1:])  # * 6.176269145886162e+17
            y_arr = np.array(table[1:, 0]) # * -1.
            z_arr = np.array(table[1:, 1:])
            #print(z_arr.shape)
            all_x_arr = x_arr
            all_y_arr = y_arr
            all_z_arr.append(z_arr)

        #print(np.array(all_z_arr).shape)
        all_z_arr = np.sum(np.array(all_z_arr), axis=0)
        delta_t = t2 - t1
        all_z_arr = np.array(all_z_arr) / delta_t
        # create a list
        all_x_arr = [all_x_arr]
        all_y_arr = [all_y_arr]
        all_z_arr = [all_z_arr]
        # git it dum value
        iterations = [0]
        alltimes = ["{:0.f}_{:0.f}".format(t1, t2)]

    ''' --- plotting --- '''
    print("plotting...")
    for it, t, x_arr, y_arr, z_arr in zip(iterations, alltimes, all_x_arr, all_y_arr, all_z_arr):
        print("{} {}".format(it, t))
        if v_n_x == "theta": x_arr = 90 - (x_arr * 180 / np.pi)
        if v_n_y == "theta": y_arr = 90 - (y_arr * 180 / np.pi)
        if v_n_y == "hu_0": y_arr = y_arr * -1.
        if v_n_x == "hu_0": x_arr = x_arr * -1.
        if v_n_x == "rho": x_arr = x_arr * constant_rho
        if v_n_y == "rho": y_arr = y_arr * constant_rho

        #
        print("x: {} -> {}".format(x_arr.min(), x_arr.max()))
        print("y: {} -> {}".format(y_arr.min(), y_arr.max()))
        #

        print("mass", np.sum(all_z_arr))
        if task["normalize"]:
            z_arr = z_arr / np.sum(z_arr)
            z_arr= np.maximum(z_arr, 1e-10)
        #
        print(x_arr.shape)
        print(y_arr.shape)
        print(z_arr.shape)

        # -------------------------------------- PLOTTING
        fig = plt.figure(figsize=plotdic["figsize"])
        ax = fig.add_subplot(111)
        #
        im = None
        norm = LogNorm(plotdic["vmin"], plotdic["vmax"])
        im = ax.pcolormesh(x_arr, y_arr, z_arr, norm=norm, cmap=plotdic["cmap"])
        if "set_under" in plotdic.keys(): im.cmap.set_over(plotdic['set_under'])
        if "set_over" in plotdic.keys(): im.cmap.set_over(plotdic['set_over'])

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
        if "text" in plotdic.keys() and len(plotdic["text"])>0:
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
        if not "title" in plotdic.keys() or plotdic["title"] == None:
            ax.set_title(r'$t-t_{merg}:$' + r'${:.1f}$'.format(t))
        else:
            ax.set_title(plotdic["title"], fontsize=plotdic["fontsize"])
        #
        clb = fig.colorbar(im, ax=ax)
        plt.tight_layout()
        clb.ax.set_title(plotdic["clabel"], fontsize=plotdic["fontsize"])
        clb.ax.tick_params(labelsize=plotdic["fontsize"])
        clb.ax.minorticks_off()
        #

        print("plotted: \n")
        if plotdic["figname"] == "it":
            figpath = plotdic["outdir"] + str(it) + ".png"
        elif plotdic["figname"] == "time":
            figpath = plotdic["outdir"] + str(t) + ".png"
        else:
            figpath = plotdic["outdir"] + plotdic["figname"]

        print("plotted: \n")
        print(figpath)
        plt.savefig(figpath, dpi=128)
        if plotdic["savepdf"]: plt.savefig(figpath.replace(".png", ".pdf"))
        plt.close()

def plot_slice_2halfs__with_morror_function(task, plotdic):

    sim = task["sim"]  # "BLh_M13641364_M0_LK_SR"
    v_n_x = task["v_n_x"]  # "Q_eff_nua"  # optd_0_nua optd_1_nua
    v_n_y = task["v_n_y"]  # "dens_unb_bern"
    v_n_left = task["v_n_left"]
    v_n_right = task["v_n_right"]
    plane = task["plane"]  # "xz"
    rl = task["rl"]
    plot_dir = plotdic["plot_dir"]

    if not os.path.isdir(plot_dir):
        print("creating: {}".format(plot_dir))
        os.mkdir(plot_dir)
    #

    d3class = MAINMETHODS_STORE_XYXZ(sim)
    d1class = ADD_METHODS_ALL_PAR(sim)
    #
    # tmerg = d1class.get_par("tmerg")
    # time_ = d3class.get_time_for_it(it, "profiles", "prof")

    if "times" in task.keys() and not "iterations" in task.keys() and len(task["times"])>0:
        times = task["times"]
        if isinstance(times, str):
            val = float(str(times[1:]))
            iterations = np.array(d3class.list_iterations, dtype=int)
            alltimes = (d3class.times - d1class.get_par("tmerg")) * 1e3
            iterations = iterations[alltimes >= val]
            alltimes = alltimes[alltimes >= val]
        else:
            raise NameError("no method set for times:{}".format(times))
    elif "iterations" in task.keys() and not "times" in task.keys() and len(task["iterations"])>0:
        iterations = np.array(task["iterations"], dtype=int)
    else:
        raise NameError("neither 'times' nor 'iterations' are set in the plotdic. ")


    #
    # print("iterations")
    # print(iterations)

    for i_it, it in enumerate(iterations):
        time_ = d3class.get_time_for_it(it, "profiles", "prof")
        print("it:{} t:{} [ms]".format(it, time_*1e3))
        if True:
            #
            tmerg = d1class.get_par("tmerg")
            #
            data_left_arr = d3class.get_comp_data(it, rl, plane, v_n_left)
            data_right_arr = d3class.get_comp_data(it, rl, plane, v_n_right)
            print("\tv_n_left: {} shape: {}".format(v_n_left, data_left_arr.shape))
            print("\tv_n_right: {} shape: {}".format(v_n_right, data_right_arr.shape))
            if "v_n_cont" in task.keys():
                cont_data_arr = d3class.get_comp_data(it, rl, plane, task["v_n_cont"])
                if task["v_n_cont"] == "rho": cont_data_arr = cont_data_arr * constant_rho
                #print(cont_data_arr)
            else: cont_data_arr = np.zeros(0,)
            #
            if v_n_left == "hu_0": data_left_arr = -1 * data_left_arr
            if v_n_right == "hu_0": data_right_arr = -1 * data_right_arr
            #
            x_arr = d3class.get_data(it, rl, plane, v_n_x) * constant_length
            z_arr = d3class.get_data(it, rl, plane, v_n_y) * constant_length
            print("initial: {}".format(x_arr.shape))
            #
            # --- mirror z and copy x to have both x[-left, +right] and z[-under, +over]
            if "mirror_z" in plotdic.keys() and plotdic["mirror_z"]:
                x_arr = np.concatenate((np.flip(x_arr,axis=1), x_arr), axis=1)
                z_arr = np.concatenate((-1.* np.flip(z_arr,axis=1), z_arr), axis=1)
                data_left_arr = np.concatenate((np.flip(data_left_arr,axis=1), data_left_arr), axis=1)
                data_right_arr = np.concatenate((np.flip(data_right_arr,axis=1), data_right_arr), axis=1)
                if len(cont_data_arr)>1:
                    cont_data_arr = np.concatenate((np.flip(cont_data_arr,axis=1), cont_data_arr), axis=1)

            #
            # print("--------x----------")
            # print(x_arr[0, :])
            # print(x_arr[-1, :])
            # print(x_arr[:, 0]) # -500 500
            # print(x_arr[:, -1])  # -500 500
            # print("--------z----------")
            # print(z_arr[0,:]) # 2 - 500
            # print(z_arr[:,0])
            # print("--------data----------")
            # print(data_left_arr)
            # print(data_right_arr)


            #data_left_arr = np.maximum(data_left_arr, 1e-15)
            data_left_arr = np.ma.masked_array(data_left_arr, x_arr > 0)
            #data_right_arr = np.maximum(data_right_arr, 1e-15)
            data_right_arr = np.ma.masked_array(data_right_arr, x_arr < 0)

                # x_arr = np.concatenate((x_arr, x_arr), axis=1)
                # z_arr = np.concatenate((-1 * z_arr, z_arr),axis=1)
                # data_left_arr = np.concatenate((data_left_arr, data_left_arr),axis=1)
                # data_right_arr = np.concatenate((data_right_arr, data_right_arr), axis=1)

            print(x_arr.shape)
            print(z_arr.shape)
            print(data_left_arr.shape)

            # -------------------------------------- PLOTTING
            fig = plt.figure(figsize=plotdic["figsize"])
            #ax = fig.add_subplot(111)
            ax = fig.add_subplot(111)

            # --- left
            if plotdic["norm_left"] == "linear": norm = Normalize(plotdic["vmin_left"], plotdic["vmax_left"])
            else: norm = LogNorm(plotdic["vmin_left"], plotdic["vmax_left"])
            im_left = ax.pcolormesh(x_arr, z_arr, data_left_arr, norm=norm, cmap=plotdic["cmap_left"])  # , vmin=dic["vmin"], vmax=dic["vmax"])
            im_left.set_rasterized(True)
            if "set_under_left" in plotdic.keys(): im_left.cmap.set_over(plotdic['set_under_left'])
            if "set_over_left" in plotdic.keys(): im_left.cmap.set_over(plotdic['set_over_left'])
            # --- right
            if plotdic["norm_right"] == "linear": norm = Normalize(plotdic["vmin_right"], plotdic["vmax_right"])
            else:  norm = LogNorm(plotdic["vmin_right"], plotdic["vmax_right"])
            im_right = ax.pcolormesh(x_arr, z_arr, data_right_arr, norm=norm, cmap=plotdic["cmap_right"])  # , vmin=dic["vmin"], vmax=dic["vmax"])
            im_right.set_rasterized(True)
            if "set_under_right" in plotdic.keys(): im_right.cmap.set_over(plotdic['set_under_right'])
            if "set_over_right" in plotdic.keys(): im_right.cmap.set_over(plotdic['set_over_right'])
            # -- countour
            if "v_n_cont" in task.keys() and "cont_plot" in plotdic.keys():
                ax.contour(x_arr, z_arr, cont_data_arr, **plotdic["cont_plot"])

            #
            ax.set_yscale(plotdic["yscale"])
            ax.set_xscale(plotdic["xscale"])
            #
            ax.set_xlabel(plotdic["xlabel"], fontsize=plotdic["fontsize"])
            ax.set_ylabel(plotdic["ylabel"], fontsize=plotdic["fontsize"])
            #
            if plotdic["xmin"] == "auto" or plotdic["xmax"] =="auto" or plotdic["ymin"] == "auto" or plotdic["ymax"]=="auto":
                xmin, xmax, _, _, ymin, ymax = UTILS.get_xmin_xmax_ymin_ymax_zmin_zmax(rl)
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
            else:
                ax.set_xlim(plotdic["xmin"], plotdic["xmax"])
                ax.set_ylim(plotdic["ymin"], plotdic["ymax"])
            #
            if "text" in plotdic.keys():
                plotdic["text"]["transform"] = ax.transAxes
                ax.text(**plotdic["text"])
            #
            if not "title" in plotdic.keys() or plotdic["title"] == None:
                ax.set_title(r'$t-t_{merg}:$' + r'${:.1f}$'.format((time_ - tmerg) * 1e3))
            else:
                ax.set_title(plotdic["title"], fontsize=plotdic["fontsize"])
            #
            ax.tick_params(axis='both', which='both', labelleft=True,
                           labelright=False, tick1On=True, tick2On=True,
                           labelsize=plotdic["fontsize"],
                           direction='in',
                           bottom=True, top=True, left=True, right=True)
            ax.minorticks_on()

            #cbaxes = fig.add_axes([0.1, 0.1, 0.03, 0.8])
            #
            clb = fig.colorbar(im_right, ax=ax)
            clb.ax.set_title(plotdic["clabel_right"], fontsize=plotdic["fontsize"])
            clb.ax.tick_params(labelsize=plotdic["fontsize"])
            #
            from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
            ax2_divider = make_axes_locatable(ax)
            cax2 = ax2_divider.append_axes("left", size="4%", pad="30%")
            clb = fig.colorbar(im_left, cax = cax2)# anchor=(0.0, -0.5)) # anchor=(2.0, 0.5)
            clb.ax.set_title(plotdic["clabel_left"], fontsize=plotdic["fontsize"])
            clb.ax.tick_params(labelsize=plotdic["fontsize"])
            clb.ax.yaxis.set_ticks_position('left')
            clb.ax.minorticks_off()
            #
            print("plotted: \n")
            if plotdic["figname"] == "it":
                figpath = plot_dir + str(it) + ".png"
            elif plotdic["figname"] == "time":
                figpath = plot_dir + str(int(time_ * 1e3)) + ".png"
            else:
                figpath = plot_dir + plotdic["figname"]
            plt.tight_layout()
            print(figpath)
            plt.savefig(figpath, dpi=128)
            if plotdic["savepdf"]: plt.savefig(figpath.replace(".png", ".pdf"))
            plt.close()
        # except NameError:
        #     Printcolor.red("NameError. Probably no neutrino data")
        # except:
        #     raise ValueError("Something is wrong.")




''' ------------------------- TASKS ----------------------- '''

def task_plot_rho_max():
    task = [
        {"sim": "BLh_M13641364_M0_LK_SR", "color": "black", "lw": 1., "ls": '-', "v_n": "rho.maximum",  "alpha": 1.,
         "label": r"\texttt{" + "BLh_M13641364_M0_LK_SR".replace('_', '\_') + "}"},

        {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "navy", "lw": 1., "ls": '-', "v_n": "rho.maximum",  "alpha": 1.,
         "label": r"\texttt{" + "DD2_M13641364_M0_LK_SR_R04".replace('_', '\_') + "}"},

        {"sim": "LS220_M14691268_M0_LK_SR", "color": "red", "lw": 1., "ls": '-', "v_n": "rho.maximum",  "alpha": 1.,
         "label": r"\texttt{" + "LS220_M14691268_M0_LK_SR".replace('_', '\_') + "}"},

        {"sim": "BLh_M11461635_M0_LK_SR", "color": "gray", "lw": 1., "ls": '-', "v_n": "rho.maximum",  "alpha": 1.,
         "label": r"\texttt{" + "BLh_M11461635_M0_LK_SR".replace('_', '\_') + "}"},

        {"sim": "DD2_M15091235_M0_LK_SR", "color": "deepskyblue", "lw": 1., "ls": '-', "v_n": "rho.maximum", "alpha": 1.,
         "label": r"\texttt{" + "DD2_M15091235_M0_LK_SR".replace('_', '\_') + "}"},

        {"sim": "SFHo_M11461635_M0_LK_SR", "color": "olive", "lw": 1., "ls": '-', "v_n": "rho.maximum", "alpha": 1.,
         "label": r"\texttt{" + "SFHo_M11461635_M0_LK_SR".replace('_', '\_') + "}"},

        {"sim": "SLy4_M11461635_M0_LK_SR", "color": "magenta", "lw": 1., "ls": '-', "v_n": "rho.maximum", "alpha": 1.,
         "label": r"\texttt{" + "SLy4_M11461635_M0_LK_SR".replace('_', '\_') + "}"},

        {"sim": "LS220_M11461635_M0_LK_SR", "color": "coral", "lw": 1., "ls": '-', "v_n": "rho.maximum", "alpha": 1.,
         "label": r"\texttt{" + "LS220_M11461635_M0_LK_SR".replace('_', '\_') + "}"},

        {"sim": "DD2_M13641364_M0_SR", "color": "blue", "lw": 1., "ls": '--', "v_n": "rho.maximum", "alpha": 1.,
         "label": r"\texttt{" + "DD2_M13641364_M0_SR".replace('_', '\_') + "}"},

        {"sim": "DD2_M14971245_M0_SR", "color": "deepskyblue", "lw": 1., "ls": '--', "v_n": "rho.maximum", "alpha": 1.,
         "label": r"\texttt{" + "DD2_M14971245_M0_SR".replace('_', '\_') + "}"},
    ]

    plot_dic = {
        "figsize": (6., 2.5),
        "xmin": 0., "xmax": 110.,
        "ymin": 8e-4, "ymax": 4e-3,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "log",
        "xlabel": r"$t-t_{\rm merg}$ [ms]",
        "ylabel": r"$\rho_{\rm max}$",
        "legend": {"fancybox": False, "loc": 'center left',
                   "bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "title": r"Maximum rest mass density",
        "figname": __outplotdir__ + "rho_max.png"
    }

    plot_rho_max(task, plot_dic)

    # --- dens_norm1
    for t in task: t["v_n"] = "dens.norm1"

    plot_dic = {
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
        "figname": __outplotdir__ + "dens_norm1.png"
    }

    plot_rho_max(task, plot_dic)

def task_plot_total_disk_mass_evo():

    task = [
        {"sim": "BLh_M13641364_M0_LK_SR", "color": "black", "ls": "-", "lw": 0.8, "alpha": 1., "type":"long"},
        # {"sim": "BLh_M11841581_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"NS"},
        {"sim": "BLh_M11461635_M0_LK_SR", "color": "black", "ls": "-.", "lw": 0.8, "alpha": 1., "type":"long"},
        {"sim": "BLh_M10651772_M0_LK_SR", "color": "black", "ls": ":", "lw": 0.8, "alpha": 1., "type":"long"},
        # {"sim": "BLh_M10201856_M0_SR", "color": "black", "ls": "-", "lw": 0.7, "alpha": 1., "outcome":"PC"},
        # {"sim": "BLh_M10201856_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"PC"},
        #
        {"sim": "DD2_M13641364_M0_SR", "color": "blue", "ls": "-", "lw": 0.8, "alpha": 1., "type":"long"},
        # {"sim": "DD2_M13641364_M0_SR_R04", "color": "blue", "ls": "--", "lw": 0.7, "alpha": 1., "t1":60},
        {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "blue", "ls": "--", "lw": 0.8, "alpha": 1., "type":"long", "t1":40},
        {"sim": "DD2_M14971245_M0_SR", "color": "blue", "ls": "-.", "lw": 0.8, "alpha": 1., "type":"long"},
        {"sim": "DD2_M15091235_M0_LK_SR", "color": "blue", "ls": ":", "lw": 0.8, "alpha": 1., "type":"long"},
        #
        # {"sim": "LS220_M13641364_M0_SR", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1.},
        {"sim": "LS220_M13641364_M0_LK_SR_restart", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "type":"short"},
        {"sim": "LS220_M14691268_M0_LK_SR", "color": "red", "ls": "--", "lw": 0.8, "alpha": 1., "type":"long"},
        {"sim": "LS220_M11461635_M0_LK_SR", "color": "red", "ls": "-.", "lw": 0.8, "alpha": 1., "type":"short"},
        {"sim": "LS220_M10651772_M0_LK_SR", "color": "red", "ls": ":", "lw": 0.7, "alpha": 1., "type":"short"},
        #
        {"sim": "SFHo_M13641364_M0_SR", "color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "type":"short"},
        {"sim": "SFHo_M14521283_M0_SR", "color": "green", "ls": "--", "lw": 0.8, "alpha": 1., "type":"short"},
        {"sim": "SFHo_M11461635_M0_LK_SR", "color": "green", "ls": "-.", "lw": 0.8, "alpha": 1., "type":"long"},
        #
        {"sim": "SLy4_M13641364_M0_SR", "color": "magenta", "ls": "-", "lw": 0.8, "alpha": 1., "type":"short"},
        {"sim": "SLy4_M14521283_M0_SR", "color": "magenta", "ls": "--", "lw": 0.8, "alpha": 1., "type":"short"},
        # {"sim": "SLy4_M10201856_M0_LK_SR", "color": "orange", "ls": "-.", "lw": 0.7, "alpha": 1.},
        {"sim": "SLy4_M11461635_M0_LK_SR", "color": "magenta", "ls": ":", "lw": 0.8, "alpha": 1., "type":"long"}
    ]

    for t in task: t["label"] = r"\texttt{" + t["sim"].replace('_', '\_') + "}"
    for t in task:
        if t["sim"] == "LS220_M13641364_M0_LK_SR_restart":
            t["label"] = t["label"].replace("\_restart", "")

    # --- LONG ---
    plot_dic = {
        "figsize": (6., 2.5),
        "type": "long",
        "xmin": -10, "xmax": 1e2,
        "ymin": 0, "ymax": 0.35,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "linear",
        "xlabel": r"$t-t_{\rm merg}$ [ms]",
        "ylabel": r"$M_{\rm disk}$ $[M_{\odot}]$",
        "legend": {"fancybox": False, "loc": 'center left',
                   "bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "total_disk_mass_evo_long.png"
    }
    plot_tot_disk_mass_evo(task, plot_dic)

    # --- SHORT ---
    plot_dic["type"] = "short"
    plot_dic["xmax"] = 40
    plot_dic["title"] = "Short-lived remnants"
    plot_dic["figname"] = __outplotdir__ + "total_disk_mass_evo_short.png"
    #
    plot_tot_disk_mass_evo(task, plot_dic)

def task_plot_final_disk_hist():
    v_n = "Ye"
    task = [
        {"sim": "BLh_M13641364_M0_LK_SR", "color": "black", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t": -1},
        # {"sim": "BLh_M11841581_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"NS", "v_n": v_n, "t": -1},
        {"sim": "BLh_M11461635_M0_LK_SR", "color": "black", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t": -1},
        {"sim": "BLh_M10651772_M0_LK_SR", "color": "black", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t": -1},
        # {"sim": "BLh_M10201856_M0_SR", "color": "black", "ls": "-", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
        # {"sim": "BLh_M10201856_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
        #
        {"sim": "DD2_M13641364_M0_SR", "color": "blue", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t": -1},
        # {"sim": "DD2_M13641364_M0_SR_R04", "color": "blue", "ls": "--", "lw": 0.7, "alpha": 1., "t1":60, "v_n": v_n, "t": -1},
        {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "blue", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long","v_n": v_n, "t": 60/1.e3},
        {"sim": "DD2_M14971245_M0_SR", "color": "blue", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t": -1},
        {"sim": "DD2_M15091235_M0_LK_SR", "color": "blue", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t": -1},
        #
        # {"sim": "LS220_M13641364_M0_SR", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "t": -1},
        {"sim": "LS220_M13641364_M0_LK_SR_restart", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "type": "short", "v_n": v_n, "t": -1},
        {"sim": "LS220_M14691268_M0_LK_SR", "color": "red", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t": 80/1.e3},
        {"sim": "LS220_M11461635_M0_LK_SR", "color": "red", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t": -1},
        {"sim": "LS220_M10651772_M0_LK_SR", "color": "red", "ls": ":", "lw": 0.7, "alpha": 1., "type": "short", "v_n": v_n, "t": -1},
        #
        {"sim": "SFHo_M13641364_M0_SR", "color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t": -1},
        {"sim": "SFHo_M14521283_M0_SR", "color": "green", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t": -1},
        {"sim": "SFHo_M11461635_M0_LK_SR", "color": "green", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t": -1},
        #
        {"sim": "SLy4_M13641364_M0_SR", "color": "magenta", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t": -1},
        {"sim": "SLy4_M14521283_M0_SR", "color": "magenta", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t": -1},
        # {"sim": "SLy4_M10201856_M0_LK_SR", "color": "orange", "ls": "-.", "lw": 0.7, "alpha": 1., "v_n": v_n, "t": -1},
        {"sim": "SLy4_M11461635_M0_LK_SR", "color": "magenta", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t": -1}
    ]

    for t in task: t["label"] = r"\texttt{" + t["sim"].replace('_', '\_') + "}"
    for t in task:
        if t["sim"] == "LS220_M13641364_M0_LK_SR_restart":
            t["label"] = t["label"].replace("\_restart", "")

    # --- Ye ---
    plot_dic = {
        "figsize": (6., 2.5),
        "type": "long",
        "xmin": 0, "xmax": 0.4,
        "ymin": 1e-4, "ymax": 1e-1,
        "normalize": True,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "log",
        "xlabel": r"$Y_e$",
        "ylabel": r"$M_{\rm disk}/M$",
        "legend": {"fancybox": False, "loc": 'center left',
                   "bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "total_disk_hist_{}_long.png".format(v_n)
    }
    # plot_final_disk_histogram(task, plot_dic)

    plot_dic["type"] = "short"
    plot_dic["title"] = "Short-lived remnants"
    plot_dic["figname"] = __outplotdir__ + "total_disk_hist_{}_short.png".format(v_n)
    #
    plot_final_disk_histogram(task, plot_dic)

    # --- temperature ---
    v_n = "temp"
    for t in task: t["v_n"] = v_n
    plot_dic = {
        "figsize": (6., 2.5),
        "type": "long",
        "xmin": 1e-1, "xmax": 1e2,
        "ymin": 1e-4, "ymax": 1e-1,
        "normalize": True,
        # "mask_below": 1e-15,
        "xscale": "log", "yscale": "log",
        "xlabel": r"$T$ [GEO]",
        "ylabel": r"$M_{\rm disk}/M$",
        "legend": {"fancybox": False, "loc": 'center left',
                   "bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "total_disk_hist_{}_long.png".format(v_n)
    }
    plot_final_disk_histogram(task, plot_dic)

    plot_dic["type"] = "short"
    plot_dic["title"] = "Short-lived remnants"
    plot_dic["figname"] = __outplotdir__ + "total_disk_hist_{}_short.png".format(v_n)
    #
    plot_final_disk_histogram(task, plot_dic)

    # --- entropy ---
    v_n = "entr"
    for t in task: t["v_n"] = v_n
    plot_dic = {
        "figsize": (6., 2.5),
        "type": "long",
        "xmin": 0, "xmax": 30,
        "ymin": 1e-4, "ymax": 1e-1,
        "normalize": True,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "log",
        "xlabel": r"$s$ [GEO]",
        "ylabel": r"$M_{\rm disk}/M$",
        "legend": {"fancybox": False, "loc": 'center left',
                   "bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "total_disk_hist_{}_long.png".format(v_n)
    }
    plot_final_disk_histogram(task, plot_dic)

    plot_dic["type"] = "short"
    plot_dic["title"] = "Short-lived remnants"
    plot_dic["figname"] = __outplotdir__ + "total_disk_hist_{}_short.png".format(v_n)
    #
    plot_final_disk_histogram(task, plot_dic)

    # --- press ---
    v_n = "press"
    for t in task: t["v_n"] = v_n
    plot_dic = {
        "figsize": (6., 2.5),
        "type": "long",
        "xmin": 1e-8, "xmax": 1e-6,
        "ymin": 1e-4, "ymax": 1e-1,
        "normalize": True,
        # "mask_below": 1e-15,
        "xscale": "log", "yscale": "log",
        "xlabel": r"$P$ [GEO]",
        "ylabel": r"$M_{\rm disk}/M$",
        "legend": {"fancybox": False, "loc": 'center left',
                   "bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "total_disk_hist_{}_long.png".format(v_n)
    }
    plot_final_disk_histogram(task, plot_dic)

    plot_dic["type"] = "short"
    plot_dic["xmin"], plot_dic["xmax"] = 10, 100
    plot_dic["title"] = "Short-lived remnants"
    plot_dic["figname"] = __outplotdir__ + "total_disk_hist_{}_short.png".format(v_n)
    #
    plot_final_disk_histogram(task, plot_dic)

    # --- theta ---
    v_n = "theta"
    for t in task: t["v_n"] = v_n
    plot_dic = {
        "figsize": (6., 2.5),
        "type": "long",
        "xmin": 0, "xmax": 90,
        "ymin": 1e-4, "ymax": 1e-1,
        "normalize": True,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "log",
        "xlabel": r"$\theta$ [GEO]",
        "ylabel": r"$M_{\rm disk}/M$",
        "legend": {"fancybox": False, "loc": 'center left',
                   "bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "total_disk_hist_{}_long.png".format(v_n)
    }
    plot_final_disk_histogram(task, plot_dic)

    plot_dic["type"] = "short"
    plot_dic["title"] = "Short-lived remnants"
    plot_dic["figname"] = __outplotdir__ + "total_disk_hist_{}_short.png".format(v_n)
    #
    plot_final_disk_histogram(task, plot_dic)

    # --- r ---
    v_n = "r"
    for t in task: t["v_n"] = v_n
    plot_dic = {
        "figsize": (6., 2.5),
        "type": "long",
        "xmin": 10, "xmax": 100,
        "ymin": 1e-4, "ymax": 1e-1,
        "normalize": True,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "log",
        "xlabel": r"$R$ [GEO]",
        "ylabel": r"$M_{\rm disk}/M$",
        "legend": {"fancybox": False, "loc": 'center left',
                   "bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "total_disk_hist_{}_long.png".format(v_n)
    }
    plot_final_disk_histogram(task, plot_dic)

    plot_dic["type"] = "short"
    plot_dic["xmin"], plot_dic["xmax"] = 10, 100
    plot_dic["title"] = "Short-lived remnants"
    plot_dic["figname"] = __outplotdir__ + "total_disk_hist_{}_short.png".format(v_n)
    #
    plot_final_disk_histogram(task, plot_dic)

def task_plot_mass_ave_val_evo():

    v_n = "Ye"
    task = [
        {"sim": "BLh_M13641364_M0_LK_SR", "color": "black", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "t": -1},
        # {"sim": "BLh_M11841581_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"NS", "v_n": v_n, "t": -1},
        {"sim": "BLh_M11461635_M0_LK_SR", "color": "black", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "t": -1},
        {"sim": "BLh_M10651772_M0_LK_SR", "color": "black", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "t": -1},
        # {"sim": "BLh_M10201856_M0_SR", "color": "black", "ls": "-", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
        # {"sim": "BLh_M10201856_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
        #
        {"sim": "DD2_M13641364_M0_SR", "color": "blue", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n,
         "t": -1},
        # {"sim": "DD2_M13641364_M0_SR_R04", "color": "blue", "ls": "--", "lw": 0.7, "alpha": 1., "t1":60, "v_n": v_n, "t": -1},
        {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "blue", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "t": 60 / 1.e3},
        {"sim": "DD2_M14971245_M0_SR", "color": "blue", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n,
         "t": -1},
        {"sim": "DD2_M15091235_M0_LK_SR", "color": "blue", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "t": -1},
        #
        # {"sim": "LS220_M13641364_M0_SR", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "t": -1},
        {"sim": "LS220_M13641364_M0_LK_SR_restart", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "type": "short",
         "v_n": v_n, "t": 25},
        {"sim": "LS220_M14691268_M0_LK_SR", "color": "red", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "t": 60},
        {"sim": "LS220_M11461635_M0_LK_SR", "color": "red", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n, "t": -1},
        {"sim": "LS220_M10651772_M0_LK_SR", "color": "red", "ls": ":", "lw": 0.7, "alpha": 1., "type": "short",
         "v_n": v_n, "t": -1},
        #
        {"sim": "SFHo_M13641364_M0_SR", "color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n, "t": -1},
        {"sim": "SFHo_M14521283_M0_SR", "color": "green", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n, "t": -1},
        {"sim": "SFHo_M11461635_M0_LK_SR", "color": "green", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "t": -1},
        #
        {"sim": "SLy4_M13641364_M0_SR", "color": "magenta", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n, "t": -1},
        {"sim": "SLy4_M14521283_M0_SR", "color": "magenta", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n, "t": -1},
        # {"sim": "SLy4_M10201856_M0_LK_SR", "color": "orange", "ls": "-.", "lw": 0.7, "alpha": 1., "v_n": v_n, "t": -1},
        {"sim": "SLy4_M11461635_M0_LK_SR", "color": "magenta", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "t": -1}
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
        "xmin": 0, "xmax": 100,
        "ymin": 0, "ymax": 0.25,
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
        "figname": __outplotdir__ + "total_disk_massave_{}_long.png".format(v_n)
    }

    plot_dic = copy.deepcopy(def_plot_dic)
    plot_mass_ave_val_evo(task, plot_dic)

    plot_dic["xmin"], plot_dic["xmax"] = 0., 40
    plot_dic["ymin"], plot_dic["ymax"] = 0., 0.3
    plot_dic["type"] = "short"
    plot_dic["title"] = "Short-lived remnants"
    plot_dic["figname"] = __outplotdir__ + "total_disk_massave_{}_short.png".format(v_n)
    #
    plot_mass_ave_val_evo(task, plot_dic)

    # --- temp
    plot_dic2 = copy.deepcopy(def_plot_dic)
    v_n = "temp"
    for t in task: t["v_n"] = v_n
    plot_dic2["title"] = "Long-lived remnants"
    plot_dic2["type"] = "long"
    plot_dic2["yscale"] = "linear"
    plot_dic2["ymin"], plot_dic2["ymax"] = 3, 10
    plot_dic2["ylabel"] = r"$\langle T \rangle$ [GEO]"
    plot_dic2["figname"] = __outplotdir__ + "total_disk_massave_{}_long.png".format(v_n)

    plot_mass_ave_val_evo(task, plot_dic2)

    plot_dic2["xmin"], plot_dic2["xmax"] = 0., 40
    plot_dic2["type"] = "short"
    plot_dic2["title"] = "Short-lived remnants"
    plot_dic2["figname"] = __outplotdir__ + "total_disk_massave_{}_short.png".format(v_n)
    #
    plot_mass_ave_val_evo(task, plot_dic2)

    # --- entropy
    plot_dic3 = copy.deepcopy(def_plot_dic)
    v_n = "entr"
    for t in task: t["v_n"] = v_n
    plot_dic3["title"] = "Long-lived remnants"
    plot_dic3["type"] = "long"
    plot_dic3["yscale"] = "linear"
    plot_dic3["ymin"], plot_dic3["ymax"] = 2.5, 10
    plot_dic3["ylabel"] = r"$\langle s \rangle$ [GEO]"
    plot_dic3["figname"] = __outplotdir__ + "total_disk_massave_{}_long.png".format(v_n)

    plot_mass_ave_val_evo(task, plot_dic3)

    plot_dic3["xmin"], plot_dic3["xmax"] = 0., 40
    plot_dic3["ymin"], plot_dic3["ymax"] = 2.5, 15
    plot_dic3["type"] = "short"
    plot_dic3["title"] = "Short-lived remnants"
    plot_dic3["figname"] = __outplotdir__ + "total_disk_massave_{}_short.png".format(v_n)
    #
    plot_mass_ave_val_evo(task, plot_dic3)

    # --- pressure
    plot_dic3 = copy.deepcopy(def_plot_dic)
    v_n = "press"
    for t in task: t["v_n"] = v_n
    plot_dic3["title"] = "Long-lived remnants"
    plot_dic3["type"] = "long"
    plot_dic3["yscale"] = "log"
    plot_dic3["ymin"], plot_dic3["ymax"] = 1e-8, 2e-7
    plot_dic3["ylabel"] = r"$\langle P \rangle$ [GEO]"
    plot_dic3["figname"] = __outplotdir__ + "total_disk_massave_{}_long.png".format(v_n)

    plot_mass_ave_val_evo(task, plot_dic3)

    plot_dic3["xmin"], plot_dic3["xmax"] = 0., 40
    plot_dic3["ymin"], plot_dic3["ymax"] = 1e-8, 2e-7
    plot_dic3["type"] = "short"
    plot_dic3["title"] = "Short-lived remnants"
    plot_dic3["figname"] = __outplotdir__ + "total_disk_massave_{}_short.png".format(v_n)
    #
    plot_mass_ave_val_evo(task, plot_dic3)

    # --- theta
    plot_dic4 = copy.deepcopy(def_plot_dic)
    v_n = "theta"
    for t in task: t["v_n"] = v_n
    plot_dic4["title"] = "Long-lived remnants"
    plot_dic4["type"] = "long"
    plot_dic4["yscale"] = "linear"
    plot_dic4["ymin"], plot_dic4["ymax"] = 10, 30.
    plot_dic4["ylabel"] = r"$\langle \theta_{\rm RMS} \rangle$"
    plot_dic4["figname"] = __outplotdir__ + "total_disk_massave_{}_long.png".format(v_n)
    #
    plot_mass_ave_val_evo(task, plot_dic4)

    plot_dic4["xmin"], plot_dic4["xmax"] = 0., 40
    plot_dic4["ymin"], plot_dic4["ymax"] = 10., 30.
    plot_dic4["type"] = "short"
    plot_dic4["title"] = "Short-lived remnants"
    plot_dic4["figname"] = __outplotdir__ + "total_disk_massave_{}_short.png".format(v_n)
    #
    plot_mass_ave_val_evo(task, plot_dic4)

    # --- r
    plot_dic5 = copy.deepcopy(def_plot_dic)
    v_n = "r"
    for t in task: t["v_n"] = v_n
    plot_dic5["title"] = "Long-lived remnants"
    plot_dic5["type"] = "long"
    plot_dic5["yscale"] = "linear"
    plot_dic5["xmin"], plot_dic5["xmax"] = 0., 100
    plot_dic5["ymin"], plot_dic5["ymax"] = 0, 60.
    plot_dic5["ylabel"] = r"$\langle R \rangle$ [GEO]"
    plot_dic5["figname"] = __outplotdir__ + "total_disk_massave_{}_long.png".format(v_n)

    plot_mass_ave_val_evo(task, plot_dic5)

    plot_dic5["xmin"], plot_dic5["xmax"] = 0., 40
    plot_dic5["ymin"], plot_dic5["ymax"] = 0., 100.
    plot_dic5["type"] = "short"
    plot_dic5["title"] = "Short-lived remnants"
    plot_dic5["figname"] = __outplotdir__ + "total_disk_massave_{}_short.png".format(v_n)
    #
    plot_mass_ave_val_evo(task, plot_dic5)

def task_plot_final_disk_corr():
    task = {
        "sim": "BLh_M13641364_M0_LK_SR",
        "v_n_x": "ang_mom_flux",
        "v_n_y": "hu_0",
        "t1": 60,
        "t2": 80,
        "normalize": True,
    }

    def_plotdic = {"vmin": 1e-7, "vmax": 1e-3,
               "xmin": 1e-10, "xmax": 1e-7,
               "ymin": 0.98, "ymax": 1.02,
               "cmap": "jet",
               "set_under": "black",
               "set_over": "red",
               "yscale": "linear",
               "xscale": "log",
               "xlabel": r"$J_{\rm flux}$ [GEO]",
               "ylabel": r"$-hu_0$",
               "title": r"\texttt{"+task["sim"].replace("_","\_")+"}" + "[{}ms]".format(task["t1"]),
               "clabel": r"$M_{\rm disk}/M$",
               # "text": {"x": 0.3, "y": 0.95, "s": r"$\theta>60\deg$", "ha": "center", "va": "top", "fontsize": 11,
               #          "color": "white",
               #          "transform": None},
               "outdir":  __outplotdir__ + "final_disk_corr_{}_{}_{}/".format(task["v_n_x"], task["v_n_y"], task["sim"]),
               "figname": "times_{}_{}.png".format(task["t1"], task["t2"])
               }

    #

    plotdic1 = copy.deepcopy(def_plotdic)
    task1 = copy.deepcopy(task)
    plot_final_disk_corr(task, plotdic1)

    task1["sim"] = "LS220_M13641364_M0_LK_SR_restart"
    task1["t1"], task1["t2"] = 30, 40
    plotdic1["title"] = r"\texttt{"+task1["sim"].replace("_","\_")+"}" + " [{}ms]".format(task1["t1"])
    plotdic1["title"] = plotdic1["title"].replace("\_restart", "")
    plotdic1["figname"] = "times_{}_{}.png".format(task1["t1"], task1["t2"])
    plotdic1["outdir"] = __outplotdir__ + "final_disk_corr_{}_{}_{}/".format(task1["v_n_x"], task1["v_n_y"], task1["sim"])

    plot_final_disk_corr(task, plotdic1)

    ''' ----------------------------------------- '''
    task2 = copy.deepcopy(task)
    task2["v_n_y"] = "dens_unb_bern"

    plotdic2 ={"vmin": 1e-9, "vmax": 1e-4,
               "xmin": 1e-10, "xmax": 1e-7,
               "ymin": 1e-10, "ymax": 1e-7,
               "cmap": "jet",
               "set_under": "black",
               "set_over": "red",
               "yscale": "log",
               "xscale": "log",
               "xlabel": r"$J_{\rm flux}$ [GEO]",
               "ylabel": r"$D(-hu_0>1)$ [GEO]",
               "title": r"\texttt{"+task["sim"].replace("_","\_")+"}" + "[{}ms]".format(task["t1"]),
               "clabel": r"$M_{\rm disk}/M$",
               # "text": {"x": 0.3, "y": 0.95, "s": r"$\theta>60\deg$", "ha": "center", "va": "top", "fontsize": 11,
               #          "color": "white",
               #          "transform": None},
               "outdir":  __outplotdir__ + "final_disk_corr_{}_{}_{}/".format(task2["v_n_x"], task2["v_n_y"], task2["sim"]),
               "figname": "times_{}_{}.png".format(task2["t1"], task2["t2"])

               }

    # print(task["sim"]); exit(1)
    plotdic2["text"] = {"x": 0.8, "y": 0.3, "s": r"$60-80$ [ms]", "ha": "center", "va": "top", "fontsize": 11,
                        "color": "white",
                        "transform": None},
    plot_final_disk_corr(task2, plotdic2)

    # task2["sim"] = "LS220_M13641364_M0_LK_SR_restart"
    # task2["t1"], task2["t2"] = 30, 40
    # plotdic2["title"] = r"\texttt{"+task2["sim"].replace("_","\_")+"}" + " [{}ms]".format(task2["t1"])
    # plotdic2["title"] = plotdic2["title"].replace("\_restart", "")
    # plotdic2["figname"] = "times_{}_{}.png".format(task2["t1"], task2["t2"])
    # plotdic2["outdir"] = __outplotdir__ + "final_disk_corr_{}_{}_{}/".format(task2["v_n_x"], task2["v_n_y"], task2["sim"])
    # plotdic2["text"] = {"x": 0.8, "y": 0.3, "s": r"$50-70$ [ms]", "ha": "center", "va": "top", "fontsize": 11,
    #                     "color": "white",
    #                     "transform": None},
    # plot_final_disk_corr(task2, plotdic2)

    task2["sim"] = "LS220_M13641364_M0_LK_SR_restart"
    task2["t1"], task2["t2"] = 30, 40
    plotdic2["title"] = r"\texttt{"+task2["sim"].replace("_","\_")+"}" + " [{}ms]".format(task2["t1"])
    plotdic2["title"] = plotdic2["title"].replace("\_restart", "")
    plotdic2["figname"] = "times_{}_{}.png".format(task2["t1"], task2["t2"])
    plotdic2["outdir"] = __outplotdir__ + "final_disk_corr_{}_{}_{}/".format(task2["v_n_x"], task2["v_n_y"], task2["sim"])
    plotdic2["text"] = {"x": 0.8, "y": 0.3, "s": r"30-40 [ms]", "ha": "center", "va": "top", "fontsize": 11,
                        "color": "white",
                        "transform": None},
    # plot_final_disk_corr(task2, plotdic2)

    task2["sim"] = "BLh_M11461635_M0_LK_SR"
    task2["t1"], task2["t2"] = 50, 70
    plotdic2["title"] = r"\texttt{"+task2["sim"].replace("_","\_")+"}" + " [{}ms]".format(task2["t1"])
    plotdic2["title"] = plotdic2["title"].replace("\_restart", "")
    plotdic2["figname"] = "times_{}_{}.png".format(task2["t1"], task2["t2"])
    plotdic2["outdir"] = __outplotdir__ + "final_disk_corr_{}_{}_{}/".format(task2["v_n_x"], task2["v_n_y"], task2["sim"])
    plotdic2["text"] = {"x": 0.8, "y": 0.3, "s": r"$50-70$ [ms]", "ha": "center", "va": "top", "fontsize": 11,
                        "color": "white",
                        "transform": None},

    plot_final_disk_corr(task2, plotdic2)

def task_plot_disk_timecorr():

    task = {
        "sim": "BLh_M13641364_M0_LK_SR",
        "v_n": "Ye",
        "mask": "disk",
        "normalize": True,
    }

    def_plotdic = {"vmin": 1e-6, "vmax": 1e-2,
                   "xmin": 0, "xmax": 90,
                   "ymin": 0, "ymax": 0.5,
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

    plot_disk_timecorr(task, def_plotdic)

    #
    plotdic1 = copy.deepcopy(def_plotdic)
    task1 = copy.deepcopy(task)
    task1["sim"] = "BLh_M11461635_M0_LK_SR"
    plotdic1["title"] = r"\texttt{" + task1["sim"].replace("_", "\_") + "}" # + "[{}ms]".format(task["t1"]),
    plotdic1["figname"] = "final_disk_timecorr_{}_{}.png".format(task1["v_n"], task1["sim"])
    plot_disk_timecorr(task1, plotdic1)

    #

    plotdic2 = copy.deepcopy(def_plotdic)
    task2 = copy.deepcopy(task)
    task2["sim"] = "DD2_M13641364_M0_SR"
    plotdic2["title"] = r"\texttt{" + task2["sim"].replace("_", "\_") + "}"  # + "[{}ms]".format(task["t1"]),
    plotdic2["figname"] = "final_disk_timecorr_{}_{}.png".format(task2["v_n"], task2["sim"])
    plot_disk_timecorr(task2, plotdic2)

def task_plot_final_disk_structure():

    task = {
        "sim": "BLh_M13641364_M0_LK_SR",
        "v_n_x": "x",
        "v_n_y": "y",
        "plane": "xy",
        "v_n_left": "entr",  # "Q_eff_nua_over_density",
        "v_n_right": "Ye",
        "v_n_cont": "rho",
        "rl": 3,
        # "times":">20"
        # "iterations": "all" # [1949696]
        "iterations": [2187264]  # 2121728
    }

    # xy blh
    plot_dic = {
        "figsize": (6., 4.),
        "xlabel": "$X$ [km]",
        "ylabel": "$Y$ [km]",
        "xmin": -120, "xmax": 120,  # "auto"
        "ymin": -120, "ymax": 120,  # "auto"
        "xscale": "linear", "yscale": "linear",

        "cont_plot": {"levels":[1e8, 1e9, 1e10, 1e11, 1e12, 1e13], "color": "black", "lw":0.4},

        "vmin_left": 0, "vmax_left": 50,
        "clabel_left": r"$s$ [$k_b/\rm{baryon}$]",  ##r"$Q_{eff}(\nu_a) / D$",
        "cmap_left": "jet_r",
        "norm_left": "linear",
        # "set_under_left": "black", "set_over_left": "blue",

        "vmin_right": 0.05, "vmax_right": 0.45,
        "clabel_right": "$Y_e$",
        "cmap_right": "RdBu",
        "norm_right": "linear",
        # "set_under_right": "black", "set_over_right": "blue",

        "plot_dir": __outplotdir__, #+ task["sim"] + "/",
        "title": r"BLh q=1.00 (SR), $t-t_{\rm merg}=88$ ms",
        # "figname": "it",
        "figname": "slice_xy_entr_ye_blh_q1_rl{}.png".format(task["rl"]),
        "savepdf": True,
        "mirror_z": False,
        "fontsize": 14,
    }
    plot_slice_2halfs__with_morror_function(task, plot_dic)

    # xz blh
    task["plane"], task["v_n_y"] = "xz", "z"
    plot_dic["ylabel"], plot_dic["mirror_z"], plot_dic["figname"] = \
        "$Z$ [km]", True, "slice_xz_entr_ye_blh_q1_rl{}.png".format(task["rl"])
    plot_slice_2halfs__with_morror_function(task, plot_dic)

    # xy dd2
    task["plane"], task["v_n_y"] = "xy", "y"
    task["sim"] = "DD2_M13641364_M0_LK_SR_R04"
    task["iterations"] = [2532732]
    plot_dic["mirror_z"] = False
    plot_dic["title"] = r"DD2 q=1.00 (SR), $t-t_{\rm merg}=109$ ms"
    plot_dic["ylabel"], plot_dic["figname"] = "$Y$ [km]", "slice_xy_entr_ye_dd2_q1_rl{}.png".format(task["rl"])

    #plot_slice_2halfs__with_morror_function(task, plot_dic)

    # xz dd2
    task["plane"], task["v_n_y"] = "xz", "z"
    plot_dic["mirror_z"] = True
    plot_dic["ylabel"], plot_dic["figname"] = "$Z$ [km]", "slice_xz_entr_ye_dd2_q1_rl{}.png".format(task["rl"])
    plot_slice_2halfs__with_morror_function(task, plot_dic)

    ''' ------ loops ------ '''

    task = {
        "sim": "BLh_M13641364_M0_LK_SR",
        "v_n_x": "x",
        "v_n_y": "y",
        "plane": "xy",
        "v_n_left": "entr",  # "Q_eff_nua_over_density",
        "v_n_right": "Ye",
        "v_n_cont": "rho",
        "rl": 3,
        # "times":">20"
        # "iterations": "all" # [1949696]
        "times": ">20"  # 2121728
    }

    plot_dic = {
        "figsize": (6., 4.),
        "xlabel": "$X$ [km]",
        "ylabel": "$Y$ [km]",
        "xmin": -120, "xmax": 120,  # "auto"
        "ymin": -120, "ymax": 120,  # "auto"
        "xscale": "linear", "yscale": "linear",

        "cont_plot": {"levels": [1e8, 1e9, 1e10, 1e11, 1e12, 1e13], "color": "black", "lw": 0.4},

        "vmin_left": 0, "vmax_left": 30,
        "clabel_left": r"$s$ [$k_b$]",  ##r"$Q_{eff}(\nu_a) / D$",
        "cmap_left": "jet_r",
        "norm_left": "linear",
        # "set_under_left": "black", "set_over_left": "blue",

        "vmin_right": 0.05, "vmax_right": 0.45,
        "clabel_right": "$Y_e$",
        "cmap_right": "RdBu",
        "norm_right": "linear",
        # "set_under_right": "black", "set_over_right": "blue",

        "plot_dir": __outplotdir__ + task["sim"] + '/movie/', # + task["sim"] + "/",
        "title": None,#r"BLh* q=1.00 (SR), $t-t_{\rm merg}=88$ [ms]",
        # "figname": "it",
        "figname": "time",
        "savepdf": False,
        "mirror_z": False,
        "fontsize": 14,
    }
    ''' --- loops | movies --- '''
    #plot_slice_2halfs__with_morror_function(task, plot_dic)
    # os.system("convert -delay 20 -loop 0 {}*.png {}.gif".format(
    #     __outplotdir__ + task["sim"] + '/movie/',
    #     __outplotdir__ + task["sim"] + "/" + \
    #     task["sim"] + "_" + task["v_n_left"]+'_'+task["v_n_right"] + '_' + "rl" + str(task["rl"])
    # ))
    # print("made: \n{}".format(
    #     __outplotdir__ + task["sim"] + "/" + \
    #     task["sim"] + "_" + task["v_n_left"] + '_' + task["v_n_right"] + '_' + "rl" + str(task["rl"]) + ".gif"
    # ))

    # DD2
    # task["sim"] = "DD2_M13641364_M0_LK_SR_R04"
    # plot_dic["plot_dir"] = __outplotdir__ + task["sim"] + '/movie/'
    # plot_slice_2halfs__with_morror_function(task, plot_dic)
    # os.system("convert -delay 20 -loop 0 {}*.png {}.gif".format(
    #     __outplotdir__ + task["sim"] + '/movie/',
    #     __outplotdir__ + task["sim"] + "/" + \
    #     task["sim"] + "_" + task["v_n_left"]+'_'+task["v_n_right"] + '_' + "rl" + str(task["rl"])
    # ))
    # print("made: \n{}".format(
    #     __outplotdir__ + task["sim"] + "/" + \
    #     task["sim"] + "_" + task["v_n_left"] + '_' + task["v_n_right"] + '_' + "rl" + str(task["rl"]) + ".gif"
    # ))

''' -------------------------- iteration 3 | tasks ----------------------- '''

def task_plot_final_disk_hist_3():
    v_n = "Ye"
    task = [
        {"sim": "BLh_M13641364_M0_LK_SR", "label":r"BLh* q=1.00 (SR)", "color": "black", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t": -1},
        # {"sim": "BLh_M11841581_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"NS", "v_n": v_n, "t": -1},
        {"sim": "BLh_M11461635_M0_LK_SR", "label":r"BLh* q=1.43 (SR)", "color": "gray", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t": -1},
        #{"sim": "BLh_M10651772_M0_LK_SR", "color": "black", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t": -1},
        # {"sim": "BLh_M10201856_M0_SR", "color": "black", "ls": "-", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
        # {"sim": "BLh_M10201856_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
        #
        #{"sim": "DD2_M13641364_M0_SR", "color": "blue", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t": -1},
        # {"sim": "DD2_M13641364_M0_SR_R04", "color": "blue", "ls": "--", "lw": 0.7, "alpha": 1., "t1":60, "v_n": v_n, "t": -1},
        {"sim": "DD2_M13641364_M0_LK_SR_R04", "label":r"DD2* q=1.00 (SR)", "color": "blue", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long","v_n": v_n, "t": 60/1.e3},
        #{"sim": "DD2_M14971245_M0_SR", "color": "blue", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t": -1},
        {"sim": "DD2_M15091235_M0_LK_SR", "label":r"DD2* q=1.22 (SR)", "color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t": -1},
        #
        # {"sim": "LS220_M13641364_M0_SR", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "t": -1},
        #{"sim": "LS220_M13641364_M0_LK_SR_restart", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "type": "short", "v_n": v_n, "t": -1},
        #{"sim": "LS220_M14691268_M0_LK_SR", "color": "red", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t": 80/1.e3},
        #{"sim": "LS220_M11461635_M0_LK_SR", "color": "red", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t": -1},
        #{"sim": "LS220_M10651772_M0_LK_SR", "color": "red", "ls": ":", "lw": 0.7, "alpha": 1., "type": "short", "v_n": v_n, "t": -1},
        #
        #{"sim": "SFHo_M13641364_M0_SR", "color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t": -1},
        #{"sim": "SFHo_M14521283_M0_SR", "color": "green", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t": -1},
        {"sim": "SFHo_M11461635_M0_LK_SR", "label":r"SFHo* q=1.42 (SR)", "color": "orange", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t": -1},
        #
        #{"sim": "SLy4_M13641364_M0_SR", "color": "magenta", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t": -1},
        #{"sim": "SLy4_M14521283_M0_SR", "color": "magenta", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t": -1},
        # {"sim": "SLy4_M10201856_M0_LK_SR", "color": "orange", "ls": "-.", "lw": 0.7, "alpha": 1., "v_n": v_n, "t": -1},
        {"sim": "SLy4_M11461635_M0_LK_SR", "label":r"SLy4* q=1.42 (SR)", "color": "magenta", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t": -1}
    ]

    for t in task:
        t["color"] = md.sim_dic_color[t["sim"]]
        t["ls"] = md.sim_dic_ls[t["sim"]]
        t["lw"] = md.sim_dic_lw[t["sim"]]

    # for t in task: t["label"] = r"\texttt{" + t["sim"].replace('_', '\_') + "}"
    # for t in task:
    #     if t["sim"] == "LS220_M13641364_M0_LK_SR_restart":
    #         t["label"] = t["label"].replace("\_restart", "")

    # --- Ye ---
    plot_dic = {
        "figsize": (6., 5.5),
        "type": "all",
        "xmin": 0, "xmax": 0.4,
        "ymin": 1e-4, "ymax": 4e-1,
        "normalize": True,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "log",
        "xlabel": r"$Y_e$",
        "ylabel": r"$M_{\rm disk}/M$",
        "legend": {},
             # {"fancybox": False, "loc": 'lower right',
             #       "bbox_to_anchor": (1.0, 1.0),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
             #       "shadow": "False", "ncol": 2, "fontsize": 12,
             #       "framealpha": 0., "borderaxespad": 0., "frameon": False},
        #"title": r"Long-lived remnants",
        "figname": __outplotdir__ + "final_structure/" + "disk_hist_{}_long.png".format(v_n),
        "savepdf": True,
        "fontsize":14
    }
    plot_final_disk_histogram(task, plot_dic)

    plot_dic["type"] = "short"
    plot_dic["title"] = "Short-lived remnants"
    plot_dic["figname"] = __outplotdir__ + "disk_hist_{}_short.png".format(v_n)
    #
    # plot_final_disk_histogram(task, plot_dic)

    # --- temperature ---
    v_n = "temp"
    for t in task: t["v_n"] = v_n
    plot_dic = {
        "figsize": (6., 5.5),
        "type": "all",
        "xmin": 5e-2, "xmax": 3e1,
        "ymin": 1e-4, "ymax": 4e-1,
        "normalize": True,
        # "mask_below": 1e-15,
        "xscale": "log", "yscale": "log",
        "xlabel": r"$T$ [MeV]",
        "ylabel": r"$M_{\rm disk}/M$",
        "legend": {"fancybox": False, "loc": 'upper left',
                   #"bbox_to_anchor": (1.0, 1.0),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        #"title": r"Long-lived remnants",
        "figname": __outplotdir__ + "final_structure/" + "disk_hist_{}_long.png".format(v_n),
        "savepdf": True,
        "fontsize": 14
    }
    plot_final_disk_histogram(task, plot_dic)

    plot_dic["type"] = "short"
    plot_dic["title"] = "Short-lived remnants"
    plot_dic["figname"] = __outplotdir__ + "final_structure/" + "disk_hist_{}_short.png".format(v_n)

    #plot_final_disk_histogram(task, plot_dic)

    # --- entropy ---
    v_n = "entr"
    for t in task: t["v_n"] = v_n
    plot_dic = {
        "figsize": (6., 5.5),
        "type": "long",
        "xmin": 0, "xmax": 30,
        "ymin": 1e-4, "ymax": 4e-1,
        "normalize": True,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "log",
        "xlabel": r"$s$ [$k_B$]",
        "ylabel": r"$M_{\rm disk}/M$",
        "legend": {},
            # {"fancybox": False, "loc": 'lower right',
            #        "bbox_to_anchor": (1.0, 1.0),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
            #        "shadow": "False", "ncol": 2, "fontsize": 12,
            #        "framealpha": 0., "borderaxespad": 0., "frameon": False},
        #"title": r"Long-lived remnants",
        "figname": __outplotdir__ + "final_structure/" + "disk_hist_{}_long.png".format(v_n),
        "savepdf": True,
        "fontsize": 14
    }
    plot_final_disk_histogram(task, plot_dic)

    plot_dic["type"] = "short"
    plot_dic["title"] = "Short-lived remnants"
    plot_dic["figname"] = __outplotdir__ + "final_structure/" + "total_disk_hist_{}_short.png".format(v_n)

    #plot_final_disk_histogram(task, plot_dic)

    # --- press ---
    v_n = "press"
    for t in task: t["v_n"] = v_n
    plot_dic = {
        "figsize": (6., 5.5),
        "type": "long",
        "xmin": 1e-8, "xmax": 1e-6,
        "ymin": 1e-4, "ymax": 4e-1,
        "normalize": True,
        # "mask_below": 1e-15,
        "xscale": "log", "yscale": "log",
        "xlabel": r"$P$ [GEO]",
        "ylabel": r"$M_{\rm disk}/M$",
        "legend": {"fancybox": False, "loc": 'lower right',
                   "bbox_to_anchor": (1.0, 1.0),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        #"title": r"Long-lived remnants",
        "figname": __outplotdir__ + "final_structure/" + "total_disk_hist_{}_long.png".format(v_n),
        "savepdf": True,
        "fontsize": 14
    }
    #plot_final_disk_histogram(task, plot_dic)

    plot_dic["type"] = "short"
    plot_dic["xmin"], plot_dic["xmax"] = 10, 100
    plot_dic["title"] = "Short-lived remnants"
    plot_dic["figname"] =__outplotdir__ + "final_structure/" + "total_disk_hist_{}_short.png".format(v_n)

    #plot_final_disk_histogram(task, plot_dic)

    # --- theta ---
    v_n = "theta"
    for t in task: t["v_n"] = v_n
    plot_dic = {
        "figsize": (6., 5.5),
        "type": "long",
        "xmin": 0, "xmax": 90,
        "ymin": 1e-4, "ymax": 4e-2,
        "normalize": True,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "log",
        "xlabel": r"Angle from the binary plane",
        "ylabel": r"$M_{\rm disk}/M$",
        "legend": {"fancybox": False, "loc": 'lower right',
                   "bbox_to_anchor": (1.0, 1.0),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        #"title": r"Long-lived remnants",
        "figname": __outplotdir__ + "final_structure/" + "total_disk_hist_{}_long.png".format(v_n),
        "savepdf": True,
        "fontsize": 14
    }
    #plot_final_disk_histogram(task, plot_dic)

    plot_dic["type"] = "short"
    plot_dic["title"] = "Short-lived remnants"
    plot_dic["figname"] =__outplotdir__ + "final_structure/" + "total_disk_hist_{}_short.png".format(v_n)
    #
    #plot_final_disk_histogram(task, plot_dic)

    # --- r ---
    v_n = "r"
    for t in task: t["v_n"] = v_n
    plot_dic = {
        "figsize": (6., 5.5),
        "type": "long",
        "xmin": 10, "xmax": 200,
        "ymin": 1e-4, "ymax": 4e-2,
        "normalize": True,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "log",
        "xlabel": r"$R$ [km]",
        "ylabel": r"$M_{\rm disk}/M$",
        "legend": {"fancybox": False, "loc": 'lower right',
                   "bbox_to_anchor": (1.0, 1.0),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        # "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "final_structure/" + "total_disk_hist_{}_long.png".format(v_n),
        "savepdf": True,
        "fontsize": 14
    }
    #plot_final_disk_histogram(task, plot_dic)

    plot_dic["type"] = "short"
    plot_dic["xmin"], plot_dic["xmax"] = 10, 100
    plot_dic["title"] = "Short-lived remnants"
    plot_dic["figname"] = __outplotdir__ + "final_structure/" + "total_disk_hist_{}_short.png".format(v_n)
    #
    #plot_final_disk_histogram(task, plot_dic)

def custom_task_plot_final_disk_hist_3():

    task = [
        {"sim": "BLh_M13641364_M0_LK_SR", "type": "long", "v_n": None, "t": -1, "ext": {}, "plot": {"color": "black", "ls": "-", "lw": 0.8, "alpha": 1., "label": r"BLh q=1.00 (SR)", "drawstyle":"steps"}},
        {"sim": "BLh_M11461635_M0_LK_SR", "type": "long", "v_n": None, "t": -1, "ext": {}, "plot": {"color": "gray", "ls": "-", "lw": 0.8, "alpha": 1., "label": r"BLh q=1.43 (SR)", "drawstyle":"steps"}},
        {"sim": "DD2_M13641364_M0_LK_SR_R04", "type": "long", "v_n": None, "t": 60/1.e3, "ext": {}, "plot": {"color": "cyan", "ls": "-", "lw": 0.8, "alpha": 1., "label": r"DD2 q=1.00 (SR)", "drawstyle":"steps"}},
        {"sim": "DD2_M15091235_M0_LK_SR", "type": "long", "v_n": None, "t": -1, "ext": {}, "plot": {"color": "orange", "ls": "-", "lw": 0.8, "alpha": 1., "label": r"DD2 q=1.22 (SR)", "drawstyle":"steps"}},
        {"sim": "SFHo_M11461635_M0_LK_SR", "type": "long", "v_n": None, "t": -1, "ext": {}, "plot": {"color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "label": r"SFHo q=1.43 (SR)","drawstyle":"steps"}},
        {"sim": "SLy4_M11461635_M0_LK_SR", "type": "long", "v_n": None, "t": -1, "ext": {}, "plot": {"color": "red", "ls": "-", "lw": 0.8, "alpha": 1., "label": r"SLy4 q=1.43 (SR)","drawstyle":"steps"}},
    ]

    for t in task:
        t["plot"]["color"] = md.sim_dic_color[t["sim"]]
        t["plot"]["ls"] = md.sim_dic_ls[t["sim"]]
        t["plot"]["lw"] = md.sim_dic_lw[t["sim"]]

    for t in task: t["v_n"] = "Y_e"
    for t in task: t["normalize"] = True

    # for t in task:
    #     if t["sim"] == "LS220_M13641364_M0_LK_SR_restart":
    #         t["label"] = t["label"].replace("\_restart", "")
    # for t in task:
    #     if t["sim"] == "SFHo_M13641364_M0_LK_SR_2019pizza":
    #         t["label"] = t["label"].replace("\_2019pizza", "")
    # ----------------------- theta

    plot_dic = {
        "task_v_n": "temp",
        "figsize": (16., 5.5),
        "type": "all",
        "xmin": 5e-2, "xmax": 22,
        "ymin": 1e-4, "ymax": 1e-1,
        "normalize": True,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "log",
        "xlabel": r"$\langle T \rangle$ [MeV]",
        "ylabel": r"$M_{\rm disk}/M_b$",
        "legend": {"fancybox": False, "loc": 'upper right',
                   #"bbox_to_anchor": (1.0, 1.0),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "figname": __outplotdir__ + "final_structure/" + "disk_hist_shared.png",
        "savepdf": True,
        "fontsize": 18,
    }


    # ---------------- Ye
    plot_dic2 = copy.deepcopy(plot_dic)
    plot_dic2["xmax"] = 0
    plot_dic2["xmax"] = 0.4
    plot_dic2["legend"] = {}
    plot_dic2["xlabel"] = r"$\langle Y_e \rangle$"
    plot_dic2["task_v_n"] = "Ye"


    # ---------------- entr
    plot_dic3 = copy.deepcopy(plot_dic)
    plot_dic3["xmax"] = 0
    plot_dic3["xmax"] = 30
    plot_dic3["legend"] = {}
    plot_dic3["xlabel"] = r"$\langle s \rangle$ [$k_B/\rm{baryon}$]"
    plot_dic3["task_v_n"] = "entr"
    # for dic in task: dic["v_n"] = "Y_e"
    # plot_total_ejecta_hist(task, plot_dic)
    custom_plot_final_disk_histogram(task, [plot_dic, plot_dic2, plot_dic3])

''' -------------------------- iteration 2 | tasks ----------------------- '''

def task_plot_total_disk_mass_evo_2():

    task = [
        # 2 prompt collapse
        #{"sim": "BLh_M10201856_M0_LK_LR_AHfix", "label": r"BLh** q=1.82 (LR)", "color": "black", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long"},
        #{"sim": "BLh_M10201856_M0_LK_LR", "label": r"BLh* q=1.82 (LR)", "color": "black", "ls": "-", "lw": 1.0, "alpha": 1., "type": "long"},
        {"sim": "BLh_M10201856_M0_SR", "label": r"BLh* q=1.82 (SR)", "color": "black", "ls": "-", "lw": 1.0, "alpha": 1., "type": "long"},
        {"sim": "BLh_M10651772_M0_LK_LR", "label": r"BLh q=1.66 (LR)", "color": "black", "ls": "-", "lw": 1.0, "alpha": 1., "type": "long"},
        #{"sim": "BLh_M11841581_M0_LK_SR", "label": r"BLh q=1.34 (SR)", "color": "black", "ls": "-.", "lw": 1.0, "alpha": 1., "type": "long"},
        # 3 short lived
        {"sim": "LS220_M11461635_M0_LK_SR", "label": r"LS220 q=1.43 (SR)",  "color": "red", "ls": "-.", "lw": 1.0, "alpha": 1., "type": "short"},
        {"sim": "SLy4_M13641364_M0_SR", "label": r"SLy4* q=1.00 (SR)", "color": "magenta", "ls": "-", "lw": 1.0, "alpha": 1., "type": "short"},
        {"sim": "SFHo_M13641364_M0_SR", "label": r"SFHo* q=1.00 (SR)", "color": "green", "ls": "-", "lw": 1.0, "alpha": 1., "type": "short"},
        # long
        {"sim": "BLh_M13641364_M0_LK_SR", "label": r"BLh q=1.00 (SR)", "color": "black", "ls": "-", "lw": 1.0, "alpha": 1., "type": "long"},
        #{"sim": "DD2_M13641364_M0_LK_SR_R04", "label": r"DD2* q=1.00 (SR)", "color": "blue", "ls": "--", "lw": 1.0, "alpha": 1., "type": "long", "t1": 40},
        {"sim": "DD2_M13641364_M0_SR", "label": r"DD2* q=1.00 (SR)", "color": "blue", "ls": ":", "lw": 1.0, "alpha": 1., "type": "long"}
    ]

    for t in task:
        t["color"] = md.sim_dic_color[t["sim"]]
        t["ls"] = md.sim_dic_ls[t["sim"]]
        t["lw"] = md.sim_dic_lw[t["sim"]]

    # task = [
    #     {"sim": "BLh_M13641364_M0_LK_SR", "color": "black", "ls": "-", "lw": 0.8, "alpha": 1., "type":"long"},
    #     # {"sim": "BLh_M11841581_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"NS"},
    #     {"sim": "BLh_M11461635_M0_LK_SR", "color": "black", "ls": "-.", "lw": 0.8, "alpha": 1., "type":"long"},
    #     {"sim": "BLh_M10651772_M0_LK_SR", "color": "black", "ls": ":", "lw": 0.8, "alpha": 1., "type":"long"},
    #     # {"sim": "BLh_M10201856_M0_SR", "color": "black", "ls": "-", "lw": 0.7, "alpha": 1., "outcome":"PC"},
    #     # {"sim": "BLh_M10201856_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"PC"},
    #     #
    #     {"sim": "DD2_M13641364_M0_SR", "color": "blue", "ls": "-", "lw": 0.8, "alpha": 1., "type":"long"},
    #     # {"sim": "DD2_M13641364_M0_SR_R04", "color": "blue", "ls": "--", "lw": 0.7, "alpha": 1., "t1":60},
    #     {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "blue", "ls": "--", "lw": 0.8, "alpha": 1., "type":"long", "t1":40},
    #     {"sim": "DD2_M14971245_M0_SR", "color": "blue", "ls": "-.", "lw": 0.8, "alpha": 1., "type":"long"},
    #     {"sim": "DD2_M15091235_M0_LK_SR", "color": "blue", "ls": ":", "lw": 0.8, "alpha": 1., "type":"long"},
    #     #
    #     # {"sim": "LS220_M13641364_M0_SR", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1.},
    #     {"sim": "LS220_M13641364_M0_LK_SR_restart", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "type":"short"},
    #     {"sim": "LS220_M14691268_M0_LK_SR", "color": "red", "ls": "--", "lw": 0.8, "alpha": 1., "type":"long"},
    #     {"sim": "LS220_M11461635_M0_LK_SR", "color": "red", "ls": "-.", "lw": 0.8, "alpha": 1., "type":"short"},
    #     {"sim": "LS220_M10651772_M0_LK_SR", "color": "red", "ls": ":", "lw": 0.7, "alpha": 1., "type":"short"},
    #     #
    #     {"sim": "SFHo_M13641364_M0_SR", "color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "type":"short"},
    #     {"sim": "SFHo_M14521283_M0_SR", "color": "green", "ls": "--", "lw": 0.8, "alpha": 1., "type":"short"},
    #     {"sim": "SFHo_M11461635_M0_LK_SR", "color": "green", "ls": "-.", "lw": 0.8, "alpha": 1., "type":"long"},
    #     #
    #     {"sim": "SLy4_M13641364_M0_SR", "color": "magenta", "ls": "-", "lw": 0.8, "alpha": 1., "type":"short"},
    #     {"sim": "SLy4_M14521283_M0_SR", "color": "magenta", "ls": "--", "lw": 0.8, "alpha": 1., "type":"short"},
    #     # {"sim": "SLy4_M10201856_M0_LK_SR", "color": "orange", "ls": "-.", "lw": 0.7, "alpha": 1.},
    #     {"sim": "SLy4_M11461635_M0_LK_SR", "color": "magenta", "ls": ":", "lw": 0.8, "alpha": 1., "type":"long"}
    # ]

    # --- LONG ---
    plot_dic = {
        "figsize": (6., 5.5),
        #"aspect": {"aspect":"equal", "adjustable":"datalim"},
        "type": "all",
        "xmin": -10, "xmax": 90,
        "ymin": 0, "ymax": 0.40,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "linear",
        "xlabel": r"$t-t_{\rm merg}$ [ms]",
        "ylabel": r"$M_{\rm disk}$ $[M_{\odot}]$",
        "legend": {"fancybox": False, "loc": 'upper right',
                   #"bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "title": r"Disk Mass evolution",
        "figname": __outplotdir__ + "total_disk_mass_evo.png",
        "fontsize":14,
        "savepdf":True
    }
    plot_tot_disk_mass_evo(task, plot_dic)

def task_plot_disk_timecorr_2():

    task1 = {
        "sim": "BLh_M13641364_M0_LK_SR",
        "v_n": "Ye",
        "mask": "disk",
        "normalize": True,
    }

    def_plotdic1 = {
                  "figsize": (6., 5.0),
                  "vmin": 1e-6, "vmax": 1e-2,
                   "xmin": 0, "xmax": 90,
                   "ymin": 0, "ymax": 0.5,
                   "cmap": "jet",
                   "fontsize": 14,
                   "set_under": "black",
                   "set_over": "red",
                   "yscale": "linear",
                   "xscale": "linear",
                   "xlabel": r"$t-t_{\rm merg}$ [ms]",
                   "ylabel": r"$\langle Y_e \rangle$",
                   "title": "BLh q=1.00 (SR)",  # + "[{}ms]".format(task["t1"]),
                   "clabel": r"$M_{\rm disk}/M_b$",
                   # "text": {"x": 0.3, "y": 0.95, "s": r"$\theta>60\deg$", "ha": "center", "va": "top", "fontsize": 11,
                   #          "color": "white",
                   #          "transform": None},
                   "outdir": __outplotdir__,
                   "figname": "final_disk_timecorr_blh_q1_Lk.png",
                   "savepdf":True,
                   }

    plot_disk_timecorr(task1, def_plotdic1)

    ''' --- --- --- '''

    task2 = {
        # "sim": "LS220_M13641364_M0_LK_SR_AHfix",
        "sim": "LS220_M11461635_M0_LK_SR",
        # "sim": "LS220_M13641364_M0_LK_SR_restart",
        "v_n": "Ye",
        "mask": "disk",
        "normalize": True,
    }

    def_plotdic2 = {
                   "figsize": (6., 5.0),
                   "vmin": 1e-6, "vmax": 1e-2,
                   "xmin": 0, "xmax": 40,
                   "ymin": 0, "ymax": 0.5,
                   "cmap": "jet",
                   "fontsize":14,
                   "set_under": "black",
                   "set_over": "red",
                   "yscale": "linear",
                   "xscale": "linear",
                   "xlabel": r"$t-t_{\rm merg}$ [ms]",
                   "ylabel": r"$\langle Y_e \rangle$",
                   "title": r"LS220 q=1.43 (SR)",# + "[{}ms]".format(task["t1"]),
                   "clabel": r"$M_{\rm disk}/M_b$",
                   # "text": {"x": 0.3, "y": 0.95, "s": r"$\theta>60\deg$", "ha": "center", "va": "top", "fontsize": 11,
                   #          "color": "white",
                   #          "transform": None},
                   "outdir": __outplotdir__ ,
                   "figname": "final_disk_timecorr_ls220_q14_LK.png",
                   "savepdf": True,
                   }

    plot_disk_timecorr(task2, def_plotdic2)

    ''' --- --- --- '''

    task2 = {
        # "sim": "LS220_M13641364_M0_LK_SR_AHfix",
        "sim": "BLh_M13651365_M0_SR",
        # "sim": "LS220_M13641364_M0_LK_SR_restart",
        "v_n": "Ye",
        "mask": "disk",
        "normalize": True,
    }

    def_plotdic2 = {
                   "figsize": (6., 5.0),
                   "vmin": 1e-6, "vmax": 1e-2,
                   "xmin": 0, "xmax": 40,
                   "ymin": 0, "ymax": 0.5,
                   "cmap": "jet",
                   "fontsize":14,
                   "set_under": "black",
                   "set_over": "red",
                   "yscale": "linear",
                   "xscale": "linear",
                   "xlabel": r"$t-t_{\rm merg}$ [ms]",
                   "ylabel": r"$\langle Y_e \rangle$",
                   "title": r"LS220 q=1.43 (SR)",# + "[{}ms]".format(task["t1"]),
                   "clabel": r"$M_{\rm disk}/M_b$",
                   # "text": {"x": 0.3, "y": 0.95, "s": r"$\theta>60\deg$", "ha": "center", "va": "top", "fontsize": 11,
                   #          "color": "white",
                   #          "transform": None},
                   "outdir": __outplotdir__ ,
                   "figname": "final_disk_timecorr_blh_q1.png",
                   "savepdf": True,
                   }

    plot_disk_timecorr(task2, def_plotdic2)

def task_plot_final_disk_hist_2():
    v_n = "Ye"
    task = [
        {"sim": "BLh_M13641364_M0_LK_SR", "label":r"BLh* q=1.00 (SR)", "color": "black", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t": -1},
        # {"sim": "BLh_M11841581_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"NS", "v_n": v_n, "t": -1},
        {"sim": "BLh_M11461635_M0_LK_SR", "label":r"BLh* q=1.43 (SR)", "color": "gray", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t": -1},
        #{"sim": "BLh_M10651772_M0_LK_SR", "color": "black", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t": -1},
        # {"sim": "BLh_M10201856_M0_SR", "color": "black", "ls": "-", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
        # {"sim": "BLh_M10201856_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
        #
        #{"sim": "DD2_M13641364_M0_SR", "color": "blue", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t": -1},
        # {"sim": "DD2_M13641364_M0_SR_R04", "color": "blue", "ls": "--", "lw": 0.7, "alpha": 1., "t1":60, "v_n": v_n, "t": -1},
        {"sim": "DD2_M13641364_M0_LK_SR_R04", "label":r"DD2* q=1.00 (SR)", "color": "blue", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long","v_n": v_n, "t": 60/1.e3},
        #{"sim": "DD2_M14971245_M0_SR", "color": "blue", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t": -1},
        {"sim": "DD2_M15091235_M0_LK_SR", "label":r"DD2* q=1.22 (SR)", "color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t": -1},
        #
        # {"sim": "LS220_M13641364_M0_SR", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "t": -1},
        #{"sim": "LS220_M13641364_M0_LK_SR_restart", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "type": "short", "v_n": v_n, "t": -1},
        #{"sim": "LS220_M14691268_M0_LK_SR", "color": "red", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t": 80/1.e3},
        #{"sim": "LS220_M11461635_M0_LK_SR", "color": "red", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t": -1},
        #{"sim": "LS220_M10651772_M0_LK_SR", "color": "red", "ls": ":", "lw": 0.7, "alpha": 1., "type": "short", "v_n": v_n, "t": -1},
        #
        #{"sim": "SFHo_M13641364_M0_SR", "color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t": -1},
        #{"sim": "SFHo_M14521283_M0_SR", "color": "green", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t": -1},
        {"sim": "SFHo_M11461635_M0_LK_SR", "label":r"SFHo* q=1.42 (SR)", "color": "orange", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t": -1},
        #
        #{"sim": "SLy4_M13641364_M0_SR", "color": "magenta", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t": -1},
        #{"sim": "SLy4_M14521283_M0_SR", "color": "magenta", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t": -1},
        # {"sim": "SLy4_M10201856_M0_LK_SR", "color": "orange", "ls": "-.", "lw": 0.7, "alpha": 1., "v_n": v_n, "t": -1},
        {"sim": "SLy4_M11461635_M0_LK_SR", "label":r"SLy4* q=1.42 (SR)", "color": "magenta", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t": -1}
    ]

    # for t in task: t["label"] = r"\texttt{" + t["sim"].replace('_', '\_') + "}"
    # for t in task:
    #     if t["sim"] == "LS220_M13641364_M0_LK_SR_restart":
    #         t["label"] = t["label"].replace("\_restart", "")

    # --- Ye ---
    plot_dic = {
        "figsize": (6., 3.5),
        "type": "all",
        "xmin": 0, "xmax": 0.4,
        "ymin": 1e-4, "ymax": 1e-1,
        "normalize": True,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "log",
        "xlabel": r"$Y_e$",
        "ylabel": r"$M_{\rm disk}/M$",
        "legend": {"fancybox": False, "loc": 'lower right',
                   "bbox_to_anchor": (1.0, 1.0),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 12,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        #"title": r"Long-lived remnants",
        "figname": __outplotdir__ + "final_structure/" + "total_disk_hist_{}_long.png".format(v_n),
        "savepdf": True,
        "fontsize":14
    }
    plot_final_disk_histogram(task, plot_dic)

    plot_dic["type"] = "short"
    plot_dic["title"] = "Short-lived remnants"
    plot_dic["figname"] = __outplotdir__ + "total_disk_hist_{}_short.png".format(v_n)
    #
    # plot_final_disk_histogram(task, plot_dic)

    # --- temperature ---
    v_n = "temp"
    for t in task: t["v_n"] = v_n
    plot_dic = {
        "figsize": (6., 3.5),
        "type": "all",
        "xmin": 1e-1, "xmax": 3e1,
        "ymin": 1e-4, "ymax": 1e-1,
        "normalize": True,
        # "mask_below": 1e-15,
        "xscale": "log", "yscale": "log",
        "xlabel": r"$T$ [MeV]",
        "ylabel": r"$M_{\rm disk}/M$",
        "legend": {"fancybox": False, "loc": 'lower right',
                   "bbox_to_anchor": (1.0, 1.0),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 12,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        #"title": r"Long-lived remnants",
        "figname": __outplotdir__ + "final_structure/" + "total_disk_hist_{}_long.png".format(v_n),
        "savepdf": True,
        "fontsize": 14
    }
    plot_final_disk_histogram(task, plot_dic)

    plot_dic["type"] = "short"
    plot_dic["title"] = "Short-lived remnants"
    plot_dic["figname"] = __outplotdir__ + "final_structure/" + "total_disk_hist_{}_short.png".format(v_n)

    #plot_final_disk_histogram(task, plot_dic)

    # --- entropy ---
    v_n = "entr"
    for t in task: t["v_n"] = v_n
    plot_dic = {
        "figsize": (6., 3.5),
        "type": "long",
        "xmin": 0, "xmax": 30,
        "ymin": 1e-4, "ymax": 3e-1,
        "normalize": True,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "log",
        "xlabel": r"$s$ [$k_B$]",
        "ylabel": r"$M_{\rm disk}/M$",
        "legend": {"fancybox": False, "loc": 'lower right',
                   "bbox_to_anchor": (1.0, 1.0),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 12,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        #"title": r"Long-lived remnants",
        "figname": __outplotdir__ + "final_structure/" + "total_disk_hist_{}_long.png".format(v_n),
        "savepdf": True,
        "fontsize": 14
    }
    plot_final_disk_histogram(task, plot_dic)

    plot_dic["type"] = "short"
    plot_dic["title"] = "Short-lived remnants"
    plot_dic["figname"] = __outplotdir__ + "final_structure/" + "total_disk_hist_{}_short.png".format(v_n)

    #plot_final_disk_histogram(task, plot_dic)

    # --- press ---
    v_n = "press"
    for t in task: t["v_n"] = v_n
    plot_dic = {
        "figsize": (6., 3.5),
        "type": "long",
        "xmin": 1e-8, "xmax": 1e-6,
        "ymin": 1e-4, "ymax": 1e-1,
        "normalize": True,
        # "mask_below": 1e-15,
        "xscale": "log", "yscale": "log",
        "xlabel": r"$P$ [GEO]",
        "ylabel": r"$M_{\rm disk}/M$",
        "legend": {"fancybox": False, "loc": 'lower right',
                   "bbox_to_anchor": (1.0, 1.0),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 12,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        #"title": r"Long-lived remnants",
        "figname": __outplotdir__ + "final_structure/" + "total_disk_hist_{}_long.png".format(v_n),
        "savepdf": True,
        "fontsize": 14
    }
    plot_final_disk_histogram(task, plot_dic)

    plot_dic["type"] = "short"
    plot_dic["xmin"], plot_dic["xmax"] = 10, 100
    plot_dic["title"] = "Short-lived remnants"
    plot_dic["figname"] =__outplotdir__ + "final_structure/" + "total_disk_hist_{}_short.png".format(v_n)

    #plot_final_disk_histogram(task, plot_dic)

    # --- theta ---
    v_n = "theta"
    for t in task: t["v_n"] = v_n
    plot_dic = {
        "figsize": (6., 3.5),
        "type": "long",
        "xmin": 0, "xmax": 90,
        "ymin": 1e-4, "ymax": 2e-2,
        "normalize": True,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "log",
        "xlabel": r"Angle from the binary plane",
        "ylabel": r"$M_{\rm disk}/M$",
        "legend": {"fancybox": False, "loc": 'lower right',
                   "bbox_to_anchor": (1.0, 1.0),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 12,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        #"title": r"Long-lived remnants",
        "figname": __outplotdir__ + "final_structure/" + "total_disk_hist_{}_long.png".format(v_n),
        "savepdf": True,
        "fontsize": 14
    }
    plot_final_disk_histogram(task, plot_dic)

    plot_dic["type"] = "short"
    plot_dic["title"] = "Short-lived remnants"
    plot_dic["figname"] =__outplotdir__ + "final_structure/" + "total_disk_hist_{}_short.png".format(v_n)
    #
    #plot_final_disk_histogram(task, plot_dic)

    # --- r ---
    v_n = "r"
    for t in task: t["v_n"] = v_n
    plot_dic = {
        "figsize": (6., 3.5),
        "type": "long",
        "xmin": 10, "xmax": 200,
        "ymin": 1e-4, "ymax": 2e-2,
        "normalize": True,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "log",
        "xlabel": r"$R$ [km]",
        "ylabel": r"$M_{\rm disk}/M$",
        "legend": {"fancybox": False, "loc": 'lower right',
                   "bbox_to_anchor": (1.0, 1.0),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 12,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        # "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "final_structure/" + "total_disk_hist_{}_long.png".format(v_n),
        "savepdf": True,
        "fontsize": 14
    }
    plot_final_disk_histogram(task, plot_dic)

    plot_dic["type"] = "short"
    plot_dic["xmin"], plot_dic["xmax"] = 10, 100
    plot_dic["title"] = "Short-lived remnants"
    plot_dic["figname"] = __outplotdir__ + "final_structure/" + "total_disk_hist_{}_short.png".format(v_n)
    #
    #plot_final_disk_histogram(task, plot_dic)

def task_plot_final_disk_corr_2():
    task = {
        "sim": "BLh_M13641364_M0_LK_SR",
        "v_n_x": "entr",
        "v_n_y": "Ye",
        "t1": 88,
        "t2": 88,
        "normalize": True,
    }

    figsize = (6.,5.0)

    def_plotdic = {
        "figsize":figsize,
        "vmin": 1e-6, "vmax": 1e-2,
        "xmin": 0, "xmax": 30, "xscale": "linear", "xlabel": r"$s$ [$k_b$]",
        "ymin": 0.05, "ymax": 0.50, "yscale": "linear", "ylabel": r"$Y_e$",
        "cmap": "jet",
        "set_under": "white",
        "set_over": "red",
        "title": r"BLh* q=1.00 (SR), $t-t_{merg}=88$ [ms]",
        "clabel": r"$M_{\rm disk}/M$",
        # "text": {"x": 0.3, "y": 0.95, "s": r"$\theta>60\deg$", "ha": "center", "va": "top", "fontsize": 11,
        #          "color": "white",
        #          "transform": None},
        "outdir": __outplotdir__,
        "figname": "corr_s_ye_blh_q1_t88.png",
        "fontsize": 14,
        "savepdf": True
        }

    plot_final_disk_corr(task, def_plotdic)

    ''' --- movie --- '''
    # task["t1"], task["t2"] = ">20", ">20"
    # def_plotdic["title"] = None
    # def_plotdic["figname"] = "it"
    # def_plotdic["savepdf"] = False
    # def_plotdic["outdir"] =  __outplotdir__ + task["sim"] + "/corr_movie/"
    #
    # plot_final_disk_corr(task, def_plotdic)
    # os.system("convert -delay 20 -loop 0 {}*.png {}.gif".format(
    #     __outplotdir__ + task["sim"] + '/corr_movie/',
    #     __outplotdir__ + task["sim"] + "/" + \
    #     task["sim"] + "_" + task["v_n_x"] + '_' + task["v_n_y"]
    # ))
    # print("made: \n{}".format(
    #     __outplotdir__ + task["sim"] + "/" + \
    #     task["sim"] + "_" + task["v_n_x"] + '_' + task["v_n_y"] + ".gif"
    # ))

    ''' --- plot --- dens vs temp ---'''

    task = {
        "sim": "BLh_M13641364_M0_LK_SR",
        "v_n_x": "rho", "v_n_y": "temp",
        "t1": 88, "t2": 88,
        "normalize": True,
    }

    def_plotdic = {
        "figsize": figsize,
        "vmin": 1e-6, "vmax": 1e-2,
        "xmin": 1e8, "xmax": 1e13, "xscale": "log", "xlabel": r"$\rho$ [cm g$^{-3}$]",
        "ymin": 1e0, "ymax": 3e1, "yscale": "log", "ylabel": r"$T$ [MeV]",
        "cmap": "jet",
        "set_under": "black",
        "set_over": "red",
        "title": r"BLh* q=1.00 (SR), $t-t_{merg}=88$ [ms]",
        "clabel": r"$M_{\rm disk}/M$",
        # "text": {"x": 0.3, "y": 0.95, "s": r"$\theta>60\deg$", "ha": "center", "va": "top", "fontsize": 11,
        #          "color": "white",
        #          "transform": None},
        "outdir": __outplotdir__,
        "figname": "corr_rho_temp_blh_q1_t88.png",
        "fontsize": 14,
        "savepdf": True
    }

    plot_final_disk_corr(task, def_plotdic)

    ''' --- movie --- '''
    # task["t1"], task["t2"] = ">20", ">20"
    # def_plotdic["title"] = None
    # def_plotdic["figname"] = "it"
    # def_plotdic["savepdf"] = False
    # def_plotdic["outdir"] =  __outplotdir__ + task["sim"] + "/corr_movie2/"

    # plot_final_disk_corr(task, def_plotdic)
    # os.system("convert -delay 20 -loop 0 {}*.png {}.gif".format(
    #     __outplotdir__ + task["sim"] + '/corr_movie2/',
    #     __outplotdir__ + task["sim"] + "/" + \
    #     task["sim"] + "_" + task["v_n_x"] + '_' + task["v_n_y"]
    # ))
    # print("made: \n{}".format(
    #     __outplotdir__ + task["sim"] + "/" + \
    #     task["sim"] + "_" + task["v_n_x"] + '_' + task["v_n_y"] + ".gif"
    # ))

    ''' -------------------------------| DD2 |------------------------------ '''

    task = {
        "sim": "DD2_M13641364_M0_LK_SR_R04",
        "v_n_x": "entr",
        "v_n_y": "Ye",
        "t1": 109,
        "t2": 109,
        "normalize": True,
    }

    def_plotdic = {
        "figsize": figsize,
        "vmin": 1e-6, "vmax": 1e-2,
        "xmin": 0, "xmax": 30, "xscale": "linear", "xlabel": r"$s$ [$k_b$]",
        "ymin": 0.05, "ymax": 0.50, "yscale": "linear", "ylabel": r"$Y_e$",
        "cmap": "jet",
        "set_under": "white",
        "set_over": "red",
        "title": r"DD2* q=1.00 (SR), $t-t_{merg}=109$ [ms]",
        "clabel": r"$M_{\rm disk}/M$",
        # "text": {"x": 0.3, "y": 0.95, "s": r"$\theta>60\deg$", "ha": "center", "va": "top", "fontsize": 11,
        #          "color": "white",
        #          "transform": None},
        "outdir": __outplotdir__,
        "figname": "corr_s_ye_dd2_q1_t109.png",
        "fontsize": 14,
        "savepdf": True
        }

    plot_final_disk_corr(task, def_plotdic)

    task = {
        "sim": "DD2_M13641364_M0_LK_SR_R04",
        "v_n_x": "rho", "v_n_y": "temp",
        "t1": 109, "t2": 109,
        "normalize": True,
    }

    def_plotdic = {
        "figsize": figsize,
        "vmin": 1e-6, "vmax": 1e-2,
        "xmin": 1e8, "xmax": 1e13, "xscale": "log", "xlabel": r"$\rho$ [cm g$^{-3}$]",
        "ymin": 1e0, "ymax": 3e1, "yscale": "log", "ylabel": r"$T$ [MeV]",
        "cmap": "jet",
        "set_under": "black",
        "set_over": "red",
        "title": r"DD2* q=1.00 (SR), $t-t_{merg}=109$ [ms]",
        "clabel": r"$M_{\rm disk}/M$",
        # "text": {"x": 0.3, "y": 0.95, "s": r"$\theta>60\deg$", "ha": "center", "va": "top", "fontsize": 11,
        #          "color": "white",
        #          "transform": None},
        "outdir": __outplotdir__,
        "figname": "corr_rho_temp_dd2_q1_t109.png",
        "fontsize": 14,
        "savepdf": True
    }

    plot_final_disk_corr(task, def_plotdic)

''' ===================== | PAPER | ========================'''

if __name__ == "__main__":

    """ -- disk mass ecolution --- """

    task_plot_total_disk_mass_evo_2()

    """ -- disk final structure histogram --- """

    # task_plot_final_disk_hist_3()
    # custom_task_plot_final_disk_hist_3()

    """ --- disk time corr --- """
    # task_plot_disk_timecorr_2()

    """ --- disk slice structure --- """

    # task_plot_final_disk_structure()

    """ --- disk 2d correlation plot --- """

    # task_plot_final_disk_corr_2()

''' ------------- 3rd iteration ----------------- '''

if __name__ == "__main__":

    """ -- disk final structure histogram --- """
    #task_plot_final_disk_hist_3()

    #exit(0)

''' ------------- 2nd iteration ----------------- '''

if __name__ == "__main__":

    """ --- disk mass evol --- """
    # task_plot_total_disk_mass_evo_2()

    """ --- disk correlation --- """
    #task_plot_final_disk_corr_2()

    """ --- disk time corr --- """
    #task_plot_disk_timecorr_2()

    """ -- disk final structure histogram --- """

    #task_plot_final_disk_hist_2()

    #exit(0)

''' --------------- 1st iteration ---------------- '''

if __name__ == "__main__":

    """ --- rho & dens.norm1 --- """
    # task_plot_rho_max()

    """ --- disk mass evol --- """
    # task_plot_total_disk_mass_evo()

    """ --- disk final hist --- """
    # task_plot_final_disk_hist()
    # task_plot_mass_ave_val_evo()

    """ --- disk final corr --- """
    #task_plot_final_disk_corr()

    """ --- disk time corr --- """
    # task_plot_disk_timecorr()

    """ --- snapshot structure --- """
    # task_plot_final_disk_structure()