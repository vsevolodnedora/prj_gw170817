from __future__ import division

import numpy as np
import pandas as pd
import sys
import os
import copy
import h5py
import csv

from h5py.tests import old
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

from model_sets import models as md

from uutils import *

__outplotdir__ = "../figs/all3/plot_nucleo/"
if not os.path.isdir(__outplotdir__):
    os.mkdir(__outplotdir__)

''' --- test --- '''

def test_solar_r():

    old_table_fname = "/data01/numrel/vsevolod.nedora/Data/skynet/solar_r.dat"
    new_table_fname = "/data01/numrel/vsevolod.nedora/Data/skynet/solar_r_Prantzos2019.dat"
    middle_table_fname = "/data01/numrel/vsevolod.nedora/Data/skynet/solar_r_Sneden2008.dat"

    old_table = np.loadtxt(old_table_fname)
    new_table = np.zeros(10)
    with open(new_table_fname) as f:
        lines = f.readlines()
        for line in lines[2:]:
            elements = line.split()
            print(len(elements))
            z = float(elements[0])
            a = float(elements[1])
            # name = elements[2] # str
            n = float(elements[3])
            s_Sne_2008 = float(elements[4])
            r_Sne_2008 = float(elements[5])
            s_Gor1999 = float(elements[6])
            r_Cor1999 = float(elements[7])
            s_Bis2014 = float(elements[8])
            s_Prantzos2019 = float(elements[9])
            r_Prantzos2019 = float(elements[10])
            new_table = np.vstack((new_table, [z, a, n,
                                               s_Sne_2008, r_Sne_2008, s_Gor1999, r_Cor1999,
                                               s_Bis2014, s_Prantzos2019, r_Prantzos2019]))
    new_table = np.delete(new_table, 0, 0)

    middle_table = np.zeros(4)
    with open(middle_table_fname) as f:
        lines = f.readlines()
        for line in lines[1:]:
            elements = line.split()
            name = elements[0]
            z = float(elements[1])
            a = float(elements[2]) # isotope
            ns = float(elements[3])
            nr = float(elements[4])
            middle_table = np.vstack((middle_table, [z, a, ns, nr]))
    middle_table = np.delete(middle_table, 0,0)

    print(old_table)
    print(new_table)

    for i, a in enumerate(old_table[:,0]):
        y_old = old_table[i, 1]
        y_middle = middle_table[find_nearest_index(middle_table[:, 1], a), 3]
        print(y_old/y_middle)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(old_table[:, 0], old_table[:, 1]/0.0005, color="gray", marker="o", label=r"Old data $(\times10^{5})$", markersize=3, linestyle="None", fillstyle="none")
    ax.plot(new_table[:, 1], new_table[:, 2]*new_table[:, 9], color="red", marker="s", label=r"Prantzos+2019", markersize=3, linestyle="None", alpha=0.7, fillstyle="none")
    ax.plot(middle_table[:, 1], middle_table[:, 3], color="green", marker="s", label=r"Sneden+2008", markersize=3, linestyle="None", alpha=0.7, fillstyle="none")
    ax.set_xlim(50, 250)
    ax.set_ylim(1e-3, 1e1)
    ax.set_yscale("log")
    ax.set_xlabel("Atomic Number")
    ax.set_ylabel("Abundances")
    ax.set_title(r"Solar system isotopic compostion (r-process)")
    ax.legend()
    plt.tight_layout()
    print("plotted:{}".format(__outplotdir__+"test_solar.png"))
    plt.savefig(__outplotdir__+"test_solar.png", dpi=250)
    plt.close()

def test_solar_r_and_model():

    old_table_fname = "/data01/numrel/vsevolod.nedora/Data/skynet/solar_r.dat"
    new_table_fname = "/data01/numrel/vsevolod.nedora/Data/skynet/solar_r_Prantzos2019.dat"
    middle_table_fname = "/data01/numrel/vsevolod.nedora/Data/skynet/solar_r_Sneden2008.dat"

    old_table = np.loadtxt(old_table_fname)
    new_table = np.zeros(10)
    with open(new_table_fname) as f:
        lines = f.readlines()
        for line in lines[2:]:
            elements = line.split()
            print(len(elements))
            z = float(elements[0])
            a = float(elements[1])
            # name = elements[2] # str
            n = float(elements[3])
            s_Sne_2008 = float(elements[4])
            r_Sne_2008 = float(elements[5])
            s_Gor1999 = float(elements[6])
            r_Cor1999 = float(elements[7])
            s_Bis2014 = float(elements[8])
            s_Prantzos2019 = float(elements[9])
            r_Prantzos2019 = float(elements[10])
            new_table = np.vstack((new_table, [z, a, n,
                                               s_Sne_2008, r_Sne_2008, s_Gor1999, r_Cor1999,
                                               s_Bis2014, s_Prantzos2019, r_Prantzos2019]))
    new_table = np.delete(new_table, 0, 0)

    middle_table = np.zeros(4)
    with open(middle_table_fname) as f:
        lines = f.readlines()
        for line in lines[1:]:
            elements = line.split()
            name = elements[0]
            z = float(elements[1])
            a = float(elements[2]) # isotope
            ns = float(elements[3])
            nr = float(elements[4])
            middle_table = np.vstack((middle_table, [z, a, ns, nr]))
    middle_table = np.delete(middle_table, 0,0)

    print(old_table)
    print(new_table)

    for i, a in enumerate(old_table[:,0]):
        y_old = old_table[i, 1]
        y_middle = middle_table[find_nearest_index(middle_table[:, 1], a), 3]
        print(y_old/y_middle)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(old_table[:, 0], old_table[:, 1]/0.0005, color="gray", marker="o", label=r"Old data $(\times10^{5})$", markersize=3, linestyle="None", fillstyle="none")
    ax.plot(new_table[:, 1], new_table[:, 2]*new_table[:, 9], color="red", marker="s", label=r"Prantzos+2019", markersize=3, linestyle="None", alpha=0.7, fillstyle="none")
    ax.plot(middle_table[:, 1], middle_table[:, 3], color="green", marker="s", label=r"Sneden+2008", markersize=3, linestyle="None", alpha=0.7, fillstyle="none")
    ax.set_xlim(50, 250)
    ax.set_ylim(1e-3, 1e1)
    ax.set_yscale("log")
    ax.set_xlabel("Atomic Number")
    ax.set_ylabel("Abundances")
    ax.set_title(r"Solar system isotopic compostion (r-process)")
    ax.legend()
    plt.tight_layout()
    print("plotted:{}".format(__outplotdir__+"test_solar.png"))
    plt.savefig(__outplotdir__+"test_solar.png", dpi=250)
    plt.close()

''' --- modules --- '''

def plot_yields(tasks, plotdic):

    fig = plt.figure(figsize=plotdic["figsize"])
    ax = fig.add_subplot(111)


    # PLOT Models
    for task in tasks:
        if task["type"] == "all" or task["type"] == plotdic["type"]:

            o_data = ADD_METHODS_ALL_PAR(task["sim"], add_mask=task["mask"])
            a_sim, y_sim = o_data.get_nucleo_outflow(task["det"], task["mask"],
                                                     method=task["method"], solardataset=task["solar"])
            #
            ax.plot(a_sim, y_sim, color=task["color"], ls=task["ls"], lw=task["lw"],
                    alpha=task["alpha"], label=task["label"], drawstyle="steps")

    # PLOT Solar
    if len(plotdic["plot_solar"].keys())>0:
        o_data = ADD_METHODS_ALL_PAR(tasks[0]["sim"])
        a_sol, y_sol = o_data.get_nucleo_solar_normed("sum", tasks[0]["solar"])
        ax.plot(a_sol, y_sol, **plotdic["plot_solar"])

    # tmp = {"color": "gray", "marker": "x", "markersize": 5, "linestyle": ':', "linewidth":0.5, "label":"Linear extrapolation"}
    # ax.plot([-1, -1],[-2., -2], **tmp)

    if len(plotdic["text"].keys()) > 0:
        plotdic["text"]["transform"] = ax.transAxes
        ax.text(**plotdic["text"])

    ax.set_yscale(plotdic["yscale"])
    ax.set_xscale(plotdic["xscale"])

    ax.set_xlabel(plotdic["xlabel"])#, fontsize=11)
    ax.set_ylabel(plotdic["ylabel"])#, fontsize=11)

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
    han, lab = ax.get_legend_handles_labels()
    ax.add_artist(ax.legend(han[:-1], lab[:-1], **plotdic["legend"])) # default
    tmp = copy.deepcopy(plotdic["legend"])
    tmp["loc"] = "upper right"
    tmp["bbox_to_anchor"] = (1, 1.)#(0., 1.)
    ax.add_artist(ax.legend([han[-1]], [lab[-1]], **tmp)) # for extapolation

    plt.tight_layout()
    #

    print("plotted: \n")
    print(plotdic["figname"])
    plt.savefig(plotdic["figname"], dpi=128)
    plt.close()
    # exit(1)

def plot_yields_2(tasks, plotdic):

    fig = plt.figure(figsize=plotdic["figsize"])
    ax = fig.add_subplot(111)

    # PLOT Models
    for task in tasks:
        if plotdic["type"] == "all" or task["type"] == plotdic["type"]:

            o_data = ADD_METHODS_ALL_PAR(task["sim"], add_mask=task["mask"])
            a_sim, y_sim = o_data.get_nucleo_outflow(task["det"], task["mask"], method=task["method"], solardataset=task["solar"])
            #
            ax.plot(a_sim, y_sim, **task["plot"])

    ### PLOT Solar
    if len(plotdic["plot_solar"].keys())>0:
        o_data = ADD_METHODS_ALL_PAR(tasks[0]["sim"])
        a_sol, y_sol = o_data.get_nucleo_solar_normed("sum", tasks[0]["solar"])
        ax.plot(a_sol, y_sol, **plotdic["plot_solar"])

    # tmp = {"color": "gray", "marker": "x", "markersize": 5, "linestyle": ':', "linewidth":0.5, "label":"Linear extrapolation"}
    # ax.plot([-1, -1],[-2., -2], **tmp)

    if len(plotdic["text"].keys()) > 0:
        plotdic["text"]["transform"] = ax.transAxes
        ax.text(**plotdic["text"])

    ax.set_yscale(plotdic["yscale"])
    ax.set_xscale(plotdic["xscale"])

    ax.set_xlabel(plotdic["xlabel"], fontsize=plotdic["fontsize"])#, fontsize=11)
    ax.set_ylabel(plotdic["ylabel"], fontsize=plotdic["fontsize"])#, fontsize=11)

    ax.set_xlim(plotdic["xmin"], plotdic["xmax"])
    ax.set_ylim(plotdic["ymin"], plotdic["ymax"])
    #
    ax.tick_params(axis='both', which='both', labelleft=True,
                   labelright=False, tick1On=True, tick2On=True,
                   labelsize=plotdic["fontsize"],
                   direction='in',
                   bottom=True, top=True, left=True, right=True)
    ax.minorticks_on()
    #
    if "title" in plotdic.keys() and plotdic["title"] != None:
        ax.set_title(plotdic["title"], fontsize=plotdic["fontsize"])

    # LEGENDS
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
    # han, lab = ax.get_legend_handles_labels()
    # ax.add_artist(ax.legend(han[:-1], lab[:-1], **plotdic["legend"])) # default
    # tmp = copy.deepcopy(plotdic["legend"])
    # tmp["loc"] = "upper right"
    # tmp["bbox_to_anchor"] = (1, 1.)#(0., 1.)
    # ax.add_artist(ax.legend([han[-1]], [lab[-1]], **tmp)) # for extapolation

    plt.tight_layout()
    #

    print("plotted: \n")
    print(plotdic["figname"])
    plt.savefig(plotdic["figname"], dpi=128)
    if "savepdf" in plotdic.keys() and plotdic["savepdf"]: plt.savefig(plotdic["figname"].replace(".png", ".pdf"))
    plt.close()
    # exit(1)

def plot_yields_colocoded(task, models, plotdic):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    #
    norm = matplotlib.colors.Normalize(vmin=plotdic["vmin"], vmax=plotdic["vmax"])
    cmap = matplotlib.cm.get_cmap(plotdic["cmap"])
    #
    # print(models.EOS); exit(1)
    for ieos, eos in enumerate(list(set(list(models.EOS)))):
        # fig = plt.figure()
        # ax = fig.add_axes([0.15, 0.12, 0.82 - 0.15, 0.92 - 0.12])
        # cax = fig.add_axes([0.84, 0.12, 0.02, 0.92 - 0.12])

        fig = plt.figure(figsize=plotdic["figsize"])
        ax = fig.add_subplot(111)

        o_data = ADD_METHODS_ALL_PAR(models.index[0])
        a_sol, y_sol = o_data.get_nucleo_solar_normed("sum")
        ax.plot(a_sol, y_sol, **plotdic["plot_solar"])
        sel = models[(models.resolution=="SR") & (models.EOS == eos) & (models["Mej_tot-geo"]>1e-3)]
        for sim, m in sel.iterrows():
            o_data = ADD_METHODS_ALL_PAR(sim, add_mask=task["mask"])
            a_sim, y_sim = o_data.get_nucleo_outflow(task["det"], task["mask"], method=task["method"])
            print(sim, y_sim.max())
            ax.step(a_sim, y_sim, where='mid', color=cmap(norm(m.q)))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
        cbar.set_label(plotdic["label"])

        #
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

        ax.set_title(r"\texttt{"+eos+"}")
        ax.legend(**plotdic["legend"])

        if len(plotdic["text"].keys()) > 0:
            plotdic["text"]["transform"] = ax.transAxes
            ax.text(**plotdic["text"])

        plt.tight_layout()

        print("plotted: \n")
        plotdic["figname"] = __outplotdir__ + \
                             "colorcoded_yeilds_{}_{}_{}.png".format(eos,task["det"],task["mask"])
        print(plotdic["figname"])
        plt.savefig(plotdic["figname"], dpi=128)
        plt.close()

    #
    # fig = plt.figure(figsize=plotdic["figsize"])
    # ax = fig.add_subplot(111)
    #
    # #
    # cmap = matplotlib.cm.get_cmap(plotdic["cmap"])
    # norm = matplotlib.colors.Normalize(vmin=plotdic["vmin"], vmax=plotdic["vmax"])
    #
    # # PLOT Models
    # for index, model in models.iterrows():
    #     eos = models["EOS"]
    #     q = models["q"]
    #     if eos == task["eos"]:
    #
    #
    #
    #     if task["type"] == "all" or task["type"] == plotdic["type"]:
    #
    #         o_data = ADD_METHODS_ALL_PAR(task["sim"], add_mask=task["mask"])
    #         a_sim, y_sim = o_data.get_nucleo_outflow(task["det"], task["mask"], method=task["method"])
    #
    #         #
    #         ax.plot(a_sim, y_sim, color=task["color"], ls=task["ls"], lw=task["lw"],
    #                 alpha=task["alpha"], label=task["label"], drawstyle="steps")
    #
    # # PLOT Solar
    # if len(plotdic["plot_solar"].keys())>0:
    #     o_data = ADD_METHODS_ALL_PAR(tasks[0]["sim"])
    #     a_sol, y_sol = o_data.get_nucleo_solar_normed("sum")
    #     ax.plot(a_sol, y_sol, **plotdic["plot_solar"])
    #
    # # tmp = {"color": "gray", "marker": "x", "markersize": 5, "linestyle": ':', "linewidth":0.5, "label":"Linear extrapolation"}
    # # ax.plot([-1, -1],[-2., -2], **tmp)
    #
    # if len(plotdic["text"].keys()) > 0:
    #     plotdic["text"]["transform"] = ax.transAxes
    #     ax.text(**plotdic["text"])
    #
    # ax.set_yscale(plotdic["yscale"])
    # ax.set_xscale(plotdic["xscale"])
    #
    # ax.set_xlabel(plotdic["xlabel"])#, fontsize=11)
    # ax.set_ylabel(plotdic["ylabel"])#, fontsize=11)
    #
    # ax.set_xlim(plotdic["xmin"], plotdic["xmax"])
    # ax.set_ylim(plotdic["ymin"], plotdic["ymax"])
    # #
    # ax.tick_params(axis='both', which='both', labelleft=True,
    #                labelright=False, tick1On=True, tick2On=True,
    #                labelsize=12,
    #                direction='in',
    #                bottom=True, top=True, left=True, right=True)
    # ax.minorticks_on()
    # #
    # ax.set_title(plotdic["title"])
    #
    # # LEGENDS
    # han, lab = ax.get_legend_handles_labels()
    # ax.add_artist(ax.legend(han[:-1], lab[:-1], **plotdic["legend"])) # default
    # tmp = copy.deepcopy(plotdic["legend"])
    # tmp["loc"] = "lower left"
    # tmp["bbox_to_anchor"] = (0, 0.)#(0., 1.)
    # ax.add_artist(ax.legend([han[-1]], [lab[-1]], **tmp)) # for extapolation
    # # plt.legend(**plotdic["legend"])
    #
    # plt.tight_layout()
    # #
    #
    # print("plotted: \n")
    # print(plotdic["figname"])
    # plt.savefig(plotdic["figname"], dpi=128)
    # plt.close()

def plot_yields_colocoded2(tasks, plotdic):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    #
    norm = matplotlib.colors.Normalize(vmin=plotdic["vmin"], vmax=plotdic["vmax"])
    cmap = matplotlib.cm.get_cmap(plotdic["cmap"])
    #
    # print(models.EOS); exit(1)
    for eos in ["DD2", "LS220", "SFHo", "SLy4", "BLh"]:
        # fig = plt.figure()
        # ax = fig.add_axes([0.15, 0.12, 0.82 - 0.15, 0.92 - 0.12])
        # cax = fig.add_axes([0.84, 0.12, 0.02, 0.92 - 0.12])

        fig = plt.figure(figsize=plotdic["figsize"])
        ax = fig.add_subplot(111)

        o_data = ADD_METHODS_ALL_PAR(tasks[0]["sim"])
        a_sol, y_sol = o_data.get_nucleo_solar_normed("sum", dataset=tasks[0]["solar"])
        ax.plot(a_sol, y_sol, **plotdic["plot_solar"])

        for task in tasks:
            o_data = ADD_METHODS_ALL_PAR(task["sim"], add_mask=task["mask"])
            meos = o_data.get_initial_data_par("EOS")
            if meos == eos:
                a_sim, y_sim = o_data.get_nucleo_outflow(task["det"], task["mask"], method=task["method"], solardataset=task["solar"])
                q = o_data.get_initial_data_par("q")
                print(task["sim"], y_sim.max())
                task["plot"]["color"] = cmap(norm(q))
                task["plot"]["where"] = 'mid'
                ax.step(a_sim, y_sim, **task["plot"])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
        cbar.set_label(plotdic["label"], fontsize=plotdic["fontsize"])
        cbar.ax.tick_params(labelsize=plotdic["fontsize"])

        #
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

        if "title" in plotdic.keys() and plotdic["title"] != None:
            ax.set_title(r"\texttt{"+eos+"}", fontsize=plotdic["fontsize"])
        ax.legend(**plotdic["legend"])

        if len(plotdic["text"].keys()) > 0:
            plotdic["text"]["transform"] = ax.transAxes
            ax.text(**plotdic["text"])

        plt.tight_layout()

        print("plotted: \n")
        if plotdic["figname"].__contains__("_EOS_"):
            figname = plotdic["figname"].replace("EOS", eos)
        else:
            figname = plotdic["figname"]

        # plotdic["figname"] = __outplotdir__ + \
        #                      "cc_yeilds_{}_{}.png".format(eos, tasks[0]["mask"])
        print(figname)
        plt.savefig(figname, dpi=128)
        if "savepdf" in plotdic.keys() and plotdic["savepdf"]: plt.savefig(figname.replace(".png", ".pdf"))
        plt.close()

    #
    # fig = plt.figure(figsize=plotdic["figsize"])
    # ax = fig.add_subplot(111)
    #
    # #
    # cmap = matplotlib.cm.get_cmap(plotdic["cmap"])
    # norm = matplotlib.colors.Normalize(vmin=plotdic["vmin"], vmax=plotdic["vmax"])
    #
    # # PLOT Models
    # for index, model in models.iterrows():
    #     eos = models["EOS"]
    #     q = models["q"]
    #     if eos == task["eos"]:
    #
    #
    #
    #     if task["type"] == "all" or task["type"] == plotdic["type"]:
    #
    #         o_data = ADD_METHODS_ALL_PAR(task["sim"], add_mask=task["mask"])
    #         a_sim, y_sim = o_data.get_nucleo_outflow(task["det"], task["mask"], method=task["method"])
    #
    #         #
    #         ax.plot(a_sim, y_sim, color=task["color"], ls=task["ls"], lw=task["lw"],
    #                 alpha=task["alpha"], label=task["label"], drawstyle="steps")
    #
    # # PLOT Solar
    # if len(plotdic["plot_solar"].keys())>0:
    #     o_data = ADD_METHODS_ALL_PAR(tasks[0]["sim"])
    #     a_sol, y_sol = o_data.get_nucleo_solar_normed("sum")
    #     ax.plot(a_sol, y_sol, **plotdic["plot_solar"])
    #
    # # tmp = {"color": "gray", "marker": "x", "markersize": 5, "linestyle": ':', "linewidth":0.5, "label":"Linear extrapolation"}
    # # ax.plot([-1, -1],[-2., -2], **tmp)
    #
    # if len(plotdic["text"].keys()) > 0:
    #     plotdic["text"]["transform"] = ax.transAxes
    #     ax.text(**plotdic["text"])
    #
    # ax.set_yscale(plotdic["yscale"])
    # ax.set_xscale(plotdic["xscale"])
    #
    # ax.set_xlabel(plotdic["xlabel"])#, fontsize=11)
    # ax.set_ylabel(plotdic["ylabel"])#, fontsize=11)
    #
    # ax.set_xlim(plotdic["xmin"], plotdic["xmax"])
    # ax.set_ylim(plotdic["ymin"], plotdic["ymax"])
    # #
    # ax.tick_params(axis='both', which='both', labelleft=True,
    #                labelright=False, tick1On=True, tick2On=True,
    #                labelsize=12,
    #                direction='in',
    #                bottom=True, top=True, left=True, right=True)
    # ax.minorticks_on()
    # #
    # ax.set_title(plotdic["title"])
    #
    # # LEGENDS
    # han, lab = ax.get_legend_handles_labels()
    # ax.add_artist(ax.legend(han[:-1], lab[:-1], **plotdic["legend"])) # default
    # tmp = copy.deepcopy(plotdic["legend"])
    # tmp["loc"] = "lower left"
    # tmp["bbox_to_anchor"] = (0, 0.)#(0., 1.)
    # ax.add_artist(ax.legend([han[-1]], [lab[-1]], **tmp)) # for extapolation
    # # plt.legend(**plotdic["legend"])
    #
    # plt.tight_layout()
    # #
    #
    # print("plotted: \n")
    # print(plotdic["figname"])
    # plt.savefig(plotdic["figname"], dpi=128)
    # plt.close()

''' --- tasks --- '''

def task_plot_yields():

    mask = "geo"
    solar = "Prantzos2019" #"Sneden2008"# "old"
    method = "Asol=195"
    task = [
        {"sim": "BLh_M13641364_M0_LK_SR", "color": "black", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "mask": mask, "t2": -1, "ext":{}},
        # {"sim": "BLh_M11841581_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"NS", "mask": mask, "t": -1, "ext":None},
        {"sim": "BLh_M11461635_M0_LK_SR", "color": "black", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "mask": mask, "t2": -1, "ext":{}},
        {"sim": "BLh_M10651772_M0_LK_SR", "color": "black", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long", "mask": mask, "t2": -1, "ext":{}},
        # {"sim": "BLh_M10201856_M0_SR", "color": "black", "ls": "-", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
        # {"sim": "BLh_M10201856_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
        #
        {"sim": "DD2_M13641364_M0_SR", "color": "blue", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "mask": mask, "t2": -1, "ext":{}},
        # {"sim": "DD2_M13641364_M0_SR_R04", "color": "blue", "ls": "--", "lw": 0.7, "alpha": 1., "t1":60, "mask": mask, "t": -1, "ext":None},
        {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "blue", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long","mask": mask, "t": 60/1.e3, "ext":{}},
        {"sim": "DD2_M14971245_M0_SR", "color": "blue", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "mask": mask, "t2": -1, "ext":{}},
        {"sim": "DD2_M15091235_M0_LK_SR", "color": "blue", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long", "mask": mask, "t2": -1, "ext":{}},
        #
        {"sim": "LS220_M13641364_M0_SR", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "type": "short", "mask": mask, "t2": -1, "ext":{}},
        {"sim": "LS220_M13641364_M0_LK_SR_restart", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "type": "short", "mask": mask, "t2": -1, "ext":{}},
        {"sim": "LS220_M14691268_M0_LK_SR", "color": "red", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long", "mask": mask, "t2": 80/1.e3, "ext":{}},
        {"sim": "LS220_M14691268_M0_SR", "color": "red", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "mask": mask, "t2": 80 / 1.e3, "ext":{}}, # long

        # {"sim": "LS220_M11461635_M0_LK_SR", "color": "red", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1, "ext":None},
        # {"sim": "LS220_M10651772_M0_LK_SR", "color": "red", "ls": ":", "lw": 0.7, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1, "ext":None},
        #
        # {"sim": "SFHo_M13641364_M0_SR", "color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1, "ext":None},
        {"sim": "SFHo_M13641364_M0_LK_SR_2019pizza", "color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short", "mask": mask, "t2": -1, "ext":{}},
        # {"sim": "SFHo_M13641364_M0_LK_SR", "color": "green", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1},

        # {"sim": "SFHo_M14521283_M0_SR", "color": "green", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short", "mask": mask, "t2": -1},
        {"sim": "SFHo_M11461635_M0_LK_SR", "color": "green", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "mask": mask, "t2": -1, "ext":{}},
        #
        {"sim": "SLy4_M13641364_M0_SR", "color": "magenta", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short", "mask": mask, "t2": -1, "ext":{}},
        # {"sim": "SLy4_M13641364_M0_LK_SR_AHfix", "color": "magenta", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "short", "mask": mask, "t2": -1},
        # {"sim": "SLy4_M14521283_M0_SR", "color": "magenta", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short", "mask": mask, "t2": -1},
        # {"sim": "SLy4_M10201856_M0_LK_SR", "color": "orange", "ls": "-.", "lw": 0.7, "alpha": 1., "mask": mask, "t": -1},
        {"sim": "SLy4_M11461635_M0_LK_SR", "color": "magenta", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long", "mask": mask, "t2": -1, "ext":{}}
    ]

    for t in task: t["label"] = r"\texttt{" + t["sim"].replace('_', '\_') + "}"
    for t in task:
        if t["sim"] == "LS220_M13641364_M0_LK_SR_restart":
            t["label"] = t["label"].replace("\_restart", "")
    for t in task:
        if t["sim"] == "SFHo_M13641364_M0_LK_SR_2019pizza":
            t["label"] = t["label"].replace("\_2019pizza", "")
    for t in task: t["det"] = 0
    for t in task: t["mask"] = mask
    for t in task: t["method"] = method
    for t in task: t["solar"] =solar


    plot_dic = {
        "type": "long",
        "plot_solar": {'color':'gray', 'marker':'o', 'ms': 4, 'alpha': 0.4, "label":"Solar", "linestyle":"None"},
        "figsize": (6., 2.5),
        "xmin": 50, "xmax": 250.,
        "ymin": 1e-5, "ymax": 2e-1,
        # "mask_below": 1e-15, 'ymin': 1e-5, 'ymax': 2e-1, 'xmin': 50, 'xmax': 210,
        "xscale": "linear", "yscale": "log",
        "xlabel": r"Mass number, A",
        "ylabel": r'Relative final abundances',
        "legend": {"fancybox": False, "loc": 'center left',
                   "bbox_to_anchor":(1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon":False},
        "title": r"Long-lived remnants",
        "text": {"x":0.7, "y":0.95, "s":"Dyn. Ej.", "ha":"center", "va":"top",  "fontsize":11, "color":"black",
                        "transform":None},
        "figname": __outplotdir__ + "total_yeilds_{}.png".format(mask)
    }


    plot_yields(task, plot_dic)

    # Wind
    mask ="geo bern_geoend"
    plot_dic["text"]["transform"] = None
    plot_dic_wind = copy.deepcopy(plot_dic)
    plot_dic_wind["text"]["s"] = "Dyn.Ej + Wind"
    plot_dic_wind["figname"] = __outplotdir__ + "total_yeilds_{}.png".format("wind")
    #
    for t in task: t["mask"] = mask
    #
    plot_yields(task, plot_dic_wind)


    # # ---- short
    # short_plot_dic = copy.deepcopy(plot_dic)
    # short_plot_dic["type"] = "short"
    # short_plot_dic["ymin"], short_plot_dic["ymax"] = 0, 1.
    # short_plot_dic["xmin"], short_plot_dic["xmax"] = 0, 40
    # short_plot_dic["title"] = "Short-lived remnants"
    # short_plot_dic["figname"] = __outplotdir__ + "total_postdyn_ejecta_flux_short.png"
    #
    # # plot_total_ejecta_flux(task, short_plot_dic)
    #
    # # ---- long ---- wind Theta
    # for dic in task: dic["mask"] = "theta60_geoend"
    # plot_dic["ymin"] = 3e-5 #0
    # plot_dic["ymax"] = 2e-3 #0.15
    # plot_dic["yscale"] = "log"
    # plot_dic["title"] = r"$\nu$-wind assuming $\theta > 60\deg$"
    # plot_dic["figname"] = __outplotdir__ + "total_neutrino_wind_flux_theta_criteroin.png"
    # # plot_total_ejecta_flux(task, plot_dic)
    #
    # # ---- long --- wind Theta
    # for dic in task: dic["mask"] = "Y_e04_geoend"
    # plot_dic["ymin"] = 1e-5 #0
    # plot_dic["ymax"] = 1e-3 #0.04
    # plot_dic["yscale"] = "log"
    # plot_dic["title"] = r"$\nu$-wind assuming $Y_e > 0.4$"
    # plot_dic["figname"] = __outplotdir__ + "total_neutrino_wind_flux_ye_criteroin.png"
    # # plot_total_ejecta_flux(task, plot_dic)

def task_plot_yields_one_modle():

    mask = "geo"
    solar = "Prantzos2019" #"Sneden2008"# "old"
    method = "Asol=195"
    task = [
        {"sim": "BLh_M13641364_M0_LK_SR", "color": "black", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "mask": mask, "t2": -1, "ext":{}},
        # {"sim": "BLh_M11841581_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"NS", "mask": mask, "t": -1, "ext":None},
        # {"sim": "BLh_M11461635_M0_LK_SR", "color": "black", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "mask": mask, "t2": -1, "ext":{}},
        # {"sim": "BLh_M10651772_M0_LK_SR", "color": "black", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long", "mask": mask, "t2": -1, "ext":{}},
        # {"sim": "BLh_M10201856_M0_SR", "color": "black", "ls": "-", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
        # {"sim": "BLh_M10201856_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
        #
        # {"sim": "DD2_M13641364_M0_SR", "color": "blue", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "mask": mask, "t2": -1, "ext":{}},
        # {"sim": "DD2_M13641364_M0_SR_R04", "color": "blue", "ls": "--", "lw": 0.7, "alpha": 1., "t1":60, "mask": mask, "t": -1, "ext":None},
        # {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "blue", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long","mask": mask, "t": 60/1.e3, "ext":{}},
        # {"sim": "DD2_M14971245_M0_SR", "color": "blue", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "mask": mask, "t2": -1, "ext":{}},
        # {"sim": "DD2_M15091235_M0_LK_SR", "color": "blue", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long", "mask": mask, "t2": -1, "ext":{}},
        #
        # {"sim": "LS220_M13641364_M0_SR", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "type": "short", "mask": mask, "t2": -1, "ext":{}},
        # {"sim": "LS220_M13641364_M0_LK_SR_restart", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "type": "short", "mask": mask, "t2": -1, "ext":{}},
        # {"sim": "LS220_M14691268_M0_LK_SR", "color": "red", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long", "mask": mask, "t2": 80/1.e3, "ext":{}},
        # {"sim": "LS220_M14691268_M0_SR", "color": "red", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "mask": mask, "t2": 80 / 1.e3, "ext":{}}, # long
        #
        # # {"sim": "LS220_M11461635_M0_LK_SR", "color": "red", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1, "ext":None},
        # # {"sim": "LS220_M10651772_M0_LK_SR", "color": "red", "ls": ":", "lw": 0.7, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1, "ext":None},
        # #
        # # {"sim": "SFHo_M13641364_M0_SR", "color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1, "ext":None},
        # {"sim": "SFHo_M13641364_M0_LK_SR_2019pizza", "color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short", "mask": mask, "t2": -1, "ext":{}},
        # {"sim": "SFHo_M13641364_M0_LK_SR", "color": "green", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1},

        # {"sim": "SFHo_M14521283_M0_SR", "color": "green", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short", "mask": mask, "t2": -1},
        # {"sim": "SFHo_M11461635_M0_LK_SR", "color": "green", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "mask": mask, "t2": -1, "ext":{}},
        #
        # {"sim": "SLy4_M13641364_M0_SR", "color": "magenta", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short", "mask": mask, "t2": -1, "ext":{}},
        # {"sim": "SLy4_M13641364_M0_LK_SR_AHfix", "color": "magenta", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "short", "mask": mask, "t2": -1},
        # {"sim": "SLy4_M14521283_M0_SR", "color": "magenta", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short", "mask": mask, "t2": -1},
        # {"sim": "SLy4_M10201856_M0_LK_SR", "color": "orange", "ls": "-.", "lw": 0.7, "alpha": 1., "mask": mask, "t": -1},
        # {"sim": "SLy4_M11461635_M0_LK_SR", "color": "magenta", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long", "mask": mask, "t2": -1, "ext":{}}
    ]

    for t in task: t["label"] = r"\texttt{" + t["sim"].replace('_', '\_') + "}"
    for t in task:
        if t["sim"] == "LS220_M13641364_M0_LK_SR_restart":
            t["label"] = t["label"].replace("\_restart", "")
    for t in task:
        if t["sim"] == "SFHo_M13641364_M0_LK_SR_2019pizza":
            t["label"] = t["label"].replace("\_2019pizza", "")
    for t in task: t["det"] = 0
    for t in task: t["mask"] = mask
    for t in task: t["method"] = method
    for t in task: t["solar"] =solar

    plot_dic = {
        "type": "long",
        "plot_solar": {'color':'gray', 'marker':'o', 'ms': 4, 'alpha': 0.4, "label":"Solar [Prantzos2019]", "linestyle":"None"},
        "figsize": (5., 2.5),
        "xmin": 50, "xmax": 250.,
        "ymin": 1e-5, "ymax": 2e-1,
        # "mask_below": 1e-15, 'ymin': 1e-5, 'ymax': 2e-1, 'xmin': 50, 'xmax': 210,
        "xscale": "linear", "yscale": "log",
        "xlabel": r"Mass number, A",
        "ylabel": r'Relative final abundances',
        "legend": {"fancybox": False, "loc": 'lower left',
                   # "bbox_to_anchor":(1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon":False},
        "title": r"Long-lived remnants",
        "text": {"x":0.4, "y":0.95, "s":"Dyn. Ej.", "ha":"center", "va":"top",  "fontsize":11, "color":"black",
                        "transform":None},
        "figname": __outplotdir__ + "total_yeilds_{}.png".format(mask)
    }

    plot_yields(task, plot_dic)

    # Wind
    mask ="geo bern_geoend"
    plot_dic["text"]["transform"] = None
    plot_dic_wind = copy.deepcopy(plot_dic)
    plot_dic_wind["text"]["s"] = "Dyn.Ej + Wind"
    plot_dic_wind["figname"] = __outplotdir__ + "total_yeilds_{}.png".format("wind")
    #
    for t in task: t["mask"] = mask
    #
    plot_yields(task, plot_dic_wind)


    # # ---- short
    # short_plot_dic = copy.deepcopy(plot_dic)
    # short_plot_dic["type"] = "short"
    # short_plot_dic["ymin"], short_plot_dic["ymax"] = 0, 1.
    # short_plot_dic["xmin"], short_plot_dic["xmax"] = 0, 40
    # short_plot_dic["title"] = "Short-lived remnants"
    # short_plot_dic["figname"] = __outplotdir__ + "total_postdyn_ejecta_flux_short.png"
    #
    # # plot_total_ejecta_flux(task, short_plot_dic)
    #
    # # ---- long ---- wind Theta
    # for dic in task: dic["mask"] = "theta60_geoend"
    # plot_dic["ymin"] = 3e-5 #0
    # plot_dic["ymax"] = 2e-3 #0.15
    # plot_dic["yscale"] = "log"
    # plot_dic["title"] = r"$\nu$-wind assuming $\theta > 60\deg$"
    # plot_dic["figname"] = __outplotdir__ + "total_neutrino_wind_flux_theta_criteroin.png"
    # # plot_total_ejecta_flux(task, plot_dic)
    #
    # # ---- long --- wind Theta
    # for dic in task: dic["mask"] = "Y_e04_geoend"
    # plot_dic["ymin"] = 1e-5 #0
    # plot_dic["ymax"] = 1e-3 #0.04
    # plot_dic["yscale"] = "log"
    # plot_dic["title"] = r"$\nu$-wind assuming $Y_e > 0.4$"
    # plot_dic["figname"] = __outplotdir__ + "total_neutrino_wind_flux_ye_criteroin.png"
    # # plot_total_ejecta_flux(task, plot_dic)

def task_plot_colocoded_yileds():

    #
    task = {"eos":"BLh", "det":0, "mask":"geo", "method":"Asol=195"}
    #
    models = md.simulations_nonblacklisted
    #
    # task = [
    #     {"sim": "BLh_M13641364_M0_LK_SR", "color": "black", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "det":0, "mask":"geo", "method":"Asol=195"},
    #     {"sim": "BLh_M11841581_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "type":"short", "det":0, "mask":"geo", "method":"Asol=195"},
    #     {"sim": "BLh_M11461635_M0_LK_SR", "color": "black", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "det":0, "mask":"geo", "method":"Asol=195"},
    #     {"sim": "BLh_M10651772_M0_LK_SR", "color": "black", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long", "det":0, "mask":"geo", "method":"Asol=195"},
    #     {"sim": "BLh_M10201856_M0_SR", "color": "black", "ls": "-", "lw": 0.7, "alpha": 1., "type":"short", "det":0, "mask":"geo", "method":"Asol=195"},
    #     {"sim": "BLh_M10201856_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "type":"short", "det":0, "mask":"geo", "method":"Asol=195"},
    #     #
    #     # {"sim": "DD2_M13641364_M0_SR", "color": "blue", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "det":0, "mask":"geo", "method":"Asol=195"},
    #     {"sim": "DD2_M13641364_M0_SR_R04", "color": "blue", "ls": "--", "lw": 0.7, "alpha": 1., "type":"long", "det":0, "mask":"geo", "method":"Asol=195"},
    #     {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "blue", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long", "det":0, "mask":"geo", "method":"Asol=195"},
    #     {"sim": "DD2_M14971245_M0_SR", "color": "blue", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "det":0, "mask":"geo", "method":"Asol=195"},
    #     {"sim": "DD2_M15091235_M0_LK_SR", "color": "blue", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long", "det":0, "mask":"geo", "method":"Asol=195"},
    #     #
    #     {"sim": "LS220_M13641364_M0_SR", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "type": "short", "det":0, "mask":"geo", "method":"Asol=195"},
    #     {"sim": "LS220_M13641364_M0_LK_SR_restart", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "type": "short", "det":0, "mask":"geo", "method":"Asol=195"},
    #     {"sim": "LS220_M14691268_M0_LK_SR", "color": "red", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long", "det":0, "mask":"geo", "method":"Asol=195"},
    #     {"sim": "LS220_M14691268_M0_SR", "color": "red", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "det":0, "mask":"geo", "method":"Asol=195"},
    #
    #     {"sim": "LS220_M11461635_M0_LK_SR", "color": "red", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "short", "det":0, "mask":"geo", "method":"Asol=195"},
    #     {"sim": "LS220_M10651772_M0_LK_SR", "color": "red", "ls": ":", "lw": 0.7, "alpha": 1., "type": "short", "det":0, "mask":"geo", "method":"Asol=195"},
    #     #
    #     {"sim": "SFHo_M13641364_M0_SR", "color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short", "det":0, "mask":"geo", "method":"Asol=195"},
    #     {"sim": "SFHo_M13641364_M0_LK_SR_2019pizza", "color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short", "det":0, "mask":"geo", "method":"Asol=195"},
    #     # {"sim": "SFHo_M13641364_M0_LK_SR", "color": "green", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1},
    #
    #     {"sim": "SFHo_M14521283_M0_SR", "color": "green", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short", "det":0, "mask":"geo", "method":"Asol=195"},
    #     {"sim": "SFHo_M11461635_M0_LK_SR", "color": "green", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "det":0, "mask":"geo", "method":"Asol=195"},
    #     #
    #     {"sim": "SLy4_M13641364_M0_SR", "color": "magenta", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short", "det":0, "mask":"geo", "method":"Asol=195"},
    #     # {"sim": "SLy4_M13641364_M0_LK_SR_AHfix", "color": "magenta", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "short", "mask": mask, "t2": -1},
    #     {"sim": "SLy4_M14521283_M0_SR", "color": "magenta", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short", "det":0, "mask":"geo", "method":"Asol=195"},
    #     # {"sim": "SLy4_M10201856_M0_LK_SR", "color": "orange", "ls": "-.", "lw": 0.7, "alpha": 1., "mask": mask, "t": -1},
    #     {"sim": "SLy4_M11461635_M0_LK_SR", "color": "magenta", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long", "det":0, "mask":"geo", "method":"Asol=195"}
    # ]


    #
    plot_dic = {
        "type": "long",
        "plot_solar": {'color':'gray', 'marker':'o', 'ms': 4, 'alpha': 0.4,
                       "label":"Solar", "linestyle":"None","zorder":100},
        "figsize": (6., 2.5),
        "cmap": "jet",
        "vmin":1.0, "vmax":1.8, "scale": "linear", "label":"$M_1/M_2$",
        "xmin": 50, "xmax": 250., "xscale": "linear", "xlabel": r"Mass number, A",
        "ymin": 1e-5, "ymax": 2e-1,"yscale": "log", "ylabel": r'Relative final abundances',
        "legend": {"fancybox": False, "loc": 'lower right',
                   "bbox_to_anchor":(1.0, 0.0),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon":False},
        "title": r"Long-lived remnants",
        "text": {"x":0.7, "y":0.95, "s":"Dyn. Ej.", "ha":"center", "va":"top",  "fontsize":11, "color":"black",
                        "transform":None},
        "figname": None
    }

    plot_yields_colocoded(task, models, plot_dic)

def task_plot_colocoded_yileds2():

    #
    # task = {"eos":"BLh", "det":0, "mask":"geo", "method":"Asol=195"}
    #
    # models = md.simulations_nonblacklisted
    #
    task = [
        {"sim": "BLh_M13641364_M0_LK_SR", "color": "black", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "det":0, "mask":"geo", "method":"Asol=195"},
        {"sim": "BLh_M13651365_M0_SR",    "color": "black", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short",  "det": 0, "mask": "geo", "method": "Asol=195"},
        {"sim": "BLh_M11841581_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "type":"short", "det":0, "mask":"geo", "method":"Asol=195"},
        {"sim": "BLh_M11461635_M0_LK_SR", "color": "black", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "det":0, "mask":"geo", "method":"Asol=195"},
        {"sim": "BLh_M10651772_M0_LK_SR", "color": "black", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long", "det":0, "mask":"geo", "method":"Asol=195"},
        {"sim": "BLh_M10201856_M0_SR", "color": "black", "ls": "-", "lw": 0.7, "alpha": 1., "type":"short", "det":0, "mask":"geo", "method":"Asol=195"},
        {"sim": "BLh_M10201856_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "type":"short", "det":0, "mask":"geo", "method":"Asol=195"},
        #
        # {"sim": "DD2_M13641364_M0_SR", "color": "blue", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "det":0, "mask":"geo", "method":"Asol=195"},
        {"sim": "DD2_M13641364_M0_SR_R04", "color": "blue", "ls": "--", "lw": 0.7, "alpha": 1., "type":"long", "det":0, "mask":"geo", "method":"Asol=195"},
        {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "blue", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long", "det":0, "mask":"geo", "method":"Asol=195"},
        {"sim": "DD2_M14971245_M0_SR", "color": "blue", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "det":0, "mask":"geo", "method":"Asol=195"},
        {"sim": "DD2_M15091235_M0_LK_SR", "color": "blue", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long", "det":0, "mask":"geo", "method":"Asol=195"},
        #
        {"sim": "LS220_M13641364_M0_SR", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "type": "short", "det":0, "mask":"geo", "method":"Asol=195"},
        {"sim": "LS220_M13641364_M0_LK_SR_restart", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "type": "short", "det":0, "mask":"geo", "method":"Asol=195"},
        {"sim": "LS220_M14691268_M0_LK_SR", "color": "red", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long", "det":0, "mask":"geo", "method":"Asol=195"},
        {"sim": "LS220_M14691268_M0_SR", "color": "red", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "det":0, "mask":"geo", "method":"Asol=195"},

        {"sim": "LS220_M11461635_M0_LK_SR", "color": "red", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "short", "det":0, "mask":"geo", "method":"Asol=195"},
        {"sim": "LS220_M10651772_M0_LK_SR", "color": "red", "ls": ":", "lw": 0.7, "alpha": 1., "type": "short", "det":0, "mask":"geo", "method":"Asol=195"},
        #
        {"sim": "SFHo_M13641364_M0_SR", "color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short", "det":0, "mask":"geo", "method":"Asol=195"},
        {"sim": "SFHo_M13641364_M0_LK_SR_2019pizza", "color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short", "det":0, "mask":"geo", "method":"Asol=195"},
        # {"sim": "SFHo_M13641364_M0_LK_SR", "color": "green", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1},

        {"sim": "SFHo_M14521283_M0_SR", "color": "green", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short", "det":0, "mask":"geo", "method":"Asol=195"},
        {"sim": "SFHo_M11461635_M0_LK_SR", "color": "green", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "det":0, "mask":"geo", "method":"Asol=195"},
        #
        {"sim": "SLy4_M13641364_M0_SR", "color": "magenta", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short", "det":0, "mask":"geo", "method":"Asol=195"},
        # {"sim": "SLy4_M13641364_M0_LK_SR_AHfix", "color": "magenta", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "short", "mask": mask, "t2": -1},
        {"sim": "SLy4_M14521283_M0_SR", "color": "magenta", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short", "det":0, "mask":"geo", "method":"Asol=195"},
        # {"sim": "SLy4_M10201856_M0_LK_SR", "color": "orange", "ls": "-.", "lw": 0.7, "alpha": 1., "mask": mask, "t": -1},
        {"sim": "SLy4_M11461635_M0_LK_SR", "color": "magenta", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long", "det":0, "mask":"geo", "method":"Asol=195"}
    ]
    for t in task:
        t["solar"] = "Prantzos2019"

    #
    plot_dic = {
        "type": "long",
        "plot_solar": {'color':'gray', 'marker':'o', 'ms': 4, 'alpha': 0.4,
                       "label":"Solar", "linestyle":"None","zorder":100},
        "figsize": (6., 2.5),
        "cmap": "jet",
        "vmin":1.0, "vmax":1.8, "scale": "linear", "label":"$M_1/M_2$",
        "xmin": 50, "xmax": 250., "xscale": "linear", "xlabel": r"Mass number, A",
        "ymin": 1e-5, "ymax": 2e-2,"yscale": "log", "ylabel": r'Relative final abundances',
        "legend": {"fancybox": False, "loc": 'lower right',
                   "bbox_to_anchor":(1.0, 0.0),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon":False},
        "title": r"Long-lived remnants",
        "text": {"x":0.7, "y":0.95, "s":"Dyn. Ej.", "ha":"center", "va":"top",  "fontsize":11, "color":"black",
                        "transform":None},
        "figname": None
    }

    plot_yields_colocoded2(task, plot_dic)

''' --- iteration 2 | tasks --- '''

def task_plot_yields_2():

    task = [
        {"sim": "BLh_M13641364_M0_LK_SR", "mask": "geo",             "plot": {"color": "black", "ls": "--", "lw": 0.8, "alpha": 1.}},
        {"sim": "BLh_M13641364_M0_LK_SR", "mask": "geo bern_geoend", "plot": {"color": "black", "ls": "-", "lw": 1.0, "label":"BLh q=1.00 (SR)", "alpha": 1.}},

        {"sim": "DD2_M13641364_M0_LK_SR_R04", "mask": "geo",             "plot": {"color": "blue", "ls": "--", "lw": 0.8, "alpha": 1.}},
        {"sim": "DD2_M13641364_M0_LK_SR_R04", "mask": "geo bern_geoend", "plot": {"color": "blue", "ls": "-", "lw": 1.0, "label":"DD2 q=1.00 (SR)", "alpha": 1.}},
    ]

    for t in task:
        t["plot"]["color"] = md.sim_dic_color[t["sim"]]
        #t["plot"]["ls"] = md.sim_dic_ls[t["sim"]]

    for t in task:
        for t in task: t["det"] = 0
        for t in task: t["plot"]["drawstyle"] = "steps"
        for t in task: t["method"] = "Asol=195"
        for t in task: t["solar"] = "Prantzos2019" #"Sneden2008"# "old"

    plot_dic = {
        "type": "all",
        "figsize": (6., 5.5),
        "xmin": 50, "xmax": 250.,
        "ymin": 1e-5, "ymax": 2e-1,
        # "mask_below": 1e-15, 'ymin': 1e-5, 'ymax': 2e-1, 'xmin': 50, 'xmax': 210,
        "xscale": "linear", "yscale": "log",
        "xlabel": r"Mass number, A",
        "ylabel": r'Relative final abundances',
        "legend": {"fancybox": False, "loc": 'upper right',
                   #"bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "multilegend": {
            "lines": [
                {"label": "Solar", 'color': 'gray', 'marker': 'o', 'ms': 4, 'alpha': 0.4, "linestyle": "None"},
                {"label": r"Dyn.", "ls": "--", "lw": 0.8, "alpha": 1., "color": "gray"},
                {"label": r"Dyn.+Wind", "ls": "-", "lw": 0.8, "alpha": 1., "color": "gray"}
            ],
            "legend": {"fancybox": False, "loc": 'lower left',
                       "shadow": "False", "ncol": 1, "fontsize": 14,
                       "framealpha": 0., "borderaxespad": 0., "frameon": False}
        },
        "title": None, #r"Long-lived remnants",
        "text": {},#{"x": 0.7, "y": 0.95, "s": "Dyn. Ej.", "ha": "center", "va": "top", "fontsize": 11, "color": "black", "transform": None},
        "plot_solar": {'color': 'gray', 'marker': 'o', 'ms': 4, 'alpha': 0.4, "linestyle": "None"},
        "fontsize":14,
        "savepdf":True,
        "figname": __outplotdir__ + "nucleo_dd2_blh.png"
    }

    plot_yields_2(task, plot_dic)

def task_plot_colocoded_yileds3():

    #
    # task = {"eos":"BLh", "det":0, "mask":"geo", "method":"Asol=195"}
    #
    # models = md.simulations_nonblacklisted
    #

    task = [
        # BLh
        {"sim": "BLh_M13641364_M0_LK_SR", "det": 0, "mask": "geo", "method": "Asol=195", "type": "short", "plot": {"color": "black", "ls": "-", "lw": 1.0, "alpha": 1.}},
        {"sim": "BLh_M13651365_M0_SR",    "det": 0, "mask": "geo", "method": "Asol=195", "type": "short", "plot": {"color": "black", "ls": "-", "lw": 1.0, "alpha": 1.}},
        {"sim": "BLh_M11841581_M0_LK_SR", "det": 0, "mask": "geo", "method": "Asol=195", "type": "short", "plot": {"color": "black", "ls": "-", "lw": 1.0, "alpha": 1.}},
        {"sim": "BLh_M11461635_M0_LK_SR", "det": 0, "mask": "geo", "method": "Asol=195", "type": "long", "plot": {"color": "black", "ls": "-", "lw": 1.0, "alpha": 1.}},
        {"sim": "BLh_M10651772_M0_LK_SR", "det": 0, "mask": "geo", "method": "Asol=195", "type": "short", "plot": {"color": "black", "ls": "-", "lw": 1.0, "alpha": 1.}},
        {"sim": "BLh_M10201856_M0_SR",    "det": 0, "mask": "geo", "method": "Asol=195", "type": "short", "plot": {"color": "black", "ls": "-", "lw": 1.0, "alpha": 1.}},
        {"sim": "BLh_M10201856_M0_LK_SR", "det": 0, "mask": "geo", "method": "Asol=195", "type": "short", "plot": {"color": "black", "ls": "-", "lw": 1.0, "alpha": 1.}},
        # DD2
        {"sim": "DD2_M13641364_M0_SR_R04", "det": 0, "mask": "geo", "method": "Asol=195", "type": "long", "plot": {"color": "black", "ls": "-", "lw": 1.0, "alpha": 1.}},
        {"sim": "DD2_M13641364_M0_LK_SR_R04","det": 0, "mask": "geo", "method": "Asol=195","type": "long", "plot":{"color": "black", "ls": "-", "lw": 1.0, "alpha": 1.}},
        {"sim": "DD2_M14971245_M0_SR",    "det": 0, "mask": "geo", "method": "Asol=195",  "type": "long", "plot":{"color": "black", "ls": "-", "lw": 1.0, "alpha": 1.}},
        {"sim": "DD2_M15091235_M0_LK_SR", "det": 0, "mask": "geo", "method": "Asol=195", "type": "long", "plot": {"color": "black", "ls": "-", "lw": 1.0, "alpha": 1.}},
        # LS220
        {"sim": "LS220_M13641364_M0_SR",   "det": 0, "mask": "geo", "method": "Asol=195", "type": "short", "plot": {"color": "black", "ls": "-", "lw": 1.0, "alpha": 1.}},
        {"sim": "LS220_M13641364_M0_LK_SR_restart","det": 0, "mask": "geo", "method": "Asol=195", "type": "short", "plot": {"color": "black", "ls": "-", "lw": 1.0, "alpha": 1.}},
        {"sim": "LS220_M14691268_M0_LK_SR", "det": 0, "mask": "geo", "method": "Asol=195", "type": "short", "plot": {"color": "black", "ls": "-", "lw": 1.0, "alpha": 1.}},
        {"sim": "LS220_M14691268_M0_SR",    "det": 0, "mask": "geo", "method": "Asol=195", "type": "short", "plot": {"color": "black", "ls": "-", "lw": 1.0, "alpha": 1.}},
        {"sim": "LS220_M11461635_M0_LK_SR", "det": 0, "mask": "geo", "method": "Asol=195", "type": "short", "plot": {"color": "black", "ls": "-", "lw": 1.0, "alpha": 1.}},
        {"sim": "LS220_M10651772_M0_LK_SR", "det": 0, "mask": "geo", "method": "Asol=195", "type": "short", "plot": {"color": "black", "ls": "-", "lw": 1.0, "alpha": 1.}},
        # SFHo
        {"sim": "SFHo_M13641364_M0_SR",     "det": 0, "mask": "geo", "method": "Asol=195", "type": "short", "plot": {"color": "black", "ls": "-", "lw": 1.0, "alpha": 1.}},
        {"sim": "SFHo_M13641364_M0_LK_SR_2019pizza", "det": 0, "mask": "geo", "method": "Asol=195", "type": "short", "plot": {"color": "black", "ls": "-", "lw": 1.0, "alpha": 1.}},
        {"sim": "SFHo_M14521283_M0_SR",     "det": 0, "mask": "geo", "method": "Asol=195", "type": "short", "plot": {"color": "black", "ls": "-", "lw": 1.0, "alpha": 1.}},
        {"sim": "SFHo_M11461635_M0_LK_SR",  "det": 0, "mask": "geo", "method": "Asol=195", "type": "long", "plot": {"color": "black", "ls": "-", "lw": 1.0, "alpha": 1.}},
        #
        {"sim": "SLy4_M13641364_M0_SR",     "det": 0, "mask": "geo", "method": "Asol=195", "type": "short", "plot": {"color": "black", "ls": "-", "lw": 1.0, "alpha": 1.}},
        {"sim": "SLy4_M14521283_M0_SR",     "det": 0, "mask": "geo", "method": "Asol=195", "type": "short", "plot": {"color": "black", "ls": "-", "lw": 1.0, "alpha": 1.}},
        {"sim": "SLy4_M11461635_M0_LK_SR",  "det": 0, "mask": "geo", "method": "Asol=195", "type": "long", "plot": {"color": "black", "ls": "-", "lw": 1.0, "alpha": 1.}},
    ]

    for t in task:
        t["solar"] = "Prantzos2019"

    for t in task:
        if t["type"] == "long": t["mask"] = "geo bern_geoend"

    #
    plot_dic = {
        "type": "all",
        "plot_solar": {'color':'gray', 'marker':'o', 'ms': 4, 'alpha': 0.4,  "label":"Solar", "linestyle":"None","zorder":100},
        "figsize": (6., 5.1),
        "cmap": "jet",
        "vmin":1.0, "vmax":1.8, "scale": "linear", "label": r"$q$",#r"$M_1/M_2$",
        "xmin": 50, "xmax": 250., "xscale": "linear", "xlabel": r"Mass number, A",
        "ymin": 1e-5, "ymax": 2e-2,"yscale": "log", "ylabel": r'Relative final abundances',
        "legend": {"fancybox": False, "loc": 'upper right',
                   #"bbox_to_anchor":(1.0, 0.0),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon":False},
        "title": r"Long-lived remnants",
        "text":{},#$ {"x":0.7, "y":0.95, "s":"Dyn. Ej.", "ha":"center", "va":"top",  "fontsize":11, "color":"black",  "transform":None},
        "figname": __outplotdir__ + "cc_nucleo_EOS_total.png",
        "savepdf":True,
        "fontsize":14
    }

    plot_yields_colocoded2(task, plot_dic)

""" ==================== PAPER ===================== """

if __name__ == "__main__":
    ''' --- separate models --- '''
    task_plot_yields_2()

    ''' --- colorcoded q --- '''

    # task_plot_colocoded_yileds3()
    # task_plot_colocoded_yileds3()

''' --- iteration 2 --- '''

if __name__ == "__main__":

    ''' --- separate models --- '''

    #task_plot_yields_2()

    ''' --- colorcoded q --- '''

    #task_plot_colocoded_yileds3()

    #exit(1)

''' --- iteration 1 --- '''

if __name__ == "__main__":

    """ -- test --- """
    # test_solar_r()

    """ --- --- """
    # task_plot_yields()
    # task_plot_yields_one_modle()

    """ --- colorcoded --- """
    # task_plot_colocoded_yileds()
    #task_plot_colocoded_yileds2()