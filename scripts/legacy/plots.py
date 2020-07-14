#
#
#
#
#

from __future__ import division

import numpy as np
import pandas as pd
import sys
import os
import copy
import h5py
import csv

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')



from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import AutoMinorLocator, FixedLocator, NullFormatter, \
    MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

import _models_old as md
import models_radice as rd
import scipy.optimize as opt # opt.curve_fit()
import statsmodels.api as sm

from uutils import x_y_z_sort

__outplotdir__ = "/data01/numrel/vsevolod.nedora/figs/all3/"

# opt.curve_fit()

def mscatter(x, y, ax=None, m=None, **kw):

    for ix, iy, im, c, s in zip(x, y, m, kw["c"], kw["s"]):
        if np.isnan(c):
            # print(ix, iy, im, s)
            ax.plot(ix, iy, color="gray", marker=im, markersize=s/5, alpha=0.6)

    import matplotlib.markers as mmarkers
    if not ax: ax = plt.gca()
    # current_cmap =.get_cmap()
    # current_cmap.set_bad(color='red')
    # cmap = kw["cmap"]
    # cmap.set_bad(color='red')
    # cmap.set_over('gray')
    # cmap.set_invalid('gray')
    # kw['cmap'] = cmap
    sc = ax.scatter(x, y, **kw)
    # sc.cmap.set_bad(color="red")
    # sc.cmap.set_over(color="red")
    # sc.cmap.set_under(color="red")
    # cmap.set_over('gray')
    # sc.set_clim(0, 1)
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    # current_cmap = matplotlib.cm.get_cmap()
    # current_cmap.set_bad(color='red')



    return sc

""" ----------------------------------| EJECTA |---------------------------- """

def __apply_mod(v_n, val, mod):
    # print(v_n, val, mod)
    if mod != None and mod != "":
        # exit(1)
        if mod[0] == '*':
            mult_by = float(mod.split("*")[-1])
            val = val * mult_by
            # print(val); exit(1)
        elif mod[0] == "/":
            dev_by = float(mod.split('/')[-1])
            val = val / dev_by
        else:
            raise NameError("v_n:{} mod:{} is invalid. Use mod '*float' or '/float' "
                            .format(v_n, mod))

    return val


def make_plot_name(v_n_x, v_n_y, v_n_col):
    figname = ''
    figname = figname + v_n_x + '_'
    figname = figname + v_n_y + '_'
    figname = figname + v_n_col + '_'

    figname = figname + '.png'
    return figname

def plot_summary_and_fit_unique():

    load_davids_data = True
    t_pc = 1.5 * 1.e-3
    x_dic = {"v_n": "Lambda",  "err": None, "mod": None, "deferr": None}
    y_dic = {"v_n": "Mdisk3D", "err": "ud", "mod": None, "deferr": 0.2}
    col_dic = {"v_n": "q",     "err": None, "mod": None, "deferr": None}
    #
    mc_bh = "o"
    mc_st = "d"
    mc_pc = "s"
    #
    plot_dic = {#"vmin": 350, "vmax": 900.0,
                "vmin": 1., "vmax": 2.0,
                # "cmap": "jet",
                "cmap": "tab10",
                "label": None, "alpha": 0.6,
                "ms": 30.,
                # "xscale":"log", "yscale":"log",
                # "xmin":1e-5, "xmax":1e-1,
                # "ymin":-0.1, "ymax":0.4,
                # "ymin": -2, "ymax": 2,
                # "ylabel":r"$\Delta M_{\rm disk} / M_{\rm disk}$"
                }
    #
    lk_edge_color = "black"
    #
    figname = make_plot_name(x_dic["v_n"], y_dic["v_n"], col_dic["v_n"], load_davids_data)

    # PLOT
    fig = plt.figure(figsize=[4, 2.5])  # <-> |
    ax = fig.add_subplot(111)
    #
    ax.scatter([-100], [-100], marker=mc_pc, s=plot_dic['ms'],
               color="gray", alpha=1., label="Prompt Collapse")
    ax.scatter([-100], [-100], marker=mc_st, s=plot_dic['ms'],
               color="gray", alpha=1., label="Stable remnant")
    ax.scatter([-100], [-100], marker=mc_bh, s=plot_dic['ms'],
               color="gray", alpha=1., label="Black Hole")
    ax.scatter([-100], [-100], marker=mc_bh, s=plot_dic['ms'],
               color="white", alpha=1., edgecolor=lk_edge_color, label="Viscosity")
    if load_davids_data:
        ax.scatter([-100], [-100], marker="*", s=plot_dic['ms'],
                   color="gray", alpha=1., edgecolor=None, label="Radice+2018")
    # # #
    if load_davids_data:
        #
        v_n_x = rd.translation[x_dic["v_n"]]
        v_n_y = rd.translation[y_dic["v_n"]]
        v_n_col = rd.translation[col_dic["v_n"]]
        #
        from models_radice import simulations, fiducial
        plot_sims = simulations[fiducial]
        eoss = sorted(list(set(plot_sims.EOS)))
        for ieos, eos in enumerate(eoss):
            qs = sorted(list(set(plot_sims.q)))
            for iq, q in enumerate(qs):
                sel = plot_sims[(plot_sims.EOS == eos) & (plot_sims.q == q)]
                cm = plt.cm.get_cmap(plot_dic["cmap"])
                norm = Normalize(vmin=plot_dic["vmin"], vmax=plot_dic["vmax"])
                #
                edgecolors = None
                #
                markers = ["*" for sim in sel.index]
                mss = [20 for sim in sel.index]
                #
                # print(fitting_func_of_lam(x_davids_fit, ))
                # print(sel.loc["BHBlp_M135135_LK"]["Lambda"]);exit(1)
                x = sel[v_n_x]
                y = sel[v_n_y]# - fitting_func_of_lam(x_davids_fit, sel["Lambda"])) / sel[v_n_y]
                #
                ax.errorbar(x, y, yerr=rd.params.MdiskPP_err(sel[v_n_y]), label=None,
                            color='gray', ecolor='gray',
                            fmt='None', elinewidth=1, capsize=1, alpha=0.5)
                #
                # ax.scatter(sel[x_dic["v_n"]], sel[y_dic["v_n"]], marker=markers, markersize=mss)
                #
                sc = mscatter(x, y, ax=ax, c=sel[v_n_col], norm=norm,
                              s=mss, cmap=cm, m=markers,
                              label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)
    # # #
    cm = plt.cm.get_cmap(plot_dic["cmap"])
    norm = Normalize(vmin=plot_dic["vmin"], vmax=plot_dic["vmax"])
    #
    from groups import groups
    sc = None
    plot_sims = groups
        #md.convert_models_to_uniquemodels_table()
    eoss = sorted(list(set(plot_sims.EOS)))
    #
    for ieos, eos in enumerate(eoss):
        print("{}".format(eos))
        sel = plot_sims[(plot_sims.EOS == eos) & (np.isfinite(plot_sims[y_dic["v_n"]]))]
        #
        print(sel["Mdisk3D"])
        #
        edgecolors = []
        for _, m in sel.iterrows():
            if m["viscosity"] == "LK":
                edgecolors.append("black")
            else:
                edgecolors.append("None")
        #
        markers = []
        mss = []
        for _, m in sel.iterrows():
            if np.isfinite(m.tcoll_gw):
                marker = mc_bh
                ms = plot_dic['ms']
                if m.tcoll_gw < t_pc:
                    marker = mc_pc
                    ms = 2 * plot_dic['ms']
            else:
                marker = mc_st
                ms = 1.5 * plot_dic['ms']
            markers.append(marker)
            mss.append(ms)
        #
        x = sel[x_dic["v_n"]]
        y = sel[y_dic["v_n"]]
        #
        ax.errorbar(x, y, yerr=sel["err-" + y_dic["v_n"]], label=None,
                    color='gray', ecolor='gray',
                    fmt='None', elinewidth=1, capsize=1, alpha=0.5)
        #
        sc = mscatter(x, y, ax=ax, c=sel[col_dic["v_n"]], norm=norm,
                      s=mss, cmap=cm, m=markers,
                      label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)

    # for ieos, eos in enumerate(eoss):
    #     print("{}".format(eos))
    #     qs = plot_sims[plot_sims["EOS"] == eos]["q"]
    #     for iq, q in enumerate(qs):
    #         print("\tq:{}".format(q))
    #         sel = plot_sims[(plot_sims.EOS == eos) & (plot_sims.q == q) & (np.isfinite(plot_sims[y_dic["v_n"]]))]
    #         cm = plt.cm.get_cmap(plot_dic["cmap"])
    #         norm = Normalize(vmin=plot_dic["vmin"], vmax=plot_dic["vmax"])
    #         #
    #         for name, sim in sel.iterrows():
    #             print("\t\t{} {}".format(name, sim["Mdisk3D"]))
    #         #
    #         edgecolors = []
    #         for _, m in sel.iterrows():
    #             if m["viscosity"] == "LK":
    #                 edgecolors.append("black")
    #             else:
    #                 edgecolors.append("None")
    #         #
    #         markers = []
    #         mss = []
    #         for _, m in sel.iterrows():
    #             if np.isfinite(m.tcoll_gw):
    #                 marker = mc_bh
    #                 ms = plot_dic['ms']
    #                 if m.tcoll_gw < t_pc:
    #                     marker = mc_pc
    #                     ms = 2 * plot_dic['ms']
    #             else:
    #                 marker = mc_st
    #                 ms = 1.5 * plot_dic['ms']
    #             markers.append(marker)
    #             mss.append(ms)
    #         #
    #         x = sel[x_dic["v_n"]]
    #         y = sel[y_dic["v_n"]]# - fitting_func_of_lam(x_davids_fit, sel["Lambda"])) / sel[y_dic["v_n"]]
    #
    #         ax.errorbar(x, y, yerr=sel["err-" + y_dic["v_n"]], label=None,
    #                     color='gray', ecolor='gray',
    #                     fmt='None', elinewidth=1, capsize=1, alpha=0.5)
    #         #
    #         sc = mscatter(x, y, ax=ax, c=sel[col_dic["v_n"]], norm=norm,
    #                       s=mss, cmap=cm, m=markers,
    #                       label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)
    #
    ax.tick_params(axis='both', which='both', labelleft=True,
                   labelright=False, tick1On=True, tick2On=True,
                   labelsize=12,
                   direction='in',
                   bottom=True, top=True, left=True, right=True)
    ax.minorticks_on()

    # limits
    if ("xmin" in plot_dic.keys() and "xmax" in plot_dic.keys()) and \
            (plot_dic["xmin"] != None and plot_dic["xmax"] != None):
        min_, max_ = plot_dic["xmin"], plot_dic["xmax"]
    else:
        min_, max_ = md.get_minmax(x_dic["v_n"], [], extra=2.)
    ax.set_xlim(min_, max_)
    if ("ymin" in plot_dic.keys() and "ymax" in plot_dic.keys()) and \
            (plot_dic["ymin"] != None and plot_dic["ymax"] != None):
        min_, max_ = plot_dic["ymin"], plot_dic["ymax"]
    else:
        min_, max_ = md.get_minmax(y_dic["v_n"], [], extra=2., oldtable=load_davids_data)
    ax.set_ylim(min_, max_)

    # scale
    if "xscale" in plot_dic.keys() and plot_dic["xscale"] != None:
        ax.set_xscale(plot_dic["xscale"])
    if "yscale" in plot_dic.keys() and plot_dic["yscale"] != None:
        ax.set_yscale(plot_dic["yscale"])

    # label
    if "ylabel" in plot_dic.keys() and plot_dic["ylabel"] != None:
        ax.set_ylabel(plot_dic["ylabel"])
    else:
        ax.set_ylabel(md.get_label(y_dic["v_n"]))
    if "xlabel" in plot_dic.keys() and plot_dic["xlabel"] != None:
        ax.set_xlabel(plot_dic["xlabel"])
    else:
        ax.set_xlabel(md.get_label(x_dic["v_n"]))

    #
    # ax.axhline(y=0, linestyle='-', linewidth=0.4, color='black')

    # colobar
    ax.legend(fancybox=True, loc='upper center',  # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
              shadow=False, ncol=2, fontsize=9,
              framealpha=0., borderaxespad=0.)
    clb = plt.colorbar(sc)
    clb.ax.set_title(md.get_label(col_dic["v_n"]), fontsize=11)
    clb.ax.tick_params(labelsize=11)
    plt.subplots_adjust(bottom=0.20, right=1.0, left=0.15)
    # plt.tight_layout()
    print(__outplotdir__ + figname)
    plt.savefig(__outplotdir__ + figname, dpi=256)
    plt.close()

def plot_summary_and_fit_unique_old_():

    load_davids_data = True
    t_pc = 1.5 * 1.e-3
    x_dic =   {"v_n": "Lambda",          "err": None, "mod": None,   "deferr": None}
    y_dic =   {"v_n": "Mdisk3D",    "err": "ud", "mod": None, "deferr": 0.2}
    # x_dic =   {"v_n": "Lambda",      "err": None, "mod": None,   "deferr": None}
    # y_dic =   {"v_n": "Mdisk3Dmax", "err": "ud", "mod": None, "deferr": 0.2}
    # y_dic =   {"v_n": "Mej_tot-geo", "err": "ud", "mod": "*1e2", "deferr": 0.2}
    # y_dic =   {"v_n": "vel_inf_ave-geo", "err": "ud", "mod": None, "deferr": 0.2}
    # x_dic =   {"v_n": "Mej_tot-geo_entropy_below_10", "err": "ud", "mod": None, "deferr": 0.2}
    # y_dic =   {"v_n": "Mej_tot-geo_entropy_above_10", "err": "ud", "mod": None, "deferr": 0.2}
    # x_dic =   {"v_n": "theta_rms-geo", "err": "ud", "mod": None, "deferr": 0.2}
    # x_dic =   {"v_n": "Ye_ave-geo", "err": "ud", "mod": None, "deferr": 0.2}
    # x_dic =   {"v_n": "theta_rms-geo", "err": "ud", "mod": None, "deferr": 0.2}
    # y_dic =   {"v_n": "Ye_ave-geo", "err": "ud", "mod": None, "deferr": 0.2}
    # y_dic =   {"v_n": "Mej_tot-geo", "err": "ud", "mod": "*1e2", "deferr": 0.2}
    # col_dic = {"v_n": "q",           "err": None, "mod": None,   "deferr": None}
    col_dic = {"v_n": "q",           "err": None, "mod": None, "deferr": None}
    #
    mc_bh = "o"
    mc_st = "d"
    mc_pc = "s"
    #
    plot_dic = {#"vmin": 350, "vmax": 900.0,
                "vmin": 1., "vmax": 2.0,
                "cmap": "tab10", # "jet",
                "label": None, "alpha": 0.8,
                "ms": 30.,
                # "xscale":"log", "yscale":"log",
                # "xmin":1e-5, "xmax":1e-1,
                # "ymin":1e-5, "ymax":1e-1
                }
    #
    lk_edge_color="black"
    plot_sims = md.unique_simulations
    unique = sorted(list(set(plot_sims["group"])))
    #
    figname = make_plot_name(x_dic["v_n"], y_dic["v_n"], col_dic["v_n"], load_davids_data)
    #
    # collect data for plotting
    x_arr = np.zeros(3)
    y_arr = np.zeros(3)
    col_arr = []
    marker_arr = []
    edgecolors = []
    #


    # for u in unique:
    #     group = plot_sims[plot_sims.group == u]
    #     x, del1, del2 = md.get_group_value3(group, x_dic)
    #     x_arr = np.vstack((x_arr, [x, del1, del2]))
    #     y, del1, del2 = md.get_group_value3(group, y_dic)
    #     y_arr = np.vstack((y_arr, [y, del1, del2]))
    #     col, _, _ = md.get_group_value3(group, col_dic)
    #     col_arr.append(col)
    #     # print(u)
    #     tcolls = list(group["tcoll_gw"])
    #     marker = mc_st
    #     for t_ in tcolls:
    #         if np.isfinite(t_):
    #             marker = mc_bh
    #             if t_ < t_pc:
    #                 marker = mc_pc
    #     # mk = list(group[marker_v_n])[0]
    #     marker_arr.append(marker)
    #     print("u:{} \t x:{:.1f} y:{:.2f} col:{:.1f} mk:{}"
    #           .format(u, x, y, col, marker))
    #     if list(group.viscosity)[0] == "LK":
    #         edgecolors.append(lk_edge_color)
    #     else:
    #         edgecolors.append("None")
    #
    # col_arr = np.array(col_arr)
    # x_arr = np.delete(x_arr, 0, 0)
    # y_arr = np.delete(y_arr, 0, 0)

    if load_davids_data:
        from models_radice import simulations, fiducial
        plot_sims = simulations[fiducial]
        # print(plot_sims.keys())
        #
        x_arr_david = plot_sims[rd.translation[x_dic["v_n"]]]
        y_arr_david = plot_sims[rd.translation[y_dic["v_n"]]]
        x_arr_david = md.__apply_mod(x_dic["v_n"], x_arr_david, x_dic["mod"])
        y_arr_david = md.__apply_mod(y_dic["v_n"], y_arr_david, y_dic["mod"])
        assert len(x_arr_david) == len(y_arr_david)
    else:
        x_arr_david = []
        y_arr_david = []

    #
    fig = plt.figure(figsize=[4, 2.5])  # <-> |
    ax = fig.add_subplot(111)

    # for labels
    ax.scatter([-100], [-100], marker=mc_pc,s=plot_dic['ms'],
               color="gray", alpha=1., label="Prompt Collapse")
    ax.scatter([-100], [-100], marker=mc_st, s=plot_dic['ms'],
               color="gray", alpha=1., label="Stable remnant")
    ax.scatter([-100], [-100], marker=mc_bh, s=plot_dic['ms'],
               color="gray", alpha=1., label="Black Hole")
    ax.scatter([-100], [-100], marker=mc_bh, s=plot_dic['ms'],
               color="white", alpha=1., edgecolor=lk_edge_color, label="Viscosity")

    # main body
    if load_davids_data:
        ax.scatter(x_arr_david, y_arr_david, marker="3", s=20,
                   color="gray", alpha=1., label="Radice+2018")

    # cm = plt.cm.get_cmap(plot_dic["cmap"])
    # norm = Normalize(vmin=plot_dic["vmin"], vmax=plot_dic["vmax"])
    # sc = mscatter(x_arr[:, 0], y_arr[:, 0], ax=ax, c=col_arr, norm=norm,
    #               s=plot_dic['ms'], cmap=cm, m=marker_arr,
    #               label=plot_dic['label'], alpha=plot_dic['alpha'], edgecolor=edgecolors)

    plot_sims = md.convert_models_to_uniquemodels_table()
    eoss = sorted(list(set(plot_sims.EOS)))
    for ieos, eos in enumerate(eoss):
        qs = sorted(list(set(plot_sims[plot_sims["EOS"] == eos].q)))
        for iq, q in enumerate(qs):
            sel = plot_sims[(plot_sims.EOS == eos) &
                            (np.round(plot_sims.q, decimals=2) == np.round(q,decimals=2)) &
                            (np.isfinite(plot_sims[y_dic["v_n"]]))]

            cm = plt.cm.get_cmap(plot_dic["cmap"])
            norm = Normalize(vmin=plot_dic["vmin"], vmax=plot_dic["vmax"])
            #
            edgecolors = []
            for _, m in sel.iterrows():
                if m["viscosity"] == "LK":
                    edgecolors.append("black")
                else:
                    edgecolors.append("None")
            #
            markers = []
            mss = []
            for _, m in sel.iterrows():
                if np.isfinite(m.tcoll_gw):
                    marker = mc_bh
                    ms = plot_dic['ms']
                    if m.tcoll_gw < t_pc:
                        marker = mc_pc
                        ms = 2 * plot_dic['ms']
                else:
                    marker = mc_st
                    ms = 1.5 * plot_dic['ms']
                markers.append(marker)
                mss.append(ms)
            #
            print(sel[y_dic["v_n"]])

            ax.errorbar(sel[x_dic["v_n"]], sel[y_dic["v_n"]], yerr=sel["err-" + y_dic["v_n"]], label=None,
                        color='gray', ecolor='gray',
                        fmt='None', elinewidth=1, capsize=1, alpha=0.5)
            #
            sc = mscatter(sel[x_dic["v_n"]], sel[y_dic["v_n"]], ax=ax, c=sel[col_dic["v_n"]], norm=norm,
                          s=mss, cmap=cm, m=markers,
                          label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)





    # error bars
    # ax.errorbar(x_arr[:, 0], y_arr[:, 0], yerr=y_arr[:, 1], color='gray', ecolor='gray',
    #             fmt='None', elinewidth=1, capsize=1, alpha=0.6)
    # ax.errorbar(x_arr[:, 0], y_arr[:, 0], xerr=x_arr[:, 1], color='gray', ecolor='gray',
    #             fmt='None', elinewidth=1, capsize=1, alpha=0.6)

    # ticks
    ax.tick_params(axis='both', which='both', labelleft=True,
                   labelright=False, tick1On=True, tick2On=True,
                   labelsize=12,
                   direction='in',
                   bottom=True, top=True, left=True, right=True)
    ax.minorticks_on()

    # limits
    if ("xmin" in plot_dic.keys() and "xmax" in plot_dic.keys()) and \
        (plot_dic["xmin"] != None and plot_dic["xmax"] != None):
            min_, max_ = plot_dic["xmin"], plot_dic["xmax"]
    else:
        min_, max_ = md.get_minmax(y_dic["v_n"], y_arr, extra=2.)
    ax.set_ylim(min_, max_)
    if ("ymin" in plot_dic.keys() and "ymax" in plot_dic.keys()) and \
            (plot_dic["ymin"] != None and plot_dic["ymax"] != None):
        min_, max_ = plot_dic["ymin"], plot_dic["ymax"]
    else:
        min_, max_ = md.get_minmax(x_dic["v_n"], x_arr, extra=2., oldtable=load_davids_data)
    ax.set_xlim(min_, max_)

    # scale
    if "xscale" in plot_dic.keys() and plot_dic["xscale"] != None:
        ax.set_xscale(plot_dic["xscale"])
    if "yscale" in plot_dic.keys() and plot_dic["yscale"] != None:
        ax.set_yscale(plot_dic["yscale"])

    # label
    ax.set_xlabel(md.get_label(x_dic["v_n"]))
    ax.set_ylabel(md.get_label(y_dic["v_n"]))

    # colobar
    ax.legend(fancybox=True, loc='upper center', #bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
              shadow=False, ncol=2, fontsize=9,
              framealpha=0., borderaxespad=0.)
    clb = plt.colorbar(sc)
    clb.ax.set_title(md.get_label(col_dic["v_n"]), fontsize=11)
    clb.ax.tick_params(labelsize=11)
    plt.subplots_adjust(bottom=0.20, right=1.0)
    # plt.tight_layout()
    print(__outplotdir__ + figname)
    plt.savefig(__outplotdir__ + figname, dpi=256)
    plt.close()

# Total Ejecta Mass
def fitting_function_mej(x, v):
    a, b, c, d, n = x

    # return (a*(v["M2"]/v["M1"])**(1.0/3.0)*(1. - 2*v["C1"])/(v["C1"]) + b*(v["M2"]/v["M1"])**n +
    #        c*(1 - v["M1"]/v["Mb1"]))*v["Mb1"] + \
    #        (a*(v["M1"]/v["M2"])**(1.0/3.0)*(1. - 2*v["C2"])/(v["C2"]) + b*(v["M1"]/v["M2"])**n +
    #        c*(1 - v["M2"]/v["Mb2"]))*v["Mb2"] + \
    #        d
    return ((a * (v.M2 / v.M1) ** (1.0 / 3.0) * (1. - 2 * v.C1) / (v.C1) + b * (v.M2 / v.M1) ** n +
            c * (1 - v.M1 / v.Mb1)) * v.Mb1 + \
           (a * (v.M1 / v.M2) ** (1.0 / 3.0) * (1. - 2 * v.C2) / (v.C2) + b * (v.M1 / v.M2) ** n +
            c * (1 - v.M2 / v.Mb2)) * v.Mb2 + \
           d) / 1.e3
def residuals(x, data, v_n = "Mej_tot-geo"):
    xi = fitting_function_mej(x, data)
    return 1e-3*(xi - 1e3*data[v_n])#/(models.params.Mej_err(data.Mej))
def dietrich_mej_coeffs():
    a = -1.35695
    b = 6.11252
    c = -49.43355
    d = 16.1144
    n = -2.5484
    return np.array((a,b,c,d,n))
def radice():
    a = -0.657
    b = 4.254
    c = -32.61
    d =  5.205
    n = -0.773
    return np.array((a,b,c,d,n))
def fit_ejecta_mass_new_data():
    #
    sims = md.convert_models_to_uniquemodels_table()
    # sims = sims[sims.viscosity=="nan"]
    # sims = rd.simulations
    #
    # print(sims.tcoll_gw)
    #
    x0 = dietrich_mej_coeffs()
    print("chi2 original: " + str(np.sum(residuals(x0, sims) ** 2)))
    sims["Mej_fit_tim"] = fitting_function_mej(x0, sims)

    res = opt.least_squares(residuals, dietrich_mej_coeffs(), args=(sims,)) # [rd.fiducial
    print("chi2 fit: " + str(np.sum(residuals(res.x, sims) ** 2)))
    print("Fit coefficients:")
    print("  a = {}".format(res.x[0]))
    print("  b = {}".format(res.x[1]))
    print("  c = {}".format(res.x[2]))
    print("  d = {}".format(res.x[3]))
    print("  n = {}".format(res.x[4]))
    sims["Mej_fit"] = fitting_function_mej(res.x, sims)
    #
    return sims
def plot_ejecta_mass_fit(sims):
    """
    :param sims: pandas.datafram with "Mej_fit" fitting functions
    :return:
    """

    v_n = "Mej_tot-geo"
    v_n_err = "err-" + v_n
    mc_bh = "o"
    mc_st = "d"
    mc_pc = "s"
    t_pc = 1.5 * 1.e-3
    figname = "summary_ejecta_mass_fit.png"

    plot_dic = {"vmin": 1., "vmax": 2.0,
                "cmap": "tab10", "alpha": 1.,
                "ms": 30.}

    fig = plt.figure(figsize=[4, 2.5])  # <-> |
    ax = fig.add_subplot(111)
    #
    plt.subplots_adjust(bottom=0.20, right=1.0)
    #
    eoss = sorted(list(set(sims.EOS)))
    for ieos, eos in enumerate(eoss):
        qs = sorted(list(set(sims.q)))
        for iq, q in enumerate(qs):
            sel = sims[(sims.EOS == eos) & (sims.q == q)]
            cm = plt.cm.get_cmap(plot_dic["cmap"])
            norm = Normalize(vmin=plot_dic["vmin"], vmax=plot_dic["vmax"])
            #
            edgecolors = []
            for _, m in sel.iterrows():
                if m["viscosity"] == "LK":
                    edgecolors.append("black")
                else:
                    edgecolors.append("None")
            #
            markers = []
            for _, m in sel.iterrows():
                if np.isfinite(m.tcoll_gw):
                    marker = mc_bh
                    if m.tcoll_gw < t_pc:
                        marker = mc_pc
                else: marker = mc_st
                markers.append(marker)
            ax.errorbar(sel["Mej_fit"], 1e3 * sel[v_n], yerr=1e3 * sel[v_n_err], label=None,
                        color='gray', ecolor='gray',
                        fmt='None', elinewidth=1, capsize=1, alpha=0.6)
            #
            sc = mscatter(sel["Mej_fit"], 1e3 * sel[v_n], ax=ax, c=sel["q"], norm=norm,
                          s=plot_dic['ms'], cmap=cm, m=markers,
                          label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)
            #

    # ax.scatter([-100], [y_arr[0, 0]], marker=mc_pc, s=plot_dic['ms'],
    #            color="gray", alpha=1., label="Prompt Collapse")

    # plot fit
    Mej_fit = np.linspace(0, 15)
    results = sm.OLS(sims[v_n], sims.Mej_fit).fit()
    fit = 1e3*results.predict(Mej_fit)
    ax.plot(Mej_fit, fit, 'k--', label="Fit")

    # ticks
    ax.tick_params(axis='both', which='both', labelleft=True,
                   labelright=False, tick1On=True, tick2On=True,
                   labelsize=12,
                   direction='in',
                   bottom=True, top=True, left=True, right=True)
    ax.minorticks_on()
    # plt.tight_layout()
    # colobar
    clb = plt.colorbar(sc)
    clb.ax.set_title(r"$M_1/M_2$", fontsize=11)
    clb.ax.tick_params(labelsize=11)

    ax.set_xlim(0., 11.)
    ax.set_ylim(0., 15.)

    ax.set_xlabel(r"$M_{\rm ej; fit}\ [10^{-3}\ M_\odot]$")
    ax.set_ylabel(r"$M_{\rm ej}\ [10^{-3}\ M_\odot]$")

    # ax.set_xlim(0, 1.5)
    # ax.set_ylim(0, 4)

    ax.legend()

    print(__outplotdir__ + figname)
    plt.savefig(__outplotdir__ + figname, dpi=256)

# Ejecta Velocity Fit
def fitting_function_vinf(x, v):
    a, b, c = x
    return a*(v.M1/v.M2)*(1. + c*v.C1) + \
           a*(v.M2/v.M1)*(1. + c*v.C2) + b
def residuals2(x, data, v_n = "vel_inf_ave-geo"):
    xi = fitting_function_vinf(x, data)
    return (xi - data[v_n])
def dietrich_vinf_coeffs():
    a = -0.219479
    b = 0.444836
    c = -2.67385
    return np.array((a,b,c))
def fit_velocity_new_data():

    sims = md.convert_models_to_uniquemodels_table()

    x0 = dietrich_vinf_coeffs()
    print("chi2 original: " + str(np.sum(residuals2(x0, sims) ** 2)))
    sims["vej_fit_tim"] = fitting_function_vinf(x0, sims)

    res = opt.least_squares(residuals2, dietrich_vinf_coeffs(), args=(sims,))
    print("chi2 fit: " + str(np.sum(residuals2(res.x, sims) ** 2)))
    print("Fit coefficients:")
    print("  a = {}".format(res.x[0]))
    print("  b = {}".format(res.x[1]))
    print("  c = {}".format(res.x[2]))
    sims["vej_fit"] = fitting_function_vinf(res.x, sims)

    return sims
def plot_velocity_fit(sims):
    """
    :param sims: pandas.datafram with "Mej_fit" fitting functions
    :return:
    """

    v_n = "vel_inf_ave-geo"
    v_n_err = "err-" + v_n
    mc_bh = "o"
    mc_st = "d"
    mc_pc = "s"
    t_pc = 1.5 * 1.e-3
    figname = "summary_velocity_fit.png"

    plot_dic = {"vmin": 1., "vmax": 2.0,
                "cmap": "tab10", "alpha": 1.,
                "ms": 30.}

    fig = plt.figure(figsize=[4, 2.5])  # <-> |
    ax = fig.add_subplot(111)
    #
    plt.subplots_adjust(bottom=0.20, right=1.0)
    #
    eoss = sorted(list(set(sims.EOS)))
    for ieos, eos in enumerate(eoss):
        qs = sorted(list(set(sims.q)))
        for iq, q in enumerate(qs):
            sel = sims[(sims.EOS == eos) & (sims.q == q)]
            cm = plt.cm.get_cmap(plot_dic["cmap"])
            norm = Normalize(vmin=plot_dic["vmin"], vmax=plot_dic["vmax"])
            #
            edgecolors = []
            for _, m in sel.iterrows():
                if m["viscosity"] == "LK":
                    edgecolors.append("black")
                else:
                    edgecolors.append("None")
            #
            markers = []
            for _, m in sel.iterrows():
                if np.isfinite(m.tcoll_gw):
                    marker = mc_bh
                    if m.tcoll_gw < t_pc:
                        marker = mc_pc
                else: marker = mc_st
                markers.append(marker)
            ax.errorbar(sel["vej_fit"], sel[v_n], yerr=sel[v_n_err], label=None,
                        color='gray', ecolor='gray',
                        fmt='None', elinewidth=1, capsize=1, alpha=0.6)
            #
            sc = mscatter(sel["vej_fit"], sel[v_n], ax=ax, c=sel["q"], norm=norm,
                          s=plot_dic['ms'], cmap=cm, m=markers,
                          label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)
            #

    # ax.scatter([-100], [y_arr[0, 0]], marker=mc_pc, s=plot_dic['ms'],
    #            color="gray", alpha=1., label="Prompt Collapse")

    # plot fit
    vej_fit = np.linspace(0, 1.)
    results = sm.OLS(sims[v_n], sims.vej_fit).fit()
    fit = results.predict(vej_fit)
    ax.plot(vej_fit, fit, 'k--', label="Fit")

    # ticks
    ax.tick_params(axis='both', which='both', labelleft=True,
                   labelright=False, tick1On=True, tick2On=True,
                   labelsize=12,
                   direction='in',
                   bottom=True, top=True, left=True, right=True)
    ax.minorticks_on()
    # plt.tight_layout()
    # colobar
    clb = plt.colorbar(sc)
    clb.ax.set_title(r"$M_1/M_2$", fontsize=11)
    clb.ax.tick_params(labelsize=11)

    ax.set_xlim(0., 0.4)
    ax.set_ylim(0., 0.4)

    ax.set_xlabel(r"$v_{\rm ej; fit}\ [c]$")
    ax.set_ylabel(r"$v_{\rm ej}\ [c]$")

    # ax.set_xlim(0, 1.5)
    # ax.set_ylim(0, 4)

    ax.legend()

    print(__outplotdir__ + figname)
    plt.savefig(__outplotdir__ + figname, dpi=256)



# Theta_rms Fit [Failed]
def fitting_function3(x, Lambda):
    a, b, c, d = x
    return a*np.tanh((Lambda - b)/c) + d
def residuals3(x, Lambda, thej):
    xi = fitting_function3(x, Lambda)
    return (xi - thej)
def initial_guess3():
    a = 10.
    b = 50.
    c = 10.
    d = 20.
    return np.array((a,b,c,d))
def fit_thetarms_new_data():

    sims = md.convert_models_to_uniquemodels_table()

    v_n1 = "k2T"
    v_n2 = "theta_rms-geo"

    res = opt.least_squares(residuals3, initial_guess3(), args=(sims[v_n1], sims[v_n2]))
    print("chi2 fit: " + str(np.sum(residuals3(res.x, sims[v_n1], sims[v_n2]) ** 2)))
    print("Fit coefficients:")
    print("  a = {}".format(res.x[0]))
    print("  b = {}".format(res.x[1]))
    print("  c = {}".format(res.x[2]))
    print("  d = {}".format(res.x[3]))
    sims["theta_ej_fit"] = fitting_function3(res.x, sims[v_n1])

    return sims
def plot_thetarms_fit(sims):

    v_n = "theta_rms-geo"
    v_nfit = "theta_ej_fit"
    v_n_err = "err-" + v_n
    mc_bh = "o"
    mc_st = "d"
    mc_pc = "s"
    t_pc = 1.5 * 1.e-3
    figname = "summary_thetarms.png"

    plot_dic = {"vmin": 1., "vmax": 2.0,
                "cmap": "tab10", "alpha": 1.,
                "ms": 30.}

    fig = plt.figure(figsize=[4, 2.5])  # <-> |
    ax = fig.add_subplot(111)
    #
    plt.subplots_adjust(bottom=0.20, right=1.0)
    #
    eoss = sorted(list(set(sims.EOS)))
    for ieos, eos in enumerate(eoss):
        qs = sorted(list(set(sims.q)))
        for iq, q in enumerate(qs):
            sel = sims[(sims.EOS == eos) & (sims.q == q)]
            cm = plt.cm.get_cmap(plot_dic["cmap"])
            norm = Normalize(vmin=plot_dic["vmin"], vmax=plot_dic["vmax"])
            #
            edgecolors = []
            for _, m in sel.iterrows():
                if m["viscosity"] == "LK":
                    edgecolors.append("black")
                else:
                    edgecolors.append("None")
            #
            markers = []
            for _, m in sel.iterrows():
                if np.isfinite(m.tcoll_gw):
                    marker = mc_bh
                    if m.tcoll_gw < t_pc:
                        marker = mc_pc
                else:
                    marker = mc_st
                markers.append(marker)
            ax.errorbar(sel[v_nfit], sel[v_n], yerr=sel[v_n_err], label=None,
                        color='gray', ecolor='gray',
                        fmt='None', elinewidth=1, capsize=1, alpha=0.6)
            #
            sc = mscatter(sel[v_nfit], sel[v_n], ax=ax, c=sel["q"], norm=norm,
                          s=plot_dic['ms'], cmap=cm, m=markers,
                          label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)
            #

    # ax.scatter([-100], [y_arr[0, 0]], marker=mc_pc, s=plot_dic['ms'],
    #            color="gray", alpha=1., label="Prompt Collapse")

    # plot fit
    theta_fit = np.linspace(0, 50.)
    results = sm.OLS(sims[v_n], sims[v_nfit]).fit()
    fit = results.predict(theta_fit)
    ax.plot(theta_fit, fit, 'k--', label="Fit")

    # ticks
    ax.tick_params(axis='both', which='both', labelleft=True,
                   labelright=False, tick1On=True, tick2On=True,
                   labelsize=12,
                   direction='in',
                   bottom=True, top=True, left=True, right=True)
    ax.minorticks_on()
    # plt.tight_layout()
    # colobar
    clb = plt.colorbar(sc)
    clb.ax.set_title(r"$M_1/M_2$", fontsize=11)
    clb.ax.tick_params(labelsize=11)

    # ax.set_xlim(0., 0.4)
    # ax.set_ylim(0., 0.4)

    ax.set_xlabel(r"$\theta_{\rm ej}$")
    ax.set_ylabel(r"$\theta_{\rm {ej; fit}}$")

    # ax.set_xlim(0, 1.5)
    # ax.set_ylim(0, 4)

    ax.legend()

    print(__outplotdir__ + figname)
    plt.savefig(__outplotdir__ + figname, dpi=256)



# Mdisk fit
# def fitting_function4(x, lam):
#     a, b, c, d = x
#     return np.maximum(a + b*(np.tanh((lam - c)/d)), 1e-3)
def fitting_function4(x, v):
    a, b, c, d = x
    return np.maximum(a + b*(np.tanh((v["Lambda"] - c)/d)), 1e-3)
def residuals4(x, lam, Mdisk):
    xi = fitting_function4(x, lam)
    return (xi - Mdisk)
def initial_guess4():
    a = 0.084
    b = 0.127
    c = 567.1
    d = 405.14
    e = 1e-3
    return np.array((a,b,c,d))
def fit_mdisk_new_data():

    sel = md.convert_models_to_uniquemodels_table()

    v_n1 = "Lambda"
    v_n2 = "Mdisk3D"

    sel = sel[np.isfinite(sel[v_n2])]

    print(" David Data ")
    res = opt.least_squares(residuals4, initial_guess4(), args=(sel[v_n1], sel[v_n2]))
    print("chi2 fit: " + str(np.sum(residuals4(res.x, sel[v_n1], sel[v_n2])**2)))
    print("Fit coefficients:")
    print("    a = {}".format(res.x[0]))
    print("    b = {}".format(res.x[1]))
    print("    c = {}".format(res.x[2]))
    print("    d = {}".format(res.x[3]))

    lam = np.linspace(1, 1500, 100)
    fit_nl = fitting_function4(initial_guess4(), lam)
    # sel["mdisk_fit"] = fit_nl
    return lam, fit_nl

def fitting_function5(x, lam):
    b, c, d = x
    return np.maximum(b * (1 + np.tanh((lam - c)/d)), 1e-3)
def residuals5(x, lam, Mdisk):
    xi = fitting_function5(x, lam)
    return (xi - Mdisk)
def initial_guess5():
    # a = 0.084
    b = 0.127
    c = 567.1
    d = 405.14
    e = 1e-3
    return np.array((b,c,d))
def fit_mdisk_old_new_data():

    sel = md.convert_models_to_uniquemodels_table()
    #
    v_n1 = "Lambda"
    v_n2 = "Mdisk3D"
    #
    sel = sel[np.isfinite(sel[v_n2])]
    #
    x_arr = np.array(sel[v_n1])
    y_arr = np.array(sel[v_n2])
    #
    table = rd.papertable
    table = table[np.isfinite(table["M_disk"])]
    from models_radice import simulations, fiducial
    plot_sims = simulations[fiducial]
    x_arr_david = np.array([float(plot_sims.loc[sim]["Lambda"]) for sim in plot_sims.index if sim in table.index])
    y_arr_david = np.array([float(table.loc[sim]["M_disk"])/1.e2 for sim in plot_sims.index if sim in table.index])
    #
    assert len(x_arr_david) > 0
    assert len(x_arr_david) == len(y_arr_david)
    #
    x_arr = np.append(x_arr, x_arr_david)
    y_arr = np.append(y_arr, y_arr_david)

    x_arr, y_arr = x_y_z_sort(x_arr, y_arr)

    print(x_arr)
    print(y_arr)

    print(" David + My data")
    res = opt.least_squares(residuals5, initial_guess5(), args=(x_arr, y_arr))
    print("chi2 fit: " + str(np.sum(residuals5(res.x, x_arr, y_arr)**2)))
    print("Fit coefficients:")
    print("    a = {}".format(res.x[0]))
    print("    b = {}".format(res.x[1]))
    print("    c = {}".format(res.x[2]))
    # print("    d = {}".format(res.x[3]))

    lam = np.linspace(1, 1500, 100)
    fit_nl = fitting_function5(res.x, lam)
    # sel["mdisk_fit"] = fit_nl
    return lam, fit_nl
# plot disk mass with old data
def print_disk_mass():

    sel = md.convert_models_to_uniquemodels_table()
    sims = sel[np.isfinite(sel["Mdisk3D"])]
    #
    load_davids_data = True
    t_pc = 1.5 * 1.e-3
    x_dic = {"v_n": "Lambda", "err": None, "mod": None, "deferr": None}
    y_dic = {"v_n": "Mdisk3D", "err": "ud", "mod": None, "deferr": 0.2}
    col_dic = {"v_n": "q", "err": None, "mod": None, "deferr": None}
    #
    mc_bh = "o"
    mc_st = "d"
    mc_pc = "s"
    #
    plot_dic = {"vmin": 1., "vmax": 2.0,
                "cmap": "tab10", "label": None, "alpha": 0.7,
                "ms": 30.,
                # "xscale":"log",
                "yscale": "log",
                # "xmin":1e-5, "xmax":1e-1,
                "ymin":3e-4, "ymax":1e0
                }
    #
    lk_edge_color = "black"

    #
    figname = make_plot_name(x_dic["v_n"], y_dic["v_n"], col_dic["v_n"], load_davids_data)
    #
    # collect data for plotting

    # x_arr = []
    # xerr_arr = []
    # y_arr = []
    # yerr_arr = []
    # col_arr = []
    # marker_arr = []
    # edgecolors = []


    if load_davids_data:

        table = rd.papertable

        from models_radice import simulations, fiducial
        plot_sims = simulations[fiducial]
        # print(plot_sims.keys())
        #
        # x_arr_david = plot_sims[rd.translation[x_dic["v_n"]]]
        # y_arr_david = plot_sims[rd.translation[y_dic["v_n"]]]
        x_arr_david = [float(plot_sims.loc[sim]["Lambda"]) for sim in plot_sims.index if sim in table.index]
        y_arr_david = [float(table.loc[sim]["M_disk"])/1.e2 for sim in plot_sims.index if sim in table.index]
        #
        x_arr_david = md.__apply_mod(x_dic["v_n"], x_arr_david, x_dic["mod"])
        y_arr_david = md.__apply_mod(y_dic["v_n"], y_arr_david, y_dic["mod"])
        assert len(x_arr_david) == len(y_arr_david)
    else:
        x_arr_david = []
        y_arr_david = []

    #
    fig = plt.figure(figsize=[4, 2.5])  # <-> |
    ax = fig.add_subplot(111)

    # for labels
    ax.scatter([-100], [-100], marker=mc_pc, s=plot_dic['ms'],
               color="gray", alpha=1., label="Prompt Collapse")
    ax.scatter([-100], [-100], marker=mc_st, s=plot_dic['ms'],
               color="gray", alpha=1., label="Stable remnant")
    ax.scatter([-100], [-100], marker=mc_bh, s=plot_dic['ms'],
               color="gray", alpha=1., label="Black Hole")
    ax.scatter([-100], [-100], marker=mc_bh, s=plot_dic['ms'],
               color="white", alpha=1., edgecolor=lk_edge_color, label="Viscosity")

    # main body
    if load_davids_data:
        ax.scatter(x_arr_david, y_arr_david, marker="3", s=20,
                   color="gray", alpha=1., label="Radice+2018")

    eoss = sorted(list(set(sims.EOS)))
    for ieos, eos in enumerate(eoss):
        qs = sorted(list(set(sims.q)))
        for iq, q in enumerate(qs):
            sel = sims[(sims.EOS == eos) & (sims.q == q)]
            cm = plt.cm.get_cmap(plot_dic["cmap"])
            norm = Normalize(vmin=plot_dic["vmin"], vmax=plot_dic["vmax"])
            #
            edgecolors = []
            for _, m in sel.iterrows():
                if m["viscosity"] == "LK":
                    edgecolors.append("black")
                else:
                    edgecolors.append("None")
            #
            markers = []
            for _, m in sel.iterrows():
                if np.isfinite(m.tcoll_gw):
                    marker = mc_bh
                    if m.tcoll_gw < t_pc:
                        marker = mc_pc
                else:
                    marker = mc_st
                markers.append(marker)
            #
            ax.errorbar(sel[x_dic["v_n"]], sel[y_dic["v_n"]], yerr=sel["err-"+y_dic["v_n"]], label=None,
                        color='gray', ecolor='gray',
                        fmt='None', elinewidth=1, capsize=1, alpha=0.5)
            #
            sc = mscatter(sel[x_dic["v_n"]], sel[y_dic["v_n"]], ax=ax, c=sel["q"], norm=norm,
                          s=plot_dic['ms'], cmap=cm, m=markers,
                          label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)

    # cm = plt.cm.get_cmap(plot_dic["cmap"])
    # norm = Normalize(vmin=plot_dic["vmin"], vmax=plot_dic["vmax"])
    # sc = mscatter(x_arr[:, 0], y_arr[:, 0], ax=ax, c=col_arr, norm=norm,
    #               s=plot_dic['ms'], cmap=cm, m=marker_arr,
    #               label=plot_dic['label'], alpha=plot_dic['alpha'], edgecolor=edgecolors)

    # error bars
    # ax.errorbar(x_arr[:, 0], y_arr[:, 0], yerr=y_arr[:, 1], color='gray', ecolor='gray',
    #             fmt='None', elinewidth=1, capsize=1, alpha=0.6)
    # ax.errorbar(x_arr[:, 0], y_arr[:, 0], xerr=x_arr[:, 1], color='gray', ecolor='gray',
    #             fmt='None', elinewidth=1, capsize=1, alpha=0.6)

    # fit
    # print(sims[x_dic["v_n"]], sims["mdisk_fit"])
    lam, fit_nl = fit_mdisk_new_data()
    ax.plot(lam, fit_nl, color='gray', ls=':', label="Old Fit")

    lam2, fit_nl2 = fit_mdisk_old_new_data()
    ax.plot(lam2, fit_nl2, color='black', ls='--', lw = 0.5, label="Fit")

    # ticks
    ax.tick_params(axis='both', which='both', labelleft=True,
                   labelright=False, tick1On=True, tick2On=True,
                   labelsize=12,
                   direction='in',
                   bottom=True, top=True, left=True, right=True)
    ax.minorticks_on()

    # limits
    if ("xmin" in plot_dic.keys() and "xmax" in plot_dic.keys()) and \
            (plot_dic["xmin"] != None and plot_dic["xmax"] != None):
        min_, max_ = plot_dic["xmin"], plot_dic["xmax"]
    else:
        min_, max_ = md.get_minmax(x_dic["v_n"], [], extra=2.)
    ax.set_xlim(min_, max_)
    #
    if ("ymin" in plot_dic.keys() and "ymax" in plot_dic.keys()) and \
            (plot_dic["ymin"] != None and plot_dic["ymax"] != None):
        min_, max_ = plot_dic["ymin"], plot_dic["ymax"]
    else:
        min_, max_ = md.get_minmax(y_dic["v_n"], [], extra=2., oldtable=load_davids_data)
    ax.set_ylim(min_, max_)

    # scale
    if "xscale" in plot_dic.keys() and plot_dic["xscale"] != None:
        ax.set_xscale(plot_dic["xscale"])
    if "yscale" in plot_dic.keys() and plot_dic["yscale"] != None:
        ax.set_yscale(plot_dic["yscale"])

    # label
    ax.set_xlabel(md.get_label(x_dic["v_n"]))
    ax.set_ylabel(md.get_label(y_dic["v_n"]))

    # colobar
    ax.legend(fancybox=True, loc='lower right',  # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
              shadow=False, ncol=1, fontsize=9,
              framealpha=0., borderaxespad=0.)
    clb = plt.colorbar(sc)
    clb.ax.set_title(r"$M_1/M_2$", fontsize=11)
    clb.ax.tick_params(labelsize=11)
    plt.subplots_adjust(bottom=0.20, right=1.0)
    # plt.tight_layout()
    print(__outplotdir__ + figname)
    plt.savefig(__outplotdir__ + figname, dpi=256)
    plt.close()



''' --- '''

def plot_disk_mass_evol():

    plot_dic = {
        "xmin":-10., "xmax":50.,
    }

    from data import ADD_METHODS_ALL_PAR

    sims = md.unique_simulations
    sr_sims = sims[sims["resolution"] == "SR"]

    diskt1, diskt2, diskn = [], [], []
    for sim in sr_sims.index:
        print(sim)
        o_data = ADD_METHODS_ALL_PAR(sim)
        disk_it, disk_times, disk_masses = o_data.get_disk_mass()
        if len(disk_times) > 0:
            print("[{:.1f}, {:.1f}] ({})".format(disk_times.min()*1.e3, disk_times.max()*1.e3, len(disk_times)))
            diskt1.append(disk_times.min())
            diskt2.append(disk_times.max())
            diskn.append(len(disk_times))
        else:
            # val = ""
            diskt1.append(np.nan)
            diskt2.append(np.nan)
            diskn.append(0)

    sr_sims["diskt1"] = diskt1
    sr_sims["diskt2"] = diskt2
    sr_sims["diskn"] = diskn

    print(sr_sims[(sr_sims["diskn"] > 10) & (sr_sims["diskt1"] < 15.*1.e-3)])

    sel = sr_sims[(sr_sims["diskn"] > 10) & (sr_sims["diskt1"] < 15.*1.e-3)]

    for sim in sel.index:
        o_data = ADD_METHODS_ALL_PAR(sim)
        disk_it, disk_times, disk_masses = o_data.get_disk_mass()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(disk_times*1e3, disk_masses, marker="x", color='black')
        ax.set_xlim(0.,60)
        ax.set_ylim(0.,0.4)
        ax.set_xlabel("$t-t_{merg}$")
        ax.set_ylabel("$M_{disk}M_{\odot}$")
        plt.savefig(__outplotdir__+"tst_disk_mass/"+sim+".png", dpi=256)
        plt.close()


    # eoss = sorted(list(set(sims["EOS"])))

def plot_mdisk_minus_mdiskfit_vs_q():
    """
    Plot Mdisk3D - Mdisk=f(Lambda) fit
    :return:
    """

    load_my_data = True
    load_vincent_data = True
    load_dietrich_data = True
    load_davids_data = True
    load_kiuchi_data = True
    t_pc = 1.5 * 1.e-3
    x_dic = {"v_n": "q", "err": None, "mod": None, "deferr": None}
    y_dic = {"v_n": "Mdisk3D", "err": "ud", "mod": None, "deferr": 0.2}
    col_dic = {"v_n": "Lambda", "err": None, "mod": None, "deferr": None}
    #
    mc_bh = "o"
    mc_st = "d"
    mc_pc = "s"
    #
    plot_dic = {"vmin": 350, "vmax": 900.0,
                #"vmin": 1., "vmax": 2.0,
                "cmap": "jet",  # "tab10",
                "label": None, "alpha": 0.6,
                "ms": 30.,
                # "xscale":"log", "yscale":"log",
                # "xmin":1e-5, "xmax":1e-1,
                # "ymin":-0.1, "ymax":0.4,
                "ymin": -1, "ymax": 3,
                "ylabel":r"$\Delta M_{\rm disk} / M_{\rm disk}$"
                }
    #
    lk_edge_color = "black"
    # Fit
    x_davids_fit = initial_guess4()
    fitting_func_of_lam = fitting_function4 # takes x_davids_fit, lam_array
    #
    figname = make_plot_name(x_dic["v_n"], y_dic["v_n"], col_dic["v_n"], load_davids_data)
    # collect data for plotting
    x_arr = np.zeros(3)
    y_arr = np.zeros(3)
    #
    fig = plt.figure(figsize=[4, 2.5])  # <-> |
    ax = fig.add_subplot(111)

    # for labels
    ax.scatter([-100], [-100], marker=mc_pc, s=plot_dic['ms'],
               color="gray", alpha=1., label="Prompt Collapse")
    ax.scatter([-100], [-100], marker=mc_st, s=plot_dic['ms'],
               color="gray", alpha=1., label="Stable remnant")
    ax.scatter([-100], [-100], marker=mc_bh, s=plot_dic['ms'],
               color="gray", alpha=1., label="Black Hole")
    ax.scatter([-100], [-100], marker=mc_bh, s=plot_dic['ms'],
               color="white", alpha=1., edgecolor=lk_edge_color, label="Viscosity")
    if load_davids_data:
        ax.scatter([-100], [-100], marker="*", s=plot_dic['ms'],
                   color="gray", alpha=1., edgecolor=None, label="Radice+2018")
    if load_kiuchi_data:
        ax.scatter([-100], [-100], marker="X", s=plot_dic['ms'],
                   color="gray", alpha=1., edgecolor=None, label="Kiuch+2019")
    if load_dietrich_data:
        ax.scatter([-100], [-100], marker="P", s=plot_dic['ms'],
                   color="gray", alpha=1., edgecolor=None, label="Dietrich+2016")
    if load_dietrich_data:
        ax.scatter([-100], [-100], marker="v", s=plot_dic['ms'],
                   color="gray", alpha=1., edgecolor=None, label="Vincent+2019")

    ''' ------------------- VINCENT ------------------ '''

    if load_vincent_data:

        import models_vincent as vi
        v_n_x = vi.translation[x_dic["v_n"]]
        v_n_y = vi.translation[y_dic["v_n"]]
        v_n_col = vi.translation[col_dic["v_n"]]
        sel = vi.simulations
        #
        markers = ["v" for sim in sel.index]
        mss = [20 for sim in sel.index]
        edgecolors = None
        #
        cm = plt.cm.get_cmap(plot_dic["cmap"])
        norm = Normalize(vmin=plot_dic["vmin"], vmax=plot_dic["vmax"])
        #
        x = np.array(sel[v_n_x])
        y = (np.array(sel[v_n_y]) - fitting_func_of_lam(x_davids_fit, sel["Lambda"])) / np.array(sel[v_n_y])
        #
        ax.errorbar(x, y, yerr=vi.params.Mdisk_err(np.array(sel[v_n_y])), label=None,
                    color='gray', ecolor='gray',
                    fmt='None', elinewidth=1, capsize=1, alpha=0.5)
        #
        sc = mscatter(x, y, ax=ax, c=sel[v_n_col], norm=norm,
                      s=mss, cmap=cm, m=markers,
                      label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)

    ''' ------------------- DIETRICH ----------------- '''

    if load_dietrich_data:
        import models_dietrich2016 as di
        v_n_x = di.translation[x_dic["v_n"]]
        v_n_y = di.translation[y_dic["v_n"]]
        v_n_col = di.translation[col_dic["v_n"]]
        sel = di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
        #
        markers = ["P" for sim in sel.index]
        mss = [20 for sim in sel.index]
        edgecolors = None
        #
        cm = plt.cm.get_cmap(plot_dic["cmap"])
        norm = Normalize(vmin=plot_dic["vmin"], vmax=plot_dic["vmax"])
        #
        x = np.array(sel[v_n_x])
        y = (np.array(sel[v_n_y]) - fitting_func_of_lam(x_davids_fit, sel["Lambda"])) / np.array(sel[v_n_y])
        #
        ax.errorbar(x, y, yerr=di.params.Mdisk_err(np.array(sel[v_n_y])), label=None,
                    color='gray', ecolor='gray',
                    fmt='None', elinewidth=1, capsize=1, alpha=0.5)
        #
        sc = mscatter(x, y, ax=ax, c=sel[v_n_col], norm=norm,
                      s=mss, cmap=cm, m=markers,
                      label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)

    ''' -------------------- DAVID ------------------- '''
    if load_davids_data:
        #
        v_n_x = rd.translation[x_dic["v_n"]]
        v_n_y = rd.translation[y_dic["v_n"]]
        v_n_col = rd.translation[col_dic["v_n"]]
        #
        # from radice_models import simulations, fiducial
        plot_sims = rd.simulations[rd.fiducial]
        eoss = sorted(list(set(plot_sims.EOS)))
        for ieos, eos in enumerate(eoss):
            qs = sorted(list(set(plot_sims.q)))
            for iq, q in enumerate(qs):
                sel = plot_sims[(plot_sims.EOS == eos) & (plot_sims.q == q)]
                cm = plt.cm.get_cmap(plot_dic["cmap"])
                norm = Normalize(vmin=plot_dic["vmin"], vmax=plot_dic["vmax"])
                #
                edgecolors = None
                #
                markers = ["*" for sim in sel.index]
                mss = [20 for sim in sel.index]
                #
                # print(fitting_func_of_lam(x_davids_fit, ))
                # print(sel.loc["BHBlp_M135135_LK"]["Lambda"]);exit(1)
                x = np.array(sel[v_n_x])
                y = (np.array(sel[v_n_y]) - fitting_func_of_lam(x_davids_fit, sel["Lambda"])) / np.array(sel[v_n_y])
                #
                # print(np.array(sel[v_n_y]))
                # print(rd.params.MdiskPP_err([1, 2,3 ])))
                ax.errorbar(x, y, yerr=rd.params.MdiskPP_err(np.array(sel[v_n_y])), label=None,
                            color='gray', ecolor='gray',
                            fmt='None', elinewidth=1, capsize=1, alpha=0.5)
                #
                # ax.scatter(sel[x_dic["v_n"]], sel[y_dic["v_n"]], marker=markers, markersize=mss)
                #
                sc = mscatter(x, y, ax=ax, c=np.array(sel[v_n_col]), norm=norm,
                              s=mss, cmap=cm, m=markers,
                              label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)

    ''' ------------------ KIUCHI ------------------'''

    if load_kiuchi_data:
        #
        import models_kiuchi as kc
        #
        v_n_x = kc.translation[x_dic["v_n"]]
        v_n_y = kc.translation[y_dic["v_n"]]
        v_n_col = kc.translation[col_dic["v_n"]]
        sel = kc.simulations
        #
        markers = ["X" for sim in sel.index]
        mss = [20 for sim in sel.index]
        edgecolors = None
        #
        cm = plt.cm.get_cmap(plot_dic["cmap"])
        norm = Normalize(vmin=plot_dic["vmin"], vmax=plot_dic["vmax"])
        #
        x = np.array(sel[v_n_x])
        y = (np.array(sel[v_n_y]) - fitting_func_of_lam(x_davids_fit, sel["Lambda"])) / np.array(sel[v_n_y])
        #
        ax.errorbar(x, y, yerr=kc.params.Mdisk_err(np.array(sel[v_n_y])), label=None,
                    color='gray', ecolor='gray',
                    fmt='None', elinewidth=1, capsize=1, alpha=0.5)
        #
        # ax.scatter(sel[x_dic["v_n"]], sel[y_dic["v_n"]], marker=markers, markersize=mss)
        #
        sc = mscatter(x, y, ax=ax, c=sel[v_n_col], norm=norm,
                      s=mss, cmap=cm, m=markers,
                      label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)

    ''' ------------------- ME -------------------- '''
    from groups import groups, get_outcome_marker
    # plot_sims = groups #= md.convert_models_to_uniquemodels_table()
    eoss = sorted(list(set(groups.EOS)))
    if load_my_data:
        #

        #
        v_n_x = x_dic["v_n"]# md.translation[x_dic["v_n"]]
        v_n_y = y_dic["v_n"]#md.translation[y_dic["v_n"]]
        v_n_col = col_dic["v_n"]# md.translation[y_dic["v_n"]]
        sel =groups
        #
        markers = ["d" for sim in sel.index]
        mss = [20 for sim in sel.index]
        edgecolors = None
        #
        cm = plt.cm.get_cmap(plot_dic["cmap"])
        norm = Normalize(vmin=plot_dic["vmin"], vmax=plot_dic["vmax"])
        #
        x = np.array(sel[v_n_x])
        y = (np.array(sel[v_n_y]) - fitting_func_of_lam(x_davids_fit, sel["Lambda"])) / np.array(sel[v_n_y])
        #
        ax.errorbar(x, y, yerr=sel["err-" + y_dic["v_n"]], label=None,
                    color='gray', ecolor='gray',
                    fmt='None', elinewidth=1, capsize=1, alpha=0.5)
        #
        # ax.scatter(sel[x_dic["v_n"]], sel[y_dic["v_n"]], marker=markers, markersize=mss)
        #
        print(sel[v_n_col])
        sc = mscatter(x, y, ax=ax, c=sel[v_n_col], norm=norm,
                      s=mss, cmap=cm, m=markers,
                      label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)

    ''' ------------------------------------------- '''

    # sc = None
    # from groups import groups, get_outcome_marker
    # plot_sims = groups #= md.convert_models_to_uniquemodels_table()
    # eoss = sorted(list(set(plot_sims.EOS)))
    # for ieos, eos in enumerate(eoss):
    #     qs = sorted(list(set(plot_sims.q)))
    #     for iq, q in enumerate(qs):
    #         sel = plot_sims[(plot_sims.EOS == eos) & (plot_sims.q == q)]
    #         cm = plt.cm.get_cmap(plot_dic["cmap"])
    #         norm = Normalize(vmin=plot_dic["vmin"], vmax=plot_dic["vmax"])
    #         #
    #         edgecolors = []
    #         for _, m in sel.iterrows():
    #             if m["viscosity"] == "LK":
    #                 edgecolors.append("black")
    #             else:
    #                 edgecolors.append("None")
    #         #
    #         markers = []
    #         mss = []
    #
    #         for _, m in sel.iterrows():
    #             marker = get_outcome_marker(m, v_n="outcome")
    #             markers.append(marker)
    #             mss.append(plot_dic['ms'])
    #         # for _, m in sel.iterrows():
    #         #     string = str(m.tcoll_gw)  # SR
    #         #     vals = string.strip('][').split(', ')
    #         #     # for val in vals:
    #         #
    #         #     if np.isfinite(m.tcoll_gw):
    #         #         marker = mc_bh
    #         #         ms = plot_dic['ms']
    #         #         if m.tcoll_gw < t_pc:
    #         #             marker = mc_pc
    #         #             ms = 2 * plot_dic['ms']
    #         #     else:
    #         #         marker = mc_st
    #         #         ms = 1.5 * plot_dic['ms']
    #         #     markers.append(marker)
    #         #     mss.append(ms)
    #         #
    #         x = sel[x_dic["v_n"]]
    #         y = (sel[y_dic["v_n"]] - fitting_func_of_lam(x_davids_fit, sel["Lambda"])) / sel[y_dic["v_n"]]
    #
    #         ax.errorbar(x, y, yerr=sel["err-" + y_dic["v_n"]], label=None,
    #                     color='gray', ecolor='gray',
    #                     fmt='None', elinewidth=1, capsize=1, alpha=0.5)
    #         #
    #         sc = mscatter(x, y, ax=ax, c=sel[col_dic["v_n"]], norm=norm,
    #                       s=mss, cmap=cm, m=markers,
    #                       label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)




    ax.tick_params(axis='both', which='both', labelleft=True,
                   labelright=False, tick1On=True, tick2On=True,
                   labelsize=12,
                   direction='in',
                   bottom=True, top=True, left=True, right=True)
    ax.minorticks_on()

    # limits
    if ("xmin" in plot_dic.keys() and "xmax" in plot_dic.keys()) and \
            (plot_dic["xmin"] != None and plot_dic["xmax"] != None):
        min_, max_ = plot_dic["xmin"], plot_dic["xmax"]
    else:
        min_, max_ = md.get_minmax(x_dic["v_n"], x_arr, extra=2.)
    ax.set_xlim(min_, max_)
    if ("ymin" in plot_dic.keys() and "ymax" in plot_dic.keys()) and \
            (plot_dic["ymin"] != None and plot_dic["ymax"] != None):
        min_, max_ = plot_dic["ymin"], plot_dic["ymax"]
    else:
        min_, max_ = md.get_minmax(y_dic["v_n"], y_arr, extra=2., oldtable=load_davids_data)
    ax.set_ylim(min_, max_)

    # scale
    if "xscale" in plot_dic.keys() and plot_dic["xscale"] != None:
        ax.set_xscale(plot_dic["xscale"])
    if "yscale" in plot_dic.keys() and plot_dic["yscale"] != None:
        ax.set_yscale(plot_dic["yscale"])

    # label
    if "ylabel" in plot_dic.keys() and plot_dic["ylabel"] != None: ax.set_ylabel(plot_dic["ylabel"])
    else: ax.set_ylabel(md.get_label(x_dic["v_n"]))
    if "xlabel" in plot_dic.keys() and plot_dic["xlabel"] != None: ax.set_xlabel(plot_dic["xlabel"])
    else: ax.set_xlabel(md.get_label(x_dic["v_n"]))

    #
    ax.axhline(y=0, linestyle='-', linewidth=0.4, color='black')

    # colobar
    ax.legend(fancybox=True, loc='upper center',  # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
              shadow=False, ncol=2, fontsize=9,
              framealpha=0., borderaxespad=0.)
    clb = plt.colorbar(sc)
    clb.ax.set_title(md.get_label(col_dic["v_n"]), fontsize=11)
    clb.ax.tick_params(labelsize=11)
    plt.subplots_adjust(bottom=0.20, right=1.0, left=0.15)
    # plt.tight_layout()
    print("plotted: \n")
    print(__outplotdir__ + figname)
    plt.savefig(__outplotdir__ + figname, dpi=256)
    plt.close()

#

def plot_mdisk_minus_mdiskfit_vs_q2():
    """
    Plot Mdisk3D - Mdisk=f(Lambda) fit
    :return:
    """

    load_my_data = True
    load_vincent_data = True
    load_dietrich_data = True
    load_davids_data = True
    load_kiuchi_data = True
    #
    t_pc = 1.5 * 1.e-3
    x_dic = {"v_n": "q", "err": None, "mod": None, "deferr": None}
    y_dic = {"v_n": "Mdisk3D", "err": "ud", "mod": None, "deferr": 0.2}
    col_dic = {"v_n": "Lambda", "err": None, "mod": None, "deferr": None}
    #
    mc_bh = "o"
    mc_st = "d"
    mc_pc = "s"
    #
    plot_dic = {"vmin": 350, "vmax": 900.0,
                #"vmin": 1., "vmax": 2.0,
                "cmap": "jet",  # "tab10",
                "label": None, "alpha": 0.8,
                "ms": 30.,
                # "xscale":"log", "yscale":"log",
                # "xmin":1e-5, "xmax":1e-1,
                # "ymin":-0.1, "ymax":0.4,
                "ymin": 0, "ymax": 0.6,
                "xlabel":  "$M_1/M_2$",
                "ylabel1": r"$M_{\rm disk}$",
                "ylabel2": r"$\Delta M_{\rm disk} / M_{\rm disk}$"
                }
    #
    lk_edge_color = "black"
    # Fit
    x_davids_fit = initial_guess4()
    fitting_func_of_lam = fitting_function4 # takes x_davids_fit, lam_array
    #
    figname = make_plot_name(x_dic["v_n"], y_dic["v_n"], col_dic["v_n"], load_davids_data)
    # collect data for plotting
    x_arr = np.zeros(3)
    y_arr = np.zeros(3)
    #


    # fig = plt.figure(figsize=[4, 2.5])  # <-> |
    # ax = fig.add_subplot(111)

    fig, ax = plt.subplots(nrows=2, figsize=[4, 3.5], sharex="all",
                           gridspec_kw={'height_ratios': [2, 1]})
    fig.subplots_adjust(left=0.15, bottom=0.12, top=0.95, right=0.95,
                        hspace=0)

    # ax = [None, None]
    # fig = plt.figure()
    # ax[0] = fig.add_subplot(2, 1, 1)
    # ax[1] = fig.add_subplot(2, 1, 2, sharex=ax[0])


    cm = plt.cm.get_cmap(plot_dic["cmap"])
    norm = Normalize(vmin=plot_dic["vmin"], vmax=plot_dic["vmax"])

    # for labels
    # ax.scatter([-100], [-100], marker=mc_pc, s=plot_dic['ms'],
    #            color="gray", alpha=1., label="Prompt Collapse")
    # ax.scatter([-100], [-100], marker=mc_st, s=plot_dic['ms'],
    #            color="gray", alpha=1., label="Stable remnant")
    # ax.scatter([-100], [-100], marker=mc_bh, s=plot_dic['ms'],
    #            color="gray", alpha=1., label="Black Hole")
    # ax.scatter([-100], [-100], marker=mc_bh, s=plot_dic['ms'],
    #            color="white", alpha=1., edgecolor=lk_edge_color, label="Viscosity")

    if load_my_data:
        ax[0].scatter([-100], [-100], marker="d", s=plot_dic['ms'],
                   color="gray", alpha=1., edgecolor=None, label="Our Data")
    if load_davids_data:
        ax[0].scatter([-100], [-100], marker="*", s=plot_dic['ms'],
                   color="gray", alpha=1., edgecolor=None, label="Radice+2018")
    if load_kiuchi_data:
        ax[0].scatter([-100], [-100], marker="X", s=plot_dic['ms'],
                   color="gray", alpha=1., edgecolor=None, label="Kiuch+2019")
    if load_dietrich_data:
        ax[0].scatter([-100], [-100], marker="P", s=plot_dic['ms'],
                   color="gray", alpha=1., edgecolor=None, label="Dietrich+2016")
    if load_dietrich_data:
        ax[0].scatter([-100], [-100], marker="v", s=plot_dic['ms'],
                   color="gray", alpha=1., edgecolor=None, label="Vincent+2019")

    ''' ------------------- VINCENT ------------------ '''

    if load_vincent_data:

        import models_vincent as vi
        v_n_x = vi.translation[x_dic["v_n"]]
        v_n_y = vi.translation[y_dic["v_n"]]
        v_n_col = vi.translation[col_dic["v_n"]]
        sel = vi.simulations
        #
        markers = ["v" for sim in sel.index]
        mss = [20 for sim in sel.index]
        edgecolors = None
        #
        # cm = plt.cm.get_cmap(plot_dic["cmap"])
        # norm = Normalize(vmin=plot_dic["vmin"], vmax=plot_dic["vmax"])
        #
        x = np.array(sel[v_n_x])
        y = (np.array(sel[v_n_y]))# - fitting_func_of_lam(x_davids_fit, sel["Lambda"])) / np.array(sel[v_n_y])
        #
        ax[0].errorbar(x, y, yerr=vi.params.Mdisk_err(np.array(sel[v_n_y])), label=None,
                    color='gray', ecolor='gray',
                    fmt='None', elinewidth=1, capsize=1, alpha=0.5)
        #
        sc = mscatter(x, y, ax=ax[0], c=sel[v_n_col], norm=norm,
                      s=mss, cmap=cm, m=markers,
                      label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)

        # ---------------------------

        x = np.array(sel[v_n_x])
        y = (np.array(sel[v_n_y]) - fitting_func_of_lam(x_davids_fit, sel["Lambda"])) / np.array(sel[v_n_y])
        delta_y = vi.params.Mdisk_err(np.array(sel[v_n_y]))/np.array(sel[v_n_y])
        #
        ax[1].errorbar(x, y, yerr=delta_y, label=None,
                       color='gray', ecolor='gray',
                       fmt='None', elinewidth=1, capsize=1, alpha=0.5)
        #
        sc = mscatter(x, y, ax=ax[1], c=sel[v_n_col], norm=norm,
                      s=mss, cmap=cm, m=markers,
                      label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)

    ''' ------------------- DIETRICH ----------------- '''

    if load_dietrich_data:
        import models_dietrich2016 as di
        v_n_x = di.translation[x_dic["v_n"]]
        v_n_y = di.translation[y_dic["v_n"]]
        v_n_col = di.translation[col_dic["v_n"]]
        sel = di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
        #
        markers = ["P" for sim in sel.index]
        mss = [20 for sim in sel.index]
        edgecolors = None
        #
        # cm = plt.cm.get_cmap(plot_dic["cmap"])
        # norm = Normalize(vmin=plot_dic["vmin"], vmax=plot_dic["vmax"])
        #
        x = np.array(sel[v_n_x])
        y = (np.array(sel[v_n_y]))# - fitting_func_of_lam(x_davids_fit, sel["Lambda"])) / np.array(sel[v_n_y])
        #
        ax[0].errorbar(x, y, yerr=di.params.Mdisk_err(np.array(sel[v_n_y])), label=None,
                    color='gray', ecolor='gray',
                    fmt='None', elinewidth=1, capsize=1, alpha=0.5)
        #
        sc = mscatter(x, y, ax=ax[0], c=sel[v_n_col], norm=norm,
                      s=mss, cmap=cm, m=markers,
                      label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)

        # ---------------------------
        x = np.array(sel[v_n_x])
        y = (np.array(sel[v_n_y]) - fitting_func_of_lam(x_davids_fit, sel["Lambda"])) / np.array(sel[v_n_y])
        delta_y = di.params.Mdisk_err(np.array(sel[v_n_y])) / np.array(sel[v_n_y])
        ax[1].errorbar(x, y, yerr=delta_y, label=None,
                    color='gray', ecolor='gray',
                    fmt='None', elinewidth=1, capsize=1, alpha=0.5)
        #
        sc = mscatter(x, y, ax=ax[1], c=sel[v_n_col], norm=norm,
                      s=mss, cmap=cm, m=markers,
                      label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)

    ''' -------------------- DAVID ------------------- '''

    if load_davids_data:
        #
        v_n_x = rd.translation[x_dic["v_n"]]
        v_n_y = rd.translation[y_dic["v_n"]]
        v_n_col = rd.translation[col_dic["v_n"]]
        #
        # from radice_models import simulations, fiducial
        plot_sims = rd.simulations[rd.fiducial]

        edgecolors = None
        #
        markers = ["*" for sim in plot_sims.index]
        mss = [20 for sim in plot_sims.index]
        #
        # print(fitting_func_of_lam(x_davids_fit, ))
        # print(sel.loc["BHBlp_M135135_LK"]["Lambda"]);exit(1)
        x = np.array(plot_sims[v_n_x])
        y = (np.array(plot_sims[v_n_y]))# - fitting_func_of_lam(x_davids_fit, plot_sims["Lambda"])) / np.array(plot_sims[v_n_y])
        #
        # print(np.array(sel[v_n_y]))
        # print(rd.params.MdiskPP_err([1, 2,3 ])))
        ax[0].errorbar(x, y, yerr=rd.params.MdiskPP_err(np.array(plot_sims[v_n_y])), label=None,
                    color='gray', ecolor='gray',
                    fmt='None', elinewidth=1, capsize=1, alpha=0.5)

        sc = mscatter(x, y, ax=ax[0], c=np.array(plot_sims[v_n_col]), norm=norm,
                      s=mss, cmap=cm, m=markers,
                      label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)

        # --------------------------------------
        x = np.array(plot_sims[v_n_x])
        y = (np.array(plot_sims[v_n_y]) - fitting_func_of_lam(x_davids_fit, plot_sims["Lambda"])) / np.array(plot_sims[v_n_y])
        delta_y = rd.params.MdiskPP_err(np.array(plot_sims[v_n_y])) / np.array(plot_sims[v_n_y])
        ax[1].errorbar(x, y, yerr=delta_y, label=None,
                    color='gray', ecolor='gray',
                    fmt='None', elinewidth=1, capsize=1, alpha=0.5)

        sc = mscatter(x, y, ax=ax[1], c=np.array(plot_sims[v_n_col]), norm=norm,
                      s=mss, cmap=cm, m=markers,
                      label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)

    ''' ------------------ KIUCHI ------------------'''

    if load_kiuchi_data:
        #
        import models_kiuchi as kc
        #
        v_n_x = kc.translation[x_dic["v_n"]]
        v_n_y = kc.translation[y_dic["v_n"]]
        v_n_col = kc.translation[col_dic["v_n"]]
        sel = kc.simulations
        #
        markers = ["X" for sim in sel.index]
        mss = [20 for sim in sel.index]
        edgecolors = None
        #
        # cm = plt.cm.get_cmap(plot_dic["cmap"])
        # norm = Normalize(vmin=plot_dic["vmin"], vmax=plot_dic["vmax"])
        #
        x = np.array(sel[v_n_x])
        y = (np.array(sel[v_n_y]))# - fitting_func_of_lam(x_davids_fit, sel["Lambda"])) / np.array(sel[v_n_y])
        #
        ax[0].errorbar(x, y, yerr=kc.params.Mdisk_err(np.array(sel[v_n_y])), label=None,
                    color='gray', ecolor='gray',
                    fmt='None', elinewidth=1, capsize=1, alpha=0.5)
        #
        # ax.scatter(sel[x_dic["v_n"]], sel[y_dic["v_n"]], marker=markers, markersize=mss)
        #
        sc = mscatter(x, y, ax=ax[0], c=sel[v_n_col], norm=norm,
                      s=mss, cmap=cm, m=markers,
                      label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)

        # --------------------------------------

        x = np.array(sel[v_n_x])
        y = (np.array(sel[v_n_y]) - fitting_func_of_lam(x_davids_fit, sel["Lambda"])) / np.array(sel[v_n_y])
        delta_y = kc.params.Mdisk_err(np.array(sel[v_n_y])) / np.array(sel[v_n_y])
        ax[1].errorbar(x, y, yerr=delta_y, label=None,
                    color='gray', ecolor='gray',
                    fmt='None', elinewidth=1, capsize=1, alpha=0.5)
        #
        # ax.scatter(sel[x_dic["v_n"]], sel[y_dic["v_n"]], marker=markers, markersize=mss)
        #
        sc = mscatter(x, y, ax=ax[1], c=sel[v_n_col], norm=norm,
                      s=mss, cmap=cm, m=markers,
                      label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)

    ''' ------------------- ME -------------------- '''

    from groups import groups, get_outcome_marker
    # plot_sims = groups #= md.convert_models_to_uniquemodels_table()
    eoss = sorted(list(set(groups.EOS)))
    if load_my_data:
        #

        #
        v_n_x = x_dic["v_n"]# md.translation[x_dic["v_n"]]
        v_n_y = y_dic["v_n"]#md.translation[y_dic["v_n"]]
        v_n_col = col_dic["v_n"]# md.translation[y_dic["v_n"]]
        sel =groups
        #
        markers = ["d" for sim in sel.index]
        mss = [20 for sim in sel.index]
        edgecolors = None
        #
        # cm = plt.cm.get_cmap(plot_dic["cmap"])
        # norm = Normalize(vmin=plot_dic["vmin"], vmax=plot_dic["vmax"])
        #
        x = np.array(sel[v_n_x])
        y = (np.array(sel[v_n_y]))# - fitting_func_of_lam(x_davids_fit, sel["Lambda"])) / np.array(sel[v_n_y])
        #
        ax[0].errorbar(x, y, yerr=sel["err-" + y_dic["v_n"]], label=None,
                    color='gray', ecolor='gray',
                    fmt='None', elinewidth=1, capsize=1, alpha=0.5)
        #
        # ax.scatter(sel[x_dic["v_n"]], sel[y_dic["v_n"]], marker=markers, markersize=mss)
        #
        print(sel[v_n_col])
        sc = mscatter(x, y, ax=ax[0], c=sel[v_n_col], norm=norm,
                      s=mss, cmap=cm, m=markers,
                      label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)

        # ------------------------------------

        x = np.array(sel[v_n_x])
        y = (np.array(sel[v_n_y]) - fitting_func_of_lam(x_davids_fit, sel["Lambda"])) / np.array(sel[v_n_y])
        #
        delta_y = sel["err-" + y_dic["v_n"]] / np.array(sel[v_n_y])
        #
        ax[1].errorbar(x, y, yerr=delta_y, label=None,
                    color='gray', ecolor='gray',
                    fmt='None', elinewidth=1, capsize=1, alpha=0.5)
        #
        # ax.scatter(sel[x_dic["v_n"]], sel[y_dic["v_n"]], marker=markers, markersize=mss)
        #
        print(sel[v_n_col])
        sc = mscatter(x, y, ax=ax[1], c=sel[v_n_col], norm=norm,
                      s=mss, cmap=cm, m=markers,
                      label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)

    ''' ------------------------------------------- '''

    for i in range(len(ax)):
        ax[i].tick_params(axis='both', which='both', labelleft=True,
                       labelright=False, tick1On=True, tick2On=True,
                       labelsize=12,
                       direction='in',
                       bottom=True, top=True, left=True, right=True)
        ax[i].minorticks_on()

        # limits
        if ("xmin" in plot_dic.keys() and "xmax" in plot_dic.keys()) and \
                (plot_dic["xmin"] != None and plot_dic["xmax"] != None):
            min_, max_ = plot_dic["xmin"], plot_dic["xmax"]
        else:
            min_, max_ = md.get_minmax(x_dic["v_n"], x_arr, extra=2.)
        ax[i].set_xlim(min_, max_)

    #
    if ("ymin" in plot_dic.keys() and "ymax" in plot_dic.keys()) and \
            (plot_dic["ymin"] != None and plot_dic["ymax"] != None):
        min_, max_ = plot_dic["ymin"], plot_dic["ymax"]
    else:
        min_, max_ = md.get_minmax(y_dic["v_n"], y_arr, extra=2., oldtable=load_davids_data)
    ax[0].set_ylim(min_, max_)

    ax[1].set_ylim(-1.2, 1.8)

    # scale
    if "xscale" in plot_dic.keys() and plot_dic["xscale"] != None:
        ax[0].set_xscale(plot_dic["xscale"])
    if "yscale" in plot_dic.keys() and plot_dic["yscale"] != None:
        ax[0].set_yscale(plot_dic["yscale"])

    # label
    if "ylabel1" in plot_dic.keys() and plot_dic["ylabel1"] != None: ax[0].set_ylabel(plot_dic["ylabel1"])
    else: ax[0].set_ylabel(md.get_label(y_dic["v_n"]))
    if "ylabel2" in plot_dic.keys() and plot_dic["ylabel2"] != None: ax[1].set_ylabel(plot_dic["ylabel2"])
    else: ax[1].set_ylabel(md.get_label(y_dic["v_n"]))
    #
    if "xlabel" in plot_dic.keys() and plot_dic["xlabel"] != None: ax[1].set_xlabel(plot_dic["xlabel"])
    else: ax[1].set_xlabel(md.get_label(x_dic["v_n"]))



    #
    ax[1].axhline(y=0, linestyle='-', linewidth=0.4, color='black')

    # colobar
    ax[0].legend(fancybox=True, loc='upper center',  # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
              shadow=False, ncol=2, fontsize=9,
              framealpha=0., borderaxespad=0.)
    clb = fig.colorbar(sc, ax=ax.ravel().tolist())
    clb.ax.set_title(md.get_label(col_dic["v_n"]), fontsize=11)
    clb.ax.tick_params(labelsize=11)
    # plt.subplots_adjust(bottom=0.20, right=1.0, left=0.15)
    # plt.tight_layout()
    print("plotted: \n")
    print(__outplotdir__ + figname)
    plt.savefig(__outplotdir__ + figname, dpi=256)
    plt.close()

def plot_mdisk_minus_mdiskfit_vs_q3(x_dic, y_dic, col_dic, plot_dic, fit_parameters, fitting_function, datasets):
    """
    Plot Mdisk3D - Mdisk=f(Lambda) fit
    :return:
    """

    x_davids_fit = fit_parameters
    fitting_func_of_lam = fitting_function # takes x_davids_fit, lam_array

    ''' ----------------------------------------- '''


    #
    if not "figname" in plot_dic.keys():
        figname = make_plot_name(x_dic["v_n"], y_dic["v_n"], col_dic["v_n"])
    else:
        figname = plot_dic["figname"]
    # collect data for plotting
    x_arr = np.zeros(3)
    y_arr = np.zeros(3)

    # plot
    fig, ax = plt.subplots(nrows=2, figsize=[4, 3.5], sharex="all",
                           gridspec_kw={'height_ratios': [2, 1]})
    fig.subplots_adjust(left=0.15, bottom=0.12, top=0.95, right=0.95,
                        hspace=0)

    cm = plt.cm.get_cmap(plot_dic["cmap"])
    norm = Normalize(vmin=plot_dic["vmin"], vmax=plot_dic["vmax"])

    for dataset_name in datasets.keys():
        dic = datasets[dataset_name]
        if dataset_name == "our":
            ax[0].scatter([-100], [-100], marker=dic['marker'], s=dic['ms'],
                          color="gray", alpha=1., edgecolor=None, label="Our Data")
        elif dataset_name == "radice":
            ax[0].scatter([-100], [-100], marker=dic['marker'], s=dic['ms'],
                          color="gray", alpha=1., edgecolor=None, label="Radice+2018")
        elif dataset_name == "kiuchi":
            ax[0].scatter([-100], [-100], marker=dic['marker'], s=dic['ms'],
                          color="gray", alpha=1., edgecolor=None, label="Kiuch+2019")
        elif dataset_name == "dietrich":
            ax[0].scatter([-100], [-100], marker=dic['marker'], s=dic['ms'],
                          color="gray", alpha=1., edgecolor=None, label="Dietrich+2016")
        elif dataset_name == "vincent":
            ax[0].scatter([-100], [-100], marker=dic['marker'], s=dic['ms'],
                          color="gray", alpha=1., edgecolor=None, label="Vincent+2019")
        else:
            raise NameError("dataset: {} is not recognized.")

    #
    for dataset_name in datasets.keys():
        dic = datasets[dataset_name]
        #
        if dataset_name == "our":
            #
            from groups import groups, get_outcome_marker
            #
            v_n_x = x_dic["v_n"]
            v_n_y = y_dic["v_n"]
            v_n_col = col_dic["v_n"]
            sel = groups
            #
            markers = [dic['marker'] for sim in sel.index]
            mss = [dic['ms'] for sim in sel.index]
            edgecolors = None

            x = np.array(sel[v_n_x])
            y = (np.array(sel[v_n_y]))  # - fitting_func_of_lam(x_davids_fit, sel["Lambda"])) / np.array(sel[v_n_y])
            y = __apply_mod(y_dic["v_n"], y, y_dic["mod"])
            #
            ax[0].errorbar(x, y, yerr=sel["err-" + y_dic["v_n"]], label=None,
                           color='gray', ecolor='gray',
                           fmt='None', elinewidth=1, capsize=1, alpha=0.5)

            # print(sel[v_n_col])
            sc = mscatter(x, y, ax=ax[0], c=sel[v_n_col], norm=norm,
                          s=mss, cmap=cm, m=markers,
                          label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)

            # ------------------------------------

            x = np.array(sel[v_n_x])
            # y = (np.array(sel[v_n_y]) - fitting_func_of_lam(x_davids_fit, sel["Lambda"])) / np.array(sel[v_n_y])
            y_ = (y - fitting_func_of_lam(x_davids_fit, sel)) / y
            #
            wrr = __apply_mod(y_dic["v_n"], sel["err-" + y_dic["v_n"]], y_dic["mod"])
            delta_y = wrr / y
            # print(fitting_func_of_lam(x_davids_fit, sel));exit(1)
            #
            ax[1].errorbar(x, y_, yerr=delta_y, label=None,
                           color='gray', ecolor='gray',
                           fmt='None', elinewidth=1, capsize=1, alpha=0.5)

            # print(sel[v_n_col])
            sc = mscatter(x, y_, ax=ax[1], c=sel[v_n_col], norm=norm,
                          s=mss, cmap=cm, m=markers,
                          label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)
        elif dataset_name == "radice":

            v_n_x = rd.translation[x_dic["v_n"]]
            v_n_y = rd.translation[y_dic["v_n"]]
            v_n_col = rd.translation[col_dic["v_n"]]

            plot_sims = rd.simulations[rd.fiducial]

            edgecolors = None
            #
            markers = [dic['marker'] for sim in plot_sims.index]
            mss = [dic['ms'] for sim in plot_sims.index]

            x = np.array(plot_sims[v_n_x])
            y = (np.array(plot_sims[v_n_y]))
            y = __apply_mod(y_dic["v_n"], y, y_dic["mod"])

            ax[0].errorbar(x, y, yerr=rd.params.MdiskPP_err(np.array(plot_sims[v_n_y])), label=None,
                           color='gray', ecolor='gray',
                           fmt='None', elinewidth=1, capsize=1, alpha=0.5)

            sc = mscatter(x, y, ax=ax[0], c=np.array(plot_sims[v_n_col]), norm=norm,
                          s=mss, cmap=cm, m=markers,
                          label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)

            # --------------------------------------
            x = np.array(plot_sims[v_n_x])
            # y = (np.array(plot_sims[v_n_y]) - fitting_func_of_lam(x_davids_fit, plot_sims["Lambda"])) / np.array(
            #     plot_sims[v_n_y])
            y_ = (y - fitting_func_of_lam(x_davids_fit, plot_sims)) / y
            delta_y = rd.params.MdiskPP_err(y) / y
            ax[1].errorbar(x, y_, yerr=delta_y, label=None,
                           color='gray', ecolor='gray',
                           fmt='None', elinewidth=1, capsize=1, alpha=0.5)

            sc = mscatter(x, y_, ax=ax[1], c=np.array(plot_sims[v_n_col]), norm=norm,
                          s=mss, cmap=cm, m=markers,
                          label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)
        elif dataset_name == "kiuchi":
            #
            import models_kiuchi as kc
            #
            v_n_x = kc.translation[x_dic["v_n"]]
            v_n_y = kc.translation[y_dic["v_n"]]
            v_n_col = kc.translation[col_dic["v_n"]]
            sel = kc.simulations
            #
            markers = [dic['marker'] for sim in sel.index]
            mss = [dic['ms'] for sim in sel.index]
            edgecolors = None

            x = np.array(sel[v_n_x])
            y = (np.array(sel[v_n_y]))
            y = __apply_mod(y_dic["v_n"], y, y_dic["mod"])
            #
            ax[0].errorbar(x, y, yerr=kc.params.Mdisk_err(np.array(sel[v_n_y])), label=None,
                           color='gray', ecolor='gray',
                           fmt='None', elinewidth=1, capsize=1, alpha=0.5)

            sc = mscatter(x, y, ax=ax[0], c=sel[v_n_col], norm=norm,
                          s=mss, cmap=cm, m=markers,
                          label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)

            # --------------------------------------

            x = np.array(sel[v_n_x])
            # y = (np.array(sel[v_n_y]) - fitting_func_of_lam(x_davids_fit, sel["Lambda"])) / np.array(sel[v_n_y])
            y_ = (y - fitting_func_of_lam(x_davids_fit, sel)) / y
            delta_y = kc.params.Mdisk_err(y) /y
            ax[1].errorbar(x, y_, yerr=delta_y, label=None,
                           color='gray', ecolor='gray',
                           fmt='None', elinewidth=1, capsize=1, alpha=0.5)

            sc = mscatter(x, y_, ax=ax[1], c=sel[v_n_col], norm=norm,
                          s=mss, cmap=cm, m=markers,
                          label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)
        elif dataset_name == "dietrich":
            import models_dietrich2016 as di
            v_n_x = di.translation[x_dic["v_n"]]
            v_n_y = di.translation[y_dic["v_n"]]
            v_n_col = di.translation[col_dic["v_n"]]
            sel = di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
            #
            markers = [dic['marker'] for sim in sel.index]
            mss = [dic['ms'] for sim in sel.index]
            edgecolors = None

            x = np.array(sel[v_n_x])
            y = (np.array(sel[v_n_y]))
            y = __apply_mod(y_dic["v_n"], y, y_dic["mod"])

            ax[0].errorbar(x, y, yerr=di.params.Mdisk_err(y), label=None,
                           color='gray', ecolor='gray',
                           fmt='None', elinewidth=1, capsize=1, alpha=0.5)
            #
            sc = mscatter(x, y, ax=ax[0], c=sel[v_n_col], norm=norm,
                          s=mss, cmap=cm, m=markers,
                          label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)

            # ---------------------------
            x = np.array(sel[v_n_x])
            # y = (np.array(sel[v_n_y]) - fitting_func_of_lam(x_davids_fit, sel["Lambda"])) / np.array(sel[v_n_y])
            y_ = (y - fitting_func_of_lam(x_davids_fit, sel)) / y
            # print(fitting_func_of_lam(x_davids_fit, sel)); exit(1)
            delta_y = di.params.Mdisk_err(y) / y
            ax[1].errorbar(x, y_, yerr=delta_y, label=None,
                           color='gray', ecolor='gray',
                           fmt='None', elinewidth=1, capsize=1, alpha=0.5)
            #
            sc = mscatter(x, y_, ax=ax[1], c=sel[v_n_col], norm=norm,
                          s=mss, cmap=cm, m=markers,
                          label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)
        elif dataset_name == "vincent":
            import models_vincent as vi
            v_n_x = vi.translation[x_dic["v_n"]]
            v_n_y = vi.translation[y_dic["v_n"]]
            v_n_col = vi.translation[col_dic["v_n"]]
            sel = vi.simulations
            #
            markers = [dic['marker'] for sim in sel.index]
            mss = [dic['ms'] for sim in sel.index]
            edgecolors = None

            x = np.array(sel[v_n_x])
            y = (np.array(sel[v_n_y]))
            y = __apply_mod(y_dic["v_n"], y, y_dic["mod"])
            #
            ax[0].errorbar(x, y, yerr=vi.params.Mdisk_err(np.array(sel[v_n_y])), label=None,
                           color='gray', ecolor='gray',
                           fmt='None', elinewidth=1, capsize=1, alpha=0.5)
            #
            sc = mscatter(x, y, ax=ax[0], c=sel[v_n_col], norm=norm,
                          s=mss, cmap=cm, m=markers,
                          label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)

            # ---------------------------

            x = np.array(sel[v_n_x])
            # y = (np.array(sel[v_n_y]) - fitting_func_of_lam(x_davids_fit, sel["Lambda"])) / np.array(sel[v_n_y])
            y_ = (y - fitting_func_of_lam(x_davids_fit, sel)) / y
            delta_y = vi.params.Mdisk_err(y) / y
            #
            ax[1].errorbar(x, y_, yerr=delta_y, label=None,
                           color='gray', ecolor='gray',
                           fmt='None', elinewidth=1, capsize=1, alpha=0.5)
            #
            sc = mscatter(x, y_, ax=ax[1], c=sel[v_n_col], norm=norm,
                          s=mss, cmap=cm, m=markers,
                          label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)
        else:
            raise NameError("dataset: {} is not recognized.")


    ''' ------------------------------------------- '''

    for i in range(len(ax)):
        ax[i].tick_params(axis='both', which='both', labelleft=True,
                       labelright=False, tick1On=True, tick2On=True,
                       labelsize=12,
                       direction='in',
                       bottom=True, top=True, left=True, right=True)
        ax[i].minorticks_on()

        # limits
        if ("xmin" in plot_dic.keys() and "xmax" in plot_dic.keys()) and \
                (plot_dic["xmin"] != None and plot_dic["xmax"] != None):
            min_, max_ = plot_dic["xmin"], plot_dic["xmax"]
        else:
            min_, max_ = md.get_minmax(x_dic["v_n"], x_arr, extra=2.)
        ax[i].set_xlim(min_, max_)

    #
    if ("ymin" in plot_dic.keys() and "ymax" in plot_dic.keys()) and \
            (plot_dic["ymin"] != None and plot_dic["ymax"] != None):
        min_, max_ = plot_dic["ymin"], plot_dic["ymax"]
    else:
        min_, max_ = md.get_minmax(y_dic["v_n"], y_arr, extra=2., oldtable=None)
    ax[0].set_ylim(min_, max_)

    ax[1].set_ylim(-3.2, 2.6)

    # scale
    if "xscale" in plot_dic.keys() and plot_dic["xscale"] != None:
        ax[0].set_xscale(plot_dic["xscale"])
    if "yscale" in plot_dic.keys() and plot_dic["yscale"] != None:
        ax[0].set_yscale(plot_dic["yscale"])

    # label
    if "ylabel1" in plot_dic.keys() and plot_dic["ylabel1"] != None: ax[0].set_ylabel(plot_dic["ylabel1"])
    else: ax[0].set_ylabel(md.get_label(y_dic["v_n"]))
    if "ylabel2" in plot_dic.keys() and plot_dic["ylabel2"] != None: ax[1].set_ylabel(plot_dic["ylabel2"])
    else: ax[1].set_ylabel(md.get_label(y_dic["v_n"]))
    #
    if "xlabel" in plot_dic.keys() and plot_dic["xlabel"] != None: ax[1].set_xlabel(plot_dic["xlabel"])
    else: ax[1].set_xlabel(md.get_label(x_dic["v_n"]))

    #
    ax[1].axhline(y=0, linestyle='-', linewidth=0.4, color='black')

    # colobar
    ax[0].legend(fancybox=True, loc='upper center',  # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
              shadow=False, ncol=2, fontsize=9,
              framealpha=0., borderaxespad=0.)
    clb = fig.colorbar(sc, ax=ax.ravel().tolist())
    clb.ax.set_title(md.get_label(col_dic["v_n"]), fontsize=11)
    clb.ax.tick_params(labelsize=11)
    # plt.subplots_adjust(bottom=0.20, right=1.0, left=0.15)
    # plt.tight_layout()
    print("plotted: \n")
    print(__outplotdir__ + figname)
    plt.savefig(__outplotdir__ + figname, dpi=256)
    plt.close()

def summary_cumulative_plots():

    """

    :return:

    """



    ''' ----------- DISK MASS --------- '''

    datasets = {}
    datasets["kiuchi"] =    {'marker': "X", "ms": 20}
    datasets["radice"] =    {'marker': '*', "ms": 20}
    datasets["dietrich"] =  {'marker': 'd', "ms": 20}
    datasets["vincent"] =   {'marker': 'v', 'ms': 20}
    datasets['our'] =       {'marker': 'o', 'ms': 20}

    x_dic = {"v_n": "q", "err": None, "mod": None, "deferr": None}
    y_dic = {"v_n": "Mdisk3D", "err": "ud", "mod": None, "deferr": 0.2}
    col_dic = {"v_n": "Lambda", "err": None, "mod": None, "deferr": None}
    #
    plot_dic = {"vmin": 350, "vmax": 900.0,
                # "vmin": 1., "vmax": 2.0,
                "cmap": "jet",  # "tab10",
                "label": None, "alpha": 0.8,
                "ms": 30.,
                # "xscale":"log", "yscale":"log",
                # "xmin":1e-5, "xmax":1e-1,
                # "ymin":-0.1, "ymax":0.4,
                "ymin": 0, "ymax": 0.6,
                "xlabel": "$M_1/M_2$",
                "ylabel1": r"$M_{\rm disk}$",
                "ylabel2": r"$\Delta M_{\rm disk} / M_{\rm disk}$"
                }

    # Fit
    x_davids_fit = initial_guess4()
    fitting_func_of_lam = fitting_function4  # takes x_davids_fit, lam_array

    #
    plot_mdisk_minus_mdiskfit_vs_q3(x_dic, y_dic, col_dic, plot_dic, x_davids_fit, fitting_func_of_lam, datasets)
    #

    ''' ----------- EJECTA MASS --------- '''

    datasets = {}
    datasets["kiuchi"] =    {'marker': "X", "ms": 20}
    datasets["radice"] =    {'marker': '*', "ms": 20}
    datasets["dietrich"] =  {'marker': 'd', "ms": 20}
    datasets["vincent"] =   {'marker': 'v', 'ms': 20}
    datasets['our'] =       {'marker': 'o', 'ms': 20}

    x_dic =   {"v_n": "q", "err": None, "mod": None, "deferr": None}
    y_dic =   {"v_n": "Mej_tot-geo", "err": "ud", "mod": "*1e3", "deferr": 0.2}
    col_dic = {"v_n": "Lambda", "err": None, "mod": None, "deferr": None}
    #
    plot_dic = {"vmin": 350, "vmax": 900.0,
                # "vmin": 1., "vmax": 2.0,
                "cmap": "jet",  # "tab10",
                "label": None, "alpha": 0.8,
                "ms": 30.,
                # "xscale":"log", "yscale":"log",
                # "xmin":1e-5, "xmax":1e-1,
                # "ymin":-0.1, "ymax":0.4,
                "ymin": 0, "ymax": 45.0,
                "xlabel": "$M_1/M_2$",
                "ylabel1": r"$M_{\rm ej}$ $[10^3M_{\odot}]$",
                "ylabel2": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
                "figname": "final_summary_ejecta_mass.png"
                }

    # Fit
    x_davids_fit = dietrich_mej_coeffs()#radice()
    fitting_func_of_lam = fitting_function_mej  # takes x_davids_fit, lam_array

    #
    # plot_mdisk_minus_mdiskfit_vs_q3(x_dic, y_dic, col_dic, plot_dic, x_davids_fit, fitting_func_of_lam, datasets)
    #




''' --- '''

def plot_mdisk_minus_mdiskfit_vs_q4(x_dic, y_dic, col_dic, plot_dic, fit_parameters, fitting_function, datasets):
    """
    Plot Mdisk3D - Mdisk=f(Lambda) fit
    :return:
    """

    x_davids_fit = fit_parameters
    fitting_func_of_lam = fitting_function # takes x_davids_fit, lam_array

    ''' ----------------------------------------- '''


    #
    if not "figname" in plot_dic.keys():
        figname = make_plot_name(x_dic["v_n"], y_dic["v_n"], col_dic["v_n"])
    else:
        figname = plot_dic["figname"]
    # collect data for plotting
    x_arr = np.zeros(3)
    y_arr = np.zeros(3)

    # plot
    if plot_dic["fit_panel"]:
        fig, ax = plt.subplots(nrows=2, figsize=[4.5, 3.5], sharex="all",
                               gridspec_kw={'height_ratios': [2, 1]})
        fig.subplots_adjust(left=0.15, bottom=0.12, top=0.95, right=0.95,
                            hspace=0)
    else:
        fig = plt.figure(figsize = (7.2, 3.6))#(nrows=1, figsize=[4.5, 3.5])
        ax = fig.add_subplot(111)
        # ax = [ax, ax]

    cm = plt.cm.get_cmap(plot_dic["cmap"])
    norm = Normalize(vmin=plot_dic["vmin"], vmax=plot_dic["vmax"])

    ''' ------------- '''

    for dataset_name in datasets.keys():
        dic = datasets[dataset_name]
        marker = dic['marker']
        label = dic['label']
        ms = dic['ms']

        if plot_dic["fit_panel"]: ax_ = ax[0]
        else: ax_ = ax
        ax_.scatter([-100], [-100], marker=marker, s=ms,
                    color="gray", alpha=1., edgecolor=None, label=label)


    ''' ------------- '''
    all_x = np.empty(1)
    all_y = np.empty(1)

    for dataset_name in datasets.keys():
        #
        dic = datasets[dataset_name]
        d_cl    = dic["data"]       # md, rd, ki ...
        err     = dic["err"]        # err lambda(y)
        models  = dic["models"]     # models DataFrame
        if "fit" in dic.keys() and dic["fit"] != None:
            do_fit  = dic["fit"]
        else:
            do_fit = False
        #
        # v_n_x   = d_cl.translation[x_dic["v_n"]]
        # v_n_y   = d_cl.translation[y_dic["v_n"]]
        v_n_col = d_cl.translation[col_dic["v_n"]]
        #
        edgecolors = None
        #
        markers = [dic['marker'] for sim in models.index]
        mss = [dic['ms'] for sim in models.index]
        #
        # x = np.array(models[v_n_x])
        # x = __apply_mod(x_dic["v_n"], x, x_dic["mod"])
        # y = np.array(models[v_n_y])
        # y = __apply_mod(y_dic["v_n"], y, y_dic["mod"])
        # c =  np.array(models[v_n_col])

        x = d_cl.get_mod_data(x_dic["v_n"], x_dic["mod"], models)
        y = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models)
        c = np.array(models[v_n_col])
        print(dataset_name)
        print(len(x), len(y), len(c))
        # print(x)
        # print(y)
        # print(c)
        # --- --- ---

        all_x = np.append(all_x, x)
        all_y = np.append(all_y, y)

        if err == "v_n":
            err = models["err-" + y_dic["v_n"]]
            err = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, err)
            # err = __apply_mod(y_dic["v_n"], err, y_dic["mod"])
        elif err ==  None:
            err = np.zeros(len(y))
        else:
            err = err(y)
            # err = __apply_mod(y_dic["v_n"], err, y_dic["mod"])
            err = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, err)

        if plot_dic["fit_panel"]: ax_ = ax[0]
        else: ax_ = ax
        ax_.errorbar(x, y, yerr=err, label=None,
                       color='gray', ecolor='gray',
                       fmt='None', elinewidth=1, capsize=1, alpha=0.5)

        sc = mscatter(x, y, ax=ax_, c=c, norm=norm,
                      s=mss, cmap=cm, m=markers,
                      label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)

        if do_fit and plot_dic["fit_panel"]:
            # --- --- ---
            fitted_values = fitting_func_of_lam(x_davids_fit, models)
            if y_dic["v_n"] == "Mej_tot-geo": fitted_values = fitted_values / 1.e3 # Tims fucking fit
            #
            y_from_fit = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, fitted_values)
            y_ = (y - y_from_fit) / y
            delta_y = err / y
            ax[1].errorbar(x, y_, yerr=delta_y, label=None,
                           color='gray', ecolor='gray',
                           fmt='None', elinewidth=1, capsize=1, alpha=0.5)
            #
            sc = mscatter(x, y_, ax=ax[1], c=c, norm=norm, s=mss, cmap=cm, m=markers,
                          label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)

    #
    #
    # #
    # for dataset_name in datasets.keys():
    #     dic = datasets[dataset_name]
    #     #
    #     if dataset_name == "our":
    #         #
    #         from groups import groups, get_outcome_marker
    #         #
    #         v_n_x = x_dic["v_n"]
    #         v_n_y = y_dic["v_n"]
    #         v_n_col = col_dic["v_n"]
    #         sel = groups
    #         #
    #         markers = [dic['marker'] for sim in sel.index]
    #         mss = [dic['ms'] for sim in sel.index]
    #         edgecolors = None
    #
    #         x = np.array(sel[v_n_x])
    #         y = (np.array(sel[v_n_y]))  # - fitting_func_of_lam(x_davids_fit, sel["Lambda"])) / np.array(sel[v_n_y])
    #         y = __apply_mod(y_dic["v_n"], y, y_dic["mod"])
    #         #
    #         ax[0].errorbar(x, y, yerr=sel["err-" + y_dic["v_n"]], label=None,
    #                        color='gray', ecolor='gray',
    #                        fmt='None', elinewidth=1, capsize=1, alpha=0.5)
    #
    #         # print(sel[v_n_col])
    #         sc = mscatter(x, y, ax=ax[0], c=sel[v_n_col], norm=norm,
    #                       s=mss, cmap=cm, m=markers,
    #                       label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)
    #
    #         # ------------------------------------
    #
    #         x = np.array(sel[v_n_x])
    #         # y = (np.array(sel[v_n_y]) - fitting_func_of_lam(x_davids_fit, sel["Lambda"])) / np.array(sel[v_n_y])
    #         y_ = (y - fitting_func_of_lam(x_davids_fit, sel)) / y
    #         #
    #         wrr = __apply_mod(y_dic["v_n"], sel["err-" + y_dic["v_n"]], y_dic["mod"])
    #         delta_y = wrr / y
    #         # print(fitting_func_of_lam(x_davids_fit, sel));exit(1)
    #         #
    #         ax[1].errorbar(x, y_, yerr=delta_y, label=None,
    #                        color='gray', ecolor='gray',
    #                        fmt='None', elinewidth=1, capsize=1, alpha=0.5)
    #
    #         # print(sel[v_n_col])
    #         sc = mscatter(x, y_, ax=ax[1], c=sel[v_n_col], norm=norm,
    #                       s=mss, cmap=cm, m=markers,
    #                       label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)
    #     elif dataset_name == "radice":
    #
    #         v_n_x = rd.translation[x_dic["v_n"]]
    #         v_n_y = rd.translation[y_dic["v_n"]]
    #         v_n_col = rd.translation[col_dic["v_n"]]
    #
    #         plot_sims = rd.simulations[rd.fiducial]
    #
    #         edgecolors = None
    #         #
    #         markers = [dic['marker'] for sim in plot_sims.index]
    #         mss = [dic['ms'] for sim in plot_sims.index]
    #
    #         x = np.array(plot_sims[v_n_x])
    #         y = (np.array(plot_sims[v_n_y]))
    #         y = __apply_mod(y_dic["v_n"], y, y_dic["mod"])
    #
    #         ax[0].errorbar(x, y, yerr=rd.params.MdiskPP_err(np.array(plot_sims[v_n_y])), label=None,
    #                        color='gray', ecolor='gray',
    #                        fmt='None', elinewidth=1, capsize=1, alpha=0.5)
    #
    #         sc = mscatter(x, y, ax=ax[0], c=np.array(plot_sims[v_n_col]), norm=norm,
    #                       s=mss, cmap=cm, m=markers,
    #                       label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)
    #
    #         # --------------------------------------
    #         x = np.array(plot_sims[v_n_x])
    #         # y = (np.array(plot_sims[v_n_y]) - fitting_func_of_lam(x_davids_fit, plot_sims["Lambda"])) / np.array(
    #         #     plot_sims[v_n_y])
    #         y_ = (y - fitting_func_of_lam(x_davids_fit, plot_sims)) / y
    #         delta_y = rd.params.MdiskPP_err(y) / y
    #         ax[1].errorbar(x, y_, yerr=delta_y, label=None,
    #                        color='gray', ecolor='gray',
    #                        fmt='None', elinewidth=1, capsize=1, alpha=0.5)
    #
    #         sc = mscatter(x, y_, ax=ax[1], c=np.array(plot_sims[v_n_col]), norm=norm,
    #                       s=mss, cmap=cm, m=markers,
    #                       label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)
    #     elif dataset_name == "kiuchi":
    #         #
    #         import kiuchi as kc
    #         #
    #         v_n_x = kc.translation[x_dic["v_n"]]
    #         v_n_y = kc.translation[y_dic["v_n"]]
    #         v_n_col = kc.translation[col_dic["v_n"]]
    #         sel = kc.simulations
    #         #
    #         markers = [dic['marker'] for sim in sel.index]
    #         mss = [dic['ms'] for sim in sel.index]
    #         edgecolors = None
    #
    #         x = np.array(sel[v_n_x])
    #         y = (np.array(sel[v_n_y]))
    #         y = __apply_mod(y_dic["v_n"], y, y_dic["mod"])
    #         #
    #         ax[0].errorbar(x, y, yerr=kc.params.Mdisk_err(np.array(sel[v_n_y])), label=None,
    #                        color='gray', ecolor='gray',
    #                        fmt='None', elinewidth=1, capsize=1, alpha=0.5)
    #
    #         sc = mscatter(x, y, ax=ax[0], c=sel[v_n_col], norm=norm,
    #                       s=mss, cmap=cm, m=markers,
    #                       label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)
    #
    #         # --------------------------------------
    #
    #         x = np.array(sel[v_n_x])
    #         # y = (np.array(sel[v_n_y]) - fitting_func_of_lam(x_davids_fit, sel["Lambda"])) / np.array(sel[v_n_y])
    #         y_ = (y - fitting_func_of_lam(x_davids_fit, sel)) / y
    #         delta_y = kc.params.Mdisk_err(y) /y
    #         ax[1].errorbar(x, y_, yerr=delta_y, label=None,
    #                        color='gray', ecolor='gray',
    #                        fmt='None', elinewidth=1, capsize=1, alpha=0.5)
    #
    #         sc = mscatter(x, y_, ax=ax[1], c=sel[v_n_col], norm=norm,
    #                       s=mss, cmap=cm, m=markers,
    #                       label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)
    #     elif dataset_name == "dietrich":
    #         import dietrich as di
    #         v_n_x = di.translation[x_dic["v_n"]]
    #         v_n_y = di.translation[y_dic["v_n"]]
    #         v_n_col = di.translation[col_dic["v_n"]]
    #         sel = di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    #         #
    #         markers = [dic['marker'] for sim in sel.index]
    #         mss = [dic['ms'] for sim in sel.index]
    #         edgecolors = None
    #
    #         x = np.array(sel[v_n_x])
    #         y = (np.array(sel[v_n_y]))
    #         y = __apply_mod(y_dic["v_n"], y, y_dic["mod"])
    #
    #         ax[0].errorbar(x, y, yerr=di.params.Mdisk_err(y), label=None,
    #                        color='gray', ecolor='gray',
    #                        fmt='None', elinewidth=1, capsize=1, alpha=0.5)
    #         #
    #         sc = mscatter(x, y, ax=ax[0], c=sel[v_n_col], norm=norm,
    #                       s=mss, cmap=cm, m=markers,
    #                       label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)
    #
    #         # ---------------------------
    #         x = np.array(sel[v_n_x])
    #         # y = (np.array(sel[v_n_y]) - fitting_func_of_lam(x_davids_fit, sel["Lambda"])) / np.array(sel[v_n_y])
    #         y_ = (y - fitting_func_of_lam(x_davids_fit, sel)) / y
    #         # print(fitting_func_of_lam(x_davids_fit, sel)); exit(1)
    #         delta_y = di.params.Mdisk_err(y) / y
    #         ax[1].errorbar(x, y_, yerr=delta_y, label=None,
    #                        color='gray', ecolor='gray',
    #                        fmt='None', elinewidth=1, capsize=1, alpha=0.5)
    #         #
    #         sc = mscatter(x, y_, ax=ax[1], c=sel[v_n_col], norm=norm,
    #                       s=mss, cmap=cm, m=markers,
    #                       label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)
    #     elif dataset_name == "vincent":
    #         import vincent as vi
    #         v_n_x = vi.translation[x_dic["v_n"]]
    #         v_n_y = vi.translation[y_dic["v_n"]]
    #         v_n_col = vi.translation[col_dic["v_n"]]
    #         sel = vi.simulations
    #         #
    #         markers = [dic['marker'] for sim in sel.index]
    #         mss = [dic['ms'] for sim in sel.index]
    #         edgecolors = None
    #
    #         x = np.array(sel[v_n_x])
    #         y = (np.array(sel[v_n_y]))
    #         y = __apply_mod(y_dic["v_n"], y, y_dic["mod"])
    #         #
    #         ax[0].errorbar(x, y, yerr=vi.params.Mdisk_err(np.array(sel[v_n_y])), label=None,
    #                        color='gray', ecolor='gray',
    #                        fmt='None', elinewidth=1, capsize=1, alpha=0.5)
    #         #
    #         sc = mscatter(x, y, ax=ax[0], c=sel[v_n_col], norm=norm,
    #                       s=mss, cmap=cm, m=markers,
    #                       label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)
    #
    #         # ---------------------------
    #
    #         x = np.array(sel[v_n_x])
    #         # y = (np.array(sel[v_n_y]) - fitting_func_of_lam(x_davids_fit, sel["Lambda"])) / np.array(sel[v_n_y])
    #         y_ = (y - fitting_func_of_lam(x_davids_fit, sel)) / y
    #         delta_y = vi.params.Mdisk_err(y) / y
    #         #
    #         ax[1].errorbar(x, y_, yerr=delta_y, label=None,
    #                        color='gray', ecolor='gray',
    #                        fmt='None', elinewidth=1, capsize=1, alpha=0.5)
    #         #
    #         sc = mscatter(x, y_, ax=ax[1], c=sel[v_n_col], norm=norm,
    #                       s=mss, cmap=cm, m=markers,
    #                       label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)
    #     else:
    #         raise NameError("dataset: {} is not recognized.")


    ''' ------------------------------------------- '''
    if plot_dic["fit_panel"]:ax_ = ax[0]
    else:  ax_ = ax
    if plot_dic["fit_panel"]:
        for i in range(len(ax)):
            ax[i].tick_params(axis='both', which='both', labelleft=True,
                              labelright=False, tick1On=True, tick2On=True,
                              labelsize=12,
                              direction='in',
                              bottom=True, top=True, left=True, right=True)
            ax[i].minorticks_on()

            # limits
            if ("xmin" in plot_dic.keys() and "xmax" in plot_dic.keys()) and \
                    (plot_dic["xmin"] != None and plot_dic["xmax"] != None):
                min_, max_ = plot_dic["xmin"], plot_dic["xmax"]
            else:
                min_, max_ = md.get_minmax(x_dic["v_n"], x_arr, extra=2.)
            ax[i].set_xlim(min_, max_)
    else:
        ax_.tick_params(axis='both', which='both', labelleft=True,
                          labelright=False, tick1On=True, tick2On=True,
                          labelsize=12,
                          direction='in',
                          bottom=True, top=True, left=True, right=True)
        ax_.minorticks_on()

        # limits
        if ("xmin" in plot_dic.keys() and "xmax" in plot_dic.keys()) and \
                (plot_dic["xmin"] != None and plot_dic["xmax"] != None):
            min_, max_ = plot_dic["xmin"], plot_dic["xmax"]
        else:
            min_, max_ = md.get_minmax(x_dic["v_n"], x_arr, extra=2.)
        ax_.set_xlim(min_, max_)

    #
    if ("ymin1" in plot_dic.keys() and "ymax1" in plot_dic.keys()) and \
            (plot_dic["ymin1"] != None and plot_dic["ymax1"] != None):
        min_, max_ = plot_dic["ymin1"], plot_dic["ymax1"]
    else:
        min_, max_ = md.get_minmax(y_dic["v_n"], all_y, extra=2., oldtable=None)
    ax_.set_ylim(min_, max_)
    #
    if plot_dic["fit_panel"]:
        if ("ymin2" in plot_dic.keys() and "ymax2" in plot_dic.keys()) and \
                (plot_dic["ymin2"] != None and plot_dic["ymax2"] != None):
            min_, max_ = plot_dic["ymin2"], plot_dic["ymax2"]
        else:
            min_, max_ = md.get_minmax(y_dic["v_n"], all_y, extra=2., oldtable=None)
        ax[1].set_ylim(min_, max_)

    # scale
    if "xscale" in plot_dic.keys() and plot_dic["xscale"] != None:
        ax_.set_xscale(plot_dic["xscale"])
    if "yscale1" in plot_dic.keys() and plot_dic["yscale1"] != None:
        ax_.set_yscale(plot_dic["yscale1"])
    if plot_dic["fit_panel"]:
        if "yscale2" in plot_dic.keys() and plot_dic["yscale2"] != None:
            ax[1].set_yscale(plot_dic["yscale2"])

    # label
    if "ylabel1" in plot_dic.keys() and plot_dic["ylabel1"] != None: ax_.set_ylabel(plot_dic["ylabel1"])
    else: ax_.set_ylabel(md.get_label(y_dic["v_n"]))
    if plot_dic["fit_panel"]:
        if "ylabel2" in plot_dic.keys() and plot_dic["ylabel2"] != None: ax[1].set_ylabel(plot_dic["ylabel2"])
        else: ax[1].set_ylabel(md.get_label(y_dic["v_n"]))
        #
        if "xlabel" in plot_dic.keys() and plot_dic["xlabel"] != None: ax[1].set_xlabel(plot_dic["xlabel"])
        else: ax[1].set_xlabel(md.get_label(x_dic["v_n"]))
    else:
        if "xlabel" in plot_dic.keys() and plot_dic["xlabel"] != None: ax_.set_xlabel(plot_dic["xlabel"])
        else: ax_.set_xlabel(md.get_label(x_dic["v_n"]))

    #
    if plot_dic["fit_panel"]:
        ax[1].axhline(y=0, linestyle='-', linewidth=0.4, color='black')

    # colobar
    ax_.legend(fancybox=True, loc='upper center',  # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
              shadow=False, ncol=2, fontsize=9,
              framealpha=0., borderaxespad=0.)
    if plot_dic["fit_panel"]:
        clb = fig.colorbar(sc, ax=ax.ravel().tolist())
    else:
        clb = fig.colorbar(sc, ax=ax_)
        plt.tight_layout()
    clb.ax.set_title(md.get_label(col_dic["v_n"]), fontsize=11)
    clb.ax.tick_params(labelsize=11)
    # plt.subplots_adjust(bottom=0.20, right=1.0, left=0.15)
    # plt.tight_layout()
    # if plot_dic["fit_panel"]:
    #     pass
    # else:
    #     plt.delaxes(ax[1])
    print("plotted: \n")
    print(__outplotdir__ + figname)
    plt.savefig(__outplotdir__ + figname, dpi=256)
    plt.close()



def ye_fit_func(x, v):
    a, b, c = x
    return a*(v.M1/v.M2)*(1. + c*v.C1) + \
           a*(v.M2/v.M1)*(1. + c*v.C2) + b

def ye_fit_coeff():
    a = 0.000259359739766
    b = 0.353384617108
    c = -2114.98171227
    return np.array((a,b,c))


def summary_cumulative_plots2():
    """

    :return:

    """

    ''' ----------- DISK MASS --------- '''

    import models_dietrich2016 as di
    import models_vincent as vi
    import models_radice as rd
    import groups as md
    import models_kiuchi as ki
    import models_bauswein2013 as bs
    import models_lehner2016 as lh
    import models_hotokezaka as hz
    import models_dietrich_ujevic2016 as du

    ''' ----------- '''

    datasets = {}
    datasets["kiuchi"] =    {'marker': "X", "ms": 20, "models": ki.simulations, "data":ki, "err":ki.params.Mdisk_err, "label":"Kiuchi+2019"}
    datasets["radice"] =    {'marker': '*', "ms": 20, "models": rd.simulations[rd.fiducial], "data":rd, "err":rd.params.MdiskPP_err, "label":"Radice+2018"}
    datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": di.simulations[di.mask_for_with_sr], "data":di, "err":di.params.Mdisk_err, "label":"Dietrich+2016"}
    datasets["vincent"] =   {'marker': 'v', 'ms': 20, "models": vi.simulations, "data":vi, "err":vi.params.Mdisk_err, "label":"Vincent+2019"} # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets['our'] =       {'marker': 'o', 'ms': 20, "models": md.groups, "data": md, "err":"v_n", "label":"We+inf"}
    #
    x_dic = {"v_n": "q", "err": None, "mod": {}, "deferr": None}
    y_dic = {"v_n": "Mdisk3D", "err": "ud", "mod": {}, "deferr": 0.2}
    col_dic = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {"vmin": 350, "vmax": 900.0,
                # "vmin": 1., "vmax": 2.0,
                "cmap": "jet",  # "tab10",
                "label": None, "alpha": 0.8,
                "ms": 30.,
                "yscale1": None,
                "yscale2": None,
                # "xmin":1e-5, "xmax":1e-1,
                # "ymin":-0.1, "ymax":0.4,
                "ymin1": 0.0, "ymax1": 0.6,
                "ymin2": -1.0, "ymax2": 1.4,
                "xlabel": "$M_1/M_2$",
                "ylabel1": r"$M_{\rm disk}$",
                "ylabel2": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
                           "figname": "final_summary_disk_mass.png"
                }

    # Fit
    from make_fit import fitting_function_mdisk, fitting_coeffs_mdisk
    x_davids_fit = fitting_coeffs_mdisk()
    fitting_func_of_lam = fitting_function_mdisk  # takes x_davids_fit, lam_array

    #
    # plot_mdisk_minus_mdiskfit_vs_q4(x_dic, y_dic, col_dic, plot_dic, x_davids_fit, fitting_func_of_lam, datasets)
    #

    ''' ----------- EJECTA MASS --------- '''

    datasets = {}
    datasets["kiuchi"] =    {'marker': "X", "ms": 20, "models": ki.simulations, "data":ki, "err":ki.params.Mej_err, "label":"Kiuchi+2019"}
    datasets["radice"] =    {'marker': '*', "ms": 20, "models": rd.simulations[rd.fiducial], "data":rd, "err":rd.params.Mej_err, "label":"Radice+2018"}
    # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": di.simulations[di.mask_for_with_sr], "data":di, "err":di.params.Mej_err, "label":"Dietrich+2016"}
    datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data":du, "err":du.params.Mej_err, "label":"Dietrich+Ujevic"}
    datasets["vincent"] =   {'marker': 'v', 'ms': 20, "models": vi.simulations, "data":vi, "err":vi.params.Mej_err, "label":"Vincent+2019"} # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets["bauswein"] =  {'marker': 's', 'ms': 20, "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label":"Bauswein+2013", "fit":False}
    datasets["lehner"] =    {'marker': 'P', 'ms': 20, "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": "Lehner+2016", "fit": False}
    datasets["hotokezaka"]= {'marker': '>', 'ms': 20, "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": "Hotokezaka+2013", "fit": False}
    datasets['our'] =       {'marker': 'o', 'ms': 20, "models": md.groups, "data": md, "err":"v_n", "label":"We+inf"}


    x_dic   = {"v_n": "q", "err": None, "mod": {}, "deferr": None}
    y_dic   = {"v_n": "Mej_tot-geo", "err": "ud", "deferr": 0.2, "mod":{}}
        # {
        #     "mult":[2.], "dev":["Mchirp"]
        # }}
    col_dic = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {"vmin": 350, "vmax": 900.0,
                # "vmin": 1., "vmax": 2.0,
                "cmap": "jet",  # "tab10",
                "label": None, "alpha": 0.8,
                "ms": 30.,
                # "xscale":"log",
                "yscale1": "log",
                "yscale2": None,
                # "xmin":1e-5, "xmax":1e-1,
                # "ymin":-0.1, "ymax":0.4,
                "ymin1": 1e-4, "ymax1": 8e-1,
                "ymin2": -4.5, "ymax2": 2.4,
                "xlabel": "$M_1/M_2$",
                "ylabel1": r"$M_{\rm ej}$ $[M_{\odot}]$",
                "ylabel2": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
                "figname": "final_summary_ejecta_mass.png",
                "fit_panel": False
                }

    # Fit
    from make_fit import fitting_function_mej
    from make_fit import fitting_coeffs_mej
    x_davids_fit = fitting_coeffs_mej()  # radice()
    fitting_func_of_lam = fitting_function_mej  # takes x_davids_fit, lam_array

    #
    plot_mdisk_minus_mdiskfit_vs_q4(x_dic, y_dic, col_dic, plot_dic, x_davids_fit, fitting_func_of_lam, datasets)
    #

    ''' ----------- EJECTA VELOCITY --------- '''

    datasets = {}
    # datasets["kiuchi"] =    {'marker': "X", "ms": 20, "models": ki.simulations, "data":ki, "err":ki.params.Mej_err, "label":"Kiuchi+2019"}
    datasets["radice"] =    {'marker': '*', "ms": 20, "models": rd.simulations[rd.fiducial], "data":rd, "err":rd.params.vej_err, "label":"Radice+2018"}
    datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": di.simulations[di.mask_for_with_sr], "data":di, "err":di.params.vej_err, "label":"Dietrich+2016"}
    datasets["vincent"] =   {'marker': 'v', 'ms': 20, "models": vi.simulations, "data":vi, "err":vi.params.vej_err, "label":"Vincent+2019"} # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets['our'] =       {'marker': 'o', 'ms': 20, "models": md.groups, "data": md, "err":"v_n", "label":"We+inf"}

    x_dic   = {"v_n": "q", "err": None, "mod": {}, "deferr": None}
    y_dic   = {"v_n": "vel_inf_ave-geo", "err": "ud", "deferr": 0.2, "mod":{}}
        # {
        #     "mult":[2.], "dev":["Mchirp"]
        # }}
    col_dic = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {"vmin": 350, "vmax": 900.0,
                # "vmin": 1., "vmax": 2.0,
                "cmap": "jet",  # "tab10",
                "label": None, "alpha": 0.8,
                "ms": 30.,
                # "xscale":"log",
                "yscale1": None,
                "yscale2": None,
                # "xmin":1e-5, "xmax":1e-1,
                # "ymin":-0.1, "ymax":0.4,
                "ymin1": 0.1, "ymax1": 0.4,
                "ymin2": -1.0, "ymax2": 0.8,
                "xlabel": "$M_1/M_2$",
                "ylabel1": r"$\upsilon_{\rm ej}$ $[c]$",
                "ylabel2": r"$\Delta \upsilon_{\rm ej} / \upsilon_{\rm ej}$",
                "figname": "final_summary_ejecta_velovity.png"
                }

    # Fit
    from make_fit import fitting_coeffs_vinf, fitting_function_vinf
    x_davids_fit = fitting_coeffs_vinf()  # radice()
    fitting_func_of_lam = fitting_function_vinf  # takes x_davids_fit, lam_array

    #
    # plot_mdisk_minus_mdiskfit_vs_q4(x_dic, y_dic, col_dic, plot_dic, x_davids_fit, fitting_func_of_lam, datasets)

    ''' ----------- EJECTA Electron Fraction --------- '''

    datasets = {}
    # datasets["kiuchi"] =    {'marker': "X", "ms": 20, "models": ki.simulations, "data":ki, "err":ki.params.Mej_err, "label":"Kiuchi+2019"}
    datasets["radice"] =    {'marker': '*', "ms": 20, "models": rd.simulations[rd.fiducial], "data":rd, "err":rd.params.Yeej_err, "label":"Radice+2018"}
    # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": di.simulations[di.mask_for_with_sr], "data":di, "err":di.params.vej_err, "label":"Dietrich+2016"}
    datasets["vincent"] =   {'marker': 'v', 'ms': 20, "models": vi.simulations, "data":vi, "err":vi.params.Yeej_err, "label":"Vincent+2019"} # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets['our'] =       {'marker': 'o', 'ms': 20, "models": md.groups, "data": md, "err":"v_n", "label":"We+inf"}

    x_dic   = {"v_n": "q", "err": None, "mod": {}, "deferr": None}
    y_dic   = {"v_n": "Ye_ave-geo", "err": "ud", "deferr": 0.2, "mod":{}}
        # {
        #     "mult":[2.], "dev":["Mchirp"]
        # }}
    col_dic = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {"vmin": 350, "vmax": 900.0,
                # "vmin": 1., "vmax": 2.0,
                "cmap": "jet",  # "tab10",
                "label": None, "alpha": 0.8,
                "ms": 30.,
                # "xscale":"log",
                "yscale1": None,
                "yscale2": None,
                # "xmin":1e-5, "xmax":1e-1,
                # "ymin":-0.1, "ymax":0.4,
                "ymin1": 0.0, "ymax1": 0.4,
                "ymin2": -1.0, "ymax2": 0.8,
                "xlabel": "$M_1/M_2$",
                "ylabel1": r"$Ye_{\rm ej}$ $[c]$",
                "ylabel2": r"$\Delta Ye_{\rm ej} / Ye_{\rm ej}$",
                "figname": "final_summary_ejecta_electron_fraction.png"
                }

    # Fit
    x_davids_fit = ye_fit_coeff()  # radice()
    fitting_func_of_lam = ye_fit_func  # takes x_davids_fit, lam_array

    #
    # plot_mdisk_minus_mdiskfit_vs_q4(x_dic, y_dic, col_dic, plot_dic, x_davids_fit, fitting_func_of_lam, datasets)


if __name__ == "__main__":
    print("hi")
     # plot_summary_and_fit_unique()
    # sel=fit_ejecta_mass_new_data()
    # plot_ejecta_mass_fit(sel)

    # plot_summary_and_fit_unique()
    # sel=fit_velocity_new_data()
    # plot_velocity_fit(sel)

    # sel = fit_thetarms_new_data()
    # plot_thetarms_fit(sel)

    # print_disk_mass()

    # plot_disk_mass_evol()
    # plot_summary_and_fit_unique()

    # plot_summary_and_fit_unique()
    # plot_mdisk_minus_mdiskfit_vs_q3()

    summary_cumulative_plots2()