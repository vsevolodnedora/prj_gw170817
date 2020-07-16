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

#from legacy.prj_visc_ej import ax

matplotlib.use('agg')
import matplotlib.pyplot as plt

from data import ADD_METHODS_ALL_PAR

__outplotdir__ = "../figs/all3/plot_postdyn_ej/"
if not os.path.isdir(__outplotdir__):
    os.mkdir(__outplotdir__)

''' data loading '''


def get_outflow_data(path, sim, det, mask, v_n):
    fpath = path + sim + '/' + "outflow_{:d}".format(det) + '/' + mask + '/' + v_n
    if not os.path.isfile(fpath):
        raise IOError("File not found for det:{} mask:{} v_n:{} -> {}"
                      .format(det, mask, v_n, fpath))
    # loading acii file
    if not v_n.__contains__(".h5"):
        data = np.loadtxt(fpath)
        return data
    # loading nucle data
    if v_n == "yields.h5":
        dfile = h5py.File(fpath, "r")
        v_ns = []
        for _v_n in dfile:
            v_ns.append(_v_n)
        assert len(v_ns) == 3
        assert "A" in v_ns
        assert "Y_final" in v_ns
        assert "Z" in v_ns
        table = np.vstack((np.array(dfile["A"], dtype=float),
                           np.array(dfile["Z"], dtype=float),
                           np.array(dfile["Y_final"], dtype=float))).T
        return table
    # loading correlation files
    if v_n.__contains__(".h5"):

        dfile = h5py.File(fpath, "r")
        v_ns = []
        for _v_n in dfile:
            v_ns.append(_v_n)
        assert len(v_ns) == 3
        assert "mass" in v_ns
        mass = np.array(dfile["mass"])
        v_ns.remove('mass')
        xarr = np.array(dfile[v_ns[0]])
        yarr = np.array(dfile[v_ns[1]])
        if len(xarr) == mass.shape[1] and len(yarr) == mass.shape[0]:
            table = UTILS.combine(xarr, yarr, mass)
        else:
            table = UTILS.combine(yarr, xarr, mass)
        return table

    raise ValueError("Loading data method for ourflow data is not found")

def get_time_data_arrs(path, sim, v_n, det=None, mask=None):


    if v_n == "Mej":
        table = get_outflow_data(path, sim, det=det, mask=mask, v_n="total_flux.dat")
        # print('time', len(table[:,0]))
        # print('mass', len(table[:, 2]))
        return table[:, 0], table[:, 2]
    elif v_n == "Mej_tot-bern_geoend":
        table = get_outflow_data(path, sim, det=0, mask="bern_geoend", v_n="total_flux.dat")
        print('time', len(table[:, 0]))
        print('mass', len(table[:, 2]))
        return table[:, 0], table[:, 2]
    elif v_n == "vel_inf_ave-bern_geoend":
        # table = self.get_outflow_data(det=0, mask="geo", v_n="total_flux.dat")
        # print(table[:,0])
        table = get_outflow_timecorr(det=0, mask="bern_geoend", v_n="vel_inf")
        time_arr = table[0, 1:] * 1.e-3  # for some reason it is in ms #
        vel_inf = table[1:, 0]
        masses = table[1:, 1:]
        assert len(time_arr) == len(masses[0, :])
        assert len(vel_inf) == len(masses[:, 0])

        vinf_aves = []
        for i in range(len(time_arr)):
            # compute average of ejecta cumulativly step by step for every timestep
            vinf_aves.append(np.sum(np.cumsum(masses, axis=1)[:, i] * vel_inf) / np.cumsum(np.sum(masses, axis=0))[i])
        vinf_aves = np.array(vinf_aves)

        # for t, m, v in zip(time_arr, np.cumsum(np.sum(masses,axis=0)), vinf_aves):
        #     print(t, m, v)
        # exit(1)
        # for i, t in enumerate(time_arr):
        #     t_masses = np.cumsum(masses[:, :i], axis=1)
        # exit(1)

        # if self.sim == "BLh_M11041699_M0_LK_LR":
        #     print(time_arr, vinf_aves)
        # exit(1)

        return time_arr[~np.isnan(vinf_aves)], vinf_aves[~np.isnan(vinf_aves)]
    elif v_n == "Ye_ave-bern_geoend":
        table = self.get_outflow_timecorr(det=0, mask="bern_geoend", v_n="Y_e")
        time_arr = table[0, 1:] * 1.e-3  # for some reason it is in ms #
        assert time_arr.max() < 1. and time_arr.max() > 0.01
        ye = table[1:, 0]
        masses = table[1:, 1:]
        assert len(time_arr) == len(masses[0, :])
        assert len(ye) == len(masses[:, 0])

        ye_aves = []
        for i in range(len(time_arr)):
            # compute average of ejecta cumulativly step by step for every timestep
            ye_aves.append(
                np.sum(np.cumsum(masses, axis=1)[:, i] * ye) / np.cumsum(np.sum(masses, axis=0))[i])
        ye_aves = np.array(ye_aves)

        return time_arr[~np.isnan(ye_aves)], ye_aves[~np.isnan(ye_aves)]
    elif v_n == "theta_rms-bern_geoend":
        table = self.get_outflow_timecorr(det=0, mask="bern_geoend", v_n="theta")
        time_arr = table[0, 1:] * 1.e-3  # for some reason it is in ms #
        assert time_arr.max() < 1. and time_arr.max() > 0.01
        theta = table[1:, 0]
        masses = table[1:, 1:]
        assert len(time_arr) == len(masses[0, :])
        assert len(theta) == len(masses[:, 0])

        theta_rmss = []
        for i in range(len(time_arr)):
            # compute average of ejecta cumulativly step by step for every timestep
            theta_rmss.append(
                np.sqrt(np.sum(np.cumsum(masses, axis=1)[:, i] * theta ** 2)) \
                / np.cumsum(np.sum(masses, axis=0))[i])
        theta_rmss = np.array(theta_rmss)

        # for t, m, theta in zip(time_arr, np.cumsum(np.sum(masses,axis=0)), theta_rmss):
        #     print(t, m, theta)
        # print(theta_rmss[~np.isnan(theta_rmss)])
        # exit(1)

        return time_arr[~np.isnan(theta_rmss)], theta_rmss[~np.isnan(theta_rmss)]

    else:
        raise NameError("v_n: {} is not recognized. No method setup."
                        .format(v_n))


''' -- modules --- '''

def plot_total_ejecta_flux(tasks, plotdic):

    fig = plt.figure(figsize=plotdic["figsize"])
    ax = fig.add_subplot(111)
    #
    # labels

    for task in tasks:
        #
        # o_data = ADD_METHODS_ALL_PAR(task["sim"])
        # o_data.path = task["path"]
        times, masses = get_time_data_arrs(task["path"], task["sim"], task["v_n"], task["det"], task["mask"])
            #o_data.get_time_data_arrs(task["v_n"], det=task["det"], mask=task["mask"])
        #
        #tmerg = o_data.get_par("tmerg")
        if task["v_n"] == "Mej" and plotdic["yscale"] != "log":
            masses_pl = masses #* 1e2  # 1e-2 Msun
        else:
            masses_pl = masses
        #times = (times - tmerg)  # ms
        if not "plot" in task.keys():
            ax.plot(times * 1e3, masses_pl, color=task["color"], ls=task["ls"], lw=task["lw"], alpha=task["alpha"], label=task["label"])
        else:
            ax.plot(times * 1e3, masses_pl, **task["plot"])

        if len(task["ext"].keys()) > 0:
            dic = task["ext"]
            # {"type": "poly", "order": 4, "t1": 20, "t2": None, "show": True}
            if dic["t1"] != None:
                masses = masses[times > float(dic["t1"]) / 1.e3]
                times = times[times > float(dic["t1"]) / 1.e3]
            if dic["t2"] != None:
                masses = masses[times <= float(dic["t2"]) / 1.e3]
                times = times[times <= float(dic["t2"]) / 1.e3]
            assert len(times) > 0
            #
            # new_x = np.array(dic["xarr"]) / 1.e3
            # if dic["type"] == "poly":
            #     fitx, fity = fit_polynomial(times, masses, dic["order"], np.zeros(0, ), new_x)
            #     if task["v_n"] == "Mej" and plotdic["yscale"] != "log": fity = fity * 1e2  # 1e-2 Msun
            #     if len(dic["plot"].keys()) > 0:
            #         ax.plot(fitx * 1e3, fity, **dic["plot"])
            # elif dic["type"] == "int1d":
            #     fitx = new_x
            #     fity = interpolate.interp1d(times, masses, kind="linear", fill_value="extrapolate")(new_x)
            #     if task["v_n"] == "Mej" and plotdic["yscale"] != "log": fity = fity * 1e2  # 1e-2 Msun
            #     if len(dic["plot"].keys()) > 0:
            #         ax.plot(fitx * 1e3, fity, **dic["plot"])


    if "add_line" in plotdic.keys() and len(plotdic["add_line"]) > 0:
        for entry in plotdic["add_line"]:
            ax.plot([-101, -102], [-101,-102], **entry)

    # if plotdic["add_legend"]:
    #     tmp = {"color": "gray", "marker": "x", "markersize": 5, "linestyle": ':', "linewidth":0.5, "label":"Linear extrapolation"}
    #     ax.plot([-1, -1],[-2., -2], **tmp)

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
    ax.set_title(plotdic["title"])

    # LEGENDS
    if "add_legend" in plotdic.keys() and len(plotdic["add_legend"].keys())>0:
        han, lab = ax.get_legend_handles_labels()
        ax.add_artist(ax.legend(han[:-2], lab[:-2], **plotdic["legend"])) # default
        # tmp = copy.deepcopy(plotdic["legend"])
        # tmp["loc"] = "upper left"
        # tmp["bbox_to_anchor"] = (0., 1.)
        ax.add_artist(ax.legend(han[-2:], lab[-2:], **plotdic["add_legend"])) # for extapolation
    else:
        ax.legend(**plotdic["legend"])

    plt.tight_layout()
    #

    print("plotted: \n")
    print(plotdic["figname"])
    plt.savefig(plotdic["figname"], dpi=128)
    if plotdic["savepdf"]: plt.savefig(plotdic["figname"].replace(".png", ".pdf"))
    plt.close()

''' --- tasks --- '''

def task_compare_mass_flux():
    v_n = "Mej"
    path = "/data01/numrel/vsevolod.nedora/postprocessed_radice2/"
    path2 = "/data01/numrel/vsevolod.nedora/postprocessed4/"
    task = [
        # DD2
        {"sim": "DD2_M13641364_M0_LK_SR_R04", "path": path2, "v_n": v_n, "t2": -1, "ext": {},
         "plot": {"color": "blue", "lw": 2.0, "ls": "-", "alpha": 1., "label": r"DD2* $(1.365+1.365)M_{\odot}$"}},
        # DD2
        {"sim": "DD2_M135135_LK", "path": path, "v_n": v_n, "t2": -1, "ext": {},
         "plot": {"color": "magenta", "lw": 0.8, "ls": ":", "alpha": 1.}},
        {"sim": "DD2_M135135_M0", "path": path, "v_n": v_n, "t2": -1, "ext": {},
         "plot": {"color": "magenta", "lw": 0.8, "ls": "-", "alpha": 1., "label": r"DD2 $(1.35+1.35)M_{\odot}$"}},
        # DD2
        {"sim": "DD2_M140120_LK", "path": path, "v_n": v_n, "t2": -1, "ext": {},
         "plot": {"color": "blue", "lw": 0.8, "ls": ":", "alpha": 1.}},
        {"sim": "DD2_M140120_M0", "path": path, "v_n": v_n, "t2": -1, "ext": {},
         "plot": {"color": "blue", "lw": 0.8, "ls": "-", "alpha": 1., "label": r"DD2 $(1.4+1.2)M_{\odot}$"}},
        # LS220
        {"sim": "LS220_M135135_LK", "path":path, "v_n":v_n, "t2":-1, "ext":{},
            "plot":{"color":"orange", "lw": 0.8, "ls":":", "alpha":1.}},
        {"sim": "LS220_M135135_M0", "path": path, "v_n": v_n, "t2": -1, "ext": {},
            "plot": {"color": "orange", "lw": 0.8, "ls": "-", "alpha": 1., "label":r"LS220 $(1.35+1.35)M_{\odot}$"}},
        # LS220
        {"sim": "LS220_M140120_LK", "path": path, "v_n": v_n, "t2": -1, "ext": {},
         "plot": {"color": "red", "lw": 0.8, "ls": ":", "alpha": 1.}},
        {"sim": "LS220_M140120_M0", "path": path, "v_n": v_n, "t2": -1, "ext": {},
         "plot": {"color": "red", "lw": 0.8, "ls": "-", "alpha": 1., "label": r"LS220 $(1.4+1.2)M_{\odot}$"}},
        # SFHo
        {"sim": "SFHo_M135135_LK", "path": path, "v_n": v_n, "t2": -1, "ext": {},
         "plot": {"color": "green", "lw": 0.8, "ls": ":", "alpha": 1.}},
        {"sim": "SFHo_M135135_M0", "path": path, "v_n": v_n, "t2": -1, "ext": {},
         "plot": {"color": "green", "lw": 0.8, "ls": "-", "alpha": 1., "label": r"SFHo $(1.35+1.35)M_{\odot}$"}},
        # SFHo
        {"sim": "BHBlp_M140120_LK", "path": path, "v_n": v_n, "t2": -1, "ext": {},
         "plot": {"color": "olive", "lw": 0.8, "ls": ":", "alpha": 1.}},
        {"sim": "BHBlp_M140120_M0", "path": path, "v_n": v_n, "t2": -1, "ext": {},
         "plot": {"color": "olive", "lw": 0.8, "ls": "-", "alpha": 1., "label": r"BHBlp $(1.4+1.2)M_{\odot}$"}},
    ]

    plot_dic = {
        "type": "long",
        "figsize": (6., 5.5),
        "xmin": 15., "xmax": 60.,
        "ymin": 0, "ymax": 2.5,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "linear",
        "xlabel": r"$t$ [ms]",# r"$t-t_{\rm merg}$ [ms]",
        "ylabel": r"$M_{\rm ej}$ $[M_{\odot}]$",# $[10^{-2}M_{\odot}]$",
        "legend": {"fancybox": False, "loc": 'lower right',
                   #"bbox_to_anchor":(1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon":False},
        "add_legend":{"fancybox": False, "loc": 'upper left',
                       #"bbox_to_anchor":(1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                       "shadow": "False", "ncol": 1, "fontsize": 13,
                       "framealpha": 0., "borderaxespad": 0., "frameon":False},
        "title": r"Wind cumulative mass",
        "figname": __outplotdir__ + "mej_flux_compare_LKvsM0.png",
        "fontsize":13,
        "savepdf":True,
        "add_line": [{"color":"gray", "ls":"-", "lw":0.8, "label":"Leackage + M0"},
                     {"color":"gray", "ls":":", "lw":0.8, "label":"Leackage"}],
    }

    for t in task: t["det"] = 0
    for t in task: t["v_n"] = "Mej"
    for dic in task: dic["mask"] = "theta60_geoend"
    plot_dic["ymin"] = 1e-5 #0
    plot_dic["ymax"] = 4e-4 #0.15
    plot_dic["yscale"] = "log"
    plot_dic["title"] = r"$\nu$-wind assuming $\theta > 60\deg$"

    plot_total_ejecta_flux(task, plot_dic)

if __name__ == '__main__':
    task_compare_mass_flux()