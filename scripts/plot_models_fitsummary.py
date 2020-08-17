from __future__ import division

import numpy as np
import pandas as pd
import sys
import os
import copy
import h5py
import csv
import scipy.stats as st
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

from collections import OrderedDict

from matplotlib.colors import LogNorm, Normalize
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from model_sets import models as ourmd

__outplotdir__ = "../figs/all3/plot_dynej_summary2/"
if not os.path.isdir(__outplotdir__):
    os.mkdir(__outplotdir__)

from make_fit2 import Fitting_Coefficients as fit_coefs, Fitting_Functions as fit_funcs

""" ----------------------------| COMPONENTS |----------------------------- """

def mscatter(x, y, ax=None, m=None, **kw):

    # for nan in data
    if "c" in kw.keys():
        for ix, iy, im, c, s in zip(x, y, m, kw["c"], kw["s"]):
            if isinstance(c, float) and np.isnan(c):
                # print(ix, iy, im, s)
                ax.plot(ix, iy, color="gray", marker=im, markersize=s/5, alpha=0.6)

    import matplotlib.markers as mmarkers
    if not ax: ax = plt.gca()

    sc = ax.scatter(x, y, **kw)

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

    return sc


""" -----------------------------| MODULUS |------------------------------- """

def plot_subplots_for_fits(plot_dic, subplot_dics, fit_dics, model_dics):

    # assuertion
    assert len(subplot_dics.keys()) == len(fit_dics.keys())

    # assign subplots
    fig, axes = plt.subplots(**plot_dic["subplots"])
    if "subplot_adjust" in plot_dic.keys():
        fig.subplots_adjust(**plot_dic["subplot_adjust"])
    #if not isinstance(axes, list): axes = [axes]

    # main loop
    for ax, key in zip(axes, subplot_dics.keys()):

        fit_dic = fit_dics[key]
        subplot_dic = subplot_dics[key]

        # labels
        if "labels" in subplot_dic.keys() and subplot_dic["labels"]:
            for model_dic in model_dics.itervalues():
                ax.scatter([-100], [-100], marker=model_dic['marker'], s=model_dic['ms'], edgecolor=model_dic['edgecolor'],
                              facecolor=model_dic['facecolor'], alpha=model_dic['alpha'], label=model_dic['label'])

        # main plotting
        all_x, all_y, all_y_ = [], [], []
        for model_dic in model_dics.itervalues():
            #
            d_cl = model_dic["data"]  # md, rd, ki ...
            err = model_dic["err"]  # err lambda(y)
            models = model_dic["models"]  # models DataFrame
            x_dic = model_dic["x_dic"]
            y_dic = model_dic["y_dic"]
            #            #
            if "v_n_x" in subplot_dic: model_dic["v_n_x"] = subplot_dic["v_n_x"] # overwrite
            if "v_n_y" in subplot_dic: model_dic["v_n_y"] = subplot_dic["v_n_y"] # overwrite
            #
            if x_dic["v_n"] == "Mej_tot-geo_fit" and y_dic["v_n"] == "Mej_tot-geo":
                func, coeffs = fit_dic["func"], fit_dic["coeffs"]
                mej = func(coeffs, models, v_n=d_cl.translation["Mej_tot-geo"]) / 1.e3  # 1e-3 Msun -> Msun
                x = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, mej)
                y = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models)
            elif x_dic["v_n"] == "vel_inf_ave-geo_fit" and y_dic["v_n"] == "vel_inf_ave-geo":
                func, coeffs = fit_dic["func"], fit_dic["coeffs"]
                mej = func(coeffs, models)  # 1e-3 Msun -> Msun
                x = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, mej)
                y = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models)
            elif x_dic["v_n"] == "Mdisk3D_fit" and y_dic["v_n"] == "Mdisk3D":
                func, coeffs = fit_dic["func"], fit_dic["coeffs"]
                mdisk = func(coeffs, models)  # 1e-3 Msun -> Msun
                x = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, mdisk)
                y = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models)
            elif x_dic["v_n"] == "Ye_ave-geo_fit" and y_dic["v_n"] == "Ye_ave-geo":
                func, coeffs = fit_dic["func"], fit_dic["coeffs"]
                mej = func(coeffs, models)  # 1e-3 Msun -> Msun
                x = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, mej)
                y = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models)
            elif x_dic["v_n"] == "theta_rms-geo_fit" and y_dic["v_n"] == "theta_rms-geo":
                func, coeffs = fit_dic["func"], fit_dic["coeffs"]
                mej = func(coeffs, models)  # 1e-3 Msun -> Msun
                x = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, mej)
                y = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models)
            else:
                x = d_cl.get_mod_data(x_dic["v_n"], x_dic["mod"], models)
                y = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models)
            #
            x = np.array(x)
            y = np.array(y)
            #
            edgecolors = [model_dic['edgecolor'] for sim in models.index]
            facecolors = [model_dic['facecolor'] for sim in models.index]
            markers = [model_dic['marker'] for sim in models.index]
            mss = [model_dic['ms'] for sim in models.index]
            #
            # sc = mscatter(x, y, ax=ax, s=mss, m=markers, label=None, alpha=plot_dic['alpha'],
            #               edgecolor=edgecolors, facecolor=facecolors)
            #
            func, coeffs = fit_dic["func"], fit_dic["coeffs"]
            fitted_values = func(coeffs, models, v_n=d_cl.translation[y_dic["v_n"]])
            #ss
            if y_dic["v_n"] == "Mej_tot-geo": fitted_values = fitted_values / 1.e3  # Tims fucking fit
            #
            y_from_fit = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, fitted_values)

            # y_ = np.array((y - y_from_fit) / y)
            if "method" in fit_dic.keys():
                if fit_dic["method"] == "delta":
                    y_ = np.array(y - y_from_fit)
                elif fit_dic["method"] == "normdelta":
                    y_ = np.array((y - y_from_fit) / y)
                else:
                    raise NameError("Only 'detla' and 'normdelta' allowed for 'method' in fitdic Given;{}"
                                    .format(fit_dic["method"]))
            else:
                y_ = np.array((y - y_from_fit) / y)  # delta_y = err / y

            sc = mscatter(x, y_, ax=ax, s=mss,
                          m=markers, label=None, alpha=model_dic['alpha'], edgecolor=edgecolors, facecolors=model_dic['facecolor'])

            if model_dic["plot_errorbar"]:
                if err == "v_n":
                    yerr = models["err-" + y_dic["v_n"]]
                    yerr = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, yerr)
                elif err == None:
                    yerr = np.zeros(len(y))
                else:
                    yerr = err(y)
                    # err = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, err)
                if "method" in fit_dic.keys():
                    if fit_dic["method"] == "delta":
                        delta_y = yerr
                    elif fit_dic["method"] == "normdelta":
                        delta_y = yerr / y
                    else:
                        raise NameError("Only 'detla' and 'normdelta' allowed for 'method' in fitdic Given;{}"
                                        .format(fit_dic["method"]))
                else:
                    delta_y = yerr / y  # delta_y = err / y
                #print(" (y - y_from_fit) / y : ")
                #print(y_)
                ax.errorbar(x, y_, yerr=delta_y, label=None,
                            color='gray', ecolor='gray', fmt='None', elinewidth=1, capsize=1, alpha=0.5)
            #
            all_x.append(x)
            all_y.append(y)
            all_y_.append(y_)

        # plot overall error
        if "add_error_bar" in plot_dic.keys() and len(plot_dic["add_error_bar"].keys()) > 0:
            all_x = np.concatenate(all_x)
            all_y = np.concatenate(all_y)
            all_y_ = np.concatenate(all_y_)
            #
            all_x = all_x[~np.isnan(all_y_)]
            all_y_ = all_y_[~np.isnan(all_y_)]
            #
            dic = plot_dic["add_error_bar"]
            #
            mean = np.mean(all_y_)
            median = np.median(all_y_)
            if "mean" in dic.keys() and dic["mean"]:
                ax.axhline(y=mean, **dic["mean"])

            if "median" in dic.keys() and dic["median"]:
                ax.axhline(y=median, **dic["median"])

            if "width" in dic.keys():# and len(dic["width"]>0):
                if dic["width"] == "1sigma":
                    width = np.std(all_y_)
                    y1 = np.full(len(all_y_), mean + width)
                    y2 = np.full(len(all_y_), mean - width)
                    ax.fill_between(all_x, y2, y1, **dic["fill_between"])
                    #print(all_y_)
                    print(mean, mean + width,mean - width)#;exit(1)

            if "confinterv" in dic.keys():

                # weights w / w.sum()
                w = np.full(len(all_y_), 1./len(all_y_))
                rv = st.rv_discrete(values=(all_y_, w))
                median = rv.median()
                interval = rv.interval(dic["confinterv"])

                _x = np.array([all_x.min(), all_x.max()])
                ax.fill_between(_x, [interval[0], interval[0]],
                                    [interval[1], interval[1]], **dic["fill_between"])

    # tend to subplots
    for axi, key in zip(axes, subplot_dics.keys()):
        subplotdic = subplot_dics[key]
        #
        axi.set_yscale(subplotdic["yscale"])
        axi.set_xscale(subplotdic["xscale"])
        #
        if "xlabel" in subplotdic.keys() and subplotdic["xlabel"] != None:
            axi.set_xlabel(subplotdic["xlabel"], fontsize=subplotdic["fontsize"])  # , fontsize=11)
        if "ylabel" in subplotdic.keys() and subplotdic["ylabel"] != None:
            axi.set_ylabel(subplotdic["ylabel"], fontsize=subplotdic["fontsize"])  # , fontsize=11)
        #
        axi.set_xlim(subplotdic["xmin"], subplotdic["xmax"])
        axi.set_ylim(subplotdic["ymin"], subplotdic["ymax"])
        #
        if "tick_params" in subplotdic.keys() and len(subplotdic["tick_params"].keys())>0:
            axi.tick_params(**subplotdic["tick_params"])
        axi.minorticks_on()
        #
        if "text" in subplotdic.keys():
            subplotdic["text"]["transform"] = axi.transAxes
            axi.text(**subplotdic["text"])
        #
        if "plot_zero" in subplotdic.keys() and subplotdic["plot_zero"]:
            axi.axhline(y=0, linestyle=':', linewidth=0.4, color='black')

        if "plot_diagonal" in subplotdic.keys() and subplotdic["plot_diagonal"]:
            axi.plot([0, 100], [0, 100], linestyle=':', linewidth=0.4, color='black',label="fit")

        if "hline" in subplotdic.keys() and len(subplotdic["hline"].keys()) > 0:
            axi.axhline(**subplotdic["hline"])

        if "legend" in subplotdic.keys() and len(subplotdic["legend"].keys()) > 0:
            axi.legend(**subplotdic["legend"])

    # for the whole plot
    if "commonaxislabel" in plot_dic.keys() and plot_dic["commonaxislabel"] != None:
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.xlabel(plot_dic["xlabel"], fontsize=plot_dic["fontsize"], labelpad=20)
        plt.ylabel(plot_dic["ylabel"], fontsize=plot_dic["fontsize"], labelpad=20)
        plt.xticks([], [])
        plt.yticks([], [])
        #plt.xticks([], minor=True)
    else:
        if "xlabel" in plot_dic.keys() and plot_dic["xlabel"] != None:
            plt.xlabel(plot_dic["xlabel"], fontsize=plot_dic["fontsize"])  # , fontsize=11)
        if "ylabel" in plot_dic.keys() and plot_dic["ylabel"] != None:
            plt.ylabel(plot_dic["ylabel"], fontsize=plot_dic["fontsize"])  # , fontsize=11)

    if "tick_params" in plot_dic.keys() and len(plot_dic["tick_params"].keys()) > 0:
        plt.tick_params(**plot_dic["tick_params"])

    if "legend" in plot_dic.keys() and len(plot_dic["legend"].keys()) > 0:
        plt.legend(**plot_dic["legend"])

    if "subplots_adjust" in plot_dic.keys() and len(plot_dic["subplots_adjust"].keys()) > 0:
        plt.subplots_adjust(**plot_dic["subplots_adjust"])

    if "figlegend" in plot_dic.keys() and len(plot_dic["figlegend"].keys()) > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        plt.figlegend(handles, labels, **plot_dic["figlegend"])

    if "tight_layout" in plot_dic.keys():
        if plot_dic["tight_layout"]: plt.tight_layout()

    # saving
    plt.savefig(plot_dic["figname"], dpi=plot_dic["dpi"])
    if plot_dic["savepdf"]: plt.savefig(plot_dic["figname"].replace(".png", ".pdf"))
    plt.close()
    print(plot_dic["figname"])

""" ------------------------------| TASKS |--------------------------------- """

def task_plot_mdisk_fits_only():
    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki
    from model_sets import models_dietrich2015 as di15  # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16  # [24] arxive:1607.06636
    # -------------------------------------------

    datasets = OrderedDict()
    # datasets["kiuchi"] =    {'marker': "X", "ms": 20, "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019", "color": "gray", "fit": False}
    # datasets["radice"] =    {'marker': '*', "ms": 40, "models": rd.simulations[rd.fiducial], "data":rd, "err":rd.params.MdiskPP_err, "label": r"Radice+2018", "color": "blue", "fit": True}
    # # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": di.simulations[di.mask_for_with_sr], "data":di, "err":di.params.Mej_err, "label":"Dietrich+2016"}
    # # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}
    # # datasets["vincent"] =   {'marker': 'v', 'ms': 20, "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019", "color": "blue", "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    # # datasets["bauswein"] =  {'marker': 's', 'ms': 20, "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "color": "blue", "fit": False}
    # # datasets["lehner"] =    {'marker': 'P', 'ms': 20, "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016", "color": "blue", "fit": False}
    # # datasets["hotokezaka"] ={'marker': '>', 'ms': 20, "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "color": "gray", "fit": False}
    # datasets['our'] =       {'marker': 'o', 'ms': 40, "models": md.groups, "data": md, "err": "v_n", "label": r"This work", "color": "red", "fit": True}
    datasets['reference'] = {"models": md.groups, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": "v_n"}
    datasets["radiceLK"] = {"models": rd.simulations[rd.fiducial], "data": rd, "fit": True, "color": None, "plot_errorbar": True, "err": rd.params.MdiskPP_err}
    datasets["radiceM0"] = {"models": rd.simulations[rd.with_m0], "data": rd, "fit": True, "color": None, "plot_errorbar": True, "err": rd.params.MdiskPP_err}
    datasets["kiuchi"] = {"models": ki.simulations[ki.mask_for_with_tov_data], "data": ki, "fit": True, "color": None, "plot_errorbar": False, "err": ki.params.Mdisk_err}
    datasets["vincent"] = {"models": vi.simulations, "data": vi, "fit": True, "color": None, "plot_errorbar": False, "err": vi.params.Mdisk_err}
    datasets["dietrich16"] = {"models": di16.simulations[di16.mask_for_with_sr], "data": di16, "fit": True, "color": None, "plot_errorbar": False, "err": di16.params.Mdisk_err}
    datasets["dietrich15"] = {"models": di15.simulations[di15.mask_for_with_sr], "data": di15, "fit": True, "color": None, "plot_errorbar": False, "err": di15.params.Mdisk_err}

    for key in datasets.keys():
        datasets[key]["x_dic"] = {"v_n": "Mdisk3D_fit", "err": None, "deferr": None, "mod": {}}
        datasets[key]["y_dic"] = {"v_n": "Mdisk3D", "err": "ud", "deferr": 0.2, "mod": {}}
        datasets[key]["mod_x"] = {}
        datasets[key]["plot_errorbar"] = False
        datasets[key]["facecolor"] = 'none'
        datasets[key]["edgecolor"] = ourmd.datasets_colors[key]
        datasets[key]["marker"] = ourmd.datasets_markers[key]
        datasets[key]["label"] = ourmd.datasets_labels[key]
        datasets[key]["ms"] = 40
        datasets[key]["alpha"] = 0.8

    fit_dics = OrderedDict()
    fit_dics["poly2"] = { # 202 --  230
        "func": fit_funcs.poly_2_qLambda, "coeffs": np.array([-8.50e-01 , 1.12e+00 , 4.21e-04 , -3.71e-01 , 3.54e-05 , -2.13e-07]),
        # "coeffs": np.array([-8.50e-01 , 1.12e+00 , 4.21e-04 , -3.71e-01 , 3.54e-05 , -2.13e-07]),
        "xmin": 0.05, "xmax": .3, "xscale": "linear", "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
        "plot_zero": True
    }
    fit_dics["Eq.15"] = { # 481.6
        "func": fit_funcs.mdisk_kruger20, "coeffs": np.array([-4.285 , 0.844 , 1.354]),
        # "coeffs": np.array([ -0.011 , 1.001 , 1932.277]),
        "xmin": 0.05, "xmax": .3, "xscale": "linear", "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
        "plot_zero": True
    }
    fit_dics["Eq.14"] = { # 630 -- 730
        "func": fit_funcs.mdisk_radice18, "coeffs": np.array([-59.490 , 59.672 , -988.615 , 379.887]),
        # "coeffs": np.array([-66.479 , 66.661 , -1009.858 , 379.943]),
        "xmin": 0.05, "xmax": .3, "xscale": "linear", "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
        "plot_zero": True
    }
    fit_dics["poly1"] = { # 640 -- 750
        "func": fit_funcs.poly_2_Lambda, "coeffs": np.array([-6.73e-02 , 4.78e-04 , -2.11e-07]),
        # "coeffs": np.array([-6.73e-02 , 4.78e-04 , -2.11e-07]),
        "xmin": 0.05, "xmax": .3, "xscale": "linear", "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
        "plot_zero": True
    }
    # fit_dics = {
    #     "Eq.14":
    #         {"func": fit_funcs.mdisk_radice18, "coeffs": np.array([0.070,0.101,305.009,189.952]),
    #         "xmin": 0.05, "xmax": .3, "xscale": "linear", "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
    #         "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
    #         "plot_zero": True},
    #     "Eq.15":
    #         {"func": fit_funcs.mdisk_kruger20, "coeffs": np.array([-0.013, 1.000, 1325]),
    #          "xmin": 0.05, "xmax": .3, "xscale": "linear", "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
    #          "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
    #          "plot_zero": True},
    #     "poly1":
    #         {"func": fit_funcs.poly_2_Lambda, "coeffs": np.array([-8.46e-02, 6.38e-04, -3.85e-07]),
    #          "xmin": 0.05, "xmax": .3, "xscale": "linear", "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
    #          "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
    #          "plot_zero": True},
    #     "poly2":
    #         {"func": fit_funcs.poly_2_qLambda, "coeffs": np.array([-8.951e-1,1.195,4.292e-4,-3.991e-1,4.778e-5,-2.266e-7]),
    #          "xmin": 0.05, "xmax": .3, "xscale": "linear", "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
    #          "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
    #          "plot_zero": True},
    # }

    subplot_dics = OrderedDict()
    subplot_dics["poly2"] = {
        "xmin": -0.02, "xmax": .35, "xscale": "linear",
        "ymin": -10.0, "ymax": 2.0, "yscale": "linear",
        # "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
        # "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
        "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False, "tick1On": True, "tick2On": True,
                        "labelsize": 14,
                        "direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        "text": {'x': 0.85, 'y': 0.85, 's':r"$P_2(q,\tilde{\Lambda})$", 'fontsize': 14, 'color': 'black',
                 'horizontalalignment': 'center'},
        "plot_zero": True,
        "labels": True,
        "legend": {"fancybox": False, "loc": 'lower right', "columnspacing": 0.4,
                   # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 11,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False}
    }
    subplot_dics["Eq.15"] = {
        "xmin": -0.02, "xmax": .35, "xscale": "linear",
        "ymin": -10.0, "ymax": 2.0, "yscale": "linear",
        # "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
        # "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
        "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False, "tick1On": True, "tick2On": True,
                        "labelsize": 14,
                        "direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        "text": {'x': 0.85, 'y': 0.90, 's': r"Eq.(15)", 'fontsize': 14, 'color': 'black',
                 'horizontalalignment': 'center'},
        "plot_zero": True,
        "labels": True
    }
    subplot_dics["Eq.14"] = {
        "xmin": -0.02, "xmax": .35, "xscale": "linear",
        "ymin": -10.0, "ymax": 2.0, "yscale": "linear",
        # "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
        # "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
        "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False, "tick1On": True, "tick2On": True,
                        "labelsize": 14,
                        "direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        "text": {'x': 0.85, 'y': 0.90, 's': r"Eq.(14)", 'fontsize': 14, 'color': 'black',
                 'horizontalalignment': 'center'},
        "plot_zero": True,
        "labels": True
    }
    subplot_dics["poly1"] = {
        "xmin": -0.02, "xmax": .35, "xscale": "linear",
        "ymin": -10.0, "ymax": 2.0, "yscale": "linear",
        # "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
        # "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
        "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False, "tick1On": True, "tick2On": True,
                        "labelsize": 14,
                        "direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        "text": {'x': 0.85, 'y': 0.85, 's': r"$P_2(\tilde{\Lambda})$", 'fontsize': 14, 'color': 'black',
                 'horizontalalignment': 'center'},
        "plot_zero": True,
        "labels": True
    }

    # subplot_dics = {
    #     "Eq.14":
    #         {"xmin": -0.02, "xmax": .3, "xscale": "linear",
    #          "ymin": -10.0, "ymax": 2.0, "yscale": "linear",
    #          #"xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
    #          #"ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
    #          "tick_params": {"axis":'both', "which":'both', "labelleft":True,
    #                            "labelright":False, "tick1On":True, "tick2On":True,
    #                            "labelsize":14,
    #                            "direction":'in',
    #                            "bottom":True, "top":True, "left":True, "right":True},
    #          "text":{'x':0.85, 'y':0.90, 's':r"Eq.(14)", 'fontsize':14, 'color':'black','horizontalalignment':'center'},
    #          "plot_zero": True,
    #          "labels": True,
    #         },
    #     "Eq.15":
    #         {"xmin": -0.02, "xmax": .3, "xscale": "linear",
    #          "ymin": -10.0, "ymax": 2.0, "yscale": "linear",
    #          #"xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
    #          #"ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
    #          "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
    #                          "labelright": False, "tick1On": True, "tick2On": True,
    #                          "labelsize": 14,
    #                          "direction": 'in',
    #                          "bottom": True, "top": True, "left": True, "right": True},
    #          "text": {'x': 0.85, 'y': 0.90, 's': r"Eq.(15)", 'fontsize': 14, 'color': 'black',
    #                   'horizontalalignment': 'center'},
    #          "plot_zero": True,
    #          "labels": True,
    #          "legend": {"fancybox": False, "loc": 'lower right', "columnspacing": 0.4,
    #                     # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
    #                     "shadow": "False", "ncol": 2, "fontsize": 13,
    #                     "framealpha": 0., "borderaxespad": 0., "frameon": False},
    #          },
    #     "poly1":
    #         {"xmin": -0.02, "xmax": .3, "xscale": "linear",
    #          "ymin": -10.0, "ymax": 2.0, "yscale": "linear",
    #          #"xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
    #          #"ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
    #          "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
    #                          "labelright": False, "tick1On": True, "tick2On": True,
    #                          "labelsize": 14,
    #                          "direction": 'in',
    #                          "bottom": True, "top": True, "left": True, "right": True},
    #          "text": {'x': 0.85, 'y': 0.90, 's': r"Poly1", 'fontsize': 14, 'color': 'black',
    #                   'horizontalalignment': 'center'},
    #          "plot_zero": True,
    #          "labels": True
    #          },
    #     "poly2":
    #         {"xmin": -0.02, "xmax": .3, "xscale": "linear",
    #          "ymin": -10.0, "ymax": 2.0, "yscale": "linear",
    #          #"xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
    #          #"ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
    #          "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
    #                          "labelright": False, "tick1On": True, "tick2On": True,
    #                          "labelsize": 14,
    #                          "direction": 'in',
    #                          "bottom": True, "top": True, "left": True, "right": True},
    #          "text": {'x': 0.85, 'y': 0.90, 's': r"Poly2", 'fontsize': 14, 'color': 'black',
    #                   'horizontalalignment': 'center'},
    #          "plot_zero": True,
    #          "labels": True,
    #          },
    # }

    plot_dic = {
        "subplots":{"figsize": (6.0, 8.0), "ncols":1,"nrows":4, "sharex":True,"sharey":False},
        "subplot_adjust": {"left": 0.10, "bottom": 0.10, "top": 0.98, "right": 0.95, "hspace": 0},
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "tight_layout": False,
        "xlabel": r"$M_{\rm disk;fit}$ $[M_{\odot}]$",
        "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$", #r"$M_{\rm disk}$ $[10^{-3}M_{\odot}]$",
        "tick_params": {"labelcolor":'none', "top":"False", "bottom":False, "left":False, "right":False},
        "savepdf": True,
        "figname": __outplotdir__ + "disk_mass_fits_cl_all.png",
        "commonaxislabel":True,
        "subplots_adjust":{"hspace":0, "wspace":0},
        # "figlegend":{"loc" : 'lower center', "bbox_to_anchor":(0.5, 0.5),
        #              "ncol":3, "labelspacing":0.}
        "add_error_bar":{
            "median":{'color':'blue','lw':0.6,'ls':':'},
            # "mean":{'color':'gray','lw':0.5,'ls':':'},
            #"width":"1sigma",
            "confinterv":0.68,
            "fill_between":{"facecolor":'gray', "alpha":0.3}

        }
    }

    plot_subplots_for_fits(plot_dic, subplot_dics, fit_dics, datasets)

def task_plot_mej_fits_only():
    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_lehner2016 as lh  # [22] arxive:1603.00501
    from model_sets import models_kiuchi2019 as ki
    from model_sets import models_dietrich2015 as di15  # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16  # [24] arxive:1607.06636
    # -------------------------------------------

    datasets = OrderedDict()
    # datasets["kiuchi"] =    {'marker': "X", "ms": 20, "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019", "color": "gray", "fit": False}
    # datasets["radice"] =    {'marker': '*', "ms": 40, "models": rd.simulations[rd.fiducial], "data":rd, "err":rd.params.MdiskPP_err, "label": r"Radice+2018", "color": "blue", "fit": True}
    # # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": di.simulations[di.mask_for_with_sr], "data":di, "err":di.params.Mej_err, "label":"Dietrich+2016"}
    # # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}
    # # datasets["vincent"] =   {'marker': 'v', 'ms': 20, "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019", "color": "blue", "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    # # datasets["bauswein"] =  {'marker': 's', 'ms': 20, "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "color": "blue", "fit": False}
    # # datasets["lehner"] =    {'marker': 'P', 'ms': 20, "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016", "color": "blue", "fit": False}
    # # datasets["hotokezaka"] ={'marker': '>', 'ms': 20, "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "color": "gray", "fit": False}
    # datasets['our'] =       {'marker': 'o', 'ms': 40, "models": md.groups, "data": md, "err": "v_n", "label": r"This work", "color": "red", "fit": True}
    datasets['reference'] = {"models": md.groups, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": "v_n"}
    datasets["radiceLK"] = {"models": rd.simulations[rd.fiducial], "data": rd, "fit": True, "color": None, "plot_errorbar": True, "err": rd.params.MdiskPP_err}
    datasets["radiceM0"] = {"models": rd.simulations[rd.with_m0], "data": rd, "fit": True, "color": None, "plot_errorbar": True, "err": rd.params.MdiskPP_err}
    datasets["lehner"] = {"models": lh.simulations, "data": lh, "fit": True, "color": None, "plot_errorbar": True, "err": lh.params.Mdisk_err}
    datasets["kiuchi"] = {"models": ki.simulations[ki.mask_for_with_tov_data], "data": ki, "fit": True, "color": None, "plot_errorbar": False, "err": ki.params.Mdisk_err}
    datasets["vincent"] = {"models": vi.simulations, "data": vi, "fit": True, "color": None, "plot_errorbar": False, "err": vi.params.Mdisk_err}
    datasets["dietrich16"] = {"models": di16.simulations[di16.mask_for_with_sr], "data": di16, "fit": True, "color": None, "plot_errorbar": False, "err": di16.params.Mdisk_err}
    datasets["dietrich15"] = {"models": di15.simulations[di15.mask_for_with_sr], "data": di15, "fit": True, "color": None, "plot_errorbar": False, "err": di15.params.Mdisk_err}

    for key in datasets.keys():
        datasets[key]["x_dic"] = {"v_n": "Mej_tot-geo_fit", "err": None, "deferr": None, "mod": {"mult": [1e3]}}
        datasets[key]["y_dic"] = {"v_n": "Mej_tot-geo",     "err": "ud", "deferr": 0.2,  "mod": {"mult": [1e3]}}
        datasets[key]["mod_x"] = {}
        datasets[key]["facecolor"] = 'none'
        datasets[key]["plot_errorbar"] = False
        datasets[key]["edgecolor"] = ourmd.datasets_colors[key]
        datasets[key]["marker"] = ourmd.datasets_markers[key]
        datasets[key]["label"] = ourmd.datasets_labels[key]
        datasets[key]["ms"] = 40
        datasets[key]["alpha"] = 0.8

    '''
    datasets & Mean & Eq.~\eqref{eq:fit_Mej} & Eq.~\eqref{eq:fit_Mej_Kruger} & $P_2(\tilde{\Lambda})$ & $P_2(q,\tilde{\Lambda})$ \\
    Reference set & 4.0 & 3.2 & 2.0 & 3.8 & 2.1 \\ 
    \& \cite{Vincent:2019kor}  & 35.7 & 6.5 & 15.9 & 39.3 & 35.6 \\ 
    \& \cite{Radice:2018pdn[M0]}  & 26.5 & 4.3 & 13.8 & 35.0 & 31.1 \\ 
    \& \cite{Radice:2018pdn}  & 73.9 & 71.0 & 33.2 & 68.0 & 34.1 \\ 
    \& \cite{Lehner:2016lxy}  & 86.0 & 45.9 & 31.9 & 71.7 & 39.1 \\ 
    \& \cite{Kiuchi:2019lls}  & 104.5 & 101.7 & 79.8 & 105.0 & 42.3 \\ 
    \& \cite{Dietrich:2016lyp}  & 372.2 & 300.8 & 50.2 & 766.7 & 75.9 \\ 
    \& \cite{Dietrich:2015iva}  & 347.6 & 301.6 & 52.2 & 625.8 & 81.5 \\
    '''

    fit_dics = OrderedDict()
    fit_dics["Eq.7"] = {  # best 53.2
        "func": fit_funcs.mej_kruger20, "coeffs": np.array([-11.074 , 140.501 , -436.828 , 1.439]),
        "xmin": 0, "xmax": 24., "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -100.0, "ymax": 100.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
        "plot_zero": True
    }
    fit_dics["poly2"] = { # 2nd best  81.5
        "func": fit_funcs.poly_2_qLambda, "coeffs": np.array([4.83e+01 , -6.89e+01 , -3.99e-02 , 2.47e+01 , 3.83e-02 , -1.03e-06]),
        "xmin": 0, "xmax": 24., "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -100.0, "ymax": 100.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
        "plot_zero": True

    }
    fit_dics["Eq.6"] = { # 153
        "func": fit_funcs.mej_dietrich16, "coeffs": np.array([0.556 , 1.030 , -5.820 , -5.560 , -4.465 ]),
        "xmin": 0, "xmax": 24., "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -100.0, "ymax": 100.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
        "plot_zero": True
    }
    fit_dics["poly1"] = { # 628
        "func": fit_funcs.poly_2_Lambda, "coeffs": np.array([ 4.93e+00 , -3.64e-03 , 5.79e-06]),
        "xmin": 0, "xmax": 24., "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -100.0, "ymax": 100.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
        "plot_zero": True
    }

    #
    # fit_dics = {
    #     "Eq.6":
    #         {"func": fit_funcs.mej_dietrich16, "coeffs": np.array([-1.234, 3.089, -31.801, 17.526, -3.146]),
    #         "xmin": 0, "xmax": 24., "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
    #         "ymin": -100.0, "ymax": 100.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
    #         "plot_zero": True},
    #     "Eq.7":
    #         {"func": fit_funcs.mej_kruger20, "coeffs": np.array([-0.981, 12.880, -35.148, 2.030]),
    #          "xmin": 0, "xmax": 24., "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
    #          "ymin": -100.0, "ymax": 100.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
    #          "plot_zero": True},
    #     "poly1":
    #         {"func": fit_funcs.poly_2_Lambda, "coeffs": np.array([-3.209e+00, 0.032, -2.759e-05]),
    #             #np.array([-1.221e-2, 0.014, 8.396e-7]),
    #          "xmin": 0, "xmax": 24., "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
    #          "ymin": -100.0, "ymax": 100.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
    #          "plot_zero": True},
    #     "poly2":
    #         {"func": fit_funcs.poly_2_qLambda, "coeffs": np.array([2.549, 2.394, -3.005e-02, -3.376e+00, 0.038, -1.149e-05]),
    #             #np.array([2.549e-03, 2.394e-03, -3.005e-05, -3.376e-03, 3.826e-05, -1.149e-08]),
    #          "xmin": 0, "xmax": 24., "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
    #          "ymin": -100.0, "ymax": 100.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
    #          "plot_zero": True},
    # }

    subplot_dics = OrderedDict()
    subplot_dics["Eq.7"] = {
        "xmin": -8, "xmax": 55., "xscale": "linear",
        "ymin": -18.0, "ymax": 8.0, "yscale": "linear",
        # "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
        # "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
        "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False, "tick1On": True, "tick2On": True,
                        "labelsize": 14,
                        "direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        "text": {'x': 0.85, 'y': 0.90, 's': r"Eq.(7)", 'fontsize': 14, 'color': 'black',
                 'horizontalalignment': 'center'},
        "plot_zero": True,
        "labels": True,
    }
    subplot_dics["poly2"] = {
        "xmin": -8, "xmax": 55., "xscale": "linear",
        "ymin": -18.0, "ymax": 8.0, "yscale": "linear",
        # "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
        # "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
        "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False, "tick1On": True, "tick2On": True,
                        "labelsize": 14,
                        "direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        "text": {'x': 0.85, 'y': 0.85, 's': r"$P_2(q,\tilde{\Lambda})$", 'fontsize': 14, 'color': 'black',
                 'horizontalalignment': 'center'},
        "plot_zero": True,
        "legend": {"fancybox": False, "loc": 'lower right', "columnspacing": 0.4,
                   # "bbox_to_anchor": (0.8, 1.1),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 11,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "labels": True
    }
    subplot_dics["Eq.6"] = {
        "xmin": -8, "xmax": 55., "xscale": "linear",
        "ymin": -18.0, "ymax": 8.0, "yscale": "linear",
        # "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
        # "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
        "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False, "tick1On": True, "tick2On": True,
                        "labelsize": 14,
                        "direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        "text": {'x': 0.85, 'y': 0.90, 's': r"Eq.(6)", 'fontsize': 14, 'color': 'black',
                 'horizontalalignment': 'center'},
        "plot_zero": True,
        "labels": True
    }
    subplot_dics["poly1"] = {
        "xmin": -8, "xmax": 55., "xscale": "linear",
        "ymin": -18, "ymax": 8.0, "yscale": "linear",
        # "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
        # "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
        "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False, "tick1On": True, "tick2On": True,
                        "labelsize": 14,
                        "direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        "text": {'x': 0.85, 'y': 0.85, 's': r"$P_2(\tilde{\Lambda})$", 'fontsize': 14, 'color': 'black',
                 'horizontalalignment': 'center'},
        "plot_zero": True,
        "labels": True
    }




    # subplot_dics = {
    #     "Eq.6":
    #         {"xmin": -4, "xmax": 40., "xscale": "linear",
    #          "ymin": -11.0, "ymax": 9.0, "yscale": "linear",
    #          #"xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
    #          #"ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
    #          "tick_params": {"axis":'both', "which":'both', "labelleft":True,
    #                            "labelright":False, "tick1On":True, "tick2On":True,
    #                            "labelsize":14,
    #                            "direction":'in',
    #                            "bottom":True, "top":True, "left":True, "right":True},
    #          "text":{'x':0.85, 'y':0.90, 's':r"Eq.(6)", 'fontsize':14, 'color':'black','horizontalalignment':'center'},
    #          "plot_zero": True,
    #          "labels": True
    #         },
    #     "Eq.7":
    #         {"xmin": -4, "xmax": 40., "xscale": "linear",
    #          "ymin": -11.0, "ymax": 9.0, "yscale": "linear",
    #          #"xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
    #          #"ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
    #          "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
    #                          "labelright": False, "tick1On": True, "tick2On": True,
    #                          "labelsize": 14,
    #                          "direction": 'in',
    #                          "bottom": True, "top": True, "left": True, "right": True},
    #          "text": {'x': 0.85, 'y': 0.90, 's': r"Eq.(7)", 'fontsize': 14, 'color': 'black', 'horizontalalignment': 'center'},
    #          "plot_zero": True,
    #          "labels": True,
    #          },
    #     "poly1":
    #         {"xmin": -4, "xmax": 40., "xscale": "linear",
    #          "ymin": -11, "ymax": 9.0, "yscale": "linear",
    #          #"xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
    #          #"ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
    #          "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
    #                          "labelright": False, "tick1On": True, "tick2On": True,
    #                          "labelsize": 14,
    #                          "direction": 'in',
    #                          "bottom": True, "top": True, "left": True, "right": True},
    #          "text": {'x': 0.85, 'y': 0.90, 's': r"Poly1", 'fontsize': 14, 'color': 'black',
    #                   'horizontalalignment': 'center'},
    #          "plot_zero": True,
    #          "legend": {"fancybox": False, "loc": 'lower right', "columnspacing": 0.4,
    #                     # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
    #                     "shadow": "False", "ncol": 2, "fontsize": 12,
    #                     "framealpha": 0., "borderaxespad": 0., "frameon": False},
    #          "labels": True
    #          },
    #     "poly2":
    #         {"xmin": -4, "xmax": 40., "xscale": "linear",
    #          "ymin": -11.0, "ymax": 9.0, "yscale": "linear",
    #          #"xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
    #          #"ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
    #          "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
    #                          "labelright": False, "tick1On": True, "tick2On": True,
    #                          "labelsize": 14,
    #                          "direction": 'in',
    #                          "bottom": True, "top": True, "left": True, "right": True},
    #          "text": {'x': 0.85, 'y': 0.90, 's': r"Poly2", 'fontsize': 14, 'color': 'black',
    #                   'horizontalalignment': 'center'},
    #          "plot_zero": True,
    #          "labels": True
    #          },
    # }

    # print(fit_dics.keys(), subplot_dics.keys())

    plot_dic = {
        "subplots":{"figsize": (6.0, 8.0), "ncols":1,"nrows":4, "sharex":True,"sharey":False},
        "subplot_adjust": {"left" : 0.10, "bottom" : 0.10, "top" : 0.98, "right" : 0.95, "hspace" : 0},
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$", #r"$M_{\rm disk}$ $[10^{-3}M_{\odot}]$",
        "tick_params": {"labelcolor":'none', "top":False, "bottom":False, "left":False, "right":False},
        "savepdf": True,
        "figname": __outplotdir__ + "mej_fits_cl_all.png",
        "commonaxislabel":True,
        "subplots_adjust":{"hspace":0, "wspace":0},
        # "figlegend":{"loc" : 'lower center', "bbox_to_anchor":(0.5, 0.5),
        #              "ncol":3, "labelspacing":0.}
        "add_error_bar": {
            "median": {'color': 'blue', 'lw': 0.6, 'ls': ':'},
            # "mean":{'color':'gray','lw':0.5,'ls':':'},
            # "width":"1sigma",
            "confinterv": 0.68,
            "fill_between": {"facecolor": 'gray', "alpha": 0.3}

        },
        "tight_layout":False
    }

    plot_subplots_for_fits(plot_dic, subplot_dics, fit_dics, datasets)

def task_plot_vej_fits_only():
    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_lehner2016 as lh  # [22] arxive:1603.00501
    from model_sets import models_kiuchi2019 as ki
    from model_sets import models_dietrich2015 as di15  # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16  # [24] arxive:1607.06636
    # -------------------------------------------

    datasets = OrderedDict()
    # datasets["kiuchi"] =    {'marker': "X", "ms": 20, "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019", "color": "gray", "fit": False}
    # datasets["radice"] =    {'marker': '*', "ms": 40, "models": rd.simulations[rd.fiducial], "data":rd, "err":rd.params.MdiskPP_err, "label": r"Radice+2018", "color": "blue", "fit": True}
    # # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": di.simulations[di.mask_for_with_sr], "data":di, "err":di.params.Mej_err, "label":"Dietrich+2016"}
    # # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}
    # # datasets["vincent"] =   {'marker': 'v', 'ms': 20, "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019", "color": "blue", "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    # # datasets["bauswein"] =  {'marker': 's', 'ms': 20, "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "color": "blue", "fit": False}
    # # datasets["lehner"] =    {'marker': 'P', 'ms': 20, "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016", "color": "blue", "fit": False}
    # # datasets["hotokezaka"] ={'marker': '>', 'ms': 20, "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "color": "gray", "fit": False}
    # datasets['our'] =       {'marker': 'o', 'ms': 40, "models": md.groups, "data": md, "err": "v_n", "label": r"This work", "color": "red", "fit": True}
    datasets['reference'] = {"models": md.groups, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": "v_n"}
    datasets["radiceLK"] = {"models": rd.simulations[rd.fiducial], "data": rd, "fit": True, "color": None, "plot_errorbar": True, "err": rd.params.vej_err}
    datasets["radiceM0"] = {"models": rd.simulations[rd.with_m0], "data": rd, "fit": True, "color": None, "plot_errorbar": True, "err": rd.params.vej_err}
    datasets["lehner"] = {"models": lh.simulations, "data": lh, "fit": True, "color": None, "plot_errorbar": True, "err": lh.params.vej_err}    #datasets["kiuchi"] = {"models": ki.simulations[ki.mask_for_with_tov_data], "data": ki, "fit": True, "color": None, "plot_errorbar": False, "err": ki.params.Mdisk_err}
    datasets["vincent"] = {"models": vi.simulations, "data": vi, "fit": True, "color": None, "plot_errorbar": False, "err": vi.params.vej_err}
    datasets["dietrich16"] = {"models": di16.simulations[di16.mask_for_with_sr], "data": di16, "fit": True, "color": None, "plot_errorbar": False, "err": di16.params.vej_err}
    datasets["dietrich15"] = {"models": di15.simulations[di15.mask_for_with_sr], "data": di15, "fit": True, "color": None, "plot_errorbar": False, "err": di15.params.vej_err}

    for key in datasets.keys():
        datasets[key]["x_dic"] = {"v_n": "vel_inf_ave-geo_fit", "err": None, "deferr": None, "mod": {}}
        datasets[key]["y_dic"] = {"v_n": "vel_inf_ave-geo",     "err": "ud", "deferr": 0.2,  "mod": {}}
        datasets[key]["mod_x"] = {}
        datasets[key]["facecolor"] = 'none'
        datasets[key]["plot_errorbar"] = False
        datasets[key]["edgecolor"] = ourmd.datasets_colors[key]
        datasets[key]["marker"] = ourmd.datasets_markers[key]
        datasets[key]["label"] = ourmd.datasets_labels[key]
        datasets[key]["ms"] = 40
        datasets[key]["alpha"] = 0.8

    ''' 
    Reference set & 3.9 & 2.8 & 3.3 & 1.0 \\ 
    \& \cite{Vincent:2019kor}  & 4.8 & 3.3 & 4.1 & 1.6 \\ 
    \& \cite{Radice:2018pdn[M0]}  & 4.4 & 3.5 & 3.7 & 1.7 \\ 
    \& \cite{Radice:2018pdn}  & 4.9 & 3.4 & 4.4 & 3.0 \\ 
    \& \cite{Lehner:2016lxy}  & 7.6 & 6.4 & 6.8 & 5.3 \\ 
    \& \cite{Dietrich:2016lyp}  & 6.4 & 5.9 & 6.3 & 5.3 \\ 
    \& \cite{Dietrich:2015iva}  & 6.6 & 6.1 & 6.5 & 5.6 \\
    '''

    fit_dics = OrderedDict()
    fit_dics["poly2"] = { # 5.6
        "func": fit_funcs.poly_2_qLambda,
        "coeffs": np.array([4.69e-01 , -3.08e-01 , -9.19e-05 , 8.30e-02 , 1.71e-05 , 3.13e-08]),
        # np.array([2.549e-03, 2.394e-03, -3.005e-05, -3.376e-03, 3.826e-05, -1.149e-08]),
        "xmin": 0.05, "xmax": .3, "xscale": "linear", "xlabel": r"$\upsilon_{\rm ej;fit}$ [c]",
        "ymin": 0.08, "ymax": .40, "yscale": "linear", "ylabel": r"$\upsilon_{\rm ej}$ [c]",
        "plot_zero": True
    }
    fit_dics["Eq.9"] = { # 6.1
        "func": fit_funcs.vej_dietrich16, "coeffs": np.array([-0.222, 0.581, -0.934]),
        "xmin": 0.05, "xmax": .3, "xscale": "linear", "xlabel": r"$\upsilon_{\rm ej;fit}$ [c]",
        "ymin": 0.08, "ymax": .40, "yscale": "linear", "ylabel": r"$\upsilon_{\rm ej}$ [c]",
        "plot_zero": True
    }
    fit_dics["poly1"] = { # 6.1
        "func": fit_funcs.poly_2_Lambda, "coeffs": np.array([2.28e-01 , -7.41e-05 , 3.04e-08]),
        # np.array([-1.221e-2, 0.014, 8.396e-7]),
        "xmin": 0.05, "xmax": .3, "xscale": "linear", "xlabel": r"$\upsilon_{\rm ej;fit}$ [c]",
        "ymin": 0.08, "ymax": .40, "yscale": "linear", "ylabel": r"$\upsilon_{\rm ej}$ [c]",
        "plot_zero": True
    }


    # fit_dics = {
    #     "Eq.9":
    #         {"func": fit_funcs.vej_dietrich16, "coeffs": np.array([-0.422, 0.834, -1.510]),
    #         "xmin":0.05, "xmax":.3, "xscale": "linear", "xlabel": r"$\upsilon_{\rm ej;fit}$ [c]",
    #         "ymin": 0.08, "ymax": .40, "yscale": "linear", "ylabel": r"$\upsilon_{\rm ej}$ [c]",
    #         "plot_zero": True},
    #     "poly1":
    #         {"func": fit_funcs.poly_2_Lambda, "coeffs": np.array([0.252, -1.723e-04, 9.481e-08]),
    #             #np.array([-1.221e-2, 0.014, 8.396e-7]),
    #          "xmin": 0.05, "xmax": .3, "xscale": "linear", "xlabel": r"$\upsilon_{\rm ej;fit}$ [c]",
    #          "ymin": 0.08, "ymax": .40, "yscale": "linear", "ylabel": r"$\upsilon_{\rm ej}$ [c]",
    #          "plot_zero": True},
    #     "poly2":
    #         {"func": fit_funcs.poly_2_qLambda, "coeffs": np.array([0.182, 0.159, -1.509e-04, -1.046e-01, 9.233e-05, -1.581e-08]),
    #             #np.array([2.549e-03, 2.394e-03, -3.005e-05, -3.376e-03, 3.826e-05, -1.149e-08]),
    #          "xmin":0.05, "xmax":.3, "xscale": "linear", "xlabel": r"$\upsilon_{\rm ej;fit}$ [c]",
    #          "ymin": 0.08, "ymax": .40, "yscale": "linear", "ylabel": r"$\upsilon_{\rm ej}$ [c]",
    #          "plot_zero": True},
    # }
    subplot_dics = OrderedDict()
    subplot_dics["poly2"] = {
        "xmin": 0.1, "xmax": .3, "xscale": "linear",  # "xlabel": r"$\upsilon_{\rm ej;fit}$ [c]",
        "ymin": -1.1, "ymax": 1.1, "yscale": "linear",  # "ylabel": r"$\Delta \upsilon_{\rm ej} / \upsilon_{\rm ej}$",
        "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False, "tick1On": True, "tick2On": True,
                        "labelsize": 14,
                        "direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        "text": {'x': 0.85, 'y': 0.85, 's': r"$P_2(q,\tilde{\Lambda})$", 'fontsize': 14, 'color': 'black',
                 'horizontalalignment': 'center'},
        "plot_zero": True,
        "labels": True
    }
    subplot_dics["Eq.9"] = {
        "xmin": 0.1, "xmax": .3, "xscale": "linear",  # "xlabel": r"$\upsilon_{\rm ej;fit}$ [c]",
        "ymin": -1.1, "ymax": 1.1, "yscale": "linear",  # "ylabel": r"$\Delta \upsilon_{\rm ej} / \upsilon_{\rm ej}$",
        "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False, "tick1On": True, "tick2On": True,
                        "labelsize": 14,
                        "direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        "text": {'x': 0.85, 'y': 0.90, 's': r"Eq.(9)", 'fontsize': 14, 'color': 'black',
                 'horizontalalignment': 'center'},
        "plot_zero": True,
        "labels": True
    }
    subplot_dics["poly1"] = {
        "xmin": 0.1, "xmax": .3, "xscale": "linear",  # "xlabel": r"$\upsilon_{\rm ej;fit}$ [c]",
        "ymin": -1.1, "ymax": 1.1, "yscale": "linear",  # "ylabel": r"$\Delta \upsilon_{\rm ej} / \upsilon_{\rm ej}$",
        "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False, "tick1On": True, "tick2On": True,
                        "labelsize": 14,
                        "direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        "text": {'x': 0.85, 'y': 0.85, 's': r"$P_2(\tilde{\Lambda})$", 'fontsize': 14, 'color': 'black',
                 'horizontalalignment': 'center'},
        "plot_zero": True,
        "legend": {"fancybox": False, "loc": 'lower left',
                   # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 11, "columnspacing": 0.4,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "labels": True
    }


    # subplot_dics = {
    #     "Eq.9":
    #         {"xmin":0.03, "xmax":.35, "xscale": "linear", #"xlabel": r"$\upsilon_{\rm ej;fit}$ [c]",
    #          "ymin": -1.1, "ymax": 1.1, "yscale": "linear", #"ylabel": r"$\Delta \upsilon_{\rm ej} / \upsilon_{\rm ej}$",
    #          "tick_params": {"axis":'both', "which":'both', "labelleft":True,
    #                            "labelright":False, "tick1On":True, "tick2On":True,
    #                            "labelsize":14,
    #                            "direction":'in',
    #                            "bottom":True, "top":True, "left":True, "right":True},
    #          "text":{'x':0.85, 'y':0.90, 's':r"Eq.(9)", 'fontsize':14, 'color':'black','horizontalalignment':'center'},
    #          "plot_zero": True,
    #          "labels": True
    #         },
    #     "poly1":
    #         {"xmin":0.03, "xmax":.35, "xscale": "linear", #"xlabel": r"$\upsilon_{\rm ej;fit}$ [c]",
    #          "ymin": -1.1, "ymax": 1.1, "yscale": "linear", #"ylabel": r"$\Delta \upsilon_{\rm ej} / \upsilon_{\rm ej}$",
    #          "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
    #                          "labelright": False, "tick1On": True, "tick2On": True,
    #                          "labelsize": 14,
    #                          "direction": 'in',
    #                          "bottom": True, "top": True, "left": True, "right": True},
    #          "text": {'x': 0.85, 'y': 0.90, 's': r"Poly1", 'fontsize': 14, 'color': 'black',
    #                   'horizontalalignment': 'center'},
    #          "plot_zero": True,
    #          "legend": {"fancybox": False, "loc": 'lower left',
    #                     # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
    #                     "shadow": "False", "ncol": 1, "fontsize": 12, "columnspacing":0.4,
    #                     "framealpha": 0., "borderaxespad": 0., "frameon": False},
    #          "labels": True
    #          },
    #     "poly2":
    #         {"xmin":0.03, "xmax":.35, "xscale": "linear", #"xlabel": r"$\upsilon_{\rm ej;fit}$ [c]",
    #          "ymin": -1.1, "ymax": 1.1, "yscale": "linear", #"ylabel": r"$\Delta \upsilon_{\rm ej} / \upsilon_{\rm ej}$",
    #          "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
    #                          "labelright": False, "tick1On": True, "tick2On": True,
    #                          "labelsize": 14,
    #                          "direction": 'in',
    #                          "bottom": True, "top": True, "left": True, "right": True},
    #          "text": {'x': 0.85, 'y': 0.90, 's': r"Poly2", 'fontsize': 14, 'color': 'black',
    #                   'horizontalalignment': 'center'},
    #          "plot_zero": True,
    #          "labels": True
    #          },
    # }

    plot_dic = {
        "subplots":{"figsize": (6.0, 6.0), "ncols":1,"nrows":3, "sharex":True,"sharey":False},
        "subplot_adjust": {"left": 0.10, "bottom": 0.10, "top": 0.98, "right": 0.95, "hspace": 0},
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "tight_layout": False,
        "xlabel": r"$\upsilon_{\rm ej;fit}$ [c]",
        "ylabel": r"$\Delta \upsilon_{\rm ej} / \upsilon_{\rm ej}$", #r"$M_{\rm disk}$ $[10^{-3}M_{\odot}]$",
        "tick_params": {"labelcolor":'none', "top":"False", "bottom":False, "left":False, "right":False},
        "savepdf": True,
        "figname": __outplotdir__ + "vej_fits_cl_all.png",
        "commonaxislabel":True,
        "subplots_adjust":{"hspace":0, "wspace":0},
        "add_error_bar": {
            "median": {'color': 'blue', 'lw': 0.6, 'ls': ':'},
            # "mean":{'color':'gray','lw':0.5,'ls':':'},
            # "width":"1sigma",
            "confinterv": 0.68,
            "fill_between": {"facecolor": 'gray', "alpha": 0.3}
        }
        # "figlegend":{"loc" : 'lower center', "bbox_to_anchor":(0.5, 0.5),
        #              "ncol":3, "labelspacing":0.}

    }

    plot_subplots_for_fits(plot_dic, subplot_dics, fit_dics, datasets)

def task_plot_yeej_fits_only():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki
    from model_sets import models_dietrich2015 as di15  # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16  # [24] arxive:1607.06636
    # -------------------------------------------

    datasets = OrderedDict()
    # datasets["kiuchi"] =    {'marker': "X", "ms": 20, "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019", "color": "gray", "fit": False}
    # datasets["radice"] =    {'marker': '*', "ms": 40, "models": rd.simulations[rd.fiducial], "data":rd, "err":rd.params.MdiskPP_err, "label": r"Radice+2018", "color": "blue", "fit": True}
    # # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": di.simulations[di.mask_for_with_sr], "data":di, "err":di.params.Mej_err, "label":"Dietrich+2016"}
    # # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}
    # # datasets["vincent"] =   {'marker': 'v', 'ms': 20, "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019", "color": "blue", "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    # # datasets["bauswein"] =  {'marker': 's', 'ms': 20, "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "color": "blue", "fit": False}
    # # datasets["lehner"] =    {'marker': 'P', 'ms': 20, "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016", "color": "blue", "fit": False}
    # # datasets["hotokezaka"] ={'marker': '>', 'ms': 20, "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "color": "gray", "fit": False}
    # datasets['our'] =       {'marker': 'o', 'ms': 40, "models": md.groups, "data": md, "err": "v_n", "label": r"This work", "color": "red", "fit": True}
    datasets['reference'] = {"models": md.groups, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": "v_n"}
    datasets["radiceLK"] = {"models": rd.simulations[rd.fiducial], "data": rd, "fit": True, "color": None, "plot_errorbar": True, "err": rd.params.vej_err}
    datasets["radiceM0"] = {"models": rd.simulations[rd.with_m0], "data": rd, "fit": True, "color": None, "plot_errorbar": True, "err": rd.params.vej_err}
    #datasets["kiuchi"] = {"models": ki.simulations[ki.mask_for_with_tov_data], "data": ki, "fit": True, "color": None, "plot_errorbar": False, "err": ki.params.Mdisk_err}
    datasets["vincent"] = {"models": vi.simulations, "data": vi, "fit": True, "color": None, "plot_errorbar": False, "err": vi.params.Mdisk_err}
    #datasets["dietrich16"] = {"models": di16.simulations[di16.mask_for_with_sr], "data": di16, "fit": True, "color": None, "plot_errorbar": False, "err": di16.params.Mdisk_err}
    #datasets["dietrich15"] = {"models": di15.simulations[di15.mask_for_with_sr], "data": di15, "fit": True, "color": None, "plot_errorbar": False, "err": di15.params.Mdisk_err}

    for key in datasets.keys():
        datasets[key]["x_dic"] = {"v_n": "Ye_ave-geo_fit", "err": None, "deferr": None, "mod": {}}
        datasets[key]["y_dic"] = {"v_n": "Ye_ave-geo",     "err": "ud", "deferr": 0.2,  "mod": {}}
        datasets[key]["mod_x"] = {}
        datasets[key]["facecolor"] = 'none'
        datasets[key]["edgecolor"] = ourmd.datasets_colors[key]
        datasets[key]["marker"] = ourmd.datasets_markers[key]
        datasets[key]["label"] = ourmd.datasets_labels[key]
        datasets[key]["ms"] = 40
        datasets[key]["alpha"] = 0.8
        datasets[key]["plot_errorbar"] = False

    fit_dics = OrderedDict()
    fit_dics["poly2"] = { # 17
        "func": fit_funcs.poly_2_qLambda,
        "method": "delta",
        "coeffs": np.array([-3.49e-01 , 7.65e-01 , 4.46e-04 , -2.94e-01 , -2.49e-04 , -1.28e-07]),
        # np.array([2.549e-03, 2.394e-03, -3.005e-05, -3.376e-03, 3.826e-05, -1.149e-08]),
        "xmin": 0.1, "xmax": .25, "xscale": "linear", "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
        "ymin": -0.3, "ymax": 0.3, "yscale": "linear", "ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
        "plot_zero": True
    }
    fit_dics["Eq.11"] = { # 23
        "method": "delta",
        "func": fit_funcs.yeej_like_vej, "coeffs": np.array([0.128 , 0.349 , -4.161]),
        "xmin": 0.1, "xmax": .25, "xscale": "linear", "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
        "ymin": -0.3, "ymax": 0.3, "yscale": "linear", "ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
        "plot_zero": True
    }
    fit_dics["poly1"] = { # 30+
        "method": "delta",
        "func": fit_funcs.poly_2_Lambda, "coeffs": np.array([1.27e-01 , 1.34e-04 , -8.64e-08]),
        # np.array([-1.221e-2, 0.014, 8.396e-7]),
        "xmin": 0.1, "xmax": .25, "xscale": "linear", "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
        "ymin": -0.3, "ymax": 0.3, "yscale": "linear", "ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
        "plot_zero": True
    }

    # fit_dics = {
    #     "Eq.11":
    #         {"func": fit_funcs.yeej_like_vej, "coeffs": np.array([0.177, 0.452, -4.611]),
    #         "xmin": 0.1, "xmax": .25, "xscale": "linear", "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
    #         "ymin": -2.0, "ymax": 0.8, "yscale": "linear", "ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
    #         "plot_zero": True},
    #     "poly1":
    #         {"func": fit_funcs.poly_2_Lambda, "coeffs": np.array([0.064, 3.485e-04, -2.638e-07]),
    #             #np.array([-1.221e-2, 0.014, 8.396e-7]),
    #          "xmin": 0.1, "xmax": .25, "xscale": "linear", "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
    #          "ymin": -2.0, "ymax": 0.8, "yscale": "linear", "ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
    #          "plot_zero": True},
    #     "poly2":
    #         {"func": fit_funcs.poly_2_qLambda, "coeffs": np.array([-4.555e-01, 0.793, 7.509e-04, -3.139e-01, -1.899e-04, -4.460e-07]),
    #             #np.array([2.549e-03, 2.394e-03, -3.005e-05, -3.376e-03, 3.826e-05, -1.149e-08]),
    #          "xmin": 0.1, "xmax": .25, "xscale": "linear", "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
    #          "ymin": -2.0, "ymax": 0.8, "yscale": "linear", "ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
    #          "plot_zero": True},
    # }
    subplot_dics = OrderedDict()
    subplot_dics["poly2"] = {
        "xmin": 0.05, "xmax": .25, "xscale": "linear",  # "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
        "ymin": -0.3, "ymax": 0.3, "yscale": "linear",  # "ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
        "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False, "tick1On": True, "tick2On": True,
                        "labelsize": 14,
                        "direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        "text": {'x': 0.85, 'y': 0.85, 's':r"$P_2(q,\tilde{\Lambda})$", 'fontsize': 14, 'color': 'black',
                 'horizontalalignment': 'center'},
        "plot_zero": True,
        "labels": True
    }
    subplot_dics["Eq.11"] = {
        "xmin": 0.05, "xmax": .25, "xscale": "linear",  # "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
        "ymin": -0.3, "ymax": 0.3, "yscale": "linear",  # "ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
        "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False, "tick1On": True, "tick2On": True,
                        "labelsize": 14,
                        "direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        "text": {'x': 0.85, 'y': 0.90, 's': r"Eq.(11)", 'fontsize': 14, 'color': 'black',
                 'horizontalalignment': 'center'},
        "plot_zero": True,
        "labels": True
    }
    subplot_dics["poly1"] = {
        "xmin": 0.05, "xmax": .25, "xscale": "linear",  # "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
        "ymin": -0.3, "ymax": 0.3, "yscale": "linear",  # "ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
        "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False, "tick1On": True, "tick2On": True,
                        "labelsize": 14,
                        "direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        "text": {'x': 0.85, 'y': 0.85, 's': r"$P_2(\tilde{\Lambda})$", 'fontsize': 14, 'color': 'black', 'horizontalalignment': 'center'},
        "plot_zero": True,
        "legend": {"fancybox": False, "loc": 'lower left',
                   # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 11, "columnspacing": 0.4,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "labels": True
    }

    # subplot_dics = {
    #     "Eq.11":
    #         {"xmin": 0.05, "xmax": .25, "xscale": "linear", #"xlabel": r"$Y_{e\: \rm{ej;fit}}$",
    #          "ymin": -3.0, "ymax": 1.2, "yscale": "linear", #"ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
    #          "tick_params": {"axis":'both', "which":'both', "labelleft":True,
    #                            "labelright":False, "tick1On":True, "tick2On":True,
    #                            "labelsize":14,
    #                            "direction":'in',
    #                            "bottom":True, "top":True, "left":True, "right":True},
    #          "text":{'x':0.85, 'y':0.90, 's':r"Eq.(11)", 'fontsize':14, 'color':'black','horizontalalignment':'center'},
    #          "plot_zero": True,
    #          "labels": True
    #         },
    #     "poly1":
    #         {"xmin": 0.05, "xmax": .25, "xscale": "linear", #"xlabel": r"$Y_{e\: \rm{ej;fit}}$",
    #          "ymin": -3.0, "ymax": 1.2, "yscale": "linear", #"ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
    #          "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
    #                          "labelright": False, "tick1On": True, "tick2On": True,
    #                          "labelsize": 14,
    #                          "direction": 'in',
    #                          "bottom": True, "top": True, "left": True, "right": True},
    #          "text": {'x': 0.85, 'y': 0.90, 's': r"Poly1", 'fontsize': 14, 'color': 'black',
    #                   'horizontalalignment': 'center'},
    #          "plot_zero": True,
    #          "legend": {"fancybox": False, "loc": 'lower right',
    #                     # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
    #                     "shadow": "False", "ncol": 1, "fontsize": 12, "columnspacing":0.4,
    #                     "framealpha": 0., "borderaxespad": 0., "frameon": False},
    #          "labels": True
    #          },
    #     "poly2":
    #         {"xmin": 0.05, "xmax": .25, "xscale": "linear", #"xlabel": r"$Y_{e\: \rm{ej;fit}}$",
    #          "ymin": -3.0, "ymax": 1.2, "yscale": "linear", #"ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
    #          "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
    #                          "labelright": False, "tick1On": True, "tick2On": True,
    #                          "labelsize": 14,
    #                          "direction": 'in',
    #                          "bottom": True, "top": True, "left": True, "right": True},
    #          "text": {'x': 0.85, 'y': 0.90, 's': r"Poly2", 'fontsize': 14, 'color': 'black',
    #                   'horizontalalignment': 'center'},
    #          "plot_zero": True,
    #          "labels": True
    #          },
    # }

    plot_dic = {
        "subplots":{"figsize": (6.0, 6.0), "ncols":1,"nrows":3, "sharex":True,"sharey":False},
        "subplot_adjust": {"left": 0.10, "bottom": 0.10, "top": 0.98, "right": 0.95, "hspace": 0},
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "tight_layout": False,
        "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
        "ylabel": r"$\Delta Y_{e\: \rm ej}$", #r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$"
        "tick_params": {"labelcolor":'none', "top":"False", "bottom":False, "left":False, "right":False},
        "savepdf": True,
        "figname": __outplotdir__ + "yeej_fits_cl_all.png",
        "commonaxislabel":True,
        "subplots_adjust":{"hspace":0, "wspace":0},
        "add_error_bar": {
            "median": {'color': 'blue', 'lw': 0.6, 'ls': ':'},
            # "mean":{'color':'gray','lw':0.5,'ls':':'},
            # "width":"1sigma",
            "confinterv": 0.68,
            "fill_between": {"facecolor": 'gray', "alpha": 0.3}
        }
        # "figlegend":{"loc" : 'lower center', "bbox_to_anchor":(0.5, 0.5),
        #              "ncol":3, "labelspacing":0.}
    }

    plot_subplots_for_fits(plot_dic, subplot_dics, fit_dics, datasets)

def task_plot_thetaej_fits_only():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki
    from model_sets import models_dietrich2015 as di15  # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16  # [24] arxive:1607.06636
    # -------------------------------------------

    datasets = OrderedDict()
    # datasets["kiuchi"] =    {'marker': "X", "ms": 20, "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019", "color": "gray", "fit": False}
    # datasets["radice"] =    {'marker': '*', "ms": 40, "models": rd.simulations[rd.fiducial], "data":rd, "err":rd.params.MdiskPP_err, "label": r"Radice+2018", "color": "blue", "fit": True}
    # # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": di.simulations[di.mask_for_with_sr], "data":di, "err":di.params.Mej_err, "label":"Dietrich+2016"}
    # # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}
    # # datasets["vincent"] =   {'marker': 'v', 'ms': 20, "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019", "color": "blue", "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    # # datasets["bauswein"] =  {'marker': 's', 'ms': 20, "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "color": "blue", "fit": False}
    # # datasets["lehner"] =    {'marker': 'P', 'ms': 20, "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016", "color": "blue", "fit": False}
    # # datasets["hotokezaka"] ={'marker': '>', 'ms': 20, "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "color": "gray", "fit": False}
    # datasets['our'] =       {'marker': 'o', 'ms': 40, "models": md.groups, "data": md, "err": "v_n", "label": r"This work", "color": "red", "fit": True}
    datasets['reference'] = {"models": md.groups, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": "v_n"}
    datasets["radiceLK"] = {"models": rd.simulations[rd.fiducial], "data": rd, "fit": True, "color": None, "plot_errorbar": True, "err": rd.params.vej_err}
    datasets["radiceM0"] = {"models": rd.simulations[rd.with_m0], "data": rd, "fit": True, "color": None, "plot_errorbar": True, "err": rd.params.vej_err}
    #datasets["kiuchi"] = {"models": ki.simulations[ki.mask_for_with_tov_data], "data": ki, "fit": True, "color": None, "plot_errorbar": False, "err": ki.params.Mdisk_err}
    # datasets["vincent"] = {"models": vi.simulations, "data": vi, "fit": True, "color": None, "plot_errorbar": False, "err": vi.params.Mdisk_err}
    #datasets["dietrich16"] = {"models": di16.simulations[di16.mask_for_with_sr], "data": di16, "fit": True, "color": None, "plot_errorbar": False, "err": di16.params.Mdisk_err}
    #datasets["dietrich15"] = {"models": di15.simulations[di15.mask_for_with_sr], "data": di15, "fit": True, "color": None, "plot_errorbar": False, "err": di15.params.Mdisk_err}

    for key in datasets.keys():
        datasets[key]["x_dic"] = {"v_n": "theta_rms-geo_fit", "err": None, "deferr": None, "mod": {}}
        datasets[key]["y_dic"] = {"v_n": "theta_rms-geo",     "err": "ud", "deferr": 0.2,  "mod": {}}
        datasets[key]["mod_x"] = {}
        datasets[key]["facecolor"] = 'none'
        datasets[key]["edgecolor"] = ourmd.datasets_colors[key]
        datasets[key]["marker"] = ourmd.datasets_markers[key]
        datasets[key]["label"] = ourmd.datasets_labels[key]
        datasets[key]["ms"] = 40
        datasets[key]["alpha"] = 0.8
        datasets[key]["plot_errorbar"] = False

    fit_dics = OrderedDict()
    fit_dics["poly2"] = { # 17
        "func": fit_funcs.poly_2_qLambda,
        "method": "delta",
        "coeffs": np.array([-1.04e+02 , 1.77e+02 , 1.10e-01 , -6.04e+01 , -6.51e-02 , -2.47e-05]), # 8.4 & 0.483
        # np.array([2.549e-03, 2.394e-03, -3.005e-05, -3.376e-03, 3.826e-05, -1.149e-08]),
        "xmin": 0.1, "xmax": .25, "xscale": "linear", "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
        "ymin": -0.3, "ymax": 0.3, "yscale": "linear", "ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
        "plot_zero": True
    }
    fit_dics["poly1"] = { # 30+
        "method": "delta",
        "func": fit_funcs.poly_2_Lambda, "coeffs": np.array([1.47e+01 , 3.37e-02 , -1.79e-05]), # 14.3 & 0.115
        # np.array([-1.221e-2, 0.014, 8.396e-7]),
        "xmin": 0.1, "xmax": .25, "xscale": "linear", "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
        "ymin": -0.3, "ymax": 0.3, "yscale": "linear", "ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
        "plot_zero": True
    }

    # fit_dics = {
    #     "Eq.11":
    #         {"func": fit_funcs.yeej_like_vej, "coeffs": np.array([0.177, 0.452, -4.611]),
    #         "xmin": 0.1, "xmax": .25, "xscale": "linear", "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
    #         "ymin": -2.0, "ymax": 0.8, "yscale": "linear", "ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
    #         "plot_zero": True},
    #     "poly1":
    #         {"func": fit_funcs.poly_2_Lambda, "coeffs": np.array([0.064, 3.485e-04, -2.638e-07]),
    #             #np.array([-1.221e-2, 0.014, 8.396e-7]),
    #          "xmin": 0.1, "xmax": .25, "xscale": "linear", "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
    #          "ymin": -2.0, "ymax": 0.8, "yscale": "linear", "ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
    #          "plot_zero": True},
    #     "poly2":
    #         {"func": fit_funcs.poly_2_qLambda, "coeffs": np.array([-4.555e-01, 0.793, 7.509e-04, -3.139e-01, -1.899e-04, -4.460e-07]),
    #             #np.array([2.549e-03, 2.394e-03, -3.005e-05, -3.376e-03, 3.826e-05, -1.149e-08]),
    #          "xmin": 0.1, "xmax": .25, "xscale": "linear", "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
    #          "ymin": -2.0, "ymax": 0.8, "yscale": "linear", "ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
    #          "plot_zero": True},
    # }
    subplot_dics = OrderedDict()
    subplot_dics["poly2"] = {
        "xmin": 0, "xmax": 40, "xscale": "linear",  # "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
        "ymin": -38, "ymax": 38, "yscale": "linear",  # "ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
        "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False, "tick1On": True, "tick2On": True,
                        "labelsize": 14,
                        "direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        "text": {'x': 0.85, 'y': 0.85, 's':r"$P_2(q,\tilde{\Lambda})$", 'fontsize': 14, 'color': 'black',
                 'horizontalalignment': 'center'},
        "plot_zero": True,
        "labels": True
    }
    subplot_dics["poly1"] = {
        "xmin": 0, "xmax": 40, "xscale": "linear",  # "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
        "ymin": -38, "ymax": 38, "yscale": "linear",  # "ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
        "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False, "tick1On": True, "tick2On": True,
                        "labelsize": 14,
                        "direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        "text": {'x': 0.85, 'y': 0.85, 's': r"$P_2(\tilde{\Lambda})$", 'fontsize': 14, 'color': 'black', 'horizontalalignment': 'center'},
        "plot_zero": True,
        "legend": {"fancybox": False, "loc": 'lower left',
                   # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 11, "columnspacing": 0.4,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "labels": True
    }

    # subplot_dics = {
    #     "Eq.11":
    #         {"xmin": 0.05, "xmax": .25, "xscale": "linear", #"xlabel": r"$Y_{e\: \rm{ej;fit}}$",
    #          "ymin": -3.0, "ymax": 1.2, "yscale": "linear", #"ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
    #          "tick_params": {"axis":'both', "which":'both', "labelleft":True,
    #                            "labelright":False, "tick1On":True, "tick2On":True,
    #                            "labelsize":14,
    #                            "direction":'in',
    #                            "bottom":True, "top":True, "left":True, "right":True},
    #          "text":{'x':0.85, 'y':0.90, 's':r"Eq.(11)", 'fontsize':14, 'color':'black','horizontalalignment':'center'},
    #          "plot_zero": True,
    #          "labels": True
    #         },
    #     "poly1":
    #         {"xmin": 0.05, "xmax": .25, "xscale": "linear", #"xlabel": r"$Y_{e\: \rm{ej;fit}}$",
    #          "ymin": -3.0, "ymax": 1.2, "yscale": "linear", #"ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
    #          "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
    #                          "labelright": False, "tick1On": True, "tick2On": True,
    #                          "labelsize": 14,
    #                          "direction": 'in',
    #                          "bottom": True, "top": True, "left": True, "right": True},
    #          "text": {'x': 0.85, 'y': 0.90, 's': r"Poly1", 'fontsize': 14, 'color': 'black',
    #                   'horizontalalignment': 'center'},
    #          "plot_zero": True,
    #          "legend": {"fancybox": False, "loc": 'lower right',
    #                     # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
    #                     "shadow": "False", "ncol": 1, "fontsize": 12, "columnspacing":0.4,
    #                     "framealpha": 0., "borderaxespad": 0., "frameon": False},
    #          "labels": True
    #          },
    #     "poly2":
    #         {"xmin": 0.05, "xmax": .25, "xscale": "linear", #"xlabel": r"$Y_{e\: \rm{ej;fit}}$",
    #          "ymin": -3.0, "ymax": 1.2, "yscale": "linear", #"ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
    #          "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
    #                          "labelright": False, "tick1On": True, "tick2On": True,
    #                          "labelsize": 14,
    #                          "direction": 'in',
    #                          "bottom": True, "top": True, "left": True, "right": True},
    #          "text": {'x': 0.85, 'y': 0.90, 's': r"Poly2", 'fontsize': 14, 'color': 'black',
    #                   'horizontalalignment': 'center'},
    #          "plot_zero": True,
    #          "labels": True
    #          },
    # }

    plot_dic = {
        "subplots":{"figsize": (6.0, 4.0), "ncols":1,"nrows":2, "sharex":True,"sharey":False}, # 6.0, 8.0
        "subplot_adjust": {"left": 0.10, "bottom": 0.12, "top": 0.98, "right": 0.95, "hspace": 0},
        # "subplot_adjust": {"left": 0.10, "bottom": 0.10, "top": 0.98, "right": 0.95, "hspace": 0},
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "tight_layout": False,
        "xlabel": r"$\theta_{\rm{RMS;\:fit}}$",
        "ylabel": r"$\Delta \theta_{\rm RMS}$", #r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$"
        "tick_params": {"labelcolor":'none', "top":"False", "bottom":False, "left":False, "right":False},
        "savepdf": True,
        "figname": __outplotdir__ + "thetaej_fits_cl_all.png",
        "commonaxislabel":True,
        "subplots_adjust":{"hspace":0, "wspace":0},
        "add_error_bar": {
            "median": {'color': 'blue', 'lw': 0.6, 'ls': ':'},
            # "mean":{'color':'gray','lw':0.5,'ls':':'},
            # "width":"1sigma",
            "confinterv": 0.68,
            "fill_between": {"facecolor": 'gray', "alpha": 0.3}
        }
        # "figlegend":{"loc" : 'lower center', "bbox_to_anchor":(0.5, 0.5),
        #              "ncol":3, "labelspacing":0.}
    }

    plot_subplots_for_fits(plot_dic, subplot_dics, fit_dics, datasets)

''' --------- MAIN --------- '''

if __name__ == "__main__":

    ''' --- Mej --- '''
    task_plot_mej_fits_only()

    ''' --- vej --- '''
    task_plot_vej_fits_only()

    ''' --- ye --- '''
    task_plot_yeej_fits_only()

    ''' --- theta --- '''
    task_plot_thetaej_fits_only()

    ''' --- Mdisk --- '''
    task_plot_mdisk_fits_only()



    ''' test '''