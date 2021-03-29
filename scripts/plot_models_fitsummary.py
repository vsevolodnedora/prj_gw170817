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
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import (cm, colors, colorbar)
from mpl_toolkits.axes_grid1 import make_axes_locatable
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif')
from collections import OrderedDict

from matplotlib.colors import LogNorm, Normalize
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from model_sets import models as ourmd

__outplotdir__ = "/home/vsevolod/GIT/bitbucket/bns_gw170817/tex/fitpaper/figs/residuals/"
if not os.path.isdir(__outplotdir__):
    os.mkdir(__outplotdir__)

#from make_fit2 import Fitting_Coefficients as fit_coefs, Fitting_Functions as fit_funcs

from make_fit4 import FittingFunctions, FittingCoefficients


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

def get_fit_val(ffunc, coeffs, models, to_val=None):
    vals = ffunc(models, *coeffs)
    if to_val is None: return vals
    elif to_val == "10**":
        return 10**vals
    elif to_val == "log10":
        return np.log10(vals)
    else:
        raise NameError("Not implmenented")

""" -----------------------------| MODULUS |------------------------------- """

def plot_subplots_for_fits(plot_dic, subplot_dics, fit_dics, model_dics):

    # assuertion
    assert len(subplot_dics.keys()) == len(fit_dics.keys())

    # assign subplots
    fig, axes = plt.subplots(**plot_dic["subplots"])
    if not hasattr(axes, '__len__'):
        axes = [axes]
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
                # func, coeffs = fit_dic["func"], fit_dic["coeffs"]
                mej = get_fit_val(fit_dic["func"], fit_dic["coeffs"], models, fit_dic["to_val"])
                # mej = func(models, *coeffs)# / 1.e3  # 1e-3 Msun -> Msun
                # if "to_val" in fit_dic.keys():
                #     if fit_dic["to_val"] == "10**":
                #         mej = 10**mej
                # if "dev" in fit_dic.keys(): mej = mej / fit_dic["dev"]
                x = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, mej)
                y = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models)
                # print(x);
                # exit(1)
            elif x_dic["v_n"] == "vel_inf_ave-geo_fit" and y_dic["v_n"] == "vel_inf_ave-geo":
                func, coeffs = fit_dic["func"], fit_dic["coeffs"]
                mej = func(models, *coeffs)  # 1e-3 Msun -> Msun
                x = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, mej)
                y = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models)
            elif x_dic["v_n"] == "Mdisk3D_fit" and y_dic["v_n"] == "Mdisk3D":
                # func, coeffs = fit_dic["func"], fit_dic["coeffs"]
                # mdisk = func(models, *coeffs)  # 1e-3 Msun -> Msun
                mdisk = get_fit_val(fit_dic["func"], fit_dic["coeffs"], models, fit_dic["to_val"])
                x = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, mdisk)
                y = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models)
            elif x_dic["v_n"] == "Ye_ave-geo_fit" and y_dic["v_n"] == "Ye_ave-geo":
                func, coeffs = fit_dic["func"], fit_dic["coeffs"]
                mej = func(models, *coeffs)  # 1e-3 Msun -> Msun
                x = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, mej)
                y = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models)
            elif x_dic["v_n"] == "theta_rms-geo_fit" and y_dic["v_n"] == "theta_rms-geo":
                func, coeffs = fit_dic["func"], fit_dic["coeffs"]
                mej = func(models, *coeffs)  # 1e-3 Msun -> Msun
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
            # func, coeffs = fit_dic["func"], fit_dic["coeffs"]
            # fitted_values = func(models, *coeffs)
            fitted_values = get_fit_val(fit_dic["func"], fit_dic["coeffs"], models, fit_dic["to_val"])
            #ss
            # if y_dic["v_n"] == "Mej_tot-geo":
            #     if "dev" in fit_dic.keys(): fitted_values = fitted_values / fit_dic["dev"]
                    #fitted_values = fitted_values / 1.e3  # Tims fucking fit
            #
            y_from_fit = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, fitted_values)

            # if "to_val" in fit_dic.keys():
            #     if fit_dic["to_val"] == "10**":
            #         y_from_fit = 10 ** y_from_fit

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
            ldic = subplotdic["legend"]
            if "first_n" in ldic.keys():
                ldic = subplotdic["legend"]
                n = ldic["first_n"]
                del ldic["first_n"]
                han, lab = axi.get_legend_handles_labels()
                axi.add_artist(axi.legend(han[n:], lab[n:], **ldic))
                # ax[0].add_artist(ax[0].legend(han[len(han) - n:], lab[len(lab) - n:], **dic_))
            elif "last_n" in ldic.keys():
                ldic = subplotdic["legend"]
                n = ldic["last_n"]
                del ldic["last_n"]
                han, lab = axi.get_legend_handles_labels()
                axi.add_artist(axi.legend(han[:-1 * n], lab[:-1 * n], **ldic))
                # ax[0].add_artist(ax[0].legend(han[len(han) - n:], lab[len(lab) - n:], **dic_))
            else:
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

    # if "legend" in plot_dic.keys() and len(plot_dic["legend"].keys()) > 0:
    #     plt.legend(**plot_dic["legend"])

    if "subplots_adjust" in plot_dic.keys() and len(plot_dic["subplots_adjust"].keys()) > 0:
        plt.subplots_adjust(**plot_dic["subplots_adjust"])

    if "figlegend" in plot_dic.keys() and len(plot_dic["figlegend"].keys()) > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        plt.figlegend(handles, labels, **plot_dic["figlegend"])

    if "tight_layout" in plot_dic.keys():
        if plot_dic["tight_layout"]: plt.tight_layout()


    # saving
    if "figname" in plot_dic.keys():
        plt.savefig(plot_dic["figname"], dpi=plot_dic["dpi"])
    if "savepdf" in plot_dic.keys() and plot_dic["savepdf"]:
        plt.savefig(plot_dic["figname"].replace(".png", ".pdf"))
    plt.show()
    plt.close()
    print(plot_dic["figname"])

def plot_subplots_for_fits_colorcoded(plot_dic, subplot_dics, fit_dics, model_dics):

    # assuertion
    assert len(subplot_dics.keys()) == len(fit_dics.keys())

    # assign subplots
    fig, axes = plt.subplots(**plot_dic["subplots"])
    if not hasattr(axes, '__len__'):
        axes = [axes]
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
            c_dic = model_dic["c_dic"]
            #            #
            if "v_n_x" in subplot_dic: model_dic["v_n_x"] = subplot_dic["v_n_x"] # overwrite
            if "v_n_y" in subplot_dic: model_dic["v_n_y"] = subplot_dic["v_n_y"] # overwrite
            if "v_n_c" in subplot_dic: model_dic["v_n_c"] = subplot_dic["v_n_c"]  # overwrite
            #
            if x_dic["v_n"] == "Mej_tot-geo_fit" and y_dic["v_n"] == "Mej_tot-geo":
                # func, coeffs = fit_dic["func"], fit_dic["coeffs"]
                mej = get_fit_val(fit_dic["func"], fit_dic["coeffs"], models, fit_dic["to_val"])
                # mej = func(models, *coeffs)# / 1.e3  # 1e-3 Msun -> Msun
                # if "to_val" in fit_dic.keys():
                #     if fit_dic["to_val"] == "10**":
                #         mej = 10**mej
                # if "dev" in fit_dic.keys(): mej = mej / fit_dic["dev"]
                x = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, mej)
                y = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models)
                # print(x);
                # exit(1)
            elif x_dic["v_n"] == "vel_inf_ave-geo_fit" and y_dic["v_n"] == "vel_inf_ave-geo":
                func, coeffs = fit_dic["func"], fit_dic["coeffs"]
                mej = func(models, *coeffs)  # 1e-3 Msun -> Msun
                x = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, mej)
                y = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models)
            elif x_dic["v_n"] == "Mdisk3D_fit" and y_dic["v_n"] == "Mdisk3D":
                # func, coeffs = fit_dic["func"], fit_dic["coeffs"]
                # mdisk = func(models, *coeffs)  # 1e-3 Msun -> Msun
                mdisk = get_fit_val(fit_dic["func"], fit_dic["coeffs"], models, fit_dic["to_val"])
                x = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, mdisk)
                y = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models)
            elif x_dic["v_n"] == "Ye_ave-geo_fit" and y_dic["v_n"] == "Ye_ave-geo":
                func, coeffs = fit_dic["func"], fit_dic["coeffs"]
                mej = func(models, *coeffs)  # 1e-3 Msun -> Msun
                x = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, mej)
                y = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models)
            elif x_dic["v_n"] == "theta_rms-geo_fit" and y_dic["v_n"] == "theta_rms-geo":
                func, coeffs = fit_dic["func"], fit_dic["coeffs"]
                mej = func(models, *coeffs)  # 1e-3 Msun -> Msun
                x = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, mej)
                y = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models)
            else:
                x = d_cl.get_mod_data(x_dic["v_n"], x_dic["mod"], models)
                y = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models)
            if c_dic["v_n"] == "q":
                c = d_cl.get_mod_data(c_dic["v_n"], c_dic["mod"], models)
            else:
                raise NameError("color code 'c_dic['v_n']': {} ' is not recognized".format(c_dic["v_n"]))
            #
            x = np.array(x)
            y = np.array(y)
            c = np.array(c)
            #
            norm = colors.Normalize(vmin=subplot_dic["vmin"], vmax=subplot_dic["vmax"])
            cmap = cm.get_cmap(subplot_dic["cmap"])
            #
            edgecolors = [cmap(norm(cval)) for cval in c]#[model_dic['edgecolor'] for sim in models.index]
            facecolors = [cmap(norm(cval)) for cval in c]#[model_dic['facecolor'] for sim in models.index]
            markers = [model_dic['marker'] for sim in models.index]
            mss = [model_dic['ms'] for sim in models.index]
            #
            subplot_dic["norm"] = norm
            subplot_dic["cmap"] = cmap
            # sc = mscatter(x, y, ax=ax, s=mss, m=markers, label=None, alpha=plot_dic['alpha'],
            #               edgecolor=edgecolors, facecolor=facecolors)
            #
            # func, coeffs = fit_dic["func"], fit_dic["coeffs"]
            # fitted_values = func(models, *coeffs)
            fitted_values = get_fit_val(fit_dic["func"], fit_dic["coeffs"], models, fit_dic["to_val"])
            #ss
            # if y_dic["v_n"] == "Mej_tot-geo":
            #     if "dev" in fit_dic.keys(): fitted_values = fitted_values / fit_dic["dev"]
                    #fitted_values = fitted_values / 1.e3  # Tims fucking fit
            #
            y_from_fit = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, fitted_values)

            # if "to_val" in fit_dic.keys():
            #     if fit_dic["to_val"] == "10**":
            #         y_from_fit = 10 ** y_from_fit

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
                          m=markers, label=None, alpha=model_dic['alpha'], edgecolor=edgecolors,
                          facecolors=model_dic['facecolor'])

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
        if "texts" in subplotdic.keys():
            for dic in subplotdic["texts"]:
                dic["transform"] = axi.transAxes
                axi.text(**dic)
        #
        if "plot_zero" in subplotdic.keys() and subplotdic["plot_zero"]:
            axi.axhline(y=0, linestyle=':', linewidth=0.4, color='black')

        if "plot_diagonal" in subplotdic.keys() and subplotdic["plot_diagonal"]:
            axi.plot([0, 100], [0, 100], linestyle=':', linewidth=0.4, color='black',label="fit")

        if "hline" in subplotdic.keys() and len(subplotdic["hline"].keys()) > 0:
            axi.axhline(**subplotdic["hline"])

        if "legend" in subplotdic.keys() and len(subplotdic["legend"].keys()) > 0:
            ldic = subplotdic["legend"]
            if "first_n" in ldic.keys():
                ldic = subplotdic["legend"]
                n = ldic["first_n"]
                del ldic["first_n"]
                han, lab = axi.get_legend_handles_labels()
                axi.add_artist(axi.legend(han[n:], lab[n:], **ldic))
                # ax[0].add_artist(ax[0].legend(han[len(han) - n:], lab[len(lab) - n:], **dic_))
            elif "last_n" in ldic.keys():
                ldic = subplotdic["legend"]
                n = ldic["last_n"]
                del ldic["last_n"]
                han, lab = axi.get_legend_handles_labels()
                axi.add_artist(axi.legend(han[:-1 * n], lab[:-1 * n], **ldic))
                # ax[0].add_artist(ax[0].legend(han[len(han) - n:], lab[len(lab) - n:], **dic_))
            else:
                axi.legend(**subplotdic["legend"])

        if "cbar_label" in subplotdic.keys():
            divider = make_axes_locatable(axi)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = colorbar.ColorbarBase(cax, cmap=subplotdic["cmap"], norm=subplotdic["norm"])
            cbar.set_label(subplotdic["cbar_label"], fontsize=plot_dic["fontsize"])
            cbar.ax.tick_params(labelsize=plot_dic["fontsize"])

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

    # if "legend" in plot_dic.keys() and len(plot_dic["legend"].keys()) > 0:
    #     plt.legend(**plot_dic["legend"])

    if "subplots_adjust" in plot_dic.keys() and len(plot_dic["subplots_adjust"].keys()) > 0:
        plt.subplots_adjust(**plot_dic["subplots_adjust"])

    if "figlegend" in plot_dic.keys() and len(plot_dic["figlegend"].keys()) > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        plt.figlegend(handles, labels, **plot_dic["figlegend"])

    if "tight_layout" in plot_dic.keys():
        if plot_dic["tight_layout"]: plt.tight_layout()


    # saving
    if "figname" in plot_dic.keys():
        plt.savefig(plot_dic["figname"], dpi=plot_dic["dpi"])
    if "savepdf" in plot_dic.keys() and plot_dic["savepdf"]:
        plt.savefig(plot_dic["figname"].replace(".png", ".pdf"))
    plt.show()
    plt.close()
    print(plot_dic["figname"])

""" ------------------------------| TASKS |--------------------------------- """

def task_plot_mdisk_fits_only():

    from model_sets import combined as md
    from make_fit4 import Fit_Data

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
    datasets['reference'] = {"models": md.reference, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": "v_n"}
    datasets["radiceLK"] =  {"models": md.radice18lk, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": md.params.MdiskPP_err}
    datasets["radiceM0"] =  {"models": md.radice18m0, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": md.params.MdiskPP_err}
    # # # datasets["lehner"] =    {"models": md.lehner16, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": md.params.MdiskPP_err}
    datasets["kiuchi"] =    {"models": md.kiuchi19, "data": md, "fit": True, "color": None, "plot_errorbar": False, "err": md.params.MdiskPP_err}
    datasets["vincent"] =   {"models": md.vincent19, "data": md, "fit": True, "color": None, "plot_errorbar": False, "err": md.params.MdiskPP_err}
    datasets["dietrich16"] ={"models": md.dietrich16, "data": md, "fit": True, "color": None, "plot_errorbar": False, "err": md.params.MdiskPP_err}
    datasets["dietrich15"] ={"models": md.dietrich15, "data": md, "fit": True, "color": None, "plot_errorbar": False, "err": md.params.MdiskPP_err}
    # # # datasets["sekiguchi15"]={"models": md.sekiguchi15, "data": md, "err": md.params.Mej_err, "label": r"Sekiguchi+2015",  "fit": True}
    datasets["sekiguchi16"]={"models": md.sekiguchi16, "data": md, "err": md.params.Mej_err, "label": r"Sekiguchi+2016", "fit": True}
    # # # datasets["hotokezaka"] ={"models": md.hotokezaka12, "data": md, "err": md.params.Mej_err, "label": r"Hotokezaka+2013", "fit": True}
    # # # datasets["bauswein"] =  {"models": md.basuwein13, "data": md, "err": md.params.Mej_err, "label": r"Bauswein+2013", "fit": True}

    for key in datasets.keys():
        datasets[key]["x_dic"] = {"v_n": "Mdisk3D_fit", "err": None, "deferr": None, "mod": {}}
        datasets[key]["y_dic"] = {"v_n": "Mdisk3D", "err": "ud", "deferr": 0.2, "mod": {}}
        datasets[key]["mod_x"] = {}
        datasets[key]["plot_errorbar"] = False
        datasets[key]["facecolor"] = 'none'
        datasets[key]["edgecolor"] = md.datasets_colors[key]
        datasets[key]["marker"] = md.datasets_markers[key]
        datasets[key]["label"] = md.datasets_labels[key]
        datasets[key]["ms"] = 40
        datasets[key]["alpha"] = 0.8

    from make_fit4 import Fit_Data
    o_fit = Fit_Data(md.simulations, "Mdisk3D", err_method="default", clean_nans=True)

    fit_dics = OrderedDict()
    coeffs, _, _, _ = o_fit.fit_curve(ff_name="poly22_qLambda", cf_name="poly22", modify=None, usesigma=False)
    fit_dics["poly2"] = { # 202 --  230
        "func": FittingFunctions.poly_2_qLambda, "coeffs": np.array(coeffs),"to_val":None,#"10**",# None,
        "xmin": 0.05, "xmax": .3, "xscale": "linear", "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
        "plot_zero": True
    }
    coeffs, _, _, _ = o_fit.fit_curve(ff_name="krug19", cf_name="krug19", modify=None, usesigma=False)
    fit_dics["Eq.15"] = { # 481.6
        "func": FittingFunctions.mdisk_kruger19, "coeffs": np.array(coeffs), "to_val":None,#"10**",# None,
        "xmin": 0.05, "xmax": .3, "xscale": "linear", "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
        "plot_zero": True
    }
    coeffs, _, _, _ = o_fit.fit_curve(ff_name="rad18", cf_name="rad18", modify=None, usesigma=False)
    fit_dics["Eq.14"] = { # 630 -- 730
        "func": FittingFunctions.mdisk_radice18, "coeffs": np.array(coeffs), "to_val":None,#"10**",# None,
        "xmin": 0.05, "xmax": .3, "xscale": "linear", "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
        "plot_zero": True
    }
    coeffs, _, _, _ = o_fit.fit_curve(ff_name="poly2_Lambda", cf_name="poly2", modify=None, usesigma=False)
    fit_dics["poly1"] = { # 640 -- 750
        "func": FittingFunctions.poly_2_Lambda, "coeffs": np.array(coeffs),  "to_val":None,#"10**",# None,
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
        "xmin": -0.02, "xmax": .30, "xscale": "linear",
        "ymin": -4.2, "ymax": 1.8, "yscale": "linear",
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
        "labels": True
    }
    subplot_dics["Eq.15"] = {
        "xmin": -0.02, "xmax": .30, "xscale": "linear",
        "ymin": -4.2, "ymax": 1.8, "yscale": "linear",
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
        "xmin": -0.02, "xmax": .30, "xscale": "linear",
        "ymin": -4.2, "ymax": 1.8, "yscale": "linear",
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
        "labels": True,
        "legend": {
            "last_n": 4,
            "fancybox": False, "loc": 'lower right', "columnspacing": 0.4,
            # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
            "shadow": "False", "ncol": 1, "fontsize": 11,
            "framealpha": 0., "borderaxespad": 0., "frameon": False}
    }
    subplot_dics["poly1"] = {
        "xmin": -0.02, "xmax": .30, "xscale": "linear",
        "ymin": -4.2, "ymax": 1.8, "yscale": "linear",
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
        "labels": True,
        "legend": {
            "first_n":4,
            "fancybox": False, "loc": 'lower right', "columnspacing": 0.4,
            # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
            "shadow": "False", "ncol": 1, "fontsize": 11,
            "framealpha": 0., "borderaxespad": 0., "frameon": False}
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
        "figname": __outplotdir__ + "residuals_disk_mass.png",
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
    # from model_sets import models_vincent2019 as vi
    # from model_sets import models_radice2018 as rd
    # from model_sets import groups as md
    # from model_sets import models_lehner2016 as lh  # [22] arxive:1603.00501
    # from model_sets import models_kiuchi2019 as ki
    # from model_sets import models_dietrich2015 as di15  # [21] arxive:1504.01266
    # from model_sets import models_dietrich2016 as di16  # [24] arxive:1607.06636
    # from model_sets import models_sekiguchi2016 as se16  # [23] arxive:1603.01918 # no Mb
    # from model_sets import models_sekiguchi2015 as se15  # [-]  arxive:1502.06660 # no Mb
    # from model_sets import models_hotokezaka2013 as hz          # [19] arxive:1212.0905
    # from model_sets import models_bauswein2013 as bs            # [20] arxive:1302.6530

    from model_sets import combined as md
    from make_fit4 import Fit_Data

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
    datasets['reference'] = {"models": md.reference, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": "v_n"}
    datasets["radiceLK"] =  {"models": md.radice18lk, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": md.params.MdiskPP_err}
    datasets["radiceM0"] =  {"models": md.radice18m0, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": md.params.MdiskPP_err}
    datasets["lehner"] =    {"models": md.lehner16, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": md.params.MdiskPP_err}
    datasets["kiuchi"] =    {"models": md.kiuchi19, "data": md, "fit": True, "color": None, "plot_errorbar": False, "err": md.params.MdiskPP_err}
    datasets["vincent"] =   {"models": md.vincent19, "data": md, "fit": True, "color": None, "plot_errorbar": False, "err": md.params.MdiskPP_err}
    datasets["dietrich16"] ={"models": md.dietrich16, "data": md, "fit": True, "color": None, "plot_errorbar": False, "err": md.params.MdiskPP_err}
    datasets["dietrich15"] ={"models": md.dietrich15, "data": md, "fit": True, "color": None, "plot_errorbar": False, "err": md.params.MdiskPP_err}
    datasets["sekiguchi15"]={"models": md.sekiguchi15, "data": md, "err": md.params.Mej_err, "label": r"Sekiguchi+2015",  "fit": True}
    datasets["sekiguchi16"]={"models": md.sekiguchi16, "data": md, "err": md.params.Mej_err, "label": r"Sekiguchi+2016", "fit": True}
    datasets["hotokezaka"] ={"models": md.hotokezaka12, "data": md, "err": md.params.Mej_err, "label": r"Hotokezaka+2013", "fit": True}
    datasets["bauswein"] =  {"models": md.basuwein13, "data": md, "err": md.params.Mej_err, "label": r"Bauswein+2013", "fit": True}



    for key in datasets.keys():
        datasets[key]["x_dic"] = {"v_n": "Mej_tot-geo_fit", "err": None, "deferr": None, "mod":{"mult": [1e3]}}#{"mult": [1e3]}}#, "mod": {"mult": [1e3]}}
        datasets[key]["y_dic"] = {"v_n": "Mej_tot-geo",     "err": "ud", "deferr": 0.2, "mod":{"mult": [1e3]}}#}{"mult": [1e3]}}#,  "mod": {"mult": [1e3]}}
        datasets[key]["mod_x"] = {}
        datasets[key]["facecolor"] = 'none'
        datasets[key]["plot_errorbar"] = False
        datasets[key]["edgecolor"] = md.datasets_colors[key]
        datasets[key]["marker"] = md.datasets_markers[key]
        datasets[key]["label"] = md.datasets_labels[key]
        datasets[key]["ms"] = 40
        datasets[key]["alpha"] = 0.8


    o_fit = Fit_Data(md.simulations, "Mej_tot-geo", err_method="default", clean_nans=True)

    # coeffs, _, _, _ = o_fit.fit_curve(ff_name="poly22_qLambda", cf_name="poly22", modify="log10")
    fit_dics = OrderedDict()
    coeffs, _, _, _ = o_fit.fit_curve(ff_name="krug19", cf_name="krug19", modify="log10")
    fit_dics["Eq.7"] = {  # best 53.2
        "func": FittingFunctions.mej_kruger19, "coeffs": np.array(coeffs), "to_val": "10**",# None,
        "xmin": 0, "xmax": 24., "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -100.0, "ymax": 100.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
        "plot_zero": True
    }
    coeffs, _, _, _ = o_fit.fit_curve(ff_name="poly22_qLambda", cf_name="poly22", modify="log10")
    fit_dics["poly2"] = { # 2nd best  81.5
        "func": FittingFunctions.poly_2_qLambda, "coeffs": np.array(coeffs), "to_val": "10**",#None,
        "xmin": 0, "xmax": 24., "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -100.0, "ymax": 100.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
        "plot_zero": True

    }
    coeffs, _, _, _ = o_fit.fit_curve(ff_name="diet16", cf_name="diet16", modify="log10")
    fit_dics["Eq.6"] = { # 153
        "func": FittingFunctions.mej_dietrich16, "coeffs": np.array(coeffs), "to_val": "10**",# None,
        "xmin": 0, "xmax": 24., "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -100.0, "ymax": 100.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
        "plot_zero": True
    }
    coeffs, _, _, _ = o_fit.fit_curve(ff_name="poly2_Lambda", cf_name="poly2", modify="log10")
    fit_dics["poly1"] = { # 628
        "func": FittingFunctions.poly_2_Lambda, "coeffs": np.array(coeffs), "to_val": "10**",# None,
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
        "xmin": -0.5, "xmax": 40., "xscale": "linear",
        "ymin": -12., "ymax": 2.2, "yscale": "linear",
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
        "xmin": -0.5, "xmax": 40., "xscale": "linear",
        "ymin": -12., "ymax": 2.2, "yscale": "linear",
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
        "labels": True
    }
    subplot_dics["Eq.6"] = {
        "xmin": -0.5, "xmax": 40., "xscale": "linear",
        "ymin": -12., "ymax": 2.2, "yscale": "linear",
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
        "xmin": -0.5, "xmax": 40., "xscale": "linear",
        "ymin": -12., "ymax": 2.2, "yscale": "linear",
        # "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
        # "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
        "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False, "tick1On": True, "tick2On": True,
                        "labelsize": 14,
                        "direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        "text": {'x': 0.85, 'y': 0.85, 's': r"$P_2(\tilde{\Lambda})$", 'fontsize': 14, 'color': 'black',
                 'horizontalalignment': 'center'},
        "legend": {"fancybox": False, "loc": 'lower right', "columnspacing": 0.4,
                   # "bbox_to_anchor": (0.8, 1.1),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 11,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
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
        "figname": __outplotdir__ + "residuals_mej.png",
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

    from model_sets import combined as md

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
    datasets['reference'] = {"models": md.reference, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": "v_n"}
    datasets["radiceLK"] =  {"models": md.radice18lk, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": md.params.vej_err}
    datasets["radiceM0"] =  {"models": md.radice18m0, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": md.params.vej_err}
    datasets["lehner"] =    {"models": md.lehner16, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": md.params.vej_err}
    # # # datasets["kiuchi"] =    {"models": md.kiuchi19, "data": md, "fit": True, "color": None, "plot_errorbar": False, "err": md.params.vej_err}
    datasets["vincent"] =   {"models": md.vincent19, "data": md, "fit": True, "color": None, "plot_errorbar": False, "err": md.params.vej_err}
    datasets["dietrich16"] ={"models": md.dietrich16, "data": md, "fit": True, "color": None, "plot_errorbar": False, "err": md.params.vej_err}
    datasets["dietrich15"] ={"models": md.dietrich15, "data": md, "fit": True, "color": None, "plot_errorbar": False, "err": md.params.vej_err}
    # # # datasets["sekiguchi15"]={"models": md.sekiguchi15, "data": md, "err": md.params.vej_err, "label": r"Sekiguchi+2015",  "fit": True}
    datasets["sekiguchi16"]={"models": md.sekiguchi16, "data": md, "err": md.params.vej_err, "label": r"Sekiguchi+2016", "fit": True}
    datasets["hotokezaka"] ={"models": md.hotokezaka12, "data": md, "err": md.params.vej_err, "label": r"Hotokezaka+2013", "fit": True}
    datasets["bauswein"] =  {"models": md.basuwein13, "data": md, "err": md.params.vej_err, "label": r"Bauswein+2013", "fit": True}

    for key in datasets.keys():
        datasets[key]["x_dic"] = {"v_n": "vel_inf_ave-geo_fit", "err": None, "deferr": None, "mod": {}}
        datasets[key]["y_dic"] = {"v_n": "vel_inf_ave-geo",     "err": "ud", "deferr": 0.2,  "mod": {}}
        datasets[key]["mod_x"] = {}
        datasets[key]["facecolor"] = 'none'
        datasets[key]["plot_errorbar"] = False
        datasets[key]["edgecolor"] = md.datasets_colors[key]
        datasets[key]["marker"] = md.datasets_markers[key]
        datasets[key]["label"] = md.datasets_labels[key]
        datasets[key]["ms"] = 40
        datasets[key]["alpha"] = 0.8

    from make_fit4 import Fit_Data
    o_fit = Fit_Data(md.simulations, "vel_inf_ave-geo", err_method="default", clean_nans=True)

    fit_dics = OrderedDict()
    coeffs, _, _, _ = o_fit.fit_curve(ff_name="poly22_qLambda", cf_name="poly22")
    fit_dics["poly2"] = { # 5.6
        "func": FittingFunctions.poly_2_qLambda,
        "coeffs": np.array(coeffs), "to_val": None,
        "xmin": 0.05, "xmax": .3, "xscale": "linear", "xlabel": r"$\upsilon_{\rm ej;fit}$ [c]",
        "ymin": 0.08, "ymax": .40, "yscale": "linear", "ylabel": r"$\upsilon_{\rm ej}$ [c]",
        "plot_zero": True
    }
    coeffs, _, _, _ = o_fit.fit_curve(ff_name="diet16", cf_name="diet16")
    fit_dics["Eq.9"] = { # 6.1
        "func": FittingFunctions.vej_dietrich16, "coeffs": np.array(coeffs), "to_val": None,
        "xmin": 0.05, "xmax": .3, "xscale": "linear", "xlabel": r"$\upsilon_{\rm ej;fit}$ [c]",
        "ymin": 0.08, "ymax": .40, "yscale": "linear", "ylabel": r"$\upsilon_{\rm ej}$ [c]",
        "plot_zero": True
    }
    coeffs, _, _, _ = o_fit.fit_curve(ff_name="poly2_Lambda", cf_name="poly2")
    fit_dics["poly1"] = { # 6.1
        "func": FittingFunctions.poly_2_Lambda, "coeffs": np.array(coeffs), "to_val": None,
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
        "figname": __outplotdir__ + "residuals_vej.png",
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

    from model_sets import combined as md

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
    datasets['reference'] = {"models": md.reference, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": "v_n"}
    datasets["radiceLK"] =  {"models": md.radice18lk, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": md.params.vej_err}
    datasets["radiceM0"] =  {"models": md.radice18m0, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": md.params.vej_err}
    # # # datasets["lehner"] =    {"models": md.lehner16, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": md.params.vej_err}
    # # # datasets["kiuchi"] =    {"models": md.kiuchi19, "data": md, "fit": True, "color": None, "plot_errorbar": False, "err": md.params.vej_err}
    datasets["vincent"] =   {"models": md.vincent19, "data": md, "fit": True, "color": None, "plot_errorbar": False, "err": md.params.vej_err}
    # # # datasets["dietrich16"] ={"models": md.dietrich16, "data": md, "fit": True, "color": None, "plot_errorbar": False, "err": md.params.vej_err}
    # # # datasets["dietrich15"] ={"models": md.dietrich15, "data": md, "fit": True, "color": None, "plot_errorbar": False, "err": md.params.vej_err}
    datasets["sekiguchi15"]={"models": md.sekiguchi15, "data": md, "err": md.params.vej_err, "label": r"Sekiguchi+2015",  "fit": True}
    datasets["sekiguchi16"]={"models": md.sekiguchi16, "data": md, "err": md.params.vej_err, "label": r"Sekiguchi+2016", "fit": True}
    # # # datasets["hotokezaka"] ={"models": md.hotokezaka12, "data": md, "err": md.params.vej_err, "label": r"Hotokezaka+2013", "fit": True}
    # # # datasets["bauswein"] =  {"models": md.basuwein13, "data": md, "err": md.params.vej_err, "label": r"Bauswein+2013", "fit": True}

    # -------------------------------------------

    for key in datasets.keys():
        datasets[key]["x_dic"] = {"v_n": "Ye_ave-geo_fit", "err": None, "deferr": None, "mod": {}}
        datasets[key]["y_dic"] = {"v_n": "Ye_ave-geo",     "err": "ud", "deferr": 0.2,  "mod": {}}
        datasets[key]["mod_x"] = {}
        datasets[key]["facecolor"] = 'none'
        datasets[key]["edgecolor"] = md.datasets_colors[key]
        datasets[key]["marker"] = md.datasets_markers[key]
        datasets[key]["label"] = md.datasets_labels[key]
        datasets[key]["ms"] = 40
        datasets[key]["alpha"] = 0.8
        datasets[key]["plot_errorbar"] = False

    from make_fit4 import Fit_Data
    o_fit = Fit_Data(md.simulations, "Ye_ave-geo", err_method="default", clean_nans=True)

    fit_dics = OrderedDict()
    coeffs, _, _, _ = o_fit.fit_curve(ff_name="poly22_qLambda", cf_name="poly22")
    fit_dics["poly2"] = { # 17
        "func": FittingFunctions.poly_2_qLambda,
        "method": "delta",
        "coeffs": np.array(coeffs), "to_val": None,
        "xmin": 0.1, "xmax": .25, "xscale": "linear", "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
        "ymin": -0.3, "ymax": 0.3, "yscale": "linear", "ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
        "plot_zero": True
    }
    coeffs, _, _, _ = o_fit.fit_curve(ff_name="our", cf_name="our")
    fit_dics["Eq.11"] = { # 23
        "method": "delta",
        "func": FittingFunctions.yeej_ours, "coeffs": np.array(coeffs), "to_val": None,
        "xmin": 0.1, "xmax": .25, "xscale": "linear", "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
        "ymin": -0.3, "ymax": 0.3, "yscale": "linear", "ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
        "plot_zero": True
    }
    coeffs, _, _, _ = o_fit.fit_curve(ff_name="poly2_Lambda", cf_name="poly2")
    fit_dics["poly1"] = { # 30+
        "method": "delta",
        "func": FittingFunctions.poly_2_Lambda, "coeffs": np.array(coeffs), "to_val": None,
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
        "xmin": 0.05, "xmax": .225, "xscale": "linear",  # "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
        "ymin": -0.18, "ymax": 0.18, "yscale": "linear",  # "ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
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
        "xmin": 0.05, "xmax": .225, "xscale": "linear",  # "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
        "ymin": -0.18, "ymax": 0.18, "yscale": "linear",  # "ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
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
        "xmin": 0.05, "xmax": .225, "xscale": "linear",  # "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
        "ymin": -0.18, "ymax": 0.18, "yscale": "linear",  # "ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
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
        "figname": __outplotdir__ + "residuals_yeej.png",
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

    from model_sets import combined as md

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
    datasets['reference'] = {"models": md.reference, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": "v_n"}
    datasets["radiceLK"] =  {"models": md.radice18lk, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": md.params.vej_err}
    datasets["radiceM0"] =  {"models": md.radice18m0, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": md.params.vej_err}
    # # # datasets["lehner"] =    {"models": md.lehner16, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": md.params.vej_err}
    # # # datasets["kiuchi"] =    {"models": md.kiuchi19, "data": md, "fit": True, "color": None, "plot_errorbar": False, "err": md.params.vej_err}
    # # # datasets["vincent"] =   {"models": md.vincent19, "data": md, "fit": True, "color": None, "plot_errorbar": False, "err": md.params.vej_err}
    # # # datasets["dietrich16"] ={"models": md.dietrich16, "data": md, "fit": True, "color": None, "plot_errorbar": False, "err": md.params.vej_err}
    # # # datasets["dietrich15"] ={"models": md.dietrich15, "data": md, "fit": True, "color": None, "plot_errorbar": False, "err": md.params.vej_err}
    # # # datasets["sekiguchi15"]={"models": md.sekiguchi15, "data": md, "err": md.params.vej_err, "label": r"Sekiguchi+2015",  "fit": True}
    # # # datasets["sekiguchi16"]={"models": md.sekiguchi16, "data": md, "err": md.params.vej_err, "label": r"Sekiguchi+2016", "fit": True}
    # # # datasets["hotokezaka"] ={"models": md.hotokezaka12, "data": md, "err": md.params.vej_err, "label": r"Hotokezaka+2013", "fit": True}
    # # # datasets["bauswein"] =  {"models": md.basuwein13, "data": md, "err": md.params.vej_err, "label": r"Bauswein+2013", "fit": True}

    # -------------------------------------------

    for key in datasets.keys():
        datasets[key]["x_dic"] = {"v_n": "theta_rms-geo_fit", "err": None, "deferr": None, "mod": {}}
        datasets[key]["y_dic"] = {"v_n": "theta_rms-geo",     "err": "ud", "deferr": 0.2,  "mod": {}}
        datasets[key]["mod_x"] = {}
        datasets[key]["facecolor"] = 'none'
        datasets[key]["edgecolor"] = md.datasets_colors[key]
        datasets[key]["marker"] = md.datasets_markers[key]
        datasets[key]["label"] = md.datasets_labels[key]
        datasets[key]["ms"] = 40
        datasets[key]["alpha"] = 0.8
        datasets[key]["plot_errorbar"] = False

    from make_fit4 import Fit_Data
    o_fit = Fit_Data(md.simulations, "theta_rms-geo", err_method="default", clean_nans=True)

    fit_dics = OrderedDict()
    coeffs, _, _, _ = o_fit.fit_curve(ff_name="poly22_qLambda", cf_name="poly22")
    fit_dics["poly2"] = { # 17
        "func": FittingFunctions.poly_2_qLambda,
        "method": "delta",
        "coeffs": np.array(coeffs), "to_val": None,
        "xmin": 0.1, "xmax": .25, "xscale": "linear", "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
        "ymin": -0.3, "ymax": 0.3, "yscale": "linear", "ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
        "plot_zero": True
    }
    coeffs, _, _, _ = o_fit.fit_curve(ff_name="poly2_Lambda", cf_name="poly2")
    fit_dics["poly1"] = { # 30+
        "method": "delta",
        "func": FittingFunctions.poly_2_Lambda, "coeffs": np.array(coeffs), "to_val": None,
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
        "xmin": 0, "xmax": 35, "xscale": "linear",  # "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
        "ymin": -18, "ymax": 18, "yscale": "linear",  # "ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
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
        "xmin": 0, "xmax": 35, "xscale": "linear",  # "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
        "ymin": -18, "ymax": 18, "yscale": "linear",  # "ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
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
        "figname": __outplotdir__ + "residual_thetaej.png",
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

# ---

def task_plot_mdisk_fits_only_group():

    from model_sets import combined as md
    from make_fit4 import Fit_Data

    # -------------------------------------------

    datasets = OrderedDict()
    datasets['refset'] =    {"models": md.group_refset, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": "v_n"}
    datasets["heatcool"] =  {"models": md.group_heatcool, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": md.params.MdiskPP_err}
    datasets["cool"] =      {"models": md.group_cool, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": md.params.MdiskPP_err}
    datasets["none"] =      {"models": md.group_none, "data": md, "fit": True, "color": None, "plot_errorbar": False, "err": md.params.MdiskPP_err}

    for key in datasets.keys():
        datasets[key]["x_dic"] = {"v_n": "Mdisk3D_fit", "err": None, "deferr": None, "mod": {}}
        datasets[key]["y_dic"] = {"v_n": "Mdisk3D", "err": "ud", "deferr": 0.2, "mod": {}}
        datasets[key]["mod_x"] = {}
        datasets[key]["plot_errorbar"] = False
        datasets[key]["facecolor"] = 'none'
        datasets[key]["edgecolor"] = md.dataset_group_colors[key]
        datasets[key]["marker"] = md.dataset_group_markers[key]
        datasets[key]["label"] = md.dataset_group_labels[key]
        datasets[key]["ms"] = 40
        datasets[key]["alpha"] = 0.8

    from make_fit4 import Fit_Data
    o_fit = Fit_Data(md.simulations, "Mdisk3D", err_method="default", clean_nans=True)

    fit_dics = OrderedDict()
    coeffs, _, _, _ = o_fit.fit_curve(ff_name="poly22_qLambda", cf_name="poly22", modify=None, usesigma=False)
    fit_dics["poly2"] = { # 202 --  230
        "func": FittingFunctions.poly_2_qLambda, "coeffs": np.array(coeffs),"to_val":None,#"10**",# None,
        "xmin": 0.05, "xmax": .3, "xscale": "linear", "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
        "plot_zero": True
    }
    coeffs, _, _, _ = o_fit.fit_curve(ff_name="krug19", cf_name="krug19", modify=None, usesigma=False)
    fit_dics["Eq.18"] = { # 481.6
        "func": FittingFunctions.mdisk_kruger19, "coeffs": np.array(coeffs), "to_val":None,#"10**",# None,
        "xmin": 0.05, "xmax": .3, "xscale": "linear", "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
        "plot_zero": True
    }
    coeffs, _, _, _ = o_fit.fit_curve(ff_name="rad18", cf_name="rad18", modify=None, usesigma=False)
    fit_dics["Eq.17"] = { # 630 -- 730
        "func": FittingFunctions.mdisk_radice18, "coeffs": np.array(coeffs), "to_val":None,#"10**",# None,
        "xmin": 0.05, "xmax": .3, "xscale": "linear", "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
        "plot_zero": True
    }
    coeffs, _, _, _ = o_fit.fit_curve(ff_name="poly2_Lambda", cf_name="poly2", modify=None, usesigma=False)
    fit_dics["poly1"] = { # 640 -- 750
        "func": FittingFunctions.poly_2_Lambda, "coeffs": np.array(coeffs),  "to_val":None,#"10**",# None,
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
        "xmin": -0.02, "xmax": .30, "xscale": "linear",
        "ymin": -4.2, "ymax": 1.8, "yscale": "linear",
        # "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
        # "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
        "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False, "tick1On": True, "tick2On": True,
                        "labelsize": 14,
                        "direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        "text": {'x': 0.85, 'y': 0.85, 's':r"$P_2^2(q,\tilde{\Lambda})$", 'fontsize': 14, 'color': 'black',
                 'horizontalalignment': 'center'},
        "plot_zero": True,
        "labels": True
    }
    subplot_dics["Eq.18"] = {
        "xmin": -0.02, "xmax": .30, "xscale": "linear",
        "ymin": -4.2, "ymax": 1.8, "yscale": "linear",
        # "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
        # "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
        "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False, "tick1On": True, "tick2On": True,
                        "labelsize": 14,
                        "direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        "text": {'x': 0.85, 'y': 0.90, 's': r"Eq.(18)", 'fontsize': 14, 'color': 'black',
                 'horizontalalignment': 'center'},
        "plot_zero": True,
        "labels": True
    }
    subplot_dics["Eq.17"] = {
        "xmin": -0.02, "xmax": .30, "xscale": "linear",
        "ymin": -4.2, "ymax": 1.8, "yscale": "linear",
        # "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
        # "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
        "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False, "tick1On": True, "tick2On": True,
                        "labelsize": 14,
                        "direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        "text": {'x': 0.85, 'y': 0.90, 's': r"Eq.(17)", 'fontsize': 14, 'color': 'black',
                 'horizontalalignment': 'center'},
        "plot_zero": True,
        "labels": True,
        "legend": {
            # "last_n": 4,
            "fancybox": False, "loc": 'lower right', "columnspacing": 0.4,
            # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
            "shadow": "False", "ncol": 1, "fontsize": 14,
            "framealpha": 0., "borderaxespad": 0., "frameon": False}
    }
    subplot_dics["poly1"] = {
        "xmin": -0.02, "xmax": .30, "xscale": "linear",
        "ymin": -4.2, "ymax": 1.8, "yscale": "linear",
        # "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
        # "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
        "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False, "tick1On": True, "tick2On": True,
                        "labelsize": 14,
                        "direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        "text": {'x': 0.85, 'y': 0.85, 's': r"$P_2^1(\tilde{\Lambda})$", 'fontsize': 14, 'color': 'black',
                 'horizontalalignment': 'center'},
        "plot_zero": True,
        "labels": True,
        # "legend": {
        #     "first_n":4,
        #     "fancybox": False, "loc": 'lower right', "columnspacing": 0.4,
        #     # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #     "shadow": "False", "ncol": 1, "fontsize": 11,
        #     "framealpha": 0., "borderaxespad": 0., "frameon": False}
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
        "dpi": 128, "fontsize": 15, "labelsize": 15,
        "tight_layout": False,
        "xlabel": r"$M_{\rm disk;fit}$ $[M_{\odot}]$",
        "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$", #r"$M_{\rm disk}$ $[10^{-3}M_{\odot}]$",
        "tick_params": {"labelcolor":'none', "top":"False", "bottom":False, "left":False, "right":False},
        "savepdf": True,
        "figname": __outplotdir__ + "residuals_sets_disk_mass.png",
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

def task_plot_mej_fits_only_group(usesigma=True, figname="residuals_sets_mej.png"):
    # from model_sets import models_vincent2019 as vi
    # from model_sets import models_radice2018 as rd
    # from model_sets import groups as md
    # from model_sets import models_lehner2016 as lh  # [22] arxive:1603.00501
    # from model_sets import models_kiuchi2019 as ki
    # from model_sets import models_dietrich2015 as di15  # [21] arxive:1504.01266
    # from model_sets import models_dietrich2016 as di16  # [24] arxive:1607.06636
    # from model_sets import models_sekiguchi2016 as se16  # [23] arxive:1603.01918 # no Mb
    # from model_sets import models_sekiguchi2015 as se15  # [-]  arxive:1502.06660 # no Mb
    # from model_sets import models_hotokezaka2013 as hz          # [19] arxive:1212.0905
    # from model_sets import models_bauswein2013 as bs            # [20] arxive:1302.6530

    from model_sets import combined as md
    from make_fit4 import Fit_Data

    # -------------------------------------------

    datasets = OrderedDict()
    datasets['refset'] = {"models": md.group_refset, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": "v_n"}
    datasets["heatcool"] = {"models": md.group_heatcool, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": md.params.MdiskPP_err}
    datasets["cool"] = {"models": md.group_cool, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": md.params.MdiskPP_err}
    datasets["none"] = {"models": md.group_none, "data": md, "fit": True, "color": None, "plot_errorbar": False, "err": md.params.MdiskPP_err}

    for key in datasets.keys():
        datasets[key]["x_dic"] = {"v_n": "Mej_tot-geo_fit", "err": None, "deferr": None, "mod":{"mult": [1e3]}}#{"mult": [1e3]}}#, "mod": {"mult": [1e3]}}
        datasets[key]["y_dic"] = {"v_n": "Mej_tot-geo",     "err": "ud", "deferr": 0.2, "mod":{"mult": [1e3]}}#}{"mult": [1e3]}}#,  "mod": {"mult": [1e3]}}
        datasets[key]["mod_x"] = {}
        datasets[key]["facecolor"] = 'none'
        datasets[key]["plot_errorbar"] = False
        datasets[key]["edgecolor"] = md.dataset_group_colors[key]
        datasets[key]["marker"] = md.dataset_group_markers[key]
        datasets[key]["label"] = md.dataset_group_labels[key]
        datasets[key]["ms"] = 40
        datasets[key]["alpha"] = 0.8


    o_fit = Fit_Data(md.simulations, "Mej_tot-geo", err_method="default", clean_nans=True)

    # coeffs, _, _, _ = o_fit.fit_curve(ff_name="poly22_qLambda", cf_name="poly22", modify="log10")
    fit_dics = OrderedDict()
    coeffs, _, _, _ = o_fit.fit_curve(ff_name="krug19", cf_name="krug19", modify="log10", usesigma=usesigma)
    fit_dics["Eq.10"] = {  # best 53.2
        "func": FittingFunctions.mej_kruger19, "coeffs": np.array(coeffs), "to_val": "10**",# None,
        "xmin": 0, "xmax": 30., "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -100.0, "ymax": 100.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
        "plot_zero": True
    }
    coeffs, _, _, _ = o_fit.fit_curve(ff_name="poly22_qLambda", cf_name="poly22", modify="log10", usesigma=usesigma)
    fit_dics["poly2"] = { # 2nd best  81.5
        "func": FittingFunctions.poly_2_qLambda, "coeffs": np.array(coeffs), "to_val": "10**",#None,
        "xmin": 0, "xmax": 30., "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -100.0, "ymax": 100.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
        "plot_zero": True

    }
    coeffs, _, _, _ = o_fit.fit_curve(ff_name="diet16", cf_name="diet16", modify="log10", usesigma=usesigma)
    fit_dics["Eq.9"] = { # 153
        "func": FittingFunctions.mej_dietrich16, "coeffs": np.array(coeffs), "to_val": "10**",# None,
        "xmin": 0, "xmax": 30., "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -100.0, "ymax": 100.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
        "plot_zero": True
    }
    coeffs, _, _, _ = o_fit.fit_curve(ff_name="poly2_Lambda", cf_name="poly2", modify="log10", usesigma=usesigma)
    fit_dics["poly1"] = { # 628
        "func": FittingFunctions.poly_2_Lambda, "coeffs": np.array(coeffs), "to_val": "10**",# None,
        "xmin": 0, "xmax": 30., "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
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
    subplot_dics["Eq.10"] = {
        "xmin": -0, "xmax": 30., "xscale": "linear",
        "ymin": -12., "ymax": 2.2, "yscale": "linear",
        # "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
        # "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
        "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False, "tick1On": True, "tick2On": True,
                        "labelsize": 14,
                        "direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        "text": {'x': 0.85, 'y': 0.90, 's': r"Eq.(10)", 'fontsize': 14, 'color': 'black',
                 'horizontalalignment': 'center'},
        "plot_zero": True,
        "labels": True,
    }
    subplot_dics["poly2"] = {
        "xmin": -0, "xmax": 30., "xscale": "linear",
        "ymin": -12., "ymax": 2.2, "yscale": "linear",
        # "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
        # "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
        "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False, "tick1On": True, "tick2On": True,
                        "labelsize": 14,
                        "direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        "text": {'x': 0.85, 'y': 0.85, 's': r"$P_2^2(q,\tilde{\Lambda})$", 'fontsize': 14, 'color': 'black',
                 'horizontalalignment': 'center'},
        "plot_zero": True,
        "labels": True
    }
    subplot_dics["Eq.9"] = {
        "xmin": -0, "xmax": 30., "xscale": "linear",
        "ymin": -12., "ymax": 2.2, "yscale": "linear",
        # "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
        # "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
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
        "xmin": -0, "xmax": 30., "xscale": "linear",
        "ymin": -12., "ymax": 2.2, "yscale": "linear",
        # "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
        # "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
        "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False, "tick1On": True, "tick2On": True,
                        "labelsize": 14,
                        "direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        "text": {'x': 0.85, 'y': 0.85, 's': r"$P_2^1(\tilde{\Lambda})$", 'fontsize': 14, 'color': 'black',
                 'horizontalalignment': 'center'},
        "legend": {"fancybox": False, "loc": 'lower right', "columnspacing": 0.4,
                   # "bbox_to_anchor": (0.8, 1.1),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
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
        "dpi": 128, "fontsize": 15, "labelsize": 15,
        "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$", #r"$M_{\rm disk}$ $[10^{-3}M_{\odot}]$",
        "tick_params": {"labelcolor":'none', "top":False, "bottom":False, "left":False, "right":False},
        "savepdf": True,
        "figname": __outplotdir__ + figname,
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

def task_plot_vej_fits_only_group(usesigma=False, figname="residuals_sets_vej.png"):

    from model_sets import combined as md # "vel_inf_ave-geo"


    # datasets["kiuchi"] =    {'marker': "X", "ms": 20, "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019", "color": "gray", "fit": False}
    # datasets["radice"] =    {'marker': '*', "ms": 40, "models": rd.simulations[rd.fiducial], "data":rd, "err":rd.params.MdiskPP_err, "label": r"Radice+2018", "color": "blue", "fit": True}
    # # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": di.simulations[di.mask_for_with_sr], "data":di, "err":di.params.Mej_err, "label":"Dietrich+2016"}
    # # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}
    # # datasets["vincent"] =   {'marker': 'v', 'ms': 20, "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019", "color": "blue", "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    # # datasets["bauswein"] =  {'marker': 's', 'ms': 20, "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "color": "blue", "fit": False}
    # # datasets["lehner"] =    {'marker': 'P', 'ms': 20, "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016", "color": "blue", "fit": False}
    # # datasets["hotokezaka"] ={'marker': '>', 'ms': 20, "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "color": "gray", "fit": False}
    # datasets['our'] =       {'marker': 'o', 'ms': 40, "models": md.groups, "data": md, "err": "v_n", "label": r"This work", "color": "red", "fit": True}
    datasets = OrderedDict()
    datasets['refset'] = {"models": md.group_refset, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": "v_n"}
    datasets["heatcool"] = {"models": md.group_heatcool, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": md.params.MdiskPP_err}
    datasets["cool"] = {"models": md.group_cool, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": md.params.MdiskPP_err}
    datasets["none"] = {"models": md.group_none, "data": md, "fit": True, "color": None, "plot_errorbar": False, "err": md.params.MdiskPP_err}

    for key in datasets.keys():
        datasets[key]["x_dic"] = {"v_n": "vel_inf_ave-geo_fit", "err": None, "deferr": None, "mod": {}}
        datasets[key]["y_dic"] = {"v_n": "vel_inf_ave-geo",     "err": "ud", "deferr": 0.2,  "mod": {}}
        datasets[key]["mod_x"] = {}
        datasets[key]["facecolor"] = 'none'
        datasets[key]["plot_errorbar"] = False
        datasets[key]["edgecolor"] = md.dataset_group_colors[key]
        datasets[key]["marker"] = md.dataset_group_markers[key]
        datasets[key]["label"] = md.dataset_group_labels[key]
        datasets[key]["ms"] = 40
        datasets[key]["alpha"] = 0.8

    from make_fit4 import Fit_Data
    o_fit = Fit_Data(md.simulations, "vel_inf_ave-geo", err_method="default", clean_nans=True)

    fit_dics = OrderedDict()
    coeffs, _, _, _ = o_fit.fit_curve(ff_name="diet16", cf_name="diet16", usesigma=usesigma)
    fit_dics["Eq.12"] = { # 6.1
        "func": FittingFunctions.vej_dietrich16, "coeffs": np.array(coeffs), "to_val": None,
        "xmin": 0.05, "xmax": .3, "xscale": "linear", "xlabel": r"$\upsilon_{\rm ej;fit}$ [c]",
        "ymin": 0.08, "ymax": .40, "yscale": "linear", "ylabel": r"$\upsilon_{\rm ej}$ [c]",
        "plot_zero": True
    }
    coeffs, _, _, _ = o_fit.fit_curve(ff_name="poly22_qLambda", cf_name="poly22", usesigma=usesigma)
    fit_dics["poly2"] = { # 5.6
        "func": FittingFunctions.poly_2_qLambda,
        "coeffs": np.array(coeffs), "to_val": None,
        "xmin": 0.05, "xmax": .3, "xscale": "linear", "xlabel": r"$\upsilon_{\rm ej;fit}$ [c]",
        "ymin": 0.08, "ymax": .40, "yscale": "linear", "ylabel": r"$\upsilon_{\rm ej}$ [c]",
        "plot_zero": True
    }
    coeffs, _, _, _ = o_fit.fit_curve(ff_name="poly2_Lambda", cf_name="poly2", usesigma=usesigma)
    fit_dics["poly1"] = { # 6.1
        "func": FittingFunctions.poly_2_Lambda, "coeffs": np.array(coeffs), "to_val": None,
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
    subplot_dics["Eq.12"] = {
        "xmin": 0.1, "xmax": .3, "xscale": "linear",  # "xlabel": r"$\upsilon_{\rm ej;fit}$ [c]",
        "ymin": -1.1, "ymax": 1.1, "yscale": "linear",  # "ylabel": r"$\Delta \upsilon_{\rm ej} / \upsilon_{\rm ej}$",
        "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False, "tick1On": True, "tick2On": True,
                        "labelsize": 14,
                        "direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        "text": {'x': 0.85, 'y': 0.90, 's': r"Eq.(12)", 'fontsize': 14, 'color': 'black',
                 'horizontalalignment': 'center'},
        "plot_zero": True,
        "labels": True
    }
    subplot_dics["poly2"] = {
        "xmin": 0.1, "xmax": .3, "xscale": "linear",  # "xlabel": r"$\upsilon_{\rm ej;fit}$ [c]",
        "ymin": -1.1, "ymax": 1.1, "yscale": "linear",  # "ylabel": r"$\Delta \upsilon_{\rm ej} / \upsilon_{\rm ej}$",
        "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False, "tick1On": True, "tick2On": True,
                        "labelsize": 14,
                        "direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        "text": {'x': 0.85, 'y': 0.85, 's': r"$P_2^2(q,\tilde{\Lambda})$", 'fontsize': 14, 'color': 'black',
                 'horizontalalignment': 'center'},
        "plot_zero": True,
        "labels": True
    }
    subplot_dics["poly1"] = {
        "xmin": 0.08, "xmax": .29, "xscale": "linear",  # "xlabel": r"$\upsilon_{\rm ej;fit}$ [c]",
        "ymin": -1.1, "ymax": 1.1, "yscale": "linear",  # "ylabel": r"$\Delta \upsilon_{\rm ej} / \upsilon_{\rm ej}$",
        "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False, "tick1On": True, "tick2On": True,
                        "labelsize": 14,
                        "direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        "text": {'x': 0.85, 'y': 0.85, 's': r"$P_2^1(\tilde{\Lambda})$", 'fontsize': 14, 'color': 'black',
                 'horizontalalignment': 'center'},
        "plot_zero": True,
        "legend": {"fancybox": False, "loc": 'lower left',
                   # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 14, "columnspacing": 0.4,
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
        "dpi": 128, "fontsize": 15, "labelsize": 15,
        "tight_layout": False,
        "xlabel": r"$\upsilon_{\rm ej;fit}$ [c]",
        "ylabel": r"$\Delta \upsilon_{\rm ej} / \upsilon_{\rm ej}$", #r"$M_{\rm disk}$ $[10^{-3}M_{\odot}]$",
        "tick_params": {"labelcolor":'none', "top":"False", "bottom":False, "left":False, "right":False},
        "savepdf": True,
        "figname": __outplotdir__ + figname,
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

def task_plot_yeej_fits_only_group(usesigma=True, figname="residuals_sets_yeej.png"):

    from model_sets import combined as md

    datasets = OrderedDict()
    datasets['refset'] = {"models": md.group_refset, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": "v_n"}
    datasets["heatcool"] = {"models": md.group_heatcool, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": md.params.MdiskPP_err}
    datasets["cool"] = {"models": md.group_cool, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": md.params.MdiskPP_err}
    datasets["none"] = {"models": md.group_none, "data": md, "fit": True, "color": None, "plot_errorbar": False, "err": md.params.MdiskPP_err}

    for key in datasets.keys():
        datasets[key]["x_dic"] = {"v_n": "Ye_ave-geo_fit", "err": None, "deferr": None, "mod": {}}
        datasets[key]["y_dic"] = {"v_n": "Ye_ave-geo",     "err": "ud", "deferr": 0.2,  "mod": {}}
        datasets[key]["mod_x"] = {}
        datasets[key]["facecolor"] = 'none'
        datasets[key]["edgecolor"] = md.dataset_group_colors[key]
        datasets[key]["marker"] = md.dataset_group_markers[key]
        datasets[key]["label"] = md.dataset_group_labels[key]
        datasets[key]["ms"] = 40
        datasets[key]["alpha"] = 0.8
        datasets[key]["plot_errorbar"] = False

    from make_fit4 import Fit_Data
    o_fit = Fit_Data(md.simulations, "Ye_ave-geo", err_method="default", clean_nans=True)

    fit_dics = OrderedDict()
    coeffs, _, _, _ = o_fit.fit_curve(ff_name="poly22_qLambda", cf_name="poly22", usesigma=usesigma)
    fit_dics["poly2"] = { # 17
        "func": FittingFunctions.poly_2_qLambda,
        "method": "delta",
        "coeffs": np.array(coeffs), "to_val": None,
        "xmin": 0.1, "xmax": .25, "xscale": "linear", "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
        "ymin": -0.3, "ymax": 0.3, "yscale": "linear", "ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
        "plot_zero": True
    }
    # coeffs, _, _, _ = o_fit.fit_curve(ff_name="our", cf_name="our")
    # fit_dics["Eq.11"] = { # 23
    #     "method": "delta",
    #     "func": FittingFunctions.yeej_ours, "coeffs": np.array(coeffs), "to_val": None,
    #     "xmin": 0.1, "xmax": .25, "xscale": "linear", "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
    #     "ymin": -0.3, "ymax": 0.3, "yscale": "linear", "ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
    #     "plot_zero": True
    # }
    coeffs, _, _, _ = o_fit.fit_curve(ff_name="poly2_Lambda", cf_name="poly2", usesigma=usesigma)
    fit_dics["poly1"] = { # 30+
        "method": "delta",
        "func": FittingFunctions.poly_2_Lambda, "coeffs": np.array(coeffs), "to_val": None,
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
        "xmin": 0.05, "xmax": .225, "xscale": "linear",  # "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
        "ymin": -0.18, "ymax": 0.18, "yscale": "linear",  # "ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
        "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False, "tick1On": True, "tick2On": True,
                        "labelsize": 14,
                        "direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        "text": {'x': 0.85, 'y': 0.85, 's':r"$P_2^2(q,\tilde{\Lambda})$", 'fontsize': 14, 'color': 'black',
                 'horizontalalignment': 'center'},
        "plot_zero": True,
        "labels": True
    }
    # subplot_dics["Eq.11"] = {
    #     "xmin": 0.05, "xmax": .225, "xscale": "linear",  # "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
    #     "ymin": -0.18, "ymax": 0.18, "yscale": "linear",  # "ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
    #     "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
    #                     "labelright": False, "tick1On": True, "tick2On": True,
    #                     "labelsize": 14,
    #                     "direction": 'in',
    #                     "bottom": True, "top": True, "left": True, "right": True},
    #     "text": {'x': 0.85, 'y': 0.90, 's': r"Eq.(11)", 'fontsize': 14, 'color': 'black',
    #              'horizontalalignment': 'center'},
    #     "plot_zero": True,
    #     "labels": True
    # }
    subplot_dics["poly1"] = {
        "xmin": 0.02, "xmax": .23, "xscale": "linear",  # "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
        "ymin": -0.18, "ymax": 0.18, "yscale": "linear",  # "ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
        "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False, "tick1On": True, "tick2On": True,
                        "labelsize": 14,
                        "direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        "text": {'x': 0.85, 'y': 0.85, 's': r"$P_2^1(\tilde{\Lambda})$",
                 'fontsize': 14, 'color': 'black', 'horizontalalignment': 'center'},
        "plot_zero": True,
        "legend": {"fancybox": False, "loc": 'lower left',
                   # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 14, "columnspacing": 0.4,
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
        "subplots":{"figsize": (6.0, 4.0), "ncols":1,"nrows":2, "sharex":True,"sharey":False}, # 6.0 6.0
        "subplot_adjust": {"left": 0.10, "bottom": 0.14, "top": 0.98, "right": 0.95, "hspace": 0},
        "dpi": 128, "fontsize": 15, "labelsize": 15,
        "tight_layout": False,
        "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
        "ylabel": r"$\Delta Y_{e\: \rm ej}$", #r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$"
        "tick_params": {"labelcolor":'none', "top":"False", "bottom":False, "left":False, "right":False},
        "savepdf": True,
        "figname": __outplotdir__ + figname,
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

def task_plot_thetaej_fits_only_group(usesigma=True, figname="residual_sets_thetaej.png"):

    from model_sets import combined as md

    datasets = OrderedDict()
    datasets['refset'] = {"models": md.group_refset, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": "v_n"}
    datasets["heatcool"] = {"models": md.group_heatcool, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": md.params.MdiskPP_err}
    datasets["cool"] = {"models": md.group_cool, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": md.params.MdiskPP_err}
    datasets["none"] = {"models": md.group_none, "data": md, "fit": True, "color": None, "plot_errorbar": False, "err": md.params.MdiskPP_err}

    for key in datasets.keys():
        datasets[key]["x_dic"] = {"v_n": "theta_rms-geo_fit", "err": None, "deferr": None, "mod": {}}
        datasets[key]["y_dic"] = {"v_n": "theta_rms-geo",     "err": "ud", "deferr": 0.2,  "mod": {}}
        datasets[key]["mod_x"] = {}
        datasets[key]["facecolor"] = 'none'
        datasets[key]["edgecolor"] = md.dataset_group_colors[key]
        datasets[key]["marker"] = md.dataset_group_markers[key]
        datasets[key]["label"] = md.dataset_group_labels[key]
        datasets[key]["ms"] = 40
        datasets[key]["alpha"] = 0.8
        datasets[key]["plot_errorbar"] = False

    from make_fit4 import Fit_Data
    o_fit = Fit_Data(md.simulations, "theta_rms-geo", err_method="default", clean_nans=True)

    fit_dics = OrderedDict()
    coeffs, _, _, _ = o_fit.fit_curve(ff_name="poly22_qLambda", cf_name="poly22", usesigma=usesigma)
    fit_dics["poly2"] = { # 17
        "func": FittingFunctions.poly_2_qLambda,
        "method": "delta",
        "coeffs": np.array(coeffs), "to_val": None,
        "xmin": 0.1, "xmax": .25, "xscale": "linear", "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
        "ymin": -0.3, "ymax": 0.3, "yscale": "linear", "ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
        "plot_zero": True
    }
    coeffs, _, _, _ = o_fit.fit_curve(ff_name="poly2_Lambda", cf_name="poly2", usesigma=usesigma)
    fit_dics["poly1"] = { # 30+
        "method": "delta",
        "func": FittingFunctions.poly_2_Lambda, "coeffs": np.array(coeffs), "to_val": None,
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
        "xmin": 0, "xmax": 35, "xscale": "linear",  # "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
        "ymin": -23, "ymax": 23, "yscale": "linear",  # "ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
        "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False, "tick1On": True, "tick2On": True,
                        "labelsize": 14,
                        "direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        "text": {'x': 0.85, 'y': 0.85, 's':r"$P_2^2(q,\tilde{\Lambda})$", 'fontsize': 14, 'color': 'black',
                 'horizontalalignment': 'center'},
        "plot_zero": True,
        "labels": True
    }
    subplot_dics["poly1"] = {
        "xmin": 5, "xmax": 35, "xscale": "linear",  # "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
        "ymin": -23, "ymax": 23, "yscale": "linear",  # "ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
        "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False, "tick1On": True, "tick2On": True,
                        "labelsize": 14,
                        "direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        "text": {'x': 0.85, 'y': 0.85, 's': r"$P_2^1(\tilde{\Lambda})$", 'fontsize': 14, 'color': 'black', 'horizontalalignment': 'center'},
        "plot_zero": True,
        "legend": {"fancybox": False, "loc": 'lower left',
                   # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 14, "columnspacing": 0.4,
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
        "dpi": 128, "fontsize": 15, "labelsize": 15,
        "tight_layout": False,
        "xlabel": r"$\theta_{\rm{RMS;\:fit}}$",
        "ylabel": r"$\Delta \theta_{\rm RMS}$", #r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$"
        "tick_params": {"labelcolor":'none', "top":"False", "bottom":False, "left":False, "right":False},
        "savepdf": True,
        "figname": __outplotdir__ + figname,
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


class Task_Post_Factum():

    @staticmethod
    def task_plot_mej_fits_only_group(
            dset = "refset",
            xmin = 1.5, xmax = 12.,
            vmin = 1., vmax = 1.4,
            modify = "log10", to_val = "10**", usesigma=True,
            note = ""
    ):
        from model_sets import combined as md
        from make_fit4 import Fit_Data

        datasets = OrderedDict()
        fit_dics = OrderedDict()

        # ---

        if dset == "refset":
            dset, dset_name = md.group_refset, 'refset'
        elif dset == "heatcool":
            dset, dset_name = md.group_heatcool, 'heatcool'
        elif dset == "cool":
            dset, dset_name = md.group_cool, 'cool'
        elif dset == "none":
            dset, dset_name = md.group_none, "none"
        else:
            raise NameError()


        v_n = "Mej_tot-geo"

        datasets[dset_name] = {"models": dset, "data": md, "fit": True, "color": None, "plot_errorbar": True,
                              "err": "v_n", "edgecolor": "blue"}
        o_fit = Fit_Data(dset, v_n, err_method="default", clean_nans=True)
        coeffs, _, chi2dof, _ = o_fit.fit_curve(ff_name="poly22_qLambda", cf_name="poly22", modify=modify,
                                                usesigma=usesigma)
        datasets[dset_name]["chi2dof"] = chi2dof

        fit_dics["poly2"] = {  # 2nd best  81.5
            "func": FittingFunctions.poly_2_qLambda, "coeffs": np.array(coeffs), "to_val": to_val,  # None,
            "xmin": 0, "xmax": 24., "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
            "ymin": -100.0, "ymax": 100.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
            "plot_zero": True
        }


        # datasets["heatcool"] = {"models": md.group_heatcool, "data": md, "fit": True, "color": None,
        #                         "plot_errorbar": True, "err": md.params.Mej_err, "edgecolor":"blue"}
        # o_fit = Fit_Data(md.group_heatcool, "Mej_tot-geo", err_method="default", clean_nans=True)
        # coeffs, _, chi2dof, _ = o_fit.fit_curve(ff_name="poly22_qLambda", cf_name="poly22", modify="log10", usesigma=True)
        # datasets["heatcool"]["chi2dof"] = chi2dof
        #
        # fit_dics["poly2"] = {  # 2nd best  81.5
        #     "func": FittingFunctions.poly_2_qLambda, "coeffs": np.array(coeffs), "to_val": "10**",  # None,
        #     "xmin": 1, "xmax": 12., "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        #     "ymin": -100.0, "ymax": 100.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
        #     "plot_zero": True
        # }

        # ---

        for key in datasets.keys():
            datasets[key]["x_dic"] = {"v_n": "Mej_tot-geo_fit", "err": None, "deferr": None,
                                      "mod": {"mult": [1e3]}}  # {"mult": [1e3]}}#, "mod": {"mult": [1e3]}}
            datasets[key]["y_dic"] = {"v_n": "Mej_tot-geo", "err": "ud", "deferr": 0.2,
                                      "mod": {"mult": [1e3]}}  # }{"mult": [1e3]}}#,  "mod": {"mult": [1e3]}}
            datasets[key]["c_dic"] = {"v_n": "q", "err": None, "deferr": None,
                                      "mod": {}}  # }{"mult": [1e3]}}#,  "mod": {"mult": [1e3]}}
            datasets[key]["mod_x"] = {}
            datasets[key]["facecolor"] = 'none'
            datasets[key]["plot_errorbar"] = False
            # datasets[key]["edgecolor"] = md.dataset_group_colors[key]
            datasets[key]["marker"] = md.dataset_group_markers[key]
            datasets[key]["label"] = md.dataset_group_labels[key] + \
                                     " $N$ = {:d}".format(len(o_fit.ds[v_n])) + '\n' \
                                     r" $\chi^2_{\nu} = $" + " {:.1f}".format(datasets[key]["chi2dof"]) + '\n' \
                                     r" $\tilde{\Lambda}$=["+str(int(min(o_fit.ds["Lambda"])))+" - "+str(int(max(o_fit.ds["Lambda"]))) + '] \n'+\
                                     r"$q$=[" + "{:.2f}".format(min(o_fit.ds["q"])) + " - " + "{:.2f}".format(max(o_fit.ds["q"])) + ']'

            datasets[key]["ms"] = 60
            datasets[key]["alpha"] = 1.0


        subplot_dics = OrderedDict()
        subplot_dics["poly2"] = {
            "vmin": vmin, "vmax": vmax, "cbar_label": "q", "cmap": "winter",
            "xmin": xmin, "xmax": xmax, "xscale": "linear",
            "ymin": -3., "ymax": 1, "yscale": "linear",
            # "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
            # "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
            "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                            "labelright": False, "tick1On": True, "tick2On": True,
                            "labelsize": 14,
                            "direction": 'in',
                            "bottom": True, "top": True, "left": True, "right": True},
            # "text": {'x': 0.85, 'y': 0.90, 's': r"$P_2^2(q,\tilde{\Lambda})$", 'fontsize': 14, 'color': 'black',
            #          'horizontalalignment': 'center'},
            "texts": [{'x': 0.85, 'y': 0.90, 's': r"$P_2^2(q,\tilde{\Lambda})$", 'fontsize': 14, 'color': 'black',
                       'horizontalalignment': 'center'}
                      # {'x': 0.15, 'y': 0.10, 's': r"$\chi_{\nu}^{\texttt{RefSet}}$",
                      #  'fontsize': 14, 'color': 'black',
                      #  'horizontalalignment': 'center'}
                      ],
            "plot_zero": True,
            "legend": {"fancybox": False, "loc": 'lower right', "columnspacing": 0.4,
                       # "bbox_to_anchor": (0.8, 1.1),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                       "shadow": "False", "ncol": 1, "fontsize": 14,
                       "framealpha": 0., "borderaxespad": 0., "frameon": False},
            "labels": True
        }

        plot_dic = {
            "subplots": {"figsize": (6.0, 4.0), "ncols": 1, "nrows": 1, "sharex": True, "sharey": False},
            "subplot_adjust": {"left": 0.10, "bottom": 0.15, "top": 0.98, "right": 0.88, "hspace": 0},
            "dpi": 128, "fontsize": 15, "labelsize": 15,
            "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
            "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",  # r"$M_{\rm disk}$ $[10^{-3}M_{\odot}]$",
            "tick_params": {"labelcolor": 'none', "top": False, "bottom": False, "left": False, "right": False},
            # "savepdf": True,
            "figname": __outplotdir__ + "residuals_{}_mej_{}.png".format(dset_name, note),
            "commonaxislabel": True,
            "subplots_adjust": {"hspace": 0, "wspace": 0},
            # "figlegend":{"loc" : 'lower center', "bbox_to_anchor":(0.5, 0.5),
            #              "ncol":3, "labelspacing":0.}
            "add_error_bar": {
                "median": {'color': 'blue', 'lw': 0.6, 'ls': ':'},
                # "mean":{'color':'gray','lw':0.5,'ls':':'},
                # "width":"1sigma",
                "confinterv": 0.68,
                "fill_between": {"facecolor": 'gray', "alpha": 0.3}

            },
            "tight_layout": False
        }

        plot_subplots_for_fits_colorcoded(plot_dic, subplot_dics, fit_dics, datasets)


    # @staticmethod
    # def task_plot_mej_fits_only_group():
    #     # from model_sets import models_vincent2019 as vi
    #     # from model_sets import models_radice2018 as rd
    #     # from model_sets import groups as md
    #     # from model_sets import models_lehner2016 as lh  # [22] arxive:1603.00501
    #     # from model_sets import models_kiuchi2019 as ki
    #     # from model_sets import models_dietrich2015 as di15  # [21] arxive:1504.01266
    #     # from model_sets import models_dietrich2016 as di16  # [24] arxive:1607.06636
    #     # from model_sets import models_sekiguchi2016 as se16  # [23] arxive:1603.01918 # no Mb
    #     # from model_sets import models_sekiguchi2015 as se15  # [-]  arxive:1502.06660 # no Mb
    #     # from model_sets import models_hotokezaka2013 as hz          # [19] arxive:1212.0905
    #     # from model_sets import models_bauswein2013 as bs            # [20] arxive:1302.6530
    #
    #     from model_sets import combined as md
    #     from make_fit4 import Fit_Data
    #
    #     # -------------------------------------------
    #
    #     datasets = OrderedDict()
    #     datasets['refset'] = {"models": md.group_refset, "data": md, "fit": True, "color": None, "plot_errorbar": True,
    #                           "err": "v_n", "edgecolor": "blue"}
    #     datasets["heatcool"] = {"models": md.group_heatcool, "data": md, "fit": True, "color": None,
    #                             "plot_errorbar": True, "err": md.params.Mej_err, "edgecolor":"blue"}
    #     # datasets["cool"] = {"models": md.group_cool, "data": md, "fit": True, "color": None, "plot_errorbar": True,
    #     #                     "err": md.params.MdiskPP_err}
    #     # datasets["none"] = {"models": md.group_none, "data": md, "fit": True, "color": None, "plot_errorbar": False,
    #     #                     "err": md.params.MdiskPP_err}
    #
    #     for key in datasets.keys():
    #         datasets[key]["x_dic"] = {"v_n": "Mej_tot-geo_fit", "err": None, "deferr": None,
    #                                   "mod": {"mult": [1e3]}}  # {"mult": [1e3]}}#, "mod": {"mult": [1e3]}}
    #         datasets[key]["y_dic"] = {"v_n": "Mej_tot-geo", "err": "ud", "deferr": 0.2,
    #                                   "mod": {"mult": [1e3]}}  # }{"mult": [1e3]}}#,  "mod": {"mult": [1e3]}}
    #         datasets[key]["c_dic"] = {"v_n": "q", "err": None, "deferr": None,
    #                                   "mod": {}}  # }{"mult": [1e3]}}#,  "mod": {"mult": [1e3]}}
    #         datasets[key]["mod_x"] = {}
    #         datasets[key]["facecolor"] = 'none'
    #         datasets[key]["plot_errorbar"] = False
    #         # datasets[key]["edgecolor"] = md.dataset_group_colors[key]
    #         datasets[key]["marker"] = md.dataset_group_markers[key]
    #         datasets[key]["label"] = md.dataset_group_labels[key]
    #         datasets[key]["ms"] = 60
    #         datasets[key]["alpha"] = 1.0
    #
    #     o_fit = Fit_Data(md.simulations, "Mej_tot-geo", err_method="default", clean_nans=True)
    #
    #     # coeffs, _, _, _ = o_fit.fit_curve(ff_name="poly22_qLambda", cf_name="poly22", modify="log10")
    #     fit_dics = OrderedDict()
    #     # coeffs, _, _, _ = o_fit.fit_curve(ff_name="krug19", cf_name="krug19", modify="log10")
    #     # fit_dics["Eq.10"] = {  # best 53.2
    #     #     "func": FittingFunctions.mej_kruger19, "coeffs": np.array(coeffs), "to_val": "10**",  # None,
    #     #     "xmin": 0, "xmax": 24., "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
    #     #     "ymin": -100.0, "ymax": 100.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
    #     #     "plot_zero": True
    #     # }
    #     coeffs, _, chi2dof, _ = o_fit.fit_curve(ff_name="poly22_qLambda", cf_name="poly22", modify="log10", usesigma=True)
    #     fit_dics["poly2"] = {  # 2nd best  81.5
    #         "func": FittingFunctions.poly_2_qLambda, "coeffs": np.array(coeffs), "to_val": "10**",  # None,
    #         "xmin": 0, "xmax": 24., "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
    #         "ymin": -100.0, "ymax": 100.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
    #         "plot_zero": True
    #
    #     }
    #     # coeffs, _, _, _ = o_fit.fit_curve(ff_name="diet16", cf_name="diet16", modify="log10")
    #     # fit_dics["Eq.9"] = {  # 153
    #     #     "func": FittingFunctions.mej_dietrich16, "coeffs": np.array(coeffs), "to_val": "10**",  # None,
    #     #     "xmin": 0, "xmax": 24., "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
    #     #     "ymin": -100.0, "ymax": 100.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
    #     #     "plot_zero": True
    #     # }
    #     # coeffs, _, _, _ = o_fit.fit_curve(ff_name="poly2_Lambda", cf_name="poly2", modify="log10")
    #     # fit_dics["poly1"] = {  # 628
    #     #     "func": FittingFunctions.poly_2_Lambda, "coeffs": np.array(coeffs), "to_val": "10**",  # None,
    #     #     "xmin": 0, "xmax": 24., "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
    #     #     "ymin": -100.0, "ymax": 100.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
    #     #     "plot_zero": True
    #     # }
    #
    #     #
    #     # fit_dics = {
    #     #     "Eq.6":
    #     #         {"func": fit_funcs.mej_dietrich16, "coeffs": np.array([-1.234, 3.089, -31.801, 17.526, -3.146]),
    #     #         "xmin": 0, "xmax": 24., "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
    #     #         "ymin": -100.0, "ymax": 100.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
    #     #         "plot_zero": True},
    #     #     "Eq.7":
    #     #         {"func": fit_funcs.mej_kruger20, "coeffs": np.array([-0.981, 12.880, -35.148, 2.030]),
    #     #          "xmin": 0, "xmax": 24., "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
    #     #          "ymin": -100.0, "ymax": 100.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
    #     #          "plot_zero": True},
    #     #     "poly1":
    #     #         {"func": fit_funcs.poly_2_Lambda, "coeffs": np.array([-3.209e+00, 0.032, -2.759e-05]),
    #     #             #np.array([-1.221e-2, 0.014, 8.396e-7]),
    #     #          "xmin": 0, "xmax": 24., "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
    #     #          "ymin": -100.0, "ymax": 100.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
    #     #          "plot_zero": True},
    #     #     "poly2":
    #     #         {"func": fit_funcs.poly_2_qLambda, "coeffs": np.array([2.549, 2.394, -3.005e-02, -3.376e+00, 0.038, -1.149e-05]),
    #     #             #np.array([2.549e-03, 2.394e-03, -3.005e-05, -3.376e-03, 3.826e-05, -1.149e-08]),
    #     #          "xmin": 0, "xmax": 24., "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
    #     #          "ymin": -100.0, "ymax": 100.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
    #     #          "plot_zero": True},
    #     # }
    #
    #     subplot_dics = OrderedDict()
    #     # subplot_dics["Eq.10"] = {
    #     #     "xmin": -0.5, "xmax": 25., "xscale": "linear",
    #     #     "ymin": -12., "ymax": 2.2, "yscale": "linear",
    #     #     # "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
    #     #     # "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
    #     #     "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
    #     #                     "labelright": False, "tick1On": True, "tick2On": True,
    #     #                     "labelsize": 14,
    #     #                     "direction": 'in',
    #     #                     "bottom": True, "top": True, "left": True, "right": True},
    #     #     "text": {'x': 0.85, 'y': 0.90, 's': r"Eq.(10)", 'fontsize': 14, 'color': 'black',
    #     #              'horizontalalignment': 'center'},
    #     #     "plot_zero": True,
    #     #     "labels": True,
    #     # }
    #     subplot_dics["poly2"] = {
    #         "vmin": 1, "vmax": 1.8, "cbar_label": "q", "cmap":"winter",
    #         "xmin": -0.5, "xmax": 20., "xscale": "linear",
    #         "ymin": -8., "ymax": 1.2, "yscale": "linear",
    #         # "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
    #         # "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
    #         "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
    #                         "labelright": False, "tick1On": True, "tick2On": True,
    #                         "labelsize": 14,
    #                         "direction": 'in',
    #                         "bottom": True, "top": True, "left": True, "right": True},
    #         # "text": {'x': 0.85, 'y': 0.90, 's': r"$P_2^2(q,\tilde{\Lambda})$", 'fontsize': 14, 'color': 'black',
    #         #          'horizontalalignment': 'center'},
    #         "texts": [{'x': 0.85, 'y': 0.90, 's': r"$P_2^2(q,\tilde{\Lambda})$", 'fontsize': 14, 'color': 'black',
    #                  'horizontalalignment': 'center'},
    #                   {'x': 0.15, 'y': 0.10, 's':
    #                       r"$\chi_{\nu}^{\texttt{RefSet}}$",
    #                    'fontsize': 14, 'color': 'black',
    #                    'horizontalalignment': 'center'}
    #                   ],
    #         "plot_zero": True,
    #         "legend": {"fancybox": False, "loc": 'lower right', "columnspacing": 0.4,
    #                                   # "bbox_to_anchor": (0.8, 1.1),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
    #                                   "shadow": "False", "ncol": 1, "fontsize": 14,
    #                                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
    #         "labels": True
    #     }
    #     # subplot_dics["Eq.9"] = {
    #     #     "xmin": -0.5, "xmax": 25., "xscale": "linear",
    #     #     "ymin": -12., "ymax": 2.2, "yscale": "linear",
    #     #     # "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
    #     #     # "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
    #     #     "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
    #     #                     "labelright": False, "tick1On": True, "tick2On": True,
    #     #                     "labelsize": 14,
    #     #                     "direction": 'in',
    #     #                     "bottom": True, "top": True, "left": True, "right": True},
    #     #     "text": {'x': 0.85, 'y': 0.90, 's': r"Eq.(9)", 'fontsize': 14, 'color': 'black',
    #     #              'horizontalalignment': 'center'},
    #     #     "plot_zero": True,
    #     #     "labels": True
    #     # }
    #     # subplot_dics["poly1"] = {
    #     #     "xmin": -0.5, "xmax": 25., "xscale": "linear",
    #     #     "ymin": -12., "ymax": 2.2, "yscale": "linear",
    #     #     # "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
    #     #     # "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
    #     #     "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
    #     #                     "labelright": False, "tick1On": True, "tick2On": True,
    #     #                     "labelsize": 14,
    #     #                     "direction": 'in',
    #     #                     "bottom": True, "top": True, "left": True, "right": True},
    #     #     "text": {'x': 0.85, 'y': 0.85, 's': r"$P_2^1(\tilde{\Lambda})$", 'fontsize': 14, 'color': 'black',
    #     #              'horizontalalignment': 'center'},
    #     #     "legend": {"fancybox": False, "loc": 'lower right', "columnspacing": 0.4,
    #     #                # "bbox_to_anchor": (0.8, 1.1),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
    #     #                "shadow": "False", "ncol": 1, "fontsize": 14,
    #     #                "framealpha": 0., "borderaxespad": 0., "frameon": False},
    #     #     "plot_zero": True,
    #     #     "labels": True
    #     # }
    #
    #     # subplot_dics = {
    #     #     "Eq.6":
    #     #         {"xmin": -4, "xmax": 40., "xscale": "linear",
    #     #          "ymin": -11.0, "ymax": 9.0, "yscale": "linear",
    #     #          #"xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
    #     #          #"ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
    #     #          "tick_params": {"axis":'both', "which":'both', "labelleft":True,
    #     #                            "labelright":False, "tick1On":True, "tick2On":True,
    #     #                            "labelsize":14,
    #     #                            "direction":'in',
    #     #                            "bottom":True, "top":True, "left":True, "right":True},
    #     #          "text":{'x':0.85, 'y':0.90, 's':r"Eq.(6)", 'fontsize':14, 'color':'black','horizontalalignment':'center'},
    #     #          "plot_zero": True,
    #     #          "labels": True
    #     #         },
    #     #     "Eq.7":
    #     #         {"xmin": -4, "xmax": 40., "xscale": "linear",
    #     #          "ymin": -11.0, "ymax": 9.0, "yscale": "linear",
    #     #          #"xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
    #     #          #"ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
    #     #          "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
    #     #                          "labelright": False, "tick1On": True, "tick2On": True,
    #     #                          "labelsize": 14,
    #     #                          "direction": 'in',
    #     #                          "bottom": True, "top": True, "left": True, "right": True},
    #     #          "text": {'x': 0.85, 'y': 0.90, 's': r"Eq.(7)", 'fontsize': 14, 'color': 'black', 'horizontalalignment': 'center'},
    #     #          "plot_zero": True,
    #     #          "labels": True,
    #     #          },
    #     #     "poly1":
    #     #         {"xmin": -4, "xmax": 40., "xscale": "linear",
    #     #          "ymin": -11, "ymax": 9.0, "yscale": "linear",
    #     #          #"xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
    #     #          #"ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
    #     #          "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
    #     #                          "labelright": False, "tick1On": True, "tick2On": True,
    #     #                          "labelsize": 14,
    #     #                          "direction": 'in',
    #     #                          "bottom": True, "top": True, "left": True, "right": True},
    #     #          "text": {'x': 0.85, 'y': 0.90, 's': r"Poly1", 'fontsize': 14, 'color': 'black',
    #     #                   'horizontalalignment': 'center'},
    #     #          "plot_zero": True,
    #     #          "legend": {"fancybox": False, "loc": 'lower right', "columnspacing": 0.4,
    #     #                     # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
    #     #                     "shadow": "False", "ncol": 2, "fontsize": 12,
    #     #                     "framealpha": 0., "borderaxespad": 0., "frameon": False},
    #     #          "labels": True
    #     #          },
    #     #     "poly2":
    #     #         {"xmin": -4, "xmax": 40., "xscale": "linear",
    #     #          "ymin": -11.0, "ymax": 9.0, "yscale": "linear",
    #     #          #"xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
    #     #          #"ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
    #     #          "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
    #     #                          "labelright": False, "tick1On": True, "tick2On": True,
    #     #                          "labelsize": 14,
    #     #                          "direction": 'in',
    #     #                          "bottom": True, "top": True, "left": True, "right": True},
    #     #          "text": {'x': 0.85, 'y': 0.90, 's': r"Poly2", 'fontsize': 14, 'color': 'black',
    #     #                   'horizontalalignment': 'center'},
    #     #          "plot_zero": True,
    #     #          "labels": True
    #     #          },
    #     # }
    #
    #     # print(fit_dics.keys(), subplot_dics.keys())
    #
    #     plot_dic = {
    #         "subplots": {"figsize": (6.0, 4.0), "ncols": 1, "nrows": 1, "sharex": True, "sharey": False},
    #         "subplot_adjust": {"left": 0.10, "bottom": 0.15, "top": 0.98, "right": 0.88, "hspace": 0},
    #         "dpi": 128, "fontsize": 15, "labelsize": 15,
    #         "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
    #         "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",  # r"$M_{\rm disk}$ $[10^{-3}M_{\odot}]$",
    #         "tick_params": {"labelcolor": 'none', "top": False, "bottom": False, "left": False, "right": False},
    #         # "savepdf": True,
    #         # "figname": __outplotdir__ + "residuals_sets_mej.png",
    #         "commonaxislabel": True,
    #         "subplots_adjust": {"hspace": 0, "wspace": 0},
    #         # "figlegend":{"loc" : 'lower center', "bbox_to_anchor":(0.5, 0.5),
    #         #              "ncol":3, "labelspacing":0.}
    #         "add_error_bar": {
    #             "median": {'color': 'blue', 'lw': 0.6, 'ls': ':'},
    #             # "mean":{'color':'gray','lw':0.5,'ls':':'},
    #             # "width":"1sigma",
    #             "confinterv": 0.68,
    #             "fill_between": {"facecolor": 'gray', "alpha": 0.3}
    #
    #         },
    #         "tight_layout": False
    #     }
    #
    #
    #
    #     plot_subplots_for_fits_colorcoded(plot_dic, subplot_dics, fit_dics, datasets)

''' --------- MAIN --------- '''

if __name__ == "__main__":

    ''' --- Mej --- '''
    # task_plot_mej_fits_only()
    # task_plot_mej_fits_only_group()
    # task_plot_mej_fits_only_group(usesigma=False, figname="residuals_sets_mej_res.png")

    ''' --- vej --- '''
    # task_plot_vej_fits_only()
    # task_plot_vej_fits_only_group()
    # task_plot_vej_fits_only_group(usesigma=False, figname="residuals_sets_vej_res.png")

    ''' --- ye --- '''
    # task_plot_yeej_fits_only()
    # task_plot_yeej_fits_only_group()
    # task_plot_yeej_fits_only_group(usesigma=False, figname="residuals_sets_yeej_res.png")

    ''' --- theta --- '''
    # task_plot_thetaej_fits_only()
    # task_plot_thetaej_fits_only_group()
    task_plot_thetaej_fits_only_group(usesigma=False, figname="residual_sets_thetaej_res.png")

    ''' --- Mdisk --- '''
    # task_plot_mdisk_fits_only()
    # task_plot_mdisk_fits_only_group()


    ''' test '''
    # Task_Post_Factum.task_plot_mej_fits_only_group(
    #     "refset", xmin=1, xmax=13, vmin=1., vmax=1.8, modify = "log10", to_val = "10**", usesigma=True, note = "log_chi"
    # )
    # Task_Post_Factum.task_plot_mej_fits_only_group(
    #     "heatcool", xmin=1, xmax=13, vmin=1., vmax=1.8, modify = "log10", to_val = "10**", usesigma=True, note = "log_chi"
    # )
    # Task_Post_Factum.task_plot_mej_fits_only_group(
    #     "cool", xmin=0.3, xmax=2, vmin=1., vmax=1.4, modify = "log10", to_val = "10**", usesigma=True, note = "log_chi"
    # )
    # Task_Post_Factum.task_plot_mej_fits_only_group(
    #     "none", xmin=0, xmax=40, vmin=1., vmax=1.4, modify = "log10", to_val = "10**", usesigma=True, note = "log_chi"
    # )
    #
    # Task_Post_Factum.task_plot_mej_fits_only_group(
    #     "refset", xmin=1, xmax=13, vmin=1., vmax=1.8, modify = "log10", to_val = "10**", usesigma=False, note = "log_res"
    # )
    # Task_Post_Factum.task_plot_mej_fits_only_group(
    #     "heatcool", xmin=1, xmax=13, vmin=1., vmax=1.8, modify = "log10", to_val = "10**", usesigma=False, note = "log_res"
    # )
    # Task_Post_Factum.task_plot_mej_fits_only_group(
    #     "cool", xmin=0.3, xmax=2, vmin=1., vmax=1.4, modify = "log10", to_val = "10**", usesigma=False, note = "log_res"
    # )
    # Task_Post_Factum.task_plot_mej_fits_only_group(
    #     "none", xmin=0, xmax=40, vmin=1., vmax=1.4, modify = "log10", to_val = "10**", usesigma=False, note = "log_res"
    # )
