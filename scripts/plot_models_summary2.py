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

''' -------------------------------------------------------------------------- '''

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

def plot_datasets_scatter2(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets):
    #
    if len(fit_dic.keys())>0:
        fig, ax = plt.subplots(nrows=2, figsize=plot_dic["figsize"], sharex="all", gridspec_kw={'height_ratios': [2, 1]})
        fig.subplots_adjust(left=plot_dic["left"], bottom=plot_dic["bottom"], top=plot_dic["top"],
                            right=plot_dic["right"], hspace=plot_dic["hspace"])
        #fig.add_subplot(hspace=0.)
        #
    else:
        ax = []
        fig = plt.figure(figsize=plot_dic["figsize"])  # (figsize=(7.2, 3.6))  # (nrows=1, figsize=[4.5, 3.5])
        ax.append(fig.add_subplot(111))

    in_v_n_x = x_dic["v_n"]
    in_v_n_y = y_dic["v_n"]

    ''' --- main loop ---'''
    sc = None
    for dataset_name in datasets.keys():
        print("processing: {}".format(dataset_name))
        model_dic = datasets[dataset_name]
        marker = model_dic['marker']
        label = model_dic['label']
        ms = model_dic['ms']
        color = model_dic["color"]


        # if "color" in model_dic.keys() and "facecolor" in model_dic.keys() and "edgecolor" in model_dic.keys():
        #     raise NameError("Specify either color or combination of 'facecolor' and 'edgecolor' ")

        # if label == "None":
        #     pass
        # if label != "None" and "facecolor" in model_dic.keys() and "edgecolor" in model_dic.keys():
        #     ax[0].scatter([-100], [-100], marker=marker, s=ms, edgecolor=model_dic["edgecolor"],
        #                   facecolor=model_dic["facecolor"], alpha=1., label=label)
        # if label != "None" and "color" in model_dic.keys():
        #     ax[0].scatter([-100], [-100], marker=marker, s=ms, color=model_dic["color"], alpha=1., label=label)
        #
        # if label == "#EOS" and "color" in model_dic.keys():
        #     if model_dic["color"] == "#EOS" and marker == "#EOS":
        #         for ieos in ourmd.eos_dic_color.keys():
        #             icolor = ourmd.eos_dic_color[ieos]
        #             imarker = ourmd.eos_dic_marker[ieos]
        #             ax[0].scatter([-100], [-100], marker=marker, s=ms, color=icolor, alpha=1., edgecolor=icolor,
        #                       label=ieos, facecolor="none")


        # if label != "None" and "facecolor" in model_dic.keys():
        #     ax[0].scatter([-100], [-100], marker=marker, s=ms, edgecolor=color,
        #                   facecolor=model_dic["facecolor"], alpha=1., label=label)
        # if label != "None" and "color" in model_dic.keys():
        #     ax[0].scatter([-100], [-100], marker=marker, s=ms, edgecolor=color,
        #                   facecolor=model_dic["color"], alpha=1., label=label)
        #
        #
        #
        if label == "None":
            pass
        elif label != "#EOS":
            if color == None:
                if "labelmarkercolor" in model_dic.keys():
                    ax[0].scatter([-100], [-100], marker=marker, s=ms,
                                  color=model_dic["labelmarkercolor"], alpha=1., edgecolor=None,label=label)
                else:
                    ax[0].scatter([-100], [-100], marker=marker, s=ms, color=color, alpha=1., edgecolor=None, label=label)
            else:
                ax[0].scatter([-100], [-100], marker=marker, s=ms, edgecolor=color, facecolor='none', alpha=1., label=label)
        else:
            for ieos in ourmd.eos_dic_color.keys():
                icolor = ourmd.eos_dic_color[ieos]
                imarker = ourmd.eos_dic_marker[ieos]
                if marker != "#EOS":
                    ax[0].scatter([-100], [-100], marker=marker, s=ms, color=icolor, alpha=1., edgecolor=icolor,
                                  label=ieos,
                                  facecolor="none")
                else:
                    ax[0].scatter([-100], [-100], marker=imarker, s=ms, color=icolor, alpha=1., edgecolor=icolor,
                                  label=ieos,
                                  facecolor="none")


        # ---------------------- plot scatter

        d_cl = model_dic["data"]  # md, rd, ki ...
        err = model_dic["err"]  # err lambda(y)
        models = model_dic["models"]  # models DataFrame
        #
        edgecolors = []
        facecolors = []
        markers = [model_dic['marker'] for sim in models.index]
        mss = [model_dic['ms'] for sim in models.index]

        try: eos = models["EOS"]
        except: eos = []

        if "v_n_x" in model_dic:
            # overwrite
            x_dic["v_n"] = model_dic["v_n_x"]
        if "v_n_y" in model_dic:
            # overwrite
            y_dic["v_n"] = model_dic["v_n_y"]

        if x_dic["v_n"] == "Mej_tot-geo_fit" and y_dic["v_n"] == "Mej_tot-geo":
            func, coeffs = fit_dic["func"], fit_dic["coeffs"]
            mej = func(coeffs, models, v_n = d_cl.translation["Mej_tot-geo"]) / 1.e3# 1e-3 Msun -> Msun
            x = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, mej)
            y = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models)
            c = np.array(models[d_cl.translation[col_dic["v_n"]]])
        elif x_dic["v_n"] == "vel_inf_ave-geo_fit" and y_dic["v_n"] == "vel_inf_ave-geo":
            func, coeffs = fit_dic["func"], fit_dic["coeffs"]
            mej = func(coeffs, models)# 1e-3 Msun -> Msun
            x = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, mej)
            y = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models)
            c = np.array(models[d_cl.translation[col_dic["v_n"]]])
        elif x_dic["v_n"] == "Mdisk3D_fit" and y_dic["v_n"] == "Mdisk3D":
            func, coeffs = fit_dic["func"], fit_dic["coeffs"]
            mdisk = func(coeffs, models)# 1e-3 Msun -> Msun
            x = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, mdisk)
            y = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models)
            c = np.array(models[d_cl.translation[col_dic["v_n"]]])
        elif x_dic["v_n"] == "Ye_ave-geo_fit" and y_dic["v_n"] == "Ye_ave-geo":
            func, coeffs = fit_dic["func"], fit_dic["coeffs"]
            mej = func(coeffs, models)# 1e-3 Msun -> Msun
            x = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, mej)
            y = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models)
            c = np.array(models[d_cl.translation[col_dic["v_n"]]])
        else:
            x = d_cl.get_mod_data(x_dic["v_n"], x_dic["mod"], models)
            y = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models)
            c = np.array(models[d_cl.translation[col_dic["v_n"]]])

        if color != "#EOS":
            # default plotting with specified colors and markers
            if 'color' in model_dic.keys() and model_dic['color'] != None:
                c = model_dic['color']
                sc = ax[0].scatter(x, y, s=mss, marker=markers[0],
                              label=None, edgecolor=model_dic["edgecolor"], facecolor=model_dic["facecolor"])
            else:
                cm = plt.cm.get_cmap(plot_dic["cmap"])
                norm = Normalize(vmin=plot_dic["vmin"], vmax=plot_dic["vmax"])
                #sc = mscatter(x, y, ax=ax[0], c=c, norm=norm, s=mss, cmap=cm, ma=markers, label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)
                sc = mscatter(x, y, ax=ax[0], c=c, norm=norm, s=mss, cmap=cm, m=markers, label=None, alpha=plot_dic['alpha'],
                              edgecolor=edgecolors)
        else:
            # use EOS to get marker and color
            assert len(eos) >0
            markers, colors = [], []
            for ieos in eos:
                if marker == "#EOS":
                    markers.append(ourmd.eos_dic_marker[ieos])
                else:
                    markers.append(marker)
                #
                if model_dic["edgecolor"] == "#EOS":
                    edgecolors.append(ourmd.eos_dic_color[ieos])
                else:
                    edgecolors.append(model_dic["edgecolor"])
                #
                if model_dic["facecolor"] == "#EOS":
                    facecolors.append(ourmd.eos_dic_color[ieos])
                else:
                    facecolors.append(model_dic["facecolor"])

            if model_dic["facecolor"] == "none":
                facecolors = "none"

            # print(edgecolors)
            # print(facecolors)
            # exit(1)
            sc = mscatter(x, y, ax=ax[0], s=mss,
                          m=markers, label=None, alpha=plot_dic['alpha'],
                          edgecolor=edgecolors, facecolor=facecolors)

        x = np.array(x)
        y = np.array(y)

        if "annotate" in model_dic.keys():
            if model_dic["annotate"] == "model":
                for _x, _y, model in zip(x, y, models.index):
                    # print(model); exit(1)
                    ax[0].annotate('{}'.format(model),
                                xy=(_x, _y),  # theta, radius
                                #xytext=(0.05, 0.05),  # fraction, fraction
                                #textcoords='figure fraction',
                                #arrowprops=dict(facecolor='black', shrink=0.05),
                                horizontalalignment='left',
                                verticalalignment='bottom')
            elif model_dic["annotate"] == "q":
                for _x, _y, q in zip(x, y, models["q"]):
                    # print(model); exit(1)
                    ax[0].annotate('{}'.format(q),
                                   xy=(_x, _y),  # theta, radius
                                   # xytext=(0.05, 0.05),  # fraction, fraction
                                   # textcoords='figure fraction',
                                   # arrowprops=dict(facecolor='black', shrink=0.05),
                                   horizontalalignment='left',
                                   verticalalignment='bottom')
            else:
                raise NameError("annotate: {} is not recognized".format(plot_dic["annotate"]))
        # --- plot error bar ---
        if "plot_errorbar" in model_dic.keys() and model_dic["plot_errorbar"]:
            #print(model_dic)
            if err == "v_n":
                yerr = models["err-" + y_dic["v_n"]]
                yerr = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, yerr)
            elif err == None:
                yerr = np.zeros(len(y))
            else:
                yerr = err(y)
                # err = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, err)
            # plot_errorbar
            ax[0].errorbar(x, y, yerr=yerr, label=None, color='gray', ecolor='gray', fmt='None', elinewidth=1, capsize=1, alpha=0.5)

        if "plot_xerrorbar" in model_dic.keys() and model_dic["plot_xerrorbar"]:
            if err == "v_n":
                xerr = models["err-" + x_dic["v_n"]]
                xerr = d_cl.get_mod_data(x_dic["v_n"], x_dic["mod"], models, xerr)
            elif err == None:
                xerr = np.zeros(len(x))
            else:
                xerr = err(x)
                # err = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, err)
            # plot_errorbar
            ax[0].errorbar(x, y, xerr=xerr, label=None, color='gray', ecolor='gray', fmt='None', elinewidth=1, capsize=1, alpha=0.5)

        # --- SECOND PANEL FOR FIT! ---
        if len(fit_dic.keys())>0 and model_dic["fit"]:
            # --- --- ---
            func, coeffs = fit_dic["func"], fit_dic["coeffs"]
            fitted_values = func(coeffs, models, v_n = d_cl.translation[y_dic["v_n"]])
            #
            # if x_dic["v_n"] == "Mej_tot-geo_fit" and y_dic["v_n"] == "Mej_tot-geo": fitted_values = fitted_values  # Tims fucking fit
            if y_dic["v_n"] == "Mej_tot-geo": fitted_values = fitted_values / 1.e3  # Tims fucking fit
            #
            y_from_fit = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, fitted_values)
            y_ = (y - y_from_fit) / y
            if model_dic["plot_errorbar"]:
                if err == "v_n":
                    yerr = models["err-" + y_dic["v_n"]]
                    yerr = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, yerr)
                elif err == None:
                    yerr = np.zeros(len(y))
                else:
                    yerr = err(y)
                    # err = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, err)
                delta_y = yerr / y#delta_y = err / y
                print(" (y - y_from_fit) / y : ")
                print(y_)
                ax[1].errorbar(x, y_, yerr=delta_y, label=None, color='gray', ecolor='gray', fmt='None', elinewidth=1, capsize=1, alpha=0.5)
            #
            if color != "#EOS":
                cm = plt.cm.get_cmap(plot_dic["cmap"])
                norm = Normalize(vmin=plot_dic["vmin"], vmax=plot_dic["vmax"])
                sc = mscatter(x, y_, ax=ax[1], c=c, norm=norm, s=mss, cmap=cm, m=markers, label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)
            else:
                assert len(eos) > 0
                markers, colors = [], []
                for ieos in eos:
                    colors.append(ourmd.eos_dic_color[ieos])

                if marker == "#EOS":
                    for ieos in eos:
                        markers.append(ourmd.eos_dic_marker[ieos])
                else:
                    for ieos in eos:
                        markers.append(marker)
                # sc = mscatter(x, y_, ax=ax[1], c=c, norm=norm, s=mss, cmap=cm, m=markers, label=None,
                #               alpha=plot_dic['alpha'], edgecolor=edgecolors)
                #print
                sc = mscatter(x, y_, ax=ax[1], s=mss,
                              m=markers, label=None, alpha=plot_dic['alpha'], edgecolor=colors, facecolors='none')
            #

        if "v_n_x" in model_dic:
            # overwrite back
            x_dic["v_n"] = in_v_n_x
        if "v_n_y" in model_dic:
            # overwrite back
            y_dic["v_n"] = in_v_n_y



    ''' ----- subplot settings --- '''
    for axi, subplotdic in zip(ax, [plot_dic, fit_dic]):
        axi.set_yscale(subplotdic["yscale"])
        axi.set_xscale(subplotdic["xscale"])
        #
        axi.set_xlabel(subplotdic["xlabel"], fontsize=plot_dic["fontsize"])  # , fontsize=11)
        axi.set_ylabel(subplotdic["ylabel"], fontsize=plot_dic["fontsize"])  # , fontsize=11)
        #
        axi.set_xlim(subplotdic["xmin"], subplotdic["xmax"])
        axi.set_ylim(subplotdic["ymin"], subplotdic["ymax"])
        #
        axi.tick_params(axis='both', which='both', labelleft=True,
                       labelright=False, tick1On=True, tick2On=True,
                       labelsize=plot_dic["labelsize"],
                       direction='in',
                       bottom=True, top=True, left=True, right=True)
        axi.minorticks_on()
        #
        if "text" in subplotdic.keys():
            subplotdic["text"]["transform"] = ax.transAxes
            axi.text(**subplotdic["text"])
        #
        if "plot_zero" in subplotdic.keys() and subplotdic["plot_zero"]:
            axi.axhline(y=0, linestyle=':', linewidth=0.4, color='black')
        if "plot_diagonal" in subplotdic.keys() and subplotdic["plot_diagonal"]:
            axi.plot([0, 100], [0, 100], linestyle=':', linewidth=0.4, color='black',label="fit")

    ''' --- additional for a subplot --- '''
    if "patch" in plot_dic.keys() and len(plot_dic["patch"])>0:
        pdics = plot_dic["patch"]
        for pdic in pdics:
            rect = Rectangle((pdic["data"]["xerr"][0],
                              pdic["data"]["yerr"][0]),
                              pdic["data"]["xerr"][1]-pdic["data"]["xerr"][0],
                              pdic["data"]["yerr"][1]-pdic["data"]["yerr"][0], **pdic["plot"])
            # # rect = Rectangle((pdic["data"]["x"] - pdic["data"]["xerr"][0],
            # #                   pdic["data"]["y"] - pdic["data"]["yerr"][0]),
            # #                   pdic["data"]["xerr"][1]-pdic["data"]["xerr"][0],
            # #                   pdic["data"]["yerr"][1]-pdic["data"]["yerr"][0])
            # pc = PatchCollection([rect])
            # ax[0].add_collection(pc)        # rect = Rectangle((pdic["data"]["xerr"][0],
            #                   pdic["data"]["yerr"][0]),
            #                   pdic["data"]["xerr"][1]-pdic["data"]["xerr"][0],
            #                   pdic["data"]["yerr"][1]-pdic["data"]["yerr"][0], **pdic["plot"])
            # # rect = Rectangle((pdic["data"]["x"] - pdic["data"]["xerr"][0],
            # #                   pdic["data"]["y"] - pdic["data"]["yerr"][0]),
            # #                   pdic["data"]["xerr"][1]-pdic["data"]["xerr"][0],
            # #                   pdic["data"]["yerr"][1]-pdic["data"]["yerr"][0])
            # pc = PatchCollection([rect])
            # ax[0].add_collection(pc)
            ax[0].add_patch(rect)

    ''' --- additioanl for label --- '''
    if "add_marker" in plot_dic.keys() and len(plot_dic["add_marker"]) > 0:
        for entry in plot_dic["add_marker"]:
            ax[0].plot(-100., -100., **entry)

    ''' -- additional for poly --- '''
    if "add_poly" in plot_dic.keys():
        for entry in plot_dic["add_poly"]:
            dic_ = dict(entry)
            x_arr = dic_["x"]
            coeffs = dic_["coeffs"]
            if len(coeffs) == 1:
                y_arr = coeffs[0]
            elif len(coeffs) == 2:
                y_arr = coeffs[0] + coeffs[1] * x_arr
            elif len(coeffs) == 3:
                y_arr = coeffs[0] + coeffs[1] * x_arr+ coeffs[2] * x_arr ** 2
            elif len(coeffs) == 4:
                y_arr = coeffs[0] + coeffs[1] * x_arr + coeffs[2] * x_arr ** 2 + coeffs[3] * x_arr ** 3
            else:
                raise ValueError("Too many coeffs")
            del dic_["x"]
            del dic_["coeffs"]
            # print(x_arr)
            # print(y_arr) ; exit(1)
            ax[0].plot(np.array(x_arr).flatten(), np.array(y_arr).flatten(), **dic_)

    #
    #handles, labels = ax[0].get_legend_handles_labels()
    ## sort both labels and handles by labels
    #labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    #ax[0].legend(handles, labels, **plot_dic["legend"])

    # handles, labels = ax[0].get_legend_handles_labels()
    # order = [2,3,4,5,6,7,8,9,10,11,0,1]
    # ax[0].legend([handles[idx] for idx in order], [labels[idx] for idx in order], **plot_dic["legend"])

    if "legend" in plot_dic.keys() and len(plot_dic["legend"].keys()) > 0:
        ax[0].legend(**plot_dic["legend"])
    #
    if "hline" in plot_dic.keys() and len(plot_dic["hline"].keys()) > 0:
        ax[0].axhline(**plot_dic["hline"])
    #
    if 'plot_cbar' in plot_dic.keys() and plot_dic['plot_cbar']:
        if len(fit_dic.keys())>0:
            clb = fig.colorbar(sc, ax=ax.ravel().tolist())
        else:
            clb = fig.colorbar(sc, ax=ax[0])
            plt.tight_layout()
        clb.ax.set_title(plot_dic["cbar_label"], fontsize=plot_dic["fontsize"])#, fontsize=plot_dic["fontsize"])
        clb.ax.tick_params(labelsize=plot_dic["labelsize"])
        #clb.ax.tick_params(plot_dic["labelsize"])
    #


    #
    print("plotted: \n")
    print(plot_dic["figname"])
    if plot_dic["tight_layout"]: plt.tight_layout()
    #
    plt.savefig(plot_dic["figname"], dpi=plot_dic["dpi"])
    if plot_dic["savepdf"]: plt.savefig(plot_dic["figname"].replace(".png", ".pdf"))
    plt.close()
    print(plot_dic["figname"])

def plot_datasets_scatter3(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets):
    #
    if len(fit_dic.keys())>0:
        fig, ax = plt.subplots(nrows=2, figsize=plot_dic["figsize"], sharex="all", gridspec_kw={'height_ratios': [2, 1]})
        fig.subplots_adjust(left=plot_dic["left"], bottom=plot_dic["bottom"], top=plot_dic["top"],
                            right=plot_dic["right"], hspace=plot_dic["hspace"])
        #fig.add_subplot(hspace=0.)
        #
    else:
        ax = []
        fig = plt.figure(figsize=plot_dic["figsize"])  # (figsize=(7.2, 3.6))  # (nrows=1, figsize=[4.5, 3.5])
        ax.append(fig.add_subplot(111))

    in_v_n_x = x_dic["v_n"]
    in_v_n_y = y_dic["v_n"]

    ''' --- main loop ---'''
    sc = None
    for dataset_name in datasets.keys():

        print("processing: {}".format(dataset_name))
        model_dic = datasets[dataset_name]
        d_cl = model_dic["data"]  # md, rd, ki ...
        err = model_dic["err"]  # err lambda(y)
        models = model_dic["models"]  # models DataFrame

        try: eos = models["EOS"]
        except: eos = []

        if "v_n_x" in model_dic:
            # overwrite
            x_dic["v_n"] = model_dic["v_n_x"]
        if "v_n_y" in model_dic:
            # overwrite
            y_dic["v_n"] = model_dic["v_n_y"]

        if x_dic["v_n"] == "Mej_tot-geo_fit" and y_dic["v_n"] == "Mej_tot-geo":
            func, coeffs = fit_dic["func"], fit_dic["coeffs"]
            mej = func(coeffs, models, v_n=d_cl.translation["Mej_tot-geo"]) / 1.e3  # 1e-3 Msun -> Msun
            x = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, mej)
            y = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models)
            c = np.array(models[d_cl.translation[col_dic["v_n"]]])
        elif x_dic["v_n"] == "vel_inf_ave-geo_fit" and y_dic["v_n"] == "vel_inf_ave-geo":
            func, coeffs = fit_dic["func"], fit_dic["coeffs"]
            mej = func(coeffs, models)  # 1e-3 Msun -> Msun
            x = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, mej)
            y = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models)
            c = np.array(models[d_cl.translation[col_dic["v_n"]]])
        elif x_dic["v_n"] == "Mdisk3D_fit" and y_dic["v_n"] == "Mdisk3D":
            func, coeffs = fit_dic["func"], fit_dic["coeffs"]
            mdisk = func(coeffs, models)  # 1e-3 Msun -> Msun
            x = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, mdisk)
            y = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models)
            c = np.array(models[d_cl.translation[col_dic["v_n"]]])
        elif x_dic["v_n"] == "Ye_ave-geo_fit" and y_dic["v_n"] == "Ye_ave-geo":
            func, coeffs = fit_dic["func"], fit_dic["coeffs"]
            mej = func(coeffs, models)  # 1e-3 Msun -> Msun
            x = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, mej)
            y = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models)
            c = np.array(models[d_cl.translation[col_dic["v_n"]]])
        else:
            x = d_cl.get_mod_data(x_dic["v_n"], x_dic["mod"], models)
            y = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models)
            c = np.array(models[d_cl.translation[col_dic["v_n"]]])

        # colors
        if plot_dic["plot_cbar"]:
            if not 'edgecolor' in model_dic.keys():
                raise NameError("Note: 'edgecolor' is required for colorcoded plot. Set 'edgecolor' for : {}"
                                .format(dataset_name))
            colors = []
            edgecolors = [model_dic['edgecolor'] for sim in models.index]
            facecolors = []
        else:
            if "color" in model_dic.keys():
                colors = model_dic["color"]
                edgecolors = []
                facecolors = []
            elif "edgecolor" in model_dic.keys() and "facecolor" in model_dic.keys():
                colors = []
                edgecolors = [model_dic['edgecolor'] for sim in models.index]
                facecolors = [model_dic['facecolor'] for sim in models.index]
            else:
                raise NameError("specifiy either 'color' or a combo 'facecolor' 'edgecolor' in model dic {}"
                                .format(dataset_name))

        # markers
        if model_dic["marker"] == "none":
            markers = []
        elif model_dic["marker"] == "#EOS":
            assert len(eos) > 0
            markers = []
            for ieos in eos:
                imarker = ourmd.eos_dic_marker[ieos]
                markers.append(imarker)
        else:
            markers = [model_dic['marker'] for sim in models.index]

        # markersize
        mss = [model_dic['ms'] for sim in models.index]

        # PLOT SCATTER
        if plot_dic["plot_cbar"]:
            if len(markers) == 0: raise NameError("No 'maker' set for scatter colorcoded plot. Set markers for : {}".format(dataset_name))
            if len(edgecolors) == 0: raise NameError("No 'edgecolor' set for colorcoded plot. Set edgecolor : {}".format(dataset_name))
            cm = plt.cm.get_cmap(plot_dic["cmap"])
            norm = Normalize(vmin=plot_dic["vmin"], vmax=plot_dic["vmax"])
            # sc = mscatter(x, y, ax=ax[0], c=c, norm=norm, s=mss, cmap=cm, ma=markers, label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)
            sc = mscatter(x, y, ax=ax[0], c=c, norm=norm, s=mss, cmap=cm, m=markers, label=None,
                          alpha=plot_dic['alpha'], edgecolor=edgecolors)
        else:
            if len(markers) == 0: raise NameError("No 'maker' set for scatter plot. Set markers for : {}".format(dataset_name))
            if len(colors) > 0:
                sc = ax[0].scatter(x, y, c=colors, s=mss, marker=markers[0],  label=None)
            elif len(colors) ==0 and len(edgecolors) != 0 and len(facecolors) != 0:
                sc = ax[0].scatter(x, y, s=mss, marker=markers[0], label=None, edgecolor=model_dic["edgecolor"], facecolor=model_dic["facecolor"])
            else:
                raise NameError("plotting is not understood. No colors set for non colocoded plot ")

        # data combine
        x = np.array(x)
        y = np.array(y)

        # annotate
        if "annotate" in model_dic.keys():
            if model_dic["annotate"] == "model":
                for _x, _y, model in zip(x, y, models.index):
                    # print(model); exit(1)
                    ax[0].annotate('{}'.format(model),
                                xy=(_x, _y),  # theta, radius
                                #xytext=(0.05, 0.05),  # fraction, fraction
                                #textcoords='figure fraction',
                                #arrowprops=dict(facecolor='black', shrink=0.05),
                                horizontalalignment='left',
                                verticalalignment='bottom')
            elif model_dic["annotate"] == "q":
                for _x, _y, q in zip(x, y, models.q):
                    # print(model); exit(1)
                    ax[0].annotate('{}'.format(q),
                                   xy=(_x, _y),  # theta, radius
                                   # xytext=(0.05, 0.05),  # fraction, fraction
                                   # textcoords='figure fraction',
                                   # arrowprops=dict(facecolor='black', shrink=0.05),
                                   horizontalalignment='left',
                                   verticalalignment='bottom')

        # ERRROR BARS
        if "plot_errorbar" in model_dic.keys() and model_dic["plot_errorbar"]:
            #print(model_dic)
            if err == "v_n":
                yerr = models["err-" + y_dic["v_n"]]
                yerr = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, yerr)
            elif err == None:
                yerr = np.zeros(len(y))
            else:
                yerr = err(y)
                # err = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, err)
            # plot_errorbar
            ax[0].errorbar(x, y, yerr=yerr, label=None, color='gray', ecolor='gray', fmt='None', elinewidth=1, capsize=1, alpha=0.5)

        if "plot_xerrorbar" in model_dic.keys() and model_dic["plot_xerrorbar"]:
            if err == "v_n":
                xerr = models["err-" + x_dic["v_n"]]
                xerr = d_cl.get_mod_data(x_dic["v_n"], x_dic["mod"], models, xerr)
            elif err == None:
                xerr = np.zeros(len(x))
            else:
                xerr = err(x)
                # err = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, err)
            # plot_errorbar
            ax[0].errorbar(x, y, xerr=xerr, label=None, color='gray', ecolor='gray', fmt='None', elinewidth=1, capsize=1, alpha=0.5)

        # # # SECOND PANEL

        # --- SECOND PANEL FOR FIT! ---
        if len(fit_dic.keys())>0 and model_dic["fit"]:
            # --- --- ---
            func, coeffs = fit_dic["func"], fit_dic["coeffs"]
            fitted_values = func(coeffs, models, v_n = d_cl.translation[y_dic["v_n"]])
            #
            # if x_dic["v_n"] == "Mej_tot-geo_fit" and y_dic["v_n"] == "Mej_tot-geo": fitted_values = fitted_values  # Tims fucking fit
            if y_dic["v_n"] == "Mej_tot-geo": fitted_values = fitted_values / 1.e3  # Tims fucking fit
            #
            y_from_fit = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, fitted_values)
            y_ = (y - y_from_fit) / y

            if plot_dic["plot_cbar"]:
                if len(markers) == 0: raise NameError("no markers set for : {}".format(dataset_name))
                if len(edgecolors) == 0: raise NameError("no edgecolors set for : {}".format(dataset_name))
                cm = plt.cm.get_cmap(plot_dic["cmap"])
                norm = Normalize(vmin=plot_dic["vmin"], vmax=plot_dic["vmax"])
                sc = mscatter(x, y_, ax=ax[1], c=c, norm=norm, s=mss, cmap=cm, m=markers, label=None,
                              alpha=plot_dic['alpha'], edgecolor=edgecolors)
            else:
                if len(colors) != 0 and len(edgecolors) == 0 and len(facecolors) ==0:
                    sc = ax[1].scatter(x, y_, c=colors, s=mss, marker=markers[0],  label=None)
                elif len(colors) == 0 and len(edgecolors) != 0 and len(facecolors) != 0:
                    sc = ax[1].scatter(x, y, s=mss, marker=markers[0], label=None, edgecolor=model_dic["edgecolor"],
                                       facecolor=model_dic["facecolor"])
                else:
                    raise NameError("plotting settings are not recognized. Neither colors nor face/edgecolors found: {}"
                                    .format(dataset_name))

            # # # SECOND PLOT errorbars
            if model_dic["plot_errorbar"]:
                if err == "v_n":
                    yerr = models["err-" + y_dic["v_n"]]
                    yerr = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, yerr)
                elif err == None:
                    yerr = np.zeros(len(y))
                else:
                    yerr = err(y)
                    # err = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, err)
                delta_y = yerr / y#delta_y = err / y
                # print(" (y - y_from_fit) / y : ")
                # print(y_)
                ax[1].errorbar(x, y_, yerr=delta_y, label=None, color='gray', ecolor='gray', fmt='None', elinewidth=1, capsize=1, alpha=0.5)

        # In case v_ns were redefined
        if "v_n_x" in model_dic:
            # overwrite back
            x_dic["v_n"] = in_v_n_x
        if "v_n_y" in model_dic:
            # overwrite back
            y_dic["v_n"] = in_v_n_y

    ''' ----- subplot settings --- '''
    for axi, subplotdic in zip(ax, [plot_dic, fit_dic]):
        axi.set_yscale(subplotdic["yscale"])
        axi.set_xscale(subplotdic["xscale"])
        #
        axi.set_xlabel(subplotdic["xlabel"], fontsize=plot_dic["fontsize"])  # , fontsize=11)
        axi.set_ylabel(subplotdic["ylabel"], fontsize=plot_dic["fontsize"])  # , fontsize=11)
        #
        axi.set_xlim(subplotdic["xmin"], subplotdic["xmax"])
        axi.set_ylim(subplotdic["ymin"], subplotdic["ymax"])
        #
        axi.tick_params(axis='both', which='both', labelleft=True,
                       labelright=False, tick1On=True, tick2On=True,
                       labelsize=plot_dic["labelsize"],
                       direction='in',
                       bottom=True, top=True, left=True, right=True)
        axi.minorticks_on()
        #
        if "text" in subplotdic.keys():
            subplotdic["text"]["transform"] = ax.transAxes
            axi.text(**subplotdic["text"])
        #
        if "plot_zero" in subplotdic.keys() and subplotdic["plot_zero"]:
            axi.axhline(y=0, linestyle=':', linewidth=0.4, color='black')
        if "plot_diagonal" in subplotdic.keys() and subplotdic["plot_diagonal"]:
            axi.plot([0, 100], [0, 100], linestyle=':', linewidth=0.4, color='black',label="fit")

    ''' --- additional for a subplot --- '''
    if "patch" in plot_dic.keys() and len(plot_dic["patch"])>0:
        pdics = plot_dic["patch"]
        for pdic in pdics:
            rect = Rectangle((pdic["data"]["xerr"][0],
                              pdic["data"]["yerr"][0]),
                              pdic["data"]["xerr"][1]-pdic["data"]["xerr"][0],
                              pdic["data"]["yerr"][1]-pdic["data"]["yerr"][0], **pdic["plot"])
            # # rect = Rectangle((pdic["data"]["x"] - pdic["data"]["xerr"][0],
            # #                   pdic["data"]["y"] - pdic["data"]["yerr"][0]),
            # #                   pdic["data"]["xerr"][1]-pdic["data"]["xerr"][0],
            # #                   pdic["data"]["yerr"][1]-pdic["data"]["yerr"][0])
            # pc = PatchCollection([rect])
            # ax[0].add_collection(pc)        # rect = Rectangle((pdic["data"]["xerr"][0],
            #                   pdic["data"]["yerr"][0]),
            #                   pdic["data"]["xerr"][1]-pdic["data"]["xerr"][0],
            #                   pdic["data"]["yerr"][1]-pdic["data"]["yerr"][0], **pdic["plot"])
            # # rect = Rectangle((pdic["data"]["x"] - pdic["data"]["xerr"][0],
            # #                   pdic["data"]["y"] - pdic["data"]["yerr"][0]),
            # #                   pdic["data"]["xerr"][1]-pdic["data"]["xerr"][0],
            # #                   pdic["data"]["yerr"][1]-pdic["data"]["yerr"][0])
            # pc = PatchCollection([rect])
            # ax[0].add_collection(pc)
            ax[0].add_patch(rect)

    ''' --- additioanl for label --- '''
    if "add_marker" in plot_dic.keys() and len(plot_dic["add_marker"]) > 0:
        for entry in plot_dic["add_marker"]:
            ax[0].plot(-100., -100., **entry)

    #
    #handles, labels = ax[0].get_legend_handles_labels()
    ## sort both labels and handles by labels
    #labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    #ax[0].legend(handles, labels, **plot_dic["legend"])

    # handles, labels = ax[0].get_legend_handles_labels()
    # order = [2,3,4,5,6,7,8,9,10,11,0,1]
    # ax[0].legend([handles[idx] for idx in order], [labels[idx] for idx in order], **plot_dic["legend"])

    if "legend" in plot_dic.keys() and len(plot_dic["legend"].keys()) > 0:
        ax[0].legend(**plot_dic["legend"])
    #
    if "hline" in plot_dic.keys() and len(plot_dic["hline"].keys()) > 0:
        ax[0].axhline(**plot_dic["hline"])
    #
    if 'plot_cbar' in plot_dic.keys() and plot_dic['plot_cbar']:
        if len(fit_dic.keys())>0:
            clb = fig.colorbar(sc, ax=ax.ravel().tolist())
        else:
            clb = fig.colorbar(sc, ax=ax[0])
            plt.tight_layout()
        clb.ax.set_title(plot_dic["cbar_label"], fontsize=plot_dic["fontsize"])#, fontsize=plot_dic["fontsize"])
        clb.ax.tick_params(labelsize=plot_dic["labelsize"])
        #clb.ax.tick_params(plot_dic["labelsize"])
    #


    #
    print("plotted: \n")
    print(plot_dic["figname"])
    if plot_dic["tight_layout"]: plt.tight_layout()
    #
    plt.savefig(plot_dic["figname"], dpi=plot_dic["dpi"])
    if plot_dic["savepdf"]: plt.savefig(plot_dic["figname"].replace(".png", ".pdf"))
    plt.close()
    print(plot_dic["figname"])

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


''' [[[[ sebastiano's request ]]]] '''
def _seba_task_plot_mej_vs_vej_all():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki                  #
    from model_sets import models_sekiguchi2016 as se16         # [23] arxive:1603.01918 # no Mb
    from model_sets import models_sekiguchi2015 as se15         # [-]  arxive:1502.06660 # no Mb
    from model_sets import models_bauswein2013 as bs            # [20] arxive:1302.6530
    from model_sets import models_lehner2016 as lh              # [22] arxive:1603.00501
    from model_sets import models_hotokezaka2013 as hz          # [19] arxive:1212.0905
    from model_sets import models_dietrich_ujevic2016 as du
    from model_sets import models_dietrich2015 as di15          # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16          # [24] arxive:1607.06636



    # -------------------------------------------
    from collections import OrderedDict
    datasets = OrderedDict() #
    datasets["bauswein"] =  {'marker': 's', 'ms': 60, "color": "slategray", "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "fit": False}
    datasets["hotokezaka"] ={'marker': '>', 'ms': 60, "color": "gray", "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "fit": False}
    datasets["dietrich15"]= {'marker': 'd', "ms": 60, "color": "gray", "models": di15.simulations[di15.mask_for_with_sr], "data": di15, "err": di15.params.Mej_err, "label": r"Dietrich+2015", "fit": False}
    #datasets["sekiguchi15"]={'marker': 'p', "ms": 20, "color": "black", "models": se15.simulations[se15.mask_for_with_sr], "data": se15, "err": se15.params.Mej_err, "label": r"Sekiguchi+2015",  "fit": False}
    datasets["dietrich16"]= {'marker': 'D', "ms": 60, "color": "gray", "models": di16.simulations[di16.mask_for_with_sr], "data": di16, "err": di16.params.Mej_err, "label": r"Dietrich+2016", "fit": False}
    datasets["sekiguchi16"]={'marker': 'h', "ms": 60, "color": "black", "models": se16.simulations[se16.mask_for_with_sr], "data": se16, "err": se16.params.Mej_err, "label": r"Sekiguchi+2016", "fit": False}
    datasets["lehner"] =    {'marker': 'P', 'ms': 60, "color": "orange", "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016",  "fit": False}
    datasets["radice"] =    {'marker': '*', "ms": 60, "color": "green", "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018",  "fit": False}
    #datasets["kiuchi"] =    {'marker': "X", "ms": 20, "color": "gray", "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019",  "fit": False}
    datasets["vincent"] =   {'marker': 'v', 'ms': 60, "color": "red", "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019",  "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets['our'] =       {'marker': 'o', 'ms': 60, "color": "blue", "models": md.groups, "data": md, "err": "v_n", "label": r"Nedora+2020",  "fit": False} # This work
    # datasets['our_total'] = {'marker': '+', 'ms': 60,
    #                          "v_n_x": "vel_inf_ave-tot", "v_n_y":"Mej_tot-tot",
    #                          "color": "blue", "models": md.groups[(md.groups.tend_wind > 0.030) | (md.groups["group"] == "BLh_M10651772_M0_LK")],
    #                          "data": md, "err": "v_n", "label": r"This work", "fit": False}
    # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}

    for t in datasets.keys():
        datasets[t]["label"] = ourmd.datasets_labels[t]
        datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["fill_style"] = "none"
        datasets[t]["plot_errorbar"] = False
        datasets[t]["plot_xerrorbar"] = False
        datasets[t]["facecolor"] = "none"
        datasets[t]["edgecolor"] = datasets[t]["color"]
    datasets["vincent"]["annotate"] = "model"

    y_dic    = {"v_n": "Mej_tot-geo",     "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    x_dic    = {"v_n": "vel_inf_ave-geo", "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (6.0, 5.5),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 16, "labelsize": 16,
        "fit_panel": False,
        "plot_diagonal":False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": False,  # "tab10",
        "xmin": 0.025, "xmax": 0.5, "xscale": "linear",
        "ymin": 1e-4, "ymax": 2e-0, "yscale": "log",
        "ylabel": r"$M_{\rm ej}$ $[M_{\odot}]$",#$[10^{-3}M_{\odot}]$",
        "xlabel": r"$\langle \upsilon_{\rm ej} \rangle$ [c]",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_mej_vej_all_mod.png",
        "legend": {"fancybox": False, "loc": 'upper right',
                   #"bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 12,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        "patch":
            [{"data":{"x":0.27,"y":0.016,"xerr":(0.2,0.3),"yerr":(1e-2,2e-2)},
             "plot":{"facecolor":"blue", "alpha":0.4, "edgecolor":"none", "label":"AT2017gfo [Siegel 2019]"}},
             {"data": {"x": 0.1, "y": 0.05, "xerr": (0.07, 0.14), "yerr": (4e-2, 6e-2)},
              "plot": {"facecolor": "red", "alpha": 0.4, "edgecolor": "none", "label": "AT2017gfo [Siegel 2019]"}},
             # {"data": {"x": None, "y": None, "xerr": (0.2, 0.23), "yerr": (1e-2, 5e-3)},
             #  "plot": {"facecolor": "none", "alpha": 0.4, "edgecolor": "red", "label": "Dyn. kN [P]"}},
             # {"data": {"x": None, "y": None, "xerr": (0.066, 0.068), "yerr": (0.001*0.08, 0.2*0.12)},
             #  "plot": {"facecolor": "none", "alpha": 0.4, "edgecolor": "blue", "label": "Wind kN [P]"}},
             # {"data": {"x": None, "y": None, "xerr": (0.027, 0.04), "yerr": (0.4 * 0.08, 0.2*0.1)},
             #  "plot": {"facecolor": "none", "alpha": 0.4, "edgecolor": "purple", "label": "Sec. kN [P]"}},
             ],
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "savepdf":True,
        "tight_layout":True
    }

    # ---------------- fit ------------------
    from make_fit import fitting_function_mej, fitting_coeffs_mej_our
    fit_dic = {}
    #     {
    #     "func":fitting_function_mej, "coeffs": fitting_coeffs_mej_our(),
    #     "xmin": 0, "xmax": 7.5, "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
    #     "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
    #     "plot_zero":True
    # }

    # Fit
    from make_fit import fitting_function_mej
    #from make_fit import fitting_coeffs_mej_david, fitting_coeffs_mej_our
    #from make_fit import complex_fic_data_mej_module
    #complex_fic_data_mej_module(datasets, fitting_coeffs_mej_our(), key_for_usable_dataset="fit")


    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    for k in datasets.keys():
        datasets[k]["plot_errorbar"] = False

    plot_datasets_scatter2(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)
def _seba_task_plot_mej_vs_ye_all():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki                  #
    from model_sets import models_sekiguchi2016 as se16         # [23] arxive:1603.01918 # no Mb
    from model_sets import models_sekiguchi2015 as se15         # [-]  arxive:1502.06660 # no Mb
    from model_sets import models_bauswein2013 as bs            # [20] arxive:1302.6530
    from model_sets import models_lehner2016 as lh              # [22] arxive:1603.00501
    from model_sets import models_hotokezaka2013 as hz          # [19] arxive:1212.0905
    from model_sets import models_dietrich_ujevic2016 as du
    from model_sets import models_dietrich2015 as di15          # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16          # [24] arxive:1607.06636

    # -------------------------------------------
    from collections import OrderedDict
    datasets = OrderedDict() #
    #datasets["bauswein"] =  {'marker': 's', 'ms': 20, "color": "slategray", "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "fit": False}
    #datasets["hotokezaka"] ={'marker': '>', 'ms': 20, "color": "gray", "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "fit": False}
    #datasets["dietrich15"]= {'marker': 'd', "ms": 20, "color": "gray", "models": di15.simulations[di15.mask_for_with_sr], "data": di15, "err": di15.params.Mej_err, "label": r"Dietrich+2015", "fit": False}
    datasets["sekiguchi15"]={'marker': 'p', "ms": 60, "color": "black", "models": se15.simulations[se15.mask_for_with_sr], "data": se15, "err": se15.params.Mej_err, "label": r"Sekiguchi+2015",  "fit": False}
    #datasets["dietrich16"]= {'marker': 'D', "ms": 20, "color": "gray", "models": di16.simulations[di16.mask_for_with_sr], "data": di16, "err": di16.params.Mej_err, "label": r"Dietrich+2016", "fit": False}
    datasets["sekiguchi16"]={'marker': 'h', "ms": 60, "color": "black", "models": se16.simulations[se16.mask_for_with_sr], "data": se16, "err": se16.params.Mej_err, "label": r"Sekiguchi+2016", "fit": False}
    #datasets["lehner"] =    {'marker': 'P', 'ms': 20, "color": "orange", "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016",  "fit": False}
    datasets["radice"] =    {'marker': '*', "ms": 60, "color": "green", "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018",  "fit": False}
    #datasets["kiuchi"] =    {'marker': "X", "ms": 20, "color": "gray", "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019",  "fit": False}
    datasets["vincent"] =   {'marker': 'v', 'ms': 60, "color": "red", "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019",  "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets['our'] =       {'marker': 'o', 'ms': 60, "color": "blue", "models": md.groups, "data": md, "err": "v_n", "label": r"This work",  "fit": False}
    # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}

    for t in datasets.keys():
        datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["label"] = ourmd.datasets_labels[t]
        datasets[t]["fill_style"] = "none"
        datasets[t]["plot_errorbar"] = False
        datasets[t]["plot_xerrorbar"] = False
        datasets[t]["facecolor"] = "none"
        datasets[t]["edgecolor"] = datasets[t]["color"]

    y_dic    = {"v_n": "Mej_tot-geo",     "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    x_dic    = {"v_n": "Ye_ave-geo",      "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (6.0, 5.5),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 16, "labelsize": 16,
        "fit_panel": False,
        "plot_diagonal":False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": False,  # "tab10",
        "xmin": 0.01, "xmax": 0.35, "xscale": "linear",
        "ymin": 1e-4, "ymax": 1e-1, "yscale": "log",
        "ylabel": r"$M_{\rm ej}$ $[M_{\odot}]$",
        "xlabel": r"$\langle Y_{e;\:\rm ej} \rangle$",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_mej_yeej_all_mod.png",
        "legend": {"fancybox": False, "loc": 'upper right',
                   #"bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "savepdf":True,
        "tight_layout":True
    }

    # ---------------- fit ------------------
    from make_fit import fitting_function_mej, fitting_coeffs_mej_our
    fit_dic = {}
    #     {
    #     "func":fitting_function_mej, "coeffs": fitting_coeffs_mej_our(),
    #     "xmin": 0, "xmax": 7.5, "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
    #     "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
    #     "plot_zero":True
    # }

    # Fit
    from make_fit import fitting_function_mej
    #from make_fit import fitting_coeffs_mej_david, fitting_coeffs_mej_our
    #from make_fit import complex_fic_data_mej_module
    #complex_fic_data_mej_module(datasets, fitting_coeffs_mej_our(), key_for_usable_dataset="fit")


    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    # for k in datasets.keys():
    #     datasets[k]["plot_errorbar"] = False

    plot_datasets_scatter2(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)
def _seba_task_plot_vej_vs_ye():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki                  #
    from model_sets import models_sekiguchi2016 as se16         # [23] arxive:1603.01918 # no Mb
    from model_sets import models_sekiguchi2015 as se15         # [-]  arxive:1502.06660 # no Mb
    from model_sets import models_bauswein2013 as bs            # [20] arxive:1302.6530
    from model_sets import models_lehner2016 as lh              # [22] arxive:1603.00501
    from model_sets import models_hotokezaka2013 as hz          # [19] arxive:1212.0905
    from model_sets import models_dietrich_ujevic2016 as du
    from model_sets import models_dietrich2015 as di15          # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16          # [24] arxive:1607.06636

    # -------------------------------------------
    from collections import OrderedDict
    datasets = OrderedDict() #
    #datasets["bauswein"] =  {'marker': 's', 'ms': 20, "color": "slategray", "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "fit": False}
    #datasets["hotokezaka"] ={'marker': '>', 'ms': 20, "color": "gray", "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "fit": False}
    #datasets["dietrich15"]= {'marker': 'd', "ms": 20, "color": "gray", "models": di15.simulations[di15.mask_for_with_sr], "data": di15, "err": di15.params.Mej_err, "label": r"Dietrich+2015", "fit": False}
    #datasets["sekiguchi15"]={'marker': 'p', "ms": 20, "color": "black", "models": se15.simulations[se15.mask_for_with_sr], "data": se15, "err": se15.params.Mej_err, "label": r"Sekiguchi+2015",  "fit": False}
    #datasets["dietrich16"]= {'marker': 'D', "ms": 20, "color": "gray", "models": di16.simulations[di16.mask_for_with_sr], "data": di16, "err": di16.params.Mej_err, "label": r"Dietrich+2016", "fit": False}
    datasets["sekiguchi16"]={'marker': 'h', "ms": 60, "color": "black", "models": se16.simulations[se16.mask_for_with_sr], "data": se16, "err": se16.params.Mej_err, "label": r"Sekiguchi+2016", "fit": False}
    #datasets["lehner"] =    {'marker': 'P', 'ms': 20, "color": "orange", "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016",  "fit": False}
    datasets["radice"] =    {'marker': '*', "ms": 60, "color": "green", "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018",  "fit": False}
    #datasets["kiuchi"] =    {'marker': "X", "ms": 20, "color": "gray", "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019",  "fit": False}
    datasets["vincent"] =   {'marker': 'v', 'ms': 60, "color": "red", "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019",  "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets['our'] =       {'marker': 'o', 'ms': 60, "color": "blue", "models": md.groups, "data": md, "err": "v_n", "label": r"This work",  "fit": False}
    # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}

    for t in datasets.keys():
        datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["label"] = ourmd.datasets_labels[t]
        datasets[t]["fill_style"] = "none"
        datasets[t]["fill_style"] = "none"
        datasets[t]["plot_errorbar"] = False
        datasets[t]["plot_xerrorbar"] = False
        datasets[t]["facecolor"] = "none"
        datasets[t]["edgecolor"] = datasets[t]["color"]

    y_dic    = {"v_n": "vel_inf_ave-geo", "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    x_dic    = {"v_n": "Ye_ave-geo",      "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (6.0, 5.5),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 16, "labelsize": 16,
        "fit_panel": False,
        "plot_diagonal":False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": False,  # "tab10",
        "xmin": 0.01, "xmax": 0.35, "xscale": "linear",
        "ymin": 0.1, "ymax": 0.35, "yscale": "linear",
        "ylabel": r"$\langle \upsilon_{\rm ej} \rangle$ [c]",
        "xlabel": r"$\langle Y_{e;\:\rm ej} \rangle$",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_vej_yeej_all_mod.png",
        "legend": {"fancybox": False, "loc": 'upper left',
                   #"bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "savepdf":True,
        "tight_layout":True
    }

    # ---------------- fit ------------------
    from make_fit import fitting_function_mej, fitting_coeffs_mej_our
    fit_dic = {}
    #     {
    #     "func":fitting_function_mej, "coeffs": fitting_coeffs_mej_our(),
    #     "xmin": 0, "xmax": 7.5, "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
    #     "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
    #     "plot_zero":True
    # }

    # Fit
    from make_fit import fitting_function_mej
    #from make_fit import fitting_coeffs_mej_david, fitting_coeffs_mej_our
    #from make_fit import complex_fic_data_mej_module
    #complex_fic_data_mej_module(datasets, fitting_coeffs_mej_our(), key_for_usable_dataset="fit")


    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    for k in datasets.keys():
        datasets[k]["plot_errorbar"] = False

    plot_datasets_scatter2(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)


''' ---------------------------| Tasks |----------------------------------- '''

''' --- mdisk --- '''

def task_plot_mdisk_vs_fit_all():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki
    from model_sets import models_dietrich2015 as di15  # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16  # [24] arxive:1607.06636
    # -------------------------------------------

    datasets = {}
    # datasets["kiuchi"] =    {'marker': "X", "ms": 20, "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019", "color": "gray", "fit": False}
    # datasets["radice"] =    {'marker': '*', "ms": 40, "models": rd.simulations[rd.fiducial], "data":rd, "err":rd.params.MdiskPP_err, "label": r"Radice+2018", "color": "blue", "fit": True}
    # # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": di.simulations[di.mask_for_with_sr], "data":di, "err":di.params.Mej_err, "label":"Dietrich+2016"}
    # # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}
    # # datasets["vincent"] =   {'marker': 'v', 'ms': 20, "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019", "color": "blue", "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    # # datasets["bauswein"] =  {'marker': 's', 'ms': 20, "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "color": "blue", "fit": False}
    # # datasets["lehner"] =    {'marker': 'P', 'ms': 20, "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016", "color": "blue", "fit": False}
    # # datasets["hotokezaka"] ={'marker': '>', 'ms': 20, "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "color": "gray", "fit": False}
    # datasets['our'] =       {'marker': 'o', 'ms': 40, "models": md.groups, "data": md, "err": "v_n", "label": r"This work", "color": "red", "fit": True}
    datasets['our'] = {"models": md.groups, "data": md, "fit": True, "color":None, "plot_errorbar":True, "err": "v_n"}
    datasets["radice"] = {"models": rd.simulations[rd.fiducial], "data": rd, "fit": True,"color":None, "plot_errorbar":True, "err": rd.params.MdiskPP_err}
    datasets["kiuchi"] = {"models": ki.simulations[ki.mask_for_with_tov_data], "data": ki, "fit": True,"color":None, "plot_errorbar":False, "err": ki.params.Mdisk_err}
    datasets["vincent"] = {"models": vi.simulations, "data": vi, "fit": True,"color":None, "plot_errorbar":False, "err": vi.params.Mdisk_err}
    datasets["dietrich16"] = {"models": di16.simulations[di16.mask_for_with_sr], "data": di16,"fit": True,"color":None, "plot_errorbar":False, "err": di16.params.Mdisk_err}
    datasets["dietrich15"] = {"models": di15.simulations[di15.mask_for_with_sr], "data": di15, "fit": True,"color":None, "plot_errorbar":False, "err": di15.params.Mdisk_err}

    for t in datasets.keys():
        datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["label"] = ourmd.datasets_labels[t]
        datasets[t]["ms"] = 40
       # datasets[t]["fill_style"] = "none"

    x_dic    = {"v_n": "Mdisk3D_fit", "err": None, "deferr": None, "mod": {}}
    y_dic    = {"v_n": "Mdisk3D",     "err": "ud", "deferr": 0.2,  "mod": {}} #"mult": [1e3]
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        # "figsize": (4.5, 3.5),
        # "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        # "fit_panel": True,
        "figsize": (5.5, 6.1),
        "left": 0.15, "bottom": 0.10, "top": 0.93, "right": 1.00, "hspace": 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "plot_diagonal":True,
        "tight_layout": False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": True,  # "tab10",
        "xmin": 0.08, "xmax": .2, "xscale": "linear",
        "ymin": 0, "ymax": .5, "yscale": "linear",
        "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
        "ylabel": r"$M_{\rm disk}$ $[10^{-3}M_{\odot}]$",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "disk_mass_fit_all.png",
        "legend": {"fancybox": False, "loc": 'upper left',
                   #"bbox_to_anchor": (0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 13,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "savepdf":True,
    }

    # ---------------- fit ------------------
    #from make_fit import fitting_function_mdisk, fitting_coeffs_mdisk_david_ours, complex_fic_data_mdisk_module

    fit_dic = {
        "func":fit_funcs.mdisk_2_2_poly, "coeffs": fit_coefs.mdisk_2_2_poly(),
        "xmin": 0.05, "xmax": .3, "xscale": "linear", "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
        "plot_zero":True
    }

    # Fit

    #complex_fic_data_mdisk_module(datasets, fitting_coeffs_mdisk_david_ours(), key_for_usable_dataset="fit")
    # exit(1)

    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    # for k in datasets.keys():
    #     datasets[k]["plot_errorbar"] = True


    plot_datasets_scatter2(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)
def task_plot_mdisk_vs_fit():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki
    from model_sets import models_dietrich2015 as di15  # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16  # [24] arxive:1607.06636
    # -------------------------------------------

    datasets = {}
    # datasets["kiuchi"] =    {'marker': "X", "ms": 20, "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019", "color": "gray", "fit": False}
    # datasets["radice"] =    {'marker': '*', "ms": 40, "models": rd.simulations[rd.fiducial], "data":rd, "err":rd.params.MdiskPP_err, "label": r"Radice+2018", "color": "blue", "fit": True}
    # # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": di.simulations[di.mask_for_with_sr], "data":di, "err":di.params.Mej_err, "label":"Dietrich+2016"}
    # # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}
    # # datasets["vincent"] =   {'marker': 'v', 'ms': 20, "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019", "color": "blue", "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    # # datasets["bauswein"] =  {'marker': 's', 'ms': 20, "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "color": "blue", "fit": False}
    # # datasets["lehner"] =    {'marker': 'P', 'ms': 20, "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016", "color": "blue", "fit": False}
    # # datasets["hotokezaka"] ={'marker': '>', 'ms': 20, "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "color": "gray", "fit": False}
    # datasets['our'] =       {'marker': 'o', 'ms': 40, "models": md.groups, "data": md, "err": "v_n", "label": r"This work", "color": "red", "fit": True}
    datasets['our'] = {"models": md.groups, "data": md, "fit": True, "color":None, "plot_errorbar":True, "err": "v_n"}
    datasets["radice"] = {"models": rd.simulations[rd.fiducial], "data": rd, "fit": True,"color":None, "plot_errorbar":True, "err": rd.params.MdiskPP_err}
    # datasets["kiuchi"] = {"models": ki.simulations[ki.mask_for_with_tov_data], "data": ki, "fit": True,"color":None, "plot_errorbar":False, "err": ki.params.Mdisk_err}
    # datasets["vincent"] = {"models": vi.simulations, "data": vi, "fit": True,"color":None, "plot_errorbar":False, "err": vi.params.Mdisk_err}
    # datasets["dietrich16"] = {"models": di16.simulations[di16.mask_for_with_sr], "data": di16,"fit": True,"color":None, "plot_errorbar":False, "err": di16.params.Mdisk_err}
    # datasets["dietrich15"] = {"models": di15.simulations[di15.mask_for_with_sr], "data": di15, "fit": True,"color":None, "plot_errorbar":False, "err": di15.params.Mdisk_err}

    for t in datasets.keys():
        datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["label"] = ourmd.datasets_labels[t]
        datasets[t]["ms"] = 40
       # datasets[t]["fill_style"] = "none"

    x_dic    = {"v_n": "Mdisk3D_fit", "err": None, "deferr": None, "mod": {}}
    y_dic    = {"v_n": "Mdisk3D",     "err": "ud", "deferr": 0.2,  "mod": {}} #"mult": [1e3]
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        # "figsize": (4.5, 3.5),
        # "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        # "fit_panel": True,
        "figsize": (5.5, 6.1),
        "left": 0.15, "bottom": 0.10, "top": 0.93, "right": 1.00, "hspace": 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "plot_diagonal":True,
        "tight_layout": False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": True,  # "tab10",
        "xmin": 0.08, "xmax": .2, "xscale": "linear",
        "ymin": 0, "ymax": .5, "yscale": "linear",
        "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
        "ylabel": r"$M_{\rm disk}$ $[10^{-3}M_{\odot}]$",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "disk_mass_fit.png",
        "legend": {"fancybox": False, "loc": 'upper left',
                   #"bbox_to_anchor": (0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 13,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "savepdf":True,
    }

    # ---------------- fit ------------------
    #from make_fit import fitting_function_mdisk, fitting_coeffs_mdisk_david_ours, complex_fic_data_mdisk_module

    fit_dic = {
        "func":fit_funcs.mdisk_radice18, "coeffs": fit_coefs.mdisk_us_radice18(),
        "xmin": 0.08, "xmax": .2, "xscale": "linear", "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
        "plot_zero":True
    }

    # Fit

    #complex_fic_data_mdisk_module(datasets, fitting_coeffs_mdisk_david_ours(), key_for_usable_dataset="fit")
    # exit(1)

    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    # for k in datasets.keys():
    #     datasets[k]["plot_errorbar"] = True


    plot_datasets_scatter2(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)

def task_plot_mdisk_q_vs_fit_all():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki
    from model_sets import models_dietrich2015 as di15  # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16  # [24] arxive:1607.06636


    # -------------------------------------------

    datasets = {}
    # datasets["kiuchi"] =    {'marker': "X", "ms": 20, "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019", "color": "gray", "fit": False}
    # datasets["radice"] =    {'marker': '*', "ms": 40, "models": rd.simulations[rd.fiducial], "data":rd, "err":rd.params.MdiskPP_err, "label": r"Radice+2018", "color": "blue", "fit": True}
    # # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": di.simulations[di.mask_for_with_sr], "data":di, "err":di.params.Mej_err, "label":"Dietrich+2016"}
    # # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}
    # # datasets["vincent"] =   {'marker': 'v', 'ms': 20, "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019", "color": "blue", "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    # # datasets["bauswein"] =  {'marker': 's', 'ms': 20, "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "color": "blue", "fit": False}
    # # datasets["lehner"] =    {'marker': 'P', 'ms': 20, "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016", "color": "blue", "fit": False}
    # # datasets["hotokezaka"] ={'marker': '>', 'ms': 20, "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "color": "gray", "fit": False}
    # datasets['our'] =       {'marker': 'o', 'ms': 40, "models": md.groups, "data": md, "err": "v_n", "label": r"This work", "color": "red", "fit": True}
    datasets['reference'] = {"models": md.groups, "data": md, "fit": True, "color": None, "plot_errorbar": False, "err": "v_n"}
    datasets["radice"] = {"models": rd.simulations[rd.fiducial], "data": rd, "fit": True, "color": None, "plot_errorbar": False, "err": rd.params.MdiskPP_err}
    datasets["kiuchi"] = {"models": ki.simulations[ki.mask_for_with_tov_data], "data": ki, "fit": True, "color": None,"plot_errorbar": False, "err": ki.params.Mdisk_err}
    datasets["vincent"] = {"models": vi.simulations, "data": vi, "fit": True, "color": None, "plot_errorbar": False, "err": vi.params.Mdisk_err}
    datasets["dietrich16"] = {"models": di16.simulations[di16.mask_for_with_sr], "data": di16, "fit": True,"color": None, "plot_errorbar": False, "err": di16.params.Mdisk_err}
    datasets["dietrich15"] = {"models": di15.simulations[di15.mask_for_with_sr], "data": di15, "fit": True, "color": None, "plot_errorbar": False, "err": di15.params.Mdisk_err}

    for t in datasets.keys():
        datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["label"] = ourmd.datasets_labels[t]
        datasets[t]["labelmarkercolor"] = "gray"
        # datasets[t]["facecolor"] = "gray"
        # datasets[t]["edgecolor"] = "none"
        # datasets[t]["labelcolor"] = "gray"
        datasets[t]["ms"] = 50
    # datasets[t]["fill_style"] = "none"

    x_dic    = {"v_n": "q", "err": None, "deferr": None, "mod": {}}
    y_dic    = {"v_n": "Mdisk3D",     "err": "ud", "deferr": 0.2,  "mod": {}} #"mult": [1e3]
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (5.5, 6.1),
        "left": 0.15, "bottom": 0.10, "top": 0.93, "right": 1.00, "hspace": 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "fit_panel": True,
        "plot_diagonal":False,
        "tight_layout": False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": True,  # "tab10",
        "xmin": 0.95, "xmax": 1.9, "xscale": "linear",
        "ymin": 0, "ymax": .5, "yscale": "linear",
        "xlabel": r"$q$",
        "ylabel": r"$M_{\rm disk}$",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "disk_q_mass_fit_all.png",
        "legend": {"fancybox": False, "loc": 'upper left',
                   #"bbox_to_anchor": (0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 13,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "savepdf":True,
    }

    # ---------------- fit ------------------
    #from make_fit import fitting_function_mdisk, fitting_coeffs_mdisk_david_ours, complex_fic_data_mdisk_module

    fit_dic = {
        "func":fit_funcs.mdisk_2_2_poly, "coeffs": np.array([-8.951e-1,1.195,4.292e-4,-3.991e-1,4.778e-5,-2.266e-7]),
        "xmin": 0.95, "xmax": 1.9, "xscale": "linear", "xlabel": r"$q$",
        "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
        "plot_zero":True
    }

    # Fit

    #complex_fic_data_mdisk_module(datasets, fitting_coeffs_mdisk_david_ours(), key_for_usable_dataset="fit")
    # exit(1)

    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    # for k in datasets.keys():
    #     datasets[k]["plot_errorbar"] = True


    plot_datasets_scatter2(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)
def task_plot_mdisk_q_vs_fit():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki
    from model_sets import models_dietrich2015 as di15  # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16  # [24] arxive:1607.06636


    # -------------------------------------------

    datasets = {}
    # datasets["kiuchi"] =    {'marker': "X", "ms": 20, "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019", "color": "gray", "fit": False}
    # datasets["radice"] =    {'marker': '*', "ms": 40, "models": rd.simulations[rd.fiducial], "data":rd, "err":rd.params.MdiskPP_err, "label": r"Radice+2018", "color": "blue", "fit": True}
    # # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": di.simulations[di.mask_for_with_sr], "data":di, "err":di.params.Mej_err, "label":"Dietrich+2016"}
    # # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}
    # # datasets["vincent"] =   {'marker': 'v', 'ms': 20, "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019", "color": "blue", "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    # # datasets["bauswein"] =  {'marker': 's', 'ms': 20, "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "color": "blue", "fit": False}
    # # datasets["lehner"] =    {'marker': 'P', 'ms': 20, "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016", "color": "blue", "fit": False}
    # # datasets["hotokezaka"] ={'marker': '>', 'ms': 20, "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "color": "gray", "fit": False}
    # datasets['our'] =       {'marker': 'o', 'ms': 40, "models": md.groups, "data": md, "err": "v_n", "label": r"This work", "color": "red", "fit": True}
    datasets['our'] = {"models": md.groups, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": "v_n"}
    datasets["radice"] = {"models": rd.simulations[rd.fiducial], "data": rd, "fit": True, "color": None, "plot_errorbar": True, "err": rd.params.MdiskPP_err}
    # datasets["kiuchi"] = {"models": ki.simulations[ki.mask_for_with_tov_data], "data": ki, "fit": True, "color": None,"plot_errorbar": False, "err": ki.params.Mdisk_err}
    # datasets["vincent"] = {"models": vi.simulations, "data": vi, "fit": True, "color": None, "plot_errorbar": False, "err": vi.params.Mdisk_err}
    # datasets["dietrich16"] = {"models": di16.simulations[di16.mask_for_with_sr], "data": di16, "fit": True,"color": None, "plot_errorbar": False, "err": di16.params.Mdisk_err}
    # datasets["dietrich15"] = {"models": di15.simulations[di15.mask_for_with_sr], "data": di15, "fit": True, "color": None, "plot_errorbar": False, "err": di15.params.Mdisk_err}

    for t in datasets.keys():
        datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["label"] = ourmd.datasets_labels[t]
        datasets[t]["ms"] = 40
    # datasets[t]["fill_style"] = "none"

    x_dic    = {"v_n": "q", "err": None, "deferr": None, "mod": {}}
    y_dic    = {"v_n": "Mdisk3D",     "err": "ud", "deferr": 0.2,  "mod": {}} #"mult": [1e3]
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (5.5, 6.1),
        "left": 0.15, "bottom": 0.10, "top": 0.93, "right": 1.00, "hspace": 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "fit_panel": True,
        "plot_diagonal":False,
        "tight_layout": False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": True,  # "tab10",
        "xmin": 0.95, "xmax": 1.9, "xscale": "linear",
        "ymin": 0, "ymax": .5, "yscale": "linear",
        "xlabel": r"$M_1/M_2$",
        "ylabel": r"$M_{\rm ej}$ $[10^{-3}M_{\odot}]$",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "disk_q_mass_fit.png",
        "legend": {"fancybox": False, "loc": 'upper left',
                   #"bbox_to_anchor": (0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 13,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "savepdf":True,
    }

    # ---------------- fit ------------------
    #from make_fit import fitting_function_mdisk, fitting_coeffs_mdisk_david_ours, complex_fic_data_mdisk_module

    fit_dic = {
        "func":fit_funcs.mdisk_radice18, "coeffs": fit_coefs.mdisk_us_radice18(),
        "xmin": 0.95, "xmax": 1.9, "xscale": "linear", "xlabel": r"$M_1/M_2$",
        "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
        "plot_zero":True
    }

    # Fit

    #complex_fic_data_mdisk_module(datasets, fitting_coeffs_mdisk_david_ours(), key_for_usable_dataset="fit")
    # exit(1)

    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    # for k in datasets.keys():
    #     datasets[k]["plot_errorbar"] = True


    plot_datasets_scatter2(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)

def task_plot_mdisk():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki                  #
    from model_sets import models_sekiguchi2016 as se16         # [23] arxive:1603.01918 # no Mb
    from model_sets import models_sekiguchi2015 as se15         # [-]  arxive:1502.06660 # no Mb
    from model_sets import models_bauswein2013 as bs            # [20] arxive:1302.6530
    from model_sets import models_lehner2016 as lh              # [22] arxive:1603.00501
    from model_sets import models_hotokezaka2013 as hz          # [19] arxive:1212.0905
    from model_sets import models_dietrich_ujevic2016 as du
    from model_sets import models_dietrich2015 as di15          # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16          # [24] arxive:1607.06636

    # -------------------------------------------
    from collections import OrderedDict
    datasets = OrderedDict() #
    #datasets["bauswein"] =  {'marker': 's', 'ms': 20, "color": "slategray", "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "fit": False}
    #datasets["hotokezaka"] ={'marker': '>', 'ms': 20, "color": "gray",  "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "fit": False}
    datasets["dietrich15"]= {'marker': 'd', "ms": 60, "color": "gray",  "models": di15.simulations[di15.mask_for_with_sr], "data": di15, "err": di15.params.Mdisk_err, "label": r"Dietrich+2015", "fit": False}
    #datasets["sekiguchi15"]={'marker': 'p', "ms": 20, "color": "black", "models": se15.simulations[se15.mask_for_with_sr], "data": se15, "err": se15.params.Mej_err, "label": r"Sekiguchi+2015",  "fit": False}
    datasets["dietrich16"]= {'marker': 'D', "ms": 60, "color": "gray",  "models": di16.simulations[di16.mask_for_with_sr], "data": di16, "err": di16.params.Mdisk_err, "label": r"Dietrich+2016", "fit": False}
    datasets["sekiguchi16"]={'marker': 'h', "ms": 60, "color": "black", "models": se16.simulations[se16.mask_for_with_sr], "data": se16, "err": se16.params.Mdisk_err, "label": r"Sekiguchi+2016", "fit": False}
    #datasets["lehner"] =    {'marker': 'P', 'ms': 20, "color": "orange", "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016",  "fit": False}
    datasets["radice"] =    {'marker': '*', "ms": 60, "color": "green", "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.MdiskPP_err, "label": r"Radice+2018",  "fit": False}
    datasets["kiuchi"] =    {'marker': "X", "ms": 60, "color": "gray",  "models": ki.simulations, "data": ki, "err": ki.params.Mdisk_err, "label": r"Kiuchi+2019",  "fit": False}
    datasets["vincent"] =   {'marker': 'v', 'ms': 60, "color": "red",   "models": vi.simulations, "data": vi, "err": vi.params.Mdisk_err,"label": r"Vincent+2019",  "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets['our'] =       {'marker': 'o', 'ms': 60, "color": "blue",  "models": md.groups, "data": md, "err": "v_n", "label": r"This work",  "fit": False}
    # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}

    for t in datasets.keys():
        datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["fill_style"] = "none"

    x_dic    = {"v_n": "q",       "err": None, "deferr": None, "mod": {}}
    y_dic    = {"v_n": "Mdisk3D","err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (6.0, 5.5),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "fit_panel": False,
        "plot_diagonal":False,
        "vmin": 350,  "vmax": 900.0, "cmap": "jet", "plot_cbar": False,  # "tab10",
        "xmin": 0.95, "xmax": 1.9,   "xscale": "linear",
        "ymin": 0,    "ymax": 0.4,   "yscale": "linear",
        "xlabel": "$M_1/M_2$",#r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ylabel": r"$M_{\rm disk}$ $M_{\odot}]$",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "disk_q_mdisk.png",
        "legend": {"fancybox": False, "loc": 'upper left',
                   #"bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "savepdf":True,
        "tight_layout":True
    }

    # ---------------- fit ------------------
    from make_fit import fitting_function_mej, fitting_coeffs_mej_our
    fit_dic = {}
    #     {
    #     "func":fitting_function_mej, "coeffs": fitting_coeffs_mej_our(),
    #     "xmin": 0, "xmax": 7.5, "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
    #     "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
    #     "plot_zero":True
    # }

    # Fit
    from make_fit import fitting_function_mej
    #from make_fit import fitting_coeffs_mej_david, fitting_coeffs_mej_our
    #from make_fit import complex_fic_data_mej_module
    #complex_fic_data_mej_module(datasets, fitting_coeffs_mej_our(), key_for_usable_dataset="fit")


    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    for k in datasets.keys():
        datasets[k]["plot_errorbar"] = False

    plot_datasets_scatter2(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)

def task_statistiscs_mdisc():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki
    from model_sets import models_bauswein2013 as bs
    from model_sets import models_lehner2016 as lh
    from model_sets import models_hotokezaka2013 as hz
    from model_sets import models_dietrich_ujevic2016 as du

    datasets = {}
    # datasets["kiuchi"] =    {'marker': "X", "ms": 20, "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019", "color": "gray", "fit": False}
    datasets["radice"] =    {'marker': '*', "ms": 40, "models": rd.simulations[rd.fiducial], "data":rd, "err":rd.params.MdiskPP_err, "label": r"Radice+2018", "color": "blue", "fit": True}
    # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": di.simulations[di.mask_for_with_sr], "data":di, "err":di.params.Mej_err, "label":"Dietrich+2016"}
    # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}
    # datasets["vincent"] =   {'marker': 'v', 'ms': 20, "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019", "color": "blue", "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    # datasets["bauswein"] =  {'marker': 's', 'ms': 20, "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "color": "blue", "fit": False}
    # datasets["lehner"] =    {'marker': 'P', 'ms': 20, "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016", "color": "blue", "fit": False}
    # datasets["hotokezaka"] ={'marker': '>', 'ms': 20, "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "color": "gray", "fit": False}
    datasets['our'] =       {'marker': 'o', 'ms': 40, "models": md.groups, "data": md, "err": "v_n", "label": r"This work", "color": "red", "fit": True}

    from make_fit import fitting_function_mdisk, fitting_coeffs_mdisk_david_ours, complex_fic_data_mdisk_module

    complex_fic_data_mdisk_module(datasets, fitting_coeffs_mdisk_david_ours(), key_for_usable_dataset="fit")

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
    fit_dics["poly2"] = {
        "func": fit_funcs.poly_2_qLambda, "coeffs": np.array([-9.03e-01, 1.20e+00, 4.61e-04, -3.92e-01, 1.46e-05, -2.29e-07]),
        "xmin": 0.05, "xmax": .3, "xscale": "linear", "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
        "plot_zero": True
    }
    fit_dics["Eq.14"] = {
        "func": fit_funcs.mdisk_radice18, "coeffs": np.array([0.080, 0.089, 362.505, 130.788]),
        "xmin": 0.05, "xmax": .3, "xscale": "linear", "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
        "plot_zero": True
    }
    fit_dics["poly1"] = {
        "func": fit_funcs.poly_2_Lambda, "coeffs": np.array([-9.44e-02, 5.98e-04, -3.11e-07]),
        "xmin": 0.05, "xmax": .3, "xscale": "linear", "xlabel": r"$M_{\rm disk;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
        "plot_zero": True
    }
    fit_dics["Eq.15"] = {
        "func": fit_funcs.mdisk_kruger20, "coeffs": np.array([-0.014, 1.001, 1388.224]),
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

''' --- yedisk --- '''

def task_plot_yedisk_cc():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki                  #
    from model_sets import models_sekiguchi2016 as se16         # [23] arxive:1603.01918 # no Mb
    from model_sets import models_sekiguchi2015 as se15         # [-]  arxive:1502.06660 # no Mb
    from model_sets import models_bauswein2013 as bs            # [20] arxive:1302.6530
    from model_sets import models_lehner2016 as lh              # [22] arxive:1603.00501
    from model_sets import models_hotokezaka2013 as hz          # [19] arxive:1212.0905
    from model_sets import models_dietrich_ujevic2016 as du
    from model_sets import models_dietrich2015 as di15          # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16          # [24] arxive:1607.06636

    # -------------------------------------------
    from collections import OrderedDict
    datasets = OrderedDict() #
    # datasets["bauswein"] =  {'marker': 's', 'ms': 60, "color": "slategray", "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "fit": False}
    # datasets["hotokezaka"] ={'marker': '>', 'ms': 60, "color": "gray", "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "fit": False}
    # datasets["dietrich15"]= {'marker': 'd', "ms": 60, "color": "gray", "models": di15.simulations[di15.mask_for_with_sr], "data": di15, "err": di15.params.Mej_err, "label": r"Dietrich+2015", "fit": False}
    # datasets["sekiguchi15"]={'marker': 'p', "ms": 60, "color": "black", "models": se15.simulations[se15.mask_for_with_sr], "data": se15, "err": se15.params.Mej_err, "label": r"Sekiguchi+2015",  "fit": False}
    # datasets["dietrich16"]= {'marker': 'D', "ms": 60, "color": "gray", "models": di16.simulations[di16.mask_for_with_sr], "data": di16, "err": di16.params.Mej_err, "label": r"Dietrich+2016", "fit": False}
    # datasets["sekiguchi16"]={'marker': 'h', "ms": 60, "color": "black", "models": se16.simulations[se16.mask_for_with_sr], "data": se16, "err": se16.params.Mej_err, "label": r"Sekiguchi+2016", "fit": False}
    # datasets["lehner"] =    {'marker': 'P', 'ms': 60, "color": "orange", "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016",  "fit": False}
    # datasets["radice"] =    {'marker': '*', "ms": 60, "color": "green", "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018",  "fit": False}
    # datasets["kiuchi"] =    {'marker': "X", "ms": 60, "color": "gray", "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019",  "fit": False}
    # datasets["vincent"] =   {'marker': 'v', 'ms': 60, "color": "red", "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019",  "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets['our'] =       {'marker': 'o', 'ms': 60, "color": "#EOS", "models": md.groups, "data": md, "err": "v_n", "label": r"#EOS",  "fit": True}
    #### datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}

    for t in datasets.keys():
        #datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["fill_style"] = "full"#"none"
        datasets[t]["edgecolor"] = "none"#"#EOS"
        #datasets[t]["facecolor"] = "#cb"#"none"
        datasets[t]["marker"] = "#EOS"#"d" # #EOS
        datasets[t]["plot_errorbar"] = True
        datasets[t]["plot_xerrorbar"] = False

    x_dic    = {"v_n": "q", "err": None, "deferr": None, "mod": {}}
    y_dic    = {"v_n": "Ye_ave-disk", "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize":  (5.5, 4.1),# (5.5, 6.1),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "fit_panel": False,
        "plot_diagonal":False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": True,  # "tab10",
        "xmin": 0.95, "xmax": 2.0, "xscale": "linear",
        "ymin": 0.01, "ymax": 0.30, "yscale": "linear",
        "xlabel": "$q$",#r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ylabel": r"$\langle Y_{e\: \rm disk} \rangle$",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_q_yedisk_our_poly22_cc.png",
        "legend": {"fancybox": False, "loc": 'upper right',
                   #"bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":1.0,
        "hline": {},#{"y":3.442 * 1.e-3, "color":'blue', "linestyle":"dashed", "label":"Mean"},
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "add_marker": [
            {"ms": 7, "color": "gray", "linestyle": "none", "label": "DD2", "marker": ourmd.eos_dic_marker["DD2"]},
            {"ms": 7, "color": "gray", "linestyle": "none", "label": "LS220", "marker": ourmd.eos_dic_marker["LS220"]},
            {"ms": 7, "color": "gray", "linestyle": "none", "label": "SFHo", "marker": ourmd.eos_dic_marker["SFHo"]},
            {"ms": 7, "color": "gray", "linestyle": "none", "label": "SLy4", "marker": ourmd.eos_dic_marker["SLy4"]},
            {"ms": 7, "color": "gray", "linestyle": "none", "label": "BLh", "marker": ourmd.eos_dic_marker["BLh"]}
        ],
        "savepdf":True,
        "tight_layout":False,
    }

    # ---------------- fit ------------------
    #from make_fit import fitting_function_mej, fitting_coeffs_mej_our
    fit_dic =  {
        # #"func":fit_funcs.yeej_like_vej, "coeffs": np.array([0.268, 0.588, -4.282]),
        # "func": fit_funcs.poly_2_qLambda, "coeffs": np.array([-6.607e-02, 0.318, 6.084e-04, -1.551e-01, -2.055e-04, -2.896e-07]),
        # "xmin": 0.95, "xmax": 2.0, "xscale": "linear", "xlabel": "$q$",#r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        # "ymin": -1.0, "ymax": 1.0, "yscale": "linear", "ylabel": r"$\Delta \langle Y_{e\: \rm ej} \rangle / \langle Y_{e\: \rm ej} \rangle$",
        # "plot_zero":True
    }

    # Fit
    from make_fit import fitting_function_mej
    #from make_fit import fitting_coeffs_mej_david, fitting_coeffs_mej_our
    #from make_fit import complex_fic_data_mej_module
    #complex_fic_data_mej_module(datasets, fitting_coeffs_mej_our(), key_for_usable_dataset="fit")


    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    for k in datasets.keys():
        datasets[k]["plot_errorbar"] = True

    plot_datasets_scatter3(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)
def task_plot_sdisk_cc():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki                  #
    from model_sets import models_sekiguchi2016 as se16         # [23] arxive:1603.01918 # no Mb
    from model_sets import models_sekiguchi2015 as se15         # [-]  arxive:1502.06660 # no Mb
    from model_sets import models_bauswein2013 as bs            # [20] arxive:1302.6530
    from model_sets import models_lehner2016 as lh              # [22] arxive:1603.00501
    from model_sets import models_hotokezaka2013 as hz          # [19] arxive:1212.0905
    from model_sets import models_dietrich_ujevic2016 as du
    from model_sets import models_dietrich2015 as di15          # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16          # [24] arxive:1607.06636

    # -------------------------------------------
    from collections import OrderedDict
    datasets = OrderedDict() #
    # datasets["bauswein"] =  {'marker': 's', 'ms': 60, "color": "slategray", "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "fit": False}
    # datasets["hotokezaka"] ={'marker': '>', 'ms': 60, "color": "gray", "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "fit": False}
    # datasets["dietrich15"]= {'marker': 'd', "ms": 60, "color": "gray", "models": di15.simulations[di15.mask_for_with_sr], "data": di15, "err": di15.params.Mej_err, "label": r"Dietrich+2015", "fit": False}
    # datasets["sekiguchi15"]={'marker': 'p', "ms": 60, "color": "black", "models": se15.simulations[se15.mask_for_with_sr], "data": se15, "err": se15.params.Mej_err, "label": r"Sekiguchi+2015",  "fit": False}
    # datasets["dietrich16"]= {'marker': 'D', "ms": 60, "color": "gray", "models": di16.simulations[di16.mask_for_with_sr], "data": di16, "err": di16.params.Mej_err, "label": r"Dietrich+2016", "fit": False}
    # datasets["sekiguchi16"]={'marker': 'h', "ms": 60, "color": "black", "models": se16.simulations[se16.mask_for_with_sr], "data": se16, "err": se16.params.Mej_err, "label": r"Sekiguchi+2016", "fit": False}
    # datasets["lehner"] =    {'marker': 'P', 'ms': 60, "color": "orange", "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016",  "fit": False}
    # datasets["radice"] =    {'marker': '*', "ms": 60, "color": "green", "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018",  "fit": False}
    # datasets["kiuchi"] =    {'marker': "X", "ms": 60, "color": "gray", "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019",  "fit": False}
    # datasets["vincent"] =   {'marker': 'v', 'ms': 60, "color": "red", "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019",  "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets['our'] =       {'marker': 'o', 'ms': 60, "color": "#EOS", "models": md.groups, "data": md, "err": "v_n", "label": r"#EOS",  "fit": True}
    #### datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}

    for t in datasets.keys():
        #datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["fill_style"] = "full"#"none"
        datasets[t]["edgecolor"] = "none"#"#EOS"
        #datasets[t]["facecolor"] = "#cb"#"none"
        datasets[t]["marker"] = "#EOS"#"d" # #EOS
        datasets[t]["plot_errorbar"] = True
        datasets[t]["plot_xerrorbar"] = False

    x_dic    = {"v_n": "q", "err": None, "deferr": None, "mod": {}}
    y_dic    = {"v_n": "s_ave-disk", "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (5.5, 4.1),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "fit_panel": False,
        "plot_diagonal":False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": True,  # "tab10",
        "xmin": 0.95, "xmax": 2.0, "xscale": "linear",
        "ymin": 1., "ymax": 15.0, "yscale": "linear",
        "xlabel": "$q$",#r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ylabel": r"$\langle s_{\rm disk} \rangle$ [$k_b$/baryon]",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_q_sdisk_our_poly22_cc.png",
        "legend": {"fancybox": False, "loc": 'upper right',
                   #"bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":1.0,
        "hline": {},#{"y":3.442 * 1.e-3, "color":'blue', "linestyle":"dashed", "label":"Mean"},
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "add_marker": [
            {"ms": 7, "color": "gray", "linestyle": "none", "label": "DD2", "marker": ourmd.eos_dic_marker["DD2"]},
            {"ms": 7, "color": "gray", "linestyle": "none", "label": "LS220", "marker": ourmd.eos_dic_marker["LS220"]},
            {"ms": 7, "color": "gray", "linestyle": "none", "label": "SFHo", "marker": ourmd.eos_dic_marker["SFHo"]},
            {"ms": 7, "color": "gray", "linestyle": "none", "label": "SLy4", "marker": ourmd.eos_dic_marker["SLy4"]},
            {"ms": 7, "color": "gray", "linestyle": "none", "label": "BLh", "marker": ourmd.eos_dic_marker["BLh"]}
        ],
        "savepdf":True,
        "tight_layout":False,
    }

    # ---------------- fit ------------------
    #from make_fit import fitting_function_mej, fitting_coeffs_mej_our
    fit_dic =  {
        # #"func":fit_funcs.yeej_like_vej, "coeffs": np.array([0.268, 0.588, -4.282]),
        # "func": fit_funcs.poly_2_qLambda, "coeffs": np.array([-6.607e-02, 0.318, 6.084e-04, -1.551e-01, -2.055e-04, -2.896e-07]),
        # "xmin": 0.95, "xmax": 2.0, "xscale": "linear", "xlabel": "$q$",#r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        # "ymin": -1.0, "ymax": 1.0, "yscale": "linear", "ylabel": r"$\Delta \langle Y_{e\: \rm ej} \rangle / \langle Y_{e\: \rm ej} \rangle$",
        # "plot_zero":True
    }

    # Fit
    from make_fit import fitting_function_mej
    #from make_fit import fitting_coeffs_mej_david, fitting_coeffs_mej_our
    #from make_fit import complex_fic_data_mej_module
    #complex_fic_data_mej_module(datasets, fitting_coeffs_mej_our(), key_for_usable_dataset="fit")


    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    for k in datasets.keys():
        datasets[k]["plot_errorbar"] = True

    plot_datasets_scatter3(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)
def task_plot_tdisk_cc():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki                  #
    from model_sets import models_sekiguchi2016 as se16         # [23] arxive:1603.01918 # no Mb
    from model_sets import models_sekiguchi2015 as se15         # [-]  arxive:1502.06660 # no Mb
    from model_sets import models_bauswein2013 as bs            # [20] arxive:1302.6530
    from model_sets import models_lehner2016 as lh              # [22] arxive:1603.00501
    from model_sets import models_hotokezaka2013 as hz          # [19] arxive:1212.0905
    from model_sets import models_dietrich_ujevic2016 as du
    from model_sets import models_dietrich2015 as di15          # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16          # [24] arxive:1607.06636

    # -------------------------------------------
    from collections import OrderedDict
    datasets = OrderedDict() #
    # datasets["bauswein"] =  {'marker': 's', 'ms': 60, "color": "slategray", "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "fit": False}
    # datasets["hotokezaka"] ={'marker': '>', 'ms': 60, "color": "gray", "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "fit": False}
    # datasets["dietrich15"]= {'marker': 'd', "ms": 60, "color": "gray", "models": di15.simulations[di15.mask_for_with_sr], "data": di15, "err": di15.params.Mej_err, "label": r"Dietrich+2015", "fit": False}
    # datasets["sekiguchi15"]={'marker': 'p', "ms": 60, "color": "black", "models": se15.simulations[se15.mask_for_with_sr], "data": se15, "err": se15.params.Mej_err, "label": r"Sekiguchi+2015",  "fit": False}
    # datasets["dietrich16"]= {'marker': 'D', "ms": 60, "color": "gray", "models": di16.simulations[di16.mask_for_with_sr], "data": di16, "err": di16.params.Mej_err, "label": r"Dietrich+2016", "fit": False}
    # datasets["sekiguchi16"]={'marker': 'h', "ms": 60, "color": "black", "models": se16.simulations[se16.mask_for_with_sr], "data": se16, "err": se16.params.Mej_err, "label": r"Sekiguchi+2016", "fit": False}
    # datasets["lehner"] =    {'marker': 'P', 'ms': 60, "color": "orange", "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016",  "fit": False}
    # datasets["radice"] =    {'marker': '*', "ms": 60, "color": "green", "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018",  "fit": False}
    # datasets["kiuchi"] =    {'marker': "X", "ms": 60, "color": "gray", "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019",  "fit": False}
    # datasets["vincent"] =   {'marker': 'v', 'ms': 60, "color": "red", "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019",  "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets['our'] =       {'marker': 'o', 'ms': 60, "color": "#EOS", "models": md.groups, "data": md, "err": "v_n", "label": r"#EOS",  "fit": True}
    #### datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}

    for t in datasets.keys():
        #datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["fill_style"] = "full"#"none"
        datasets[t]["edgecolor"] = "none"#"#EOS"
        #datasets[t]["facecolor"] = "#cb"#"none"
        datasets[t]["marker"] = "#EOS"#"d" # #EOS
        datasets[t]["plot_errorbar"] = True
        datasets[t]["plot_xerrorbar"] = False

    x_dic    = {"v_n": "q", "err": None, "deferr": None, "mod": {}}
    y_dic    = {"v_n": "T_ave-disk", "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (5.5, 4.1),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "fit_panel": False,
        "plot_diagonal":False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": True,  # "tab10",
        "xmin": 0.95, "xmax": 2.0, "xscale": "linear",
        "ymin": 0, "ymax": 1, "yscale": "linear",
        "xlabel": "$q$",#r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ylabel": r"$\langle T_{\rm disk} \rangle$ [MeV]",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_q_tdisk_our_poly22_cc.png",
        "legend": {"fancybox": False, "loc": 'upper right',
                   #"bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":1.0,
        "hline": {},#{"y":3.442 * 1.e-3, "color":'blue', "linestyle":"dashed", "label":"Mean"},
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "add_marker": [
            {"ms": 7, "color": "gray", "linestyle": "none", "label": "DD2", "marker": ourmd.eos_dic_marker["DD2"]},
            {"ms": 7, "color": "gray", "linestyle": "none", "label": "LS220", "marker": ourmd.eos_dic_marker["LS220"]},
            {"ms": 7, "color": "gray", "linestyle": "none", "label": "SFHo", "marker": ourmd.eos_dic_marker["SFHo"]},
            {"ms": 7, "color": "gray", "linestyle": "none", "label": "SLy4", "marker": ourmd.eos_dic_marker["SLy4"]},
            {"ms": 7, "color": "gray", "linestyle": "none", "label": "BLh", "marker": ourmd.eos_dic_marker["BLh"]}
        ],
        "savepdf":True,
        "tight_layout":False,
    }

    # ---------------- fit ------------------
    #from make_fit import fitting_function_mej, fitting_coeffs_mej_our
    fit_dic =  {
        # #"func":fit_funcs.yeej_like_vej, "coeffs": np.array([0.268, 0.588, -4.282]),
        # "func": fit_funcs.poly_2_qLambda, "coeffs": np.array([-6.607e-02, 0.318, 6.084e-04, -1.551e-01, -2.055e-04, -2.896e-07]),
        # "xmin": 0.95, "xmax": 2.0, "xscale": "linear", "xlabel": "$q$",#r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        # "ymin": -1.0, "ymax": 1.0, "yscale": "linear", "ylabel": r"$\Delta \langle Y_{e\: \rm ej} \rangle / \langle Y_{e\: \rm ej} \rangle$",
        # "plot_zero":True
    }

    # Fit
    from make_fit import fitting_function_mej
    #from make_fit import fitting_coeffs_mej_david, fitting_coeffs_mej_our
    #from make_fit import complex_fic_data_mej_module
    #complex_fic_data_mej_module(datasets, fitting_coeffs_mej_our(), key_for_usable_dataset="fit")


    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    for k in datasets.keys():
        datasets[k]["plot_errorbar"] = True

    plot_datasets_scatter3(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)
def task_plot_thetadisk_cc():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki                  #
    from model_sets import models_sekiguchi2016 as se16         # [23] arxive:1603.01918 # no Mb
    from model_sets import models_sekiguchi2015 as se15         # [-]  arxive:1502.06660 # no Mb
    from model_sets import models_bauswein2013 as bs            # [20] arxive:1302.6530
    from model_sets import models_lehner2016 as lh              # [22] arxive:1603.00501
    from model_sets import models_hotokezaka2013 as hz          # [19] arxive:1212.0905
    from model_sets import models_dietrich_ujevic2016 as du
    from model_sets import models_dietrich2015 as di15          # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16          # [24] arxive:1607.06636

    # -------------------------------------------
    from collections import OrderedDict
    datasets = OrderedDict() #
    # datasets["bauswein"] =  {'marker': 's', 'ms': 60, "color": "slategray", "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "fit": False}
    # datasets["hotokezaka"] ={'marker': '>', 'ms': 60, "color": "gray", "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "fit": False}
    # datasets["dietrich15"]= {'marker': 'd', "ms": 60, "color": "gray", "models": di15.simulations[di15.mask_for_with_sr], "data": di15, "err": di15.params.Mej_err, "label": r"Dietrich+2015", "fit": False}
    # datasets["sekiguchi15"]={'marker': 'p', "ms": 60, "color": "black", "models": se15.simulations[se15.mask_for_with_sr], "data": se15, "err": se15.params.Mej_err, "label": r"Sekiguchi+2015",  "fit": False}
    # datasets["dietrich16"]= {'marker': 'D', "ms": 60, "color": "gray", "models": di16.simulations[di16.mask_for_with_sr], "data": di16, "err": di16.params.Mej_err, "label": r"Dietrich+2016", "fit": False}
    # datasets["sekiguchi16"]={'marker': 'h', "ms": 60, "color": "black", "models": se16.simulations[se16.mask_for_with_sr], "data": se16, "err": se16.params.Mej_err, "label": r"Sekiguchi+2016", "fit": False}
    # datasets["lehner"] =    {'marker': 'P', 'ms': 60, "color": "orange", "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016",  "fit": False}
    # datasets["radice"] =    {'marker': '*', "ms": 60, "color": "green", "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018",  "fit": False}
    # datasets["kiuchi"] =    {'marker': "X", "ms": 60, "color": "gray", "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019",  "fit": False}
    # datasets["vincent"] =   {'marker': 'v', 'ms': 60, "color": "red", "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019",  "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets['our'] =       {'marker': 'o', 'ms': 60, "color": "#EOS", "models": md.groups, "data": md, "err": "v_n", "label": r"#EOS",  "fit": True}
    #### datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}

    for t in datasets.keys():
        #datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["fill_style"] = "full"#"none"
        datasets[t]["edgecolor"] = "none"#"#EOS"
        #datasets[t]["facecolor"] = "#cb"#"none"
        datasets[t]["marker"] = "#EOS"#"d" # #EOS
        datasets[t]["plot_errorbar"] = True
        datasets[t]["plot_xerrorbar"] = False

    x_dic    = {"v_n": "q", "err": None, "deferr": None, "mod": {}}
    y_dic    = {"v_n": "theta_rms-disk", "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (5.5, 4.1),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "fit_panel": False,
        "plot_diagonal":False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": True,  # "tab10",
        "xmin": 0.95, "xmax": 2.0, "xscale": "linear",
        "ymin": 0, "ymax": 90.0, "yscale": "linear",
        "xlabel": "$q$",#r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ylabel": r"$\langle \theta_{RMS\:\rm disk} \rangle$",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_q_thetadisk_our_poly22_cc.png",
        "legend": {"fancybox": False, "loc": 'upper right',
                   #"bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":1.0,
        "hline": {},#{"y":3.442 * 1.e-3, "color":'blue', "linestyle":"dashed", "label":"Mean"},
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "add_marker": [
            {"ms": 7, "color": "gray", "linestyle": "none", "label": "DD2", "marker": ourmd.eos_dic_marker["DD2"]},
            {"ms": 7, "color": "gray", "linestyle": "none", "label": "LS220", "marker": ourmd.eos_dic_marker["LS220"]},
            {"ms": 7, "color": "gray", "linestyle": "none", "label": "SFHo", "marker": ourmd.eos_dic_marker["SFHo"]},
            {"ms": 7, "color": "gray", "linestyle": "none", "label": "SLy4", "marker": ourmd.eos_dic_marker["SLy4"]},
            {"ms": 7, "color": "gray", "linestyle": "none", "label": "BLh", "marker": ourmd.eos_dic_marker["BLh"]}
        ],
        "savepdf":True,
        "tight_layout":False,
    }

    # ---------------- fit ------------------
    #from make_fit import fitting_function_mej, fitting_coeffs_mej_our
    fit_dic =  {
        # #"func":fit_funcs.yeej_like_vej, "coeffs": np.array([0.268, 0.588, -4.282]),
        # "func": fit_funcs.poly_2_qLambda, "coeffs": np.array([-6.607e-02, 0.318, 6.084e-04, -1.551e-01, -2.055e-04, -2.896e-07]),
        # "xmin": 0.95, "xmax": 2.0, "xscale": "linear", "xlabel": "$q$",#r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        # "ymin": -1.0, "ymax": 1.0, "yscale": "linear", "ylabel": r"$\Delta \langle Y_{e\: \rm ej} \rangle / \langle Y_{e\: \rm ej} \rangle$",
        # "plot_zero":True
    }

    # Fit
    from make_fit import fitting_function_mej
    #from make_fit import fitting_coeffs_mej_david, fitting_coeffs_mej_our
    #from make_fit import complex_fic_data_mej_module
    #complex_fic_data_mej_module(datasets, fitting_coeffs_mej_our(), key_for_usable_dataset="fit")


    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    for k in datasets.keys():
        datasets[k]["plot_errorbar"] = True

    plot_datasets_scatter3(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)
""" ========================================================== """

''' --- mej --- '''

def task_plot_mej_vs_fit_all():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki
    from model_sets import models_dietrich2015 as di15  # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16  # [24] arxive:1607.06636

    # -------------------------------------------

    datasets = {}
    # # datasets["kiuchi"] =    {'marker': "X", "ms": 20, "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019", "color": "gray", "fit": False}
    # datasets["radice"] =    {'marker': '*', "ms": 40, "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018", "color": "blue", "fit": True}
    # # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": di.simulations[di.mask_for_with_sr], "data":di, "err":di.params.Mej_err, "label":"Dietrich+2016"}
    # # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}
    # # datasets["vincent"] =   {'marker': 'v', 'ms': 20, "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019", "color": "blue", "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    # # datasets["bauswein"] =  {'marker': 's', 'ms': 20, "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "color": "blue", "fit": False}
    # # datasets["lehner"] =    {'marker': 'P', 'ms': 20, "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016", "color": "blue", "fit": False}
    # # datasets["hotokezaka"] ={'marker': '>', 'ms': 20, "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "color": "gray", "fit": False}
    # datasets['our'] =       {'marker': 'o', 'ms': 40, "models": md.groups, "data": md, "err": "v_n", "label": r"This work", "color": "red", "fit": True}
    datasets['our'] = {"models": md.groups, "data": md, "fit": True, "color":None, "plot_errorbar":True, "err": "v_n"}
    datasets["radice"] = {"models": rd.simulations[rd.fiducial], "data": rd, "fit": True,"color":None, "plot_errorbar":True, "err": rd.params.Mej_err}
    datasets["kiuchi"] = {"models": ki.simulations[ki.mask_for_with_tov_data], "data": ki, "fit": True,"color":None, "plot_errorbar":False, "err": ki.params.Mej_err}
    datasets["vincent"] = {"models": vi.simulations, "data": vi, "fit": True,"color":None, "plot_errorbar":False, "err": vi.params.Mej_err}
    datasets["dietrich16"] = {"models": di16.simulations[di16.mask_for_with_sr], "data": di16,"fit": True,"color":None, "plot_errorbar":False, "err": di16.params.Mej_err}
    datasets["dietrich15"] = {"models": di15.simulations[di15.mask_for_with_sr], "data": di15, "fit": True,"color":None, "plot_errorbar":False, "err": di15.params.Mej_err}

    for t in datasets.keys():
        datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["label"] = ourmd.datasets_labels[t]
        datasets[t]["ms"] = 40
       # datasets[t]["fill_style"] = "none"

    x_dic    = {"v_n": "Mej_tot-geo_fit", "err": None, "deferr": None, "mod": {"mult": [1e3]}}
    y_dic    = {"v_n": "Mej_tot-geo",     "err": "ud", "deferr": 0.2,  "mod": {"mult": [1e3]}}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (5.5, 6.1),
        "left": 0.15, "bottom": 0.10, "top": 0.93, "right": 1.00, "hspace": 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "fit_panel": True,
        "plot_diagonal":True,
        "tight_layout": False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": True,  # "tab10",
        "xmin": 1, "xmax": 24, "xscale": "linear", # 7.5
        "ymin": 0, "ymax": 28, "yscale": "linear",
        "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ylabel": r"$M_{\rm ej}$ $[10^{-3}M_{\odot}]$",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_mass_fit_all_mean.png",
        "legend": {"fancybox": False, "loc": 'upper center',
                   #"bbox_to_anchor": (0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "savepdf":True,
    }

    # ---------------- fit ------------------
    #from make_fit import fitting_function_mej, fitting_coeffs_mej_our
    #fit = Fit_Data(datasets, "Mej_tot-geo")

    fit_dic = {
        "func":fit_funcs.mej_flat_mean, "coeffs": fit_coefs.mej_all_2(), #
        "xmin": 0, "xmax": 24., "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -10.0, "ymax": 2.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
        "plot_zero":True
    }

    # Fit
    #from make_fit import fitting_function_mej
    # from make_fit import fitting_coeffs_mej_david, fitting_coeffs_mej_our
    # from make_fit import complex_fic_data_mej_module
    # complex_fic_data_mej_module(datasets, fitting_coeffs_mej_our(), key_for_usable_dataset="fit")


    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    # for k in datasets.keys():
    #     datasets[k]["plot_errorbar"] = True


    plot_datasets_scatter2(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)
def task_plot_mej_vs_fit():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki
    from model_sets import models_dietrich2015 as di15  # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16  # [24] arxive:1607.06636

    # -------------------------------------------

    datasets = {}
    # # datasets["kiuchi"] =    {'marker': "X", "ms": 20, "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019", "color": "gray", "fit": False}
    # datasets["radice"] =    {'marker': '*', "ms": 40, "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018", "color": "blue", "fit": True}
    # # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": di.simulations[di.mask_for_with_sr], "data":di, "err":di.params.Mej_err, "label":"Dietrich+2016"}
    # # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}
    # # datasets["vincent"] =   {'marker': 'v', 'ms': 20, "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019", "color": "blue", "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    # # datasets["bauswein"] =  {'marker': 's', 'ms': 20, "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "color": "blue", "fit": False}
    # # datasets["lehner"] =    {'marker': 'P', 'ms': 20, "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016", "color": "blue", "fit": False}
    # # datasets["hotokezaka"] ={'marker': '>', 'ms': 20, "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "color": "gray", "fit": False}
    # datasets['our'] =       {'marker': 'o', 'ms': 40, "models": md.groups, "data": md, "err": "v_n", "label": r"This work", "color": "red", "fit": True}
    datasets['our'] = {"models": md.groups, "data": md, "fit": True, "color":None, "plot_errorbar":True, "err": "v_n"}
    #datasets["radice"] = {"models": rd.simulations[rd.fiducial], "data": rd, "fit": True,"color":None, "plot_errorbar":True, "err": rd.params.Mej_err}
    # datasets["kiuchi"] = {"models": ki.simulations[ki.mask_for_with_tov_data], "data": ki, "fit": True,"color":None, "plot_errorbar":False, "err": ki.params.Mej_err}
    # datasets["vincent"] = {"models": vi.simulations, "data": vi, "fit": True,"color":None, "plot_errorbar":False, "err": vi.params.Mej_err}
    # datasets["dietrich16"] = {"models": di16.simulations[di16.mask_for_with_sr], "data": di16,"fit": True,"color":None, "plot_errorbar":False, "err": di16.params.Mej_err}
    # datasets["dietrich15"] = {"models": di15.simulations[di15.mask_for_with_sr], "data": di15, "fit": True,"color":None, "plot_errorbar":False, "err": di15.params.Mej_err}

    for t in datasets.keys():
        datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["label"] = ourmd.datasets_labels[t]
        datasets[t]["ms"] = 40
       # datasets[t]["fill_style"] = "none"

    x_dic    = {"v_n": "Mej_tot-geo_fit", "err": None, "deferr": None, "mod": {"mult": [1e3]}}
    y_dic    = {"v_n": "Mej_tot-geo",     "err": "ud", "deferr": 0.2,  "mod": {"mult": [1e3]}}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (5.5, 6.1),
        "left": 0.15, "bottom": 0.10, "top": 0.93, "right": 1.00, "hspace": 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "fit_panel": True,
        "plot_diagonal":True,
        "tight_layout": False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": True,  # "tab10",
        "xmin": 1, "xmax": 10, "xscale": "linear", # 7.5
        "ymin": 0, "ymax": 15, "yscale": "linear",
        "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ylabel": r"$M_{\rm ej}$ $[10^{-3}M_{\odot}]$",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_mass_fit.png",
        "legend": {"fancybox": False, "loc": 'upper right',
                   #"bbox_to_anchor": (0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 13,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "savepdf":True,
    }

    # ---------------- fit ------------------
    #from make_fit import fitting_function_mej, fitting_coeffs_mej_our
    #fit = Fit_Data(datasets, "Mej_tot-geo")

    fit_dic = {
        "func":fit_funcs.mej_dietrich16, "coeffs": fit_coefs.mej_us_radice18(),
        "xmin": 0, "xmax": 10., "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -10.0, "ymax": 2.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
        "plot_zero":True
    }

    # Fit
    #from make_fit import fitting_function_mej
    # from make_fit import fitting_coeffs_mej_david, fitting_coeffs_mej_our
    # from make_fit import complex_fic_data_mej_module
    # complex_fic_data_mej_module(datasets, fitting_coeffs_mej_our(), key_for_usable_dataset="fit")


    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    # for k in datasets.keys():
    #     datasets[k]["plot_errorbar"] = True


    plot_datasets_scatter2(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)

def task_plot_mej_q_vs_fit():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki
    from model_sets import models_dietrich2015 as di15  # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16  # [24] arxive:1607.06636


    # -------------------------------------------

    datasets = {}
    # datasets["kiuchi"] =    {'marker': "X", "ms": 20, "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019", "color": "gray", "fit": False}
    # datasets["radice"] =    {'marker': '*', "ms": 40, "models": rd.simulations[rd.fiducial], "data":rd, "err":rd.params.MdiskPP_err, "label": r"Radice+2018", "color": "blue", "fit": True}
    # # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": di.simulations[di.mask_for_with_sr], "data":di, "err":di.params.Mej_err, "label":"Dietrich+2016"}
    # # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}
    # # datasets["vincent"] =   {'marker': 'v', 'ms': 20, "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019", "color": "blue", "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    # # datasets["bauswein"] =  {'marker': 's', 'ms': 20, "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "color": "blue", "fit": False}
    # # datasets["lehner"] =    {'marker': 'P', 'ms': 20, "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016", "color": "blue", "fit": False}
    # # datasets["hotokezaka"] ={'marker': '>', 'ms': 20, "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "color": "gray", "fit": False}
    # datasets['our'] =       {'marker': 'o', 'ms': 40, "models": md.groups, "data": md, "err": "v_n", "label": r"This work", "color": "red", "fit": True}
    datasets['our'] = {"models": md.groups, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": "v_n"}
    # datasets["radice"] = {"models": rd.simulations[rd.fiducial], "data": rd, "fit": True, "color": None, "plot_errorbar": True, "err": rd.params.MdiskPP_err}
    # datasets["kiuchi"] = {"models": ki.simulations[ki.mask_for_with_tov_data], "data": ki, "fit": True, "color": None,"plot_errorbar": False, "err": ki.params.Mdisk_err}
    # datasets["vincent"] = {"models": vi.simulations, "data": vi, "fit": True, "color": None, "plot_errorbar": False, "err": vi.params.Mdisk_err}
    # datasets["dietrich16"] = {"models": di16.simulations[di16.mask_for_with_sr], "data": di16, "fit": True,"color": None, "plot_errorbar": False, "err": di16.params.Mdisk_err}
    # datasets["dietrich15"] = {"models": di15.simulations[di15.mask_for_with_sr], "data": di15, "fit": True, "color": None, "plot_errorbar": False, "err": di15.params.Mdisk_err}

    for t in datasets.keys():
        datasets[t]["marker"] = "d"#ourmd.datasets_markers[t]
        datasets[t]["label"] = ourmd.datasets_labels[t]
        datasets[t]["ms"] = 40
    # datasets[t]["fill_style"] = "none"

    x_dic    = {"v_n": "q", "err": None, "deferr": None, "mod": {}}
    y_dic    = {"v_n": "Mej_tot-geo", "err": None, "deferr": None, "mod": {}}#{"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (5.5, 6.1),
        "left": 0.15, "bottom": 0.10, "top": 0.93, "right": 1.00, "hspace": 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "fit_panel": True,
        "plot_diagonal":False,
        "tight_layout": False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": True,  # "tab10",
        "xmin": 1, "xmax": 10.0, "xscale": "linear",
        "ymin": 0, "ymax": .5, "yscale": "linear",
        "xlabel": r"$q$",
        "ylabel": r"$M_{\rm ej;fit}$ $[M_{\odot}]$",#r"$M_{\rm ej}$ $[10^{-3}M_{\odot}]$",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "mej_q_fit_our.png",
        "legend": {"fancybox": False, "loc": 'upper left',
                   #"bbox_to_anchor": (0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 13,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "savepdf":True,
    }

    # ---------------- fit ------------------
    #from make_fit import fitting_function_mdisk, fitting_coeffs_mdisk_david_ours, complex_fic_data_mdisk_module

    fit_dic = {
        "func":fit_funcs.mej_flat_mean, "coeffs": fit_coefs.mej_all(),
        "xmin": 0.95, "xmax": 1.9, "xscale": "linear", "xlabel": r"$M_1/M_2$",
        "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej}/M_{\rm ej}$",#r"$\Delta M_{\rm disk} / M_{\rm disk}$",
        "plot_zero":True
    }

    # Fit

    #complex_fic_data_mdisk_module(datasets, fitting_coeffs_mdisk_david_ours(), key_for_usable_dataset="fit")
    # exit(1)

    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    # for k in datasets.keys():
    #     datasets[k]["plot_errorbar"] = True


    plot_datasets_scatter2(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)

def task_plot_mej_all():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki                  #
    from model_sets import models_sekiguchi2016 as se16         # [23] arxive:1603.01918 # no Mb
    from model_sets import models_sekiguchi2015 as se15         # [-]  arxive:1502.06660 # no Mb
    from model_sets import models_bauswein2013 as bs            # [20] arxive:1302.6530
    from model_sets import models_lehner2016 as lh              # [22] arxive:1603.00501
    from model_sets import models_hotokezaka2013 as hz          # [19] arxive:1212.0905
    from model_sets import models_dietrich_ujevic2016 as du
    from model_sets import models_dietrich2015 as di15          # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16          # [24] arxive:1607.06636

    # -------------------------------------------
    from collections import OrderedDict
    datasets = OrderedDict() #
    datasets["bauswein"] =  {'marker': 's', 'ms': 60, "color": "slategray", "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "fit": False}
    datasets["hotokezaka"] ={'marker': '>', 'ms': 60, "color": "gray", "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "fit": False}
    datasets["dietrich15"]= {'marker': 'd', "ms": 60, "color": "gray", "models": di15.simulations[di15.mask_for_with_sr], "data": di15, "err": di15.params.Mej_err, "label": r"Dietrich+2015", "fit": False}
    datasets["sekiguchi15"]={'marker': 'p', "ms": 60, "color": "black", "models": se15.simulations[se15.mask_for_with_sr], "data": se15, "err": se15.params.Mej_err, "label": r"Sekiguchi+2015",  "fit": False}
    datasets["dietrich16"]= {'marker': 'D', "ms": 60, "color": "gray", "models": di16.simulations[di16.mask_for_with_sr], "data": di16, "err": di16.params.Mej_err, "label": r"Dietrich+2016", "fit": False}
    datasets["sekiguchi16"]={'marker': 'h', "ms": 60, "color": "black", "models": se16.simulations[se16.mask_for_with_sr], "data": se16, "err": se16.params.Mej_err, "label": r"Sekiguchi+2016", "fit": False}
    datasets["lehner"] =    {'marker': 'P', 'ms': 60, "color": "orange", "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016",  "fit": False}
    datasets["radice"] =    {'marker': '*', "ms": 60, "color": "green", "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018",  "fit": False}
    datasets["kiuchi"] =    {'marker': "X", "ms": 60, "color": "gray", "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019",  "fit": False}
    datasets["vincent"] =   {'marker': 'v', 'ms': 60, "color": "red", "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019",  "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets['our'] =       {'marker': 'o', 'ms': 60, "color": "blue", "models": md.groups, "data": md, "err": "v_n", "label": r"This work",  "fit": False}
    #### datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}

    for t in datasets.keys():
        datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["fill_style"] = "none"

    x_dic    = {"v_n": "q", "err": None, "deferr": None, "mod": {}}
    y_dic    = {"v_n": "Mej_tot-geo",     "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (6.0, 5.5),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "fit_panel": False,
        "plot_diagonal":False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": False,  # "tab10",
        "xmin": 0.95, "xmax": 2.4, "xscale": "linear",
        "ymin": 1e-4, "ymax": 1e-1, "yscale": "log",
        "xlabel": "$q$",#r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ylabel": r"$M_{\rm ej}$ $[10^{-3}M_{\odot}]$",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_q_mej.png",
        "legend": {"fancybox": False, "loc": 'lower right',
                   #"bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        "hline": {"y":5.220 * 1.e-3, "color":'blue', "linestyle":"dashed", "label":"Mean"},
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "savepdf":True,
        "tight_layout":True
    }

    # ---------------- fit ------------------
    from make_fit import fitting_function_mej, fitting_coeffs_mej_our
    fit_dic = {}
    #     {
    #     "func":fitting_function_mej, "coeffs": fitting_coeffs_mej_our(),
    #     "xmin": 0, "xmax": 7.5, "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
    #     "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
    #     "plot_zero":True
    # }

    # Fit
    from make_fit import fitting_function_mej
    #from make_fit import fitting_coeffs_mej_david, fitting_coeffs_mej_our
    #from make_fit import complex_fic_data_mej_module
    #complex_fic_data_mej_module(datasets, fitting_coeffs_mej_our(), key_for_usable_dataset="fit")


    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    for k in datasets.keys():
        datasets[k]["plot_errorbar"] = False

    plot_datasets_scatter2(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)
def task_plot_mej():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki                  #
    from model_sets import models_sekiguchi2016 as se16         # [23] arxive:1603.01918 # no Mb
    from model_sets import models_sekiguchi2015 as se15         # [-]  arxive:1502.06660 # no Mb
    from model_sets import models_bauswein2013 as bs            # [20] arxive:1302.6530
    from model_sets import models_lehner2016 as lh              # [22] arxive:1603.00501
    from model_sets import models_hotokezaka2013 as hz          # [19] arxive:1212.0905
    from model_sets import models_dietrich_ujevic2016 as du
    from model_sets import models_dietrich2015 as di15          # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16          # [24] arxive:1607.06636

    # -------------------------------------------
    from collections import OrderedDict
    datasets = OrderedDict() #
    # datasets["bauswein"] =  {'marker': 's', 'ms': 60, "color": "slategray", "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "fit": False}
    # datasets["hotokezaka"] ={'marker': '>', 'ms': 60, "color": "gray", "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "fit": False}
    # datasets["dietrich15"]= {'marker': 'd', "ms": 60, "color": "gray", "models": di15.simulations[di15.mask_for_with_sr], "data": di15, "err": di15.params.Mej_err, "label": r"Dietrich+2015", "fit": False}
    # datasets["sekiguchi15"]={'marker': 'p', "ms": 60, "color": "black", "models": se15.simulations[se15.mask_for_with_sr], "data": se15, "err": se15.params.Mej_err, "label": r"Sekiguchi+2015",  "fit": False}
    # datasets["dietrich16"]= {'marker': 'D', "ms": 60, "color": "gray", "models": di16.simulations[di16.mask_for_with_sr], "data": di16, "err": di16.params.Mej_err, "label": r"Dietrich+2016", "fit": False}
    # datasets["sekiguchi16"]={'marker': 'h', "ms": 60, "color": "black", "models": se16.simulations[se16.mask_for_with_sr], "data": se16, "err": se16.params.Mej_err, "label": r"Sekiguchi+2016", "fit": False}
    # datasets["lehner"] =    {'marker': 'P', 'ms': 60, "color": "orange", "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016",  "fit": False}
    # datasets["radice"] =    {'marker': '*', "ms": 60, "color": "green", "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018",  "fit": False}
    # datasets["kiuchi"] =    {'marker': "X", "ms": 60, "color": "gray", "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019",  "fit": False}
    # datasets["vincent"] =   {'marker': 'v', 'ms': 60, "color": "red", "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019",  "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets['our'] =       {'marker': 'o', 'ms': 60, "color": "#EOS", "models": md.groups, "data": md, "err": "v_n", "label": r"#EOS",  "fit": True}
    #### datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}

    for t in datasets.keys():
        #datasets[t]["marker"] = ourmd.datasets_markers[t]
        #datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["fill_style"] = "none"
        datasets[t]["edgecolor"] = "#EOS"
        datasets[t]["facecolor"] = "none"
        datasets[t]["marker"] = "d" #EOS"
        datasets[t]["plot_errorbar"] = True
        datasets[t]["plot_xerrorbar"] = False

    x_dic    = {"v_n": "q", "err": None, "deferr": None, "mod": {}}
    y_dic    = {"v_n": "Mej_tot-geo",     "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (5.5, 6.1),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "fit_panel": True,
        "plot_diagonal":False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": False,  # "tab10",
        "xmin": 0.95, "xmax": 2.0, "xscale": "linear",
        "ymin": 3e-4, "ymax": 2e-2, "yscale": "log",
        "xlabel": "$q$",#r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ylabel": r"$M_{\rm ej}$ $[M_{\odot}]$",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_q_mej_our_poly22.png",
        "legend": {"fancybox": False, "loc": 'lower right',
                   #"bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":1.0,
        #"hline": {"y":3.442 * 1.e-3, "color":'blue', "linestyle":"dashed", "label":"Mean"},
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        # "add_marker":[
        #     {"ms": 7, "color": "gray", "linestyle":"none", "label": "DD2", "marker": ourmd.eos_dic_marker["DD2"]},
        #     {"ms": 7, "color": "gray", "linestyle":"none", "label": "LS220", "marker": ourmd.eos_dic_marker["LS220"]},
        #     {"ms": 7, "color": "gray", "linestyle":"none", "label": "SFHo", "marker": ourmd.eos_dic_marker["SFHo"]},
        #     {"ms": 7, "color": "gray", "linestyle":"none", "label": "SLy4", "marker": ourmd.eos_dic_marker["SLy4"]},
        #     {"ms": 7, "color": "gray", "linestyle":"none", "label": "BLh", "marker": ourmd.eos_dic_marker["BLh"]}
        # ],
        "savepdf":True,
        "tight_layout":False,
    }

    # ---------------- fit ------------------
    #from make_fit import fitting_function_mej, fitting_coeffs_mej_our
    fit_dic =  {
        #"func":fit_funcs.mej_flat_mean, "coeffs": fit_coefs.mej_all_2(),
        "func": fit_funcs.poly_2_qLambda, "coeffs": np.array([54.247, -5.732e+01, -6.887e-02, 13.604, 0.048, 1.067e-05]),#fit_coefs.mej_all_2(),
        "xmin": 0.95, "xmax": 2.0, "xscale": "linear", "xlabel": "$q$",#r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -2.0, "ymax": 2.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
        "plot_zero":True
    }

    # Fit
    from make_fit import fitting_function_mej
    #from make_fit import fitting_coeffs_mej_david, fitting_coeffs_mej_our
    #from make_fit import complex_fic_data_mej_module
    #complex_fic_data_mej_module(datasets, fitting_coeffs_mej_our(), key_for_usable_dataset="fit")


    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    for k in datasets.keys():
        datasets[k]["plot_errorbar"] = True

    plot_datasets_scatter2(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)
def task_plot_mej_cc():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki                  #
    from model_sets import models_sekiguchi2016 as se16         # [23] arxive:1603.01918 # no Mb
    from model_sets import models_sekiguchi2015 as se15         # [-]  arxive:1502.06660 # no Mb
    from model_sets import models_bauswein2013 as bs            # [20] arxive:1302.6530
    from model_sets import models_lehner2016 as lh              # [22] arxive:1603.00501
    from model_sets import models_hotokezaka2013 as hz          # [19] arxive:1212.0905
    from model_sets import models_dietrich_ujevic2016 as du
    from model_sets import models_dietrich2015 as di15          # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16          # [24] arxive:1607.06636

    # -------------------------------------------
    from collections import OrderedDict
    datasets = OrderedDict() #
    # datasets["bauswein"] =  {'marker': 's', 'ms': 60, "color": "slategray", "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "fit": False}
    # datasets["hotokezaka"] ={'marker': '>', 'ms': 60, "color": "gray", "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "fit": False}
    # datasets["dietrich15"]= {'marker': 'd', "ms": 60, "color": "gray", "models": di15.simulations[di15.mask_for_with_sr], "data": di15, "err": di15.params.Mej_err, "label": r"Dietrich+2015", "fit": False}
    # datasets["sekiguchi15"]={'marker': 'p', "ms": 60, "color": "black", "models": se15.simulations[se15.mask_for_with_sr], "data": se15, "err": se15.params.Mej_err, "label": r"Sekiguchi+2015",  "fit": False}
    # datasets["dietrich16"]= {'marker': 'D', "ms": 60, "color": "gray", "models": di16.simulations[di16.mask_for_with_sr], "data": di16, "err": di16.params.Mej_err, "label": r"Dietrich+2016", "fit": False}
    # datasets["sekiguchi16"]={'marker': 'h', "ms": 60, "color": "black", "models": se16.simulations[se16.mask_for_with_sr], "data": se16, "err": se16.params.Mej_err, "label": r"Sekiguchi+2016", "fit": False}
    # datasets["lehner"] =    {'marker': 'P', 'ms': 60, "color": "orange", "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016",  "fit": False}
    # datasets["radice"] =    {'marker': '*', "ms": 60, "color": "green", "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018",  "fit": False}
    # datasets["kiuchi"] =    {'marker': "X", "ms": 60, "color": "gray", "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019",  "fit": False}
    # datasets["vincent"] =   {'marker': 'v', 'ms': 60, "color": "red", "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019",  "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets['our'] =       {'marker': 'o', 'ms': 60, "color": "#EOS", "models": md.groups, "data": md, "err": "v_n", "label": r"#EOS",  "fit": True}
    #### datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}

    for t in datasets.keys():
        #datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["fill_style"] = "full"#"none"
        datasets[t]["edgecolor"] = "none"#"#EOS"
        #datasets[t]["facecolor"] = "#cb"#"none"
        datasets[t]["marker"] = "#EOS"#"d" # #EOS
        datasets[t]["plot_errorbar"] = True
        datasets[t]["plot_xerrorbar"] = False

    x_dic    = {"v_n": "q", "err": None, "deferr": None, "mod": {}}
    y_dic    = {"v_n": "Mej_tot-geo",     "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (5.5, 6.1),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "fit_panel": True,
        "plot_diagonal":False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": True,  # "tab10",
        "xmin": 0.95, "xmax": 2.0, "xscale": "linear",
        "ymin": 3e-4, "ymax": 2e-2, "yscale": "log",
        "xlabel": "$q$",#r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ylabel": r"$M_{\rm ej}$ $[M_{\odot}]$",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_q_mej_our_poly22_cc.png",
        "legend": {"fancybox": False, "loc": 'lower right',
                   #"bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":1.0,
        #"hline": {"y":3.442 * 1.e-3, "color":'blue', "linestyle":"dashed", "label":"Mean"},
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "add_marker":[
            {"ms": 7, "color": "gray", "linestyle":"none", "label": "DD2", "marker": ourmd.eos_dic_marker["DD2"]},
            {"ms": 7, "color": "gray", "linestyle":"none", "label": "LS220", "marker": ourmd.eos_dic_marker["LS220"]},
            {"ms": 7, "color": "gray", "linestyle":"none", "label": "SFHo", "marker": ourmd.eos_dic_marker["SFHo"]},
            {"ms": 7, "color": "gray", "linestyle":"none", "label": "SLy4", "marker": ourmd.eos_dic_marker["SLy4"]},
            {"ms": 7, "color": "gray", "linestyle":"none", "label": "BLh", "marker": ourmd.eos_dic_marker["BLh"]}
        ],
        "savepdf":True,
        "tight_layout":False,
    }

    # ---------------- fit ------------------
    #from make_fit import fitting_function_mej, fitting_coeffs_mej_our
    fit_dic =  {
        #"func":fit_funcs.mej_flat_mean, "coeffs": fit_coefs.mej_all_2(),
        "func": fit_funcs.poly_2_qLambda, "coeffs": np.array([54.247, -5.732e+01, -6.887e-02, 13.604, 0.048, 1.067e-05]),#fit_coefs.mej_all_2(),
        "xmin": 0.95, "xmax": 2.0, "xscale": "linear", "xlabel": "$q$",#r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -2.0, "ymax": 2.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
        "plot_zero":True
    }

    # Fit
    from make_fit import fitting_function_mej
    #from make_fit import fitting_coeffs_mej_david, fitting_coeffs_mej_our
    #from make_fit import complex_fic_data_mej_module
    #complex_fic_data_mej_module(datasets, fitting_coeffs_mej_our(), key_for_usable_dataset="fit")


    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    for k in datasets.keys():
        datasets[k]["plot_errorbar"] = True

    plot_datasets_scatter3(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)
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

    fit_dics = OrderedDict()
    fit_dics["poly1"] = { # BEST FIT
        "func": fit_funcs.poly_2_Lambda, "coeffs": np.array([4.05e+00, -9.92e-04, 4.05e-06]),
        "xmin": 0, "xmax": 24., "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -100.0, "ymax": 100.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
        "plot_zero": True
    }
    fit_dics["poly2"] = { # 3rd best after the mean which I do not plot
        "func": fit_funcs.poly_2_qLambda, "coeffs": np.array([4.92e+01, -6.72e+01, -4.65e-02, 2.33e+01, 4.06e-02, 1.59e-06]),
        "xmin": 0, "xmax": 24., "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -100.0, "ymax": 100.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
        "plot_zero": True

    }
    fit_dics["Eq.6"] = { # Best of 2 fitting fucntions
        "func": fit_funcs.mej_dietrich16, "coeffs": np.array([-0.442, 6.148, -8.703, -8.853, -2.527]),
        "xmin": 0, "xmax": 24., "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -100.0, "ymax": 100.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
        "plot_zero": True
    }
    fit_dics["Eq.7"] = {  # worst of 2 fitting fucntions
        "func": fit_funcs.mej_kruger20, "coeffs": np.array([0.328, 0.785, -11.008, 6.072]),
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
    subplot_dics["poly1"] = {
        "xmin": -4, "xmax": 40., "xscale": "linear",
        "ymin": -13, "ymax": 4.0, "yscale": "linear",
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
    subplot_dics["poly2"] = {
        "xmin": -4, "xmax": 40., "xscale": "linear",
        "ymin": -13.0, "ymax": 4.0, "yscale": "linear",
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
    subplot_dics["Eq.6"] ={
        "xmin": -4, "xmax": 40., "xscale": "linear",
        "ymin": -13.0, "ymax": 4.0, "yscale": "linear",
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
    subplot_dics["Eq.7"] = {
        "xmin": -4, "xmax": 40., "xscale": "linear",
        "ymin": -13.0, "ymax": 4.0, "yscale": "linear",
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

def task_plot_mej_all_q():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki                  #
    from model_sets import models_sekiguchi2016 as se16         # [23] arxive:1603.01918 # no Mb
    from model_sets import models_sekiguchi2015 as se15         # [-]  arxive:1502.06660 # no Mb
    from model_sets import models_bauswein2013 as bs            # [20] arxive:1302.6530
    from model_sets import models_lehner2016 as lh              # [22] arxive:1603.00501
    from model_sets import models_hotokezaka2013 as hz          # [19] arxive:1212.0905
    from model_sets import models_dietrich_ujevic2016 as du
    from model_sets import models_dietrich2015 as di15          # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16          # [24] arxive:1607.06636

    # -------------------------------------------
    from collections import OrderedDict
    datasets = OrderedDict() #
    datasets["bauswein"] =  {'marker': 's', 'ms': 60, "color": "slategray", "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "fit": False}
    datasets["hotokezaka"] ={'marker': '>', 'ms': 60, "color": "gray", "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "fit": False}
    datasets["dietrich15"]= {'marker': 'd', "ms": 60, "color": "gray", "models": di15.simulations[di15.mask_for_with_sr], "data": di15, "err": di15.params.Mej_err, "label": r"Dietrich+2015", "fit": False}
    datasets["sekiguchi15"]={'marker': 'p', "ms": 60, "color": "black", "models": se15.simulations[se15.mask_for_with_sr], "data": se15, "err": se15.params.Mej_err, "label": r"Sekiguchi+2015",  "fit": False}
    datasets["dietrich16"]= {'marker': 'D', "ms": 60, "color": "gray", "models": di16.simulations[di16.mask_for_with_sr], "data": di16, "err": di16.params.Mej_err, "label": r"Dietrich+2016", "fit": False}
    datasets["sekiguchi16"]={'marker': 'h', "ms": 60, "color": "black", "models": se16.simulations[se16.mask_for_with_sr], "data": se16, "err": se16.params.Mej_err, "label": r"Sekiguchi+2016", "fit": False}
    datasets["lehner"] =    {'marker': 'P', 'ms': 60, "color": "orange", "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016",  "fit": False}
    datasets["radice"] =    {'marker': '*', "ms": 60, "color": "green", "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018",  "fit": False}
    datasets["kiuchi"] =    {'marker': "X", "ms": 60, "color": "gray", "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019",  "fit": False}
    datasets["vincent"] =   {'marker': 'v', 'ms': 60, "color": "red", "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019",  "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets['reference'] =       {'marker': 'o', 'ms': 60, "color": "blue", "models": md.groups, "data": md, "err": "v_n", "label": r"This work",  "fit": False}
    #### datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}

    for t in datasets.keys():
        datasets[t]["label"] = ourmd.datasets_labels[t]
        datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["fill_style"] = "none"
        datasets[t]["plot_errorbar"] = False
        datasets[t]["plot_xerrorbar"] = False
        datasets[t]["facecolor"] = "none"
        datasets[t]["edgecolor"] = datasets[t]["color"]

    x_dic    = {"v_n": "q", "err": None, "deferr": None, "mod": {}}
    y_dic    = {"v_n": "Mej_tot-geo",     "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    from make_fit3 import Fitting_Functions
    plot_dic = {
        "figsize": (6.0, 5.5),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "fit_panel": False,
        "plot_diagonal":False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": False,  # "tab10",
        "xmin": 0.95, "xmax": 2.4, "xscale": "linear",
        "ymin": 1e-4, "ymax": 5e-1, "yscale": "log",
        "xlabel": "$q$",#r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ylabel": r"$M_{\rm ej}$ [$M_{\odot}$]",# $[10^{-3}M_{\odot}]$",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_q_mej.png",
        "legend": {"fancybox": False, "loc": 'lower right',
                   #"bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 13,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        "add_poly": [{"x": np.arange(1., 2., 0.1), "coeffs": [-1e-3 * 2.30e+01 , 1e-3 * 3.81e+01 , -1e-3 * 1.34e+01], # 213.1 & 0.145 \\
                      "color": "blue", "lw": 0.8, "ls": "-.", "label": r"$P_2 ^{\rm micro}$"},
                     {"x": np.arange(1., 2., 0.1), "coeffs": [1e-3 * 5.44e+01 , -1e-3 * 1.05e+02 , 1e-3 * 5.42e+01], # 12.3 & 0.839 \\
                      "color": "black", "lw": 0.8, "ls": "-.", "label": r"$P_2 ^{\rm poly}$"}
                     ],
        # "hline": {"y":5.220 * 1.e-3, "color":'blue', "linestyle":"dashed", "label":"Mean"},
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "savepdf":True,
        "tight_layout":True
    }

    # ---------------- fit ------------------
    from make_fit import fitting_function_mej, fitting_coeffs_mej_our
    fit_dic = {}
    #     {
    #     "func":fitting_function_mej, "coeffs": fitting_coeffs_mej_our(),
    #     "xmin": 0, "xmax": 7.5, "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
    #     "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
    #     "plot_zero":True
    # }

    # Fit
    from make_fit import fitting_function_mej
    #from make_fit import fitting_coeffs_mej_david, fitting_coeffs_mej_our
    #from make_fit import complex_fic_data_mej_module
    #complex_fic_data_mej_module(datasets, fitting_coeffs_mej_our(), key_for_usable_dataset="fit")


    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    for k in datasets.keys():
        datasets[k]["plot_errorbar"] = False

    plot_datasets_scatter2(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)
def task_plot_mej_all_Lambda():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki                  #
    from model_sets import models_sekiguchi2016 as se16         # [23] arxive:1603.01918 # no Mb
    from model_sets import models_sekiguchi2015 as se15         # [-]  arxive:1502.06660 # no Mb
    from model_sets import models_bauswein2013 as bs            # [20] arxive:1302.6530
    from model_sets import models_lehner2016 as lh              # [22] arxive:1603.00501
    from model_sets import models_hotokezaka2013 as hz          # [19] arxive:1212.0905
    from model_sets import models_dietrich_ujevic2016 as du
    from model_sets import models_dietrich2015 as di15          # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16          # [24] arxive:1607.06636

    # -------------------------------------------
    from collections import OrderedDict
    datasets = OrderedDict() #
    # datasets["bauswein"] =  {'marker': 's', 'ms': 60, "color": "slategray", "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "fit": False}
    # datasets["hotokezaka"] ={'marker': '>', 'ms': 60, "color": "gray", "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "fit": False}
    datasets["dietrich15"]= {'marker': 'd', "ms": 60, "color": "gray", "models": di15.simulations[di15.mask_for_with_sr], "data": di15, "err": di15.params.Mej_err, "label": r"Dietrich+2015", "fit": False}
    # datasets["sekiguchi15"]={'marker': 'p', "ms": 60, "color": "black", "models": se15.simulations[se15.mask_for_with_sr], "data": se15, "err": se15.params.Mej_err, "label": r"Sekiguchi+2015",  "fit": False}
    datasets["dietrich16"]= {'marker': 'D', "ms": 60, "color": "gray", "models": di16.simulations[di16.mask_for_with_sr], "data": di16, "err": di16.params.Mej_err, "label": r"Dietrich+2016", "fit": False}
    # datasets["sekiguchi16"]={'marker': 'h', "ms": 60, "color": "black", "models": se16.simulations[se16.mask_for_with_sr], "data": se16, "err": se16.params.Mej_err, "label": r"Sekiguchi+2016", "fit": False}
    datasets["lehner"] =    {'marker': 'P', 'ms': 60, "color": "orange", "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016",  "fit": False}
    datasets["radice"] =    {'marker': '*', "ms": 60, "color": "green", "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018",  "fit": False}
    datasets["kiuchi"] =    {'marker': "X", "ms": 60, "color": "gray", "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019",  "fit": False}
    datasets["vincent"] =   {'marker': 'v', 'ms': 60, "color": "red", "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019",  "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets['reference'] =       {'marker': 'o', 'ms': 60, "color": "blue", "models": md.groups, "data": md, "err": "v_n", "label": r"This work",  "fit": False}
    #### datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}

    for t in datasets.keys():
        datasets[t]["label"] = ourmd.datasets_labels[t]
        datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["fill_style"] = "none"
        datasets[t]["plot_errorbar"] = False
        datasets[t]["plot_xerrorbar"] = False
        datasets[t]["facecolor"] = "none"
        datasets[t]["edgecolor"] = datasets[t]["color"]

    x_dic    = {"v_n": "Lambda", "err": None, "deferr": None, "mod": {}}
    y_dic    = {"v_n": "Mej_tot-geo",     "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (6.0, 5.5),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "fit_panel": False,
        "plot_diagonal":False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": False,  # "tab10",
        "xmin": 0, "xmax": 2000, "xscale": "linear",
        "ymin": 1e-4, "ymax": 5e-1, "yscale": "log",
        "xlabel": r"$\tilde{\Lambda}$",#r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ylabel": r"$M_{\rm ej}$ [$M_{\odot}$]",# $[10^{-3}M_{\odot}]$",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_lambda_mej.png",
        "add_poly": [{"x": np.arange(200, 2000, 100), "coeffs": [1e-3 * 3.28e+00, -1e-3 * 2.54e-03, 1e-3 * 1.32e-06],
                      "color": "blue", "lw": 0.8, "ls": "-.", "label": r"$P_2 ^{\rm micro}$"},
                     {"x": np.arange(200, 2000, 100), "coeffs": [-1e-3 * 1.15e+00 , 1e-3 *  2.78e-02 , -1e-3 * 8.71e-06],
                      "color": "black", "lw": 0.8, "ls": "-.", "label": r"$P_2 ^{\rm poly}$"}
                     ],
        "legend": {"fancybox": False, "loc": 'upper left',
                   #"bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 13,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        # "hline": {"y":5.220 * 1.e-3, "color":'blue', "linestyle":"dashed", "label":"Mean"},
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "savepdf":True,
        "tight_layout":True
    }

    # ---------------- fit ------------------
    from make_fit import fitting_function_mej, fitting_coeffs_mej_our
    fit_dic = {}
    #     {
    #     "func":fitting_function_mej, "coeffs": fitting_coeffs_mej_our(),
    #     "xmin": 0, "xmax": 7.5, "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
    #     "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
    #     "plot_zero":True
    # }

    # Fit
    from make_fit import fitting_function_mej
    #from make_fit import fitting_coeffs_mej_david, fitting_coeffs_mej_our
    #from make_fit import complex_fic_data_mej_module
    #complex_fic_data_mej_module(datasets, fitting_coeffs_mej_our(), key_for_usable_dataset="fit")


    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    for k in datasets.keys():
        datasets[k]["plot_errorbar"] = False

    plot_datasets_scatter2(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)

''' --- vej --- '''

def task_plot_vej_vs_fit_all():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki                  #
    from model_sets import models_sekiguchi2016                 # [23] arxive:1603.01918
    from model_sets import models_sekiguchi2015                 # [-]  arxive:1502.06660
    from model_sets import models_bauswein2013 as bs            # [20] arxive:1302.6530
    from model_sets import models_lehner2016 as lh              # [22] arxive:1603.00501
    from model_sets import models_hotokezaka2013 as hz          # [19] arxive:1212.0905
    from model_sets import models_dietrich_ujevic2016 as du
    from model_sets import models_dietrich2015 as di15          # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16          # [24] arxive:1607.06636

    # -------------------------------------------

    datasets = {}
    # # datasets["kiuchi"] =    {'marker': "X", "ms": 20, "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019", "color": "gray", "fit": False}
    # datasets["radice"] =    {'marker': '*', "ms": 40, "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.vej_err, "label": r"Radice+2018", "color": "blue", "fit": True}
    # # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": di.simulations[di.mask_for_with_sr], "data":di, "err":di.params.Mej_err, "label":"Dietrich+2015"}
    # # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": di.simulations[di.mask_for_with_sr], "data":di, "err":di.params.Mej_err, "label":"Dietrich+2016"}
    # # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}
    # # datasets["vincent"] =   {'marker': 'v', 'ms': 20, "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019", "color": "blue", "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    # # datasets["bauswein"] =  {'marker': 's', 'ms': 20, "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "color": "blue", "fit": False}
    # # datasets["lehner"] =    {'marker': 'P', 'ms': 20, "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016", "color": "blue", "fit": False}
    # # datasets["hotokezaka"] ={'marker': '>', 'ms': 20, "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "color": "gray", "fit": False}
    # datasets['our'] =       {'marker': 'o', 'ms': 40, "models": md.groups, "data": md, "err": "v_n", "label": r"This work", "color": "red", "fit": True}

    datasets['our'] = {"models": md.groups, "data": md, "fit": True, "color":None, "plot_errorbar":True, "err": "v_n"}
    datasets["radice"] = {"models": rd.simulations[rd.fiducial], "data": rd, "fit": True,"color":None, "plot_errorbar":True, "err": rd.params.vej_err}
    #datasets["kiuchi"] = {"models": ki.simulations[ki.mask_for_with_tov_data], "data": ki, "fit": True,"color":None, "plot_errorbar":False, "err": ki.params.vej_err}
    datasets["vincent"] = {"models": vi.simulations, "data": vi, "fit": True,"color":None, "plot_errorbar":False, "err": vi.params.vej_err}
    datasets["dietrich16"] = {"models": di16.simulations[di16.mask_for_with_sr], "data": di16,"fit": True,"color":None, "plot_errorbar":False, "err": di16.params.vej_err}
    datasets["dietrich15"] = {"models": di15.simulations[di15.mask_for_with_sr], "data": di15, "fit": True,"color":None, "plot_errorbar":False, "err": di15.params.vej_err}

    for t in datasets.keys():
        datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["label"] = ourmd.datasets_labels[t]
        datasets[t]["ms"] = 40
       # datasets[t]["fill_style"] = "none"

    x_dic   = {"v_n": "vel_inf_ave-geo_fit", "err": None, "deferr": None, "mod": {}}
    y_dic   = {"v_n": "vel_inf_ave-geo",     "err": "ud", "deferr": 0.2,  "mod": {}}
    col_dic = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (5.5, 6.1),
        "left": 0.15, "bottom": 0.10, "top": 0.93, "right": 1.00, "hspace": 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        # "figsize": (4.5, 3.5),
        # "left": 0.15, "bottom": 0.14, "top": 0.92, "right": 0.95, "hspace": 0,
        "fit_panel": True,
        "plot_diagonal":True,
        "tight_layout": False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": True,  # "tab10",
        "xmin":0.18, "xmax":.30, "xscale": "linear",
        "ymin": 0.08, "ymax": .40, "yscale": "linear",
        "xlabel": r"$\upsilon_{\rm ej;fit}$ [c]",
        "ylabel": r"$\upsilon_{\rm ej}$ [c]",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_vel_fit_all.png",
        "legend": {"fancybox": False, "loc": 'upper right',
                   #"bbox_to_anchor": (0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 13.,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "savepdf":True
    }

    # -------------- fit ----------------
    #from make_fit import complex_fic_data_vinf_module, fitting_function_vinf, fitting_coeffs_vinf_our

    fit_dic = {
        "func": fit_funcs.vej_dietrich16, "coeffs": fit_coefs.vel_all_2(),
        "xmin": 0.05, "xmax": .30, "xscale": "linear", "xlabel": r"$\upsilon_{\rm ej;fit}$ [c]",
        "ymin": -0.9, "ymax": 0.9, "yscale": "linear", "ylabel": r"$\Delta \upsilon_{\rm ej} / \upsilon_{\rm ej}$",
        "plot_zero":True
    }

    # Fit
    # from make_fit import fitting_function_mej
    # from make_fit import fitting_coeffs_mej_david # fitting_coeffs_mej
    # from make_fit import complex_fic_data_vinf_module
    # x_davids_fit = fitting_coeffs_mej_david()# fitting_coeffs_mej()  # radice()
    # fitting_func_of_lam = fitting_function_mej  # takes x_davids_fit, lam_array


    # plot_datasets_scatter(x_dic, y_dic, col_dic, plot_dic, x_davids_fit, fitting_func_of_lam, datasets)

    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "sekiguchi15": datasets["sekiguchi15"]["color"] = "gray" # no Mb
    #     if k == "sekiguchi16": datasets["sekiguchi16"]["color"] = "gray" # no Mb
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    #     if k == "dietrich15": datasets["dietrich15"]["color"]   = None
    #     if k == "dietrich16": datasets["dietrich16"]["color"]   = None
    # for k in datasets.keys():
    #     datasets[k]["plot_errorbar"] = True

    plot_datasets_scatter2(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)

    #complex_fic_data_vinf_module(datasets, fitting_coeffs_vinf_our, key_for_usable_dataset="fit")
def task_plot_vej_vs_fit():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki                  #
    from model_sets import models_sekiguchi2016                 # [23] arxive:1603.01918
    from model_sets import models_sekiguchi2015                 # [-]  arxive:1502.06660
    from model_sets import models_bauswein2013 as bs            # [20] arxive:1302.6530
    from model_sets import models_lehner2016 as lh              # [22] arxive:1603.00501
    from model_sets import models_hotokezaka2013 as hz          # [19] arxive:1212.0905
    from model_sets import models_dietrich_ujevic2016 as du
    from model_sets import models_dietrich2015 as di15          # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16          # [24] arxive:1607.06636

    # -------------------------------------------

    datasets = {}
    # # datasets["kiuchi"] =    {'marker': "X", "ms": 20, "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019", "color": "gray", "fit": False}
    # datasets["radice"] =    {'marker': '*', "ms": 40, "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.vej_err, "label": r"Radice+2018", "color": "blue", "fit": True}
    # # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": di.simulations[di.mask_for_with_sr], "data":di, "err":di.params.Mej_err, "label":"Dietrich+2015"}
    # # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": di.simulations[di.mask_for_with_sr], "data":di, "err":di.params.Mej_err, "label":"Dietrich+2016"}
    # # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}
    # # datasets["vincent"] =   {'marker': 'v', 'ms': 20, "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019", "color": "blue", "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    # # datasets["bauswein"] =  {'marker': 's', 'ms': 20, "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "color": "blue", "fit": False}
    # # datasets["lehner"] =    {'marker': 'P', 'ms': 20, "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016", "color": "blue", "fit": False}
    # # datasets["hotokezaka"] ={'marker': '>', 'ms': 20, "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "color": "gray", "fit": False}
    # datasets['our'] =       {'marker': 'o', 'ms': 40, "models": md.groups, "data": md, "err": "v_n", "label": r"This work", "color": "red", "fit": True}

    datasets['our'] = {"models": md.groups, "data": md, "fit": True, "color":None, "plot_errorbar":True, "err": "v_n"}
    datasets["radice"] = {"models": rd.simulations[rd.fiducial], "data": rd, "fit": True,"color":None, "plot_errorbar":True, "err": rd.params.vej_err}
    #datasets["kiuchi"] = {"models": ki.simulations[ki.mask_for_with_tov_data], "data": ki, "fit": True,"color":None, "plot_errorbar":False, "err": ki.params.vej_err}
    # datasets["vincent"] = {"models": vi.simulations, "data": vi, "fit": True,"color":None, "plot_errorbar":False, "err": vi.params.vej_err}
    # datasets["dietrich16"] = {"models": di16.simulations[di16.mask_for_with_sr], "data": di16,"fit": True,"color":None, "plot_errorbar":False, "err": di16.params.vej_err}
    # datasets["dietrich15"] = {"models": di15.simulations[di15.mask_for_with_sr], "data": di15, "fit": True,"color":None, "plot_errorbar":False, "err": di15.params.vej_err}

    for t in datasets.keys():
        datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["label"] = ourmd.datasets_labels[t]
        datasets[t]["ms"] = 40
       # datasets[t]["fill_style"] = "none"

    x_dic   = {"v_n": "vel_inf_ave-geo_fit", "err": None, "deferr": None, "mod": {}}
    y_dic   = {"v_n": "vel_inf_ave-geo",     "err": "ud", "deferr": 0.2,  "mod": {}}
    col_dic = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (5.5, 6.1),
        "left": 0.15, "bottom": 0.10, "top": 0.93, "right": 1.00, "hspace": 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        # "figsize": (4.5, 3.5),
        # "left": 0.15, "bottom": 0.14, "top": 0.92, "right": 0.95, "hspace": 0,
        "fit_panel": True,
        "plot_diagonal":True,
        "tight_layout": False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": True,  # "tab10",
        "xmin":0.05, "xmax":.3, "xscale": "linear", "xlabel": r"$\upsilon_{\rm ej;fit}$ [c]",
        "ymin": 0.08, "ymax": .40, "yscale": "linear", "ylabel": r"$\upsilon_{\rm ej}$ [c]",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_vel_fit.png",
        "legend": {"fancybox": False, "loc": 'upper right',
                   #"bbox_to_anchor": (0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 13.,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "savepdf":True
    }

    # -------------- fit ----------------
    #from make_fit import complex_fic_data_vinf_module, fitting_function_vinf, fitting_coeffs_vinf_our

    fit_dic = {
        "func": fit_funcs.vej_dietrich16, "coeffs": fit_coefs.vej_us_radice(),
        "xmin": 0.05, "xmax": .3, "xscale": "linear", "xlabel": r"$\upsilon_{\rm ej;fit}$ [c]",
        "ymin": -0.9, "ymax": 0.9, "yscale": "linear", "ylabel": r"$\Delta \upsilon_{\rm ej} / \upsilon_{\rm ej}$",
        "plot_zero":True
    }

    # Fit
    # from make_fit import fitting_function_mej
    # from make_fit import fitting_coeffs_mej_david # fitting_coeffs_mej
    # from make_fit import complex_fic_data_vinf_module
    # x_davids_fit = fitting_coeffs_mej_david()# fitting_coeffs_mej()  # radice()
    # fitting_func_of_lam = fitting_function_mej  # takes x_davids_fit, lam_array


    # plot_datasets_scatter(x_dic, y_dic, col_dic, plot_dic, x_davids_fit, fitting_func_of_lam, datasets)

    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "sekiguchi15": datasets["sekiguchi15"]["color"] = "gray" # no Mb
    #     if k == "sekiguchi16": datasets["sekiguchi16"]["color"] = "gray" # no Mb
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    #     if k == "dietrich15": datasets["dietrich15"]["color"]   = None
    #     if k == "dietrich16": datasets["dietrich16"]["color"]   = None
    # for k in datasets.keys():
    #     datasets[k]["plot_errorbar"] = True

    plot_datasets_scatter2(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)

    #complex_fic_data_vinf_module(datasets, fitting_coeffs_vinf_our, key_for_usable_dataset="fit")

def task_plot_vej_all():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki                  #
    from model_sets import models_sekiguchi2016 as se16         # [23] arxive:1603.01918 # no Mb
    from model_sets import models_sekiguchi2015 as se15         # [-]  arxive:1502.06660 # no Mb
    from model_sets import models_bauswein2013 as bs            # [20] arxive:1302.6530
    from model_sets import models_lehner2016 as lh              # [22] arxive:1603.00501
    from model_sets import models_hotokezaka2013 as hz          # [19] arxive:1212.0905
    from model_sets import models_dietrich_ujevic2016 as du
    from model_sets import models_dietrich2015 as di15          # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16          # [24] arxive:1607.06636

    # -------------------------------------------
    from collections import OrderedDict
    datasets = OrderedDict() #
    datasets["bauswein"] =  {'marker': 's', 'ms': 60, "color": "slategray", "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "fit": False}
    datasets["hotokezaka"] ={'marker': '>', 'ms': 60, "color": "gray", "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "fit": False}
    datasets["dietrich15"]= {'marker': 'd', "ms": 60, "color": "gray", "models": di15.simulations[di15.mask_for_with_sr], "data": di15, "err": di15.params.Mej_err, "label": r"Dietrich+2015", "fit": False}
    #datasets["sekiguchi15"]={'marker': 'p', "ms": 20, "color": "black", "models": se15.simulations[se15.mask_for_with_sr], "data": se15, "err": se15.params.Mej_err, "label": r"Sekiguchi+2015",  "fit": False}
    datasets["dietrich16"]= {'marker': 'D', "ms": 60, "color": "gray", "models": di16.simulations[di16.mask_for_with_sr], "data": di16, "err": di16.params.Mej_err, "label": r"Dietrich+2016", "fit": False}
    datasets["sekiguchi16"]={'marker': 'h', "ms": 60, "color": "black", "models": se16.simulations[se16.mask_for_with_sr], "data": se16, "err": se16.params.Mej_err, "label": r"Sekiguchi+2016", "fit": False}
    datasets["lehner"] =    {'marker': 'P', 'ms': 60, "color": "orange", "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016",  "fit": False}
    datasets["radice"] =    {'marker': '*', "ms": 60, "color": "green", "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018",  "fit": False}
    #datasets["kiuchi"] =    {'marker': "X", "ms": 20, "color": "gray", "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019",  "fit": False}
    datasets["vincent"] =   {'marker': 'v', 'ms': 60, "color": "red", "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019",  "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets['our'] =       {'marker': 'o', 'ms': 60, "color": "blue", "models": md.groups, "data": md, "err": "v_n", "label": r"This work",  "fit": False}
    # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}

    for t in datasets.keys():
        datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["fill_style"] = "none"

    x_dic    = {"v_n": "q", "err": None, "deferr": None, "mod": {}}
    y_dic    = {"v_n": "vel_inf_ave-geo", "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (6.0, 5.5),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "fit_panel": False,
        "plot_diagonal":False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": False,  # "tab10",
        "xmin": 0.95, "xmax": 2.4, "xscale": "linear",
        "ymin": 0.1, "ymax": 0.5, "yscale": "linear",
        "xlabel": "$M_1/M_2$",#r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ylabel": r"$\langle \upsilon_{\rm ej} \rangle$ [c]",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_q_vej.png",
        "legend": {"fancybox": False, "loc": 'upper right',
                   #"bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "savepdf":True,
        "tight_layout":True
    }

    # ---------------- fit ------------------
    from make_fit import fitting_function_mej, fitting_coeffs_mej_our
    fit_dic = {}
    #     {
    #     "func":fitting_function_mej, "coeffs": fitting_coeffs_mej_our(),
    #     "xmin": 0, "xmax": 7.5, "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
    #     "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
    #     "plot_zero":True
    # }

    # Fit
    from make_fit import fitting_function_mej
    #from make_fit import fitting_coeffs_mej_david, fitting_coeffs_mej_our
    #from make_fit import complex_fic_data_mej_module
    #complex_fic_data_mej_module(datasets, fitting_coeffs_mej_our(), key_for_usable_dataset="fit")


    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    for k in datasets.keys():
        datasets[k]["plot_errorbar"] = False

    plot_datasets_scatter2(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)
def task_plot_vej():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki                  #
    from model_sets import models_sekiguchi2016 as se16         # [23] arxive:1603.01918 # no Mb
    from model_sets import models_sekiguchi2015 as se15         # [-]  arxive:1502.06660 # no Mb
    from model_sets import models_bauswein2013 as bs            # [20] arxive:1302.6530
    from model_sets import models_lehner2016 as lh              # [22] arxive:1603.00501
    from model_sets import models_hotokezaka2013 as hz          # [19] arxive:1212.0905
    from model_sets import models_dietrich_ujevic2016 as du
    from model_sets import models_dietrich2015 as di15          # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16          # [24] arxive:1607.06636

    # -------------------------------------------
    from collections import OrderedDict
    datasets = OrderedDict() #
    # datasets["bauswein"] =  {'marker': 's', 'ms': 60, "color": "slategray", "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "fit": False}
    # datasets["hotokezaka"] ={'marker': '>', 'ms': 60, "color": "gray", "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "fit": False}
    # datasets["dietrich15"]= {'marker': 'd', "ms": 60, "color": "gray", "models": di15.simulations[di15.mask_for_with_sr], "data": di15, "err": di15.params.Mej_err, "label": r"Dietrich+2015", "fit": False}
    # datasets["sekiguchi15"]={'marker': 'p', "ms": 60, "color": "black", "models": se15.simulations[se15.mask_for_with_sr], "data": se15, "err": se15.params.Mej_err, "label": r"Sekiguchi+2015",  "fit": False}
    # datasets["dietrich16"]= {'marker': 'D', "ms": 60, "color": "gray", "models": di16.simulations[di16.mask_for_with_sr], "data": di16, "err": di16.params.Mej_err, "label": r"Dietrich+2016", "fit": False}
    # datasets["sekiguchi16"]={'marker': 'h', "ms": 60, "color": "black", "models": se16.simulations[se16.mask_for_with_sr], "data": se16, "err": se16.params.Mej_err, "label": r"Sekiguchi+2016", "fit": False}
    # datasets["lehner"] =    {'marker': 'P', 'ms': 60, "color": "orange", "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016",  "fit": False}
    # datasets["radice"] =    {'marker': '*', "ms": 60, "color": "green", "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018",  "fit": False}
    # datasets["kiuchi"] =    {'marker': "X", "ms": 60, "color": "gray", "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019",  "fit": False}
    # datasets["vincent"] =   {'marker': 'v', 'ms': 60, "color": "red", "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019",  "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets['our'] =       {'marker': 'o', 'ms': 60, "color": "#EOS", "models": md.groups, "data": md, "err": "v_n", "label": r"#EOS",  "fit": True}
    #### datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}

    for t in datasets.keys():
        #datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["fill_style"] = "none"
        datasets[t]["edgecolor"] = "#EOS"
        datasets[t]["facecolor"] = "none"
        datasets[t]["marker"] = "d" #EOS"
        datasets[t]["plot_errorbar"] = True
        datasets[t]["plot_xerrorbar"] = False


    x_dic    = {"v_n": "q", "err": None, "deferr": None, "mod": {}}
    y_dic    = {"v_n": "vel_inf_ave-geo", "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (5.5, 6.1),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "fit_panel": True,
        "plot_diagonal":False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": False,  # "tab10",
        "xmin": 0.95, "xmax": 2.0, "xscale": "linear",
        "ymin": 0.1, "ymax": 0.30, "yscale": "linear",
        "xlabel": "$M_1/M_2$",#r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ylabel": r"$\langle \upsilon_{\rm ej} \rangle$ [c]",
        #"cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_q_vej_our_poly22.png",
        "legend": {"fancybox": False, "loc": 'upper right',
                   #"bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":1.0,
        "hline": {},#{"y":3.442 * 1.e-3, "color":'blue', "linestyle":"dashed", "label":"Mean"},
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "savepdf":True,
        "tight_layout":False,
    }

    # ---------------- fit ------------------
    #from make_fit import fitting_function_mej, fitting_coeffs_mej_our
    fit_dic =  {
        "func":fit_funcs.vej_poly_22, "coeffs": np.array([0.677, -1.819e-1, -1.083e-3, -4.912e-2, 3.893e-4, 4.239e-7]),
        "xmin": 0.95, "xmax": 2.0, "xscale": "linear", "xlabel": "$q$",#r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -0.45, "ymax": 0.45, "yscale": "linear", "ylabel": r"$\Delta \langle \upsilon_{\rm ej} \rangle / \langle \upsilon_{\rm ej} \rangle$",
        "plot_zero":True
    }

    # Fit
    from make_fit import fitting_function_mej
    #from make_fit import fitting_coeffs_mej_david, fitting_coeffs_mej_our
    #from make_fit import complex_fic_data_mej_module
    #complex_fic_data_mej_module(datasets, fitting_coeffs_mej_our(), key_for_usable_dataset="fit")


    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    for k in datasets.keys():
        datasets[k]["plot_errorbar"] = True

    plot_datasets_scatter2(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)
def task_plot_vej_cc():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki                  #
    from model_sets import models_sekiguchi2016 as se16         # [23] arxive:1603.01918 # no Mb
    from model_sets import models_sekiguchi2015 as se15         # [-]  arxive:1502.06660 # no Mb
    from model_sets import models_bauswein2013 as bs            # [20] arxive:1302.6530
    from model_sets import models_lehner2016 as lh              # [22] arxive:1603.00501
    from model_sets import models_hotokezaka2013 as hz          # [19] arxive:1212.0905
    from model_sets import models_dietrich_ujevic2016 as du
    from model_sets import models_dietrich2015 as di15          # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16          # [24] arxive:1607.06636

    # -------------------------------------------
    from collections import OrderedDict
    datasets = OrderedDict() #
    # datasets["bauswein"] =  {'marker': 's', 'ms': 60, "color": "slategray", "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "fit": False}
    # datasets["hotokezaka"] ={'marker': '>', 'ms': 60, "color": "gray", "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "fit": False}
    # datasets["dietrich15"]= {'marker': 'd', "ms": 60, "color": "gray", "models": di15.simulations[di15.mask_for_with_sr], "data": di15, "err": di15.params.Mej_err, "label": r"Dietrich+2015", "fit": False}
    # datasets["sekiguchi15"]={'marker': 'p', "ms": 60, "color": "black", "models": se15.simulations[se15.mask_for_with_sr], "data": se15, "err": se15.params.Mej_err, "label": r"Sekiguchi+2015",  "fit": False}
    # datasets["dietrich16"]= {'marker': 'D', "ms": 60, "color": "gray", "models": di16.simulations[di16.mask_for_with_sr], "data": di16, "err": di16.params.Mej_err, "label": r"Dietrich+2016", "fit": False}
    # datasets["sekiguchi16"]={'marker': 'h', "ms": 60, "color": "black", "models": se16.simulations[se16.mask_for_with_sr], "data": se16, "err": se16.params.Mej_err, "label": r"Sekiguchi+2016", "fit": False}
    # datasets["lehner"] =    {'marker': 'P', 'ms': 60, "color": "orange", "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016",  "fit": False}
    # datasets["radice"] =    {'marker': '*', "ms": 60, "color": "green", "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018",  "fit": False}
    # datasets["kiuchi"] =    {'marker': "X", "ms": 60, "color": "gray", "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019",  "fit": False}
    # datasets["vincent"] =   {'marker': 'v', 'ms': 60, "color": "red", "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019",  "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets['our'] =       {'marker': 'o', 'ms': 60, "color": "#EOS", "models": md.groups, "data": md, "err": "v_n", "label": r"#EOS",  "fit": True}
    #### datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}

    for t in datasets.keys():
        #datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["fill_style"] = "full"#"none"
        datasets[t]["edgecolor"] = "none"#"#EOS"
        #datasets[t]["facecolor"] = "#cb"#"none"
        datasets[t]["marker"] = "#EOS"#"d" # #EOS
        datasets[t]["plot_errorbar"] = True
        datasets[t]["plot_xerrorbar"] = False


    x_dic    = {"v_n": "q", "err": None, "deferr": None, "mod": {}}
    y_dic    = {"v_n": "vel_inf_ave-geo", "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (5.5, 6.1),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "fit_panel": True,
        "plot_diagonal":False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": True,  # "tab10",
        "xmin": 0.95, "xmax": 2.0, "xscale": "linear",
        "ymin": 0.1, "ymax": 0.30, "yscale": "linear",
        "xlabel": "$M_1/M_2$",#r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ylabel": r"$\langle \upsilon_{\rm ej} \rangle$ [c]",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_q_vej_our_poly22_cc.png",
        "legend": {"fancybox": False, "loc": 'upper right',
                   #"bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":1.0,
        "add_marker": [
            {"ms": 7, "color": "gray", "linestyle": "none", "label": "DD2", "marker": ourmd.eos_dic_marker["DD2"]},
            {"ms": 7, "color": "gray", "linestyle": "none", "label": "LS220", "marker": ourmd.eos_dic_marker["LS220"]},
            {"ms": 7, "color": "gray", "linestyle": "none", "label": "SFHo", "marker": ourmd.eos_dic_marker["SFHo"]},
            {"ms": 7, "color": "gray", "linestyle": "none", "label": "SLy4", "marker": ourmd.eos_dic_marker["SLy4"]},
            {"ms": 7, "color": "gray", "linestyle": "none", "label": "BLh", "marker": ourmd.eos_dic_marker["BLh"]}
        ],
        "hline": {},#{"y":3.442 * 1.e-3, "color":'blue', "linestyle":"dashed", "label":"Mean"},
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "savepdf":True,
        "tight_layout":False,
    }

    # ---------------- fit ------------------
    #from make_fit import fitting_function_mej, fitting_coeffs_mej_our
    fit_dic =  {
        "func":fit_funcs.vej_poly_22, "coeffs": np.array([0.677, -1.819e-1, -1.083e-3, -4.912e-2, 3.893e-4, 4.239e-7]),
        "xmin": 0.95, "xmax": 2.0, "xscale": "linear", "xlabel": "$q$",#r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -0.45, "ymax": 0.45, "yscale": "linear", "ylabel": r"$\Delta \langle \upsilon_{\rm ej} \rangle / \langle \upsilon_{\rm ej} \rangle$",
        "plot_zero":True
    }

    # Fit
    from make_fit import fitting_function_mej
    #from make_fit import fitting_coeffs_mej_david, fitting_coeffs_mej_our
    #from make_fit import complex_fic_data_mej_module
    #complex_fic_data_mej_module(datasets, fitting_coeffs_mej_our(), key_for_usable_dataset="fit")


    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    for k in datasets.keys():
        datasets[k]["plot_errorbar"] = True

    plot_datasets_scatter3(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)
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

    fit_dics = OrderedDict()
    fit_dics["poly2"] = {
        "func": fit_funcs.poly_2_qLambda,
        "coeffs": np.array([4.35e-01, -2.56e-01, -1.10e-04, 6.50e-02, 1.78e-05, 4.59e-08]),
        # np.array([2.549e-03, 2.394e-03, -3.005e-05, -3.376e-03, 3.826e-05, -1.149e-08]),
        "xmin": 0.05, "xmax": .3, "xscale": "linear", "xlabel": r"$\upsilon_{\rm ej;fit}$ [c]",
        "ymin": 0.08, "ymax": .40, "yscale": "linear", "ylabel": r"$\upsilon_{\rm ej}$ [c]",
        "plot_zero": True
    }
    fit_dics["Eq.9"] = {
        "func": fit_funcs.vej_dietrich16, "coeffs": np.array([-0.185, 0.529, -0.713]),
        "xmin": 0.05, "xmax": .3, "xscale": "linear", "xlabel": r"$\upsilon_{\rm ej;fit}$ [c]",
        "ymin": 0.08, "ymax": .40, "yscale": "linear", "ylabel": r"$\upsilon_{\rm ej}$ [c]",
        "plot_zero": True
    }
    fit_dics["poly1"] = {
        "func": fit_funcs.poly_2_Lambda, "coeffs": np.array([2.33e-01, -1.05e-04, 5.35e-08]),
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

''' --- ye --- '''

def task_plot_ye_vs_fit_all():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md

    # -------------------------------------------

    datasets = {}
    # datasets["radice"] =  {'marker': '*', "ms": 40, "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Yeej_err, "label": r"Radice+2018", "color": "blue", "fit": True}
    # #datasets["vincent"] = {'marker': 'v', 'ms': 40, "models": vi.simulations, "data": vi, "err": vi.params.Yeej_err, "label": r"Vincent+2019", "color": "blue", "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    # datasets['our'] =     {'marker': 'o', 'ms': 40, "models": md.groups, "data": md, "err": "v_n", "label": r"This work", "color": "red", "fit": True}
    datasets['our'] = {"models": md.groups, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": "v_n"}
    datasets["radice"] = {"models": rd.simulations[rd.fiducial], "data": rd, "fit": True, "color": None, "plot_errorbar": True, "err": rd.params.Yeej_err}
    # datasets["kiuchi"] = {"models": ki.simulations[ki.mask_for_with_tov_data], "data": ki, "fit": True,"color":None, "plot_errorbar":False, "err": ki.params.vej_err}
    datasets["vincent"] = {"models": vi.simulations, "data": vi, "fit": True, "color": None, "plot_errorbar": False, "err": vi.params.Yeej_err}

    for t in datasets.keys():
        datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["label"] = ourmd.datasets_labels[t]
        datasets[t]["ms"] = 40

    x_dic   = {"v_n": "Ye_ave-geo_fit", "err": None, "deferr": None, "mod": {}}
    y_dic   = {"v_n": "Ye_ave-geo",     "err": "ud", "deferr": 0.2,  "mod": {}}
    col_dic = {"v_n": "Lambda",         "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (5.5, 6.1),
        "left": 0.15, "bottom": 0.10, "top": 0.93, "right": 1.00, "hspace": 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "fit_panel": True,
        "plot_diagonal":True,
        "tight_layout": False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": True,  # "tab10",
        "xmin":0.1, "xmax":.21, "xscale": "linear",
        "ymin": 0.05, "ymax": .32, "yscale": "linear",
        "xlabel": r"$ Y_{e\: \rm{ej; fit}} $",
        "ylabel": r"$ Y_{e\: \rm ej} $",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_ye_fit_all.png",
        "legend": {"fancybox": False, "loc": 'upper right',
                   #"bbox_to_anchor": (0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 13,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        # "dpi": 128, "fontsize":12, "labelsize":12,
        "savepdf":True
    }

    # fit
    # from make_fit import complex_fic_data_ye_module, fitting_function_ye, fitting_coeffs_ye2

    fit_dic = {
        "func":fit_funcs.yeej_like_vej, "coeffs": fit_coefs.yeej_all_2(),
        "xmin": 0.1, "xmax": .25, "xscale": "linear", "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
        "ymin": -2.0, "ymax": 0.8, "yscale": "linear", "ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
        "plot_zero":True
    }

    # complex_fic_data_ye_module(datasets, fitting_function_ye, fitting_coeffs_ye2(), "fit")#; exit(1)

    # Fit
    # from make_fit import fitting_function_mej
    # from make_fit import fitting_coeffs_mej_david # fitting_coeffs_mej
    # from make_fit import complex_fic_data_vinf_module
    # x_davids_fit = fitting_coeffs_mej_david()# fitting_coeffs_mej()  # radice()
    # fitting_func_of_lam = fitting_function_mej  # takes x_davids_fit, lam_array


    # # # plot_datasets_scatter(x_dic, y_dic, col_dic, plot_dic, x_davids_fit, fitting_func_of_lam, datasets)

    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    # for k in datasets.keys():
    #     datasets[k]["plot_errorbar"] = True

    plot_datasets_scatter2(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)

    #complex_fic_data_vinf_module(datasets, fitting_coeffs_vinf_our, key_for_usable_dataset="fit")
def task_plot_ye_vs_fit():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md

    # -------------------------------------------

    datasets = {}
    # datasets["radice"] =  {'marker': '*', "ms": 40, "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Yeej_err, "label": r"Radice+2018", "color": "blue", "fit": True}
    # #datasets["vincent"] = {'marker': 'v', 'ms': 40, "models": vi.simulations, "data": vi, "err": vi.params.Yeej_err, "label": r"Vincent+2019", "color": "blue", "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    # datasets['our'] =     {'marker': 'o', 'ms': 40, "models": md.groups, "data": md, "err": "v_n", "label": r"This work", "color": "red", "fit": True}
    datasets['our'] = {"models": md.groups, "data": md, "fit": True, "color": None, "plot_errorbar": True, "err": "v_n"}
    datasets["radice"] = {"models": rd.simulations[rd.fiducial], "data": rd, "fit": True, "color": None, "plot_errorbar": True, "err": rd.params.Yeej_err}
    # datasets["kiuchi"] = {"models": ki.simulations[ki.mask_for_with_tov_data], "data": ki, "fit": True,"color":None, "plot_errorbar":False, "err": ki.params.vej_err}
    # datasets["vincent"] = {"models": vi.simulations, "data": vi, "fit": True, "color": None, "plot_errorbar": False, "err": vi.params.Yeej_err}

    for t in datasets.keys():
        datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["label"] = ourmd.datasets_labels[t]
        datasets[t]["ms"] = 40

    x_dic   = {"v_n": "Ye_ave-geo_fit", "err": None, "deferr": None, "mod": {}}
    y_dic   = {"v_n": "Ye_ave-geo",     "err": "ud", "deferr": 0.2,  "mod": {}}
    col_dic = {"v_n": "Lambda",         "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (5.5, 6.1),
        "left": 0.15, "bottom": 0.10, "top": 0.93, "right": 1.00, "hspace": 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "fit_panel": True,
        "plot_diagonal":True,
        "tight_layout": False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": True,  # "tab10",
        "xmin":0.1, "xmax":.21, "xscale": "linear",
        "ymin": 0.05, "ymax": .32, "yscale": "linear",
        "xlabel": r"$\langle Y_e \rangle _{\rm ej;fit}$",
        "ylabel": r"$Ye_{\rm ej}$",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_ye_fit.png",
        "legend": {"fancybox": False, "loc": 'upper right',
                   #"bbox_to_anchor": (0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 13,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        # "dpi": 128, "fontsize":12, "labelsize":12,
        "savepdf":True
    }

    # fit
    # from make_fit import complex_fic_data_ye_module, fitting_function_ye, fitting_coeffs_ye2

    fit_dic = {
        "func":fit_funcs.yeej_like_vej, "coeffs": fit_coefs.yeej_us_radice(),
        "xmin": 0.1, "xmax": .21, "xscale": "linear", "xlabel": r"$Ye_{\rm ej;fit}$",
        "ymin": -2.0, "ymax": 0.8, "yscale": "linear", "ylabel": r"$\Delta Ye_{\rm ej} / Ye_{\rm ej}$",
        "plot_zero":True
    }

    # complex_fic_data_ye_module(datasets, fitting_function_ye, fitting_coeffs_ye2(), "fit")#; exit(1)

    # Fit
    # from make_fit import fitting_function_mej
    # from make_fit import fitting_coeffs_mej_david # fitting_coeffs_mej
    # from make_fit import complex_fic_data_vinf_module
    # x_davids_fit = fitting_coeffs_mej_david()# fitting_coeffs_mej()  # radice()
    # fitting_func_of_lam = fitting_function_mej  # takes x_davids_fit, lam_array


    # # # plot_datasets_scatter(x_dic, y_dic, col_dic, plot_dic, x_davids_fit, fitting_func_of_lam, datasets)

    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    # for k in datasets.keys():
    #     datasets[k]["plot_errorbar"] = True

    plot_datasets_scatter2(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)

    #complex_fic_data_vinf_module(datasets, fitting_coeffs_vinf_our, key_for_usable_dataset="fit")

def task_plot_yeej_all():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki                  #
    from model_sets import models_sekiguchi2016 as se16         # [23] arxive:1603.01918 # no Mb
    from model_sets import models_sekiguchi2015 as se15         # [-]  arxive:1502.06660 # no Mb
    from model_sets import models_bauswein2013 as bs            # [20] arxive:1302.6530
    from model_sets import models_lehner2016 as lh              # [22] arxive:1603.00501
    from model_sets import models_hotokezaka2013 as hz          # [19] arxive:1212.0905
    from model_sets import models_dietrich_ujevic2016 as du
    from model_sets import models_dietrich2015 as di15          # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16          # [24] arxive:1607.06636

    # -------------------------------------------

    # fill_styule = ('full', 'left', 'right', 'bottom', 'top', 'none')
    from collections import OrderedDict
    datasets = OrderedDict() #
    #datasets["bauswein"] =  {'marker': 's', 'ms': 20, "color": "slategray", "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "fit": False}
    #datasets["hotokezaka"] ={'marker': '>', 'ms': 20, "color": "gray", "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "fit": False}
    #datasets["dietrich15"]= {'marker': 'd', "ms": 20, "color": "gray", "models": di15.simulations[di15.mask_for_with_sr], "data": di15, "err": di15.params.Mej_err, "label": r"Dietrich+2015", "fit": False}
    datasets["sekiguchi15"]={'marker': 'p', "ms": 60, "color": "black", "models": se15.simulations[se15.mask_for_with_sr], "data": se15, "err": se15.params.Mej_err, "label": r"Sekiguchi+2015",  "fit": False}
    #datasets["dietrich16"]= {'marker': 'D', "ms": 20, "color": "gray", "models": di16.simulations[di16.mask_for_with_sr], "data": di16, "err": di16.params.Mej_err, "label": r"Dietrich+2016", "fit": False}
    datasets["sekiguchi16"]={'marker': 'h', "ms": 60, "color": "black", "models": se16.simulations[se16.mask_for_with_sr], "data": se16, "err": se16.params.Mej_err, "label": r"Sekiguchi+2016", "fit": False}
    #datasets["lehner"] =    {'marker': 'P', 'ms': 20, "color": "orange", "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016",  "fit": False}
    datasets["radice"] =    {'marker': '*', "ms": 60, "color": "green", "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018",  "fit": False}
    #datasets["kiuchi"] =    {'marker': "X", "ms": 20, "color": "gray", "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019",  "fit": False}
    datasets["vincent"] =   {'marker': 'v', 'ms': 60, "color": "red", "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019",  "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets['our'] =       {'marker': 'o', 'ms': 60, "color": "blue", "models": md.groups, "data": md, "err": "v_n", "label": r"This work",  "fit": False}
    # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}

    for t in datasets.keys():
        datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["fill_style"] = "none"


    x_dic    = {"v_n": "q", "err": None, "deferr": None, "mod": {}}
    y_dic    = {"v_n": "Ye_ave-geo", "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (6.0, 5.5),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "fit_panel": False,
        "plot_diagonal":False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": False,  # "tab10",
        "xmin": 0.95, "xmax": 2.0, "xscale": "linear",
        "ymin": 0.01, "ymax": 0.35, "yscale": "linear",
        "xlabel": "$M_1/M_2$",#r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ylabel": r"$\langle Ye_{\rm ej} \rangle$",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_q_yeej.png",
        "legend": {"fancybox": False, "loc": 'upper right',
                   #"bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "savepdf":True,
        "tight_layout":True
    }

    # ---------------- fit ------------------
    from make_fit import fitting_function_mej, fitting_coeffs_mej_our
    fit_dic = {}
    #     {
    #     "func":fitting_function_mej, "coeffs": fitting_coeffs_mej_our(),
    #     "xmin": 0, "xmax": 7.5, "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
    #     "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
    #     "plot_zero":True
    # }

    # Fit
    from make_fit import fitting_function_mej
    #from make_fit import fitting_coeffs_mej_david, fitting_coeffs_mej_our
    #from make_fit import complex_fic_data_mej_module
    #complex_fic_data_mej_module(datasets, fitting_coeffs_mej_our(), key_for_usable_dataset="fit")


    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    for k in datasets.keys():
        datasets[k]["plot_errorbar"] = False

    plot_datasets_scatter2(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)
def task_plot_yeej():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki                  #
    from model_sets import models_sekiguchi2016 as se16         # [23] arxive:1603.01918 # no Mb
    from model_sets import models_sekiguchi2015 as se15         # [-]  arxive:1502.06660 # no Mb
    from model_sets import models_bauswein2013 as bs            # [20] arxive:1302.6530
    from model_sets import models_lehner2016 as lh              # [22] arxive:1603.00501
    from model_sets import models_hotokezaka2013 as hz          # [19] arxive:1212.0905
    from model_sets import models_dietrich_ujevic2016 as du
    from model_sets import models_dietrich2015 as di15          # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16          # [24] arxive:1607.06636

    # -------------------------------------------
    from collections import OrderedDict
    datasets = OrderedDict() #
    # datasets["bauswein"] =  {'marker': 's', 'ms': 60, "color": "slategray", "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "fit": False}
    # datasets["hotokezaka"] ={'marker': '>', 'ms': 60, "color": "gray", "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "fit": False}
    # datasets["dietrich15"]= {'marker': 'd', "ms": 60, "color": "gray", "models": di15.simulations[di15.mask_for_with_sr], "data": di15, "err": di15.params.Mej_err, "label": r"Dietrich+2015", "fit": False}
    # datasets["sekiguchi15"]={'marker': 'p', "ms": 60, "color": "black", "models": se15.simulations[se15.mask_for_with_sr], "data": se15, "err": se15.params.Mej_err, "label": r"Sekiguchi+2015",  "fit": False}
    # datasets["dietrich16"]= {'marker': 'D', "ms": 60, "color": "gray", "models": di16.simulations[di16.mask_for_with_sr], "data": di16, "err": di16.params.Mej_err, "label": r"Dietrich+2016", "fit": False}
    # datasets["sekiguchi16"]={'marker': 'h', "ms": 60, "color": "black", "models": se16.simulations[se16.mask_for_with_sr], "data": se16, "err": se16.params.Mej_err, "label": r"Sekiguchi+2016", "fit": False}
    # datasets["lehner"] =    {'marker': 'P', 'ms': 60, "color": "orange", "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016",  "fit": False}
    # datasets["radice"] =    {'marker': '*', "ms": 60, "color": "green", "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018",  "fit": False}
    # datasets["kiuchi"] =    {'marker': "X", "ms": 60, "color": "gray", "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019",  "fit": False}
    # datasets["vincent"] =   {'marker': 'v', 'ms': 60, "color": "red", "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019",  "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets['our'] =       {'marker': 'o', 'ms': 60, "color": "#EOS", "models": md.groups, "data": md, "err": "v_n", "label": r"#EOS",  "fit": True}
    #### datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}

    for t in datasets.keys():
        datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["fill_style"] = "none"
        datasets[t]["edgecolor"] = "#EOS"
        datasets[t]["facecolor"] = "none"
        datasets[t]["marker"] = "d" #EOS"
        datasets[t]["plot_errorbar"] = True
        datasets[t]["plot_xerrorbar"] = False

    x_dic    = {"v_n": "q", "err": None, "deferr": None, "mod": {}}
    y_dic    = {"v_n": "Ye_ave-geo", "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (5.5, 6.1),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "fit_panel": True,
        "plot_diagonal":False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": False,  # "tab10",
        "xmin": 0.95, "xmax": 2.0, "xscale": "linear",
        "ymin": 0.01, "ymax": 0.30, "yscale": "linear",
        "xlabel": "$M_1/M_2$",#r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ylabel": r"$\langle Y_{e\: \rm ej} \rangle$",
        #"cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_q_yeej_our_poly22.png",
        "legend": {"fancybox": False, "loc": 'upper right',
                   #"bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":1.0,
        "hline": {},#{"y":3.442 * 1.e-3, "color":'blue', "linestyle":"dashed", "label":"Mean"},
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "savepdf":True,
        "tight_layout":False,
    }

    # ---------------- fit ------------------
    #from make_fit import fitting_function_mej, fitting_coeffs_mej_our
    fit_dic =  {
        #"func":fit_funcs.yeej_like_vej, "coeffs": np.array([0.268, 0.588, -4.282]),
        "func": fit_funcs.poly_2_qLambda, "coeffs": np.array([-6.607e-02, 0.318, 6.084e-04, -1.551e-01, -2.055e-04, -2.896e-07]),
        "xmin": 0.95, "xmax": 2.0, "xscale": "linear", "xlabel": "$q$",#r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -1.0, "ymax": 1.0, "yscale": "linear", "ylabel": r"$\Delta Y_{e\: \rm ej} / \langle Y_{e\: \rm ej} \rangle$",
        "plot_zero":True
    }

    # Fit
    from make_fit import fitting_function_mej
    #from make_fit import fitting_coeffs_mej_david, fitting_coeffs_mej_our
    #from make_fit import complex_fic_data_mej_module
    #complex_fic_data_mej_module(datasets, fitting_coeffs_mej_our(), key_for_usable_dataset="fit")


    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    for k in datasets.keys():
        datasets[k]["plot_errorbar"] = True

    plot_datasets_scatter2(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)
def task_plot_yeej_cc():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki                  #
    from model_sets import models_sekiguchi2016 as se16         # [23] arxive:1603.01918 # no Mb
    from model_sets import models_sekiguchi2015 as se15         # [-]  arxive:1502.06660 # no Mb
    from model_sets import models_bauswein2013 as bs            # [20] arxive:1302.6530
    from model_sets import models_lehner2016 as lh              # [22] arxive:1603.00501
    from model_sets import models_hotokezaka2013 as hz          # [19] arxive:1212.0905
    from model_sets import models_dietrich_ujevic2016 as du
    from model_sets import models_dietrich2015 as di15          # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16          # [24] arxive:1607.06636

    # -------------------------------------------
    from collections import OrderedDict
    datasets = OrderedDict() #
    # datasets["bauswein"] =  {'marker': 's', 'ms': 60, "color": "slategray", "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "fit": False}
    # datasets["hotokezaka"] ={'marker': '>', 'ms': 60, "color": "gray", "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "fit": False}
    # datasets["dietrich15"]= {'marker': 'd', "ms": 60, "color": "gray", "models": di15.simulations[di15.mask_for_with_sr], "data": di15, "err": di15.params.Mej_err, "label": r"Dietrich+2015", "fit": False}
    # datasets["sekiguchi15"]={'marker': 'p', "ms": 60, "color": "black", "models": se15.simulations[se15.mask_for_with_sr], "data": se15, "err": se15.params.Mej_err, "label": r"Sekiguchi+2015",  "fit": False}
    # datasets["dietrich16"]= {'marker': 'D', "ms": 60, "color": "gray", "models": di16.simulations[di16.mask_for_with_sr], "data": di16, "err": di16.params.Mej_err, "label": r"Dietrich+2016", "fit": False}
    # datasets["sekiguchi16"]={'marker': 'h', "ms": 60, "color": "black", "models": se16.simulations[se16.mask_for_with_sr], "data": se16, "err": se16.params.Mej_err, "label": r"Sekiguchi+2016", "fit": False}
    # datasets["lehner"] =    {'marker': 'P', 'ms': 60, "color": "orange", "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016",  "fit": False}
    # datasets["radice"] =    {'marker': '*', "ms": 60, "color": "green", "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018",  "fit": False}
    # datasets["kiuchi"] =    {'marker': "X", "ms": 60, "color": "gray", "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019",  "fit": False}
    # datasets["vincent"] =   {'marker': 'v', 'ms': 60, "color": "red", "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019",  "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets['our'] =       {'marker': 'o', 'ms': 60, "color": "#EOS", "models": md.groups, "data": md, "err": "v_n", "label": r"#EOS",  "fit": True}
    #### datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}

    for t in datasets.keys():
        #datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["fill_style"] = "full"#"none"
        datasets[t]["edgecolor"] = "none"#"#EOS"
        #datasets[t]["facecolor"] = "#cb"#"none"
        datasets[t]["marker"] = "#EOS"#"d" # #EOS
        datasets[t]["plot_errorbar"] = True
        datasets[t]["plot_xerrorbar"] = False

    x_dic    = {"v_n": "q", "err": None, "deferr": None, "mod": {}}
    y_dic    = {"v_n": "Ye_ave-geo", "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (5.5, 6.1),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "fit_panel": True,
        "plot_diagonal":False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": True,  # "tab10",
        "xmin": 0.95, "xmax": 2.0, "xscale": "linear",
        "ymin": 0.01, "ymax": 0.30, "yscale": "linear",
        "xlabel": "$M_1/M_2$",#r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ylabel": r"$\langle Y_{e\: \rm ej} \rangle$",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_q_yeej_our_poly22_cc.png",
        "legend": {"fancybox": False, "loc": 'upper right',
                   #"bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":1.0,
        "hline": {},#{"y":3.442 * 1.e-3, "color":'blue', "linestyle":"dashed", "label":"Mean"},
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "add_marker": [
            {"ms": 7, "color": "gray", "linestyle": "none", "label": "DD2", "marker": ourmd.eos_dic_marker["DD2"]},
            {"ms": 7, "color": "gray", "linestyle": "none", "label": "LS220", "marker": ourmd.eos_dic_marker["LS220"]},
            {"ms": 7, "color": "gray", "linestyle": "none", "label": "SFHo", "marker": ourmd.eos_dic_marker["SFHo"]},
            {"ms": 7, "color": "gray", "linestyle": "none", "label": "SLy4", "marker": ourmd.eos_dic_marker["SLy4"]},
            {"ms": 7, "color": "gray", "linestyle": "none", "label": "BLh", "marker": ourmd.eos_dic_marker["BLh"]}
        ],
        "savepdf":True,
        "tight_layout":False,
    }

    # ---------------- fit ------------------
    #from make_fit import fitting_function_mej, fitting_coeffs_mej_our
    fit_dic =  {
        #"func":fit_funcs.yeej_like_vej, "coeffs": np.array([0.268, 0.588, -4.282]),
        "func": fit_funcs.poly_2_qLambda, "coeffs": np.array([-6.607e-02, 0.318, 6.084e-04, -1.551e-01, -2.055e-04, -2.896e-07]),
        "xmin": 0.95, "xmax": 2.0, "xscale": "linear", "xlabel": "$q$",#r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -1.0, "ymax": 1.0, "yscale": "linear", "ylabel": r"$\Delta \langle Y_{e\: \rm ej} \rangle / \langle Y_{e\: \rm ej} \rangle$",
        "plot_zero":True
    }

    # Fit
    from make_fit import fitting_function_mej
    #from make_fit import fitting_coeffs_mej_david, fitting_coeffs_mej_our
    #from make_fit import complex_fic_data_mej_module
    #complex_fic_data_mej_module(datasets, fitting_coeffs_mej_our(), key_for_usable_dataset="fit")


    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    for k in datasets.keys():
        datasets[k]["plot_errorbar"] = True

    plot_datasets_scatter3(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)
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
    fit_dics["poly2"] = {
        "func": fit_funcs.poly_2_qLambda,
        "method": "delta",
        "coeffs": np.array([-4.48e-01, 8.60e-01, 5.51e-04, -3.12e-01, -3.08e-04, -1.50e-07]),
        # np.array([2.549e-03, 2.394e-03, -3.005e-05, -3.376e-03, 3.826e-05, -1.149e-08]),
        "xmin": 0.1, "xmax": .25, "xscale": "linear", "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
        "ymin": -0.3, "ymax": 0.3, "yscale": "linear", "ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
        "plot_zero": True
    }
    fit_dics["Eq.11"] = {
        "method": "delta",
        "func": fit_funcs.yeej_like_vej, "coeffs": np.array([0.137, 0.361, -4.160]),
        "xmin": 0.1, "xmax": .25, "xscale": "linear", "xlabel": r"$Y_{e\: \rm{ej;fit}}$",
        "ymin": -0.3, "ymax": 0.3, "yscale": "linear", "ylabel": r"$\Delta Y_{e\: \rm ej} / Y_{e\: \rm ej}$",
        "plot_zero": True
    }
    fit_dics["poly1"] = {
        "method": "delta",
        "func": fit_funcs.poly_2_Lambda, "coeffs": np.array([1.10e-01, 1.75e-04, -1.08e-07]),
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

''' --- other --- '''

def task_plot_mej_vs_theta_all():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki                  #
    from model_sets import models_sekiguchi2016 as se16         # [23] arxive:1603.01918 # no Mb
    from model_sets import models_sekiguchi2015 as se15         # [-]  arxive:1502.06660 # no Mb
    from model_sets import models_bauswein2013 as bs            # [20] arxive:1302.6530
    from model_sets import models_lehner2016 as lh              # [22] arxive:1603.00501
    from model_sets import models_hotokezaka2013 as hz          # [19] arxive:1212.0905
    from model_sets import models_dietrich_ujevic2016 as du
    from model_sets import models_dietrich2015 as di15          # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16          # [24] arxive:1607.06636



    # -------------------------------------------
    from collections import OrderedDict
    datasets = OrderedDict() #
    # datasets["bauswein"] =  {'marker': 's', 'ms': 60, "color": "slategray", "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "fit": False}
    # datasets["hotokezaka"] ={'marker': '>', 'ms': 60, "color": "gray", "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "fit": False}
    # datasets["dietrich15"]= {'marker': 'd', "ms": 60, "color": "gray", "models": di15.simulations[di15.mask_for_with_sr], "data": di15, "err": di15.params.Mej_err, "label": r"Dietrich+2015", "fit": False}
    #datasets["sekiguchi15"]={'marker': 'p', "ms": 20, "color": "black", "models": se15.simulations[se15.mask_for_with_sr], "data": se15, "err": se15.params.Mej_err, "label": r"Sekiguchi+2015",  "fit": False}
    # datasets["dietrich16"]= {'marker': 'D', "ms": 60, "color": "gray", "models": di16.simulations[di16.mask_for_with_sr], "data": di16, "err": di16.params.Mej_err, "label": r"Dietrich+2016", "fit": False}
    # datasets["sekiguchi16"]={'marker': 'h', "ms": 60, "color": "black", "models": se16.simulations[se16.mask_for_with_sr], "data": se16, "err": se16.params.Mej_err, "label": r"Sekiguchi+2016", "fit": False}
    # datasets["lehner"] =    {'marker': 'P', 'ms': 60, "color": "orange", "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016",  "fit": False}
    # datasets["radice"] =    {'marker': '*', "ms": 60, "color": "green", "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018",  "fit": False}
    datasets["radiceLK"] =    {'marker': '*', "ms": 60, "color": "lime", "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018",  "fit": False}
    datasets["radiceM0"] =    {'marker': '*', "ms": 60, "color": "green", "models": rd.simulations[rd.with_m0], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018",  "fit": False}
    #datasets["kiuchi"] =    {'marker': "X", "ms": 20, "color": "gray", "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019",  "fit": False}
    # datasets["vincent"] =   {'marker': 'v', 'ms': 60, "color": "red", "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019",  "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets['reference'] = {'marker': 'o', 'ms': 60, "color": "blue", "models": md.groups, "data": md, "err": "v_n", "label": r"Nedora+2020",  "fit": False} # This work
    # datasets['our_total'] = {'marker': '+', 'ms': 60,
    #                          "v_n_x": "vel_inf_ave-tot", "v_n_y":"Mej_tot-tot",
    #                          "color": "blue", "models": md.groups[(md.groups.tend_wind > 0.030) | (md.groups["group"] == "BLh_M10651772_M0_LK")],
    #                          "data": md, "err": "v_n", "label": r"This work", "fit": False}
    # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}

    for t in datasets.keys():
        datasets[t]["label"] = ourmd.datasets_labels[t]
        datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["fill_style"] = "none"
        datasets[t]["plot_errorbar"] = False
        datasets[t]["plot_xerrorbar"] = False
        datasets[t]["facecolor"] = "none"
        datasets[t]["edgecolor"] = datasets[t]["color"]


    y_dic    = {"v_n": "Mej_tot-geo",     "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    x_dic    = {"v_n": "theta_rms-geo", "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (6.0, 5.5),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 18, "labelsize": 18,
        "fit_panel": False,
        "plot_diagonal":False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": False,  # "tab10",
        "xmin": 0, "xmax": 50, "xscale": "linear",
        "ymin": 1e-4, "ymax": 2e-2, "yscale": "log",
        "ylabel": r"$M_{\rm ej}$ $[M_{\odot}]$",#$[10^{-3}M_{\odot}]$",
        "xlabel": r"$\theta_{\rm RMS}$",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_mej_theta_all2.png",
        "legend": {"fancybox": False, "loc": 'upper right',
                   #"bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 12,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        # "patch":
        #     [{"data":{"x":0.27,"y":0.016,"xerr":(0.2,0.3),"yerr":(1e-2,2e-2)},
        #      "plot":{"facecolor":"blue", "alpha":0.4, "edgecolor":"none", "label":"Blue kN"}},
        #      {"data": {"x": 0.1, "y": 0.05, "xerr": (0.07, 0.14), "yerr": (4e-2, 6e-2)},
        #       "plot": {"facecolor": "red", "alpha": 0.4, "edgecolor": "none", "label": "Red kN"}},
        #      # {"data": {"x": None, "y": None, "xerr": (0.2, 0.23), "yerr": (1e-2, 5e-3)},
        #      #  "plot": {"facecolor": "none", "alpha": 0.4, "edgecolor": "red", "label": "Dyn. kN [P]"}},
        #      # {"data": {"x": None, "y": None, "xerr": (0.066, 0.068), "yerr": (0.001*0.08, 0.2*0.12)},
        #      #  "plot": {"facecolor": "none", "alpha": 0.4, "edgecolor": "blue", "label": "Wind kN [P]"}},
        #      # {"data": {"x": None, "y": None, "xerr": (0.027, 0.04), "yerr": (0.4 * 0.08, 0.2*0.1)},
        #      #  "plot": {"facecolor": "none", "alpha": 0.4, "edgecolor": "purple", "label": "Sec. kN [P]"}},
        #      ],
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "savepdf":True,
        "tight_layout":True
    }

    # ---------------- fit ------------------
    from make_fit import fitting_function_mej, fitting_coeffs_mej_our
    fit_dic = {}
    #     {
    #     "func":fitting_function_mej, "coeffs": fitting_coeffs_mej_our(),
    #     "xmin": 0, "xmax": 7.5, "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
    #     "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
    #     "plot_zero":True
    # }

    # Fit
    from make_fit import fitting_function_mej
    #from make_fit import fitting_coeffs_mej_david, fitting_coeffs_mej_our
    #from make_fit import complex_fic_data_mej_module
    #complex_fic_data_mej_module(datasets, fitting_coeffs_mej_our(), key_for_usable_dataset="fit")


    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    for k in datasets.keys():
        datasets[k]["plot_errorbar"] = False

    plot_datasets_scatter2(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)
def task_plot_vej_vs_theta():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki                  #
    from model_sets import models_sekiguchi2016 as se16         # [23] arxive:1603.01918 # no Mb
    from model_sets import models_sekiguchi2015 as se15         # [-]  arxive:1502.06660 # no Mb
    from model_sets import models_bauswein2013 as bs            # [20] arxive:1302.6530
    from model_sets import models_lehner2016 as lh              # [22] arxive:1603.00501
    from model_sets import models_hotokezaka2013 as hz          # [19] arxive:1212.0905
    from model_sets import models_dietrich_ujevic2016 as du
    from model_sets import models_dietrich2015 as di15          # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16          # [24] arxive:1607.06636

    # -------------------------------------------
    from collections import OrderedDict
    datasets = OrderedDict() #
    #datasets["bauswein"] =  {'marker': 's', 'ms': 20, "color": "slategray", "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "fit": False}
    #datasets["hotokezaka"] ={'marker': '>', 'ms': 20, "color": "gray", "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "fit": False}
    #datasets["dietrich15"]= {'marker': 'd', "ms": 20, "color": "gray", "models": di15.simulations[di15.mask_for_with_sr], "data": di15, "err": di15.params.Mej_err, "label": r"Dietrich+2015", "fit": False}
    #datasets["sekiguchi15"]={'marker': 'p', "ms": 20, "color": "black", "models": se15.simulations[se15.mask_for_with_sr], "data": se15, "err": se15.params.Mej_err, "label": r"Sekiguchi+2015",  "fit": False}
    #datasets["dietrich16"]= {'marker': 'D', "ms": 20, "color": "gray", "models": di16.simulations[di16.mask_for_with_sr], "data": di16, "err": di16.params.Mej_err, "label": r"Dietrich+2016", "fit": False}
    #datasets["sekiguchi16"]={'marker': 'h', "ms": 60, "color": "black", "models": se16.simulations[se16.mask_for_with_sr], "data": se16, "err": se16.params.Mej_err, "label": r"Sekiguchi+2016", "fit": False}
    #datasets["lehner"] =    {'marker': 'P', 'ms': 20, "color": "orange", "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016",  "fit": False}
    datasets["radiceLK"] =    {'marker': '*', "ms": 60, "color": "lime", "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018",  "fit": False}
    datasets["radiceM0"] =    {'marker': '*', "ms": 60, "color": "green", "models": rd.simulations[rd.with_m0], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018",  "fit": False}
    #datasets["kiuchi"] =    {'marker': "X", "ms": 20, "color": "gray", "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019",  "fit": False}
    # datasets["vincent"] =   {'marker': 'v', 'ms': 60, "color": "red", "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019",  "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets['reference'] = {'marker': 'o', 'ms': 60, "color": "blue", "models": md.groups, "data": md, "err": "v_n", "label": r"This work",  "fit": False}
    # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}

    for t in datasets.keys():
        datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["label"] = ourmd.datasets_labels[t]
        datasets[t]["fill_style"] = "none"
        datasets[t]["fill_style"] = "none"
        datasets[t]["plot_errorbar"] = False
        datasets[t]["plot_xerrorbar"] = False
        datasets[t]["facecolor"] = "none"
        datasets[t]["edgecolor"] = datasets[t]["color"]

    y_dic    = {"v_n": "vel_inf_ave-geo", "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    x_dic    = {"v_n": "theta_rms-geo",      "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (6.0, 5.5),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 18, "labelsize": 18,
        "fit_panel": False,
        "plot_diagonal":False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": False,  # "tab10",
        "xmin": 0, "xmax": 50, "xscale": "linear",
        "ymin": 0.1, "ymax": 0.35, "yscale": "linear",
        "ylabel": r"$\langle \upsilon_{\rm ej} \rangle$ [c]",
        "xlabel": r"$\theta_{\rm RMS}$",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_vej_theta_all2.png",
        "legend": {"fancybox": False, "loc": 'upper right',
                   #"bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "savepdf":True,
        "tight_layout":True
    }

    # ---------------- fit ------------------
    from make_fit import fitting_function_mej, fitting_coeffs_mej_our
    fit_dic = {}
    #     {
    #     "func":fitting_function_mej, "coeffs": fitting_coeffs_mej_our(),
    #     "xmin": 0, "xmax": 7.5, "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
    #     "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
    #     "plot_zero":True
    # }

    # Fit
    from make_fit import fitting_function_mej
    #from make_fit import fitting_coeffs_mej_david, fitting_coeffs_mej_our
    #from make_fit import complex_fic_data_mej_module
    #complex_fic_data_mej_module(datasets, fitting_coeffs_mej_our(), key_for_usable_dataset="fit")


    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    for k in datasets.keys():
        datasets[k]["plot_errorbar"] = False

    plot_datasets_scatter2(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)
def task_plot_theta_vs_ye():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki                  #
    from model_sets import models_sekiguchi2016 as se16         # [23] arxive:1603.01918 # no Mb
    from model_sets import models_sekiguchi2015 as se15         # [-]  arxive:1502.06660 # no Mb
    from model_sets import models_bauswein2013 as bs            # [20] arxive:1302.6530
    from model_sets import models_lehner2016 as lh              # [22] arxive:1603.00501
    from model_sets import models_hotokezaka2013 as hz          # [19] arxive:1212.0905
    from model_sets import models_dietrich_ujevic2016 as du
    from model_sets import models_dietrich2015 as di15          # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16          # [24] arxive:1607.06636

    # -------------------------------------------
    from collections import OrderedDict
    datasets = OrderedDict() #
    #datasets["bauswein"] =  {'marker': 's', 'ms': 20, "color": "slategray", "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "fit": False}
    #datasets["hotokezaka"] ={'marker': '>', 'ms': 20, "color": "gray", "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "fit": False}
    #datasets["dietrich15"]= {'marker': 'd', "ms": 20, "color": "gray", "models": di15.simulations[di15.mask_for_with_sr], "data": di15, "err": di15.params.Mej_err, "label": r"Dietrich+2015", "fit": False}
    #datasets["sekiguchi15"]={'marker': 'p', "ms": 20, "color": "black", "models": se15.simulations[se15.mask_for_with_sr], "data": se15, "err": se15.params.Mej_err, "label": r"Sekiguchi+2015",  "fit": False}
    #datasets["dietrich16"]= {'marker': 'D', "ms": 20, "color": "gray", "models": di16.simulations[di16.mask_for_with_sr], "data": di16, "err": di16.params.Mej_err, "label": r"Dietrich+2016", "fit": False}
    # datasets["sekiguchi16"]={'marker': 'h', "ms": 60, "color": "black", "models": se16.simulations[se16.mask_for_with_sr], "data": se16, "err": se16.params.Mej_err, "label": r"Sekiguchi+2016", "fit": False}
    #datasets["lehner"] =    {'marker': 'P', 'ms': 20, "color": "orange", "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016",  "fit": False}
    datasets["radiceLK"] =    {'marker': '*', "ms": 60, "color": "lime", "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018",  "fit": False}
    datasets["radiceM0"] =    {'marker': '*', "ms": 60, "color": "green", "models": rd.simulations[rd.with_m0], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018",  "fit": False}
    #datasets["kiuchi"] =    {'marker': "X", "ms": 20, "color": "gray", "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019",  "fit": False}
    # datasets["vincent"] =   {'marker': 'v', 'ms': 60, "color": "red", "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019",  "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets['reference'] = {'marker': 'o', 'ms': 60, "color": "blue", "models": md.groups, "data": md, "err": "v_n", "label": r"This work",  "fit": False}
    # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}

    for t in datasets.keys():
        datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["label"] = ourmd.datasets_labels[t]
        datasets[t]["fill_style"] = "none"
        datasets[t]["fill_style"] = "none"
        datasets[t]["plot_errorbar"] = False
        datasets[t]["plot_xerrorbar"] = False
        datasets[t]["facecolor"] = "none"
        datasets[t]["edgecolor"] = datasets[t]["color"]

    y_dic    = {"v_n": "theta_rms-geo", "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    x_dic    = {"v_n": "Ye_ave-geo",      "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (6.0, 5.5),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 18, "labelsize": 18,
        "fit_panel": False,
        "plot_diagonal":False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": False,  # "tab10",
        "xmin": 0.01, "xmax": 0.30, "xscale": "linear",
        "ymin": 0, "ymax": 50, "yscale": "linear",
        "ylabel": r"$\theta_{\rm RMS}$",
        "xlabel": r"$\langle Y_{e;\:\rm ej} \rangle$",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_theta_yeej_all2.png",
        "legend": {"fancybox": False, "loc": 'upper left',
                   #"bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "savepdf":True,
        "tight_layout":True
    }

    # ---------------- fit ------------------
    from make_fit import fitting_function_mej, fitting_coeffs_mej_our
    fit_dic = {}
    #     {
    #     "func":fitting_function_mej, "coeffs": fitting_coeffs_mej_our(),
    #     "xmin": 0, "xmax": 7.5, "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
    #     "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
    #     "plot_zero":True
    # }

    # Fit
    from make_fit import fitting_function_mej
    #from make_fit import fitting_coeffs_mej_david, fitting_coeffs_mej_our
    #from make_fit import complex_fic_data_mej_module
    #complex_fic_data_mej_module(datasets, fitting_coeffs_mej_our(), key_for_usable_dataset="fit")


    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    for k in datasets.keys():
        datasets[k]["plot_errorbar"] = False

    plot_datasets_scatter2(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)

def task_plot_mej_vs_vej_all():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki                  #
    from model_sets import models_sekiguchi2016 as se16         # [23] arxive:1603.01918 # no Mb
    from model_sets import models_sekiguchi2015 as se15         # [-]  arxive:1502.06660 # no Mb
    from model_sets import models_bauswein2013 as bs            # [20] arxive:1302.6530
    from model_sets import models_lehner2016 as lh              # [22] arxive:1603.00501
    from model_sets import models_hotokezaka2013 as hz          # [19] arxive:1212.0905
    from model_sets import models_dietrich_ujevic2016 as du
    from model_sets import models_dietrich2015 as di15          # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16          # [24] arxive:1607.06636



    # -------------------------------------------
    from collections import OrderedDict
    datasets = OrderedDict() #
    datasets["bauswein"] =  {'marker': 's', 'ms': 60, "color": "slategray", "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "fit": False}
    datasets["hotokezaka"] ={'marker': '>', 'ms': 60, "color": "gray", "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "fit": False}
    datasets["dietrich15"]= {'marker': 'd', "ms": 60, "color": "gray", "models": di15.simulations[di15.mask_for_with_sr], "data": di15, "err": di15.params.Mej_err, "label": r"Dietrich+2015", "fit": False}
    #datasets["sekiguchi15"]={'marker': 'p', "ms": 20, "color": "black", "models": se15.simulations[se15.mask_for_with_sr], "data": se15, "err": se15.params.Mej_err, "label": r"Sekiguchi+2015",  "fit": False}
    datasets["dietrich16"]= {'marker': 'D', "ms": 60, "color": "gray", "models": di16.simulations[di16.mask_for_with_sr], "data": di16, "err": di16.params.Mej_err, "label": r"Dietrich+2016", "fit": False}
    datasets["sekiguchi16"]={'marker': 'h', "ms": 60, "color": "black", "models": se16.simulations[se16.mask_for_with_sr], "data": se16, "err": se16.params.Mej_err, "label": r"Sekiguchi+2016", "fit": False}
    datasets["lehner"] =    {'marker': 'P', 'ms': 60, "color": "orange", "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016",  "fit": False}
    # datasets["radice"] =    {'marker': '*', "ms": 60, "color": "green", "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018",  "fit": False}
    datasets["radiceLK"] =    {'marker': '*', "ms": 60, "color": "lime", "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018",  "fit": False}
    datasets["radiceM0"] =    {'marker': '*', "ms": 60, "color": "green", "models": rd.simulations[rd.with_m0], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018",  "fit": False}
    #datasets["kiuchi"] =    {'marker': "X", "ms": 20, "color": "gray", "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019",  "fit": False}
    datasets["vincent"] =   {'marker': 'v', 'ms': 60, "color": "red", "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019",  "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets['reference'] = {'marker': 'o', 'ms': 60, "color": "blue", "models": md.groups, "data": md, "err": "v_n", "label": r"Nedora+2020",  "fit": False} # This work
    # datasets['our_total'] = {'marker': '+', 'ms': 60,
    #                          "v_n_x": "vel_inf_ave-tot", "v_n_y":"Mej_tot-tot",
    #                          "color": "blue", "models": md.groups[(md.groups.tend_wind > 0.030) | (md.groups["group"] == "BLh_M10651772_M0_LK")],
    #                          "data": md, "err": "v_n", "label": r"This work", "fit": False}
    # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}

    for t in datasets.keys():
        datasets[t]["label"] = ourmd.datasets_labels[t]
        datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["fill_style"] = "none"
        datasets[t]["plot_errorbar"] = False
        datasets[t]["plot_xerrorbar"] = False
        datasets[t]["facecolor"] = "none"
        datasets[t]["edgecolor"] = datasets[t]["color"]


    y_dic    = {"v_n": "Mej_tot-geo",     "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    x_dic    = {"v_n": "vel_inf_ave-geo", "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (6.0, 5.5),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 18, "labelsize": 18,
        "fit_panel": False,
        "plot_diagonal":False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": False,  # "tab10",
        "xmin": 0.025, "xmax": 0.5, "xscale": "linear",
        "ymin": 1e-4, "ymax": 2e-0, "yscale": "log",
        "ylabel": r"$M_{\rm ej}$ $[M_{\odot}]$",#$[10^{-3}M_{\odot}]$",
        "xlabel": r"$\langle \upsilon_{\rm ej} \rangle$ [c]",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_mej_vej_all2.png",
        "legend": {"fancybox": False, "loc": 'upper right',
                   #"bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 12,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        "patch":
            [{"data":{"x":0.27,"y":0.016,"xerr":(0.2,0.3),"yerr":(1e-2,2e-2)},
             "plot":{"facecolor":"blue", "alpha":0.4, "edgecolor":"none", "label":"Blue kN"}},
             {"data": {"x": 0.1, "y": 0.05, "xerr": (0.07, 0.14), "yerr": (4e-2, 6e-2)},
              "plot": {"facecolor": "red", "alpha": 0.4, "edgecolor": "none", "label": "Red kN"}},
             # {"data": {"x": None, "y": None, "xerr": (0.2, 0.23), "yerr": (1e-2, 5e-3)},
             #  "plot": {"facecolor": "none", "alpha": 0.4, "edgecolor": "red", "label": "Dyn. kN [P]"}},
             # {"data": {"x": None, "y": None, "xerr": (0.066, 0.068), "yerr": (0.001*0.08, 0.2*0.12)},
             #  "plot": {"facecolor": "none", "alpha": 0.4, "edgecolor": "blue", "label": "Wind kN [P]"}},
             # {"data": {"x": None, "y": None, "xerr": (0.027, 0.04), "yerr": (0.4 * 0.08, 0.2*0.1)},
             #  "plot": {"facecolor": "none", "alpha": 0.4, "edgecolor": "purple", "label": "Sec. kN [P]"}},
             ],
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "savepdf":True,
        "tight_layout":True
    }

    # ---------------- fit ------------------
    from make_fit import fitting_function_mej, fitting_coeffs_mej_our
    fit_dic = {}
    #     {
    #     "func":fitting_function_mej, "coeffs": fitting_coeffs_mej_our(),
    #     "xmin": 0, "xmax": 7.5, "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
    #     "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
    #     "plot_zero":True
    # }

    # Fit
    from make_fit import fitting_function_mej
    #from make_fit import fitting_coeffs_mej_david, fitting_coeffs_mej_our
    #from make_fit import complex_fic_data_mej_module
    #complex_fic_data_mej_module(datasets, fitting_coeffs_mej_our(), key_for_usable_dataset="fit")


    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    for k in datasets.keys():
        datasets[k]["plot_errorbar"] = False

    plot_datasets_scatter2(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)
def task_plot_mej_vs_vej():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki                  #
    from model_sets import models_sekiguchi2016 as se16         # [23] arxive:1603.01918 # no Mb
    from model_sets import models_sekiguchi2015 as se15         # [-]  arxive:1502.06660 # no Mb
    from model_sets import models_bauswein2013 as bs            # [20] arxive:1302.6530
    from model_sets import models_lehner2016 as lh              # [22] arxive:1603.00501
    from model_sets import models_hotokezaka2013 as hz          # [19] arxive:1212.0905
    from model_sets import models_dietrich_ujevic2016 as du
    from model_sets import models_dietrich2015 as di15          # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16          # [24] arxive:1607.06636

    # -------------------------------------------
    from collections import OrderedDict
    datasets = OrderedDict() #
    # datasets["bauswein"] =  {'marker': 's', 'ms': 60, "color": "slategray", "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "fit": False}
    # datasets["hotokezaka"] ={'marker': '>', 'ms': 60, "color": "gray", "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "fit": False}
    # datasets["dietrich15"]= {'marker': 'd', "ms": 60, "color": "gray", "models": di15.simulations[di15.mask_for_with_sr], "data": di15, "err": di15.params.Mej_err, "label": r"Dietrich+2015", "fit": False}
    # datasets["sekiguchi15"]={'marker': 'p', "ms": 60, "color": "black", "models": se15.simulations[se15.mask_for_with_sr], "data": se15, "err": se15.params.Mej_err, "label": r"Sekiguchi+2015",  "fit": False}
    # datasets["dietrich16"]= {'marker': 'D', "ms": 60, "color": "gray", "models": di16.simulations[di16.mask_for_with_sr], "data": di16, "err": di16.params.Mej_err, "label": r"Dietrich+2016", "fit": False}
    # datasets["sekiguchi16"]={'marker': 'h', "ms": 60, "color": "black", "models": se16.simulations[se16.mask_for_with_sr], "data": se16, "err": se16.params.Mej_err, "label": r"Sekiguchi+2016", "fit": False}
    # datasets["lehner"] =    {'marker': 'P', 'ms': 60, "color": "orange", "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016",  "fit": False}
    # datasets["radice"] =    {'marker': '*', "ms": 60, "color": "green", "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018",  "fit": False}
    # datasets["kiuchi"] =    {'marker': "X", "ms": 60, "color": "gray", "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019",  "fit": False}
    # datasets["vincent"] =   {'marker': 'v', 'ms': 60, "color": "red", "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019",  "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets['our'] =       {"models": md.groups,  "data": md, "err": "v_n", "fit": False, "plot_errorbar":True, "plot_xerrorbar":True,
                             "color": "#EOS", "label": r"None",#r"#EOS",
                             "marker":"d", 'ms': 60,
                             "facecolor":"none","edgecolor":"#EOS"}

    datasets['our_total'] = {"v_n_x": "vel_inf_ave-tot", "v_n_y":"Mej_tot-tot", "data": md, "err": "v_n", "plot_errorbar":False, "plot_xerrorbar":False,
                             "models": md.groups[(md.groups.tend_wind > 0.035) | (md.groups["group"] == "BLh_M10651772_M0_LK")],
                             'marker': 'P', 'ms': 60, "color": "#EOS",  "label": r"None",#r"None",
                             "fit": False, "edgecolor":"none", "facecolor":"#EOS"}

    datasets['our_sec'] = {"v_n_x": "vel_inf_ave-tot", "v_n_y": "0.4Mdisk3D", "data": md, "err": "v_n", "plot_errorbar":False, "plot_xerrorbar":False,
                             "models": md.groups[(md.groups.tend_wind > 0.035) | (md.groups["group"] == "BLh_M10651772_M0_LK")],
                             'marker': 'v', 'ms': 60, "color": "#EOS", "label": r"None", "fit": False,
                             "edgecolor": "none", "facecolor": "#EOS"}

    #### datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}

    for t in datasets.keys():
        # datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["fill_style"] = "none"

    y_dic    = {"v_n": "Mej_tot-geo",     "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    x_dic    = {"v_n": "vel_inf_ave-geo", "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (6.0, 5.5),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 16, "labelsize": 16,
        "fit_panel": False,
        "plot_diagonal":False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": False,  # "tab10",
        "xmin": 0.025, "xmax": 0.35, "xscale": "linear",
        "ymin": 6e-4, "ymax": 2e-1, "yscale": "log",
        "ylabel": r"$M_{\rm ej}$ $[M_{\odot}]$",#$[10^{-3}M_{\odot}]$",
        "xlabel": r"$\langle \upsilon_{\rm ej} \rangle$ [c]",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_mej_vej_our2.png",
        "legend": {"fancybox": False, "loc": 'upper right',
                    ##"bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                    "shadow": "False", "ncol": 1, "fontsize": 12,
                    "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,

        # "add_marker":
        #     [{"marker": 'P', "color": "gray", "ms": 8, "alpha": 1., "label": "Tot.Ej.", "linestyle": "none",
        #       'zorder': -1},
        #      {"marker": 'v', "color": "gray", "ms": 8, "alpha": 1., "label": r"Sec.Ej",
        #       "linestyle": "none", 'zorder': -1},
        #      ],

        "patch":
            [
             {"data":{"x":0.27,"y":0.016,"xerr":(0.2,0.3),"yerr":(1e-2,2e-2)},
             "plot":{"facecolor":"blue", "alpha":0.4, "edgecolor":"none", "label":"Blue kN"}},
             {"data": {"x": 0.1, "y": 0.05, "xerr": (0.07, 0.14), "yerr": (4e-2, 6e-2)},
              "plot": {"facecolor": "red", "alpha": 0.4, "edgecolor": "none", "label": "Red kN"}},
             # {"data": {"x": None, "y": None, "xerr": (0.2, 0.23), "yerr": (1e-2, 5e-3)},
             #  "plot": {"facecolor": "none", "alpha": 0.4, "edgecolor": "red", "label": "Dyn. kN [P]"}},
             # {"data": {"x": None, "y": None, "xerr": (0.066, 0.068), "yerr": (0.001*0.08, 0.2*0.12)},
             #  "plot": {"facecolor": "none", "alpha": 0.4, "edgecolor": "blue", "label": "Wind kN [P]"}},
             # {"data": {"x": None, "y": None, "xerr": (0.027, 0.04), "yerr": (0.4 * 0.08, 0.2*0.1)},
             #  "plot": {"facecolor": "none", "alpha": 0.4, "edgecolor": "purple", "label": "Sec. kN [P]"}},
             ],

        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "savepdf":True,
        "tight_layout":True
    }

    # ---------------- fit ------------------
    #from make_fit import fitting_function_mej, fitting_coeffs_mej_our
    fit_dic =  {}
    # {
    #     "func":fit_funcs.mej_flat_mean, "coeffs": fit_coefs.mej_all_2(),
    #     "xmin": 0.95, "xmax": 2.0, "xscale": "linear", "xlabel": "$M_1/M_2$",#r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
    #     "ymin": -2.0, "ymax": 2.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
    #     "plot_zero":True
    # }

    # Fit
    from make_fit import fitting_function_mej
    #from make_fit import fitting_coeffs_mej_david, fitting_coeffs_mej_our
    #from make_fit import complex_fic_data_mej_module
    #complex_fic_data_mej_module(datasets, fitting_coeffs_mej_our(), key_for_usable_dataset="fit")


    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    # for k in datasets.keys():
    #     if k!="our_total":
    #         datasets[k]["plot_errorbar"] = True
    #         datasets[k]["plot_xerrorbar"] = True

    plot_datasets_scatter2(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)

def task_plot_mej_vs_ye_all():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki                  #
    from model_sets import models_sekiguchi2016 as se16         # [23] arxive:1603.01918 # no Mb
    from model_sets import models_sekiguchi2015 as se15         # [-]  arxive:1502.06660 # no Mb
    from model_sets import models_bauswein2013 as bs            # [20] arxive:1302.6530
    from model_sets import models_lehner2016 as lh              # [22] arxive:1603.00501
    from model_sets import models_hotokezaka2013 as hz          # [19] arxive:1212.0905
    from model_sets import models_dietrich_ujevic2016 as du
    from model_sets import models_dietrich2015 as di15          # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16          # [24] arxive:1607.06636

    # -------------------------------------------
    from collections import OrderedDict
    datasets = OrderedDict() #
    #datasets["bauswein"] =  {'marker': 's', 'ms': 20, "color": "slategray", "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "fit": False}
    #datasets["hotokezaka"] ={'marker': '>', 'ms': 20, "color": "gray", "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "fit": False}
    #datasets["dietrich15"]= {'marker': 'd', "ms": 20, "color": "gray", "models": di15.simulations[di15.mask_for_with_sr], "data": di15, "err": di15.params.Mej_err, "label": r"Dietrich+2015", "fit": False}
    datasets["sekiguchi15"]={'marker': 'p', "ms": 60, "color": "black", "models": se15.simulations[se15.mask_for_with_sr], "data": se15, "err": se15.params.Mej_err, "label": r"Sekiguchi+2015",  "fit": False}
    #datasets["dietrich16"]= {'marker': 'D', "ms": 20, "color": "gray", "models": di16.simulations[di16.mask_for_with_sr], "data": di16, "err": di16.params.Mej_err, "label": r"Dietrich+2016", "fit": False}
    datasets["sekiguchi16"]={'marker': 'h', "ms": 60, "color": "black", "models": se16.simulations[se16.mask_for_with_sr], "data": se16, "err": se16.params.Mej_err, "label": r"Sekiguchi+2016", "fit": False}
    #datasets["lehner"] =    {'marker': 'P', 'ms': 20, "color": "orange", "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016",  "fit": False}
    # datasets["radice"] =    {'marker': '*', "ms": 60, "color": "green", "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018",  "fit": False}
    datasets["radiceM0"] =    {'marker': '*', "ms": 60, "color": "green", "models": rd.simulations[rd.with_m0], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018",  "fit": False}
    datasets["radiceLK"] =    {'marker': '*', "ms": 60, "color": "lime", "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018",  "fit": False}
    #datasets["kiuchi"] =    {'marker': "X", "ms": 20, "color": "gray", "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019",  "fit": False}
    datasets["vincent"] =   {'marker': 'v', 'ms': 60, "color": "red", "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019",  "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets['reference'] = {'marker': 'o', 'ms': 60, "color": "blue", "models": md.groups, "data": md, "err": "v_n", "label": r"This work",  "fit": False}
    # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}

    for t in datasets.keys():
        datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["label"] = ourmd.datasets_labels[t]
        datasets[t]["fill_style"] = "none"
        datasets[t]["plot_errorbar"] = False
        datasets[t]["plot_xerrorbar"] = False
        datasets[t]["facecolor"] = "none"
        datasets[t]["edgecolor"] = datasets[t]["color"]

    y_dic    = {"v_n": "Mej_tot-geo",     "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    x_dic    = {"v_n": "Ye_ave-geo",      "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (6.0, 5.5),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 18, "labelsize": 18,
        "fit_panel": False,
        "plot_diagonal":False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": False,  # "tab10",
        "xmin": 0.01, "xmax": 0.35, "xscale": "linear",
        "ymin": 1e-4, "ymax": 1e-1, "yscale": "log",
        "ylabel": r"$M_{\rm ej}$ $[M_{\odot}]$",
        "xlabel": r"$\langle Y_{e;\:\rm ej} \rangle$",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_mej_yeej_all2.png",
        "legend": {"fancybox": False, "loc": 'upper right',
                   #"bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "savepdf":True,
        "tight_layout":True
    }

    # ---------------- fit ------------------
    from make_fit import fitting_function_mej, fitting_coeffs_mej_our
    fit_dic = {}
    #     {
    #     "func":fitting_function_mej, "coeffs": fitting_coeffs_mej_our(),
    #     "xmin": 0, "xmax": 7.5, "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
    #     "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
    #     "plot_zero":True
    # }

    # Fit
    from make_fit import fitting_function_mej
    #from make_fit import fitting_coeffs_mej_david, fitting_coeffs_mej_our
    #from make_fit import complex_fic_data_mej_module
    #complex_fic_data_mej_module(datasets, fitting_coeffs_mej_our(), key_for_usable_dataset="fit")


    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    # for k in datasets.keys():
    #     datasets[k]["plot_errorbar"] = False

    plot_datasets_scatter2(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)
def task_plot_mej_vs_yeej():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki                  #
    from model_sets import models_sekiguchi2016 as se16         # [23] arxive:1603.01918 # no Mb
    from model_sets import models_sekiguchi2015 as se15         # [-]  arxive:1502.06660 # no Mb
    from model_sets import models_bauswein2013 as bs            # [20] arxive:1302.6530
    from model_sets import models_lehner2016 as lh              # [22] arxive:1603.00501
    from model_sets import models_hotokezaka2013 as hz          # [19] arxive:1212.0905
    from model_sets import models_dietrich_ujevic2016 as du
    from model_sets import models_dietrich2015 as di15          # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16          # [24] arxive:1607.06636

    # -------------------------------------------
    from collections import OrderedDict
    datasets = OrderedDict() #
    # datasets["bauswein"] =  {'marker': 's', 'ms': 60, "color": "slategray", "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "fit": False}
    # datasets["hotokezaka"] ={'marker': '>', 'ms': 60, "color": "gray", "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "fit": False}
    # datasets["dietrich15"]= {'marker': 'd', "ms": 60, "color": "gray", "models": di15.simulations[di15.mask_for_with_sr], "data": di15, "err": di15.params.Mej_err, "label": r"Dietrich+2015", "fit": False}
    # datasets["sekiguchi15"]={'marker': 'p', "ms": 60, "color": "black", "models": se15.simulations[se15.mask_for_with_sr], "data": se15, "err": se15.params.Mej_err, "label": r"Sekiguchi+2015",  "fit": False}
    # datasets["dietrich16"]= {'marker': 'D', "ms": 60, "color": "gray", "models": di16.simulations[di16.mask_for_with_sr], "data": di16, "err": di16.params.Mej_err, "label": r"Dietrich+2016", "fit": False}
    # datasets["sekiguchi16"]={'marker': 'h', "ms": 60, "color": "black", "models": se16.simulations[se16.mask_for_with_sr], "data": se16, "err": se16.params.Mej_err, "label": r"Sekiguchi+2016", "fit": False}
    # datasets["lehner"] =    {'marker': 'P', 'ms': 60, "color": "orange", "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016",  "fit": False}
    # datasets["radice"] =    {'marker': '*', "ms": 60, "color": "green", "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018",  "fit": False}
    # datasets["kiuchi"] =    {'marker': "X", "ms": 60, "color": "gray", "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019",  "fit": False}
    # datasets["vincent"] =   {'marker': 'v', 'ms': 60, "color": "red", "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019",  "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets['our'] =       {"models": md.groups,  "data": md, "err": "v_n", "fit": False, "plot_errorbar":True, "plot_xerrorbar":True,
                             "color": "#EOS", "label": r"#EOS", "marker":"d", 'ms': 60,
                             "facecolor":"none","edgecolor":"#EOS"}

    datasets['our_total'] = {"v_n_x": "Ye_ave-geo-tot", "v_n_y":"Mej_tot-tot", "data": md, "err": "v_n", "plot_errorbar":False, "plot_xerrorbar":False,
                             "models": md.groups[(md.groups.tend_wind > 0.035) | (md.groups["group"] == "BLh_M10651772_M0_LK")],
                             'marker': 'P', 'ms': 60, "color": "#EOS",  "label": r"None", "fit": False, "edgecolor":"none", "facecolor":"#EOS"}

    # datasets['our_sec'] = {"v_n_x": "vel_inf_ave-tot", "v_n_y": "0.4Mdisk3D", "data": md, "err": "v_n", "plot_errorbar":False, "plot_xerrorbar":False,
    #                          "models": md.groups[(md.groups.tend_wind > 0.035) | (md.groups["group"] == "BLh_M10651772_M0_LK")],
    #                          'marker': 'v', 'ms': 60, "color": "#EOS", "label": r"None", "fit": False,
    #                          "edgecolor": "none", "facecolor": "#EOS"}

    #### datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}

    for t in datasets.keys():
        # datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["fill_style"] = "none"

    # datasets['our_total']["annotate"] = "q"

    y_dic    = {"v_n": "Mej_tot-geo",     "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    x_dic    = {"v_n": "Ye_ave-geo", "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (6.0, 5.5),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 16, "labelsize": 16,
        "fit_panel": False,
        "plot_diagonal":False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": False,  # "tab10",
        "xmin": 0.025, "xmax": 0.35, "xscale": "linear",
        "ymin": 6e-4, "ymax": 2e-1, "yscale": "log",
        "ylabel": r"$M_{\rm ej}$ $[M_{\odot}]$",#$[10^{-3}M_{\odot}]$",
        "xlabel": r"$\langle Y_{e;\:\rm ej} \rangle$",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_mej_yeej_our2.png",
        "legend": {"fancybox": False, "loc": 'upper left',
                   #"bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 4, "fontsize": 12, "columnspacing":0.3,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,

        "add_marker":
            [
             {"marker": 'P', "color": "gray", "ms": 8, "alpha": 1., "label": "Tot.Ej.", "linestyle": "none",
              'zorder': -1},
             {"marker": 'v', "color": "gray", "ms": 8, "alpha": 1., "label": r"Sec.Ej.",
              "linestyle": "none", 'zorder': -1},
             ],

        "patch":
            [
             # {"data":{"x":0.27,"y":0.016,"xerr":(0.25, 0.50),"yerr":(1e-2,2e-2)},
             # "plot":{"facecolor":"blue", "alpha":0.4, "edgecolor":"none", "label":"Blue kN"}},
             # {"data": {"x": 0.1, "y": 0.05, "xerr": (0.0, 0.25), "yerr": (4e-2, 6e-2)},
             #  "plot": {"facecolor": "red", "alpha": 0.4, "edgecolor": "none", "label": "Red kN"}},
             #
             # {"data": {"x": None, "y": None, "xerr": (0, 0.5), "yerr": (1e-2, 5e-3)},
             #  "plot": {"facecolor": "none", "alpha": 0.4, "edgecolor": "red", "label": "Dyn. kN [P]"}},
             # {"data": {"x": None, "y": None, "xerr": (0.25, 0.5), "yerr": (0.001*0.08, 0.2*0.12)},
             #  "plot": {"facecolor": "none", "alpha": 0.4, "edgecolor": "blue", "label": "Wind kN [P]"}},
             # {"data": {"x": None, "y": None, "xerr": (0.25, 0.5), "yerr": (0.4 * 0.08, 0.2*0.1)},
             #  "plot": {"facecolor": "none", "alpha": 0.4, "edgecolor": "purple", "label": "Sec. kN [P]"}},
             ],

        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "savepdf":True,
        "tight_layout":True
    }

    # ---------------- fit ------------------
    #from make_fit import fitting_function_mej, fitting_coeffs_mej_our
    fit_dic =  {}
    # {
    #     "func":fit_funcs.mej_flat_mean, "coeffs": fit_coefs.mej_all_2(),
    #     "xmin": 0.95, "xmax": 2.0, "xscale": "linear", "xlabel": "$M_1/M_2$",#r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
    #     "ymin": -2.0, "ymax": 2.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
    #     "plot_zero":True
    # }

    # Fit
    from make_fit import fitting_function_mej
    #from make_fit import fitting_coeffs_mej_david, fitting_coeffs_mej_our
    #from make_fit import complex_fic_data_mej_module
    #complex_fic_data_mej_module(datasets, fitting_coeffs_mej_our(), key_for_usable_dataset="fit")


    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    # for k in datasets.keys():
    #     if k!="our_total":
    #         datasets[k]["plot_errorbar"] = True
    #         datasets[k]["plot_xerrorbar"] = True

    plot_datasets_scatter2(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)

def task_plot_vej_vs_ye():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki                  #
    from model_sets import models_sekiguchi2016 as se16         # [23] arxive:1603.01918 # no Mb
    from model_sets import models_sekiguchi2015 as se15         # [-]  arxive:1502.06660 # no Mb
    from model_sets import models_bauswein2013 as bs            # [20] arxive:1302.6530
    from model_sets import models_lehner2016 as lh              # [22] arxive:1603.00501
    from model_sets import models_hotokezaka2013 as hz          # [19] arxive:1212.0905
    from model_sets import models_dietrich_ujevic2016 as du
    from model_sets import models_dietrich2015 as di15          # [21] arxive:1504.01266
    from model_sets import models_dietrich2016 as di16          # [24] arxive:1607.06636

    # -------------------------------------------
    from collections import OrderedDict
    datasets = OrderedDict() #
    #datasets["bauswein"] =  {'marker': 's', 'ms': 20, "color": "slategray", "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "fit": False}
    #datasets["hotokezaka"] ={'marker': '>', 'ms': 20, "color": "gray", "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "fit": False}
    #datasets["dietrich15"]= {'marker': 'd', "ms": 20, "color": "gray", "models": di15.simulations[di15.mask_for_with_sr], "data": di15, "err": di15.params.Mej_err, "label": r"Dietrich+2015", "fit": False}
    #datasets["sekiguchi15"]={'marker': 'p', "ms": 20, "color": "black", "models": se15.simulations[se15.mask_for_with_sr], "data": se15, "err": se15.params.Mej_err, "label": r"Sekiguchi+2015",  "fit": False}
    #datasets["dietrich16"]= {'marker': 'D', "ms": 20, "color": "gray", "models": di16.simulations[di16.mask_for_with_sr], "data": di16, "err": di16.params.Mej_err, "label": r"Dietrich+2016", "fit": False}
    datasets["sekiguchi16"]={'marker': 'h', "ms": 60, "color": "black", "models": se16.simulations[se16.mask_for_with_sr], "data": se16, "err": se16.params.Mej_err, "label": r"Sekiguchi+2016", "fit": False}
    #datasets["lehner"] =    {'marker': 'P', 'ms': 20, "color": "orange", "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016",  "fit": False}
    datasets["radiceLK"] =    {'marker': '*', "ms": 60, "color": "lime", "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018",  "fit": False}
    datasets["radiceM0"] =    {'marker': '*', "ms": 60, "color": "green", "models": rd.simulations[rd.with_m0], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018",  "fit": False}
    #datasets["kiuchi"] =    {'marker': "X", "ms": 20, "color": "gray", "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019",  "fit": False}
    datasets["vincent"] =   {'marker': 'v', 'ms': 60, "color": "red", "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019",  "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets['reference'] = {'marker': 'o', 'ms': 60, "color": "blue", "models": md.groups, "data": md, "err": "v_n", "label": r"This work",  "fit": False}
    # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}

    for t in datasets.keys():
        datasets[t]["marker"] = ourmd.datasets_markers[t]
        datasets[t]["label"] = ourmd.datasets_labels[t]
        datasets[t]["fill_style"] = "none"
        datasets[t]["fill_style"] = "none"
        datasets[t]["plot_errorbar"] = False
        datasets[t]["plot_xerrorbar"] = False
        datasets[t]["facecolor"] = "none"
        datasets[t]["edgecolor"] = datasets[t]["color"]

    y_dic    = {"v_n": "vel_inf_ave-geo", "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    x_dic    = {"v_n": "Ye_ave-geo",      "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (6.0, 5.5),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 18, "labelsize": 18,
        "fit_panel": False,
        "plot_diagonal":False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": False,  # "tab10",
        "xmin": 0.01, "xmax": 0.35, "xscale": "linear",
        "ymin": 0.1, "ymax": 0.35, "yscale": "linear",
        "ylabel": r"$\langle \upsilon_{\rm ej} \rangle$ [c]",
        "xlabel": r"$\langle Y_{e;\:\rm ej} \rangle$",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_vej_yeej_all2.png",
        "legend": {"fancybox": False, "loc": 'upper left',
                   #"bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "savepdf":True,
        "tight_layout":True
    }

    # ---------------- fit ------------------
    from make_fit import fitting_function_mej, fitting_coeffs_mej_our
    fit_dic = {}
    #     {
    #     "func":fitting_function_mej, "coeffs": fitting_coeffs_mej_our(),
    #     "xmin": 0, "xmax": 7.5, "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
    #     "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
    #     "plot_zero":True
    # }

    # Fit
    from make_fit import fitting_function_mej
    #from make_fit import fitting_coeffs_mej_david, fitting_coeffs_mej_our
    #from make_fit import complex_fic_data_mej_module
    #complex_fic_data_mej_module(datasets, fitting_coeffs_mej_our(), key_for_usable_dataset="fit")


    # -------------- colorcoding models with data ------------
    # for k in datasets.keys():
    #     if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
    #     if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
    #     if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
    #     if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
    #     if k == "lehner": datasets["lehner"]["color"]           = "gray"
    #     if k == "radice": datasets["radice"]["color"]           = None
    #     if k == "vincent": datasets["vincent"]["color"]         = None
    #     if k == "our": datasets["our"]["color"]                 = None
    for k in datasets.keys():
        datasets[k]["plot_errorbar"] = False

    plot_datasets_scatter2(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)

""" --- --- --- PAPER --- --- --- """









""" --- --- --- """

if __name__ == "__main__":

    ''' --- Mej --- '''
    #task_plot_mej_vs_fit_all()
    #task_plot_mej_vs_fit()
    # task_plot_mej_all()
    # task_plot_mej()
    #task_plot_mej_cc()
    # # # task_plot_mej_q_vs_fit()

    # task_plot_mej_fits_only()
    task_plot_mej_all_q()
    task_plot_mej_all_Lambda()

    ''' --- vej --- '''
    #task_plot_vej_vs_fit_all()
    # task_plot_vej_vs_fit()
    # task_plot_vej_all()
    #task_plot_vej()
    #task_plot_vej_cc()

    # task_plot_vej_fits_only()

    ''' --- theta --- '''
    # task_plot_mej_vs_theta_all()
    # task_plot_vej_vs_theta()
    # task_plot_theta_vs_ye()

    ''' --- ye --- '''
    #task_plot_ye_vs_fit_all()
    # task_plot_ye_vs_fit()
    #task_plot_yeej_all()
    #task_plot_yeej()
    #task_plot_yeej_cc()
    # task_plot_yeej_fits_only()
    ''' --- Mej vs vej ---  '''
    # task_plot_mej_vs_vej_all()
    # task_plot_mej_vs_vej()

    ''' --- Mej vs ye ---  '''
    # task_plot_mej_vs_ye_all()
    # task_plot_mej_vs_yeej()
    ''' --- vej vs ye ---  '''
    # task_plot_vej_vs_ye()

    ''' --- Mdisk --- '''
    # task_plot_mdisk_vs_fit_all()
    # task_plot_mdisk_vs_fit()
    #task_plot_mdisk_q_vs_fit_all()
    # task_plot_mdisk_q_vs_fit()
    # task_plot_mdisk()
    # task_plot_mdisk_fits_only()

    # task_plot_yedisk_cc()
    # task_plot_sdisk_cc()
    # task_plot_thetadisk_cc()
    # task_plot_tdisk_cc()


    ''' test '''
    # from matplotlib import rcParams
    #
    # rcParams['font.family'] = 'sans-serif'
    # rcParams['font.sans-serif'] = ['Noto Sans Tibetan Regular']

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot([1, 1], [2, 2], 'black', '-.', label=r'$\theta$')
    # plt.legend()
    # plt.savefig(__outplotdir__ + "tmp.png", dpi=128)


    ''' [[[ sebastiano request ]]] '''
    #_seba_task_plot_mej_vs_vej_all()
    #_seba_task_plot_mej_vs_ye_all()
    #_seba_task_plot_vej_vs_ye()

