from __future__ import division

import numpy as np
import os
import matplotlib
import matplotlib.font_manager
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif')
# fm = matplotlib.font_manager.json_load("./fontlist-v300-before.json")
# fm.findfont("serif", rebuild_if_missing=True)
# fm.findfont("serif", fontext="afm", rebuild_if_missing=True)

from collections import OrderedDict

from model_sets import combined as md

__outplotdir__ = "/home/vsevolod/GIT/bitbucket/bns_gw170817/tex/fitpaper/figs/summary/"
if not os.path.isdir(__outplotdir__):
    os.mkdir(__outplotdir__)

from make_fit4 import FittingFunctions, FittingCoefficients, Fit_Data

''' -----------------------------| Components |--------------------------------- '''

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

''' -----------------------------| Modules |------------------------------------- '''

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
    ''' --- hline and vline --- '''
    if "hline" in plot_dic.keys() and len(plot_dic["hline"].keys()) > 0:
        ax[0].axhline(**plot_dic["hline"])
    #
    ''' --- additional legend --- '''
    if "add_legend" in plot_dic.keys():
        dic_ = plot_dic["add_legend"]
        n = dic_["last_n"]
        del dic_["last_n"]
        han, lab = ax[0].get_legend_handles_labels()
        ax[0].add_artist(ax[0].legend(han[:-1 * n], lab[:-1 * n], **plot_dic["legend"]))
        ax[0].add_artist(ax[0].legend(han[len(han) - n:], lab[len(lab) - n:], **dic_))
    elif  "legend" in plot_dic.keys() and len(plot_dic["legend"].keys()) > 0:
        ax[0].legend(**plot_dic["legend"])
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
                edgecolors = [model_dic['edgecolor'] for _ in list(models.iteritems())]
                facecolors = [model_dic['facecolor'] for _ in list(models.iteritems())]
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
                imarker = md.eos_dic_marker[ieos]
                markers.append(imarker)
        else:
            markers = [model_dic['marker'] for _ in list(models.iteritems())]
            assert len(markers) > 0

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
            if len(markers) == 0:
                raise NameError("No 'maker' set for scatter plot. Set markers for : {}".format(dataset_name))
            if len(colors) > 0:
                sc = ax[0].scatter(x, y, c=colors, s=mss, marker=markers[0],  label=None)
            elif len(colors) == 0 and len(edgecolors) != 0 and len(facecolors) != 0:
                sc = ax[0].scatter(x, y, s=mss, marker=markers[0], label=model_dic["label"],
                                   edgecolor=model_dic["edgecolor"],
                                   facecolor=model_dic["facecolor"], )
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

    ''' -- additional for poly --- '''
    if "add_poly" in plot_dic.keys():
        for entry in plot_dic["add_poly"]:
            dic_ = dict(entry)
            x_arr = dic_["x"]
            coeffs = dic_["coeffs"]
            if "func" in dic_.keys() and dic_["func"] == "rad18":
                del dic_["func"]
                a, b, c, d = coeffs
                y_arr = np.array(np.maximum(a + b * (np.tanh((x_arr - c) / d)), 1.e-3))
            elif len(coeffs) == 1:
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
            if "to_val" in dic_.keys():
                if dic_["to_val"] == "10**": y_arr = 10**y_arr
                else:
                    raise NameError("Not implemented")
                del dic_["to_val"]
            # print(x_arr)
            # print(y_arr) ; exit(1)
            ax[0].plot(np.array(x_arr).flatten(), np.array(y_arr).flatten(), **dic_)
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
    if "add_legend" in plot_dic.keys():
        dic_ = plot_dic["add_legend"]
        n = dic_["last_n"]
        del dic_["last_n"]
        han, lab = ax[0].get_legend_handles_labels()
        ax[0].add_artist(ax[0].legend(han[:-1 * n], lab[:-1 * n], **plot_dic["legend"]))
        ax[0].add_artist(ax[0].legend(han[len(han) - n:], lab[len(lab) - n:], **dic_))
    elif "legend" in plot_dic.keys() and len(plot_dic["legend"].keys()) > 0:
        ax[0].legend(**plot_dic["legend"])

    #
    print("plotted: \n")
    print(plot_dic["figname"])
    if plot_dic["tight_layout"]: plt.tight_layout()
    #
    plt.draw()
    plt.savefig(plot_dic["figname"], dpi=plot_dic["dpi"])
    if plot_dic["savepdf"]:
        plt.savefig(plot_dic["figname"].replace(".png", ".pdf"))
    if plot_dic["show"]: plt.show()
    print(plot_dic["figname"])
    plt.close()
''' -----------------------------| Tasks |------------------------------------------'''

def task_plot_mej_all_q():

    # bibblacklist = []
    v_n = "Mej_tot-geo"
    # sel = md.simulations[~np.isnan(md.simulations[v_n])]
    # bibkeys = list(set(sel["bibkey"]))
    # datasets = OrderedDict()
    # for key in bibkeys:
    #     if not key in bibblacklist:
    #         datasets[key] = {"models":sel[sel["bibkey"]==key], "data":md, "fit":True,
    #                          "color":None,"marker":None,"ms":None,"label":None, "err":md.params.Mej_err}
    # assert len(datasets.keys()) > 0
    datasets = OrderedDict()
    datasets['reference'] = {"models": md.reference, "data": md, "fit": True, "plot_errorbar": True, "err": "v_n"}
    datasets["radiceLK"] =  {"models": md.radice18lk, "data": md, "fit": True, "plot_errorbar": True, "err": md.params.MdiskPP_err}
    datasets["radiceM0"] =  {"models": md.radice18m0, "data": md, "fit": True, "plot_errorbar": True, "err": md.params.MdiskPP_err}
    datasets["lehner"] =    {"models": md.lehner16, "data": md, "fit": True, "plot_errorbar": True, "err": md.params.MdiskPP_err}
    datasets["kiuchi"] =    {"models": md.kiuchi19, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.MdiskPP_err}
    datasets["vincent"] =   {"models": md.vincent19, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.MdiskPP_err}
    datasets["dietrich16"] ={"models": md.dietrich16, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.MdiskPP_err}
    datasets["dietrich15"] ={"models": md.dietrich15, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.MdiskPP_err}
    datasets["sekiguchi15"]={"models": md.sekiguchi15, "data": md, "err": md.params.Mej_err, "label": r"Sekiguchi+2015",  "fit": True}
    datasets["sekiguchi16"]={"models": md.sekiguchi16, "data": md, "err": md.params.Mej_err, "label": r"Sekiguchi+2016", "fit": True}
    datasets["hotokezaka"] ={"models": md.hotokezaka12, "data": md, "err": md.params.Mej_err, "label": r"Hotokezaka+2013", "fit": True}
    datasets["bauswein"] =  {"models": md.basuwein13, "data": md, "err": md.params.Mej_err, "label": r"Bauswein+2013", "fit": True}

    label_whitelist = ["bauswein", "hotokezaka", "dietrich15", "dietrich16", "kiuchi"]

    for t in datasets.keys():
        datasets[t]["ms"] = 60
        datasets[t]["label"] = md.datasets_labels[t]
        datasets[t]["marker"] = md.datasets_markers[t]
        datasets[t]["fill_style"] = "none"
        datasets[t]["plot_errorbar"] = False
        datasets[t]["plot_xerrorbar"] = False
        datasets[t]["facecolor"] = "none"
        datasets[t]["edgecolor"] = md.datasets_colors[t]
        datasets[t]["plot_errorbar"] = False
        datasets[t]["labelmarkercolor"] = md.datasets_colors[t]
        if not t in label_whitelist:
            datasets[t]["label"] = None

    x_dic    = {"v_n": "q", "err": None, "deferr": None, "mod": {}}
    y_dic    = {"v_n": "Mej_tot-geo",     "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}

    # fits for different neutrino schemes
    o_fit = Fit_Data(md.simulations[md.simulations["nus"]=="leak"], v_n, "default", clean_nans=True)
    coeffs_l, _, _, _ = o_fit.fit_curve(ff_name="poly2_q", cf_name="poly2")
    o_fit = Fit_Data(md.simulations[(md.simulations["nus"] == "leakM0")|
                                    (md.simulations["nus"] == "M1")|
                                    (md.simulations["nus"] == "leakM1")], v_n, "default", clean_nans=True)
    coeffs_m, _, _, _ = o_fit.fit_curve(ff_name="poly2_q", cf_name="poly2")
    o_fit = Fit_Data(md.simulations[md.simulations["nus"] == "none"], v_n, "default", clean_nans=True)
    coeffs_n, _, _, _ = o_fit.fit_curve(ff_name="poly2_q", cf_name="poly2")

    plot_dic = {
        "figsize": (6.0, 5.5),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "fit_panel": False,
        "plot_diagonal":False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": False,  # "tab10",
        "xmin": 0.95, "xmax": 2.1, "xscale": "linear",
        "ymin": 1e-4, "ymax": 4e-1, "yscale": "log",
        "xlabel": "$q$",#r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ylabel": r"$M_{\rm ej}$ [$M_{\odot}$]",# $[10^{-3}M_{\odot}]$",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_q_mej.png",
        "legend": {"fancybox": False, "loc": 'lower right',
                   #"bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 12,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        "add_poly": [
            # {"x": np.arange(1., 2., 0.1), "coeffs": np.array(coeffs_l), # 52.3 & 0.549
            # "color": "black", "lw": 0.8, "ls": "-.", "label": r"$P_2 ^{\rm leak}$"},
            #
            # {"x": np.arange(1., 2., 0.1), "coeffs": np.array(coeffs_m), # 36.3 & 0.238
            # "color": "blue", "lw": 0.8, "ls": "-.", "label": r"$P_2 ^{\rm M0/M1}$"},
            #
            # {"x": np.arange(1., 2., 0.1), "coeffs": np.array(coeffs_n), # 12.3 & 0.830
            # "color": "gray", "lw": 0.8, "ls": "-.", "label": r"$P_2 ^{\rm none}$"}
        ],
        # "add_legend":{"last_n":12,
        #               "fancybox": False, "loc": 'uppwer left',
        #               # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #               "shadow": "False", "ncol": 2, "fontsize": 12,
        #               "framealpha": 0., "borderaxespad": 0., "frameon": False
        #               },
        # "hline": {"y":5.220 * 1.e-3, "color":'blue', "linestyle":"dashed", "label":"Mean"},
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "show":True,
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


    plot_datasets_scatter3(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)

def task_plot_mej_all_Lambda():

    v_n = "Mej_tot-geo"
    # sel = md.simulations[~np.isnan(md.simulations[v_n])]
    # bibkeys = list(set(sel["bibkey"]))
    # datasets = OrderedDict()
    # for key in bibkeys:
    #     if not key in bibblacklist:
    #         datasets[key] = {"models":sel[sel["bibkey"]==key], "data":md, "fit":True,
    #                          "color":None,"marker":None,"ms":None,"label":None, "err":md.params.Mej_err}
    # assert len(datasets.keys()) > 0
    datasets = OrderedDict()
    datasets['reference'] = {"models": md.reference, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    datasets["radiceLK"] =  {"models": md.radice18lk, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    datasets["radiceM0"] =  {"models": md.radice18m0, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    datasets["lehner"] =    {"models": md.lehner16, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    datasets["kiuchi"] =    {"models": md.kiuchi19, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    datasets["vincent"] =   {"models": md.vincent19, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    datasets["dietrich16"] ={"models": md.dietrich16, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    datasets["dietrich15"] ={"models": md.dietrich15, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    datasets["sekiguchi15"]={"models": md.sekiguchi15, "data": md, "err": md.params.Mej_err, "label": r"Sekiguchi+2015",  "fit": True,"plot_errorbar": False}
    datasets["sekiguchi16"]={"models": md.sekiguchi16, "data": md, "err": md.params.Mej_err, "label": r"Sekiguchi+2016", "fit": True,"plot_errorbar": False}
    datasets["hotokezaka"] ={"models": md.hotokezaka12, "data": md, "err": md.params.Mej_err, "label": r"Hotokezaka+2013", "fit": True,"plot_errorbar": False}
    datasets["bauswein"] =  {"models": md.basuwein13, "data": md, "err": md.params.Mej_err, "label": r"Bauswein+2013", "fit": True,"plot_errorbar": False}

    label_whitelist = ['reference', "radiceLK", "radiceM0", "lehner", "vincent", "sekiguchi15", "sekiguchi16"]

    for t in datasets.keys():
        datasets[t]["ms"] = 60
        datasets[t]["label"] = md.datasets_labels[t]
        datasets[t]["marker"] = md.datasets_markers[t]
        datasets[t]["fill_style"] = "none"
        #datasets[t]["plot_errorbar"] = False
        #datasets[t]["plot_xerrorbar"] = False
        datasets[t]["facecolor"] = "none"
        datasets[t]["edgecolor"] = md.datasets_colors[t]
        #datasets[t]["plot_errorbar"] = False
        datasets[t]["labelmarkercolor"] = md.datasets_colors[t]
        if not t in label_whitelist:
            datasets[t]["label"] = None


    x_dic    = {"v_n": "Lambda", "err": None, "deferr": None, "mod": {}}
    y_dic    = {"v_n": "Mej_tot-geo",     "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}

    # fits for different neutrino schemes
    o_fit = Fit_Data(md.simulations[(md.simulations["bibkey"]=="Reference set")|
                                    (md.simulations["bibkey"]=="Vincent:2019kor")], v_n, "default", clean_nans=True) # [md.simulations["nus"]=="leak"] md.simulations[md.simulations["nus"]!="leak"]
    coeffs_l, _, chi2, _ = o_fit.fit_curve(ff_name="poly2_Lambda", cf_name="poly2", modify="log10")
    # o_fit = Fit_Data(md.simulations[(md.simulations["nus"] == "leakM0")|
    #                                 (md.simulations["nus"] == "M1")|
    #                                 (md.simulations["nus"] == "leakM1")], v_n, "default", clean_nans=True)
    # coeffs_m, _, _, _ = o_fit.fit_curve(ff_name="poly2_Lambda", cf_name="poly2")
    # o_fit = Fit_Data(md.simulations[md.simulations["nus"] == "none"], v_n, "default", clean_nans=True)
    # coeffs_n, _, _, _ = o_fit.fit_curve(ff_name="poly2_Lambda", cf_name="poly2")

    plot_dic = {
        "figsize": (6.0, 5.5),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "fit_panel": False,
        "plot_diagonal":False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": False,  # "tab10",
        "xmin": 0, "xmax": 2000, "xscale": "linear",
        "ymin": 5e-5, "ymax": 4e-1, "yscale": "log",
        "xlabel": r"$\tilde{\Lambda}$",#r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ylabel": r"$M_{\rm ej}$ [$M_{\odot}$]",# $[10^{-3}M_{\odot}]$",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_lambda_mej.png",
        "add_poly": [
            # {"x": np.arange(0, 1600, 100), "coeffs": np.array(coeffs_l), "to_val":"10**", # 625.8 & 0.125
            # "color": "black", "lw": 0.8, "ls": "-.", "label": r"$P_2 ^{\rm tot}$ $\chi_{\nu}^2=$"+"{:.1f}".format(chi2)},
            #
            # {"x": np.arange(0, 1600, 100), "coeffs": 1.e-2*np.array([-2.93319697e-02 , 1.63389243e-04 ,-4.63276134e-08]), # 71.6 & 0.031
            # "color": "red", "lw": 0.8, "ls": "-.", "label": r"$P_2 ^{Bernuzzi}$"},
            #
            # {"x": np.arange(0, 1600, 100), "coeffs": np.array(coeffs_n), # 96.8 & 0.228
            #  "color": "gray", "lw": 0.8, "ls": "-.", "label": r"$P_2 ^{\rm none}$"}
        ],
        # "add_legend":{"last_n":11,
        #               "fancybox": False, "loc": 'uppwer right',
        #               # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #               "shadow": "False", "ncol": 2, "fontsize": 12,
        #               "framealpha": 0., "borderaxespad": 0., "frameon": False
        #               },
        "legend": {"fancybox": False, "loc": 'upper left',
                   #"bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 12,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        # "hline": {"y":5.220 * 1.e-3, "color":'blue', "linestyle":"dashed", "label":"Mean"},
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "show":True,
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

    plot_datasets_scatter3(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)

# mdisk

def task_plot_mdisk_all_q():

    # bibblacklist = []
    v_n = "Mdisk3D"
    # sel = md.simulations[~np.isnan(md.simulations[v_n])]
    # bibkeys = list(set(sel["bibkey"]))
    # datasets = OrderedDict()
    # for key in bibkeys:
    #     if not key in bibblacklist:
    #         datasets[key] = {"models":sel[sel["bibkey"]==key], "data":md, "fit":True,
    #                          "color":None,"marker":None,"ms":None,"label":None, "err":md.params.Mej_err}
    # assert len(datasets.keys()) > 0
    datasets = OrderedDict()
    datasets['reference'] = {"models": md.reference, "data": md, "fit": True, "plot_errorbar": True, "err": "v_n"}
    datasets["radiceLK"] =  {"models": md.radice18lk, "data": md, "fit": True, "plot_errorbar": True, "err": md.params.MdiskPP_err}
    datasets["radiceM0"] =  {"models": md.radice18m0, "data": md, "fit": True, "plot_errorbar": True, "err": md.params.MdiskPP_err}
    datasets["lehner"] =    {"models": md.lehner16, "data": md, "fit": True, "plot_errorbar": True, "err": md.params.MdiskPP_err}
    datasets["kiuchi"] =    {"models": md.kiuchi19, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.MdiskPP_err}
    datasets["vincent"] =   {"models": md.vincent19, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.MdiskPP_err}
    datasets["dietrich16"] ={"models": md.dietrich16, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.MdiskPP_err}
    datasets["dietrich15"] ={"models": md.dietrich15, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.MdiskPP_err}
    datasets["sekiguchi15"]={"models": md.sekiguchi15, "data": md, "err": md.params.Mej_err, "label": r"Sekiguchi+2015",  "fit": True}
    datasets["sekiguchi16"]={"models": md.sekiguchi16, "data": md, "err": md.params.Mej_err, "label": r"Sekiguchi+2016", "fit": True}
    datasets["hotokezaka"] ={"models": md.hotokezaka12, "data": md, "err": md.params.Mej_err, "label": r"Hotokezaka+2013", "fit": True}
    datasets["bauswein"] =  {"models": md.basuwein13, "data": md, "err": md.params.Mej_err, "label": r"Bauswein+2013", "fit": True}

    label_whitelist = ["dietrich15", "dietrich16", "kiuchi"] # "bauswein", "hotokezaka",

    for t in datasets.keys():
        datasets[t]["ms"] = 60
        datasets[t]["label"] = md.datasets_labels[t]
        datasets[t]["marker"] = md.datasets_markers[t]
        datasets[t]["fill_style"] = "none"
        datasets[t]["plot_errorbar"] = False
        datasets[t]["plot_xerrorbar"] = False
        datasets[t]["facecolor"] = "none"
        datasets[t]["edgecolor"] = md.datasets_colors[t]
        datasets[t]["plot_errorbar"] = False
        datasets[t]["labelmarkercolor"] = md.datasets_colors[t]
        if not t in label_whitelist:
            datasets[t]["label"] = None

    x_dic    = {"v_n": "q", "err": None, "deferr": None, "mod": {}}
    y_dic    = {"v_n": "Mdisk3D",     "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}

    # fits for different neutrino schemes
    o_fit = Fit_Data(md.simulations, v_n, "default", clean_nans=True) # [md.simulations["nus"]=="leak"]
    coeffs_l, _, chi2, _ = o_fit.fit_curve(ff_name="poly2_q", cf_name="poly2", modify="log10")
    # o_fit = Fit_Data(md.simulations[(md.simulations["nus"] == "leakM0")|
    #                                 (md.simulations["nus"] == "M1")|
    #                                 (md.simulations["nus"] == "leakM1")], v_n, "default", clean_nans=True)
    # coeffs_m, _, _, _ = o_fit.fit_curve(ff_name="poly2_q", cf_name="poly2")
    # o_fit = Fit_Data(md.simulations[md.simulations["nus"] == "none"], v_n, "default", clean_nans=True)
    # coeffs_n, _, _, _ = o_fit.fit_curve(ff_name="poly2_q", cf_name="poly2")

    plot_dic = {
         "figsize": (6.0, 5.5),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "fit_panel": False,
        "plot_diagonal":False,
        "vmin": 350,  "vmax": 900.0, "cmap": "jet", "plot_cbar": False,  # "tab10",
        "xmin": 0.95, "xmax": 1.9,   "xscale": "linear",
        "ymin": 0,    "ymax": 0.41,   "yscale": "linear",
        "xlabel": "$q$",#r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ylabel": r"$M_{\rm disk}$ [$M_{\odot}$]",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "disk_q_mdisk.png",
        "legend": {"fancybox": False, "loc": 'upper left',
                   #"bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 13,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        "add_poly": [
            # {"x": np.arange(1., 2., 0.1), "coeffs": np.array(coeffs_l), "to_val": "10**", # 52.3 & 0.549
            # "color": "black", "lw": 0.8, "ls": "-.", "label": r"$P_2 ^{\rm tot}$ $\chi_{\nu}^2=$"+"{:.1f}".format(chi2)},
            #
            # {"x": np.arange(1., 2., 0.1), "coeffs": np.array(coeffs_m), # 36.3 & 0.238
            # "color": "blue", "lw": 0.8, "ls": "-.", "label": r"$P_2 ^{\rm M0/M1}$"},
            #
            # {"x": np.arange(1., 2., 0.1), "coeffs": np.array(coeffs_n), # 12.3 & 0.830
            # "color": "gray", "lw": 0.8, "ls": "-.", "label": r"$P_2 ^{\rm none}$"}
        ],
        # "add_legend":{"last_n":12,
        #               "fancybox": False, "loc": 'uppwer left',
        #               # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #               "shadow": "False", "ncol": 2, "fontsize": 12,
        #               "framealpha": 0., "borderaxespad": 0., "frameon": False
        #               },
        # "hline": {"y":5.220 * 1.e-3, "color":'blue', "linestyle":"dashed", "label":"Mean"},
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "show":True,
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


    plot_datasets_scatter3(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)

def task_plot_mdisk_all_Lambda():

    v_n = "Mdisk3D"
    # sel = md.simulations[~np.isnan(md.simulations[v_n])]
    # bibkeys = list(set(sel["bibkey"]))
    # datasets = OrderedDict()
    # for key in bibkeys:
    #     if not key in bibblacklist:
    #         datasets[key] = {"models":sel[sel["bibkey"]==key], "data":md, "fit":True,
    #                          "color":None,"marker":None,"ms":None,"label":None, "err":md.params.Mej_err}
    # assert len(datasets.keys()) > 0
    datasets = OrderedDict()
    datasets['reference'] = {"models": md.reference, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    datasets["radiceLK"] =  {"models": md.radice18lk, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    datasets["radiceM0"] =  {"models": md.radice18m0, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    datasets["lehner"] =    {"models": md.lehner16, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    datasets["kiuchi"] =    {"models": md.kiuchi19, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    datasets["vincent"] =   {"models": md.vincent19, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    datasets["dietrich16"] ={"models": md.dietrich16, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    datasets["dietrich15"] ={"models": md.dietrich15, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    datasets["sekiguchi15"]={"models": md.sekiguchi15, "data": md, "err": md.params.Mej_err, "label": r"Sekiguchi+2015",  "fit": True,"plot_errorbar": False}
    datasets["sekiguchi16"]={"models": md.sekiguchi16, "data": md, "err": md.params.Mej_err, "label": r"Sekiguchi+2016", "fit": True,"plot_errorbar": False}
    datasets["hotokezaka"] ={"models": md.hotokezaka12, "data": md, "err": md.params.Mej_err, "label": r"Hotokezaka+2013", "fit": True,"plot_errorbar": False}
    datasets["bauswein"] =  {"models": md.basuwein13, "data": md, "err": md.params.Mej_err, "label": r"Bauswein+2013", "fit": True,"plot_errorbar": False}

    label_whitelist = ['reference', "radiceLK", "radiceM0", "vincent", "sekiguchi16"] #  "sekiguchi15" "lehner"

    for t in datasets.keys():
        datasets[t]["ms"] = 60
        datasets[t]["label"] = md.datasets_labels[t]
        datasets[t]["marker"] = md.datasets_markers[t]
        datasets[t]["fill_style"] = "none"
        #datasets[t]["plot_errorbar"] = False
        #datasets[t]["plot_xerrorbar"] = False
        datasets[t]["facecolor"] = "none"
        datasets[t]["edgecolor"] = md.datasets_colors[t]
        #datasets[t]["plot_errorbar"] = False
        datasets[t]["labelmarkercolor"] = md.datasets_colors[t]
        if not t in label_whitelist:
            datasets[t]["label"] = None


    x_dic    = {"v_n": "Lambda", "err": None, "deferr": None, "mod": {}}
    y_dic    = {"v_n": "Mdisk3D",     "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}

    # fits for different neutrino schemes
    o_fit = Fit_Data(md.simulations, v_n, "default", clean_nans=True) # [md.simulations["nus"]=="leak"] md.simulations[md.simulations["nus"]!="leak"]
    coeffs_l, _, chi2, _ = o_fit.fit_curve(ff_name="poly2_Lambda", cf_name="poly2")#, modify="log10", )
    coeffs_d, _, chi2d, _ = o_fit.fit_curve(ff_name="rad18", cf_name="rad18")#, modify=None)
    # o_fit = Fit_Data(md.simulations[(md.simulations["nus"] == "leakM0")|
    #                                 (md.simulations["nus"] == "M1")|
    #                                 (md.simulations["nus"] == "leakM1")], v_n, "default", clean_nans=True)
    # coeffs_m, _, _, _ = o_fit.fit_curve(ff_name="poly2_Lambda", cf_name="poly2")
    # o_fit = Fit_Data(md.simulations[md.simulations["nus"] == "none"], v_n, "default", clean_nans=True)
    # coeffs_n, _, _, _ = o_fit.fit_curve(ff_name="poly2_Lambda", cf_name="poly2")

    plot_dic = {
        "figsize": (6.0, 5.5),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "fit_panel": False,
        "plot_diagonal":False,
        "vmin": 350,  "vmax": 900.0, "cmap": "jet", "plot_cbar": False,  # "tab10",
        "xmin": 0, "xmax": 2000,   "xscale": "linear",
        "ymin": 0,    "ymax": 0.41,   "yscale": "linear",
        "xlabel": r"$\tilde{\Lambda}$",#r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ylabel": r"$M_{\rm disk}$ [$M_{\odot}$]",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "disk_lambda_mdisk.png",
        "add_poly": [
            # {"x": np.arange(0, 1600, 100), "coeffs": np.array(coeffs_l), #"to_val":"10**",#"10**", # 625.8 & 0.125
            # "color": "black", "lw": 0.8, "ls": "--", "label": r"$P_2 ^{\rm tot}$ $\chi_{\nu}^2=$"+"{:.1f}".format(chi2)},
            # {"x": np.arange(0, 1600, 100), "coeffs": np.array(coeffs_d), "func":"rad18", #"to_val": None,#"10**",  # "10**", # 625.8 & 0.125
            #  "color": "black", "lw": 0.8, "ls": "-.", "label": r"$Eq.(R) ^{\rm tot}$ $\chi_{\nu}^2=$" + "{:.1f}".format(chi2d)},
            #
            # {"x": np.arange(0, 1600, 100), "coeffs": 1.e-2*np.array([-2.93319697e-02 , 1.63389243e-04 ,-4.63276134e-08]), # 71.6 & 0.031
            # "color": "red", "lw": 0.8, "ls": "-.", "label": r"$P_2 ^{Bernuzzi}$"},
            #
            # {"x": np.arange(0, 1600, 100), "coeffs": np.array(coeffs_n), # 96.8 & 0.228
            #  "color": "gray", "lw": 0.8, "ls": "-.", "label": r"$P_2 ^{\rm none}$"}
        ],
        # "add_legend":{"last_n":11,
        #               "fancybox": False, "loc": 'uppwer right',
        #               # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #               "shadow": "False", "ncol": 2, "fontsize": 12,
        #               "framealpha": 0., "borderaxespad": 0., "frameon": False
        #               },
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
        "show":True,
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

    plot_datasets_scatter3(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)

# mixed

def task_plot_mej_mdisk():

    v_n = "Mdisk3D"
    # sel = md.simulations[~np.isnan(md.simulations[v_n])]
    # bibkeys = list(set(sel["bibkey"]))
    # datasets = OrderedDict()
    # for key in bibkeys:
    #     if not key in bibblacklist:
    #         datasets[key] = {"models":sel[sel["bibkey"]==key], "data":md, "fit":True,
    #                          "color":None,"marker":None,"ms":None,"label":None, "err":md.params.Mej_err}
    # assert len(datasets.keys()) > 0
    datasets = OrderedDict()
    datasets['reference'] = {"models": md.reference, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    datasets["radiceLK"] =  {"models": md.radice18lk, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    datasets["radiceM0"] =  {"models": md.radice18m0, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    datasets["lehner"] =    {"models": md.lehner16, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    datasets["kiuchi"] =    {"models": md.kiuchi19, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    datasets["vincent"] =   {"models": md.vincent19, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    datasets["dietrich16"] ={"models": md.dietrich16, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    datasets["dietrich15"] ={"models": md.dietrich15, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    datasets["sekiguchi15"]={"models": md.sekiguchi15, "data": md, "err": md.params.Mej_err, "label": r"Sekiguchi+2015",  "fit": True,"plot_errorbar": False}
    datasets["sekiguchi16"]={"models": md.sekiguchi16, "data": md, "err": md.params.Mej_err, "label": r"Sekiguchi+2016", "fit": True,"plot_errorbar": False}
    datasets["hotokezaka"] ={"models": md.hotokezaka12, "data": md, "err": md.params.Mej_err, "label": r"Hotokezaka+2013", "fit": True,"plot_errorbar": False}
    datasets["bauswein"] =  {"models": md.basuwein13, "data": md, "err": md.params.Mej_err, "label": r"Bauswein+2013", "fit": True,"plot_errorbar": False}

    label_whitelist = ['reference', "radiceLK", "radiceM0", "vincent", "sekiguchi16"] #  "sekiguchi15" "lehner"

    for t in datasets.keys():
        datasets[t]["ms"] = 60
        datasets[t]["label"] = md.datasets_labels[t]
        datasets[t]["marker"] = md.datasets_markers[t]
        datasets[t]["fill_style"] = "none"
        #datasets[t]["plot_errorbar"] = False
        #datasets[t]["plot_xerrorbar"] = False
        datasets[t]["facecolor"] = "none"
        datasets[t]["edgecolor"] = md.datasets_colors[t]
        #datasets[t]["plot_errorbar"] = False
        datasets[t]["labelmarkercolor"] = md.datasets_colors[t]
        if not t in label_whitelist:
            datasets[t]["label"] = None


    x_dic    = {"v_n": "Mej_tot-geo", "err": None, "deferr": None, "mod": {}}
    y_dic    = {"v_n": "Mdisk3D",     "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}

    # fits for different neutrino schemes
    o_fit = Fit_Data(md.simulations, v_n, "default", clean_nans=True) # [md.simulations["nus"]=="leak"] md.simulations[md.simulations["nus"]!="leak"]
    coeffs_l, _, chi2, _ = o_fit.fit_curve(ff_name="poly2_Lambda", cf_name="poly2")#, modify="log10", )
    coeffs_d, _, chi2d, _ = o_fit.fit_curve(ff_name="rad18", cf_name="rad18")#, modify=None)
    # o_fit = Fit_Data(md.simulations[(md.simulations["nus"] == "leakM0")|
    #                                 (md.simulations["nus"] == "M1")|
    #                                 (md.simulations["nus"] == "leakM1")], v_n, "default", clean_nans=True)
    # coeffs_m, _, _, _ = o_fit.fit_curve(ff_name="poly2_Lambda", cf_name="poly2")
    # o_fit = Fit_Data(md.simulations[md.simulations["nus"] == "none"], v_n, "default", clean_nans=True)
    # coeffs_n, _, _, _ = o_fit.fit_curve(ff_name="poly2_Lambda", cf_name="poly2")

    plot_dic = {
        "figsize": (6.0, 5.5),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "fit_panel": False,
        "plot_diagonal":False,
        "vmin": 350,  "vmax": 900.0, "cmap": "jet", "plot_cbar": False,  # "tab10",
        "xmin": 1e-4, "xmax": 1e-1,   "xscale": "log",
        "ymin": 1e-3,    "ymax": 1e0,   "yscale": "log",
        "xlabel": r"$M_{\rm ej}$ [$M_{\odot}$]",#r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ylabel": r"$M_{\rm disk}$ [$M_{\odot}$]",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_mass_disk_mass.png",
        "add_poly": [
            # {"x": np.arange(0, 1600, 100), "coeffs": np.array(coeffs_l), #"to_val":"10**",#"10**", # 625.8 & 0.125
            # "color": "black", "lw": 0.8, "ls": "--", "label": r"$P_2 ^{\rm tot}$ $\chi_{\nu}^2=$"+"{:.1f}".format(chi2)},
            # {"x": np.arange(0, 1600, 100), "coeffs": np.array(coeffs_d), "func":"rad18", #"to_val": None,#"10**",  # "10**", # 625.8 & 0.125
            #  "color": "black", "lw": 0.8, "ls": "-.", "label": r"$Eq.(R) ^{\rm tot}$ $\chi_{\nu}^2=$" + "{:.1f}".format(chi2d)},
            #
            # {"x": np.arange(0, 1600, 100), "coeffs": 1.e-2*np.array([-2.93319697e-02 , 1.63389243e-04 ,-4.63276134e-08]), # 71.6 & 0.031
            # "color": "red", "lw": 0.8, "ls": "-.", "label": r"$P_2 ^{Bernuzzi}$"},
            #
            # {"x": np.arange(0, 1600, 100), "coeffs": np.array(coeffs_n), # 96.8 & 0.228
            #  "color": "gray", "lw": 0.8, "ls": "-.", "label": r"$P_2 ^{\rm none}$"}
        ],
        # "add_legend":{"last_n":11,
        #               "fancybox": False, "loc": 'uppwer right',
        #               # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #               "shadow": "False", "ncol": 2, "fontsize": 12,
        #               "framealpha": 0., "borderaxespad": 0., "frameon": False
        #               },
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
        "show":True,
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

    plot_datasets_scatter3(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)

def task_plot_mej_vs_vej_all():
    v_n = "Mej_tot-geo"
    # sel = md.simulations[~np.isnan(md.simulations[v_n])]
    # bibkeys = list(set(sel["bibkey"]))
    # datasets = OrderedDict()
    # for key in bibkeys:
    #     if not key in bibblacklist:
    #         datasets[key] = {"models":sel[sel["bibkey"]==key], "data":md, "fit":True,
    #                          "color":None,"marker":None,"ms":None,"label":None, "err":md.params.Mej_err}
    # assert len(datasets.keys()) > 0
    datasets = OrderedDict()
    datasets['reference'] = {"models": md.reference, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    datasets["radiceLK"] = {"models": md.radice18lk, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    datasets["radiceM0"] = {"models": md.radice18m0, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    datasets["lehner"] = {"models": md.lehner16, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    # # datasets["kiuchi"] = {"models": md.kiuchi19, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    datasets["vincent"] = {"models": md.vincent19, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    datasets["dietrich16"] = {"models": md.dietrich16, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    datasets["dietrich15"] = {"models": md.dietrich15, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    # # datasets["sekiguchi15"] = {"models": md.sekiguchi15, "data": md, "err": md.params.Mej_err, "label": r"Sekiguchi+2015", "fit": True, "plot_errorbar": False}
    datasets["sekiguchi16"] = {"models": md.sekiguchi16, "data": md, "err": md.params.Mej_err, "label": r"Sekiguchi+2016", "fit": True, "plot_errorbar": False}
    datasets["hotokezaka"] = {"models": md.hotokezaka12, "data": md, "err": md.params.Mej_err, "label": r"Hotokezaka+2013", "fit": True, "plot_errorbar": False}
    datasets["bauswein"] = {"models": md.basuwein13, "data": md, "err": md.params.Mej_err, "label": r"Bauswein+2013", "fit": True, "plot_errorbar": False}

    # ['reference', 'radiceLK', 'radiceM0', 'lehner', 'vincent', 'dietrich16',  'dietrich15', 'sekiguchi16', 'hotokezaka', 'bauswein']
    whitelist_labels1 = ["bauswein", "hotokezaka", "dietrich15", "dietrich16", "lehner"]
    for t in datasets.keys():
        datasets[t]["ms"] = 60
        datasets[t]["label"] = md.datasets_labels[t]
        datasets[t]["marker"] = md.datasets_markers[t]
        datasets[t]["fill_style"] = "none"
        # datasets[t]["plot_errorbar"] = False
        # datasets[t]["plot_xerrorbar"] = False
        datasets[t]["facecolor"] = "none"
        datasets[t]["edgecolor"] = md.datasets_colors[t]
        # datasets[t]["plot_errorbar"] = False
        datasets[t]["labelmarkercolor"] = md.datasets_colors[t]
        if not t in whitelist_labels1:
            datasets[t]["label"] = None

    y_dic    = {"v_n": "Mej_tot-geo",     "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    x_dic    = {"v_n": "vel_inf_ave-geo", "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (6.0, 5.5),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "fit_panel": False,
        "plot_diagonal":False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": False,  # "tab10",
        "xmin": 0.025, "xmax": 0.5, "xscale": "linear",
        "ymin": 1e-4, "ymax": 3e-1, "yscale": "log",
        "ylabel": r"$M_{\rm ej}$ $[M_{\odot}]$",#$[10^{-3}M_{\odot}]$",
        "xlabel": r"$\langle \upsilon_{\rm ej} \rangle$ [c]",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_mej_vej_all2.png",
        "legend": {"fancybox": False, "loc": 'upper right',
                   #"bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 12,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        # "add_legend": {"n_last",:4,
        #             "fancybox": False, "loc": 'upper left',
        #            # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #            "shadow": "False", "ncol": 1, "fontsize": 11.5,
        #            "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        "patch":
            [{"data":{"x":0.27,"y":0.016,"xerr":(0.2,0.3),"yerr":(1e-2,2e-2)}, # xerr - velocity   yerr - mass
             "plot":{"facecolor":"blue", "alpha":0.4, "edgecolor":"none", "label":None}},#"Blue kN"}},
             {"data": {"x": 0.1, "y": 0.05, "xerr": (0.07, 0.14), "yerr": (4e-2, 6e-2)}, # xerr - velocity   yerr - mass
              "plot": {"facecolor": "red", "alpha": 0.4, "edgecolor": "none", "label":None}},# "Red kN"}},
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
        "show":True,
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

    plot_datasets_scatter3(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)

def task_plot_mej_vs_ye_all():
    v_n = "Mej_tot-geo"
    # sel = md.simulations[~np.isnan(md.simulations[v_n])]
    # bibkeys = list(set(sel["bibkey"]))
    # datasets = OrderedDict()
    # for key in bibkeys:
    #     if not key in bibblacklist:
    #         datasets[key] = {"models":sel[sel["bibkey"]==key], "data":md, "fit":True,
    #                          "color":None,"marker":None,"ms":None,"label":None, "err":md.params.Mej_err}
    # assert len(datasets.keys()) > 0
    datasets = OrderedDict()
    datasets['reference'] = {"models": md.reference, "data": md, "fit": True, "plot_errorbar": False,  "err": md.params.Mej_err}
    datasets["radiceLK"] = {"models": md.radice18lk, "data": md, "fit": True, "plot_errorbar": False,"err": md.params.Mej_err}
    datasets["radiceM0"] = {"models": md.radice18m0, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    # # datasets["lehner"] = {"models": md.lehner16, "data": md, "fit": True, "plot_errorbar": True,"err": md.params.Mej_err}
    # # datasets["kiuchi"] = {"models": md.kiuchi19, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    datasets["vincent"] = {"models": md.vincent19, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    # # datasets["dietrich16"] = {"models": md.dietrich16, "data": md, "fit": True, "plot_errorbar": True,"err": md.params.Mej_err}
    datasets["dietrich15"] = {"models": md.dietrich15, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    # # datasets["sekiguchi15"] = {"models": md.sekiguchi15, "data": md, "err": md.params.Mej_err, "label": r"Sekiguchi+2015", "fit": True, "plot_errorbar": False}
    datasets["sekiguchi16"] = {"models": md.sekiguchi16, "data": md, "err": md.params.Mej_err, "label": r"Sekiguchi+2016", "fit": False, "plot_errorbar": False}
    # # datasets["hotokezaka"] = {"models": md.hotokezaka12, "data": md, "err": md.params.Mej_err, "label": r"Hotokezaka+2013", "fit": True, "plot_errorbar": False}
    # # datasets["bauswein"] = {"models": md.basuwein13, "data": md, "err": md.params.Mej_err, "label": r"Bauswein+2013", "fit": True, "plot_errorbar": False}

    # ['reference', 'radiceLK', 'radiceM0', 'lehner', 'vincent', 'dietrich16',  'dietrich15', 'sekiguchi16', 'hotokezaka', 'bauswein']
    whitelist_labels = ["sekiguchi15", "sekiguchi16", "radiceM0", "radiceLK", 'lehner', "vincent"]
    for t in datasets.keys():
        datasets[t]["ms"] = 60
        datasets[t]["label"] = md.datasets_labels[t]
        datasets[t]["marker"] = md.datasets_markers[t]
        datasets[t]["fill_style"] = "none"
        # datasets[t]["plot_errorbar"] = False
        # datasets[t]["plot_xerrorbar"] = False
        datasets[t]["facecolor"] = "none"
        datasets[t]["edgecolor"] = md.datasets_colors[t]
        # datasets[t]["plot_errorbar"] = False
        datasets[t]["labelmarkercolor"] = md.datasets_colors[t]
        if not t in whitelist_labels:
            datasets[t]["label"] = None

    y_dic    = {"v_n": "Mej_tot-geo",     "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    x_dic    = {"v_n": "Ye_ave-geo",      "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (6.0, 5.5),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "fit_panel": False,
        "plot_diagonal":False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": False,  # "tab10",
        "xmin": 0.01, "xmax": 0.35, "xscale": "linear",
        "ymin": 1e-4, "ymax": 3e-1, "yscale": "log",
        "ylabel": r"$M_{\rm ej}$ $[M_{\odot}]$",
        "xlabel": r"$\langle Y_{e;\:\rm ej} \rangle$",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_mej_yeej_all2.png",
        "legend": {"fancybox": False, "loc": 'lower right',
                   #"bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "add_legend": {"last_n":4,
                   "fancybox": False, "loc": 'upper right',
                   # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 12,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        "patch":
            [{"data": {"x": 0.27, "y": 0.016, "xerr": (0., 0.25), "yerr": (1e-2, 2e-2)}, # xerr - velocity   yerr - mass
              "plot": {"facecolor": "blue", "alpha": 0.4, "edgecolor": "none", "label": "Blue kN"}},
             {"data": {"x": 0.1, "y": 0.05, "xerr": (0.25, 0.50), "yerr": (4e-2, 6e-2)}, # xerr - velocity   yerr - mass
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
        "show":True,
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

    plot_datasets_scatter3(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)

def task_plot_vej_vs_ye_all():

    v_n = "Mej_tot-geo"
    # sel = md.simulations[~np.isnan(md.simulations[v_n])]
    # bibkeys = list(set(sel["bibkey"]))
    # datasets = OrderedDict()
    # for key in bibkeys:
    #     if not key in bibblacklist:
    #         datasets[key] = {"models":sel[sel["bibkey"]==key], "data":md, "fit":True,
    #                          "color":None,"marker":None,"ms":None,"label":None, "err":md.params.Mej_err}
    # assert len(datasets.keys()) > 0
    datasets = OrderedDict()
    datasets['reference'] = {"models": md.reference, "data": md, "fit": True, "plot_errorbar": False,  "err": md.params.Mej_err}
    datasets["radiceLK"] = {"models": md.radice18lk, "data": md, "fit": True, "plot_errorbar": False,"err": md.params.Mej_err}
    datasets["radiceM0"] = {"models": md.radice18m0, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    # # datasets["lehner"] = {"models": md.lehner16, "data": md, "fit": True, "plot_errorbar": True,"err": md.params.Mej_err}
    # # datasets["kiuchi"] = {"models": md.kiuchi19, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    datasets["vincent"] = {"models": md.vincent19, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    # # datasets["dietrich16"] = {"models": md.dietrich16, "data": md, "fit": True, "plot_errorbar": True,"err": md.params.Mej_err}
    # datasets["dietrich15"] = {"models": md.dietrich15, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    # # datasets["sekiguchi15"] = {"models": md.sekiguchi15, "data": md, "err": md.params.Mej_err, "label": r"Sekiguchi+2015", "fit": True, "plot_errorbar": False}
    datasets["sekiguchi16"] = {"models": md.sekiguchi16, "data": md, "err": md.params.Mej_err, "label": r"Sekiguchi+2016", "fit": True, "plot_errorbar": False}
    # # datasets["hotokezaka"] = {"models": md.hotokezaka12, "data": md, "err": md.params.Mej_err, "label": r"Hotokezaka+2013", "fit": True, "plot_errorbar": False}
    # # datasets["bauswein"] = {"models": md.basuwein13, "data": md, "err": md.params.Mej_err, "label": r"Bauswein+2013", "fit": True, "plot_errorbar": False}

    whitelist_labels2 = ['reference']
    for t in datasets.keys():
        datasets[t]["ms"] = 60
        datasets[t]["label"] = md.datasets_labels[t]
        datasets[t]["marker"] = md.datasets_markers[t]
        datasets[t]["fill_style"] = "none"
        # datasets[t]["plot_errorbar"] = False
        # datasets[t]["plot_xerrorbar"] = False
        datasets[t]["facecolor"] = "none"
        datasets[t]["edgecolor"] = md.datasets_colors[t]
        # datasets[t]["plot_errorbar"] = False
        datasets[t]["labelmarkercolor"] = md.datasets_colors[t]

        #datasets['reference']["label"] = r"Perego+2019, Nedora+2019,\\ Bernuzzi+2020, Nedora+2020"
        if not t in whitelist_labels2:
            datasets[t]["label"] = None

    y_dic    = {"v_n": "vel_inf_ave-geo", "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    x_dic    = {"v_n": "Ye_ave-geo",      "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (6.0, 5.5),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "fit_panel": False,
        "plot_diagonal":False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": False,  # "tab10",
        "xmin": 0.01, "xmax": 0.35, "xscale": "linear",
        "ymin": 0.1, "ymax": 0.38, "yscale": "linear",
        "ylabel": r"$\langle \upsilon_{\rm ej} \rangle$ [c]",
        "xlabel": r"$\langle Y_{e;\:\rm ej} \rangle$",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_vej_yeej_all2.png",
        "legend": {"fancybox": False, "loc": 'upper left',
                   #"bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 13,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "add_legend": {"last_n": 5,
                       "fancybox": False, "loc": 'upper right',
                       # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                       "shadow": "False", "ncol": 1, "fontsize": 12,
                       "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        "patch":
            [{"data": {"x": 0.27, "y": 0.016, "xerr": (0.0, 0.25), "yerr": (0.2, 0.3)}, # xerr - velocity   yerr - mass
              "plot": {"facecolor": "blue", "alpha": 0.4, "edgecolor": "none", "label":None}},#: "Blue kN"}},
             {"data": {"x": 0.1, "y": 0.05, "xerr": (0.25, 0.50), "yerr": (0.07, 0.14)}, # xerr - velocity   yerr - mass
              "plot": {"facecolor": "red", "alpha": 0.4, "edgecolor": "none", "label":None}},#: "Red kN"}},
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
        "show":True,
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

    plot_datasets_scatter3(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)

def task_plot_mej_vs_theta_all():
    v_n = "Mej_tot-geo"

    # -------------------------------------------
    datasets = OrderedDict()
    datasets['reference'] = {"models": md.reference, "data": md, "fit": True, "plot_errorbar": False,  "err": md.params.Mej_err}
    datasets["radiceLK"] = {"models": md.radice18lk, "data": md, "fit": True, "plot_errorbar": False,"err": md.params.Mej_err}
    datasets["radiceM0"] = {"models": md.radice18m0, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    # # datasets["lehner"] = {"models": md.lehner16, "data": md, "fit": True, "plot_errorbar": True,"err": md.params.Mej_err}
    # # datasets["kiuchi"] = {"models": md.kiuchi19, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    ## datasets["vincent"] = {"models": md.vincent19, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    # # datasets["dietrich16"] = {"models": md.dietrich16, "data": md, "fit": True, "plot_errorbar": True,"err": md.params.Mej_err}
    # datasets["dietrich15"] = {"models": md.dietrich15, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    # # datasets["sekiguchi15"] = {"models": md.sekiguchi15, "data": md, "err": md.params.Mej_err, "label": r"Sekiguchi+2015", "fit": True, "plot_errorbar": False}
    ## datasets["sekiguchi16"] = {"models": md.sekiguchi16, "data": md, "err": md.params.Mej_err, "label": r"Sekiguchi+2016", "fit": True, "plot_errorbar": False}
    # # datasets["hotokezaka"] = {"models": md.hotokezaka12, "data": md, "err": md.params.Mej_err, "label": r"Hotokezaka+2013", "fit": True, "plot_errorbar": False}
    # # datasets["bauswein"] = {"models": md.basuwein13, "data": md, "err": md.params.Mej_err, "label": r"Bauswein+2013", "fit": True, "plot_errorbar": False}

    for t in datasets.keys():
        datasets[t]["ms"] = 60
        datasets[t]["label"] = md.datasets_labels[t]
        datasets[t]["marker"] = md.datasets_markers[t]
        datasets[t]["fill_style"] = "none"
        # datasets[t]["plot_errorbar"] = False
        # datasets[t]["plot_xerrorbar"] = False
        datasets[t]["facecolor"] = "none"
        datasets[t]["edgecolor"] = md.datasets_colors[t]
        # datasets[t]["plot_errorbar"] = False
        datasets[t]["labelmarkercolor"] = md.datasets_colors[t]

    y_dic    = {"v_n": "Mej_tot-geo",     "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    x_dic    = {"v_n": "theta_rms-geo", "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (6.0, 5.5),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "fit_panel": False,
        "plot_diagonal":False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": False,  # "tab10",
        "xmin": 0, "xmax": 50, "xscale": "linear",
        "ymin": 1e-4, "ymax": 2e-2, "yscale": "log",
        "ylabel": r"$M_{\rm ej}$ $[M_{\odot}]$",#$[10^{-3}M_{\odot}]$",
        "xlabel": r"$\langle\theta_{\rm RMS}\rangle$ [deg]",
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
        "show": True,
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

    plot_datasets_scatter3(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)

def task_plot_vej_vs_theta_all():
    datasets = OrderedDict()
    datasets['reference'] = {"models": md.reference, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    datasets["radiceLK"] = {"models": md.radice18lk, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    datasets["radiceM0"] = {"models": md.radice18m0, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}

    for t in datasets.keys():
        datasets[t]["ms"] = 60
        datasets[t]["label"] = md.datasets_labels[t]
        datasets[t]["marker"] = md.datasets_markers[t]
        datasets[t]["fill_style"] = "none"
        # datasets[t]["plot_errorbar"] = False
        # datasets[t]["plot_xerrorbar"] = False
        datasets[t]["facecolor"] = "none"
        datasets[t]["edgecolor"] = md.datasets_colors[t]
        # datasets[t]["plot_errorbar"] = False
        datasets[t]["labelmarkercolor"] = md.datasets_colors[t]

    y_dic    = {"v_n": "vel_inf_ave-geo", "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    x_dic    = {"v_n": "theta_rms-geo",      "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (6.0, 5.5),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "fit_panel": False,
        "plot_diagonal":False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": False,  # "tab10",
        "xmin": 0, "xmax": 50, "xscale": "linear",
        "ymin": 0.1, "ymax": 0.35, "yscale": "linear",
        "ylabel": r"$\langle \upsilon_{\rm ej} \rangle$ [c]",
        "xlabel": r"$\langle\theta_{\rm RMS}\rangle$ [deg]",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_vej_theta_all2.png",
        # "legend": {"fancybox": False, "loc": 'upper right',
        #            #"bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #            "shadow": "False", "ncol": 1, "fontsize": 14,
        #            "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "savepdf":True,
        "show":True,
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

    plot_datasets_scatter3(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)

def task_plot_theta_vs_ye_all():

    datasets = OrderedDict()
    datasets['reference'] = {"models": md.reference, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    datasets["radiceLK"] = {"models": md.radice18lk, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}
    datasets["radiceM0"] = {"models": md.radice18m0, "data": md, "fit": True, "plot_errorbar": False, "err": md.params.Mej_err}

    for t in datasets.keys():
        datasets[t]["ms"] = 60
        datasets[t]["label"] = md.datasets_labels[t]
        datasets[t]["marker"] = md.datasets_markers[t]
        datasets[t]["fill_style"] = "none"
        # datasets[t]["plot_errorbar"] = False
        # datasets[t]["plot_xerrorbar"] = False
        datasets[t]["facecolor"] = "none"
        datasets[t]["edgecolor"] = md.datasets_colors[t]
        # datasets[t]["plot_errorbar"] = False
        datasets[t]["labelmarkercolor"] = md.datasets_colors[t]

    y_dic    = {"v_n": "theta_rms-geo", "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    x_dic    = {"v_n": "Ye_ave-geo",      "err": "ud", "deferr": 0.2,  "mod": {}}#"mod": {"mult": [1e3]}
    col_dic  = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (6.0, 5.5),
        "left" : 0.15, "bottom" : 0.14, "top" : 0.92, "right" : 0.95, "hspace" : 0,
        "dpi": 128, "fontsize": 14, "labelsize": 14,
        "fit_panel": False,
        "plot_diagonal":False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": False,  # "tab10",
        "xmin": 0.01, "xmax": 0.30, "xscale": "linear",
        "ymin": 0, "ymax": 50, "yscale": "linear",
        "ylabel": r"$\langle\theta_{\rm RMS}\rangle$ [deg]",
        "xlabel": r"$\langle Y_{e;\:\rm ej} \rangle$",
        "cbar_label": r"$\tilde{\Lambda}$",
        "figname": __outplotdir__ + "ej_theta_yeej_all2.png",
        # "legend": {"fancybox": False, "loc": 'upper left',
        #            #"bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #            "shadow": "False", "ncol": 1, "fontsize": 14,
        #            "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "savepdf":True,
        "show":True,
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

    plot_datasets_scatter3(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)

if __name__ == "__main__":

    ''' --- ejecta mass --- '''

    # task_plot_mej_all_q()
    # task_plot_mej_all_Lambda()

    ''' --- disk mass --- '''

    # task_plot_mdisk_all_q()
    # task_plot_mdisk_all_Lambda()

    ''' --- mixed --- '''
    task_plot_mej_mdisk()

    #task_plot_mej_vs_vej_all()
    #task_plot_mej_vs_ye_all()
    # task_plot_vej_vs_ye_all()
    #
    # task_plot_mej_vs_theta_all()
    # task_plot_vej_vs_theta_all()
    # task_plot_theta_vs_ye_all()
