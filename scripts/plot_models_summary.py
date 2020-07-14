from __future__ import division

import numpy as np
import pandas as pd
import sys
import os
import copy
import h5py
import csv

# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
rc('text', usetex=True)

from matplotlib.colors import LogNorm, Normalize


from uutils import x_y_z_sort

__outplotdir__ = "../figs/all3/plot_dynej_summary/"
if not os.path.isdir(__outplotdir__):
    os.mkdir(__outplotdir__)


def mscatter(x, y, ax=None, m=None, **kw):

    # for nan in data
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

def get_minmax(v_n, arr, extra = 2., oldtable=True):
    if v_n == "Lambda":
        if oldtable:
            min_, max_ = 0, 1500
        else:
            min_, max_ = 380, 880
    elif v_n == "k2T":
        if oldtable:
            min_, max_ = 0, 320
        else:
            min_, max_ = 50, 200
    elif v_n == "Mej_tot-geo" or v_n == "Mej":
        min_, max_ = 0, 2.
    elif v_n == "Mej_tot-geo_entropy_below_10":
        min_, max_ = 0, 1
    elif v_n == "Mej_tot-geo_entropy_above_10":
        min_, max_ = 0, 1
    elif v_n == "Ye_ave-geo" or v_n == "Yeej":
        min_, max_ = 0., 0.5
    elif v_n == "vel_inf_ave-geo" or v_n == "vej":
        min_, max_ = 0.1, 0.4
    elif v_n == "Ye_ave-geo":
        return 0., 0.4
    elif  v_n == "theta_rms-geo":
        return 0., 50
    elif v_n == "Mdisk3D":
        return 0., 0.4
    elif v_n == "Mdisk3Dmax":
        return 0., 0.4
    elif v_n == "q":
        return 0.9, 2.
    else:
        min_, max_ = np.array(arr).min(), np.array(arr).max() + (extra * (np.array(arr).max() - np.array(arr).min()))
        print("xlimits are not set for v_n_x:{}".format(v_n))
    print(v_n, min_, max_)
    return min_, max_

def get_label(v_n):
    if v_n == "q":
        return r"$M_a/M_b$"
    if v_n == "mtot":
        return r"$M_b + M_a$"
    if v_n == "mtot2":
        return r"$(M_b + M_a)^2$"
    if v_n == "Mej_tot-geo" or v_n == "Mej":
        return r"$M_{\rm{ej}}$ $[10^{-2}M_{\odot}]$"
    if v_n == "Lambda":
        return r"$\tilde{\Lambda}$"
    if v_n == "k2T":
        return r"$k_2^T$"
    if v_n == "mchirp":
        return r"$\mathcal{M}$"
    if v_n == "qmchirp" or v_n == "mchirpq":
        return r"$q \mathcal{M}$"
    if v_n == "q1mchirp":
        return r"$q / \mathcal{M}$"
    if v_n == "mchirp2":
        return r"$\mathcal{M} ^2$"
    if v_n == "Mej":
        return r"M_{\rm{ej}}"
    if v_n == "symq":
        return r"$\eta$"
    if v_n == "symq2":
        return r"$\eta^2$"
    if v_n == "q":
        return r"$q$"
    if v_n == "q1mtot":
        return r"$q/M_{\rm{tot}}$"
    if v_n == "qmtot":
        return r"$q M_{\rm{tot}}$"
    if v_n == "q2":
        return r"$q^2$"
    if v_n == "vel_inf_ave-geo" or v_n == "vej":
        return r"$v_{\rm ej}\ [c]$"
    if v_n == "symqmtot" or v_n == "mtotsymq":
        return r"$\eta M_{\rm{\tot}}$"
    if v_n == "symqmchirp":
        return r"$\eta\mathcal{M}$"
    if v_n == "mtotsymqmchirp":
        return r"$\eta M_{\rm{tot}}\mathcal{M}$"
    if v_n == "Mej_tot-geo_entropy_below_10"or v_n == "Mej_tidal":
        return r"$M_{\rm{ej;s<10}}$" # $[10^{-2}M_{\odot}]$
    if v_n == "Mej_tot-geo_entropy_above_10" or v_n == "Mej_shocked":
        return r"$M_{\rm{ej;s>10}}$" # $[10^{-2}M_{\odot}]$
    if v_n == "Ye_ave-geo":
        return r"$\langle Y_e \rangle$"
    if v_n == "theta_rms-geo":
        return r"$\theta_{\rm ej}$"
    if v_n == "Mdisk3D":
        return r"$M_{\rm{disk}}$ $[M_{\odot}]$"
    if v_n == "Mdisk3Dmax":
        return r"$M_{\rm{disk;max}}$ $[M_{\odot}]$"
    #
    elif str(v_n).__contains__("_mult_"):
        v_n1 = v_n.split("_mult_")[0]
        v_n2 = v_n.split("_mult_")[-1]
        lbl1 = get_label(v_n1)
        lbl2 = get_label(v_n2)
        return lbl1 + r"$\times$" + lbl2
    elif str(v_n).__contains__("_dev_"):
        v_n1 = v_n.split("_dev_")[0]
        v_n2 = v_n.split("_dev_")[-1]
        lbl1 = get_label(v_n1)
        lbl2 = get_label(v_n2)
        return lbl1 + r"$/$" + lbl2

    raise NameError("Np label for v_n: {}".format(v_n))

def make_plot_name(v_n_x, v_n_y, v_n_col):
    figname = ''
    figname = figname + v_n_x + '_'
    figname = figname + v_n_y + '_'
    figname = figname + v_n_col + '_'

    figname = figname + '.png'
    return figname

def plot_one_dataset_scatter(ax, x_dic, y_dic, col_dic, models_dic, plot_dic, fitting_func, fitting_coeffs):

    # -------------------- label

    marker = models_dic['marker']
    label = models_dic['label']
    ms = models_dic['ms']
    color = models_dic["color"]

    if color == None: color = "gray"
    else: pass

    if plot_dic["fit_panel"]: ax_ = ax[0]
    else: ax_ = ax
    ax_.scatter([-100], [-100], marker=marker, s=ms,
                color=color, alpha=1., edgecolor=None, label=label)

    # ---------------------- plot scatter

    d_cl = models_dic["data"]  # md, rd, ki ...
    err = models_dic["err"]  # err lambda(y)
    models = models_dic["models"]  # models DataFrame
    if "fit" in models_dic.keys() and models_dic["fit"] != None:
        do_fit = models_dic["fit"]
    else:
        do_fit = False

    v_n_col = d_cl.translation[col_dic["v_n"]]
    #
    edgecolors = None
    #
    markers = [models_dic['marker'] for sim in models.index]
    mss = [models_dic['ms'] for sim in models.index]

    x = d_cl.get_mod_data(x_dic["v_n"], x_dic["mod"], models)
    y = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models)
    c = np.array(models[v_n_col])

    # --- --- ---

    if err == "v_n":
        err = models["err-" + y_dic["v_n"]]
        err = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, err)
        # err = __apply_mod(y_dic["v_n"], err, y_dic["mod"])
    elif err == None:
        err = np.zeros(len(y))
    else:
        err = err(y)
        # err = __apply_mod(y_dic["v_n"], err, y_dic["mod"])
        err = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, err)

    if plot_dic["fit_panel"]:
        ax_ = ax[0]
    else:
        ax_ = ax

    cm = plt.cm.get_cmap(plot_dic["cmap"])
    norm = Normalize(vmin=plot_dic["vmin"], vmax=plot_dic["vmax"])

    ax_.errorbar(x, y, yerr=err, label=None,
                 color='gray', ecolor='gray',
                 fmt='None', elinewidth=1, capsize=1, alpha=0.5)

    if 'color' in models_dic.keys() and models_dic['color']!=None: c = models_dic['color']
    sc = mscatter(x, y, ax=ax_, c=c, norm=norm,
                  s=mss, cmap=cm, m=markers,
                  label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)

    if do_fit and plot_dic["fit_panel"]:
        # --- --- ---
        fitted_values = fitting_func(fitting_coeffs, models)
        if y_dic["v_n"] == "Mej_tot-geo": fitted_values = fitted_values / 1.e3  # Tims fucking fit
        #
        y_from_fit = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, fitted_values)
        y_ = (y - y_from_fit) / y
        delta_y = err / y
        print("error: Y ")
        print(y_)
        ax[1].errorbar(x, y_, yerr=delta_y, label=None,
                       color='gray', ecolor='gray',
                       fmt='None', elinewidth=1, capsize=1, alpha=0.5)
        #
        sc = mscatter(x, y_, ax=ax[1], c=c, norm=norm, s=mss, cmap=cm, m=markers,
                      label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)

    return sc, x, y, c

def plot_datasets_scatter(x_dic, y_dic, col_dic, plot_dic, fit_parameters, fitting_function, datasets):

    all_x = np.empty(1)
    all_y = np.empty(1)
    all_c = np.empty(1)

    if not "figname" in plot_dic.keys():
        figname = make_plot_name(x_dic["v_n"], y_dic["v_n"], col_dic["v_n"])
    else:
        figname = plot_dic["figname"]

    if plot_dic["fit_panel"]:
        fig, ax = plt.subplots(nrows=2, figsize=plot_dic["figsize"], sharex="all",
                               gridspec_kw={'height_ratios': [2, 1]})
        fig.subplots_adjust(left=0.15, bottom=0.12, top=0.95, right=0.95,
                            hspace=0)
    else:
        fig = plt.figure(figsize=plot_dic["figsize"])#(figsize=(7.2, 3.6))  # (nrows=1, figsize=[4.5, 3.5])
        ax = fig.add_subplot(111)

    # ------------ MAIN LOOP -----------
    sc = None
    for dataset_name in datasets.keys():
        dic = datasets[dataset_name]
        sc, x, y, c = plot_one_dataset_scatter(ax, x_dic, y_dic, col_dic,
                                               dic, plot_dic, fitting_function, fit_parameters)
        all_x = np.append(all_x, x)
        all_y = np.append(all_y, y)
        all_c = np.append(all_y, c)

    # ------------ ticks

    if plot_dic["fit_panel"]:ax_ = ax[0]
    else:  ax_ = ax
    if plot_dic["fit_panel"]:
        for i in range(len(ax)):
            ax[i].tick_params(axis='both', which='both', labelleft=True,
                              labelright=False, tick1On=True, tick2On=True,
                              labelsize=plot_dic["labelsize"], # 12
                              direction='in',
                              bottom=True, top=True, left=True, right=True)
            ax[i].minorticks_on()

            # limits
            if ("xmin" in plot_dic.keys() and "xmax" in plot_dic.keys()) and \
                    (plot_dic["xmin"] != None and plot_dic["xmax"] != None):
                min_, max_ = plot_dic["xmin"], plot_dic["xmax"]
            else:
                min_, max_ = get_minmax(x_dic["v_n"], all_x, extra=2.)
            ax[i].set_xlim(min_, max_)
    else:
        ax_.tick_params(axis='both', which='both', labelleft=True,
                          labelright=False, tick1On=True, tick2On=True,
                          labelsize=plot_dic["labelsize"], # 12
                          direction='in',
                          bottom=True, top=True, left=True, right=True)
        ax_.minorticks_on()

    # --------------- xlim ylim

    if ("ymin1" in plot_dic.keys() and "ymax1" in plot_dic.keys()) and \
            (plot_dic["ymin1"] != None and plot_dic["ymax1"] != None):
        min_, max_ = plot_dic["ymin1"], plot_dic["ymax1"]
    else:
        min_, max_ = get_minmax(y_dic["v_n"], all_y, extra=2., oldtable=True)
    ax_.set_ylim(min_, max_)
    #
    if plot_dic["fit_panel"]:
        if ("ymin2" in plot_dic.keys() and "ymax2" in plot_dic.keys()) and \
                (plot_dic["ymin2"] != None and plot_dic["ymax2"] != None):
            min_, max_ = plot_dic["ymin2"], plot_dic["ymax2"]
        else:
            min_, max_ = get_minmax(y_dic["v_n"], all_y, extra=2., oldtable=True)
        ax[1].set_ylim(min_, max_)

    if ("xmin" in plot_dic.keys() and "xmax" in plot_dic.keys()) and \
            (plot_dic["xmin"] != None and plot_dic["xmax"] != None):
        min_, max_ = plot_dic["xmin"], plot_dic["xmax"]
    else:
        min_, max_ = get_minmax(x_dic["v_n"], all_x, extra=1.)
    ax_.set_xlim(min_, max_)

    # --------------- scale

    if "xscale" in plot_dic.keys() and plot_dic["xscale"] != None:
        ax_.set_xscale(plot_dic["xscale"])
    if "yscale1" in plot_dic.keys() and plot_dic["yscale1"] != None:
        ax_.set_yscale(plot_dic["yscale1"])
    if plot_dic["fit_panel"]:
        if "yscale2" in plot_dic.keys() and plot_dic["yscale2"] != None:
            ax[1].set_yscale(plot_dic["yscale2"])

    # ------------------ xy labels

    if "ylabel1" in plot_dic.keys() and plot_dic["ylabel1"] != None: ax_.set_ylabel(plot_dic["ylabel1"], fontsize=plot_dic["fontsize"])
    else: ax_.set_ylabel(get_label(y_dic["v_n"]), fontsize=plot_dic["fontsize"])
    if plot_dic["fit_panel"]:
        if "ylabel2" in plot_dic.keys() and plot_dic["ylabel2"] != None: ax[1].set_ylabel(plot_dic["ylabel2"], fontsize=plot_dic["fontsize"])
        else: ax[1].set_ylabel(get_label(y_dic["v_n"]), fontsize=plot_dic["fontsize"])
        #
        if "xlabel" in plot_dic.keys() and plot_dic["xlabel"] != None: ax[1].set_xlabel(plot_dic["xlabel"], fontsize=plot_dic["fontsize"])
        else: ax[1].set_xlabel(get_label(x_dic["v_n"]), fontsize=plot_dic["fontsize"])
    else:
        if "xlabel" in plot_dic.keys() and plot_dic["xlabel"] != None: ax_.set_xlabel(plot_dic["xlabel"], fontsize=plot_dic["fontsize"])
        else: ax_.set_xlabel(get_label(x_dic["v_n"]), fontsize=plot_dic["fontsize"])

    # ----------------

    if plot_dic["fit_panel"]:
        ax[1].axhline(y=0, linestyle='-', linewidth=0.4, color='black')

    # ----------------

    ax_.legend(**plot_dic["legend"])
    if 'plot_cbar' in plot_dic.keys() and plot_dic['plot_cbar']:
        if plot_dic["fit_panel"]:
            clb = fig.colorbar(sc, ax=ax.ravel().tolist())
        else:
            clb = fig.colorbar(sc, ax=ax_)
            plt.tight_layout()
        clb.ax.set_title(get_label(col_dic["v_n"]), fontsize=plot_dic["fontsize"])
        clb.ax.tick_params(plot_dic["labelsize"])

    print("plotted: \n")
    print(__outplotdir__ + figname)
    if plot_dic["tight_layout"]: plt.tight_layout()
    plt.savefig(__outplotdir__ + figname, dpi=plot_dic["dpi"])
    plt.close()

def plot_datasets_scatter2(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets):
    #
    if len(fit_dic.keys())>0:
        fig, ax = plt.subplots(nrows=2, figsize=plot_dic["figsize"], sharex="all", gridspec_kw={'height_ratios': [2, 1]})
        fig.subplots_adjust(left=0.15, bottom=0.12, top=0.95, right=0.95,hspace=0)
        #
        # fig = plt.figure(figsize=plot_dic["figsize"])
        # ax = fig.subplots(nrows=2, ncols=1, sharex=True, sharey=False, gridspec_kw={'height_ratios': [2, 1]})
    else:
        ax = []
        fig = plt.figure(figsize=plot_dic["figsize"])  # (figsize=(7.2, 3.6))  # (nrows=1, figsize=[4.5, 3.5])
        ax.append(fig.add_subplot(111))

    ''' --- main loop ---'''
    sc = None
    for dataset_name in datasets.keys():
        model_dic = datasets[dataset_name]
        marker = model_dic['marker']
        label = model_dic['label']
        ms = model_dic['ms']
        color = model_dic["color"]
        #
        if color == None: color = "gray"

        ax[0].scatter([-100], [-100], marker=marker, s=ms, color=color, alpha=1., edgecolor=None, label=label)

        # ---------------------- plot scatter

        d_cl = model_dic["data"]  # md, rd, ki ...
        err = model_dic["err"]  # err lambda(y)
        models = model_dic["models"]  # models DataFrame
        #
        edgecolors = None
        markers = [model_dic['marker'] for sim in models.index]
        mss = [model_dic['ms'] for sim in models.index]
        if x_dic["v_n"] == "Mej_tot-geo_fit" and y_dic["v_n"] == "Mej_tot-geo":
            func, coeffs = fit_dic["func"], fit_dic["coeffs"]
            mej = func(coeffs, models) / 1.e3# 1e-3 Msun -> Msun
            x = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, mej)
            y = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models)
            c = np.array(models[d_cl.translation[col_dic["v_n"]]])
        elif x_dic["v_n"] == "vel_inf_ave-geo_fit" and y_dic["v_n"] == "vel_inf_ave-geo":
            func, coeffs = fit_dic["func"], fit_dic["coeffs"]
            mej = func(coeffs, models)# 1e-3 Msun -> Msun
            x = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, mej)
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
        #
        if err == "v_n":
            err = models["err-" + y_dic["v_n"]]
            err = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, err)
        elif err == None:
            err = np.zeros(len(y))
        else:
            err = err(y)
            # err = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, err)
        #
        cm = plt.cm.get_cmap(plot_dic["cmap"])
        norm = Normalize(vmin=plot_dic["vmin"], vmax=plot_dic["vmax"])
        ax[0].errorbar(x, y, yerr=err, label=None, color='gray', ecolor='gray', fmt='None', elinewidth=1, capsize=1, alpha=0.5)

        if 'color' in model_dic.keys() and model_dic['color'] != None: c = model_dic['color']
        sc = mscatter(x, y, ax=ax[0], c=c, norm=norm, s=mss, cmap=cm, m=markers, label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)
        # --- SECOND PANEL FOR FIT! ---
        if len(fit_dic.keys())>0 and model_dic["fit"]:
            # --- --- ---
            func, coeffs = fit_dic["func"], fit_dic["coeffs"]
            fitted_values = func(coeffs, models)
            #
            # if x_dic["v_n"] == "Mej_tot-geo_fit" and y_dic["v_n"] == "Mej_tot-geo": fitted_values = fitted_values  # Tims fucking fit
            if y_dic["v_n"] == "Mej_tot-geo": fitted_values = fitted_values / 1.e3  # Tims fucking fit
            #
            y_from_fit = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, fitted_values)
            y_ = (y - y_from_fit) / y
            delta_y = err / y
            print(" (y - y_from_fit) / y : ")
            print(y_)
            ax[1].errorbar(x, y_, yerr=delta_y, label=None, color='gray', ecolor='gray', fmt='None', elinewidth=1, capsize=1, alpha=0.5)
            #
            sc = mscatter(x, y_, ax=ax[1], c=c, norm=norm, s=mss, cmap=cm, m=markers, label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)
            #
    #

    ''' ----- subplot settings --- '''
    for axi, subplotdic in zip(ax, [plot_dic, fit_dic]):
        axi.set_yscale(subplotdic["yscale"])
        axi.set_xscale(subplotdic["xscale"])
        #
        axi.set_xlabel(subplotdic["xlabel"])  # , fontsize=11)
        axi.set_ylabel(subplotdic["ylabel"])  # , fontsize=11)
        #
        axi.set_xlim(subplotdic["xmin"], subplotdic["xmax"])
        axi.set_ylim(subplotdic["ymin"], subplotdic["ymax"])
        #
        axi.tick_params(axis='both', which='both', labelleft=True,
                       labelright=False, tick1On=True, tick2On=True,
                       labelsize=12,
                       direction='in',
                       bottom=True, top=True, left=True, right=True)
        axi.minorticks_on()
        #
        if "text" in subplotdic.keys():
            subplotdic["text"]["transform"] = ax.transAxes
            axi.text(**subplotdic["text"])
        #
        if "plot_zero" in subplotdic.keys(): axi.axhline(y=0, linestyle=':', linewidth=0.4, color='black')
        if "plot_diagonal" in subplotdic.keys(): axi.plot([0, 100], [0, 100], linestyle=':', linewidth=0.4, color='black',label="fit")
    #
    ax[0].legend(**plot_dic["legend"])
    #
    if 'plot_cbar' in plot_dic.keys() and plot_dic['plot_cbar']:
        if len(fit_dic.keys())>0:
            clb = fig.colorbar(sc, ax=ax.ravel().tolist())
        else:
            clb = fig.colorbar(sc, ax=ax[0])
            plt.tight_layout()
        clb.ax.set_title(get_label(col_dic["v_n"]))#, fontsize=plot_dic["fontsize"])
        #clb.ax.tick_params(plot_dic["labelsize"])
    #
    print("plotted: \n")
    print(__outplotdir__ + plot_dic["figname"])
    if plot_dic["tight_layout"]: plt.tight_layout()

    plt.savefig(__outplotdir__ + plot_dic["figname"], dpi=128)
    plt.close()

def summary_cumulative_plots2():
    """

    :return:

    """

    ''' ----------- DISK MASS --------- '''

    from model_sets import models_dietrich2016 as di
    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki
    from model_sets import models_bauswein2013 as bs
    from model_sets import models_lehner2016 as lh
    from model_sets import models_hotokezaka2013 as hz
    from model_sets import models_dietrich_ujevic2016 as du

    ''' ----------- '''

    datasets = {}
    datasets["kiuchi"] =    {'marker': "X", "ms": 20, "models": ki.simulations, "data":ki, "err":ki.params.Mdisk_err, "label":"Kiuchi+2019", "fit":True}
    datasets["radice"] =    {'marker': '*', "ms": 20, "models": rd.simulations[rd.fiducial], "data":rd, "err":rd.params.MdiskPP_err, "label":"Radice+2018", "fit":True}
    datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": di.simulations[di.mask_for_with_sr], "data":di, "err":di.params.Mdisk_err, "label":"Dietrich+2016", "fit":True}
    datasets["vincent"] =   {'marker': 'v', 'ms': 20, "models": vi.simulations, "data":vi, "err":vi.params.Mdisk_err, "label":"Vincent+2019", "fit":True} # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets['our'] =       {'marker': 'o', 'ms': 20, "models": md.groups, "data": md, "err":"v_n", "label":"Our Models", "fit":True}
    #
    x_dic = {"v_n": "q", "err": None, "mod": {}, "deferr": None}
    y_dic = {"v_n": "Mdisk3D", "err": "ud", "mod": {}, "deferr": 0.2}
    col_dic = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {"figsize":(4.5, 3.5),
                "fit_panel": True,
                "plot_cbar": True,
                "vmin": 350, "vmax": 900.0,
                # "vmin": 1., "vmax": 2.0,
                "cmap": "jet",  # "tab10",
                "label": None, "alpha": 0.8,
                "ms": 30.,
                "yscale1": None,
                "yscale2": None,
                # "xmin":1e-5, "xmax":1e-1,
                # "ymin":-0.1, "ymax":0.4,
                "ymin1": 0.0, "ymax1": 0.6,
                "ymin2": -1.4, "ymax2": 1.4,
                "xlabel": "$M_1/M_2$",
                "ylabel1": r"$M_{\rm disk}$",
                "ylabel2": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
                           "figname": "final_summary_disk_mass.png",
                "dpi":128,
                "tight_layout":False
                }

    # Fit
    from make_fit import fitting_function_mdisk, fitting_coeffs_mdisk
    x_davids_fit = fitting_coeffs_mdisk()
    fitting_func_of_lam = fitting_function_mdisk  # takes x_davids_fit, lam_array


    # plot_datasets_scatter(x_dic, y_dic, col_dic, plot_dic, x_davids_fit, fitting_func_of_lam, datasets)


    ''' ----------- EJECTA MASS --------- '''

    datasets = {}
    datasets["kiuchi"] =    {'marker': "X", "ms": 20, "models": ki.simulations, "data":ki, "err":ki.params.Mej_err, "label":"Kiuchi+2019", "color":"gray", "fit": False}
    datasets["radice"] =    {'marker': '*', "ms": 20, "models": rd.simulations[rd.fiducial], "data":rd, "err":rd.params.Mej_err, "label":"Radice+2018", "color":"blue", "fit": False}
    # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": di.simulations[di.mask_for_with_sr], "data":di, "err":di.params.Mej_err, "label":"Dietrich+2016"}
    datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data":du, "err": du.params.Mej_err, "label":"Dietrich+Ujevic+2016","color":"black",  "fit":False}
    datasets["vincent"] =   {'marker': 'v', 'ms': 20, "models": vi.simulations, "data":vi, "err": vi.params.Mej_err, "label":"Vincent+2019","color":"blue", "fit":False} # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets["bauswein"] =  {'marker': 's', 'ms': 20, "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label":"Bauswein+2013","color":"blue", "fit":False}
    datasets["lehner"] =    {'marker': 'P', 'ms': 20, "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": "Lehner+2016","color":"blue", "fit": False}
    datasets["hotokezaka"]= {'marker': '>', 'ms': 20, "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": "Hotokezaka+2013", "color":"gray", "fit": False}
    datasets['our'] =       {'marker': 'o', 'ms': 20, "models": md.groups, "data": md, "err":"v_n", "label":"Our Models", "color":"red", "fit":False}


    x_dic   = {"v_n": "q", "err": None, "mod": {}, "deferr": None}
    y_dic   = {"v_n": "Mej_tot-geo", "err": "ud", "deferr": 0.2, "mod":{}}
        # {
        #     "mult":[2.], "dev":["Mchirp"]
        # }}
    col_dic = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
                "figsize": (4.5, 3.5),
                "fit_panel": False,
                "plot_cbar": False,
                "tight_layout": True,
                "vmin": 350, "vmax": 900.0,
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
                "dpi": 128,
                }

    # Fit
    from make_fit import fitting_function_mej
    from make_fit import fitting_coeffs_mej
    x_davids_fit = fitting_coeffs_mej()  # radice()
    fitting_func_of_lam = fitting_function_mej  # takes x_davids_fit, lam_array


    plot_datasets_scatter(x_dic, y_dic, col_dic, plot_dic, x_davids_fit, fitting_func_of_lam, datasets)

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

    # plot_datasets_scatter(x_dic, y_dic, col_dic, plot_dic, x_davids_fit, fitting_func_of_lam, datasets)

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
    from make_fit import fitting_function_ye, fitting_coeffs_ye
    x_davids_fit = fitting_coeffs_ye()  # radice()
    fitting_func_of_lam = fitting_function_ye  # takes x_davids_fit, lam_array

    # plot_datasets_scatter(x_dic, y_dic, col_dic, plot_dic, x_davids_fit, fitting_func_of_lam, datasets)

def summary_fit_diviation_plots(x_dic, y_dic, col_dic, plot_dic, fit_dics, datasets):

    """

    :param x_dic:
    :param y_dic:
    :param col_dic:
    :param plot_dic:
    :param fit_dic:   {"name:{"func":f, "coeffs":c ...}, "nextname"...}
    :param datasets:
    :return:
    """

    fig = plt.figure(figsize=plot_dic["figsize"])
    list_axes=[]
    if len(fit_dics.keys())==1:
        list_axes.append(fig.add_subplot(111))
    else:
        list_axes = fig.subplots(nrows=len(fit_dics.keys()),ncols=1,sharex=True,sharey=False)
        # fig.subplots_adjust(left=0.15, bottom=0.12, top=0.95, right=0.95, hspace=0)
    #
    all_x = np.empty(1)
    all_y = np.empty(1)
    all_c = np.empty(1)

    ''' --- main plotting --- '''
    sc = None
    for dataset_name in datasets.keys():
        model_dic = datasets[dataset_name]
        marker = model_dic['marker']
        label = model_dic['label']
        ms = model_dic['ms']
        color = model_dic["color"]
        #
        if color == None: color = "gray"
        # -- LABELS ---
        if plot_dic["plot_label"]:
            list_axes[0].scatter([-100], [-100], marker=marker, s=ms, color=color, alpha=1., edgecolor=None, label=label)
        #
        d_cl    = model_dic["data"]        # md, rd, ki ...
        err     = model_dic["err"]          # err lambda(y)
        models  = model_dic["models"]    # models DataFrame
        # --- Extracting data for plotting points ---
        edgecolors = None
        markers = [model_dic['marker'] for sim in models.index]
        mss = [model_dic['ms'] for sim in models.index]
        x = d_cl.get_mod_data(x_dic["v_n"], x_dic["mod"], models)
        y = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models)
        c = np.array(models[d_cl.translation[col_dic["v_n"]]])
        # --- Getting Error. Either from the data or from lambda function
        if err == "v_n":
            err = models["err-" + y_dic["v_n"]] # if error is avilable in the data itself.
            err = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, err)
        elif err == None:
            err = np.zeros(len(y)) # If there is no Error
        else:
            err = err(y) # using lambda function
            err = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, err)
        # -- PLOT error bar
        cm = plt.cm.get_cmap(plot_dic["cmap"])
        norm = Normalize(vmin=plot_dic["vmin"], vmax=plot_dic["vmax"])

        # ax_.errorbar(x, y, yerr=err, label=None, color='gray', ecolor='gray',  fmt='None', elinewidth=1, capsize=1, alpha=0.5)

        if 'color' in model_dic.keys() and model_dic['color'] != None: c = model_dic['color']
        # sc = mscatter(x, y, ax=ax_, c=c, norm=norm, s=mss, cmap=cm, m=markers, label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)

        ''' ---------------- for every fitting function ------------------- '''
        for fit_key, ax in zip(fit_dics.keys(), list_axes):
            fit_dic = fit_dics[fit_key]
            func = fit_dic["func"]
            coeffs=fit_dic["coeffs"]
            fitted_values = func(coeffs, models) # models contain data as md.Mej or something
            if y_dic["v_n"] == "Mej_tot-geo": fitted_values = fitted_values / 1.e3 # since the ejecta fits are for 1e3 Msun
            y_from_fit = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models, fitted_values) #
            y_ = (y - y_from_fit) / y # normalize
            delta_y = err / y # for errorbars
            ax.errorbar(x, y_, yerr=delta_y, label=None, color='gray', ecolor='gray', fmt='None', elinewidth=1, capsize=1, alpha=0.5)
            sc = mscatter(x, y_, ax=ax, c=c, norm=norm, s=mss, cmap=cm, m=markers, label=None, alpha=plot_dic['alpha'], edgecolor=edgecolors)
        #

    ''' ----- subplot settings --- '''
    for fit_key, ax in zip(fit_dics.keys(), list_axes):
        subplotdic = fit_dics[fit_key]["plot"]
        ax.set_yscale(subplotdic["yscale"])
        ax.set_xscale(subplotdic["xscale"])

        ax.set_xlabel(subplotdic["xlabel"])  # , fontsize=11)
        ax.set_ylabel(subplotdic["ylabel"])  # , fontsize=11)

        ax.set_xlim(subplotdic["xmin"], subplotdic["xmax"])
        ax.set_ylim(subplotdic["ymin"], subplotdic["ymax"])
        #
        ax.tick_params(axis='both', which='both', labelleft=True,
                       labelright=False, tick1On=True, tick2On=True,
                       labelsize=12,
                       direction='in',
                       bottom=True, top=True, left=True, right=True)
        ax.minorticks_on()
        #
        if "text" in subplotdic.keys():
            subplotdic["text"]["transform"] = ax.transAxes
            ax.text(**subplotdic["text"])
        #
        if "plot_zero" in subplotdic.keys():
            ax.axhline(y=0, linestyle='-', linewidth=0.4, color='black')

    ''' --- overall --- '''
    if len(fit_dics.keys())>1:
        clb = fig.colorbar(sc, ax=list_axes.ravel().tolist(),  anchor=(1.4,0.0))
        # from mpl_toolkits.axes_grid1 import make_axes_locatable
        # ax_to_use = list_axes.ravel().tolist()
        # # divider = make_axes_locatable(ax_to_use)
        # pos1 = ax_to_use.get_position()
        # pos2 = [pos1.x0 + 0.1,
        #         pos1.y0 + 0,
        #         pos1.width,
        #         0.05]
        # # cax1 = divider.append_axes("right", size="5%", pad=0.05)
        #
        #
        # cax = plt.axes(pos2) # ``[left, bottom, width, height]``.
        # clb = plt.colorbar(sc, cax=cax)
    else:
        clb = fig.colorbar(sc, ax=list_axes[0])
    clb.ax.set_title(get_label(col_dic["v_n"]), fontsize=plot_dic["fontsize"])
    clb.ax.tick_params(plot_dic["labelsize"])
    #
    list_axes[0].legend(**plot_dic["legend"])

    if "wspace" in plot_dic.keys():
        plt.subplots_adjust(wspace=plot_dic["wspace"])
    if "hspace" in plot_dic.keys():
        plt.subplots_adjust(hspace=plot_dic["hspace"])

    if "tight_layout" in plot_dic.keys() and plot_dic["tight_layout"]:
        plt.tight_layout()
    print("plotted: \n")
    print(__outplotdir__ + plot_dic["figname"])
    if plot_dic["tight_layout"]: plt.tight_layout()
    plt.savefig(__outplotdir__ + plot_dic["figname"], dpi=plot_dic["dpi"])
    plt.close()

""" ---------------- tasks ------------------- """

def task_plot_ejecta_mass():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki
    from model_sets import models_bauswein2013 as bs
    from model_sets import models_lehner2016 as lh
    from model_sets import models_hotokezaka2013 as hz
    from model_sets import models_dietrich_ujevic2016 as du

    # -------------------------------------------

    datasets = {}
    datasets["kiuchi"] =    {'marker': "X", "ms": 20, "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019", "color": "gray", "fit": False}
    datasets["radice"] =    {'marker': '*', "ms": 20, "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018", "color": "blue", "fit": False}
    # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": di.simulations[di.mask_for_with_sr], "data":di, "err":di.params.Mej_err, "label":"Dietrich+2016"}
    datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}
    datasets["vincent"] =   {'marker': 'v', 'ms': 20, "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019", "color": "blue", "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets["bauswein"] =  {'marker': 's', 'ms': 20, "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "color": "blue", "fit": False}
    datasets["lehner"] =    {'marker': 'P', 'ms': 20, "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016", "color": "blue", "fit": False}
    datasets["hotokezaka"] ={'marker': '>', 'ms': 20, "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "color": "gray", "fit": False}
    datasets['our'] =       {'marker': 'o', 'ms': 40, "models": md.groups, "data": md, "err": "v_n", "label": r"Our Models", "color": "red", "fit": False}

    x_dic = {"v_n": "q", "err": None, "mod": {}, "deferr": None}
    y_dic = {"v_n": "Mej_tot-geo", "err": "ud", "deferr": 0.2, "mod": {}}
    col_dic = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (4.5, 3.5),
        "fit_panel": False,
        "plot_cbar": False,
        "tight_layout": True,
        "vmin": 350, "vmax": 900.0,
        # "vmin": 1., "vmax": 2.0,
        "cmap": "jet",  # "tab10",
        "label": None, "alpha": 0.8,
        "ms": 30.,
        # "xscale":"log",
        "yscale1": "log",
        "yscale2": None,
        "xmin":0.9, "xmax":2.1,
        # "ymin":-0.1, "ymax":0.4,
        "ymin1": 1e-4, "ymax1": 1e-1,
        "ymin2": -4.5, "ymax2": 2.4,
        "xlabel": "$M_1/M_2$",
        "ylabel1": r"$M_{\rm ej}$ $[M_{\odot}]$",
        "ylabel2": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
        "figname": "final_summary_ejecta_mass.png",

        "legend": {"fancybox": False, "loc": 'center',
                   "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 3, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},

        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "dpi": 128, "fontsize":12, "labelsize":12
    }

    # Fit
    from make_fit import fitting_function_mej
    from make_fit import fitting_coeffs_mej
    x_davids_fit = fitting_coeffs_mej()  # radice()
    fitting_func_of_lam = fitting_function_mej  # takes x_davids_fit, lam_array

    # plot_datasets_scatter(x_dic, y_dic, col_dic, plot_dic, x_davids_fit, fitting_func_of_lam, datasets)

    # -------------- colorcoding models with data ------------
    for k in datasets.keys():
        if k == "kiuchi": datasets["kiuchi"]["color"] = "gray"
        if k == "dietrich": datasets["dietrich"]["color"] = "gray"
        if k == "bauswein": datasets["bauswein"]["color"] = "gray"
        if k == "hotokezaka": datasets["hotokezaka"]["color"] = "gray"
        if k == "lehner": datasets["lehner"]["color"] = "gray"
        if k == "radice": datasets["radice"]["color"] = None
        if k == "vincent": datasets["vincent"]["color"] = None
        if k == "our": datasets["our"]["color"] = None
    # for k in datasets.keys():
    #     datasets[k]["alpha"]
    plot_dic_i = copy.deepcopy(plot_dic)
    plot_dic_i["figsize"] = (6., 4.)
    plot_dic_i["plot_cbar"] = True
    plot_dic_i["figname"] = "final_summary_ejecta_mass_colorcoded.png"

    plot_datasets_scatter(x_dic, y_dic, col_dic, plot_dic_i, x_davids_fit, fitting_func_of_lam, datasets)
def task_plot_ejecta_mass__with_fit():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki
    from model_sets import models_bauswein2013 as bs
    from model_sets import models_lehner2016 as lh
    from model_sets import models_hotokezaka2013 as hz
    from model_sets import models_dietrich_ujevic2016 as du

    # -------------------------------------------

    datasets = {}
    datasets["kiuchi"] =    {'marker': "X", "ms": 20, "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019", "color": "gray", "fit": False}
    datasets["radice"] =    {'marker': '*', "ms": 20, "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018", "color": "blue", "fit": True}
    # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": di.simulations[di.mask_for_with_sr], "data":di, "err":di.params.Mej_err, "label":"Dietrich+2016"}
    datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}
    datasets["vincent"] =   {'marker': 'v', 'ms': 20, "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019", "color": "blue", "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets["bauswein"] =  {'marker': 's', 'ms': 20, "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "color": "blue", "fit": False}
    datasets["lehner"] =    {'marker': 'P', 'ms': 20, "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016", "color": "blue", "fit": False}
    datasets["hotokezaka"] ={'marker': '>', 'ms': 20, "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "color": "gray", "fit": False}
    datasets['our'] =       {'marker': 'o', 'ms': 40, "models": md.groups, "data": md, "err": "v_n", "label": r"Our Models", "color": "red", "fit": True}

    x_dic = {"v_n": "q", "err": None, "mod": {}, "deferr": None}
    y_dic = {"v_n": "Mej_tot-geo", "err": "ud", "deferr": 0.2, "mod": {}}
    col_dic = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (4.5, 3.5),
        "fit_panel": True,
        "plot_cbar": False,
        "tight_layout": False,
        "vmin": 350, "vmax": 900.0,
        "cmap": "jet",  # "tab10",
        "label": None, "alpha": 0.8,
        "ms": 30.,
        "yscale1": "log",
        "yscale2": None,
        "xmin":0.9, "xmax":2.1,
        "ymin1": 1e-4, "ymax1": 1e-1,
        "ymin2": -4.5, "ymax2": 2.4,
        "xlabel": "$M_1/M_2$",
        "ylabel1": r"$M_{\rm ej}$ $[M_{\odot}]$",
        "ylabel2": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
        "figname": "final_summary_ejecta_mass.png",

        "legend": {"fancybox": False, "loc": 'center',
                   "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 3, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},

        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "dpi": 128, "fontsize":12, "labelsize":12
    }

    # Fit
    from make_fit import fitting_function_mej, fitting_coeffs_mej_david, fitting_coeffs_mej_our, complex_fic_data_mej_module

    x_davids_fit = fitting_coeffs_mej_david()# fitting_coeffs_mej()  # radice()
    fitting_func_of_lam = fitting_function_mej  # takes x_davids_fit, lam_array
    # complex_fic_data_mej_module(datasets, fitting_coeffs_mej_our(), key_for_usable_dataset="fit")

    # plot_datasets_scatter(x_dic, y_dic, col_dic, plot_dic, x_davids_fit, fitting_func_of_lam, datasets)

    # -------------- colorcoding models with data ------------
    for k in datasets.keys():
        if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
        if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
        if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
        if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
        if k == "lehner": datasets["lehner"]["color"]           = "gray"
        if k == "radice": datasets["radice"]["color"]           = None
        if k == "vincent": datasets["vincent"]["color"]         = None
        if k == "our": datasets["our"]["color"]                 = None
    # for k in datasets.keys():
    #     datasets[k]["alpha"]
    plot_dic_i = copy.deepcopy(plot_dic)
    plot_dic_i["figsize"] = (6., 4.)
    plot_dic_i["plot_cbar"] = True
    plot_dic_i["figname"] = "final_summary_ejecta_mass_colorcoded_with_fit.png"

    # plot_datasets_scatter(x_dic, y_dic, col_dic, plot_dic_i, x_davids_fit, fitting_func_of_lam, datasets)
def task_plot_ejecta_mass_fit_vs_mass_ejecta():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki
    from model_sets import models_bauswein2013 as bs
    from model_sets import models_lehner2016 as lh
    from model_sets import models_hotokezaka2013 as hz
    from model_sets import models_dietrich_ujevic2016 as du

    # -------------------------------------------

    datasets = {}
    # datasets["kiuchi"] =    {'marker': "X", "ms": 20, "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019", "color": "gray", "fit": False}
    datasets["radice"] =    {'marker': '*', "ms": 20, "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018", "color": "blue", "fit": True}
    # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": di.simulations[di.mask_for_with_sr], "data":di, "err":di.params.Mej_err, "label":"Dietrich+2016"}
    # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}
    # datasets["vincent"] =   {'marker': 'v', 'ms': 20, "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019", "color": "blue", "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    # datasets["bauswein"] =  {'marker': 's', 'ms': 20, "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "color": "blue", "fit": False}
    # datasets["lehner"] =    {'marker': 'P', 'ms': 20, "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016", "color": "blue", "fit": False}
    # datasets["hotokezaka"] ={'marker': '>', 'ms': 20, "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "color": "gray", "fit": False}
    datasets['our'] =       {'marker': 'o', 'ms': 40, "models": md.groups, "data": md, "err": "v_n", "label": r"Our Models", "color": "red", "fit": True}

    x_dic    = {"v_n": "Mej_tot-geo_fit", "err": None, "deferr": None, "mod": {"mult": [1e3]}}
    y_dic    = {"v_n": "Mej_tot-geo",     "err": "ud", "deferr": 0.2,  "mod": {"mult": [1e3]}}
    col_dic = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (4.5, 3.5),
        "fit_panel": True,
        "plot_diagonal":True,
        "tight_layout": False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": True,  # "tab10",
        "xmin":1, "xmax":7.5, "xscale": "linear",
        "ymin": 0, "ymax": 14, "yscale": "linear",
        "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ylabel": r"$M_{\rm ej}$ $[10^{-3}M_{\odot}]$",
        "figname": "final_summary_ejecta_mass_fit_vs_ejecta_mass.png",
        "legend": {"fancybox": False, "loc": 'upper left',
                   #"bbox_to_anchor": (0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "dpi": 128, "fontsize":12, "labelsize":12
    }

    # fit
    from make_fit import fitting_function_mej, fitting_coeffs_mej_our
    fit_dic = {
        "func":fitting_function_mej, "coeffs": fitting_coeffs_mej_our(),
        "xmin": 0, "xmax": 7.5, "xscale": "linear", "xlabel": r"$M_{\rm ej;fit}$ $[10^{-3}M_{\odot}]$",
        "ymin": -5.0, "ymax": 3.0, "yscale": "linear", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$",
        "plot_zero":True
    }

    # Fit
    # from make_fit import fitting_function_mej
    # from make_fit import fitting_coeffs_mej_david # fitting_coeffs_mej
    # from make_fit import complex_fic_data_mej_module
    # x_davids_fit = fitting_coeffs_mej_david()# fitting_coeffs_mej()  # radice()
    # fitting_func_of_lam = fitting_function_mej  # takes x_davids_fit, lam_array
    # complex_fic_data_mej_module(datasets, key_for_usable_dataset="fit")

    # plot_datasets_scatter(x_dic, y_dic, col_dic, plot_dic, x_davids_fit, fitting_func_of_lam, datasets)

    # -------------- colorcoding models with data ------------
    for k in datasets.keys():
        if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
        if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
        if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
        if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
        if k == "lehner": datasets["lehner"]["color"]           = "gray"
        if k == "radice": datasets["radice"]["color"]           = None
        if k == "vincent": datasets["vincent"]["color"]         = None
        if k == "our": datasets["our"]["color"]                 = None

    plot_datasets_scatter2(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)
def task_plot_ejecta_mass_fits():
    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki
    from model_sets import models_bauswein2013 as bs
    from model_sets import models_lehner2016 as lh
    from model_sets import models_hotokezaka2013 as hz
    from model_sets import models_dietrich_ujevic2016 as du

    # -------------------------------------------

    datasets = {}
    # datasets["kiuchi"] = {'marker': "X", "ms": 20, "models": ki.simulations, "data": ki, "err": ki.params.Mej_err,  "label": r"Kiuchi+2019", "color": "gray", "fit": False}
    datasets["radice"] = {'marker': '*', "ms": 20, "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Mej_err, "label": r"Radice+2018", "color": "blue", "fit": True}
    # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": di.simulations[di.mask_for_with_sr], "data":di, "err":di.params.Mej_err, "label":"Dietrich+2016"}
    # datasets["dietrich"] = {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err,  "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}
    # datasets["vincent"] = {'marker': 'v', 'ms': 20, "models": vi.simulations, "data": vi, "err": vi.params.Mej_err, "label": r"Vincent+2019", "color": "blue", "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    # datasets["bauswein"] = {'marker': 's', 'ms': 20, "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "color": "blue", "fit": False}
    # datasets["lehner"] = {'marker': 'P', 'ms': 20, "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016", "color": "blue", "fit": False}
    # datasets["hotokezaka"] = {'marker': '>', 'ms': 20, "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "color": "gray", "fit": False}
    datasets['our'] = {'marker': 'o', 'ms': 40, "models": md.groups, "data": md, "err": "v_n", "label": r"Our Models", "color": "red", "fit": True}

    x_dic = {"v_n": "q", "err": None, "mod": {}, "deferr": None}
    y_dic = {"v_n": "Mej_tot-geo", "err": "ud", "deferr": 0.2, "mod": {}}
    col_dic = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    # {"x":0.3, "y":0.95, "s":r"Dietrich\&Ujevic+2016", "ha":"center", "va":"top",  "fontsize":11, "color":"white", "transform":None}
    from make_fit import fitting_function_mej
    from make_fit import fitting_coeffs_mej_david
    from make_fit import fitting_coeffs_mej_tim
    from make_fit import fitting_coeffs_mej_our

    #porn = { "good": { "lesbo": {"bettwer" : {"a":1, "b":2} } } }
    # exit(1)
    fit_dics = {
        "radice": {"func": fitting_function_mej, "coeffs": fitting_coeffs_mej_david(), "plot":
            {"xmin": 0.9, "xmax": 2.1, "ymin": -4.5, "ymax": 4.5, "xlabel": "$M_1/M_2$",
             "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$", "xscale": "linear", "yscale": "linear", "text":
                 {"x": 0.3, "y": 0.95, "s": r"Radice+2018", "ha": "center", "va": "top", "fontsize": 11,
                  "color": "black", "transform": None}, "plot_zero": True
             }},
        "dietrich":{"func":fitting_function_mej, "coeffs":fitting_coeffs_mej_tim(), "plot":
            {"xmin": 0.9, "xmax": 2.1, "ymin": -4.5, "ymax": 4.5, "xlabel": "$M_1/M_2$", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$", "xscale":"linear", "yscale":"linear", "text":
                 {"x":0.3, "y":0.95, "s":r"Dietrich\&Ujevic+2016", "ha":"center", "va":"top",  "fontsize":11, "color":"black", "transform":None}, "plot_zero": True
             }},
        "Our": {"func": fitting_function_mej, "coeffs": fitting_coeffs_mej_our(), "plot":
            {"xmin": 0.9, "xmax": 2.1, "ymin": -4.5, "ymax": 4.5, "xlabel": "$M_1/M_2$", "ylabel": r"$\Delta M_{\rm ej} / M_{\rm ej}$","xscale":"linear", "yscale":"linear", "text":
                 {"x": 0.3, "y": 0.95, "s": r"Our Fit", "ha": "center", "va": "top", "fontsize": 11, "color": "black", "transform": None}, "plot_zero": True
             }}
    }

    #
    plot_dic = {
        "figsize": (6.5, 4.5), #
        "tight_layout": False,
        "vmin": 350, "vmax": 900.0,
        "cmap": "jet",  # "tab10",
        "label": None, "alpha": 0.8,
        "plot_label": True,
        "legend": {"fancybox": False, "loc": 'upper right',
                   #"bbox_to_anchor": (0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 3, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},

        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "dpi": 128, "fontsize": 12, "labelsize": 12, "hspace":0,
        "figname": "final_summary_ejecta_mass_colorcoded_fits.png"
    }

    # -------------- colorcoding models with data ------------
    for k in datasets.keys():
        if k == "kiuchi": datasets["kiuchi"]["color"] = "gray"
        if k == "dietrich": datasets["dietrich"]["color"] = "gray"
        if k == "bauswein": datasets["bauswein"]["color"] = "gray"
        if k == "hotokezaka": datasets["hotokezaka"]["color"] = "gray"
        if k == "lehner": datasets["lehner"]["color"] = "gray"
        if k == "radice": datasets["radice"]["color"] = None
        if k == "vincent": datasets["vincent"]["color"] = None
        if k == "our": datasets["our"]["color"] = None

    summary_fit_diviation_plots(x_dic, y_dic, col_dic, plot_dic, fit_dics, datasets)
    #
    from make_fit import complex_fic_data_mej_module
    complex_fic_data_mej_module(datasets, fitting_coeffs_mej_our(), key_for_usable_dataset="fit")

def task_plot_ejecta_velocity():

    ''' ----------- EJECTA VELOCITY --------- '''

    from model_sets import models_dietrich2016 as di
    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki
    from model_sets import models_bauswein2013 as bs
    from model_sets import models_lehner2016 as lh
    from model_sets import models_hotokezaka2013 as hz
    from model_sets import models_dietrich_ujevic2016 as du

    # -------------------------------------------


    datasets = {}
    # datasets["kiuchi"] =    {'marker': "X", "ms": 20, "models": ki.simulations, "data": ki, "err": ki.params.vej_err, "label": "Kiuchi+2019", "color": "gray", "fit": False}
    datasets["radice"] =    {'marker': '*', "ms": 20, "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.vej_err, "label": r"Radice+2018", "color": "blue", "fit": True}
    # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": di.simulations[di.mask_for_with_sr], "data":di, "err":di.params.Mej_err, "label":"Dietrich+2016"}
    datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.vej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}
    datasets["vincent"] =   {'marker': 'v', 'ms': 20, "models": vi.simulations, "data": vi, "err": vi.params.vej_err,"label": r"Vincent+2019", "color": "blue", "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets["bauswein"] =  {'marker': 's', 'ms': 20, "models": bs.simulations, "data": bs, "err": bs.params.vej_err, "label": r"Bauswein+2013", "color": "blue", "fit": False}
    datasets["lehner"] =    {'marker': 'P', 'ms': 20, "models": lh.simulations, "data": lh, "err": lh.params.vej_err, "label": r"Lehner+2016", "color": "blue", "fit": False}
    datasets["hotokezaka"] ={'marker': '>', 'ms': 20, "models": hz.simulations, "data": hz, "err": hz.params.vej_err, "label": r"Hotokezaka+2013", "color": "gray", "fit": False}
    datasets['our'] =       {'marker': 'o', 'ms': 20, "models": md.groups, "data": md, "err": "v_n", "label": r"Our Models", "color": "red", "fit": True}

    # datasets = {}
    # # datasets["kiuchi"] =    {'marker': "X", "ms": 20, "models": ki.simulations, "data":ki, "err":ki.params.Mej_err, "label":"Kiuchi+2019"}
    # datasets["radice"] = {'marker': '*', "ms": 20, "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.vej_err, "label": "Radice+2018"}
    # datasets["dietrich"] = {'marker': 'd', "ms": 20, "models": di.simulations[di.mask_for_with_sr], "data": di, "err": di.params.vej_err, "label": "Dietrich+2016"}
    # datasets["vincent"] = {'marker': 'v', 'ms': 20, "models": vi.simulations, "data": vi, "err": vi.params.vej_err, "label": "Vincent+2019"}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    # datasets['our'] = {'marker': 'o', 'ms': 20, "models": md.groups, "data": md, "err": "v_n", "label": "We+inf"}

    x_dic = {"v_n": "q", "err": None, "mod": {}, "deferr": None}
    y_dic = {"v_n": "vel_inf_ave-geo", "err": "ud", "deferr": 0.2, "mod": {}}
    # {
    #     "mult":[2.], "dev":["Mchirp"]
    # }}
    col_dic = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
                "figsize": (4.5, 3.5),
                "fit_panel": False,
                "plot_cbar": False,
                "tight_layout": True,
                "vmin": 350, "vmax": 900.0,
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
                "ylabel1": r"$\langle \upsilon_{\rm ej} \rangle$ [c]",
                "ylabel2": r"$\Delta \upsilon_{\rm ej} / \upsilon_{\rm ej}$",
                "figname": "final_summary_ejecta_velovity.png",
                "legend": {"fancybox": True, "loc": 'upper right',
                           # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                           "shadow": False, "ncol": 1, "fontsize": 9,
                           "framealpha": 0., "borderaxespad": 0.},
                "dpi": 128, "fontsize":12, "labelsize":12
                }

    # Fit
    from make_fit import fitting_coeffs_vinf_our, fitting_function_vinf, complex_fic_data_vinf_module
    x_davids_fit = fitting_coeffs_vinf_our()  # radice()
    fitting_func_of_lam = fitting_function_vinf  # takes x_davids_fit, lam_array
    complex_fic_data_vinf_module(datasets, fitting_coeffs_vinf_our(), "fit")

    # plot_datasets_scatter(x_dic, y_dic, col_dic, plot_dic, x_davids_fit, fitting_func_of_lam, datasets)

    # -------------- colorcoding models with data ------------

    for k in datasets.keys():
        if k == "dietrich": datasets["dietrich"]["color"] = "gray"
        if k == "bauswein": datasets["bauswein"]["color"] = "gray"
        if k == "hotokezaka": datasets["hotokezaka"]["color"] = "gray"
        if k == "lehner": datasets["lehner"]["color"] = "gray"
        if k == "radice": datasets["radice"]["color"] = None
        if k == "vincent": datasets["vincent"]["color"] = None
        if k == "our": datasets["our"]["color"] = None
    # for k in datasets.keys():
    #     datasets[k]["alpha"]
    plot_dic_i = copy.deepcopy(plot_dic)
    plot_dic_i["figsize"] = (6., 4.)
    plot_dic_i["plot_cbar"] = True
    plot_dic_i["xmin"], plot_dic_i["xmax"] = 0.9, 1.9
    plot_dic_i["legend"] = {"fancybox": False, "loc": 'center',
                           "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                           "shadow": "False", "ncol": 3, "fontsize": 10,
                           "framealpha": 0., "borderaxespad": 0., "frameon": False}
    plot_dic_i["figname"] = "final_summary_ejecta_velovity_colorcoded.png"

    # plot_datasets_scatter(x_dic, y_dic, col_dic, plot_dic_i, x_davids_fit, fitting_func_of_lam, datasets)
def task_plot_ejecta_vel_vs_vel_fit():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki
    from model_sets import models_bauswein2013 as bs
    from model_sets import models_lehner2016 as lh
    from model_sets import models_hotokezaka2013 as hz
    from model_sets import models_dietrich_ujevic2016 as du

    # -------------------------------------------

    datasets = {}
    # datasets["kiuchi"] =    {'marker': "X", "ms": 20, "models": ki.simulations, "data": ki, "err": ki.params.Mej_err, "label": r"Kiuchi+2019", "color": "gray", "fit": False}
    datasets["radice"] =    {'marker': '*', "ms": 20, "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.vej_err, "label": r"Radice+2018", "color": "blue", "fit": True}
    # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": di.simulations[di.mask_for_with_sr], "data":di, "err":di.params.Mej_err, "label":"Dietrich+2016"}
    # datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": du.simulations, "data": du, "err": du.params.Mej_err, "label": r"Dietrich\&Ujevic+2016", "color": "black", "fit": False}
    # datasets["vincent"] =   {'marker': 'v', 'ms': 20, "models": vi.simulations, "data": vi, "err": vi.params.Mej_err,"label": r"Vincent+2019", "color": "blue", "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    # datasets["bauswein"] =  {'marker': 's', 'ms': 20, "models": bs.simulations, "data": bs, "err": bs.params.Mej_err, "label": r"Bauswein+2013", "color": "blue", "fit": False}
    # datasets["lehner"] =    {'marker': 'P', 'ms': 20, "models": lh.simulations, "data": lh, "err": lh.params.Mej_err, "label": r"Lehner+2016", "color": "blue", "fit": False}
    # datasets["hotokezaka"] ={'marker': '>', 'ms': 20, "models": hz.simulations, "data": hz, "err": hz.params.Mej_err, "label": r"Hotokezaka+2013", "color": "gray", "fit": False}
    datasets['our'] =       {'marker': 'o', 'ms': 40, "models": md.groups, "data": md, "err": "v_n", "label": r"Our Models", "color": "red", "fit": True}

    x_dic = {"v_n": "vel_inf_ave-geo_fit", "err": None, "deferr": None, "mod": {}}
    y_dic = {"v_n": "vel_inf_ave-geo",     "err": "ud", "deferr": 0.2,  "mod": {}}
    col_dic = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (4.5, 3.5),
        "fit_panel": True,
        "plot_diagonal":True,
        "tight_layout": False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": True,  # "tab10",
        "xmin":0.1, "xmax":.3, "xscale": "linear",
        "ymin": 0.1, "ymax": .3, "yscale": "linear",
        "xlabel": r"$\upsilon_{\rm ej;fit}$ [c]",
        "ylabel": r"$\upsilon_{\rm ej}$ [c]",
        "figname": "final_summary_ejecta_vel_fit_vs_ejecta_vel.png",
        "legend": {"fancybox": False, "loc": 'upper left',
                   #"bbox_to_anchor": (0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "dpi": 128, "fontsize":12, "labelsize":12
    }

    # fit
    from make_fit import complex_fic_data_vinf_module, fitting_function_vinf, fitting_coeffs_vinf_our
    fit_dic = {
        "func":fitting_function_vinf, "coeffs": fitting_coeffs_vinf_our(),
        "xmin": 0.1, "xmax": .3, "xscale": "linear", "xlabel": r"$\upsilon_{\rm ej;fit}$ [c]",
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
    for k in datasets.keys():
        if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
        if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
        if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
        if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
        if k == "lehner": datasets["lehner"]["color"]           = "gray"
        if k == "radice": datasets["radice"]["color"]           = None
        if k == "vincent": datasets["vincent"]["color"]         = None
        if k == "our": datasets["our"]["color"]                 = None

    plot_datasets_scatter2(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)

    complex_fic_data_vinf_module(datasets, fitting_coeffs_vinf_our, key_for_usable_dataset="fit")

def task_plot_ejecta_ye():

    from model_sets import models_dietrich2016 as di
    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki
    from model_sets import models_bauswein2013 as bs
    from model_sets import models_lehner2016 as lh
    from model_sets import models_hotokezaka2013 as hz
    from model_sets import models_dietrich_ujevic2016 as du

    # -------------------------------------------

    datasets = {}
    datasets["radice"] =    {'marker': '*', "ms": 20, "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Yeej_err, "label": "Radice+2018", "color": "blue", "fit": True}
    datasets["vincent"] =   {'marker': 'v', 'ms': 20, "models": vi.simulations, "data": vi, "err": vi.params.Yeej_err,"label": "Vincent+2019", "color": "blue", "fit": False}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets['our'] =       {'marker': 'o', 'ms': 20, "models": md.groups, "data": md, "err": "v_n", "label": "Our Models", "color": "red", "fit": True}

    x_dic = {"v_n": "q", "err": None, "mod": {}, "deferr": None}
    y_dic = {"v_n": "Ye_ave-geo", "err": "ud", "deferr": 0.2, "mod": {}}

    col_dic = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
                "figsize": (4.5, 3.5),
                "fit_panel": False,
                "plot_cbar": False,
                "tight_layout": True,
                "vmin": 350, "vmax": 900.0,
                # "vmin": 1., "vmax": 2.0,
                "cmap": "jet",  # "tab10",
                "label": None, "alpha": 0.8,
                "ms": 30.,
                # "xscale":"log",
                "yscale1": None,
                "yscale2": None,
                # "xmin":1e-5, "xmax":1e-1,
                # "ymin":-0.1, "ymax":0.4,
                "ymin1": 0.0, "ymax1": 0.3,
                "ymin2": -1.0, "ymax2": 0.8,
                "xlabel": "$M_1/M_2$",
                "ylabel1": r"$\langle Y_{e} \rangle$",
                "ylabel2": r"$\Delta Ye_{\rm ej} / Ye_{\rm ej}$",
                "figname": "final_summary_ejecta_electron_fraction.png",
                "legend": {"fancybox": True, "loc": 'upper right',
                           # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                           "shadow": False, "ncol": 1, "fontsize": 9,
                           "framealpha": 0., "borderaxespad": 0.},
                "dpi": 128, "fontsize":12, "labelsize":12
                }

    # Fit
    from make_fit import fitting_function_ye, fitting_coeffs_ye_our_david, complex_fic_data_ye_module
    x_davids_fit = fitting_coeffs_ye_our_david()  # radice()
    fitting_func_of_lam = fitting_function_ye  # takes x_davids_fit, lam_array

    complex_fic_data_ye_module(datasets, fitting_coeffs_ye_our_david(), "fit")

    # plot_datasets_scatter(x_dic, y_dic, col_dic, plot_dic, x_davids_fit, fitting_func_of_lam, datasets)

    # -------------- colorcoding models with data ------------

    for k in datasets.keys():
        if k == "radice": datasets["radice"]["color"] = None
        if k == "vincent": datasets["vincent"]["color"] = None
        if k == "our": datasets["our"]["color"] = None

    plot_dic_i = copy.deepcopy(plot_dic)
    plot_dic_i["figsize"] = (6., 4)
    plot_dic_i["plot_cbar"] = True
    plot_dic_i["xmin"], plot_dic_i["xmax"] = 0.9, 1.9
    plot_dic_i["legend"] = {"fancybox": False, "loc": 'center',
                           "bbox_to_anchor": (0.5, 1.05),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                           "shadow": "False", "ncol": 3, "fontsize": 10,
                           "framealpha": 0., "borderaxespad": 0., "frameon": False}
    plot_dic_i["figname"] = "final_summary_ejecta_electron_fraction_colorcoded.png"

    #plot_datasets_scatter(x_dic, y_dic, col_dic, plot_dic_i, x_davids_fit, fitting_func_of_lam, datasets)
def task_plot_ejecta_ye_vs_ye_fit():

    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md

    # -------------------------------------------

    datasets = {}
    datasets["radice"] =  {'marker': '*', "ms": 20, "models": rd.simulations[rd.fiducial], "data": rd, "err": rd.params.Yeej_err, "label": r"Radice+2018", "color": "blue", "fit": True}
    datasets["vincent"] = {'marker': 'v', 'ms': 20, "models": vi.simulations, "data": vi, "err": vi.params.Yeej_err, "label": r"Vincent+2019", "color": "blue", "fit": True}  # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets['our'] =     {'marker': 'o', 'ms': 20, "models": md.groups, "data": md, "err": "v_n", "label": r"This work", "color": "red", "fit": True}

    x_dic   = {"v_n": "Ye_ave-geo_fit", "err": None, "deferr": None, "mod": {}}
    y_dic   = {"v_n": "Ye_ave-geo",     "err": "ud", "deferr": 0.2,  "mod": {}}
    col_dic = {"v_n": "Lambda",         "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
        "figsize": (4.5, 3.5),
        "fit_panel": True,
        "plot_diagonal":True,
        "tight_layout": False,
        "vmin": 350, "vmax": 900.0, "cmap": "jet", "plot_cbar": True,  # "tab10",
        "xmin":0.1, "xmax":.3, "xscale": "linear",
        "ymin": 0.1, "ymax": .3, "yscale": "linear",
        "xlabel": r"$\langle Y_e \rangle _{\rm ej;fit}$",
        "ylabel": r"$Ye_{\rm ej}$",
        "figname": "final_summary_ejecta_ye_fit_vs_ejecta_ye.png",
        "legend": {"fancybox": False, "loc": 'upper left',
                   #"bbox_to_anchor": (0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "alpha":0.8,
        # "legend":{"fancybox":True, "loc":'upper right',
        #        # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
        #        "shadow":False, "ncol":2, "fontsize":9,
        #        "framealpha":0., "borderaxespad":0.},
        "dpi": 128, "fontsize":12, "labelsize":12
    }

    # fit
    from make_fit import complex_fic_data_ye_module, fitting_function_ye, fitting_coeffs_ye_our_david
    fit_dic = {
        "func":fitting_function_ye, "coeffs": fitting_coeffs_ye_our_david(),
        "xmin": 0.1, "xmax": .3, "xscale": "linear", "xlabel": r"$Ye_{\rm ej;fit}$",
        "ymin": -1.0, "ymax": 0.8, "yscale": "linear", "ylabel": r"$\Delta Ye_{\rm ej} / Ye_{\rm ej}$",
        "plot_zero":True
    }

    # Fit
    # from make_fit import fitting_function_mej
    # from make_fit import fitting_coeffs_mej_david # fitting_coeffs_mej
    # from make_fit import complex_fic_data_vinf_module
    # x_davids_fit = fitting_coeffs_mej_david()# fitting_coeffs_mej()  # radice()
    # fitting_func_of_lam = fitting_function_mej  # takes x_davids_fit, lam_array


    # # # plot_datasets_scatter(x_dic, y_dic, col_dic, plot_dic, x_davids_fit, fitting_func_of_lam, datasets)

    # -------------- colorcoding models with data ------------
    for k in datasets.keys():
        if k == "kiuchi": datasets["kiuchi"]["color"]           = "gray"
        if k == "dietrich": datasets["dietrich"]["color"]       = "gray"
        if k == "bauswein": datasets["bauswein"]["color"]       = "gray"
        if k == "hotokezaka": datasets["hotokezaka"]["color"]   = "gray"
        if k == "lehner": datasets["lehner"]["color"]           = "gray"
        if k == "radice": datasets["radice"]["color"]           = None
        if k == "vincent": datasets["vincent"]["color"]         = None
        if k == "our": datasets["our"]["color"]                 = None

    plot_datasets_scatter2(x_dic, y_dic, col_dic, plot_dic, fit_dic, datasets)

    #complex_fic_data_vinf_module(datasets, fitting_coeffs_vinf_our, key_for_usable_dataset="fit")

def task_plot_disk_mass():
    ''' ----------- DISK MASS --------- '''

    from model_sets import models_dietrich2016 as di
    from model_sets import models_vincent2019 as vi
    from model_sets import models_radice2018 as rd
    from model_sets import groups as md
    from model_sets import models_kiuchi2019 as ki
    from model_sets import models_bauswein2013 as bs
    from model_sets import models_lehner2016 as lh
    from model_sets import models_hotokezaka2013 as hz
    from model_sets import models_dietrich_ujevic2016 as du

    datasets = {}
    datasets["kiuchi"] =    {'marker': "X", "ms": 20, "models": ki.simulations, "data":ki, "err":ki.params.Mdisk_err, "label":"Kiuchi+2019", "color":"gray", "fit": False}
    datasets["radice"] =    {'marker': '*', "ms": 20, "models": rd.simulations[rd.fiducial], "data":rd, "err":rd.params.MdiskPP_err, "label":"Radice+2018", "color":"blue", "fit": False}
    datasets["dietrich"] =  {'marker': 'd', "ms": 20, "models": di.simulations[di.mask_for_with_sr], "data":di, "err":di.params.Mej_err, "label":"Dietrich+2016", "color":"black",  "fit":False}
    datasets["vincent"] =   {'marker': 'v', 'ms': 20, "models": vi.simulations, "data":vi, "err": vi.params.Mdisk_err, "label":"Vincent+2019","color":"blue", "fit":False} # di.simulations[di.mask_for_with_sr & di.mask_for_with_disk]
    datasets['our'] =       {'marker': 'o', 'ms': 20, "models": md.groups, "data": md, "err":"v_n", "label":"Our Models", "color":"red", "fit":False}

    #
    x_dic = {"v_n": "q", "err": None, "mod": {}, "deferr": None}
    y_dic = {"v_n": "Mdisk3D", "err": "ud", "mod": {}, "deferr": 0.2}
    col_dic = {"v_n": "Lambda", "err": None, "mod": {}, "deferr": None}
    #
    plot_dic = {
                "figsize": (4.5, 3.5),
                "fit_panel": False,
                "plot_cbar": False,
                "tight_layout": True,
                "vmin": 350, "vmax": 900.0,
                # "vmin": 1., "vmax": 2.0,
                "cmap": "jet",  # "tab10",
                "label": None, "alpha": 0.8,
                "ms": 30.,
                "yscale1": None,
                "yscale2": None,
                # "xmin":1e-5, "xmax":1e-1,
                # "ymin":-0.1, "ymax":0.4,
                "ymin1": 0.0, "ymax1": 0.5,
                "ymin2": -1.4, "ymax2": 1.4,
                "xlabel": "$M_1/M_2$",
                "ylabel1": r"$M_{\rm disk}$",
                "ylabel2": r"$\Delta M_{\rm disk} / M_{\rm disk}$",
                "figname": "final_summary_disk_mass.png",
                "legend": {"fancybox": True, "loc": 'upper center',
                           # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                           "shadow": False, "ncol": 2, "fontsize": 9,
                           "framealpha": 0., "borderaxespad": 0.},
                "dpi": 128, "fontsize":12, "labelsize":12
                }

    # Fit
    from make_fit import fitting_function_mdisk, fitting_coeffs_mdisk
    x_davids_fit = fitting_coeffs_mdisk()
    fitting_func_of_lam = fitting_function_mdisk  # takes x_davids_fit, lam_array

    # plot_datasets_scatter(x_dic, y_dic, col_dic, plot_dic, x_davids_fit, fitting_func_of_lam, datasets)

    # -------------- colorcoding models with data ------------

    for k in datasets.keys():
        if k == "kiuchi": datasets["kiuchi"]["color"] = "gray"
        if k == "radice": datasets["radice"]["color"] = None
        if k == "vincent": datasets["vincent"]["color"] = None
        if k == "our": datasets["our"]["color"] = None

    plot_dic_i = copy.deepcopy(plot_dic)
    plot_dic_i["figsize"] = (6., 4)
    plot_dic_i["plot_cbar"] = True
    plot_dic_i["xmin"], plot_dic_i["xmax"] = 0.9, 1.9
    plot_dic_i["legend"] = {"fancybox": False, "loc": 'center',
                           "bbox_to_anchor": (0.5, 1.1),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                           "shadow": "False", "ncol": 3, "fontsize": 10,
                           "framealpha": 0., "borderaxespad": 0., "frameon": False}
    plot_dic_i["figname"] = "final_summary_disk_mass_colorcoded.png"

    plot_datasets_scatter(x_dic, y_dic, col_dic, plot_dic_i, x_davids_fit, fitting_func_of_lam, datasets)

    # -------------- colorcoding models with data ------------

    for k in datasets.keys():
        if k == "kiuchi": datasets["kiuchi"]["color"] = "gray"; datasets["kiuchi"]["fit"] = True
        if k == "radice": datasets["radice"]["color"] = None;  datasets["radice"]["fit"] = True
        if k == "vincent": datasets["vincent"]["color"] = None;  datasets["vincent"]["fit"] = True
        if k == "our": datasets["our"]["color"] = None;  datasets["our"]["fit"] = True

    plot_dic_i = copy.deepcopy(plot_dic)
    plot_dic_i["fit_panel"] = True
    plot_dic_i["tight_layout"] = False
    plot_dic_i["ymin1"], plot_dic_i["ymax1"] = 0., 0.6
    # plot_dic_i["figsize"] = (6., 4)
    plot_dic_i["plot_cbar"] = True
    plot_dic_i["xmin"], plot_dic_i["xmax"] = 0.9, 1.9
    plot_dic_i["legend"] = {"fancybox": False, "loc": 'center',
                           "bbox_to_anchor": (0.5, 0.85),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                           "shadow": "False", "ncol": 2, "fontsize": 10,
                           "framealpha": 0., "borderaxespad": 0., "frameon": False}
    plot_dic_i["figname"] = "final_summary_disk_mass_colorcoded_with_fit.png"

    plot_datasets_scatter(x_dic, y_dic, col_dic, plot_dic_i, x_davids_fit, fitting_func_of_lam, datasets)

if __name__ == "__main__":

    ''' --- Mej --- '''
    task_plot_ejecta_mass()
    # task_plot_ejecta_mass__with_fit()
    # task_plot_ejecta_mass_fit_vs_mass_ejecta()
    # task_plot_ejecta_mass_fits()

    ''' --- vej --- '''
    # task_plot_ejecta_velocity()
    # task_plot_ejecta_vel_vs_vel_fit()

    ''' --- Ye --- '''
    # task_plot_ejecta_ye()
    # task_plot_ejecta_ye_vs_ye_fit()

    ''' --- Mdisk --- '''
    # task_plot_disk_mass()



    # print("hi")
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

    # summary_cumulative_plots2()