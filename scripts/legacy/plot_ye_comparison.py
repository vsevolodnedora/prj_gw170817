# ---
#
# compares and plots Ye histogams/correlations
#
# ---
from __future__ import division

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import numpy as np
import pandas as pd
import sys
import os
import copy
import h5py
import csv
from scipy import interpolate
sys.path.append('/data01/numrel/vsevolod.nedora/bns_ppr_tools/')
from preanalysis import LOAD_INIT_DATA
from outflowed import EJECTA_PARS
from preanalysis import LOAD_ITTIME
from plotting_methods import PLOT_MANY_TASKS
from profile import LOAD_PROFILE_XYXZ, LOAD_RES_CORR, LOAD_DENSITY_MODES
from utils import Paths, Lists, Labels, Constants, Printcolor, UTILS, Files, PHYSICS
from data import *
from tables import *
from settings import simulations, old_simulations, resolutions

__outplotdir__ = "/data01/numrel/vsevolod.nedora/figs/all3/ye_comparison/"

def plot_correlation():

    sims = ['DD2_M13641364_M0_SR_R04', 'DD2_M13641364_M0_LK_SR_R04']

    v_ns = ["Y_e", "Y_e"]
    det = 0
    masks = ["geo", "geo"]

    #

    dic_data = {}
    for sim, v_n, mask in zip(sims, v_ns, masks):
        o = ADD_METHODS_ALL_PAR(sim)
        corr_table = o.get_outflow_corr(det, mask, "{}_{}".format(v_n, "theta"))
        dic_data[sim] = {}
        dic_data[sim]["data"] = corr_table.T
        hist = o.get_outflow_hist(det, mask, "theta")
        dic_data[sim]['hist'] = hist.T
        print(hist.T.shape)
    Printcolor.green("data in collected")

    #
    # for sim in sims:
    #     dic_data[sim]['data'] /= np.sum(dic_data[sim]['data'][1:, 1:])

    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = __outplotdir__
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (8., 5.6)  # <->, |]
    o_plot.gen_set["figname"] = "comparison.png".format("Ye")
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = False
    o_plot.gen_set["dpi"] = 128
    o_plot.gen_set["subplots_adjust_h"] = 0.0
    o_plot.gen_set["subplots_adjust_w"] = 0.0
    o_plot.set_plot_dics = []

    i = 1
    for sim in sims:
        # HISTOGRAMS
        plot_dic = {
            'task': 'hist1d', 'ptype': 'cartesian',
            'position': (1, i),
            'data': dic_data[sim]["hist"], 'normalize': True,
            'v_n_x': "theta", 'v_n_y': "mass",
            'color': "black", 'ls': '-', 'lw': 1., 'ds': 'steps', 'alpha': 1.0,
            'xmin': 0., 'xamx': 90., 'ymin': 1e-4, 'ymax': 1e0,
            'xlabel': Labels.labels("theta"), 'ylabel': r"$M_{\rm{ej}}/M$",
            'label': None, 'yscale': 'log',
            'fancyticks': True, 'minorticks': True,
            'fontsize': 14,
            'labelsize': 14,
            'sharex': True,  # removes angular citkscitks
            'sharey': False,
            'title':{'text':sim.replace('_', '\_'), 'fontsize':12},
            # 'textold': {'coords': (0.1, 0.1), 'text':sim.replace('_', '\_'), 'color': 'black', 'fs': 10},
            'legend': {}  # 'loc': 'best', 'ncol': 2, 'fontsize': 18
        }
        # if v_n == "tempterature":
        if i!=1:
            plot_dic['sharey'] = True

        o_plot.set_plot_dics.append(plot_dic)
        # CORRELATIONS
        corr_dic2 = {  # relies on the "get_res_corr(self, it, v_n): " method of data object
            'task': 'corr2d', 'dtype': 'corr', 'ptype': 'cartesian',
            'data': dic_data[sim]['data'],
            'position': (2, i),
            'v_n_x': 'theta', 'v_n_y': 'ye', 'v_n': 'mass', 'normalize': True,
            'cbar': {},
            'cmap': 'viridis', #'set_under': 'white', #'set_over': 'black',
            'xlabel': r"Angle from binary plane", 'ylabel': r"$Y_e$",
            'xmin': 0., 'xmax': 90., 'ymin': 0.05, 'ymax': 0.5, 'vmin': 1e-4, 'vmax': 1e-2,
            'xscale': "linear", 'yscale': "linear", 'norm': 'log',
            'mask_below': None, 'mask_above': None,
            'title': {},  # {"text": o_corr_data.sim.replace('_', '\_'), 'fontsize': 14},
            'fancyticks': True,
            'minorticks': True,
            'sharex': False,  # removes angular citkscitks
            'sharey': False,
            'fontsize': 14,
            'labelsize': 14,
        }
        corr_dic2["axhline"] = {"y": 0.25, "linestyle": "-", "linewidth": 0.5, "color": "black"}
        # corr_dic2["axvline"] = {"x": 0.25, "linestyle": "-", "linewidth": 0.5, "color": "black"}
        if i!=1:
            corr_dic2['sharey'] = True
        if sim == sims[-1]:
            corr_dic2['cbar'] = \
                {'location': 'right .03 .0', 'label': r"$M_{\rm{ej}}/M$",  # 'fmt': '%.1f',
                'labelsize': 14, 'fontsize': 14}
        o_plot.set_plot_dics.append(corr_dic2)
        i = i + 1
    o_plot.main()

def plot_correlation_from_two_datadirs():

    sims = ['SFHo_M135135_LK', 'SFHo_M135135_M0', "SFHo_M13641364_M0_LK_SR"]
    rowdata = ["/data1/numrel/WhiskyTHC/Backup/2017/",
                 "/data1/numrel/WhiskyTHC/Backup/2017/",
                 "/data1/numrel/WhiskyTHC/Backup/2018/GW170817/"]
    datapaths = ["/data01/numrel/vsevolod.nedora/postprocessed_radice2/",
                 "/data01/numrel/vsevolod.nedora/postprocessed_radice2/",
                 "/data01/numrel/vsevolod.nedora/postprocessed4/"]
    v_ns = ["Y_e", "Y_e", "Y_e"]
    det = 0
    masks = ["geo", "geo", "geo"]

    #

    dic_data = {}
    for sim, v_n, mask, path in zip(sims, v_ns, masks, datapaths):
        Paths.ppr_sims = path
        o = ADD_METHODS_ALL_PAR(sim)
        corr_table = o.get_outflow_corr(det, mask, "{}_{}".format(v_n, "theta"))
        dic_data[sim] = {}
        dic_data[sim]["data"] = corr_table.T
        hist = o.get_outflow_hist(det, mask, "theta")
        dic_data[sim]['hist'] = hist.T
        print(hist.T.shape)
    Printcolor.green("data in collected")

    #
    # for sim in sims:
    #     dic_data[sim]['data'] /= np.sum(dic_data[sim]['data'][1:, 1:])

    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = __outplotdir__
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (4.2*len(sims), 4.)  # <->, |]
    o_plot.gen_set["figname"] = "comparison.png".format("Ye")
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = False
    o_plot.gen_set["dpi"] = 128
    o_plot.gen_set["subplots_adjust_h"] = 0.0
    o_plot.gen_set["subplots_adjust_w"] = 0.0
    o_plot.set_plot_dics = []

    i = 1
    for sim in sims:
        # HISTOGRAMS
        plot_dic = {
            'task': 'hist1d', 'ptype': 'cartesian',
            'position': (1, i),
            'data': dic_data[sim]["hist"], 'normalize': True,
            'v_n_x': "theta", 'v_n_y': "mass",
            'color': "black", 'ls': '-', 'lw': 1., 'ds': 'steps', 'alpha': 1.0,
            'xmin': 0., 'xamx': 90., 'ymin': 1e-4, 'ymax': 1e0,
            'xlabel': Labels.labels("theta"), 'ylabel': r"$M_{\rm{ej}}/M$",
            'label': None, 'yscale': 'log',
            'fancyticks': True, 'minorticks': True,
            'fontsize': 14,
            'labelsize': 14,
            'sharex': True,  # removes angular citkscitks
            'sharey': False,
            'title':{'text':sim.replace('_', '\_'), 'fontsize':12},
            # 'textold': {'coords': (0.1, 0.1), 'text':sim.replace('_', '\_'), 'color': 'black', 'fs': 10},
            'legend': {}  # 'loc': 'best', 'ncol': 2, 'fontsize': 18
        }
        # if v_n == "tempterature":
        if i!=1:
            plot_dic['sharey'] = True

        o_plot.set_plot_dics.append(plot_dic)
        # CORRELATIONS
        corr_dic2 = {  # relies on the "get_res_corr(self, it, v_n): " method of data object
            'task': 'corr2d', 'dtype': 'corr', 'ptype': 'cartesian',
            'data': dic_data[sim]['data'],
            'position': (2, i),
            'v_n_x': 'theta', 'v_n_y': 'ye', 'v_n': 'mass', 'normalize': True,
            'cbar': {},
            'cmap': 'viridis', #'set_under': 'white', #'set_over': 'black',
            'xlabel': r"Angle from binary plane", 'ylabel': r"$Y_e$",
            'xmin': 0., 'xmax': 90., 'ymin': 0.05, 'ymax': 0.5, 'vmin': 1e-4, 'vmax': 1e-2,
            'xscale': "linear", 'yscale': "linear", 'norm': 'log',
            'mask_below': None, 'mask_above': None,
            'title': {},  # {"text": o_corr_data.sim.replace('_', '\_'), 'fontsize': 14},
            'fancyticks': True,
            'minorticks': True,
            'sharex': False,  # removes angular citkscitks
            'sharey': False,
            'fontsize': 14,
            'labelsize': 14,
        }
        corr_dic2["axhline"] = {"y": 0.25, "linestyle": "-", "linewidth": 0.5, "color": "black"}
        # corr_dic2["axvline"] = {"x": 0.25, "linestyle": "-", "linewidth": 0.5, "color": "black"}
        if i!=1:
            corr_dic2['sharey'] = True
        if sim == sims[-1]:
            corr_dic2['cbar'] = \
                {'location': 'right .03 .0', 'label': r"$M_{\rm{ej}}/M$",  # 'fmt': '%.1f',
                'labelsize': 14, 'fontsize': 14}
        o_plot.set_plot_dics.append(corr_dic2)
        i = i + 1
    o_plot.main()

def plot_correlation_from_two_datadirs_vertical():

    sims = ['SFHo_M135135_LK', 'SFHo_M135135_M0', "SFHo_M13641364_M0_LK_SR"]
    colors = ["blue", "red", "green"]#, "orange"]
    lss = ["-", "--", "-."]#, ":"]
    rowdata = ["/data1/numrel/WhiskyTHC/Backup/2017/",
                 "/data1/numrel/WhiskyTHC/Backup/2017/",
                 "/data1/numrel/WhiskyTHC/Backup/2018/GW170817/"]
    datapaths = ["/data01/numrel/vsevolod.nedora/postprocessed_radice2/",
                 "/data01/numrel/vsevolod.nedora/postprocessed_radice2/",
                 "/data01/numrel/vsevolod.nedora/postprocessed4/"]
    v_ns = ["Y_e", "Y_e", "Y_e"]
    det = 0
    masks = ["geo", "geo", "geo"]

    #

    dic_data = {}
    for sim, v_n, mask, path in zip(sims, v_ns, masks, datapaths):
        Paths.ppr_sims = path
        o = ADD_METHODS_ALL_PAR(sim)
        corr_table = o.get_outflow_corr(det, mask, "{}_{}".format(v_n, "theta"))
        dic_data[sim] = {}
        dic_data[sim]["data"] = corr_table.T
        hist = o.get_outflow_hist(det, mask, "theta")
        dic_data[sim]['hist'] = hist.T
        print(hist.T.shape)
    Printcolor.green("data in collected")

    #
    # for sim in sims:
    #     dic_data[sim]['data'] /= np.sum(dic_data[sim]['data'][1:, 1:])

    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = __outplotdir__
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (4.2, 3.6*len(sims))  # <->, |]
    o_plot.gen_set["figname"] = "comparison.png".format("Ye")
    o_plot.gen_set["sharex"] = True
    o_plot.gen_set["sharey"] = False
    o_plot.gen_set["dpi"] = 128
    o_plot.gen_set["subplots_adjust_h"] = 0.0
    o_plot.gen_set["subplots_adjust_w"] = 0.0
    o_plot.set_plot_dics = []

    i = 1
    for sim, ls, color in zip(sims, lss, colors):
        # HISTOGRAMS
        plot_dic = {
            'task': 'hist1d', 'ptype': 'cartesian',
            'position': (1, 1),
            'data': dic_data[sim]["hist"], 'normalize': True,
            'v_n_x': "theta", 'v_n_y': "mass",
            'color': color, 'ls': ls, 'lw': 1., 'ds': 'steps', 'alpha': 1.0,
            'xmin': 50., 'xamx': 90., 'ymin': 1e-4, 'ymax': 1e0,
            'xlabel': Labels.labels("theta"), 'ylabel': r"$M_{\rm{ej}}/M$",
            'label': sim.replace('_', '\_'), 'yscale': 'log',
            'fancyticks': True, 'minorticks': True,
            'fontsize': 14,
            'labelsize': 14,
            'sharex': True,  # removes angular citkscitks
            'sharey': False,
            'title':{},#{'text':sim.replace('_', '\_'), 'fontsize':12},
            # 'textold': {'coords': (0.1, 0.1), 'text':sim.replace('_', '\_'), 'color': 'black', 'fs': 10},
            'legend': {'loc': 'best', 'ncol': 1, 'fontsize': 10}  # 'loc': 'best', 'ncol': 2, 'fontsize': 18
        }
        # if v_n == "tempterature":
        # if i!=1:
        #     plot_dic['sharey'] = True

        o_plot.set_plot_dics.append(plot_dic)
    i = 2
    for sim in sims:
        # CORRELATIONS
        corr_dic2 = {  # relies on the "get_res_corr(self, it, v_n): " method of data object
            'task': 'corr2d', 'dtype': 'corr', 'ptype': 'cartesian',
            'data': dic_data[sim]['data'],
            'position': (i, 1),
            'v_n_x': 'theta', 'v_n_y': 'ye', 'v_n': 'mass', 'normalize': True,
            'cbar': {},
            'cmap': 'viridis', #'set_under': 'white', #'set_over': 'black',
            'xlabel': r"Angle from binary plane", 'ylabel': r"$Y_e$",
            'xmin': 0., 'xmax': 90., 'ymin': 0.05, 'ymax': 0.5, 'vmin': 1e-4, 'vmax': 1e-2,
            'xscale': "linear", 'yscale': "linear", 'norm': 'log',
            'mask_below': None, 'mask_above': None,
            'title': {},  # {"text": o_corr_data.sim.replace('_', '\_'), 'fontsize': 14},
            'fancyticks': True,
            'minorticks': True,
            'sharex': True,  # removes angular citkscitks
            'sharey': False,
            'fontsize': 14,
            'labelsize': 14,
        }
        corr_dic2["axhline"] = {"y": 0.25, "linestyle": "-", "linewidth": 0.5, "color": "black"}
        # corr_dic2["axvline"] = {"x": 0.25, "linestyle": "-", "linewidth": 0.5, "color": "black"}
        if sim == sims[-1]:
            corr_dic2['sharex'] = False
        if sim == sims[-1]:
            corr_dic2['cbar'] = \
                {'location': 'right .03 .0', 'label': r"$M_{\rm{ej}}/M$",  # 'fmt': '%.1f',
                'labelsize': 14, 'fontsize': 14}
        o_plot.set_plot_dics.append(corr_dic2)
        i = i + 1
    o_plot.main()



if __name__ == '__main__':
    plot_correlation_from_two_datadirs_vertical()
    exit(1)
    plot_correlation_from_two_datadirs()
    exit(1)
    plot_correlation()
    pass