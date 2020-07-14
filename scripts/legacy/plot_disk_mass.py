# ---
#
# Needs .csv tables to plot quantities
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
from scipy import stats
#
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

def plot_disk_mass_evol_SR():
    # 11

    sims = ["BLh_M13641364_M0_LK_SR", "BLh_M11841581_M0_LK_SR", "BLh_M11461635_M0_LK_SR", "BLh_M10651772_M0_LK_SR", "BLh_M10201856_M0_SR", "BLh_M10201856_M0_LK_SR"] + \
           ["DD2_M13641364_M0_SR", "DD2_M13641364_M0_SR_R04", "DD2_M13641364_M0_LK_SR_R04", "DD2_M14971245_M0_SR", "DD2_M15091235_M0_LK_SR"] + \
           ["LS220_M13641364_M0_SR", "LS220_M13641364_M0_LK_SR_restart", "LS220_M14691268_M0_LK_SR", "LS220_M11461635_M0_LK_SR", "LS220_M10651772_M0_LK_SR"] + \
           ["SFHo_M13641364_M0_SR", "SFHo_M14521283_M0_SR"] + \
           ["SLy4_M13641364_M0_SR", "SLy4_M14521283_M0_SR", "SLy4_M10201856_M0_LK_SR"]
    colors = ["black", "black", "black", "black", "black", "black"] + \
            ["blue", "blue", "blue", "blue", "blue"] + \
            ["red", "red", "red", "red", "red"] + \
            ["green", "green"] + \
            ["orange", "orange", "orange"]
    lss = ["-", "--", "-.", ":", '-', '--'] + \
          ["-", "-", "--", "-.", ":"] + \
          ["-", "--", "-.", ":", '-'] + \
          ["-", "--"] + \
          ['-', '--', "-."]
    lws = [0.7 for sim in sims]
    alphas = [1. for sim in sims]


    #
    # sims = ["DD2_M13641364_M0_LK_SR_R04", "BLh_M13641364_M0_LK_SR", "BLh_M13641364_M0_LK_SR"] + \
    #        ["DD2_M15091235_M0_LK_SR", "LS220_M14691268_M0_LK_SR"] + \
    #        ["DD2_M13641364_M0_SR", "LS220_M13641364_M0_SR", "SFHo_M13641364_M0_SR", "SLy4_M13641364_M0_SR"] + \
    #        ["DD2_M14971245_M0_SR", "SFHo_M14521283_M0_SR", "SLy4_M14521283_M0_SR"]
    # #
    # colors = ["blue", "black"] + \
    #        ["blue", "red"] + \
    #        ["blue", "red", "green", "orange"] + \
    #        ["blue", "green", "orange"]
    # #
    # lss=["-", "-"] + \
    #     ["--", "--"] + \
    #     [":", ":", ":", ":"] + \
    #     ["-.", "-."]
    # #
    # lws = [1., 1.] + \
    #     [1., 1.] + \
    #     [1., 1., 1., 1.] + \
    #     [1., 1.]
    # alphas=[1., 1.] + \
    #     [1., 1.] + \
    #     [1., 1., 1., 1.] + \
    #     [1., 1.]
    #
    # ----

    from scipy import interpolate

    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = Paths.plots + "all3/"
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (4.2, 3.6)  # <->, |]
    o_plot.gen_set["figname"] = "disk_mass_evol_SR.png"
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = True
    o_plot.gen_set["dpi"] = 128
    o_plot.gen_set["subplots_adjust_h"] = 0.3
    o_plot.gen_set["subplots_adjust_w"] = 0.0
    o_plot.set_plot_dics = []

    for sim, color, ls, lw, alpha in zip(sims, colors, lss, lws, alphas):
        print("{}".format(sim))
        o_data = ADD_METHODS_ALL_PAR(sim)
        data = o_data.get_disk_mass()
        tmerg = o_data.get_par("tmerg")
        tarr = (data[:, 0] - tmerg) * 1e3
        marr = data[:, 1]

        if sim == "DD2_M13641364_M0_LK_SR_R04":
            tarr = tarr[3:] # 3ms, 6ms, 51ms.... Removing initial profiles
            marr = marr[3:] #
        #
        tcoll = o_data.get_par("tcoll_gw")
        if not np.isnan(tcoll) and tcoll < tarr[-1]:
            tcoll = (tcoll - tmerg) * 1e3
            print(tcoll, tarr[0])
            mcoll = interpolate.interp1d(tarr,marr,kind="linear", bounds_error=False)(tcoll)
            tcoll_dic = {
                'task': 'line', 'ptype': 'cartesian',
                'position': (1, 1),
                'xarr': [tcoll], 'yarr': [mcoll],
                'v_n_x': "time", 'v_n_y': "mass",
                'color': color, 'marker': "x", 'ms': 5., 'alpha': alpha,
                'xmin': -10, 'xmax': 100, 'ymin': 0, 'ymax': .3,
                'xlabel': Labels.labels("t-tmerg"), 'ylabel': Labels.labels("diskmass"),
                'label': None, 'yscale': 'linear',
                'fancyticks': True, 'minorticks': True,
                'fontsize': 14,
                'labelsize': 14,
                'legend': {}  # 'loc': 'best', 'ncol': 2, 'fontsize': 18
            }
            o_plot.set_plot_dics.append(tcoll_dic)
        #
        plot_dic = {
            'task': 'line', 'ptype': 'cartesian',
            'position': (1, 1),
            'xarr': tarr, 'yarr': marr,
            'v_n_x': "time", 'v_n_y': "mass",
            'color': color, 'ls': ls, 'lw': 0.8, 'ds': 'steps', 'alpha': 1.0,
            'xmin': -10, 'xmax': 30, 'ymin': 0, 'ymax': .35,
            'xlabel': Labels.labels("t-tmerg"), 'ylabel': Labels.labels("diskmass"),
            'label': str(sim).replace('_', '\_'), 'yscale': 'linear',
            'fancyticks': True, 'minorticks': True,
            'fontsize': 14,
            'labelsize': 14,
            'legend': {'bbox_to_anchor':(1.1,1.05),
                'loc': 'lower right', 'ncol': 2, 'fontsize': 8}  # 'loc': 'best', 'ncol': 2, 'fontsize': 18
        }
        if sim == sims[-1]:
            plot_dic['legend'] = {'bbox_to_anchor':(1.1,1.05),
                'loc': 'lower right', 'ncol': 2, 'fontsize': 8}
        o_plot.set_plot_dics.append(plot_dic)

    o_plot.main()
    exit(1)
#
def plot_disk_mass_evol_LR():

    sims = ["BLh_M16351146_M0_LK_LR", "BLh_M13641364_M0_LK_LR", "SLy4_M10651772_M0_LK_LR",  "SFHo_M10651772_M0_LK_LR", "SFHo_M16351146_M0_LK_LR",
            "LS220_M10651772_M0_LK_LR", "LS220_M16351146_M0_LK_LR", "DD2_M16351146_M0_LK_LR"] + \
           ["DD2_M13641364_M0_LR", "LS220_M13641364_M0_LR"] + \
           ["DD2_M14971246_M0_LR", "DD2_M14861254_M0_LR", "DD2_M14351298_M0_LR", "DD2_M14321300_M0_LR"]
    #
    colors = ["black", "gray", "orange", "pink", "olive", "red", "purple", "blue"] + \
            ["blue", "red"] + \
            ["green", "blue", "lightblue", "cyan"]
    #
    lss = ["-", "-", "-", "-", "-", "-", "-", "-"] +\
          ['--', '--', '--'] + \
          [":", ":", ":", ":"]
    #
    lws = [1., 1., 1., 1., 1., 1., 1., 1.] + \
          [1., 1.] + \
          [1., 1., 1., 1.]
    #
    alphas = [1., 1., 1., 1., 1., 1., 1., 1.] + \
          [1., 1.] + \
          [1., 1., 1., 1.]


    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = Paths.plots + "all2/"
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (4.2, 3.6)  # <->, |]
    o_plot.gen_set["figname"] = "disk_mass_evol_LR.png"
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = True
    o_plot.gen_set["dpi"] = 128
    o_plot.gen_set["subplots_adjust_h"] = 0.3
    o_plot.gen_set["subplots_adjust_w"] = 0.0
    o_plot.set_plot_dics = []

    from scipy import interpolate

    for sim, color, ls, lw, alpha in zip(sims, colors, lss, lws, alphas):
        print("{}".format(sim))
        o_data = ADD_METHODS_ALL_PAR(sim)
        data = o_data.get_disk_mass()
        assert len(data) > 0
        tmerg = o_data.get_par("tmerg")
        tarr = (data[:, 0] - tmerg) * 1e3
        marr = data[:, 1]

        if sim == "DD2_M13641364_M0_LK_SR_R04":
            tarr = tarr[3:]  # 3ms, 6ms, 51ms.... Removing initial profiles
            marr = marr[3:]  #
        #
        tcoll = o_data.get_par("tcoll_gw")
        if not np.isnan(tcoll) and tcoll < tarr[-1]:
            tcoll = (tcoll - tmerg) * 1e3
            print(tcoll, tarr[0])
            mcoll = interpolate.interp1d(tarr, marr, kind="linear")(tcoll)
            tcoll_dic = {
                'task': 'line', 'ptype': 'cartesian',
                'position': (1, 1),
                'xarr': [tcoll], 'yarr': [mcoll],
                'v_n_x': "time", 'v_n_y': "mass",
                'color': color, 'marker': "x", 'ms': 5., 'alpha': alpha,
                'xmin': -10, 'xmax': 40, 'ymin': 0, 'ymax': .3,
                'xlabel': Labels.labels("t-tmerg"), 'ylabel': Labels.labels("diskmass"),
                'label': None, 'yscale': 'linear',
                'fancyticks': True, 'minorticks': True,
                'fontsize': 14,
                'labelsize': 14,
                'legend': {}  # 'loc': 'best', 'ncol': 2, 'fontsize': 18
            }
            o_plot.set_plot_dics.append(tcoll_dic)
        #
        plot_dic = {
            'task': 'line', 'ptype': 'cartesian',
            'position': (1, 1),
            'xarr': tarr, 'yarr': marr,
            'v_n_x': "time", 'v_n_y': "mass",
            'color': color, 'ls': ls, 'lw': 0.8, 'ds': 'steps', 'alpha': 1.0,
            'xmin': -10, 'xmax': 40, 'ymin': 0, 'ymax': .35,
            'xlabel': Labels.labels("t-tmerg"), 'ylabel': Labels.labels("diskmass"),
            'label': str(sim).replace('_', '\_'), 'yscale': 'linear',
            'fancyticks': True, 'minorticks': True,
            'fontsize': 14,
            'labelsize': 14,
            'legend': {'bbox_to_anchor': (1.1, 1.05),
                       'loc': 'lower right', 'ncol': 2, 'fontsize': 8}  # 'loc': 'best', 'ncol': 2, 'fontsize': 18
        }
        if sim == sims[-1]:
            plot_dic['legend'] = {'bbox_to_anchor': (1.1, 1.05),
                                  'loc': 'lower right', 'ncol': 2, 'fontsize': 8}
        o_plot.set_plot_dics.append(plot_dic)


    o_plot.main()
    exit(1)

if __name__ == '__main__':
    plot_disk_mass_evol_SR()