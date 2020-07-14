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

__outplotdir__ = "/data01/numrel/vsevolod.nedora/figs/all3/histograms/"

def plot_hists_from_different_dirs():

    # sims = ['SFHo_M135135_LK', 'SFHo_M135135_M0', 'SFHo_M13641364_M0_SR', "SFHo_M13641364_M0_LK_SR"]
    # sims = ['LS220_M135135_LK', 'LS220_M135135_M0', 'LS220_M13641364_M0_SR', "LS220_M13641364_M0_LK_SR"]
    # sims = ['DD2_M135135_LK', 'DD2_M135135_M0', 'DD2_M13641364_M0_SR_R04', "DD2_M13641364_M0_LK_SR_R04"]
    # sims = ['BHBlp_M130130_LK', 'BHBlp_M135135_M0']
    # sims = ['BLh_M13651365_M0_SR', 'BLh_M13641364_M0_LK_SR']
    sims = ['SLy4_M13641364_M0_SR', 'SLy4_M13641364_M0_LK_SR']
    colors = ["blue", "red"]#, "green", "purple"]
    lss = ["-", "--"]#, "-.", ":"]
    # datapaths = ["/data01/numrel/vsevolod.nedora/postprocessed_radice2/",
    #                "/data01/numrel/vsevolod.nedora/postprocessed_radice2/"]
    datapaths = ["/data01/numrel/vsevolod.nedora/postprocessed4/",
                  "/data01/numrel/vsevolod.nedora/postprocessed4/"]
    #

    v_ns = ["Y_e", 'theta', 'entropy']
    det = 0
    masks = ['geo', 'geo', 'geo']

    #

    dic_data = {}
    for v_n, mask in zip(v_ns, masks):
        dic_data[v_n+mask] = {}
        for sim, path in zip(sims, datapaths):
            dic_data[v_n+mask][sim] = {}
            Paths.ppr_sims = path
            o = ADD_METHODS_ALL_PAR(sim)
            # corr_table = o.get_outflow_corr(det, mask, "{}_{}".format(v_n, "theta"))
            # dic_data[v_n+mask][sim]["data"] = corr_table.T
            hist = o.get_outflow_hist(det, mask, v_n)
            dic_data[v_n+mask][sim]['hist'] = hist.T
            dic_data[v_n + mask][sim]['mej'] = sum(hist.T[:,1])
            print(hist.T.shape)
    Printcolor.green("data in collected")

    #
    # for sim in sims:
    #     dic_data[sim]['data'] /= np.sum(dic_data[sim]['data'][1:, 1:])

    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = __outplotdir__
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (4.2*len(v_ns), 3.6)  # <->, |]
    o_plot.gen_set["figname"] = "comparison_sly4.png"
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = False
    o_plot.gen_set["dpi"] = 128
    o_plot.gen_set["subplots_adjust_h"] = 0.0
    o_plot.gen_set["subplots_adjust_w"] = 0.0
    o_plot.set_plot_dics = []

    i = 1
    for v_n, mask in zip(v_ns, masks):
        # textdic = {'task': 'text', 'ptype': 'cartesian',
        #            'position': (1, i), 'x': 0.5, 'y': 0.5,
        #            'text': r"{}".format(dic_data[v_n + mask][sim]['mej'] * 1e2) + "$[10^2M_{\odot}]$", 'fs': 12,
        #            'color': 'black', 'horal': True, 'transform': True}

        for sim, ls, color in zip(sims, lss, colors):
            # HISTOGRAMS
            mej = np.sum(dic_data[v_n + mask][sim]["hist"][:, 1])
            plot_dic = {
                'task': 'hist1d', 'ptype': 'cartesian',
                'position': (1, i),
                'data': dic_data[v_n+mask][sim]["hist"], 'normalize': False,
                'v_n_x': v_n, 'v_n_y': "mass",
                'color': color, 'ls': ls, 'lw': 1., 'ds': 'steps', 'alpha': 1.0,
                'xmin': 0., 'xamx': 90., 'ymin': 1e-5, 'ymax': 1e-3,
                'xlabel': Labels.labels(v_n), 'ylabel': r"$M_{\rm{ej}}$",
                'label': None, 'yscale': 'log',
                'fancyticks': True, 'minorticks': True,
                'fontsize': 14,
                'labelsize': 14,
                'sharex': False,  # removes angular citkscitks
                'sharey': False,
                # 'yticks': [],
                'title':{},#{'text':sim.replace('_', '\_'), 'fontsize':12},
                # 'textold': {'coords': (0.1, 0.1), 'text':sim.replace('_', '\_'), 'color': 'black', 'fs': 10},
                'legend': {'loc': 'best', 'ncol': 1, 'fontsize': 10}  # 'loc': 'best', 'ncol': 2, 'fontsize': 18
            }

            # textdic = {'task': 'text', 'ptype': 'cartesian',
            #            'position': (1, i), 'x': 0.5, 'y': 0.5,
            #            'text': r"{}".format(mej * 1e2) + "$[10^2M_{\odot}]$", 'fs': 12,
            #            'color': 'black', 'horal': True, 'transform': True}

            # if v_n == "tempterature":
            # if i!=1:
            #     plot_dic['sharey'] = True
            if v_n != v_ns[0]:
                plot_dic['sharey'] = False
                plot_dic['yticks'] = []
                plot_dic['rmylbls'] = True
                plot_dic['ylabel'] = None
            if v_n == v_ns[0]:
                # mej = np.sum(dic_data[v_n+mask][sim]["hist"][:,1])
                # o_plot.set_plot_dics.append(textdic)
                plot_dic['label'] = sim.replace('_', '\_')
            elif v_n == v_ns[-1]:
                mej = np.sum(dic_data[v_n + mask][sim]["hist"][:, 1])
                plot_dic['label'] = r"$M_{\rm{ej}}$ "+"{:.2f}".format(mej * 1e2) + "$ [10^2M_{\odot}]$"

            if v_n == "theta":
                plot_dic['xmin'], plot_dic['xmax'] = 0., 90.
                plot_dic['xmajorticks'] = np.arange(5) * 90. / 4.
                plot_dic['xminorticks'] = np.arange(17) * 90. / 16
                plot_dic['xmajorlabels'] = [r"$0^\circ$", r"$22.5^\circ$", r"$45^\circ$",
                            r"$67.5^\circ$", r"$90^\circ$"]
            elif v_n == "entropy":
                plot_dic['xmin'], plot_dic['xmax'] = 5, 95.
            elif v_n == "Y_e":
                plot_dic['xmin'], plot_dic['xmax'] = 0.05, 0.45



            o_plot.set_plot_dics.append(plot_dic)
        i = i + 1
    o_plot.main()
    exit(1)

if __name__ == '__main__':
    plot_hists_from_different_dirs()

