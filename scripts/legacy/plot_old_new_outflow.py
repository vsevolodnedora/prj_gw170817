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

''' code comparison '''
# comaprison of two codes
def plot_total_fluxes_2sims():

    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = Paths.plots
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (4.2, 3.6)  # <->, |]
    o_plot.gen_set["figname"] = "compairing_py_cc.png"
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = True
    o_plot.gen_set["dpi"] = 128
    o_plot.gen_set["subplots_adjust_h"] = 0.3
    o_plot.gen_set["subplots_adjust_w"] = 0.01
    o_plot.set_plot_dics = []

    det = 0

    # sims = ["DD2_M13641364_M0_LK_SR_R04", "BLh_M13641364_M0_LK_SR", "LS220_M13641364_M0_LK_SR", "SLy4_M13641364_M0_LK_SR", "SFHo_M13641364_M0_LK_SR"]
    # lbls = ["DD2", "BLh", "LS220", "SLy4", "SFHo"]
    # masks= [mask, mask, mask, mask, mask]
    # colors=["black", "gray", "red", "blue", "green"]
    # lss   =["-", "-", "-", "-", "-"]
    #
    # sims += ["DD2_M15091235_M0_LK_SR", "LS220_M14691268_M0_LK_SR", "SFHo_M14521283_M0_LK_SR"]
    # lbls += ["DD2 151 124", "LS220 150 127", "SFHo 145 128"]
    # masks+= [mask, mask, mask, mask, mask]
    # colors+=["black", "red", "green"]
    # lss   +=["--", "--", "--"]
    do_load_tmerg = False
    sims = ["BHBlp_M130130_LK", "BHBlp_M130130_LK"]
    paths = ["/data01/numrel/vsevolod.nedora/postprocessed_radice2/BHBlp_M130130_LK/outflow_0/geo/",
             "/data01/numrel/vsevolod.nedora/postprocessed_radice/BHBlp_M130130_LK/outflow_0/"]
    lbls = ["outflowed.py", "outflowed.cc"]
    masks= ["geo", "geo"]
    colors=["blue", "red"]
    lss   =["--", ":"]
    fname = "total_flux.dat"
    # sims += ["DD2_M15091235_M0_LK_SR", "LS220_M14691268_M0_LK_SR"]
    # lbls += ["DD2 151 124", "LS220 150 127"]
    # masks+= [mask, mask]
    # colors+=["blue", "red"]
    # lss   +=["--", "--"]


    i_x_plot = 1
    for sim, lbl, mask, path, color, ls in zip(sims, lbls, masks, paths, colors, lss):

        # fpath = Paths.ppr_sims + sim + "/" + "outflow_{}/".format(det) + mask + '/' + "total_flux.dat"
        fpath = path + fname
        if not os.path.isfile(fpath):
            raise IOError("File does not exist: {}".format(fpath))
        print("loading: {}".format(fpath))
        timearr, massarr = np.loadtxt(fpath, usecols=(0, 2), unpack=True)

        if not path.__contains__("geo"):
            timearr *= Constants.time_constant * 1e-3
        if do_load_tmerg:
            fpath = Paths.ppr_sims + sim + "/" + "waveforms/" + "tmerger.dat"
            if not os.path.isfile(fpath):
                raise IOError("File does not exist: {}".format(fpath))
            tmerg = np.float(np.loadtxt(fpath, unpack=True))
            timearr = timearr - (tmerg * Constants.time_constant * 1e-3)

        plot_dic = {
            'task': 'line', 'ptype': 'cartesian',
            'position': (1, 1),
            'xarr': timearr * 1e3, 'yarr': massarr * 1e2,
            'v_n_x': "time", 'v_n_y': "mass",
            'color': color, 'ls': ls, 'lw': 0.8, 'ds': 'default', 'alpha': 1.0,
            'xmin': 0, 'xmax': 25, 'ymin': 0, 'ymax': 0.07,
            'xlabel': Labels.labels("t"), 'ylabel': Labels.labels("ejmass"),
            'label': lbl, 'yscale': 'linear',
            'fancyticks': True, 'minorticks': True,
            'fontsize': 14,
            'labelsize': 14,
            'title':{'text':r"{}".format(sim).replace('_','\_'), 'fontsize':12},
            'legend': {'loc': 'best', 'ncol': 1, 'fontsize': 11} # 'loc': 'best', 'ncol': 2, 'fontsize': 18
        }
        # if mask == "geo": plot_dic["ymax"] = 11.

        if sim >= sims[-1]:
            plot_dic['legend'] = {'loc': 'best', 'ncol': 1, 'fontsize': 12}

        o_plot.set_plot_dics.append(plot_dic)




        #
        #


        i_x_plot += 1
    o_plot.main()
    exit(1)

if __name__ == '__main__':
    plot_total_fluxes_2sims()