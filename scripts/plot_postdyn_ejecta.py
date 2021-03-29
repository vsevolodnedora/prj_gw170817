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
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')


from matplotlib.colors import LogNorm, Normalize

sys.path.append('/data01/numrel/vsevolod.nedora/bns_ppr_tools/')
from preanalysis import LOAD_INIT_DATA
from outflowed import EJECTA_PARS
from preanalysis import LOAD_ITTIME
from plotting_methods import PLOT_MANY_TASKS
from profile import LOAD_PROFILE_XYXZ, LOAD_RES_CORR, LOAD_DENSITY_MODES, MAINMETHODS_STORE, MAINMETHODS_STORE_XYXZ
from utils import Paths, Lists, Labels, Constants, Printcolor, UTILS, Files, PHYSICS

from data import ADD_METHODS_ALL_PAR
from comparison import TWO_SIMS, THREE_SIMS
from settings import resolutions
from uutils import *

import model_sets.models as md

__outplotdir__ = "..//figs/all3/plot_postdyn_ej/"
if not os.path.isdir(__outplotdir__):
    os.mkdir(__outplotdir__)


class Limits:

    @staticmethod
    def lim(v_n):
        if v_n in ["Y_e", "ye", "Ye"]:
            return 0., 0.5
        elif v_n in  ["vel_inf", "vinf", "velinf"]:
            return 0, 1.1
        elif v_n in ["theta"]:
            return 0, 90.
        elif v_n in ["phi"]:
            return 0., 360
        elif v_n in ["entropy", "s"]:
            return 0, 120.
        elif v_n in ["temperature", "temp"]:
            return 0, 5.
        else:
            raise NameError("limit for v_n:{} is not found"
                            .format(v_n))

    @staticmethod
    def in_dic(dic):
        # if "v_n" in dic.keys():
        #     if dic["v_n"] != None:
        #         lim1, lim2 = Limits.lim(dic["v_n"])
        #         dic["zmin"], dic["zmax"] = lim1, lim2

        if "v_n_x" in dic.keys():
            if dic["xmin"] != None and dic["xmax"] != None:
                pass
            else:
                # print(dic["xmin"], dic["xmin"])
                if dic["v_n_x"] != None:
                    try:
                        lim1, lim2 = Limits.lim(dic["v_n_x"])
                    except:
                        raise NameError("X limits for {} are not set and not found".format(dic["v_n_x"]))

                    dic["xmin"], dic["xmax"] = lim1, lim2

        if "v_n_y" in dic.keys():
            if dic["ymin"] != None and dic["ymax"] != None:
                pass
            else:
                if dic["v_n_y"] != None:
                    try:
                        lim1, lim2 = Limits.lim(dic["v_n_y"])
                    except:
                        raise NameError("Y limits for {} are not set and not found".format(dic["v_n_y"]))
                    dic["ymin"], dic["ymax"] = lim1, lim2
        return dic

def plot_ejecta_time_corr_properites():

    det = 0

    # sims = ["SLy4_M13641364_M0_SR", "SFHo_M13641364_M0_LK_SR", "LS220_M13641364_M0_SR"]
    sims = ["BLh_M13641364_M0_LK_SR", "DD2_M13641364_M0_LK_SR_R04", "LS220_M14691268_M0_LK_SR"]
    lbls = [print_fancy_label(sim) for sim in sims]
    masks= ["bern_geoend", "bern_geoend", "bern_geoend", "bern_geoend", "bern_geoend"]
    # v_ns = ["vel_inf", "vel_inf", "vel_inf", "vel_inf", "vel_inf"]
    v_ns = ["Y_e", "Y_e", "Y_e", "Y_e", "Y_e"]

    plotname = ""
    for sim in sims:
        eos = simulations.loc[sim]["EOS"]
        if not plotname.__contains__(eos.lower()):
            plotname = plotname + eos.lower() + '_'
    for v_n in v_ns:
        if not plotname.__contains__(v_n.lower()):
            plotname = plotname + v_n.lower() + '_'
    plotname  = plotname + '.png'


    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = __outplotdir__
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (9.5, 3.1)  # <->, |]
    o_plot.gen_set["figname"] = plotname
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = True
    o_plot.gen_set["dpi"] = 128
    o_plot.gen_set["subplots_adjust_h"] = 0.3
    o_plot.gen_set["subplots_adjust_w"] = 0.01
    o_plot.set_plot_dics = []

    i_x_plot = 1
    for sim, lbl, mask, v_n in zip(sims,lbls,masks,v_ns):

        o_data = ADD_METHODS_ALL_PAR(sim)

        fpath = Paths.ppr_sims+sim+"/"+"outflow_{}/".format(det) + mask + '/' + "timecorr_{}.h5".format(v_n)
        if not os.path.isfile(fpath):
            raise IOError("File does not exist: {}".format(fpath))

        dfile = h5py.File(fpath, "r")
        timearr = np.array(dfile["time"])
        v_n_arr = np.array(dfile[v_n])
        mass    = np.array(dfile["mass"])

        tmerg = o_data.get_par("tmerg_r")
        timearr = timearr - (tmerg * 1.e3)

        # data = o_data.get_outflow_corr(det, mask, v_n)


        corr_dic2 = {  # relies on the "get_res_corr(self, it, v_n): " method of data object
            'task': 'corr2d', 'dtype': 'corr', 'ptype': 'cartesian',
            'xarr':timearr, 'yarr':v_n_arr, 'zarr':mass,
            'position': (1, i_x_plot),
            'v_n_x': "time", 'v_n_y': v_n, 'v_n': 'mass', 'normalize': True,
            'cbar': {},
            'cmap': 'inferno_r',
            'xlabel': Labels.labels("t-tmerg"), 'ylabel': Labels.labels(v_n),
            'xmin': 10., 'xmax': timearr[-1], 'ymin': None, 'ymax': None, 'vmin': 1e-4, 'vmax': 1e-1,
            'xscale': "linear", 'yscale': "linear", 'norm': 'log',
            'mask_below': None, 'mask_above': None,
            'title': {},  # {"text": o_corr_data.sim.replace('_', '\_'), 'fontsize': 14},
            'fancyticks': True,
            'minorticks': True,
            'sharex': False,  # removes angular citkscitks
            'sharey': False,
            'fontsize': 14,
            'labelsize': 14
        }
        text_dic = {
            'task': 'text', 'dtype': 'corr', 'ptype': 'cartesian',
            'position': (1, i_x_plot),
            'x':0.5, 'y':0.9, 'text':lbl, 'fs':11, 'color':'black','horizontalalignment':'center', 'transform':True,
        }
        o_plot.set_plot_dics.append(text_dic)

        if i_x_plot > 1:
            corr_dic2['sharey']=True
        # if i_x_plot == 1:
        #     corr_dic2['text'] = {'text': lbl.replace('_', '\_'), 'coords': (0.1, 0.9),  'color': 'white', 'fs': 14}
        if sim == sims[-1]:
            corr_dic2['cbar'] = {
                'location': 'right .03 .0', 'label': r"$M/M_{\rm{ej}}$",  # 'fmt': '%.1f',
                'labelsize': 14, 'fontsize': 14}
        i_x_plot += 1
        corr_dic2 = Limits.in_dic(corr_dic2)
        o_plot.set_plot_dics.append(corr_dic2)

    o_plot.main()
    exit(1)

''' --------------------- TRASH --------------------------- '''

def plot_qeff_dunb_slices():

    rl = 0
    v_n = "Q_eff_nua"
    # v_n2 = "dens_unb_bern"

    sims = ["LS220_M14691268_M0_LK_SR"]
    iterations = [1515520, 1843200] # 1302528

    sims = ["LS220_M13641364_M0_LK_SR_restart"]
    iterations = [696320]

    sims = ["BLh_M13641364_M0_LK_SR"]

    sims = ["DD2_M13641364_M0_LK_SR_R04"]
    v_n = "density"


    #
    d3class = LOAD_PROFILE_XYXZ(sims[-1])
    d1class = ADD_METHODS_ALL_PAR(sims[-1])

    for it in d3class.list_iterations:


        if not os.path.isdir(__outplotdir__ + sims[-1] + '/'):
            os.mkdir(__outplotdir__ + sims[-1] + '/')
        #
        o_plot = PLOT_MANY_TASKS()
        o_plot.gen_set["figdir"] = __outplotdir__
        o_plot.gen_set["type"] = "cartesian"
        o_plot.gen_set["figsize"] = (9., 3.2)  # <->, |] # to match hists with (8.5, 2.7)
        o_plot.gen_set["figname"] = "{}/{}.png".format(sims[-1], it)
        o_plot.gen_set["sharex"] = False
        o_plot.gen_set["sharey"] = False
        o_plot.gen_set["subplots_adjust_h"] = 0.2
        o_plot.gen_set["subplots_adjust_w"] = 0.1
        o_plot.set_plot_dics = []

        #
        i_x_plot = 1
        i_y_plot = 1

        if True:


            tmerg = d1class.get_par("tmerg")
            time_ = d3class.get_time_for_it(it, "profiles", "prof")

            dens_arr = d3class.get_data(it, rl, "xz", "density")
            data_arr = d3class.get_data(it, rl, "xz", v_n)
            data_arr = data_arr / dens_arr
            x_arr = d3class.get_data(it, rl, "xz", "x")
            z_arr = d3class.get_data(it, rl, "xz", "z")

            def_dic_xz = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                          'xarr': x_arr, "yarr": z_arr, "zarr": data_arr,
                          'position': (i_y_plot, i_x_plot),  # 'title': '[{:.1f} ms]'.format(time_),
                          'cbar': {},
                          'v_n_x': 'x', 'v_n_y': 'z', 'v_n': v_n,
                          'xmin': None, 'xmax': None, 'ymin': None, 'ymax': None, 'vmin': 1e-10, 'vmax': 1e-4,
                          'fill_vmin': False,  # fills the x < vmin with vmin
                          'xscale': None, 'yscale': None,
                          'mask': 'x>0', 'cmap': 'inferno_r', 'norm': "log",
                          'fancyticks': True,
                          'minorticks': True,
                          'title': {"text": r'$t-t_{merg}:$' + r'${:.1f}$'.format((time_ - tmerg) * 1e3),
                                    'fontsize': 14},
                          # 'sharex': True,  # removes angular citkscitks
                          'fontsize': 14,
                          'labelsize': 14,
                          'sharex': False,
                          'sharey': True,
                          }

            def_dic_xz["xmin"], def_dic_xz["xmax"], _, _, def_dic_xz["ymin"], def_dic_xz["ymax"] \
                = UTILS.get_xmin_xmax_ymin_ymax_zmin_zmax(rl)

            if v_n == 'Q_eff_nua':

                def_dic_xz['v_n'] = 'Q_eff_nua/D'
                def_dic_xz['vmin'] = 1e-13
                def_dic_xz['vmax'] = 1e-6
                # def_dic_xz['norm'] = None
            elif v_n == 'Q_eff_nue':

                def_dic_xz['v_n'] = 'Q_eff_nue/D'
                def_dic_xz['vmin'] = 1e-7
                def_dic_xz['vmax'] = 1e-3
                # def_dic_xz['norm'] = None
            elif v_n == 'Q_eff_nux':

                def_dic_xz['v_n'] = 'Q_eff_nux/D'
                def_dic_xz['vmin'] = 1e-10
                def_dic_xz['vmax'] = 1e-4
                # def_dic_xz['norm'] = None
                # print("v_n: {} [{}->{}]".format(v_n, def_dic_xz['zarr'].min(), def_dic_xz['zarr'].max()))
            elif v_n == "R_eff_nua":

                def_dic_xz['v_n'] = 'R_eff_nua/D'
                def_dic_xz['vmin'] = 1e2
                def_dic_xz['vmax'] = 1e6
                # def_dic_xz['norm'] = None

                print("v_n: {} [{}->{}]".format(v_n, def_dic_xz['zarr'].min(), def_dic_xz['zarr'].max()))
                # exit(1)
            elif v_n == "Density":
                def_dic_xz['v_n'] = 'D'
                def_dic_xz['vmin'] = 1e-13
                def_dic_xz['vmax'] = 1e-6


            #if it == iterations[0]:
            def_dic_xz["sharey"] = False

            #if it == iterations[-1]:
            def_dic_xz['cbar'] = {'location': 'right -.08 0.', 'label': Labels.labels(v_n) + "/D",
                                      # 'right .02 0.' 'fmt': '%.1e',
                                      'labelsize': 14,  # 'aspect': 10., # 6
                                      'fontsize': 14}

            o_plot.set_plot_dics.append(def_dic_xz)

            ''' ----------| density unbound | ---------- '''

            u_0 = d3class.get_data(it, rl, "xz", "u_0")
            enthalpy = d3class.get_data(it, rl, "xz", "enthalpy")
            data_arr = -1. * enthalpy * u_0

            print(data_arr.sum())
            print(data_arr.min(), data_arr.max())
            # print(data_arr[100,:])

            def_dic_xz2 = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                           'xarr': x_arr, "yarr": z_arr, "zarr": data_arr,
                           'position': (i_y_plot, i_x_plot),  # 'title': '[{:.1f} ms]'.format(time_),
                           'cbar': {},
                           'v_n_x': 'x', 'v_n_y': 'z', 'v_n': v_n,
                           'xmin': None, 'xmax': None, 'ymin': None, 'ymax': None, 'vmin': 0.98, 'vmax': 1.02,
                           'fill_vmin': False,  # fills the x < vmin with vmin
                           'xscale': None, 'yscale': None,
                           'mask': 'x<0', 'cmap': 'RdBu', 'norm': None,  # "log",
                           'fancyticks': True,
                           'minorticks': True,
                           'title': {"text": r'$t-t_{merg}:$' + r'${:.1f}$'.format((time_ - tmerg) * 1e3),
                                     'fontsize': 14},
                           # 'sharex': True,  # removes angular citkscitks
                           'fontsize': 14,
                           'labelsize': 14,
                           'sharex': False,
                           'sharey': True,
                           }

            def_dic_xz2["xmin"], def_dic_xz2["xmax"], _, _, def_dic_xz2["ymin"], def_dic_xz2["ymax"] \
                = UTILS.get_xmin_xmax_ymin_ymax_zmin_zmax(rl)

            #if it == iterations[0]:
            def_dic_xz2["sharey"] = False

            #if it == iterations[0]:
            def_dic_xz2['cbar'] = {'location': 'left -0.78 0.', 'label': r"$-hu_0$",
                                       # 'right .02 0.' 'fmt': '%.1e',
                                       'labelsize': 14,  # 'aspect': 10.,
                                       'fontsize': 14}

            o_plot.set_plot_dics.append(def_dic_xz2)
            o_plot.main()

        # except:
        #     print (it)

    #
    # for sim in sims:
    #
    #     d3class = LOAD_PROFILE_XYXZ(sim)
    #     d1class = ADD_METHODS_ALL_PAR(sim)
    #
    #     for it in iterations:
    #
    #         ''' ----------| Neutrinos |--------- '''
    #
    #         tmerg = d1class.get_par("tmerg")
    #         time_ = d3class.get_time_for_it(it, "profiles", "prof")
    #
    #         dens_arr = d3class.get_data(it, rl, "xz", "density")
    #         data_arr = d3class.get_data(it, rl, "xz", v_n)
    #         data_arr = data_arr / dens_arr
    #         x_arr = d3class.get_data(it, rl, "xz", "x")
    #         z_arr = d3class.get_data(it, rl, "xz", "z")
    #
    #         def_dic_xz = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
    #                       'xarr': x_arr, "yarr": z_arr, "zarr": data_arr,
    #                       'position': (i_y_plot, i_x_plot),  # 'title': '[{:.1f} ms]'.format(time_),
    #                       'cbar': {},
    #                       'v_n_x': 'x', 'v_n_y': 'z', 'v_n': v_n,
    #                       'xmin': None, 'xmax': None, 'ymin': None, 'ymax': None, 'vmin': 1e-10, 'vmax': 1e-4,
    #                       'fill_vmin': False,  # fills the x < vmin with vmin
    #                       'xscale': None, 'yscale': None,
    #                       'mask': 'x>0', 'cmap': 'inferno_r', 'norm': "log",
    #                       'fancyticks': True,
    #                       'minorticks': True,
    #                       'title': {"text": r'$t-t_{merg}:$' + r'${:.1f}$'.format((time_ - tmerg) * 1e3),
    #                                 'fontsize': 14},
    #                       # 'sharex': True,  # removes angular citkscitks
    #                       'fontsize': 14,
    #                       'labelsize': 14,
    #                       'sharex': False,
    #                       'sharey': True,
    #                       }
    #
    #         def_dic_xz["xmin"], def_dic_xz["xmax"], _, _, def_dic_xz["ymin"], def_dic_xz["ymax"] \
    #             = UTILS.get_xmin_xmax_ymin_ymax_zmin_zmax(rl)
    #
    #         if v_n == 'Q_eff_nua':
    #
    #             def_dic_xz['v_n'] = 'Q_eff_nua/D'
    #             def_dic_xz['vmin'] = 1e-13
    #             def_dic_xz['vmax'] = 1e-6
    #             # def_dic_xz['norm'] = None
    #         elif v_n == 'Q_eff_nue':
    #
    #             def_dic_xz['v_n'] = 'Q_eff_nue/D'
    #             def_dic_xz['vmin'] = 1e-7
    #             def_dic_xz['vmax'] = 1e-3
    #             # def_dic_xz['norm'] = None
    #         elif v_n == 'Q_eff_nux':
    #
    #             def_dic_xz['v_n'] = 'Q_eff_nux/D'
    #             def_dic_xz['vmin'] = 1e-10
    #             def_dic_xz['vmax'] = 1e-4
    #             # def_dic_xz['norm'] = None
    #             # print("v_n: {} [{}->{}]".format(v_n, def_dic_xz['zarr'].min(), def_dic_xz['zarr'].max()))
    #         elif v_n == "R_eff_nua":
    #
    #             def_dic_xz['v_n'] = 'R_eff_nua/D'
    #             def_dic_xz['vmin'] = 1e2
    #             def_dic_xz['vmax'] = 1e6
    #             # def_dic_xz['norm'] = None
    #
    #             print("v_n: {} [{}->{}]".format(v_n, def_dic_xz['zarr'].min(), def_dic_xz['zarr'].max()))
    #             # exit(1)
    #
    #         if it == iterations[0]:
    #             def_dic_xz["sharey"] = False
    #
    #         if it == iterations[-1]:
    #             def_dic_xz['cbar'] = {'location': 'right -.08 0.', 'label': Labels.labels(v_n) + "/D",
    #                                   # 'right .02 0.' 'fmt': '%.1e',
    #                                   'labelsize': 14, #'aspect': 10., # 6
    #                                   'fontsize': 14}
    #
    #         o_plot.set_plot_dics.append(def_dic_xz)
    #
    #         ''' ----------| density unbound | ---------- '''
    #
    #         u_0 = d3class.get_data(it, rl, "xz", "u_0")
    #         enthalpy = d3class.get_data(it, rl, "xz", "enthalpy")
    #         data_arr = -1. * enthalpy * u_0
    #
    #         print(data_arr.sum())
    #         print(data_arr.min(), data_arr.max())
    #         # print(data_arr[100,:])
    #
    #         def_dic_xz2 = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
    #                       'xarr': x_arr, "yarr": z_arr, "zarr": data_arr,
    #                       'position': (i_y_plot, i_x_plot),  # 'title': '[{:.1f} ms]'.format(time_),
    #                       'cbar': {},
    #                       'v_n_x': 'x', 'v_n_y': 'z', 'v_n': v_n,
    #                       'xmin': None, 'xmax': None, 'ymin': None, 'ymax': None, 'vmin': 0.98, 'vmax': 1.02,
    #                       'fill_vmin': False,  # fills the x < vmin with vmin
    #                       'xscale': None, 'yscale': None,
    #                       'mask': 'x<0', 'cmap': 'RdBu', 'norm': None,#"log",
    #                       'fancyticks': True,
    #                       'minorticks': True,
    #                       'title': {"text": r'$t-t_{merg}:$' + r'${:.1f}$'.format((time_ - tmerg) * 1e3),
    #                                 'fontsize': 14},
    #                       # 'sharex': True,  # removes angular citkscitks
    #                       'fontsize': 14,
    #                       'labelsize': 14,
    #                       'sharex': False,
    #                       'sharey': True,
    #                       }
    #
    #         def_dic_xz2["xmin"], def_dic_xz2["xmax"], _, _, def_dic_xz2["ymin"], def_dic_xz2["ymax"] \
    #             = UTILS.get_xmin_xmax_ymin_ymax_zmin_zmax(rl)
    #
    #         if it == iterations[0]:
    #             def_dic_xz2["sharey"] = False
    #
    #         if it == iterations[0]:
    #             def_dic_xz2['cbar'] = {'location': 'left -0.78 0.', 'label': r"$-hu_0$",
    #                                   # 'right .02 0.' 'fmt': '%.1e',
    #                                   'labelsize': 14, #'aspect': 10.,
    #                                   'fontsize': 14}
    #
    #         o_plot.set_plot_dics.append(def_dic_xz2)
    #
    #         i_x_plot = i_x_plot + 1
    #     i_y_plot = i_y_plot + 1
    # o_plot.main()
    exit(0)

def plot_ye_theta_correlation_ejecta():
    # sim = "LS220_M14691268_M0_LK_SR"
    sim = "BLh_M13641364_M0_LK_SR"

    v_n1 = "Y_e"  # optd_0_nua optd_1_nua
    v_n2 = "theta"
    det = 0
    mask = "bern_geoend"

    plotdic = {"vmin": 1e-4, "vmax": 1.e-1,
               "xmin": 0, "xmax": 90.,
               "ymin": 0, "ymax": 0.5,
               "cmap": "jet",
               "set_under": "black",
               "set_over": "red",
               "yscale": "linear",
               "xscale": "linear",
               "figname": "outflow_corr_{}_{}_{}.png".format(sim, v_n1, v_n2),
               "xlabel": "Angle from binary plane",
               "ylabel": "Ye"
               }
    #
    o_methods = ADD_METHODS_ALL_PAR(sim)
    #
    table = o_methods.get_outflow_corr(det, mask, v_n1+'_'+v_n2)
    x_arr = table[:, 0]
    y_arr = table[0, :]
    mass = table[1:, 1:].T
    #
    x_arr = 90 - (x_arr * 180 / np.pi)
    mass = np.maximum(mass, 1e-10)
    mass = mass / np.sum(mass)

    #
    fig = plt.figure(figsize=(4.2, 3.6))
    ax = fig.add_subplot(111)
    #
    ax.axvline(x=60, color="white", linestyle="-", label=r'Expected $\nu$-wind')
    ax.axhline(y=0.4, color="white", linestyle="-")
    #
    norm = LogNorm(plotdic["vmin"], plotdic["vmax"])
    im = ax.pcolormesh(x_arr, y_arr, mass, norm=norm,
                       cmap=plotdic["cmap"])  # , vmin=dic["vmin"], vmax=dic["vmax"])
    im.set_rasterized(True)
    im.cmap.set_over(plotdic['set_under'])
    im.cmap.set_over(plotdic['set_over'])
    #
    ax.set_yscale(plotdic["yscale"])
    ax.set_xscale(plotdic["xscale"])
    #
    ax.set_xlabel(plotdic["xlabel"], fontsize="large")
    ax.set_ylabel(plotdic["ylabel"], fontsize="large")
    #
    ax.set_xlim(plotdic["xmin"], plotdic["xmax"])
    ax.set_ylim(plotdic["ymin"], plotdic["ymax"])
    #
    ax.tick_params(axis='both', which='both', labelleft=True,
                   labelright=False, tick1On=True, tick2On=True,
                   labelsize=12,
                   direction='in',
                   bottom=True, top=True, left=True, right=True)
    ax.minorticks_on()
    #
    ax.set_title("Spiral-Wave Wind Correlation")

    leg = ax.legend(fancybox=True, loc='upper left',
              # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
              shadow=False, ncol=1, fontsize=9,
              framealpha=0., borderaxespad=0.)
    for text in leg.get_texts():
        plt.setp(text, color='w')
    ax.text(0.95, 0.05, sim.replace('_', '\_'),
            ha="right", va="bottom", transform=ax.transAxes, color='white')

    clb = fig.colorbar(im, ax=ax)
    plt.tight_layout()
    clb.ax.set_title(r"M/sum(M)", fontsize=11)
    clb.ax.tick_params(labelsize=11)
    #
    print("plotted: \n")
    print(__outplotdir__ + plotdic["figname"])
    plt.savefig(__outplotdir__ + plotdic["figname"], dpi=128)
    plt.close()

def plot_total_flux_of_matter():

    task = [
        {"sim": "LS220_M14691268_M0_LK_SR", "v_n": "Mej", "det":0, "mask": "bern_geoend", "alpha":1.}
    ]

    sims = ["LS220_M14691268_M0_LK_SR", "BLh_M13641364_M0_LK_SR"]
    colors = ["red", "blue"]
    t1, t2 = 60, 80
    det = 0
    plane = "xz"
    plotdic = {
               "xmin": 20, "xmax": 100,
               "ymin": 0, "ymax": 1.5,
               "yscale": "linear",
               "xscale": "linear",
               "figname": "neutrino_driven_totflux.png",
               "xlabel": r"$time [ms]$",
               "ylabel": r"$M_{\rm ej}$ $[10^{-3} M_{\odot}]$"
               }

    fig = plt.figure(figsize=(4.2, 3.6))
    ax = fig.add_subplot(111)
    #
    # labels

    for sim, color in zip(sims, colors):
        ax.plot([-1, -2], [-1, -2], ls='-',lw=1.,color=color, label=sim.replace('_', '\_'))
    for ls, mask in zip(['-','--'], ["theta60_geoend", "Y_e04_geoend"]):
        ax.plot([-1, -2], [-1, -2], ls=ls, lw=1., color='gray', label=mask.replace('_', '\_'))

    # for sim, color in zip(sims, colors):
    #     ax.plot([-1, -2], [-1, -2], ls='-',lw=1.,color=color, label=sim.replace('_', '\_'))
    #     ax.plot([-1, -2], [-1, -2], ls='-', lw=1., color=color, label=mask.replace('_', '\_'))
    #
    #     lines = axes.get_lines()
    #     legend1 = plt.legend([lines[i] for i in [0, 1, 2]], ["algo1", "algo2", "algo3"], loc=1)
    #     legend2 = plt.legend([lines[i] for i in [0, 3, 6]], parameters, loc=4)
    #     axes.add_artist(legend1)
    #     axes.add_artist(legend2)


    #
    for sim, color in zip(sims, colors):
        o_data = ADD_METHODS_ALL_PAR(sim)
        t_ej_theta, m_ej_theta = o_data.get_time_data_arrs("Mej", det=det, mask="theta60_geoend")
        t_ej_ye, m_ej_ye = o_data.get_time_data_arrs("Mej", det=det, mask="Y_e04_geoend")
        #
        ax.plot(t_ej_theta*1e3, m_ej_theta*1e3,ls='-',lw=1.,color=color)
        ax.plot(t_ej_ye*1e3, m_ej_ye*1e3,ls='--',lw=1.,color=color)
    #
    ax.fill_between([t1, t2], [0, 0], [100, 100], alpha=0.8, color='gray', label=r"$D_{\rm unb}/Q_{\rm eff}$ study")

    #
    ax.set_yscale(plotdic["yscale"])
    ax.set_xscale(plotdic["xscale"])

    ax.set_xlabel(plotdic["xlabel"], fontsize="large")
    ax.set_ylabel(plotdic["ylabel"], fontsize="large")

    ax.set_xlim(plotdic["xmin"], plotdic["xmax"])
    ax.set_ylim(plotdic["ymin"], plotdic["ymax"])
    #
    ax.tick_params(axis='both', which='both', labelleft=True,
                   labelright=False, tick1On=True, tick2On=True,
                   labelsize=12,
                   direction='in',
                   bottom=True, top=True, left=True, right=True)
    ax.minorticks_on()
    # #
    ax.set_title(r"Ejecta mass suspected fo $\nu$-wind")
    # #

    #
    ax.legend(fancybox=True, loc='upper left',
               # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
               shadow=False, ncol=1, fontsize=9,
               framealpha=0., borderaxespad=0.)

    plt.tight_layout()
    #

    print("plotted: \n")
    print(__outplotdir__ + plotdic["figname"])
    plt.savefig(__outplotdir__ + plotdic["figname"], dpi=128)
    plt.close()
    exit(1)

def corr_nu_heat_rate_and_wind():

    # sim = "LS220_M14691268_M0_LK_SR"
    sim = "BLh_M13641364_M0_LK_SR"
    t1, t2 = 60, 80

    v_n1 = "Q_eff_nua" # optd_0_nua optd_1_nua
    v_n2 = "dens_unb_bern"
    plane = "xz"
    plotdic={"vmin":1e-5, "vmax":1.e-2,
             "xmin":1e-15, "xmax":1e-13,
             "ymin":5e-10, "ymax":1e-8,
             "cmap": "jet",
             "set_under":"black",
             "set_over":"red",
             "yscale":"log",
             "xscale":"log",
             "figname": "corr_{}_{}_{}_plane_{}.png".format(sim, v_n1, v_n2, plane)
             }

    o_methods = LOAD_RES_CORR(sim)
    o_methods.set_corr_fname_intro = "{}_corr_".format(plane)
    _, iterations, times = o_methods.get_ittime("profiles", "prof")
    iterations = iterations[(times >= t1*1.e-3) & (times < t2*1.e-3)]
    times = times[(times >= t1*1.e-3) & (times < t2*1.e-3)]
    assert len(iterations) == len(times)
    all_x_arr = np.zeros(0,)
    all_y_arr = np.zeros(0,)
    all_z_arr = []
    #
    for it, t in zip(iterations, times):
        print("loading: {:d} {:.1f}".format(it, t*1.e3)),
        table = o_methods.get_res_corr(it, v_n1, v_n2)
        x_arr = np.array(table[0, 1:])  # * 6.176269145886162e+17
        y_arr = np.array(table[1:, 0])
        z_arr = np.array(table[1:, 1:])
        print(z_arr.shape)
        all_x_arr = x_arr
        all_y_arr = y_arr
        all_z_arr.append(z_arr)
    #
    print("x: {} -> {}".format(all_x_arr.min(), all_x_arr.max()))
    print("y: {} -> {}".format(all_y_arr.min(), all_y_arr.max()))
    #
    print(np.array(all_z_arr).shape)
    all_z_arr = np.sum(np.array(all_z_arr), axis=0)
    delta_t = t2-t1
    all_z_arr = np.array(all_z_arr)/delta_t
    print(np.sum(all_z_arr))
    all_z_arr = all_z_arr / np.sum(all_z_arr)
    #
    print(all_x_arr.shape)
    print(all_y_arr.shape)
    print(all_z_arr.shape)
    #
    #
    fig = plt.figure(figsize=(4.2, 3.6))
    ax = fig.add_subplot(111)
    #
    norm = LogNorm(plotdic["vmin"], plotdic["vmax"])
    im = ax.pcolormesh(all_x_arr, all_y_arr, all_z_arr, norm=norm, cmap=plotdic["cmap"])  # , vmin=dic["vmin"], vmax=dic["vmax"])
    im.set_rasterized(True)
    im.cmap.set_over(plotdic['set_under'])
    im.cmap.set_over(plotdic['set_over'])
    #
    ax.set_yscale(plotdic["yscale"])
    ax.set_xscale(plotdic["xscale"])
    #
    ax.set_xlabel(v_n1.replace('_', '\_'), fontsize="large")
    ax.set_ylabel(v_n2.replace('_', '\_'), fontsize="large")
    #
    ax.set_xlim(plotdic["xmin"], plotdic["xmax"])
    ax.set_ylim(plotdic["ymin"], plotdic["ymax"])
    #
    ax.tick_params(axis='both', which='both', labelleft=True,
                      labelright=False, tick1On=True, tick2On=True,
                      labelsize=12,
                      direction='in',
                      bottom=True, top=True, left=True, right=True)
    ax.minorticks_on()
    #
    ax.set_title("Correlation in 'xz' plane")
    #
    clb = fig.colorbar(im, ax=ax)
    plt.tight_layout()
    clb.ax.set_title(r"M/sum(M)", fontsize=11)
    clb.ax.tick_params(labelsize=11)
    #
    print("plotted: \n")
    print(__outplotdir__ + plotdic["figname"])
    plt.savefig(__outplotdir__ + plotdic["figname"], dpi=256)
    plt.close()

    # corr_dic = o_methods.corr_task_dic_q_eff_nua_ye
    #
    # for it in iterations:
    #     o_data = LOAD_PROFILE_XYXZ(sim)
    #     x_data = o_data.get_data(it=it, rl=3, plane="xz", v_n=v_n1)
    #     y_data = o_data.get_data(it=it, rl=3, plane="xz", v_n=v_n2)
    #     mask = o_data.get_data(it=it, rl=3, plane="xz", v_n="rl_mask")
    #     print("{}: {} -> {}".format(v_n1, x_data.min(), x_data.max()))
    #     print("{}: {} -> {}".format(v_n2, y_data.min(), y_data.max()))
    #     edges, corr = __corr_for_a_slice()
    #     #
    #     dfile = h5py.File(fpath, "w")
    #     dfile.create_dataset("mass", data=mass, dtype=np.float32)
    #     for dic, edge in zip(corr_task_dic, edges):
    #         dfile.create_dataset("{}".format(dic["v_n"]), data=edge)
    #     dfile.close()

    pass

''' ----------------------- MODULES ------------------------- '''

def plot_total_ejecta_flux(tasks, plotdic):

    fig = plt.figure(figsize=plotdic["figsize"])
    ax = fig.add_subplot(111)
    #
    # labels

    for task in tasks:
        if task["type"] == "all" or task["type"] == plotdic["type"]:
            o_data = ADD_METHODS_ALL_PAR(task["sim"])
            times, masses = o_data.get_time_data_arrs(task["v_n"], det=task["det"], mask=task["mask"])
            #
            tmerg = o_data.get_par("tmerg")
            if task["v_n"] == "Mej" and plotdic["yscale"] != "log": masses_pl = masses * 1e2 # 1e-2 Msun
            else: masses_pl = masses
            times = (times-tmerg) # ms
            if not "plot" in task.keys():
                ax.plot(times*1e3, masses_pl, color=task["color"], ls=task["ls"], lw=task["lw"],
                        alpha=task["alpha"], label=task["label"])
            else:
                ax.plot(times*1e3, masses_pl, **task["plot"])

            if len(task["ext"].keys()) > 0:
                dic = task["ext"]
                # {"type": "poly", "order": 4, "t1": 20, "t2": None, "show": True}
                if dic["t1"] != None:
                    masses = masses[times>float(dic["t1"])/1.e3]
                    times = times[times>float(dic["t1"])/1.e3]
                if dic["t2"] != None:
                    masses = masses[times<=float(dic["t2"])/1.e3]
                    times = times[times<=float(dic["t2"])/1.e3]
                assert len(times) > 0
                #
                new_x = np.array(dic["xarr"]) / 1.e3
                if dic["type"] == "poly":
                    fitx, fity = fit_polynomial(times, masses, dic["order"], np.zeros(0,), new_x)
                    if task["v_n"] == "Mej" and plotdic["yscale"] != "log": fity = fity * 1e2  # 1e-2 Msun
                    if len(dic["plot"].keys()) > 0:
                        ax.plot(fitx*1e3, fity, **dic["plot"])
                elif dic["type"] == "int1d":
                    fitx = new_x
                    fity = interpolate.interp1d(times, masses, kind="linear", fill_value="extrapolate")(new_x)
                    if task["v_n"] == "Mej" and plotdic["yscale"] != "log": fity = fity * 1e2  # 1e-2 Msun
                    if len(dic["plot"].keys()) > 0:
                        ax.plot(fitx*1e3, fity, **dic["plot"])

    if plotdic["add_legend"]:
        tmp = {"color": "gray", "marker": "x", "markersize": 5, "linestyle": ':', "linewidth":0.5, "label":"Linear extrapolation"}
        ax.plot([-1, -1],[-2., -2], **tmp)

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
    if plotdic["add_legend"]:
        han, lab = ax.get_legend_handles_labels()
        ax.add_artist(ax.legend(han[:-1], lab[:-1], **plotdic["legend"])) # default
        tmp = copy.deepcopy(plotdic["legend"])
        tmp["loc"] = "upper left"
        tmp["bbox_to_anchor"] = (0., 1.)
        ax.add_artist(ax.legend([han[-1]], [lab[-1]], **tmp)) # for extapolation
    else:
        ax.legend(**plotdic["legend"])

    plt.tight_layout()
    #

    print("plotted: \n")
    print(plotdic["figname"])
    plt.savefig(plotdic["figname"], dpi=128)
    if plotdic["savepdf"]: plt.savefig(plotdic["figname"].replace(".png", ".pdf"))
    plt.close()
    # exit(1)

def plot_total_ejecta_hist(tasks, plotdic):

    fig = plt.figure(figsize=plotdic["figsize"])
    ax = fig.add_subplot(111)
    #
    # labels

    for task in tasks:
        if task["type"] == "all" or task["type"] == plotdic["type"]:

            # os.system("python /data01/numrel/vsevolod.nedora/bns_ppr_tools/outflowed.py -s {} -t hist --v_n {} "
            #           "--overwrite yes -d {} -m {}".format(task["sim"], task["v_n"], task["det"], task["mask"]))

            o_data = ADD_METHODS_ALL_PAR(task["sim"])
            hist = o_data.get_outflow_hist(det=task["det"], mask=task["mask"], v_n = task["v_n"])
            dataarr = hist[0, :]
            massarr = hist[1, :]
            #
            if task["v_n"] == "theta": dataarr = 90 - (dataarr * 180 / np.pi)
            if task["v_n"] == "phi": dataarr = dataarr / np.pi * 180.
            #
            if task['normalize']: massarr /= np.sum(massarr)
            #
            # print(dataarr); exit(1)
            if not "plot" in task.keys():
                ax.plot(dataarr, massarr, color=task["color"], ls=task["ls"], lw=task["lw"], alpha=task["alpha"],
                        label=task["label"], drawstyle="steps")
            else:
                ax.plot(dataarr, massarr, **task["plot"])

    ax.set_yscale(plotdic["yscale"])
    ax.set_xscale(plotdic["xscale"])

    ax.set_xlabel(plotdic["xlabel"], fontsize = plotdic["fontsize"])  # , fontsize=11)
    ax.set_ylabel(plotdic["ylabel"], fontsize = plotdic["fontsize"])  # , fontsize=11)

    ax.set_xlim(plotdic["xmin"], plotdic["xmax"])
    ax.set_ylim(plotdic["ymin"], plotdic["ymax"])
    #
    ax.tick_params(axis='both', which='both', labelleft=True,
                   labelright=False, tick1On=True, tick2On=True,
                   labelsize=plotdic["fontsize"],
                   direction='in',
                   bottom=True, top=True, left=True, right=True)
    ax.minorticks_on()
    # #
    if "title" in plotdic.keys(): ax.set_title(plotdic["title"])
    # #
    if len(plotdic["legend"].keys())>0 : ax.legend(**plotdic["legend"])
    #
    plt.tight_layout()
    #
    print("plotted: \n")
    print(plotdic["figname"])
    plt.savefig(plotdic["figname"], dpi=128)
    if "savepdf" in plotdic.keys() and plotdic["savepdf"]:
        plt.savefig(plotdic["figname"].replace(".png", ".pdf"))
    plt.close()
    # exit(1)

def custom_plot_total_ejecta_hist(tasks, plotdics):

    fig, axes = plt.subplots(figsize=plotdics[0]["figsize"], ncols=len(plotdics), nrows=1, sharey=True)
    # ax = fig.add_subplot(111)
    i = 0
    for ax, plotdic in zip(axes, plotdics):
        #
        # labels

        for task in tasks:
            if task["type"] == "all" or task["type"] == plotdic["type"]:

                # os.system("python /data01/numrel/vsevolod.nedora/bns_ppr_tools/outflowed.py -s {} -t hist --v_n {} "
                #           "--overwrite yes -d {} -m {}".format(task["sim"], task["v_n"], task["det"], task["mask"]))

                o_data = ADD_METHODS_ALL_PAR(task["sim"])
                hist = o_data.get_outflow_hist(det=task["det"], mask=task["mask"], v_n = plotdic["task_v_n"])
                dataarr = hist[0, :]
                massarr = hist[1, :]
                #
                if plotdic["task_v_n"] == "theta": dataarr = 90 - (dataarr * 180 / np.pi)
                if plotdic["task_v_n"] == "phi": dataarr = dataarr / np.pi * 180.
                #
                if task['normalize']: massarr /= np.sum(massarr)
                #
                # print(dataarr); exit(1)
                if not "plot" in task.keys():
                    ax.plot(dataarr, massarr, color=task["color"], ls=task["ls"], lw=task["lw"], alpha=task["alpha"],
                            label=task["label"], drawstyle="steps")
                else:
                    ax.plot(dataarr, massarr, **task["plot"])

        ax.set_yscale(plotdic["yscale"])
        ax.set_xscale(plotdic["xscale"])

        ax.set_xlabel(plotdic["xlabel"], fontsize = plotdic["fontsize"])  # , fontsize=11)
        if i == 0: ax.set_ylabel(plotdic["ylabel"], fontsize = plotdic["fontsize"])  # , fontsize=11)

        ax.set_xlim(plotdic["xmin"], plotdic["xmax"])
        ax.set_ylim(plotdic["ymin"], plotdic["ymax"])
        #
        ax.tick_params(axis='both', which='both', labelleft=True,
                       labelright=False, tick1On=True, tick2On=True,
                       labelsize=plotdic["fontsize"],
                       direction='in',
                       bottom=True, top=True, left=True, right=True)
        ax.minorticks_on()
        # #
        if "title" in plotdic.keys(): ax.set_title(plotdic["title"])
        # #
        if i > 0:
            ax.tick_params(labelleft=False)
            #ax.get_yaxis().set_ticks([])
            # ax.get_yaxis().set_visible(False)
        if len(plotdic["legend"].keys())>0 : ax.legend(**plotdic["legend"])
        i = i + 1
    #
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.)
    #
    print("plotted: \n")
    print(plotdics[0]["figname"])
    plt.savefig(plotdics[0]["figname"], dpi=128)
    if "savepdf" in plotdics[0].keys() and plotdics[0]["savepdf"]:
        plt.savefig(plotdics[0]["figname"].replace(".png", ".pdf"))
    plt.close()
    # exit(1)

def plot_total_ejecta_corr(task, plotdic):

    sim = task["sim"]  # "BLh_M13641364_M0_LK_SR"

    v_n_x = task["v_n_x"]  # "Q_eff_nua"  # optd_0_nua optd_1_nua
    v_n_y= task["v_n_y"]  # "dens_unb_bern"
    mask = task["mask"]
    det = task["det"]

    o_methods = ADD_METHODS_ALL_PAR(sim)
    #
    table = o_methods.load_outflow_corr(det, mask, v_n_x, v_n_y)
    x_arr = table[:, 0]
    y_arr = table[0, :]
    mass = table[1:, 1:].T

    if task["normalize"]:
        mass = mass / np.sum(mass)
    mass = np.maximum(mass, 1e-10)

    if v_n_x == "theta":
        all_x_arr = 90 - (x_arr * 180 / np.pi)
    if v_n_y == "theta":
        all_y_arr = 90 - (y_arr * 180 / np.pi)

    #
    print("x: {} -> {}".format(x_arr.min(), x_arr.max()))
    print("y: {} -> {}".format(y_arr.min(), y_arr.max()))
    #
    print(np.array(mass).shape)

    #
    print(x_arr.shape)
    print(y_arr.shape)
    print(mass.shape)


    # -------------------------------------- PLOTTING
    fig = plt.figure(figsize=(4., 2.5))
    ax = fig.add_subplot(111)
    #
    norm = LogNorm(plotdic["vmin"], plotdic["vmax"])
    im = ax.pcolormesh(x_arr, y_arr, mass, norm=norm, cmap=plotdic["cmap"])  # , vmin=dic["vmin"], vmax=dic["vmax"])
    im.set_rasterized(True)
    im.cmap.set_over(plotdic['set_under'])
    im.cmap.set_over(plotdic['set_over'])
    #
    ax.set_yscale(plotdic["yscale"])
    ax.set_xscale(plotdic["xscale"])
    #
    ax.set_xlabel(plotdic["xlabel"], fontsize=11)
    ax.set_ylabel(plotdic["ylabel"], fontsize=11)
    #
    ax.set_xlim(plotdic["xmin"], plotdic["xmax"])
    ax.set_ylim(plotdic["ymin"], plotdic["ymax"])
    #
    if "text" in plotdic.keys():
        plotdic["text"]["transform"] = ax.transAxes
        ax.text(**plotdic["text"])
    #
    ax.tick_params(axis='both', which='both', labelleft=True,
                   labelright=False, tick1On=True, tick2On=True,
                   labelsize=12,
                   direction='in',
                   bottom=True, top=True, left=True, right=True)
    ax.minorticks_on()
    #
    if "title" in plotdic.keys(): ax.set_title(plotdic["title"])
    #
    clb = fig.colorbar(im, ax=ax)
    plt.tight_layout()
    clb.ax.set_title(plotdic["clabel"], fontsize=11)
    clb.ax.tick_params(labelsize=11)
    #
    print("plotted: \n")
    print(__outplotdir__ + plotdic["figname"])
    plt.savefig(__outplotdir__ + plotdic["figname"], dpi=128)
    plt.close()

def plot_total_ejecta_timecorr(task, plotdic):

    o_data = ADD_METHODS_ALL_PAR(task["sim"])
    table = o_data.get_outflow_timecorr(task["det"], task["mask"], task["v_n"])

    x_arr = np.array(table[0, 1:])  # * 6.176269145886162e+17
    y_arr = np.array(table[1:, 0])
    z_arr = np.array(table[1:, 1:])

    if task["v_n"] == "theta": y_arr = 90 - (y_arr * 180 / np.pi)

    #
    print("x: {} -> {}".format(x_arr.min(), x_arr.max()))
    print("y: {} -> {}".format(y_arr.min(), y_arr.max()))
    #
    print(np.array(z_arr).shape)

    if task["normalize"]:
        z_arr = z_arr / np.sum(z_arr)
        z_arr = np.maximum(z_arr, 1e-10)
    #
    print(x_arr.shape)
    print(y_arr.shape)
    print(z_arr.shape)

    # -------------------------------------- PLOTTING
    fig = plt.figure(figsize=(4., 2.5))
    ax = fig.add_subplot(111)
    #
    norm = LogNorm(plotdic["vmin"], plotdic["vmax"])
    im = ax.pcolormesh(x_arr, y_arr, z_arr, norm=norm,
                       cmap=plotdic["cmap"])  # , vmin=dic["vmin"], vmax=dic["vmax"])
    im.set_rasterized(True)
    im.cmap.set_over(plotdic['set_under'])
    im.cmap.set_over(plotdic['set_over'])
    #
    ax.set_yscale(plotdic["yscale"])
    ax.set_xscale(plotdic["xscale"])
    #
    ax.set_xlabel(plotdic["xlabel"], fontsize=11)
    ax.set_ylabel(plotdic["ylabel"], fontsize=11)
    #
    ax.set_xlim(plotdic["xmin"], plotdic["xmax"])
    ax.set_ylim(plotdic["ymin"], plotdic["ymax"])
    #
    if "text" in plotdic.keys():
        plotdic["text"]["transform"] = ax.transAxes
        ax.text(**plotdic["text"])
    #
    ax.tick_params(axis='both', which='both', labelleft=True,
                   labelright=False, tick1On=True, tick2On=True,
                   labelsize=12,
                   direction='in',
                   bottom=True, top=True, left=True, right=True)
    ax.minorticks_on()
    #
    ax.set_title(plotdic["title"])
    #
    clb = fig.colorbar(im, ax=ax)
    plt.tight_layout()
    clb.ax.set_title(plotdic["clabel"], fontsize=11)
    clb.ax.tick_params(labelsize=11)
    #
    print("plotted: \n")
    print(__outplotdir__ + plotdic["figname"])
    plt.savefig(__outplotdir__ + plotdic["figname"], dpi=128)
    plt.close()

def plot_corr_qeff_u_0(task, plotdic):


    sim = task["sim"] # "BLh_M13641364_M0_LK_SR"
    t1, t2 = task["t1"], task["t2"]# 60, 80

    v_n_x = task["v_n_x"] # "Q_eff_nua"  # optd_0_nua optd_1_nua
    v_n_y = task["v_n_y"] # "dens_unb_bern"
    plane = task["plane"]# "xz"
    mask = task["mask"]

    o_methods = LOAD_RES_CORR(sim)
    o_methods.set_corr_fname_intro = "{}/{}_corr_".format(mask, plane)
    _, iterations, times = o_methods.get_ittime("profiles", "prof")
    iterations = iterations[(times >= t1 * 1.e-3) & (times < t2 * 1.e-3)]
    times = times[(times >= t1 * 1.e-3) & (times < t2 * 1.e-3)]
    assert len(iterations) == len(times)
    all_x_arr = np.zeros(0, )
    all_y_arr = np.zeros(0, )
    all_z_arr = []
    #
    for it, t in zip(iterations, times):
        print("loading: {:d} {:.1f}".format(it, t * 1.e3)),
        table = o_methods.get_res_corr(it, v_n_x, v_n_y)
        x_arr = np.array(table[0, 1:])  # * 6.176269145886162e+17
        y_arr = np.array(table[1:, 0]) # * -1.
        z_arr = np.array(table[1:, 1:])
        print(z_arr.shape)
        all_x_arr = x_arr
        all_y_arr = y_arr
        all_z_arr.append(z_arr)
    #
    if v_n_x == "theta":
        all_x_arr = 90 - (all_x_arr * 180 / np.pi)
    if v_n_y == "theta":
        all_y_arr = 90 - (all_y_arr * 180 / np.pi)
    if v_n_y == "hu_0":
        all_y_arr = all_y_arr * -1.
    if v_n_x == "hu_0":
        all_y_arr = all_y_arr * -1.
    #
    print("x: {} -> {}".format(all_x_arr.min(), all_x_arr.max()))
    print("y: {} -> {}".format(all_y_arr.min(), all_y_arr.max()))
    #
    print(np.array(all_z_arr).shape)
    all_z_arr = np.sum(np.array(all_z_arr), axis=0)
    delta_t = t2 - t1
    all_z_arr = np.array(all_z_arr) / delta_t
    print("mass", np.sum(all_z_arr))
    if task["normalize"]:
        all_z_arr = all_z_arr / np.sum(all_z_arr)
        all_z_arr= np.maximum(all_z_arr, 1e-10)
    #
    print(all_x_arr.shape)
    print(all_y_arr.shape)
    print(all_z_arr.shape)


    # -------------------------------------- PLOTTING
    fig = plt.figure(figsize=(4., 2.5))
    ax = fig.add_subplot(111)
    #
    norm = LogNorm(plotdic["vmin"], plotdic["vmax"])
    im = ax.pcolormesh(all_x_arr, all_y_arr, all_z_arr, norm=norm,
                       cmap=plotdic["cmap"])  # , vmin=dic["vmin"], vmax=dic["vmax"])
    im.set_rasterized(True)
    im.cmap.set_over(plotdic['set_under'])
    im.cmap.set_over(plotdic['set_over'])
    #
    ax.set_yscale(plotdic["yscale"])
    ax.set_xscale(plotdic["xscale"])
    #
    ax.set_xlabel(plotdic["xlabel"], fontsize=11)
    ax.set_ylabel(plotdic["ylabel"], fontsize=11)
    #
    ax.set_xlim(plotdic["xmin"], plotdic["xmax"])
    ax.set_ylim(plotdic["ymin"], plotdic["ymax"])
    #
    if "text" in plotdic.keys():
        plotdic["text"]["transform"] = ax.transAxes
        ax.text(**plotdic["text"])
    #
    ax.tick_params(axis='both', which='both', labelleft=True,
                   labelright=False, tick1On=True, tick2On=True,
                   labelsize=12,
                   direction='in',
                   bottom=True, top=True, left=True, right=True)
    ax.minorticks_on()
    #
    ax.set_title("Correlation in 'xz' plane")
    #
    clb = fig.colorbar(im, ax=ax)
    plt.tight_layout()
    clb.ax.set_title(plotdic["clabel"], fontsize=11)
    clb.ax.tick_params(labelsize=11)
    #
    print("plotted: \n")
    print(__outplotdir__ + plotdic["figname"])
    plt.savefig(__outplotdir__ + plotdic["figname"], dpi=128)
    plt.close()

def plot_slice_2halfs(task, plotdic):

    sim = task["sim"]  # "BLh_M13641364_M0_LK_SR"
    v_n_x = task["v_n_x"]  # "Q_eff_nua"  # optd_0_nua optd_1_nua
    v_n_y = task["v_n_y"]  # "dens_unb_bern"
    v_n_left = task["v_n_left"]
    v_n_right = task["v_n_right"]
    plane = task["plane"]  # "xz"
    rl = task["rl"]
    plot_dir = plotdic["plot_dir"]

    if not os.path.isdir(plot_dir):
        print("creating: {}".format(plot_dir))
        os.mkdir(plot_dir)
    #

    d3class = MAINMETHODS_STORE_XYXZ(sim)
    d1class = ADD_METHODS_ALL_PAR(sim)
    #
    # tmerg = d1class.get_par("tmerg")
    # time_ = d3class.get_time_for_it(it, "profiles", "prof")

    if "times" in task.keys() and not "iterations" in task.keys():
        times = task["times"]
        if isinstance(times, str):
            val = float(str(times[1:]))
            iterations = np.array(d3class.list_iterations, dtype=int)
            alltimes = (d3class.times - d1class.get_par("tmerg")) * 1e3
            iterations = iterations[alltimes >= val]
            alltimes = alltimes[alltimes >= val]
        else:
            raise NameError("no method set for times:{}".format(times))
    elif "iterations" in task.keys() and not "times" in task.keys():
        iterations = np.array(task["iterations"], dtype=int)
    else:
        raise NameError("neither 'times' nor 'iterations' are set in the plotdic. ")


    #
    # print("iterations")
    # print(iterations)

    for i_it, it in enumerate(iterations):
        time_ = d3class.get_time_for_it(it, "profiles", "prof")
        print("it:{} t:{} [ms]".format(it, time_*1e3))
        try:
            #
            tmerg = d1class.get_par("tmerg")
            #
            data_left_arr = d3class.get_comp_data(it, rl, plane, v_n_left)
            data_right_arr = d3class.get_comp_data(it, rl, plane, v_n_right)
            #
            if v_n_left == "hu_0": data_left_arr = data_left_arr * -1.
            if v_n_right == "hu_0": data_right_arr = data_right_arr * -1.
            #
            x_arr = d3class.get_data(it, rl, plane, v_n_x) * constant_length
            z_arr = d3class.get_data(it, rl, plane, v_n_y) * constant_length
            #
            data_left_arr = np.maximum(data_left_arr, 1e-15)
            data_left_arr = np.ma.masked_array(data_left_arr, x_arr > 0)
            data_right_arr = np.maximum(data_right_arr, 1e-15)
            data_right_arr = np.ma.masked_array(data_right_arr, x_arr < 0)

            # -------------------------------------- PLOTTING
            fig = plt.figure(figsize=(6., 2.5))
            ax = fig.add_subplot(111)

            # left
            if plotdic["norm_left"] == "linear":
                norm = Normalize(plotdic["vmin_left"], plotdic["vmax_left"])
            else:
                norm = LogNorm(plotdic["vmin_left"], plotdic["vmax_left"])
            im_left = ax.pcolormesh(x_arr, z_arr, data_left_arr, norm=norm, cmap=plotdic["cmap_left"])  # , vmin=dic["vmin"], vmax=dic["vmax"])
            im_left.set_rasterized(True)
            if "set_under_left" in plotdic.keys(): im_left.cmap.set_over(plotdic['set_under_left'])
            if "set_over_left" in plotdic.keys(): im_left.cmap.set_over(plotdic['set_over_left'])
            #
            if plotdic["norm_right"] == "linear":
                norm = Normalize(plotdic["vmin_right"], plotdic["vmax_right"])
            else:
                norm = LogNorm(plotdic["vmin_right"], plotdic["vmax_right"])
            im_right = ax.pcolormesh(x_arr, z_arr, data_right_arr, norm=norm, cmap=plotdic["cmap_right"])  # , vmin=dic["vmin"], vmax=dic["vmax"])
            im_right.set_rasterized(True)
            if "set_under_right" in plotdic.keys(): im_right.cmap.set_over(plotdic['set_under_right'])
            if "set_over_right" in plotdic.keys(): im_right.cmap.set_over(plotdic['set_over_right'])

            #
            ax.set_yscale(plotdic["yscale"])
            ax.set_xscale(plotdic["xscale"])
            #
            ax.set_xlabel(plotdic["xlabel"], fontsize=11)
            ax.set_ylabel(plotdic["ylabel"], fontsize=11)
            #
            if plotdic["xmin"] == "auto" or plotdic["xmax"] =="auto" or plotdic["ymin"] == "auto" or plotdic["ymax"]=="auto":
                xmin, xmax, _, _, ymin, ymax = UTILS.get_xmin_xmax_ymin_ymax_zmin_zmax(rl)
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
            else:
                ax.set_xlim(plotdic["xmin"], plotdic["xmax"])
                ax.set_ylim(plotdic["ymin"], plotdic["ymax"])
            #
            if "text" in plotdic.keys():
                plotdic["text"]["transform"] = ax.transAxes
                ax.text(**plotdic["text"])
            #
            ax.set_title(r'$t-t_{merg}:$' + r'${:.1f}$'.format((time_ - tmerg) * 1e3))
            #
            ax.tick_params(axis='both', which='both', labelleft=True,
                           labelright=False, tick1On=True, tick2On=True,
                           labelsize=12,
                           direction='in',
                           bottom=True, top=True, left=True, right=True)
            ax.minorticks_on()
            #
            clb = fig.colorbar(im_left, ax=ax, anchor=(0.0, 0.5))
            clb.ax.set_title(plotdic["clabel_left"], fontsize=11)
            clb.ax.tick_params(labelsize=11)
            clb.ax.minorticks_off()
            #
            clb = fig.colorbar(im_right, ax=ax)
            clb.ax.set_title(plotdic["clabel_right"], fontsize=11)
            clb.ax.tick_params(labelsize=11)
            # clb.ax.minorticks_off()
            #
            print("plotted: \n")
            if plotdic["figname"] == "it":
                figpath = plot_dir + str(it) + ".png"
            elif plotdic["figname"] == "time":
                figpath = plotdic + str(int(time_)) + ".png"
            else:
                figpath = plot_dir + plotdic["figname"]
            plt.tight_layout()
            print(figpath)
            plt.savefig(figpath, dpi=128)
            plt.close()
        except NameError:
            Printcolor.red("NameError. Probably no neutrino data")
        # except:
        #     raise ValueError("Something is wrong.")

def old_plot_slice_2halfs__with_morror_function(task, plotdic):

    sim = task["sim"]  # "BLh_M13641364_M0_LK_SR"
    v_n_x = task["v_n_x"]  # "Q_eff_nua"  # optd_0_nua optd_1_nua
    v_n_y = task["v_n_y"]  # "dens_unb_bern"
    v_n_left = task["v_n_left"]
    v_n_right = task["v_n_right"]
    plane = task["plane"]  # "xz"
    rl = task["rl"]
    plot_dir = plotdic["plot_dir"]

    if not os.path.isdir(plot_dir):
        print("creating: {}".format(plot_dir))
        os.mkdir(plot_dir)
    #

    d3class = MAINMETHODS_STORE_XYXZ(sim)
    d1class = ADD_METHODS_ALL_PAR(sim)
    #
    # tmerg = d1class.get_par("tmerg")
    # time_ = d3class.get_time_for_it(it, "profiles", "prof")

    if "times" in task.keys() and not "iterations" in task.keys():
        times = task["times"]
        if isinstance(times, str):
            val = float(str(times[1:]))
            iterations = np.array(d3class.list_iterations, dtype=int)
            alltimes = (d3class.times - d1class.get_par("tmerg")) * 1e3
            iterations = iterations[alltimes >= val]
            alltimes = alltimes[alltimes >= val]
        else:
            raise NameError("no method set for times:{}".format(times))
    elif "iterations" in task.keys() and not "times" in task.keys():
        iterations = np.array(task["iterations"], dtype=int)
    else:
        raise NameError("neither 'times' nor 'iterations' are set in the plotdic. ")


    #
    # print("iterations")
    # print(iterations)

    for i_it, it in enumerate(iterations):
        time_ = d3class.get_time_for_it(it, "profiles", "prof")
        print("it:{} t:{} [ms]".format(it, time_*1e3))
        try:
            #
            tmerg = d1class.get_par("tmerg")
            #
            data_left_arr = d3class.get_comp_data(it, rl, plane, v_n_left)
            data_right_arr = d3class.get_comp_data(it, rl, plane, v_n_right)
            #
            if v_n_left == "hu_0": data_left_arr = data_left_arr * -1.
            if v_n_right == "hu_0": data_right_arr = data_right_arr * -1.
            #
            x_arr = d3class.get_data(it, rl, plane, v_n_x) * constant_length
            z_arr = d3class.get_data(it, rl, plane, v_n_y) * constant_length
            #
            data_left_arr = np.maximum(data_left_arr, 1e-15)
            data_left_arr = np.ma.masked_array(data_left_arr, x_arr > 0)
            data_right_arr = np.maximum(data_right_arr, 1e-15)
            data_right_arr = np.ma.masked_array(data_right_arr, x_arr < 0)

            # -------------------------------------- PLOTTING
            fig = plt.figure(figsize=plotdic["figsize"])
            #ax = fig.add_subplot(111)
            if "mirror_z" in plotdic.keys() and plotdic["mirror_z"]:
                ax = fig.add_subplot(211)
            else:
                ax = fig.add_subplot(111)

            # --- left
            if plotdic["norm_left"] == "linear": norm = Normalize(plotdic["vmin_left"], plotdic["vmax_left"])
            else: norm = LogNorm(plotdic["vmin_left"], plotdic["vmax_left"])
            im_left = ax.pcolormesh(x_arr, z_arr, data_left_arr, norm=norm, cmap=plotdic["cmap_left"])  # , vmin=dic["vmin"], vmax=dic["vmax"])
            im_left.set_rasterized(True)
            if "set_under_left" in plotdic.keys(): im_left.cmap.set_over(plotdic['set_under_left'])
            if "set_over_left" in plotdic.keys(): im_left.cmap.set_over(plotdic['set_over_left'])
            # --- right
            if plotdic["norm_right"] == "linear": norm = Normalize(plotdic["vmin_right"], plotdic["vmax_right"])
            else:  norm = LogNorm(plotdic["vmin_right"], plotdic["vmax_right"])
            im_right = ax.pcolormesh(x_arr, z_arr, data_right_arr, norm=norm, cmap=plotdic["cmap_right"])  # , vmin=dic["vmin"], vmax=dic["vmax"])
            im_right.set_rasterized(True)
            if "set_under_right" in plotdic.keys(): im_right.cmap.set_over(plotdic['set_under_right'])
            if "set_over_right" in plotdic.keys(): im_right.cmap.set_over(plotdic['set_over_right'])

            #
            ax.set_yscale(plotdic["yscale"])
            ax.set_xscale(plotdic["xscale"])
            #
            ax.set_xlabel(plotdic["xlabel"], fontsize=plotdic["fontsize"])
            ax.set_ylabel(plotdic["ylabel"], fontsize=plotdic["fontsize"])
            #
            if plotdic["xmin"] == "auto" or plotdic["xmax"] =="auto" or plotdic["ymin"] == "auto" or plotdic["ymax"]=="auto":
                xmin, xmax, _, _, ymin, ymax = UTILS.get_xmin_xmax_ymin_ymax_zmin_zmax(rl)
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
            else:
                ax.set_xlim(plotdic["xmin"], plotdic["xmax"])
                ax.set_ylim(plotdic["ymin"], plotdic["ymax"])
            #
            if "text" in plotdic.keys():
                plotdic["text"]["transform"] = ax.transAxes
                ax.text(**plotdic["text"])
            #
            ax.set_title(r'$t-t_{merg}:$' + r'${:.1f}$'.format((time_ - tmerg) * 1e3))
            #
            ax.tick_params(axis='both', which='both', labelleft=True,
                           labelright=False, tick1On=True, tick2On=True,
                           labelsize=plotdic["fontsize"],
                           direction='in',
                           bottom=True, top=True, left=True, right=True)
            ax.minorticks_on()

            if "mirror_z" in plotdic.keys() and plotdic["mirror_z"]:
                ax = fig.add_subplot(212)

                # --- left
                if plotdic["norm_left"] == "linear": norm = Normalize(plotdic["vmin_left"], plotdic["vmax_left"])
                else: norm = LogNorm(plotdic["vmin_left"], plotdic["vmax_left"])
                im_left2 = ax.pcolormesh(x_arr, -1 * z_arr, data_left_arr, norm=norm,
                                        cmap=plotdic["cmap_left"])  # , vmin=dic["vmin"], vmax=dic["vmax"])
                im_left2.set_rasterized(True)
                if "set_under_left" in plotdic.keys(): im_left2.cmap.set_over(plotdic['set_under_left'])
                if "set_over_left" in plotdic.keys(): im_left2.cmap.set_over(plotdic['set_over_left'])
                # --- right
                if plotdic["norm_right"] == "linear":
                    norm = Normalize(plotdic["vmin_right"], plotdic["vmax_right"])
                else:
                    norm = LogNorm(plotdic["vmin_right"], plotdic["vmax_right"])
                im_right2 = ax.pcolormesh(x_arr, -1 * z_arr, data_right_arr, norm=norm,
                                         cmap=plotdic["cmap_right"])  # , vmin=dic["vmin"], vmax=dic["vmax"])
                im_right2.set_rasterized(True)
                if "set_under_right" in plotdic.keys(): im_right2.cmap.set_over(plotdic['set_under_right'])
                if "set_over_right" in plotdic.keys(): im_right2.cmap.set_over(plotdic['set_over_right'])

                #
                ax.set_yscale(plotdic["yscale"])
                ax.set_xscale(plotdic["xscale"])
                #
                if not "mirror_z" in plotdic.keys() or not plotdic["mirror_z"]:
                    ax.set_xlabel(plotdic["xlabel"], fontsize=plotdic["fontsize"])
                ax.set_ylabel(plotdic["ylabel"], fontsize=plotdic["fontsize"])
                #
                if plotdic["xmin"] == "auto" or plotdic["xmax"] == "auto" or plotdic["ymin"] == "auto" or plotdic[
                    "ymax"] == "auto":
                    xmin, xmax, _, _, ymin, ymax = UTILS.get_xmin_xmax_ymin_ymax_zmin_zmax(rl)
                    ax.set_xlim(xmin, xmax)
                    ax.set_ylim(-1 * ymin, -1 * ymax)
                else:
                    ax.set_xlim(plotdic["xmin"], plotdic["xmax"])
                    ax.set_ylim(-1 * plotdic["ymax"], -1 * plotdic["ymin"])
                #
                if "text" in plotdic.keys():
                    plotdic["text"]["transform"] = ax.transAxes
                    ax.text(**plotdic["text"])
                #
                ax.set_title(r'$t-t_{merg}:$' + r'${:.1f}$'.format((time_ - tmerg) * 1e3))
                #
                ax.tick_params(axis='both', which='both', labelleft=True,
                               labelright=False, tick1On=True, tick2On=True,
                               labelsize=plotdic["fontsize"],
                               direction='in',
                               bottom=True, top=True, left=True, right=True)
                ax.minorticks_on()

            plt.subplots_adjust(**{"hspace":0, "wspace":0})



            # clb = fig.colorbar(im_left, ax=ax, anchor=(0.0, 0.5))
            # clb.ax.set_title(plotdic["clabel_left"], fontsize=11)
            # clb.ax.tick_params(labelsize=11)
            # clb.ax.minorticks_off()
            # #
            # clb = fig.colorbar(im_right, ax=ax)
            # clb.ax.set_title(plotdic["clabel_right"], fontsize=11)
            # clb.ax.tick_params(labelsize=11)

            #
            print("plotted: \n")
            if plotdic["figname"] == "it":
                figpath = plot_dir + str(it) + ".png"
            elif plotdic["figname"] == "time":
                figpath = plotdic + str(int(time_)) + ".png"
            else:
                figpath = plot_dir + plotdic["figname"]
            plt.tight_layout()
            print(figpath)
            plt.savefig(figpath, dpi=128)
            plt.close()
        except NameError:
            Printcolor.red("NameError. Probably no neutrino data")
        # except:
        #     raise ValueError("Something is wrong.")

def plot_slice_2halfs__with_morror_function(task, plotdic):

    sim = task["sim"]  # "BLh_M13641364_M0_LK_SR"
    v_n_x = task["v_n_x"]  # "Q_eff_nua"  # optd_0_nua optd_1_nua
    v_n_y = task["v_n_y"]  # "dens_unb_bern"
    v_n_left = task["v_n_left"]
    v_n_right = task["v_n_right"]
    plane = task["plane"]  # "xz"
    rl = task["rl"]
    plot_dir = plotdic["plot_dir"]

    if not os.path.isdir(plot_dir):
        print("creating: {}".format(plot_dir))
        os.mkdir(plot_dir)
    #

    d3class = MAINMETHODS_STORE_XYXZ(sim)
    d1class = ADD_METHODS_ALL_PAR(sim)
    #
    # tmerg = d1class.get_par("tmerg")
    # time_ = d3class.get_time_for_it(it, "profiles", "prof")

    if "times" in task.keys() and not "iterations" in task.keys():
        times = task["times"]
        if isinstance(times, str):
            val = float(str(times[1:]))
            iterations = np.array(d3class.list_iterations, dtype=int)
            alltimes = (d3class.times - d1class.get_par("tmerg")) * 1e3
            iterations = iterations[alltimes >= val]
            alltimes = alltimes[alltimes >= val]
        else:
            raise NameError("no method set for times:{}".format(times))
    elif "iterations" in task.keys() and not "times" in task.keys():
        iterations = np.array(task["iterations"], dtype=int)
    else:
        raise NameError("neither 'times' nor 'iterations' are set in the plotdic. ")


    #
    # print("iterations")
    # print(iterations)

    for i_it, it in enumerate(iterations):
        time_ = d3class.get_time_for_it(it, "profiles", "prof")
        print("it:{} t:{} [ms]".format(it, time_*1e3))
        if True:
            #
            tmerg = d1class.get_par("tmerg")
            #
            data_left_arr = d3class.get_comp_data(it, rl, plane, v_n_left)
            data_right_arr = d3class.get_comp_data(it, rl, plane, v_n_right)
            print("\tv_n_left: {} shape: {}".format(v_n_left ,data_left_arr.shape))
            print("\tv_n_right: {} shape: {}".format(v_n_right, data_right_arr.shape))
            #
            if "v_n_cont" in task.keys():
                cont_data_arr = d3class.get_comp_data(it, rl, plane, task["v_n_cont"])
                if task["v_n_cont"] == "rho": cont_data_arr = cont_data_arr * constant_rho
                #print(cont_data_arr)
            else: cont_data_arr = np.zeros(0, )
            #
            if v_n_left == "hu_0": data_left_arr = -1 * data_left_arr
            if v_n_right == "hu_0": data_right_arr = -1 * data_right_arr
            #
            x_arr = d3class.get_data(it, rl, plane, v_n_x) * constant_length
            z_arr = d3class.get_data(it, rl, plane, v_n_y) * constant_length
            print("initial: {}".format(x_arr.shape))
            #
            # --- mirror z and copy x to have both x[-left, +right] and z[-under, +over]
            if "mirror_z" in plotdic.keys() and plotdic["mirror_z"]:
                x_arr = np.concatenate((np.flip(x_arr,axis=1), x_arr), axis=1)
                z_arr = np.concatenate((-1.* np.flip(z_arr,axis=1), z_arr), axis=1)
                data_left_arr = np.concatenate((np.flip(data_left_arr,axis=1), data_left_arr), axis=1)
                data_right_arr = np.concatenate((np.flip(data_right_arr,axis=1), data_right_arr), axis=1)
                #
                if len(cont_data_arr)>1:
                    cont_data_arr = np.concatenate((np.flip(cont_data_arr,axis=1), cont_data_arr), axis=1)

            #
            # print("--------x----------")
            # print(x_arr[0, :])
            # print(x_arr[-1, :])
            # print(x_arr[:, 0]) # -500 500
            # print(x_arr[:, -1])  # -500 500
            # print("--------z----------")
            # print(z_arr[0,:]) # 2 - 500
            # print(z_arr[:,0])
            # print("--------data----------")
            # print(data_left_arr)
            # print(data_right_arr)


            #data_left_arr = np.maximum(data_left_arr, 1e-15)
            data_left_arr = np.ma.masked_array(data_left_arr, x_arr > 0)
            #data_right_arr = np.maximum(data_right_arr, 1e-15)
            data_right_arr = np.ma.masked_array(data_right_arr, x_arr < 0)

                # x_arr = np.concatenate((x_arr, x_arr), axis=1)
                # z_arr = np.concatenate((-1 * z_arr, z_arr),axis=1)
                # data_left_arr = np.concatenate((data_left_arr, data_left_arr),axis=1)
                # data_right_arr = np.concatenate((data_right_arr, data_right_arr), axis=1)

            print(x_arr.shape)
            print(z_arr.shape)
            print(data_left_arr.shape)

            # -------------------------------------- PLOTTING
            fig = plt.figure(figsize=plotdic["figsize"])
            #ax = fig.add_subplot(111)
            ax = fig.add_subplot(111)

            # --- left
            if plotdic["norm_left"] == "linear": norm = Normalize(plotdic["vmin_left"], plotdic["vmax_left"])
            else: norm = LogNorm(plotdic["vmin_left"], plotdic["vmax_left"])
            im_left = ax.pcolormesh(x_arr, z_arr, data_left_arr, norm=norm, cmap=plotdic["cmap_left"])  # , vmin=dic["vmin"], vmax=dic["vmax"])
            im_left.set_rasterized(True)
            if "set_under_left" in plotdic.keys(): im_left.cmap.set_over(plotdic['set_under_left'])
            if "set_over_left" in plotdic.keys(): im_left.cmap.set_over(plotdic['set_over_left'])
            # --- right
            if plotdic["norm_right"] == "linear": norm = Normalize(plotdic["vmin_right"], plotdic["vmax_right"])
            else:  norm = LogNorm(plotdic["vmin_right"], plotdic["vmax_right"])
            im_right = ax.pcolormesh(x_arr, z_arr, data_right_arr, norm=norm, cmap=plotdic["cmap_right"])  # , vmin=dic["vmin"], vmax=dic["vmax"])
            im_right.set_rasterized(True)
            if "set_under_right" in plotdic.keys(): im_right.cmap.set_over(plotdic['set_under_right'])
            if "set_over_right" in plotdic.keys(): im_right.cmap.set_over(plotdic['set_over_right'])
            # --- contour plot
            if "v_n_cont" in task.keys() and "cont_plot" in plotdic.keys():
                ax.contour(x_arr, z_arr, cont_data_arr, **plotdic["cont_plot"])
            #
            ax.set_yscale(plotdic["yscale"])
            ax.set_xscale(plotdic["xscale"])
            #
            ax.set_xlabel(plotdic["xlabel"], fontsize=plotdic["fontsize"])
            ax.set_ylabel(plotdic["ylabel"], fontsize=plotdic["fontsize"])
            #
            if plotdic["xmin"] == "auto" or plotdic["xmax"] =="auto" or plotdic["ymin"] == "auto" or plotdic["ymax"]=="auto":
                xmin, xmax, _, _, ymin, ymax = UTILS.get_xmin_xmax_ymin_ymax_zmin_zmax(rl)
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
            else:
                ax.set_xlim(plotdic["xmin"], plotdic["xmax"])
                ax.set_ylim(plotdic["ymin"], plotdic["ymax"])
            #
            if "text" in plotdic.keys():
                plotdic["text"]["transform"] = ax.transAxes
                ax.text(**plotdic["text"])
            #
            ax.set_title(r'$t-t_{merg} = $' + r'${:.1f}$ ms'.format((time_ - tmerg) * 1e3))
            #
            ax.tick_params(axis='both', which='both', labelleft=True,
                           labelright=False, tick1On=True, tick2On=True,
                           labelsize=plotdic["fontsize"],
                           direction='in',
                           bottom=True, top=True, left=True, right=True)
            ax.minorticks_on()

            #cbaxes = fig.add_axes([0.1, 0.1, 0.03, 0.8])
            #
            clb = fig.colorbar(im_right, ax=ax)
            clb.ax.set_title(plotdic["clabel_right"], fontsize=plotdic["fontsize"])
            clb.ax.tick_params(labelsize=plotdic["fontsize"])
            #
            from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
            ax2_divider = make_axes_locatable(ax)
            cax2 = ax2_divider.append_axes("left", size="5%", pad="30%")
            clb = fig.colorbar(im_left, cax = cax2)# anchor=(0.0, -0.5)) # anchor=(2.0, 0.5)
            clb.ax.set_title(plotdic["clabel_left"], fontsize=plotdic["fontsize"])
            clb.ax.tick_params(labelsize=plotdic["fontsize"])
            clb.ax.yaxis.set_ticks_position('left')
            clb.ax.minorticks_off()
            #
            print("plotted: \n")
            if plotdic["figname"] == "it":
                figpath = plot_dir + str(it) + ".png"
            elif plotdic["figname"] == "time":
                figpath = plotdic + str(int(time_)) + ".png"
            else:
                figpath = plot_dir + plotdic["figname"]
            plt.tight_layout()
            print(figpath)
            plt.savefig(figpath, dpi=128)
            if plotdic["savepdf"]: plt.savefig(figpath.replace(".png", ".pdf"))
            plt.close()
        # except NameError:
        #     Printcolor.red("NameError. Probably no neutrino data")
        # except:
        #     raise ValueError("Something is wrong.")

"""  ---------- ---------- ------- """

def plot_total_mass_within_thera_60_and_total_wind(tasks, plotdic):

    fig = plt.figure(figsize=plotdic["figsize"])
    ax = fig.add_subplot(111)
    #
    # labels

    for task in tasks:
        if task["type"] == "all" or task["type"] == plotdic["type"]:
            o_data = ADD_METHODS_ALL_PAR(task["sim"])
            its, times, datas = o_data.get_3d_data("hist_{}.dat".format(task["v_n"]))
            tmerg = o_data.get_par("tmerg")
            times = (times - tmerg)  # ms
            tot_mass = []
            for it, t, arr in zip(its, times, datas):
                # print(len(arr))
                if np.array(arr).ndim>1:
                    thetas = 90. - (180 * arr[0, :] / np.pi)
                    print("theta,min:{} theta.max:{}".format(thetas.min(), thetas.max()))
                    masses = arr[1, :]
                    tot_mass.append(np.sum(masses[thetas >= task["value"]]))
                else:
                    print("sim:{} it:{} -- NO data".format(task["sim"], it))
                    tot_mass.append(np.nan)
            ax.plot(times * 1e3, np.array(tot_mass) * 1e3, **task["plot"])

            #

            times, masses = o_data.get_time_data_arrs("Mej", det=0, mask="theta60_geoend")
            task["plot"]["ls"] = ":"
            ax.plot(times * 1e3, np.array(masses) * 1e3, **task["plot"])


    # tmp = {"color": "gray", "marker": "x", "markersize": 5, "linestyle": ':', "linewidth": 0.5,
    #        "label": "Linear extrapolation"}
    # ax.plot([-1, -1], [-2., -2], **tmp)

    ax.set_yscale(plotdic["yscale"])
    ax.set_xscale(plotdic["xscale"])

    ax.set_xlabel(plotdic["xlabel"])  # , fontsize=11)
    ax.set_ylabel(plotdic["ylabel"])  # , fontsize=11)

    ax.set_xlim(plotdic["xmin"], plotdic["xmax"])
    ax.set_ylim(plotdic["ymin"], plotdic["ymax"])
    #
    ax.tick_params(axis='both', which='both', labelleft=True,
                   labelright=False, tick1On=True, tick2On=True,
                   labelsize=12,
                   direction='in',
                   bottom=True, top=True, left=True, right=True)
    ax.minorticks_on()
    #
    ax.set_title(plotdic["title"])

    # LEGENDS
    ax.legend(**plotdic["legend"])

    # han, lab = ax.get_legend_handles_labels()
    # ax.add_artist(ax.legend(han[:-1], lab[:-1], **plotdic["legend"]))  # default
    # tmp = copy.deepcopy(plotdic["legend"])
    # tmp["loc"] = "upper left"
    # tmp["bbox_to_anchor"] = (0., 1.)
    # ax.add_artist(ax.legend([han[-1]], [lab[-1]], **tmp))  # for extapolation

    plt.tight_layout()
    #

    print("plotted: \n")
    print(plotdic["figname"])
    plt.savefig(plotdic["figname"], dpi=128)
    plt.close()

''' ------------------------- TASKS ----------------------- '''

def task_plot_total_ejecta_flux():

    v_n = "Mej"
    task = [
        {"sim": "BLh_M13641364_M0_LK_SR", "color": "black", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t2": -1, "ext":{}},
        # {"sim": "BLh_M11841581_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"NS", "v_n": v_n, "t": -1, "ext":None},
        {"sim": "BLh_M11461635_M0_LK_SR", "color": "black", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t2": -1, "ext":{}},
        {"sim": "BLh_M10651772_M0_LK_SR", "color": "black", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t2": -1, "ext":{}},
        # {"sim": "BLh_M10201856_M0_SR", "color": "black", "ls": "-", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
        # {"sim": "BLh_M10201856_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
        #
        {"sim": "DD2_M13641364_M0_SR", "color": "blue", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t2": -1, "ext":{}},
        # {"sim": "DD2_M13641364_M0_SR_R04", "color": "blue", "ls": "--", "lw": 0.7, "alpha": 1., "t1":60, "v_n": v_n, "t": -1, "ext":None},
        {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "blue", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long","v_n": v_n, "t": 60/1.e3, "ext":{}},
        {"sim": "DD2_M14971245_M0_SR", "color": "blue", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t2": -1, "ext":{}},
        {"sim": "DD2_M15091235_M0_LK_SR", "color": "blue", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t2": -1, "ext":{}},
        #
        {"sim": "LS220_M13641364_M0_SR", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1, "ext":{}},
        {"sim": "LS220_M13641364_M0_LK_SR_restart", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1, "ext":{}},
        {"sim": "LS220_M14691268_M0_LK_SR", "color": "red", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t2": 80/1.e3, "ext":{}},
        {"sim": "LS220_M14691268_M0_SR", "color": "red", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t2": 80 / 1.e3, "ext":{}}, # long

        # {"sim": "LS220_M11461635_M0_LK_SR", "color": "red", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1, "ext":None},
        # {"sim": "LS220_M10651772_M0_LK_SR", "color": "red", "ls": ":", "lw": 0.7, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1, "ext":None},
        #
        # {"sim": "SFHo_M13641364_M0_SR", "color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1, "ext":None},
        {"sim": "SFHo_M13641364_M0_LK_SR_2019pizza", "color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1, "ext":{}},
        # {"sim": "SFHo_M13641364_M0_LK_SR", "color": "green", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1},

        # {"sim": "SFHo_M14521283_M0_SR", "color": "green", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1},
        {"sim": "SFHo_M11461635_M0_LK_SR", "color": "green", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t2": -1, "ext":{}},
        #
        {"sim": "SLy4_M13641364_M0_SR", "color": "magenta", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1, "ext":{}},
        # {"sim": "SLy4_M13641364_M0_LK_SR_AHfix", "color": "magenta", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1},
        # {"sim": "SLy4_M14521283_M0_SR", "color": "magenta", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1},
        # {"sim": "SLy4_M10201856_M0_LK_SR", "color": "orange", "ls": "-.", "lw": 0.7, "alpha": 1., "v_n": v_n, "t": -1},
        {"sim": "SLy4_M11461635_M0_LK_SR", "color": "magenta", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t2": -1, "ext":{}}
    ]

    for t in task: t["label"] = r"\texttt{" + t["sim"].replace('_', '\_') + "}"
    for t in task:
        if t["sim"] == "LS220_M13641364_M0_LK_SR_restart":
            t["label"] = t["label"].replace("\_restart", "")
    for t in task:
        if t["sim"] == "SFHo_M13641364_M0_LK_SR_2019pizza":
            t["label"] = t["label"].replace("\_2019pizza", "")
    for t in task: t["det"] = 0
    for t in task: t["v_n"] = "Mej"
    for t in task: t["mask"] = "bern_geoend"

    plot_dic = {
        "type": "long",
        "figsize": (6., 2.5),
        "xmin": 0., "xmax": 110.,
        "ymin": 0, "ymax": 2.5,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "linear",
        "xlabel": r"$t-t_{\rm merg}$ [ms]",
        "ylabel": r"$M_{\rm ej}$ $[10^{-2}M_{\odot}]$",
        "legend": {"fancybox": False, "loc": 'center left',
                   "bbox_to_anchor":(1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon":False},
        "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "total_postdyn_ejecta_flux_long.png"
    }


    plot_total_ejecta_flux(task, plot_dic)

    # ---- short
    short_plot_dic = copy.deepcopy(plot_dic)
    short_plot_dic["type"] = "short"
    short_plot_dic["ymin"], short_plot_dic["ymax"] = 0, 1.
    short_plot_dic["xmin"], short_plot_dic["xmax"] = 0, 40
    short_plot_dic["title"] = "Short-lived remnants"
    short_plot_dic["figname"] = __outplotdir__ + "total_postdyn_ejecta_flux_short.png"

    # plot_total_ejecta_flux(task, short_plot_dic)

    # ---- long ---- wind Theta
    for dic in task: dic["mask"] = "theta60_geoend"
    plot_dic["ymin"] = 3e-5 #0
    plot_dic["ymax"] = 2e-3 #0.15
    plot_dic["yscale"] = "log"
    plot_dic["title"] = r"$\nu$-wind assuming $\theta > 60\deg$"
    plot_dic["figname"] = __outplotdir__ + "total_neutrino_wind_flux_theta_criteroin.png"
    # plot_total_ejecta_flux(task, plot_dic)

    # ---- long --- wind Theta
    for dic in task: dic["mask"] = "Y_e04_geoend"
    plot_dic["ymin"] = 1e-5 #0
    plot_dic["ymax"] = 1e-3 #0.04
    plot_dic["yscale"] = "log"
    plot_dic["title"] = r"$\nu$-wind assuming $Y_e > 0.4$"
    plot_dic["figname"] = __outplotdir__ + "total_neutrino_wind_flux_ye_criteroin.png"
    # plot_total_ejecta_flux(task, plot_dic)

def task_plot_total_ejecta_flux_extapolate():

    v_n = "Mej"
    task = [
        {"sim": "BLh_M13641364_M0_LK_SR", "color": "black", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t2": -1,
         "ext":{"type":"int1d","order":5, "t1":20,"t2":None,"show":True, "xarr":[100, 150, 200, 250, 300, 350, 400, 450, 500],
                "plot":{"color":"black", "marker":"x", "markersize":5, "linestyle": ':', "linewidth":0.5} }},

        {"sim": "BLh_M11461635_M0_LK_SR", "color": "gray", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t2": -1,
         "ext":{"type":"int1d","order":5, "t1":20,"t2":50, "show":True, "xarr":[100, 150, 200, 250, 300, 350, 400, 450, 500],
                "plot":{"color":"gray", "marker":"x", "markersize":5, "linestyle": ':', "linewidth":0.5} }},

        {"sim": "DD2_M13641364_M0_SR_R04", "color": "cyan", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t2": -1,
         "ext": {"type": "int1d", "order": 5, "t1": 20, "t2": None, "show": True, "xarr": [100, 150, 200, 250, 300, 350, 400, 450, 500],
                 "plot": {"color": "cyan", "marker": "x", "markersize": 5, "linestyle": ':', "linewidth":0.5}  }},

        {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "royalblue", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long","v_n": v_n, "t": 60/1.e3,
         "ext": {"type": "int1d", "order": 5, "t1": 20, "t2": None, "show": True, "xarr": [100, 150, 200, 250, 300, 350, 400, 450, 500],
                 "plot": {"color": "royalblue", "marker": "x", "markersize": 5, "linestyle": ':', "linewidth":0.5}  }},

        {"sim": "DD2_M14971245_M0_SR", "color": "slateblue", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t2": -1,
         "ext": {"type": "int1d", "order": 5, "t1": 20, "t2": None, "show": True, "xarr": [100, 150, 200, 250, 300, 350, 400, 450, 500],
                 "plot": {"color": "cyan", "marker": "x", "markersize": 5, "linestyle": ':', "linewidth": 0.5} }},

        {"sim": "DD2_M15091235_M0_LK_SR", "color": "blueviolet", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t2": -1,
         "ext": {"type": "int1d", "order": 5, "t1": 20, "t2": 90, "show": True, "xarr": [100, 150, 200, 250, 300, 350, 400, 450, 500],
                 "plot": {"color": "blueviolet", "marker": "x", "markersize": 5, "linestyle": ':', "linewidth": 0.5}  }},

        {"sim": "LS220_M14691268_M0_LK_SR", "color": "red", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t2": 80 / 1.e3,
         "ext": {"type": "int1d", "order": 5, "t1": 20, "t2": None, "show": True, "xarr": [100, 150, 200, 250, 300, 350, 400, 450, 500],
                 "plot": {"color": "red", "marker": "x", "markersize": 5, "linestyle": ':', "linewidth": 0.5}}},

        {"sim": "SFHo_M11461635_M0_LK_SR", "color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t2": -1,
         "ext": {"type": "int1d", "order": 5, "t1": 20, "t2": None, "show": True, "xarr": [100, 150, 200, 250, 300, 350, 400, 450, 500],
                 "plot": {"color": "green", "marker": "x", "markersize": 5, "linestyle": ':', "linewidth": 0.5} }},

        {"sim": "SLy4_M11461635_M0_LK_SR", "color": "magenta", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t2": -1,
         "ext": {"type": "int1d", "order": 5, "t1": 20, "t2": None, "show": True, "xarr": [100, 150, 200, 250, 300, 350, 400, 450, 500],
                 "plot": {"color": "magenta", "marker": "x", "markersize": 5, "linestyle": ':', "linewidth": 0.5}}},

    ]

    for t in task: t["label"] = r"\texttt{" + t["sim"].replace('_', '\_') + "}"
    for t in task: t["det"] = 0
    for t in task: t["v_n"] = "Mej"
    for t in task: t["mask"] = "bern_geoend"

    plot_dic = {
        "type": "long",
        "figsize": (6., 2.5),
        "xmin": 0., "xmax": 500.,
        "ymin": 0, "ymax": 10.0,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "linear",
        "xlabel": r"$t-t_{\rm merg}$ [ms]",
        "ylabel": r"$M_{\rm ej}$ $[10^{-2}M_{\odot}]$",
        "legend": {"fancybox": False, "loc": 'center left',
                   "bbox_to_anchor":(1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon":False},
        "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "total_postdyn_ejecta_flux_extrapolation_long.png"
    }


    plot_total_ejecta_flux(task, plot_dic)

    # ---- short
    short_plot_dic = copy.deepcopy(plot_dic)
    short_plot_dic["type"] = "short"
    short_plot_dic["ymin"], short_plot_dic["ymax"] = 0, 1.
    short_plot_dic["xmin"], short_plot_dic["xmax"] = 0, 40
    short_plot_dic["title"] = "Short-lived remnants"
    short_plot_dic["figname"] = __outplotdir__ + "total_postdyn_ejecta_flux_short.png"

    # plot_total_ejecta_flux(task, short_plot_dic)

    # ---- long ---- wind Theta
    for dic in task: dic["mask"] = "theta60_geoend"
    plot_dic["ymin"] = 3e-5 #0
    plot_dic["ymax"] = 2e-3 #0.15
    plot_dic["yscale"] = "log"
    plot_dic["title"] = r"$\nu$-wind assuming $\theta > 60\deg$"
    plot_dic["figname"] = __outplotdir__ + "total_neutrino_wind_flux_theta_criteroin.png"
    # plot_total_ejecta_flux(task, plot_dic)

    # ---- long --- wind Theta
    for dic in task: dic["mask"] = "Y_e04_geoend"
    plot_dic["ymin"] = 1e-5 #0
    plot_dic["ymax"] = 1e-3 #0.04
    plot_dic["yscale"] = "log"
    plot_dic["title"] = r"$\nu$-wind assuming $Y_e > 0.4$"
    plot_dic["figname"] = __outplotdir__ + "total_neutrino_wind_flux_ye_criteroin.png"
    # plot_total_ejecta_flux(task, plot_dic)

def task_plot_total_ejecta_hist():
    v_n = "None"
    task = [
        {"sim": "BLh_M13641364_M0_LK_SR", "color": "black", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "t2": -1},
        # {"sim": "BLh_M11841581_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"NS", "v_n": v_n, "t": -1},
        {"sim": "BLh_M11461635_M0_LK_SR", "color": "black", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "t2": -1},
        {"sim": "BLh_M10651772_M0_LK_SR", "color": "black", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "t2": -1},
        # {"sim": "BLh_M10201856_M0_SR", "color": "black", "ls": "-", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
        # {"sim": "BLh_M10201856_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
        #
        {"sim": "DD2_M13641364_M0_SR", "color": "blue", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n,
         "t2": -1},
        # {"sim": "DD2_M13641364_M0_SR_R04", "color": "blue", "ls": "--", "lw": 0.7, "alpha": 1., "t1":60, "v_n": v_n, "t": -1},
        {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "blue", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "t": 60 / 1.e3},
        {"sim": "DD2_M14971245_M0_SR", "color": "blue", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n,
         "t2": -1},
        {"sim": "DD2_M15091235_M0_LK_SR", "color": "blue", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "t2": -1},
        #
        {"sim": "LS220_M13641364_M0_SR", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "type": "short", "v_n": v_n,
         "t2": -1},
        {"sim": "LS220_M13641364_M0_LK_SR_restart", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "type": "short",
         "v_n": v_n, "t2": -1},
        {"sim": "LS220_M14691268_M0_LK_SR", "color": "red", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "t2": 80 / 1.e3},
        {"sim": "LS220_M14691268_M0_SR", "color": "red", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n,
         "t2": 80 / 1.e3},  # long

        # {"sim": "LS220_M11461635_M0_LK_SR", "color": "red", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1},
        # {"sim": "LS220_M10651772_M0_LK_SR", "color": "red", "ls": ":", "lw": 0.7, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1},
        #
        # {"sim": "SFHo_M13641364_M0_SR", "color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1},
        {"sim": "SFHo_M13641364_M0_LK_SR_2019pizza", "color": "green", "ls": "-", "lw": 0.8, "alpha": 1.,
         "type": "short", "v_n": v_n, "t2": -1},
        # {"sim": "SFHo_M13641364_M0_LK_SR", "color": "green", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1},

        # {"sim": "SFHo_M14521283_M0_SR", "color": "green", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1},
        {"sim": "SFHo_M11461635_M0_LK_SR", "color": "green", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "t2": -1},
        #
        {"sim": "SLy4_M13641364_M0_SR", "color": "magenta", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short",
         "v_n": v_n, "t2": -1},
        # {"sim": "SLy4_M13641364_M0_LK_SR_AHfix", "color": "magenta", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1},
        # {"sim": "SLy4_M14521283_M0_SR", "color": "magenta", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1},
        # {"sim": "SLy4_M10201856_M0_LK_SR", "color": "orange", "ls": "-.", "lw": 0.7, "alpha": 1., "v_n": v_n, "t": -1},
        {"sim": "SLy4_M11461635_M0_LK_SR", "color": "magenta", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long",
         "v_n": v_n, "t2": -1}
    ]

    for t in task: t["det"] = 0
    for t in task: t["v_n"] = "theta"
    for t in task: t["normalize"] = True
    for t in task: t["mask"] = "bern_geoend"
    for t in task: t["label"] = r"\texttt{" + t["sim"].replace('_', '\_') + "}"
    for t in task:
        if t["sim"] == "LS220_M13641364_M0_LK_SR_restart":
            t["label"] = t["label"].replace("\_restart", "")
    for t in task:
        if t["sim"] == "SFHo_M13641364_M0_LK_SR_2019pizza":
            t["label"] = t["label"].replace("\_2019pizza", "")
    # ----------------------- theta
    plot_dic = {
        "type": "long",
        "figsize": (6., 2.5),
        "xmin": 0., "xmax": 90.,
        "ymin": 1e-4, "ymax": 1e-1,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "log",
        "xlabel": r"Angle from binary plane",
        "ylabel": r"$M_{\rm ej}/M$",
        "legend": {"fancybox": False, "loc": 'center left',
                   "bbox_to_anchor":(1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon":False},
        "title": r"Postdynamical ejecta properties",
        "figname": __outplotdir__ + "total_postdyn_ejecta_hist_theta.png"
    }
    plot_total_ejecta_hist(task, plot_dic)

    # ---------------- velinf
    plot_dic["xmax"] = 0.5
    plot_dic["ymax"] = 5e-1
    plot_dic["xlabel"] = r"$\upsilon_{\infty}$"
    plot_dic["figname"] = __outplotdir__ + "total_postdyn_ejecta_hist_vinf.png"
    for dic in task: dic["v_n"] = "vel_inf"
    plot_total_ejecta_hist(task, plot_dic)

    # ---------------- Ye
    plot_dic["xmax"] = 0.5
    plot_dic["ymax"] = 5e-1
    plot_dic["xlabel"] = r"$Y_e$"
    plot_dic["figname"] = __outplotdir__ + "total_postdyn_ejecta_hist_ye.png"
    for dic in task: dic["v_n"] = "Y_e"
    plot_total_ejecta_hist(task, plot_dic)

def task_plot_total_ejecta_corr():
    task = {
        "sim": "BLh_M13641364_M0_LK_SR",
        "v_n_x": "vel_inf",
        "v_n_y": "Y_e",
        "normalize": True,
        "mask": "theta60_geoend",
        "det": 0
    }

    plotdic = {"vmin": 1e-5, "vmax": 1e-1,
               "xmin": 0., "xmax": 0.2,
               "ymin": 0.2, "ymax": 0.5,
               "cmap": "jet",
               "set_under": "black",
               "set_over": "red",
               "yscale": "linear",
               "xscale": "linear",
               "xlabel": r"$\upsilon_{\infty}$ [c]",
               "ylabel": r"$Y_e$",
               "title": r"\texttt{"+task["sim"].replace("_","\_")+"}",
               "clabel": r"$M_{\rm ej}/M$",
               "text": {"x": 0.3, "y": 0.95, "s": r"$\theta>60\deg$", "ha": "center", "va": "top", "fontsize": 11,
                        "color": "white",
                        "transform": None},
               "figname": "total_ejecta_corr_{}_{}_{}.png".format(task["v_n_x"], task["v_n_y"], task["sim"])
               }

    plot_total_ejecta_corr(task, plotdic)

def tast_plot_total_ejecta_timecorr():

    # default

    simlist = ["BLh_M13641364_M0_LK_SR", "DD2_M13641364_M0_LK_SR_R04",
        "LS220_M14691268_M0_LK_SR", "BLh_M11461635_M0_LK_SR",
        "DD2_M15091235_M0_LK_SR", "SFHo_M11461635_M0_LK_SR",
        "SLy4_M11461635_M0_LK_SR", "LS220_M11461635_M0_LK_SR",
        "DD2_M13641364_M0_SR", "DD2_M14971245_M0_SR"]

    goodsims = ["DD2_M15091235_M0_LK_SR", "DD2_M13641364_M0_LK_SR_R04", "BLh_M13641364_M0_LK_SR"]

    task = {
        "sim": "BLh_M13641364_M0_LK_SR",
        "v_n": "theta",
        "det": 0,
        "mask": "bern_geoend",
        "normalize": True,
    }
    plotdic = {"vmin": 1e-5, "vmax": 1.e-2,
               "xmin": 0, "xmax": 100,
               "ymin": 0, "ymax": 85,
               "cmap": "jet",
               "set_under": "black",
               "set_over": "red",
               "yscale": "linear",
               "xscale": "linear",
               "xlabel": r"$t-t_{\rm merg}$ [ms]",
               "ylabel": r"Angle from binary plane",
               "clabel": r"M/sum(M)",
               "title":r"$\texttt{"+task["sim"].replace('_', '\_') + "}$",
               # "text": {"x": 0.35, "y": 0.95, "s": r"$\texttt{"+task["sim"].replace('_', '\_') + "}$",
               #          "ha": "center", "va": "top", "fontsize": 11, "color": "white", "transform": None},
               "figname": "total_ejecta_timecorr_{}_{}.png".format(task["v_n"], task["sim"])
               }
    for sim in goodsims:
        i_plotdic = copy.deepcopy(plotdic)
        #
        task["sim"] = sim
        i_plotdic["title"] = r"$\texttt{"+task["sim"].replace('_', '\_') + "}$"
        i_plotdic["figname"] = "total_ejecta_timecorr_{}_{}.png".format(task["v_n"], task["sim"])
        plot_total_ejecta_timecorr(task, i_plotdic)

        # -- vel inf
        task["v_n"] = "vel_inf"
        i_plotdic["ylabel"] = "$\upsilon_{\infty}$"
        i_plotdic["ymax"] = .4
        i_plotdic["figname"] = "total_ejecta_timecorr_{}_{}.png".format(task["v_n"], task["sim"])
        plot_total_ejecta_timecorr(task, i_plotdic)

        # -- ye
        task["v_n"] = "Y_e"
        i_plotdic["ylabel"] = "$Y_e$"
        i_plotdic["ymin"] = .1
        i_plotdic["ymax"] = .5
        i_plotdic["figname"] = "total_ejecta_timecorr_{}_{}.png".format("ye", task["sim"])
        plot_total_ejecta_timecorr(task, i_plotdic)

def task_plot_corr_qeff_u_0():

    # default
    task = {
        "sim": "BLh_M13641364_M0_LK_SR",
        "v_n_x": "Q_eff_nua_over_density",
        "v_n_y": "hu_0",
        "plane": "xz",
        "t1": 60, "t2": 80,
        "normalize": True,
        "mask": "rl",
    }
    plotdic = {"vmin": 1e-9, "vmax": 1.e-6,
               "xmin": 1e-9, "xmax": 1e-4,
               "ymin": 0.98, "ymax": 1.04,
               "cmap": "jet",
               "set_under": "black",
               "set_over": "red",
               "yscale": "linear",
               "xscale": "log",
               "xlabel": r"$Q_{eff}(\nu_a) / D$ [GEO]",
               "ylabel": r"-$h u_t$",
               "clabel": r"M/sum(M)",
               "text": {"x":0.3, "y":0.95, "s":"Total", "ha":"center", "va":"top",  "fontsize":11, "color":"white",
                        "transform":None},
               "figname": "corr_{}_{}_{}_plane_{}_mask_{}.png".format(task["sim"], task["v_n_x"],
                                                              task["v_n_y"], task["plane"], task["mask"])
               }
    # plot_corr_qeff_u_0(task, plotdic)

    # mask
    task["mask"] = "rl_Ye04"
    plotdic["vmin"], plotdic["vmax"] = 1e-7, 1e-3
    plotdic["figname"] = "corr_{}_{}_{}_plane_{}_mask_{}.png".format(task["sim"], task["v_n_x"],
                                                              task["v_n_y"], task["plane"], task["mask"])
    plotdic["text"]["s"] = r"$Y_e > 0.4$"
    plotdic["text"]["y"] = 0.1
    # plot_corr_qeff_u_0(task, plotdic)

    # mask
    task["mask"] = "rl_theta60"
    plotdic["vmin"], plotdic["vmax"] = 1e-6, 1e-3
    plotdic["text"]["s"] = r"$\theta > 60\deg$"
    plotdic["figname"] = "corr_{}_{}_{}_plane_{}_mask_{}.png".format(task["sim"], task["v_n_x"],
                                                              task["v_n_y"], task["plane"], task["mask"])
    plot_corr_qeff_u_0(task, plotdic)

    #

def task_plot_corr_qeff_theta():
    # default
    task = {
        "sim": "BLh_M13641364_M0_LK_SR",
        "v_n_x": "Q_eff_nua_over_density",
        "v_n_y": "theta",
        "plane": "xz",
        "t1": 60, "t2": 80,
        "normalize": True,
        "mask": "rl",
    }
    plotdic = {"vmin": 1e-7, "vmax": 1.e-4,
               "xmin": 1e-9, "xmax": 1e-4,
               "ymin": 0., "ymax": 90.,
               "cmap": "jet",
               "set_under": "black",
               "set_over": "red",
               "yscale": "linear",
               "xscale": "log",
               "xlabel": r"$Q_{eff}(\nu_a) / D$ [GEO]",
               "ylabel": r"Angle from binary plane",
               "clabel": r"M/sum(M)",
               "text": {"x": 0.3, "y": 0.95, "s": "Total", "ha": "center", "va": "top", "fontsize": 11,
                        "color": "white",
                        "transform": None},
               "figname": "corr_{}_{}_{}_plane_{}_mask_{}.png".format(task["sim"], task["v_n_x"],
                                                                      task["v_n_y"], task["plane"], task["mask"])
               }
    plot_corr_qeff_u_0(task, plotdic)

    # mask
    task["mask"] = "rl_hu0"
    plotdic["text"]["s"] = r"$-hu_0 > 1$"
    plotdic["vmin"], plotdic["vmax"] = 1e-6, 1e-3
    plotdic["figname"] = "corr_{}_{}_{}_plane_{}_mask_{}.png".format(task["sim"], task["v_n_x"],
                                                                     task["v_n_y"], task["plane"], task["mask"])
    plot_corr_qeff_u_0(task, plotdic)

def task_plot_corr_qeff_ye():
    # default
    task = {
        "sim": "BLh_M13641364_M0_LK_SR",
        "v_n_x": "Q_eff_nua_over_density",
        "v_n_y": "Ye",
        "plane": "xz",
        "t1": 60, "t2": 80,
        "normalize": True,
        "mask": "rl",
    }
    plotdic = {"vmin": 1e-9, "vmax": 1.e-6,
               "xmin": 1e-9, "xmax": 1e-4,
               "ymin": 0., "ymax": 0.5,
               "cmap": "jet",
               "set_under": "black",
               "set_over": "red",
               "yscale": "linear",
               "xscale": "log",
               "xlabel": r"$Q_{eff}(\nu_a) / D$ [GEO]",
               "ylabel": r"$Y_e$",
               "clabel": r"M/sum(M)",
               "text": {"x": 0.3, "y": 0.95, "s": "Total", "ha": "center", "va": "top", "fontsize": 11,
                        "color": "white",
                        "transform": None},
               "figname": "corr_{}_{}_{}_plane_{}_mask_{}.png".format(task["sim"], task["v_n_x"],
                                                                      task["v_n_y"], task["plane"], task["mask"])
               }
    plot_corr_qeff_u_0(task, plotdic)

    # mask
    task["mask"] = "rl_hu0"
    plotdic["text"]["s"] = r"$-hu_0 > 1$"
    plotdic["vmin"], plotdic["vmax"] = 1e-6, 1e-3
    plotdic["figname"] = "corr_{}_{}_{}_plane_{}_mask_{}.png".format(task["sim"], task["v_n_x"],
                                                                     task["v_n_y"], task["plane"], task["mask"])
    plot_corr_qeff_u_0(task, plotdic)

def tasl_plot_slice_2halfs():

    task = {
        "sim": "BLh_M13641364_M0_LK_SR",
        "v_n_x": "x",
        "v_n_y": "z",
        "plane": "xz",
        "v_n_left": "Q_eff_nua_over_density",
        "v_n_right": "hu_0",
        "rl": 1,
        "times":">20"
        # "iterations": "all" # [1949696]
    }

    plot_dic = {
        "xlabel": "$X$ $[M_{\odot}]$",
        "ylabel": "$Y$ $[M_{\odot}]$",

        "vmin_left": 1e-12, "vmax_left": 1e-6,
        "clabel_left": r"$Q_{eff}(\nu_a) / D$",
        "cmap_left": "jet_r",
        "norm_left": "log",
        # "set_under_left": "black", "set_over_left": "blue",

        "vmin_right": 0.98, "vmax_right": 1.02,
        "clabel_right": "$-hu_0$",
        "cmap_right": "RdBu",
        "norm_right": "linear",
        # "set_under_right": "black", "set_over_right": "blue",

        "xmin": "auto", "xmax": "auto",
        "ymin": "auto", "ymax": "auto",
        "xscale":"linear", "yscale":"linear",
        "plot_dir": __outplotdir__ + task["sim"] + '/{}__{}/'.format(task["v_n_left"], task["v_n_right"]),
        "figname": "it"

    }
    # plot_slice_2halfs(task, plot_dic)

    # -- Ye
    task["v_n_left"]="Ye"
    plot_dic["clabel_left"]="$Y_e$"
    plot_dic["vmin_left"], plot_dic["vmax_left"]=0.2, 0.45
    plot_dic["norm_left"] = "linear"
    plot_dic["plot_dir"] = __outplotdir__ + task["sim"] + '/{}__{}/'.format(task["v_n_left"], task["v_n_right"])
    #
    # plot_slice_2halfs(task, plot_dic)
    #

    # close to the remnant
    task["rl"] = 3
    task["v_n_left"] = "Q_eff_nua_over_density"
    plot_dic["clabel_left"] = r"$Q_{eff}(\nu_a) / D$"
    plot_dic["norm_left"] = "log"
    plot_dic["vmin_left"], plot_dic["vmax_left"] = 1e-9, 1e-4
    plot_dic["plot_dir"] = __outplotdir__ + task["sim"] + \
                           '/rl{}__{}__{}/'.format(task["rl"], task["v_n_left"], task["v_n_right"])
    plot_slice_2halfs(task, plot_dic)

def task_plot_total_mass_within_thera_60_and_total_wind():

    task = [
        {"sim": "BLh_M13641364_M0_LK_SR", "plot":{"color": "black", "ls": "-", "lw": 0.8, "alpha": 1.}, "type": "long"},
        # {"sim": "BLh_M11841581_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"NS"},
        {"sim": "BLh_M11461635_M0_LK_SR", "plot":{"color": "black", "ls": "-.", "lw": 0.8, "alpha": 1.}, "type": "long"},
        # {"sim": "BLh_M10651772_M0_LK_SR", "plot":{"color": "black", "ls": ":", "lw": 0.8, "alpha": 1.}, "type": "long"},
        # {"sim": "BLh_M10201856_M0_SR", "color": "black", "ls": "-", "lw": 0.7, "alpha": 1., "outcome":"PC"},
        # {"sim": "BLh_M10201856_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"PC"},
        #
        {"sim": "DD2_M13641364_M0_SR", "plot":{"color": "blue", "ls": "-", "lw": 0.8, "alpha": 1.}, "type": "long"},
        # {"sim": "DD2_M13641364_M0_SR_R04", "color": "blue", "ls": "--", "lw": 0.7, "alpha": 1., "t1":60},
        #{"sim": "DD2_M13641364_M0_LK_SR_R04", "plot":{"color": "blue", "ls": "--", "lw": 0.8, "alpha": 1.}, "type": "long",
         #"t1": 40},
        #{"sim": "DD2_M14971245_M0_SR", "plot":{"color": "blue", "ls": "-.", "lw": 0.8, "alpha": 1.}, "type": "long"},
        #{"sim": "DD2_M15091235_M0_LK_SR", "plot":{"color": "blue", "ls": ":", "lw": 0.8, "alpha": 1.}, "type": "long"},
        #
        # {"sim": "LS220_M13641364_M0_SR", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1.},
        #{"sim": "LS220_M13641364_M0_LK_SR_restart", "plot":{"color": "red", "ls": "-", "lw": 0.7, "alpha": 1.}, "type": "short"},
        #{"sim": "LS220_M14691268_M0_LK_SR", "plot":{"color": "red", "ls": "--", "lw": 0.8, "alpha": 1.}, "type": "long"},
        #{"sim": "LS220_M11461635_M0_LK_SR", "plot":{"color": "red", "ls": "-.", "lw": 0.8, "alpha": 1.}, "type": "short"},
        #{"sim": "LS220_M10651772_M0_LK_SR", "plot":{"color": "red", "ls": ":", "lw": 0.7, "alpha": 1.}, "type": "short"},
        #
        #{"sim": "SFHo_M13641364_M0_SR", "plot":{"color": "green", "ls": "-", "lw": 0.8, "alpha": 1.}, "type": "short"},
        #{"sim": "SFHo_M14521283_M0_SR", "plot":{"color": "green", "ls": "--", "lw": 0.8, "alpha": 1.}, "type": "short"},
        {"sim": "SFHo_M11461635_M0_LK_SR", "plot":{"color": "green", "ls": "-.", "lw": 0.8, "alpha": 1.}, "type": "long"},
        #
        #{"sim": "SLy4_M13641364_M0_SR", "plot":{"color": "magenta", "ls": "-", "lw": 0.8, "alpha": 1.}, "type": "short"},
        #{"sim": "SLy4_M14521283_M0_SR", "plot":{"color": "magenta", "ls": "--", "lw": 0.8, "alpha": 1.}, "type": "short"},
        # {"sim": "SLy4_M10201856_M0_LK_SR", "color": "orange", "ls": "-.", "lw": 0.7, "alpha": 1.},
        #{"sim": "SLy4_M11461635_M0_LK_SR", "plot":{"color": "magenta", "ls": ":", "lw": 0.8, "alpha": 1.}, "type": "long"}
    ]

    for t in task: t["plot"]["label"] = r"\texttt{" + t["sim"].replace('_', '\_') + "}"
    for t in task: t["v_n"] = "theta"
    for t in task: t["value"] = 70.
    for t in task:
        if t["sim"] == "LS220_M13641364_M0_LK_SR_restart":
            t["plot"]["label"] = t["plot"]["label"].replace("\_restart", "")

    # --- LONG ---
    plot_dic = {
        "figsize": (4.2, 3.6),
        "type": "long",
        "xmin": 0, "xmax": 1e2,
        "ymin": 0, "ymax": 2.0,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "linear",
        "xlabel": r"$t-t_{\rm merg}$ [ms]",
        "ylabel": r"$M_{\rm disk}|_{\theta>70}$ $[10^{-3}M_{\odot}]$",
        "legend": {"fancybox": False, "loc": 'upper right',
                   #"bbox_to_anchor": (1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "title": r"Baryon Loading of the $\theta>70\deg$",
        "figname": __outplotdir__ + "total_baryon_loading_polar_region.png"
    }
    plot_total_mass_within_thera_60_and_total_wind(task, plot_dic)

''' ---------------- RESOLUTION COMPARISON TASKS ------------ '''

def task_resolution_plot_total_ejecta_flux():

    v_n = "Mej"
    task = [
        {"sim": "BLh_M13641364_M0_LK_SR", "color": "blue", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t2": -1, "ext":{}},
        {"sim": "BLh_M13641364_M0_LK_LR", "color": "blue", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t2": -1, "ext":{}},
        {"sim": "BLh_M13641364_M0_LK_HR", "color": "blue", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t2": -1, "ext":{}},

        {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "red", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t2": -1, "ext":{}},
        {"sim": "DD2_M13641364_M0_LK_LR_R04", "color": "red", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long","v_n": v_n, "t": -1, "ext":{}},
        {"sim": "DD2_M13641364_M0_LK_HR_R04", "color": "red", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t2": -1, "ext":{}},
    ]

    for t in task: t["label"] = r"\texttt{" + t["sim"].replace('_', '\_') + "}"
    for t in task:
        if t["sim"] == "LS220_M13641364_M0_LK_SR_restart":
            t["label"] = t["label"].replace("\_restart", "")
    for t in task:
        if t["sim"] == "SFHo_M13641364_M0_LK_SR_2019pizza":
            t["label"] = t["label"].replace("\_2019pizza", "")
    for t in task: t["det"] = 0
    for t in task: t["v_n"] = "Mej"
    for t in task: t["mask"] = "bern_geoend"

    plot_dic = {
        "type": "long",
        "figsize": (6., 2.5),
        "xmin": 0., "xmax": 110.,
        "ymin": 0, "ymax": 2.5,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "linear",
        "xlabel": r"$t-t_{\rm merg}$ [ms]",
        "ylabel": r"$M_{\rm ej}$ $[10^{-2}M_{\odot}]$",
        "legend": {"fancybox": False, "loc": 'center left',
                   "bbox_to_anchor":(1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 10,
                   "framealpha": 0., "borderaxespad": 0., "frameon":False},
        "title": r"Long-lived remnants",
        "figname": __outplotdir__ + "resolution_postdyn_ejecta_flux_long.png"
    }

    plot_total_ejecta_flux(task, plot_dic)

    # ---- short
    short_plot_dic = copy.deepcopy(plot_dic)
    short_plot_dic["type"] = "short"
    short_plot_dic["ymin"], short_plot_dic["ymax"] = 0, 1.
    short_plot_dic["xmin"], short_plot_dic["xmax"] = 0, 40
    short_plot_dic["title"] = "Short-lived remnants"
    short_plot_dic["figname"] = __outplotdir__ + "resolution_postdyn_ejecta_flux_short.png"

    # plot_total_ejecta_flux(task, short_plot_dic)

    # ---- long ---- wind Theta
    for dic in task: dic["mask"] = "theta60_geoend"
    plot_dic["ymin"] = 3e-5 #0
    plot_dic["ymax"] = 2e-3 #0.15
    plot_dic["yscale"] = "log"
    plot_dic["title"] = r"$\nu$-wind assuming $\theta > 60\deg$"
    plot_dic["figname"] = __outplotdir__ + "resolution_neutrino_wind_flux_theta_criteroin.png"
    plot_total_ejecta_flux(task, plot_dic)

    # ---- long --- wind Theta
    for dic in task: dic["mask"] = "Y_e04_geoend"
    plot_dic["ymin"] = 1e-5 #0
    plot_dic["ymax"] = 1e-3 #0.04
    plot_dic["yscale"] = "log"
    plot_dic["title"] = r"$\nu$-wind assuming $Y_e > 0.4$"
    plot_dic["figname"] = __outplotdir__ + "resolution_neutrino_wind_flux_ye_criteroin.png"
    plot_total_ejecta_flux(task, plot_dic)

''' --- Tasks --- iteration 2 --- '''

def task_plot_total_ejecta_flux_2():

    v_n = "Mej"

    task = [
        {"sim": "BLh_M13641364_M0_LK_SR",   "type": "long", "v_n": v_n, "t2": -1, "ext": {}, "plot": {"color": "black", "ls": "-", "lw": 0.8, "alpha": 1., "label": r"BLh q=1.00 (SR)"}},
        {"sim": "BLh_M11461635_M0_LK_SR",   "type": "long", "v_n": v_n, "t2": -1, "ext": {}, "plot": {"color": "gray", "ls": "-", "lw": 0.8, "alpha": 1., "label":  r"BLh q=1.43 (SR)"}},
        {"sim": "BLh_M10651772_M0_LK_LR",   "type": "long", "v_n": v_n, "t2": -1, "ext": {}, "plot": {"color": "gray", "ls": "-", "lw": 0.8, "alpha": 1., "label":  r"BLh q=1.66 (LR)"}},
        {"sim": "DD2_M13641364_M0_SR",      "type": "long", "v_n": v_n, "t2": -1, "ext": {}, "plot": {"color": "blue", "ls": "-", "lw": 0.8, "alpha": 1., "label":  r"DD2* q=1.00 (SR)"}},
        {"sim": "DD2_M13641364_M0_LK_SR_R04","type": "long", "v_n": v_n, "t2": -1, "ext": {}, "plot": {"color": "cyan", "ls": "-", "lw": 0.8, "alpha": 1., "label": r"DD2 q=1.00 (SR)"}},
        {"sim": "DD2_M14971245_M0_SR",      "type": "long", "v_n": v_n, "t2": -1, "ext": {}, "plot": {"color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "label": r"DD2* q=1.19 (SR)"}},
        {"sim": "DD2_M15091235_M0_LK_SR",   "type": "long", "v_n": v_n, "t2": -1, "ext": {}, "plot": {"color": "orange", "ls": "-", "lw": 0.8, "alpha": 1., "label": r"DD2 q=1.22 (SR)"}},
        #{"sim": "SFHo_M11461635_M0_LK_SR", "type": "long", "v_n": v_n, "t2": -1, "ext": {}, "plot": {"color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "label": r"SFHo* q=1.43 (SR)"}},
        #{"sim": "SLy4_M11461635_M0_LK_SR", "type": "long", "v_n": v_n, "t2": -1, "ext": {}, "plot": {"color": "red", "ls": "-", "lw": 0.8, "alpha": 1., "label": r"SLy4* q=1.43 (SR)"}},
    ]

    for t in task:
        t["plot"]["color"] = md.sim_dic_color[t["sim"]]
        t["plot"]["ls"] = md.sim_dic_ls[t["sim"]]
        t["plot"]["lw"] = md.sim_dic_lw[t["sim"]]

    # task = [
    #     {"sim": "BLh_M13641364_M0_LK_SR", "color": "black", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t2": -1, "ext":{}},
    #     # {"sim": "BLh_M11841581_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"NS", "v_n": v_n, "t": -1, "ext":None},
    #     {"sim": "BLh_M11461635_M0_LK_SR", "color": "black", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t2": -1, "ext":{}},
    #     {"sim": "BLh_M10651772_M0_LK_SR", "color": "black", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t2": -1, "ext":{}},
    #     # {"sim": "BLh_M10201856_M0_SR", "color": "black", "ls": "-", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
    #     # {"sim": "BLh_M10201856_M0_LK_SR", "color": "black", "ls": "--", "lw": 0.7, "alpha": 1., "outcome":"PC", "t": -1},
    #     #
    #     {"sim": "DD2_M13641364_M0_SR", "color": "blue", "ls": "-", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t2": -1, "ext":{}},
    #     # {"sim": "DD2_M13641364_M0_SR_R04", "color": "blue", "ls": "--", "lw": 0.7, "alpha": 1., "t1":60, "v_n": v_n, "t": -1, "ext":None},
    #     {"sim": "DD2_M13641364_M0_LK_SR_R04", "color": "blue", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long","v_n": v_n, "t": 60/1.e3, "ext":{}},
    #     {"sim": "DD2_M14971245_M0_SR", "color": "blue", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t2": -1, "ext":{}},
    #     {"sim": "DD2_M15091235_M0_LK_SR", "color": "blue", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t2": -1, "ext":{}},
    #     #
    #     {"sim": "LS220_M13641364_M0_SR", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1, "ext":{}},
    #     {"sim": "LS220_M13641364_M0_LK_SR_restart", "color": "red", "ls": "-", "lw": 0.7, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1, "ext":{}},
    #     {"sim": "LS220_M14691268_M0_LK_SR", "color": "red", "ls": "--", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t2": 80/1.e3, "ext":{}},
    #     {"sim": "LS220_M14691268_M0_SR", "color": "red", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t2": 80 / 1.e3, "ext":{}}, # long
    #
    #     # {"sim": "LS220_M11461635_M0_LK_SR", "color": "red", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1, "ext":None},
    #     # {"sim": "LS220_M10651772_M0_LK_SR", "color": "red", "ls": ":", "lw": 0.7, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1, "ext":None},
    #     #
    #     # {"sim": "SFHo_M13641364_M0_SR", "color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1, "ext":None},
    #     {"sim": "SFHo_M13641364_M0_LK_SR_2019pizza", "color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1, "ext":{}},
    #     # {"sim": "SFHo_M13641364_M0_LK_SR", "color": "green", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1},
    #
    #     # {"sim": "SFHo_M14521283_M0_SR", "color": "green", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1},
    #     {"sim": "SFHo_M11461635_M0_LK_SR", "color": "green", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t2": -1, "ext":{}},
    #     #
    #     {"sim": "SLy4_M13641364_M0_SR", "color": "magenta", "ls": "-", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1, "ext":{}},
    #     # {"sim": "SLy4_M13641364_M0_LK_SR_AHfix", "color": "magenta", "ls": "-.", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1},
    #     # {"sim": "SLy4_M14521283_M0_SR", "color": "magenta", "ls": "--", "lw": 0.8, "alpha": 1., "type": "short", "v_n": v_n, "t2": -1},
    #     # {"sim": "SLy4_M10201856_M0_LK_SR", "color": "orange", "ls": "-.", "lw": 0.7, "alpha": 1., "v_n": v_n, "t": -1},
    #     {"sim": "SLy4_M11461635_M0_LK_SR", "color": "magenta", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long", "v_n": v_n, "t2": -1, "ext":{}}
    # ]

    # for t in task:
    #     t["label"] = r"\texttt{" + t["sim"].replace('_', '\_') + "}"
    # for t in task:
    #     if t["sim"] == "LS220_M13641364_M0_LK_SR_restart":
    #         t["label"] = t["label"].replace("\_restart", "")
    # for t in task:
    #     if t["sim"] == "SFHo_M13641364_M0_LK_SR_2019pizza":
    #         t["label"] = t["label"].replace("\_2019pizza", "")
    for t in task: t["det"] = 0
    for t in task: t["v_n"] = "Mej"
    for t in task: t["mask"] = "bern_geoend"

    plot_dic = {
        "type": "long",
        "figsize": (6., 5.5),
        "xmin": 0., "xmax": 110.,
        "ymin": 0, "ymax": 2.5,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "linear",
        "xlabel": r"$t-t_{\rm merg}$ [ms]",
        "ylabel": r"$M_{\rm ej}$ $[10^{-2}M_{\odot}]$",
        "legend": {"fancybox": False, "loc": 'upper left',
                   #"bbox_to_anchor":(1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon":False},
        "add_legend": False,
        "title": r"Wind cumulative mass",
        "figname": __outplotdir__ + "wind_mass_flux.png",
        "fontsize":14,
        "savepdf":True
    }


    plot_total_ejecta_flux(task, plot_dic)

    # ---- short
    # short_plot_dic = copy.deepcopy(plot_dic)
    # short_plot_dic["type"] = "short"
    # short_plot_dic["ymin"], short_plot_dic["ymax"] = 0, 1.
    # short_plot_dic["xmin"], short_plot_dic["xmax"] = 0, 40
    # short_plot_dic["title"] = "Short-lived remnants"
    # short_plot_dic["figname"] = __outplotdir__ + "total_postdyn_ejecta_flux_short.png"

    # plot_total_ejecta_flux(task, short_plot_dic)

    # ---- long ---- wind Theta
    for dic in task: dic["mask"] = "theta60_geoend"
    plot_dic["ymin"] = 3e-5 #0
    plot_dic["ymax"] = 2e-3 #0.15
    plot_dic["yscale"] = "log"
    plot_dic["title"] = r"Wind cumulative mass with $\theta > 60\deg$"
    plot_dic["figname"] = __outplotdir__ + "wind_mass_flux_theta60.png"
    plot_total_ejecta_flux(task, plot_dic)

    # ---- long --- wind Theta
    for dic in task: dic["mask"] = "Y_e04_geoend"
    plot_dic["ymin"] = 1e-5 #0
    plot_dic["ymax"] = 1e-3 #0.04
    plot_dic["yscale"] = "log"
    plot_dic["title"] = r"Wind cumulative mass with $Y_e>0.4$"
    plot_dic["figname"] = __outplotdir__ + "wind_mass_flux_ye04.png"
    plot_total_ejecta_flux(task, plot_dic)

def task_plot_total_ejecta_hist_2():
    v_n = "None"
    task = [
        {"sim": "BLh_M13641364_M0_LK_SR",   "type": "long", "v_n": v_n, "t2": -1, "ext": {}, "plot": {"color": "black", "ls": "-", "lw": 0.8, "alpha": 1., "label": r"BLh* q=1.00 (SR)", "drawstyle":"steps"}},
        {"sim": "BLh_M11461635_M0_LK_SR",   "type": "long", "v_n": v_n, "t2": -1, "ext": {}, "plot": {"color": "gray", "ls": "-", "lw": 0.8, "alpha": 1., "label": r"BLh* q=1.43 (SR)", "drawstyle":"steps"}},
        {"sim": "BLh_M10651772_M0_LK_SR", "type": "long", "v_n": v_n, "t2": -1, "ext": {}, "plot": {"color": "gray", "ls": "-", "lw": 0.8, "alpha": 1., "label": r"BLh* q=1.66 (SR)","drawstyle": "steps"}},
        {"sim": "DD2_M13641364_M0_SR",      "type": "long", "v_n": v_n, "t2": -1, "ext": {}, "plot": {"color": "blue", "ls": "-", "lw": 0.8, "alpha": 1., "label": r"DD2 q=1.00 (SR)", "drawstyle":"steps"}},
        {"sim": "DD2_M13641364_M0_LK_SR_R04","type": "long", "v_n": v_n, "t2": -1, "ext": {}, "plot": {"color": "cyan", "ls": "-", "lw": 0.8, "alpha": 1., "label": r"DD2* q=1.00 (SR)", "drawstyle":"steps"}},
        {"sim": "DD2_M14971245_M0_SR",      "type": "long", "v_n": v_n, "t2": -1, "ext": {}, "plot": {"color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "label": r"DD2 q=1.19 (SR)", "drawstyle":"steps"}},
        {"sim": "DD2_M15091235_M0_LK_SR",   "type": "long", "v_n": v_n, "t2": -1, "ext": {}, "plot": {"color": "orange", "ls": "-", "lw": 0.8, "alpha": 1., "label": r"DD2* q=1.22 (SR)", "drawstyle":"steps"}},
        # {"sim": "SFHo_M11461635_M0_LK_SR", "type": "long", "v_n": v_n, "t2": -1, "ext": {}, "plot": {"color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "label": r"SFHo* q=1.43 (SR)"}},
        # {"sim": "SLy4_M11461635_M0_LK_SR", "type": "long", "v_n": v_n, "t2": -1, "ext": {}, "plot": {"color": "red", "ls": "-", "lw": 0.8, "alpha": 1., "label": r"SLy4* q=1.43 (SR)"}},
    ]

    for t in task:
        t["plot"]["color"] = md.sim_dic_color[t["sim"]]
        t["plot"]["ls"] = md.sim_dic_ls[t["sim"]]
        t["plot"]["lw"] = md.sim_dic_lw[t["sim"]]

    for t in task: t["det"] = 0
    for t in task: t["v_n"] = "theta"
    for t in task: t["normalize"] = True
    for t in task: t["mask"] = "bern_geoend"

    # for t in task:
    #     if t["sim"] == "LS220_M13641364_M0_LK_SR_restart":
    #         t["label"] = t["label"].replace("\_restart", "")
    # for t in task:
    #     if t["sim"] == "SFHo_M13641364_M0_LK_SR_2019pizza":
    #         t["label"] = t["label"].replace("\_2019pizza", "")
    # ----------------------- theta
    plot_dic = {
        "type": "long",
        "figsize": (6., 5.5),
        "xmin": 0., "xmax": 90.,
        "ymin": 1e-4, "ymax": 1e-1,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "log",
        "xlabel": r"Angle from binary plane",
        "ylabel": r"$M_{\rm ej}/M$",
        "legend": {"fancybox": False, "loc": 'lower center',
                   #"bbox_to_anchor":(1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon":False},
        #"title": r"Postdynamical ejecta properties",
        "figname": __outplotdir__ + "wind_hist_theta.png",
        "fontsize": 14,
        "savepdf": True
    }
    plot_total_ejecta_hist(task, plot_dic)

    # ---------------- velinf
    plot_dic["xmax"] = 0.3
    plot_dic["ymax"] = 5e-1
    plot_dic["legend"] = {}
    plot_dic["xlabel"] = r"$\langle \upsilon_{\infty} \rangle$"
    plot_dic["figname"] = __outplotdir__ + "wind_hist_vinf.png"
    for dic in task: dic["v_n"] = "vel_inf"
    plot_total_ejecta_hist(task, plot_dic)

    # ---------------- Ye
    plot_dic["xmax"] = 0.5
    plot_dic["ymax"] = 5e-1
    plot_dic["legend"] = {}
    plot_dic["xlabel"] = r"$\langle Y_e \rangle$"
    plot_dic["figname"] = __outplotdir__ + "wind_hist_ye.png"
    for dic in task: dic["v_n"] = "Y_e"
    plot_total_ejecta_hist(task, plot_dic)

def custom_task_plot_total_ejecta_hist_2():
    v_n = "None"
    task = [
        {"sim": "BLh_M13641364_M0_LK_SR",   "type": "long", "v_n": v_n, "t2": -1, "ext": {}, "plot": {"color": "black", "ls": "-", "lw": 0.8, "alpha": 1., "label": r"BLh q=1.00 (SR)", "drawstyle":"steps"}},
        {"sim": "BLh_M11461635_M0_LK_SR",   "type": "long", "v_n": v_n, "t2": -1, "ext": {}, "plot": {"color": "gray", "ls": "-", "lw": 0.8, "alpha": 1., "label": r"BLh q=1.43 (SR)", "drawstyle":"steps"}},
        {"sim": "BLh_M10651772_M0_LK_SR", "type": "long", "v_n": v_n, "t2": -1, "ext": {}, "plot": {"color": "gray", "ls": "-", "lw": 0.8, "alpha": 1., "label": r"BLh q=1.66 (SR)","drawstyle": "steps"}},
        {"sim": "DD2_M13641364_M0_SR",      "type": "long", "v_n": v_n, "t2": -1, "ext": {}, "plot": {"color": "blue", "ls": "-", "lw": 0.8, "alpha": 1., "label": r"DD2* q=1.00 (SR)", "drawstyle":"steps"}},
        {"sim": "DD2_M13641364_M0_LK_SR_R04","type": "long", "v_n": v_n, "t2": -1, "ext": {}, "plot": {"color": "cyan", "ls": "-", "lw": 0.8, "alpha": 1., "label": r"DD2 q=1.00 (SR)", "drawstyle":"steps"}},
        {"sim": "DD2_M14971245_M0_SR",      "type": "long", "v_n": v_n, "t2": -1, "ext": {}, "plot": {"color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "label": r"DD2* q=1.19 (SR)", "drawstyle":"steps"}},
        {"sim": "DD2_M15091235_M0_LK_SR",   "type": "long", "v_n": v_n, "t2": -1, "ext": {}, "plot": {"color": "orange", "ls": "-", "lw": 0.8, "alpha": 1., "label": r"DD2 q=1.22 (SR)", "drawstyle":"steps"}},
        # {"sim": "SFHo_M11461635_M0_LK_SR", "type": "long", "v_n": v_n, "t2": -1, "ext": {}, "plot": {"color": "green", "ls": "-", "lw": 0.8, "alpha": 1., "label": r"SFHo* q=1.43 (SR)"}},
        # {"sim": "SLy4_M11461635_M0_LK_SR", "type": "long", "v_n": v_n, "t2": -1, "ext": {}, "plot": {"color": "red", "ls": "-", "lw": 0.8, "alpha": 1., "label": r"SLy4* q=1.43 (SR)"}},
    ]

    for t in task:
        t["plot"]["color"] = md.sim_dic_color[t["sim"]]
        t["plot"]["ls"] = md.sim_dic_ls[t["sim"]]
        t["plot"]["lw"] = md.sim_dic_lw[t["sim"]]

    for t in task: t["det"] = 0
    for t in task: t["v_n"] = "theta"
    for t in task: t["normalize"] = True
    for t in task: t["mask"] = "bern_geoend"

    # for t in task:
    #     if t["sim"] == "LS220_M13641364_M0_LK_SR_restart":
    #         t["label"] = t["label"].replace("\_restart", "")
    # for t in task:
    #     if t["sim"] == "SFHo_M13641364_M0_LK_SR_2019pizza":
    #         t["label"] = t["label"].replace("\_2019pizza", "")
    # ----------------------- theta
    plot_dic = {
        "task_v_n" : "theta",
        "type": "long",
        "figsize": (16., 5.5),
        "xmin": 0., "xmax": 90.,
        "ymin": 1e-4, "ymax": 1e-1,
        # "mask_below": 1e-15,
        "xscale": "linear", "yscale": "log",
        "xlabel": r"Angle from binary plane",
        "ylabel": r"$M_{\rm ej}/M_{\rm ej;tot}$",
        "legend": {"fancybox": False, "loc": 'lower center',
                   #"bbox_to_anchor":(1.0, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 14,
                   "framealpha": 0., "borderaxespad": 0., "frameon":False},
        #"title": r"Postdynamical ejecta properties",
        "figname": __outplotdir__ + "wind_hists_shared.png",
        "fontsize": 18,
        "savepdf": True
    }
    # plot_total_ejecta_hist(task, plot_dic)

    # ---------------- velinf
    plot_dic2 = copy.deepcopy(plot_dic)
    plot_dic2["xmax"] = 0.37
    plot_dic2["ymax"] = 5e-1
    plot_dic2["legend"] = {}
    plot_dic2["xlabel"] = r"$\langle \upsilon_{\infty} \rangle$"
    plot_dic2["figname"] = __outplotdir__ + "wind_hist_vinf.png"
    plot_dic2["task_v_n"] = "vel_inf"
    # for dic in task: dic["v_n"] = "vel_inf"
    # plot_total_ejecta_hist(task, plot_dic)

    # ---------------- Ye
    plot_dic3 = copy.deepcopy(plot_dic)
    plot_dic3["xmax"] = 0.5
    plot_dic3["ymax"] = 5e-1
    plot_dic3["legend"] = {}
    plot_dic3["xlabel"] = r"$\langle Y_e \rangle$"
    plot_dic3["figname"] = __outplotdir__ + "wind_hist_ye.png"
    plot_dic3["task_v_n"] = "Y_e"
    # for dic in task: dic["v_n"] = "Y_e"
    # plot_total_ejecta_hist(task, plot_dic)
    custom_plot_total_ejecta_hist(task, [plot_dic, plot_dic2, plot_dic3])

def tasl_plot_slice_2halfs_xz_2():

    task = {
        "sim": "BLh_M13641364_M0_LK_SR",
        "v_n_x": "x",
        "v_n_y": "z",
        "plane": "xz",
        "v_n_left": "abs_energy_over_density", #"Q_eff_nua_over_density",
        "v_n_right": "hu_0",
        "v_n_cont": "rho",
        "rl": 1,
        # "times":">20"
        # "iterations": "all" # [1949696]
        "iterations": [2187264] # 2121728
    }

    plot_dic = {
        "figsize":(6., 4.),
        "xlabel": "$X$ [km]",
        "ylabel": "$Z$ [km]",
        "xmin": -500, "xmax": 500, #   "auto"
        "ymin": -500, "ymax": 500, #   "auto"
        "xscale": "linear", "yscale": "linear",

        "vmin_left": 1e-9, "vmax_left": 1e-4,
        "clabel_left": r"$Q_{\rm abs; \:a}/D$ [$c^{5}/(G M_{\odot})$]",##r"$Q_{eff}(\nu_a) / D$",
        "cmap_left": "jet",
        "norm_left": "log",
        # "set_under_left": "black", "set_over_left": "blue",

        "vmin_right": 0.98, "vmax_right": 1.02,
        "clabel_right": "$-hu_0$",
        "cmap_right": "RdBu",
        "norm_right": "linear",
        # "set_under_right": "black", "set_over_right": "blue",

        "plot_dir": __outplotdir__ + task["sim"] + "/",
        #"figname": "it",
        "figname": "slice_xz_abs_energy_hu_{}.png".format(task["rl"]),
        "savepdf":True,
        "mirror_z": True,
        "fontsize":14,

        "cont_plot": {"levels": [1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13], "color": "black", "linewidth": 0.7} # [1e8, 1e9, 1e10, 1e11, 1e12, 1e13]
    }
    plot_slice_2halfs__with_morror_function(task, plot_dic)

    task["rl"] = 2
    plot_dic["xmin"], plot_dic["xmax"] = -220, 220
    plot_dic["ymin"], plot_dic["ymax"] = -220, 220
    plot_dic["figname"] = "slice_xz_abs_energy_hu_{}.png".format(task["rl"])

    plot_slice_2halfs__with_morror_function(task, plot_dic)

    task["rl"] = 3
    plot_dic["xmin"], plot_dic["xmax"] = -120, 120
    plot_dic["ymin"], plot_dic["ymax"] = -120, 120
    plot_dic["figname"] = "slice_xz_abs_energy_hu_{}.png".format(task["rl"])

    plot_slice_2halfs__with_morror_function(task, plot_dic)

    task["rl"] = 5
    plot_dic["xmin"], plot_dic["xmax"] = -30, 30
    plot_dic["ymin"], plot_dic["ymax"] = -30, 30
    plot_dic["figname"] = "slice_xz_abs_energy_hu_{}.png".format(task["rl"])

    plot_slice_2halfs__with_morror_function(task, plot_dic)

    ''' --- Ye --- '''

    task["rl"] = 1
    task["v_n_left"] = "Ye"
    plot_dic["xmin"], plot_dic["xmax"] = -500, 500
    plot_dic["ymin"], plot_dic["ymax"] = -500, 500
    plot_dic["clabel_left"] = "$Y_e$"
    plot_dic["vmin_left"], plot_dic["vmax_left"] = 0.2, 0.45
    plot_dic["norm_left"] = "linear"
    plot_dic["figname"] = "slice_xz_ye_hu_{}.png".format(task["rl"])

    plot_slice_2halfs__with_morror_function(task, plot_dic)

    task["rl"] = 2
    plot_dic["xmin"], plot_dic["xmax"] = -220, 220
    plot_dic["ymin"], plot_dic["ymax"] = -220, 220
    plot_dic["figname"] = "slice_xz_ye_hu_{}.png".format(task["rl"])

    plot_slice_2halfs__with_morror_function(task, plot_dic)

    task["rl"] = 3
    plot_dic["xmin"], plot_dic["xmax"] = -120, 120
    plot_dic["ymin"], plot_dic["ymax"] = -120, 120
    plot_dic["figname"] = "slice_xz_ye_hu_{}.png".format(task["rl"])

    plot_slice_2halfs__with_morror_function(task, plot_dic)

    task["rl"] = 5
    plot_dic["xmin"], plot_dic["xmax"] = -30, 30
    plot_dic["ymin"], plot_dic["ymax"] = -30, 30
    plot_dic["figname"] = "slice_xz_ye_hu_{}.png".format(task["rl"])

    plot_slice_2halfs__with_morror_function(task, plot_dic)

    #
    # # -- Ye
    # task["v_n_left"]="Ye"
    # plot_dic["clabel_left"]="$Y_e$"
    # plot_dic["vmin_left"], plot_dic["vmax_left"]=0.2, 0.45
    # plot_dic["norm_left"] = "linear"
    # plot_dic["plot_dir"] = __outplotdir__ + task["sim"] + '/{}__{}/'.format(task["v_n_left"], task["v_n_right"])
    # #
    # # plot_slice_2halfs(task, plot_dic)
    # #
    #
    # # close to the remnant
    # task["rl"] = 3
    # task["v_n_left"] = "abs_energy_over_density"#"Q_eff_nua_over_density" 2056192
    # plot_dic["clabel_left"] = r"$E_{\nu}/D$"## r"$Q_{eff}(\nu_a) / D$"
    # plot_dic["norm_left"] = "log"
    # plot_dic["xmin"], plot_dic["xmax"] = -120, 120
    # plot_dic["ymin"], plot_dic["ymax"] = -120, 120
    # plot_dic["vmin_left"], plot_dic["vmax_left"] = 1e-8, 1e-4
    # plot_dic["plot_dir"] = __outplotdir__ + task["sim"] + \
    #                        '/rl{}__{}__{}/'.format(task["rl"], task["v_n_left"], task["v_n_right"])
    # plot_dic["figname"] = "slice_xz_abs_energy_hu.png",
    # plot_slice_2halfs__with_morror_function(task, plot_dic)

    v_ns = ['abs_energy', 'abs_nua', 'abs_nue', 'abs_number', 'eave_nua', 'eave_nue',
           'eave_nux', 'E_nua', 'E_nue', 'E_nux', 'flux_fac', 'ndens_nua', 'ndens_nue',
           'ndens_nux', 'N_nua', 'N_nue', 'N_nux']
    #for v_n in v_ns:

def tasl_plot_slice_2halfs_xy_2():

    task = {
        "sim": "BLh_M13641364_M0_LK_SR",
        "v_n_x": "x",
        "v_n_y": "y",
        "plane": "xy",
        "v_n_left": "abs_energy_over_density", #"Q_eff_nua_over_density",
        "v_n_right": "hu_0",
        "v_n_cont": "rho",
        "rl": 1,
        # "times":">20"
        # "iterations": "all" # [1949696]
        "iterations": [2187264] # 2121728
    }

    plot_dic = {
        "figsize":(6., 4.),
        "xlabel": "$X$ [km]",
        "ylabel": "$Y$ [km]",
        "xmin": -500, "xmax": 500, #   "auto"
        "ymin": -500, "ymax": 500, #   "auto"
        "xscale": "linear", "yscale": "linear",

        "vmin_left": 1e-9, "vmax_left": 1e-4,
        "clabel_left": r"$Q_{\rm abs; \:a}/D$ [$c^{5}/(G M_{\odot})$]",##r"$Q_{eff}(\nu_a) / D$", [$c^5/GM_{\odot}^2$]
        "cmap_left": "jet",
        "norm_left": "log",
        # "set_under_left": "black", "set_over_left": "blue",

        "vmin_right": 0.98, "vmax_right": 1.02,
        "clabel_right": "$-hu_0$",
        "cmap_right": "RdBu",
        "norm_right": "linear",
        # "set_under_right": "black", "set_over_right": "blue",

        "plot_dir": __outplotdir__ + task["sim"] + "/",
        #"figname": "it",
        "figname": "slice_xy_abs_energy_hu_{}.png".format(task["rl"]),
        "savepdf":True,
        "mirror_z": False,
        "fontsize":14,

        "cont_plot": {"levels": [1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13], "color": "black", "linewidth": 0.7} # [1e8, 1e9, 1e10, 1e11, 1e12, 1e13]
    }
    plot_slice_2halfs__with_morror_function(task, plot_dic)

    task["rl"] = 2
    plot_dic["xmin"], plot_dic["xmax"] = -220, 220
    plot_dic["ymin"], plot_dic["ymax"] = -220, 220
    plot_dic["figname"] = "slice_xy_abs_energy_hu_{}.png".format(task["rl"])

    plot_slice_2halfs__with_morror_function(task, plot_dic)

    task["rl"] = 3
    plot_dic["xmin"], plot_dic["xmax"] = -120, 120
    plot_dic["ymin"], plot_dic["ymax"] = -120, 120
    plot_dic["figname"] = "slice_xy_abs_energy_hu_{}.png".format(task["rl"])

    plot_slice_2halfs__with_morror_function(task, plot_dic)

    task["rl"] = 5
    plot_dic["xmin"], plot_dic["xmax"] = -30, 30
    plot_dic["ymin"], plot_dic["ymax"] = -30, 30
    plot_dic["figname"] = "slice_xz_abs_energy_hu_{}.png".format(task["rl"])

    plot_slice_2halfs__with_morror_function(task, plot_dic)

    ''' --- Ye --- '''

    task["rl"] = 1
    task["v_n_left"] = "Ye"
    plot_dic["xmin"], plot_dic["xmax"] = -500, 500
    plot_dic["ymin"], plot_dic["ymax"] = -500, 500
    plot_dic["clabel_left"] = "$Y_e$"
    plot_dic["vmin_left"], plot_dic["vmax_left"] = 0.2, 0.45
    plot_dic["norm_left"] = "linear"
    plot_dic["figname"] = "slice_xy_ye_hu_{}.png".format(task["rl"])

    plot_slice_2halfs__with_morror_function(task, plot_dic)

    task["rl"] = 2
    plot_dic["xmin"], plot_dic["xmax"] = -220, 220
    plot_dic["ymin"], plot_dic["ymax"] = -220, 220
    plot_dic["figname"] = "slice_xy_ye_hu_{}.png".format(task["rl"])

    plot_slice_2halfs__with_morror_function(task, plot_dic)

    task["rl"] = 3
    plot_dic["xmin"], plot_dic["xmax"] = -120, 120
    plot_dic["ymin"], plot_dic["ymax"] = -120, 120
    plot_dic["figname"] = "slice_xy_ye_hu_{}.png".format(task["rl"])

    plot_slice_2halfs__with_morror_function(task, plot_dic)

    task["rl"] = 5
    plot_dic["xmin"], plot_dic["xmax"] = -30, 30
    plot_dic["ymin"], plot_dic["ymax"] = -30, 30
    plot_dic["figname"] = "slice_xy_ye_hu_{}.png".format(task["rl"])

    plot_slice_2halfs__with_morror_function(task, plot_dic)

    #
    # # -- Ye
    # task["v_n_left"]="Ye"
    # plot_dic["clabel_left"]="$Y_e$"
    # plot_dic["vmin_left"], plot_dic["vmax_left"]=0.2, 0.45
    # plot_dic["norm_left"] = "linear"
    # plot_dic["plot_dir"] = __outplotdir__ + task["sim"] + '/{}__{}/'.format(task["v_n_left"], task["v_n_right"])
    # #
    # # plot_slice_2halfs(task, plot_dic)
    # #
    #
    # # close to the remnant
    # task["rl"] = 3
    # task["v_n_left"] = "abs_energy_over_density"#"Q_eff_nua_over_density" 2056192
    # plot_dic["clabel_left"] = r"$E_{\nu}/D$"## r"$Q_{eff}(\nu_a) / D$"
    # plot_dic["norm_left"] = "log"
    # plot_dic["xmin"], plot_dic["xmax"] = -120, 120
    # plot_dic["ymin"], plot_dic["ymax"] = -120, 120
    # plot_dic["vmin_left"], plot_dic["vmax_left"] = 1e-8, 1e-4
    # plot_dic["plot_dir"] = __outplotdir__ + task["sim"] + \
    #                        '/rl{}__{}__{}/'.format(task["rl"], task["v_n_left"], task["v_n_right"])
    # plot_dic["figname"] = "slice_xz_abs_energy_hu.png",
    # plot_slice_2halfs__with_morror_function(task, plot_dic)

    v_ns = ['abs_energy', 'abs_nua', 'abs_nue', 'abs_number', 'eave_nua', 'eave_nue',
           'eave_nux', 'E_nua', 'E_nue', 'E_nux', 'flux_fac', 'ndens_nua', 'ndens_nue',
           'ndens_nux', 'N_nua', 'N_nue', 'N_nux']
    #for v_n in v_ns:

""" ==================== PAPER ===================== """

if __name__ == '__main__':

    ''' ------ Ejecta mass evolution ------- '''

    task_plot_total_ejecta_flux_2()

    ''' ------ Ejecta 1D histrograms ------- '''

    #task_plot_total_ejecta_hist_2()
    # custom_task_plot_total_ejecta_hist_2() # subplots

    ''' ------ Slices XZ Ye & hu ----------- '''

    tasl_plot_slice_2halfs_xz_2()
    tasl_plot_slice_2halfs_xy_2()

''' --- iteration 2 --- '''

if __name__ == '__main__':

    """ --- Ejecta Mass --- """
    # task_plot_total_ejecta_flux_2()

    # task_plot_total_ejecta_hist_2()

    #tasl_plot_slice_2halfs__2()

    #exit(1)

''' --- iteration 1 --- '''

if __name__ == '__main__':

    """ --- Correlate Q_eff_nua and hu_0, theta, Ye --- """
    # task_plot_corr_qeff_u_0() #
    # task_plot_corr_qeff_theta() # bad
    # task_plot_corr_qeff_ye()  #

    """ --- D2 slices --- """
    # tasl_plot_slice_2halfs()

    """ --- Ejecta Mass --- """
    # task_plot_total_ejecta_flux()
    # task_plot_total_ejecta_flux_extapolate()

    """ --- Ejecta Properties --- """
    # task_plot_total_ejecta_hist()
    # tast_plot_total_ejecta_timecorr()

    """ --- Ejecta Correlation --- """
    # task_plot_total_ejecta_corr()
    # plot_ye_theta_correlation_ejecta()

    """ --- baryon loading -- """
    # task_plot_total_mass_within_thera_60_and_total_wind()

    """ ---------------------- RESOLUTION ------------------------ """
    # task_resolution_plot_total_ejecta_flux()


    #exit(1)

    #plot_qeff_dunb_slices()
    # plot_ye_theta_correlation_ejecta()
    # plot_total_flux_of_matter()
    #corr_nu_heat_rate_and_wind()
    # plot_ejecta_time_corr_properites()
