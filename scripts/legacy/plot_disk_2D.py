# ---
#
# Disk structure in 2D
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

__outplotdir__ = "/data01/numrel/vsevolod.nedora/figs/all3/disk_2D/"

def plot_den_unb__vel_z_sly4_evol():

    # tmp = d3class.get_data(688128, 3, "xy", "ang_mom_flux")
    # print(tmp.min(), tmp.max())
    # print(tmp)
    # exit(1) # dens_unb_geo

    """ --- --- --- """


    '''sly4 '''
    simlist = ["SLy4_M13641364_M0_SR", "SLy4_M13641364_M0_SR", "SLy4_M13641364_M0_SR", "SLy4_M13641364_M0_SR"]
    # itlist = [434176, 475136, 516096, 565248]
    # itlist = [606208, 647168, 696320, 737280]
    # itlist = [434176, 516096, 647168, 737280]
    ''' ls220 '''
    simlist = ["LS220_M14691268_M0_LK_SR", "LS220_M14691268_M0_LK_SR", "LS220_M14691268_M0_LK_SR"]#, "LS220_M14691268_M0_LK_SR"]
    itlist = [1515520, 1728512, 1949696]#, 2162688]
    ''' dd2 '''
    simlist = ["DD2_M13641364_M0_LK_SR_R04", "DD2_M13641364_M0_LK_SR_R04", "DD2_M13641364_M0_LK_SR_R04"]#, "DD2_M13641364_M0_LK_SR_R04"]
    itlist = [1111116,1741554,2213326]#,2611022]
    #
    simlist = ["DD2_M13641364_M0_LK_SR_R04", "BLh_M13641364_M0_LK_SR", "LS220_M14691268_M0_LK_SR", "SLy4_M13641364_M0_SR"]
    itlist = [2611022, 1974272, 1949696, 737280]
    #

    #
    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = __outplotdir__
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (4*len(simlist), 6.0)  # <->, |] # to match hists with (8.5, 2.7)
    o_plot.gen_set["figname"] = "disk_structure_last.png".format(simlist[0])#"DD2_1512_slices.png" # LS_1412_slices
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = True
    o_plot.gen_set["dpi"] = 128
    o_plot.gen_set["subplots_adjust_h"] = -0.35
    o_plot.gen_set["subplots_adjust_w"] = 0.05
    o_plot.set_plot_dics = []
    #
    rl = 3
    #
    o_plot.gen_set["figsize"] = (4.2*len(simlist), 8.0)  # <->, |] # to match hists with (8.5, 2.7)

    plot_x_i = 1
    for sim, it in zip(simlist, itlist):
        print("sim:{} it:{}".format(sim, it))
        d3class = LOAD_PROFILE_XYXZ(sim)
        d1class = ADD_METHODS_ALL_PAR(sim)

        t = d3class.get_time_for_it(it, d1d2d3prof="prof")
        tmerg = d1class.get_par("tmerg")
        xmin, xmax, ymin, ymax, zmin, zmax = UTILS.get_xmin_xmax_ymin_ymax_zmin_zmax(rl)



        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        mask = "x>0"
        #
        v_n = "rho"
        data_arr = d3class.get_data(it, rl, "xz", v_n)
        x_arr = d3class.get_data(it, rl, "xz", "x")
        z_arr = d3class.get_data(it, rl, "xz", "z")
        # print(data_arr); exit(1)

        contour_dic_xz = {
            'task': 'contour',
            'ptype': 'cartesian', 'aspect': 1.,
            'xarr': x_arr, "yarr": z_arr, "zarr": data_arr, 'levels': [1.e13 / 6.176e+17],
            'position': (1, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
            'colors': ['white'], 'lss': ["-"], 'lws': [1.],
            'v_n_x': 'x', 'v_n_y': 'y', 'v_n': 'rho',
            'xscale': None, 'yscale': None,
            'fancyticks': True,
            'sharey': False,
            'sharex': True,  # removes angular citkscitks
            'fontsize': 14,
            'labelsize': 14}
        o_plot.set_plot_dics.append(contour_dic_xz)

        rho_dic_xz = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                      'xarr': x_arr, "yarr": z_arr, "zarr": data_arr,
                      'position': (1, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
                      'cbar': {},
                      'v_n_x': 'x', 'v_n_y': 'z', 'v_n': v_n,
                      'xmin': xmin, 'xmax': xmax, 'ymin': zmin, 'ymax': zmax, 'vmin': 1e-9, 'vmax': 1e-5,
                      'fill_vmin': False,  # fills the x < vmin with vmin
                      'xscale': None, 'yscale': None,
                      'mask': mask, 'cmap': 'Greys', 'norm': "log",
                      'fancyticks': True,
                      'minorticks':True,
                      'title': {"text": sim.replace('_', '\_'), 'fontsize': 12},
                      #'title': {"text": r'$t-t_{merg}:$' + r'${:.1f}$ [ms]'.format((t - tmerg) * 1e3), 'fontsize': 14},
                      'sharey': False,
                      'sharex': True,  # removes angular citkscitks
                      'fontsize': 14,
                      'labelsize': 14
                      }
        #
        data_arr = d3class.get_data(it, rl, "xy", v_n)
        x_arr = d3class.get_data(it, rl, "xy", "x")
        y_arr = d3class.get_data(it, rl, "xy", "y")

        contour_dic_xy = {
            'task': 'contour',
            'ptype': 'cartesian', 'aspect': 1.,
            'xarr': x_arr, "yarr": y_arr, "zarr": data_arr, 'levels': [1.e13 / 6.176e+17],
            'position': (2, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
            'colors': ['white'], 'lss': ["-"], 'lws': [1.],
            'v_n_x': 'x', 'v_n_y': 'y', 'v_n': 'rho',
            'xscale': None, 'yscale': None,
            'fancyticks': True,
            'sharey': False,
            'sharex': True,  # removes angular citkscitks
            'fontsize': 14,
            'labelsize': 14}
        o_plot.set_plot_dics.append(contour_dic_xy)

        rho_dic_xy = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                      'xarr': x_arr, "yarr": y_arr, "zarr": data_arr,
                      'position': (2, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
                      'cbar': {},
                      'v_n_x': 'x', 'v_n_y': 'y', 'v_n': v_n,
                      'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax, 'vmin': 1e-9, 'vmax': 1e-5,
                      'fill_vmin': False,  # fills the x < vmin with vmin
                      'xscale': None, 'yscale': None,
                      'mask': mask, 'cmap': 'Greys', 'norm': "log",
                      'fancyticks': True,
                      'minorticks': True,
                      'title': {},
                      'sharey': False,
                      'sharex': False,  # removes angular citkscitks
                      'fontsize': 14,
                      'labelsize': 14
                      }
        #
        if plot_x_i == 1:
            rho_dic_xy['cbar'] = {'location': 'bottom -.05 .00', 'label': r'$\rho$ [GEO]',  # 'fmt': '%.1e',
                          'labelsize': 14,
                          'fontsize': 14}
        if plot_x_i > 1:
            rho_dic_xz['sharey'] = True
            rho_dic_xy['sharey'] = True

        o_plot.set_plot_dics.append(rho_dic_xz)
        o_plot.set_plot_dics.append(rho_dic_xy)

        # ----------------------------------------------------------------------
        v_n = "dens_unb_bern"
        #
        data_arr = d3class.get_data(it, rl, "xz", v_n)
        x_arr = d3class.get_data(it, rl, "xz", "x")
        z_arr = d3class.get_data(it, rl, "xz", "z")
        dunb_dic_xz = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                      'xarr': x_arr, "yarr": z_arr, "zarr": data_arr,
                      'position': (1, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
                      'cbar': {},
                      'v_n_x': 'x', 'v_n_y': 'z', 'v_n': v_n,
                      'xmin': xmin, 'xmax': xmax, 'ymin': zmin, 'ymax': zmax, 'vmin': 1e-10, 'vmax': 1e-7,
                      'fill_vmin': False,  # fills the x < vmin with vmin
                      'xscale': None, 'yscale': None,
                      'mask': mask, 'cmap': 'Blues', 'norm': "log",
                      'fancyticks': True,
                       'minorticks': True,
                       'title': {},#{"text": r'$t-t_{merg}:$' + r'${:.1f}$ [ms]'.format((t - tmerg) * 1e3), 'fontsize': 14},
                      'sharex': True,  # removes angular citkscitks
                      'sharey': False,
                      'fontsize': 14,
                      'labelsize': 14
                      }
        #
        data_arr = d3class.get_data(it, rl, "xy", v_n)
        x_arr = d3class.get_data(it, rl, "xy", "x")
        y_arr = d3class.get_data(it, rl, "xy", "y")
        dunb_dic_xy = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                      'xarr': x_arr, "yarr": y_arr, "zarr": data_arr,
                      'position': (2, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
                      'cbar': {},
                      'fill_vmin': False,  # fills the x < vmin with vmin
                      'v_n_x': 'x', 'v_n_y': 'y', 'v_n': v_n,
                      'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax, 'vmin': 1e-10, 'vmax': 1e-7,
                      'xscale': None, 'yscale': None,
                      'mask': mask, 'cmap': 'Blues', 'norm': "log",
                      'fancyticks': True,
                       'minorticks': True,
                       'title': {},
                      'sharey': False,
                      'sharex': False,  # removes angular citkscitks
                      'fontsize': 14,
                      'labelsize': 14
                      }
        #
        if plot_x_i == 2:
            dunb_dic_xy['cbar'] = {'location': 'bottom -.05 .00', 'label': r'$D_{\rm{unb}}$ [GEO]',  # 'fmt': '%.1e',
                          'labelsize': 14,
                          'fontsize': 14}
        if plot_x_i > 1:
            dunb_dic_xz['sharey'] = True
            dunb_dic_xy['sharey'] = True

        o_plot.set_plot_dics.append(dunb_dic_xz)
        o_plot.set_plot_dics.append(dunb_dic_xy)

        # ----------------------------------------------------------------------
        mask = "x<0"
        #
        v_n = "Ye"
        cmap = "bwr_r"
        #
        data_arr = d3class.get_data(it, rl, "xz", v_n)
        x_arr = d3class.get_data(it, rl, "xz", "x")
        z_arr = d3class.get_data(it, rl, "xz", "z")
        ye_dic_xz = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                       'xarr': x_arr, "yarr": z_arr, "zarr": data_arr,
                       'position': (1, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
                       'cbar': {},
                       'fill_vmin': False,  # fills the x < vmin with vmin
                       'v_n_x': 'x', 'v_n_y': 'z', 'v_n': v_n,
                       'xmin': xmin, 'xmax': xmax, 'ymin': zmin, 'ymax': zmax, 'vmin': 0.05, 'vmax': 0.5,
                       'xscale': None, 'yscale': None,
                       'mask': mask, 'cmap': cmap, 'norm': None,
                       'fancyticks': True,
                       'minorticks': True,
                       'title': {},#{"text": r'$t-t_{merg}:$' + r'${:.1f}$ [ms]'.format((t - tmerg) * 1e3), 'fontsize': 14},
                       'sharey': False,
                       'sharex': True,  # removes angular citkscitks
                       'fontsize': 14,
                       'labelsize': 14
                       }
        #
        data_arr = d3class.get_data(it, rl, "xy", v_n)
        x_arr = d3class.get_data(it, rl, "xy", "x")
        y_arr = d3class.get_data(it, rl, "xy", "y")
        ye_dic_xy = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                       'xarr': x_arr, "yarr": y_arr, "zarr": data_arr,
                       'position': (2, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
                       'cbar': {},
                       'fill_vmin': False,  # fills the x < vmin with vmin
                       'v_n_x': 'x', 'v_n_y': 'y', 'v_n': v_n,
                       'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax, 'vmin': 0.01, 'vmax': 0.5,
                       'xscale': None, 'yscale': None,
                       'mask': mask, 'cmap': cmap, 'norm': None,
                       'fancyticks': True,
                       'minorticks': True,
                       'title': {},
                       'sharey': False,
                       'sharex': False,  # removes angular citkscitks
                       'fontsize': 14,
                       'labelsize': 14
                       }
        #
        if plot_x_i == 3:
            ye_dic_xy['cbar'] = {'location': 'bottom -.05 .00', 'label': r'$Y_e$',   'fmt': '%.1f',
                          'labelsize': 14,
                          'fontsize': 14}
        if plot_x_i > 1:
            ye_dic_xz['sharey'] = True
            ye_dic_xy['sharey'] = True

        o_plot.set_plot_dics.append(ye_dic_xz)
        o_plot.set_plot_dics.append(ye_dic_xy)

        # ----------------------------------------------------------
        tcoll = d1class.get_par("tcoll_gw")
        if not np.isnan(tcoll) and t >= tcoll:
            print(tcoll, t)
            v_n = "lapse"
            mask = "z>0.15"
            data_arr = d3class.get_data(it, rl, "xz", v_n)
            x_arr = d3class.get_data(it, rl, "xz", "x")
            z_arr = d3class.get_data(it, rl, "xz", "z")
            lapse_dic_xz = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                            'xarr': x_arr, "yarr": z_arr, "zarr": data_arr,
                            'position': (1, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
                            'cbar': {},
                            'v_n_x': 'x', 'v_n_y': 'z', 'v_n': v_n,
                            'xmin': xmin, 'xmax': xmax, 'ymin': zmin, 'ymax': zmax, 'vmin': 0., 'vmax': 0.15,
                            'fill_vmin': False,  # fills the x < vmin with vmin
                            'xscale': None, 'yscale': None,
                            'mask': mask, 'cmap': 'Greys', 'norm': None,
                            'fancyticks': True,
                            'minorticks': True,
                            'title': {},#,{"text": r'$t-t_{merg}:$' + r'${:.1f}$ [ms]'.format((t - tmerg) * 1e3),
                                      #'fontsize': 14},
                            'sharey': False,
                            'sharex': True,  # removes angular citkscitks
                            'fontsize': 14,
                            'labelsize': 14
                            }
            #
            data_arr = d3class.get_data(it, rl, "xy", v_n)
            # print(data_arr.min(), data_arr.max()); exit(1)
            x_arr = d3class.get_data(it, rl, "xy", "x")
            y_arr = d3class.get_data(it, rl, "xy", "y")
            lapse_dic_xy = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                            'xarr': x_arr, "yarr": y_arr, "zarr": data_arr,
                            'position': (2, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
                            'cbar': {},
                            'v_n_x': 'x', 'v_n_y': 'y', 'v_n': v_n,
                            'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax, 'vmin': 0, 'vmax': 0.15,
                            'fill_vmin': False,  # fills the x < vmin with vmin
                            'xscale': None, 'yscale': None,
                            'mask': mask, 'cmap': 'Greys', 'norm': None,
                            'fancyticks': True,
                            'minorticks': True,
                            'title': {},
                            'sharey': False,
                            'sharex': False,  # removes angular citkscitks
                            'fontsize': 14,
                            'labelsize': 14
                            }
            #
            # if plot_x_i == 1:
            #     rho_dic_xy['cbar'] = {'location': 'bottom -.05 .00', 'label': r'$\rho$ [GEO]',  # 'fmt': '%.1e',
            #                           'labelsize': 14,
            #                           'fontsize': 14}
            if plot_x_i > 1:
                lapse_dic_xz['sharey'] = True
                lapse_dic_xy['sharey'] = True

            o_plot.set_plot_dics.append(lapse_dic_xz)
            o_plot.set_plot_dics.append(lapse_dic_xy)


        plot_x_i += 1




    o_plot.main()

    exit(0)

def plot_disk_2d():

    # tmp = d3class.get_data(688128, 3, "xy", "ang_mom_flux")
    # print(tmp.min(), tmp.max())
    # print(tmp)
    # exit(1) # dens_unb_geo

    """ --- --- --- """


    '''sly4 '''
    simlist = ["SLy4_M13641364_M0_SR", "SLy4_M13641364_M0_SR", "SLy4_M13641364_M0_SR", "SLy4_M13641364_M0_SR"]
    # itlist = [434176, 475136, 516096, 565248]
    # itlist = [606208, 647168, 696320, 737280]
    # itlist = [434176, 516096, 647168, 737280]
    ''' ls220 '''
    simlist = ["LS220_M14691268_M0_LK_SR", "LS220_M14691268_M0_LK_SR", "LS220_M14691268_M0_LK_SR"]#, "LS220_M14691268_M0_LK_SR"]
    itlist = [1515520, 1728512, 1949696]#, 2162688]
    ''' dd2 '''
    simlist = ["DD2_M13641364_M0_LK_SR_R04", "DD2_M13641364_M0_LK_SR_R04", "DD2_M13641364_M0_LK_SR_R04"]#, "DD2_M13641364_M0_LK_SR_R04"]
    itlist = [1111116,1741554,2213326]#,2611022]
    #
    simlist = ["DD2_M13641364_M0_LK_SR_R04", "BLh_M13641364_M0_LK_SR", "LS220_M14691268_M0_LK_SR", "SLy4_M13641364_M0_SR"]
    itlist = [2611022, 1974272, 1949696, 737280]
    #

    #


    simlist = ["BLh_M13641364_M0_LK_SR"]
    itlist = [737280]
    v_ns = ["rho", "Ye"]
    rl = 3
    plane = "xy"

    data_dic = {}
    Printcolor.blue("Collecting data...")
    for sim, it in zip(simlist, itlist):
        simit = str(sim) + str(it)
        data_dic[simit] = {}
        print("sim:{} it:{}".format(sim, it))
        d3class = LOAD_PROFILE_XYXZ(sim)
        # d1class = ADD_METHODS_ALL_PAR(sim)
        x_arr = d3class.get_data(it, rl, plane, "x")
        y_arr = d3class.get_data(it, rl, plane, "y")
        data_dic[simit]["x_arr"] = x_arr
        data_dic[simit]["y_arr"] = y_arr
        for v_n in v_ns:
            data_arr = d3class.get_data(it, rl, plane, v_n)
            data_dic[simit][v_n] = data_arr
    Printcolor.green("Data is collected")
    #

    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = __outplotdir__
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (4.2, 3.6)#(4 * len(simlist), 6.0)  # <->, |] # to match hists with (8.5, 2.7)
    o_plot.gen_set["figname"] = "disk_densmodes.png"  # "DD2_1512_slices.png" # LS_1412_slices
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = True
    o_plot.gen_set["dpi"] = 128
    o_plot.gen_set["subplots_adjust_h"] = -0.35
    o_plot.gen_set["subplots_adjust_w"] = 0.05
    o_plot.set_plot_dics = []

    plot_x_i = 1
    for sim, it in zip(simlist, itlist):
        simit = str(sim) + str(it)
        #
        mask = "x>0"
        v_n = "rho"
        cmap = 'Greys'
        xmin, xmax, ymin, ymax, zmin, zmax = UTILS.get_xmin_xmax_ymin_ymax_zmin_zmax(rl)
        rho_dic_xy = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                      'xarr': data_dic[simit]["x_arr"], "yarr": data_dic[simit]["y_arr"], "zarr": data_dic[simit][v_n],
                      'position': (1, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
                      'cbar': {},
                      'v_n_x': 'x', 'v_n_y': 'y', 'v_n': v_n,
                      'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax, 'vmin': 1e-9, 'vmax': 1e-5,
                      'fill_vmin': False,  # fills the x < vmin with vmin
                      'xscale': None, 'yscale': None,
                      'mask': mask, 'cmap': cmap, 'norm': "log",
                      'fancyticks': True,
                      'minorticks': True,
                      'title': {},
                      'sharey': False,
                      'sharex': False,  # removes angular citkscitks
                      'fontsize': 14,
                      'labelsize': 14
        }
        rho_dic_xy['cbar'] = {'location': 'left -0.6 .00', 'label': r'$\rho$ [GEO]',  # 'fmt': '%.1e',
                              'labelsize': 14,
                              'fontsize': 14}
        o_plot.set_plot_dics.append(rho_dic_xy)
        #
        contour_dic_xy = {
            'position': (1, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
            'task': 'contour',
            'ptype': 'cartesian', 'aspect': 1.,
            'xarr': data_dic[simit]["x_arr"], "yarr": data_dic[simit]["y_arr"], "zarr": data_dic[simit][v_n],
            'levels': [1.e13 / 6.176e+17],
            'colors': ['white'], 'lss': ["-"], 'lws': [1.],
            'v_n_x': 'x', 'v_n_y': 'y', 'v_n': 'rho',
            'xscale': None, 'yscale': None,
            'fancyticks': True,
            'sharey': False,
            'sharex': False,  # removes angular citkscitks
            'fontsize': 14,
            'labelsize': 14}
        o_plot.set_plot_dics.append(contour_dic_xy)

        # ---

        mask = "x<0"
        v_n = "Ye"
        cmap = "bwr_r"
        ye_dic_xy = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                     'xarr': data_dic[simit]["x_arr"], "yarr": data_dic[simit]["y_arr"], "zarr": data_dic[simit][v_n],
                     'position': (1, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
                     'cbar': {},
                     'fill_vmin': False,  # fills the x < vmin with vmin
                     'v_n_x': 'x', 'v_n_y': 'y', 'v_n': v_n,
                     'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax, 'vmin': 0.01, 'vmax': 0.5,
                     'xscale': None, 'yscale': None,
                     'mask': mask, 'cmap': cmap, 'norm': None,
                     'fancyticks': True,
                     'minorticks': True,
                     'title': {},
                     'sharey': False,
                     'sharex': False,  # removes angular citkscitks
                     'fontsize': 14,
                     'labelsize': 14
                     }
        ye_dic_xy['cbar'] = {'location': 'right -.02 .00', 'label': r'$Y_e$', 'fmt': '%.1f',
                             'labelsize': 14,
                             'fontsize': 14}
        o_plot.set_plot_dics.append(ye_dic_xy)
        plot_x_i += 1
    o_plot.main()
    exit(1)









    o_plot = PLOT_MANY_TASKS()
    o_plot.gen_set["figdir"] = __outplotdir__
    o_plot.gen_set["type"] = "cartesian"
    o_plot.gen_set["figsize"] = (4*len(simlist), 6.0)  # <->, |] # to match hists with (8.5, 2.7)
    o_plot.gen_set["figname"] = "disk_structure_last.png".format(simlist[0])#"DD2_1512_slices.png" # LS_1412_slices
    o_plot.gen_set["sharex"] = False
    o_plot.gen_set["sharey"] = True
    o_plot.gen_set["dpi"] = 128
    o_plot.gen_set["subplots_adjust_h"] = -0.35
    o_plot.gen_set["subplots_adjust_w"] = 0.05
    o_plot.set_plot_dics = []
    #
    rl = 3
    #
    o_plot.gen_set["figsize"] = (4.2*len(simlist), 8.0)  # <->, |] # to match hists with (8.5, 2.7)

    plot_x_i = 1
    for sim, it in zip(simlist, itlist):
        print("sim:{} it:{}".format(sim, it))
        d3class = LOAD_PROFILE_XYXZ(sim)
        d1class = ADD_METHODS_ALL_PAR(sim)

        t = d3class.get_time_for_it(it, d1d2d3prof="prof")
        tmerg = d1class.get_par("tmerg")
        xmin, xmax, ymin, ymax, zmin, zmax = UTILS.get_xmin_xmax_ymin_ymax_zmin_zmax(rl)



        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        mask = "x>0"
        #
        v_n = "rho"
        data_arr = d3class.get_data(it, rl, "xz", v_n)
        x_arr = d3class.get_data(it, rl, "xz", "x")
        z_arr = d3class.get_data(it, rl, "xz", "z")
        # print(data_arr); exit(1)

        contour_dic_xz = {
            'task': 'contour',
            'ptype': 'cartesian', 'aspect': 1.,
            'xarr': x_arr, "yarr": z_arr, "zarr": data_arr, 'levels': [1.e13 / 6.176e+17],
            'position': (1, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
            'colors': ['white'], 'lss': ["-"], 'lws': [1.],
            'v_n_x': 'x', 'v_n_y': 'y', 'v_n': 'rho',
            'xscale': None, 'yscale': None,
            'fancyticks': True,
            'sharey': False,
            'sharex': True,  # removes angular citkscitks
            'fontsize': 14,
            'labelsize': 14}
        o_plot.set_plot_dics.append(contour_dic_xz)

        rho_dic_xz = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                      'xarr': x_arr, "yarr": z_arr, "zarr": data_arr,
                      'position': (1, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
                      'cbar': {},
                      'v_n_x': 'x', 'v_n_y': 'z', 'v_n': v_n,
                      'xmin': xmin, 'xmax': xmax, 'ymin': zmin, 'ymax': zmax, 'vmin': 1e-9, 'vmax': 1e-5,
                      'fill_vmin': False,  # fills the x < vmin with vmin
                      'xscale': None, 'yscale': None,
                      'mask': mask, 'cmap': 'Greys', 'norm': "log",
                      'fancyticks': True,
                      'minorticks':True,
                      'title': {"text": sim.replace('_', '\_'), 'fontsize': 12},
                      #'title': {"text": r'$t-t_{merg}:$' + r'${:.1f}$ [ms]'.format((t - tmerg) * 1e3), 'fontsize': 14},
                      'sharey': False,
                      'sharex': True,  # removes angular citkscitks
                      'fontsize': 14,
                      'labelsize': 14
                      }
        #
        data_arr = d3class.get_data(it, rl, "xy", v_n)
        x_arr = d3class.get_data(it, rl, "xy", "x")
        y_arr = d3class.get_data(it, rl, "xy", "y")

        contour_dic_xy = {
            'task': 'contour',
            'ptype': 'cartesian', 'aspect': 1.,
            'xarr': x_arr, "yarr": y_arr, "zarr": data_arr, 'levels': [1.e13 / 6.176e+17],
            'position': (2, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
            'colors': ['white'], 'lss': ["-"], 'lws': [1.],
            'v_n_x': 'x', 'v_n_y': 'y', 'v_n': 'rho',
            'xscale': None, 'yscale': None,
            'fancyticks': True,
            'sharey': False,
            'sharex': True,  # removes angular citkscitks
            'fontsize': 14,
            'labelsize': 14}
        o_plot.set_plot_dics.append(contour_dic_xy)

        rho_dic_xy = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                      'xarr': x_arr, "yarr": y_arr, "zarr": data_arr,
                      'position': (2, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
                      'cbar': {},
                      'v_n_x': 'x', 'v_n_y': 'y', 'v_n': v_n,
                      'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax, 'vmin': 1e-9, 'vmax': 1e-5,
                      'fill_vmin': False,  # fills the x < vmin with vmin
                      'xscale': None, 'yscale': None,
                      'mask': mask, 'cmap': 'Greys', 'norm': "log",
                      'fancyticks': True,
                      'minorticks': True,
                      'title': {},
                      'sharey': False,
                      'sharex': False,  # removes angular citkscitks
                      'fontsize': 14,
                      'labelsize': 14
                      }
        #
        if plot_x_i == 1:
            rho_dic_xy['cbar'] = {'location': 'bottom -.05 .00', 'label': r'$\rho$ [GEO]',  # 'fmt': '%.1e',
                          'labelsize': 14,
                          'fontsize': 14}
        if plot_x_i > 1:
            rho_dic_xz['sharey'] = True
            rho_dic_xy['sharey'] = True

        o_plot.set_plot_dics.append(rho_dic_xz)
        o_plot.set_plot_dics.append(rho_dic_xy)

        # ----------------------------------------------------------------------
        v_n = "dens_unb_bern"
        #
        data_arr = d3class.get_data(it, rl, "xz", v_n)
        x_arr = d3class.get_data(it, rl, "xz", "x")
        z_arr = d3class.get_data(it, rl, "xz", "z")
        dunb_dic_xz = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                      'xarr': x_arr, "yarr": z_arr, "zarr": data_arr,
                      'position': (1, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
                      'cbar': {},
                      'v_n_x': 'x', 'v_n_y': 'z', 'v_n': v_n,
                      'xmin': xmin, 'xmax': xmax, 'ymin': zmin, 'ymax': zmax, 'vmin': 1e-10, 'vmax': 1e-7,
                      'fill_vmin': False,  # fills the x < vmin with vmin
                      'xscale': None, 'yscale': None,
                      'mask': mask, 'cmap': 'Blues', 'norm': "log",
                      'fancyticks': True,
                       'minorticks': True,
                       'title': {},#{"text": r'$t-t_{merg}:$' + r'${:.1f}$ [ms]'.format((t - tmerg) * 1e3), 'fontsize': 14},
                      'sharex': True,  # removes angular citkscitks
                      'sharey': False,
                      'fontsize': 14,
                      'labelsize': 14
                      }
        #
        data_arr = d3class.get_data(it, rl, "xy", v_n)
        x_arr = d3class.get_data(it, rl, "xy", "x")
        y_arr = d3class.get_data(it, rl, "xy", "y")
        dunb_dic_xy = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                      'xarr': x_arr, "yarr": y_arr, "zarr": data_arr,
                      'position': (2, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
                      'cbar': {},
                      'fill_vmin': False,  # fills the x < vmin with vmin
                      'v_n_x': 'x', 'v_n_y': 'y', 'v_n': v_n,
                      'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax, 'vmin': 1e-10, 'vmax': 1e-7,
                      'xscale': None, 'yscale': None,
                      'mask': mask, 'cmap': 'Blues', 'norm': "log",
                      'fancyticks': True,
                       'minorticks': True,
                       'title': {},
                      'sharey': False,
                      'sharex': False,  # removes angular citkscitks
                      'fontsize': 14,
                      'labelsize': 14
                      }
        #
        if plot_x_i == 2:
            dunb_dic_xy['cbar'] = {'location': 'bottom -.05 .00', 'label': r'$D_{\rm{unb}}$ [GEO]',  # 'fmt': '%.1e',
                          'labelsize': 14,
                          'fontsize': 14}
        if plot_x_i > 1:
            dunb_dic_xz['sharey'] = True
            dunb_dic_xy['sharey'] = True

        o_plot.set_plot_dics.append(dunb_dic_xz)
        o_plot.set_plot_dics.append(dunb_dic_xy)

        # ----------------------------------------------------------------------
        mask = "x<0"
        #
        v_n = "Ye"
        cmap = "bwr_r"
        #
        data_arr = d3class.get_data(it, rl, "xz", v_n)
        x_arr = d3class.get_data(it, rl, "xz", "x")
        z_arr = d3class.get_data(it, rl, "xz", "z")
        ye_dic_xz = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                       'xarr': x_arr, "yarr": z_arr, "zarr": data_arr,
                       'position': (1, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
                       'cbar': {},
                       'fill_vmin': False,  # fills the x < vmin with vmin
                       'v_n_x': 'x', 'v_n_y': 'z', 'v_n': v_n,
                       'xmin': xmin, 'xmax': xmax, 'ymin': zmin, 'ymax': zmax, 'vmin': 0.05, 'vmax': 0.5,
                       'xscale': None, 'yscale': None,
                       'mask': mask, 'cmap': cmap, 'norm': None,
                       'fancyticks': True,
                       'minorticks': True,
                       'title': {},#{"text": r'$t-t_{merg}:$' + r'${:.1f}$ [ms]'.format((t - tmerg) * 1e3), 'fontsize': 14},
                       'sharey': False,
                       'sharex': True,  # removes angular citkscitks
                       'fontsize': 14,
                       'labelsize': 14
                       }
        #
        data_arr = d3class.get_data(it, rl, "xy", v_n)
        x_arr = d3class.get_data(it, rl, "xy", "x")
        y_arr = d3class.get_data(it, rl, "xy", "y")
        ye_dic_xy = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                       'xarr': x_arr, "yarr": y_arr, "zarr": data_arr,
                       'position': (2, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
                       'cbar': {},
                       'fill_vmin': False,  # fills the x < vmin with vmin
                       'v_n_x': 'x', 'v_n_y': 'y', 'v_n': v_n,
                       'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax, 'vmin': 0.01, 'vmax': 0.5,
                       'xscale': None, 'yscale': None,
                       'mask': mask, 'cmap': cmap, 'norm': None,
                       'fancyticks': True,
                       'minorticks': True,
                       'title': {},
                       'sharey': False,
                       'sharex': False,  # removes angular citkscitks
                       'fontsize': 14,
                       'labelsize': 14
                       }
        #
        if plot_x_i == 3:
            ye_dic_xy['cbar'] = {'location': 'bottom -.05 .00', 'label': r'$Y_e$',   'fmt': '%.1f',
                          'labelsize': 14,
                          'fontsize': 14}
        if plot_x_i > 1:
            ye_dic_xz['sharey'] = True
            ye_dic_xy['sharey'] = True

        o_plot.set_plot_dics.append(ye_dic_xz)
        o_plot.set_plot_dics.append(ye_dic_xy)

        # ----------------------------------------------------------
        tcoll = d1class.get_par("tcoll_gw")
        if not np.isnan(tcoll) and t >= tcoll:
            print(tcoll, t)
            v_n = "lapse"
            mask = "z>0.15"
            data_arr = d3class.get_data(it, rl, "xz", v_n)
            x_arr = d3class.get_data(it, rl, "xz", "x")
            z_arr = d3class.get_data(it, rl, "xz", "z")
            lapse_dic_xz = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                            'xarr': x_arr, "yarr": z_arr, "zarr": data_arr,
                            'position': (1, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
                            'cbar': {},
                            'v_n_x': 'x', 'v_n_y': 'z', 'v_n': v_n,
                            'xmin': xmin, 'xmax': xmax, 'ymin': zmin, 'ymax': zmax, 'vmin': 0., 'vmax': 0.15,
                            'fill_vmin': False,  # fills the x < vmin with vmin
                            'xscale': None, 'yscale': None,
                            'mask': mask, 'cmap': 'Greys', 'norm': None,
                            'fancyticks': True,
                            'minorticks': True,
                            'title': {},#,{"text": r'$t-t_{merg}:$' + r'${:.1f}$ [ms]'.format((t - tmerg) * 1e3),
                                      #'fontsize': 14},
                            'sharey': False,
                            'sharex': True,  # removes angular citkscitks
                            'fontsize': 14,
                            'labelsize': 14
                            }
            #
            data_arr = d3class.get_data(it, rl, "xy", v_n)
            # print(data_arr.min(), data_arr.max()); exit(1)
            x_arr = d3class.get_data(it, rl, "xy", "x")
            y_arr = d3class.get_data(it, rl, "xy", "y")
            lapse_dic_xy = {'task': 'colormesh', 'ptype': 'cartesian', 'aspect': 1.,
                            'xarr': x_arr, "yarr": y_arr, "zarr": data_arr,
                            'position': (2, plot_x_i),  # 'title': '[{:.1f} ms]'.format(time_),
                            'cbar': {},
                            'v_n_x': 'x', 'v_n_y': 'y', 'v_n': v_n,
                            'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax, 'vmin': 0, 'vmax': 0.15,
                            'fill_vmin': False,  # fills the x < vmin with vmin
                            'xscale': None, 'yscale': None,
                            'mask': mask, 'cmap': 'Greys', 'norm': None,
                            'fancyticks': True,
                            'minorticks': True,
                            'title': {},
                            'sharey': False,
                            'sharex': False,  # removes angular citkscitks
                            'fontsize': 14,
                            'labelsize': 14
                            }
            #
            # if plot_x_i == 1:
            #     rho_dic_xy['cbar'] = {'location': 'bottom -.05 .00', 'label': r'$\rho$ [GEO]',  # 'fmt': '%.1e',
            #                           'labelsize': 14,
            #                           'fontsize': 14}
            if plot_x_i > 1:
                lapse_dic_xz['sharey'] = True
                lapse_dic_xy['sharey'] = True

            o_plot.set_plot_dics.append(lapse_dic_xz)
            o_plot.set_plot_dics.append(lapse_dic_xy)


        plot_x_i += 1




    o_plot.main()

    exit(0)




if __name__ == '__main__':
    plot_disk_2d()
    plot_den_unb__vel_z_sly4_evol()