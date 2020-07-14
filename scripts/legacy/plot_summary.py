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

# v_n + _modmtot : v_n / (M1 + M2)
# v_n + _modmchirp : v_n / [(M1 * M2) / (M1 + M2) ** (1. / 5.)] which is Mchirp
# v_n + _modq : v_n / [(M1 * M2) / (M1 + M2) ** 2] which is q
# v_n + _modq2 : v_n / [ [(M1 * M2) / (M1 + M2) ** 2] ** 2] which is q
# v_n + _modqmtot2 : v_n / [ [(M1 * M2) / (M1 + M2) ** 2] * [M1 + M2] ** 2 ]
#
plot_fit1 = True
plot_fit2 = True
plot_fit_total = True
plot_old_table = True
v_n_x = "Lambda"#"Mej_tot-geo_entropy_above_10_dev_mtot"#"Mej_tot-geo_entropy_above_10"#"Lambda"
v_n_y = "Mej_tot-geo_dev_mtotsymqmchirp"#"Mej_tot-geo_entropy_below_10_dev_mtot"#"Mej_tot-geo_entropy_below_10"#"Mej_tot-geo_6"#"Mej_tot-geo_Mchirp"
v_n_col = "q"
simlist = simulations
simlist2 = old_simulations
simtable = Paths.output + "models3.csv"#"models2.csv"
simtable2 = Paths.output + "radice2018_summary2.csv"#"radice2018_summary.csv"
deferr = 0.2
__outplotdir__ = "/data01/numrel/vsevolod.nedora/figs/all3/"
xyscales = None#"log"
prompt_bhtime = 1.5

marker_pc = 's'
marker_bh = 'o'
marker_long = 'd'
plot_legend = True

rs, rhos = [], []

def get_table_label(v_n):
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
    if v_n == "mchirp":
        return r"$\mathcal{M}$"
    if v_n == "mchirp2":
        return r"$\mathcal{M} ^2$"
    if v_n == "Mej":
        return r"M_{\rm{ej}}"
    if v_n == "symq":
        return r"$\eta$"
    if v_n == "symq2":
        return r"$\eta^2$"
    if v_n == "symqmchirp":
        return r"$\eta\mathcal{M}$"
    if v_n == "mtotsymqmchirp":
        return r"$\eta M_{\rm{tot}}\mathcal{M}$"
    if v_n == "Mej_tot-geo_entropy_below_10"or v_n == "Mej_tidal":
        return r"$M_{\rm{ej;s<10}}$" # $[10^{-2}M_{\odot}]$
    if v_n == "Mej_tot-geo_entropy_above_10" or v_n == "Mej_shocked":
        return r"$M_{\rm{ej;s>10}}$" # $[10^{-2}M_{\odot}]$
    #
    elif str(v_n).__contains__("_mult_"):
        v_n1 = v_n.split("_mult_")[0]
        v_n2 = v_n.split("_mult_")[-1]
        lbl1 = get_table_label(v_n1)
        lbl2 = get_table_label(v_n2)
        return lbl1 + r"$\times$" + lbl2
    elif str(v_n).__contains__("_dev_"):
        v_n1 = v_n.split("_dev_")[0]
        v_n2 = v_n.split("_dev_")[-1]
        lbl1 = get_table_label(v_n1)
        lbl2 = get_table_label(v_n2)
        return lbl1 + r"$/$" + lbl2

    raise NameError("Np label for v_n: {}".format(v_n))

    # if v_n == "Lambda":
    #     return r"$\tilde{\Lambda}$"
    # if v_n == "Mej_tot-geo":
    #     return r"$M_{\rm{ej}}$ $[10^{-2}M_{\odot}]$"
    # if v_n == "Mej_tot-geo_entropy_above_10" or v_n_y == "Mej_shocked":
    #     return r"$M_{\rm{ej;s>10}}$ $[10^{-2}M_{\odot}]$"
    # if v_n == "Mej_tot-geo_entropy_below_10"or v_n_y == "Mej_tidal":
    #     return r"$M_{\rm{ej;s<10}}$ $[10^{-2}M_{\odot}]$"
    # if v_n == "Mej_tot-geo" or v_n == "Mej":
    #     return r"$M_{\rm{ej}}$ $[10^{-2}M_{\odot}]$"
    # if v_n == "Mej_tot-geo_5" or v_n == "Mej5":
    #     return r"$M_{\rm{ej}} / (\eta^2)$" #  $\eta=(M_1 \times M_2) / (M_1 + M_2)^2$
    # if v_n == "Mej_tot-geo_1" or v_n == "Mej1":
    #     return r"$M_{\rm{ej}} / M_{\rm{chirp}}$" #  $\eta=(M_1 \times M_2) / (M_1 + M_2)^2$
    # if v_n == "Mej_tot-geo_6" or v_n == "Mej6":
    #     return r"$M_{\rm{ej}} \times \eta^2$" #  $\eta=(M_1 \times M_2) / (M_1 + M_2)^2$
    # if v_n == "q":
    #     return r"$M_a/M_b$"
    # return str(v_n).replace('_','\_')

def set_dic_xminxmax(v_n, dic, xarr):
    #
    if v_n == "Mej_tot-geo" or v_n == "Mej":
        dic['xmin'], dic['xmax'] = 0, 1.5
    elif v_n == "Lambda":
        dic['xmin'], dic['xmax'] = 5, 1500
    elif v_n == "Mej_tot-geo_entropy_above_10" or v_n == "Mej_shocked":
        dic['xmin'], dic['xmax'] = 0, 0.7
    elif v_n == "Mej_tot-geo_entropy_below_10" or v_n == "Mej_tidal":
        dic['xmin'], dic['xmax'] = 0, 0.5
    else:
        dic['xmin'], dic['xmax'] = np.array(xarr).min(), np.array(xarr).max()

        Printcolor.yellow("xlimits are not set for v_n_x:{}".format(v_n))
    return dic

def set_dic_yminymax(v_n, dic, yarr):
    #
    if v_n == "Mej_tot-geo" or v_n == "Mej":
        dic['ymin'], dic['ymax'] = 0, 1.5
    elif v_n == "Mej_tot-geo_2" or v_n == "Mej2":
        dic['ymin'], dic['ymax'] = 0, 2
    elif v_n == "Mej_tot-geo_1" or v_n == "Mej1":
        dic['ymin'], dic['ymax'] = 0, 0.7
    elif v_n == "Mej_tot-geo_3" or v_n == "Mej3":
        dic['ymin'], dic['ymax'] = 0, 2
    elif v_n == "Mej_tot-geo_4" or v_n == "Mej4":
        dic['ymin'], dic['ymax'] = 0, .75
    elif v_n == "Mej_tot-geo_5" or v_n == "Mej5":
        dic['ymin'], dic['ymax'] = 0, 10.
    elif v_n == "Mej_tot-geo_6" or v_n == "Mej6":
        dic['ymin'], dic['ymax'] = 0, 0.06
    elif v_n == "Mej_tot-geo_entropy_above_10" or v_n == "Mej_shocked":
        dic['ymin'], dic['ymax'] = 0, 0.7
    elif v_n == "Mej_tot-geo_entropy_below_10" or v_n == "Mej_tidal":
        dic['ymin'], dic['ymax'] = 0, 0.5
    else:
        dic['ymin'], dic['ymax'] = np.array(yarr).min(), np.array(yarr).max()
        Printcolor.yellow("xlimits are not set for v_n_x:{}".format(v_n))
    return dic

''' --------------------------------------------------------------- '''

total_x = [] # for fits
total_y = [] # for fits

Printcolor.blue("Collecting Data")
o_tbl = GET_PAR_FROM_TABLE()
o_tbl.set_intable = simtable
o_tbl.load_table()
data = {}
all_x = []
all_y = []
all_col = []
all_marker = []
for eos in simlist.keys():
    data[eos] = {}
    for usim in simlist[eos].keys():
        data[eos][usim] = {}
        sims = simlist[eos][usim]
        print("\t{} [{}]".format(usim, len(sims)))
        x, x1, x2 = o_tbl.get_par_with_error(sims, v_n_x, deferr=deferr)
        y, y1, y2 = o_tbl.get_par_with_error(sims, v_n_y, deferr=deferr)
        col = o_tbl.get_par(sims[0], v_n_col)
        data[eos][usim]["x"] = x
        data[eos][usim]['x1'] = x1
        data[eos][usim]['x2'] = x2
        data[eos][usim]['y'] = y
        data[eos][usim]['y1'] = y1
        data[eos][usim]['y2'] = y2
        data[eos][usim]['col'] = col
        all_x.append(x)
        all_y.append(y)
        all_col.append(col)
        isbh, ispromtcoll = o_tbl.get_is_prompt_coll(sims, delta_t=prompt_bhtime, v_n_tmerg="tmerg_r")
        data[eos][usim]["isprompt"] = ispromtcoll
        data[eos][usim]["isbh"] = isbh
        if isbh and not ispromtcoll:
            marker = marker_bh
        elif isbh and ispromtcoll:
            marker = marker_pc
        else:
            marker = marker_long
        all_marker.append(marker)
        data[eos][usim]["marker"] = marker
data["allx"] = np.array(all_x)
data["ally"] = np.array(all_y)
data["allcol"] = np.array(all_col)
data["allmarker"] = all_marker

# for fits
for eos in simlist.keys():
    for usim in simlist[eos].keys():
        if not data[eos][usim]["isprompt"]:
            total_x.append(data[eos][usim]["x"])
            total_y.append(data[eos][usim]["y"])
#
Printcolor.green("Data is collected")
Printcolor.blue("Plotting Data")
#
def make_plot_name(v_n_x, v_n_y, v_n_col, do_plot_old_table):
    figname = ''
    figname = figname + v_n_x + '_'
    figname = figname + v_n_y + '_'
    figname = figname + v_n_col + '_'
    if do_plot_old_table:
        figname = figname + '_InclOldTbl'
    figname = figname + '.png'
    return figname
figname = make_plot_name(v_n_x, v_n_y, v_n_col, False)
#
def get_custom_colormap(cmap_name = 'newCmap', n_bin=8):

    from matplotlib.colors import LinearSegmentedColormap
    # colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    colors=([(1, 0, 0), (0, 0, 1)],[1., 1.8])
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)
    return cm
#
o_plot = PLOT_MANY_TASKS()
o_plot.gen_set["figdir"] = __outplotdir__
o_plot.gen_set["type"] = "cartesian"
o_plot.gen_set["figsize"] = (4.2, 3.6)  # <->, |]
o_plot.gen_set["figname"] = figname
o_plot.gen_set["sharex"] = True
o_plot.gen_set["sharey"] = False
o_plot.gen_set["subplots_adjust_h"] = 0.0
o_plot.gen_set["subplots_adjust_w"] = 0.0
o_plot.set_plot_dics = []
#
assert len(data["allx"]) == len(data["ally"])
assert len(data["ally"]) == len(data["allcol"])
assert len(data["allcol"]) > 0

if v_n_y.__contains__("Mej"):
    data["ally"] =  data["ally"] * 1e2
if v_n_x.__contains__("Mej"):
    data["allx"] =  data["allx"] * 1e2
#

    # if  eos == "BLh" and u_sim == simulations2[eos][q].keys()[-1]:
    #     print('-0--------------------')
if plot_fit1:
    total_x1, total_y1 = UTILS.x_y_z_sort(total_x, total_y)
    # fit_polynomial(x, y, order, depth, new_x=np.empty(0, ), print_formula=True):
    Printcolor.blue("New data fit")
    if xyscales == "log":
        fit_x1, fit_y1 = UTILS.fit_polynomial(total_x, total_y, order=1, depth=100)
    else:
        fit_x1, fit_y1 = UTILS.fit_polynomial(total_x, total_y, order=1, depth=100)
    #
    if v_n_y.__contains__("Mej"):
        fit_y1 = fit_y1 * 1e2
    if v_n_x.__contains__("Mej"):
        fit_x1 = fit_x1 * 1e2
    # print(fit_x, fit_y)
    linear_fit = {
        'task': 'line', 'ptype': 'cartesian',
        'position': (1, 1),
        'xarr': fit_x1, "yarr": fit_y1,
        'xlabel': None, "ylabel": None,
        'label': "New Data",
        'ls': '-', 'color': 'red', 'lw': 1., 'alpha': 0.8, 'ds': 'default',
        'sharey': False,
        'sharex': False,  # removes angular citkscitks
        'fontsize': 14,
        'labelsize': 14,
        # 'text':{'x':1., 'y':1., 'text':'my_text', 'fs':14, 'color':'black','horal':True}
    }
    o_plot.set_plot_dics.append(linear_fit)

    r, rho = stats.spearmanr(total_x1, total_y1)
    print("r: {} rho: {}".format(r, rho))
    rs.append(r)
    rhos.append(rho)
    text_dic = {
        'task': 'text', 'ptype': 'cartesian',
        'position': (1, 1),
        'x': 0.45, 'y': 0.9,
        'text': r'New: $r:{:.2f}$ $\rho:{:.2e}$'.format(r, rho),
        'fs': 10, 'color': 'black', 'horizontalalignment': "left",
        'transform': True
    }
    o_plot.set_plot_dics.append(text_dic)

dic = {
        'task': 'scatter', 'ptype': 'cartesian',  # 'aspect': 1.,
        'xarr': data["allx"], "yarr": data["ally"], "zarr": data["allcol"],
        'position': (1, 1),  # 'title': '[{:.1f} ms]'.format(time_),
        'cbar': {},
        'v_n_x': v_n_x, 'v_n_y': v_n_y, 'v_n': v_n_col,
        'xlabel': get_table_label(v_n_x), "ylabel": get_table_label(v_n_y),
        'xmin': 300, 'xmax': 900, 'ymin': None, 'ymax': None, 'vmin': 1.0, 'vmax': 1.9,
        'fill_vmin': False,  # fills the x < vmin with vmin
        'xscale': None, 'yscale': None,
        'cmap': 'tab10', 'norm': None, 'ms': 60, 'markers': data["allmarker"], 'alpha': 0.7, "edgecolors": None,
        'tick_params': {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False,  # "tick1On":True, "tick2On":True,
                        "labelsize": 12,
                        "direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        'yaxiscolor': {'bottom': 'black', 'top': 'black', 'right': 'black', 'left': 'black'},
        'minorticks': True,
        'title': {},  # {"text": eos, "fontsize": 12},
        'label': None,
        'legend': {},
        'sharey': False,
        'sharex': False,  # removes angular citkscitks
        'fontsize': 14,
        'labelsize': 14
    }
dic = set_dic_xminxmax(v_n_x, dic, data["allx"])
dic = set_dic_yminymax(v_n_y, dic, data["ally"])

dic['cbar'] = {'location': 'right .03 .0', 'label': Labels.labels(v_n_col),  # 'fmt': '%.1f',
                       'labelsize': 14, 'fontsize': 14}
o_plot.set_plot_dics.append(dic)

''' ------------------------------------------------------------------------------------------ '''

if plot_old_table:


    translation = {"Mej_tot-geo":"Mej",
                   "Lambda":"Lambda",
                   "Mej_tot-geo_Mchirp":"Mej_Mchirp",
                   "Mej_tot-geo_1":"Mej1",
                   "Mej_tot-geo_2":"Mej2",
                   "Mej_tot-geo_3": "Mej3",
                   "Mej_tot-geo_4": "Mej4",
                   "Mej_tot-geo_5": "Mej5",
                   "Mej_tot-geo_6": "Mej6",
                   "tcoll_gw":"tcoll",
                   "Mej_tot-geo_entropy_above_10":"Mej_shocked",
                   "Mej_tot-geo_entropy_below_10":"Mej_tidal",
                   "Mej_tot-geo_entropy_above_10_dev_mtot":"Mej_shocked_dev_mtot",
                   "Mej_tot-geo_entropy_below_10_dev_mtot":"Mej_tidal_dev_mtot",
                   "Mej_tot-geo_dev_mtot":"Mej_dev_mtot",
                   "Mej_tot-geo_dev_mtot2":"Mej_dev_mtot2",
                   "Mej_tot-geo_mult_mtot":"Mej_mult_mtot",
                   "Mej_tot-geo_mult_mtot2":"Mej_mult_mtot2",
                   "Mej_tot-geo_dev_symq":"Mej_dev_symq",
                   "Mej_tot-geo_dev_symq2":"Mej_dev_symq2",
                   "Mej_tot-geo_mult_symq":"Mej_mult_symq",
                   "Mej_tot-geo_mult_symq2":"Mej_mult_symq2",
                   "Mej_tot-geo_dev_mchirp":"Mej_dev_mchirp",
                   "Mej_tot-geo_dev_mchirp2":"Mej_dev_mchirp2",
                   "Mej_tot-geo_mult_mchirp":"Mej_mult_mchirp",
                   "Mej_tot-geo_dev_symqmchirp":"Mej_dev_symqmchirp",
                   "Mej_tot-geo_dev_mtotsymqmchirp":"Mej_dev_mtotsymqmchirp"}

    v_n_x = translation[v_n_x]
    v_n_y = translation[v_n_y]

    total_x2 = []  # for fits
    total_y2 = []  # for fits

    Printcolor.blue("Collecting Data")
    o_tbl = GET_PAR_FROM_TABLE()
    o_tbl.set_intable = simtable2
    o_tbl.load_table()
    data2 = {}
    all_x = []
    all_y = []
    all_col = []
    all_marker = []
    for eos in simlist2.keys():
        data2[eos] = {}
        for usim in simlist2[eos].keys():
            data2[eos][usim] = {}
            sims = simlist2[eos][usim]
            print("\t{} [{}]".format(usim, len(sims)))
            x, x1, x2 = o_tbl.get_par_with_error(sims, v_n_x, deferr=deferr)
            y, y1, y2 = o_tbl.get_par_with_error(sims, v_n_y, deferr=deferr)
            # col = o_tbl.get_par(sims[0], v_n_col)
            col = o_tbl.get_par(sims[0], v_n_col)

            # print(col); exit(1)
            data2[eos][usim]["x"] = x
            data2[eos][usim]['x1'] = x1
            data2[eos][usim]['x2'] = x2
            data2[eos][usim]['y'] = y
            data2[eos][usim]['y1'] = y1
            data2[eos][usim]['y2'] = y2
            data2[eos][usim]['col'] = col
            all_x.append(x)
            all_y.append(y)
            all_col.append(col)
            isbh, ispromtcoll = o_tbl.get_is_prompt_coll(sims, delta_t=3., v_n_tcoll="tcoll", v_n_tmerg="tmerg_r")
            data2[eos][usim]["isprompt"] = ispromtcoll
            data2[eos][usim]["isbh"] = isbh
            if isbh and not ispromtcoll:
                marker = marker_bh
            elif isbh and ispromtcoll:
                marker = marker_pc
            else:
                marker = marker_long
            all_marker.append(marker)
            data2[eos][usim]["marker"] = marker
    data2["allx"] = np.array(all_x)
    data2["ally"] = np.array(all_y)
    data2["allcol"] = np.array(all_col)
    data2["allmarker"] = all_marker
    #

    #
    Printcolor.green("Data is collected")
    Printcolor.blue("Plotting Data")
    #
    def make_plot_name(v_n_x, v_n_y, v_n_col, do_plot_old_table):
        figname = ''
        figname = figname + v_n_x + '_'
        figname = figname + v_n_y + '_'
        figname = figname + v_n_col + '_'
        if do_plot_old_table:
            figname = figname + '_InclOldTbl'
        figname = figname + '.png'
        return figname
    figname = make_plot_name(v_n_x, v_n_y, v_n_col, False)
    #
    def get_custom_colormap(cmap_name = 'newCmap', n_bin=8):

        from matplotlib.colors import LinearSegmentedColormap
        # colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        colors=([(1, 0, 0), (0, 0, 1)],[1., 1.8])
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)
        return cm
    #
    #
    assert len(data2["allx"]) == len(data2["ally"])
    assert len(data2["ally"]) == len(data2["allcol"])
    assert len(data2["allcol"]) > 0

    if v_n_y.__contains__("Mej"):
        data2["ally"] = data2["ally"] * 1e2
    if v_n_x.__contains__("Mej"):
        data2["allx"] = data2["allx"] * 1e2
    #
    # if plot_legend:
    #     x = -1.
    #     y = -1.
    #     marker_dic_lr = {
    #         'task': 'line', 'ptype': 'cartesian',
    #         'position': (1, 1),
    #         'xarr': [x], "yarr": [y],
    #         'xlabel': None, "ylabel": None,
    #         'label': "BH formation",
    #         'marker': marker_bh, 'color': 'gray', 'ms': 10., 'alpha': 0.4,
    #         'sharey': False,
    #         'sharex': False,  # removes angular citkscitks
    #         'fontsize': 14,
    #         'labelsize': 14
    #     }
    #
    #     o_plot.set_plot_dics.append(marker_dic_lr)
    #     marker_dic_lr = {
    #         'task': 'line', 'ptype': 'cartesian',
    #         'position': (1, 1),
    #         'xarr': [x], "yarr": [y],
    #         'xlabel': None, "ylabel": None,
    #         'label': "Prompt collapse",
    #         'marker': marker_pc, 'color': 'gray', 'ms': 10., 'alpha': 0.4,
    #         'sharey': False,
    #         'sharex': False,  # removes angular citkscitks
    #         'fontsize': 14,
    #         'labelsize': 14
    #     }
    #
    #     o_plot.set_plot_dics.append(marker_dic_lr)
    #     marker_dic_lr = {
    #         'task': 'line', 'ptype': 'cartesian',
    #         'position': (1, 1),
    #         'xarr': [x], "yarr": [y],
    #         'xlabel': None, "ylabel": None,
    #         'label': "Long lived",
    #         'marker': marker_long, 'color': 'gray', 'ms': 10., 'alpha': 0.4,
    #         'sharey': False,
    #         'sharex': False,  # removes angular citkscitks
    #         'fontsize': 14,
    #         'labelsize': 14
    #     }
    #     marker_dic_lr['legend'] = {'loc': 'upper left', 'ncol': 1, 'shadow': False, 'framealpha': 0.,
    #                                'borderaxespad': 0., 'fontsize': 11}
    #     o_plot.set_plot_dics.append(marker_dic_lr)
    #     # if  eos == "BLh" and u_sim == simulations2[eos][q].keys()[-1]:
    #     #     print('-0--------------------')

    # print(data2["ally"]); exit(1)

    dic2 = {
            'task': 'scatter', 'ptype': 'cartesian',  # 'aspect': 1.,
            'xarr': data2["allx"], "yarr": data2["ally"], "zarr": data2["allcol"],
            'position': (1, 1),  # 'title': '[{:.1f} ms]'.format(time_),
            'cbar': {},
            'v_n_x': v_n_x, 'v_n_y': v_n_y, 'v_n': v_n_col,
            'xlabel': get_table_label(v_n_x), "ylabel": get_table_label(v_n_y),
            'xmin': 300, 'xmax': 900, 'ymin': 0.03, 'ymax': 0.5, 'vmin': 1.0, 'vmax': 1.9,
            'fill_vmin': False,  # fills the x < vmin with vmin
            'xscale': None, 'yscale': None,
            'cmap': 'tab10', 'norm': None, 'ms': 40, 'marker': '*', 'alpha': 0.4, "edgecolors": None, #data2["allmarker"]
            'tick_params': {"axis": 'both', "which": 'both', "labelleft": True,
                            "labelright": False,  # "tick1On":True, "tick2On":True,
                            "labelsize": 12,
                            "direction": 'in',
                            "bottom": True, "top": True, "left": True, "right": True},
            'yaxiscolor': {'bottom': 'black', 'top': 'black', 'right': 'black', 'left': 'black'},
            'minorticks': True,
            'title': {},  # {"text": eos, "fontsize": 12},
            'label': None,
            'legend': {},
            'sharey': False,
            'sharex': False,  # removes angular citkscitks
            'fontsize': 14,
            'labelsize': 14
        }
    dic2 = set_dic_xminxmax(v_n_x, dic2, data2["allx"])
    dic2 = set_dic_yminymax(v_n_y, dic2, data2["ally"])

    dic2['cbar'] = {'location': 'right .03 .0', 'label': Labels.labels(v_n_col),  # 'fmt': '%.1f',
                           'labelsize': 14, 'fontsize': 14}

    if xyscales == "log":
        dic2["xscale"] = "log"
        dic2["yscale"] = "log"
        dic2["xmin"], dic2["xmax"] = 5e-3, 1e0
        dic2["ymin"], dic2["ymax"] = 5e-3, 1e0
    o_plot.set_plot_dics.append(dic2)

    # for fits
    for eos in simlist2.keys():
        for usim in simlist2[eos].keys():
            if not data2[eos][usim]["isprompt"]:
                total_x2.append(data2[eos][usim]["x"])
                total_y2.append(data2[eos][usim]["y"])

    if plot_fit2:
        total_x2, total_y2 = UTILS.x_y_z_sort(total_x2, total_y2)
        # fit_polynomial(x, y, order, depth, new_x=np.empty(0, ), print_formula=True):
        Printcolor.blue("Old data fit")
        if xyscales == "log":
            fit_x2, fit_y2 = UTILS.fit_polynomial(total_x2, total_y2, order=1, depth=100)
        else:
            fit_x2, fit_y2 = UTILS.fit_polynomial(total_x2, total_y2, order=1, depth=100)
        #
        if v_n_y.__contains__("Mej"):
            fit_y2 = fit_y2 * 1e2
        if v_n_x.__contains__("Mej"):
            fit_x2 = fit_x2 * 1e2
        # print(fit_x, fit_y)
        linear_fit = {
            'task': 'line', 'ptype': 'cartesian',
            'position': (1, 1),
            'xarr': fit_x2, "yarr": fit_y2,
            'xlabel': None, "ylabel": None,
            'label': "Old Data",
            'ls': '--', 'color': 'blue', 'lw': 1., 'alpha': 0.8, 'ds': 'default',
            'sharey': False,
            'sharex': False,  # removes angular citkscitks
            'fontsize': 14,
            'labelsize': 14,
            # 'text':{'x':1., 'y':1., 'text':'my_text', 'fs':14, 'color':'black','horal':True}
        }
        o_plot.set_plot_dics.append(linear_fit)

        r, rho = stats.spearmanr(total_x2, total_y2)
        rs.append(r)
        rhos.append(rho)
        print("r: {} rho: {}".format(r, rho))
        text_dic = {
            'task': 'text', 'ptype': 'cartesian',
            'position': (1, 1),
            'x': 0.45, 'y': 0.8,
            'text': r'Old: $r:{:.2f}$ $\rho:{:.2e}$'.format(r, rho),
            'fs': 10, 'color': 'black', 'horizontalalignment': "left",
            'transform': True
        }
        o_plot.set_plot_dics.append(text_dic)


if plot_fit2 and plot_old_table:
    total_x3, total_y3 = np.append(total_x, total_x2), np.append(total_y, total_y2)
    # print(len(total_x3)); exit(1)
    total_x3, total_y3 = UTILS.x_y_z_sort(total_x3, total_y3)
    # fit_polynomial(x, y, order, depth, new_x=np.empty(0, ), print_formula=True):

    Printcolor.blue("All data fit")
    if xyscales == "log":
        fit_x3, fit_y3 = UTILS.fit_polynomial(total_x3, total_y3, order=1, depth=100)
    else:
        fit_x3, fit_y3 = UTILS.fit_polynomial(total_x3, total_y3, order=1, depth=100)
    #
    if v_n_y.__contains__("Mej"):
        fit_y3 = fit_y3 * 1e2
    if v_n_x.__contains__("Mej"):
        fit_x3 = fit_x3 * 1e2
    # print(fit_x, fit_y)
    linear_fit = {
        'task': 'line', 'ptype': 'cartesian',
        'position': (1, 1),
        'xarr': fit_x3, "yarr": fit_y3,
        'xlabel': None, "ylabel": None,
        'label': "All Data",
        'ls': ':', 'color': 'black', 'lw': 1., 'alpha': 1., 'ds': 'default',
        'sharey': False,
        'sharex': False,  # removes angular citkscitks
        'fontsize': 14,
        'labelsize': 14,
        # 'text':{'x':1., 'y':1., 'text':'my_text', 'fs':14, 'color':'black','horal':True}
    }
    o_plot.set_plot_dics.append(linear_fit)

    r, rho = stats.spearmanr(total_x3, total_y3)
    print("r: {} rho: {}".format(r, rho))
    text_dic = {
        'task': 'text', 'ptype': 'cartesian',
        'position': (1, 1),
        'x': 0.45, 'y': 0.7,
        'text': r'All: $r:{:.2f}$ $\rho:{:.2e}$'.format(r, rho),
        'fs': 10, 'color': 'black', 'horizontalalignment': "left",
        'transform': True
    }
    o_plot.set_plot_dics.append(text_dic)
    rs.append(r)
    rhos.append(rho)


if plot_legend:
    x = -1.
    y = -1.
    marker_dic_lr = {
        'task': 'line', 'ptype': 'cartesian',
        'position': (1, 1),
        'xarr': [x], "yarr": [y],
        'xlabel': None, "ylabel": None,
        'label': "BH formation",
        'marker': marker_bh, 'color': 'gray', 'ms': 8., 'alpha': 0.4,
        'sharey': False,
        'sharex': False,  # removes angular citkscitks
        'fontsize': 14,
        'labelsize': 14
    }

    o_plot.set_plot_dics.append(marker_dic_lr)
    marker_dic_lr = {
        'task': 'line', 'ptype': 'cartesian',
        'position': (1, 1),
        'xarr': [x], "yarr": [y],
        'xlabel': None, "ylabel": None,
        'label': "Prompt collapse",
        'marker': marker_pc, 'color': 'gray', 'ms': 8., 'alpha': 0.4,
        'sharey': False,
        'sharex': False,  # removes angular citkscitks
        'fontsize': 14,
        'labelsize': 14
    }

    o_plot.set_plot_dics.append(marker_dic_lr)
    marker_dic_lr = {
        'task': 'line', 'ptype': 'cartesian',
        'position': (1, 1),
        'xarr': [x], "yarr": [y],
        'xlabel': None, "ylabel": None,
        'label': "Long lived",
        'marker': marker_long, 'color': 'gray', 'ms': 8., 'alpha': 0.4,
        'sharey': False,
        'sharex': False,  # removes angular citkscitks
        'fontsize': 14,
        'labelsize': 14
    }
    marker_dic_lr['legend'] = {'loc': 'upper left', 'ncol': 1, 'shadow': False, 'framealpha': 0.,
                               'borderaxespad': 0., 'fontsize': 11}
    o_plot.set_plot_dics.append(marker_dic_lr)

print("\n")
Printcolor.blue("Spearman's Rank Coefficients for: ")
Printcolor.green("v_n_x: {}".format(v_n_x))
Printcolor.green("v_n_y: {}".format(v_n_y))
Printcolor.blue("New data: ", comma=True)
Printcolor.green("{:.2f}".format(rs[0]))
Printcolor.blue("Old data: ", comma=True)
Printcolor.green("{:.2f}".format(rs[1]))
Printcolor.blue("All data: ", comma=True)
Printcolor.green("{:.2f}".format(rs[2]))

o_plot.main()
exit(0)






















__outplotdir__ = "/data01/numrel/vsevolod.nedora/figs/all3/"

#/data01/numrel/vsevolod.nedora/bns_ppr_tools
# import imp
# LOAD_INIT_DATA = imp.load_source("LOAD_INIT_DATA", "/data01/numrel/vsevolod.nedora/bns_ppr_tools/preanalysis.py")
# LOAD_INIT_DATA.

def __get_value(o_init, o_par, det=None, mask=None, v_n=None):

    if v_n in o_init.list_v_ns and mask == None:
        value = o_init.get_par(v_n)
    elif not v_n in o_init.list_v_ns and mask == None:
        value = o_par.get_par(v_n)
    elif v_n == "Mej_tot_scaled":
        ma = __get_value(o_init, o_par, None, None, "Mb1")
        mb = __get_value(o_init, o_par, None, None, "Mb2")
        mej = __get_value(o_init, o_par, det, mask, "Mej_tot")
        return mej / (ma + mb)
    elif v_n == "Mej_tot_scaled2":
        # M1 * M2 / (M1 + M2) ^ 2
        ma = __get_value(o_init, o_par, None, None, "Mb1")
        mb = __get_value(o_init, o_par, None, None, "Mb2")
        eta = ma * mb / (ma + mb) ** 2
        mej = __get_value(o_init, o_par, det, mask, "Mej_tot")
        return mej / (eta * (ma + mb))

    elif not v_n in o_init.list_v_ns and mask != None:
        value = o_par.get_outflow_par(det, mask, v_n)
    else:
        raise NameError("unrecognized: v_n_x:{} mask_x:{} det:{} combination"
                        .format(v_n, mask, det))
    if value == None or np.isinf(value) or np.isnan(value):
        raise ValueError("sim: {} det:{} mask:{} v_n:{} --> value:{} wrong!"
                         .format(o_par.sim,det,mask,v_n, value))
    return value


def __get_val_err(sims, o_inits, o_pars, v_n, det=0, mask="geo", error=0.2):

    if v_n == "nsims":
        return len(sims), len(sims), len(sims)
    elif v_n == "pizzaeos":
        pizza_eos = ''
        for sim, o_init, o_par in zip(sims, o_inits, o_pars):
            _pizza_eos = o_init.get_par("pizza_eos")
            if pizza_eos != '' and pizza_eos != _pizza_eos:
                raise NameError("sim:{} pizza_eos:{} \n sim:{} pizza_eos: {} \n MISMATCH"
                                .format(sim, pizza_eos, sims[0], _pizza_eos))
        pizza_eos = _pizza_eos
        return pizza_eos, pizza_eos, pizza_eos
    if len(sims) == 0:
        raise ValueError("no simualtions passed")
    _resols, _values = [], []
    assert len(sims) == len(o_inits)
    assert len(sims) == len(o_pars)
    for sim, o_init, o_par in zip(sims, o_inits, o_pars):
        _val = __get_value(o_init, o_par, det, mask, v_n)
        # print(sim, _val)
        _res = "fuck"
        for res in resolutions.keys():
            if sim.__contains__(res):
                _res = res
                break
        if _res == "fuck":
            raise NameError("fuck")
        _resols.append(resolutions[_res])
        _values.append(_val)
    if len(sims) == 1:
        return _values[0], _values[0] - error * _values[0], _values[0] + error * _values[0]
    elif len(sims) == 2:
        delta = np.abs(_values[0] - _values[1])
        if _resols[0] < _resols[1]:
            return _values[0], _values[0] - delta, _values[0] + delta
        else:
            return _values[1], _values[1] - delta, _values[1] + delta
    elif len(sims) == 3:
        _resols_, _values_ = UTILS.x_y_z_sort(_resols, _values) # 123, 185, 236
        delta1 = np.abs(_values_[0] - _values_[1])
        delta2 = np.abs(_values_[1] - _values_[2])
        # print(_values, _values_); exit(0)
        return _values_[1], _values_[1] - delta1, _values_[1] + delta2
    else:
        raise ValueError("Too many simulations")

def __get_is_prompt_coll(sims, o_inits, o_pars, delta_t = 3.):

    isprompt = False
    isbh = False
    for sim, o_init, o_par in zip(sims, o_inits, o_pars):
        tcoll = o_par.get_par("tcoll_gw")
        if np.isinf(tcoll):
            pass
        else:
            isbh = True
            tmerg = o_par.get_par("tmerg")
            assert tcoll > tmerg
            if float(tcoll - tmerg) < delta_t * 1e-3:
                isprompt = True

    return isbh, isprompt

def __get_custom_descrete_colormap(n):
    # n = 5
    import matplotlib.colors as col
    from_list = col.LinearSegmentedColormap.from_list
    cm = from_list(None, plt.cm.Set1(range(0, n)), n)
    x = np.arange(99)
    y = x % 11
    z = x % n
    return cm

v_n_x = "Lambda"
v_n_y = "Ye_ave"
v_n_col = "q"
det = 0
do_plot_linear_fit = True
do_plot_promptcoll = True
do_plot_bh = True
do_plot_error_bar_y = True
do_plot_error_bar_x = False
do_plot_old_table = True
do_plot_annotations = False
mask_x, mask_y, mask_col = None, "geo", None  # geo_entropy_above_10
data2 = {}
error = 0.2  # in * 100 percent
delta_t_prompt = 2.  # ms


''' --- collect data for table 1 --- '''

old_data = {}
if do_plot_old_table:
    #
    if mask_x != None and mask_x != "geo":
        raise NameError("old table des not contain data for mask_x: {}".format(mask_x))
    if mask_y != None and mask_y != "geo":
        raise NameError("old table des not contain data for mask_x: {}".format(mask_y))
    if mask_col != None and mask_col != "geo":
        raise NameError("old table des not contain data for mask_x: {}".format(mask_col))
    #
    new_old_dic = {'Mej_tot': "Mej",
                   "Lambda": "Lambda",
                   "vel_inf_ave": "vej",
                   "Ye_ave": "Yeej"}

    old_tbl = ALL_SIMULATIONS_TABLE()
    old_tbl.set_list_neut = ["LK", "M0"]
    old_tbl.set_list_vis = ["L5", "L25", "L50"]
    old_tbl.set_list_eos.append("BHBlp")
    old_tbl.set_intable = Paths.output + "radice2018_summary.csv"
    old_tbl.load_input_data()
    old_all_x = []
    old_all_y = []
    old_all_col = []
    for run in old_tbl.table:
        sim = run['name']
        old_data[sim] = {}
        if not sim.__contains__("HR") \
                and not sim.__contains__("OldM0") \
                and not sim.__contains__("LR") \
                and not sim.__contains__("L5") \
                and not sim.__contains__("L25") \
                and not sim.__contains__("L50"):
            x = float(run[new_old_dic[v_n_x]])
            y = float(run[new_old_dic[v_n_y]])
            col = "gray"
            old_all_col.append(col)
            old_all_x.append(x)
            old_all_y.append(y)
            old_data[sim][v_n_x] = x
            old_data[sim][v_n_y] = y

    Printcolor.green("old data is collected")
    old_all_x = np.array(old_all_x)
    old_all_y = np.array(old_all_y)

''' --- --- --- '''
new_data = {}

# collect old data
old_data = {}
if do_plot_old_table:
    #
    if mask_x != None and mask_x != "geo":
        raise NameError("old table des not contain data for mask_x: {}".format(mask_x))
    if mask_y != None and mask_y != "geo":
        raise NameError("old table des not contain data for mask_x: {}".format(mask_y))
    if mask_col != None and mask_col != "geo":
        raise NameError("old table des not contain data for mask_x: {}".format(mask_col))
    #
    new_old_dic = {'Mej_tot': "Mej",
                   "Lambda": "Lambda",
                   "vel_inf_ave": "vej",
                   "Ye_ave": "Yeej"}
    old_tbl = ALL_SIMULATIONS_TABLE()
    old_tbl.set_list_neut = ["LK", "M0"]
    old_tbl.set_list_vis = ["L5", "L25", "L50"]
    old_tbl.set_list_eos.append("BHBlp")
    old_tbl.set_intable = Paths.output + "radice2018_summary.csv"
    old_tbl.load_input_data()
    old_all_x = []
    old_all_y = []
    old_all_col = []
    for run in old_tbl.table:
        sim = run['name']
        old_data[sim] = {}
        if not sim.__contains__("HR") \
                and not sim.__contains__("OldM0") \
                and not sim.__contains__("LR") \
                and not sim.__contains__("L5") \
                and not sim.__contains__("L25") \
                and not sim.__contains__("L50"):
            x = float(run[new_old_dic[v_n_x]])
            y = float(run[new_old_dic[v_n_y]])
            col = "gray"
            old_all_col.append(col)
            old_all_x.append(x)
            old_all_y.append(y)
            old_data[sim][v_n_x] = x
            old_data[sim][v_n_y] = y

    Printcolor.green("old data is collected")
    old_all_x = np.array(old_all_x)
    old_all_y = np.array(old_all_y)

# exit(1)
# collect data
for eos in simulations.keys():
    data2[eos] = {}
    for q in simulations[eos]:
        data2[eos][q] = {}
        for u_sim in simulations[eos][q]:
            data2[eos][q][u_sim] = {}
            sims = simulations[eos][q][u_sim]
            o_inits = [LOAD_INIT_DATA(sim) for sim in sims]
            o_pars = [ADD_METHODS_ALL_PAR(sim) for sim in sims]
            x_coord, x_err1, x_err2 = __get_val_err(sims, o_inits, o_pars, v_n_x, det, mask_x, error)
            y_coord, y_err1, y_err2 = __get_val_err(sims, o_inits, o_pars, v_n_y, det, mask_y, error)
            col_coord, col_err1, col_err2 = __get_val_err(sims, o_inits, o_pars, v_n_col, det, mask_col, error)
            data2[eos][q][u_sim]["lserr"] = len(sims)
            data2[eos][q][u_sim]["x"] = x_coord
            data2[eos][q][u_sim]["xe1"] = x_err1
            data2[eos][q][u_sim]["xe2"] = x_err2
            data2[eos][q][u_sim]["y"] = y_coord
            data2[eos][q][u_sim]["ye1"] = y_err1
            data2[eos][q][u_sim]["ye2"] = y_err2
            data2[eos][q][u_sim]["c"] = col_coord
            data2[eos][q][u_sim]["ce1"] = col_err1
            data2[eos][q][u_sim]["ce2"] = col_err2
            #
            isbh, ispromtcoll = __get_is_prompt_coll(sims, o_inits, o_pars, delta_t=delta_t_prompt)
            data2[eos][q][u_sim]["isprompt"] = ispromtcoll
            data2[eos][q][u_sim]["isbh"] = isbh
            if isbh and not ispromtcoll:
                marker = 'o'
            elif isbh and ispromtcoll:
                marker = 's'
            else:
                marker = 'd'
            data2[eos][q][u_sim]["marker"] = marker
            #
            pizzaeos = False
            if eos == "SFHo":
                pizzaeos, _, _ = __get_val_err(sims, o_inits, o_pars, "pizzaeos")
                if pizzaeos.__contains__("2019"):
                    _pizzaeos = True
                    data2[eos][q][u_sim]['pizza2019'] = True
                else:
                    _pizzaeos = False
                    data2[eos][q][u_sim]['pizza2019'] = False
            #
            Printcolor.print_colored_string([u_sim, "({})".format(len(sims)),
                                             "x:[", "{:.1f}".format(x_coord),
                                             "v:", "{:.1f}".format(x_err1),
                                             "^:", "{:.1f}".format(x_err2),
                                             "|",
                                             "y:", "{:.5f}".format(y_coord),
                                             "v:", "{:.5f}".format(y_err1),
                                             "^:",
                                             "{:.5f}".format(y_err2),
                                             "] col: {} BH:".format(col_coord),
                                             "{}".format(ispromtcoll),
                                             "pizza2019:",
                                             "{}".format(pizzaeos)],
                                            ["blue", "green", "blue", "green", "blue", "green",
                                             "blue", "green", "yellow", "blue", "green", "blue",
                                             "green", "blue", "green", "blue", "green", "blue", "green"])

            # Printcolor.blue("Processing {} ({} sims) x:[{:.1f}, v:{:.1f} ^{:.1f}] y:[{:.5f}, v{:.5f} ^{:.5f}] col:{:.1f}"
            #                 .format(u_sim, len(sims), x_coord, x_err1, x_err2, y_coord, y_err1, y_err2, col_coord))
Printcolor.green("Data is collaected")

# FIT
print(" =============================== ")
all_x = []
all_y = []
for eos in data2.keys():
    for q in data2[eos].keys():
        for u_sim in data2[eos][q].keys():
            ispc = data2[eos][q][u_sim]["isprompt"]
            if not ispc:
                all_x.append(data2[eos][q][u_sim]["x"])
                all_y.append(data2[eos][q][u_sim]['y'])
all_x = np.array(all_x)
all_y = np.array(all_y)

# print(all_x)
all_x, all_y = UTILS.x_y_z_sort(all_x, all_y)
# print(all_x);
print("_log(lambda) as x")
UTILS.fit_polynomial(np.log10(all_x), all_y, 1, 100)
print("lamda as x")
fit_x, fit_y = UTILS.fit_polynomial(all_x, all_y, 1, 100)
# print(fit_x); exit(1)
print("ave: {}".format(np.sum(all_y) / len(all_y)))
print(" =============================== ")

# stuck data for scatter plot
for eos in simulations.keys():
    for v_n in ["x", "y", "c", "marker"]:
        arr = []
        for q in simulations[eos].keys():
            for u_sim in simulations[eos][q]:
                arr.append(data2[eos][q][u_sim][v_n])
        data2[eos][v_n + "s"] = arr

Printcolor.green("Data is stacked")
# plot the scatter points
figname = ''
if mask_x == None:
    figname = figname + v_n_x + '_'
else:
    figname = figname + v_n_x + '_' + mask_x + '_'
if mask_y == None:
    figname = figname + v_n_y + '_'
else:
    figname = figname + v_n_y + '_' + mask_y + '_'
if mask_col == None:
    figname = figname + v_n_col + '_'
else:
    figname = figname + v_n_col + '_' + mask_col + '_'
if det == None:
    figname = figname + ''
else:
    figname = figname + str(det)
if do_plot_old_table:
    figname = figname + '_InclOldTbl'
figname = figname + '.png'
#
o_plot = PLOT_MANY_TASKS()
o_plot.gen_set["figdir"] = __outplotdir__
o_plot.gen_set["type"] = "cartesian"
o_plot.gen_set["figsize"] = (4.2, 3.6)  # <->, |]
o_plot.gen_set["figname"] = figname
o_plot.gen_set["sharex"] = True
o_plot.gen_set["sharey"] = False
o_plot.gen_set["subplots_adjust_h"] = 0.0
o_plot.gen_set["subplots_adjust_w"] = 0.0
o_plot.set_plot_dics = []

# FOR LEGENDS
if do_plot_promptcoll:
    x = -1.
    y = -1.
    marker_dic_lr = {
        'task': 'line', 'ptype': 'cartesian',
        'position': (1, 1),
        'xarr': [x], "yarr": [y],
        'xlabel': None, "ylabel": None,
        'label': "Prompt collapse",
        'marker': 's', 'color': 'gray', 'ms': 10., 'alpha': 0.4,
        'sharey': False,
        'sharex': False,  # removes angular citkscitks
        'fontsize': 14,
        'labelsize': 14
    }
    # if  eos == "BLh" and u_sim == simulations2[eos][q].keys()[-1]:
    #     print('-0--------------------')
    marker_dic_lr['legend'] = {'loc': 'upper left', 'ncol': 1, 'shadow': False, 'framealpha': 0.,
                               'borderaxespad': 0., 'fontsize': 11}
    o_plot.set_plot_dics.append(marker_dic_lr)
if do_plot_bh:
    x = -1.
    y = -1.
    marker_dic_lr = {
        'task': 'line', 'ptype': 'cartesian',
        'position': (1, 1),
        'xarr': [x], "yarr": [y],
        'xlabel': None, "ylabel": None,
        'label': "BH formation",
        'marker': 'o', 'color': 'gray', 'ms': 10., 'alpha': 0.4,
        'sharey': False,
        'sharex': False,  # removes angular citkscitks
        'fontsize': 14,
        'labelsize': 14
    }
    # if  eos == "BLh" and u_sim == simulations2[eos][q].keys()[-1]:
    #     print('-0--------------------')
    marker_dic_lr['legend'] = {'loc': 'upper left', 'ncol': 1, 'shadow': False, 'framealpha': 0.,
                               'borderaxespad': 0., 'fontsize': 11}
    o_plot.set_plot_dics.append(marker_dic_lr)
if do_plot_bh:
    x = -1.
    y = -1.
    marker_dic_lr = {
        'task': 'line', 'ptype': 'cartesian',
        'position': (1, 1),
        'xarr': [x], "yarr": [y],
        'xlabel': None, "ylabel": None,
        'label': "Long Lived",
        'marker': 'd', 'color': 'gray', 'ms': 10., 'alpha': 0.4,
        'sharey': False,
        'sharex': False,  # removes angular citkscitks
        'fontsize': 14,
        'labelsize': 14
    }
    # if  eos == "BLh" and u_sim == simulations2[eos][q].keys()[-1]:
    #     print('-0--------------------')
    marker_dic_lr['legend'] = {'loc': 'upper right', 'ncol': 1, 'shadow': False, 'framealpha': 0.,
                               'borderaxespad': 0., 'fontsize': 11}
    o_plot.set_plot_dics.append(marker_dic_lr)
if do_plot_old_table:
    x = -1.
    y = -1.
    marker_dic_lr = {
        'task': 'line', 'ptype': 'cartesian',
        'position': (1, 1),
        'xarr': [x], "yarr": [y],
        'xlabel': None, "ylabel": None,
        'label': "Radice+2018",
        'marker': '*', 'color': 'gray', 'ms': 10., 'alpha': 0.4,
        'sharey': False,
        'sharex': False,  # removes angular citkscitks
        'fontsize': 14,
        'labelsize': 14
    }
    # if  eos == "BLh" and u_sim == simulations2[eos][q].keys()[-1]:
    #     print('-0--------------------')
    marker_dic_lr['legend'] = {'loc': 'upper right', 'ncol': 1, 'shadow': False, 'framealpha': 0.,
                               'borderaxespad': 0., 'fontsize': 11}
    o_plot.set_plot_dics.append(marker_dic_lr)

# FOR FITS
if do_plot_linear_fit:
    if v_n_y == "Mej_tot" or v_n_y == "Mej_tot_scaled":
        fit_y = fit_y * 1e2
    if v_n_x == "Mej_tot" or v_n_x == "Mej_tot_scaled":
        fit_x = fit_x * 1e2
    # print(fit_x, fit_y)
    linear_fit = {
        'task': 'line', 'ptype': 'cartesian',
        'position': (1, 1),
        'xarr': fit_x, "yarr": fit_y,
        'xlabel': None, "ylabel": None,
        'label': "Linear fit",
        'ls': '-', 'color': 'black', 'lw': 1., 'alpha': 1., 'ds': 'default',
        'sharey': False,
        'sharex': False,  # removes angular citkscitks
        'fontsize': 14,
        'labelsize': 14
    }
    o_plot.set_plot_dics.append(linear_fit)

    #
if do_plot_old_table:

    if v_n_y == "Mej_tot" or v_n_y == "Mej_tot_scaled":
        old_all_y = old_all_y * 1e2
    if v_n_x == "Mej_tot" or v_n_x == "Mej_tot_scaled":
        old_all_x = old_all_x * 1e2
    dic = {
        'task': 'scatter', 'ptype': 'cartesian',  # 'aspect': 1.,
        'xarr': old_all_x, "yarr": old_all_y, "zarr": old_all_col,
        'position': (1, 1),  # 'title': '[{:.1f} ms]'.format(time_),
        'cbar': {},
        'v_n_x': v_n_x, 'v_n_y': v_n_y, 'v_n': v_n_col,
        'xlabel': None, "ylabel": Labels.labels(v_n_y, mask_y),
        'xmin': 300, 'xmax': 900, 'ymin': 0.03, 'ymax': 0.3, 'vmin': 1.0, 'vmax': 1.9,
        'fill_vmin': False,  # fills the x < vmin with vmin
        'xscale': None, 'yscale': None,
        'cmap': 'tab10', 'norm': None, 'ms': 60, 'marker': '*', 'alpha': 0.7, "edgecolors": None,
        'tick_params': {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False,  # "tick1On":True, "tick2On":True,
                        "labelsize": 12,
                        "direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        'yaxiscolor': {'bottom': 'black', 'top': 'black', 'right': 'black', 'left': 'black'},
        'minorticks': True,
        'title': {},  # {"text": eos, "fontsize": 12},
        'label': None,
        'legend': {},
        'sharey': False,
        'sharex': False,  # removes angular citkscitks
        'fontsize': 14,
        'labelsize': 14
    }
    o_plot.set_plot_dics.append(dic)
if do_plot_annotations:
    for eos in ["SFHo"]:
        print(eos)
        for q in simulations[eos].keys():
            for u_sim in simulations[eos][q].keys():
                x = data2[eos][q][u_sim]["x"]
                y = data2[eos][q][u_sim]["y"]
                y1 = data2[eos][q][u_sim]["ye1"]
                y2 = data2[eos][q][u_sim]["ye2"]
                if data2[eos][q][u_sim]["pizza2019"]:
                    if v_n_x == "Mej_tot" or v_n_x == "Mej_tot_scaled":
                        x = x * 1e2
                    if v_n_y == "Mej_tot" or v_n_y == "Mej_tot_scaled":
                        y1 = y1 * 1e2
                        y2 = y2 * 1e2
                        y = y * 1e2
                    marker_dic_lr = {
                        'task': 'line', 'ptype': 'cartesian',
                        'position': (1, 1),
                        'xarr': [x], "yarr": [y],
                        'xlabel': None, "ylabel": None,
                        'label': None,
                        'marker': '2', 'color': 'blue', 'ms': 15, 'alpha': 1.,
                        # 'ls': ls, 'color': 'gray', 'lw': 1.5, 'alpha': 1., 'ds': 'default',
                        'sharey': False,
                        'sharex': False,  # removes angular citkscitks
                        'fontsize': 14,
                        'labelsize': 14
                    }
                    o_plot.set_plot_dics.append(marker_dic_lr)

# PLOTS
i_col = 1
for eos in ["SLy4", "SFHo", "BLh", "LS220", "DD2"]:
    print(eos)
    # Error Bar
    if do_plot_error_bar_y:
        for q in simulations[eos].keys():
            for u_sim in simulations[eos][q].keys():
                x = data2[eos][q][u_sim]["x"]
                y = data2[eos][q][u_sim]["y"]
                y1 = data2[eos][q][u_sim]["ye1"]
                y2 = data2[eos][q][u_sim]["ye2"]
                nsims = data2[eos][q][u_sim]["lserr"]
                if v_n_x == "Mej_tot" or v_n_x == "Mej_tot_scaled":
                    x = x * 1e2
                if v_n_y == "Mej_tot" or v_n_y == "Mej_tot_scaled":
                    y1 = y1 * 1e2
                    y2 = y2 * 1e2
                    y = y * 1e2
                if nsims == 1:
                    ls = ':'
                elif nsims == 2:
                    ls = '--'
                elif nsims == 3:
                    ls = '-'
                else:
                    raise ValueError("too many sims >3")
                marker_dic_lr = {
                    'task': 'line', 'ptype': 'cartesian',
                    'position': (1, i_col),
                    'xarr': [x, x], "yarr": [y1, y2],
                    'xlabel': None, "ylabel": None,
                    'label': None,
                    'ls': ls, 'color': 'gray', 'lw': 1.5, 'alpha': 0.6, 'ds': 'default',
                    'sharey': False,
                    'sharex': False,  # removes angular citkscitks
                    'fontsize': 14,
                    'labelsize': 14
                }
                o_plot.set_plot_dics.append(marker_dic_lr)
    if do_plot_error_bar_x:
        for q in simulations[eos].keys():
            for u_sim in simulations[eos][q].keys():
                x = data2[eos][q][u_sim]["x"]
                x1 = data2[eos][q][u_sim]["xe1"]
                x2 = data2[eos][q][u_sim]["xe2"]
                y = data2[eos][q][u_sim]["y"]
                nsims = data2[eos][q][u_sim]["lserr"]
                if v_n_y == "Mej_tot" or v_n_y == "Mej_tot_scaled":
                    y = y * 1e2
                if v_n_x == "Mej_tot" or v_n_x == "Mej_tot_scaled":
                    x1 = x1 * 1e2
                    x2 = x2 * 1e2
                    x = x * 1e2
                if nsims == 1:
                    ls = ':'
                elif nsims == 2:
                    ls = '--'
                elif nsims == 3:
                    ls = '-'
                else:
                    raise ValueError("too many sims >3")
                marker_dic_lr = {
                    'task': 'line', 'ptype': 'cartesian',
                    'position': (1, i_col),
                    'xarr': [x1, x2], "yarr": [y, y],
                    'xlabel': None, "ylabel": None,
                    'label': None,
                    'ls': ls, 'color': 'gray', 'lw': 1.5, 'alpha': 1., 'ds': 'default',
                    'sharey': False,
                    'sharex': False,  # removes angular citkscitks
                    'fontsize': 14,
                    'labelsize': 14
                }
                o_plot.set_plot_dics.append(marker_dic_lr)
    # if do_plot_promptcoll:
    #     for q in simulations2[eos].keys():
    #         for u_sim in simulations2[eos][q].keys():
    #             x = data[eos][q][u_sim]["x"]
    #             y = data[eos][q][u_sim]["y"]
    #             isprompt = data[eos][q][u_sim]["isprompt"]
    #             if v_n_y == "Mej_tot" or v_n_y == "Mej_tot_scaled":
    #                 y = y * 1e2
    #             if v_n_x == "Mej_tot" or v_n_x == "Mej_tot_scaled":
    #                 x = x * 1e2
    #             if isprompt:
    #                 marker_dic_lr = {
    #                     'task': 'line', 'ptype': 'cartesian',
    #                     'position': (1, i_col),
    #                     'xarr': [x], "yarr": [y],
    #                     'xlabel': None, "ylabel": None,
    #                     'label': None,
    #                     'marker': 's', 'color': 'gray', 'ms': 10., 'alpha': 0.4,
    #                     'sharey': False,
    #                     'sharex': False,  # removes angular citkscitks
    #                     'fontsize': 14,
    #                     'labelsize': 14
    #                 }
    #                 # if  eos == "BLh" and u_sim == simulations2[eos][q].keys()[-1]:
    #                 #     print('-0--------------------')
    #                 marker_dic_lr['legend'] = {'loc':'upper left', 'ncol':1, 'shadow': False, 'framealpha':0., 'borderaxespad':0., 'fontsize':11}
    #                 o_plot.set_plot_dics.append(marker_dic_lr)
    # if do_plot_bh:
    #     for q in simulations2[eos].keys():
    #         for u_sim in simulations2[eos][q].keys():
    #             x = data[eos][q][u_sim]["x"]
    #             y = data[eos][q][u_sim]["y"]
    #             isbh = data[eos][q][u_sim]["isbh"]
    #             if v_n_y == "Mej_tot" or v_n_y == "Mej_tot_scaled":
    #                 y = y * 1e2
    #             if v_n_x == "Mej_tot" or v_n_x == "Mej_tot_scaled":
    #                 x = x * 1e2
    #             if isbh:
    #                 marker_dic_lr = {
    #                     'task': 'line', 'ptype': 'cartesian',
    #                     'position': (1, i_col),
    #                     'xarr': [x], "yarr": [y],
    #                     'xlabel': None, "ylabel": None,
    #                     'label': None,
    #                     'marker': 'o', 'color': 'gray', 'ms': 10., 'alpha': 0.4,
    #                     'sharey': False,
    #                     'sharex': False,  # removes angular citkscitks
    #                     'fontsize': 14,
    #                     'labelsize': 14
    #                 }
    #                 # if  eos == "BLh" and u_sim == simulations2[eos][q].keys()[-1]:
    #                 #     print('-0--------------------')
    #                 marker_dic_lr['legend'] = {'loc':'upper left', 'ncol':1, 'shadow': False, 'framealpha':0., 'borderaxespad':0., 'fontsize':11}
    #                 o_plot.set_plot_dics.append(marker_dic_lr)

    # LEGEND
    # if eos == "DD2" and plot_legend:
    #     for res in ["HR", "LR", "SR"]:
    #         marker_dic_lr = {
    #             'task': 'line', 'ptype': 'cartesian',
    #             'position': (1, i_col),
    #             'xarr': [-1], "yarr": [-1],
    #             'xlabel': None, "ylabel": None,
    #             'label': res,
    #             'marker': 'd', 'color': 'gray', 'ms': 8, 'alpha': 1.,
    #             'sharey': False,
    #             'sharex': False,  # removes angular citkscitks
    #             'fontsize': 14,
    #             'labelsize': 14
    #         }
    #         if res == "HR": marker_dic_lr['marker'] = "v"
    #         if res == "SR": marker_dic_lr['marker'] = "d"
    #         if res == "LR": marker_dic_lr['marker'] = "^"
    #         # if res == "BH": marker_dic_lr['marker'] = "x"
    #         if res == "SR":
    #             if v_n_y == "Ye_ave":
    #                 loc = 'lower right'
    #             else:
    #                 loc = 'upper right'
    #             marker_dic_lr['legend'] = {'loc': loc, 'ncol': 1, 'fontsize': 12, 'shadow': False,
    #                                        'framealpha': 0.5, 'borderaxespad': 0.0}
    #         o_plot.set_plot_dics.append(marker_dic_lr)
    #
    xarr = np.array(data2[eos]["xs"])
    yarr = np.array(data2[eos]["ys"])
    colarr = data2[eos]["cs"]
    markers = data2[eos]['markers']
    # marker = data[eos]["res" + 's']
    # edgecolor = data[eos]["vis" + 's']
    # bh_marker = data[eos]["tcoll" + 's']
    #
    # UTILS.fit_polynomial(xarr, yarr, 1, 100)
    #
    # print(xarr, yarr); exit(1)
    if v_n_y == "Mej_tot" or v_n_y == "Mej_tot_scaled":
        yarr = yarr * 1e2
    if v_n_x == "Mej_tot" or v_n_x == "Mej_tot_scaled":
        xarr = xarr * 1e2

    #
    #
    #
    # dic_bh = {
    #     'task': 'scatter', 'ptype': 'cartesian',  # 'aspect': 1.,
    #     'xarr': xarr, "yarr": yarr, "zarr": colarr,
    #     'position': (1, i_col),  # 'title': '[{:.1f} ms]'.format(time_),
    #     'cbar': {},
    #     'v_n_x': v_n_x, 'v_n_y': v_n_y, 'v_n': v_n_col,
    #     'xlabel': None, "ylabel": None, 'label': eos,
    #     'xmin': 300, 'xmax': 900, 'ymin': 0.03, 'ymax': 0.3, 'vmin': 1.0, 'vmax': 1.5,
    #     'fill_vmin': False,  # fills the x < vmin with vmin
    #     'xscale': None, 'yscale': None,
    #     'cmap': 'viridis', 'norm': None, 'ms': 80, 'marker': bh_marker, 'alpha': 1.0, "edgecolors": edgecolor,
    #     'fancyticks': True,
    #     'minorticks': True,
    #     'title': {},
    #     'legend': {},
    #     'sharey': False,
    #     'sharex': False,  # removes angular citkscitks
    #     'fontsize': 14,
    #     'labelsize': 14
    # }
    #
    # if mask_y != None and mask_y.__contains__("bern"):
    #     o_plot.set_plot_dics.append(dic_bh)
    #

    #

    #
    # print("marker: {}".format(marker))
    dic = {
        'task': 'scatter', 'ptype': 'cartesian',  # 'aspect': 1.,
        'xarr': xarr, "yarr": yarr, "zarr": colarr,
        'position': (1, i_col),  # 'title': '[{:.1f} ms]'.format(time_),
        'cbar': {},
        'v_n_x': v_n_x, 'v_n_y': v_n_y, 'v_n': v_n_col,
        'xlabel': None, "ylabel": Labels.labels(v_n_y, mask_y),
        'xmin': 300, 'xmax': 900, 'ymin': 0.03, 'ymax': 0.3, 'vmin': 1.0, 'vmax': 1.9,
        'fill_vmin': False,  # fills the x < vmin with vmin
        'xscale': None, 'yscale': None,
        'cmap': 'tab10', 'norm': None, 'ms': 60, 'markers': markers, 'alpha': 0.6, "edgecolors": None,
        'tick_params': {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False,  # "tick1On":True, "tick2On":True,
                        "labelsize": 12,
                        "direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        'yaxiscolor': {'bottom': 'black', 'top': 'black', 'right': 'black', 'left': 'black'},
        'minorticks': True,
        'title': {},  # {"text": eos, "fontsize": 12},
        'label': None,
        'legend': {},
        'sharey': False,
        'sharex': False,  # removes angular citkscitks
        'fontsize': 14,
        'labelsize': 14
    }
    #

    if v_n_y == "q":
        dic['ymin'], dic['ymax'] = 0.9, 2.0
    if v_n_col == "nsims":
        dic['vmin'], dic['vmax'] = 1, 3.9
        dic['cmap'] = __get_custom_descrete_colormap(3)
        # dic['cmap'] = 'RdYlBu'

    if v_n_y == "Mdisk3Dmax":
        dic['ymin'], dic['ymax'] = 0.03, 0.30
    if v_n_y == "Mb":
        dic['ymin'], dic['ymax'] = 2.8, 3.4
    if v_n_y == "Mej_tot" and mask_y == "geo":
        dic['ymin'], dic['ymax'] = 0, 1.2
    if v_n_y == "Mej_tot_scaled" and mask_y == "geo":
        dic['ymin'], dic['ymax'] = 0, 0.5

    if v_n_y == "Mej_tot_scaled2" and mask_y == "geo":
        dic['ymin'], dic['ymax'] = 0, 1.
    if v_n_y == "Mej_tot_scaled2" and mask_y == "geo_entropy_above_10":
        dic['ymin'], dic['ymax'] = 0, 0.01
    if v_n_y == "Mej_tot_scaled2" and mask_y == "geo_entropy_below_10":
        dic['ymin'], dic['ymax'] = 0, 0.02

    if v_n_y == "Mej_tot" and mask_y == "bern_geoend":
        if dic['yscale'] == "log":
            dic['ymin'], dic['ymax'] = 1e-3, 2e0
        else:
            dic['ymin'], dic['ymax'] = 0, 3.2
    if v_n_y == "Mej_tot" and mask_y == "geo_entropy_above_10":
        if dic['yscale'] == "log":
            dic['ymin'], dic['ymax'] = 1e-3, 2e0
        else:
            dic['ymin'], dic['ymax'] = 0, .6
    if v_n_y == "Mej_tot" and mask_y == "geo_entropy_below_10":
        if dic['yscale'] == "log":
            dic['ymin'], dic['ymax'] = 1e-2, 2e0
        else:
            dic['ymin'], dic['ymax'] = 0, 1.2
    if v_n_y == "Mej_tot_scaled" and mask_y == "bern_geoend":
        dic['ymin'], dic['ymax'] = 0, 3.

    if v_n_y == "Ye_ave" and mask_y == "geo":
        dic['ymin'], dic['ymax'] = 0.01, 0.35
    if v_n_y == "Ye_ave" and mask_y == "bern_geoend":
        dic['ymin'], dic['ymax'] = 0.1, 0.4
    if v_n_y == "vel_inf_ave" and mask_y == "geo":
        dic['ymin'], dic['ymax'] = 0.1, 0.3
    if v_n_y == "vel_inf_ave" and mask_y == "bern_geoend":
        dic['ymin'], dic['ymax'] = 0.05, 0.25
    #

    #
    if v_n_x == "Mdisk3Dmax":
        dic['xmin'], dic['xmax'] = 0.03, 0.30
    if v_n_x == "Mb":
        dic['xmin'], dic['xmax'] = 2.8, 3.4
    if v_n_x == "Mej_tot" and mask_x == "geo":
        dic['xmin'], dic['xmax'] = 0, 1.5
    if v_n_x == "Mej_tot_scaled" and mask_x == "geo":
        dic['xmin'], dic['xmax'] = 0, 0.5
    if v_n_x == "Mej_tot" and mask_x == "bern_geoend":
        dic['xmin'], dic['xmax'] = 0, 3.2
    if v_n_x == "Mej_tot" and mask_x == "geo_entropy_above_10":
        if dic['xscale'] == "log":
            dic['xmin'], dic['xmax'] = 1e-3, 2e0
        else:
            dic['xmin'], dic['xmax'] = 0, .6
    if v_n_x == "Mej_tot" and mask_x == "geo_entropy_below_10":
        if dic['xscale'] == "log":
            dic['xmin'], dic['xmax'] = 1e-2, 2e0
        else:
            dic['xmin'], dic['xmax'] = 0, 1.2
    if v_n_x == "Mej_tot_scaled" and mask_x == "bern_geoend":
        dic['xmin'], dic['xmax'] = 0, 3.
    if v_n_x == "Ye_ave" and mask_x == "geo":
        dic['xmin'], dic['xmax'] = 0.01, 0.30
    if v_n_x == "Ye_ave" and mask_x == "bern_geoend":
        dic['xmin'], dic['xmax'] = 0.1, 0.4
    if v_n_x == "vel_inf_ave" and mask_x == "geo":
        dic['xmin'], dic['xmax'] = 0.1, 0.3
    if v_n_x == "vel_inf_ave" and mask_x == "bern_geoend":
        dic['xmin'], dic['xmax'] = 0.05, 0.25

    #
    # if eos == "SLy4":
    #     dic['xmin'], dic['xmax'] = 380, 420
    #     dic['xticks'] = [390, 410]
    # if eos == "SFHo":
    #     dic['xmin'], dic['xmax'] = 390, 430
    #     dic['xticks'] = [400, 420]
    # if eos == "BLh":
    #     dic['xmin'], dic['xmax'] = 510, 550
    #     dic['xticks'] = [520, 540]
    # if eos == "LS220":
    #     dic['xmin'], dic['xmax'] = 690, 730
    #     dic['xticks'] = [700, 720]
    # if eos == "DD2":
    #     dic['xmin'], dic['xmax'] = 820, 860
    #     dic['xticks'] = [830, 850]
    # if eos == "SLy4":
    #     dic['tick_params']['right'] = False
    #     dic['yaxiscolor']["right"] = "lightgray"
    # elif eos == "DD2":
    #     dic['tick_params']['left'] = False
    #     dic['yaxiscolor']["left"] = "lightgray"
    # else:
    #     dic['tick_params']['left'] = False
    #     dic['tick_params']['right'] = False
    #     dic['yaxiscolor']["left"] = "lightgray"
    #     dic['yaxiscolor']["right"] = "lightgray"

    #
    # if eos != "SLy4" and eos != "DD2":
    #     dic['yaxiscolor'] = {'left':'lightgray','right':'lightgray', 'label': 'black'}
    #     dic['ytickcolor'] = {'left':'lightgray','right':'lightgray'}
    #     dic['yminortickcolor'] = {'left': 'lightgray', 'right': 'lightgray'}
    # elif eos == "DD2":
    #     dic['yaxiscolor'] = {'left': 'lightgray', 'right': 'black', 'label': 'black'}
    #     # dic['ytickcolor'] = {'left': 'lightgray'}
    #     # dic['yminortickcolor'] = {'left': 'lightgray'}
    # elif eos == "SLy4":
    #     dic['yaxiscolor'] = {'left': 'black', 'right': 'lightgray', 'label': 'black'}
    #     # dic['ytickcolor'] = {'right': 'lightgray'}
    #     # dic['yminortickcolor'] = {'right': 'lightgray'}

    #
    # if eos != "SLy4":
    #     dic['sharey'] = True
    if eos == "BLh":
        dic['xlabel'] = Labels.labels(v_n_x, mask_x)
    if eos == 'DD2':
        dic['cbar'] = {'location': 'right .03 .0', 'label': Labels.labels(v_n_col),  # 'fmt': '%.1f',
                       'labelsize': 14, 'fontsize': 14}
        if v_n_col == "nsims":
            dic['cbar']['fmt'] = '%d'
    #
    o_plot.set_plot_dics.append(dic)
    #

    # i_col = i_col + 1

    if do_plot_old_table:
        if v_n_x == 'Lambda':
            dic['xmin'], dic['xmax'] = 5, 1500

# LEGEND


#
o_plot.main()
exit(0)