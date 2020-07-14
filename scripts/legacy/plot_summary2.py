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
v_n_x = "Mej_tot-geo_entropy_above_10_dev_mchirp" #"Lambda"#"Mej_tot-geo_entropy_above_10_dev_mtot"#"Mej_tot-geo_entropy_above_10"#"Lambda"
v_n_y = "Mej_tot-geo_entropy_below_10_dev_mchirp" #"Mej_tot-geo_dev_mtotsymqmchirp"#"Mej_tot-geo_entropy_below_10_dev_mtot"#"Mej_tot-geo_entropy_below_10"#"Mej_tot-geo_6"#"Mej_tot-geo_Mchirp"
v_n_col = "q"
simlist = simulations
simlist2 = old_simulations
simtable = Paths.output + "models3.csv"#"models2.csv"
simtable2 = Paths.output + "radice2018_summary3.csv"#"radice2018_summary.csv"
deferr = 0.2
__outplotdir__ = "/data01/numrel/vsevolod.nedora/figs/all3/"
xyscales = None#"log"
prompt_bhtime = 1.5

limit_mchirp_to = [1.14, 1.22]

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

def set_dic_yminymax(v_n, dic, yarr, extra=2.):
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
        dic['ymin'], dic['ymax'] = np.array(yarr).min(), np.array(yarr).max() + (extra * (np.array(yarr).max() - np.array(yarr).min()))
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
        print("\t{} [{}] Chrip:{}".format(usim, len(sims), o_tbl.get_par(sims[0], "mchirp")))
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
        figname = figname + '_InclOldTbl2'
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
        'cmap': 'tab10', 'norm': None, 'ms': 60, 'markers': data["allmarker"], 'alpha': 0.6, "edgecolors": None,
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


    # translation = {"Mej_tot-geo":"Mej",
    #                "Lambda":"Lambda",
    #                "Mej_tot-geo_Mchirp":"Mej_Mchirp",
    #                "Mej_tot-geo_1":"Mej1",
    #                "Mej_tot-geo_2":"Mej2",
    #                "Mej_tot-geo_3": "Mej3",
    #                "Mej_tot-geo_4": "Mej4",
    #                "Mej_tot-geo_5": "Mej5",
    #                "Mej_tot-geo_6": "Mej6",
    #                "tcoll_gw":"tcoll",
    #                "Mej_tot-geo_entropy_above_10":"Mej_shocked",
    #                "Mej_tot-geo_entropy_below_10":"Mej_tidal",
    #                "Mej_tot-geo_entropy_above_10_dev_mtot":"Mej_shocked_dev_mtot",
    #                "Mej_tot-geo_entropy_below_10_dev_mtot":"Mej_tidal_dev_mtot",
    #                "Mej_tot-geo_dev_mtot":"Mej_dev_mtot",
    #                "Mej_tot-geo_dev_mtot2":"Mej_dev_mtot2",
    #                "Mej_tot-geo_mult_mtot":"Mej_mult_mtot",
    #                "Mej_tot-geo_mult_mtot2":"Mej_mult_mtot2",
    #                "Mej_tot-geo_dev_symq":"Mej_dev_symq",
    #                "Mej_tot-geo_dev_symq2":"Mej_dev_symq2",
    #                "Mej_tot-geo_mult_symq":"Mej_mult_symq",
    #                "Mej_tot-geo_mult_symq2":"Mej_mult_symq2",
    #                "Mej_tot-geo_dev_mchirp":"Mej_dev_mchirp",
    #                "Mej_tot-geo_dev_mchirp2":"Mej_dev_mchirp2",
    #                "Mej_tot-geo_mult_mchirp":"Mej_mult_mchirp",
    #                "Mej_tot-geo_dev_symqmchirp":"Mej_dev_symqmchirp",
    #                "Mej_tot-geo_dev_mtotsymqmchirp":"Mej_dev_mtotsymqmchirp"}
    #
    # v_n_x = translation[v_n_x]
    # v_n_y = translation[v_n_y]

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
    plotted = []
    for eos in simlist2.keys():
        data2[eos] = {}
        for usim in simlist2[eos].keys():
            data2[eos][usim] = {}
            sims = simlist2[eos][usim]
            mchirp = o_tbl.get_par(sims[0], "mchirp")
            if len(limit_mchirp_to) > 0 and mchirp > limit_mchirp_to[0] and mchirp < limit_mchirp_to[-1]:
                print("\t{} [{}]".format(usim, len(sims)))
                x, x1, x2 = o_tbl.get_par_with_error(sims, v_n_x, deferr=deferr)
                y, y1, y2 = o_tbl.get_par_with_error(sims, v_n_y, deferr=deferr)
                # col = o_tbl.get_par(sims[0], v_n_col)
                col = o_tbl.get_par(sims[0], v_n_col)
                if np.isnan(x) or np.isnan(y) or np.isnan(col):
                    break
                else:
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
                    plotted.append(usim)
            else:
                Printcolor.blue("skipping: {} with chirp: {} not in [{}, {}]"
                                .format(usim, mchirp, limit_mchirp_to[0], limit_mchirp_to[-1]))
    Printcolor.green("Plotted {} sims".format(len(plotted)))
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
        dic2["xmin"], dic2["xmax"] = 3e-3, 1e1
        dic2["ymin"], dic2["ymax"] = 3e-3, 1e1
    o_plot.set_plot_dics.append(dic2)

    # for fits
    for eos in simlist2.keys():
        for usim in data2[eos].keys():
            if len(data2[eos][usim].keys()) > 0:
                print(data2[eos][usim])
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
















