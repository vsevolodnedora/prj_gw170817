#
# plots summary data
# taking data from .csv table
#

from __future__ import division

import numpy as np
import pandas as pd
import sys
import os
import copy
import h5py
import csv

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import AutoMinorLocator, FixedLocator, NullFormatter, \
    MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from _models_old import *

#
load_davids_data = True
t_pc = 1.5 * 1.e-3
x_dic   = {"v_n": "Lambda",       "err": None,  "mod": None,   "deferr": None}
y_dic   = {"v_n": "Mej_tot-geo",  "err": "ud",  "mod": "*1e2", "deferr": 0.2}
col_dic = {"v_n": "q",            "err": None,  "mod": None,   "deferr": None}
marker_v_n = "EOS"
mc_bh = "o"
mc_st = "d"
mc_pc = "s"
#
plot_dic= {"vmin":1., "vmax":2.0,
           "cmap":"tab10", "label":None, "alpha":1.,
           "ms":20.}
#
eos_marker = {"DD2":   marker_list[0],
              "BLh":   marker_list[1],
              "SLy4":  marker_list[2],
              "SFHo":  marker_list[3],
              "LS220": marker_list[4]}
#
plot_sims = unique_simulations
#
__outplotdir__ = "/data01/numrel/vsevolod.nedora/figs/all3/"
#

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

def standard_div(x_arr):
    x_arr = np.array(x_arr, dtype=float)
    n = 1. * len(x_arr)
    mean = sum(x_arr) / n
    tmp = (1 / (n-1)) * np.sum((x_arr - mean) ** 2)
    return mean, np.sqrt(tmp)

def get_minmax(v_n, arr, extra = 2.):
    if v_n == "Mej_tot-geo" or v_n == "Mej":
        min_, max_ = 0, 2.
    elif v_n == "Ye_ave-geo" or "Yeej":
        min_, max_ = 0., 0.5
    elif v_n == "vel_inf_ave-geo" or v_n == "vej":
        min_, max_= 0., 0.7
    else:
        min_, max_ = np.array(arr).min(), np.array(arr).max() + (extra * (np.array(arr).max() - np.array(arr).min()))
        print("xlimits are not set for v_n_x:{}".format(v_n))

    return min_, max_

def __apply_mod(v_n, val, mod):
    # print(v_n, val, mod)
    if mod != None and mod != "":
        # exit(1)
        if mod[0] == '*':
            mult_by = float(mod.split("*")[-1])
            val = val * mult_by
            # print(val); exit(1)
        elif mod[0] == "/":
            dev_by = float(mod.split('/')[-1])
            val = val / dev_by
        else:
            raise NameError("v_n:{} mod:{} is invalid. Use mod '*float' or '/float' "
                            .format(v_n, mod))

    return val

def get_group_value2(group, task_dic):

    v_n = task_dic["v_n"]
    # fmt = task_dic["fmt"]
    mod = task_dic["mod"]
    err = task_dic["err"]
    deferr = task_dic["deferr"]

    # print(type(group.resolution)); print(list(group.resolution)); exit(1)

    ress = sorted(list(group.resolution))
    if len(ress) > 3:
        raise ValueError("too many resolutions! {}".format(group.resolutons))

    if "SR" in ress and "LR" in ress and "HR" in ress:
        # HR - SR , SR - LR
        # print("\t 1 ress:{} v_n:{} fmt:{} err:{}".format(ress, v_n, fmt, err))
        if v_n == "group":
            val = list(group.group)[0]
            val = val.replace("_", "--") + r"\textit{[S,H,L]}"
            return val
        elif v_n == "resolution":
            val = r"\texttt{" +  " ".join(ress) + "}"
            return val
        elif err == None or err == "":
            sr = group[group.resolution == "SR"]
            val_sr = float(list(sr[v_n])[0])
            val_sr = __apply_mod(v_n, val_sr, mod)
            return val_sr, 0., 0.
        else:
            hr =  group[group.resolution == "HR"]
            sr =  group[group.resolution == "SR"]
            lr =  group[group.resolution == "LR"]
            #
            # print(hr); exit(1)
            #
            val_sr = float(sr[v_n])
            val_lr = float(lr[v_n])
            val_hr = float(hr[v_n])
            #
            # print(float(val_sr)); exit(1)
            #
            val_sr = float(__apply_mod(v_n, val_sr, mod))
            val_lr = float(__apply_mod(v_n, val_lr, mod))
            val_hr = float(__apply_mod(v_n, val_hr, mod))
            #
            if val_hr < val_sr and val_sr < val_lr:
                # convergence
                conv = True
                del1 = val_lr - val_sr # +
                del2 = val_sr - val_hr # -
            elif val_hr > val_sr and val_sr > val_lr:
                # oppoiste of convergence:
                conv = False
                del1 = val_sr - val_lr # +
                del2 = val_sr - val_hr # -
            elif val_hr < val_sr and val_sr > val_lr:
                # SR is the top
                conv = False
                del1 = 0.
                del2 = max([val_sr-val_hr, val_sr-val_lr])
            elif val_hr > val_sr and val_sr < val_lr:
                # SR lowest
                del1 = max([val_hr-val_sr, val_lr-val_sr])
                del2 = 0.
            else:
                raise ValueError("Wrong. Eveything is wrong.")

            rx = 1.35
            p = np.log((val_sr - val_lr) / (val_hr - val_sr)) / rx
            if p > 0.:
                conv = True
            else:
                conv = False
            ref = (val_hr * (rx ** p) - val_sr) / ((rx ** p) - 1.)
            #if conv: print("v_n:{} {} {} {} -> {}".format(v_n, val_lr, val_sr, val_hr, ref)); exit(1)


            del1 = np.abs(del1)
            del2 = np.abs(del2)

            # return str("$" + val_sr + r"^{+" + str(del1) + r"} _{-" + str(del2)) + "} $"
            return val_sr, del1, del2
    elif len(ress) == 2 and "SR" in ress and "LR" in ress:
        # SR - LR, SR - LR
        # print("\t 2 ress:{} v_n:{} fmt:{} err:{}".format(ress, v_n, fmt, err))
        if v_n == "group":
            val = list(group.group)[0]
            val = val.replace("_", "--") + r"\textit{[S,L]}"
            return val
        elif v_n == "resolution":
            val = r"\texttt{" +  " ".join(ress) + "}"
            return val
        elif err == None or err == "":
            sr = group[group.resolution == "SR"]
            val_sr = float(list(sr[v_n])[0])
            val_sr = __apply_mod(v_n, val_sr, mod)
            return val_sr, 0., 0.
        else:
            # hr = group[group.resolution == "HR"]
            sr = group[group.resolution == "SR"]
            lr = group[group.resolution == "LR"]
            #
            # print(hr); exit(1)
            #
            val_sr = float(sr[v_n])
            val_lr = float(lr[v_n])

            val_sr = __apply_mod(v_n, val_sr, mod)
            val_lr = __apply_mod(v_n, val_lr, mod)

            if val_sr < val_lr:
                # conv = True
                del1 = val_lr - val_sr
                del2 = del1
            else:
                # conv = False
                del1 = val_sr - val_lr
                del2 = del1

            return val_sr, del1, del2
            # return str("$" + val_sr + r"^{+" + str(del1) + r"} _{-" + str(del2)) + "} $"
    elif len(ress) == 2 and "SR" in ress and "HR" in ress:
        # print("\t 3 ress:{} v_n:{} fmt:{} err:{}".format(ress, v_n, fmt, err))
        if v_n == "group":
            val = list(group.group)[0]
            val = val.replace("_", "--") + r"\textit{[S,H]}"
            return val
        elif v_n == "resolution":
            val = r"\texttt{" +  " ".join(ress) + "}"
            return val
        elif err == None or err == "":
            sr = group[group.resolution == "SR"]
            val_sr = float(list(sr[v_n])[0])
            val_sr = __apply_mod(v_n, val_sr, mod)
            return val_sr, 0., 0.
        else:
            hr = group[group.resolution == "HR"]
            sr = group[group.resolution == "SR"]

            val_sr = float(sr[v_n])
            val_hr = float(hr[v_n])

            val_sr = __apply_mod(v_n, val_sr, mod)
            val_hr = __apply_mod(v_n, val_hr, mod)

            if val_sr > val_hr:
                # conv = True
                del1 = del2 = val_sr - val_hr
            else:
                # conv = False
                del1 = del2 = val_hr - val_sr
            # conv = False
            return val_sr, del1, del2
            # return str("$" + val_sr + r"^{+" + str(del1) + r"} _{-" + str(del2)) + "} $"
        pass
    elif len(ress) == 2 and "LR" in ress and "HR" in ress:
        # print("\t 3 ress:{} v_n:{} fmt:{} err:{}".format(ress, v_n, fmt, err))
        if v_n == "group":
            val = list(group.group)[0]
            val = val.replace("_", "--") + r"\textit{[L,H]}"
            return val
        elif v_n == "resolution":
            val = r"\texttt{" +  " ".join(ress) + "}"
            return val
        elif err == None or err == "":
            lr = group[group.resolution == "LR"]
            val_lr = float(list(lr[v_n])[0])
            val_lr = __apply_mod(v_n, val_lr, mod)
            return val_lr, 0., 0.
        else:
            hr = group[group.resolution == "HR"]
            lr = group[group.resolution == "LR"]

            val_lr = float(lr[v_n])
            val_hr = float(hr[v_n])

            val_lr = __apply_mod(v_n, val_lr, mod)
            val_hr = __apply_mod(v_n, val_hr, mod)

            if val_lr > val_hr:
                # conv = True
                del1 = del2 = val_lr - val_hr
            else:
                # conv = False
                del1 = del2 = val_hr - val_lr

            return val_hr, del1, del2
            # return str("$" + val_sr + r"^{+" + str(del1) + r"} _{-" + str(del2)) + "} $"
        pass
    elif len(ress) == 1 and "SR" in ress:
        # SR - X*SR, SR - X*SR
        # print("\t 4 ress:{} v_n:{} fmt:{} err:{}".format(ress, v_n, fmt, err))
        if v_n == "group":
            val = list(group.group)[0]
            val = val.replace("_", "--") + r"\textit{[S]}"
            return val
        elif v_n == "resolution":
            val = r"\texttt{" +  " ".join(ress) + "}"
            return val
        elif err == None or err == "":
            sr = group[group.resolution == "SR"]
            val_sr = float(list(sr[v_n])[0])
            val_sr = __apply_mod(v_n, val_sr, mod)
            return val_sr, 0., 0.
        else:
            # hr = group[group.resolution == "HR"]
            sr = group[group.resolution == "SR"]

            val_sr = float(sr[v_n])

            val_sr = __apply_mod(v_n, val_sr, mod)

            del1 = del2 = val_sr * deferr

            return val_sr, del1, del2
            # return str("$" + val_sr + "$")#+ r"^{" + str(del1) + r"} _{" + str(del2)) + "} $"
    elif len(ress) == 1 and "LR" in ress:
        # LR - Y*LR, SR - Y*LR
        # print("\t 5 ress:{} v_n:{} fmt:{} err:{}".format(ress, v_n, fmt, err))
        if v_n == "group":
            val = list(group.group)[0]
            val = val.replace("_", "--") + r"\textit{[L]}"
            return val
        elif v_n == "resolution":
            val = r"\texttt{" +  " ".join(ress) + "}"
            return val
        elif err == None or err == "":
            lr = group[group.resolution == "LR"]
            val_lr = float(list(lr[v_n])[0])
            val_lr = __apply_mod(v_n, val_lr, mod)
            return val_lr, 0., 0.
        else:

            lr = group[group.resolution == "LR"]

            val_lr = float(lr[v_n])

            val_lr = __apply_mod(v_n, val_lr, mod)

            del1 = del2 = val_lr * deferr

            return val_lr, del1, del2
            # return str("$" + val_lr + "$")#r"^{" + str(del1) + r"} _{" + str(del2)) + "} $"
    elif len(ress) == 1 and "HR" in ress:
        # HR - Z*HR, SR - Z*HR
        # print("\t 6 ress:{} v_n:{} fmt:{} err:{}".format(ress, v_n, fmt, err))
        if v_n == "group":
            val = list(group.group)[0]
            val = val.replace("_", "--") + r"\textit{[H]}"
            return val
        elif v_n == "resolution":
            val = r"\texttt{" +  " ".join(ress) + "}"
            return val
        elif err == None or err == "":
            hr = group[group.resolution == "HR"]
            val_hr = float(list(hr[v_n])[0])
            val_hr = __apply_mod(v_n, val_hr, mod)
            return val_hr, 0., 0.
        else:
            hr = group[group.resolution == "HR"]

            val_hr = float(hr[v_n])

            val_hr = __apply_mod(v_n, val_hr, mod)

            del1 = del2 = val_hr * deferr

            return val_hr, del1, del2
            # return str("$" + val_hr + "$")#r"^{" + str(del1) + r"} _{" + str(del2)) + "} $"
    else:
        print(group)
        raise ValueError("\t 7. Unrecognized resoltion setup \n "
                         "ress:{} v_n:{} err:{}".format(ress, v_n, err))
    # exit(1)

def get_group_value3(group, task_dic):

    v_n = task_dic["v_n"]
    # fmt = task_dic["fmt"]
    mod = task_dic["mod"]
    err = task_dic["err"]
    deferr = task_dic["deferr"]

    # print(type(group.resolution)); print(list(group.resolution)); exit(1)

    ress = sorted(list(group.resolution))
    if len(ress) > 3:
        raise ValueError("too many resolutions! {}".format(group.resolutons))

    if "SR" in ress and "LR" in ress and "HR" in ress:
        # HR - SR , SR - LR
        # print("\t 1 ress:{} v_n:{} fmt:{} err:{}".format(ress, v_n, fmt, err))
        if v_n == "group":
            val = list(group.group)[0]
            val = val.replace("_", "--") + r"\textit{[S,H,L]}"
            return val
        elif v_n == "resolution":
            val = r"\texttt{" +  " ".join(ress) + "}"
            return val
        elif err == None or err == "":
            sr = group[group.resolution == "SR"]
            val_sr = float(list(sr[v_n])[0])
            val_sr = __apply_mod(v_n, val_sr, mod)
            return val_sr, 0., 0.
        else:
            hr =  group[group.resolution == "HR"]
            sr =  group[group.resolution == "SR"]
            lr =  group[group.resolution == "LR"]
            #
            # print(hr); exit(1)
            #
            val_sr = float(sr[v_n])
            val_lr = float(lr[v_n])
            val_hr = float(hr[v_n])
            #
            # print(float(val_sr)); exit(1)
            #
            val_sr = float(__apply_mod(v_n, val_sr, mod))
            val_lr = float(__apply_mod(v_n, val_lr, mod))
            val_hr = float(__apply_mod(v_n, val_hr, mod))
            #
            mean, stddiv = standard_div([val_hr, val_sr, val_lr])
            return mean, stddiv, stddiv

    elif len(ress) == 2 and "SR" in ress and "LR" in ress:
        # SR - LR, SR - LR
        # print("\t 2 ress:{} v_n:{} fmt:{} err:{}".format(ress, v_n, fmt, err))
        if v_n == "group":
            val = list(group.group)[0]
            val = val.replace("_", "--") + r"\textit{[S,L]}"
            return val
        elif v_n == "resolution":
            val = r"\texttt{" +  " ".join(ress) + "}"
            return val
        elif err == None or err == "":
            sr = group[group.resolution == "SR"]
            val_sr = float(list(sr[v_n])[0])
            val_sr = __apply_mod(v_n, val_sr, mod)
            return val_sr, 0., 0.
        else:
            # hr = group[group.resolution == "HR"]
            sr = group[group.resolution == "SR"]
            lr = group[group.resolution == "LR"]
            #
            # print(hr); exit(1)
            #
            val_sr = float(sr[v_n])
            val_lr = float(lr[v_n])

            val_sr = __apply_mod(v_n, val_sr, mod)
            val_lr = __apply_mod(v_n, val_lr, mod)

            mean, stddiv = standard_div([val_sr, val_lr])
            return mean, stddiv, stddiv
            # return str("$" + val_sr + r"^{+" + str(del1) + r"} _{-" + str(del2)) + "} $"
    elif len(ress) == 2 and "SR" in ress and "HR" in ress:
        # print("\t 3 ress:{} v_n:{} fmt:{} err:{}".format(ress, v_n, fmt, err))
        if v_n == "group":
            val = list(group.group)[0]
            val = val.replace("_", "--") + r"\textit{[S,H]}"
            return val
        elif v_n == "resolution":
            val = r"\texttt{" +  " ".join(ress) + "}"
            return val
        elif err == None or err == "":
            sr = group[group.resolution == "SR"]
            val_sr = float(list(sr[v_n])[0])
            val_sr = __apply_mod(v_n, val_sr, mod)
            return val_sr, 0., 0.
        else:
            hr = group[group.resolution == "HR"]
            sr = group[group.resolution == "SR"]

            val_sr = float(sr[v_n])
            val_hr = float(hr[v_n])

            val_sr = __apply_mod(v_n, val_sr, mod)
            val_hr = __apply_mod(v_n, val_hr, mod)

            mean, stddiv = standard_div([val_hr, val_sr])
            return mean, stddiv, stddiv
            # return str("$" + val_sr + r"^{+" + str(del1) + r"} _{-" + str(del2)) + "} $"
        pass
    elif len(ress) == 2 and "LR" in ress and "HR" in ress:
        # print("\t 3 ress:{} v_n:{} fmt:{} err:{}".format(ress, v_n, fmt, err))
        if v_n == "group":
            val = list(group.group)[0]
            val = val.replace("_", "--") + r"\textit{[L,H]}"
            return val
        elif v_n == "resolution":
            val = r"\texttt{" +  " ".join(ress) + "}"
            return val
        elif err == None or err == "":
            lr = group[group.resolution == "LR"]
            val_lr = float(list(lr[v_n])[0])
            val_lr = __apply_mod(v_n, val_lr, mod)
            return val_lr, 0., 0.
        else:
            hr = group[group.resolution == "HR"]
            lr = group[group.resolution == "LR"]

            val_lr = float(lr[v_n])
            val_hr = float(hr[v_n])

            val_lr = __apply_mod(v_n, val_lr, mod)
            val_hr = __apply_mod(v_n, val_hr, mod)

            mean, stddiv = standard_div([val_hr, val_lr])
            return mean, stddiv, stddiv
            # return str("$" + val_sr + r"^{+" + str(del1) + r"} _{-" + str(del2)) + "} $"
        pass
    elif len(ress) == 1 and "SR" in ress:
        # SR - X*SR, SR - X*SR
        # print("\t 4 ress:{} v_n:{} fmt:{} err:{}".format(ress, v_n, fmt, err))
        if v_n == "group":
            val = list(group.group)[0]
            val = val.replace("_", "--") + r"\textit{[S]}"
            return val
        elif v_n == "resolution":
            val = r"\texttt{" +  " ".join(ress) + "}"
            return val
        elif err == None or err == "":
            sr = group[group.resolution == "SR"]
            val_sr = float(list(sr[v_n])[0])
            val_sr = __apply_mod(v_n, val_sr, mod)
            return val_sr, 0., 0.
        else:
            # hr = group[group.resolution == "HR"]
            sr = group[group.resolution == "SR"]

            val_sr = float(sr[v_n])

            val_sr = __apply_mod(v_n, val_sr, mod)

            del1 = del2 = val_sr * deferr

            return val_sr, del1, del2
            # return str("$" + val_sr + "$")#+ r"^{" + str(del1) + r"} _{" + str(del2)) + "} $"
    elif len(ress) == 1 and "LR" in ress:
        # LR - Y*LR, SR - Y*LR
        # print("\t 5 ress:{} v_n:{} fmt:{} err:{}".format(ress, v_n, fmt, err))
        if v_n == "group":
            val = list(group.group)[0]
            val = val.replace("_", "--") + r"\textit{[L]}"
            return val
        elif v_n == "resolution":
            val = r"\texttt{" +  " ".join(ress) + "}"
            return val
        elif err == None or err == "":
            lr = group[group.resolution == "LR"]
            val_lr = float(list(lr[v_n])[0])
            val_lr = __apply_mod(v_n, val_lr, mod)
            return val_lr, 0., 0.
        else:

            lr = group[group.resolution == "LR"]

            val_lr = float(lr[v_n])

            val_lr = __apply_mod(v_n, val_lr, mod)

            del1 = del2 = val_lr * deferr

            return val_lr, del1, del2
            # return str("$" + val_lr + "$")#r"^{" + str(del1) + r"} _{" + str(del2)) + "} $"
    elif len(ress) == 1 and "HR" in ress:
        # HR - Z*HR, SR - Z*HR
        # print("\t 6 ress:{} v_n:{} fmt:{} err:{}".format(ress, v_n, fmt, err))
        if v_n == "group":
            val = list(group.group)[0]
            val = val.replace("_", "--") + r"\textit{[H]}"
            return val
        elif v_n == "resolution":
            val = r"\texttt{" +  " ".join(ress) + "}"
            return val
        elif err == None or err == "":
            hr = group[group.resolution == "HR"]
            val_hr = float(list(hr[v_n])[0])
            val_hr = __apply_mod(v_n, val_hr, mod)
            return val_hr, 0., 0.
        else:
            hr = group[group.resolution == "HR"]

            val_hr = float(hr[v_n])

            val_hr = __apply_mod(v_n, val_hr, mod)

            del1 = del2 = val_hr * deferr

            return val_hr, del1, del2
            # return str("$" + val_hr + "$")#r"^{" + str(del1) + r"} _{" + str(del2)) + "} $"
    else:
        print(group)
        raise ValueError("\t 7. Unrecognized resoltion setup \n "
                         "ress:{} v_n:{} err:{}".format(ress, v_n, err))
    # exit(1)

def mscatter(x, y, ax=None, m=None, **kw):
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

unique = sorted(list(set(plot_sims["group"])))

# collect data for plotting
x_arr = np.zeros(3)
y_arr = np.zeros(3)
col_arr = []
marker_arr = []
for u in unique:
    group = plot_sims[plot_sims.group == u]
    x, del1, del2 = get_group_value3(group, x_dic)
    x_arr = np.vstack((x_arr, [x, del1, del2]))
    y, del1, del2 = get_group_value3(group, y_dic)
    y_arr = np.vstack((y_arr, [y, del1, del2]))
    col, _, _ = get_group_value3(group, col_dic)
    col_arr.append(col)
    # print(u)
    tcolls = list(group["tcoll_gw"])
    marker = mc_st
    for t_ in tcolls:
        if np.isfinite(t_):
            marker = mc_bh
            if t_ < t_pc:
                marker = mc_pc
    # mk = list(group[marker_v_n])[0]
    marker_arr.append(marker)
    print("u:{} \t x:{:.1f} y:{:.2f} col:{:.1f} mk:{}"
          .format(u, x, y, col, marker))

col_arr = np.array(col_arr)
x_arr = np.delete(x_arr, 0, 0)
y_arr = np.delete(y_arr, 0, 0)
#
print("\tdata is located {} {} {}".format(x_arr.shape, y_arr.shape, col_arr.shape))
# exit(1)

# plotting


translation = {
    "Lambda": "Lambda",
    "Mej_tot-geo":"Mej"
}
if load_davids_data:
    from models_radice import simulations, fiducial
    plot_sims = simulations[fiducial]
    # print(plot_sims.keys())
    #
    x_arr_david = plot_sims[translation[x_dic["v_n"]]]
    y_arr_david = plot_sims[translation[y_dic["v_n"]]]
    x_arr_david = __apply_mod(x_dic["v_n"], x_arr_david, x_dic["mod"])
    y_arr_david = __apply_mod(y_dic["v_n"], y_arr_david, y_dic["mod"])
    assert len(x_arr_david) == len(y_arr_david)
else:
    x_arr_david = []
    y_arr_david = []



# print(x_arr)


fig = plt.figure(figsize=[4, 2.5]) # <-> |
ax = fig.add_subplot(111)
# ax = fig.add_axes([0.14, 0.15, 1.0 - 0.14 * 2, 0.95 - 0.15])

if load_davids_data:
    ax.scatter(x_arr_david, y_arr_david, marker="3", s = 20,
               color="gray", alpha=1., label="Radice+2018")



''' -----------------| PLOTTING |----------------------- '''

# for labels
ax.scatter([0], [y_arr[0,0]], marker=mc_pc,
        color="gray", alpha=1., label="Prompt Collapse")
ax.scatter([0], [y_arr[0,0]], marker=mc_st,
        color="gray", alpha=1., label="Stable remnant")
ax.scatter([0], [y_arr[0,0]], marker=mc_bh,
        color="gray", alpha=1., label="Black Hole")

# main body
cm = plt.cm.get_cmap(plot_dic["cmap"])
norm = Normalize(vmin=plot_dic["vmin"], vmax=plot_dic["vmax"])
sc = mscatter(x_arr[:,0], y_arr[:,0], ax=ax, c=col_arr, norm=norm,
                          s=plot_dic['ms'], cmap=cm, m=marker_arr,
                          label=plot_dic['label'], alpha=plot_dic['alpha'])

# error bars
ax.errorbar(x_arr[:,0], y_arr[:,0], yerr=y_arr[:,1], color='gray', ecolor='gray',
            fmt='None', elinewidth=1, capsize=1, alpha=0.6)

# ticks
ax.tick_params(axis='both', which='both', labelleft=True,
               labelright=False, tick1On=True, tick2On=True,
               labelsize=12,
               direction='in',
               bottom=True, top=True, left=True, right=True)
ax.minorticks_on()
min_, max_ = get_minmax(y_dic["v_n"], y_arr, extra=2.)
ax.set_ylim(min_, max_)
if load_davids_data:
    ax.set_xlim(20, 1500)
else:
    ax.set_xlim(380, 880)

# label
ax.set_xlabel(get_label(x_dic["v_n"]))
ax.set_ylabel(get_label(y_dic["v_n"]))

# colobar
ax.legend(fancybox=True, loc='lower right', bbox_to_anchor=(0.5, 0.5),#loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
         shadow=False, ncol=1, fontsize=9,
         framealpha=0., borderaxespad=0.)
clb = plt.colorbar(sc)
clb.ax.set_title(r"$M_1/M_2$", fontsize=11)
clb.ax.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig(__outplotdir__ + "summary.png", dpi=256)
plt.close()

