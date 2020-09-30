#
# prints table in tex format
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
import ast

''' Parameters '''
# csv_table = "../output/models3.csv"

''' Modules '''

def get_label(v_n):
    if v_n == "Mdisk3D":
        return r"$M_{\text{disk}} ^{\text{last}}$"
    elif v_n == "Mdisk":
        return r"$M_{\text{disk}} ^{\text{BH}}$"
    elif v_n == "M1":
        return "$M_a$"
    elif v_n == "M2":
        return "$M_b$"
    elif v_n == "tcoll_gw" or v_n == "tcoll":
        return r"$t_{\text{BH}}$"
    elif v_n == "tend":
        return r"$t_{\text{end}}$"
    elif v_n == "tend_wind":
        return r"$t_{\text{wind}}$"
    elif v_n == "tdisk3D":
        return r"$t_{\text{disk}}$"
    elif v_n == "q":
        return r"$M_a/M_b$"
    elif v_n == "EOS":
        return r"EOS"
    elif v_n == "group":
        return "Model"
    elif v_n == "res" or v_n == "resolution":
        return r"res"
    elif v_n == "vis" or v_n == "viscosity":
        return "GRLES"
    elif v_n == "note":
        return r"note"
    elif v_n == "Mbi/Mb" or v_n == "Mbi_Mb":
        return r"$Mb_i/Mb$"
    elif v_n == "note" or v_n == "comment":
        return "Note"
    elif v_n == "Lambda" or v_n == r"$\tilde{\Lambda}$":
        return "Note"

    if v_n == "Mwinddot":
        return r"$M_{\rm ej}^{\rm w}/\Delta t$"

    if v_n == "theta_rms" + '-' + "geo":
        return "$\\langle \\theta_{\\text{ej}} ^{\\text{d}} \\rangle$"
    elif v_n == "Mej_tot" + '-' + "geo":
        return "$M_{\\text{ej}} ^{\\text{d}}$"
    elif v_n == "Ye_ave" + '-' + "geo":
        return "$\\langle Y_e ^{\\text{d}} \\rangle$"
    elif v_n == "vel_inf_ave" + '-' + "geo":
        return "$\\langle \\upsilon_{\\text{ej}} ^{\\text{d}} \\rangle$"
    elif v_n == "theta_rms" + '-' + "bern_geoend":
        return "$\\langle \\theta_{\\text{ej}}^{\\text{w}} \\rangle$"
    elif v_n == "Mej_tot" + '-' + "bern_geoend":
        return "$M_{\\text{ej}}^{\\text{w}}$"
    elif v_n == "Ye_ave" + '-' + "bern_geoend":
        return "$\\langle Y_e ^{\\text{w}}  \\rangle$"
    elif v_n == "vel_inf_ave" + '-' + "bern_geoend":
        return "$\\langle \\upsilon_{\\text{ej}}^{\\text{w}} \\rangle$"
    elif v_n == "Mej_tot" + '-' + "theta60_geoend":
        return "$M_{\\text{ej}}^{\\theta>60}$"
    elif v_n == "Ye_ave" + '-' + "theta60_geoend":
        return "$\\langle Y_e ^{\\theta>60}  \\rangle$"
    elif v_n == "vel_inf_ave" + '-' + "theta60_geoend":
        return "$\\langle \\upsilon_{\\text{ej}}^{\\theta>60} \\rangle$"
    elif v_n == "Mej_tot" + '-' + "Y_e04_geoend":
        return "$M_{\\text{ej}}^{Ye>0.4}$"
    elif v_n == "Ye_ave" + '-' + "Y_e04_geoend":
        return "$\\langle Y_e ^{Ye>0.4}  \\rangle$"
    elif v_n == "vel_inf_ave" + '-' + "Y_e04_geoend":
        return "$\\langle \\upsilon_{\\text{ej}}^{Ye>0.4} \\rangle$"
    elif v_n == "res":
        return "res"

    raise NameError("No label found for outflow v_n: {} ".format(v_n))


def get_unit_label(v_n):

    if v_n in ["M1", "M2"]:
        return "$[M_{\odot}]$"
    elif v_n == "Mwinddot":
        return r"$[M_{\odot}/s]$"
    elif v_n.__contains__("Mej_tot"):
        return "$[10^{-2} M_{\odot}]$"
    elif v_n.__contains__("vel_inf_ave"):
        return "$[c]$"
    elif v_n in ["tcoll_gw", "tmerg_gw", "tmerg", "tcoll", "tend", "tdisk3D", "tend_wind"]:
        return "[ms]"
    else:
        return " "

    # if v_n in ["M1", "M2"]:
    #     return "$[M_{\odot}]$"
    # elif v_n in ["Mej_tot"]:
    #     return "$[10^{-2} M_{\odot}]$"
    # elif v_n in ["Mdisk3D", "Mdisk"]:
    #     return "$[M_{\odot}]$"
    # elif v_n in ["vel_inf_ave"]:
    #     return "$[c]$"
    # elif v_n in ["tcoll_gw", "tmerg_gw", "tmerg", "tcoll", "tend", "tdisk3D"]:
    #     return "[ms]"
    # else:
    #     return " "


def get_string_value(model_dic, task_dic):
    #
    v_n = task_dic["v_n"]
    fmt = task_dic["fmt"]
    mod = task_dic["mod"]
    err = task_dic["err"]
    deferr = task_dic["deferr"]
    #
    val = model_dic[v_n]
    if mod != None and mod != "":
        if mod[0] == '*':
            mult_by = float(mod.split("*")[-1])
            val = val * mult_by
        elif mod[0] == "/":
            dev_by = float(mod.split('/')[-1])
            val = val / dev_by
        else:
            raise NameError("v_n:{} mod:{} is invalid. Use mod '*float' or '/float' "
                            .format(v_n, mod))

    if fmt != None and fmt != "":
        val = ("%{}".format(fmt) % val)
    else:
        if str(val) == "nan":
            val = " "
    return val


def standard_div(x_arr):
    x_arr = np.array(x_arr, dtype=float)
    n = 1. * len(x_arr)
    mean = sum(x_arr) / n
    tmp = (1 / (n-1)) * np.sum((x_arr - mean) ** 2)
    return mean, np.sqrt(tmp)


def __apply_mod(v_n, val, mod):
    # print(v_n, val, mod)
    if mod != None and mod != "":
        # exit(1)
        val = np.float(val)

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


def __apply_fmt(v_n, val, fmt, color="black"):
    if fmt != None and fmt != "":
        _val = str(("%{}".format(fmt) % float(val)))
    else:
        if str(val) == "nan":
            _val = " "
            # exit(1)
        else:
            _val = val
    #
    if color != "black":
        return r"\textcolor{"+color+"}{" + _val + "}"
    else:
        return _val


def get_group_string_value(group, task_dic):

    v_n = task_dic["v_n"]
    fmt = task_dic["fmt"]
    mod = task_dic["mod"]
    err = task_dic["err"]
    deferr = task_dic["deferr"]

    # print(type(group.resolution)); print(list(group.resolution)); exit(1)

    ress = list(group.resolution)
    if len(ress) > 3:
        raise ValueError("too many resolutions! {}".format(group.resolutons))

    if "SR" in ress and "LR" in ress and "HR" in ress:
        # HR - SR , SR - LR
        print("\t 1 ress:{} v_n:{} fmt:{} err:{}".format(ress, v_n, fmt, err))
        if v_n == "group":
            val = list(group.group)[0]
            val = val.replace("_", "--") + r"\textit{[S,H,L]}"
            return val
        elif v_n == "resolution":
            val = r"\texttt{" +  " ".join(ress) + "}"
            return val
        elif fmt == None or fmt == "":
            sr = group[group.resolution == "SR"]
            val_sr = str(list(sr[v_n])[0])
            val_sr = __apply_fmt(v_n, val_sr,fmt)
            return val_sr
        elif err == None or err == "":
            sr = group[group.resolution == "SR"]
            val_sr = float(list(sr[v_n])[0])
            val_sr = __apply_fmt(v_n, val_sr, fmt)
            return val_sr
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
            val_sr = __apply_mod(v_n, val_sr, mod)
            val_lr = __apply_mod(v_n, val_lr, mod)
            val_hr = __apply_mod(v_n, val_hr, mod)
            #
            del1 = float(val_sr - val_hr)
            del2 = float(val_sr - val_lr)
            #

            #
            val_sr = __apply_fmt(v_n, val_sr, fmt)
            del1   = __apply_fmt(v_n, del1, fmt)
            del2   = __apply_fmt(v_n, del2, fmt)
            #
            # print(str(val_sr + r"^{" + str(del1) + r"} _{" + str(del2)) + "}")
            #
            return str("$" + val_sr + r"^{" + str(del1) + r"} _{" + str(del2)) + "} $"
    elif len(ress) == 2 and "SR" in ress and "LR" in ress:
        # SR - LR, SR - LR
        print("\t 2 ress:{} v_n:{} fmt:{} err:{}".format(ress, v_n, fmt, err))
        if v_n == "group":
            val = list(group.group)[0]
            val = val.replace("_", "--") + r"\textit{[S,L]}"
            return val
        elif v_n == "resolution":
            val = r"\texttt{" +  " ".join(ress) + "}"
            return val
        elif fmt == None or fmt == "":
            sr = group[group.resolution == "SR"]
            val_sr = str(list(sr[v_n])[0])
            val_sr = __apply_fmt(v_n, val_sr, fmt)
            return val_sr
        elif err == None or err == "":
            sr = group[group.resolution == "SR"]
            val_sr = float(list(sr[v_n])[0])
            val_sr = __apply_fmt(v_n, val_sr, fmt)
            return val_sr
        else:
            # hr = group[group.resolution == "HR"]
            sr = group[group.resolution == "SR"]
            lr = group[group.resolution == "LR"]
            #
            # print(hr); exit(1)
            #
            val_sr = float(sr[v_n])
            val_lr = float(lr[v_n])
            # val_hr = float(hr[v_n])
            #
            # print(float(val_sr)); exit(1)
            #
            val_sr = __apply_mod(v_n, val_sr, mod)
            val_lr = __apply_mod(v_n, val_lr, mod)
            # val_hr = __apply_mod(v_n, val_hr, mod)
            #
            # del1 = float(val_sr - val_hr)
            del2 = float(val_sr - val_lr)
            #

            #
            val_sr = __apply_fmt(v_n, val_sr, fmt)
            # del1 = __apply_fmt(v_n, del1, fmt)
            del2 = __apply_fmt(v_n, del2, fmt)
            #
            # print(str(val_sr + r"^{" + str(del1) + r"} _{" + str(del2)) + "}")
            #
            return str("$" + val_sr + " _{" + str(del2) + "} $")
    elif len(ress) == 2 and "SR" in ress and "HR" in ress:
        print("\t 3 ress:{} v_n:{} fmt:{} err:{}".format(ress, v_n, fmt, err))
        if v_n == "group":
            val = list(group.group)[0]
            val = val.replace("_", "--") + r"\textit{[S,H]}"
            return val
        elif v_n == "resolution":
            val = r"\texttt{" +  " ".join(ress) + "}"
            return val
        elif fmt == None or fmt == "":
            sr = group[group.resolution == "SR"]
            val_sr = str(list(sr[v_n])[0])
            val_sr = __apply_fmt(v_n, val_sr, fmt)
            return val_sr
        elif err == None or err == "":
            sr = group[group.resolution == "SR"]
            val_sr = float(list(sr[v_n])[0])
            val_sr = __apply_fmt(v_n, val_sr, fmt)
            return val_sr
        else:
            hr = group[group.resolution == "HR"]
            sr = group[group.resolution == "SR"]
            # lr = group[group.resolution == "LR"]
            #
            # print(hr); exit(1)
            #
            val_sr = float(sr[v_n])
            # val_lr = float(lr[v_n])
            val_hr = float(hr[v_n])
            #
            # print(float(val_sr)); exit(1)
            #
            val_sr = __apply_mod(v_n, val_sr, mod)
            # val_lr = __apply_mod(v_n, val_lr, mod)
            val_hr = __apply_mod(v_n, val_hr, mod)
            #
            del1 = float(val_sr - val_hr)
            # del2 = float(val_sr - val_lr)
            #

            #
            val_sr = __apply_fmt(v_n, val_sr, fmt)
            del1 = __apply_fmt(v_n, del1, fmt)
            # del2 = __apply_fmt(v_n, del2, fmt)
            #
            # print(str(val_sr + r"^{" + str(del1) + r"} _{" + str(del2)) + "}")
            #
            return str("$" + val_sr + r"^{" + str(del1) + r"} $") #_{" + str(del2)) + "} $"
        pass
    elif len(ress) == 2 and "LR" in ress and "HR" in ress:
        print("\t 3 ress:{} v_n:{} fmt:{} err:{}".format(ress, v_n, fmt, err))
        if v_n == "group":
            val = list(group.group)[0]
            val = val.replace("_", "--") + r"\textit{[L,H]}"
            return val
        elif v_n == "resolution":
            val = r"\texttt{" +  " ".join(ress) + "}"
            return val
        elif fmt == None or fmt == "":
            lr = group[group.resolution == "LR"]
            val_lr = str(list(lr[v_n])[0])
            val_lr = __apply_fmt(v_n, val_lr, fmt)
            return val_lr
        elif err == None or err == "":
            lr = group[group.resolution == "LR"]
            val_lr = float(list(lr[v_n])[0])
            val_lr = __apply_fmt(v_n, val_lr, fmt)
            return val_lr
        else:
            hr = group[group.resolution == "HR"]
            lr = group[group.resolution == "LR"]
            # lr = group[group.resolution == "LR"]
            #
            # print(hr); exit(1)
            #
            val_lr = float(lr[v_n])
            # val_lr = float(lr[v_n])
            val_hr = float(hr[v_n])
            #
            # print(float(val_sr)); exit(1)
            #
            val_lr = __apply_mod(v_n, val_lr, mod)
            # val_lr = __apply_mod(v_n, val_lr, mod)
            val_hr = __apply_mod(v_n, val_hr, mod)
            #
            del1 = float(val_lr - val_hr)
            # del2 = float(val_sr - val_lr)
            #

            #
            val_lr = __apply_fmt(v_n, val_lr, fmt)
            del1 = __apply_fmt(v_n, del1, fmt)
            # del2 = __apply_fmt(v_n, del2, fmt)
            #
            # print(str(val_sr + r"^{" + str(del1) + r"} _{" + str(del2)) + "}")
            #
            return str("$" + val_lr + r"^{" + str(del1) + r"} $") #_{" + str(del2)) + "} $"
        pass
    elif len(ress) == 1 and "SR" in ress:
        # SR - X*SR, SR - X*SR
        print("\t 4 ress:{} v_n:{} fmt:{} err:{}".format(ress, v_n, fmt, err))
        if v_n == "group":
            val = list(group.group)[0]
            val = val.replace("_", "--") + r"\textit{[S]}"
            return val
        elif v_n == "resolution":
            val = r"\texttt{" +  " ".join(ress) + "}"
            return val
        elif fmt == None or fmt == "":
            sr = group[group.resolution == "SR"]
            val_sr = str(list(sr[v_n])[0])
            val_sr = __apply_fmt(v_n, val_sr, fmt)
            return val_sr
        elif err == None or err == "":
            sr = group[group.resolution == "SR"]
            val_sr = float(list(sr[v_n])[0])
            val_sr = __apply_fmt(v_n, val_sr, fmt)
            return val_sr
        else:
            # hr = group[group.resolution == "HR"]
            sr = group[group.resolution == "SR"]
            # lr = group[group.resolution == "LR"]
            #
            # print(hr); exit(1)
            #
            val_sr = float(sr[v_n])
            # val_lr = float(lr[v_n])
            # val_hr = float(hr[v_n])
            #
            # print(float(val_sr)); exit(1)
            #
            val_sr = __apply_mod(v_n, val_sr, mod)
            # val_lr = __apply_mod(v_n, val_lr, mod)
            # val_hr = __apply_mod(v_n, val_hr, mod)
            #
            # del1 = float(val_sr - val_hr)
            # del2 = float(val_sr - val_lr)
            #

            #
            val_sr = __apply_fmt(v_n, val_sr, fmt)
            # del1 = __apply_fmt(v_n, del1, fmt)
            # del2 = __apply_fmt(v_n, del2, fmt)
            #
            # print(str(val_sr + r"^{" + str(del1) + r"} _{" + str(del2)) + "}")
            #
            return str("$" + val_sr + "$")#+ r"^{" + str(del1) + r"} _{" + str(del2)) + "} $"
    elif len(ress) == 1 and "LR" in ress:
        # LR - Y*LR, SR - Y*LR
        print("\t 5 ress:{} v_n:{} fmt:{} err:{}".format(ress, v_n, fmt, err))
        if v_n == "group":
            val = list(group.group)[0]
            val = val.replace("_", "--") + r"\textit{[L]}"
            return val
        elif v_n == "resolution":
            val = r"\texttt{" +  " ".join(ress) + "}"
            return val
        elif fmt == None or fmt == "":
            lr = group[group.resolution == "LR"]
            val_lr = str(list(lr[v_n])[0])
            val_lr = __apply_fmt(v_n, val_lr, fmt)
            return val_lr
        elif err == None or err == "":
            lr = group[group.resolution == "LR"]
            val_lr = float(list(lr[v_n])[0])
            val_lr = __apply_fmt(v_n, val_lr, fmt)
            return val_lr
        else:
            # hr = group[group.resolution == "HR"]
            # sr = group[group.resolution == "SR"]
            lr = group[group.resolution == "LR"]
            #
            # print(hr); exit(1)
            #
            # val_sr = float(sr[v_n])
            val_lr = float(lr[v_n])
            # val_hr = float(hr[v_n])
            #
            # print(float(val_sr)); exit(1)
            #
            # val_sr = __apply_mod(v_n, val_sr, mod)
            val_lr = __apply_mod(v_n, val_lr, mod)
            # val_hr = __apply_mod(v_n, val_hr, mod)
            #
            # del1 = float(val_sr - val_hr)
            # del2 = float(val_sr - val_lr)
            #

            #
            val_lr = __apply_fmt(v_n, val_lr, fmt)
            # del1 = __apply_fmt(v_n, del1, fmt)
            # del2 = __apply_fmt(v_n, del2, fmt)
            #
            # print(str(val_sr + r"^{" + str(del1) + r"} _{" + str(del2)) + "}")
            #
            return str("$" + val_lr + "$")#r"^{" + str(del1) + r"} _{" + str(del2)) + "} $"
    elif len(ress) == 1 and "HR" in ress:
        # HR - Z*HR, SR - Z*HR
        print("\t 6 ress:{} v_n:{} fmt:{} err:{}".format(ress, v_n, fmt, err))
        if v_n == "group":
            val = list(group.group)[0]
            val = val.replace("_", "--") + r"\textit{[H]}"
            return val
        elif v_n == "resolution":
            val = r"\texttt{" +  " ".join(ress) + "}"
            return val
        elif fmt == None or fmt == "":
            hr = group[group.resolution == "HR"]
            val_hr = str(list(hr[v_n])[0])
            val_hr = __apply_fmt(v_n, val_hr, fmt)
            return val_hr
        elif err == None or err == "":
            hr = group[group.resolution == "HR"]
            val_hr = float(list(hr[v_n])[0])
            val_hr = __apply_fmt(v_n, val_hr, fmt)
            return val_hr
        else:
            hr = group[group.resolution == "HR"]
            # sr = group[group.resolution == "SR"]
            # lr = group[group.resolution == "LR"]
            #
            # print(hr); exit(1)
            #
            # val_sr = float(sr[v_n])
            # val_lr = float(lr[v_n])
            val_hr = float(hr[v_n])
            #
            # print(float(val_sr)); exit(1)
            #
            # val_sr = __apply_mod(v_n, val_sr, mod)
            # val_lr = __apply_mod(v_n, val_lr, mod)
            val_hr = __apply_mod(v_n, val_hr, mod)
            #
            # del1 = float(val_sr - val_hr)
            # del2 = float(val_sr - val_lr)
            #

            #
            val_hr = __apply_fmt(v_n, val_hr, fmt)
            # del1 = __apply_fmt(v_n, del1, fmt)
            # del2 = __apply_fmt(v_n, del2, fmt)
            #
            # print(str(val_sr + r"^{" + str(del1) + r"} _{" + str(del2)) + "}")
            #
            return str("$" + val_hr + "$")#r"^{" + str(del1) + r"} _{" + str(del2)) + "} $"
    else:
        print(group)
        raise ValueError("\t 7. Unrecognized resoltion setup \n "
                         "ress:{} v_n:{} fmt:{} err:{}".format(ress, v_n, fmt, err))
    # exit(1)


def get_group_string_value2(group, task_dic, color_conv="green", color_notconv="red"):

    v_n = task_dic["v_n"]
    fmt = task_dic["fmt"]
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
        elif fmt == None or fmt == "":
            sr = group[group.resolution == "SR"]
            val_sr = str(list(sr[v_n])[0])
            val_sr = __apply_fmt(v_n, val_sr,fmt)
            return val_sr
        elif err == None or err == "":
            sr = group[group.resolution == "SR"]
            val_sr = float(list(sr[v_n])[0])
            val_sr = __apply_mod(v_n, val_sr, mod)
            val_sr = __apply_fmt(v_n, val_sr, fmt)
            return val_sr
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
                del1 = val_lr - val_sr # +
                del2 = val_sr - val_hr # -
            elif val_hr > val_sr and val_sr > val_lr:
                # oppoiste of convergence:
                del1 = val_sr - val_lr # +
                del2 = val_sr - val_hr # -
            elif val_hr < val_sr and val_sr > val_lr:
                # SR is the top
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

            if conv:
                val_sr = __apply_fmt(v_n, val_sr, fmt, color_conv)
            else:
                val_sr = __apply_fmt(v_n, val_sr, fmt, color_notconv)

            # mean, stddiv = standard_div([val_hr, val_sr, val_lr])

            del1 = np.abs(del1)
            del2 = np.abs(del2)

            del1   = __apply_fmt(v_n, del1, fmt)
            del2   = __apply_fmt(v_n, del2, fmt)

            return str("$" + val_sr + r"^{+" + str(del1) + r"} _{-" + str(del2)) + "} $"
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
        elif fmt == None or fmt == "":
            sr = group[group.resolution == "SR"]
            val_sr = str(list(sr[v_n])[0])
            val_sr = __apply_fmt(v_n, val_sr, fmt)
            return val_sr
        elif err == None or err == "":
            sr = group[group.resolution == "SR"]
            val_sr = float(list(sr[v_n])[0])
            val_sr = __apply_mod(v_n, val_sr, mod)
            val_sr = __apply_fmt(v_n, val_sr, fmt)
            return val_sr
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


            val_sr = __apply_fmt(v_n, val_sr, fmt)

            del1 = __apply_fmt(v_n, del1, fmt)
            del2 = __apply_fmt(v_n, del2, fmt)

            return str("$" + val_sr + r"^{+" + str(del1) + r"} _{-" + str(del2)) + "} $"
    elif len(ress) == 2 and "SR" in ress and "HR" in ress:
        # print("\t 3 ress:{} v_n:{} fmt:{} err:{}".format(ress, v_n, fmt, err))
        if v_n == "group":
            val = list(group.group)[0]
            val = val.replace("_", "--") + r"\textit{[S,H]}"
            return val
        elif v_n == "resolution":
            val = r"\texttt{" +  " ".join(ress) + "}"
            return val
        elif fmt == None or fmt == "":
            sr = group[group.resolution == "SR"]
            val_sr = str(list(sr[v_n])[0])
            val_sr = __apply_fmt(v_n, val_sr, fmt)
            return val_sr
        elif err == None or err == "":
            sr = group[group.resolution == "SR"]
            val_sr = float(list(sr[v_n])[0])
            val_sr = __apply_mod(v_n, val_sr, mod)
            val_sr = __apply_fmt(v_n, val_sr, fmt)
            return val_sr
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
            val_sr = __apply_fmt(v_n, val_sr, fmt)

            del1 = __apply_fmt(v_n, del1, fmt)
            del2 = __apply_fmt(v_n, del2, fmt)

            return str("$" + val_sr + r"^{+" + str(del1) + r"} _{-" + str(del2)) + "} $"
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
        elif fmt == None or fmt == "":
            lr = group[group.resolution == "LR"]
            val_lr = str(list(lr[v_n])[0])
            val_lr = __apply_fmt(v_n, val_lr, fmt)
            return val_lr
        elif err == None or err == "":
            lr = group[group.resolution == "LR"]
            val_lr = float(list(lr[v_n])[0])
            val_lr = __apply_mod(v_n, val_lr, mod)
            val_lr = __apply_fmt(v_n, val_lr, fmt)
            return val_lr
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

            val_sr = __apply_fmt(v_n, val_hr, fmt)

            del1 = __apply_fmt(v_n, del1, fmt)
            del2 = __apply_fmt(v_n, del2, fmt)

            return str("$" + val_sr + r"^{+" + str(del1) + r"} _{-" + str(del2)) + "} $"
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
        elif fmt == None or fmt == "":
            sr = group[group.resolution == "SR"]
            val_sr = str(list(sr[v_n])[0])
            val_sr = __apply_fmt(v_n, val_sr, fmt)
            return val_sr
        elif err == None or err == "":
            sr = group[group.resolution == "SR"]
            val_sr = float(list(sr[v_n])[0])
            val_sr = __apply_mod(v_n, val_sr, mod)
            val_sr = __apply_fmt(v_n, val_sr, fmt)
            return val_sr
        else:
            # hr = group[group.resolution == "HR"]
            sr = group[group.resolution == "SR"]
            # lr = group[group.resolution == "LR"]
            #
            # print(hr); exit(1)
            #
            val_sr = float(sr[v_n])
            # val_lr = float(lr[v_n])
            # val_hr = float(hr[v_n])
            #
            # print(float(val_sr)); exit(1)
            #
            val_sr = __apply_mod(v_n, val_sr, mod)
            # val_lr = __apply_mod(v_n, val_lr, mod)
            # val_hr = __apply_mod(v_n, val_hr, mod)
            #
            # del1 = float(val_sr - val_hr)
            # del2 = float(val_sr - val_lr)
            #

            #
            val_sr = __apply_fmt(v_n, val_sr, fmt)
            # del1 = __apply_fmt(v_n, del1, fmt)
            # del2 = __apply_fmt(v_n, del2, fmt)
            #
            # print(str(val_sr + r"^{" + str(del1) + r"} _{" + str(del2)) + "}")
            #
            return str("$" + val_sr + "$")#+ r"^{" + str(del1) + r"} _{" + str(del2)) + "} $"
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
        elif fmt == None or fmt == "":
            lr = group[group.resolution == "LR"]
            val_lr = str(list(lr[v_n])[0])
            val_lr = __apply_fmt(v_n, val_lr, fmt)
            return val_lr
        elif err == None or err == "":
            lr = group[group.resolution == "LR"]
            val_lr = float(list(lr[v_n])[0])
            val_lr = __apply_mod(v_n, val_lr, mod)
            val_lr = __apply_fmt(v_n, val_lr, fmt)
            return val_lr
        else:
            # hr = group[group.resolution == "HR"]
            # sr = group[group.resolution == "SR"]
            lr = group[group.resolution == "LR"]
            #
            # print(hr); exit(1)
            #
            # val_sr = float(sr[v_n])
            val_lr = float(lr[v_n])
            # val_hr = float(hr[v_n])
            #
            # print(float(val_sr)); exit(1)
            #
            # val_sr = __apply_mod(v_n, val_sr, mod)
            val_lr = __apply_mod(v_n, val_lr, mod)
            # val_hr = __apply_mod(v_n, val_hr, mod)
            #
            # del1 = float(val_sr - val_hr)
            # del2 = float(val_sr - val_lr)
            #

            #
            val_lr = __apply_fmt(v_n, val_lr, fmt)
            # del1 = __apply_fmt(v_n, del1, fmt)
            # del2 = __apply_fmt(v_n, del2, fmt)
            #
            # print(str(val_sr + r"^{" + str(del1) + r"} _{" + str(del2)) + "}")
            #
            return str("$" + val_lr + "$")#r"^{" + str(del1) + r"} _{" + str(del2)) + "} $"
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
        elif fmt == None or fmt == "":
            hr = group[group.resolution == "HR"]
            val_hr = str(list(hr[v_n])[0])
            val_hr = __apply_fmt(v_n, val_hr, fmt)
            return val_hr
        elif err == None or err == "":
            hr = group[group.resolution == "HR"]
            val_hr = float(list(hr[v_n])[0])
            val_hr = __apply_mod(v_n, val_hr, mod)
            val_hr = __apply_fmt(v_n, val_hr, fmt)
            return val_hr
        else:
            hr = group[group.resolution == "HR"]
            # sr = group[group.resolution == "SR"]
            # lr = group[group.resolution == "LR"]
            #
            # print(hr); exit(1)
            #
            # val_sr = float(sr[v_n])
            # val_lr = float(lr[v_n])
            val_hr = float(hr[v_n])
            #
            # print(float(val_sr)); exit(1)
            #
            # val_sr = __apply_mod(v_n, val_sr, mod)
            # val_lr = __apply_mod(v_n, val_lr, mod)
            val_hr = __apply_mod(v_n, val_hr, mod)
            #
            # del1 = float(val_sr - val_hr)
            # del2 = float(val_sr - val_lr)
            #

            #
            val_hr = __apply_fmt(v_n, val_hr, fmt)
            # del1 = __apply_fmt(v_n, del1, fmt)
            # del2 = __apply_fmt(v_n, del2, fmt)
            #
            # print(str(val_sr + r"^{" + str(del1) + r"} _{" + str(del2)) + "}")
            #
            return str("$" + val_hr + "$")#r"^{" + str(del1) + r"} _{" + str(del2)) + "} $"
    else:
        print(group)
        raise ValueError("\t 7. Unrecognized resoltion setup \n "
                         "ress:{} v_n:{} fmt:{} err:{}".format(ress, v_n, fmt, err))
    # exit(1)

# used for group table
def get_fancy_str_value(group, task_dic, color_conv=None, color_notconv=None, err_v_n="err-"):

    #
    v_n = task_dic["v_n"]
    fmt = task_dic["fmt"]
    mod = task_dic["mod"]
    err = task_dic["err"]
    deferr = task_dic["deferr"]

    # print(type(group.resolution)); print(list(group.resolution)); exit(1)

    res = str(group["resolution"])
    ress = res.split()
    assert len(ress) > 0 and len(ress) <= 3
    #
    if v_n == "res" or v_n == "resolution" or v_n == "resoltuions":
        return r"\texttt{" + res + "}"

    if v_n == "viscosity" or v_n == "vis" or v_n == "GRLES":
        lk = str(group["viscosity"])
        if lk == "LK": return r"\cmark"
        else: return r"\xmark"


    if err == None:
        val = group[v_n]
        val = __apply_mod(v_n, val, mod)
        val = __apply_fmt(v_n, val, fmt)
        return val
    elif err == "ud":
        mean = float(group[v_n])
        errp = errn = float(group[err_v_n + v_n])
        mean = __apply_mod(v_n, mean, mod)
        errp, errn = __apply_mod(err_v_n + v_n, errp, mod), __apply_mod(err_v_n + v_n, errn, mod)
        if color_conv != None and color_conv != "black":
            # TODO get a list of conv v_ns from the table. If v_n is in the list - pring green. else -- black
            raise NameError("Printing convergence is not set")
        else:
            mean = __apply_fmt(v_n, mean, fmt, "black")
        errp, errn = __apply_fmt(err_v_n + v_n, errp, fmt), __apply_fmt(err_v_n + v_n, errn, fmt)
        return str("$" + mean + r"^{+" + str(errp) + r"} _{-" + str(errn)) + "} $"
    elif err == "SR^HR_LR":
        #
        string = str(group[v_n]) # SR
        vals = string.strip('][').split(', ')
        # x = ast.literal_eval(x)
        # vals = [n.strip() for n in x]
        if not len(ress) == len(vals):
            raise ValueError("ress:{} array != vals:{} (v_n:{})"
                             .format(ress, vals, v_n))
        #
        val_sr, val_hr, val_lr = " ", " ", " "
        if "SR" in ress:
            val_sr = vals[ress.index("SR")]
            if fmt == "": pass
            else:
                val_sr = np.float(val_sr)
                if np.isnan(val_sr):
                    val_sr = "-"
                elif np.isfinite(val_sr):
                    val_sr = __apply_mod(v_n, val_sr, mod)
                    val_sr = __apply_fmt(v_n, val_sr, fmt)
                else: val_sr = r"\infty"
        if "HR" in ress:
            if fmt == "": pass
            else:
                val_hr = np.float(vals[ress.index("HR")])
                if np.isnan(val_hr):
                    val_hr = "-"
                elif np.isfinite(val_hr):
                    val_hr = __apply_mod(v_n, val_hr, mod)
                    val_hr = __apply_fmt(v_n, val_hr, fmt)
                else:
                    val_hr = r"\infty"
        if "LR" in ress:
            if fmt == "": pass
            else:
                val_lr = np.float(vals[ress.index("LR")])
                if np.isnan(val_lr):
                    val_lr = "-"
                elif np.isfinite(val_lr):
                    val_lr = __apply_mod(v_n, val_lr, mod)
                    val_lr = __apply_fmt(v_n, val_lr, fmt)
                else:
                    val_lr = r"\infty"
        #
        res = "$" + val_sr + "^{" + val_hr + "} _{" + val_lr + "}$"
        #
        return res

    elif err == "LR SR HR":
        #
        if v_n == "tcoll_gw":
            str_tend = str(group["tend"])
            tends =str_tend.strip('][').split(', ')
        else:
            tends = []

        string = str(group[v_n]) # SR
        vals = string.strip('][').split(', ')
        # x = ast.literal_eval(x)
        # vals = [n.strip() for n in x]
        if not len(ress) == len(vals):
            raise ValueError("ress:{} array != vals:{} (v_n:{}) -- data might be missing .csv"
                             .format(ress, vals, v_n))
        #
        val_sr, val_hr, val_lr = " ", " ", " "
        if "SR" in ress:
            val_sr = vals[ress.index("SR")]
            if fmt == "": pass
            else:
                val_sr = np.float(val_sr)
                if np.isnan(val_sr):
                    val_sr = "-"
                elif np.isfinite(val_sr):
                    val_sr = __apply_mod(v_n, val_sr, mod)
                    val_sr = __apply_fmt(v_n, val_sr, fmt)
                else:
                    if v_n == "tcoll_gw":
                        tend_sr = np.float(tends[ress.index("SR")])
                        tend_sr = __apply_mod("tend", tend_sr, mod)
                        tend_sr = __apply_fmt("tend", tend_sr, fmt)
                        val_sr = r">{}".format(tend_sr)
                    else:
                        val_sr = r"\text{N/A}"
        if "HR" in ress:
            if fmt == "": pass
            else:
                val_hr = np.float(vals[ress.index("HR")])
                if np.isnan(val_hr):
                    val_hr = "-"
                elif np.isfinite(val_hr):
                    val_hr = __apply_mod(v_n, val_hr, mod)
                    val_hr = __apply_fmt(v_n, val_hr, fmt)
                else:
                    if v_n == "tcoll_gw":
                        tend_hr = np.float(tends[ress.index("HR")])
                        tend_hr = __apply_mod("tend", tend_hr, mod)
                        tend_hr = __apply_fmt("tend", tend_hr, fmt)
                        val_hr = r">{}".format(tend_hr)
                    else:
                        val_hr = r"\text{N/A}"
        if "LR" in ress:
            if fmt == "": pass
            else:
                val_lr = np.float(vals[ress.index("LR")])
                if np.isnan(val_lr):
                    val_lr = "-"
                elif np.isfinite(val_lr):
                    val_lr = __apply_mod(v_n, val_lr, mod)
                    val_lr = __apply_fmt(v_n, val_lr, fmt)
                else:
                    if v_n == "tcoll_gw":
                        tend_lr = np.float(tends[ress.index("LR")])
                        tend_lr = __apply_mod("tend", tend_lr, mod)
                        tend_lr = __apply_fmt("tend", tend_lr, fmt)
                        val_lr = r">{}".format(tend_lr)
                    else:
                        val_lr = r"\text{N/A}"
        #
        res = '$' + val_lr + "$ $" + val_sr + "$ $" + val_hr + '$'
        #
        return res

''' Main '''

def basic_table(simulations):

    table_sims = simulations  # simulations[long_runs]

    plot_unique = True
    set_print_eos_separators = True
    set_print_q_separators = True
    grouped_by = ["EOS", "q"]

    tasks = [
        {"v_n": "EOS", "fmt": "", "mod": None, "err": None, "deferr": None},
        {"v_n": "q", "fmt": ".1f", "mod": None, "err": None, "deferr": None},
        {"v_n": "comment", "fmt": "", "mod": None, "err": None, "deferr": None},
        {"v_n": "resolution", "fmt": "", "mod": None, "err": None, "deferr": None},
        {"v_n": "viscosity", "fmt": "", "mod": None, "err": None, "deferr": None},
        # ---
        {"v_n": "Mbi_Mb", "fmt": ".2f", "mod": None, "err": None, "deferr": None},
        {"v_n": "tend", "fmt": ".1f", "mod": "*1e3", "err": None, "deferr": None},
        {"v_n": "tdisk3D", "fmt": ".1f", "mod": "*1e3", "err": None, "deferr": None},
        {"v_n": "Mdisk3D", "fmt": ".3f", "mod": None, "err": None, "deferr": None},
        # ---
        {"v_n": "Mej_tot-geo", "fmt": ".2f", "mod": "*1e2", "err": "ud", "deferr": None},
        {"v_n": "Ye_ave-geo", "fmt": ".2f", "mod": None, "err": "ud", "deferr": None},
        {"v_n": "vel_inf_ave-geo", "fmt": ".2f", "mod": None, "err": "ud", "deferr": None},
        {"v_n": "theta_rms-geo", "fmt": ".2f", "mod": None, "err": "ud", "deferr": None},
        # ---
        {"v_n": "Mej_tot-bern_geoend", "fmt": ".2f", "mod": "*1e2", "err": "ud", "deferr": None},
    ]

    # HEADER

    print("\n")
    size = '{'
    head = ''
    # i = 0

    for task_dic in tasks:
        v_n = get_label(task_dic["v_n"])
        size = size + 'c'
        head = head + '{}'.format(v_n)
        if task_dic != tasks[-1]: size = size + ' '
        if task_dic != tasks[-1]: head = head + ' & '
        # i = i + 1

    size = size + '}'

    head = head + ' \\\\'  # = \\

    # print(size)
    # print(head)

    # UNIT BAR

    unit_bar = ''
    for i, task in enumerate(tasks):
        unit = get_unit_label(task["v_n"])

        unit_bar = unit_bar + '{}'.format(unit)
        # if v_ns.index(v_n) != len(v_ns): unit_bar = unit_bar + ' & '
        if i != len(tasks) - 1: unit_bar = unit_bar + ' & '

    unit_bar = unit_bar + ' \\\\ '

    # print(unit_bar)

    # TABLE [ROWS] [NO GROUPED]

    rows = []
    eoss = []
    qs = {}
    for _, m in table_sims.iterrows():
        # m is a dictionary
        row = ''
        for i_t, task in enumerate(tasks):
            #
            val = get_string_value(m, task)
            row = row + val
            if task != tasks[-1]: row = row + " & "
        #
        if set_print_eos_separators and not m.EOS in eoss:
            eoss.append(m.EOS)
            qs[m.EOS] = []
            rows.append('\\hline')

        if set_print_q_separators and not "%.1f" % m.q in qs[m.EOS]:
            qs[m.EOS].append("%.1f" % m.q)
            rows.append('\\hline')

        row = row + ' \\\\'
        rows.append(row)

    " --- printing -- "

    print('\\begin{table*}[t]')
    print('\\begin{center}')
    print('\\begin{tabular}' + '{}'.format(size))
    print('\\hline')
    print(head)
    print(unit_bar)
    print('\\hline\\hline')

    for row in rows:
        print(row)

    print('\\hline')
    print('\\end{tabular}')
    print('\\end{center}')
    print('\\caption{I am your table! }')
    print('\\label{tbl:1}')
    print('\\end{table*}')


def unique_sim_table(groups, tasks, settings):
    # table_sims = groups
    # | \# (simulations.group == "BLh_M10651772_M0_LK")]  # simulations[long_runs]
    # print(table_sims.index)
    # unique = sorted(list(set(table_sims["group"])))

    # print(unique)

    set_print_eos_separators = settings["set_print_eos_separators"]
    set_print_q_separators = settings["set_print_q_separators"]

    # tasks = [
    #     # {"v_n": "group", "fmt": None, "mod": None, "err":None, "deferr":None},
    #     {"v_n": "EOS",          "fmt": "",      "mod": None,    "err": None,        "deferr": None},
    #     {"v_n": "q",            "fmt": ".2f",   "mod": None,    "err": None,        "deferr": None},
    #     #{"v_n": "comment",      "fmt": "",      "mod": None,    "err": None,        "deferr": None},
    #     {"v_n": "resolution",   "fmt": "",      "mod": None,    "err": None,        "deferr": None},
    #     {"v_n": "viscosity",    "fmt": "",      "mod": None,    "err": None,        "deferr": None},
    #     # ---
    #     #{"v_n": "Mbi_Mb",       "fmt": ".2f",   "mod": None,    "err": "SR^HR_LR",  "deferr": None},
    #     {"v_n": "tend",         "fmt": ".1f",   "mod": "*1e3",  "err": "LR SR HR",  "deferr": None},
    #     # {"v_n": "tcoll_gw",     "fmt": ".1f",   "mod": "*1e3",  "err": "SR^HR_LR",  "deferr":None},
    #     {"v_n": "tcoll_gw",     "fmt": ".1f",   "mod": "*1e3",  "err": "LR SR HR",  "deferr": None},
    #     {"v_n": "tdisk3D",      "fmt": ".1f",   "mod": "*1e3",  "err": None,        "deferr": None},
    #     {"v_n": "Mdisk3D",      "fmt": ".3f",   "mod": None,    "err": "ud",        "deferr": None},
    #     # ---
    #     {"v_n": "Mej_tot-geo",  "fmt": ".2f",   "mod": "*1e2",  "err": "ud",        "deferr": None},
    #     {"v_n": "Ye_ave-geo",   "fmt": ".2f",   "mod": None,    "err": "ud",        "deferr": None},
    #     {"v_n": "vel_inf_ave-geo", "fmt": ".2f", "mod": None,   "err": "ud",        "deferr": None},
    #     {"v_n": "theta_rms-geo", "fmt": ".2f",  "mod": None,    "err": "ud",        "deferr": None},
    #     # ---
    #     # {"v_n": "Mej_tot-bern_geoend", "fmt": ".2f", "mod": "*1e2", "err": "ud",    "deferr": None},
    # ]

    # HEADER

    print("\n")
    size = '{'
    head = ''
    # i = 0

    for task_dic in tasks:
        v_n = get_label(task_dic["v_n"])
        size = size + 'c'
        head = head + '{}'.format(v_n)
        if task_dic != tasks[-1]: size = size + ' '
        if task_dic != tasks[-1]: head = head + ' & '
        # i = i + 1

    size = size + '}'

    head = head + ' \\\\'  # = \\

    #

    unit_bar = ''
    for i, task in enumerate(tasks):
        unit = get_unit_label(task["v_n"])

        unit_bar = unit_bar + '{}'.format(unit)
        # if v_ns.index(v_n) != len(v_ns): unit_bar = unit_bar + ' & '
        if i != len(tasks) - 1: unit_bar = unit_bar + ' & '

    unit_bar = unit_bar + ' \\\\ '

    #

    rows = []
    eoss = []
    qs = {}
    for group_name, group_dic in groups.iterrows():
        row = ''
        for i_t, task in enumerate(tasks):
            #
            val = get_fancy_str_value(group_dic, task, color_conv=None, color_notconv=None, err_v_n="err-")
            print(task["v_n"], val)
            row = row + val
            if task != tasks[-1]: row = row + " & "
        #
        q = float(group_dic.q)
        eos = str(group_dic.EOS)
        #
        if set_print_eos_separators and not eos in eoss:
            eoss.append(eos)
            qs[eos] = []
            rows.append('\\hline')
        #
        if set_print_q_separators and not "%.1f" % q in qs[eos]:
            qs[eos].append("%.1f" % q)
            rows.append('\\hline')
        #
        row = row + ' \\\\'
        rows.append(row)

    " --- printing -- "
    print("\n")
    print('\\begin{table*}[t]')
    print('\\begin{center}')
    print('\\begin{tabular}' + '{}'.format(size))
    print('\\hline')
    print(head)
    print(unit_bar)
    #
    for row in rows:
        print(row)
    #
    print('\\hline')
    print('\\end{tabular}')
    print('\\end{center}')
    print('\\caption{I am your table! }')
    print('\\label{tbl:1}')
    print('\\end{table*}')


if __name__ == '__main__':

    fpath = "../datasets/groups.csv"
    print("loading {}...".format(fpath)),
    groups = pd.read_csv(fpath)
    groups.set_index("group")
    print("done.")


    ''' -------------------------- Main Table --------------------- '''

    settings = {
        "set_print_eos_separators": True,
        "set_print_q_separators": True
    }

    tasks = [
        # {"v_n":              "group", "fmt": None, "mod": None, "err":None, "deferr":None},
        {"v_n": "EOS",        "fmt": "", "mod": None, "err": None, "deferr": None},
        {"v_n": "q",          "fmt": ".2f", "mod": None, "err": None, "deferr": None},
        # {"v_n": "comment",    "fmt": "",      "mod": None,    "err": None,        "deferr": None},
        {"v_n": "resolution", "fmt": "", "mod": None, "err": None, "deferr": None},
        {"v_n": "viscosity",  "fmt": "", "mod": None, "err": None, "deferr": None},
        # ---
        # {"v_n": "Mbi_Mb",       "fmt": ".2f",   "mod": None,    "err": "SR^HR_LR",  "deferr": None},
        {"v_n": "tend", "fmt": ".1f", "mod": "*1e3", "err": "LR SR HR", "deferr": None},
        # {"v_n": "tcoll_gw",     "fmt": ".1f",   "mod": "*1e3",  "err": "SR^HR_LR",  "deferr":None},
        {"v_n": "tcoll_gw","fmt": ".1f", "mod": "*1e3", "err": "LR SR HR", "deferr": None},
        {"v_n": "tdisk3D", "fmt": ".1f", "mod": "*1e3", "err": None, "deferr": None},
        {"v_n": "Mdisk3D", "fmt": ".3f", "mod": None, "err": "ud", "deferr": None},
        # ---
        {"v_n": "Mej_tot-geo", "fmt": ".2f", "mod": "*1e2", "err": "ud", "deferr": None},
        {"v_n": "Ye_ave-geo", "fmt": ".2f", "mod": None, "err": "ud", "deferr": None},
        {"v_n": "vel_inf_ave-geo", "fmt": ".2f", "mod": None, "err": "ud", "deferr": None},
        {"v_n": "theta_rms-geo", "fmt": ".2f", "mod": None, "err": "ud", "deferr": None},
        # ---
        # {"v_n": "Mej_tot-bern_geoend", "fmt": ".2f", "mod": "*1e2", "err": "ud",    "deferr": None},
    ]
    # groups = groups
    #print(groups["group"] == "BLh_M10651772_M0_LK"); exit(1)
    # print("-"*20)
    # unique_sim_table(groups, tasks, settings)
    # unique_sim_table(groups, tasks, settings)

    ''' -------------------------- Wind Table --------------------- '''

    settings = {
        "set_print_eos_separators": True,
        "set_print_q_separators": True
    }

    tasks = [
        # {"v_n":              "group", "fmt": None, "mod": None, "err":None, "deferr":None},
        {"v_n": "EOS", "fmt": "", "mod": None, "err": None, "deferr": None},
        {"v_n": "q", "fmt": ".2f", "mod": None, "err": None, "deferr": None},
        {"v_n": "Lambda", "fmt": ".2f", "mod": None, "err": None, "deferr": None},
        # {"v_n": "comment",    "fmt": "",      "mod": None,    "err": None,        "deferr": None},
        {"v_n": "resolution", "fmt": "", "mod": None, "err": None, "deferr": None},
        {"v_n": "viscosity", "fmt": "", "mod": None, "err": None, "deferr": None},
        # ---
        # {"v_n": "Mbi_Mb",       "fmt": ".2f",   "mod": None,    "err": "SR^HR_LR",  "deferr": None},
        {"v_n": "tend", "fmt": ".1f", "mod": "*1e3", "err": "LR SR HR", "deferr": None},
        #{"v_n": "tend_wind", "fmt": ".1f", "mod": "*1e3", "err": None, "deferr": None},
        # {"v_n": "tcoll_gw",     "fmt": ".1f",   "mod": "*1e3",  "err": "SR^HR_LR",  "deferr":None},
        #{"v_n": "tcoll_gw", "fmt": ".1f", "mod": "*1e3", "err": "LR SR HR", "deferr": None},
        #{"v_n": "tdisk3D", "fmt": ".1f", "mod": "*1e3", "err": None, "deferr": None},
        #{"v_n": "Mdisk3D", "fmt": ".3f", "mod": None, "err": "ud", "deferr": None},
        # ---
        {"v_n": "Mej_tot-bern_geoend", "fmt": ".2f", "mod": "*1e2", "err": "ud", "deferr": None},
        {"v_n": "Mwinddot", "fmt": ".2f", "mod": None, "err": "ud", "deferr": None},
        {"v_n": "Ye_ave-bern_geoend", "fmt": ".2f", "mod": None, "err": "ud", "deferr": None},
        {"v_n": "vel_inf_ave-bern_geoend", "fmt": ".2f", "mod": None, "err": "ud", "deferr": None},
        {"v_n": "theta_rms-bern_geoend", "fmt": ".2f", "mod": None, "err": "ud", "deferr": None},
        # ---
        # {"v_n": "Mej_tot-bern_geoend", "fmt": ".2f", "mod": "*1e2", "err": "ud",    "deferr": None},
    ]
    # groups = groups[(groups.tend_wind > 0.030) | (groups["group"] == "BLh_M10651772_M0_LK")]
    # print(groups["group"] == "BLh_M10651772_M0_LK"); exit(1)
    print("-"*20)
    unique_sim_table(groups, tasks, settings)

