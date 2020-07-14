#!/usr/bin/env python

import csv
from itertools import cycle
import h5py
import pandas
import matplotlib
import matplotlib.pyplot as plt
from math import pi, sqrt
import numpy as np
import os
import copy
# import units as ut
# from utils import get_advanced_time, get_retarded_time

class Paths(object):
    to_rns_sequences = "../Data/RNS/RNS.dat.gz"
    to_tovs = "../Data/TOVs/" # EOS_sequence.txt
    to_summarytable = "../output/models3.csv"
    to_simsource = "/data1/numrel/WhiskyTHC/Backup/2018/GW170817/"
    # to_summarytable = "../output/radice2018_summary.csv"
    pass


class Struct(object):
    pass
#
params = Struct()
params.Mej_min = 5e-5
params.Mej_err = lambda Mej: 0.5*Mej + params.Mej_min
params.Yeej_err = lambda Ye: 0.01
params.vej_err = lambda v: 0.02
params.Sej_err = lambda Sej: 1.5
params.theta_ej_err = lambda theta_ej: 2.0
params.MdiskPP_min = 5e-4
params.MdiskPP_err = lambda MdiskPP: 0.5*MdiskPP + params.MdiskPP_min
params.Anrm_range = [180, 200]
params.vej_fast = 0.6
params.Mej_fast_min = 1e-8
params.Mej_fast_err = lambda Mej_fast: 0.5*Mej_fast + params.Mej_fast_min
params.Mej_v_n = "Mej_tot-geo" # Mej_tot-geo
params.tend_min = 10. # ms
params.blacklist = ["LS220_M13641364_M0_LK_SR_restart",
                    "LS220_M14691268_M0_LK_SR_AHfix",
                    "SLy4_M10201856_M0_LK_SR"]
# Plot styles
color_list = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
marker_list = ["o", "s", "^", "d", "*", "1", "2", "3", "4", "x"]
cmap_list = [
    plt.get_cmap("Purples"),
    plt.get_cmap("Blues"),
    plt.get_cmap("Oranges"),
    plt.get_cmap("Reds")
]

#
color_cycle  = cycle(color_list)
marker_cycle = cycle(marker_list)
cmap_cycle   = cycle(cmap_list)
#
TOV_M_max = {
    "BLh"   : np.nan,
    "SLy4"  : np.nan,
    "BHBlp" : 2.11,
    "DD2"   : 2.42,
    "LS220" : 2.05,
    "SFHo"  : 2.03,
}
#
def print_label(label):
    return r"\texttt{%s}" % label.replace("_", r"\_")

def print_eos(eos):
    aliases = {
        "BHBlp": r"BHB$\Lambda\phi$"
    }
    if aliases.has_key(eos):
        return aliases[eos]
    else:
        return eos

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
    if v_n == "k2T":
        return r"$k_2^T$"
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
    if v_n == "vel_inf_ave-geo" or v_n == "vej":
        return r"$v_{\rm ej}\ [c]$"
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
    if v_n == "Ye_ave-geo":
        return r"$\langle Y_e \rangle$"
    if v_n == "theta_rms-geo":
        return r"$\theta_{\rm ej}$"
    if v_n == "Mdisk3D":
        return r"$M_{\rm{disk}}$ $[M_{\odot}]$"
    if v_n == "Mdisk3Dmax":
        return r"$M_{\rm{disk;max}}$ $[M_{\odot}]$"
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

def get_minmax(v_n, arr, extra = 2., oldtable=True):
    if v_n == "Lambda":
        if oldtable:
            min_, max_ = 0, 1500
        else:
            min_, max_ = 380, 880
    elif v_n == "k2T":
        if oldtable:
            min_, max_ = 0, 320
        else:
            min_, max_ = 50, 200
    elif v_n == "Mej_tot-geo" or v_n == "Mej":
        min_, max_ = 0, 2.
    elif v_n == "Mej_tot-geo_entropy_below_10":
        min_, max_ = 0, 1
    elif v_n == "Mej_tot-geo_entropy_above_10":
        min_, max_ = 0, 1
    elif v_n == "Ye_ave-geo" or v_n == "Yeej":
        min_, max_ = 0., 0.5
    elif v_n == "vel_inf_ave-geo" or v_n == "vej":
        min_, max_ = 0.1, 0.4
    elif v_n == "Ye_ave-geo":
        return 0., 0.4
    elif  v_n == "theta_rms-geo":
        return 0., 50
    elif v_n == "Mdisk3D":
        return 0., 0.4
    elif v_n == "Mdisk3Dmax":
        return 0., 0.4
    elif v_n == "q":
        return 0.9, 2.
    else:
        min_, max_ = np.array(arr).min(), np.array(arr).max() + (extra * (np.array(arr).max() - np.array(arr).min()))
        print("xlimits are not set for v_n_x:{}".format(v_n))
    print(v_n, min_, max_)
    return min_, max_

def check_initial_table_models():

    missing_from_table = {}
    sims = os.listdir(Paths.to_simsource)
    for eos in EOS:
        missing_from_table[eos] = []
        eos_sims = []
        for sim in sims:
            if sim.__contains__(eos):
                eos_sims.append(sim)
        assert len(eos_sims) > 0
        for sim in eos_sims:
            if not sim in simulations.index.values:
                missing_from_table[eos].append(sim)
    #
    print("---------------------------------------------------------")
    print("Missing from the table: {}".format(Paths.to_summarytable))
    for eos in EOS:
        print("\t{}".format(eos))
        for isim, sim in enumerate(missing_from_table[eos]):
            print("\t\t{}".format(sim))
    print("---------------------------------------------------------")
    #
    print("---------------------------------------------------------")
    print("Unique simulations:")
    unique = sorted(list(set(simulations["group"])))
    for u in unique:
        group = simulations[simulations.group == u]
        # print("\t{} | {} (res)".format(u, len(group)))
        print(group["tcoll_gw"])
        print("-"*20)
    print(unique)
    print("---------------------------------------------------------")
    print(" No Dynamical Ejecta")
    print(simulations[simulations[params.Mej_v_n] <= params.Mej_min])

    print("---------------------------------------------------------")
    print(" Shorter than {:.1f} ms".format(params.tend_min))
    print(simulations[simulations["tend"] < float(params.tend_min / 1.e3)])

    # print(sims)
    # print(len(sims))

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

def get_group_value3(group, task_dic):
    """
    group = pandas.dataframe
    task_dic = {"v_n":"Mej", "mod":"*1.e2", "err":"ud", "deferr":0.2}
    """
    v_n = task_dic["v_n"]
    mod = task_dic["mod"]
    err = task_dic["err"]
    deferr = task_dic["deferr"]

    # print(type(group.resolution)); print(list(group.resolution)); exit(1)

    ress = sorted(list(group.resolution))
    if len(ress) > 3:
        raise ValueError("too many resolutions! {}".format(group.resolutons))
    #
    if v_n == "group":
        val = list(group.group)[0]
        val = val.replace("_", "--") + r"\textit{[S,H,L]}"
        return val
    #
    if v_n == "resolution":
        val = r"\texttt{" + " ".join(ress) + "}"
        return val
    #
    if "SR" in ress and "LR" in ress and "HR" in ress:
        # HR - SR , SR - LR
        # print("\t 1 ress:{} v_n:{} fmt:{} err:{}".format(ress, v_n, fmt, err))
        if err == None or err == "":
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
        if err == None or err == "":
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
        if err == None or err == "":
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
        if err == None or err == "":
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
        if err == None or err == "":
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
        if err == None or err == "":
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
        if err == None or err == "":
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

def get_group_vals_sigams(sims, v_n_dic):
    """
    sims = pandas.dataframe with all simualtions
    v_n_dic = {"v_n":"Mej", "mod":"*1.e2", "err":"ud", "deferr":0.2}
    """
    unique = sorted(list(set(sims["group"])))
    x_arr = np.zeros(3)
    for u in unique:
        group = sims[sims.group == u]
        x, del1, del2 = get_group_value3(group, v_n_dic)
        x_arr = np.vstack((x_arr, [x, del1, del2]))
    x_arr = np.delete(x_arr, 0, 0)
    return x_arr


# Read the simulation list
simulations = pandas.read_csv(Paths.to_summarytable)
simulations = simulations.set_index("name")
# simulations = simulations[simulations["publish"] == "yes"]
# simulations["label"] = [get_label(m.name) for _, m in simulations.iterrows()]
simulations["C1"] = simulations["M1"]/simulations["R1"]
simulations["C2"] = simulations["M2"]/simulations["R2"]
simulations["M"] = simulations["M1"] + simulations["M2"]
simulations["C"] = simulations["M"]/(simulations["R1"] + simulations["R2"])

def print_fancy_label(name):
    sim = simulations.loc[name]
    if sim["viscosity"] == "LK":
        viscousisty = ", LK"
    else:
        viscousisty = ""
    return print_eos(sim["EOS"]) + \
           r": $({} + {})\, M_\odot$".format(sim["M1"], sim["M2"]) + ", M0" + viscousisty


    # if sim["comment"] == "M0":
    #     leakage = r"; $\nu$ cooling and heating"
    # else:
    #     leakage = r"; $\nu$ cooling only"
    # return print_eos(sim["EOS"]) +\
    #         r": $({} + {})\, M_\odot$".format(sim["M1"], sim["M2"]) +\
    #         leakage

# def print_fancy_label(name):
#     sim = simulations.loc[name]
#     if sim["comment"] == "M0":
#         leakage = r"; $\nu$ cooling and heating"
#     else:
#         leakage = r"; $\nu$ cooling only"
#     return print_eos(sim["EOS"]) +\
#             r": $({} + {})\, M_\odot$".format(sim["M1"], sim["M2"]) +\
#             leakage

EOS = sorted(list(set(simulations["EOS"])))

wrong_init_data = abs(simulations["Mbi_Mb"] - 1.) > 0.04
correct_init_data = abs(simulations["Mbi_Mb"] - 1.) < 0.04

fiducial = correct_init_data & (simulations.resolution == "SR")
well_resolved =  (fiducial) & (simulations[params.Mej_v_n] > params.Mej_min)
with_disk_mass = (fiducial) & (simulations.Mdisk3D.notnull())
viscous = (fiducial) & (simulations.viscosity == "LK")
with_m0 = (fiducial) & (simulations.viscosity == "LK")
non_viscous = (fiducial) & (simulations.viscosity != "LK")
long_runs = (fiducial) & (simulations.tend > 50. / 1.e3)
with_bh = (fiducial) &  (np.isfinite(simulations.tcoll_gw))
# not_blacklist =
unique_simulations = copy.deepcopy(simulations)
unique_simulations = unique_simulations.drop(params.blacklist, axis=0)
# print(unique_simulations.loc["LS220_M13641364_M0_LK_SR_restart"]["tend"]); exit(1)
# for blacklisted in params.blacklist:
#     unique_simulations.drop(blacklisted)
    # unique_simulations = unique_simulations[unique_simulations.index!=blacklisted]

def __average_value_for_group(group, v_n, method="st.div", deferr=0.2):
    """
    :param group: panda.dataframe of atmost 3 simualtions
    :param v_n: name of the variable in the dataframe
    :param method: None or st.div
    :return: mean, err1, err2
    """
    ress = sorted(list(group.resolution))
    if len(ress) > 3:
        raise ValueError("too many resolutions! {}".format(group.resolutons))
    #
    if v_n == "group":
        val = list(group.group)[0]
        return val
    #
    if v_n == "resolution" or v_n == "res":
        val = " ".join(ress)
        return val
    #
    if "SR" in ress and "LR" in ress and "HR" in ress:
        if method == None or method == "":
            sr = group[group.resolution == "SR"]
            val_sr = float(list(sr[v_n])[0])
            return val_sr, 0., 0.
        elif method == "st.div":
            hr = group[group.resolution == "HR"]
            sr = group[group.resolution == "SR"]
            lr = group[group.resolution == "LR"]

            val_sr = float(sr[v_n])
            val_lr = float(lr[v_n])
            val_hr = float(hr[v_n])

            mean, stddiv = standard_div([val_hr, val_sr, val_lr])
            return mean, stddiv, stddiv
        else:
            raise NameError("method:{} is not recognised. Use None of st.div".format(method))
    elif len(ress) == 2 and "SR" in ress and "LR" in ress:
        if method == None or method == "":
            sr = group[group.resolution == "SR"]
            val_sr = float(list(sr[v_n])[0])
            return val_sr, 0., 0.
        elif method == "st.div":
            # hr = group[group.resolution == "HR"]
            sr = group[group.resolution == "SR"]
            lr = group[group.resolution == "LR"]

            val_sr = float(sr[v_n])
            val_lr = float(lr[v_n])

            mean, stddiv = standard_div([val_sr, val_lr])
            return mean, stddiv, stddiv
        else:
            raise NameError("method:{} is not recognised. Use None of st.div".format(method))
            # return str("$" + val_sr + r"^{+" + str(del1) + r"} _{-" + str(del2)) + "} $"
    elif len(ress) == 2 and "SR" in ress and "HR" in ress:
        if method == None or method == "":
            sr = group[group.resolution == "SR"]
            val_sr = float(list(sr[v_n])[0])
            return val_sr, 0., 0.
        elif method == "st.div":
            hr = group[group.resolution == "HR"]
            sr = group[group.resolution == "SR"]

            val_sr = float(sr[v_n])
            val_hr = float(hr[v_n])

            mean, stddiv = standard_div([val_hr, val_sr])
            return mean, stddiv, stddiv
        else:
            raise NameError("method:{} is not recognised. Use None of st.div".format(method))
            # return str("$" + val_sr + r"^{+" + str(del1) + r"} _{-" + str(del2)) + "} $"
        pass
    elif len(ress) == 2 and "LR" in ress and "HR" in ress:
        # print("\t 3 ress:{} v_n:{} fmt:{} err:{}".format(ress, v_n, fmt, err))
        if method == None or method == "":
            lr = group[group.resolution == "LR"]
            val_lr = float(list(lr[v_n])[0])
            return val_lr, 0., 0.
        elif method == "st.div":
            hr = group[group.resolution == "HR"]
            lr = group[group.resolution == "LR"]

            val_lr = float(lr[v_n])
            val_hr = float(hr[v_n])

            mean, stddiv = standard_div([val_hr, val_lr])
            return mean, stddiv, stddiv
        else:
            raise NameError("method:{} is not recognised. Use None of st.div".format(method))
        pass
    elif len(ress) == 1 and "SR" in ress:
        # SR - X*SR, SR - X*SR
        if method == None or method == "":
            sr = group[group.resolution == "SR"]
            val_sr = float(list(sr[v_n])[0])
            return val_sr, 0., 0.
        elif method == "st.div":
            # hr = group[group.resolution == "HR"]
            sr = group[group.resolution == "SR"]

            val_sr = float(sr[v_n])

            del1 = del2 = val_sr * deferr

            return val_sr, del1, del2
        else:
            raise NameError("method:{} is not recognised. Use None of st.div".format(method))
    elif len(ress) == 1 and "LR" in ress:
        # LR - Y*LR, SR - Y*LR
        # print("\t 5 ress:{} v_n:{} fmt:{} err:{}".format(ress, v_n, fmt, err))
        if method == None or method == "":
            lr = group[group.resolution == "LR"]
            val_lr = float(list(lr[v_n])[0])
            return val_lr, 0., 0.
        elif method == "st.div":

            lr = group[group.resolution == "LR"]

            val_lr = float(lr[v_n])

            del1 = del2 = val_lr * deferr

            return val_lr, del1, del2
        else:
            raise NameError("method:{} is not recognised. Use None of st.div".format(method))
            # return str("$" + val_lr + "$")#r"^{" + str(del1) + r"} _{" + str(del2)) + "} $"
    elif len(ress) == 1 and "HR" in ress:
        # HR - Z*HR, SR - Z*HR
        # print("\t 6 ress:{} v_n:{} fmt:{} err:{}".format(ress, v_n, fmt, err))
        if method == None or method == "":
            hr = group[group.resolution == "HR"]
            val_hr = float(list(hr[v_n])[0])
            return val_hr, 0., 0.
        elif method == "st.div":
            hr = group[group.resolution == "HR"]

            val_hr = float(hr[v_n])

            del1 = del2 = val_hr * deferr

            return val_hr, del1, del2
        else:
            raise NameError("method:{} is not recognised. Use None of st.div".format(method))
            # return str("$" + val_hr + "$")#r"^{" + str(del1) + r"} _{" + str(del2)) + "} $"
    else:
        print(group)
        raise ValueError("\t 7. Unrecognized resoltion setup \n "
                         "ress:{} v_n:{} err:{}".format(ress, v_n, method))

def convert_models_to_uniquemodels_table():
    """
    Loads simulation table, extract data for groups of simualtion (with different resolution)
    computes average quantites of selected v_ns and saves as a new .csv and returns dataframe
    :return: pandas.dataframe
    """
    import pandas
    # intable = Paths.output + "models3.csv"
    # outtable = Paths.output + "unique_models.csv"
    #
    const_v_ns = ["q", "viscosity", "EOS", "M1", "M2", "Mb1", "Mb2", "Mb", "Mg1", "Mg2", "R1", "R2", "C1",
                  "C2", "k21", "k22", "lam21", "lam22", "Lambda", "k2T", "MADM", "EGW", "JADM"]
    var_v_ns = ["Mej_tot-geo", "vel_inf_ave-geo", "theta_rms-geo", "Mdisk3D", "Mdisk3Dmax"]
    min_val_v_ns = ["tcoll_gw"]
    outfname = "../output/groups.csv"
    #
    # from models import unique_simulations
    #
    simulations = unique_simulations
    #
    new_data_frame = {}
    #
    groups = sorted(list(set(simulations["group"])))
    new_data_frame["group"] = groups
    new_data_frame["resolution"] = \
        [" ".join(simulations[simulations["group"] == group].resolution) for group in groups]
    #
    for v_n in const_v_ns:
        values = []
        for group in groups:
            sims = simulations[simulations["group"] == group]
            assert len(sims) > 0 and len(sims) <= 3
            value = list(sims[v_n])[0]
            values.append(value)
        #
        new_data_frame[v_n] = values
    #
    for v_n in min_val_v_ns:
        values = []
        for group in groups:
            sims = simulations[simulations["group"] == group]
            assert len(sims) > 0 and len(sims) <= 3
            value = np.array(sims[v_n])
            # values = values[values < 1e10] # remove inf
            # if len(values) > 0:
            values.append(min(value))
            # else:
            #     values.append()
        #
        new_data_frame[v_n] = values
    #
    for v_n in var_v_ns:
        values = []
        errors = []
        for group in groups:
            sims = simulations[simulations["group"] == group]
            assert len(sims) > 0 and len(sims) <= 3
            value, err1, err2 = __average_value_for_group(sims, v_n, "st.div", 0.2)
            values.append(value)
            errors.append(err1)
        new_data_frame[v_n] = values
        new_data_frame["err-"+v_n] = errors
    #
    df = pandas.DataFrame(new_data_frame, index=groups)
    df.set_index("group")
    df.to_csv(outfname)
    print("saved as {}".format(outfname))

    print(df.loc["SFHo_M13641364_M0"]["Mdisk3D"]); exit(1)

    return df




if __name__ == '__main__':

    convert_models_to_uniquemodels_table()


    check_initial_table_models()


    # SLy4_M10651772_M0_LK_SR SFHo_M14521283_M0_LR SLy4_M13641364_M0_LR SFHo_M14521283_M0_LK_LR (last has ejecta not finished)
    # SFHo_M13641364_M0_LK_LR LS220_M13641364_LK_LR DD2_M15091235_M0_LK_LR SLy4_M10651772_M0_LK_SR_AHfix
    # SLy4_M10651772_M0_LK_SR SFHo_M11461635_M0_LK_SR SFHo_M10651772_M0_LK_SR_AHfix DD2_M16351146_M0_LK_LR
    # DD2_M11461635_M0_LK_SR BLh_M16351146_M0_LK_LR BLh_M13641364_M0_LK_SR BLh_M13641364_M0_LK_HR
    # BLh_M11461635_M0_LK_SR BLh_M10651772_M0_LK_SR BLh_M10201856_M0_LK_HR

    # print(simulations["EOS"] == "SLy4"); exit(1)
    # print(simulations.comment)
    # print(simulations.index.values)
    print("# Simulations       = {}".format(len(simulations)))
    print("# correct_init_data = {}".format(len(simulations[correct_init_data])))
    print("# wrong_init_data   = {}".format(len(simulations[wrong_init_data])))
    # print("# Leakage           = {}".format(len(simulations[leakage])))
    print("# Fiducial          = {}".format(len(simulations[fiducial])))
    print("# Long              = {}".format(len(simulations[long_runs])))
    print("# Resolved          = {}".format(len(simulations[well_resolved])))
    print("# Disks             = {}".format(len(simulations[with_disk_mass])))
    # BH
    print("# BH forming  ")
    for eos in EOS:
        coll = (simulations.EOS == eos) & (np.isfinite(simulations.tcoll_gw))
        stable=(simulations.EOS == eos) & (np.isinf(simulations.tcoll_gw))
        print("{} BH: {}/{} Stable {}/{}".format(eos, len(simulations[coll]), len(simulations),
                                                 len(simulations[stable]), len(simulations))),
        sr = (simulations.EOS == eos) & (simulations.resolution == "SR")
        coll = sr & (np.isfinite(simulations.tcoll_gw))
        stable = sr & (np.isinf(simulations.tcoll_gw))
        print("\t[SR] {} BH: {}/{} Stable {}/{}".format(eos, len(simulations[coll]), len(simulations[sr]),
                                                 len(simulations[stable]), len(simulations[sr]))),
        print('')
    print('---------|LONG & DISK|----------')
    print(simulations[long_runs & with_disk_mass]["Mdisk3D"])
    print('---------|BH & DISK|----------')
    print(simulations[with_bh & with_disk_mass]["Mdisk3D"])

    # print("rm -r"),
    # for _, m in simulations.iterrows():
    #     print(m.name),
    # print('')
    # print("# Kilonova    = {}".format(len(simulations[kilonova])))
    # print("# M0          = {}".format(len(simulations[M0])))


