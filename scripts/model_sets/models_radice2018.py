#!/usr/bin/env python

'''
https://arxiv.org/pdf/1809.11161.pdf
m1 > m2

'''


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
    to_rns_sequences = "../../Data/RNS/RNS.dat.gz"
    to_tovs = "../../Data/TOVs/" # EOS_sequence.txt
    to_summarytable = "../datasets/models3.csv"
    to_radicetable = "../datasets/radice2018_summary3.csv"
    to_simsource = "/data1/numrel/WhiskyTHC/Backup/2018/GW170817/"
    # to_summarytable = "../output/radice2018_summary.csv"
    to_radicepapertable = "../datasets/radice_papertable.txt"
    pass


class Struct(object):
    Mej_min = 5e-5
    Mej_err = lambda _, Mej: 0.5 * Mej + Struct.Mej_min
    # Yeej_err = lambda Ye: 0.01
    ye_def_err = 0.01
    Yeej_err = lambda _, v: 1 * np.full(len(v), Struct.ye_def_err)
    vej_def_err = 0.02
    vej_err = lambda _, v: 1 * np.full(len(v), Struct.vej_def_err)
    Sej_err = lambda Sej: 1.5
    theta_ej_err = lambda theta_ej: 2.0
    MdiskPP_min = 5e-4
    MdiskPP_err = lambda _, MdiskPP: 0.5 * MdiskPP + Struct.MdiskPP_min
    Anrm_range = [180, 200]
    vej_fast = 0.6
    Mej_fast_min = 1e-8
    Mej_fast_err = lambda Mej_fast: 0.5 * Mej_fast + Struct.Mej_fast_min
    Mej_v_n = "Mej"
    pass
#
params = Struct()
# params.Mej_min = 5e-5
# params.Mej_err = lambda Mej: 0.5*Mej + params.Mej_min
# params.Yeej_err = lambda Ye: 0.01
# params.vej_err = lambda v: 0.02
# params.Sej_err = lambda Sej: 1.5
# params.theta_ej_err = lambda theta_ej: 2.0
# params.MdiskPP_min = 5e-4
# params.MdiskPP_err = lambda MdiskPP: 0.5*MdiskPP + params.MdiskPP_min
# params.Anrm_range = [180, 200]
# params.vej_fast = 0.6
# params.Mej_fast_min = 1e-8
# params.Mej_fast_err = lambda Mej_fast: 0.5*Mej_fast + params.Mej_fast_min
# params.Mej_v_n = "Mej" # Mej_tot-geo


# print(params.MdiskPP_err(np.array([ 2.])))

# Plot styles
color_list = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
marker_list = ["o", "s", "^", "d", "*", "triangle_down", "plus", "triangle_up", "triangle_left", "triangle_right"]
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
def get_label(name):
    aliases = {
        "BHBlp_M1251365_LK"    : "BHBlp_M1365125_LK",
        "DD2_M1251365_LK"      : "DD2_M1365125_LK",
        "LS220_M1251365_LK"    : "LS220_M1365125_LK",
        "LS220_M135135_M0_L5"  : "LS220_M135135_M0_L05",
        "LS220_M140120_M0_L5"  : "LS220_M140120_M0_L05",
        "LS220_M135135_OldM0"  : "LS220_M135135_M0_LTE",
        "SFHo_M1251365_LK"     : "SFHo_M1365125_LK",
        "SFHo_M140120_M0_v4"   : "SFHo_M140120_M0"
    }
    if aliases.has_key(name):
        return aliases[name]
    else:
        return name

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

translation = {
    "Lambda":       "Lambda",
    "Mej_tot-geo":  "Mej",
    "k2T":          "k2T",
    "q":            "q",
    "vel_inf_ave-geo": "vej",
    "Mej_tot-geo_entropy_below_10":"Mej_tot-geo_entropy_below_10",
    "Mej_tot-geo_entropy_above_10":"Mej_tot-geo_entropy_above_10",
    "theta_rms-geo":"theta_rms-geo",
    "Mdisk3D":    "M_disk", #"Mdisk",
    "Mdisk3Dmax": "M_disk",  #"Mdisk"
    "Mtot": "M",
    "Mchirp":"Mchirp",
    "Ye_ave-geo":"Yeej",
    "M1":"M1",
    "M2":"M2",
    "C1":"C1",
    "C2":"C2",
    "Mb1":"Mb1",
    "Mb2":"Mb2",
    "EOS": "EOS",
    "nus": "nus",
    "arxiv": "arxiv"
}

# Read the simulation list
simulations = pandas.read_csv(Paths.to_radicetable)
simulations = simulations.set_index("name")

simulations["arxiv"] = "https://arxiv.org/abs/1809.11161"

# simulations = simulations[simulations["publish"] == "yes"]
simulations["label"] = [get_label(m.name) for _, m in simulations.iterrows()]
simulations["C1"] = simulations["M1"]/simulations["R1"]
simulations["q"] = simulations["M1"]/simulations["M2"]
simulations["C2"] = simulations["M2"]/simulations["R2"]
simulations["M"] = simulations["M1"] + simulations["M2"]
simulations["C"] = simulations["M"]/(simulations["R1"] + simulations["R2"])
simulations["Mchirp"] = ((simulations["M1"] * simulations["M2"]) ** (3./5.)) / (simulations["M"]**(1./5.))
simulations["nus"] = "none"
nus = []
for name, model in simulations.iterrows():
    if model.comment == "leakage": nus.append("leak")
    else: nus.append("leakM0")
simulations["nus"] = nus


def get_mod_err(v_n, mod_dic, simulations, arr=np.zeros(0,)):
    if len(arr) == 0:
        val_arr = np.array(simulations[translation[v_n]], dtype=float)
        if v_n == "Mej_tot-geo": arr = params.Mej_err(val_arr)
        elif v_n == "vel_inf_ave-geo": arr = params.vej_err(val_arr)
        elif v_n == "Ye_ave-geo": arr = params.Yeej_err(val_arr)
        elif v_n == "Mdisk3D" or v_n == "Mdisk3Dmax": arr = params.MdiskPP_err(val_arr)
        else:
            raise NameError("No error prescription for v_n:{}".format(v_n))

    # print ("--arr:{}".format(arr))
    if "mult" in mod_dic.keys():
        print("mult, {}".format(mod_dic["mult"]))
        for entry in mod_dic["mult"]:
            if isinstance(entry, float):
                arr = arr * float(entry)
            else:
                another_array = np.array(simulations[translation[entry]])
                arr = arr * another_array
    if "dev" in mod_dic.keys():
        print("dev {}".format(mod_dic["dev"]))
        for entry in mod_dic["dev"]:
            if isinstance(entry, float):
                arr = arr / float(entry)
            else:
                another_array = np.array(simulations[translation[entry]])
                arr = arr / another_array
    # print ("--arr:{}".format(arr))
    return arr

def get_mod_data(v_n, mod_dic, simulations, arr=np.zeros(0,)):
    if len(arr) == 0:
        if v_n in ["EOS", "nus", "arxiv"]:
            arr = list(simulations[translation[v_n]])
        else:
            arr = np.array(simulations[translation[v_n]], dtype=float)
    if "mult" in mod_dic.keys():
        print("mult, {}".format(mod_dic["mult"]))
        for entry in mod_dic["mult"]:
            if isinstance(entry, float):
                arr = arr * float(entry)
            else:
                another_array = np.array(simulations[translation[entry]])
                arr = arr * another_array
    if "dev" in mod_dic.keys():
        print("dev {}".format(mod_dic["dev"]))
        for entry in mod_dic["dev"]:
            if isinstance(entry, float):
                arr = arr / float(entry)
            else:
                another_array = np.array(simulations[translation[entry]])
                arr = arr / another_array
    return arr

def print_fancy_label(name):
    sim = simulations.loc[name]
    if sim["viscosity"] == "LK":
        viscousisty = ", LK"
    else:
        viscousisty = ""
    return print_eos(sim["EOS"]) + \
           r": $({} + {})\, M_\odot$".format(sim["M1"], sim["M2"]) + ", M0" + viscousisty

# LOAD Paper Table (it contains some absent Data)
papertable = pandas.read_table(Paths.to_radicepapertable, sep=" ", header=None,
                               names=["model", "h", "M_a", "M_b", "t_BH", "M_disk", "M_ej", "M_ej06",
                               "Ye", "s", "v_ej", "theta_ej", "T_ej"], index_col=0)
# Append absent in .csv data from .text table
append_v_ns = ["M_disk"]
tmp = lambda (v_n, val): val / 1.e2 if v_n == "M_disk" else val
for v_n in append_v_ns:
    L = []
    for name, dic in simulations.iterrows():
        if name in papertable.index:
            val = tmp((v_n, float(papertable.loc[name][v_n])))
            L.append(val)
        else:
            L.append(np.nan)
    L = np.array(L)
    nonnanL = L[np.logical_not(np.isnan(L))]
    print("Appending v_n:{} for {}/{}".format(v_n, len(nonnanL), len(L)))
    simulations[v_n] = L


# papertable[1]
# papertable.set_index("model")
# print(simulations["M_disk"])

''' end '''

EOS = sorted(list(set(simulations["EOS"])))

leakage  = (simulations.comment == "leakage")
fiducial = (leakage) & (simulations.resolution == 0.125)
well_resolved =  (fiducial) & (simulations.Mej > Struct.Mej_min)
with_disk_mass = (fiducial) & (simulations.Mdisk.notnull())
with_table_disk_mass = (fiducial) & (simulations.M_disk.notnull())
# viscous = (fiducial) & (simulations.viscosity == "LK")
with_m0 = (fiducial) & (simulations.comment == "M0")
# non_viscous = (fiducial) & (simulations.viscosity != "LK")
# not_blacklist =
# unique_simulations = copy.deepcopy(simulations)
# print(unique_simulations.loc["LS220_M13641364_M0_LK_SR_restart"]["tend"]); exit(1)
# for blacklisted in params.blacklist:
#     unique_simulations.drop(blacklisted)
    # unique_simulations = unique_simulations[unique_simulations.index!=blacklisted]



if __name__ == '__main__':
    print(simulations[["EOS", "q", "Lambda", "M1", "M2", "Mb1", "Mb2", "Mej", "nus"]])

    # check_initial_table_models()

    # SLy4_M10651772_M0_LK_SR SFHo_M14521283_M0_LR SLy4_M13641364_M0_LR SFHo_M14521283_M0_LK_LR (last has ejecta not finished)
    # SFHo_M13641364_M0_LK_LR LS220_M13641364_LK_LR DD2_M15091235_M0_LK_LR SLy4_M10651772_M0_LK_SR_AHfix
    # SLy4_M10651772_M0_LK_SR SFHo_M11461635_M0_LK_SR SFHo_M10651772_M0_LK_SR_AHfix DD2_M16351146_M0_LK_LR
    # DD2_M11461635_M0_LK_SR BLh_M16351146_M0_LK_LR BLh_M13641364_M0_LK_SR BLh_M13641364_M0_LK_HR
    # BLh_M11461635_M0_LK_SR BLh_M10651772_M0_LK_SR BLh_M10201856_M0_LK_HR

    # print(simulations["EOS"] == "SLy4"); exit(1)
    # print(simulations.comment)
    # print(simulations.index.values)
    print("# Simulations       = {}".format(len(simulations)))
    # print("# correct_init_data = {}".format(len(simulations[correct_init_data])))
    # print("# wrong_init_data   = {}".format(len(simulations[wrong_init_data])))
    print("# Leakage           = {}".format(len(simulations[leakage])))
    print("# Fiducial          = {}".format(len(simulations[fiducial])))
    # print("# Long              = {}".format(len(simulations[long_runs])))
    print("# Resolved          = {}".format(len(simulations[well_resolved])))
    print("# Disks             = {}".format(len(simulations[with_disk_mass])))
    print("# Disks Table       = {}".format(len(simulations[with_table_disk_mass])))
    print(simulations[fiducial][with_table_disk_mass])
    # BH
    # print("# BH forming  ")
    # for eos in EOS:
    #     coll = (simulations.EOS == eos) & (np.isfinite(simulations.tcoll_gw))
    #     stable=(simulations.EOS == eos) & (np.isinf(simulations.tcoll_gw))
    #     print("{} BH: {}/{} Stable {}/{}".format(eos, len(simulations[coll]), len(simulations),
    #                                              len(simulations[stable]), len(simulations))),
    #     sr = (simulations.EOS == eos) & (simulations.resolution == "SR")
    #     coll = sr & (np.isfinite(simulations.tcoll_gw))
    #     stable = sr & (np.isinf(simulations.tcoll_gw))
    #     print("\t[SR] {} BH: {}/{} Stable {}/{}".format(eos, len(simulations[coll]), len(simulations[sr]),
    #                                              len(simulations[stable]), len(simulations[sr]))),
    #     print('')
    # print('---------|LONG & DISK|----------')
    # print(simulations[long_runs & with_disk_mass])

    # print("rm -r"),
    # for _, m in simulations.iterrows():
    #     print(m.name),
    # print('')
    # print("# Kilonova    = {}".format(len(simulations[kilonova])))
    # print("# M0          = {}".format(len(simulations[M0])))

    ''' statistics '''

    print(simulations["Mej"].describe(percentiles=[0.9]))
