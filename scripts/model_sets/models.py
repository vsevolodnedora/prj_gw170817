from __future__ import division
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
# from data import ADD_METHODS_ALL_PAR, AVERAGE_PAR
# import units as ut
# from utils import get_advanced_time, get_retarded_time

# from uutils import standard_div

class Paths(object):
    to_rns_sequences = "../../Data/RNS/RNS.dat.gz"
    to_tovs = "../../Data/TOVs/" # EOS_sequence.txt
    to_summarytable = "../datasets/models3.csv"
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
params.blacklist = ["LS220_M13641364_M0_LK_SR", # copy of what we have
                    "SLy4_M10201856_M0_LK_SR", # endless dyn. ej growth
                    # sebastiano's choice
                    "DD2_M13641364_M0_SR", "DD2_M13641364_M0_HR", "DD2_M13641364_M0_LR",
                    "DD2_M14321300_M0_LR", "DD2_M14351298_M0_LR",
                    "DD2_M14861254_M0_LR", "DD2_M14861254_M0_HR",
                    "SFHo_M13641364_M0_LK_SR", "SFHo_M13641364_M0_LK_LR",
                    "SFHo_M14521283_M0_LK_SR", "SFHo_M14521283_M0_LK_LR",
                    #_restart",
                    # "LS220_M14691268_M0_LK_SR_AHfix",
                    # "SLy4_M10201856_M0_LK_SR",
                    # "SLy4_M13641364_M0_LK_SR_AHfix",
                    # "LS220_M10651772_M0_LK_SR_R05_AHfix",
                    # "LS220_M10651772_M0_SR_R05_AHfix",
                    # "LS220_M10651772_M0_LR_AHfix"
                    #  BLh_M16351146_M0_LK_LR --- fix merger time!
                    #  SFHo_M13641364_M0_LR -- Fix Merger Time!
                    # SFHo_M10651772_M0_LK_SR_AHfix -- Fix Merger time!
                    # SLy4_M14521283_M0_LR same111111
                    ]
params.good_disk_evol=["LS220_M11461635_M0_LK_SR", "SFHo_M14521283_M0_SR"]
# SFHo_M13641364_M0_LK_LR/profiles/disk_mass.png  -- remove last disk mass
# SFHo_M13641364_M0_LK_SR_2019pizza n --- disk mass reduced to 0


# Maximum times
maxtime = {
    "SLy4_M13641364_M0_LK_SR_AHfix":    22,
    "SLy4_M10651772_M0_LK_SR":          18
}

# Plot styles
color_list = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
marker_list = ["o", "s", "^", "d", "*", "1", "2", "3", "4", "x"]
cmap_list = [
    plt.get_cmap("Purples"),
    plt.get_cmap("Blues"),
    plt.get_cmap("Oranges"),
    plt.get_cmap("Reds")
]

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

def get_marker_for_outcome(outcome):

    from rns import Options

    if outcome == Options.list_merger_outcomes[0]:
      return "s"
    elif outcome == Options.list_merger_outcomes[1]:
        return 'o'
    elif outcome == Options.list_merger_outcomes[2]:
        return 'd'
    elif outcome == Options.list_merger_outcomes[3]:
        return 'P'
    elif outcome == Options.list_merger_outcomes[4]:
        return '*'
    else:
        raise NameError("unrecognized outcome:{} no marker set in "
                        "Options.list_of_markers".format(outcome))



""" -------- METHODS & TREATMENTS -------- """

def get_comment_for_a_group(group):
    #
    eos = list(group["EOS"])[0]
    L = []
    for name, dic in group.iterrows():
        o_data = ADD_METHODS_ALL_PAR(name, )
        if eos == "DD2":
            run = o_data.get_initial_data_par("run")
            L.append(run)
        elif eos == "SFHo":
            pz = o_data.get_initial_data_par("pizza_eos")
            if pz.__contains__("2019"):
                L.append("pz2019")
            else:
                L.append("old_pz")
        else:
            L.append("")
    return L

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

def check_initial_table_models(simulations):

    EOS = sorted(list(set(simulations["EOS"])))

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
        print("\tgroup: {}".format(u))
        group = simulations[simulations.group == u]
        # print("\t{} | {} (res)".format(u, len(group)))
        names = group.index
        ress = list(group["resolution"])
        ress_uniq = sorted(set(list(group["resolution"])))
        if len(ress) != len(ress_uniq):
            print("\tResolution Error: {}".format(ress))
        print(group[["Mbi_Mb","tcoll_gw", "nprofs", "Mdisk3D", "tdisk3D"]])
        print("-"*20)
    print(unique)
    print("---------------------------------------------------------")
    print(" No Dynamical Ejecta")
    print(simulations[simulations[params.Mej_v_n] <= params.Mej_min])

    print("---------------------------------------------------------")
    print(" Shorter than {:.1f} ms".format(params.tend_min))
    print(simulations[simulations["tend"] < float(params.tend_min / 1.e3)])

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

def test_group_disk(simulations):

    groups = sorted(list(set(simulations["group"])))
    for group in groups:
        sel = simulations[simulations["group"] == group]
        print(sel[["Mdisk3D","tdisk3D"]])

def get_q(simulations):
    qs = []
    for _, m in simulations.iterrows():
        q = m.M1 / m.M2
        if q >= 1.: qs.append(q)
        else: qs.append(1./q)
    return qs

# --- printing ---

def print_label(sim):
    return r"\texttt{%s}" % sim.replace("_", r"\_")

def print_eos(eos):
    aliases = {
        "BHBlp": r"BHB$\Lambda\phi$"
    }
    if aliases.has_key(eos):
        return aliases[eos]
    else:
        return eos

def print_fancy_label(name):
    name = str(name)
    sim = simulations.loc[name]
    eos = name.split("_")[0]
    if name.__contains__("_M0"):
        leakage = r"; M0"  # r"; $\nu$ cooling and heating"
    else:
        leakage = r"; Leakage"  # r"; $\nu$ cooling only"
    #
    if name.__contains__("_LK"):
        viscosity = r"; Vis."  # r"; $\nu$ cooling and heating"
    else:
        viscosity = r";"  # r"; $\nu$ cooling only"
    #
    return print_eos(eos) + r": $({} + {})\, M_\odot$".format(sim["M1"], sim["M2"]) + leakage + viscosity

""" ------- COLORs LSs LABELs --------"""

sim_dic_color = {
    # BLh -> "green"
    "BLh_M10201856_M0_LK_LR":  "green",
    "BLh_M13641364_M0_LK_SR":  "green",
    "BLh_M11461635_M0_LK_SR":  "green",
    "BLh_M10201856_M0_SR":     'green',
    "BLh_M10651772_M0_LK_SR":  "green",
    "BLh_M10651772_M0_LK_LR":  "green",
    # DD2 -> blue
    "DD2_M13641364_M0_SR":        "navy",
    "DD2_M13641364_M0_SR_R04":    "navy",
    "DD2_M13641364_M0_LK_SR_R04": "blue",
    "DD2_M15091235_M0_LK_SR":     "blue",
    "DD2_M14971245_M0_SR":        "navy",
    # LS220 -> red
    "LS220_M11461635_M0_LK_SR":    "red",
    "LS220_M13641364_M0_LK_SR_restart":"red",
    "LS220_M13641364_M0_SR":      "maroon",
    # SLy4 -> magenta
    "SLy4_M13641364_M0_SR":      "magenta",
    "SLy4_M14521283_M0_SR":      "magenta",
    "SLy4_M11461635_M0_LK_SR":   "magenta",
    # SFHo -> purple
    "SFHo_M13641364_M0_SR":       "purple",
    "SFHo_M11461635_M0_LK_SR":    "purple"
}

sim_dic_ls = {
    # q=1.00 -> "-"
    "BLh_M13641364_M0_LK_SR": "-",
    "DD2_M13641364_M0_SR":    "-",
    "SLy4_M13641364_M0_SR":   "-",
    "SFHo_M13641364_M0_SR":   "-",
    "DD2_M13641364_M0_LK_SR_R04": "-",
    "LS220_M13641364_M0_LK_SR_restart": "-",
    # q = 1.82 -> ":"
    "BLh_M10201856_M0_LK_LR": ":",
    "BLh_M10201856_M0_SR":    ":",
    "BLh_M10651772_M0_LK_SR": ":",
    "BLh_M10651772_M0_LK_LR": ":",
    # q = 1.43
    "LS220_M11461635_M0_LK_SR": "-.",
    "BLh_M11461635_M0_LK_SR": "-.",
    "SFHo_M11461635_M0_LK_SR": "-.",
    "SLy4_M11461635_M0_LK_SR": "-.",
    # q = 1.22
    "DD2_M14971245_M0_SR": "--",
    "DD2_M15091235_M0_LK_SR":":",
}

sim_dic_lw = {
# BLh -> "green"
    "BLh_M10201856_M0_LK_LR":  1.,
    "BLh_M13641364_M0_LK_SR":  1.,
    "BLh_M11461635_M0_LK_SR":  1.,
    "BLh_M10201856_M0_SR":     2.,
    "BLh_M10651772_M0_LK_SR":  1.,
    "BLh_M10651772_M0_LK_LR":  1.,
    # DD2 -> blue
    "DD2_M13641364_M0_SR":        2.,
    "DD2_M13641364_M0_LK_SR_R04": 1.,
    "DD2_M13641364_M0_SR_R04":    1.,
    "DD2_M15091235_M0_LK_SR":     1.,
    "DD2_M14971245_M0_SR":        2.,
    # LS220 -> red
    "LS220_M11461635_M0_LK_SR": 1.,
    "LS220_M13641364_M0_LK_SR_restart":1.,
    "LS220_M13641364_M0_SR":2,
    # SLy4 -> magenta
    "SLy4_M13641364_M0_SR":      2.,
    "SLy4_M14521283_M0_SR":      2.,
    "SLy4_M11461635_M0_LK_SR":   1.,
    # SFHo -> purple
    "SFHo_M13641364_M0_SR":       2.,
    "SFHo_M11461635_M0_LK_SR":    1.
}

datasets_markers = {
    "bauswein":     "h", #"s",
    "hotokezaka":   "d",  #">",
    "dietrich15":   "<",    #"d",
    "sekiguchi15":  "v",  #"p",
    "dietrich16":   ">",  #"D",
    "sekiguchi16":  "^",  #"h",
    "lehner":       "P", #"P",
    "radice":       "*", #"*",
    "kiuchi":       "D", #"X",
    "vincent":      "s", #"v",
    "our":          "o",
    "our_total":    ".",
    "reference":    "o"
}

datasets_labels = {
    "bauswein":     "Bauswein+2013", #"s",
    "hotokezaka":   "Hotokezaka+2013",  #">",
    "dietrich15":   "Dietrich+2015",    #"d",
    "sekiguchi15":  "Sekiguchi+2015",  #"p",
    "dietrich16":   "Dietrich+2016",  #"D",
    "sekiguchi16":  "Sekiguchi+2016",  #"h",
    "lehner":       "Lehner+2016", #"P",
    "radice":       "Radice+2018", #"*",
    "kiuchi":       "Kiuchi+2019", #"X",
    "vincent":      "Vincent+2019", #"v",
    "our":          "This work",
    "our_total":    "This work Total",
    "reference":    "Reference"
}

datasets_colors = {
    "dietrich15":"gray",
    "dietrich16":"gray",
    "sekiguchi16":"black",
    "radice":"green",
    "kiuchi":"gray",
    "vincent":"red",
    "our":"blue",
    "reference": "blue"
}

eos_dic_marker = {
    "DD2": "s",
    "BLh": "d",
    "LS220": "P",
    "SLy4": "o",
    "SFHo": "h"
}


eos_dic_color = {
    "DD2": "blue",
    "BLh": "green",
    "LS220": "red",
    "SLy4": "orange",
    "SFHo": "magenta"
}

""" ------- READING .CSV Table ------ """

simulations = pandas.read_csv(Paths.to_summarytable)
simulations = simulations.set_index("name")

""" ------- MODIFYING DATAFRAME ----- """

EOS = sorted(list(set(simulations["EOS"])))
simulations["C1"] = simulations["M1"]/simulations["R1"]
simulations["C2"] = simulations["M2"]/simulations["R2"]
simulations["M"] = simulations["M1"] + simulations["M2"]
simulations["C"] = simulations["M"]/(simulations["R1"] + simulations["R2"])

# simulations["M"] = simulations["M1"] + simulations["M2"]
simulations["nu"] = simulations["M1"]*simulations["M2"]/(simulations["M"]**2)
simulations["q"] = get_q(simulations)

simulations["Eb"] = (simulations["M"] - simulations["MADM"] + simulations["EGW"])/(simulations["M"]*simulations["nu"])
simulations["j"] = (simulations["JADM"] - simulations["JGW"])/(simulations["M"]**2*simulations["nu"])

simulations["Mfinal"] = simulations["MADM"] - simulations["EGW"]
simulations["Jfinal"] = simulations["JADM"] - simulations["JGW"]
simulations["afinal"] = simulations["Jfinal"]/(simulations["Mfinal"]**2)

simulations["Mwinddot"] = simulations["Mej_tot-bern_geoend"] / simulations["t98mass-geo"]
simulations["Mnuwinddot_theta"] = simulations["Mej_tot-bern_geoend"] / simulations["t98mass-geo"]

simulations["0.4Mdisk3D"] = simulations["Mdisk3D"] * 0.4
""" --------- ADD DATA FROM RNS ----------- """


# def add_merger_outcome(sims):
#
#     import rns as rns
#
#     L = []
#     for _, m in sims.iterrows():
#
#         if len(seq.keys()) == 0: compute_bounding_sequences_in_J_M0_for_EOS(m.EOS)
#         if not "Jmax" in seq.keys(): get_Jmax_and_M0_sup(m.EOS)
#
#         rns.
#
#         if m.tcoll_gw * 1.e3 < 1.5:  # ms
#             L.append(rns.Options.list_merger_outcomes[0])
#         elif m.tcoll_gw * 1.e3 > 1.5 and not np.isinf(m.tcoll_gw):  # ms
#             L.append(rns.Options.list_merger_outcomes[1])
#         elif m.Mb < seq[m.EOS]["M0_TOV"]:
#             L.append(rns.Options.list_merger_outcomes[2])
#         elif m.Mb < seq[m.EOS]["M0_sup"]:
#             L.append(rns.Options.list_merger_outcomes[3])
#         else:
#             L.append(Options.list_merger_outcomes[4])
#     sims["outcome"] = L

""" ------- MASKS FOR DATAFRAME ------ """

mask_for_sr = simulations.resolution == "SR"
mask_for_wrong_mbi = abs(simulations["Mbi_Mb"] - 1.) > 0.04
mask_for_correct_mbi = abs(simulations["Mbi_Mb"] - 1.) < 0.04
mask_for_with_disk3D = simulations.Mdisk3D.notnull()
mask_for_visous = simulations.viscosity == "LK"
mask_for_long = (simulations.tend > 50. / 1.e3 ) #| (simulations.index == "BLh_M10651772_M0_LK_SR")
mask_for_with_bh = np.isfinite(simulations.tcoll_gw)
mask_for_with_dynej = simulations[params.Mej_v_n] > params.Mej_min
mask_for_non_ahfix = simulations.AHfix!="yes"
mask_for_ahfix = simulations.AHfix=="yes"
#
simulations_nonblacklisted = simulations.drop(params.blacklist, axis=0)

"""----------------------------------- """


print(simulations.keys())

if __name__ == '__main__':




    # BLh_M12591482_M0_LR

    # failed = []
    # for name, sim in simulations.iterrows():
    #     try:
    #         print("{}".format(name))
    #         # os.system("python /data01/numrel/vsevolod.nedora/bns_ppr_tools/profile.py -s {} -t plotmass --it all --overwrite yes".format(name))
    #         os.system(
    #             "python /data01/numrel/vsevolod.nedora/bns_ppr_tools/preanalysis.py -s {} -t update_status --overwrite yes".format(
    #                 name))
    #     except KeyboardInterrupt:
    #         exit(1)
    #     except:
    #         failed.append(name)

    # print("Failed: ")
    # print(failed)
    # exit(1)

    # print(simulations.loc["SFHo_M13641364_M0_LK_LR"][["Mej_tot-geo", "tmerg_r"]])
    # print(simulations.loc["LS220_M13641364_M0_HR"][["Mej_tot-geo", "tmerg_r"]])
    # exit(1)

    check_initial_table_models(simulations)
    # exit(1)
    print("------------------------------")
    print("# Simulations       = {}".format(len(simulations)))
    print("# correct_init_data = {}".format(len(simulations[mask_for_correct_mbi])))
    print("# wrong_init_data   = {}".format(len(simulations[mask_for_wrong_mbi])))
    print("# BH                = {}".format(len(simulations[mask_for_with_bh])))
    print("# SR                = {}".format(len(simulations[mask_for_sr])))
    print("# SR + Long         = {}".format(len(simulations[mask_for_sr & mask_for_long])))
    print("# SR + Disks        = {}".format(len(simulations[mask_for_sr & mask_for_with_disk3D])))
    print("# SR + BH           = {}".format(len(simulations[mask_for_sr & mask_for_with_bh])))
    print("# SR + Long + Disk  = {}".format(len(simulations[mask_for_sr & mask_for_long & mask_for_with_disk3D])))
    print("# SR + BH + Disk    = {}".format(len(simulations[mask_for_sr & mask_for_with_disk3D & mask_for_with_bh])))
    print("------------ BH --------------")
    for eos in EOS:
        coll = (simulations.EOS == eos) & (np.isfinite(simulations.tcoll_gw))
        stable = (simulations.EOS == eos) & (np.isinf(simulations.tcoll_gw))
        print("{} BH: {}/{} Stable {}/{}".format(eos, len(simulations[coll]), len(simulations),
                                                 len(simulations[stable]), len(simulations))),
        sr = (simulations.EOS == eos) & (simulations.resolution == "SR")
        coll = sr & (np.isfinite(simulations.tcoll_gw))
        stable = sr & (np.isinf(simulations.tcoll_gw))
        print("\t[SR] {} BH: {}/{} Stable {}/{}".format(eos, len(simulations[coll]), len(simulations[sr]),
                                                        len(simulations[stable]), len(simulations[sr]))),
        print('')
    print('---------|LONG & DISK|----------')
    print(simulations[mask_for_sr & mask_for_long & mask_for_with_disk3D]["Mdisk3D"])
    print('---------|BH & DISK|----------')
    print(simulations[mask_for_sr & mask_for_with_disk3D & mask_for_with_bh]["Mdisk3D"])
    print("---------|wind|-----------")
    print(simulations[mask_for_sr & mask_for_long][["Mej_tot-bern_geoend", "t98mass-geo", "Mwinddot"]])

    #
    # test_group_disk(sims)

