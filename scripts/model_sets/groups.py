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

#

class Paths:
    to_group_table = "../datasets/groups.csv"

list_long = [
    ""
]

""" ------- READING .CSV Table ------ """

groups = pandas.read_csv(Paths.to_group_table)
#groups = groups.set_index("group")

""" ------- MODIFYING DATAFRAME ----- """

groups["Mtot"] = groups["M1"] + groups["M2"]
groups["Mchirp"] = ((groups["M1"] * groups["M2"]) ** (3./5.)) / (groups["Mtot"]**(1./5.))
groups["Mej_tot-tot"] = groups["Mej_tot-geo"] + groups["Mej_tot-bern_geoend"]
groups["vel_inf_ave-tot"] = (groups["vel_inf_ave-geo"] * groups["Mej_tot-geo"] + \
                             groups["vel_inf_ave-bern_geoend"] * groups["Mej_tot-bern_geoend"]) \
                            / groups["Mej_tot-tot"]
groups["0.4Mdisk3D"] = groups["Mdisk3D"] * 0.4

translation = {
    'models':"groups",
    "Mdisk3D":"Mdisk3D",
    "0.4Mdisk3D":"0.4Mdisk3D",
    "Lambda":"Lambda",
    "q":"q",
    "Mej_tot-geo":"Mej_tot-geo",
    "Mtot": "Mtot",
    "Mchirp":"Mchirp",
    "vel_inf_ave-geo":"vel_inf_ave-geo",
    "Ye_ave-geo":"Ye_ave-geo",
    "vel_inf_ave-bern_geoend":"vel_inf_ave-bern_geoend",
    "Mej_tot-bern_geoend":"Mej_tot-bern_geoend",
    "vel_inf_ave-tot": "vel_inf_ave-tot",
    "Mej_tot-tot": "Mej_tot-tot",
    "M1":"M1",
    "M2":"M2",
    "C1":"C1",
    "C2":"C2",
    "Mb1":"Mb1",
    "Mb2":"Mb2"
}

def get_outcome_marker(group, v_n = "outcome"):
    #
    marker_dic = {
        "NS": "d",
        "BH": "o",
        "PC": "s",
        "mix": "p",
        "unclear": "X"
    }
    #
    string = str(group[v_n])
    vals = list(string.strip('][').split(', '))
    #
    u_val = list(set(vals))
    if len(u_val) == 1:
        return marker_dic[u_val[0].strip("''")]
    elif len(u_val) == 2:
        return marker_dic["mix"]
    else:
        raise NameError("Unrecognized set for {} : {} | group: {}"
                        .format(v_n, vals, group))

def get_mod_err(v_n, mod_dic, simulations, arr=np.zeros(0,)):
    if len(arr) == 0:
        arr = np.array(simulations["err-" + translation[v_n]])

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


def get_mod_data(v_n, mod_dic, simulations, arr=np.zeros(0,)):
    if len(arr) == 0: arr = np.array(simulations[translation[v_n]])
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
    # print(arr)
    return arr



if __name__ == '__main__':
    print(groups.index)
    print(groups["Mej_tot-geo"])
    print(groups.keys())

    ''' -- average --- '''
    print(groups["Mej_tot-geo"])
    print(groups["Mej_tot-geo"].describe(percentiles=[0.9]))

    print(groups["vel_inf_ave-geo"])
    print(groups["vel_inf_ave-geo"].describe(percentiles=[0.9]))

    print(groups["Ye_ave-geo"])
    print(groups["Ye_ave-geo"].describe(percentiles=[0.9]))

    #print(groups[["C2", "M2"]])

    #print("average dyn. ejecta mass: {} {} ".format(np.mean(groups["Mej_tot-geo"]), np.percentile(groups["Mej_tot-geo"], 90)))

