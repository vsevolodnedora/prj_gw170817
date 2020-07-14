#!/usr/bin/env python

"""

https://arxiv.org/abs/1903.01466

Attempt to show that Lambda<400 can still explain the AT2017gfo

Models with Low Lambda ~ 200
Q : 1, 0.774
EOS: piece-wise polytropes with 3 segemnts  (claim to be good to investigate genergic models and not exact nuclear model)

Neutrinos : unknown
m1>m2

"""


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
    to_csv_table = "../datasets/kiuchi_summary_2019.csv"

class Struct(object):
    Mej_min = 5e-5
    Mej_err = lambda _, Mej: 0.5 * Mej + Struct.Mej_min
    Mdisk_min = 5e-4
    Mdisk_err = lambda _, MdiskPP: 0.5 * MdiskPP + Struct.Mdisk_min
    vej_def_err = 0.02
    vej_err = lambda _, v: 1 * np.full(len(v), Struct.vej_def_err)
    pass

params = Struct()

""" ------- READING .CSV Table ------ """

simulations = pandas.read_csv(Paths.to_csv_table, " ")
simulations = simulations.set_index("model")

translation = {
    "Lambda":"Lambda",
    "Mdisk3Dmax":"Mdisk",
    "Mdisk3D":"Mdisk",
    "Mdisk":"Mdisk",
    "q":"q",
    "Mej_tot-geo": "Mdyn",
    "Mtot": "Mtot",
    "Mchirp":"Mchirp",
    "M1": "M1",
    "M2": "M2",
    "C1": "C1",
    "C2": "C2",
    "Mb1": "Mb1",
    "Mb2": "Mb2"
    # "vel_inf_ave-geo":"None"  # does not have vel_inf
}

""" ------- MODIFYING DATAFRAME ----- """

simulations["q"] = 1. / simulations["q"]
simulations["Mtot"] = simulations["M1"]  + simulations["M2"]
simulations["Mchirp"] = ((simulations["M1"] * simulations["M2"]) ** (3./5.)) / (simulations["Mtot"]**(1./5.))

#simulations["Mdisk"] = simulations["Mdisk"] / 1.e2

mask_for_resolved_disk = simulations["Mdisk"] > 1.e-3
mask_for_with_tov_data = simulations["Mb1"] > 0.

def get_mod_err(v_n, mod_dic, simulations, arr=np.zeros(0,)):
    if len(arr) == 0:
        val_arr = np.array(simulations[translation[v_n]], dtype=float)
        if v_n == "Mej_tot-geo": arr = params.Mej_err(val_arr)
        elif v_n == "Mdisk3D" or v_n == "Mdisk3Dmax": arr = params.Mdisk_err(val_arr)
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
    return arr


if __name__ == "__main__":

    print("{} {} {}".format(simulations["M1"], simulations["M2"], simulations["Mtot"]))

    print(" ---------- ")
    print("simulations:        {}".format(len(simulations)))
    print("with initial data   {}".format(len(simulations[mask_for_with_tov_data])))
    print("with resolved disk: {}".format(len(simulations[mask_for_resolved_disk])))
    print("with in.data and disk: {}".format(len(simulations[mask_for_resolved_disk & mask_for_with_tov_data])))
    #
    # print(simulations.keys())
    print(simulations[mask_for_with_tov_data][["q", "Mdyn"]])
    print(simulations.keys())