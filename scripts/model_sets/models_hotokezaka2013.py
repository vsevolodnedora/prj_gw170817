#!/usr/bin/env python

"""

https://arxiv.org/pdf/1212.0905.pdf

EOS
piece-wise polytropes
separate tratment of cold and shocked parts of EOS
m1 > m2

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

class Paths:
    to_csv_table = "../output/hotokezaka2013_summary.csv"

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

""" ------- MODIFYING DATAFRAME ----- """

simulations["q"] = 1. / (simulations["q"])# / simulations["M2"])
simulations["Mtot"] = simulations["M1"] + simulations["M2"]
simulations["Mchirp"] = ((simulations["M1"] * simulations["M2"]) ** (3./5.)) / (simulations["Mtot"]**(1./5.))
simulations["Mej"] = simulations["Mesc"] / 1.e3
simulations["Lambda"] = np.full(len(simulations["q"]), np.nan)
simulations["v"] = np.sqrt(simulations["vr_esc"]**2 + simulations["vz_esc"]**2)

""" --------------------------------- """

def get_mod_err(v_n, mod_dic, simulations, arr=np.zeros(0,)):
    if len(arr) == 0:
        val_arr = np.array(simulations[translation[v_n]], dtype=float)
        if v_n == "Mej_tot-geo": arr = params.Mej_err(val_arr)
        elif v_n == "vel_inf_ave-geo": arr = params.vej_err(val_arr)
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
    if len(arr) == 0: arr = np.array(simulations[translation[v_n]], dtype=float)
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

translation = {
    "Lambda":"Lambda",
    # "Mdisk3Dmax":"Mdisk",
    # "Mdisk3D":"Mdisk",
    # "Mdisk":"Mdisk",
    "q":"q",
    "Mej_tot-geo": "Mej",
    "Mtot":"Mtot",
    "Mchirp":"Mchirp",
    "vel_inf_ave-geo":"v",
    "M1": "M1",
    "M2": "M2",
    # "C1": "C1",
    # "C2": "C2",
    # "Mb1": "Mb1",
    # "Mb2": "Mb2"
}


if __name__ == '__main__':

    # new_lines = []
    # with open(Paths.to_csv_table, "r") as f:
    #     for line in f.readlines():
    #         # print(line)
    #         elements = line.split(' ')
    #         eos = elements[0].split('-')[0]
    #         name = elements[0].replace('-','_') + '_' + elements[1]
    #         # print(name)
    #         new_line = line.replace(elements[0], name + ' ' + eos)
    #         new_lines.append(new_line)
    #         print(new_line)
    #         # new_lines.append(name + ' ' + line)
    # with open(Paths.to_csv_table.replace("sammary", "summary"), "w") as f:
    #     f.writelines(new_lines)

    print(simulations[["q", "Mej"]])
    print(simulations.keys())
    print(list(set(simulations.EOS)))
