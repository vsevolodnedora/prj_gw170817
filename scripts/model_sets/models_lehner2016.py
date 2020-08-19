#!/usr/bin/env python

"""

3 realistic microphysical EOS  + neutrino cooling effects
NL3   DD2   SFHo

Varsios Q:
1, 0.85, 0.76

Lambda - FALSE - reason -- NL3 EOS is unknown

conclusion:
higher q -> more ejecta

# url : https://arxiv.org/pdf/1603.00501.pdf
# m1>,2

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
    to_csv_table = "../datasets/lehner2016_summary.csv"

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

simulations = pandas.read_csv(Paths.to_csv_table)
# print(simulations)
simulations = simulations.set_index("model")

simulations["arxiv"] = "https://arxiv.org/abs/1603.00501"
simulations["nus"] = "leak"

""" ------- MODIFYING DATAFRAME ----- """

# simulations["q"] = 1. / (simulations["Mg1"] / simulations["Mg2"])
# simulations["Mtot"] = simulations["Mg1"] + simulations["Mg2"]
# simulations["Mchirp"] = ((simulations["Mg1"] * simulations["Mg2"]) ** (3./5.)) / (simulations["Mtot"]**(1./5.))
# simulations["Mej"] = simulations["Mej"] / 1.e3
# simulations["Lambda"] = np.full(len(simulations["q"]), np.nan)
#
""" --------------------------------- """

def get_mod_err(v_n, mod_dic, simulations, arr=np.zeros(0,)):
    if len(arr) == 0:
        val_arr = np.array(simulations[translation[v_n]], dtype=float)
        if v_n == "Mej_tot-geo": arr = params.Mej_err(val_arr)
        elif v_n == "vel_inf_ave-geo": arr = params.vej_err(val_arr)
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
    "M1": "Mg1",
    "M2": "Mg2",
    "C1": "C1",
    "C2": "C2",
    "Mb1": "Mb1",
    "Mb2": "Mb2",
    "EOS": "EOS",
    "nus": "nus",
    "arxiv": "arxiv"
}


if __name__ == '__main__':

    print(simulations[["EOS", "q", "Lambda", "Mg1", "Mg2", "Mb1", "Mb2", "Mej"]])

    # new_lines = []
    # with open(Paths.to_csv_table, "r") as f:
    #     for line in f.readlines():
    #         print(line)
    #         elements = line.split(' ')
    #         name = elements[0] + '_' + elements[3] + '_' + elements[5]
    #         new_lines.append(name + ' ' + line)
    # with open(Paths.to_csv_table.replace("sammary", "summary"), "w") as f:
    #     f.writelines(new_lines)

    # print(simulations[["q", "Lambda", "Mej"]])
    print(simulations.keys())
    print(list(set(simulations.EOS)))