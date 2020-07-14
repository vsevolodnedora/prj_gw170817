#!/usr/bin/env python

"""
https://arxiv.org/pdf/1603.01918.pdf

 no bayonic masses

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
    to_csv_table = "../datasets/sekiguchi2016_summary.csv"

class Struct(object):
    Mej_min = 5e-5
    Mej_err = lambda _, Mej: 0.5 * Mej + Struct.Mej_min
    Mdisk_min = 5e-4
    Mdisk_err = lambda _, MdiskPP: 0.5 * MdiskPP + Struct.Mdisk_min
    vej_def_err = 0.02
    vej_err = lambda _, v: 1 * np.full(len(v), Struct.vej_def_err)
    ye_def_err = 0.01
    Yeej_err = lambda _, v: 1 * np.full(len(v), Struct.ye_def_err)
    pass

params = Struct()


""" ------- READING .CSV Table ------ """

simulations = pandas.read_csv(Paths.to_csv_table, " ")
simulations = simulations.set_index("model")

translation = {
    "Lambda":"Lambda", #-- no bayonic masses
    "Mdisk3Dmax":"Mtorus",
    "Mdisk3D":"Mtorus",
    "Mdisk": "Mtorus",
    "q":"q",
    "Mej_tot-geo": "Mej",
    "Mtot": "Mtot",
    "Mchirp": "Mchirp",
    "vel_inf_ave-geo": "vej",
    "Ye_ave-geo":"Yeej",
    "M1": "M1",
    "M2": "M2",
    # "C1": "C1",
    # "C2": "C2",
    # "Mb1": "Mb1",
    # "Mb2": "Mb2"
}

""" ------- MODIFYING DATAFRAME ----- """

# simulations["q"] = 1. / simulations["q"]
# simulations["Mej"] = simulations["Mej"] / 1.e2
# simulations["Tej"] = simulations["Tej"] / 1.e4
simulations["q"] = 1. / simulations["q"]
simulations["Mtot"] = simulations["M1"] + simulations["M2"]
simulations["Mej"] = simulations["Mej"] / 1.e2
simulations["Mchirp"] = ((simulations["M1"] * simulations["M2"]) ** (3./5.)) / (simulations["Mtot"]**(1./5.))
# simulations["vej"] = np.sqrt((simulations["v_ave_rho"]**2) + (simulations["v_ave_z"]**2))
simulations["Lambda"] = np.full(len(simulations["q"]), np.nan)

def get_mod_err(v_n, mod_dic, simulations, arr=np.zeros(0,)):
    if len(arr) == 0:
        val_arr = np.array(simulations[translation[v_n]], dtype=float)
        if v_n == "Mej_tot-geo": arr = params.Mej_err(val_arr)
        elif v_n == "vel_inf_ave-geo": arr = params.vej_err(val_arr)
        elif v_n == "Ye_ave-geo": arr = params.Yeej_err(val_arr)
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

# simulations["Mdisk"] = simulations["Mdisk"] / 1.e2

# """ Matteo's summary """

# datatable = pandas.read_csv("../output/DataTablePostMerger.csv", "\t")
# datatable = datatable.set_index("#name")
# print(datatable)

""" --------------------------------- """

# mask_for_with_disk = simulations.Mdisk > 0.
# mask_for_with_sr = (simulations.resolution == "R2s") | (simulations.resolution == "R2b") | (simulations.resolution == "R2")
mask_for_with_sr = (simulations.resolution == "high")

if __name__ == "__main__":

    print(" all models:            {}".format(len(simulations)))

    print(simulations[["Mej", "vej"]])

    print(simulations.keys())

    # print(simulations["M2"])

    #
    # simulations = get_lambda_tilde(simulations)
    # print(simulations[["Lambda","Mdisk"]])
    # simulations.to_csv(Paths.to_csv_table)




    # datatable["newname"] = [name.replace('.', '') for name in datatable.index]
    # datatable["M1"] = [np.nan for name in datatable.index]
    #
    # m1 = lambda name: float(name.split('_')[1]) / 1.e3 if name.split('_')[1][0] != "M" else np.nan
    # m2 = lambda name: float(name.split('_')[2]) / 1.e3 if name.split('_')[1][0] != "M" else np.nan
    #
    # datatable["M1"] = [m1(name) for name in datatable["newname"]]
    # datatable["M2"] = [m2(name) for name in datatable["newname"]]
    #
    # for name, sim in simulations.iterrows():
    #     print("name")
    #     loc = datatable[(datatable.eos == sim["eos"]) & (datatable.M1 == sim["M1"]) & (datatable.M2 == sim["M2"])]
    #     lambda_t = sorted(set(list(loc["lambda_t"])))
    #     if len(lambda_t) == 1:
    #         lambda_t = float(lambda_t[0])
    #         # simulations["Lambda"] = lambda_t
    #     elif len(lambda_t) > 1:
    #         lambda_t = np.nan
    #         print("more than one lambda tilde found: {} {} {}".format(name, loc, lambda_t))
    #     else:
    #         lambda_t = np.nan
    #         print("NO lambda tilde found: {} {} {}".format(name, loc, lambda_t))
    #     simulations["Lambda"] = lambda_t
    #     # print(set(list(loc["lambda_t"]))); exit(1)
    #
    # print(datatable[["newname", "M1", "M2"]])
    # print(simulations[["Lambda", "Mdisk"]])


    # print(datatable)

    # print(simulations.keys())
    # print(simulations)
