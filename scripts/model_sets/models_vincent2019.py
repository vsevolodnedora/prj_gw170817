#!/usr/bin/env python

'''

https://arxiv.org/abs/1908.00655

m1>m2

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


class Paths:
    to_csv_table = "../output/vincent_summary_2019.csv"

class Struct(object):
    ye_def_err = 0.01
    Yeej_err = lambda _, v: 1 * np.full(len(v), Struct.ye_def_err)
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
simulations = simulations.set_index("model")

translation = {
    "Lambda":"Lambda",
    "Mdisk3Dmax":"Mdisk",
    "Mdisk3D":"Mdisk",
    "Mdisk":"Mdisk",
    "q":"q",
    "Mej_tot-geo": "Mej_tot",
    "Mtot":"Mtot",
    "Mchirp":"Mchirp",
    "vel_inf_ave-geo":"vel_inf_tot",
    "Ye_ave-geo":"Ye_tot",
    "M1":"M1",
    "M2":"M2",
    "C1":"C1",
    "C2":"C2",
    "Mb1":"Mb1",
    "Mb2":"Mb2"
}

""" ------- MODIFYING DATAFRAME ----- """

simulations["Mtot"] = simulations["M1"] + simulations["M2"]
simulations["Mchirp"] = ((simulations["M1"] * simulations["M2"]) ** (3./5.)) / (simulations["Mtot"]**(1./5.))
# simulations["q"] = 1. / simulations["q"]
# simulations["Mej_pol"] = simulations["Mej_pol"] / 1.e2
# simulations["Mej_eq"] = simulations["Mej_eq"] / 1.e2
# simulations["Mej_tot"] = simulations["Mej_tot"] / 1.e2
# simulations["Mdisk"] = simulations["Mdisk"] / 1.e1
# simulations["Mdisk"] = simulations["Mdisk"] / 1.e2

# """ Matteo's summary """

# datatable = pandas.read_csv("../output/DataTablePostMerger.csv", "\t")
# datatable = datatable.set_index("#name")
# print(datatable)

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


""" --------------------------------- """


def get_lambda_tilde(sims):

    eos_dic = {
        "DD2":      "/data01/numrel/vsevolod.nedora/Data/TOVs/franks/DD2_sequence.txt",
        "LS220":    "/data01/numrel/vsevolod.nedora/Data/TOVs/franks/LS200_sequence.txt",
        "SFHo":     "/data01/numrel/vsevolod.nedora/Data/TOVs/franks/SFHo_sequence.txt",
    }

    from scipy import interpolate

    lambda_tilde = []
    mb1 = []
    mb2 = []
    r1 = []
    r2 = []

    for name, par_dic in sims.iterrows():
        #
        eos = par_dic["eos"]
        #
        tov_table = np.loadtxt(eos_dic[eos])
        #
        m_grav = tov_table[:, 1]
        m_bary = tov_table[:, 2]
        r = tov_table[:, 3]
        comp = tov_table[:, 4]  # compactness
        kl = tov_table[:, 5]
        lamb = tov_table[:, 6]  # lam

        idx = np.argmax(m_grav)
        #
        m_grav = m_grav[:idx]
        m_bary = m_bary[:idx]
        r = r[:idx]
        comp = comp[:idx]
        kl = kl[:idx]
        lamb = lamb[:idx]

        kind = "linear"
        #
        # import matplotlib.pyplot as plt
        # fig = plt.figure()

        #
        interp_grav_bary =  interpolate.interp1d(m_bary, m_grav, kind=kind)
        interp_lamb_bary =  interpolate.interp1d(m_bary, lamb, kind=kind)
        # interp_comp_bary =  interpolate.interp1d(m_bary, comp, kind=kind)
        interp_k_bary =     interpolate.interp1d(m_bary, kl, kind=kind)
        interp_r_bary =     interpolate.interp1d(m_bary, r, kind=kind)
        #
        interp_comp_bary = interpolate.interp1d(comp, m_bary, kind=kind)
        par_dic["Mb1"] = float(interp_comp_bary(float(par_dic["C1"])))
        par_dic["Mb2"] = float(interp_comp_bary(float(par_dic["C2"])))

        #
        if par_dic["Mb1"] != '':
            par_dic["lam21"] = float(interp_lamb_bary(float(par_dic["Mb1"])))  # lam21
            par_dic["Mg1"] = float(interp_grav_bary(float(par_dic["Mb1"])))
            # par_dic["C1"] = float(interp_comp_bary(float(par_dic["Mb1"])))  # C1
            par_dic["k21"] = float(interp_k_bary(float(par_dic["Mb1"])))
            par_dic["R1"] = float(interp_r_bary(float(par_dic["Mb1"])))
            # run["R1"] = run["M1"] / run["C1"]
        #
        if par_dic["Mb2"] != '':
            par_dic["lam22"] = float(interp_lamb_bary(float(par_dic["Mb2"])))  # lam22
            par_dic["Mg2"] = float(interp_grav_bary(float(par_dic["Mb2"])))
            # par_dic["C2"] = float(interp_comp_bary(float(par_dic["Mb2"])))  # C2
            par_dic["k22"] = float(interp_k_bary(float(par_dic["Mb2"])))
            par_dic["R2"] = float(interp_r_bary(float(par_dic["Mb2"])))
            # run["R2"] = run["M2"] / run["C2"]
        #
        if par_dic["Mg1"] != '' and par_dic["Mg2"] != '':
            mg1 = float(par_dic["Mg1"])
            mg2 = float(par_dic["Mg2"])
            mg_tot = mg1 + mg2
            k21 = float(par_dic["k21"])
            k22 = float(par_dic["k22"])
            c1 = float(par_dic["C1"])
            c2 = float(par_dic["C2"])
            lam1 = float(par_dic["lam21"])
            lam2 = float(par_dic["lam22"])

            kappa21 = 2 * ((mg1 / mg_tot) ** 5) * (mg2 / mg1) * (k21 / (c1 ** 5))

            kappa22 = 2 * ((mg2 / mg_tot) ** 5) * (mg1 / mg2) * (k22 / (c2 ** 5))

            par_dic["k2T"] = kappa21 + kappa22

            tmp1 = (mg1 + (12 * mg2)) * (mg1 ** 4) * lam1
            tmp2 = (mg2 + (12 * mg1)) * (mg2 ** 4) * lam2
            lambda_t = (16. / 13.) * (tmp1 + tmp2) / (mg_tot ** 5.)
            lambda_tilde.append(lambda_t)
            #
            mb1.append(par_dic["Mb1"])
            mb2.append(par_dic["Mb2"])
            #
            r1.append(par_dic["R1"])
            r2.append(par_dic["R2"])
    #
    # sims["Lambda"] = lambda_tilde
    sims["Mb1"] = mb1
    sims["Mb2"] = mb2
    #
    sims["R1"] = r1
    sims["R2"] = r2

    return sims

if __name__ == '__main__':

    print(simulations)

    # sims = simulations
    # sims = get_lambda_tilde(sims)
    # sims.to_csv(Paths.to_csv_table)
    # print(simulations[["eos", "q", "Lambda", "Mdisk"]])

    print(simulations.keys())