#!/usr/bin/env python

'''

https://arxiv.org/pdf/1504.01266.pdf

[21]

Model EOS M1 M2 Mb1 Mb2 C1 C2 k2T Momega22 MADM JADM tmerg Momega22 fmerg
remnant tauHMNS Momega22* f* Momega222 f2 Mej Tej Mdisk Mbh jBH

Mdisk = M at 200Msun after BH forms (~1ms) after BH formation

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
    to_csv_table = "../datasets/dietrich_summary_2015.csv"
    to_csv_table_compilation = "../datasets/dietrich_ujevic_sammary.csv"

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

simulations = pandas.read_csv(Paths.to_csv_table, sep=" ", index_col="model")
#simulations = simulations.set_index("model")

# tmp_simulations = pandas.read_csv(Paths.to_csv_table)
# tmp_simulations = tmp_simulations.set_index("model")

translation = {
    "vel_inf_ave-geo": "vej",
    "Lambda": "Lambda", # dataset does not have # lam1 and lam2 -- wel... sucks to be you
    "Mdisk3Dmax":"Mdisk",
    "Mdisk3D":"Mdisk",
    "Mdisk":"Mdisk",
    "q":"q",
    "Mej_tot-geo": "Mej",
    "Mtot":"Mtot",
    "Mchirp":"Mchirp",
    "M1": "M1",
    "M2": "M2",
    "C1": "C1",
    "C2": "C2",
    "Mb1": "Mb1",
    "Mb2": "Mb2"
}

""" ------- MODIFYING DATAFRAME ----- """

# simulations["q"] = 1. / simulations["q"]
simulations["Mej"] = simulations["Mej"] / 1.e3
simulations["Tej"] = simulations["Tej"] / 1.e4
simulations["Mdisk"] = simulations["Mdisk"] / 1.e2
simulations["q"] = simulations["M1"] / simulations["M2"]
simulations["Mtot"] = simulations["M1"] + simulations["M2"]
simulations["Mchirp"] = ((simulations["M1"] * simulations["M2"]) ** (3./5.)) / (simulations["Mtot"]**(1./5.))


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

# simulations["Mdisk"] = simulations["Mdisk"] / 1.e2

# """ Matteo's summary """

# datatable = pandas.read_csv("../output/DataTablePostMerger.csv", "\t")
# datatable = datatable.set_index("#name")
# print(datatable)

""" --------------------------------- """

mask_for_with_disk = simulations.Mdisk > 0.
mask_for_with_sr = ((simulations.resolution == "R2n") | (simulations.resolution == "R2n2"))


def get_lambda_tilde(sims):

    eos_dic = {
        "ALF2": "/data01/numrel/vsevolod.nedora/Data/TOVs/franks/ALF2_sequence.txt",
        "H4":   "/data01/numrel/vsevolod.nedora/Data/TOVs/franks/H4_sequence.txt",
        "MS1b": "/data01/numrel/vsevolod.nedora/Data/TOVs/franks/MS1b_sequence.txt",
        "MS1": "/data01/numrel/vsevolod.nedora/Data/TOVs/franks/MS1_sequence.txt",
        "SLy":  "/data01/numrel/vsevolod.nedora/Data/TOVs/franks/SLy_sequence.txt"
    }

    from scipy import interpolate

    lambda_tilde = []

    for name, par_dic in sims.iterrows():
        #
        eos = par_dic["EOS"]
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

        m_grav = m_grav[:idx]
        m_bary = m_bary[:idx]
        r = r[:idx]
        comp = comp[:idx]
        kl = kl[:idx]
        lamb = lamb[:idx]

        kind = "linear"
        interp_grav_bary =  interpolate.interp1d(m_bary, m_grav, kind=kind)
        interp_lamb_bary =  interpolate.interp1d(m_bary, lamb, kind=kind)
        interp_comp_bary =  interpolate.interp1d(m_bary, comp, kind=kind)
        interp_k_bary =     interpolate.interp1d(m_bary, kl, kind=kind)
        interp_r_bary =     interpolate.interp1d(m_bary, r, kind=kind)
        #
        if par_dic["Mb1"] != '':
            par_dic["lam21"] = float(interp_lamb_bary(float(par_dic["Mb1"])))  # lam21
            par_dic["Mg1"] = float(interp_grav_bary(float(par_dic["Mb1"])))
            par_dic["C1"] = float(interp_comp_bary(float(par_dic["Mb1"])))  # C1
            par_dic["k21"] = float(interp_k_bary(float(par_dic["Mb1"])))
            par_dic["R1"] = float(interp_r_bary(float(par_dic["Mb1"])))
            # run["R1"] = run["M1"] / run["C1"]
        #
        if par_dic["Mb2"] != '':
            par_dic["lam22"] = float(interp_lamb_bary(float(par_dic["Mb2"])))  # lam22
            par_dic["Mg2"] = float(interp_grav_bary(float(par_dic["Mb2"])))
            par_dic["C2"] = float(interp_comp_bary(float(par_dic["Mb2"])))  # C2
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
    sims["Lambda"] = lambda_tilde
    return sims

def get_ejecta_vel_from_diet_and_ujecvic():

    import models_dietrich_ujevic2016 as du
    comp_sel = du.simulations[du.simulations["Ref"] == "[21]"]

    velocities = []
    for i, m in simulations.iterrows():
        print("\n{}".format(i))
        sel = comp_sel[(comp_sel["EOS"] == m.EOS) & (((comp_sel["M1"] == m.M1)  & (comp_sel["M2"] == m.M2)) | ((comp_sel["M1"] == m.M2)  & (comp_sel["M2"] == m.M1)))]
        print("\t{}".format(len(sel)))
        velocities.append(float(sel["vej"]))
    simulations["vej"] = velocities
    print(simulations["vej"])

if __name__ == "__main__":

    # print("models resolution")
    # print(simulations["resolution"])
    print(" all models:            {}".format(len(simulations)))
    print(" resolved simualtions:  {}".format(len(simulations[mask_for_with_sr])))
    print(" models with disk mass: {}".format(len(simulations[mask_for_with_disk])))

    #print(simulations[["Mej", "vinf"]])

    # print(simulations.keys())

    # get_ejecta_vel_from_diet_and_ujecvic()

    print(simulations[mask_for_with_sr][["Lambda", "Mej", "vej"]])

    #
    #simulations = get_lambda_tilde(simulations)
    #print(simulations[["Lambda","Mdisk"]])
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
