#
#
#
#
#

from __future__ import division

import numpy as np
import pandas as pd
import sys
import os
import copy
import h5py
import csv

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')



from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import AutoMinorLocator, FixedLocator, NullFormatter, \
    MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

# from legacy import _models_old as md
# import models_radice as rd
import scipy.optimize as opt # opt.curve_fit()
import statsmodels.api as sm

from uutils import x_y_z_sort

__outplotdir__ = "/data01/numrel/vsevolod.nedora/figs/all3/"

# opt.curve_fit()




# def fitting_function(x, lam):
#     a, b = x
#     return a + b * lam
# def residuals(x, lam, Mdisk):
#     xi = fitting_function(x, lam)
#     return (xi - Mdisk)
# def initial_guess():
#     # a = 0.084
#     a = 0.29
#     b = -0.11
#     return np.array((a,b))
# def collate_data(datasets, x_dic, y_dic):
#     all_x = []
#     all_y = []
#
#
#     for dataset_name in datasets.keys():
#         #
#         dic = datasets[dataset_name]
#         d_cl = dic["data"]  # md, rd, ki ...
#         models = dic["models"]  # models DataFrame
#         #
#         mask = (~np.isnan(models[d_cl.translation[y_dic["v_n"]]])) & \
#                (~np.isnan(models[d_cl.translation[x_dic["v_n"]]]))
#         #
#         models = models[mask]
#         x = d_cl.get_mod_data(x_dic["v_n"], x_dic["mod"], models)
#         y = d_cl.get_mod_data(y_dic["v_n"], y_dic["mod"], models)
#
#         # print(set(np.isnan(y)))
#
#         print(dataset_name)
#         print(len(x), len(y))
#
#         all_x = np.append(all_x, x)
#         all_y = np.append(all_y, y)
#     #
#     all_x, all_y = x_y_z_sort(all_x, all_y)
#     #
#     return all_x, all_y

''' ------------------------------------------------- '''

def create_combine_dataframe(datasets, v_ns, special_instructions, key_for_usable_dataset="fit"):
    import pandas
    new_data_frame = {}
    #

    new_data_frame["models"] = []
    #
    index_arr = []
    for name in datasets.keys():
        dic = datasets[name]
        flag = dic[key_for_usable_dataset]
        if flag:
            d_cl = dic["data"]  # md, rd, ki ...
            # models = dic["models"]
            # print(dic["models"].index); exit(1)
            models = list(dic["models"].index)
            for model in models:
                index_arr.append(model)
            # index_arr.append(list(dic["models"].index))
    new_data_frame['models'] = index_arr
    print(len(index_arr))
    #
    for v_n in v_ns:
        value_arr = []
        for name in datasets.keys():
            dic = datasets[name]
            flag = dic[key_for_usable_dataset]
            if flag:
                d_cl = dic["data"]  # md, rd, ki ...
                models = dic["models"]
                print("appending dataset: {}".format(name))
                x = d_cl.get_mod_data(v_n, special_instructions, models)
                value_arr = np.append(value_arr, x)
                #
        print(len(value_arr))
        new_data_frame[v_n] = value_arr
    #
    df = pandas.DataFrame(new_data_frame, index=new_data_frame["models"])

    return df

''' ------------------------------------------------- '''
###  ye -- lambda
# def fitting_function_ye_lambda(x, v):
#     a, b = x
#     return a * 1.e-5 * v.Lambda + b
# def residuals_ye(x, data, v_n ="Ye_ave-geo"):
#     xi = fitting_function_ye_lambda(x, data)
#     return (xi - data[v_n])
# def fitting_coeffs_ye_lambda():
#     a = -1.09883225742
#     b = 0.16797259046
#     return np.array((a, b))
# def complex_fic_data_ye_module(datasets, fitting_function, coefs, key_for_usable_dataset="fit"):
#
#
#     #
#     v_ns = ["M1", "M2", "C1", "C2", "Mb1", "Mb2", "Lambda", "q", "Ye_ave-geo"]
#     #
#     dataframe = create_combine_dataframe(datasets, v_ns, {})
#     #
#     dataframe = dataframe[np.isfinite(dataframe["Ye_ave-geo"])]
#     print(np.array(np.isfinite(dataframe["Ye_ave-geo"])))
#
#     x0 = coefs#fitting_coeffs_vinf()
#     fitting_function_ye = fitting_function
#     print("chi2 original: " + str(np.sum(residuals_ye(x0, dataframe) ** 2)))
#     dataframe["Ye_ave_fit"] = fitting_function_ye(x0, dataframe)
#
#     res = opt.least_squares(residuals_ye, coefs, args=(dataframe,))
#     print("chi2 fit: " + str(np.sum(residuals_ye(res.x, dataframe) ** 2)))
#     print("Fit coefficients:")
#     print("  a = {}".format(res.x[0]))
#     print("  b = {}".format(res.x[1]))
#     # print("  c = {}".format(res.x[2]))
#     # print("  d = {}".format(res.x[3]))
#     # print("  n = {}".format(res.x[4]))
#     # print("  c = {}".format(res.x[2]))
#     dataframe["vej_fit"] = fitting_function_ye(res.x, dataframe)
#
#     return dataframe

# ye
def fitting_function_ye(x, v):
    a, b, c = x
    return a*1e-5*(v.M1/v.M2)*(1. + c*1e5*v.C1) + \
           a*1e-5*(v.M2/v.M1)*(1. + c*1e5*v.C2) + b

def residuals_ye(x, data, v_n ="Ye_ave-geo"):
    xi = fitting_function_ye(x, data)
    return (xi - data[v_n])

def fitting_coeffs_ye():
    a = 0.139637775679
    b = 0.33996686385
    c = -3.70301958353
    return np.array((a,b,c))
def fitting_coeffs_ye_our_david():
    a = -0.327870598759
    b = 0.806936090228
    c = -0.228575548358
    return np.array((a,b,c))
def fitting_coeffs_ye2():
    a = 0.117330004048
    b = 0.328304694267
    c = -4.18036435314

    return np.array((a,b,c))

def complex_fic_data_ye():
    import models_dietrich2016 as di
    import models_vincent as vi
    import models_radice as rd
    import groups as md
    import models_kiuchi as ki
    #
    datasets = {}
    # datasets["kiuchi"] =    {"models": ki.simulations, "data": ki}
    datasets["radice"] = {"models": rd.simulations[rd.fiducial], "data": rd}
    # datasets["dietrich"] =  {"models": di.simulations[di.mask_for_with_sr], "data": di}
    datasets["vincent"] = {"models": vi.simulations, "data": vi}
    datasets['our'] = {"models": md.groups, "data": md}

    #
    v_ns = ["M1", "M2", "C1", "C2", "Mb1", "Mb2", "Lambda", "q",
            "vel_inf_ave-geo", "Mej_tot-geo", "Ye_ave-geo", "Mdisk3D"]
    #
    dataframe = create_combine_dataframe(datasets, v_ns, {})
    #


    x0 = fitting_coeffs_ye()
    print("chi2 original: " + str(np.sum(residuals_ye(x0, dataframe) ** 2)))
    dataframe["ye_fit_tim"] = fitting_function_ye(x0, dataframe)

    res = opt.least_squares(residuals_ye(), fitting_coeffs_ye(), args=(dataframe,))
    print("chi2 fit: " + str(np.sum(residuals_ye(res.x, dataframe) ** 2)))
    print("Fit coefficients:")
    print("  a = {}".format(res.x[0]))
    print("  b = {}".format(res.x[1]))
    print("  c = {}".format(res.x[2]))
    # print("  d = {}".format(res.x[3]))
    # print("  n = {}".format(res.x[4]))
    # print("  c = {}".format(res.x[2]))
    dataframe["vej_fit"] = fitting_function_ye(res.x, dataframe)

    return dataframe
def complex_fic_data_ye_module(datasets, fitting_function, coefs, key_for_usable_dataset="fit"):


    #
    v_ns = ["M1", "M2", "C1", "C2", "Mb1", "Mb2", "Lambda", "q", "Ye_ave-geo"]
    #
    dataframe = create_combine_dataframe(datasets, v_ns, {})
    #
    dataframe = dataframe[np.isfinite(dataframe["Ye_ave-geo"])]
    print(np.array(np.isfinite(dataframe["Ye_ave-geo"])))

    x0 = coefs#fitting_coeffs_vinf()
    fitting_function_ye = fitting_function
    print("chi2 original: " + str(np.sum(residuals_ye(x0, dataframe) ** 2)))
    dataframe["Ye_ave_fit"] = fitting_function_ye(x0, dataframe)

    res = opt.least_squares(residuals_ye, coefs, args=(dataframe,))
    print("chi2 fit: " + str(np.sum(residuals_ye(res.x, dataframe) ** 2)))
    print("Fit coefficients:")
    print("  a = {}".format(res.x[0]))
    print("  b = {}".format(res.x[1]))
    print("  c = {}".format(res.x[2]))
    # print("  d = {}".format(res.x[3]))
    # print("  n = {}".format(res.x[4]))
    # print("  c = {}".format(res.x[2]))
    dataframe["vej_fit"] = fitting_function_ye(res.x, dataframe)

    return dataframe

''' ----------- '''
# mej
def fitting_function_mej(x, v):
    a, b, c, d, n = x

    return (a * (v.M2 / v.M1) ** (1.0 / 3.0) * (1. - 2 * v.C1) / (v.C1) + b * (v.M2 / v.M1) ** n +
             c * (1 - v.M1 / v.Mb1)) * v.Mb1 + \
            (a * (v.M1 / v.M2) ** (1.0 / 3.0) * (1. - 2 * v.C2) / (v.C2) + b * (v.M1 / v.M2) ** n +
             c * (1 - v.M2 / v.Mb2)) * v.Mb2 + \
            d
def residuals_mej(x, data, v_n = "Mej_tot-geo"):
    xi = fitting_function_mej(x, data)
    return 1e-3*(xi - 1e3*data[v_n])#/(models.params.Mej_err(data.Mej))
def fitting_coeffs_mej():
    a = 1.74906283433
    b = 14.3723379753
    c = -19.4680941386
    d = -54.807479436
    n = -0.503884305092
    return np.array((a, b, c, d, n))
def fitting_coeffs_mej_tim():
    a = -1.35695
    b = 6.11252
    c = 49.43355
    d = 16.1144
    n = -2.5484
    return np.array((a, b, c, d, n))
def fitting_coeffs_mej_david():
    a = -0.657
    b = 4.254
    c = -32.61
    d = 5.205
    n = -0.773
    return np.array((a, b, c, d, n))
def fitting_coeffs_mej_our():
    a = 0.73981628215
    b = -16.1216173699
    c = 88.9549552785
    d = 17.1927414116
    n = 0.122190535015

    return np.array((a, b, c, d, n))

def complex_fic_data_mej():
    import models_dietrich2016 as di
    import models_vincent as vi
    import models_radice as rd
    import groups as md
    import models_kiuchi as ki
    #
    datasets = {}
    datasets["kiuchi"] =    {"models": ki.simulations[ki.mask_for_with_tov_data], "data": ki}
    datasets["radice"] =    {"models": rd.simulations[rd.fiducial], "data": rd}
    datasets["dietrich"] =  {"models": di.simulations[di.mask_for_with_sr], "data": di}
    datasets["vincent"] =   {"models": vi.simulations, "data": vi}
    datasets['our'] =       {"models": md.groups, "data": md}

    #
    v_ns = ["M1", "M2", "C1", "C2", "Mb1", "Mb2", "Lambda", "q",
           "Mej_tot-geo"]
    #
    dataframe = create_combine_dataframe(datasets, v_ns, {})
    #
    dataframe = dataframe[np.isfinite(dataframe["Mej_tot-geo"])]
    print(np.array(np.isfinite(dataframe["Mej_tot-geo"])))

    x0 = fitting_coeffs_mej()
    print("chi2 original: " + str(np.sum(residuals_mej(x0, dataframe) ** 2)))
    dataframe["ye_fit_tim"] = fitting_function_mej(x0, dataframe)

    res = opt.least_squares(residuals_mej, fitting_coeffs_mej(), args=(dataframe,))
    print("chi2 fit: " + str(np.sum(residuals_mej(res.x, dataframe) ** 2)))
    print("Fit coefficients:")
    print("  a = {}".format(res.x[0]))
    print("  b = {}".format(res.x[1]))
    print("  c = {}".format(res.x[2]))
    print("  d = {}".format(res.x[3]))
    print("  n = {}".format(res.x[4]))
    # print("  c = {}".format(res.x[2]))
    dataframe["vej_fit"] = fitting_function_mej(res.x, dataframe)

    return dataframe
def complex_fic_data_mej_module(datasets, coefs, key_for_usable_dataset="fit"):
    """
    key_for_usable_dataset -- if dataset[key] == True -- use it
    :param datasets: {"name":{"models}:md.simulations, "data":md}

    """
    #
    #
    # datasets = {}
    # datasets["kiuchi"] =    {"models": ki.simulations[ki.mask_for_with_tov_data], "data": ki}
    # datasets["radice"] =    {"models": rd.simulations[rd.fiducial], "data": rd}
    # datasets["dietrich"] =  {"models": di.simulations[di.mask_for_with_sr], "data": di}
    # datasets["vincent"] =   {"models": vi.simulations, "data": vi}
    # datasets['our'] =       {"models": md.groups, "data": md}

    #
    v_ns = ["M1", "M2", "C1", "C2", "Mb1", "Mb2", "Lambda", "q", "Mej_tot-geo"]
    #
    dataframe = create_combine_dataframe(datasets, v_ns, {}, key_for_usable_dataset)
    print(dataframe.statistics)
    #
    dataframe = dataframe[np.isfinite(dataframe["Mej_tot-geo"])]
    print(np.array(np.isfinite(dataframe["Mej_tot-geo"])))

    x0 =  coefs# fitting_coeffs_mej()
    print("chi2 original: " + str(np.sum(residuals_mej(x0, dataframe) ** 2)))
    dataframe["ye_fit_tim"] = fitting_function_mej(x0, dataframe)

    res = opt.least_squares(residuals_mej, coefs, args=(dataframe,))
    print("chi2 fit: " + str(np.sum(residuals_mej(res.x, dataframe) ** 2)))
    print("Fit coefficients:")
    print("  a = {}".format(res.x[0]))
    print("  b = {}".format(res.x[1]))
    print("  c = {}".format(res.x[2]))
    print("  d = {}".format(res.x[3]))
    print("  n = {}".format(res.x[4]))
    # print("  c = {}".format(res.x[2]))
    dataframe["vej_fit"] = fitting_function_mej(res.x, dataframe)

    return dataframe

''' ----------- '''
# vinf
def fitting_function_vinf(x, v):
    a, b, c = x
    return a*(v.M1/v.M2)*(1. + c*v.C1) + \
           a*(v.M2/v.M1)*(1. + c*v.C2) + b
def residuals_vinf(x, data, v_n = "vel_inf_ave-geo"):
    xi = fitting_function_vinf(x, data)
    return (xi - data[v_n])
def fitting_coeffs_vinf_david():
    a = -0.287
    b = 0.494
    c = -3.000
    return np.array((a,b,c))
def fitting_coeffs_vinf_tim():
    a = -0.219479
    b = 0.444836
    c = -2.67385
    return np.array((a,b,c))
def fitting_coeffs_vinf_our():
    a = -0.53653590802
    b = 1.02532188568
    c = -1.39029278804
    return np.array((a,b,c))

def complex_fic_data_vinf():
    import models_dietrich2016 as di
    import models_vincent as vi
    import models_radice as rd
    import groups as md
    # import kiuchi as ki
    #
    datasets = {}
    # datasets["kiuchi"] =    {"models": ki.simulations[ki.mask_for_with_tov_data], "data": ki}
    datasets["radice"] =    {"models": rd.simulations[rd.fiducial], "data": rd}
    datasets["dietrich"] =  {"models": di.simulations[di.mask_for_with_sr], "data": di}
    datasets["vincent"] =   {"models": vi.simulations, "data": vi}
    datasets['our'] =       {"models": md.groups, "data": md}

    #
    v_ns = ["M1", "M2", "C1", "C2", "Mb1", "Mb2", "Lambda", "q",
           "vel_inf_ave-geo"]
    #
    dataframe = create_combine_dataframe(datasets, v_ns, {})
    #
    dataframe = dataframe[np.isfinite(dataframe["vel_inf_ave-geo"])]
    print(np.array(np.isfinite(dataframe["vel_inf_ave-geo"])))

    x0 = fitting_coeffs_vinf()
    print("chi2 original: " + str(np.sum(residuals_vinf(x0, dataframe) ** 2)))
    dataframe["vel_inf_ave_fit"] = fitting_function_vinf(x0, dataframe)

    res = opt.least_squares(residuals_vinf, fitting_coeffs_vinf(), args=(dataframe,))
    print("chi2 fit: " + str(np.sum(residuals_vinf(res.x, dataframe) ** 2)))
    print("Fit coefficients:")
    print("  a = {}".format(res.x[0]))
    print("  b = {}".format(res.x[1]))
    print("  c = {}".format(res.x[2]))
    # print("  d = {}".format(res.x[3]))
    # print("  n = {}".format(res.x[4]))
    # print("  c = {}".format(res.x[2]))
    dataframe["vej_fit"] = fitting_function_vinf(res.x, dataframe)

    return dataframe

def complex_fic_data_vinf_module(datasets, coefs, key_for_usable_dataset="fit"):


    #
    v_ns = ["M1", "M2", "C1", "C2", "Mb1", "Mb2", "Lambda", "q",
           "vel_inf_ave-geo"]
    #
    dataframe = create_combine_dataframe(datasets, v_ns, {})
    #
    dataframe = dataframe[np.isfinite(dataframe["vel_inf_ave-geo"])]
    print(np.array(np.isfinite(dataframe["vel_inf_ave-geo"])))

    x0 = coefs#fitting_coeffs_vinf()
    print("chi2 original: " + str(np.sum(residuals_vinf(x0, dataframe) ** 2)))
    dataframe["vel_inf_ave_fit"] = fitting_function_vinf(x0, dataframe)

    res = opt.least_squares(residuals_vinf, coefs, args=(dataframe,))
    print("chi2 fit: " + str(np.sum(residuals_vinf(res.x, dataframe) ** 2)))
    print("Fit coefficients:")
    print("  a = {}".format(res.x[0]))
    print("  b = {}".format(res.x[1]))
    print("  c = {}".format(res.x[2]))
    # print("  d = {}".format(res.x[3]))
    # print("  n = {}".format(res.x[4]))
    # print("  c = {}".format(res.x[2]))
    dataframe["vej_fit"] = fitting_function_vinf(res.x, dataframe)

    return dataframe
''' ------------- '''
# mdisk

def fitting_function_mdisk(x, v):
    a, b, c, d = x
    return np.maximum(a + b*(np.tanh((v["Lambda"] - c)/d)), 1e-3)
def residuals_mdisk(x, data, v_n = "Mdisk3D"):
    xi = fitting_function_mdisk(x, data)
    return (xi - data[v_n])
def fitting_coeffs_mdisk():
    a = -0.243087590223
    b = 0.436980750624
    c = 30.4790977667
    d = 332.568017486
    return np.array((a,b,c,d))
def fitting_coeffs_mdisk_david_ours():
    a = 0.0213678698585
    b = 0.13786459117
    c = 340.018532311
    d = 96.2565909525
    return np.array((a,b,c,d))
def complex_fic_data_mdisk_module(datasets, coefs, key_for_usable_dataset="fit"):
    #
    v_ns = ["M1", "M2", "C1", "C2", "Mb1", "Mb2", "Lambda", "q", "Mdisk3D"]
    #
    dataframe = create_combine_dataframe(datasets, v_ns, {})
    #
    dataframe = dataframe[np.isfinite(dataframe["Mdisk3D"])]
    print(np.array(np.isfinite(dataframe["Mdisk3D"])))

    x0 = coefs  # fitting_coeffs_vinf()
    print("chi2 original: " + str(np.sum(residuals_mdisk(x0, dataframe) ** 2)))
    dataframe["mdisk_fit"] = fitting_function_mdisk(x0, dataframe)

    res = opt.least_squares(residuals_mdisk, coefs, args=(dataframe,))
    print("chi2 fit: " + str(np.sum(residuals_mdisk(res.x, dataframe) ** 2)))
    print("Fit coefficients:")
    print("  a = {}".format(res.x[0]))
    print("  b = {}".format(res.x[1]))
    print("  c = {}".format(res.x[2]))
    print("  d = {}".format(res.x[3]))
    # print("  n = {}".format(res.x[4]))
    # print("  c = {}".format(res.x[2]))
    dataframe["mdisk_fit"] = fitting_function_mdisk(res.x, dataframe)

    return dataframe
def complex_fic_data_mdisk():
    import models_dietrich2016 as di
    import models_vincent as vi
    import models_radice as rd
    import groups as md
    import models_kiuchi as ki
    #
    datasets = {}
    datasets["kiuchi"] =    {"models": ki.simulations[ki.mask_for_with_tov_data], "data": ki}
    datasets["radice"] =    {"models": rd.simulations[rd.fiducial], "data": rd}
    datasets["dietrich"] =  {"models": di.simulations[di.mask_for_with_sr], "data": di}
    datasets["vincent"] =   {"models": vi.simulations, "data": vi}
    datasets['our'] =       {"models": md.groups, "data": md}

    #
    v_ns = ["M1", "M2", "C1", "C2", "Mb1", "Mb2", "Lambda", "q",
           "Mdisk3D"]
    #
    dataframe = create_combine_dataframe(datasets, v_ns, {})
    #
    dataframe = dataframe[np.isfinite(dataframe["Mdisk3D"])]
    print(np.array(np.isfinite(dataframe["Mdisk3D"])))

    x0 = fitting_coeffs_mdisk()
    print("chi2 original: " + str(np.sum(residuals_mdisk(x0, dataframe) ** 2)))
    dataframe["mdisk_fit"] = fitting_function_mdisk(x0, dataframe)

    res = opt.least_squares(residuals_mdisk, fitting_coeffs_mdisk(), args=(dataframe,))
    print("chi2 fit: " + str(np.sum(residuals_mdisk(res.x, dataframe) ** 2)))
    print("Fit coefficients:")
    print("  a = {}".format(res.x[0]))
    print("  b = {}".format(res.x[1]))
    print("  c = {}".format(res.x[2]))
    print("  d = {}".format(res.x[3]))
    # print("  n = {}".format(res.x[4]))
    # print("  c = {}".format(res.x[2]))
    dataframe["mdisk_fit"] = fitting_function_mdisk(res.x, dataframe)

    return dataframe

if __name__ == '__main__':
    # complex_fic_data_mdisk()
    # complex_fic_data_vinf()
    complex_fic_data_mej()
    # complex_fic_data_ye()
    # fit_data()