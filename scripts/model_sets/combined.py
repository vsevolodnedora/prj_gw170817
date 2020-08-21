# python

import numpy as np
import scipy.optimize as opt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats
from collections import OrderedDict
import pandas


class Paths:
    to_table = "../datasets/summary_table.csv"

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

params = Struct()

translation = {
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
    "theta_rms-geo":"theta_rms-geo",
    "M1":"M1",
    "M2":"M2",
    "C1":"C1",
    "C2":"C2",
    "Mb1":"Mb1",
    "Mb2":"Mb2",
    "EOS":"EOS",
    "nus":"nus",
    "arxiv":"arxiv"
}

simulations = pandas.read_csv(Paths.to_table)
# simulations = simulations.set_index("name")

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





def get_err(vals, fit_v_n, error_method="default"):
    Mej_min = 5e-5
    vej_def_err = 0.02
    ye_def_err = 0.01
    MdiskPP_min = 5e-4
    theta_ej_min = 2.0

    # vals = np.array(self.dataframe[self.fit_v_n])
    # if v_n == "Mej_tot-geo": vals = vals / 1e3

    # res = np.zeros(0, )
    if error_method == "std":
        res = np.std(vals)
    elif error_method == "2std":
        res = 2. * np.std(vals)
    elif error_method == "default":
        if fit_v_n == "Mej_tot-geo":
            lambda_err = lambda Mej: 0.5 * Mej + Mej_min
        elif fit_v_n == "vel_inf_ave-geo":
            lambda_err = lambda v: 1. * np.full(len(v), vej_def_err)
        elif fit_v_n == "Ye_ave-geo":
            lambda_err = lambda v: 1. * np.full(len(v), ye_def_err)
        elif fit_v_n == "Mdisk3D":
            lambda_err = lambda MdiskPP: 0.5 * MdiskPP + MdiskPP_min
        elif fit_v_n == "theta_rms-geo":
            lambda_err = lambda v: 1. * np.full(len(v), theta_ej_min)
        else:
            raise NameError("No error method for v_n: {}".format(fit_v_n))
        res = lambda_err(vals)
    elif error_method == "arr":
        raise NameError("aaa")
    else:
        raise NameError("no err method: {}".format(error_method))

    # if v_n == "Mej_tot-geo": res = res * 1e3
    return res

def get_chi2(y_vals, y_expets, y_errs):
    assert len(y_vals) == len(y_expets)
    z = (y_vals - y_expets) / y_errs
    chi2 = np.sum(z ** 2.)
    return chi2

def get_ch2dof(chi2, n, k):
    """
    :param chi2: chi squared
    :param n: number of elements in a sample
    :param k: n of independedn parameters (1 -- mean, 2 -- poly1 fit, etc)
    :return:
    """
    return chi2 / (n - k)

def polyfit(models, v_ns_x="Lambda", v_n_y="Mej_tot-geo", degree=2):

    v_ns_x = list(v_ns_x.split())

    # clearing the dataset to get only physcial values
    models = models[~np.isnan(models[v_n_y])]
    for v_n in v_ns_x:
        models = models[~np.isnan(models[v_n])]

    # getting the quantity that we want fit. and its errors
    y = np.array(models[v_n_y], dtype=float)
    errs = get_err(y, v_n_y)
    # for ejecta mass all fits are for a Mej*1e3
    if v_n_y[0] == "Mej_tot-geo":
        y = 1e3 * y
        errs = 1e3 * errs

    # getting quantities that we want to get fit of. (Trapose for 'transfomer')
    x = []
    for v_n_x in v_ns_x:
        x_ = np.array(models[v_n_x], dtype=float)
        x.append(x_)
        # print(len(x_))
    x = np.reshape(np.array(x), (len(v_ns_x), len(y))).T

    transformer = PolynomialFeatures(degree=degree, include_bias=False)
    transformer.fit(x)
    x_ = transformer.transform(x)
    model = LinearRegression().fit(x_, y)
    r_sq = model.score(x_, y)
    y_pred = model.predict(x_)

    print('coefficient of determination R2: {}'.format(r_sq))
    print('intercept b0: {}'.format(model.intercept_))
    print('coefficients bi: {}'.format(model.coef_))

    # reverting back Mej to compute chi2
    # if v_n_y[0] == "Mej_tot-geo":
    #     y = 1e-3 * y
    #     errs = 1e-3 * errs
    #     y_pred = 1e-3 * y_pred

    # computing Mej
    chi2 = get_chi2(y, y_pred, errs)
    chi2dof = get_ch2dof(chi2, len(y), degree + 1)

    print("{} = f({}) : [{}] : {} : {}".format(v_n_y, v_ns_x, len(y), chi2, chi2dof))

''' subsets '''

basuwein13 = simulations[simulations["bibkey"] == "Bauswein:2013yna"]
reference = simulations[simulations["bibkey"] == "Reference set"]
vincent19 = simulations[simulations["bibkey"] == "Vincent:2019kor"]
# radice18lk = simulations[(simulations["bibkey"] == "Radice:2018pdn")&(simulations["nus"] == "leak")]
# radice18m0 = simulations[(simulations["bibkey"] == "Radice:2018pdn")&(simulations["nus"] == "leakM0")]
lehner16 = simulations[simulations["bibkey"] == "Lehner:2016lxy"]
kiuchi19 = simulations[simulations["bibkey"] == "Kiuchi:2019lls"]
dietrich16 = simulations[simulations["bibkey"] == "Dietrich:2016lyp"]
dietrich15 = simulations[simulations["bibkey"] == "Dietrich:2015iva"]
sekiguchi16 = simulations[simulations["bibkey"] == "Sekiguchi:2016bjd"]
sekiguchi15 = simulations[simulations["bibkey"] == "Sekiguchi:2015dma"]
hotokezaka12 = simulations[simulations["bibkey"] == "Hotokezaka:2012ze"]
radice18lk = simulations[(simulations["bibkey"] == "Radice:2018pdn(LK)")]
radice18m0 = simulations[(simulations["bibkey"] == "Radice:2018pdn(M0)")]

datasets_markers = {
    "bauswein":     "h", #"s",
    "hotokezaka":   "d",  #">",
    "dietrich15":   "<",    #"d",
    "sekiguchi15":  "v",  #"p",
    "dietrich16":   ">",  #"D",
    "sekiguchi16":  "^",  #"h",
    "lehner":       "P", #"P",
    "radice":       "*", #"*",
    "radiceLK":     "*", #"*",
    "radiceM0":     "*", #"*",
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
    "radiceLK":     "Radice+2018(LK)", #"*",
    "radiceM0":     "Radice+2018(M0)",
    "kiuchi":       "Kiuchi+2019", #"X",
    "vincent":      "Vincent+2019", #"v",
    "our":          "Nedora+2020", #"This work",
    "our_total":    "This work Total",
    "reference":    "Reference set"
}

datasets_colors = {
    "bauswein":     "gray",
    "hotokezaka":   "gray",
    "dietrich15":   "gray",
    "dietrich16":   "gray",
    "sekiguchi16":  "black",
    "sekiguchi15":  "black",
    "radice":       "green",
    "kiuchi":       "gray",
    "lehner":       "magenta",
    "radiceLK":     "lime",
    "radiceM0":     "green",
    "vincent":      "red",
    "our":          "blue",
    "reference":    "blue"
}

eos_dic_marker = {
    "DD2": "s",
    "BLh": "d",
    "LS220": "P",
    "SLy4": "o",
    "SFHo": "h",
    # "BHBlp": "v"
}

eos_dic_color = {
    "DD2": "blue",
    "BLh": "green",
    "LS220": "red",
    "SLy4": "orange",
    "SFHo": "magenta",
    # "BHBlp": "lime"
}

# ---

# simulations = simulations[simulations["Mej_tot-geo"] > 5e-5]

if __name__ == "__main__":

    print(len(simulations))

    # i_simulations = simulations[np.isnan(simulations["Mdisk3D"])]
    # print(list(set(i_simulations["bibkey"])))
    # i_simulations = simulations[~np.isnan(simulations["Mdisk3D"])]
    # print(list(set(i_simulations["bibkey"])))
    # for key in simulations["bibkey"]:
    #     print(key)

    # dfname2 = "/home/vsevolod/GIT/bitbucket/bns_gw170817/data/dynej_disc_literature/LiteratureData.csv"
    # keys = {
    #     "Mej_tot-geo": "Mej1e-2",
    #     "Lambda": "tLam",
    #     "q": "q"
    # }
    # allmodels = pandas.read_csv(dfname2)
    # allmodels["Mej_tot-geo"] = allmodels[keys["Mej_tot-geo"]] * 1.e-2 # -> Msun
    # allmodels["Lambda"] = allmodels[keys["Lambda"]]
    # polyfit(models=allmodels, v_ns_x="Lambda", v_n_y="Mej_tot-geo")
    #
    #
    # dfname = "../../datasets/summary_table.csv"
    #
    # allmodels = pandas.read_csv(dfname)
    # print(allmodels.keys())
    # polyfit(models=allmodels, v_ns_x="Lambda", v_n_y="Mej_tot-geo")

    # import models_hotokezaka2013 as hs
    # import models_bauswein2013 as bs
    # hs_models = hs.simulations.sort_values(hs.translation["Mej_tot-geo"])
    # bs_models = bs.simulations.sort_values(bs.translation["Mej_tot-geo"])
    # print(bs_models[hs.translation["Mej_tot-geo"]])
    # print(hs_models[bs.translation["Mej_tot-geo"]])



