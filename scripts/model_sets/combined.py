# python

import numpy as np
import scipy.optimize as opt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats
from collections import OrderedDict
import pandas

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



if __name__ == "__main__":
    dfname2 = "/home/vsevolod/GIT/bitbucket/bns_gw170817/data/dynej_disc_literature/LiteratureData.csv"
    keys = {
        "Mej_tot-geo": "Mej1e-2",
        "Lambda": "tLam",
        "q": "q"
    }
    allmodels = pandas.read_csv(dfname2)
    allmodels["Mej_tot-geo"] = allmodels[keys["Mej_tot-geo"]] * 1.e-2 # -> Msun
    allmodels["Lambda"] = allmodels[keys["Lambda"]]
    polyfit(models=allmodels, v_ns_x="Lambda", v_n_y="Mej_tot-geo")


    dfname = "../../datasets/summary_table.csv"

    allmodels = pandas.read_csv(dfname)
    print(allmodels.keys())
    polyfit(models=allmodels, v_ns_x="Lambda", v_n_y="Mej_tot-geo")




