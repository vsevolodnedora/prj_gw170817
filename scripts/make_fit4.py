#!/usr/bin/env python2
from __future__ import division

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, least_squares
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class FittingCoefficients:

    @staticmethod
    def default_poly22():
        return (0.1, 0.01, 0.01, 0.001, 0.001, 0.00001)

    @staticmethod
    def default_poly2():
        return (0.1, 0.01, 0.001)

    @staticmethod
    def vej_diet16_default():
        return (-0.3, 0.5, -3.)

    @staticmethod
    def yeej_our_default():
        return (0.1, 0.5, -8.)

    @staticmethod
    def mej_diet16_default():
        # return(-.6, 4.2, -32., 5., 1.) # fails to converge
        return(-0.657, 4.254, -32.61,  5.205, -0.773)

    @staticmethod
    def mej_krug19_default():
        return (-9., 100., -300., 1.)

    @staticmethod
    def mdisk_rad18_default():
        return (0.08, 0.08, 567., 405.)

    @staticmethod
    def mdisk_krug19_default():
        return (-8., 1., 1.)

class FittingFunctions:

    @staticmethod
    def poly_2_qLambda(v, b0, b1, b2, b3, b4, b5):
        #b0, b1, b2, b3, b4, b5 = x
        return b0 + b1 * v.q + b2 * v.Lambda + b3 * v.q ** 2 + b4 * v.q * v.Lambda + b5 * v.Lambda ** 2

    @staticmethod
    def poly_2_Lambda(v, b0, b1, b2):
        #b0, b1, b2 = x
        return b0 + b1*v.Lambda + b2*v.Lambda**2

    @staticmethod
    def poly_2_q(v, b0, b1, b2):
        #b0, b1, b2 = x
        return b0 + b1*v.q + b2*v.q**2

    @staticmethod
    def vej_dietrich16(v, a, b, c):
        return a * (v.M1 / v.M2) * (1. + c * v.C1) + \
               a * (v.M2 / v.M1) * (1. + c * v.C2) + b

    @staticmethod
    def yeej_ours(v, a, b, c):
        return a * 1e-5 * (v.M1 / v.M2) * (1. + c * 1e5 * v.C1) + \
               a * 1e-5 * (v.M2 / v.M1) * (1. + c * 1e5 * v.C2) + b

    @staticmethod
    def mej_dietrich16(v, a, b, c, d, n):
        return (a * (v.M2 / v.M1) ** (1.0 / 3.0) * (1. - 2 * v.C1) / (v.C1) + b * (v.M2 / v.M1) ** n +
                c * (1 - v.M1 / v.Mb1)) * v.Mb1 + \
               (a * (v.M1 / v.M2) ** (1.0 / 3.0) * (1. - 2 * v.C2) / (v.C2) + b * (v.M1 / v.M2) ** n +
                c * (1 - v.M2 / v.Mb2)) * v.Mb2 + \
               d

    @staticmethod
    def mej_kruger19(v, a, b, c, n):
        return ((a / v.C1) + b * ((v.M2 ** n) / (v.M1 ** n)) + c * v.C1) * v.M1 + \
               ((a / v.C2) + b * ((v.M1 ** n) / (v.M2 ** n)) + c * v.C2) * v.M2

    @staticmethod
    def mdisk_radice18(v, a, b, c, d):
        return np.maximum(a + b * (np.tanh((v["Lambda"] - c) / d)), 1.e-3)

    @staticmethod
    def mdisk_kruger19(v, a, c, d):
        val = 5. * 10 ** (-4)
        # print("lighter? {} then {}".format(v["M2"], v["M1"])); exit(1)
        arr = v["M2"] * np.maximum(val, ((a * v["C2"]) + c) ** d)
        arr[np.isnan(arr)] = val

        return arr

class Fit:

    def __init__(self, dataframe, fit_v_n, err_method="default", clean_nans=True):

        self.fit_v_n = fit_v_n
        self.err_meth = err_method
        print("Initial dataset for ' {} ' contains {} entries".format(self.fit_v_n, len(dataframe)))
        # clear dataset of nans
        used_v_ns = ["M1", "M2", "Mb1", "Mb2", "C1", "C2", "Lambda", "q"] ### This is for my dataset
        # used_v_ns = ["MA", "MB", "MbA", "MbB", "q", "CA", "CB", "tLam", fit_v_n] ### This is for Sebastiano dataset
        dataframe = dataframe[~np.isnan(dataframe[self.fit_v_n])]
        if clean_nans:
            for v_n in used_v_ns:
                dataframe = dataframe[~np.isnan(dataframe[v_n])]

        print("Dataset cleared of nans for ' {} ' contains {} entries".format(self.fit_v_n, len(dataframe)))

        self.ds = dataframe

    @staticmethod
    def get_chi2(y_vals, y_expets, y_errs):
        assert len(y_vals) == len(y_expets)
        z = (y_vals - y_expets) / y_errs
        chi2 = np.sum(z ** 2.)
        return chi2

    @staticmethod
    def get_ch2dof(chi2, n, k):
        """
        :param chi2: chi squared
        :param n: number of elements in a sample
        :param k: n of independedn parameters (1 -- mean, 2 -- poly1 fit, etc)
        :return:
        """
        return chi2 / (n - k)

    @staticmethod
    def get_score(y_true, y_pred):
        """
        uses https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.score
        :param y_true:
        :param y_pred:
        :return: 0-1 value float

        The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
        A constant model that always predicts the expected value of y,
        disregarding the input features, would get a R^2 score of 0.0.

        """

        y_true = np.array(y_true, dtype=np.float)
        y_pred = np.array(y_pred, dtype=np.float)

        y_true = y_true[np.isfinite(y_true)]
        y_pred = y_pred[np.isfinite(y_pred)]

        assert len(y_true) == len(y_pred)
        u = np.sum((y_true - y_pred) ** 2.)
        v = np.sum((y_true - np.mean(y_true)) ** 2.)
        res = (1. - u / v)

        if res > 1.: return np.nan

        return res

    def get_err(self, vals):

        Mej_min = 5.e-5
        vej_def_err = 0.02
        ye_def_err = 0.01
        MdiskPP_min = 5e-4
        theta_ej_min = 2.0

        # vals = self.y_arr

        if self.err_meth == "std":
            res = np.std(vals)

        elif self.err_meth == "2std":
            res = 2. * np.std(vals)

        elif self.err_meth == "default":
            if self.fit_v_n in ["Mej_tot-geo", "Mej1e-2"]:
                lambda_err = lambda Mej: 0.5 * Mej + Mej_min # Msun

            elif self.fit_v_n in ["vel_inf_ave-geo", "Vej"]:
                lambda_err = lambda v: 1. * np.full(len(v), vej_def_err) # c

            elif self.fit_v_n in ["Ye_ave-geo", "Yeej"]:
                lambda_err = lambda v: 1. * np.full(len(v), ye_def_err) # NOne

            elif self.fit_v_n in ["Mdisk3D", "Mdisk1e-2"]:
                lambda_err = lambda MdiskPP: 0.5 * MdiskPP + MdiskPP_min # Msun

            elif self.fit_v_n == "theta_rms-geo":
                lambda_err = lambda v: 1. * np.full(len(v), theta_ej_min) # degrees

            else:
                raise NameError("No error method for v_n: {}".format(self.fit_v_n))

            res = lambda_err(vals)

        elif self.err_meth == "arr":
            raise NameError("Not implemented")

        else:
            raise NameError("no err method: {}".format(self.err_meth))

        # if v_n == "Mej_tot-geo": res = res * 1e3
        return res

    def get_coeffs(self, name):
        if name == "poly22":
            return FittingCoefficients.default_poly22()
        elif name == "poly2":
            return FittingCoefficients.default_poly2()

        elif self.fit_v_n in ["Vej", "vel_inf_ave-geo"]:
            if name == "diet16":
                return FittingCoefficients.vej_diet16_default()
            else:
                raise NameError("Not implemented")

        elif self.fit_v_n in ["Yeej", "Ye_ave-geo"]:
            if name == "our":
                return FittingCoefficients.yeej_our_default()
            else:
                raise NameError("Not implemented")

        elif self.fit_v_n in ["Mej", "Mej_tot-geo"]:
            if name == "diet16":
                return FittingCoefficients.mej_diet16_default()
            elif name == "krug19":
                return FittingCoefficients.mej_krug19_default()
            else:
                raise NameError("Not implememnted")

        elif self.fit_v_n in ["Mdisk", "Mdisk3D"]:
            if name == "rad18":
                return FittingCoefficients.mdisk_rad18_default()
            elif name == "krug19":
                return FittingCoefficients.mdisk_krug19_default()
            else:
                raise NameError("Not implemented")
        else:
            raise NameError("Not implemented")

    def get_fitfunc(self, name):
        if name == "poly22_qLam":
            return FittingFunctions.poly_2_qLambda
        elif name == "poly2_Lam":
            return FittingFunctions.poly_2_Lambda

        elif self.fit_v_n in ["Vej", "vel_inf_ave-geo"]:
            if name == "diet16":
                return FittingFunctions.vej_dietrich16
            else:
                raise NameError("Not implemented")

        elif self.fit_v_n in ["Yeej", "Ye_ave-geo"]:
            if name == "our":
                return FittingFunctions.yeej_ours
            else:
                raise NameError("Not implemented")

        elif self.fit_v_n in ["Mej", "Mej_tot-geo"]:
            if name == "diet16":
                return FittingFunctions.mej_dietrich16
            elif name == "krug19":
                return FittingFunctions.mej_kruger19
            else:
                raise NameError("Not implemented")

        elif self.fit_v_n in ["Mdisk", "Mdisk3D"]:
            if name == "rad18":
                return FittingFunctions.mdisk_radice18
            elif name == "krug19":
                return FittingFunctions.mdisk_kruger19

        else:
            raise NameError("Not implemneted")

    def get_chi2dof_for_mean(self):

        ydata = self.ds[self.fit_v_n]
        yerrs = self.get_err(ydata)

        mean = np.float(np.mean(ydata))
        ypred = np.full(len(ydata), mean)

        chi2 = self.get_chi2(ypred, ydata, yerrs)
        chi2dof = self.get_ch2dof(chi2, len(ydata), 0)

        print(chi2dof)

    def residuals(self, x, data, ffname):

        fitfunc = self.get_fitfunc(ffname)
        xi = fitfunc(data, *x)
        return (xi - data[self.fit_v_n])

    def git_func(self, ff_name="poly22_qLam", cf_name="default"):

        fitfunc = self.get_fitfunc(ff_name)
        init_coeffs = self.get_coeffs(cf_name)

        ydata = self.ds[self.fit_v_n]
        yerrs = self.get_err(ydata)

        # Note: Dietrich and Kruger fitformulas are for 1e3 Msun, so we adapt
        if self.fit_v_n in ["Mej", "Mej_tot-geo"]:
            ydata *= 1e3
            yerrs *= 1e3

        res = least_squares(self.residuals, init_coeffs, args=(self.ds, ff_name))
        ypred = fitfunc(self.ds, *res.x)

        if self.fit_v_n in ["Mej", "Mej_tot-geo"]:
            ydata *= 1e-3
            yerrs *= 1e-3
            ypred *= 1e-3

        chi2 = self.get_chi2(ypred, ydata, yerrs)
        chi2dof = self.get_ch2dof(chi2, len(ydata), len(res.x))

        print(chi2dof)

    def fit_poly(self, ff_name="poly22_tLam", degree=2):

        if ff_name == "poly22_qLam": v_ns_x = ["q", "Lambda"]
        elif ff_name == "poly2_Lam": v_ns_x = ["Lambda"]
        elif ff_name == "poly2_q": v_ns_x = ["q"]
        else:
            raise NameError("Not implemented")

        fitdata = self.ds
        ydata = self.ds[self.fit_v_n]
        yerrs = self.get_err(ydata)

        if self.fit_v_n in ["Mej", "Mej_tot-geo"]:
            ydata *= 1e3
            yerrs *= 1e3

        xdata = []
        for v_n_x in v_ns_x:
            x_ = np.array(fitdata[v_n_x], dtype=float)
            xdata.append(x_)
            # print(len(x_))
        xdata = np.reshape(np.array(xdata), (len(v_ns_x), len(ydata))).T

        transformer = PolynomialFeatures(degree=degree, include_bias=False)
        transformer.fit(xdata)
        x_ = transformer.transform(xdata)
        model = LinearRegression().fit(x_, ydata)
        r_sq = model.score(x_, ydata)
        y_pred = model.predict(x_)

        # print('coefficient of determination R2: {}'.format(r_sq))
        # print('intercept b0: {}'.format(model.intercept_))
        # print('coefficients bi: {}'.format(model.coef_))

        if self.fit_v_n in ["Mej", "Mej_tot-geo"]:
            ydata *= 1e-3
            yerrs *= 1e-3
            y_pred *= 1e-3

        chi2 = self.get_chi2(ydata, y_pred, yerrs)
        chi2dof = self.get_ch2dof(chi2, len(ydata), degree + 1)

        print("{} = f({}) : [{}] : {} : {}".format(self.fit_v_n, v_ns_x, len(ydata), chi2, chi2dof))

    def fit_curve(self, ff_name="poly22_qLam", cf_name="default"):

        fitfunc = self.get_fitfunc(ff_name)
        init_coeffs = self.get_coeffs(cf_name)

        fitdata = self.ds
        ydata = self.ds[self.fit_v_n]
        yerrs = self.get_err(ydata)

        # Note: Dietrich and Kruger fitformulas are for 1e3 Msun, so we adapt
        if self.fit_v_n in ["Mej", "Mej_tot-geo"]:
            ydata *= 1e3
            yerrs *= 1e3

        pred_coeffs, pcov = curve_fit(f=fitfunc, xdata=fitdata, ydata=ydata, p0=init_coeffs, sigma=yerrs, maxfev=5000)
        perr = np.sqrt(np.diag(pcov))
        ypred = fitfunc(fitdata, *pred_coeffs)

        if self.fit_v_n in ["Mej", "Mej_tot-geo"]:
            ydata *= 1e-3
            yerrs *= 1e-3
            ypred *= 1e-3

        res = ypred - ydata # residuals
        chi2 = self.get_chi2(ypred, ydata, yerrs)
        chi2dof = self.get_ch2dof(chi2, len(ydata), len(pred_coeffs))

        print(chi2dof)

        # chi2 = self.chi2dof(data, bfit, len(popt), sigma)
        return pred_coeffs, pcov, perr, res, chi2dof, ypred

if __name__ == "__main__":
    dfname2 = "/home/vsevolod/GIT/bitbucket/bns_gw170817/data/dynej_disc_literature/LiteratureData.csv"
    allmodels = pd.read_csv(dfname2)
    print(allmodels.keys())
    allmodels["Lambda"] = allmodels["tLam"]
    allmodels["M1"] = allmodels["MA"]
    allmodels["M2"] = allmodels["MB"]
    allmodels["C1"] = allmodels["CA"]
    allmodels["C2"] = allmodels["CB"]
    allmodels["Mb1"] = allmodels["MbA"]
    allmodels["Mb2"] = allmodels["MbB"]
    allmodels["vel_inf_ave-geo"] = allmodels["Vej"]
    allmodels["Ye_ave-geo"] = allmodels["Yeej"]
    allmodels["Mdisk3D"] = allmodels["Mdisk1e-2"] * 1.e-2 # Msun
    allmodels["Mej_tot-geo"] = allmodels["Mej1e-2"] * 1.e-2 # Msun

    o_fit = Fit(allmodels, fit_v_n="vel_inf_ave-geo", err_method="default")
    o_fit.fit_curve(ff_name="poly22_qLam", cf_name="poly22")
    o_fit.fit_curve(ff_name="diet16", cf_name="diet16")

    o_fit = Fit(allmodels, fit_v_n="Ye_ave-geo", err_method="default")
    o_fit.fit_curve(ff_name="poly22_qLam", cf_name="poly22")
    o_fit.fit_curve(ff_name="our", cf_name="our")

    o_fit = Fit(allmodels, fit_v_n="Mej_tot-geo", err_method="default", clean_nans=True)
    o_fit.get_chi2dof_for_mean()
    o_fit.fit_curve(ff_name="poly2_Lam", cf_name="poly2")
    o_fit.fit_curve(ff_name="poly22_qLam", cf_name="poly22")
    o_fit.fit_curve(ff_name="diet16", cf_name="diet16")
    o_fit.fit_curve(ff_name="krug19", cf_name="krug19")

    o_fit = Fit(allmodels, fit_v_n="Mdisk3D", err_method="default", clean_nans=True)
    o_fit.get_chi2dof_for_mean()
    o_fit.fit_curve(ff_name="poly2_Lam", cf_name="poly2")
    o_fit.fit_curve(ff_name="poly22_qLam", cf_name="poly22")
    o_fit.fit_poly(ff_name="poly22_qLam")
    o_fit.fit_curve(ff_name="rad18", cf_name="rad18")
    o_fit.fit_curve(ff_name="krug19", cf_name="krug19")
    o_fit.git_func(ff_name="krug19", cf_name="krug19")