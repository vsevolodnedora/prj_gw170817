#!/usr/bin/env python2
from __future__ import division

import pandas as pd
import numpy as np
from collections import OrderedDict

from docutils.utils.math.latex2mathml import mo
from scipy.optimize import curve_fit, least_squares
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class FittingCoefficients:
    """
    Collection of fitting coefficients
    """

    @staticmethod
    def default_poly22():
        return (0.1, 0.01, 0.01, 0.001, 0.001, 0.00001)

    @staticmethod
    def default_poly2():
        return (0.1, 0.01, 0.001)

    @staticmethod
    def default_log_poly2():
        return tuple(2 * 10 ** np.array(FittingCoefficients.default_poly2()))

    @staticmethod
    def default_log_poly22():
        return tuple(2 * 10 ** np.array(FittingCoefficients.default_poly22()))

    @staticmethod
    def vej_diet16_default():
        return (-0.3, 0.5, -3.)

    @staticmethod
    def yeej_our_default():
        return (0.1, 0.5, -8.)

    @staticmethod
    def mej_diet16_default():
        #return(-1.08554642e-02,  6.02898026e+00, -5.92689660e+00, -1.63266494e+01, 7.64071836e-01)
        return(-.6, 4.2, -32., 5., 1.) # fails to converge
        # return(-0.657, 4.254, -32.61,  5.205, -0.773)
        # return (-0.7, 4.0, -132.0, 5.0, -2.)

        # return (1., 1., 1., 1., 1.)

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
    """
    Collection of fitting functions
    """

    @staticmethod
    def poly_2_qLambda(v, b0, b1, b2, b3, b4, b5):
        #b0, b1, b2, b3, b4, b5 = x
        return b0 + b1 * v.q + b2 * v.Lambda + b3 * v.q ** 2 + b4 * v.q * v.Lambda + b5 * v.Lambda ** 2

    @staticmethod
    def poly_2_Lambda(v, b0, b1, b2):
        #b0, b1, b2 = x
        return b0 + b1*v.Lambda + b2*v.Lambda**2

    @staticmethod
    def log_poly_2_Lambda(v, b0, b1, b2):
        #b0, b1, b2 = x
        return np.log10(b0 + b1*v.Lambda + b2*v.Lambda**2)

    @staticmethod
    def log_poly_2_qLambda(v, b0, b1, b2, b3, b4, b5):
        #b0, b1, b2 = x
        return np.log10(b0 + b1 * v.q + b2 * v.Lambda + b3 * v.q ** 2 + b4 * v.q * v.Lambda + b5 * v.Lambda ** 2)

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
        return ((a * (v.M2 / v.M1) ** (1.0 / 3.0) * (1. - 2 * v.C1) / (v.C1) + b * (v.M2 / v.M1) ** n +
                c * (1 - v.M1 / v.Mb1)) * v.Mb1 + \
               (a * (v.M1 / v.M2) ** (1.0 / 3.0) * (1. - 2 * v.C2) / (v.C2) + b * (v.M1 / v.M2) ** n +
                c * (1 - v.M2 / v.Mb2)) * v.Mb2 +  d)

    @staticmethod
    def mej_kruger19(v, a, b, c, n):
        return 1.e3*(((a / v.C1) + b * ((v.M2 ** n) / (v.M1 ** n)) + c * v.C1) * v.M1 + \
               ((a / v.C2) + b * ((v.M1 ** n) / (v.M2 ** n)) + c * v.C2) * v.M2)

    @staticmethod
    def mdisk_radice18(v, a, b, c, d):
        return np.maximum(a + b * (np.tanh((v["Lambda"] - c) / d)), 1.e-3)

    @staticmethod
    def mdisk_kruger19(v, a, c, d):
        val = 5. * 10 ** (-4)

        # print("lighter? {} then {}".format(v["M2"], v["M1"])); exit(1)
        arr = (v["M2"] * np.maximum(val, ((a * v["C2"]) + c) ** d) )
        arr[np.isnan(arr)] = val
        # print(np.array(arr))
        return arr

class Fit_Data:
    """
    Collection of methods to fit data from :

    Parameters
    -------
        dataframe : pandas.dataframe
            contains table of data to fit, to use for fitting, i.e., Mej, Vej, Mdisk, etc

        fit_v_n : str
            name of the variable that is to be fitted with a function i.e., "Mej"

        err_method="default" : str
            method for error estimation for a given data, e.g., 'defatult' assumes
            errors from 'get_err()' for each varaible according to Radice+2018

        clean_nans : bool
            Remove nans from dataframe (rows of data, that is used for
             fitting that contain at least one nan) If 'False' errors might occure
             if data with nans inside is passed into fitting libraries
    """

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

    def get_mean(self):
        return float(np.mean(self.ds[self.fit_v_n]))

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
        https://en.wikipedia.org/wiki/Coefficient_of_determination#:~:text=In%20statistics%2C%20the%20coefficient%20of,the%20independent%20variable(s).&text=In%20both%20such%20cases%2C%20the,ranges%20from%200%20to%201.

        The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
        A constant model that always predicts the expected value of y,
        disregarding the input features, would get a R^2 score of 0.0.

        """

        y_true = np.array(y_true, dtype=np.float)
        y_pred = np.array(y_pred, dtype=np.float)

        y_true = y_true[np.isfinite(y_true)]
        y_pred = y_pred[np.isfinite(y_pred)]

        assert len(y_true) == len(y_pred)
        ss_res = np.sum((y_true - y_pred) ** 2.)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2.)
        res = (1. - ss_res / ss_tot)

        if res > 1.: return np.nan

        return res

    def get_chi2dof_for_mean(self):

        ydata = self.ds[self.fit_v_n]
        yerrs = self.get_err(ydata)

        mean = np.float(np.mean(ydata))
        ypred = np.full(len(ydata), mean)

        chi2 = self.get_chi2(ypred, ydata, yerrs)
        chi2dof = self.get_ch2dof(chi2, len(ydata), 0)

        # print(chi2dof)

        return chi2, chi2dof

    def get_stats(self, v_ns=("n", "mean", "std", "80", "90", "95", "chi2", "chi2dof")):
        v_ns = list(v_ns)
        res = []
        for v_n in v_ns:
            if v_n == "n":
                res.append(len(self.ds[self.fit_v_n]))
            if v_n == "mean":
                res.append(np.mean(self.ds[self.fit_v_n]))
            if v_n == "std":
                res.append(np.std(self.ds[self.fit_v_n]))
            if v_n == "90" or v_n == "80" or v_n == "95":
                dic = self.ds[self.fit_v_n].describe(percentiles=[0.8, 0.9, 0.95])
                val = float(dic[v_n+"%"])
                res.append(val)
            if v_n == "chi2":
                chi2, chi2dof = self.get_chi2dof_for_mean()
                res.append(chi2)
            if v_n == "chi2dof":
                chi2, chi2dof = self.get_chi2dof_for_mean()
                res.append(chi2dof)

        if len(res) == 1: return float(res[0])
        else: return res

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
        elif name == "log_poly2":
            return FittingCoefficients.default_log_poly2()
        elif name == "log_poly22":
            return FittingCoefficients.default_log_poly22()

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
        if name == "poly22_qLambda":
            return FittingFunctions.poly_2_qLambda
        elif name == "poly2_Lambda":
            return FittingFunctions.poly_2_Lambda
        elif name == "poly2_q":
            return FittingFunctions.poly_2_q
        elif name == "log_poly2_Lambda":
            return FittingFunctions.log_poly_2_Lambda
        elif name == "log_poly22_qLambda":
            return FittingFunctions.log_poly_2_qLambda

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
                raise NameError("Not implemented: {}".format(name))

        elif self.fit_v_n in ["Mdisk", "Mdisk3D"]:
            if name == "rad18":
                return FittingFunctions.mdisk_radice18
            elif name == "krug19":
                return FittingFunctions.mdisk_kruger19

        else:
            raise NameError("Not implemneted")

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

    def fit_curve(self, ff_name="poly22_qLam", cf_name="default", modify=None, usesigma=True):

        fitfunc = self.get_fitfunc(ff_name)
        init_coeffs = self.get_coeffs(cf_name)

        fitdata = self.ds
        ydata = np.array(self.ds[self.fit_v_n],dtype=np.float64)
        yerrs = self.get_err(ydata)

        if modify == "log10":
            ydata = np.log10(ydata)
            yerrs = np.log10(yerrs)
        elif modify == "log2":
            ydata = np.log2(ydata)
            yerrs = np.log2(yerrs)
        elif modify == "10**":
            ydata = 10**(ydata)
            yerrs = 10**(yerrs)
            # init_coeffs = tuple(100*(np.array(init_coeffs)))

        # Note: Dietrich and Kruger fitformulas are for 1e3 Msun, so we adapt
        # if self.fit_v_n in ["Mej", "Mej_tot-geo"]:
        #     ydata *= 1e3
        #     yerrs *= 1e3

        if usesigma: sigma = yerrs
        else: sigma = None
        # pred_coeffs, pcov = curve_fit(f=fitfunc, xdata=fitdata, ydata=ydata, p0=init_coeffs, sigma=yerrs, maxfev=20000)
        pred_coeffs, pcov = curve_fit(f=fitfunc, xdata=fitdata, ydata=ydata, p0=init_coeffs, sigma=sigma, maxfev=20000)

        perr = np.sqrt(np.diag(pcov))
        ypred = fitfunc(fitdata, *pred_coeffs)

        # if self.fit_v_n in ["Mej", "Mej_tot-geo"]:
        #     ydata *= 1e-3
        #     yerrs *= 1e-3
        #     ypred *= 1e-3
        if  modify == "log10":
            ydata = 10**(ydata)
            yerrs = 10**(yerrs)
            ypred = 10**(ypred)
        elif modify == "log2":
            ydata = np.exp(ydata)
            yerrs = np.exp(yerrs)
            ypred = np.exp(ypred)
        elif modify == "10**":
            ydata = np.log10(ydata)
            yerrs = np.log10(yerrs)
            ypred = np.log10(ypred)

        res = np.array((ydata - ypred)/ydata) # residuals
        chi2 = self.get_chi2(ypred, ydata, yerrs)
        chi2dof = self.get_ch2dof(chi2, len(ydata), len(pred_coeffs))

        # if modify == "log10":
        #     chi2dof = np.log10(chi2dof)

        # print(chi2dof)
        # print(res);
        # exit(1)

        rs = self.get_score(ydata, ypred)
        # print(chi2dof)

        # chi2 = self.chi2dof(data, bfit, len(popt), sigma)
        # return pred_coeffs, pcov, perr, res, chi2dof, ypred
        return pred_coeffs, chi2, chi2dof, rs

class Fit_Tex_Tables:

    def __init__(self, dataframe, fit_v_n, err_method="default", clean_nans=True, deliminator='&'):

        # print(dataframe[np.isnan(dataframe[fit_v_n])])
        # exit(1)

        self.df = dataframe[~np.isnan(dataframe[fit_v_n])] # clean datasets for which there is no data
        self.v_n = fit_v_n
        self.err_m = err_method
        self.clean = clean_nans
        self.dlm = deliminator
        # my dataset
        self.dsets_order = [
            # leakage + M0/M1
            "Reference set",
            "Vincent:2019kor",
            "Radice:2018pdn(M0)",
            "Sekiguchi:2016bjd",
            "Sekiguchi:2015dma",
            # leakage
            "Radice:2018pdn(LK)",
            "Lehner:2016lxy",
            # None
            "Kiuchi:2019lls",
            "Dietrich:2016lyp",
            "Dietrich:2015iva",
            "Hotokezaka:2012ze",
            "Bauswein:2013yna",
        ]
        # sebastiano Dataset
        # self.dsets_order = [
        #     # Leakage + M0 / M1
        #     "Nedora:2020",
        #     "Bernuzzi:2020txg",
        #     "Vincent:2019kor",
        #     "Sekiguchi:2016bjd",
        #     "Sekiguchi:2015dma",
        #     # Leakage
        #     "Radice:2018pdn",
        #     "Lehner:2016lxy",
        #     # None
        #     "Kiuchi:2019lls",
        #     "Dietrich:2016hky",
        #     "Dietrich:2015iva",
        #     "Hotokezaka:2012ze",
        #     "Bauswein:2013yna"
        # ]
        #

    @staticmethod
    def __get_str_val(val, fmt, fancy=False):
        if fmt != None and fmt != "":
            _val = str(("%{}".format(fmt) % float(val)))
        else:
            if str(val) == "nan":
                _val = " "
                # exit(1)
            else:
                _val = val

        if fancy:
            if _val.__contains__("e-") or _val.__contains__("e+"):
                # power = str(_val).split("e")[-1]
                # power = str(power[1:]) # remove +/-
                # if power[0] == "0": power = str(power[1:])
                _val = "$" + str(_val).replace("e", r'\times10^{') + "}$"

        # if fancy:
        #     if _val.__contains__("e-"):
        #         _val = "$"+str(_val).replace("e-", r'\times10^{-')+"}$"
        #     elif _val.__contains__("e+"):
        #         _val = "$" + str(_val).replace("e+", r'\times10^{') + "}$"
        #     else:
        #         pass

        return _val

    def get_dataframe_subset(self, key, vals):
        """
            select a sub-dataframe with given values for a certain key
        """
        mask = np.zeros(len(self.df["q"]), dtype=bool)
        for val in vals:
            mask = mask | (self.df[key] == val)
        sel_df = self.df[mask]
        assert len(sel_df) > 0
        return sel_df

    def print_stats(self, v_ns=("n", "mean", "std", "80", "90", "95", "chi2", "chi2dof")):
        v_ns = list(v_ns)
        sel_dsets = []
        row_labels, vals = [], []
        for i in range(len(self.dsets_order)):
            if self.dsets_order[i] in list(self.df["bibkey"]):
                sel_dsets.append(self.dsets_order[i])
                dataframe = self.get_dataframe_subset("bibkey", sel_dsets)
                print("\t Adding {} : {}".format(sel_dsets[-1],len(dataframe["bibkey"])))
                #
                df = Fit_Data(dataframe, self.v_n, err_method=self.err_m, clean_nans=self.clean)
                i_vals = df.get_stats(v_ns)
                #
                row_labels.append(sel_dsets[-1])
                vals.append(i_vals)
                #
            else:
                print("\t Neglecting {} ".format(sel_dsets[-1]))

        print("Data is collected")

        ''' --- --- --- printing a table --- --- --- '''
        pre_names = ["datasets"]
        v_labels = ["Datasets", r"$N$", r"$\mu$", r"$\sigma$", r"$80\%$", r"$90\%$", r"$95\%$", "$\chi^2$",
                    r"$\chi^2 _{\text{dof}}$"]
        coeff_fmt = ".3f"
        coeff_small_fmt = ".3e"

        cells = "c" * (len(pre_names) + len(v_ns))
        print("\n")
        print(r"\begin{table*}")
        print(r"\caption{I am your little table}")
        print(r"\begin{tabular}{l|" + cells + "}")

        line = ''
        for name, label in zip(pre_names + v_ns, v_labels):
            if name != v_ns[-1]: line = line + label + ' & '
            else: line = line + label + r' \\'
        # line[-2] = r"\\"
        print(line)

        for row_name, coeff in zip(row_labels, vals):

            # row_names = row_labels[i]
            #row_name = row_names[-1]

            if row_name == row_labels[0]:
                pass
            else:
                row_name = "\& " + "\cite{" + row_name + "} "

            row = row_name + " & "
            for i_coeff in coeff:
                if i_coeff < 1.e-2:  ifmt = coeff_small_fmt
                else: ifmt = coeff_fmt
                val = str(("%{}".format(ifmt) % float(i_coeff)))
                if i_coeff != coeff[-1]:
                    row = row + val + " & "
                else:
                    row = row + val + r" \\ "
            print(row)
            # row[-2] = r" \\ "
        print(r"\end{tabular}")
        print(r"\end{table*}")

    def print_polyfit_table(self, ff_name="poly22_qLambda", cf_name="default", fancy=False, modify=None, usesigma=True):

        sel_dsets = []
        row_labels = []
        all_coeffs = []
        all_pars = []
        for i in range(len(self.dsets_order)):
            if self.dsets_order[i] in list(self.df["bibkey"]):
                sel_dsets.append(self.dsets_order[i])
                dataframe = self.get_dataframe_subset("bibkey", sel_dsets)
                print("\t Adding {} : {}".format(sel_dsets[-1],len(dataframe["bibkey"])))
                #
                df = Fit_Data(dataframe, self.v_n, err_method=self.err_m, clean_nans=self.clean)
                i_coeffs, i_chi, i_chi2dof, i_rs = df.fit_curve(ff_name=ff_name, cf_name=cf_name, modify=modify, usesigma=usesigma)
                #
                row_labels.append(sel_dsets[-1])
                all_coeffs.append(i_coeffs)
                all_pars.append([i_chi2dof, i_rs])  # i_chi
                #
            else:
                print("\t Neglecting {} ".format(sel_dsets[-1]))
        #
        print("data is collected")

        ''' --- --- --- table --- --- --- '''

        dataset_label = "Datasets"
        coefs_labels = [r"$b_0$", r"$b_1$", r"$b_2$", r"$b_3$", r"$b_4$", r"$b_5$"]
        coefs_fmt = [".2e", ".2e", ".2e", ".2e", ".2e", ".2e"]
        other_labels = [r"$\chi^2_{\nu}$", r"$R^2$"]
        other_fmt = [".1f", ".3f"]

        label_line = dataset_label + ' '
        for name in coefs_labels[:len(all_coeffs[0])] + other_labels:
            if name != other_labels[-1]:
                label_line = label_line + name + ' & '
            else:
                label_line = label_line + name + r' \\'

        lines = []

        for i in range(len(row_labels)):
            # fiest element -- name of the dataset
            row_name = row_labels[i]
            #row_name = row_names[-1]

            if i == 0:
                pass
                #row_name = row_names[-1]
            else:
                row_name = "\& " + "\cite{" + row_name + "} "

            row = row_name + " & "

            # add coefficients
            i_coeffs = all_coeffs[i]
            for coeff, fmt in zip(i_coeffs, coefs_fmt[:len(i_coeffs)]):
                val = self.__get_str_val(coeff, fmt, fancy)
                if coeff != i_coeffs[-1]:
                    row = row + val + " {} ".format(self.dlm)
                else:
                    if len(other_labels) == 0:
                        row = row + val + r" \\ "
                    else:
                        row = row + val + " {} ".format(self.dlm)

            # add other values
            i_pars = all_pars[i]
            assert len(i_pars) == len(other_fmt)
            for par, fmt in zip(i_pars, other_fmt):
                val = self.__get_str_val(par, fmt, fancy)
                if par == i_pars[-1]:
                    row = row + val + r" \\ "
                else:
                    row = row + val + r" {} ".format(self.dlm)

            # done
            lines.append(row)

        ''' --- printing --- '''

        cells = "c" * (1 + len(coefs_labels) + len(other_labels))
        #
        print("\n")
        print(r"\begin{table}")
        print(r"\caption{I am your little table}")
        print(r"\begin{tabular}{l|" + cells + "}")

        print(label_line)

        for line in lines:
            print(line)

        print(r"\end{tabular}")
        print(r"\end{table}")

    def print_fitfunc_table(self,  ff_name="diet16", cf_name="default", fancy=False, modify=None, usesigma=True):

        sel_dsets = []
        row_labels = []
        all_coeffs = []
        all_pars = []
        for i in range(len(self.dsets_order)):
            if self.dsets_order[i] in list(self.df["bibkey"]):
                sel_dsets.append(self.dsets_order[i])
                dataframe = self.get_dataframe_subset("bibkey", sel_dsets)
                print("\t Adding {} : {}".format(sel_dsets[-1],len(dataframe["bibkey"])))
                #
                df = Fit_Data(dataframe, self.v_n, err_method=self.err_m, clean_nans=self.clean)
                i_coeffs, i_chi, i_chi2dof, i_rs = df.fit_curve(ff_name=ff_name, cf_name=cf_name, modify=modify, usesigma=usesigma)
                #
                row_labels.append(sel_dsets[-1])
                all_coeffs.append(i_coeffs)
                all_pars.append([i_chi2dof, i_rs])  # i_chi
                #
            else:
                print("\t Neglecting {} ".format(sel_dsets[-1]))
        #
        print("data is collected")

        ''' --- table --- '''
        dataset_label = "Datasets"
        coefs_labels = [r"$\alpha$", r"$\beta$", r"$\delta$", r"$\gamma$", r"$\eta$", r"$\phi$"]
        # coefs_fmt = [".2e", ".2e", ".2e", ".2e", ".2e", ".2e"]
        coefs_fmt = [".3e", ".3e", ".3e", ".3e", ".3e", ".3e"]
        other_labels = [r"$\chi^2_{\nu}$", r"$R^2$"]
        other_fmt = [".1f", ".3f"]

        # label line
        label_line = dataset_label + ' '
        for name in coefs_labels[:len(all_coeffs[0])] + other_labels:
            if name == other_labels[-1]:
                label_line = label_line + name + r' \\ '
            else:
                label_line = label_line + name + r' {}'.format(self.dlm)

        lines = []
        # collect table lines
        for i in range(len(row_labels)):
            # fiest element -- name of the dataset
            row_name = row_labels[i]
            #row_name = row_names[-1]

            if i == 0:
                pass
                #row_name = row_names[-1]
            else:
                row_name = "\& " + "\cite{" + row_name + "} "

            row = row_name + " {} ".format(self.dlm)

            # add coefficients
            i_coeffs = all_coeffs[i]
            for coeff, fmt in zip(i_coeffs, coefs_fmt[:len(i_coeffs)]):
                val = self.__get_str_val(coeff, fmt, fancy)
                if coeff != i_coeffs[-1]:
                    row = row + val + " {} ".format(self.dlm)
                else:
                    if len(other_labels) == 0:
                        row = row + val + r" \\ "
                    else:
                        row = row + val + " {} ".format(self.dlm)

            # add other values
            i_pars = all_pars[i]
            assert len(i_pars) == len(other_fmt)
            for par, fmt in zip(i_pars, other_fmt):
                val = self.__get_str_val(par, fmt, fancy)
                if par == i_pars[-1]:
                    row = row + val + r" \\ "
                else:
                    row = row + val + r" {} ".format(self.dlm)

            # done
            lines.append(row)

        ''' --- printing --- '''

        cells = "c" * (1 + len(coefs_labels) + len(other_labels))
        #
        print("\n")
        print(r"\begin{table}")
        print(r"\caption{I am your little table}")
        print(r"\begin{tabular}{l|" + cells + "}")

        print(label_line)

        for line in lines:
            print(line)

        print(r"\end{tabular}")
        print(r"\end{table}")

    # def print_mej_chi2dofs(self):
    #
    #     v_ns = ["datasets", "mean-chi2dof", "diet16-chi2dof", "krug19-chi2dof", "poly2-chi2dof", "poly22-chi2dof"]
    #
    #     row_labels, all_vals = [], []
    #     sel_dsets = []
    #     for i in range(len(self.dsets_order)):
    #         if self.dsets_order[i] in list(self.df["bibkey"]):
    #             sel_dsets.append(self.dsets_order[i])
    #             dataframe = self.get_dataframe_subset("bibkey", sel_dsets)
    #             print("\t Adding {} : {}".format(sel_dsets[-1],len(dataframe["bibkey"])))
    #             #
    #             df = Fit_Data(dataframe, self.v_n, err_method=self.err_m, clean_nans=self.clean)
    #             row_labels.append(sel_dsets[-1])
    #             # i_coeffs, i_chi, i_chi2dof, i_rs = df.fit_curve(ff_name=ff_name, cf_name=cf_name)
    #             #
    #             vals = []
    #             for v_n in v_ns:
    #                 if v_n.__contains__("mean-"):
    #                     if v_n.__contains__("chi2dof"):
    #                         _, chi2dof = df.get_chi2dof_for_mean()
    #                         vals.append(chi2dof)
    #
    #                 if v_n.__contains__("diet16-"):
    #                     print("\tTask: {}".format(v_n))
    #                     i_coeffs, chi2, chi2dof, R2 = df.fit_curve(ff_name="diet16", cf_name="diet16")
    #                     # print(chi2dof); exit(1)
    #                     if v_n.__contains__("chi2dof"): vals.append(chi2dof)
    #
    #                 if v_n.__contains__("krug19-"):
    #                     print("\tTask: {}".format(v_n))
    #                     i_coeffs, chi2, chi2dof, R2 = df.fit_curve(ff_name="krug19", cf_name="krug19")
    #                     if v_n.__contains__("chi2dof"): vals.append(chi2dof)
    #
    #                 if v_n.__contains__("poly2-"):
    #                     print("\tTask: {}".format(v_n))
    #                     coeffs, chi2, chi2dof, r2 = df.fit_curve(ff_name="poly2_Lambda", cf_name="poly2")
    #                     if v_n.__contains__("chi2dof"): vals.append(chi2dof)
    #
    #                 if v_n.__contains__("poly22-"):
    #                     print("\tTask: {}".format(v_n))
    #                     coeffs, chi2, chi2dof, r2 = df.fit_curve(ff_name="poly22_qLambda", cf_name="poly22")
    #                     if v_n.__contains__("chi2dof"): vals.append(chi2dof)
    #             all_vals.append(vals)
    #             #
    #         else:
    #             print("\t Neglecting {} ".format(sel_dsets[-1]))
    #
    #     print("\t---<DataCollected>---")
    #
    #     ''' --- --- --- table --- --- --- '''
    #
    #     fmts = [".1f", ".2f", ".2f", ".2f", ".2f"]
    #     v_ns_labels = ["datasets", r"Mean", r"Eq.~\eqref{eq:fit_Mej}", r"Eq.~\eqref{eq:fit_Mej_Kruger}",
    #                    r"$P_2(\tilde{\Lambda})$", r"$P_2(q,\tilde{\Lambda})$"]
    #
    #     cells = "c" * len(v_ns_labels)
    #     print("\n")
    #     print(r"\begin{table*}")
    #     print(r"\caption{I am your little table}")
    #     print(r"\begin{tabular}{l|" + cells + "}")
    #     line = ''
    #     # HEADER
    #     for name, label in zip(v_ns, v_ns_labels):
    #         if name != v_ns[-1]:
    #             line = line + label + ' {} '.format(self.dlm)
    #         else:
    #             line = line + label + r' \\'
    #     print(line)
    #     # TABLE
    #
    #     #
    #     for i in range(len(row_labels)):
    #         # DATA SET NAME
    #         row_name = row_labels[i]
    #
    #         #row_name = row_names[-1]
    #
    #         if row_name == row_labels[0]:
    #             # row_name = row_names[-1]
    #             pass
    #         else:
    #             row_name = "\& " + "\cite{" + row_name + "} "
    #
    #         # DATA ITSELF
    #         vals = all_vals[i]
    #         row = row_name + " {} ".format(self.dlm)
    #         assert len(vals) == len(fmts)
    #         for val, fmt in zip(vals, fmts):
    #             if val != vals[-1]:
    #                 val = self.__get_str_val(val, fmt)
    #                 # if val < 1e-2:  val = str(("%{}".format(coeff_small_fmt) % float(val)))
    #                 # else: val = __get_str_val()#str(("%{}".format(coeff_fmt) % float(val)))
    #                 row = row + val + " {} ".format(self.dlm)
    #             else:
    #                 val = self.__get_str_val(val, fmt)
    #                 # if val < 1e-2:  val = str(("%{}".format(coeff_small_fmt) % float(val)))
    #                 # else: val = str(("%{}".format(coeff_fmt) % float(val)))
    #                 row = row + val + r" \\ "
    #
    #         print(row)
    #         # row[-2] = r" \\ "
    #
    #     print(r"\end{tabular}")
    #     print(r"\end{table}")

    def print_chi2dofs(self, v_ns, v_ns_labels, fmts, modify=None, usesigma=True):

        #v_ns = ["datasets", "mean-chi2dof", "diet16-chi2dof", "poly2-chi2dof", "poly22-chi2dof"]

        row_labels, all_vals = [], []
        sel_dsets = []
        for i in range(len(self.dsets_order)):
            if self.dsets_order[i] in list(self.df["bibkey"]):
                sel_dsets.append(self.dsets_order[i])
                dataframe = self.get_dataframe_subset("bibkey", sel_dsets)
                print("\t Adding {} : {}".format(sel_dsets[-1], len(dataframe["bibkey"])))
                #
                df = Fit_Data(dataframe, self.v_n, err_method=self.err_m, clean_nans=self.clean)
                row_labels.append(sel_dsets[-1])
                # i_coeffs, i_chi, i_chi2dof, i_rs = df.fit_curve(ff_name=ff_name, cf_name=cf_name)
                #
                vals = []
                for v_n in v_ns:
                    if v_n.__contains__("mean-"):
                        if v_n.__contains__("chi2dof"):
                            _, chi2dof = df.get_chi2dof_for_mean()
                            vals.append(chi2dof)

                    if v_n.__contains__("our-"):
                        print("\tTask: {}".format(v_n))
                        i_coeffs, chi2, chi2dof, R2 = df.fit_curve(ff_name="our", cf_name="our", modify=modify, usesigma=usesigma)
                        # print(chi2dof); exit(1)
                        if v_n.__contains__("chi2dof"): vals.append(chi2dof)

                    if v_n.__contains__("diet16-"):
                        print("\tTask: {}".format(v_n))
                        i_coeffs, chi2, chi2dof, R2 = df.fit_curve(ff_name="diet16", cf_name="diet16", modify=modify, usesigma=usesigma)
                        # print(chi2dof); exit(1)
                        if v_n.__contains__("chi2dof"): vals.append(chi2dof)

                    if v_n.__contains__("rad18-"):
                        print("\tTask: {}".format(v_n))
                        i_coeffs, chi2, chi2dof, R2 = df.fit_curve(ff_name="rad18", cf_name="rad18", modify=modify, usesigma=usesigma)
                        # print(chi2dof); exit(1)
                        if v_n.__contains__("chi2dof"): vals.append(chi2dof)

                    if v_n.__contains__("krug19-"):
                        print("\tTask: {}".format(v_n))
                        i_coeffs, chi2, chi2dof, R2 = df.fit_curve(ff_name="krug19", cf_name="krug19", modify=modify, usesigma=usesigma)
                        if v_n.__contains__("chi2dof"): vals.append(chi2dof)

                    if v_n.__contains__("poly2-"):
                        print("\tTask: {}".format(v_n))
                        coeffs, chi2, chi2dof, r2 = df.fit_curve(ff_name="poly2_Lambda", cf_name="poly2", modify=modify, usesigma=usesigma)
                        if v_n.__contains__("chi2dof"): vals.append(chi2dof)

                    if v_n.__contains__("poly22-"):
                        print("\tTask: {}".format(v_n))
                        coeffs, chi2, chi2dof, r2 = df.fit_curve(ff_name="poly22_qLambda", cf_name="poly22", modify=modify, usesigma=usesigma)
                        if v_n.__contains__("chi2dof"): vals.append(chi2dof)
                all_vals.append(vals)
                #
            else:
                print("\t Neglecting {} ".format(sel_dsets[-1]))

        print("\t---<DataCollected>---")

        ''' --- --- --- table --- --- --- '''

        #fmts = [ ".2f", ".2f", ".2f", ".2f"]
        #v_ns_labels = ["datasets", r"Mean", r"Eq.~\eqref{eq:fit_vej}", r"$P_2(\tilde{\Lambda})$",
        #               r"$P_2(q,\tilde{\Lambda})$"]

        cells = "c" * len(v_ns_labels)
        print("\n")
        print(r"\begin{table*}")
        print(r"\caption{I am your little table}")
        print(r"\begin{tabular}{l|" + cells + "}")
        line = ''
        # HEADER
        for name, label in zip(v_ns, v_ns_labels):
            if name != v_ns[-1]:
                line = line + label + ' & '
            else:
                line = line + label + r' \\'
        print(line)
        # TABLE

        #
        for i in range(len(row_labels)):
            # DATA SET NAME
            row_name = row_labels[i]

            # row_name = row_names[-1]

            if row_name == row_labels[0]:
                # row_name = row_names[-1]
                pass
            else:
                row_name = "\& " + "\cite{" + row_name + "} "

            # DATA ITSELF
            vals = all_vals[i]
            row = row_name + " {} ".format(self.dlm)
            assert len(vals) == len(fmts)
            for val, fmt in zip(vals, fmts):
                if val != vals[-1]:
                    val = self.__get_str_val(val, fmt)
                    # if val < 1e-2:  val = str(("%{}".format(coeff_small_fmt) % float(val)))
                    # else: val = __get_str_val()#str(("%{}".format(coeff_fmt) % float(val)))
                    row = row + val + " {} ".format(self.dlm)
                else:
                    val = self.__get_str_val(val, fmt)
                    # if val < 1e-2:  val = str(("%{}".format(coeff_small_fmt) % float(val)))
                    # else: val = str(("%{}".format(coeff_fmt) % float(val)))
                    row = row + val + r" \\ "

            print(row)
            # row[-2] = r" \\ "

        print(r"\end{tabular}")
        print(r"\end{table}")

class BestFits:
    def __init__(self, dataframe, err_method="default", clean_nans=True):

        self.ds = dataframe
        self.errm = err_method
        self.clean = clean_nans
        # ejecta mass

        self.mass_ds = self.compute_ejecta_mass_fits(modify="log10", usesigma=True).sort_values(by="chi2dof")
        self.vel_ds = self.compute_ejecta_vel_fits(modify=None, usesigma=True).sort_values(by="chi2dof")
        self.ye_ds = self.compute_ejecta_ye_fits(modify=None, usesigma=True).sort_values(by="chi2dof")
        self.theta_ds = self.compute_ejecta_theta_fits(modify=None, usesigma=True).sort_values(by="chi2dof")
        self.diskmass_ds = self.compute_ejecta_diskmass_fits(modify=None, usesigma=False).sort_values(by="chi2dof")

    def compute_ejecta_mass_fits(self, modify=None, usesigma=True):

        names = []
        vals_coeffs = []
        chi2dods = []

        o_fit = Fit_Data(self.ds, fit_v_n="Mej_tot-geo", err_method=self.errm, clean_nans=self.clean)

        # methods::mean
        val_coff, chi2dof = o_fit.get_stats(v_ns=("mean", "chi2dof"))
        names.append("mean")
        vals_coeffs.append(val_coff)
        chi2dods.append(chi2dof)

        # methods::poly2_lam
        val_coff, _, chi2dof, _ = o_fit.fit_curve(ff_name="poly2_Lambda", cf_name="poly2", modify=modify, usesigma=usesigma)
        names.append("poly2_Lambda")
        vals_coeffs.append(val_coff)
        chi2dods.append(chi2dof)

        # methods::poly2_qlam
        val_coff, _, chi2dof, _ = o_fit.fit_curve(ff_name="poly22_qLambda", cf_name="poly22", modify=modify, usesigma=usesigma)
        names.append("poly22_qLambda")
        vals_coeffs.append(val_coff)
        chi2dods.append(chi2dof)

        # methods::Kruger+2019
        val_coff, _, chi2dof, _ = o_fit.fit_curve(ff_name="krug19", cf_name="krug19", modify=modify, usesigma=usesigma)
        names.append("krug19")
        vals_coeffs.append(val_coff)
        chi2dods.append(chi2dof)

        # methods::Kruger+2019
        val_coff, _, chi2dof, _ = o_fit.fit_curve(ff_name="diet16", cf_name="diet16", modify=modify, usesigma=usesigma)
        names.append("diet16")
        vals_coeffs.append(val_coff)
        chi2dods.append(chi2dof)

        dic = {"fitmodel":names, "vals_coeffs":vals_coeffs, "chi2dof":chi2dods}
        df = pd.DataFrame(dic)
        df = df.set_index("fitmodel")
        df["modify"] = modify
        df["usesigma"] = usesigma
        print(df)
        return df

    def compute_ejecta_vel_fits(self, modify=None, usesigma=True):

        names = []
        vals_coeffs = []
        chi2dods = []

        o_fit = Fit_Data(self.ds, fit_v_n="vel_inf_ave-geo", err_method=self.errm, clean_nans=self.clean)

        # methods::mean
        val_coff, chi2dof = o_fit.get_stats(v_ns=("mean", "chi2dof"))
        names.append("mean")
        vals_coeffs.append(val_coff)
        chi2dods.append(chi2dof)

        # methods::poly2_lam
        val_coff, _, chi2dof, _ = o_fit.fit_curve(ff_name="poly2_Lambda", cf_name="poly2", modify=modify, usesigma=usesigma)
        names.append("poly2_Lambda")
        vals_coeffs.append(val_coff)
        chi2dods.append(chi2dof)

        # methods::poly2_qlam
        val_coff, _, chi2dof, _ = o_fit.fit_curve(ff_name="poly22_qLambda", cf_name="poly22", modify=modify, usesigma=usesigma)
        names.append("poly22_qLambda")
        vals_coeffs.append(val_coff)
        chi2dods.append(chi2dof)

        # methods::Dietrich+16
        val_coff, _, chi2dof, _ = o_fit.fit_curve(ff_name="diet16", cf_name="diet16", modify=modify, usesigma=usesigma)
        names.append("diet16")
        vals_coeffs.append(val_coff)
        chi2dods.append(chi2dof)

        dic = {"fitmodel":names, "vals_coeffs":vals_coeffs, "chi2dof":chi2dods}
        df = pd.DataFrame(dic)
        df = df.set_index("fitmodel")
        df["modify"] = modify
        df["usesigma"] = usesigma
        print(df)
        return df

    def compute_ejecta_ye_fits(self, modify=None, usesigma=True):

        names = []
        vals_coeffs = []
        chi2dods = []

        o_fit = Fit_Data(self.ds, fit_v_n="Ye_ave-geo", err_method=self.errm, clean_nans=self.clean)

        # methods::mean
        val_coff, chi2dof = o_fit.get_stats(v_ns=("mean", "chi2dof"))
        names.append("mean")
        vals_coeffs.append(val_coff)
        chi2dods.append(chi2dof)

        # methods::poly2_lam
        val_coff, _, chi2dof, _ = o_fit.fit_curve(ff_name="poly2_Lambda", cf_name="poly2", modify=modify, usesigma=usesigma)
        names.append("poly2_Lambda")
        vals_coeffs.append(val_coff)
        chi2dods.append(chi2dof)

        # methods::poly2_qlam
        val_coff, _, chi2dof, _ = o_fit.fit_curve(ff_name="poly22_qLambda", cf_name="poly22", modify=modify, usesigma=usesigma)
        names.append("poly22_qLambda")
        vals_coeffs.append(val_coff)
        chi2dods.append(chi2dof)

        # methods::our
        val_coff, _, chi2dof, _ = o_fit.fit_curve(ff_name="our", cf_name="our", modify=modify, usesigma=usesigma)
        names.append("our")
        vals_coeffs.append(val_coff)
        chi2dods.append(chi2dof)

        dic = {"fitmodel":names, "vals_coeffs":vals_coeffs, "chi2dof":chi2dods}
        df = pd.DataFrame(dic)
        df = df.set_index("fitmodel")
        df["modify"] = modify
        df["usesigma"] = usesigma
        print(df)
        return df

    def compute_ejecta_theta_fits(self, modify=None, usesigma=True):

        names = []
        vals_coeffs = []
        chi2dods = []

        o_fit = Fit_Data(self.ds, fit_v_n="theta_rms-geo", err_method=self.errm, clean_nans=self.clean)

        # methods::mean
        val_coff, chi2dof = o_fit.get_stats(v_ns=("mean", "chi2dof"))
        names.append("mean")
        vals_coeffs.append(val_coff)
        chi2dods.append(chi2dof)

        # methods::poly2_lam
        val_coff, _, chi2dof, _ = o_fit.fit_curve(ff_name="poly2_Lambda", cf_name="poly2", modify=modify, usesigma=usesigma)
        names.append("poly2_Lambda")
        vals_coeffs.append(val_coff)
        chi2dods.append(chi2dof)

        # methods::poly2_qlam
        val_coff, _, chi2dof, _ = o_fit.fit_curve(ff_name="poly22_qLambda", cf_name="poly22", modify=modify, usesigma=usesigma)
        names.append("poly22_qLambda")
        vals_coeffs.append(val_coff)
        chi2dods.append(chi2dof)

        dic = {"fitmodel":names, "vals_coeffs":vals_coeffs, "chi2dof":chi2dods}
        df = pd.DataFrame(dic)
        df = df.set_index("fitmodel")
        df["modify"] = modify
        df["usesigma"] = usesigma
        print(df)
        return df

    def compute_ejecta_diskmass_fits(self, modify=None, usesigma=True):

        names = []
        vals_coeffs = []
        chi2dods = []

        o_fit = Fit_Data(self.ds, fit_v_n="Mdisk3D", err_method=self.errm, clean_nans=self.clean)

        # methods::mean
        val_coff, chi2dof = o_fit.get_stats(v_ns=("mean", "chi2dof"))
        names.append("mean")
        vals_coeffs.append(val_coff)
        chi2dods.append(chi2dof)

        # methods::poly2_lam
        val_coff, _, chi2dof, _ = o_fit.fit_curve(ff_name="poly2_Lambda", cf_name="poly2", modify=modify, usesigma=usesigma)
        names.append("poly2_Lambda")
        vals_coeffs.append(val_coff)
        chi2dods.append(chi2dof)

        # methods::poly2_qlam
        val_coff, _, chi2dof, _ = o_fit.fit_curve(ff_name="poly22_qLambda", cf_name="poly22", modify=modify, usesigma=usesigma)
        names.append("poly22_qLambda")
        vals_coeffs.append(val_coff)
        chi2dods.append(chi2dof)

        # methods::Kruger+2019
        val_coff, _, chi2dof, _ = o_fit.fit_curve(ff_name="rad18", cf_name="rad18", modify=modify, usesigma=usesigma)
        names.append("rad18")
        vals_coeffs.append(val_coff)
        chi2dods.append(chi2dof)

        # methods::Kruger+2019
        val_coff, _, chi2dof, _ = o_fit.fit_curve(ff_name="krug19", cf_name="krug19", modify=modify, usesigma=usesigma)
        names.append("krug19")
        vals_coeffs.append(val_coff)
        chi2dods.append(chi2dof)

        dic = {"fitmodel":names, "vals_coeffs":vals_coeffs, "chi2dof":chi2dods}
        df = pd.DataFrame(dic)
        df = df.set_index("fitmodel")
        df["modify"] = modify
        df["usesigma"] = usesigma
        print(df)
        return df

    # ---
    @staticmethod
    def get_fit_val(ffunc, coeffs, models, modify=None):
        # print(ffunc, coeffs, models, modify); exit(1)
        vals = ffunc(models, *coeffs)
        if modify is None:
            return vals
        elif modify == "10**":
            return np.log10(vals)
        elif modify == "log10":
            return 10**(vals)
        else:
            raise NameError("Not implmenented")

    def predict_mass(self, model):
        #print(model)
        vals = []
        mass_ds = self.mass_ds
        for key, m in mass_ds.iterrows():
            if key == "mean":
                vals.append(float(m["vals_coeffs"]))
            elif key == "poly2_Lambda":
                coeffs = m["vals_coeffs"]
                # val = float(FittingFunctions.poly_2_Lambda(model, *coeffs))
                val = self.get_fit_val(FittingFunctions.poly_2_Lambda, coeffs, model, str(mass_ds["modify"][-1]))
                vals.append(val)
            elif key == "poly22_qLambda":
                coeffs = m["vals_coeffs"]
                # val = float(FittingFunctions.poly_2_qLambda(model, *coeffs))
                val = self.get_fit_val(FittingFunctions.poly_2_qLambda, coeffs, model, str(mass_ds["modify"][-1]))
                vals.append(val)
            elif key == "diet16":
                coeffs = m["vals_coeffs"]
                # val = float(FittingFunctions.mej_dietrich16(model, *coeffs))
                val = self.get_fit_val(FittingFunctions.mej_dietrich16, coeffs, model, str(mass_ds["modify"][-1]))
                vals.append(val)
            elif key == "krug19":
                coeffs = m["vals_coeffs"]
                # val = float(FittingFunctions.mej_kruger19(model, *coeffs))
                val = self.get_fit_val(FittingFunctions.mej_kruger19, coeffs, model, str(mass_ds["modify"][-1]))
                vals.append(val)
            else:
                raise NameError("not implemented: {}".format(key))
        mass_ds[model.index[0]] = vals


        return mass_ds

    def predict_vel(self, model):
        #print(model)
        vals = []
        vel_ds = self.vel_ds
        for key, m in vel_ds.iterrows():
            if key == "mean":
                vals.append(float(m["vals_coeffs"]))
            elif key == "poly2_Lambda":
                coeffs = m["vals_coeffs"]
                val = float(FittingFunctions.poly_2_Lambda(model, *coeffs))
                vals.append(val)
            elif key == "poly22_qLambda":
                coeffs = m["vals_coeffs"]
                val = float(FittingFunctions.poly_2_qLambda(model, *coeffs))
                vals.append(val)
            elif key == "diet16":
                coeffs = m["vals_coeffs"]
                val = float(FittingFunctions.vej_dietrich16(model, *coeffs))
                vals.append(val)
            else:
                raise NameError("not implemented: {}".format(key))
        vel_ds[model.index[0]] = vals

        return vel_ds

    def predict_ye(self, model):
        #print(model)
        vals = []
        ye_ds = self.ye_ds
        for key, m in ye_ds.iterrows():
            if key == "mean":
                vals.append(float(m["vals_coeffs"]))
            elif key == "poly2_Lambda":
                coeffs = m["vals_coeffs"]
                val = float(FittingFunctions.poly_2_Lambda(model, *coeffs))
                vals.append(val)
            elif key == "poly22_qLambda":
                coeffs = m["vals_coeffs"]
                val = float(FittingFunctions.poly_2_qLambda(model, *coeffs))
                vals.append(val)
            elif key == "our":
                coeffs = m["vals_coeffs"]
                val = float(FittingFunctions.yeej_ours(model, *coeffs))
                vals.append(val)
            else:
                raise NameError("not implemented: {}".format(key))
        ye_ds[model.index[0]] = vals

        return ye_ds

    def predict_theta(self, model):
        # print(model)
        vals = []
        theta_ds = self.theta_ds
        for key, m in theta_ds.iterrows():
            if key == "mean":
                vals.append(float(m["vals_coeffs"]))
            elif key == "poly2_Lambda":
                coeffs = m["vals_coeffs"]
                val = float(FittingFunctions.poly_2_Lambda(model, *coeffs))
                vals.append(val)
            elif key == "poly22_qLambda":
                coeffs = m["vals_coeffs"]
                val = float(FittingFunctions.poly_2_qLambda(model, *coeffs))
                vals.append(val)
            else:
                raise NameError("not implemented: {}".format(key))
        theta_ds[model.index[0]] = vals

        return theta_ds

    def predict_diskmass(self, model):
        # print(model)
        vals = []
        diskmass_ds = self.diskmass_ds
        for key, m in diskmass_ds.iterrows():
            if key == "mean":
                vals.append(float(m["vals_coeffs"]))
            elif key == "poly2_Lambda":
                coeffs = m["vals_coeffs"]
                val = float(FittingFunctions.poly_2_Lambda(model, *coeffs))
                vals.append(val)
            elif key == "poly22_qLambda":
                coeffs = m["vals_coeffs"]
                val = float(FittingFunctions.poly_2_qLambda(model, *coeffs))
                vals.append(val)
            elif key == "rad18":
                coeffs = m["vals_coeffs"]
                val = float(FittingFunctions.mdisk_radice18(model, *coeffs))
                vals.append(val)
            elif key == "krug19":
                coeffs = m["vals_coeffs"]
                val = float(FittingFunctions.mdisk_kruger19(model, *coeffs))
                vals.append(val)
            else:
                raise NameError("not implemented: {}".format(key))
        diskmass_ds[model.index[0]] = vals

        return diskmass_ds

    # ---

    def get_best(self, v_n, model):
        if v_n == "Mej_tot-geo":
            ds = self.predict_mass(model)
            return np.array(ds[model.index[0]])[0] # sorted, -- 0's best
        elif v_n == "vel_inf_ave-geo":
            ds = self.predict_vel(model)
            return np.array(ds[model.index[0]])[0]
        elif v_n == "Ye_ave-geo":
            ds = self.predict_ye(model)
            return np.array(ds[model.index[0]])[0]
        elif v_n == "theta_rms-geo":
            ds = self.predict_theta(model)
            return np.array(ds[model.index[0]])[0]
        elif v_n == "Mdisk3D":
            ds = self.predict_diskmass(model)
            return np.array(ds[model.index[0]])[0]
        else:
            raise("not implemented")

    def get_worst(self, v_n, model):
        if v_n == "Mej_tot-geo":
            ds = self.predict_mass(model)
            return np.array(ds[model.index[0]])[-1] # sorted, -- -1's worst
        elif v_n == "vel_inf_ave-geo":
            ds = self.predict_vel(model)
            return np.array(ds[model.index[0]])[-1]
        elif v_n == "Ye_ave-geo":
            ds = self.predict_ye(model)
            return np.array(ds[model.index[0]])[-1]
        elif v_n == "theta_rms-geo":
            ds = self.predict_theta(model)
            return np.array(ds[model.index[0]])[-1]
        elif v_n == "Mdisk3D":
            ds = self.predict_diskmass(model)
            return np.array(ds[model.index[0]])[-1]
        else:
            raise("not implemented")

if __name__ == "__main__":
    dfname2 = "/home/vsevolod/GIT/bitbucket/bns_gw170817/data/dynej_disc_literature/LiteratureData.csv"
    allmodels = pd.read_csv(dfname2)
    print(allmodels.keys())
    print(list(set(allmodels["bibkey"])))
    # print(allmodels.loc["bibkey","Nedora:2020"])

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

    dfname2 = "/home/vsevolod/GIT/GitHub/prj_gw170817/datasets/summary_table.csv"
    allmodels = pd.read_csv(dfname2)
    print(allmodels.keys())
    print(list(set(allmodels["bibkey"])))
    #
    # from model_sets import models as mds
    # models = mds.simulations[mds.mask_for_with_dynej]
    # o_bf = BestFits(allmodels, "default", True)
    # model = models[models.index == "BLh_M13641364_M0_LK_SR"]
    # masses = o_bf.predict_mass(model)
    # print(masses['BLh_M13641364_M0_LK_SR'])
    # vels = o_bf.predict_vel(model)
    # yes = o_bf.predict_ye(model)
    # thetas = o_bf.predict_theta(model)
    # mdisk = o_bf.predict_diskmass(model)
    # print(vels)
    # print(o_bf.get_best("vel_inf_ave-geo", model))
    # print(vels)
    # print(yes)
    # print(thetas)
    # print(mdisk)

    #


    # with_lambda = (~np.isnan(allmodels["Lambda"])) & \
    #               (~np.isnan(allmodels["M1"])) & \
    #               (~np.isnan(allmodels["M2"])) & \
    #               (~np.isnan(allmodels["Mb1"])) & \
    #               (~np.isnan(allmodels["Mb2"])) & \
    #               (~np.isnan(allmodels["C1"])) & \
    #               (~np.isnan(allmodels["C2"]))
    # allmodels = allmodels[with_lambda]
    # print(len(allmodels[allmodels["bibkey"] == "Bauswein:2013yna"]))
    # print(np.mean(allmodels[allmodels["bibkey"] == "Bauswein:2013yna"]["Mej_tot-geo"]))
    # print(len(allmodels[allmodels["bibkey"] == "Hotokezaka:2012ze"]))
    # print(np.mean(allmodels[allmodels["bibkey"] == "Hotokezaka:2012ze"]["Mej_tot-geo"]))
    ''' ejecta mass'''
    # o_tbl = Fit_Tex_Tables(allmodels, "Mej_tot-geo", "default", True, deliminator='&')
    # o_tbl.print_stats()
    # o_tbl.print_polyfit_table(ff_name="poly2_Lambda", cf_name="poly2", modify="log10",usesigma=True)
    # o_tbl.print_polyfit_table(ff_name="poly22_qLambda", cf_name="poly22")
    # o_tbl.print_polyfit_table(ff_name="poly22_qLambda", cf_name="poly22")
    # o_tbl.print_fitfunc_table(ff_name="diet16", cf_name="diet16")
    # o_tbl.print_fitfunc_table(ff_name="krug19", cf_name="krug19")
    # o_tbl.print_chi2dofs(["datasets", "mean-chi2dof", "diet16-chi2dof", "krug19-chi2dof", "poly2-chi2dof", "poly22-chi2dof"],
    #                      ["datasets", r"Mean", r"Eq.~\eqref{eq:fit_Mej}", r"Eq.~\eqref{eq:fit_Mej_Kruger}",
    #                                   r"$P_2(\tilde{\Lambda})$", r"$P_2(q,\tilde{\Lambda})$"],
    #                      [".2f", ".2f", ".2f", ".2f", ".2f"], modify="log10",usesigma=True)
    # exit(1)
    ''' ejecta velocity '''
    # o_tbl = Fit_Tex_Tables(allmodels, "vel_inf_ave-geo", "default", True, deliminator="&")
    # o_tbl.print_stats()
    # o_tbl.print_polyfit_table(ff_name="poly2_Lambda", cf_name="poly2")
    # o_tbl.print_polyfit_table(ff_name="poly22_qLambda", cf_name="poly22")
    # o_tbl.print_fitfunc_table(ff_name="diet16", cf_name="diet16")
    # o_tbl.print_chi2dofs(["datasets", "mean-chi2dof", "diet16-chi2dof", "poly2-chi2dof", "poly22-chi2dof"],
    #                      ["datasets", r"Mean", r"Eq.~\eqref{eq:fit_Mej}",
    #                                   r"$P_2(\tilde{\Lambda})$", r"$P_2(q,\tilde{\Lambda})$"],
    #                      [".2f", ".2f", ".2f", ".2f"])

    ''' ejecta electron fraction '''
    # o_tbl = Fit_Tex_Tables(allmodels, "Ye_ave-geo", "default", True, deliminator="&")
    # o_tbl.print_stats()
    # o_tbl.print_polyfit_table(ff_name="poly2_Lambda", cf_name="poly2")
    # o_tbl.print_polyfit_table(ff_name="poly22_qLambda", cf_name="poly22")
    # o_tbl.print_fitfunc_table(ff_name="our", cf_name="our")
    # o_tbl.print_chi2dofs(["datasets", "mean-chi2dof", "our-chi2dof", "poly2-chi2dof", "poly22-chi2dof"],
    #                      ["datasets", r"Mean", r"Eq.~\eqref{eq:fit_Yeej}", r"$P_2(\tilde{\Lambda})$",
    #                       r"$P_2(q,\tilde{\Lambda})$"],
    #                      [".2f", ".2f", ".2f", ".2f"])
    #
    # print(list(set(allmodels["bibkey"])))

    ''' ejecta rms angle '''
    # o_tbl = Fit_Tex_Tables(allmodels, "theta_rms-geo", "default", True, deliminator="&")
    # o_tbl.print_stats()
    # o_tbl.print_polyfit_table(ff_name="poly2_Lam", cf_name="poly2")
    # o_tbl.print_polyfit_table(ff_name="poly22_qLam", cf_name="poly22")
    # o_tbl.print_fitfunc_table(ff_name="krug19", cf_name="krug19")
    # o_tbl.print_chi2dofs(["datasets", "mean-chi2dof", "poly2-chi2dof", "poly22-chi2dof"],
    #                      ["datasets", r"Mean", r"$P_2(\tilde{\Lambda})$", r"$P_2(q,\tilde{\Lambda})$"],
    #                      [".2f",  ".2f", ".2f"], usesigma=True)

    ''' disk mass '''
    o_tbl = Fit_Tex_Tables(allmodels, "Mdisk3D", "default", True, deliminator="&")
    o_tbl.print_stats()
    # o_tbl.print_polyfit_table(ff_name="poly2_Lam", cf_name="poly2")
    # o_tbl.print_polyfit_table(ff_name="poly22_qLambda", cf_name="poly22", modify="log10")
    # o_tbl.print_fitfunc_table(ff_name="krug19", cf_name="krug19", modify=None)
    # o_tbl.print_chi2dofs(["datasets", "mean-chi2dof", "rad18-chi2dof", "krug19-chi2dof", "poly2-chi2dof", "poly22-chi2dof"],
    #                      ["datasets", r"Mean", r"Eq.~\eqref{eq:fit_Mdisk}", r"Eq.~\eqref{eq:fit_Mdisk_Kruger}",
    #                       r"$P_2(\tilde{\Lambda})$", r"$P_2(q,\tilde{\Lambda})$"],
    #                      [".2f", ".2f", ".2f", ".2f", ".2f"],usesigma=False)
    #
    # print(list(set(allmodels["bibkey"])))



    # o_fit = Fit_Data(allmodels, fit_v_n="vel_inf_ave-geo", err_method="default")
    # o_fit.fit_curve(ff_name="poly22_qLam", cf_name="poly22")
    # o_fit.fit_curve(ff_name="diet16", cf_name="diet16")
    #
    # o_fit = Fit_Data(allmodels, fit_v_n="Ye_ave-geo", err_method="default")
    # o_fit.fit_curve(ff_name="poly22_qLam", cf_name="poly22")
    # o_fit.fit_curve(ff_name="our", cf_name="our")

    # o_fit = Fit_Data(allmodels, fit_v_n="Mej_tot-geo", err_method="default", clean_nans=True)
    # o_fit.get_chi2dof_for_mean()
    # o_fit.fit_curve(ff_name="poly2_Lam", cf_name="poly2")
    # o_fit.fit_curve(ff_name="poly22_qLam", cf_name="poly22")
    # o_fit.fit_curve(ff_name="diet16", cf_name="diet16")
    # o_fit.fit_curve(ff_name="krug19", cf_name="krug19")
    #
    # o_fit = Fit_Data(allmodels, fit_v_n="Mdisk3D", err_method="default", clean_nans=True)
    # o_fit.get_chi2dof_for_mean()
    # o_fit.fit_curve(ff_name="poly2_Lam", cf_name="poly2")
    # o_fit.fit_curve(ff_name="poly22_qLam", cf_name="poly22")
    # o_fit.fit_poly(ff_name="poly22_qLam")
    # o_fit.fit_curve(ff_name="rad18", cf_name="rad18")
    # o_fit.fit_curve(ff_name="krug19", cf_name="krug19")
    # o_fit.git_func(ff_name="krug19", cf_name="krug19")

    ''' --- '''