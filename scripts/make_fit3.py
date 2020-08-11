from __future__ import division

import numpy as np
import scipy.optimize as opt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats

# datasets
from model_sets import models_vincent2019 as vi
from model_sets import models_radice2018 as rd
from model_sets import groups as md
from model_sets import models_kiuchi2019 as ki  #
from model_sets import models_sekiguchi2016 as se16  # [23] arxive:1603.01918 # no Mb
from model_sets import models_sekiguchi2015 as se15  # [-]  arxive:1502.06660 # no Mb
from model_sets import models_bauswein2013 as bs  # [20] arxive:1302.6530
from model_sets import models_lehner2016 as lh  # [22] arxive:1603.00501
from model_sets import models_hotokezaka2013 as hz  # [19] arxive:1212.0905
from model_sets import models_dietrich_ujevic2016 as du
from model_sets import models_dietrich2015 as di15  # [21] arxive:1504.01266
from model_sets import models_dietrich2016 as di16  # [24] arxive:1607.06636

from collections import OrderedDict

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
            _val = "$"+str(_val).replace("e", r'\times10^{')+"}$"

    # if fancy:
    #     if _val.__contains__("e-"):
    #         _val = "$"+str(_val).replace("e-", r'\times10^{-')+"}$"
    #     elif _val.__contains__("e+"):
    #         _val = "$" + str(_val).replace("e+", r'\times10^{') + "}$"
    #     else:
    #         pass

    return _val

def create_combine_dataframe2(datasets, v_ns, v_ns_err,
                             special_instructions,
                             key_for_usable_dataset="fit",
                             ifabsent=np.nan):
    print("Datasets: {}".format(datasets.keys()))

    import pandas
    new_data_frame = {}
    #

    new_data_frame["models"] = []
    #
    index_arr = []
    datasets_names = []
    for name in datasets.keys():
        dic = datasets[name]
        flag = dic[key_for_usable_dataset]
        if flag:
            d_cl = dic["data"]  # md, rd, ki ...
            # models = dic["models"]
            # print(dic["models"].index); exit(1)
            models = list(dic["models"].index)
            for model in models:
                datasets_names.append(name)
                index_arr.append(model)
            # index_arr.append(list(dic["models"].index))
    new_data_frame['models'] = index_arr
    new_data_frame['dset_name'] = datasets_names
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

                if not v_n in d_cl.translation.keys():
                    print("\tWarning: v_n:{} is not in keys for {}".format(v_n, name))
                    shape = len(models.index)
                    value_arr = np.append(value_arr, np.full(shape, ifabsent))
                else:
                    print("appending dataset: {}".format(name))
                    x = d_cl.get_mod_data(v_n, special_instructions, models)
                    value_arr = np.append(value_arr, x)
                #
        print(len(value_arr))
        new_data_frame[v_n] = value_arr
    #
    print("concencating errors")
    for v_n in v_ns_err:
        error_arr = []
        for name in datasets.keys():
            dic = datasets[name]
            flag = dic[key_for_usable_dataset]
            if flag:
                d_cl = dic["data"]
                models = dic["models"]
                print("appending dataset errors: {}".format(name))
                err = d_cl.get_mod_err(v_n, special_instructions, models)
                error_arr = np.append(error_arr, err)
        print(len(error_arr))
        new_data_frame["err_" + v_n] = error_arr

    df = pandas.DataFrame(new_data_frame, index=new_data_frame["models"])

    # print(df[["Lambda","q","Mej_tot-geo","err_Mej_tot-geo"]]); exit(1)

    return df

class Fitting_Coefficients:

    def __init__(self):
        pass

    @staticmethod
    def mej_radice18():
        a = -0.657
        b = 4.254
        c = -32.61
        d = 5.205
        n = -0.773
        return np.array((a, b, c, d, n))

    @staticmethod
    def mej_kruger20():
        a = -9.3335
        b = 114.17
        c = -337.56
        n = 1.5465
        return np.array((a, b, c, n))

    @staticmethod
    def mej_us_radice18():
        a = -0.559
        b = 0.384
        c = -13.431
        d = 10.972
        n = -5.605
        return np.array((a, b, c, d, n))

    @staticmethod
    def mej_all():
        a = 3.985
        b = 26.286
        c = -41.628
        d = -110.723
        n = -0.440
        return np.array((a, b, c, d, n))

    @staticmethod
    def mej_all_2():
        a = 1.573
        b = 14.153
        c = -19.973
        d = -51.632
        n = -0.500
        return np.array((a, b, c, d, n))

    #

    @staticmethod
    def vej_radice():
        a = -0.287
        b = 0.494
        c = -3.000
        return np.array((a, b, c))

    @staticmethod
    def vej_us_radice():
        # -0.547 & 1.043 & -1.388 & 0.389
        a = -0.547
        b = 1.043
        c = -1.388
        return np.array((a, b, c))

    @staticmethod
    def vej_all():
        a = 0.033
        b = 0.184
        c = -5.071
        return np.array((a, b, c))

    @staticmethod
    def vel_all_2():
        a = -0.422
        b = 0.834
        c = -1.510
        return np.array((a, b, c))

    #

    @staticmethod
    def yeej_all():
        a = 0.099
        b = 0.347
        c = -5.610
        return np.array((a, b, c))

    @staticmethod
    def yeej_us_radice():
        # 0.118 & 0.328 & -4.160 & 1.094
        a = 0.118
        b = 0.328
        c = -4.160
        return np.array((a, b, c))

    @staticmethod
    def yeej_all_2():
        a = 0.177
        b = 0.452
        c = -4.611
        return np.array((a, b, c))

    #

    @staticmethod
    def mdisk_kruger20():
        # https://arxiv.org/pdf/2002.07728.pdf
        a = -8.1324
        c = 1.4820
        d = 1.7784
        return np.array((a, c, d))

    @staticmethod
    def mdisk_radice18():
        a = -0.243087590223
        b = 0.436980750624
        c = 30.4790977667
        d = 332.568017486
        return np.array((a, b, c, d))

    @staticmethod
    def mdisk_us_radice18():
        # 0.091 & 0.078 & 395.362 & 88.938 & 2.306
        a = 0.091
        b = 0.078
        c = 395.362
        d = 88.938
        return np.array((a, b, c, d))

    @staticmethod
    def mdisk_all():
        a = 0.137
        b = 0.036
        c = 516.135
        d = 0.277
        return np.array((a, b, c, d))

    @staticmethod
    def mdisk_all_2():
        a = 0.137
        b = 0.036
        c = 516.135
        d = 0.277
        return np.array((a, b, c, d))

    @staticmethod
    def mdisk_2_2_poly():
        a =-0.8951
        b =1.195
        c =4.292 * 1.e-4
        d =-0.3991
        e = 4.778 * 1.e-5
        f = -2.266 * 1e-7
        return np.array((a, b, c ,d , e, f))

class Fitting_Functions:

    def __init__(self):
        pass

    @staticmethod
    def mej_flat_mean(x, v, v_n = "Mej_tot-geo"):
        mean = float(np.mean(v[v_n])) / 1.e-3 #
        return np.full(len(v.M2), mean)

    @staticmethod
    def mej_dietrich16(x, v, v_n = "Mej_tot-geo"):
        a, b, c, d, n = x
        return (a * (v.M2 / v.M1) ** (1.0 / 3.0) * (1. - 2 * v.C1) / (v.C1) + b * (v.M2 / v.M1) ** n +
                c * (1 - v.M1 / v.Mb1)) * v.Mb1 + \
               (a * (v.M1 / v.M2) ** (1.0 / 3.0) * (1. - 2 * v.C2) / (v.C2) + b * (v.M1 / v.M2) ** n +
                c * (1 - v.M2 / v.Mb2)) * v.Mb2 + \
               d

    @staticmethod
    def mej_kruger20(x, v, v_n = "Mej_tot-geo"):
        a, b, c, n = x
        return ((a / v.C1) + b * ((v.M2 ** n) / (v.M1 ** n)) + c * v.C1) * v.M1 + \
               ((a / v.C2) + b * ((v.M1 ** n) / (v.M2 ** n)) + c * v.C2) * v.M2

    @staticmethod
    def vej_dietrich16(x, v, v_n="vel_inf_ave-geo"):
        a, b, c = x
        return a * (v.M1 / v.M2) * (1. + c * v.C1) + \
               a * (v.M2 / v.M1) * (1. + c * v.C2) + b

    @staticmethod
    def vej_poly_22(x, v, v_n="vel_inf_ave-geo"):
        b0, b1, b2, b3, b4, b5 = x
        x1 = v.q
        x2 = v.Lambda
        return  b0 + b1*x1 + b2*x2 + b3*x1**2 + b4*x1*x2 + b5*x2**2

    @staticmethod
    def yeej_like_vej(x, v, v_n = "Ye_ave-geo"):
        a, b, c = x
        return a * 1e-5 * (v.M1 / v.M2) * (1. + c * 1e5 * v.C1) + \
               a * 1e-5 * (v.M2 / v.M1) * (1. + c * 1e5 * v.C2) + b

    @staticmethod
    def mdisk_radice18(x, v, v_n = "Mdisk3D"):
        a, b, c, d = x
        return np.maximum(a + b * (np.tanh((v["Lambda"] - c) / d)), 1e-3)

    @staticmethod
    def mdisk_kruger20(x, v, v_n = "Mdisk3D"):
        a, c, d = x
        val = 5. * 10 ** (-4)
        arr = v["M2"] * np.maximum(val, ((a * v["C2"]) + c) ** d)
        arr[np.isnan(arr)] = val

        # print(" --- M2 ---  ")
        # print(v.M2)
        # # print(" --- C2 ---+")
        # print(v.C2)
        # print(" ---0 ")
        # print(a * v.C2 + c)
        # print(" ---1 ")
        # print((a * v.C2 + c)**d)
        # print(" ---2 ")
        # print(np.maximum((a * v.C2 + c)**d, 5.e-4))
        # print(" ---3 ")
        #print(v["M2"] * np.maximum((a * v["C2"] + c) ** d, 5.e-4))
        # exit(1)
        return arr

    @staticmethod
    def mdisk_2_2_poly(x, v, v_n = "Mdisk3D"):
        a, b, c, d, e, f = x
        return a + b * v.q + c * v.Lambda + \
               d * (v.q) ** 2 + e * v.q * v.Lambda + f * (v.Lambda) ** 2

    # ---
    @staticmethod
    def poly_2_Lambda(x, v, v_n = None):
        b0, b1, b2 = x
        #print(b0 + b1*v.Lambda + b2*v.Lambda**2s)
        return b0 + b1*v.Lambda + b2*v.Lambda**2

    @staticmethod
    def poly_2_qLambda(x, v, v_n = None):
        b0, b1, b2, b3, b4, b5 = x
        return b0 + b1 * v.q + b2 * v.Lambda + b3 * v.q ** 2 + b4 * v.q * v.Lambda + b5 * v.Lambda ** 2
        #return b0 + b1 * v.Lambda + b2 * v.q + b3 * v.Lambda ** 2 + b4 * v.Lambda * v.q + b5 * v.q ** 2

class Fit_Data_old:
    """
    Fit the data with least square  and with polynomial regrression
    """

    def __init__(self, datasets, v_n="Mej_tot-geo", chi_method="scipy", error_method="1std"):

        self.fit_v_n = v_n
        self.chi_method = chi_method
        self.error_method = error_method

        v_ns = ["M1", "M2", "C1", "C2", "Mb1", "Mb2", "Lambda", "q", self.fit_v_n]
        #v_ns = [self.fit_v_n]
        v_ns_err = [self.fit_v_n]

        dataframe = create_combine_dataframe2(datasets, v_ns, v_ns_err, {})
        dataframe = dataframe[np.isfinite(dataframe[self.fit_v_n])]
        dataframe = dataframe[dataframe["Lambda"] <= 2000]
        self.dataframe = dataframe
        # dataframe = dataframe[dataframe["q"] < 1.29]
        # dataframe = dataframe[(dataframe["C2"] > 0.136) & (dataframe["C2"] < 0.218)]
        print(np.array(np.isfinite(dataframe[self.fit_v_n])))
        # if self.fit_v_n == "Mej_tot-geo":
        #     dataframe[self.fit_v_n] = dataframe[self.fit_v_n] * 1.e3

        print(" --- {} ---- ".format(self.fit_v_n))

        print(dataframe[self.fit_v_n].describe(percentiles=[0.8, 0.9, 0.95]))

        self.std = dataframe[self.fit_v_n].std()
        # self.mean = dataframe[self.fit_v_n].mean()
        n = len(np.array(dataframe[self.fit_v_n]))
        #
        # if chi_method == "scipy":
        #     print("[fittig mean] Using scipy norm with ddof=0")
        #     chi2, p = stats.chisquare(np.array(dataframe[self.fit_v_n]),
        #                              np.full(self.num, self.mean),
        #                              ddof=1)
        # else:
        #     chi2 = np.sum(((np.array(dataframe[self.fit_v_n]) - self.mean) ** 2) / self.std ** 2)

        y_vals = np.array(dataframe[self.fit_v_n], dtype=float)
        # if self.error_method == "1std": y_errs = float(self.std)
        # else: raise NameError("Unknown error_method: {}".format(error_method))
        y_errs = self.get_err(y_vals)
        mean = np.mean(y_vals)

        z = (y_vals - mean) / y_errs
        chi2 = np.sum(z ** 2.)

        chi2dof = chi2 / float(n - 1) # 1 -- ddof for a mean
        sigma = np.sqrt(2. / float(n - 1))
        nsig = (chi2dof - 1) / sigma

        '''  -------------------------  '''

        self.chi2 = chi2
        self.chi2dof = chi2dof
        self.num = n
        self.mean = mean

        print("-----------------------------------------------")
        print("\t num:     {:d} in the sample".format(self.num))
        print("\t std:     {}".format(self.std))
        print("\t mean:    {}".format(self.mean))
        print("\t chi2:    {} [{}]".format(self.chi2, self.chi_method))
        print("\t cho2dof: {}".format(self.chi2dof))
        print("\t nsig:    {} sigma's".format(nsig))
        print("-----------------------------------------------")

        #self.dataframe = dataframe
        #dataframe.to_csv("/data01/numrel/vsevolod.nedora/tmp/dataset.csv")


    def get_err(self, vals):
        res = np.zeros(0,)
        if self.error_method == "std":
            res =  np.std(vals)
        elif self.error_method == "2std":
            res = 2. * np.std(vals)
        elif self.error_method == "arr":
            res = self.dataframe["err_" + self.fit_v_n]
        else:
            raise NameError("no err method: {}".format(self.error_method))

        # if self.fit_v_n == "Mej_tot-geo": res = res * 1e3
        return res

    def get_chi2(self, y_vals, y_expets, y_errs):
        assert len(y_vals) == len(y_expets)
        z = (y_vals - y_expets) / y_errs
        chi2 = np.sum(z ** 2.)
        return chi2

    def get_score(self, y_true, y_pred):
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

        #y_true = y_true[~np.isnan(y_true)]
        #y_pred = y_pred[~np.isnan(y_pred)]

        y_true = y_true[np.isfinite(y_true)]
        y_pred = y_pred[np.isfinite(y_pred)]
        # print("--------")
        # print(y_true)
        # print(y_pred)
        # print("--------")

        assert len(y_true) == len(y_pred)
        u = np.sum((y_true - y_pred)**2.)
        v = np.sum((y_true - np.mean(y_true))**2.)
        res = (1.-u/v)
        #print(res)
        # assert res >= 0
        # assert res <= 1

        #if res < 0.: return np.nan # neg
        if res > 1.: return np.nan

        return res

    def get_ch2dof(self, chi2, n, k):
        """
        :param chi2: chi squared
        :param n: number of elements in a sample
        :param k: n of independedn parameters (1 -- mean, 2 -- poly1 fit, etc)
        :return:
        """
        return chi2 / (n - k)

    def get_ch2dof_mean(self):
        y_vals = np.array(self.dataframe[self.fit_v_n], dtype=float)
        if self.fit_v_n == "Mej_tot-geo": y_vals = y_vals * 1e3
        mean = float(np.mean(y_vals))
        y_errs = self.get_err(y_vals)
        chi2 = self.get_chi2(y_vals, np.full(len(y_vals), mean), y_errs)
        n = len(np.array(self.dataframe[self.fit_v_n]))
        chi2dof = self.get_ch2dof(chi2, n, 1) # k=1 -- ddof for a mean
        return chi2dof

    def get_stats(self, v_ns=["n", "mean", "std", "80", "90", "95", "chi2", "chi2dof"]):

        res = []
        for v_n in v_ns:
            if v_n == "n":
                res.append(self.num)
            if v_n == "mean":
                res.append(self.mean)
            if v_n == "std":
                res.append(self.std)
            if v_n == "90" or v_n == "80" or v_n == "95":
                dic = self.dataframe[self.fit_v_n].describe(percentiles=[0.8, 0.9, 0.95])
                val = float(dic[v_n+"%"])
                res.append(val)
            if v_n == "chi2":
                res.append(self.chi2)
            if v_n == "chi2dof":
                res.append(self.chi2dof)

        return res

    def fitting_functions(self, x, v, name="Radice+2018"):

        if self.fit_v_n == "Mej_tot-geo":
            if name == "Radice+2018":
                return Fitting_Functions.mej_dietrich16(x, v)
            elif name == "Kruger+2020":
                return Fitting_Functions.mej_kruger20(x, v)
            else:
                raise NameError("no fitting function for: {} ".format(name))

        elif self.fit_v_n == "vel_inf_ave-geo":
            if name == "Radice+2018":
                return Fitting_Functions.vej_dietrich16(x, v)
            else:
                raise NameError("no fitting function for: {} ".format(name))

        elif self.fit_v_n == "Ye_ave-geo":
            return Fitting_Functions.yeej_like_vej(x, v)

        elif self.fit_v_n == "Mdisk3D":
            if name == "Radice+2018":
                return Fitting_Functions.mdisk_radice18(x, v)

            if name == "Kruger+2020":
                return Fitting_Functions.mdisk_kruger20(x, v)

            if name == "flat":
                return np.full(len(v["Lambda"]), 0.126323)

            else:
                raise NameError("no fitting function for: {} ".format(name))

        else:
            raise NameError("No fit. funct. for v_n:{}".format(self.fit_v_n))

    def coefficients(self, name="Radice+2018"):

        if self.fit_v_n == "Mej_tot-geo":
            if name == "Radice+2018":
                # Radice:2018pdn
                return Fitting_Coefficients.mej_radice18()
            elif name == "Kruger+2020":
                # Radice:2018pdn
                return Fitting_Coefficients.mej_kruger20()
            elif name == "all":
                # {Radice:2018pdn,Dietrich:2015iva,Dietrich:2016lyp,Kiuchi:2019lls,Vincent:2019kor}
                  return Fitting_Coefficients.mej_all()
            else:
                raise NameError(" coeff name: {} is not recognized v_n:{}".format(name, self.fit_v_n))
        elif self.fit_v_n == "vel_inf_ave-geo":
            if name == "Radice+2018":
                # Radice:2018pdn
                return Fitting_Coefficients.vej_radice()
            elif name == "all":
                # Dietrich:2016lyp,Dietrich:2015iva,Radice:2018pdn,Vincent:2019kor
                return Fitting_Coefficients.vej_all()
            else:
                raise NameError("no fitting function for: {} v_n:{}".format(name, self.fit_v_n))
        elif self.fit_v_n == "Ye_ave-geo":
            if name == "us":
                a = 0.139637775679
                b = 0.33996686385
                c = -3.70301958353
                return np.array((a, b, c))
            elif name == "all":
                # Radice:2018pdn,Vincent:2019kor
                return Fitting_Coefficients.yeej_all()
            else:
                raise NameError("no fitting coeffs for: {} v_n:{}".format(name, self.fit_v_n))
        elif self.fit_v_n == "Mdisk3D":
            if name == "Radice+2018":
                # Radice:2018pdn + us
                return Fitting_Coefficients.mdisk_radice18()
            elif name == "Kruger+2020":
                return Fitting_Coefficients.mdisk_kruger20()
            elif name == "all":
                # Radice:2018pdn,Dietrich:2015iva,Dietrich:2016lyp,Kiuchi:2019lls,Vincent:2019kor,Dietrich:2015iva,Dietrich:2016lyp,Kiuchi:2019lls,Vincent:2019kor
                return Fitting_Coefficients.mdisk_all()
            elif name == "flat":
                return np.array((0, 0, 0, 0))
            else:
                raise NameError(" coeff name: {} is not recognized".format(name))
        else:
            raise NameError("No coeffs found for v_n: {}".format(self.fit_v_n))

    def residuals(self, x, data, ffname ="Radice+2018", rs_name="Radice+2018"):

        if self.fit_v_n == "Mej_tot-geo":
            xi = self.fitting_functions(x, data, ffname)
            return 1.e-3 * (xi - 1.e3 * data[self.fit_v_n])

        xi = self.fitting_functions(x, data, ffname)
        return (xi - data[self.fit_v_n])

        # if self.fit_v_n == "Mej_tot-geo":
        #     if name == "Radice+2018":
        #         xi = self.fitting_functions(x, data, name)
        #         return 1.e-3*(xi - 1.e3*data[self.fit_v_n])
        #         #return ((data[self.fit_v_n] - xi) ** 2)
        #         # return ((1.e-3*(1.e3*data[] - xi)) ** 2)
        #     else:
        #         raise NameError("no residuals for name: {}".format(name))
        # else:
        #     if name == "Radice+2018":
        #         xi = self.fitting_functions(x, data, name)
        #         return (xi - data[self.fit_v_n])
        #         # return ((data[self.fit_v_n] - xi) ** 2)
        #     else:
        #         raise NameError("no residuals for name: {}".format(name))

    def fit2(self, ff_name, cf_name, rs_name):

        print("ff_name:{} cf_name:{} rs_name:{}".format(ff_name, cf_name, rs_name))

        y_vals = np.array(self.dataframe[self.fit_v_n])
        print("y_vals: {}".format(y_vals))
        x0 = self.coefficients(cf_name)
        print("coeffs: {}".format(x0))
        xi = self.fitting_functions(x0, self.dataframe, ff_name)
        print("y inferred: {}".format(xi))
        ### if self.fit_v_n == "Mej_tot-geo": xi = xi / 1.e3

        # if self.chi_method == "scipy":
        #     chi2, p = stats.chisquare(np.array(self.dataframe[self.fit_v_n]), xi, ddof=len(x0))
        # else:
        #     chi2 = np.sum((self.dataframe[self.fit_v_n] - xi) ** 2 / stats.tstd(xi)**2)
        # else: chi2 = np.sum((xi - self.dataframe[self.fit_v_n]) ** 2)
        chi2 = self.get_chi2(y_vals, xi, self.get_err(y_vals))
        print("chi2 original: {}".format(chi2))

        res = opt.least_squares(self.residuals, x0, args=(self.dataframe, ff_name, rs_name)) # res.x -- new coeffs
        xi = self.fitting_functions(res.x, self.dataframe, ff_name)
        ### if self.fit_v_n == "Mej_tot-geo": xi = xi / 1.e3
        # if self.chi_method == "scipy":
        #     chi2, p = stats.chisquare(np.array(self.dataframe[self.fit_v_n]), xi, ddof=len(x0))
        # else:
        #     chi2 = np.sum(((self.dataframe[self.fit_v_n] - xi) ** 2) / stats.tstd(xi)**2)
        chi2 = self.get_chi2(y_vals, xi, self.get_err(y_vals))
        chi2dof = self.get_ch2dof(chi2, len(y_vals), len(x0))

        print("chi2    fit: {}".format(chi2))
        print("chi2dof fit: {}".format(chi2dof))

        print("Fit coefficients:")
        for i in range(len(x0)):
            print("  coeff[{}] = {}".format(i, res.x[i]))

        #print(y_vals, xi);exit(1)
        if self.fit_v_n == "Mej_tot-geo": y_vals = y_vals * 1e3
        score = self.get_score(y_vals, xi) # R^2 coefficient

        return res.x, chi2, chi2dof, score

    def linear_regression(self, degree=1, v_n_x = "Mej_tot-geo", v_n_y="Lambda"):
        '''
        given x - Mdisk and y - Lambda,
        find y = b0 + b1 * x

        Here x - dependent variables, outputs, or responses (predicted responses)
             y - independent variables, inputs, or predictors.
             b0 - intercept (shows the point where the estimated regression line crosses the 'y' axis)
             b1 - determines the slope of the estimated regression line.

        y1 - f(x1) = y1 - b0 - b1*x1 for i = 1... n
        Here y1 -- residuals

        Data fromat
        y = [ 5 20 14 32 22 38]
        x = [[ 5]
             [15]
             [25]
             [35]
             [45]
             [55]]


        :return:
        '''
        # x = np.array(self.dataframe["Mdisk3D"], dtype=float).reshape((-1, 1)) # dependent variables, outputs, or responses.
        # y = np.array(self.dataframe["Lambda"], dtype=float) # independent variables, inputs, or predictors.
        #
        # model = LinearRegression() # fits the model
        #
        # model.fit(x, y)
        #
        # r_sq = model.score(x, y)
        #
        # print('coefficient of determination:', r_sq)
        #
        # print('intercept:', model.intercept_) # which represents the coefficient, b[0]
        # print('slope:', model.coef_) # which represents b[1]
        #
        # y_pred = model.predict(x)
        # print('predicted response:', y_pred)

        x = np.array(self.dataframe[v_n_x], dtype=float).reshape((-1, 1))
        # y = np.array(self.dataframe[v_n_y], dtype=float)
        if v_n_y == "Mej_tot-geo":
            y = np.array(self.dataframe[v_n_y], dtype=float) * 1.e3
        else:
            y = np.array(self.dataframe[v_n_y], dtype=float)
        # # new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
        # # print('intercept:', new_model.intercept_)
        print("------ Polynomial regression x:{} y:{} deg:{} ------ ".format(v_n_x, v_n_y, degree))
        transformer = PolynomialFeatures(degree=degree, include_bias=False)
        transformer.fit(x)
        x_ = transformer.transform(x)
        model = LinearRegression().fit(x_, y)
        r_sq = model.score(x_, y)

        std = stats.tstd(y)
        y_pred = model.predict(x_)

        print('coefficient of determination R2: {}'.format(r_sq))
        print('intercept b0: {}'.format(model.intercept_))
        print('coefficients bi: {}'.format(model.coef_))

        y_pred = model.predict(x_)
        chi2 = self.get_chi2(y, y_pred, self.get_err(y))
        chi2dof = self.get_ch2dof(chi2, self.num, degree+1)
        #print('predicted response:', y_pred)
        # if self.chi_method == "scipy":
        #     print("[poly regres] deg:{} using scipy for chi ddof:{}".format(degree, degree+1))
        #     chi2, p = stats.chisquare(y, y_pred, ddof=degree+1)
        # else:
        #     chi2 = np.sum(((y - y_pred)**2) / std**2)
        #else: chi2 = np.sum((y - y_pred)**2)
        print('chi2: {}'.format(chi2))

        print("------------------------------------------------------")

        coeffs = np.zeros(len(model.coef_)+1)
        coeffs[0] = model.intercept_
        coeffs[1:] = model.coef_

        # return(model.intercept_, model.coef_, chi2, chi2dof, r_sq)
        return(coeffs, chi2, chi2dof, r_sq)

    def linear_regression2(self, degree=1, v_ns_x = ["Mej_tot-geo"], v_ns_y=["Lambda"]):
        '''
        given x - Mdisk and y - Lambda,
        find y = b0 + b1 * x

        Here x - dependent variables, outputs, or responses (predicted responses)
             y - independent variables, inputs, or predictors.
             b0 - intercept (shows the point where the estimated regression line crosses the 'y' axis)
             b1 - determines the slope of the estimated regression line.

        y1 - f(x1) = y1 - b0 - b1*x1 for i = 1... n
        Here y1 -- residuals

        Data fromat
        y = [ 5 20 14 32 22 38]
        x = [[ 5]
             [15]
             [25]
             [35]
             [45]
             [55]]


        :return:
        '''
        # x = np.array(self.dataframe["Mdisk3D"], dtype=float).reshape((-1, 1)) # dependent variables, outputs, or responses.
        # y = np.array(self.dataframe["Lambda"], dtype=float) # independent variables, inputs, or predictors.
        #
        # model = LinearRegression() # fits the model
        #
        # model.fit(x, y)
        #
        # r_sq = model.score(x, y)
        #
        # print('coefficient of determination:', r_sq)
        #
        # print('intercept:', model.intercept_) # which represents the coefficient, b[0]
        # print('slope:', model.coef_) # which represents b[1]
        #
        # y_pred = model.predict(x)
        # print('predicted response:', y_pred)

        #x = np.array(self.dataframe[v_n_x], dtype=float).reshape((-1, 1))
        if len(v_ns_x) == 1:
            x = np.array(self.dataframe[v_ns_x[0]], dtype=float)
        else:
            x = []
            for v_n_x in v_ns_x:
                x_ = np.array(self.dataframe[v_n_x], dtype=float)
                x.append(x_)
                #print(len(x_))
            x = np.reshape(np.array(x), (len(v_ns_x), len(x_))).T
            #print(x)
            #exit(1)
        if v_ns_y[0] == "Mej_tot-geo":
            y = np.array(self.dataframe[v_ns_y[0]], dtype=float) * 1e3
        else:
            y = np.array(self.dataframe[v_ns_y[0]], dtype=float)
        # if len(v_ns_y) == 1:
        #     y = np.array(self.dataframe[v_ns_y[0]], dtype=float)
        # else:
        #     y = []
        #     for v_n_y in v_ns_y:
        #         y_ = np.array(self.dataframe[v_n_y], dtype=float)
        #         y.append(y_)
        #     y = np.reshape(np.array(y), (len(v_ns_y), len(y_)))
        #     print(y)
        #     exit(1)

        print("------ Polynomial regression2 xs:{} ys:{} deg:{} ------ ".format(v_ns_x, v_ns_y, degree))
        transformer = PolynomialFeatures(degree=degree, include_bias=False)
        transformer.fit(x)
        x_ = transformer.transform(x)
        model = LinearRegression().fit(x_, y)
        r_sq = model.score(x_, y)

        std = stats.tstd(y)

        print('coefficient of determination R2: {}'.format(r_sq))
        print('intercept b0: {}'.format(model.intercept_))
        print('coefficients bi: {}'.format(model.coef_))

        y_pred = model.predict(x_)
        chi2 = self.get_chi2(y, y_pred, self.get_err(y))
        chi2dof = self.get_ch2dof(chi2, self.num, degree+1)
        #print('predicted response:', y_pred)
        # if self.chi_method == "scipy":
        #     print("[poly regres] deg:{} using scipy for chi ddof:{}".format(degree, degree+1))
        #     chi2, p = stats.chisquare(y, y_pred, ddof=degree+1)
        # else:
        #     chi2 = np.sum(((y - y_pred)**2) / std**2)
        #else: chi2 = np.sum((y - y_pred)**2)
        print('chi2: {}'.format(chi2))

        print("------------------------------------------------------")

        coeffs = np.zeros(len(model.coef_)+1)
        coeffs[0] = model.intercept_
        coeffs[1:] = model.coef_

        # return(model.intercept_, model.coef_, chi2, chi2dof, r_sq)
        return(coeffs, chi2, chi2dof, r_sq)

    #

    def mean_predict(self):
        """

        :param x_values:
        :param err:
        :return: (mean, mean-std, mean+std)
        """

        y_vals = np.array(self.dataframe[self.fit_v_n], dtype=float)
        y_errs = self.get_err(y_vals)
        mean = np.mean(y_vals)
        std = stats.tstd(y_vals)
        if std > mean:
            return mean, std, mean
        else:
            return mean, std, std

    def fit2_predict(self, x_values, ff_name, cf_name, rs_name):
        """

        :param ff_name:
        :param cf_name:
        :param rs_name:
        :param x_values: [x_val, x_lower, x_upper] or just [x_val]
        :return: (x, chi2, chi2dof, score, xi)
        """
        #assert len(x_values) == 1 or len(x_values) == 3
        x, chi2, chi2dof, score = self.fit2(ff_name, cf_name, rs_name) # x - coefficients
        xi = self.fitting_functions(x, x_values, ff_name) # get fitted values
        if self.fit_v_n == "Mej_tot-geo": xi = xi / 1.e3 # for ejecta mass fitting functions is for 1e3
        return xi

    def linear_regression_predict(self, vals, degree=1, v_n_x = "Mej_tot-geo", v_n_y="Lambda"):

        #assert len(vals) == 1 or len(vals) == 3

        x = np.array(self.dataframe[v_n_x], dtype=float).reshape((-1, 1))
        y = np.array(self.dataframe[v_n_y], dtype=float)
        # # new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
        # # print('intercept:', new_model.intercept_)
        print("------ Polynomial regression x:{} y:{} deg:{} ------ ".format(v_n_x, v_n_y, degree))
        transformer = PolynomialFeatures(degree=degree, include_bias=False)
        transformer.fit(x)
        x_ = transformer.transform(x)
        model = LinearRegression().fit(x_, y)
        r_sq = model.score(x_, y)

        # std = stats.tstd(y)

        print('coefficient of determination R2: {}'.format(r_sq))
        print('intercept b0: {}'.format(model.intercept_))
        print('coefficients bi: {}'.format(model.coef_))

        # y_pred = model.predict(x_)
        # chi2 = self.get_chi2(y, y_pred, self.get_err(y))
        # chi2dof = self.get_ch2dof(chi2, self.num, degree + 1)

        y_pred = model.predict(np.array(vals).reshape(-1,1))

        return y_pred

    def linear_regression2_predict(self, vals, degree=1, v_ns_x = ["Mej_tot-geo"], v_ns_y=["Lambda"]):

        '''
                given x - Mdisk and y - Lambda,
                find y = b0 + b1 * x

                Here x - dependent variables, outputs, or responses (predicted responses)
                     y - independent variables, inputs, or predictors.
                     b0 - intercept (shows the point where the estimated regression line crosses the 'y' axis)
                     b1 - determines the slope of the estimated regression line.

                y1 - f(x1) = y1 - b0 - b1*x1 for i = 1... n
                Here y1 -- residuals

                Data fromat
                y = [ 5 20 14 32 22 38]
                x = [[ 5]
                     [15]
                     [25]
                     [35]
                     [45]
                     [55]]


                :return:
                '''
        # x = np.array(self.dataframe["Mdisk3D"], dtype=float).reshape((-1, 1)) # dependent variables, outputs, or responses.
        # y = np.array(self.dataframe["Lambda"], dtype=float) # independent variables, inputs, or predictors.
        #
        # model = LinearRegression() # fits the model
        #
        # model.fit(x, y)
        #
        # r_sq = model.score(x, y)
        #
        # print('coefficient of determination:', r_sq)
        #
        # print('intercept:', model.intercept_) # which represents the coefficient, b[0]
        # print('slope:', model.coef_) # which represents b[1]
        #
        # y_pred = model.predict(x)
        # print('predicted response:', y_pred)

        # x = np.array(self.dataframe[v_n_x], dtype=float).reshape((-1, 1))
        n_fit_vals = 1
        if len(v_ns_x) == 1:
            x = np.array(self.dataframe[v_ns_x[0]], dtype=float).reshape((-1,1))
            fit_vals = np.array(vals).reshape((-1, 1))
            #print("---", fit_vals)
        else:
            # x = []
            # for i, v_n_x in enumerate(v_ns_x):
            #     x_ = np.array(self.dataframe[v_n_x], dtype=float)
            #     x.append(x_.T)
            # x = np.hstack((x)).T
            # fit_vals = vals

            #
            x = []
            fit_vals = vals
            for i, v_n_x in enumerate(v_ns_x):
                x_ = np.array(self.dataframe[v_n_x], dtype=float)
                x.append(x_)
                #fit_vals.append(vals[i])
                #n_fit_vals = len(vals[i])
                # print(len(x_))
            x = np.reshape(np.array(x), (len(v_ns_x), len(x_))).T
            #fit_vals = np.reshape(np.array(fit_vals), (len(v_ns_x), n_fit_vals))
            # print(x)
            # exit(1)
        y = np.array(self.dataframe[v_ns_y[0]], dtype=float)
        # if len(v_ns_y) == 1:
        #     y = np.array(self.dataframe[v_ns_y[0]], dtype=float)
        # else:
        #     y = []
        #     for v_n_y in v_ns_y:
        #         y_ = np.array(self.dataframe[v_n_y], dtype=float)
        #         y.append(y_)
        #     y = np.reshape(np.array(y), (len(v_ns_y), len(y_)))
        #     print(y)
        #     exit(1)

        print("------ Polynomial regression2 xs:{} ys:{} deg:{} ------ ".format(v_ns_x, v_ns_y, degree))
        transformer = PolynomialFeatures(degree=degree, include_bias=False)
        transformer.fit(x)
        x_ = transformer.transform(x)
        model = LinearRegression().fit(x_, y)
        r_sq = model.score(x_, y)
        #print(x_.shape)
        #print(model.predict(x_))
        std = stats.tstd(y)

        print('coefficient of determination R2: {}'.format(r_sq))
        print('intercept b0: {}'.format(model.intercept_))
        print('coefficients bi: {}'.format(model.coef_))

        #assert len(vals) == len(v_ns_x)
        #print(x)
        #print(x_)

        x2 = transformer.transform(fit_vals)
        y_pred1 = model.predict(x2)
        #y_pred2 = model.predict(fit_vals[1].reshape(1, -1))

        return y_pred1

class Fit_Data:
    """
    Fit the data with least square  and with polynomial regrression
    """

    def __init__(self, datasets, v_n="Mej_tot-geo", chi_method="scipy", error_method="1std"):

        self.fit_v_n = v_n
        self.chi_method = chi_method
        self.error_method = error_method

        v_ns = ["M1", "M2", "C1", "C2", "Mb1", "Mb2", "Lambda", "q", self.fit_v_n]
        #v_ns = [self.fit_v_n]
        v_ns_err = [self.fit_v_n]

        dataframe = create_combine_dataframe2(datasets, v_ns, v_ns_err, {})
        dataframe = dataframe[np.isfinite(dataframe[self.fit_v_n])]
        dataframe = dataframe[dataframe["Lambda"] <= 2000]
        self.dataframe = dataframe
        # dataframe = dataframe[dataframe["q"] < 1.29]
        # dataframe = dataframe[(dataframe["C2"] > 0.136) & (dataframe["C2"] < 0.218)]
        print(np.array(np.isfinite(dataframe[self.fit_v_n])))
        # if self.fit_v_n == "Mej_tot-geo":
        #     dataframe[self.fit_v_n] = dataframe[self.fit_v_n] * 1.e3

        print(" --- {} ---- ".format(self.fit_v_n))

        print(dataframe[self.fit_v_n].describe(percentiles=[0.8, 0.9, 0.95]))

        self.std = dataframe[self.fit_v_n].std()
        # self.mean = dataframe[self.fit_v_n].mean()
        n = len(np.array(dataframe[self.fit_v_n]))
        #
        # if chi_method == "scipy":
        #     print("[fittig mean] Using scipy norm with ddof=0")
        #     chi2, p = stats.chisquare(np.array(dataframe[self.fit_v_n]),
        #                              np.full(self.num, self.mean),
        #                              ddof=1)
        # else:
        #     chi2 = np.sum(((np.array(dataframe[self.fit_v_n]) - self.mean) ** 2) / self.std ** 2)

        y_vals = np.array(dataframe[self.fit_v_n], dtype=float)
        # if self.error_method == "1std": y_errs = float(self.std)
        # else: raise NameError("Unknown error_method: {}".format(error_method))
        y_errs = self.get_err(y_vals)
        mean = np.mean(y_vals)

        z = (y_vals - mean) / y_errs
        chi2 = np.sum(z ** 2.)

        chi2dof = chi2 / float(n - 1) # 1 -- ddof for a mean
        sigma = np.sqrt(2. / float(n - 1))
        nsig = (chi2dof - 1) / sigma

        '''  -------------------------  '''

        self.chi2 = chi2
        self.chi2dof = chi2dof
        self.num = n
        self.mean = mean

        print("-----------------------------------------------")
        print("\t num:     {:d} in the sample".format(self.num))
        print("\t std:     {}".format(self.std))
        print("\t mean:    {}".format(self.mean))
        print("\t chi2:    {} [{}]".format(self.chi2, self.chi_method))
        print("\t cho2dof: {}".format(self.chi2dof))
        print("\t nsig:    {} sigma's".format(nsig))
        print("-----------------------------------------------")

        #self.dataframe = dataframe
        #dataframe.to_csv("/data01/numrel/vsevolod.nedora/tmp/dataset.csv")


    def get_err(self, vals):
        res = np.zeros(0,)
        if self.error_method == "std":
            res =  np.std(vals)
        elif self.error_method == "2std":
            res = 2. * np.std(vals)
        elif self.error_method == "arr":
            res = self.dataframe["err_" + self.fit_v_n]
        else:
            raise NameError("no err method: {}".format(self.error_method))

        # if self.fit_v_n == "Mej_tot-geo": res = res * 1e3
        return res

    def get_chi2(self, y_vals, y_expets, y_errs):
        assert len(y_vals) == len(y_expets)
        z = (y_vals - y_expets) / y_errs
        chi2 = np.sum(z ** 2.)
        return chi2

    def get_score(self, y_true, y_pred):
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

        #y_true = y_true[~np.isnan(y_true)]
        #y_pred = y_pred[~np.isnan(y_pred)]

        y_true = y_true[np.isfinite(y_true)]
        y_pred = y_pred[np.isfinite(y_pred)]
        # print("--------")
        # print(y_true)
        # print(y_pred)
        # print("--------")

        assert len(y_true) == len(y_pred)
        u = np.sum((y_true - y_pred)**2.)
        v = np.sum((y_true - np.mean(y_true))**2.)
        res = (1.-u/v)
        #print(res)
        # assert res >= 0
        # assert res <= 1

        #if res < 0.: return np.nan # neg
        if res > 1.: return np.nan

        return res

    def get_ch2dof(self, chi2, n, k):
        """
        :param chi2: chi squared
        :param n: number of elements in a sample
        :param k: n of independedn parameters (1 -- mean, 2 -- poly1 fit, etc)
        :return:
        """
        return chi2 / (n - k)

    def get_ch2dof_mean(self):
        y_vals = np.array(self.dataframe[self.fit_v_n], dtype=float)
        # if self.fit_v_n == "Mej_tot-geo": y_vals = y_vals * 1e3
        mean = float(np.mean(y_vals))
        y_errs = self.get_err(y_vals)
        chi2 = self.get_chi2(y_vals, np.full(len(y_vals), mean), y_errs)
        n = len(np.array(self.dataframe[self.fit_v_n]))
        chi2dof = self.get_ch2dof(chi2, n, 1) # k=1 -- ddof for a mean
        return chi2dof

    def get_stats(self, v_ns=["n", "mean", "std", "80", "90", "95", "chi2", "chi2dof"]):

        res = []
        for v_n in v_ns:
            if v_n == "n":
                res.append(self.num)
            if v_n == "mean":
                res.append(self.mean)
            if v_n == "std":
                res.append(self.std)
            if v_n == "90" or v_n == "80" or v_n == "95":
                dic = self.dataframe[self.fit_v_n].describe(percentiles=[0.8, 0.9, 0.95])
                val = float(dic[v_n+"%"])
                res.append(val)
            if v_n == "chi2":
                res.append(self.chi2)
            if v_n == "chi2dof":
                res.append(self.chi2dof)

        return res

    def fitting_functions(self, x, v, name="Radice+2018"):

        if self.fit_v_n == "Mej_tot-geo":
            if name == "Radice+2018":
                return Fitting_Functions.mej_dietrich16(x, v)
            elif name == "Kruger+2020":
                return Fitting_Functions.mej_kruger20(x, v)
            else:
                raise NameError("no fitting function for: {} ".format(name))

        elif self.fit_v_n == "vel_inf_ave-geo":
            if name == "Radice+2018":
                return Fitting_Functions.vej_dietrich16(x, v)
            else:
                raise NameError("no fitting function for: {} ".format(name))

        elif self.fit_v_n == "Ye_ave-geo":
            return Fitting_Functions.yeej_like_vej(x, v)

        elif self.fit_v_n == "Mdisk3D":
            if name == "Radice+2018":
                return Fitting_Functions.mdisk_radice18(x, v)

            if name == "Kruger+2020":
                return Fitting_Functions.mdisk_kruger20(x, v)

            if name == "flat":
                return np.full(len(v["Lambda"]), 0.126323)

            else:
                raise NameError("no fitting function for: {} ".format(name))

        else:
            raise NameError("No fit. funct. for v_n:{}".format(self.fit_v_n))

    def coefficients(self, name="Radice+2018"):

        if self.fit_v_n == "Mej_tot-geo":
            if name == "Radice+2018":
                # Radice:2018pdn
                return Fitting_Coefficients.mej_radice18()
            elif name == "Kruger+2020":
                # Radice:2018pdn
                return Fitting_Coefficients.mej_kruger20()
            elif name == "all":
                # {Radice:2018pdn,Dietrich:2015iva,Dietrich:2016lyp,Kiuchi:2019lls,Vincent:2019kor}
                  return Fitting_Coefficients.mej_all()
            else:
                raise NameError(" coeff name: {} is not recognized v_n:{}".format(name, self.fit_v_n))
        elif self.fit_v_n == "vel_inf_ave-geo":
            if name == "Radice+2018":
                # Radice:2018pdn
                return Fitting_Coefficients.vej_radice()
            elif name == "all":
                # Dietrich:2016lyp,Dietrich:2015iva,Radice:2018pdn,Vincent:2019kor
                return Fitting_Coefficients.vej_all()
            else:
                raise NameError("no fitting function for: {} v_n:{}".format(name, self.fit_v_n))
        elif self.fit_v_n == "Ye_ave-geo":
            if name == "us":
                a = 0.139637775679
                b = 0.33996686385
                c = -3.70301958353
                return np.array((a, b, c))
            elif name == "all":
                # Radice:2018pdn,Vincent:2019kor
                return Fitting_Coefficients.yeej_all()
            else:
                raise NameError("no fitting coeffs for: {} v_n:{}".format(name, self.fit_v_n))
        elif self.fit_v_n == "Mdisk3D":
            if name == "Radice+2018":
                # Radice:2018pdn + us
                return Fitting_Coefficients.mdisk_radice18()
            elif name == "Kruger+2020":
                return Fitting_Coefficients.mdisk_kruger20()
            elif name == "all":
                # Radice:2018pdn,Dietrich:2015iva,Dietrich:2016lyp,Kiuchi:2019lls,Vincent:2019kor,Dietrich:2015iva,Dietrich:2016lyp,Kiuchi:2019lls,Vincent:2019kor
                return Fitting_Coefficients.mdisk_all()
            elif name == "flat":
                return np.array((0, 0, 0, 0))
            else:
                raise NameError(" coeff name: {} is not recognized".format(name))
        else:
            raise NameError("No coeffs found for v_n: {}".format(self.fit_v_n))

    def residuals(self, x, data, ffname ="Radice+2018", rs_name="Radice+2018"):

        if self.fit_v_n == "Mej_tot-geo":
            xi = self.fitting_functions(x, data, ffname)
            return 1.e-3 * (xi - 1.e3 * data[self.fit_v_n])

        xi = self.fitting_functions(x, data, ffname)
        return (xi - data[self.fit_v_n])

        # if self.fit_v_n == "Mej_tot-geo":
        #     if name == "Radice+2018":
        #         xi = self.fitting_functions(x, data, name)
        #         return 1.e-3*(xi - 1.e3*data[self.fit_v_n])
        #         #return ((data[self.fit_v_n] - xi) ** 2)
        #         # return ((1.e-3*(1.e3*data[] - xi)) ** 2)
        #     else:
        #         raise NameError("no residuals for name: {}".format(name))
        # else:
        #     if name == "Radice+2018":
        #         xi = self.fitting_functions(x, data, name)
        #         return (xi - data[self.fit_v_n])
        #         # return ((data[self.fit_v_n] - xi) ** 2)
        #     else:
        #         raise NameError("no residuals for name: {}".format(name))

    def fit2(self, ff_name, cf_name, rs_name):

        print("ff_name:{} cf_name:{} rs_name:{}".format(ff_name, cf_name, rs_name))

        y_vals = np.array(self.dataframe[self.fit_v_n])
        print("y_vals: {}".format(y_vals))
        x0 = self.coefficients(cf_name)
        print("coeffs: {}".format(x0))
        xi = self.fitting_functions(x0, self.dataframe, ff_name)
        print("y inferred: {}".format(xi))
        ### if self.fit_v_n == "Mej_tot-geo": xi = xi / 1.e3

        # if self.chi_method == "scipy":
        #     chi2, p = stats.chisquare(np.array(self.dataframe[self.fit_v_n]), xi, ddof=len(x0))
        # else:
        #     chi2 = np.sum((self.dataframe[self.fit_v_n] - xi) ** 2 / stats.tstd(xi)**2)
        # else: chi2 = np.sum((xi - self.dataframe[self.fit_v_n]) ** 2)
        chi2 = self.get_chi2(y_vals, xi, self.get_err(y_vals))
        print("chi2 original: {}".format(chi2))

        res = opt.least_squares(self.residuals, x0, args=(self.dataframe, ff_name, rs_name)) # res.x -- new coeffs
        xi = self.fitting_functions(res.x, self.dataframe, ff_name)
        ### if self.fit_v_n == "Mej_tot-geo": xi = xi / 1.e3
        # if self.chi_method == "scipy":
        #     chi2, p = stats.chisquare(np.array(self.dataframe[self.fit_v_n]), xi, ddof=len(x0))
        # else:
        #     chi2 = np.sum(((self.dataframe[self.fit_v_n] - xi) ** 2) / stats.tstd(xi)**2)
        chi2 = self.get_chi2(y_vals, xi, self.get_err(y_vals))
        chi2dof = self.get_ch2dof(chi2, len(y_vals), len(x0))

        print("chi2    fit: {}".format(chi2))
        print("chi2dof fit: {}".format(chi2dof))

        print("Fit coefficients:")
        for i in range(len(x0)):
            print("  coeff[{}] = {}".format(i, res.x[i]))

        #print(y_vals, xi);exit(1)
        # if self.fit_v_n == "Mej_tot-geo": y_vals = y_vals * 1e3
        score = self.get_score(y_vals, xi) # R^2 coefficient

        return res.x, chi2, chi2dof, score

    def linear_regression(self, degree=1, v_n_x = "Lambda", v_n_y="Mej_tot-geo"):
        '''
        given x - Mdisk and y - Lambda,
        find y = b0 + b1 * x

        Here x - dependent variables, outputs, or responses (predicted responses)
             y - independent variables, inputs, or predictors.
             b0 - intercept (shows the point where the estimated regression line crosses the 'y' axis)
             b1 - determines the slope of the estimated regression line.

        y1 - f(x1) = y1 - b0 - b1*x1 for i = 1... n
        Here y1 -- residuals

        Data fromat
        y = [ 5 20 14 32 22 38]
        x = [[ 5]
             [15]
             [25]
             [35]
             [45]
             [55]]


        :return:
        '''
        # x = np.array(self.dataframe["Mdisk3D"], dtype=float).reshape((-1, 1)) # dependent variables, outputs, or responses.
        # y = np.array(self.dataframe["Lambda"], dtype=float) # independent variables, inputs, or predictors.
        #
        # model = LinearRegression() # fits the model
        #
        # model.fit(x, y)
        #
        # r_sq = model.score(x, y)
        #
        # print('coefficient of determination:', r_sq)
        #
        # print('intercept:', model.intercept_) # which represents the coefficient, b[0]
        # print('slope:', model.coef_) # which represents b[1]
        #
        # y_pred = model.predict(x)
        # print('predicted response:', y_pred)

        x = np.array(self.dataframe[v_n_x], dtype=float).reshape((-1, 1))
        # y = np.array(self.dataframe[v_n_y], dtype=float)
        # if v_n_y == "Mej_tot-geo":
        #     y = np.array(self.dataframe[v_n_y], dtype=float) * 1.e3
        # else:
        # print(v_n_x, v_n_y); exit(1)
        y = np.array(self.dataframe[v_n_y], dtype=float)
        errs = self.get_err(y)
        if v_n_y == "Mej_tot-geo":
            y = 1e3 * y
            errs = 1e3 * errs

        # # new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
        # # print('intercept:', new_model.intercept_)
        print("------ Polynomial regression x:{} y:{} deg:{} ------ ".format(v_n_x, v_n_y, degree))
        transformer = PolynomialFeatures(degree=degree, include_bias=False)
        transformer.fit(x)
        x_ = transformer.transform(x)
        model = LinearRegression().fit(x_, y)
        r_sq = model.score(x_, y)

        print('coefficient of determination R2: {}'.format(r_sq))
        print('intercept b0: {}'.format(model.intercept_))
        print('coefficients bi: {}'.format(model.coef_))

        y_pred = model.predict(x_)
        chi2 = self.get_chi2(y, y_pred, errs)
        chi2dof = self.get_ch2dof(chi2, self.num, degree+1)
        #print('predicted response:', y_pred)
        # if self.chi_method == "scipy":
        #     print("[poly regres] deg:{} using scipy for chi ddof:{}".format(degree, degree+1))
        #     chi2, p = stats.chisquare(y, y_pred, ddof=degree+1)
        # else:
        #     chi2 = np.sum(((y - y_pred)**2) / std**2)
        #else: chi2 = np.sum((y - y_pred)**2)
        print('chi2: {}'.format(chi2))

        print("------------------------------------------------------")

        coeffs = np.zeros(len(model.coef_)+1)
        coeffs[0] = model.intercept_
        coeffs[1:] = model.coef_

        # return(model.intercept_, model.coef_, chi2, chi2dof, r_sq)
        return(coeffs, chi2, chi2dof, r_sq)

    def linear_regression2(self, degree=1, v_ns_x = ["Mej_tot-geo"], v_ns_y=["Lambda"]):
        '''
        given x - Mdisk and y - Lambda,
        find y = b0 + b1 * x

        Here x - dependent variables, outputs, or responses (predicted responses)
             y - independent variables, inputs, or predictors.
             b0 - intercept (shows the point where the estimated regression line crosses the 'y' axis)
             b1 - determines the slope of the estimated regression line.

        y1 - f(x1) = y1 - b0 - b1*x1 for i = 1... n
        Here y1 -- residuals

        Data fromat
        y = [ 5 20 14 32 22 38]
        x = [[ 5]
             [15]
             [25]
             [35]
             [45]
             [55]]


        :return:
        '''
        # x = np.array(self.dataframe["Mdisk3D"], dtype=float).reshape((-1, 1)) # dependent variables, outputs, or responses.
        # y = np.array(self.dataframe["Lambda"], dtype=float) # independent variables, inputs, or predictors.
        #
        # model = LinearRegression() # fits the model
        #
        # model.fit(x, y)
        #
        # r_sq = model.score(x, y)
        #
        # print('coefficient of determination:', r_sq)
        #
        # print('intercept:', model.intercept_) # which represents the coefficient, b[0]
        # print('slope:', model.coef_) # which represents b[1]
        #
        # y_pred = model.predict(x)
        # print('predicted response:', y_pred)

        #x = np.array(self.dataframe[v_n_x], dtype=float).reshape((-1, 1))
        if len(v_ns_x) == 1:
            x = np.array(self.dataframe[v_ns_x[0]], dtype=float)
        else:
            x = []
            for v_n_x in v_ns_x:
                x_ = np.array(self.dataframe[v_n_x], dtype=float)
                x.append(x_)
                #print(len(x_))
            x = np.reshape(np.array(x), (len(v_ns_x), len(x_))).T
            #print(x)
            #exit(1)
        # if v_ns_y[0] == "Mej_tot-geo":
        #    y = np.array(self.dataframe[v_ns_y[0]], dtype=float) * 1e3
        # else:
        y = np.array(self.dataframe[v_ns_y[0]], dtype=float)
        errs = self.get_err(y)
        if v_ns_y[0] == "Mej_tot-geo":
            y = 1e3 * y
            errs = 1e3 * errs

        print("------ Polynomial regression2 xs:{} ys:{} deg:{} ------ ".format(v_ns_x, v_ns_y, degree))
        transformer = PolynomialFeatures(degree=degree, include_bias=False)
        transformer.fit(x)
        x_ = transformer.transform(x)
        model = LinearRegression().fit(x_, y)
        r_sq = model.score(x_, y)
        y_pred = model.predict(x_)


        print('coefficient of determination R2: {}'.format(r_sq))
        print('intercept b0: {}'.format(model.intercept_))
        print('coefficients bi: {}'.format(model.coef_))

        chi2 = self.get_chi2(y, y_pred, errs)
        chi2dof = self.get_ch2dof(chi2, self.num, degree+1)
        #print('predicted response:', y_pred)
        # if self.chi_method == "scipy":
        #     print("[poly regres] deg:{} using scipy for chi ddof:{}".format(degree, degree+1))
        #     chi2, p = stats.chisquare(y, y_pred, ddof=degree+1)
        # else:
        #     chi2 = np.sum(((y - y_pred)**2) / std**2)
        #else: chi2 = np.sum((y - y_pred)**2)
        print('chi2: {}'.format(chi2))

        print("------------------------------------------------------")

        coeffs = np.zeros(len(model.coef_)+1)
        coeffs[0] = model.intercept_
        coeffs[1:] = model.coef_

        # return(model.intercept_, model.coef_, chi2, chi2dof, r_sq)
        return(coeffs, chi2, chi2dof, r_sq)

    #

    def mean_predict(self):
        """

        :param x_values:
        :param err:
        :return: (mean, mean-std, mean+std)
        """

        y_vals = np.array(self.dataframe[self.fit_v_n], dtype=float)
        y_errs = self.get_err(y_vals)
        mean = np.mean(y_vals)
        std = stats.tstd(y_vals)
        if std > mean:
            return mean, std, mean
        else:
            return mean, std, std

    def fit2_predict(self, x_values, ff_name, cf_name, rs_name):
        """

        :param ff_name:
        :param cf_name:
        :param rs_name:
        :param x_values: [x_val, x_lower, x_upper] or just [x_val]
        :return: (x, chi2, chi2dof, score, xi)
        """
        #assert len(x_values) == 1 or len(x_values) == 3
        x, chi2, chi2dof, score = self.fit2(ff_name, cf_name, rs_name) # x - coefficients
        xi = self.fitting_functions(x, x_values, ff_name) # get fitted values
        if self.fit_v_n == "Mej_tot-geo": xi = xi / 1.e3 # for ejecta mass fitting functions is for 1e3
        return xi

    def linear_regression_predict(self, vals, degree=1, v_n_x = "Mej_tot-geo", v_n_y="Lambda"):

        #assert len(vals) == 1 or len(vals) == 3

        x = np.array(self.dataframe[v_n_x], dtype=float).reshape((-1, 1))
        y = np.array(self.dataframe[v_n_y], dtype=float)
        # # new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
        # # print('intercept:', new_model.intercept_)
        print("------ Polynomial regression x:{} y:{} deg:{} ------ ".format(v_n_x, v_n_y, degree))
        transformer = PolynomialFeatures(degree=degree, include_bias=False)
        transformer.fit(x)
        x_ = transformer.transform(x)
        model = LinearRegression().fit(x_, y)
        r_sq = model.score(x_, y)

        # std = stats.tstd(y)

        print('coefficient of determination R2: {}'.format(r_sq))
        print('intercept b0: {}'.format(model.intercept_))
        print('coefficients bi: {}'.format(model.coef_))

        # y_pred = model.predict(x_)
        # chi2 = self.get_chi2(y, y_pred, self.get_err(y))
        # chi2dof = self.get_ch2dof(chi2, self.num, degree + 1)

        y_pred = model.predict(np.array(vals).reshape(-1,1))

        return y_pred

    def linear_regression2_predict(self, vals, degree=1, v_ns_x = ["Mej_tot-geo"], v_ns_y=["Lambda"]):

        '''
                given x - Mdisk and y - Lambda,
                find y = b0 + b1 * x

                Here x - dependent variables, outputs, or responses (predicted responses)
                     y - independent variables, inputs, or predictors.
                     b0 - intercept (shows the point where the estimated regression line crosses the 'y' axis)
                     b1 - determines the slope of the estimated regression line.

                y1 - f(x1) = y1 - b0 - b1*x1 for i = 1... n
                Here y1 -- residuals

                Data fromat
                y = [ 5 20 14 32 22 38]
                x = [[ 5]
                     [15]
                     [25]
                     [35]
                     [45]
                     [55]]


                :return:
                '''
        # x = np.array(self.dataframe["Mdisk3D"], dtype=float).reshape((-1, 1)) # dependent variables, outputs, or responses.
        # y = np.array(self.dataframe["Lambda"], dtype=float) # independent variables, inputs, or predictors.
        #
        # model = LinearRegression() # fits the model
        #
        # model.fit(x, y)
        #
        # r_sq = model.score(x, y)
        #
        # print('coefficient of determination:', r_sq)
        #
        # print('intercept:', model.intercept_) # which represents the coefficient, b[0]
        # print('slope:', model.coef_) # which represents b[1]
        #
        # y_pred = model.predict(x)
        # print('predicted response:', y_pred)

        # x = np.array(self.dataframe[v_n_x], dtype=float).reshape((-1, 1))
        n_fit_vals = 1
        if len(v_ns_x) == 1:
            x = np.array(self.dataframe[v_ns_x[0]], dtype=float).reshape((-1,1))
            fit_vals = np.array(vals).reshape((-1, 1))
            #print("---", fit_vals)
        else:
            # x = []
            # for i, v_n_x in enumerate(v_ns_x):
            #     x_ = np.array(self.dataframe[v_n_x], dtype=float)
            #     x.append(x_.T)
            # x = np.hstack((x)).T
            # fit_vals = vals

            #
            x = []
            fit_vals = vals
            for i, v_n_x in enumerate(v_ns_x):
                x_ = np.array(self.dataframe[v_n_x], dtype=float)
                x.append(x_)
                #fit_vals.append(vals[i])
                #n_fit_vals = len(vals[i])
                # print(len(x_))
            x = np.reshape(np.array(x), (len(v_ns_x), len(x_))).T
            #fit_vals = np.reshape(np.array(fit_vals), (len(v_ns_x), n_fit_vals))
            # print(x)
            # exit(1)
        y = np.array(self.dataframe[v_ns_y[0]], dtype=float)
        # if len(v_ns_y) == 1:
        #     y = np.array(self.dataframe[v_ns_y[0]], dtype=float)
        # else:
        #     y = []
        #     for v_n_y in v_ns_y:
        #         y_ = np.array(self.dataframe[v_n_y], dtype=float)
        #         y.append(y_)
        #     y = np.reshape(np.array(y), (len(v_ns_y), len(y_)))
        #     print(y)
        #     exit(1)

        print("------ Polynomial regression2 xs:{} ys:{} deg:{} ------ ".format(v_ns_x, v_ns_y, degree))
        transformer = PolynomialFeatures(degree=degree, include_bias=False)
        transformer.fit(x)
        x_ = transformer.transform(x)
        model = LinearRegression().fit(x_, y)
        r_sq = model.score(x_, y)
        #print(x_.shape)
        #print(model.predict(x_))
        std = stats.tstd(y)

        print('coefficient of determination R2: {}'.format(r_sq))
        print('intercept b0: {}'.format(model.intercept_))
        print('coefficients bi: {}'.format(model.coef_))

        #assert len(vals) == len(v_ns_x)
        #print(x)
        #print(x_)

        x2 = transformer.transform(fit_vals)
        y_pred1 = model.predict(x2)
        #y_pred2 = model.predict(fit_vals[1].reshape(1, -1))

        return y_pred1

""" ========================| TASKS |====================== """

def task_print_stats(v_n = "Mej_tot-geo", v_ns = ["n", "mean", "std", "80", "90", "95", "chi2", "chi2dof"], error_method = "arr"):

    # v_n = "Mej_tot-geo"
    scale_by = None#1.e2
    pre_names = ["datasets"]
    # v_ns = ["n", "mean", "std", "80", "90", "95", "chi2", "chi2dof"]
    v_labels = ["Datasets", r"$N$", r"$\mu$", r"$\sigma$", r"$80\%$", r"$90\%$", r"$95\%$", "$\chi^2$", r"$\chi^2 _{\text{dof}}$"]
    coeff_fmt = ".3f"
    coeff_small_fmt = ".3e"
    # error_method = "arr"

    row_labels, vals = [], []
    mdisk_datasets = OrderedDict()
    for i in range(10):

        if i >= 0: mdisk_datasets['Reference set']      = {"models": md.groups, "data": md, "fit": True}
        if i >= 1: mdisk_datasets["Vincent:2019kor"]    = {"models": vi.simulations, "data": vi, "fit": True}
        if i >= 2: mdisk_datasets["Radice:2018pdn[M0]"] = {"models": rd.simulations[rd.with_m0], "data": rd, "fit": True}
        if i >= 3: mdisk_datasets["Radice:2018pdn"]     = {"models": rd.simulations[rd.fiducial], "data": rd, "fit": True}
        if i >= 4: mdisk_datasets["Lehner:2016lxy"]     = {"models": lh.simulations, "data": lh, "fit": True}
        if i >= 5: mdisk_datasets["Kiuchi:2019lls"] =     {"models": ki.simulations[ki.mask_for_with_tov_data], "data": ki, "fit": True}
        if i >= 6: mdisk_datasets["Dietrich:2016lyp"]   = {"models": di16.simulations[di16.mask_for_with_sr],"data": di16, "fit": True}
        if i >= 7: mdisk_datasets["Dietrich:2015iva"]   = {"models": di15.simulations[di15.mask_for_with_sr], "data": di15, "fit": True}
        if i == 8: break

        if v_n == "vel_inf_ave-geo":
            if "Kiuchi:2019lls" in mdisk_datasets.keys():
                del mdisk_datasets["Kiuchi:2019lls"]
        elif v_n == "Ye_ave-geo":
            blacklist = ["Dietrich:2016lyp", "Dietrich:2015iva", "Kiuchi:2019lls", "Lehner:2016lxy"]
            for key in blacklist:
                if key in mdisk_datasets.keys():
                    del mdisk_datasets[key]
                    # print("~{}".format(key))
        elif v_n == "Mdisk3D":
            if "Lehner:2016lxy" in mdisk_datasets.keys():
                del mdisk_datasets["Lehner:2016lxy"]

        row_labels.append(mdisk_datasets.keys())
        print(row_labels[-1])

        df = Fit_Data(mdisk_datasets, v_n, error_method=error_method)
        i_vals = df.get_stats(v_ns)
        vals.append(i_vals)

    print("Data is collected")
    #
    cells = "c" * (len(pre_names) + len(v_ns))
    print("\n")
    print(r"\begin{table*}")
    print(r"\caption{I am your little table}")
    print(r"\begin{tabular}{l|" + cells + "}")

    line = ''
    for name, label in zip(pre_names + v_ns, v_labels):
        if name != v_ns[-1]:
            line = line + label + ' & '
        else:
            line = line + label + r' \\'
    # line[-2] = r"\\"
    print(line)

    for row_names, coeff in zip(row_labels, vals):

        #row_names = row_labels[i]
        row_name = row_names[-1]

        if row_names == row_labels[0]:
            row_name = row_names[-1]
        else:
            row_name = "\& " + "\cite{" + row_name + "} "

        # if row_names == row_labels[0]:
        #     row_name = ""
        #     for dsname in row_names:
        #         if not dsname == row_names[-1]: row_name = row_name + dsname + ','
        #         else: row_name = row_name + dsname + ''
        # else:
        #     row_name = 'with \cite{'
        #     for dsname in row_names:
        #         if dsname == "This work": pass
        #         else:
        #             if not dsname == row_names[-1]: row_name = row_name + dsname + ','
        #             else: row_name = row_name + dsname + '}'


        row = row_name + " & "
        for i_coeff in coeff:
            if i_coeff < 1.e-2:
                ifmt = coeff_small_fmt
            else:
                ifmt = coeff_fmt
            val = str(("%{}".format(ifmt) % float(i_coeff)))
            if i_coeff != coeff[-1]:
                row = row + val + " & "
            else:
                row = row + val + r" \\ "

        # val = str(("%{}".format(coeff_fmt) % float(chi2)))
        # row = row + val + r" \\ "

        print(row)
        #row[-2] = r" \\ "
    print(r"\end{tabular}")
    print(r"\end{table*}")

def task_table_linear_regresion(v_n_y="Mej_tot-geo", v_n_x=["Lambda"], degree=1,error_method="arr", fancy=True):

    # degree = 1
    # v_n_y = "Mej_tot-geo"
    # error_method = "arr"

    dataset_label = "Datasets"
    coefs_labels = [r"$b_0$", r"$b_1$", r"$b_2$", r"$b_3$", r"$b_4$", r"$b_5$"]
    coefs_fmt    = [".2e", ".2e", ".2e", ".2e", ".2e", ".2e"]
    other_labels = [r"$\chi^2_{\nu}$", r"$R^2$"]
    other_fmt =    [".1f", ".3f"]

    ''' ----------- '''

    all_coeffs = []
    all_pars = []

    row_labels, all_vals = [], []
    mdisk_datasets = OrderedDict()
    for i in range(0,8):
        if i >= 0: mdisk_datasets['Reference set']      = {"models": md.groups, "data": md, "fit": True}
        if i >= 1: mdisk_datasets["Vincent:2019kor"]    = {"models": vi.simulations[vi.with_mej], "data": vi, "fit": True}
        if i >= 2: mdisk_datasets["Radice:2018pdn[M0]"] = {"models": rd.simulations[rd.with_m0], "data": rd, "fit": True}
        if i >= 3: mdisk_datasets["Radice:2018pdn"]     = {"models": rd.simulations[rd.fiducial], "data": rd, "fit": True}
        if i >= 4: mdisk_datasets["Lehner:2016lxy"]     = {"models": lh.simulations, "data": lh, "fit": True}
        # if i >= 5: mdisk_datasets["Kiuchi:2019lls"] =     {"models": ki.simulations[ki.mask_for_with_tov_data], "data": ki, "fit": True}
        # if i >= 6: mdisk_datasets["Dietrich:2015iva"]   = {"models": di15.simulations[di15.mask_for_with_sr], "data": di15, "fit": True}
        # if i >= 7: mdisk_datasets["Dietrich:2016lyp"]   = {"models": di16.simulations[di16.mask_for_with_sr],"data": di16, "fit": True}
        if i == 8: break

        # print(md.groups[["q", "Mej_tot-geo"]]); exit(1)
        # print(di15.simulations[di15.mask_for_with_sr][["q","Mej"]]); exit(1)

        # cleaning
        if v_n_y == "vel_inf_ave-geo":
            if "Kiuchi:2019lls" in mdisk_datasets.keys():
                del mdisk_datasets["Kiuchi:2019lls"]
        elif v_n_y == "Ye_ave-geo":
            blacklist = ["Dietrich:2016lyp", "Dietrich:2015iva", "Kiuchi:2019lls", "Lehner:2016lxy"]
            for key in blacklist:
                if key in mdisk_datasets.keys():
                    del mdisk_datasets[key]
                    # print("~{}".format(key))
        elif v_n_y == "Mdisk3D":
            if "Lehner:2016lxy" in mdisk_datasets.keys():
                del mdisk_datasets["Lehner:2016lxy"]

        print(mdisk_datasets.keys())
        row_labels.append(mdisk_datasets.keys())
        #
        df = Fit_Data(mdisk_datasets, v_n_y, error_method=error_method)
        #
        if len(v_n_x) == 1:
            i_coeffs, i_chi, i_chi2dof, i_rs = df.linear_regression(degree=degree, v_n_x=v_n_x[0], v_n_y=v_n_y)
        else:
            i_coeffs, i_chi, i_chi2dof, i_rs = df.linear_regression2(degree=degree, v_ns_x=v_n_x, v_ns_y=[v_n_y])
        #
        all_coeffs.append(i_coeffs)
        all_pars.append([i_chi2dof, i_rs])# i_chi


    ''' table '''

    # label line
    label_line = dataset_label + ' '
    for name in coefs_labels[:len(all_coeffs[0])] + other_labels:
        if name != other_labels[-1]: label_line = label_line + name + ' & '
        else: label_line = label_line + name + r' \\'



    lines = []
    # collect table lines
    for i in range(len(row_labels)):
        # fiest element -- name of the dataset
        row_names = row_labels[i]
        row_name = row_names[-1]

        if i == 0:
            row_name = row_names[-1]
        else:
            row_name = "\& " + "\cite{" + row_name + "} "

        # if row_names == row_labels[0]:
        #     row_name = ""
        #     for dsname in row_names:
        #         if not dsname == row_names[-1]:
        #             row_name = row_name + dsname + ','
        #         else:
        #             row_name = row_name + dsname + ''
        # else:
        #     row_name = 'with \cite{'
        #     for dsname in row_names:
        #         if dsname == "This work":
        #             pass
        #         else:
        #             if not dsname == row_names[-1]:
        #                 row_name = row_name + dsname + ','
        #             else:
        #                 row_name = row_name + dsname + '}'

        row = row_name + " & "

        # add coefficients
        i_coeffs = all_coeffs[i]
        for coeff, fmt in zip(i_coeffs, coefs_fmt[:len(i_coeffs)]):
            val = __get_str_val(coeff, fmt, fancy)
            if coeff != i_coeffs[-1]:
                row = row + val + " & "
            else:
                if len(other_labels) == 0:
                    row = row + val + r" \\ "
                else:
                    row = row + val + " & "

        # add other values
        i_pars = all_pars[i]
        assert len(i_pars) == len(other_fmt)
        for par, fmt in zip(i_pars, other_fmt):
            val = __get_str_val(par, fmt, fancy)
            if par == i_pars[-1]:
                row = row + val + r" \\ "
            else:
                row = row + val + r" & "

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

def task_fitfunc_print_table(v_n = "Mej_tot-geo", error_method = "arr",
                             ff_name = "Radice+2018", cf_name = "Radice+2018", rs_name = "Radice+2018", fancy = True):

    # v_n = "Mej_tot-geo"
    # error_method = "arr"
    # ff_name = "Radice+2018"
    # cf_name = "Radice+2018"
    # rs_name = "Radice+2018"

    dataset_label = "Datasets"
    coefs_labels = [r"$\alpha$", r"$\beta$", r"$\delta$", r"$\gamma$", r"$\eta$", r"$\phi$"]
    # coefs_fmt = [".2e", ".2e", ".2e", ".2e", ".2e", ".2e"]
    coefs_fmt = [".3f", ".3f", ".3f", ".3f", ".3f", ".3f"]
    other_labels = [r"$\chi^2_{\nu}$", r"$R^2$"]
    other_fmt = [".1f", ".3f"]

    # -- coefficients

    all_coeffs = []
    all_pars = []

    row_labels, all_vals = [], []
    mdisk_datasets = OrderedDict()
    for i in range(10):
        if i == 0 :
            df = Fit_Data(mdisk_datasets, v_n=v_n, error_method=error_method)
            all_coeffs.append(df.coefficients(cf_name))
            all_pars.append([np.nan, np.nan]) # np.nan
            row_labels.append(["Prior"])
        if i >= 1: mdisk_datasets['Reference set']      = {"models": md.groups, "data": md, "fit": True}
        if i >= 2: mdisk_datasets["Vincent:2019kor"]    = {"models": vi.simulations, "data": vi, "fit": True}
        if i >= 3: mdisk_datasets["Radice:2018pdn[M0]"] = {"models": rd.simulations[rd.with_m0], "data": rd, "fit": True}
        if i >= 4: mdisk_datasets["Radice:2018pdn"]     = {"models": rd.simulations[rd.fiducial], "data": rd, "fit": True}
        if i >= 5: mdisk_datasets["Lehner:2016lxy"]     = {"models": lh.simulations, "data": lh, "fit": True}
        if i >= 6: mdisk_datasets["Kiuchi:2019lls"] =     {"models": ki.simulations[ki.mask_for_with_tov_data], "data": ki, "fit": True}
        if i >= 7: mdisk_datasets["Dietrich:2016lyp"]   = {"models": di16.simulations[di16.mask_for_with_sr],"data": di16, "fit": True}
        if i >= 8: mdisk_datasets["Dietrich:2015iva"]   = {"models": di15.simulations[di15.mask_for_with_sr], "data": di15, "fit": True}
        if i == 9: break

        if i >= 1:
            # cleaning
            if v_n == "vel_inf_ave-geo":
                if "Kiuchi:2019lls" in mdisk_datasets.keys():
                    del mdisk_datasets["Kiuchi:2019lls"]
            elif v_n == "Ye_ave-geo":
                blacklist = ["Dietrich:2016lyp", "Dietrich:2015iva", "Kiuchi:2019lls", "Lehner:2016lxy"]
                for key in blacklist:
                    if key in mdisk_datasets.keys():
                        del mdisk_datasets[key]
                        # print("~{}".format(key))
            elif v_n == "Mdisk3D":
                if "Lehner:2016lxy" in mdisk_datasets.keys():
                    del mdisk_datasets["Lehner:2016lxy"]

            print(mdisk_datasets.keys())
            row_labels.append(mdisk_datasets.keys())
            #
            df = Fit_Data(mdisk_datasets, v_n, error_method=error_method)
            #
            i_coeffs, chi2, chi2dof, r2 = df.fit2(ff_name, cf_name, rs_name)
            #
            all_coeffs.append(i_coeffs)
            all_pars.append([chi2dof, r2])# i_chi


    ''' table '''

    # label line
    label_line = dataset_label + ' '
    for name in coefs_labels[:len(all_coeffs[0])] + other_labels:
        if name == other_labels[-1]:
            label_line = label_line + name + r' \\ '
        else:
            label_line = label_line + name + r' &'

    lines = []
    # collect table lines
    for i in range(len(row_labels)):
        # fiest element -- name of the dataset
        row_names = row_labels[i]
        row_name = row_names[-1]

        if i == 0:
            row_name = row_names[-1]
        else:
            row_name = "\& " + "\cite{" + row_name + "} "

        row = row_name + " & "

        # add coefficients
        i_coeffs = all_coeffs[i]
        for coeff, fmt in zip(i_coeffs, coefs_fmt[:len(i_coeffs)]):
            val = __get_str_val(coeff, fmt, fancy)
            if coeff != i_coeffs[-1]:
                row = row + val + " & "
            else:
                if len(other_labels) == 0:
                    row = row + val + r" \\ "
                else:
                    row = row + val + " & "

        # add other values
        i_pars = all_pars[i]
        assert len(i_pars) == len(other_fmt)
        for par, fmt in zip(i_pars, other_fmt):
            val = __get_str_val(par, fmt, fancy)
            if par == i_pars[-1]:
                row = row + val + r" \\ "
            else:
                row = row + val + r" & "

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

""" ------------------------| COMMON |--------------------- """

def task_mej_chi2dofs():

    v_n_y = "Mej_tot-geo"
    v_ns_x = ["q", "Lambda"]
    degree = 2
    v_ns = ["datasets", "mean-chi2dof", "Radice-chi2dof", "Kruger-chi2dof", "poly2-chi2dof", "poly22-chi2dof"]
    fmts = [            ".1f", ".1f", ".1f", ".1f", ".1f"]
    #v_ns = ["Radice-chi2dof"]
    #v_ns_labels = ["datasets", r"$\chi^2 _{\text{dof}}$", r"$\chi^2 _{\text{dof}}$", r"$\chi^2 _{\text{dof}}$", r"$\chi^2 _{\text{dof}}$", r"$\chi^2 _{\text{dof}}$"]
    v_ns_labels = ["datasets", r"Mean", r"Eq.~\eqref{eq:fit_Mej}", r"Eq.~\eqref{eq:fit_Mej_Kruger}", r"$P_2(\tilde{\Lambda})$", r"$P_2(q,\tilde{\Lambda})$"]
    error_method = "arr"

    # ---
    row_labels, all_vals = [], []
    mdisk_datasets = OrderedDict()

    for i in range(9):
        if i >= 0: mdisk_datasets['Reference set']      = {"models": md.groups, "data": md, "fit": True}
        if i >= 1: mdisk_datasets["Vincent:2019kor"]    = {"models": vi.simulations, "data": vi, "fit": True}
        if i >= 2: mdisk_datasets["Radice:2018pdn[M0]"] = {"models": rd.simulations[rd.with_m0], "data": rd, "fit": True}
        if i >= 3: mdisk_datasets["Radice:2018pdn"]     = {"models": rd.simulations[rd.fiducial], "data": rd, "fit": True}
        if i >= 4: mdisk_datasets["Lehner:2016lxy"]     = {"models": lh.simulations, "data": lh, "fit": True}
        if i >= 5: mdisk_datasets["Kiuchi:2019lls"] =     {"models": ki.simulations[ki.mask_for_with_tov_data], "data": ki, "fit": True}
        if i >= 6: mdisk_datasets["Dietrich:2016lyp"]   = {"models": di16.simulations[di16.mask_for_with_sr],"data": di16, "fit": True}
        if i >= 7: mdisk_datasets["Dietrich:2015iva"]   = {"models": di15.simulations[di15.mask_for_with_sr], "data": di15, "fit": True}
        if i == 8: break
        print(mdisk_datasets.keys())
        row_labels.append(mdisk_datasets.keys())

        df = Fit_Data(mdisk_datasets, v_n_y, error_method=error_method)

        vals = []
        for v_n in v_ns:
            if v_n.__contains__("mean-"):
                if v_n.__contains__("chi2dof"):  vals.append(df.get_ch2dof_mean())
            if v_n.__contains__("Radice-"):
                print("\tTask: {}".format(v_n))
                i_coeffs, chi2, chi2dof, R2 = df.fit2("Radice+2018", "Radice+2018", "Radice+2018")
                # print(chi2dof); exit(1)
                if v_n.__contains__("chi2dof"):  vals.append(chi2dof)
            if v_n.__contains__("Kruger-"):
                print("\tTask: {}".format(v_n))
                i_coeffs, chi2, chi2dof, R2 = df.fit2("Kruger+2020", "Kruger+2020", "Kruger+2020")
                if v_n.__contains__("chi2dof"):  vals.append(chi2dof)
            if v_n.__contains__("poly2-"):
                print("\tTask: {}".format(v_n))
                b0, bi, chi2, chi2dof, r2 = df.linear_regression(degree=degree, v_n_x=v_ns_x[-1], v_n_y=v_n_y)
                if v_n.__contains__("chi2dof"):  vals.append(chi2dof)
            if v_n.__contains__("poly22-"):
                print("\tTask: {}".format(v_n))
                b0, bi, chi2, chi2dof, r2 = df.linear_regression2(degree=degree, v_ns_x=v_ns_x, v_ns_y=[v_n_y])
                if v_n.__contains__("chi2dof"):  vals.append(chi2dof)
        all_vals.append(vals)

    print("\t---<DataCollected>---")

    #
    # for label in row_labels:
    #     print(label)
    # exit(1)

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
        row_names = row_labels[i]
        if row_names == row_labels[0]:
            row_name = ""
            for dsname in row_names:
                if not dsname == row_names[-1]: row_name = row_name + dsname + ','
                else: row_name = row_name + dsname + ''
        else:
            row_name = '\& \cite{'
            for dsname in row_names:
                if dsname == "This work": pass
                else:
                    if not dsname == row_names[-1]: row_name = row_name + dsname + ','
                    else: row_name = row_name + dsname + '}'
        # DATA ITSELF
        vals = all_vals[i]
        row = row_name + " & "
        assert len(vals) == len(fmts)
        for val, fmt in zip(vals, fmts):
            if val != vals[-1]:
                val = __get_str_val(val, fmt)
                # if val < 1e-2:  val = str(("%{}".format(coeff_small_fmt) % float(val)))
                # else: val = __get_str_val()#str(("%{}".format(coeff_fmt) % float(val)))
                row = row + val + " & "
            else:
                val = __get_str_val(val, fmt)
                # if val < 1e-2:  val = str(("%{}".format(coeff_small_fmt) % float(val)))
                # else: val = str(("%{}".format(coeff_fmt) % float(val)))
                row = row + val + r" \\ "

        print(row)
        #row[-2] = r" \\ "


    print(r"\end{tabular}")
    print(r"\end{table}")

def Oldtask_mej_print_table_overall():


    pre_names = ["datasets"]
    coefs_names = [r"$\alpha$", r"$\beta$", r"$\gamma$", r"$\delta$", r"$n$"]
    coeff_fmt = ".3f"
    add_names = [r"$\chi^2$", r"$\chi^2 _{\text{dof}}$", "R^2"]
    coeffs = []
    chi2s = []
    chi2dofs = []
    row_labels = []
    r2s = []
    v_n = "Mej_tot-geo"
    error_method = "arr"

    # ### mdisk_datasets["bauswein"] = {"models": bs.simulations, "data": bs, "fit": True}
    # ### mdisk_datasets["hotokezaka"] = {"models": hz.simulations, "data": hz, "fit": True}
    # mdisk_datasets["dietrich15"] = {"models": di15.simulations[di15.mask_for_with_sr], "data": di15, "fit": True}
    # ### mdisk_datasets["sekiguchi15"] = {"models": se15.simulations[se15.mask_for_with_sr], "data": se15, "fit": True}
    # mdisk_datasets["dietrich16"] = {"models": di16.simulations[di16.mask_for_with_sr], "data": di16, "fit": True}
    # ### mdisk_datasets["sekiguchi16"] = {"models": se16.simulations[se16.mask_for_with_sr], "data": se16, "fit": True}
    # ### mdisk_datasets["lehner"] = {"err": lh.params.Mej_err, "label": r"Lehner+2016", "fit": True}
    # mdisk_datasets["radice"] =    {"models": rd.simulations[rd.fiducial], "data": rd, "fit": True}
    # mdisk_datasets["kiuchi"] =    {"models": ki.simulations, "data": ki, "fit": True}
    # mdisk_datasets["vincent"] =   {"models": vi.simulations, "data": vi,  "fit": True}
    #mdisk_datasets['our'] = {"models": md.groups, "data": md, "fit": True}

    mdisk_datasets = {}
    for i in range(8):
        if i == 0 :
            df = Fit_Data(mdisk_datasets, v_n="Mej_tot-geo", error_method=error_method)
            coeffs.append(df.coefficients("Radice+2018"))
            chi2s.append(np.nan)
            chi2dofs.append(np.nan)
            r2s.append(np.nan)
            row_labels.append(["Prior"])
        if i >= 1 : mdisk_datasets['This work'] = {"models": md.groups, "data": md, "fit": True}
        if i >= 2 : mdisk_datasets["Radice:2018pdn"] =    {"models": rd.simulations[rd.fiducial], "data": rd, "fit": True}
        if i >= 3 : mdisk_datasets["Kiuchi:2019lls"] =    {"models": ki.simulations[ki.mask_for_with_tov_data], "data": ki, "fit": True}
        if i >= 4 : mdisk_datasets["Vincent:2019kor"] =   {"models": vi.simulations, "data": vi,  "fit": True}
        if i >= 5 : mdisk_datasets["Dietrich:2016lyp"] =  {"models": di16.simulations[di16.mask_for_with_sr], "data": di16, "fit": True}
        if i >= 6 : mdisk_datasets["Dietrich:2015iva"] =  {"models": di15.simulations[di15.mask_for_with_sr], "data": di15, "fit": True}
        if i == 7: break

        if i >= 1:
            row_labels.append(mdisk_datasets.keys())
            print(row_labels[-1])

            df = Fit_Data(mdisk_datasets, v_n=v_n, error_method = error_method)
            i_coeffs, chi2, chi2dof, r2 = df.fit2("Radice+2018", "Radice+2018", "Radice+2018")
            coeffs.append(i_coeffs)
            chi2s.append(chi2)
            r2s.append(r2)
            chi2dofs.append(chi2dof)
        #row_labels.append(mdisk_datasets.keys())
    #
    print("Data is collected")
    #
    cells = "c" * (len(pre_names) + len(coefs_names) + len(add_names))
    #
    print("\n")
    print(r"\begin{table}")
    print(r"\caption{I am your little table}")
    print(r"\begin{tabular}{l|" + cells + "}")

    line = ''
    for name in pre_names + coefs_names + add_names:
        if name != add_names[-1]: line = line + name + ' & '
        else: line = line + name + r' \\'
    #line[-2] = r"\\"
    print(line)

    assert len(coeffs) == len(chi2s)
    for row_names, coeff, chi2, chi2dof, r2 in zip(row_labels, coeffs, chi2s, chi2dofs, r2s):

        if row_names == row_labels[0]:
            row_name = ""
            for dsname in row_names:
                if not dsname == row_names[-1]: row_name = row_name + dsname + ','
                else: row_name = row_name + dsname + ''
        else:
            row_name = 'with \cite{'
            for dsname in row_names:
                if dsname == "This work": pass
                else:
                    if not dsname == row_names[-1]: row_name = row_name + dsname + ','
                    else: row_name = row_name + dsname + '}'


        row = row_name + " & "
        for i_coeff in coeff:
            val = str(("%{}".format(coeff_fmt) % float(i_coeff)))
            row = row + val + " & "

        val = str(("%{}".format(coeff_fmt) % float(chi2)))
        row = row + val + r" & "
        val = str(("%{}".format(coeff_fmt) % float(chi2dof)))
        row = row + val + r" & "
        val = str(("%{}".format(coeff_fmt) % float(r2)))
        row = row + val + r" \\ "
        print(row)


        #row[-2] = r" \\ "


    print(r"\end{tabular}")
    print(r"\end{table}")

### vej
def task_vej_chi2dofs():

    v_n_y = "vel_inf_ave-geo"
    v_ns_x = ["q", "Lambda"]
    degree = 2
    v_ns = ["datasets", "mean-chi2dof", "Radice-chi2dof", "poly2-chi2dof", "poly22-chi2dof"]
    fmts = [            ".1f", ".1f", ".1f", ".1f"]
    #v_ns = ["Radice-chi2dof"]
    #v_ns_labels = ["datasets", r"$\chi^2 _{\text{dof}}$", r"$\chi^2 _{\text{dof}}$", r"$\chi^2 _{\text{dof}}$", r"$\chi^2 _{\text{dof}}$", r"$\chi^2 _{\text{dof}}$"]
    v_ns_labels = ["datasets", r"Mean", r"Eq.~\eqref{eq:fit_vej}", r"$P_2(\tilde{\Lambda})$", r"$P_2(q,\tilde{\Lambda})$"]
    error_method = "arr"

    # ---
    row_labels, all_vals = [], []
    mdisk_datasets = OrderedDict()

    for i in range(9):
        if i >= 0: mdisk_datasets['Reference set']      = {"models": md.groups, "data": md, "fit": True}
        if i >= 1: mdisk_datasets["Vincent:2019kor"]    = {"models": vi.simulations, "data": vi, "fit": True}
        if i >= 2: mdisk_datasets["Radice:2018pdn[M0]"] = {"models": rd.simulations[rd.with_m0], "data": rd, "fit": True}
        if i >= 3: mdisk_datasets["Radice:2018pdn"]     = {"models": rd.simulations[rd.fiducial], "data": rd, "fit": True}
        if i >= 4: mdisk_datasets["Lehner:2016lxy"]     = {"models": lh.simulations, "data": lh, "fit": True}
        # if i >= 5: mdisk_datasets["Kiuchi:2019lls"] =     {"models": ki.simulations[ki.mask_for_with_tov_data], "data": ki, "fit": True}
        if i >= 5: mdisk_datasets["Dietrich:2016lyp"]   = {"models": di16.simulations[di16.mask_for_with_sr],"data": di16, "fit": True}
        if i >= 6: mdisk_datasets["Dietrich:2015iva"]   = {"models": di15.simulations[di15.mask_for_with_sr], "data": di15, "fit": True}
        if i == 7: break
        print(mdisk_datasets.keys())
        row_labels.append(mdisk_datasets.keys())

        df = Fit_Data(mdisk_datasets, v_n_y, error_method=error_method)

        vals = []
        for v_n in v_ns:
            if v_n.__contains__("mean-"):
                if v_n.__contains__("chi2dof"):  vals.append(df.get_ch2dof_mean())
            if v_n.__contains__("Radice-"):
                print("\tTask: {}".format(v_n))
                i_coeffs, chi2, chi2dof, R2 = df.fit2("Radice+2018", "Radice+2018", "Radice+2018")
                # print(chi2dof); exit(1)
                if v_n.__contains__("chi2dof"):  vals.append(chi2dof)
            if v_n.__contains__("Kruger-"):
                print("\tTask: {}".format(v_n))
                i_coeffs, chi2, chi2dof, R2 = df.fit2("Kruger+2020", "Kruger+2020", "Kruger+2020")
                if v_n.__contains__("chi2dof"):  vals.append(chi2dof)
            if v_n.__contains__("poly2-"):
                print("\tTask: {}".format(v_n))
                b0, bi, chi2, chi2dof, r2 = df.linear_regression(degree=degree, v_n_x=v_ns_x[-1], v_n_y=v_n_y)
                if v_n.__contains__("chi2dof"):  vals.append(chi2dof)
            if v_n.__contains__("poly22-"):
                print("\tTask: {}".format(v_n))
                b0, bi, chi2, chi2dof, r2 = df.linear_regression2(degree=degree, v_ns_x=v_ns_x, v_ns_y=[v_n_y])
                if v_n.__contains__("chi2dof"):  vals.append(chi2dof)
        all_vals.append(vals)

    print("\t---<DataCollected>---")

    #
    # for label in row_labels:
    #     print(label)
    # exit(1)

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
        row_names = row_labels[i]
        if row_names == row_labels[0]:
            row_name = ""
            for dsname in row_names:
                if not dsname == row_names[-1]: row_name = row_name + dsname + ','
                else: row_name = row_name + dsname + ''
        else:
            row_name = '\& \cite{'
            for dsname in row_names:
                if dsname == "This work": pass
                else:
                    if not dsname == row_names[-1]: row_name = row_name + dsname + ','
                    else: row_name = row_name + dsname + '}'
        # DATA ITSELF
        vals = all_vals[i]
        row = row_name + " & "
        assert len(vals) == len(fmts)
        for val, fmt in zip(vals, fmts):
            if val != vals[-1]:
                val = __get_str_val(val, fmt)
                # if val < 1e-2:  val = str(("%{}".format(coeff_small_fmt) % float(val)))
                # else: val = __get_str_val()#str(("%{}".format(coeff_fmt) % float(val)))
                row = row + val + " & "
            else:
                val = __get_str_val(val, fmt)
                # if val < 1e-2:  val = str(("%{}".format(coeff_small_fmt) % float(val)))
                # else: val = str(("%{}".format(coeff_fmt) % float(val)))
                row = row + val + r" \\ "

        print(row)
        #row[-2] = r" \\ "


    print(r"\end{tabular}")
    print(r"\end{table}")

### yeej
def task_yeej_chi2dofs():

    v_n_y = "Ye_ave-geo"
    v_ns_x = ["q", "Lambda"]
    degree = 2
    v_ns = ["datasets", "mean-chi2dof", "our-chi2dof", "poly2-chi2dof", "poly22-chi2dof"]
    fmts = [            ".1f", ".1f", ".1f", ".1f"]
    #v_ns = ["Radice-chi2dof"]
    #v_ns_labels = ["datasets", r"$\chi^2 _{\text{dof}}$", r"$\chi^2 _{\text{dof}}$", r"$\chi^2 _{\text{dof}}$", r"$\chi^2 _{\text{dof}}$", r"$\chi^2 _{\text{dof}}$"]
    v_ns_labels = ["datasets", r"Mean", r"Eq.~\eqref{eq:fit_Yeej}", r"$P_2(\tilde{\Lambda})$", r"$P_2(q,\tilde{\Lambda})$"]
    error_method = "arr"

    # ---
    row_labels, all_vals = [], []
    mdisk_datasets = OrderedDict()

    for i in range(9):
        if i >= 0: mdisk_datasets['Reference set']      = {"models": md.groups, "data": md, "fit": True}
        if i >= 1: mdisk_datasets["Vincent:2019kor"]    = {"models": vi.simulations, "data": vi, "fit": True}
        if i >= 2: mdisk_datasets["Radice:2018pdn[M0]"] = {"models": rd.simulations[rd.with_m0], "data": rd, "fit": True}
        if i >= 3: mdisk_datasets["Radice:2018pdn"]     = {"models": rd.simulations[rd.fiducial], "data": rd, "fit": True}
        #if i >= 4: mdisk_datasets["Lehner:2016lxy"]     = {"models": lh.simulations, "data": lh, "fit": True}
        # if i >= 5: mdisk_datasets["Kiuchi:2019lls"] =     {"models": ki.simulations[ki.mask_for_with_tov_data], "data": ki, "fit": True}
        #if i >= 5: mdisk_datasets["Dietrich:2016lyp"]   = {"models": di16.simulations[di16.mask_for_with_sr],"data": di16, "fit": True}
        #if i >= 6: mdisk_datasets["Dietrich:2015iva"]   = {"models": di15.simulations[di15.mask_for_with_sr], "data": di15, "fit": True}
        if i == 4: break
        print(mdisk_datasets.keys())
        row_labels.append(mdisk_datasets.keys())

        df = Fit_Data(mdisk_datasets, v_n_y, error_method=error_method)

        vals = []
        for v_n in v_ns:
            if v_n.__contains__("mean-"):
                if v_n.__contains__("chi2dof"):  vals.append(df.get_ch2dof_mean())
            if v_n.__contains__("our-"):
                print("\tTask: {}".format(v_n))
                i_coeffs, chi2, chi2dof, R2 = df.fit2("our", "us", "Radice+2018")
                # print(chi2dof); exit(1)
                if v_n.__contains__("chi2dof"):  vals.append(chi2dof)
            if v_n.__contains__("Radice-"):
                print("\tTask: {}".format(v_n))
                i_coeffs, chi2, chi2dof, R2 = df.fit2("Radice+2018", "Radice+2018", "Radice+2018")
                # print(chi2dof); exit(1)
                if v_n.__contains__("chi2dof"):  vals.append(chi2dof)
            if v_n.__contains__("Kruger-"):
                print("\tTask: {}".format(v_n))
                i_coeffs, chi2, chi2dof, R2 = df.fit2("Kruger+2020", "Kruger+2020", "Kruger+2020")
                if v_n.__contains__("chi2dof"):  vals.append(chi2dof)
            if v_n.__contains__("poly2-"):
                print("\tTask: {}".format(v_n))
                b0, bi, chi2, chi2dof, r2 = df.linear_regression(degree=degree, v_n_x=v_ns_x[-1], v_n_y=v_n_y)
                if v_n.__contains__("chi2dof"):  vals.append(chi2dof)
            if v_n.__contains__("poly22-"):
                print("\tTask: {}".format(v_n))
                b0, bi, chi2, chi2dof, r2 = df.linear_regression2(degree=degree, v_ns_x=v_ns_x, v_ns_y=[v_n_y])
                if v_n.__contains__("chi2dof"):  vals.append(chi2dof)
        all_vals.append(vals)

    print("\t---<DataCollected>---")

    #
    # for label in row_labels:
    #     print(label)
    # exit(1)

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
        row_names = row_labels[i]
        if row_names == row_labels[0]:
            row_name = ""
            for dsname in row_names:
                if not dsname == row_names[-1]: row_name = row_name + dsname + ','
                else: row_name = row_name + dsname + ''
        else:
            row_name = '\& \cite{'
            for dsname in row_names:
                if dsname == "This work": pass
                else:
                    if not dsname == row_names[-1]: row_name = row_name + dsname + ','
                    else: row_name = row_name + dsname + '}'
        # DATA ITSELF
        vals = all_vals[i]
        row = row_name + " & "
        assert len(vals) == len(fmts)
        for val, fmt in zip(vals, fmts):
            if val != vals[-1]:
                val = __get_str_val(val, fmt)
                # if val < 1e-2:  val = str(("%{}".format(coeff_small_fmt) % float(val)))
                # else: val = __get_str_val()#str(("%{}".format(coeff_fmt) % float(val)))
                row = row + val + " & "
            else:
                val = __get_str_val(val, fmt)
                # if val < 1e-2:  val = str(("%{}".format(coeff_small_fmt) % float(val)))
                # else: val = str(("%{}".format(coeff_fmt) % float(val)))
                row = row + val + r" \\ "

        print(row)
        #row[-2] = r" \\ "


    print(r"\end{tabular}")
    print(r"\end{table}")

### Mdisk
def task_mdisk_chi2dofs():

    v_n_y = "Mdisk3D"
    v_ns_x = ["q", "Lambda"]
    degree = 2
    v_ns = ["datasets", "mean-chi2dof", "Radice-chi2dof", "Kruger-chi2dof", "poly2-chi2dof", "poly22-chi2dof"]
    fmts = [            ".1f", ".1f", ".1f", ".1f", ".1f"]
    #v_ns = ["Radice-chi2dof"]
    #v_ns_labels = ["datasets", r"$\chi^2 _{\text{dof}}$", r"$\chi^2 _{\text{dof}}$", r"$\chi^2 _{\text{dof}}$", r"$\chi^2 _{\text{dof}}$", r"$\chi^2 _{\text{dof}}$"]
    v_ns_labels = ["datasets", r"Mean", r"Eq.~\eqref{eq:fit_Mdisk}", r"Eq.~\eqref{eq:fit_Mdisk_Kruger}", r"$P_2(\tilde{\Lambda})$", r"$P_2(q,\tilde{\Lambda})$"]
    error_method = "arr"

    # ---
    row_labels, all_vals = [], []
    mdisk_datasets = OrderedDict()

    for i in range(9):
        if i >= 0: mdisk_datasets['Reference set']      = {"models": md.groups, "data": md, "fit": True}
        if i >= 1: mdisk_datasets["Vincent:2019kor"]    = {"models": vi.simulations, "data": vi, "fit": True}
        if i >= 2: mdisk_datasets["Radice:2018pdn[M0]"] = {"models": rd.simulations[rd.with_m0], "data": rd, "fit": True}
        if i >= 3: mdisk_datasets["Radice:2018pdn"]     = {"models": rd.simulations[rd.fiducial], "data": rd, "fit": True}
        # if i >= 4: mdisk_datasets["Lehner:2016lxy"]     = {"models": lh.simulations, "data": lh, "fit": True}
        if i >= 4: mdisk_datasets["Kiuchi:2019lls"] =     {"models": ki.simulations[ki.mask_for_with_tov_data], "data": ki, "fit": True}
        if i >= 5: mdisk_datasets["Dietrich:2016lyp"]   = {"models": di16.simulations[di16.mask_for_with_sr],"data": di16, "fit": True}
        if i >= 6: mdisk_datasets["Dietrich:2015iva"]   = {"models": di15.simulations[di15.mask_for_with_sr], "data": di15, "fit": True}
        if i == 7: break
        print(mdisk_datasets.keys())
        row_labels.append(mdisk_datasets.keys())

        df = Fit_Data(mdisk_datasets, v_n_y, error_method=error_method)

        vals = []
        for v_n in v_ns:
            if v_n.__contains__("mean-"):
                if v_n.__contains__("chi2dof"):  vals.append(df.get_ch2dof_mean())
            if v_n.__contains__("Radice-"):
                print("\tTask: {}".format(v_n))
                i_coeffs, chi2, chi2dof, R2 = df.fit2("Radice+2018", "Radice+2018", "Radice+2018")
                # print(chi2dof); exit(1)
                if v_n.__contains__("chi2dof"):  vals.append(chi2dof)
            if v_n.__contains__("Kruger-"):
                print("\tTask: {}".format(v_n))
                i_coeffs, chi2, chi2dof, R2 = df.fit2("Kruger+2020", "Kruger+2020", "Kruger+2020")
                if v_n.__contains__("chi2dof"):  vals.append(chi2dof)
            if v_n.__contains__("poly2-"):
                print("\tTask: {}".format(v_n))
                b0, bi, chi2, chi2dof, r2 = df.linear_regression(degree=degree, v_n_x=v_ns_x[-1], v_n_y=v_n_y)
                if v_n.__contains__("chi2dof"):  vals.append(chi2dof)
            if v_n.__contains__("poly22-"):
                print("\tTask: {}".format(v_n))
                b0, bi, chi2, chi2dof, r2 = df.linear_regression2(degree=degree, v_ns_x=v_ns_x, v_ns_y=[v_n_y])
                if v_n.__contains__("chi2dof"):  vals.append(chi2dof)
        all_vals.append(vals)

    print("\t---<DataCollected>---")

    #
    # for label in row_labels:
    #     print(label)
    # exit(1)

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
        row_names = row_labels[i]
        if row_names == row_labels[0]:
            row_name = ""
            for dsname in row_names:
                if not dsname == row_names[-1]: row_name = row_name + dsname + ','
                else: row_name = row_name + dsname + ''
        else:
            row_name = '\& \cite{'
            for dsname in row_names:
                if dsname == "This work": pass
                else:
                    if not dsname == row_names[-1]: row_name = row_name + dsname + ','
                    else: row_name = row_name + dsname + '}'
        # DATA ITSELF
        vals = all_vals[i]
        row = row_name + " & "
        assert len(vals) == len(fmts)
        for val, fmt in zip(vals, fmts):
            if val != vals[-1]:
                val = __get_str_val(val, fmt)
                # if val < 1e-2:  val = str(("%{}".format(coeff_small_fmt) % float(val)))
                # else: val = __get_str_val()#str(("%{}".format(coeff_fmt) % float(val)))
                row = row + val + " & "
            else:
                val = __get_str_val(val, fmt)
                # if val < 1e-2:  val = str(("%{}".format(coeff_small_fmt) % float(val)))
                # else: val = str(("%{}".format(coeff_fmt) % float(val)))
                row = row + val + r" \\ "

        print(row)
        #row[-2] = r" \\ "


    print(r"\end{tabular}")
    print(r"\end{table}")

''' ------------------------------ '''
def task_save_csv_of_all_datasets():

    # print(md.groups.keys())
    # md.groups.set_index("groups")
    # md.groups["models"] = md.groups.index
    # print(md.groups["models"]); exit(1)

    datasets = OrderedDict()
    datasets['Reference set'] = {"models": md.groups, "data": md, "fit": True}
    datasets["Vincent:2019kor"] = {"models": vi.simulations, "data": vi, "fit": True}
    datasets["Radice:2018pdn[M0]"] = {"models": rd.simulations[rd.with_m0], "data": rd, "fit": True}
    datasets["Radice:2018pdn"] = {"models": rd.simulations[rd.fiducial], "data": rd, "fit": True}
    datasets["Lehner:2016lxy"] = {"models": lh.simulations, "data": lh, "fit": True}
    datasets["Kiuchi:2019lls"] = {"models": ki.simulations[ki.mask_for_with_tov_data], "data": ki, "fit": True}
    datasets["Dietrich:2016lyp"] = {"models": di16.simulations[di16.mask_for_with_sr], "data": di16, "fit": True}
    datasets["Dietrich:2015iva"] = {"models": di15.simulations[di15.mask_for_with_sr], "data": di15, "fit": True}
    #
    datasets["Bauswein:2013yna"] =  {"models": bs.simulations, "data": bs, "label": r"Bauswein+2013", "fit": True}
    datasets["Hotokezaka:2012ze"] = {"models": hz.simulations, "data": hz, "label": r"Hotokezaka+2013", "fit": True}
    datasets["Sekiguchi:2015dma"] = {"models": se15.simulations[se15.mask_for_with_sr], "data": se15, "label": r"Sekiguchi+2015",  "fit": True}
    datasets["Sekiguchi:2016bjd"] = {"models": se16.simulations[se16.mask_for_with_sr], "data": se16, "label": r"Sekiguchi+2016", "fit": True}
    #
    v_ns = ["EOS", "M1", "M2", "q", "C1", "C2", "Lambda", "Mej_tot-geo", "vel_inf_ave-geo", "Ye_ave-geo", "Mdisk3D"]
    #
    dataframe = create_combine_dataframe2(datasets, v_ns, [], {}, "fit", ifabsent=np.nan)
    dataframe = dataframe[["dset_name"]+v_ns]
    dataframe.to_csv("../datasets/combined.csv")

if __name__ == '__main__':

    ''' --- ejecta mass --- '''
    ### Mej
    # task_print_stats(v_n = "Mej_tot-geo")
    # task_mej_chi2dofs()
    task_table_linear_regresion(v_n_y="Mej_tot-geo", v_n_x=["q"], degree=2, error_method="arr", fancy=False)
    # task_table_linear_regresion(v_n_y="Mej_tot-geo", degree=2, error_method="arr", fancy=False)
    # task_table_linear_regresion(v_n_y="Mej_tot-geo", v_n_x=["q", "Lambda"], degree=2, error_method="arr", fancy=False)
    # task_fitfunc_print_table(v_n="Mej_tot-geo", error_method="arr",
    #                          ff_name="Radice+2018", cf_name="Radice+2018", rs_name="Radice+2018", fancy=False)
    # task_fitfunc_print_table(v_n="Mej_tot-geo", error_method="arr",
    #                          ff_name="Kruger+2020", cf_name="Kruger+2020", rs_name="Radice+2018", fancy=False)

    ### vej
    # prtask_print_stats(v_n="vel_inf_ave-geo")
    # task_vej_chi2dofs()
    # task_table_linear_regresion(v_n_y="vel_inf_ave-geo", degree=2, error_method="arr", fancy=False)
    # task_table_linear_regresion(v_n_y="vel_inf_ave-geo", v_n_x=["q","Lambda"], degree=2, error_method="arr", fancy=False)
    # task_fitfunc_print_table(v_n="vel_inf_ave-geo", error_method="arr",
    #                          ff_name="Radice+2018", cf_name="Radice+2018", rs_name="Radice+2018", fancy=False)

    ### yeej
    # task_print_stats(v_n="Ye_ave-geo")
    # task_yeej_chi2dofs()
    # task_table_linear_regresion(v_n_y="Ye_ave-geo", degree=2, error_method="arr", fancy=False)
    # task_table_linear_regresion(v_n_y="Ye_ave-geo", v_n_x=["q", "Lambda"], degree=2, error_method="arr", fancy=False)
    # task_fitfunc_print_table(v_n="Ye_ave-geo", error_method="arr",
    #                          ff_name="us", cf_name="us", rs_name="Radice+2018", fancy=False)

    ### Mdsik
    # task_print_stats(v_n="Mdisk3D")
    # task_mdisk_chi2dofs()
    # task_table_linear_regresion(v_n_y="Mdisk3D", degree=2, error_method="arr", fancy=False)
    # task_table_linear_regresion(v_n_y="Mdisk3D", v_n_x=["q", "Lambda"], degree=2, error_method="arr", fancy=False)
    # task_fitfunc_print_table(v_n="Mdisk3D", error_method="arr",
    #                          ff_name="Radice+2018", cf_name="Radice+2018", rs_name="Radice+2018", fancy=False)
    # task_fitfunc_print_table(v_n="Mdisk3D", error_method="arr",
    #                          ff_name="Kruger+2020", cf_name="Kruger+2020", rs_name="Radice+2018", fancy=False)

    ''' --- '''
    # task_save_csv_of_all_datasets()