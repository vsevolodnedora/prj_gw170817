#!/usr/bin/env python3

# TODO UNIFNISHED
# Errors: 1. Wrong errors calls for all variable except mass
# Errors: 2. Wrong Error for Yeej -- 0.01 is the correct value
# Errors: 3. Not working mdisk fit

"""
Analyse the data in the .csv with various fitting formulas

SB 08/2020
"""



from __future__ import division

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class DataCSV(object):
    def __init__(self, fname=None, data=None):
        """
        Read & Work with data in the .csv
        """
        self.data = []
        if fname:
            self.csv2dict(fname)
        elif data:
            self.data = data

    def csv2dict(self, fname):
        """
        Read .csv as a list of dict into data
        """
        self.data = pd.read_csv(fname, delimiter=',').to_dict(orient='records')

    def group(self, key, val):
        """
        Return a sublist for the given (key, val) pair
        """
        if key not in self.data[0].keys(): return self.data
        return [d for d in self.data if d[key] == val]

    def add(self, newdata):
        """
        Add data, does not check for double entries
        """
        self.data += newdata

    def rm(self, olddata):
        """
        Remove data matching the input
        """
        self.data.remove(olddata)

    def model_poly2_1d(self, x, b0, b1, b2):
        """
        2nd order poly in x
        """
        return b0 + b1 * x + b2 * x ** 2

    def model_poly2_2d(self, X, b0, b1, b2, b3, b4, b5):
        """
        2nd order poly in (x,y)
        """
        x, y = X
        return b0 + b1 * x + b2 * y + b3 * x ** 2 + b4 * x * y + b5 * y ** 2

    def model_mass_KDR(self, X, alp, bet, gam, dlt, n):
        """
        Mass fit model of Kawaguchi+, Dietrich+, Radice+
        Note the formula refers to Mdyn[Mo]/1e-3
        """
        MA, MB, CA, CB, MbA, MbB = X
        XBA = MB / MA
        XAB = MA / MB
        factA = alp * XBA ** (1. / 3.) * (1. - 2 * CA) / CA + bet * XBA ** n + gam * (1 - MA / MbA)
        factB = alp * XAB ** (1. / 3.) * (1. - 2 * CB) / CB + bet * XAB ** n + gam * (1 - MB / MbB)
        return 10 * (factA * MbA + factB * MbB + dlt)

    def model_mass_KF(self, X, alp, bet, gam, n):
        """
        Mass fit model of Krueger&Foucart
        Note the formula refers to Mdyn[Mo]/1e-3
        """
        MA, MB, CA, CB = X
        XBA = MB / MA
        XAB = MA / MB
        factA = alp / CA + bet * XBA ** n + gam * CA
        factB = alp / CB + bet * XAB ** n + gam * CB
        return 10. * (factA * MA + factB * MB)

    def model_vel_KDR(self, X, alp, bet, gam):
        """
        Vel fit model of Kawaguchi+, Dietrich+, Radice+
        """
        MA, MB, CA, CB = X
        XBA = MB / MA
        XAB = MA / MB
        factA = alp * XAB * (1 + gam * CA)
        factB = alp * XBA * (1 + gam * CB)
        return factA + factB + bet

    def model_Ye_KDR(self, X, alp, bet, gam):
        """
        Ye fit model KDR-like
        """
        MA, MB, CA, CB = X
        XBA = MB / MA
        XAB = MA / MB
        factA = 1e-5 * alp * XAB * (1 + 1e5 * gam * CA)
        factB = 1e-5 * alp * XBA * (1 + 1e5 * gam * CB)
        return factA + factB + bet

    def model_disc_mass_R(self, tLam, alp, bet, gam, dlt):
        """
        Disc mass fit model of Radice+
        Note the original formula refers to Mdyn[Mo]
        """
        floor = 1e-3 * np.ones_like(tLam)
        return np.maximum(floor, alp + bet * np.tanh((tLam - gam) / dlt)) * 100.

    def model_disc_mass_KF(self, X, alp, bet, gam):
        """
        Disc mass fit model of KF+
        Note the original formula refers to Mdyn[Mo]
        """
        MA, MB, CA, CB = X
        floor = 5e-4 * np.ones_like(CA)
        factA = np.maximum(floor, (alp * CA + bet) ** gam)
        factB = np.maximum(floor, (alp * CB + bet) ** gam)
        return (MA * factA + MB * factB) * 100.

    def chi2dof(self, o, c, npars, s=None):
        """
        Reduced chi^2
        """
        if not s.any(): s = np.ones_like(o)
        nu = len(o) - npars
        return np.sum(((o - c) / s) ** 2) / nu

    def error_me(self, me):
        """
        Assign uncertainty on Mej
        Note this is on Mej, while we use Mej * 100
        """
        return 0.5 * me + 5e-5

    def error_ve(self, ve):
        """
        Assign uncertainty on Vej
        """
        return 0.02 * np.ones(len(ve))

    def error_ye(self, ye):
        """
        Assign uncertainty on Ye
        """
        return 0.1 * np.ones(len(ye)) # ??? # TODO ERROR = 0.01

    def error_mdisc(self, me):
        """
        Assign uncertainty on Mdisc
        Note this is on Mdisc, while we use Mdisc * 100
        """
        return 0.5 * me + 5e-4

    def fitme(self, model, X, data, p0=None, sigma=None):
        """
        Standard fitting routine based on curve_fit()
        """
        print(len(data))
        popt, pcov = curve_fit(model, X, data, p0, sigma)
        perr = np.sqrt(np.diag(pcov))
        bfit = model(X, *popt)
        res = bfit - data
        chi2 = self.chi2dof(data, bfit, len(popt), sigma)
        return popt, pcov, perr, res, chi2, bfit

    def plot_fit(self, yres, yfit, lab=None, fout=None):
        """
        Make 1d fit plots
        """
        fig, ax = plt.subplots()
        # ax.plot(yfit,yres/yfit,'o') #,label=)
        ax.plot(yfit, yres, 'o')  # ,label=)
        ax.grid()
        ax.set_xlabel('Fit')  # +lab)
        ax.set_ylabel('Residuals')  # +lab)
        if fout:
            fig.savefig(fout + ".pdf")
            plt.close(fig)
        else:
            plt.show()

    def fit(self, fit_to_do=None, make_plot=None, save_file=None, verbose=None):
        """
        Do the required fits
        """

        # TODO check order

        # Parameters
        q = np.array([d['q'] for d in self.data], dtype='float64')
        MA = np.array([d['MA'] for d in self.data], dtype='float64')
        MB = np.array([d['MB'] for d in self.data], dtype='float64')
        MbA = np.array([d['MbA'] for d in self.data], dtype='float64')
        MbB = np.array([d['MbB'] for d in self.data], dtype='float64')
        tLam = np.array([d['tLam'] for d in self.data], dtype='float64')
        CA = np.array([d['CA'] for d in self.data], dtype='float64')
        CB = np.array([d['CB'] for d in self.data], dtype='float64')

        # Data to fit
        Mej = np.array([d['Mej1e-2'] for d in self.data], dtype='float64')  # Note this is Mej/1e-2
        Vej = np.array([d['Vej'] for d in self.data], dtype='float64')
        Yeej = np.array([d['Yeej'] for d in self.data], dtype='float64')
        Mdisc = np.array([d['Mdisk1e-2'] for d in self.data], dtype='float64')  # Note this is Mdisc/1e-2

        Mej_Mo = Mej / 100.  # Mo
        Mdisc_Mo = Mdisc / 100.  # Mo

        # Errors
        sigma_Mej = self.error_me(Mej_Mo) * 100
        sigma_Vej = self.error_ve(Vej)
        sigma_Yeej = self.error_ye(Yeej)
        sigma_Mdisc = self.error_mdisc(Mdisc_Mo) * 100

        # Reduced datasets (rm NaNs)
        i_vnn = ~np.isnan(Vej)
        i_ynn = ~np.isnan(Yeej)
        i_dnn = ~np.isnan(Mdisc)

        # Do the fitting and store things in 'result'
        result = {}

        if 'model_mass_mean' in fit_to_do:
            mm = np.mean(Mej)
            result['model_mass_mean'] = {}
            result['model_mass_mean']['bfit'] = mm * np.ones_like(Mej)
            result['model_mass_mean']['chi2'] = self.chi2dof(Mej, result['model_mass_mean']['bfit'], 0, sigma_Mej)
            result['model_mass_mean']['res'] = Mej - mm
            result['model_mass_mean']['lab'] = '$M_{ej\ 100}$'

        if 'model_vel_mean' in fit_to_do:
            mm = np.mean(Vej[i_vnn])
            result['model_vel_mean'] = {}
            result['model_vel_mean']['bfit'] = mm * np.ones_like(Vej[i_vnn])
            result['model_vel_mean']['chi2'] = self.chi2dof(Vej[i_vnn], result['model_vel_mean']['bfit'], 0,
                                                            sigma_Vej[i_vnn])
            result['model_vel_mean']['res'] = Vej[i_vnn] - mm
            result['model_vel_mean']['lab'] = '$v_{\rm ej}$'

        if 'model_Ye_mean' in fit_to_do:
            mm = np.mean(Yeej[i_ynn])
            result['model_Ye_mean'] = {}
            result['model_Ye_mean']['bfit'] = mm * np.ones_like(Yeej[i_ynn])
            result['model_Ye_mean']['chi2'] = self.chi2dof(Yeej[i_ynn], result['model_Ye_mean']['bfit'], 0,
                                                           sigma_Yeej[i_ynn])
            result['model_Ye_mean']['res'] = Yeej[i_ynn] - mm
            result['model_Ye_mean']['lab'] = '$Y_{e\ ej}$'

        if 'model_disc_mass_mean' in fit_to_do:
            mm = np.mean(Mdisc[i_dnn])
            result['model_disc_mass_mean'] = {}
            result['model_disc_mass_mean']['bfit'] = mm * np.ones_like(Mdisc[i_dnn])
            result['model_disc_mass_mean']['chi2'] = self.chi2dof(Mdisc[i_dnn], result['model_disc_mass_mean']['bfit'],
                                                                  0, sigma_Mdisc[i_dnn])
            result['model_disc_mass_mean']['res'] = Mdisc[i_dnn] - mm
            result['model_disc_mass_mean']['lab'] = '$M_{disc\ 100}$'

        if 'model_mass_poly2_tLam' in fit_to_do:
            p0 = 0.1, 0.01, 0.,
            popt, pcov, perr, res, chi2, bfit = self.fitme(self.model_poly2_1d, tLam, Mej, p0, sigma_Mej)
            result['model_mass_poly2_tLam'] = {}
            result['model_mass_poly2_tLam']['opt'] = popt
            result['model_mass_poly2_tLam']['perr'] = perr
            result['model_mass_poly2_tLam']['res'] = res
            result['model_mass_poly2_tLam']['bfit'] = bfit
            result['model_mass_poly2_tLam']['lab'] = '$M_{ej\ 100}$'
            result['model_mass_poly2_tLam']['chi2'] = chi2

        if 'model_mass_poly2_tLamq' in fit_to_do:
            p0 = 0.1, 0.01, 0.01, 0.001, 0.001, 0.00001
            popt, pcov, perr, res, chi2, bfit = self.fitme(self.model_poly2_2d, (tLam, q), Mej, p0, sigma_Mej)
            result['model_mass_poly2_tLamq'] = {}
            result['model_mass_poly2_tLamq']['opt'] = popt
            result['model_mass_poly2_tLamq']['perr'] = perr
            result['model_mass_poly2_tLamq']['res'] = res
            result['model_mass_poly2_tLamq']['bfit'] = bfit
            result['model_mass_poly2_tLamq']['lab'] = '$M_{ej\ 100}$'
            result['model_mass_poly2_tLamq']['chi2'] = chi2

        if 'model_vel_poly2_tLamq' in fit_to_do:
            p0 = 0.1, 0.01, 0.01, 0.001, 0.001, 0.00001
            popt, pcov, perr, res, chi2, bfit = self.fitme(self.model_poly2_2d, (tLam[i_vnn], q[i_vnn]), Vej[i_vnn], p0,
                                                           sigma_Vej[i_vnn])
            result['model_vel_poly2_tLamq'] = {}
            result['model_vel_poly2_tLamq']['opt'] = popt
            result['model_vel_poly2_tLamq']['perr'] = perr
            result['model_vel_poly2_tLamq']['res'] = res
            result['model_vel_poly2_tLamq']['bfit'] = bfit
            result['model_vel_poly2_tLamq']['lab'] = '$v_{ej}$'
            result['model_vel_poly2_tLamq']['chi2'] = chi2

        if 'model_Ye_poly2_tLamq' in fit_to_do:
            p0 = 0.1, 0.01, 0.01, 0.001, 0.001, 0.00001
            popt, pcov, perr, res, chi2, bfit = self.fitme(self.model_poly2_2d, (tLam[i_ynn], q[i_ynn]), Yeej[i_ynn],
                                                           p0, sigma_Yeej[i_ynn])
            result['model_Ye_poly2_tLamq'] = {}
            result['model_Ye_poly2_tLamq']['opt'] = popt
            result['model_Ye_poly2_tLamq']['perr'] = perr
            result['model_Ye_poly2_tLamq']['res'] = res
            result['model_Ye_poly2_tLamq']['bfit'] = bfit
            result['model_Ye_poly2_tLamq']['lab'] = '$Y_{e\ ej}$'
            result['model_Ye_poly2_tLamq']['chi2'] = chi2

        if 'model_disc_mass_poly2_tLamq' in fit_to_do:
            p0 = 0.1, 0.01, 0.01, 0.001, 0.001, 0.00001
            popt, pcov, perr, res, chi2, bfit = self.fitme(self.model_poly2_2d, (tLam[i_dnn], q[i_dnn]), Mdisc[i_dnn],
                                                           p0, sigma_Mdisc[i_dnn])
            result['model_disc_mass_poly2_tLamq'] = {}
            result['model_disc_mass_poly2_tLamq']['opt'] = popt
            result['model_disc_mass_poly2_tLamq']['perr'] = perr
            result['model_disc_mass_poly2_tLamq']['res'] = res
            result['model_disc_mass_poly2_tLamq']['bfit'] = bfit
            result['model_disc_mass_poly2_tLamq']['lab'] = '$M_{ej\ 100}$'
            result['model_disc_mass_poly2_tLamq']['chi2'] = chi2

        if 'model_mass_KDR' in fit_to_do:
            p0 = -.6, 4.2, -32., 5., 1.
            popt, pcov, perr, res, chi2, bfit = self.fitme(self.model_mass_KDR, (MA, MB, CA, CB, MbA, MbB), Mej, p0,
                                                           sigma_Mej)
            # print(popt)
            result['model_mass_KDR'] = {}
            result['model_mass_KDR']['opt'] = popt
            result['model_mass_KDR']['perr'] = perr
            result['model_mass_KDR']['res'] = res
            result['model_mass_KDR']['bfit'] = bfit
            result['model_mass_KDR']['lab'] = '$M_{ej\ 100}$'
            result['model_mass_KDR']['chi2'] = chi2

        if 'model_vel_KDR' in fit_to_do:
            p0 = -0.3, 0.5, -3.
            popt, pcov, perr, res, chi2, bfit = self.fitme(self.model_vel_KDR,
                                                           (MA[i_vnn], MB[i_vnn], CA[i_vnn], CB[i_vnn]), Vej[i_vnn], p0,
                                                           sigma_Vej[i_vnn])
            result['model_vel_KDR'] = {}
            result['model_vel_KDR']['opt'] = popt
            result['model_vel_KDR']['perr'] = perr
            result['model_vel_KDR']['res'] = res
            result['model_vel_KDR']['bfit'] = bfit
            result['model_vel_KDR']['lab'] = '$v_{ej}$'
            result['model_vel_KDR']['chi2'] = chi2

        if 'model_Ye_KDR' in fit_to_do:
            p0 = 0.1, 0.5, -8.
            popt, pcov, perr, res, chi2, bfit = self.fitme(self.model_Ye_KDR,
                                                           (MA[i_ynn], MB[i_ynn], CA[i_ynn], CB[i_ynn]), Yeej[i_ynn],
                                                           p0, sigma_Yeej[i_ynn])
            result['model_Ye_KDR'] = {}
            result['model_Ye_KDR']['opt'] = popt
            result['model_Ye_KDR']['perr'] = perr
            result['model_Ye_KDR']['res'] = res
            result['model_Ye_KDR']['bfit'] = bfit
            result['model_Ye_KDR']['lab'] = '$Y_{e\ ej}$'
            result['model_Ye_KDR']['chi2'] = chi2

        if 'model_mass_KF' in fit_to_do:
            p0 = -9., 100., -300., 1.
            popt, pcov, perr, res, chi2, bfit = self.fitme(self.model_mass_KF, (MA, MB, CA, CB), Mej, p0, sigma_Mej)
            result['model_mass_KF'] = {}
            result['model_mass_KF']['opt'] = popt
            result['model_mass_KF']['perr'] = perr
            result['model_mass_KF']['res'] = res
            result['model_mass_KF']['bfit'] = bfit
            result['model_mass_KF']['lab'] = '$M_{ej\ 100}$'
            result['model_mass_KF']['chi2'] = chi2

        if 'model_disc_mass_R' in fit_to_do:
            p0 = 0.08, 0.08, 567., 405.
            popt, pcov, perr, res, chi2, bfit = self.fitme(self.model_disc_mass_R, tLam[i_dnn], Mdisc[i_dnn], p0,
                                                           sigma_Mdisc[i_dnn])
            result['model_disc_mass_R'] = {}
            result['model_disc_mass_R']['opt'] = popt
            result['model_disc_mass_R']['perr'] = perr
            result['model_disc_mass_R']['res'] = res
            result['model_disc_mass_R']['bfit'] = bfit
            result['model_disc_mass_R']['lab'] = '$M_{{\rm disc}\ 100}$'
            result['model_disc_mass_R']['chi2'] = chi2

        if 'model_disc_mass_KF' in fit_to_do:
            p0 = -8., 1., 1.
            popt, pcov, perr, res, chi2, bfit = self.fitme(self.model_disc_mass_KF,
                                                           (MA[i_dnn], MB[i_dnn], CA[i_dnn], CB[i_dnn]), Mdisc[i_dnn],
                                                           p0, sigma_Mdisc[i_dnn])
            result['model_disc_mass_KF'] = {}
            result['model_disc_mass_KF']['opt'] = popt
            result['model_disc_mass_KF']['perr'] = perr
            result['model_disc_mass_KF']['res'] = res
            result['model_disc_mass_KF']['bfit'] = bfit
            result['model_disc_mass_KF']['lab'] = '$M_{{\rm disc}\ 100}$'
            result['model_disc_mass_KF']['chi2'] = bfit

        # Output
        if verbose:
            for m in fit_to_do:
                print('{} {} {}'.format(save_file, m, result[m]['chi2']))
        if save_file: np.savez(save_file, result=result)
        if make_plot:
            for m in fit_to_do:
                ##print('{} {}'.format(save_file,m)) # DEBUG
                if save_file:
                    sv = save_file + "_" + m
                else:
                    sv = None
                self.plot_fit(result[m]['res'], result[m]['bfit'], result[m]['lab'], sv)


if __name__ == "__main__":

    # Fits list
    # ##############################################

    dynej_mass_fits = []#['model_mass_poly2_tLam', 'model_mass_poly2_tLamq', 'model_mass_KDR', 'model_mass_KF']
    dynej_vel_fits = ["model_vel_poly2_tLamq","model_vel_KDR"]
    dynej_ye_fits = []#['model_Ye_poly2_tLamq', 'model_Ye_KDR']
    dynej_mean = []#['model_mass_mean', 'model_vel_mean', 'model_Ye_mean']
    dynej_all = dynej_mass_fits + dynej_vel_fits + dynej_ye_fits + dynej_mean

    disc_fits = ['model_disc_mass_R', 'model_disc_mass_KF', 'model_disc_mass_poly2_tLamq', 'model_disc_mass_mean']

    make_plot = 0
    save_file = None
    verbose = 1

    if 0:
        # Ejecta
        # ##############################################

        # All data
        # ds = DataCSV()
        # ds = ds.read_from_csv('test.csv') # DEBUG
        # ds = DataCSV(fname='test.csv') # DEBUG

        ds = DataCSV(fname='LiteratureData.csv')
        # print(ds.data[0].keys()) # DEBUG
        # ds = ds[ds["bibref"] == "Nedora:2020"]

        # Select reference set
        dsref = DataCSV(data=ds.group(*('nus', 'leakM0')))
        # dsref = DataCSV(data=ds.group(*('bibkey', 'Nedora:2020')))

        # DEBUG to test indivdual fits
        # dsref.fit(fit_to_do = ['model_mass_poly2_tLamq'], make_plot = make_plot ) # DEBUG
        # dsref.fit(fit_to_do = ['model_mass_KDR'], make_plot = make_plot ) # DEBUG
        # dsref.fit(fit_to_do = ['model_vel_KDR'], make_plot = make_plot ) # DEBUG
        # dsref.fit(fit_to_do = ['model_Ye_KDR'], make_plot = make_plot ) # DEBUG
        # sys.exit() # DEBUG

        # Fit ref. set ...
        #dsref.fit(fit_to_do=dynej_all, make_plot=make_plot, save_file="dynejset_M0", verbose=verbose)

        # ... add M1
        #dsref.add(ds.group(*('nus', 'M1')))
        dsref.add(ds.group(*('nus', 'leakM1')))
        #dsref.fit(fit_to_do=dynej_all, make_plot=make_plot, save_file="dynejset_M0_M1", verbose=verbose)

        # ... add leak
        dsref.add(ds.group(*('nus', 'leak')))
        #dsref.fit(fit_to_do=dynej_all, make_plot=make_plot, save_file="dynejset_M0_M1_leak", verbose=verbose)

        # ... add no nus
        dsref.add(ds.group(*('nus', 'none')))
        dsref.fit(fit_to_do=dynej_all, make_plot=make_plot, save_file="dynejset_M0_M1_leak_none", verbose=verbose)

    if 1:
        # Disc mass
        # ##############################################

        ds = DataCSV(fname='LiteratureData.csv')
        dsdisc = DataCSV(data=ds.group(*('disc_data', 1)))
        dsdiscref = DataCSV(data=dsdisc.group(*('nus', 'leakM0')))

        # TODO check individual fits first, not yet working

        # DEBUG to test indivdual fits
        # dsdiscref.fit(fit_to_do = ['model_disc_mass_R'], make_plot=1, save_file=None, verbose=1) # DEBUG
        # dsdiscref.fit(fit_to_do = ['model_disc_mass_KF'], make_plot=1, save_file=None, verbose=1) # DEBUG
        # dsdiscref.fit(fit_to_do = ['model_disc_mass_poly2_tLamq'], make_plot=1, save_file=None, verbose=1) # DEBUG
        # sys.exit() # DEBUG

        # Fit ref. set
        #dsdiscref.fit(fit_to_do=disc_fits)
        #dsdiscref.fit(fit_to_do=disc_fits, make_plot=make_plot, save_file="discset_M0", verbose=verbose)

        # ... add M1
        dsdiscref.add(dsdisc.group(*('nus', 'M1')))
        dsdiscref.add(dsdisc.group(*('nus', 'leakM1')))
        #dsdiscref.fit(fit_to_do=disc_fits, make_plot=make_plot, save_file="discset_M0_M1", verbose=verbose)

        # ... add leak
        dsdiscref.add(dsdisc.group(*('nus', 'leak')))
        #dsdiscref.fit(fit_to_do=disc_fits, make_plot=make_plot, save_file="discset_M0_M1_leak", verbose=verbose)

        # ... add no nus
        dsdiscref.add(dsdisc.group(*('nus', 'none')))
        dsdiscref.fit(fit_to_do=disc_fits, make_plot=make_plot, save_file="discset_M0_M1_leak_none", verbose=verbose)