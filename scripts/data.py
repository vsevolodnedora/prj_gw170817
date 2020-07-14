from __future__ import division

import numpy as np
import pandas as pd
import sys
import os
import copy
import h5py
import csv
from scipy import interpolate
sys.path.append('/data01/numrel/vsevolod.nedora/bns_ppr_tools/')
from preanalysis import LOAD_INIT_DATA
from outflowed import EJECTA_PARS
from preanalysis import LOAD_ITTIME
from plotting_methods import PLOT_MANY_TASKS
from profile import LOAD_PROFILE_XYXZ, LOAD_RES_CORR, LOAD_DENSITY_MODES
from utils import Paths, Lists, Labels, Constants, Printcolor, UTILS, Files, PHYSICS
from uutils import *

class LOAD_FILES(LOAD_ITTIME):

    list_outflowed_files = [
        "total_flux.dat",
        "hist_temperature.dat",
        "hist_theta.dat",
        "hist_Y_e.dat",
        "hist_log_rho.dat", #hist_rho.dat",
        "hist_entropy.dat",
        "hist_vel_inf.dat",
        "hist_vel_inf_bern.dat",

        "mass_averages.dat",

        "ejecta_profile.dat",
        "ejecta_profile_bern.dat",

        "corr_vel_inf_bern_theta.h5",
        "corr_vel_inf_theta.h5",
        "corr_ye_entropy.h5",
        "corr_ye_theta.h5",
        "corr_Y_e_theta.h5",
        "corr_vel_inf_Y_e.h5",
        "corr_Y_e_vel_inf.h5",

        "timecorr_vel_inf.h5",
        "timecorr_entropy.h5",
        "timecorr_Y_e.h5",
        "timecorr_theta.h5",

        "yields.h5"
    ]

    list_collated_files = [
        "dens_unbnd.norm1.asc",
        "dens_unbnd_bernoulli.norm1.asc",
        "dens.norm1.asc",
        "rho.maximum.asc"
    ]

    list_gw_files = [
        "postmerger_psd_l2_m2.dat",
        "waveform_l2_m2.dat",
        "tmerger.dat",
        "tcoll.dat",
        "EJ.dat"
    ]

    list_3d_data_files = [
        "remnant_mass.txt",
        "disk_mass.txt",
        "MJ_encl.txt",
        "hist_Ye.dat",
        "hist_temp.dat",
        "hist_entr.dat",
        "hist_theta.dat",
        "hist_press.dat",
        "hist_r.dat"
    ]

    def __init__(self, sim, add_mask):

        LOAD_ITTIME.__init__(self, sim)

        self.sim = sim

        self.r_gw = 400. # radius of the detector for GW

        self.list_detectors = [0, 1]

        self.list_outflow_masks = ["geo", "bern", "bern_geoend", "Y_e04_geoend", "theta60_geoend",
                           "geo_entropy_above_10", "geo_entropy_below_10"]

        self.list_3d_masks = ['', "disk", "remnant"]

        if add_mask != None and not add_mask in self.list_outflow_masks:
            self.list_outflow_masks.append(add_mask)

        self.matrix_outflow_data = [[[np.zeros(0,)
                                    for i in range(len(self.list_outflowed_files))]
                                     for k in range(len(self.list_outflow_masks))]
                                    for j in range(len(self.list_detectors))]

        self.matrix_gw_data = [np.zeros(0,)
                                     for k in range(len(self.list_outflow_masks))]

        self.matrix_collated_data = [np.zeros(0,)
                                     for i in range(len(self.list_collated_files))]

        self.matrix_3d_data = [[np.zeros(0,)
                               for i in range(len(self.list_3d_data_files))]
                               for k in range(len(self.list_3d_masks))]

    def check_v_n(self, v_n):
        if not v_n in self.list_outflowed_files:
            if not v_n in self.list_gw_files:
                if not v_n in self.list_collated_files:
                    if not v_n in self.list_3d_data_files:
                        raise NameError("v_n: {} is not in the any list: \n"
                                        "outflow: [ {} ], \ngw: [ {} ], \ncolated: [ {} ], \n3D: [ {} ]"
                                        .format(v_n,
                                                self.list_outflowed_files,
                                                self.list_gw_files,
                                                self.list_collated_files,
                                                self.list_3d_data_files))

    def check_det(self, det):
        if not det in self.list_detectors:
            raise NameError("det: {} is not in the list: {}"
                            .format(det, self.list_detectors))

    def check_outflow_mask(self, mask):
        if not mask in self.list_outflow_masks:
            return NameError("mask: {} is not in the list {} "
                             .format(mask, self.list_outflow_masks))

    def check_3d_mask(self, mask):
        if not mask in self.list_3d_masks:
            return NameError("mask: {} is not in the list {} "
                             .format(mask, self.list_3d_masks))
    def i_fn_outflow(self, v_n):
        return int(self.list_outflowed_files.index(v_n))

    def i_fn_col(self, v_n):
        return int(self.list_collated_files.index(v_n))

    def i_fn_gw(self, v_n):
        return int(self.list_gw_files.index(v_n))

    def i_fn_3d(self, v_n):
        return int(self.list_3d_data_files.index(v_n))

    def i_det(self, det):
        return int(self.list_detectors.index(int(det)))

    def i_o_mask(self, mask):
        return int(self.list_outflow_masks.index(mask))

    def i_3d_mask(self, mask):
        return int(self.list_3d_masks.index(mask))

    # --------------------------- LOADING DATA METHODS ----------------------------

    def load_outflow_corr(self, det, mask, v_n_x, v_n_y):

        fpath_1 = Paths.ppr_sims + self.sim + '/' + "outflow_{:d}".format(
            det) + '/' + mask + '/corr_' + v_n_x + '_' + v_n_y + ".h5"
        fpath_2 = Paths.ppr_sims + self.sim + '/' + "outflow_{:d}".format(
            det) + '/' + mask + '/corr_' + v_n_y + '_' + v_n_x + ".h5"

        if os.path.isfile(fpath_1):
            dfile = h5py.File(fpath_1, "r")
            v_ns = []
            for _v_n in dfile:
                v_ns.append(_v_n)
            assert len(v_ns) == 3
            assert "mass" in v_ns
            mass = np.array(dfile["mass"])
            v_ns.remove('mass')
            xarr = np.array(dfile[v_ns[0]]) #
            yarr = np.array(dfile[v_ns[1]]) #
            if len(xarr) == mass.shape[1] and len(yarr) == mass.shape[0]:
                table = UTILS.combine(xarr, yarr, mass)
            else:
                table = UTILS.combine(yarr, xarr, mass)
            return table
        elif not os.path.isfile(fpath_1) and os.path.isfile(fpath_2):
            dfile = h5py.File(fpath_2, "r")
            v_ns = []
            for _v_n in dfile:
                v_ns.append(_v_n)
            assert len(v_ns) == 3
            assert "mass" in v_ns
            mass = np.array(dfile["mass"])
            v_ns.remove('mass')
            xarr = np.array(dfile[v_ns[0]]) #
            yarr = np.array(dfile[v_ns[1]]) #
            if len(xarr) == mass.shape[1] and len(yarr) == mass.shape[0]:
                table = UTILS.combine(xarr, yarr, mass)
            else:
                table = UTILS.combine(yarr, xarr, mass)
            return table
        else:
            raise IOError("Neither one of the corr files is found: \n\t{}\n\t{}".format(fpath_1,fpath_2))

    def load_outflow_data(self, det, mask, v_n):

        fpath = Paths.ppr_sims+self.sim+'/'+ "outflow_{:d}".format(det) + '/' + mask + '/' + v_n
        if not os.path.isfile(fpath):
            raise IOError("File not found for det:{} mask:{} v_n:{} -> {}"
                          .format(det, mask, v_n, fpath))
        # loading acii file
        if not v_n.__contains__(".h5"):
            data = np.loadtxt(fpath)
            return data
        # loading nucle data
        if v_n == "yields.h5":
            dfile = h5py.File(fpath, "r")
            v_ns = []
            for _v_n in dfile:
                v_ns.append(_v_n)
            assert len(v_ns) == 3
            assert "A" in v_ns
            assert "Y_final" in v_ns
            assert "Z" in v_ns
            table = np.vstack((np.array(dfile["A"], dtype=float),
                               np.array(dfile["Z"], dtype=float),
                               np.array(dfile["Y_final"], dtype=float))).T
            return table
        # loading correlation files
        if v_n.__contains__(".h5"):


            dfile = h5py.File(fpath, "r")
            v_ns = []
            for _v_n in dfile:
                v_ns.append(_v_n)
            assert len(v_ns) == 3
            assert "mass" in v_ns
            mass = np.array(dfile["mass"])
            v_ns.remove('mass')
            xarr = np.array(dfile[v_ns[0]])
            yarr = np.array(dfile[v_ns[1]])
            if len(xarr) == mass.shape[1] and len(yarr) == mass.shape[0]:
                table = UTILS.combine(xarr, yarr, mass)
            else:
                table = UTILS.combine(yarr, xarr, mass)
            return table

        raise ValueError("Loading data method for ourflow data is not found")

    def load_collated_data(self, v_n):
        fpath = Paths.ppr_sims + self.sim + '/' + "collated/" + v_n
        if not os.path.isfile(fpath):
            raise IOError("File not found: collated v_n:{} -> {}"
                          .format(v_n, fpath))

        data = np.loadtxt(fpath)
        return data

    def load_gw_data(self, v_n):
        fpath =  Paths.ppr_sims + self.sim + '/' + "waveforms/" + v_n
        if not os.path.isfile(fpath):
            raise IOError("File not found: gw v_n:{} -> {}"
                          .format(v_n, fpath))

        data = np.loadtxt(fpath)
        return np.array([data])

    def load_3d_data_old(self, v_n):
        # ispar, itpar, tpar = self.get_ittime("profiles", "prof")

        if not os.path.isdir(Paths.ppr_sims+self.sim+'/' + 'profiles/'):
            # print("No dir: {}".format())
            return np.zeros(0,)

        list_iterations = Paths.get_list_iterations_from_res_3d(self.sim, "profiles/")
        if len(list_iterations) == 0:
            return np.zeros(0,)

        # empty = np.array(list_iterations)
        # empty.fill(np.nan)
        # return empty

        time_arr = []
        data_arr = []

        if v_n == "disk_mass.txt":
            for it in list_iterations:
                fpath = Paths.ppr_sims+self.sim+'/'+"profiles/" + str(int(it))  + '/' + v_n
                time_ = self.get_time_for_it(it, "prof")
                time_arr.append(time_)
                if os.path.isfile(fpath):
                    data_ = np.float(np.loadtxt(fpath, unpack=True))
                    data_arr.append(data_)
                else:
                    data_arr.append(np.nan)
            data_arr = np.array(data_arr)
            time_arr = np.array(time_arr)
            assert len(data_arr) == len(time_arr)
            res = np.vstack((time_arr, data_arr))
        else:
            raise NameError("no name for : {}".format(v_n))
            # n = 0
            # for it in list_iterations:
            #     fpath = Paths.ppr_sims + self.sim + '/' + "profiles/" + str(int(it)) + '/' + v_n
            #     time_ = self.get_time_for_it(it, "prof")
            #     time_arr.append(time_)
            #     if os.path.isfile(fpath):
            #         data_ = np.float(np.loadtxt(fpath, unpack=True))
            #         data_arr.append(data_)
            #         n = len(data_[0,:])
            #     else:
            #         data_arr.append([])
            # if len(data_arr) > 0:
            #     data_arr = np.reshape(np.array(data_arr), newshape=(n, self.mjfile_ncols, len(time_arr)))
            # data_arr = np.array(data_arr)

        return res

    def load_3d_data(self, v_n, mask="disk"):
        # ispar, itpar, tpar = self.get_ittime("profiles", "prof")

        data_path = Paths.ppr_sims+self.sim+'/' + 'profiles/'

        if not os.path.isdir(data_path):
            # print("No dir: {}".format())
            print("\t No 3D data found for {}".format(self.sim))
            return np.zeros(0,), np.zeros(0,), np.zeros(0,)

        list_iterations = Paths.get_list_iterations_from_res_3d(self.sim, "profiles/")
        if len(list_iterations) == 0:
            print("\t empty list of iterations 3D data for {}".format(self.sim))
            return np.zeros(0,), np.zeros(0,), np.zeros(0,)

        # empty = np.array(list_iterations)
        # empty.fill(np.nan)
        # return empty

        time_arr = []
        data_arr = []
        iter_arr = []

        if v_n == "disk_mass.txt" or v_n == "remnant_mass.txt":
            for it in list_iterations:
                fpath = Paths.ppr_sims+self.sim+'/'+"profiles/" + str(int(it))  + '/' + mask + '/' + v_n
                time_ = self.get_time_for_it(it, "profiles", "prof")
                time_arr.append(time_)
                iter_arr.append(it)
                if os.path.isfile(fpath):
                    data_ = np.float(np.loadtxt(fpath, unpack=True))
                    data_arr.append(data_)
                else:
                    print("\tMissing {}".format(fpath))
                    data_arr.append(np.nan)
            data_arr = np.array(data_arr)
            time_arr = np.array(time_arr)
            iter_arr = np.array(iter_arr, dtype=int)
            assert len(data_arr) == len(time_arr)
            # res = (iter_arr, time_arr, data_arr)
        else:
            for it in list_iterations:
                fpath = Paths.ppr_sims+self.sim+'/'+"profiles/" + str(int(it)) + '/' + mask + '/' + v_n
                time_ = self.get_time_for_it(it, "profiles", "prof")
                time_arr.append(time_)
                iter_arr.append(it)
                if os.path.isfile(fpath):
                    data_ = np.loadtxt(fpath, unpack=True)
                    data_arr.append(data_)
                else:
                    print("not found: {}".format(fpath))
                    data_arr.append(np.empty(2,))
            # data_arr = np.array(data_arr)
            time_arr = np.array(time_arr)
            iter_arr = np.array(iter_arr, dtype=int)
            assert len(data_arr) == len(time_arr)
            # res = (iter_arr, time_arr, data_arr)


            # raise NameError("no name for : {}".format(v_n))


            # n = 0
            # for it in list_iterations:
            #     fpath = Paths.ppr_sims + self.sim + '/' + "profiles/" + str(int(it)) + '/' + v_n
            #     time_ = self.get_time_for_it(it, "prof")
            #     time_arr.append(time_)
            #     if os.path.isfile(fpath):
            #         data_ = np.float(np.loadtxt(fpath, unpack=True))
            #         data_arr.append(data_)
            #         n = len(data_[0,:])
            #     else:
            #         data_arr.append([])
            # if len(data_arr) > 0:
            #     data_arr = np.reshape(np.array(data_arr), newshape=(n, self.mjfile_ncols, len(time_arr)))
            # data_arr = np.array(data_arr)
        if len(iter_arr) == 0: print("warning! no data found: {} {}".format(self.sim, v_n))
        return iter_arr, time_arr, data_arr

    # -----------------------------------------------------------------------------
    def is_outflow_data_loaded(self, det, mask, v_n):
        data = self.matrix_outflow_data[self.i_det(det)][self.i_o_mask(mask)][self.i_fn_outflow(v_n)]
        if len(data) == 0:
            data = self.load_outflow_data(det, mask, v_n)
            self.matrix_outflow_data[self.i_det(det)][self.i_o_mask(mask)][self.i_fn_outflow(v_n)] = data

        data = self.matrix_outflow_data[self.i_det(det)][self.i_o_mask(mask)][self.i_fn_outflow(v_n)]
        if len(data) == 0:
            raise ValueError("Loading outflow data has failed. Array is empty. det:{} mask:{} v_n:{}"
                             .format(det, mask, v_n))

    def get_outflow_data(self, det, mask, v_n):
        self.check_v_n(v_n)
        self.check_det(det)
        self.check_outflow_mask(mask)
        self.is_outflow_data_loaded(det, mask, v_n)
        data = self.matrix_outflow_data[self.i_det(det)][self.i_o_mask(mask)][self.i_fn_outflow(v_n)]
        return data
    # --- --- #
    def is_collated_loaded(self, v_n):
        data = self.matrix_collated_data[self.i_fn_col(v_n)]
        if len(data) == 0:
            data = self.load_collated_data(v_n)
            self.matrix_collated_data[self.i_fn_col(v_n)] = data
        # --- #
        data = self.matrix_collated_data[self.i_fn_col(v_n)]
        if len(data) == 0:
            raise ValueError("Failed loading collated data. Array is empty. v_n:{}"
                             .format(v_n))

    def get_collated_data(self, v_n):
        self.check_v_n(v_n)
        self.is_collated_loaded(v_n)
        data = self.matrix_collated_data[self.i_fn_col(v_n)]
        return data
    # --- --- #
    def is_gw_data_loaded(self, v_n):
        data = self.matrix_gw_data[self.i_fn_gw(v_n)]
        if len(data) == 0:
            data = self.load_gw_data(v_n)
            self.matrix_gw_data[self.i_fn_gw(v_n)] = data

        data = self.matrix_gw_data[self.i_fn_gw(v_n)]
        # print(data)
        if len(data) == 0:
            raise ValueError("Failed to load GW data v_n:{}")

    def get_gw_data(self, v_n):
        self.check_v_n(v_n)
        self.is_gw_data_loaded(v_n)
        data = self.matrix_gw_data[self.i_fn_gw(v_n)]
        return data
    # --- --- #
    def is_3d_data_loaded(self, v_n, mask):
        data = self.matrix_3d_data[self.i_3d_mask(mask)][self.i_fn_3d(v_n)]
        if len(data) == 0:
            data = self.load_3d_data(v_n, mask)
            self.matrix_3d_data[self.i_3d_mask(mask)][self.i_fn_3d(v_n)] = data

        # data = self.matrix_3d_data[self.i_fn_3d(v_n)]
        # if len(data) == 0:
        #     raise ValueError("Failed to load 3D ")

    def get_3d_data(self, v_n, it=0, t=0, mask=''):
        self.check_v_n(v_n)
        self.check_3d_mask(mask)
        self.is_3d_data_loaded(v_n, mask)
        # data = self.matrix_3d_data[self.i_fn_3d(v_n)]
        if it == 0 and t == 0:
            iters, times, data = self.matrix_3d_data[self.i_3d_mask(mask)][self.i_fn_3d(v_n)]
            return iters, times, data
        elif (it == 0 and t == -1) or (it == -1 and t == 0):
            iters, times, data = self.matrix_3d_data[self.i_3d_mask(mask)][self.i_fn_3d(v_n)]
            data = data[UTILS.find_nearest_index(times, times.max())]
            return data
        elif it == 0 and t != 0:
            iters, times, data = self.matrix_3d_data[self.i_3d_mask(mask)][self.i_fn_3d(v_n)]
            if t > times.max(): raise ValueError("sim:{} t:{} > tmax:{}".format(self.sim, t, times.max()))
            if t < times.min(): raise ValueError("sim:{} t:{} < tmin:{}".format(self.sim, t, times.min()))
            data = data[UTILS.find_nearest_index(times, t)]
            # print(data.T.shape)
            return data
        elif it != 0 and t ==0:
            iters, times, data = self.matrix_3d_data[self.i_3d_mask(mask)][self.i_fn_3d(v_n)]
            if not it in iters:
                raise ValueError("Requested it:{} not in the list of iteration from 3D data loaded:\n{}"
                                 .format(it, iters))
            data = data[list(iters).index(it)]
            # print(data.T.shape)
            return data
        else:
            raise ValueError("use either it or t. Given both.")
    # -------------------------------------------------------------------------------


class COMPUTE_ARR(LOAD_FILES):

    def __init__(self, sim, add_mask=None):
        LOAD_FILES.__init__(self, sim, add_mask)

    def get_outflow_hist(self, det, mask, v_n):
        data = self.get_outflow_data(det, mask, "hist_{}.dat".format(v_n))
        return data.T

    def get_outflow_corr(self, det, mask, v_n1_vn2):
        data = self.get_outflow_data(det, mask, "corr_{}.h5".format(v_n1_vn2))
        return data

    def get_outflow_timecorr(self, det, mask, v_n):
        data = self.get_outflow_data(det, mask, "timecorr_{}.h5".format(v_n))
        return data

    #
    @staticmethod
    def get_nucleo_solar_normed(method='sum', dataset="old"):

        if dataset == "old":
            fpath = Paths.skynet + Files.solar_r
            As, Ys = np.loadtxt(fpath, unpack=True)
        elif dataset == "Prantzos2019":
            fpath = Paths.skynet + "solar_r_Prantzos2019.dat"
            new_table = np.zeros(10)
            with open(fpath) as f:
                lines = f.readlines()
                for line in lines[2:]:
                    elements = line.split()
                    #print(len(elements))
                    z = float(elements[0])
                    a = float(elements[1])
                    # name = elements[2] # str
                    n = float(elements[3])
                    s_Sne_2008 = float(elements[4])
                    r_Sne_2008 = float(elements[5])
                    s_Gor1999 = float(elements[6])
                    r_Cor1999 = float(elements[7])
                    s_Bis2014 = float(elements[8])
                    s_Prantzos2019 = float(elements[9])
                    r_Prantzos2019 = float(elements[10])
                    new_table = np.vstack((new_table, [z, a, n,
                                                       s_Sne_2008, r_Sne_2008, s_Gor1999, r_Cor1999,
                                                       s_Bis2014, s_Prantzos2019, r_Prantzos2019]))
            new_table = np.delete(new_table, 0, 0)
            As, Ys = new_table[:, 1], new_table[:, 2]*new_table[:, 9]
            Ys = Ys * 1.e-5 # normalize for skynet tables -- Crude! To be improved
        elif dataset == "Sneden2008":
            middle_table = np.zeros(4)
            fpath = Paths.skynet + "solar_r_Sneden2008.dat"
            with open(fpath) as f:
                lines = f.readlines()
                for line in lines[1:]:
                    elements = line.split()
                    name = elements[0]
                    z = float(elements[1])
                    a = float(elements[2])  # isotope
                    ns = float(elements[3])
                    nr = float(elements[4])
                    middle_table = np.vstack((middle_table, [z, a, ns, nr]))
            middle_table = np.delete(middle_table, 0, 0)
            As, Ys = middle_table[:, 1], middle_table[:, 3]
            Ys = Ys * 1.e-5 # normalize for skynet tables -- Crude! To be improved
        else: raise NameError("dataset (SOlar Nucleo) name: {} is not recognized".format(dataset))

        '''Sums all Ys for a given A (for all Z)'''
        Anrm = np.arange(As.max() + 1)
        Ynrm = np.zeros(int(As.max()) + 1)
        for i in range(Ynrm.shape[0]):  # changed xrange to range
            Ynrm[i] = Ys[As == i].sum()

        if method == 'sum':
            Ynrm /= np.sum(Ynrm)
            return Anrm, Ynrm
        else:
            raise NameError("Normalisation method '{}' for the solar is not recognized."
                            .format(method))


    def get_nucleo_outflow(self, det, mask, method ="Asol=195", solardataset="old"):
        """

        :param det:
        :param mask: I mask == "geo bern" -- i.e. of 2 musks, the yields would be combined
        :param method:
        :return:
        """
        As = []
        all_Ys = []
        for i_mask in mask.split(' '): # combining yeilds for many masks, like geo + wind
            arr = self.get_outflow_data(det, i_mask, "yields.h5")
            As = arr[:,0]
            all_Ys.append(arr[:,2])
        all_Ys = np.reshape(all_Ys, (len(mask.split(' ')), len(As)))
        Ys = np.sum(all_Ys, axis=0)

        '''Sums all Ys for a given A (for all Z)'''
        Anrm = np.arange(As.max() + 1)
        Ynrm = np.zeros(int(As.max()) + 1)
        for i in range(Ynrm.shape[0]):  # changed xrange to range
            Ynrm[i] = Ys[As == i].sum()
        #
        if method == '':
            return Anrm, Ynrm
        #
        elif method == 'sum':
            ''' Normalizes to a sum of all A '''
            norm = Ynrm.sum()
            Ynrm /= norm
            return Anrm, Ynrm
        #
        elif method == "Asol=195":
            ''' Normalize to the solar abundances of a given element'''
            a_sol, y_sol = self.get_nucleo_solar_normed("sum", solardataset)
            #
            element_a = int(method.split("=")[-1])
            if element_a not in a_sol: raise ValueError('Element: a:{} not in solar A\n{}'.format(element_a, a_sol))
            if element_a not in Anrm: raise ValueError('Element: a:{} not in a_arr\n{}'.format(element_a, Anrm))
            #
            delta = np.float(y_sol[np.where(a_sol == element_a)] / Ynrm[np.where(Anrm == element_a)])
            Ynrm *= delta
            #
            return Anrm, Ynrm
        else:
            raise NameError("Normalisation method '{}' for the simulation yields is not recognized"
                            .format(method))

    # ------

    @staticmethod
    def __d1order__(arr, dx):
        idx = 1. / (2 * dx)
        df = []
        for i in range(1, len(arr) - 1):
            # df.append(idx * idx * (arr[i + 1] + arr[i - 1] - 2 * arr[i]))
            df.append( idx * (arr[i + 1] - arr[i - 1]))
        return np.array(df)

    @staticmethod
    def __df__(arr_x, arr_y, reinterpolate="None"):
        #
        arr_x = np.array(arr_x)
        arr_y = np.array(arr_y)
        #
        if len(arr_x) == 0: return np.zeros(0,), np.zeros(0,)
        #
        if reinterpolate != "None":
            _arr_x = np.linspace(arr_x[0], arr_x[-1], len(arr_x))
            _arr_y = interpolate.InterpolatedUnivariateSpline(arr_x, arr_y)(_arr_x)
            # _arr_y = interpolate.interp1d(arr_x, arr_y, kind=reinterpolate)(_arr_x)
            arr_x = _arr_x
            arr_y = _arr_y
        #
        dx = (arr_x[-1] - arr_x[0]) / (1. * len(arr_x))
        x0 = arr_x[0] - dx
        xN = arr_x[-1] + dx
        #
        _f_ = interpolate.interp1d(arr_x, arr_y, kind="linear", fill_value="extrapolate")
        #
        arr_y = np.insert(arr_y, 0, _f_(x0))
        arr_y = np.append(arr_y, _f_(xN))
        #
        arr_x = np.insert(arr_x, 0, x0)
        arr_x = np.append(arr_x, xN)
        #
        d_arr_y = COMPUTE_ARR.__d1order__(arr_y, dx)
        arr_x = arr_x[1:-1]
        #
        assert len(d_arr_y) == len(arr_x)
        #
        return arr_x, d_arr_y

    def get_disk_mass(self):
        data = self.get_3d_data("disk_mass.txt")
        return data

    def get_remnant_mass(self):
        data = self.get_3d_data("remnant_mass.txt")
        return data

    def get_summed_disk_remn_mass(self):
        it, t1, m_disk = self.get_3d_data("disk_mass.txt")
        it, t2, m_remnant = self.get_3d_data("remnant_mass.txt")
        assert len(t1) == len(t2)
        return it, t1, m_disk + m_remnant

    def get_disk_hist(self, v_n, it=0, t=0, mask=''):
        data = self.get_3d_data("hist_{}.dat".format(v_n), it=it, t=t, mask=mask)
        return data

    def get_disk_mass_ave_par_evo(self, v_n, mask=''):
        _, iterations, times = self.get_ittime("profiles", "prof")
        final_times, final_its, final_vals = [], [], []
        for it, t in zip(iterations, times):
            data = self.get_disk_hist(v_n, it, mask=mask)
            # print(data.shape)
            if len(data) >= 2:
                final_its.append(it)
                final_times.append(t)
                if v_n == "theta":
                    data[0, :] -= np.pi / 2.
                    ave = (180. / np.pi) * np.sqrt(np.sum(data[1, :] * data[0, :] ** 2) / np.sum(data[1,:]))
                else:
                    ave = np.sum(data[0, :] * data[1, :]) / np.sum(data[1,:])
                final_vals.append(ave)
        if len(final_vals) > 2:
            return np.vstack((np.array(final_its),
                              np.array(final_times),
                              np.array(final_vals))).T
        else:
            print("\tNo hist data found: v_n:{} sim:{}".format(v_n, self.sim))
            return np.zeros((0, 0, 0))

    def get_disk_timecorr(self, v_n, mask=''):
        _, iterations, times = self.get_ittime("profiles", "prof")
        final_times, final_its, bins, masses = [], [], [], []
        for it, t in zip(iterations, times):
            data = self.get_disk_hist(v_n, it, mask=mask)
            if len(data) >= 2:
                final_its.append(it)
                final_times.append(t)
                bins = data[0, :]
                masses.append(data[1, :])
        if len(final_times) >= 2:
            final_times = np.array(final_times)
            bins = np.array(bins)
            masses = np.reshape(np.array(masses), newshape=(len(final_its), len(bins))).T
            return final_times, bins, masses
        else:
            return np.zeros(0,), np.zeros(0,), np.zeros(2,)

    def get_enclosed_mj(self, reshape=False):
        """
        Angualr momentum, mass, I, within a given shell
        To obtain the total quantity -- use
        #
        Jout_max = np.cumsum(J)
        #
        To account for inneficienty, use
        #
        Jout_fidu = np.cumsum(J * options_Jfac_fidu(rc))
        #
        :param options_Jfac_fidu:
        :return:
        """
        iterations, times, dataset = self.get_3d_data("MJ_encl.txt")
        dic = {}
        for i, t in enumerate(times):
            dic[t] = dataset[i]
        #
        times = np.sort(times)
        #
        rc, drc = np.zeros(0,), np.zeros(0,)
        all_j, all_jf, all_m, all_i = [], [], [], []
        all_iterations, all_times = [], []
        for i, t in enumerate(times):
            if len(dataset[i]) <= 2:
                print("\tmissing MJ_encl.txt for it:{} sim:{}"
                      .format(iterations[i], self.sim))
            elif len(dataset[i]) != 6:
                print("\told MJ_encl.txt for it:{} sim:{}"
                      .format(iterations[i], self.sim))
            else:
                mj = np.array(dic[t]).T
                # print(mj.shape)
                rc, drc, Mb, Jb, Jfb, I = mj[:, 0], mj[:, 1], mj[:, 2], mj[:, 3], mj[:, 4], mj[:, 5]
                #
                tot_j_max   = np.array(Jb * rc * drc)   #[::-1]
                tot_jf      = np.array(Jfb * 2 * rc)       #[::-1]
                tot_m       = np.array(Mb * rc * drc)   #[::-1])
                tot_i       = np.array(I * rc * drc)    #[::-1])
                #
                all_j.append(tot_j_max)
                all_jf.append(tot_jf)
                all_m.append(tot_m)
                all_i.append(tot_i)
                all_iterations.append(iterations[i])
                all_times.append(times[i])
            #
        #
        all_iterations = np.array(all_iterations, dtype=int)
        all_times = np.array(all_times)
        #
        if not reshape:
            return all_iterations, all_times, rc, all_j, all_jf, all_m, all_i
        else:
            #
            all_j   = np.array(all_j)
            all_jf  = np.array(all_jf)
            all_m   = np.array(all_m)
            all_i   = np.array(all_i)
            #
            all_j   = np.reshape(all_j, (len(all_iterations), len(rc)))
            all_jf  = np.reshape(all_jf,(len(all_iterations), len(rc)))
            all_m   = np.reshape(all_m, (len(all_iterations), len(rc)))
            all_i   = np.reshape(all_i, (len(all_iterations), len(rc)))
            #
            return all_iterations, all_times, rc, all_j, all_jf, all_m, all_i

    def get_total_enclosed_j_jf_mb(self, extraction_radius=None):

        iterations, times, rc, all_j, all_jf, all_m, all_i = \
            self.get_enclosed_mj(reshape=False)
        tot_j, tot_jf, tot_mb = [], [], []
        for i in range(len(iterations)):
            #
            if extraction_radius != None:
                if isinstance(extraction_radius, str):
                    sign = extraction_radius[0]
                    val = float(''.join(extraction_radius[1:]))
                    if sign == ">": mask = rc > val
                    elif sign == "<": mask = rc <= val
                    else: raise NameError("Unknown sign: {}".format(sign))
                    #
                    tot_j.append(np.cumsum(all_j[i][mask])[-1])  # -1 - outer shell
                    tot_jf.append(all_jf[i][mask][-1])  # flux through outer most shell (r=500)
                    tot_mb.append(np.cumsum(all_m[i][mask])[-1])
                    #
                else:
                    idx = find_nearest_index(rc, extraction_radius)
                    #
                    tot_j.append(np.cumsum(all_j[i])[idx])  # -1 - outer shell
                    tot_jf.append(all_jf[i][idx])  # flux through outer most shell (r=500)
                    tot_mb.append(np.cumsum(all_m[i])[idx])
            else:
                idx = -1
                #
                tot_j.append(np.cumsum(all_j[i])[idx]) # -1 - outer shell
                tot_jf.append(all_jf[i][idx]) # flux through outer most shell (r=500)
                tot_mb.append(np.cumsum(all_m[i])[idx])
        #
        tot_j = np.array(tot_j)
        tot_jf = np.array(tot_jf)
        tot_mb = np.array(tot_mb)
        #
        return iterations, times, tot_j, tot_jf, tot_mb

    def get_total_enclosed_djdt(self, reinterpolate="None", extraction_radius=None):

        iterations, times, tot_j, tot_jf, tot_mb = \
            self.get_total_enclosed_j_jf_mb(extraction_radius)
        # convert times into GEO units (for the dJ/dt to be in the same units)


        times_ms = times * 1e3
        times_geo = times_ms / 0.004925794970773136

        #


        # for t1, t2 in zip(times_ms, times_geo):
        #     print("t1: {:.1f} t2: {:.1f}".format(t1,t2))
        # exit(1)

        assert len(times) == len(tot_j)
        ts, djevo = self.__df__(times_geo, tot_j, reinterpolate)
        ts = ts * 0.004925794970773136 / 1e3 # convert time back to seconds

        return ts, djevo

    def get_total_mass(self):
        tmp = self.get_collated_data("dens.norm1.asc")
        t_total_mass, dens = tmp[:, 1], tmp[:, 2]
        t_total_mass = t_total_mass * Constants.time_constant / 1000  # [s]
        m_tot = dens * Constants.volume_constant ** 3
        return np.vstack((t_total_mass, m_tot)).T

    def get_tot_unb_mass(self):
        tmp2 = self.get_collated_data("dens_unbnd.norm1.asc")
        t_unb_mass, dens_unb = tmp2[:, 1], tmp2[:, 2]
        t_unb_mass *= Constants.time_constant / 1000
        unb_mass = dens_unb * (Constants.volume_constant ** 3)
        return np.vstack((t_unb_mass, unb_mass)).T

    def get_unb_bern_mass(self):
        tmp2 = self.get_collated_data("dens_unbnd_bernoulli.norm1.asc")
        t_unb_mass, dens_unb = tmp2[:, 1], tmp2[:, 2]
        t_unb_mass *= Constants.time_constant / 1000
        unb_mass = dens_unb * (Constants.volume_constant ** 3)
        return np.vstack((t_unb_mass, unb_mass)).T

    def get_fpeak(self):

        data = self.get_gw_data("postmerger_psd_l2_m2.dat").T

        f, hr, hi = data[:,0], data[:,1], data[:,2]
        idx = f > 0.0
        f, h = f[idx], np.sqrt(f[idx]) * np.sqrt(hr[idx] ** 2 + hi[idx] ** 2)
        ipeak = np.argmax(np.abs(h))

        fpeak = f[ipeak]
        return fpeak


    # --- meta


class ALL_PAR(COMPUTE_ARR):

    def __init__(self, sim, add_mask=None):

        self.o_init = LOAD_INIT_DATA(sim)

        COMPUTE_ARR.__init__(self, sim, add_mask)

    def get_time_data_arrs(self, v_n, det=None, mask=None):
        if v_n == "Mdisk3D":
            its, times, masses = self.get_disk_mass()
            return times, masses
        elif v_n == "Mej":
            table = self.get_outflow_data(det=det, mask=mask, v_n="total_flux.dat")
            #print('time', len(table[:,0]))
            #print('mass', len(table[:, 2]))
            return table[:,0], table[:,2]
        elif v_n == "Mej_tot-bern_geoend":
            table = self.get_outflow_data(det=0, mask="bern_geoend", v_n="total_flux.dat")
            print('time', len(table[:, 0]))
            print('mass', len(table[:, 2]))
            return table[:, 0], table[:, 2]
        elif v_n == "vel_inf_ave-bern_geoend":
            # table = self.get_outflow_data(det=0, mask="geo", v_n="total_flux.dat")
            # print(table[:,0])
            table = self.get_outflow_timecorr(det=0, mask="bern_geoend", v_n="vel_inf")
            time_arr = table[0, 1:] * 1.e-3 # for some reason it is in ms #
            vel_inf = table[1:, 0]
            masses = table[1:, 1:]
            assert len(time_arr) == len(masses[0, :])
            assert len(vel_inf) == len(masses[:,0 ])

            vinf_aves = []
            for i in range(len(time_arr)):
                # compute average of ejecta cumulativly step by step for every timestep
                vinf_aves.append(np.sum(np.cumsum(masses, axis=1)[:, i] * vel_inf) / np.cumsum(np.sum(masses, axis=0))[i])
            vinf_aves = np.array(vinf_aves)

            # for t, m, v in zip(time_arr, np.cumsum(np.sum(masses,axis=0)), vinf_aves):
            #     print(t, m, v)
            # exit(1)
            # for i, t in enumerate(time_arr):
            #     t_masses = np.cumsum(masses[:, :i], axis=1)
            # exit(1)

            # if self.sim == "BLh_M11041699_M0_LK_LR":
            #     print(time_arr, vinf_aves)
            # exit(1)

            return time_arr[~np.isnan(vinf_aves)], vinf_aves[~np.isnan(vinf_aves)]
        elif v_n == "Ye_ave-bern_geoend":
            table = self.get_outflow_timecorr(det=0, mask="bern_geoend", v_n="Y_e")
            time_arr = table[0, 1:] * 1.e-3  # for some reason it is in ms #
            assert time_arr.max() < 1. and time_arr.max() > 0.01
            ye = table[1:, 0]
            masses = table[1:, 1:]
            assert len(time_arr) == len(masses[0, :])
            assert len(ye) == len(masses[:, 0])

            ye_aves = []
            for i in range(len(time_arr)):
                # compute average of ejecta cumulativly step by step for every timestep
                ye_aves.append(
                    np.sum(np.cumsum(masses, axis=1)[:, i] * ye) / np.cumsum(np.sum(masses, axis=0))[i])
            ye_aves = np.array(ye_aves)

            return time_arr[~np.isnan(ye_aves)], ye_aves[~np.isnan(ye_aves)]
        elif v_n == "theta_rms-bern_geoend":
            table = self.get_outflow_timecorr(det=0, mask="bern_geoend", v_n="theta")
            time_arr = table[0, 1:] * 1.e-3  # for some reason it is in ms #
            assert time_arr.max() < 1. and time_arr.max() > 0.01
            theta = table[1:, 0]
            masses = table[1:, 1:]
            assert len(time_arr) == len(masses[0, :])
            assert len(theta) == len(masses[:, 0])

            theta_rmss = []
            for i in range(len(time_arr)):
                # compute average of ejecta cumulativly step by step for every timestep
                theta_rmss.append(
                    np.sqrt(np.sum(np.cumsum(masses, axis=1)[:, i] * theta ** 2)) \
                    / np.cumsum(np.sum(masses, axis=0))[i])
            theta_rmss = np.array(theta_rmss)

            # for t, m, theta in zip(time_arr, np.cumsum(np.sum(masses,axis=0)), theta_rmss):
            #     print(t, m, theta)
            # print(theta_rmss[~np.isnan(theta_rmss)])
            # exit(1)


            return time_arr[~np.isnan(theta_rmss)], theta_rmss[~np.isnan(theta_rmss)]

        else:
            raise NameError("v_n: {} is not recognized. No method setup."
                            .format(v_n))

    def get_outflow_par(self, det, mask, v_n):

        if v_n == "Mej_tot":
            data = np.array(self.get_outflow_data(det, mask, "total_flux.dat"))
            mass_arr = data[:, 2]
            res = mass_arr[-1]
        elif v_n == "Ye_ave":
            mej = self.get_outflow_par(det, mask, "Mej_tot")
            hist = self.get_outflow_hist(det, mask, "Y_e").T
            res = EJECTA_PARS.compute_ave_ye(mej, hist)
        elif v_n == "s_ave":
            mej = self.get_outflow_par(det, mask, "Mej_tot")
            hist = self.get_outflow_hist(det, mask, "entropy").T
            res = EJECTA_PARS.compute_ave_s(mej, hist)
        elif v_n == "vel_inf_ave":
            mej = self.get_outflow_par(det, mask, "Mej_tot")
            hist = self.get_outflow_hist(det, mask, "vel_inf").T
            res = EJECTA_PARS.compute_ave_vel_inf(mej, hist)
        elif v_n == "E_kin_ave":
            mej = self.get_outflow_par(det, mask, "Mej_tot")
            hist = self.get_outflow_hist(det, mask, "vel_inf").T
            res = EJECTA_PARS.compute_ave_ekin(mej, hist)
        elif v_n == "theta_rms":
            hist = self.get_outflow_hist(det, mask, "theta").T
            res = EJECTA_PARS.compute_ave_theta_rms(hist)
        elif v_n == "tend":
            data = np.array(self.get_outflow_data(det, mask, "total_flux.dat"))
            mass_arr = data[:, 0]
            res = mass_arr[-1]
        elif v_n == "t98mass":
            data = np.array(self.get_outflow_data(det, mask, "total_flux.dat"))
            mass_arr = data[:, 2]
            time_arr = data[:, 0]
            fraction = 0.98
            i_t98mass = int(np.where(mass_arr >= fraction * mass_arr[-1])[0][0])
            # print(i_t98mass)
            assert i_t98mass < len(time_arr)
            res = time_arr[i_t98mass]
        else:
            raise NameError("no method for estimation det:{} mask:{} v_n:{}"
                            .format(det, mask, v_n))
        return res

    def get_initial_data_par(self, v_n):

        val = self.o_init.get_par(v_n)

        return val

    def get_par(self, v_n):

        if v_n == "tcoll_gw":
            try:
                data = self.get_gw_data("tcoll.dat")
            except IOError:
                Printcolor.yellow("\tWarning! No tcoll.dat found for sim:{}".format(self.sim))
                return np.inf
            #
            try:
                mtot = self.o_init.get_par("M1") + self.o_init.get_par("M1")
            except:
                Printcolor.yellow("\tWarning! Failed to get mtot from init.data using M_Inf=2.748")
                mtot = 2.748
            ret_time = PHYSICS.get_retarded_time(data, M_Inf=mtot, R_GW=self.r_gw)
            # tcoll = ut.conv_time(ut.cactus, ut.cgs, ret_time)
            return float(ret_time * Constants.time_constant * 1e-3)
        elif v_n == "tend":
            total_mass = self.get_total_mass()
            t, Mtot = total_mass[:, 0], total_mass[:, 1]
            # print(t)
            return t[-1]
        elif v_n == "tmerg" or v_n == "tmerger" or v_n == "tmerg_r":
            try:
                data = self.get_gw_data("tmerger.dat")
            except IOError:
                Printcolor.yellow("\tWarning! No tmerger.dat found for sim:{}".format(self.sim))
                return np.nan
            try:
                mtot = self.o_init.get_par("M1") + self.o_init.get_par("M1")
            except:
                Printcolor.yellow("\tWarning! Failed to get mtot from init.data using M_Inf=2.748")
                mtot = 2.748
            ret_time = PHYSICS.get_retarded_time(data, M_Inf=mtot, R_GW=self.r_gw)
            return float(ret_time * Constants.time_constant * 1e-3)
        elif v_n == "tcoll" or v_n == "Mdisk":
            total_mass = self.get_total_mass()
            unb_mass = self.get_tot_unb_mass()
            t, Mtot = total_mass[:, 0]*Constants.time_constant*1e-3, total_mass[:, 1]
            _, Munb = unb_mass[:, 0]*Constants.time_constant*1e-3, unb_mass[:, 1]
            # print(Mtot.min()); exit(1)
            if Mtot[-1] > 1.0:
                Mdisk = np.nan
                tcoll = np.inf
                Printcolor.yellow("Warning: Mtot[-1] > 1 Msun. -> Either no disk or wrong .ascii")
            else:
                i_BH = np.argmin(Mtot > 1.0)
                tcoll = t[i_BH]  # *1000 #* utime
                i_disk = i_BH + int(1.0 / (t[1] * 1000))  #
                # my solution to i_disk being out of bound:
                if i_disk > len(Mtot): i_disk = len(Mtot) - 1
                if i_disk > len(Munb): i_disk = len(Munb) - 1
                Mdisk = Mtot[i_disk] - Munb[i_disk]
            if v_n == "tcoll":
                return tcoll
            else:
                return Mdisk
        elif v_n == "Munb_tot":
            unb_mass = self.get_tot_unb_mass()
            _, Munb = unb_mass[:, 0], unb_mass[:, 1]
            print(unb_mass.shape)
            return Munb[-1]
        elif v_n == "Munb_bern_tot":
            unb_mass = self.get_unb_bern_mass()
            _, Munb = unb_mass[:, 0], unb_mass[:, 1]
            return Munb[-1]
        elif v_n == "Mdisk3D":
            itersations, times, masses = self.get_disk_mass()

            if len(itersations) > 0:
                return masses[-1]
            else:
                return np.nan
        elif v_n == "Mdisk3Dmax":
            itersations, times, masses = self.get_disk_mass()
            if len(itersations) > 0:
                return np.array(masses).max()
            else:
                return np.nan
        elif v_n == "tdisk3D":
            itersations, times, masses = self.get_disk_mass()
            if len(itersations) > 0:
                return times[-1]
            else:
                return np.nan
        elif v_n == "tdisk3Dmax":
            itersations, times, masses = self.get_disk_mass()
            if len(itersations) > 0:
                return times[UTILS.find_nearest_index(masses, masses.max())]
            else:
                return np.nan
        elif v_n == "fpeak":
            try:
                data = self.get_gw_data("postmerger_psd_l2_m2.dat").T
            except IOError:
                Printcolor.yellow("File not found: {} in sim: {}"
                                  .format("postmerger_psd_l2_m2.dat", self.sim))
                return np.nan
            f, hr, hi = data[:, 0], data[:, 1], data[:, 2]
            idx = f > 0.0
            f, h = f[idx], np.sqrt(f[idx]) * np.sqrt(hr[idx] ** 2 + hi[idx] ** 2)
            ipeak = np.argmax(np.abs(h))
            fpeak = f[ipeak]
            return fpeak
        elif v_n == "EGW" or v_n == "JGW":
            try:
                data = self.get_gw_data("EJ.dat").T
            except IOError:
                Printcolor.yellow("File not found: {} in sim: {}"
                                  .format("EJ.dat", self.sim))
                return np.nan
            # print(data.shape)
            # data = data[:, :]
            # print(data.shape)
            # tmerg = self.get_par("tmerg") # retarded
            # t, EGW, JGW = data[:, 0], data[:, 2], data[:, 4]
            t, EGW, JGW = data[0, :], data[2, :], data[4, :]
            # print(EGW)
            # print(EGW)
            # print(data)
            idx = -1
            egw, jgw = EGW[idx], JGW[idx] # total emitted
            if v_n == "EGW": return float(egw)
            else: return float(jgw)
        elif v_n == "EGW20" or v_n == "JGW20":
            try:
                data = self.get_gw_data("EJ.dat").T
            except IOError:
                Printcolor.yellow("File not found: {} in sim: {}"
                                  .format("EJ.dat", self.sim))
                return np.nan
            tmerg = self.get_par("tmerg") # retarded
            # t, EGW, JGW = data[:, 0], data[:, 2], data[:, 4]
            t, EGW, JGW = data[0, :], data[2, :], data[4, :]
            idx = np.argmin(np.abs(t - tmerg - 20.0))
            egw, jgw = EGW[idx], JGW[idx] # emitted by some time
            if v_n == "EGW": return float(egw)
            else: return float(jgw)
        elif v_n == "nprofs":
            _, itpros, timeprofs = self.get_ittime("profiles", "prof")
            return len(itpros)
        elif v_n == "nnuprofs":
            _, itpros, timeprofs = self.get_ittime("nuprofiles", "nuprof")
            return len(itpros)
        else:
            raise NameError("no parameter found for v_n:{}".format(v_n))


class ADD_METHODS_ALL_PAR(ALL_PAR):

    def __init__(self, sim, add_mask=None):
        ALL_PAR.__init__(self, sim, add_mask)

    def get_int_par(self, v_n, t):

        if v_n == "Mdisk3D":
            dislmasses = self.get_disk_mass()
            if len(dislmasses) == 0:
                Printcolor.red("no disk mass data found (empty get_disk_mass()): "
                               "{}".format(self.sim))
                return np.nan
                # raise ValueError("no disk mass data found")
            tarr = dislmasses[:,0]
            marr = dislmasses[:,1]
            if t > tarr.max():
                raise ValueError("t: {} is above DiskMass time array max: {}, sim:{}"
                                 .format(t, tarr.max(), self.sim))
            if t < tarr.min():
                raise ValueError("t: {} is below DiskMass time array min: {}, sim:{}"
                                 .format(t, tarr.min(), self.sim))
            f = interpolate.interp1d(tarr, marr, kind="linear", bounds_error=True)
            return f(t)


class AVERAGE_PAR():

    def __init__(self, sims):
        assert len(sims) > 0
        self.sims = sims
        self.data_dic = {
            "SR":None,
            "HR":None,
            "LR":None
        }
        for sim in sims:
            if sim.__contains__("LR"):
                self.data_dic["LR"] = ADD_METHODS_ALL_PAR(sim)
            elif sim.__contains__("HR"):
                self.data_dic["HR"] = ADD_METHODS_ALL_PAR(sim)
            elif sim.__contains__("SR"):
                self.data_dic["SR"] = ADD_METHODS_ALL_PAR(sim)
            else:
                raise NameError("sim resolution is nor recognized: {}"
                                .format(sim))
        self.res_dic = {}
        self.res_outflow_dic = {}
        # settings
        self.Mej_min        = 5e-5
        self.Mej_err        = lambda Mej: 0.5 * Mej + self.Mej_min
        self.Yeej_err       = lambda Ye: 0.01
        self.vej_err        = lambda v: 0.02
        self.Sej_err        = lambda Sej: 1.5
        self.theta_ej_err   = lambda theta_ej: 2.0
        self.MdiskPP_min    = 5e-4
        self.MdiskPP_err    = lambda MdiskPP: 0.5 * MdiskPP + self.MdiskPP_min
        self.Anrm_range     = [180, 200]
        self.vej_fast       = 0.6
        self.Mej_fast_min   = 1e-8
        self.Mej_fast_err   = lambda Mej_fast: 0.5 * Mej_fast + self.Mej_fast_min
        self.tend_min       = 10.  # ms
        #
        self.error_multplier_for_1res = 1
        self.error_multplier_for_2res = 2
        #

    def __get_err_for_v_n(self, v_n, val):
        if v_n == "Mej":
            return self.Mej_err(val)
        if v_n == "Mej_tot-bern_geoend":
            return self.Mej_err(val)
        if v_n == "vel_inf_ave-bern_geoend":
            return self.vej_err(val)
        if v_n == "Ye_ave-bern_geoend":
            return self.Yeej_err(val)
        if v_n == "theta_rms-bern_geoend":
            return self.theta_ej_err(val)
        if v_n in ["Mdisk", "Mdisk3D", "Mdisk3Dmax"]:
            return self.MdiskPP_err(val)
        if v_n == "s_ave":
            return self.Sej_err(val)
        if v_n == "Ye_ave":
            return self.Yeej_err(val)
        if v_n == "theta_ave":
            return self.theta_ej_err(val)
        raise NameError("v_n: {} has no error setup"
                        .format(v_n))

    def __get_list_vals(self, v_n):
        vals = []
        for res in self.data_dic.keys():
            if self.data_dic[res] != None:
                vals.append(self.data_dic[res].get_par(v_n))
        return vals

    # def get_ave_val(self, v_n, method = 'st.div'):
    #     #
    #     if v_n in self.res_dic.keys():
    #         return self.res_dic[v_n]
    #     #
    #     if method == "st.div":
    #         if len(self.sims) == 3:
    #             vals = self.__get_list_vals(v_n)
    #             mean, err1, err2 = standard_div(vals)
    #             self.res_dic[v_n] = (mean, err1, err2)
    #         elif len(self.sims) == 2:
    #             vals = self.__get_list_vals(v_n)
    #             mean, err1, err2 = standard_div(vals)
    #             err1 = self.error_multplier_for_2res * err1
    #             err2 = self.error_multplier_for_2res * err2
    #             self.res_dic[v_n] = (mean, err1, err2)
    #         elif len(self.sims) == 1:
    #             vals = self.__get_list_vals(v_n)[0]
    #             err1 = err2 = self.__get_err_for_v_n(v_n, vals)
    #             self.res_dic[v_n] = (vals[0], err1, err2)
    #         return self.res_dic[v_n]
    #     else:
    #         raise NameError("method: {} is not recognized".format(method))

    def __get_list_times_vals_for_last_common_time(self, v_n,
                                                   substract_tmerg=False,
                                                   substract_tcoll=False):
        #
        times, vals = [], []
        #
        for res in self.data_dic.keys():
            if self.data_dic[res] != None:
                _times, _vals = self.data_dic[res].get_time_data_arrs(v_n)
                if substract_tmerg and not substract_tcoll:
                    tmerg = self.data_dic[res].get_par("tmerg")
                    _times = _times - tmerg
                elif substract_tcoll and not substract_tmerg:
                    tcoll = self.data_dic[res].get_par("tcoll_gw")
                    _times = _times - tcoll
                elif substract_tcoll and substract_tmerg:
                    raise NameError("use either 'substract_tmerg' or 'substract_tcoll', give: {} {}"
                                    .format(substract_tmerg, substract_tcoll))
                times.append(_times)
                vals.append(_vals)
        return times, vals

    def get_vals_last_common_time(self, v_n,
                                     substract_tmerg=True,
                                     substract_tcoll=False,
                                     interpolemethod="linear",
                                     extrapolate=True):
        """
        :param v_n:
        :param substract_tmerg:
        :param substract_tcoll:
        :param interpolemethod:
        :param extrapolate:
        :return: [maxtime, (val1, val2, val3)]
        """
        #
        res_time = np.nan
        res_vals = (np.nan, np.nan, np.nan)
        #
        if v_n in self.res_dic.keys():
            return self.res_dic[v_n]
        #
        times, vals = self.__get_list_times_vals_for_last_common_time(v_n,
                                                                      substract_tmerg=substract_tmerg,
                                                                      substract_tcoll=substract_tcoll)
        #
        assert len(times) == len(vals)
        #
        if len(times) == 0:
            raise ValueError("no data loaded")
        #
        maxtimes = []
        for _times in times:
            if len(_times) > 0:
                maxtime = np.max(_times)
                assert np.isfinite(maxtime)
                maxtimes.append(maxtime)
        #
        if len(maxtimes) == 0:
            print("\tWarning. No maxtimes (3D data) found for v_n:{} for sims:\n{}"
                  .format(v_n, self.sims))
            #self.res_dic[v_n] = (np.nan, np.nan, np.nan)
        #
        if len(maxtimes) == 1:
            if len(self.sims) != 1:
                print("\tWarning. 1 maxtimes (3D data) found for v_n:{} for NOT 1 sims:\n{}"
                      .format(v_n, self.sims))
            # extract 1 value at maximum time (if postmerger, if postcollapse)
            # print(times)
            # print(vals)
            _times = np.concatenate(np.array(times))
            _vals = np.concatenate(np.array(vals))
            # assert len(_times) == len(_vals)
            # err = self.__get_err_for_v_n(v_n, _vals.max())
            #
            res_time = _times.max()
            res_vals = _vals[np.where(_times == res_time)]#  .max()
        #
        if len(maxtimes) == 2 or len(maxtimes) == 3:
            #
            if len(maxtimes) == 2 and len(self.sims) != 2:
                print("\tWarning. 2 maxtimes (3D data) found for v_n:{} for NOT 2 sims:\n{}"
                  .format(v_n, self.sims))
            #
            if len(maxtimes) == 3 and len(self.sims) != 3:
                print("\tWarning. 3 maxtimes (3D data) found for v_n:{} for NOT 3 sims:\n{}"
                  .format(v_n, self.sims))
            # extract 2 values at maximum common time (if both postmerger and if both postcollapse)
            mmaxtime = float(np.min(maxtimes))
            #
            int_vals = []
            for _times, _vals in zip(times, vals):

                if len(_times) > 0:
                    _times, _vals = np.array(_times), np.array(_vals)
                    #
                    # print(len(_times))
                    if _times.min() <= mmaxtime and _times.max() >= mmaxtime:

                        if len(_times) == 1:
                            # TODO if _time[0] == mmaxtime: just append times and mass, do not interpoalte
                            if _times[0] == mmaxtime:
                                int_vals.append(_vals[0])
                            else:
                                int_vals.append(np.nan)
                        else:
                            assert len(_times) == len(_vals)
                            assert len(_times) >= 2
                            f = interpolate.interp1d(_times, _vals, kind=interpolemethod)
                            int_val = f(mmaxtime)
                            int_vals.append(int_val)
                    elif mmaxtime < _times.min() or mmaxtime > _times.max():
                        if len(_times) == 1:
                            int_vals.append(np.nan)
                        else:
                            if extrapolate:
                                assert len(_times) == len(_vals)
                                assert len(_times) >= 2
                                print("\tWarning: mmaxtime:{:.3f} is above max:{:.3f} or below min:{:.3f}: Extrapolating \n\t{}"
                                        .format(mmaxtime, _times.max(), _times.min(), self.sims))
                                f = interpolate.interp1d(_times, _vals, kind=interpolemethod, fill_value="extrapolate")
                                int_val = f(mmaxtime)
                                int_vals.append(int_val)
                            else:
                                print(
                                    "\tWarning: mmaxtime:{:.3f} is above max:{:.3f} or below min:{:.3f} Appending nan \n\t{}"
                                    .format(mmaxtime, _times.max(), _times.min(), self.sims))
                                int_vals.append(np.nan)
                    else:
                        raise ValueError("something is wrong "
                                         "mmaxtime:{:.3f} _times.min():{:.3f} _times.max():{:.3f} "
                                         .format(mmaxtime, _times.min(), _times.max()))
            #
            res_time = mmaxtime
            res_vals = int_vals
        #
        if len(maxtimes) > 3:
            raise ValueError("More than 3 maxtimes: {} for sims: \n{}"
                             .format(maxtimes, self.sims))

        return res_time, res_vals

    def get_ave_val_last_common_time(self, v_n, method="st.div",
                                     substract_tmerg=True,
                                     substract_tcoll=False,
                                     interpolemethod="linear",
                                     extrapolate=True):
        """

        :param v_n:
        :param method:
        :param substract_tmerg:
        :param substract_tcoll:
        :param interpolemethod:
        :param extrapolate:
        :return: [time, value, err1, err2]
        """
        try:
            maxtime, vals = self.get_vals_last_common_time(v_n,
                                         substract_tmerg,
                                         substract_tcoll,
                                         interpolemethod,
                                         extrapolate)
        except IOError:
            print("IOError: sims: {} v_n: {}".format(self.sims, v_n))
            return np.nan, np.nan, np.nan, np.nan
        except:
            print("-------------- ")
            raise AssertionError("something went wrong with getting ave for last common time \n"
                                 "sims: {} v_n: {}".format(self.sims, v_n))

        #
        vals = np.array(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        if method == "st.div":
            if len(vals) == 3:
                mean, err = standard_div(vals)
                return maxtime, mean, err, err
            elif len(vals) == 2:
                mean, err = standard_div(vals)
                err = err*self.error_multplier_for_2res
                return maxtime, mean, err, err
            elif len(vals) == 1:
                err = self.__get_err_for_v_n(v_n, vals[0])
                return maxtime, vals[0], err, err
            elif len(vals) == 0:
                return np.nan, np.nan, np.nan, np.nan
            else:
                raise ValueError("too many vals:{} for v_n:{}".format(vals, v_n))
        else:
            raise NameError("method:{} is not recognized".format(method))

    # TODO #

    def get_ave_val_spec_common_time(self, v_n, t, tpostmerger=False, method="st.div"):
        pass
    #
    def get_ave_val_close_to_last_time(self, v_n, delta_t,
                                       assurepostmerger=True,
                                       assurepostcollapse=False, method="st.div"):
        pass
    #
    def get_ave_outflow_par(self, det, mask, v_n, method="st.div"):
        pass


def test_plot_disk_masses_evol_and_average():

    output_dir = "../figs/all3/tst_disk_mass/"

    colors = {
        "SR": "black",
        "LR": "red",
        "HR": "blue"
    }
    lss = {
        "SR": '-',
        "LR": ':',
        "HR": '--'
    }

    alpha = 1.
    labels = True

    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    from model_sets.models import simulations_nonblacklisted

    simulations = simulations_nonblacklisted

    groups = sorted(list(set(simulations["group"])))
    for group in groups:
        print(" ---------------------------------------- ")
        sel = simulations[simulations["group"] == group]
        print(sel[["Mdisk3D", "tdisk3D"]])
        #
        o_group = AVERAGE_PAR(sel.index)
        tmmax, ave_mass, err1, err2 = o_group.get_ave_val_last_common_time("Mdisk3D", "st.div")
        #
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #
        for name, simdic in sel.iterrows():
            res = simdic["resolution"]
            color = colors[res]
            ls = lss[res]
            #
            o_data = ADD_METHODS_ALL_PAR(name)
            times, masses = o_data.get_time_data_arrs("Mdisk3D")
            if len(times)>0:
                tmerg = o_data.get_par("tmerg")
                times = times - tmerg
                #
                ax.plot(times * 1e3, masses, color=color, ls=ls, alpha=alpha, label = res)

        ax.plot(tmmax * 1e3, ave_mass, color="black", marker="+", alpha=1.)
        ax.errorbar(tmmax * 1e3, ave_mass, yerr=err1, label=None,
                    color='gray', ecolor='gray',
                    fmt='None', elinewidth=1, capsize=1, alpha=0.5)
        #
        ax.set_xlabel(r"$t-t_{\rm{merg}}$ [ms]")
        ax.set_ylabel(r"$M_{\rm{disk}}$ $[M_{\odot}]$")
        #
        ax.tick_params(axis='both', which='both', labelleft=True,
                       labelright=False, tick1On=True, tick2On=True,
                       labelsize=12,
                       direction='in',
                       bottom=True, top=True, left=True, right=True)
        ax.minorticks_on()

        ax.legend(fancybox=True, loc='upper center',
                  # bbox_to_anchor=(0.5, 0.5),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                  shadow=False, ncol=2, fontsize=9,
                  framealpha=0., borderaxespad=0.)

        print(output_dir + group + ".png")
        plt.savefig(output_dir + group + ".png", dpi=128)
        plt.close()




if __name__ == '__main__':

    o_data = ADD_METHODS_ALL_PAR("BLh_M13641364_M0_LK_SR")
    print(o_data.get_par("JGW"))

    # o_data = ADD_METHODS_ALL_PAR("BLh_M13641364_M0_LK_SR")
    # print(o_data.get_par("nprofs"))
    # tcoll = data.get_par("tcoll_gw")
    # tmerg = data.get_par("tmerg")
    # print("tmerg:{} tcoll:{}".format(tmerg, tcoll))

    # egw = data.get_par("EGW")
    # print(egw)
    # exit(1)

    # mj = data.get_3d_data("MJ_encl.txt", it=2211840).T
    # rc, drc, Mb, Jb = mj[:, 0], mj[:, 1], mj[:, 2], mj[:, 3]
    # print(mj.shape)

    # mdisk = data.get_par("Mdisk3D")
    # print(mdisk)
    #
    # it, t, masses = data.get_3d_data("disk_mass.txt")
    # print(["{:d}, {:.1f}, {:.2f} | ".format(it, time*1e3, mass) for it, time, mass in zip(it, t, masses)])
    print("\n")
    # group = AVERAGE_PAR(["DD2_M14971246_M0_LR", "DD2_M14971245_M0_SR", "DD2_M14971245_M0_HR"])
    # print(group.get_vals_last_common_time("Mdisk3D"))
    # print(group.get_ave_val_last_common_time("Mdisk3D"))

    # group = AVERAGE_PAR(["BLh_M10201856_M0_HR", "BLh_M10201856_M0_LR", "BLh_M10201856_M0_SR"])
    # print(group.get_vals_last_common_time("Mdisk3D"))
    # print(group.get_ave_val_last_common_time("Mdisk3D"))

    # group = AVERAGE_PAR(["BLh_M11841581_M0_LK_SR", "BLh_M11841581_M0_LK_LR"])
    # print(group.get_vals_last_common_time("Mdisk3D"))
    # print(group.get_ave_val_last_common_time("Mdisk3D"))

    # group = AVERAGE_PAR(["SFHo_M14521283_M0_HR","SFHo_M14521283_M0_LR", "SFHo_M14521283_M0_SR"])
    # print(group.get_vals_last_common_time("Mdisk3D"))
    # print(group.get_ave_val_last_common_time("Mdisk3D"))

    # test_plot_disk_masses_evol_and_average()

    #pars = AVERAGE_PAR(["BLh_M11041699_M0_LK_LR"])
    #print(pars.get_ave_val_last_common_time("theta_rms-bern_geoend"))








