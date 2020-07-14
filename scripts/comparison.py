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
from outflowed import EJECTA_PARS, outflowed_historgrams, outflowed_correlations, outflowed_totmass, outflowed_timecorr
from preanalysis import LOAD_ITTIME
from plotting_methods import PLOT_MANY_TASKS
from profile import LOAD_PROFILE_XYXZ, LOAD_RES_CORR, LOAD_DENSITY_MODES
from utils import Paths, Lists, Labels, Constants, Printcolor, UTILS, Files, PHYSICS

from data import *


class TWO_SIMS():

    def __init__(self, sim1, sim2):
        self.sim1 = sim1
        self.sim2 = sim2
        self.o_par1 = ADD_METHODS_ALL_PAR(self.sim1)
        self.o_par2 = ADD_METHODS_ALL_PAR(self.sim2)
        self.outflow_tasks = ["totflux", "hist"]

    def compute_outflow_new_mask(self, det, sim, mask, rewrite):

        # get_tmax60 # ms
        print("\tAdding mask:{}".format(mask))
        o_outflow = EJECTA_PARS(sim, add_mask=mask)

        if not os.path.isdir(Paths.ppr_sims + sim +"/" +"outflow_{:d}/".format(det) + mask + '/'):
            os.mkdir(Paths.ppr_sims + sim +"/" +"outflow_{:d}/".format(det) + mask + '/')

        Printcolor.blue("Creating new outflow mask dir:{}"
                        .format(sim +"/" +"outflow_{:d}/".format(det) + mask + '/'))

        for task in self.outflow_tasks:
            if task == "hist":
                # from outflowed import outflowed_historgrams
                outflowed_historgrams(o_outflow, [det], [mask], o_outflow.list_hist_v_ns, rewrite=rewrite)
            elif task == "corr":
                # from outflowed import outflowed_correlations
                outflowed_correlations(o_outflow, [det], [mask], o_outflow.list_corr_v_ns, rewrite=rewrite)
            elif task == "totflux":
                # from outflowed import outflowed_totmass
                outflowed_totmass(o_outflow, [det], [mask], rewrite=rewrite)
            elif task == "timecorr":
                # from outflowed import outflowed_timecorr
                outflowed_timecorr(o_outflow, [det], [mask], o_outflow.list_hist_v_ns, rewrite=rewrite)
            else:
                raise NameError("method for computing outflow with new mask is not setup for task:{}".format(task))

    def get_post_geo_delta_t(self, det):

        # o_par1 = ALL_PAR(self.sim1)
        # o_par2 = ALL_PAR(self.sim2)

        # tmerg1 = self.o_par1.get_par("tmerger")
        # tmerg2 = self.o_par2.get_par("tmerger")

        t98geomass1 = self.o_par1.get_outflow_par(det, "geo", "t98mass")
        t98geomass2 = self.o_par2.get_outflow_par(det, "geo", "t98mass")

        tend1 = self.o_par1.get_outflow_par(det, "geo", "tend")
        tend2 = self.o_par2.get_outflow_par(det, "geo", "tend")

        if tend1 < t98geomass1:
            Printcolor.red("tend1:{} < t98geomass1:{}".format(tend1, t98geomass1))
            return np.nan
        if tend2 < t98geomass2:
            Printcolor.red("tend2:{} < t98geomass2:{}".format(tend2, t98geomass2))
            return np.nan

        Printcolor.yellow("Relaxing the time limits criteria")
        # if tend1 < t98geomass2:
        #     Printcolor.red("Delta t does not overlap tend1:{} < t98geomass2:{}".format(tend1, t98geomass2))
        #     return np.nan
        # if tend2 < t98geomass1:
        #     Printcolor.red("Delta t does not overlap tend2:{} < t98geomass1:{}".format(tend2, t98geomass1))
        #     return np.nan
        # assert tmerg1 < t98geomass1
        # assert tmerg2 < t98geomass2

        # tend1 = tend1 - tmerg1
        # tend2 = tend2 - tmerg2
        # t98geomass1 = t98geomass1 - tmerg1
        # t98geomass2 = t98geomass2 - tmerg2

        delta_t1 = tend1 - t98geomass1
        delta_t2 = tend2 - t98geomass2

        print("\tTime window for bernoulli ")
        print("\t{} {:.2f} [ms]".format(self.sim1, delta_t1*1e3))
        print("\t{} {:.2f} [ms]".format(self.sim2, delta_t2*1e3))
        # exit(1)

        delta_t = np.min([delta_t1, delta_t2])

        if delta_t < 0.005:
            return np.nan# ms

        return delta_t

    def get_tmax_d3_data(self):

        isd3_1, itd3_1, td3_1 = self.o_par1.get_ittime("profiles", "prof")
        isd3_2, itd3_2, td3_2 = self.o_par2.get_ittime("profiles", "prof")

        if len(td3_1) == 0:
            Printcolor.red("D3 data not found for sim1:{}".format(self.sim1))
            return np.nan
        if len(td3_2) == 0:
            Printcolor.red("D3 data not found for sim2:{}".format(self.sim2))
            return np.nan

        tmerg1 = self.o_par1.get_par("tmerger")
        tmerg2 = self.o_par2.get_par("tmerger")

        Printcolor.blue("\ttd3_1[-1]:{} tmerg1:{} -> {}".format(td3_1[-1], tmerg1, td3_1[-1] - tmerg1))
        Printcolor.blue("\ttd3_2[-1]:{} tmerg2:{} -> {}".format(td3_2[-1], tmerg2, td3_2[-1] - tmerg2))

        td3_1 = np.array(td3_1 - tmerg1)
        td3_2 = np.array(td3_2 - tmerg2)

        if td3_1.min() > td3_2.max():
            Printcolor.red("D3 data does not overlap. sim1 has min:{} that is > than sim2 max: {}"
                           .format(td3_1.min(), td3_2.max()))
            return np.nan

        if td3_1.max() < td3_2.min():
            Printcolor.red("D3 data does not overlap. sim1 has max:{} that is < than sim2 min: {}"
                           .format(td3_1.max(), td3_2.min()))
            return np.nan

        tmax = np.min([td3_1.max(), td3_2.max()])
        Printcolor.blue("\ttmax for D3 data: {}".format(tmax))
        return float(tmax)

    def get_outflow_par_err(self, det, new_mask, v_n):

        o_par1 = ALL_PAR(self.sim1, add_mask=new_mask)
        o_par2 = ALL_PAR(self.sim2, add_mask=new_mask)

        val1 = o_par1.get_outflow_par(det, new_mask, v_n)
        val2 = o_par2.get_outflow_par(det, new_mask, v_n)

        # err = np.abs(val1 - val2) / val1

        return val1, val2

    # --- --- --- --- ---

    def get_outflow_pars(self, det, mask, v_n, rewrite=False):

        if mask == "geo":
            self.compute_outflow_new_mask(det, self.sim1, mask, rewrite=rewrite)
            self.compute_outflow_new_mask(det, self.sim2, mask, rewrite=rewrite)
            return self.get_outflow_par_err(det, mask, v_n)

        elif mask.__contains__("bern_"):
            delta_t = self.get_post_geo_delta_t(det)
            if not np.isnan(delta_t):
                mask = "bern_geoend" + "_length{:.0f}".format(delta_t * 1e5)  # [1e2 ms]
                self.compute_outflow_new_mask(det, self.sim1, mask, rewrite=rewrite)
                self.compute_outflow_new_mask(det, self.sim2, mask, rewrite=rewrite)
                return self.get_outflow_par_err(det, mask, v_n)
            else:
                return np.nan, np.nan
        else:
            raise NameError("No method exists for mask:{} ".format(mask))

    def get_3d_pars(self, v_n):
        td3 = self.get_tmax_d3_data()
        if not np.isnan(td3):
            tmerg1 = self.o_par1.get_par("tmerger")
            tmerg2 = self.o_par2.get_par("tmerger")
            print("\n{} and {}".format(td3+tmerg1, td3+tmerg2))
            val1 = self.o_par1.get_int_par(v_n, td3+tmerg1)
            val2 = self.o_par2.get_int_par(v_n, td3+tmerg2)
            return val1, val2
        else:
            return np.nan, np.nan


class THREE_SIMS():

    def __init__(self, sim1, sim2, sim3):
        self.sim1 = sim1
        self.sim2 = sim2
        self.sim3 = sim3
        self.o_par1 = ADD_METHODS_ALL_PAR(self.sim1)
        self.o_par2 = ADD_METHODS_ALL_PAR(self.sim2)
        self.o_par3 = ADD_METHODS_ALL_PAR(self.sim3)
        self.outflow_tasks = ["totflux", "hist"]

    def compute_outflow_new_mask(self, det, sim, mask, rewrite):

        # get_tmax60 # ms
        print("\tAdding mask:{}".format(mask))
        o_outflow = EJECTA_PARS(sim, add_mask=mask)

        if not os.path.isdir(Paths.ppr_sims + sim +"/" +"outflow_{:d}/".format(det) + mask + '/'):
            os.mkdir(Paths.ppr_sims + sim +"/" +"outflow_{:d}/".format(det) + mask + '/')

        Printcolor.blue("Creating new outflow mask dir:{}"
                        .format(sim +"/" +"outflow_{:d}/".format(det) + mask + '/'))

        for task in self.outflow_tasks:
            if task == "hist":
                # from outflowed import outflowed_historgrams
                outflowed_historgrams(o_outflow, [det], [mask], o_outflow.list_hist_v_ns, rewrite=rewrite)
            elif task == "corr":
                # from outflowed import outflowed_correlations
                outflowed_correlations(o_outflow, [det], [mask], o_outflow.list_corr_v_ns, rewrite=rewrite)
            elif task == "totflux":
                # from outflowed import outflowed_totmass
                outflowed_totmass(o_outflow, [det], [mask], rewrite=rewrite)
            elif task == "timecorr":
                # from outflowed import outflowed_timecorr
                outflowed_timecorr(o_outflow, [det], [mask], o_outflow.list_hist_v_ns, rewrite=rewrite)
            else:
                raise NameError("method for computing outflow with new mask is not setup for task:{}".format(task))

    def get_post_geo_delta_t(self, det):

        # o_par1 = ALL_PAR(self.sim1)
        # o_par2 = ALL_PAR(self.sim2)

        # tmerg1 = self.o_par1.get_par("tmerger")
        # tmerg2 = self.o_par2.get_par("tmerger")

        t98geomass1 = self.o_par1.get_outflow_par(det, "geo", "t98mass")
        t98geomass2 = self.o_par2.get_outflow_par(det, "geo", "t98mass")
        t98geomass3 = self.o_par3.get_outflow_par(det, "geo", "t98mass")

        tend1 = self.o_par1.get_outflow_par(det, "geo", "tend")
        tend2 = self.o_par2.get_outflow_par(det, "geo", "tend")
        tend3 = self.o_par3.get_outflow_par(det, "geo", "tend")

        if tend1 < t98geomass1:
            Printcolor.red("tend1:{} < t98geomass1:{}".format(tend1, t98geomass1))
            return np.nan
        if tend2 < t98geomass2:
            Printcolor.red("tend2:{} < t98geomass2:{}".format(tend2, t98geomass2))
            return np.nan
        if tend3 < t98geomass3:
            Printcolor.red("tend3:{} < t98geomass3:{}".format(tend3, t98geomass3))
            return np.nan

        Printcolor.yellow("Relaxing the time limits criteria")
        # if tend1 < t98geomass2:
        #     Printcolor.red("Delta t does not overlap tend1:{} < t98geomass2:{}".format(tend1, t98geomass2))
        #     return np.nan
        # if tend2 < t98geomass1:
        #     Printcolor.red("Delta t does not overlap tend2:{} < t98geomass1:{}".format(tend2, t98geomass1))
        #     return np.nan
        # assert tmerg1 < t98geomass1
        # assert tmerg2 < t98geomass2

        # tend1 = tend1 - tmerg1
        # tend2 = tend2 - tmerg2
        # t98geomass1 = t98geomass1 - tmerg1
        # t98geomass2 = t98geomass2 - tmerg2

        delta_t1 = tend1 - t98geomass1
        delta_t2 = tend2 - t98geomass2
        delta_t3 = tend3 - t98geomass3

        print("\tTime window for bernoulli ")
        print("\t{} {:.2f} [ms]".format(self.sim1, delta_t1 * 1e3))
        print("\t{} {:.2f} [ms]".format(self.sim2, delta_t2 * 1e3))
        print("\t{} {:.2f} [ms]".format(self.sim3, delta_t3 * 1e3))
        # exit(1)

        delta_t = np.min([delta_t1, delta_t2, delta_t3])

        if delta_t < 0.005:
            return np.nan# ms

        return delta_t

    def get_tmax_d3_data(self):

        isd3_1, itd3_1, td3_1 = self.o_par1.get_ittime("profiles", "prof")
        isd3_2, itd3_2, td3_2 = self.o_par2.get_ittime("profiles", "prof")
        isd3_3, itd3_3, td3_3 = self.o_par3.get_ittime("profiles", "prof")

        if len(td3_1) == 0:
            Printcolor.red("D3 data not found for sim1:{}".format(self.sim1))
            return np.nan
        if len(td3_2) == 0:
            Printcolor.red("D3 data not found for sim2:{}".format(self.sim2))
            return np.nan
        if len(td3_3) == 0:
            Printcolor.red("D3 data not found for sim3:{}".format(self.sim3))
            return np.nan

        tmerg1 = self.o_par1.get_par("tmerger")
        tmerg2 = self.o_par2.get_par("tmerger")
        tmerg3 = self.o_par3.get_par("tmerger")

        Printcolor.blue("\ttd3_1[-1]:{} tmerg1:{} -> {}".format(td3_1[-1], tmerg1, td3_1[-1] - tmerg1))
        Printcolor.blue("\ttd3_2[-1]:{} tmerg2:{} -> {}".format(td3_2[-1], tmerg2, td3_2[-1] - tmerg2))
        Printcolor.blue("\ttd3_3[-1]:{} tmerg3:{} -> {}".format(td3_3[-1], tmerg3, td3_3[-1] - tmerg3))

        td3_1 = np.array(td3_1 - tmerg1)
        td3_2 = np.array(td3_2 - tmerg2)
        td3_3 = np.array(td3_3 - tmerg3)
        #
        if td3_1.min() > td3_2.max():
            Printcolor.red("D3 data does not overlap. sim1 has min:{} that is > than sim2 max: {}"
                           .format(td3_1.min(), td3_2.max()))
            return np.nan

        if td3_1.min() > td3_3.max():
            Printcolor.red("D3 data does not overlap. sim1 has min:{} that is > than sim3 max: {}"
                           .format(td3_1.min(), td3_3.max()))
            return np.nan

        if td3_2.min() > td3_3.max():
            Printcolor.red("D3 data does not overlap. sim2 has min:{} that is > than sim3 max: {}"
                           .format(td3_2.min(), td3_3.max()))
            return np.nan
        # ---
        if td3_1.max() < td3_2.min():
            Printcolor.red("D3 data does not overlap. sim1 has max:{} that is < than sim2 min: {}"
                           .format(td3_1.max(), td3_2.min()))
            return np.nan

        if td3_1.max() < td3_3.min():
            Printcolor.red("D3 data does not overlap. sim1 has max:{} that is < than sim3 min: {}"
                           .format(td3_1.max(), td3_3.min()))
            return np.nan

        if td3_2.max() < td3_3.min():
            Printcolor.red("D3 data does not overlap. sim2 has max:{} that is < than sim3 min: {}"
                           .format(td3_2.max(), td3_3.min()))
            return np.nan

        tmax = np.min([td3_1.max(), td3_2.max(), td3_3.max()])
        Printcolor.blue("\ttmax for D3 data: {}".format(tmax))
        return float(tmax)

    def get_outflow_par_err(self, det, new_mask, v_n):

        o_par1 = ALL_PAR(self.sim1, add_mask=new_mask)
        o_par2 = ALL_PAR(self.sim2, add_mask=new_mask)
        o_par3 = ALL_PAR(self.sim3, add_mask=new_mask)

        val1 = o_par1.get_outflow_par(det, new_mask, v_n)
        val2 = o_par2.get_outflow_par(det, new_mask, v_n)
        val3 = o_par3.get_outflow_par(det, new_mask, v_n)

        # err = np.abs(val1 - val2) / val1

        return val1, val2, val3

    # --- --- --- --- ---

    def get_outflow_pars(self, det, mask, v_n, rewrite=False):

        if mask == "geo":
            self.compute_outflow_new_mask(det, self.sim1, mask, rewrite=rewrite)
            self.compute_outflow_new_mask(det, self.sim2, mask, rewrite=rewrite)
            self.compute_outflow_new_mask(det, self.sim3, mask, rewrite=rewrite)
            return self.get_outflow_par_err(det, mask, v_n)

        elif mask.__contains__("bern_"):
            delta_t = self.get_post_geo_delta_t(det)
            if not np.isnan(delta_t):
                mask = "bern_geoend" + "_length{:.0f}".format(delta_t * 1e5)  # [1e2 ms]
                self.compute_outflow_new_mask(det, self.sim1, mask, rewrite=rewrite)
                self.compute_outflow_new_mask(det, self.sim2, mask, rewrite=rewrite)
                self.compute_outflow_new_mask(det, self.sim3, mask, rewrite=rewrite)
                return self.get_outflow_par_err(det, mask, v_n)
            else:
                return np.nan, np.nan, np.nan
        else:
            raise NameError("No method exists for mask:{} ".format(mask))

    def get_3d_pars(self, v_n):
        td3 = self.get_tmax_d3_data()
        if not np.isnan(td3):
            tmerg1 = self.o_par1.get_par("tmerger")
            tmerg2 = self.o_par2.get_par("tmerger")
            tmerg3 = self.o_par3.get_par("tmerger")
            print("\n{} and {} and {}".format(td3+tmerg1, td3+tmerg2, td3+tmerg3))
            val1 = self.o_par1.get_int_par(v_n, td3 + tmerg1)
            val2 = self.o_par2.get_int_par(v_n, td3 + tmerg2)
            val3 = self.o_par3.get_int_par(v_n, td3 + tmerg3)
            return val1, val2, val3
        else:
            return np.nan, np.nan, np.nan
