from __future__ import division

import numpy as np
import sys
import os
import csv
from scipy import interpolate
sys.path.append('/data01/numrel/vsevolod.nedora/bns_ppr_tools/')
from argparse import ArgumentParser
from preanalysis import LOAD_INIT_DATA
from preanalysis import LOAD_ITTIME
from utils import Paths, Labels, Constants, Printcolor, UTILS

from data import ADD_METHODS_ALL_PAR, AVERAGE_PAR
from comparison import TWO_SIMS, THREE_SIMS
from settings import resolutions
from uutils import *

# there is a better in csv2tex_old.py
class TEX_TABLES:

    def __init__(self):
        self.sim_list = []

        # setting up parameters

        self.init_data_v_ns = ["EOS", "q", "note", "res", "vis"]
        self.init_data_prec = ["", ".1f", "", "", ""]
        #
        self.col_d3_gw_data_v_ns = ["Mbi/Mb", 'tend', "tdisk3D", "Mdisk3D", 'tcoll_gw', "Mdisk"]
        self.col_d3_gw_data_prec = [".2f", ".1f", ".1f",       ".2f",       ".2f",     ".2f"]
        #
        self.outflow_data_v_ns = ['Mej_tot', 'Ye_ave', 'vel_inf_ave', 'theta_rms',
                                  'Mej_tot', 'Ye_ave', 'vel_inf_ave', 'theta_rms']
        self.outflow_data_prec = [".2f", ".2f", ".2f", ".2f",
                                  ".2f", ".2f", ".2f", ".2f"]
        self.outflow_data_mask = ["geo", "geo", "geo", "geo",
                                  "bern_geoend", "bern_geoend", "bern_geoend", "bern_geoend"]


        pass

    # --- UNITS --- #

    @staticmethod
    def get_unit_lbl(v_n):
        if v_n in ["M1", "M2"]: return "$[M_{\odot}]$"
        elif v_n in ["Mej_tot"]: return "$[10^{-2} M_{\odot}]$"
        elif v_n in ["Mdisk3D", "Mdisk"]: return "$[M_{\odot}]$"
        elif v_n in ["vel_inf_ave"]: return "$[c]$"
        elif v_n in ["tcoll_gw", "tmerg_gw", "tmerg", "tcoll", "tend", "tdisk3D"]: return "[ms]"
        else:
            return " "

    # --- LABELS --- #
    @staticmethod
    def get_other_lbl(v_n):
        if v_n == "Mdisk3D": return r"$M_{\text{disk}} ^{\text{last}}$"
        elif v_n == "Mdisk": return r"$M_{\text{disk}} ^{\text{BH}}$"
        elif v_n == "M1": return "$M_a$"
        elif v_n == "M2": return "$M_b$"
        elif v_n == "tcoll_gw" or v_n == "tcoll": return r"$t_{\text{BH}}$"
        elif v_n == "tend": return r"$t_{\text{end}}$"
        elif v_n == "tdisk3D": return r"$t_{\text{disk}}$"
        elif v_n == "q": return r"$M_a/M_b$"
        elif v_n == "EOS": return r"EOS"
        elif v_n == "res": return r"res"
        elif v_n == "vis": return "LK"
        elif v_n == "note": return r"note"
        elif v_n == "Mbi/Mb": return r"$Mb_i/Mb$"
        else:
            raise NameError("No label found for other v_n: {}".format(v_n))
    @staticmethod
    def get_outflow_lbl(v_n, mask):

        if v_n == "theta_rms" and mask=="geo": return "$\\langle \\theta_{\\text{ej}} \\rangle$"
        elif v_n == "Mej_tot" and mask=="geo": return "$M_{\\text{ej}}$"
        elif v_n == "Ye_ave" and mask=="geo": return "$\\langle Y_e \\rangle$"
        elif v_n == "vel_inf_ave" and mask=="geo": return "$\\langle \\upsilon_{\\text{ej}} \\rangle$"
        elif v_n == "theta_rms" and mask=="bern_geoend": return "$\\langle \\theta_{\\text{ej}}^{\\text{w}} \\rangle$"
        elif v_n == "Mej_tot" and mask=="bern_geoend": return "$M_{\\text{ej}}^{\\text{w}}$"
        elif v_n == "Ye_ave" and mask=="bern_geoend": return "$\\langle Y_e ^{\\text{w}}  \\rangle$"
        elif v_n == "vel_inf_ave" and mask=="bern_geoend": return "$\\langle \\upsilon_{\\text{ej}}^{\\text{w}} \\rangle$"
        elif v_n == "Mej_tot" and mask=="theta60_geoend": return "$M_{\\text{ej}}^{\\theta>60}$"
        elif v_n == "Ye_ave" and mask=="theta60_geoend": return "$\\langle Y_e ^{\\theta>60}  \\rangle$"
        elif v_n == "vel_inf_ave" and mask=="theta60_geoend": return "$\\langle \\upsilon_{\\text{ej}}^{\\theta>60} \\rangle$"
        elif v_n == "Mej_tot" and mask=="Y_e04_geoend": return "$M_{\\text{ej}}^{Ye>0.4}$"
        elif v_n == "Ye_ave" and mask=="Y_e04_geoend": return "$\\langle Y_e ^{Ye>0.4}  \\rangle$"
        elif v_n == "vel_inf_ave" and mask=="Y_e04_geoend": return "$\\langle \\upsilon_{\\text{ej}}^{Ye>0.4} \\rangle$"
        elif v_n == "res": return "res"
        else:
            raise NameError("No label found for outflow v_n: {} and mask: {} ".format(v_n, mask))

    # --- DATA --- #

    def get_mixed_data_val(self, o_init_data, o_coll_data, v_n, prec):

        if v_n == "Mbi/Mb":
            mbi = float(o_init_data.get_par("Mb1") + o_init_data.get_par("Mb2"))
            mass = o_coll_data.get_total_mass()
            mb0 = mass[0,1]

            val = (mbi / mb0) * 100

            if prec == "":
                return str(val)
            else:
                return ("%{}".format(prec) % float(val))

        else:raise NameError("Unknown v_n for mixed data: v_n:{}".format(v_n))

    def get_inital_data_val(self, o_data, v_n, prec):
        #
        if v_n == "note":
            eos = o_data.get_par("EOS")
            if eos == "DD2":
                val = o_data.get_par("run")
            elif eos == "SFHo":
                pizza = o_data.get_par("pizza_eos")
                if pizza.__contains__("2019"):
                    val = "pz19"
                else:
                    val = ""
            elif eos == "LS220":
                val = ""
            elif eos == "SLy4":
                val = ""
            elif eos == "BLh":
                val = ""
            else:
                raise NameError("no notes for EOS:{}".format(eos))

        else:
            val = o_data.get_par(v_n)
        #

        #
        if prec == "":
            return str(val)
        else:
            return ("%{}".format(prec) % float(val))

    def get_col_gw_d3_val(self, o_init_data, o_data, v_n, prec):

        if v_n == "Mbi/Mb":
            mbi = float(o_init_data.get_par("Mb1") + o_init_data.get_par("Mb2"))
            mass = o_data.get_total_mass()
            mb0 = mass[0,1]

            val = (mbi / mb0)

            if prec == "":
                return str(val)
            else:
                return ("%{}".format(prec) % float(val))

        val = o_data.get_par(v_n)

        if v_n == "tcoll_gw":
            tmerg = float(o_data.get_par("tmerger"))
            if np.isinf(val):
                tend = o_data.get_par("tend")
                # print(o_data.sim, tend, tmerg)
                assert tend > tmerg
                return str(r"$>{:.1f}$".format((tend-tmerg) * 1e3))
            else:
                print(val)
                # tcoll = o_data.get_par("tcoll")
                assert val > tmerg
                val = (val-tmerg) * 1e3

        if v_n == "tend":
            tmerg = o_data.get_par("tmerg")
            val = (val - tmerg) * 1e3

        if v_n == "Mdisk3D" or v_n == "Mdisk":
            if np.isnan(val):
                return r"N/A"
            else:
                val = val# * 1e2

        if v_n == "tdisk3D":

            if np.isnan(val):
                return r"N/A"
            else:
                tmerg = o_data.get_par("tmerg")
                val = (val - tmerg) * 1e3# * 1e2

        if prec == "":
            return str(val)
        else:
            return ("%{}".format(prec) % val)

    def get_ouflow_data(self, o_data, v_n, mask, prec):

        val = o_data.get_outflow_par(0, mask, v_n)
        if v_n == "Mej_tot":
            val = val * 1e2
            if mask == "bern_geoend":
                tcoll = o_data.get_par("tcoll_gw")
                if np.isinf(tcoll):
                    return("$>~%{}$".format(prec) % val)
                else:
                    return("$\propto~%{}$".format(prec) % val)

        if prec == "":
            return str(val)
        else:
            return ("%{}".format(prec) % val)

    # --- MAIN --- #

    def get_table_size_head(self):

        print("\n")
        size = '{'
        head = ''
        i = 0

        all_v_ns = self.init_data_v_ns + self.col_d3_gw_data_v_ns + self.outflow_data_v_ns
        if len(self.init_data_v_ns) > 0:
            for init_data_v in self.init_data_v_ns:
                v_n = self.get_other_lbl(init_data_v)
                size = size + 'c'
                head = head + '{}'.format(v_n)
                if init_data_v != all_v_ns[-1]: size = size + ' '
                if i != len(all_v_ns) - 1: head = head + ' & '
                i = i + 1
        if len(self.col_d3_gw_data_v_ns) > 0:
            for other_v_n in self.col_d3_gw_data_v_ns:
                v_n = self.get_other_lbl(other_v_n)
                size = size + 'c'
                head = head + '{}'.format(v_n)
                if other_v_n != all_v_ns[-1]: size = size + ' '
                if i != len(all_v_ns) - 1: head = head + ' & '
                i = i + 1
        if len(self.outflow_data_v_ns) > 0:
            for outflow_v_n, outflow_mask in zip(self.outflow_data_v_ns, self.outflow_data_mask):
                v_n = self.get_outflow_lbl(outflow_v_n, outflow_mask)
                size = size + 'c'
                head = head + '{}'.format(v_n)
                if outflow_v_n != all_v_ns[-1]: size = size + ' '
                if i != len(all_v_ns) - 1: head = head + ' & '
                i = i + 1

        size = size + '}'

        head = head + ' \\\\'  # = \\

        return size, head

    def get_unit_bar(self):

        all_v_ns = self.init_data_v_ns + self.col_d3_gw_data_v_ns + self.outflow_data_v_ns

        unit_bar = ''
        for i, v_n in enumerate(all_v_ns):
            unit = self.get_unit_lbl(v_n)

            unit_bar = unit_bar + '{}'.format(unit)
            # if v_ns.index(v_n) != len(v_ns): unit_bar = unit_bar + ' & '
            if i != len(all_v_ns) - 1: unit_bar = unit_bar + ' & '

        unit_bar = unit_bar + ' \\\\ '

        return unit_bar

    def get_rows(self, sim_list):

        all_v_ns = self.init_data_v_ns + self.col_d3_gw_data_v_ns + self.outflow_data_v_ns

        rows = []
        for i, sim in enumerate(sim_list):
            row = ''
            j = 0
            # add init_data_val:
            if len(self.init_data_v_ns) > 0:
                o_init_data = LOAD_INIT_DATA(sim)
                for init_data_v, init_data_p in zip(self.init_data_v_ns, self.init_data_prec):
                    print("\tPrinting Initial Data {}".format(init_data_v))
                    val = self.get_inital_data_val(o_init_data, v_n=init_data_v, prec=init_data_p)
                    row = row + val
                    if j != len(all_v_ns) - 1: row = row + ' & '
                    j = j + 1

            # add coll gw d3 data:
            if len(self.col_d3_gw_data_v_ns) > 0 or len(self.outflow_data_v_ns) > 0:
                o_data = ADD_METHODS_ALL_PAR(sim)
                o_init_data = LOAD_INIT_DATA(sim)
                for other_v_n, other_prec in zip(self.col_d3_gw_data_v_ns, self.col_d3_gw_data_prec):
                    print("\tPrinting Initial Data {}".format(other_v_n))
                    val = self.get_col_gw_d3_val(o_init_data, o_data, v_n=other_v_n, prec=other_prec)
                    row = row + val
                    if j != len(all_v_ns) - 1: row = row + ' & '
                    j = j + 1

                # add outflow data:
                for outflow_v_n, outflow_prec, outflow_mask in zip(self.outflow_data_v_ns, self.outflow_data_prec,
                                                                   self.outflow_data_mask):
                    print("\tPrinting Outflow Data {} (mask: {})".format(outflow_v_n, outflow_mask))
                    val = self.get_ouflow_data(o_data, v_n=outflow_v_n, mask=outflow_mask, prec=outflow_prec)
                    row = row + val
                    if j != len(all_v_ns) - 1: row = row + ' & '
                    j = j + 1
            row = row + ' \\\\'  # = \\
            rows.append(row)

        return rows

    def print_intro_table(self):

        size, head = self.get_table_size_head()
        unit_bar = self.get_unit_bar()

        print('\\begin{table*}[t]')
        print('\\begin{center}')
        print('\\begin{tabular}' + '{}'.format(size))
        print('\\hline')
        print(head)
        print(unit_bar)
        print('\\hline\\hline')

    def print_end_table(self):
        print('\\hline')
        print('\\end{tabular}')
        print('\\end{center}')
        print('\\caption{I am your table! }')
        print('\\label{tbl:1}')
        print('\\end{table*}')

    def print_one_table(self, sim_list, print_head=True, print_end=True):

        # setting up parameters
        init_data_v_ns = self.init_data_v_ns
        init_data_prec = self.init_data_prec
        #
        col_d3_gw_data_v_ns = self.col_d3_gw_data_v_ns
        col_d3_gw_data_prec = self.col_d3_gw_data_prec
        #
        outflow_data_v_ns = self.outflow_data_v_ns
        outflow_data_prec = self.outflow_data_prec
        outflow_data_mask = self.outflow_data_mask
        #
        assert len(init_data_prec) == len(init_data_v_ns)
        assert len(col_d3_gw_data_prec) == len(col_d3_gw_data_v_ns)
        assert len(outflow_data_mask) == len(outflow_data_prec)
        assert len(outflow_data_prec) == len(outflow_data_v_ns)
        #
        all_v_ns = init_data_v_ns + col_d3_gw_data_v_ns + outflow_data_v_ns
        #
        rows = []
        for i, sim in enumerate(sim_list):
            row = ''
            j = 0
            # add init_data_val:
            if len(init_data_v_ns) > 0:
                o_init_data = LOAD_INIT_DATA(sim)
                for init_data_v, init_data_p in zip(init_data_v_ns, init_data_prec):
                    print("\tPrinting Initial Data {}".format(init_data_v))
                    val = self.get_inital_data_val(o_init_data, v_n=init_data_v, prec=init_data_p)
                    row = row + val
                    if j != len(all_v_ns) - 1: row = row + ' & '
                    j = j + 1

            # add coll gw d3 data:
            if len(col_d3_gw_data_v_ns) > 0 or len(outflow_data_v_ns) > 0:
                o_data = ADD_METHODS_ALL_PAR(sim)
                o_init_data = LOAD_INIT_DATA(sim)
                for other_v_n, other_prec in zip(col_d3_gw_data_v_ns, col_d3_gw_data_prec):
                    print("\tPrinting Initial Data {}".format(other_v_n))
                    val = self.get_col_gw_d3_val(o_init_data, o_data, v_n=other_v_n, prec=other_prec)
                    row = row + val
                    if j != len(all_v_ns) - 1: row = row + ' & '
                    j = j + 1

                # add outflow data:
                for outflow_v_n, outflow_prec, outflow_mask in zip(outflow_data_v_ns, outflow_data_prec, outflow_data_mask):
                    print("\tPrinting Outflow Data {} (mask: {})".format(outflow_v_n, outflow_mask))
                    val = self.get_ouflow_data(o_data, v_n=outflow_v_n, mask=outflow_mask, prec=outflow_prec)
                    row = row + val
                    if j != len(all_v_ns) - 1: row = row + ' & '
                    j = j + 1
            row = row + ' \\\\'  # = \\
            rows.append(row)

        # --- HEAD --- #

        print("\n")
        size = '{'
        head = ''
        i = 0
        if len(init_data_v_ns) > 0:
            for init_data_v in init_data_v_ns:
                v_n = self.get_other_lbl(init_data_v)
                size = size + 'c'
                head = head + '{}'.format(v_n)
                if init_data_v != all_v_ns[-1]: size = size + ' '
                if i != len(all_v_ns) - 1: head = head + ' & '
                i = i + 1
        if len(col_d3_gw_data_v_ns) > 0:
            for other_v_n in col_d3_gw_data_v_ns:
                v_n = self.get_other_lbl(other_v_n)
                size = size + 'c'
                head = head + '{}'.format(v_n)
                if other_v_n != all_v_ns[-1]: size = size + ' '
                if i != len(all_v_ns) - 1: head = head + ' & '
                i = i + 1
        if len(outflow_data_v_ns) > 0:
            for outflow_v_n, outflow_mask in zip(outflow_data_v_ns, outflow_data_mask):
                v_n = self.get_outflow_lbl(outflow_v_n, outflow_mask)
                size = size + 'c'
                head = head + '{}'.format(v_n)
                if outflow_v_n != all_v_ns[-1]: size = size + ' '
                if i != len(all_v_ns) - 1: head = head + ' & '
                i = i + 1

        size = size + '}'

        # --- UNIT BAR --- #

        unit_bar = ''
        for i, v_n in enumerate(all_v_ns):
            unit = self.get_unit_lbl(v_n)

            unit_bar = unit_bar + '{}'.format(unit)
            # if v_ns.index(v_n) != len(v_ns): unit_bar = unit_bar + ' & '
            if i != len(all_v_ns) - 1: unit_bar = unit_bar + ' & '

        head = head + ' \\\\'  # = \\
        unit_bar = unit_bar + ' \\\\ '

        # ====================== PRINT TABLE ================== #



        if print_head:
            print('\\begin{table*}[t]')
            print('\\begin{center}')
            print('\\begin{tabular}' + '{}'.format(size))
            print('\\hline')
            print(head)
            print(unit_bar)
            print('\\hline\\hline')

        for row in rows:
            print(row)

        if print_end:
            print('\\hline')
            print('\\end{tabular}')
            print('\\end{center}')
            print('\\caption{I am your table! }')
            print('\\label{tbl:1}')
            print('\\end{table*}')

        exit(0)

    def print_mult_table(self, list_simgroups, separateors):

        assert len(list_simgroups) == len(separateors)

        group_rows = []
        for sim_group in list_simgroups:
            rows = self.get_rows(sim_group)
            group_rows.append(rows)

        print("data colleted. Printing...")

        self.print_intro_table()
        for i in range(len(list_simgroups)):
            for row in group_rows[i]:
                print(row)
            print(separateors[i])
            # print("\\hline")

        self.print_end_table()


class COMPARISON_TABLE:

    def __init__(self):
        self.sim_list = []

        # setting up parameters
        self.init_data_v_ns = ["EOS", "q", "note", "res", "vis"]
        self.init_data_prec = ["", ".1f", "", "", ""]
        #
        self.col_d3_gw_data_v_ns = ["Mdisk3D", "tdisk3D"]
        self.col_d3_gw_data_prec = [".2f", ".1f"]
        #
        self.outflow_data_v_ns = ['Mej_tot', 'Ye_ave', 'vel_inf_ave', 'theta_rms', 'delta_t',
                                  'Mej_tot', 'Ye_ave', 'vel_inf_ave', 'theta_rms']
        self.outflow_data_prec = [".2f", ".3f", ".3f", ".2f", ".1f",
                                  ".2f", ".3f", ".3f", ".2f"]
        self.outflow_data_mask = ["geo", "geo", "geo", "geo", "bern_geoend",
                                  "bern_geoend", "bern_geoend", "bern_geoend", "bern_geoend"]
        #

        pass

    # --- UNITS --- #

    @staticmethod
    def get_unit_lbl(v_n):
        if v_n in ["M1", "M2"]:
            return "$[M_{\odot}]$"
        elif v_n in ["Mej_tot"]:
            return "$[10^{-2} M_{\odot}]$"
        elif v_n in ["Mdisk3D", "Mdisk"]:
            return "$[M_{\odot}]$"
        elif v_n in ["vel_inf_ave"]:
            return "$[c]$"
        elif v_n in ["tcoll_gw", "tmerg_gw", "tmerg", "tcoll", "tend", "tdisk3D", "delta_t"]:
            return "[ms]"
        else:
            return " "

    # --- LABELS --- #
    @staticmethod
    def get_other_lbl(v_n):
        if v_n == "Mdisk3D":
            return r"$M_{\text{disk}} ^{\text{last}}$"
        elif v_n == "Mdisk":
            return r"$M_{\text{disk}} ^{\text{BH}}$"
        elif v_n == "M1":
            return "$M_a$"
        elif v_n == "M2":
            return "$M_b$"
        elif v_n == "tcoll_gw" or v_n == "tcoll":
            return r"$t_{\text{BH}}$"
        elif v_n == "tend":
            return r"$t_{\text{end}}$"
        elif v_n == "tdisk3D":
            return r"$t_{\text{disk}}$"
        elif v_n == "q":
            return r"$M_a/M_b$"
        elif v_n == "EOS":
            return r"EOS"
        elif v_n == "res":
            return r"res"
        elif v_n == "vis":
            return "LK"
        elif v_n == "note":
            return r"note"
        else:
            raise NameError("No label found for other v_n: {}".format(v_n))

    @staticmethod
    def get_outflow_lbl(v_n, mask):

        if v_n == "theta_rms" and mask == "geo":
            return "$\\langle \\theta_{\\text{ej}} \\rangle$"
        elif v_n == "Mej_tot" and mask == "geo":
            return "$M_{\\text{ej}}$"
        elif v_n == "Ye_ave" and mask == "geo":
            return "$\\langle Y_e \\rangle$"
        elif v_n == "vel_inf_ave" and mask == "geo":
            return "$\\langle \\upsilon_{\\text{ej}} \\rangle$"
        elif v_n == "theta_rms" and mask == "bern_geoend":
            return "$\\langle \\theta_{\\text{ej}}^{\\text{w}} \\rangle$"
        elif v_n == "Mej_tot" and mask == "bern_geoend":
            return "$M_{\\text{ej}}^{\\text{w}}$"
        elif v_n == "Ye_ave" and mask == "bern_geoend":
            return "$\\langle Y_e ^{\\text{w}}  \\rangle$"
        elif v_n == "vel_inf_ave" and mask == "bern_geoend":
            return "$\\langle \\upsilon_{\\text{ej}}^{\\text{w}} \\rangle$"
        elif v_n == "delta_t" and mask.__contains__("geoend"):
            return r"$\Delta t_{\text{wind}}$"
        elif v_n == "res":
            return "res"
        else:
            raise NameError("No label found for outflow v_n: {} and mask: {} ".format(v_n, mask))

    # --- DATA --- #

    def get_inital_data_val(self, o_data, v_n, prec):
        #
        if v_n == "note":
            eos = o_data.get_par("EOS")
            if eos == "DD2":
                val = o_data.get_par("run")
            elif eos == "SFHo":
                pizza = o_data.get_par("pizza_eos")
                if pizza.__contains__("2019"):
                    val = "pz19"
                else:
                    val = ""
            elif eos == "LS220":
                val = ""
            elif eos == "SLy4":
                val = ""
            elif eos == "BLh":
                val = ""
            else:
                raise NameError("no notes for EOS:{}".format(eos))

        else:
            val = o_data.get_par(v_n)
        #

        #
        if prec == "":
            return str(val)
        else:
            return ("%{}".format(prec) % float(val))

    def get_col_gw_d3_val(self, o_data, v_n, prec):

        val = o_data.get_par(v_n)

        if v_n == "tcoll_gw":
            tmerg = float(o_data.get_par("tmerger"))
            if np.isinf(val):
                tend = o_data.get_par("tend")
                # print(o_data.sim, tend, tmerg)
                assert tend > tmerg
                return str(r"$>{:.1f}$".format((tend - tmerg) * 1e3))
            else:
                print(val)
                # tcoll = o_data.get_par("tcoll")
                assert val > tmerg
                val = (val - tmerg) * 1e3

        if v_n == "tend":
            tmerg = o_data.get_par("tmerg")
            val = (val - tmerg) * 1e3

        if v_n == "Mdisk3D" or v_n == "Mdisk":
            if np.isnan(val):
                return r"N/A"
            else:
                val = val  # * 1e2

        if v_n == "tdisk3D":

            if np.isnan(val):
                return r"N/A"
            else:
                tmerg = o_data.get_par("tmerg")
                val = (val - tmerg) * 1e3  # * 1e2

        if prec == "":
            return str(val)
        else:
            return ("%{}".format(prec) % val)

    def get_ouflow_data(self, o_data, v_n, mask, prec):

        val = o_data.get_outflow_par(0, mask, v_n)
        if v_n == "Mej_tot":
            val = val * 1e2
            if mask == "bern_geoend":
                tcoll = o_data.get_par("tcoll")
                if np.isinf(tcoll):
                    return ("$>~%{}$".format(prec) % val)
                else:
                    return ("$\propto~%{}$".format(prec) % val)

        if prec == "":
            return str(val)
        else:
            return ("%{}".format(prec) % val)

    # ---- Comparison Data ---

    def get_comp_other_data(self, o_2sim, v_n, prec, sims=2):

        if sims == 2:
            val1, val2 = o_2sim.get_3d_pars(v_n)

            if v_n == "Mdisk3D" or v_n == "Mdisk":
                if np.isnan(val1) or np.isnan(val2):
                    err = "N/A"
                else:
                    err = 100 * (val1 - val2) / val1
                    err = "{:.0f}".format(err)
                #
                if np.isnan(val1):
                    res1 = "N/A"
                else:
                    res1 = "%{}".format(prec) % val1
                #
                if np.isnan(val2):
                    res2 = "N/A"
                else:
                    res2 = "%{}".format(prec) % val2
                #
                return res1, res2, err

            elif v_n == "tdisk3D" or v_n == "tdisk3D":

                val = o_2sim.get_tmax_d3_data()
                if np.isnan(val):
                    res1 = res2 = "N/A"
                else:
                    res1 = res2 = "%{}".format(prec) % (val * 1e3)
                #
                return res1, res2, " "

            else:
                raise NameError("np method for comp_other_data v_n:{} is set".format(v_n))
        elif sims == 3:
            val1, val2, val3 = o_2sim.get_3d_pars(v_n)

            if v_n == "Mdisk3D" or v_n == "Mdisk":
                if np.isnan(val1) or np.isnan(val2) or np.isnan(val2):
                    err = "N/A"
                else:
                    err1 = 100 * (val1 - val2) / val1
                    err2 = 100 * (val1 - val3) / val1
                    err = "{:.0f} {:.0f}".format(err1, err2)
                #
                if np.isnan(val1):
                    res1 = "N/A"
                else:
                    res1 = "%{}".format(prec) % val1
                #
                if np.isnan(val2):
                    res2 = "N/A"
                else:
                    res2 = "%{}".format(prec) % val2
                #
                if np.isnan(val3):
                    res3 = "N/A"
                else:
                    res3 = "%{}".format(prec) % val3
                #
                return res1, res2, res3, err

            elif v_n == "tdisk3D" or v_n == "tdisk3D":

                val = o_2sim.get_tmax_d3_data()
                if np.isnan(val):
                    res1 = res2 = res3 = "N/A"
                else:
                    res1 = res2 = res3 = "%{}".format(prec) % (val * 1e3)
                #
                return res1, res2, res3, " "

            else:
                raise NameError("np method for comp_other_data v_n:{} is set".format(v_n))
        else:
            raise ValueError("No get_comp_other_data for {} sims".format(sims))

    def get_comp_ouflow_data(self, o_2sim, v_n, mask, prec, sims=2):

        if sims == 2:
            if v_n == "delta_t" and mask.__contains__("geoend"):
                val = o_2sim.get_post_geo_delta_t(0)
                if np.isnan(val):
                    res1 = res2 = "N/A"
                else:
                    res1 = res2 = "%{}".format(prec) % (val * 1e3)
                return res1, res2, ""

            val1, val2 = o_2sim.get_outflow_pars(0, mask, v_n, rewrite=False)

            if v_n == "Mej_tot" or v_n == "Mej_tot":
                if np.isnan(val1) or np.isnan(val2):
                    err = "N/A"
                else:
                    err = 100 * (val1 - val2) / val1
                    err = "{:.0f}".format(err)
                #
                if np.isnan(val1):
                    res1 = "N/A"
                else:
                    res1 = "%{}".format(prec) % (val1 * 1e2)
                #
                if np.isnan(val2):
                    res2 = "N/A"
                else:
                    res2 = "%{}".format(prec) % (val2 * 1e2)
                #
                return res1, res2, err

            elif v_n in ['Ye_ave', 'vel_inf_ave', 'theta_rms']:
                if np.isnan(val1) or np.isnan(val2):
                    err = "N/A"
                else:
                    err = 100 * (val1 - val2) / val1
                    err = "{:.0f}".format(err)
                #
                if np.isnan(val1):
                    res1 = "N/A"
                else:
                    res1 = "%{}".format(prec) % val1
                #
                if np.isnan(val2):
                    res2 = "N/A"
                else:
                    res2 = "%{}".format(prec) % val2
                #
                return res1, res2, err
            else:
                raise NameError("no method setup for geting a str(val1, val2, err) for v_n:{} mask:{}"
                                .format(v_n, mask))
        elif sims == 3:
            if v_n == "delta_t" and mask.__contains__("geoend"):
                val = o_2sim.get_post_geo_delta_t(0)
                if np.isnan(val):
                    res1 = res2 = res3 = "N/A"
                else:
                    res1 = res2 = res3 = "%{}".format(prec) % (val * 1e3)
                return res1, res2, res3, ""

            val1, val2, val3 = o_2sim.get_outflow_pars(0, mask, v_n, rewrite=False)

            if v_n == "Mej_tot" or v_n == "Mej_tot":
                if np.isnan(val1) or np.isnan(val2) or np.isnan(val3) :
                    err = "N/A"
                else:
                    err1 = 100 * (val1 - val2) / val1
                    err2 = 100 * (val1 - val3) / val1
                    err = "{:.0f} {:.0f}".format(err1, err2)
                #
                if np.isnan(val1):
                    res1 = "N/A"
                else:
                    res1 = "%{}".format(prec) % (val1 * 1e2)
                #
                if np.isnan(val2):
                    res2 = "N/A"
                else:
                    res2 = "%{}".format(prec) % (val2 * 1e2)
                #
                if np.isnan(val3):
                    res3 = "N/A"
                else:
                    res3 = "%{}".format(prec) % (val3 * 1e2)
                #
                return res1, res2, res3, err

            elif v_n in ['Ye_ave', 'vel_inf_ave', 'theta_rms']:
                if np.isnan(val1) or np.isnan(val2) or np.isnan(val3):
                    err = "N/A"
                else:
                    err1 = 100 * (val1 - val2) / val1
                    err2 = 100 * (val1 - val3) / val1
                    err = "{:.0f} {:.0f}".format(err1, err2)
                #
                if np.isnan(val1):
                    res1 = "N/A"
                else:
                    res1 = "%{}".format(prec) % val1
                #
                if np.isnan(val2):
                    res2 = "N/A"
                else:
                    res2 = "%{}".format(prec) % val2
                #
                if np.isnan(val3):
                    res3 = "N/A"
                else:
                    res3 = "%{}".format(prec) % val3
                #
                return res1, res2, res3, err
            else:
                raise NameError("no method setup for geting a str(val1, val2, err) for v_n:{} mask:{}"
                                .format(v_n, mask))
        else:
            raise ValueError("only sims=2 and sims=3 supporte. Given:{}".format(sims))

    # --- MAIN --- #

    def get_table_size_head(self):

        print("\n")
        size = '{'
        head = ''
        i = 0

        all_v_ns = self.init_data_v_ns + self.col_d3_gw_data_v_ns + self.outflow_data_v_ns
        if len(self.init_data_v_ns) > 0:
            for init_data_v in self.init_data_v_ns:
                v_n = self.get_other_lbl(init_data_v)
                size = size + 'c'
                head = head + '{}'.format(v_n)
                if init_data_v != all_v_ns[-1]: size = size + ' '
                if i != len(all_v_ns) - 1: head = head + ' & '
                i = i + 1
        if len(self.col_d3_gw_data_v_ns) > 0:
            for other_v_n in self.col_d3_gw_data_v_ns:
                v_n = self.get_other_lbl(other_v_n)
                size = size + 'c'
                head = head + '{}'.format(v_n)
                if other_v_n != all_v_ns[-1]: size = size + ' '
                if i != len(all_v_ns) - 1: head = head + ' & '
                i = i + 1
        if len(self.outflow_data_v_ns) > 0:
            for outflow_v_n, outflow_mask in zip(self.outflow_data_v_ns, self.outflow_data_mask):
                v_n = self.get_outflow_lbl(outflow_v_n, outflow_mask)
                size = size + 'c'
                head = head + '{}'.format(v_n)
                if outflow_v_n != all_v_ns[-1]: size = size + ' '
                if i != len(all_v_ns) - 1: head = head + ' & '
                i = i + 1

        size = size + '}'

        head = head + ' \\\\'  # = \\

        return size, head

    def get_unit_bar(self):

        all_v_ns = self.init_data_v_ns + self.col_d3_gw_data_v_ns + self.outflow_data_v_ns

        unit_bar = ''
        for i, v_n in enumerate(all_v_ns):
            unit = self.get_unit_lbl(v_n)

            unit_bar = unit_bar + '{}'.format(unit)
            # if v_ns.index(v_n) != len(v_ns): unit_bar = unit_bar + ' & '
            if i != len(all_v_ns) - 1: unit_bar = unit_bar + ' & '

        unit_bar = unit_bar + ' \\\\ '

        return unit_bar

    def get_rows(self, sim_list):

        all_v_ns = self.init_data_v_ns + self.col_d3_gw_data_v_ns + self.outflow_data_v_ns

        rows = []

        for i, sim in enumerate(sim_list):
            row = ''
            j = 0
            # add init_data_val:
            if len(self.init_data_v_ns) > 0:
                o_init_data = LOAD_INIT_DATA(sim)
                for init_data_v, init_data_p in zip(self.init_data_v_ns, self.init_data_prec):
                    print("\tPrinting Initial Data {}".format(init_data_v))
                    val = self.get_inital_data_val(o_init_data, v_n=init_data_v, prec=init_data_p)
                    row = row + val
                    if j != len(all_v_ns) - 1: row = row + ' & '
                    j = j + 1

            # add coll gw d3 data:
            if len(self.col_d3_gw_data_v_ns) > 0 or len(self.outflow_data_v_ns) > 0:
                o_data = ADD_METHODS_ALL_PAR(sim)
                for other_v_n, other_prec in zip(self.col_d3_gw_data_v_ns, self.col_d3_gw_data_prec):
                    print("\tPrinting Initial Data {}".format(other_v_n))
                    val = self.get_col_gw_d3_val(o_data, v_n=other_v_n, prec=other_prec)
                    row = row + val
                    if j != len(all_v_ns) - 1: row = row + ' & '
                    j = j + 1

                # add outflow data:
                for outflow_v_n, outflow_prec, outflow_mask in zip(self.outflow_data_v_ns, self.outflow_data_prec,
                                                                   self.outflow_data_mask):
                    print("\tPrinting Outflow Data {} (mask: {})".format(outflow_v_n, outflow_mask))
                    val = self.get_ouflow_data(o_data, v_n=outflow_v_n, mask=outflow_mask, prec=outflow_prec)
                    row = row + val
                    if j != len(all_v_ns) - 1: row = row + ' & '
                    j = j + 1
            row = row + ' \\\\'  # = \\
            rows.append(row)

        return rows

    def get_compartison_rows_for_2(self, two_sims):

        assert len(two_sims) == 2

        all_v_ns = self.init_data_v_ns + self.col_d3_gw_data_v_ns + self.outflow_data_v_ns

        # rows = []

        # o_2sim = TWO_SIMS(two_sims[0], two_sims[1])

        # if len(self.init_data_v_ns) > 0:
        #     o_init_data = LOAD_INIT_DATA(sim)
        #     for init_data_v, init_data_p in zip(self.init_data_v_ns, self.init_data_prec):
        #         print("\tPrinting Initial Data {}".format(init_data_v))
        #         val = self.get_inital_data_val(o_init_data, v_n=init_data_v, prec=init_data_p)
        #         row = row + val
        #         if j != len(all_v_ns) - 1: row = row + ' & '
        #         j = j + 1

        row1 = ''
        row2 = ''
        row3 = ''
        j = 0
        if len(self.init_data_v_ns) > 0:
            o_init_data1 = LOAD_INIT_DATA(two_sims[0])
            o_init_data2 = LOAD_INIT_DATA(two_sims[1])
            for init_data_v, init_data_p in zip(self.init_data_v_ns, self.init_data_prec):
                print("\tPrinting Initial Data {}".format(init_data_v))
                val1 = self.get_inital_data_val(o_init_data1, v_n=init_data_v, prec=init_data_p)
                val2 = self.get_inital_data_val(o_init_data2, v_n=init_data_v, prec=init_data_p)
                row1 = row1 + val1
                row2 = row2 + val2
                if init_data_v == self.init_data_v_ns[0]:
                    row3 = row3 + r"$\Delta$ [\%]"
                else:
                    row3 = row3 + ""
                if j != len(all_v_ns) - 1: row1 = row1 + ' & '
                if j != len(all_v_ns) - 1: row2 = row2 + ' & '
                if j != len(all_v_ns) - 1: row3 = row3 + ' & '
                j = j + 1

        if len(self.col_d3_gw_data_v_ns) > 0 or len(self.outflow_data_v_ns) > 0:
            o_2sim = TWO_SIMS(two_sims[0], two_sims[1])
            for other_v_n, other_prec in zip(self.col_d3_gw_data_v_ns, self.col_d3_gw_data_prec):
                print("\tPrinting Other Data {}".format(other_v_n))
                val1, val2, err = self.get_comp_other_data(o_2sim, v_n=other_v_n, prec=other_prec)
                row1 = row1 + val1
                row2 = row2 + val2
                row3 = row3 + err
                if j != len(all_v_ns) - 1: row1 = row1 + ' & '
                if j != len(all_v_ns) - 1: row2 = row2 + ' & '
                if j != len(all_v_ns) - 1: row3 = row3 + ' & '
                j = j + 1

            for outflow_v_n, outflow_prec, outflow_mask in zip(self.outflow_data_v_ns, self.outflow_data_prec,
                                                               self.outflow_data_mask):
                print("\tPrinting Outflow Data {} (mask: {})".format(outflow_v_n, outflow_mask))
                val1, val2, err = self.get_comp_ouflow_data(o_2sim, v_n=outflow_v_n, mask=outflow_mask,
                                                           prec=outflow_prec)
                row1 = row1 + val1
                row2 = row2 + val2
                row3 = row3 + err
                if j != len(all_v_ns) - 1: row1 = row1 + ' & '
                if j != len(all_v_ns) - 1: row2 = row2 + ' & '
                if j != len(all_v_ns) - 1: row3 = row3 + ' & '
                j = j + 1

            row1 = row1 + ' \\\\'  # = \\
            row2 = row2 + ' \\\\'  # = \\
            row3 = row3 + ' \\\\'  # = \\
        rows = [row1, row2, row3]

        return rows

    def get_compartison_rows_for_3(self, three_sims):

        assert len(three_sims) == 3

        all_v_ns = self.init_data_v_ns + self.col_d3_gw_data_v_ns + self.outflow_data_v_ns

        # rows = []

        # o_2sim = TWO_SIMS(two_sims[0], two_sims[1])

        # if len(self.init_data_v_ns) > 0:
        #     o_init_data = LOAD_INIT_DATA(sim)
        #     for init_data_v, init_data_p in zip(self.init_data_v_ns, self.init_data_prec):
        #         print("\tPrinting Initial Data {}".format(init_data_v))
        #         val = self.get_inital_data_val(o_init_data, v_n=init_data_v, prec=init_data_p)
        #         row = row + val
        #         if j != len(all_v_ns) - 1: row = row + ' & '
        #         j = j + 1

        row1 = ''
        row2 = ''
        row3 = ''
        row4 = ''

        j = 0
        if len(self.init_data_v_ns) > 0:
            o_init_data1 = LOAD_INIT_DATA(three_sims[0])
            o_init_data2 = LOAD_INIT_DATA(three_sims[1])
            o_init_data3 = LOAD_INIT_DATA(three_sims[2])
            for init_data_v, init_data_p in zip(self.init_data_v_ns, self.init_data_prec):
                print("\tPrinting Initial Data {}".format(init_data_v))
                val1 = self.get_inital_data_val(o_init_data1, v_n=init_data_v, prec=init_data_p)
                val2 = self.get_inital_data_val(o_init_data2, v_n=init_data_v, prec=init_data_p)
                val3 = self.get_inital_data_val(o_init_data3, v_n=init_data_v, prec=init_data_p)
                row1 = row1 + val1
                row2 = row2 + val2
                row3 = row3 + val3
                if init_data_v == self.init_data_v_ns[0]:
                    row4 = row4 + r"$\Delta$ [\%]"
                else:
                    row4 = row4 + ""
                if j != len(all_v_ns) - 1: row1 = row1 + ' & '
                if j != len(all_v_ns) - 1: row2 = row2 + ' & '
                if j != len(all_v_ns) - 1: row3 = row3 + ' & '
                if j != len(all_v_ns) - 1: row4 = row4 + ' & '
                j = j + 1
        #
        if len(self.col_d3_gw_data_v_ns) > 0 or len(self.outflow_data_v_ns) > 0:
            o_3sim = THREE_SIMS(three_sims[0], three_sims[1], three_sims[2])
            for other_v_n, other_prec in zip(self.col_d3_gw_data_v_ns, self.col_d3_gw_data_prec):
                print("\tPrinting Other Data {}".format(other_v_n))
                val1, val2, val3, err = self.get_comp_other_data(o_3sim, v_n=other_v_n, prec=other_prec, sims=3)
                row1 = row1 + val1
                row2 = row2 + val2
                row3 = row3 + val3
                row4 = row4 + err
                if j != len(all_v_ns) - 1: row1 = row1 + ' & '
                if j != len(all_v_ns) - 1: row2 = row2 + ' & '
                if j != len(all_v_ns) - 1: row3 = row3 + ' & '
                if j != len(all_v_ns) - 1: row4 = row4 + ' & '
                j = j + 1

            for outflow_v_n, outflow_prec, outflow_mask in zip(self.outflow_data_v_ns, self.outflow_data_prec,
                                                               self.outflow_data_mask):
                print("\tPrinting Outflow Data {} (mask: {})".format(outflow_v_n, outflow_mask))
                val1, val2, val3, err = self.get_comp_ouflow_data(o_3sim, v_n=outflow_v_n, mask=outflow_mask,
                                                           prec=outflow_prec, sims=3)
                row1 = row1 + val1
                row2 = row2 + val2
                row3 = row3 + val3
                row4 = row4 + err
                if j != len(all_v_ns) - 1: row1 = row1 + ' & '
                if j != len(all_v_ns) - 1: row2 = row2 + ' & '
                if j != len(all_v_ns) - 1: row3 = row3 + ' & '
                if j != len(all_v_ns) - 1: row4 = row4 + ' & '
                j = j + 1

            row1 = row1 + ' \\\\'  # = \\
            row2 = row2 + ' \\\\'  # = \\
            row3 = row3 + ' \\\\'  # = \\
            row4 = row4 + ' \\\\'  # = \\
        rows = [row1, row2, row3, row4]

        return rows

    def print_intro_table(self):

        size, head = self.get_table_size_head()
        unit_bar = self.get_unit_bar()

        print('\\begin{table*}[t]')
        print('\\begin{center}')
        print('\\begin{tabular}' + '{}'.format(size))
        print('\\hline')
        print(head)
        print(unit_bar)
        print('\\hline\\hline')

    def print_end_table(self, comment, label):
        print(r'\hline')
        print(r'\end{tabular}')
        print(r'\end{center}')
        print(r'\caption{}'.format(comment))
        print(r'\label{}'.format(label))
        print(r'\end{table*}')

    def print_one_table(self, sim_list, print_head=True, print_end=True):

        # setting up parameters
        init_data_v_ns = self.init_data_v_ns
        init_data_prec = self.init_data_prec
        #
        col_d3_gw_data_v_ns = self.col_d3_gw_data_v_ns
        col_d3_gw_data_prec = self.col_d3_gw_data_prec
        #
        outflow_data_v_ns = self.outflow_data_v_ns
        outflow_data_prec = self.outflow_data_prec
        outflow_data_mask = self.outflow_data_mask
        #
        assert len(init_data_prec) == len(init_data_v_ns)
        assert len(col_d3_gw_data_prec) == len(col_d3_gw_data_v_ns)
        assert len(outflow_data_mask) == len(outflow_data_prec)
        assert len(outflow_data_prec) == len(outflow_data_v_ns)
        #
        all_v_ns = init_data_v_ns + col_d3_gw_data_v_ns + outflow_data_v_ns
        #
        rows = []
        for i, sim in enumerate(sim_list):
            row = ''
            j = 0
            # add init_data_val:
            if len(init_data_v_ns) > 0:
                o_init_data = LOAD_INIT_DATA(sim)
                for init_data_v, init_data_p in zip(init_data_v_ns, init_data_prec):
                    print("\tPrinting Initial Data {}".format(init_data_v))
                    val = self.get_inital_data_val(o_init_data, v_n=init_data_v, prec=init_data_p)
                    row = row + val
                    if j != len(all_v_ns) - 1: row = row + ' & '
                    j = j + 1

            # add coll gw d3 data:
            if len(col_d3_gw_data_v_ns) > 0 or len(outflow_data_v_ns) > 0:
                o_data = ADD_METHODS_ALL_PAR(sim)
                for other_v_n, other_prec in zip(col_d3_gw_data_v_ns, col_d3_gw_data_prec):
                    print("\tPrinting Initial Data {}".format(other_v_n))
                    val = self.get_col_gw_d3_val(o_data, v_n=other_v_n, prec=other_prec)
                    row = row + val
                    if j != len(all_v_ns) - 1: row = row + ' & '
                    j = j + 1

                # add outflow data:
                for outflow_v_n, outflow_prec, outflow_mask in zip(outflow_data_v_ns, outflow_data_prec,
                                                                   outflow_data_mask):
                    print("\tPrinting Outflow Data {} (mask: {})".format(outflow_v_n, outflow_mask))
                    val = self.get_ouflow_data(o_data, v_n=outflow_v_n, mask=outflow_mask, prec=outflow_prec)
                    row = row + val
                    if j != len(all_v_ns) - 1: row = row + ' & '
                    j = j + 1
            row = row + ' \\\\'  # = \\
            rows.append(row)

        # --- HEAD --- #

        print("\n")
        size = '{'
        head = ''
        i = 0
        if len(init_data_v_ns) > 0:
            for init_data_v in init_data_v_ns:
                v_n = self.get_other_lbl(init_data_v)
                size = size + 'c'
                head = head + '{}'.format(v_n)
                if init_data_v != all_v_ns[-1]: size = size + ' '
                if i != len(all_v_ns) - 1: head = head + ' & '
                i = i + 1
        if len(col_d3_gw_data_v_ns) > 0:
            for other_v_n in col_d3_gw_data_v_ns:
                v_n = self.get_other_lbl(other_v_n)
                size = size + 'c'
                head = head + '{}'.format(v_n)
                if other_v_n != all_v_ns[-1]: size = size + ' '
                if i != len(all_v_ns) - 1: head = head + ' & '
                i = i + 1
        if len(outflow_data_v_ns) > 0:
            for outflow_v_n, outflow_mask in zip(outflow_data_v_ns, outflow_data_mask):
                v_n = self.get_outflow_lbl(outflow_v_n, outflow_mask)
                size = size + 'c'
                head = head + '{}'.format(v_n)
                if outflow_v_n != all_v_ns[-1]: size = size + ' '
                if i != len(all_v_ns) - 1: head = head + ' & '
                i = i + 1

        size = size + '}'

        # --- UNIT BAR --- #

        unit_bar = ''
        for i, v_n in enumerate(all_v_ns):
            unit = self.get_unit_lbl(v_n)

            unit_bar = unit_bar + '{}'.format(unit)
            # if v_ns.index(v_n) != len(v_ns): unit_bar = unit_bar + ' & '
            if i != len(all_v_ns) - 1: unit_bar = unit_bar + ' & '

        head = head + ' \\\\'  # = \\
        unit_bar = unit_bar + ' \\\\ '

        # ====================== PRINT TABLE ================== #

        if print_head:
            print('\\begin{table*}[t]')
            print('\\begin{center}')
            print('\\begin{tabular}' + '{}'.format(size))
            print('\\hline')
            print(head)
            print(unit_bar)
            print('\\hline\\hline')

        for row in rows:
            print(row)

        if print_end:
            print('\\hline')
            print('\\end{tabular}')
            print('\\end{center}')
            print('\\caption{I am your table! }')
            print('\\label{' + "tbl:1" + '}')
            print('\\end{table*}')

        exit(0)

    def print_mult_table(self, list_simgroups, separateors, comment, label):

        assert len(list_simgroups) == len(separateors)

        group_rows = []
        for sim_group in list_simgroups:
            if len(sim_group) == 2: rows = self.get_compartison_rows_for_2(sim_group)
            elif len(sim_group) == 3: rows = self.get_compartison_rows_for_3(sim_group)
            else:
                raise ValueError("only 2 and 3 simulations can be compared.")
            group_rows.append(rows)

        print("data colleted. Printing...")

        self.print_intro_table()
        for i in range(len(list_simgroups)):
            # print(len(group_rows[i])); exit(1)
            for i_row, row in enumerate(group_rows[i]):
                print(row)
            print(separateors[i])
            # print("\\hline")

        self.print_end_table(comment, label)


class ALL_SIMULATIONS_TABLE:

    def __init__(self):
        #
        self.set_intable = Paths.output + "models_tmp2.csv"
        self.set_outtable = Paths.output + "models2.csv"
        #

        self.set_bern_passed = ["LS220_M14691268_M0_LK_SR"]
        self.set_list_eos = ["BLh", "SFHo", "SLy4", "DD2", "LS220"]
        self.set_list_res = ["HR", "SR", "LR", "VLR"]
        self.set_list_vis = ["LK"] # L5 L25 L50
        self.set_list_neut= ["M0"] # LK OldM0
        self.header = []
        self.table = []
        self.set_tmerg_v_n = "tmerg"
        self.set_tmerg_to_0 = False

        # assert len(simlist) > 0
        # self.sims = simlist

        #
        # self.header, self.table = self.load_table(self.intable)
        # Printcolor.blue("\tInitial table is loaded")
        # Printcolor.blue("------------------------------------")
        # print(self.header)
        # Printcolor.blue("------------------------------------")
        # print([run["name"] for run in self.table])
        # Printcolor.blue("------------------------------------")
        #

    @staticmethod
    def __load_table(file_name):
        table = []
        assert os.path.isfile(file_name)
        with open(file_name, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                table.append(row)
                # print(row)
            header = reader.fieldnames

        # self.out_header = header
        # self.out_table = table
        return header, table

    @staticmethod
    def save_table(header, table, file_name):

        if os.path.exists(file_name):
            os.remove(file_name)

        with open(file_name, "w") as csvfile:
            writer = csv.DictWriter(csvfile, header)
            writer.writeheader()
            for run in table:
                writer.writerow(run)
        Printcolor.green("> Table saved: {}".format(file_name))

    def check_sim_name(self):

        assert len(self.table) > 0
        for run in self.table:
            sim = run["name"]

            _eos = None
            for eos in self.set_list_eos:
                if sim.__contains__(eos):
                    _eos = eos
                    break
            if _eos == None: eos_color = "red"
            else: eos_color = "green"
            _res = None
            for res in self.set_list_res:
                if sim.__contains__(res):
                    _res = res
                    break
            if _res == None: res_color = "red"
            else: res_color = "green"
            _vis = None
            for vis in self.set_list_vis:
                if sim.__contains__(vis):
                    _vis = vis
                    break
            if _vis == None: vis_color = "yellow"
            else: vis_color = "green"
            _neut = None
            for neut in self.set_list_neut:
                if sim.__contains__(neut):
                    _neut = neut
                    break
            if _neut == None: neut_color = "yellow"
            else: neut_color = "green"

            Printcolor.print_colored_string(["\tCheck", sim, "eos:", _eos, "res:", _res, "vis:", _vis, "neut:", _neut],
                                        ["blue", "green", "blue", eos_color, "blue", res_color, "blue", vis_color, "blue", neut_color])

    def fill_self_parameters(self):
        for run in self.table:
            sim = run["name"]
            eos = run["name"].split('_')[0]
            m1m2 = run["name"].split('_')[1]
            if m1m2[0] != 'M':
                Printcolor.yellow(
                    "\tWarning. m1m2 is not [1] component of name. Using [2] (run:{})".format(run["name"]))
                # print("Warning. m1m2 is not [1] component of name. Using [2] (run:{})".format(run["name"]))
                m1m2 = run["name"].split('_')[2]
            else:
                m1m2 = ''.join(m1m2[1:])
            m1 = float(''.join(m1m2[:4])) / 1000
            m2 = float(''.join(m1m2[4:])) / 1000
            # print(run["name"])
            _eos = None
            for eos in self.set_list_eos:
                if sim.__contains__(eos):
                    _eos = eos
                    break
            if _eos == None: eos_color = "red"
            else: eos_color = "green"
            _res = None
            for res in self.set_list_res:
                if sim.__contains__(res):
                    _res = res
                    break
            if _res == None: res_color = "red"
            else: res_color = "green"
            _vis = None
            for vis in self.set_list_vis:
                if sim.__contains__(vis):
                    _vis = vis
                    break
            if _vis == None: vis_color = "yellow"
            else: vis_color = "green"
            _neut = None
            for neut in self.set_list_neut:
                if sim.__contains__(neut):
                    _neut = neut
                    break
            if _neut == None: neut_color = "yellow"
            else: neut_color = "green"
            #
            run["resolution"] = _res
            run["viscosity"] = _vis
            run["neutrinos"] = _neut
            run["M1"] = m1
            run["M2"] = m2
            run["EOS"] = eos
        Printcolor.green("> EOS, M1, M2 are extracted form the simulation name")

    def fill_self_bern_phase_status(self):
        assert len(self.set_bern_passed) > 0
        for run in self.table:
            if run["name"] in self.set_bern_passed:
                run["bern_phase"] = "passed"
            else:
                run["bern_phase"] = "not passed"
        Printcolor.green("> 'bern phase' is added from the list of ({}) simulations"
                         .format(len(self.set_bern_passed)))

    def fill_initial_data(self):
        '''

        '''
        v_ns = ["f0", "Mb1", "Mb2", "Mb", "MADM", "JADM", "q",
                "lam21", "Mg1", "C1", "k21", "R1", "lam22", "Mg2", "C2", "k22", "R2", "k2T", "Lambda",
                ]
        data = {}
        #
        for run in self.table:
            sim = run["name"]
            try:
                #
                o_init_data = LOAD_INIT_DATA(sim)
                #
                data[sim] = {}
                for v_n in v_ns:
                    data[sim][v_n] = o_init_data.get_par(v_n)
                #
            except IOError:
                Printcolor.red("{} Failed to locate initial data".format(sim))
        Printcolor.green("> Initial data collected")
        #
        missing_data = []
        for run in self.table:
            sim = run["name"]
            if sim in data.keys():
                for v_n in v_ns:
                    if not v_n in self.header:
                        raise NameError("v_n:{} is not in the file header".format(v_n))
                    run[v_n] = data[sim][v_n]
            else:
                missing_data.append(sim)
        Printcolor.green("> Initial data appended.")
        if len(missing_data) > 0:
            Printcolor.red("\t Data is not found for ({}) runs".format(len(missing_data)))

    def fill_comment(self):

        # custom:

        for run in self.table:
            sim = run["name"]
            o_init_data = LOAD_INIT_DATA(sim)
            eos = o_init_data.get_par("EOS")
            if eos == "DD2":
                n = o_init_data.get_par("run")
                run["comment"] = n
            elif eos == "SFHo":
                pz = o_init_data.get_par("pizza_eos")
                if pz.__contains__("2019"):
                    run["comment"] = "pz2019"
            else:
                pass
        Printcolor.green("> Comment filled")

    def fill_par_data(self):
        """

        """
        v_ns = ["nprofs", "nnuprofs", "fpeak", "EGW", "JGW",
                "tend","tcoll_gw","Mdisk3D", "tmerg_r", "Munb_tot", "Munb_bern_tot",
                "tcoll", "Mdisk", "tdisk3D","Mdisk3D", "tdisk3Dmax", "Mdisk3Dmax"]
        #
        data = {}
        #
        complete = True
        missing_data = []
        #
        for run in self.table:
            sim = run["name"]
            o_par = ADD_METHODS_ALL_PAR(sim)
            #
            data[sim] = {}
            for v_n in v_ns:
                # rescaling time to postmerger
                if v_n in ["tcoll", "tend", "tcoll_gw", "tdisk3Dmax", "tdisk3D"]:
                    tmerg = o_par.get_par("tmerg")
                    val = o_par.get_par(v_n) - tmerg
                    if val < 0.:
                        Printcolor.red("{}:{} - {}:{} < 0. sim:{}".format(v_n, o_par.get_par(v_n), "tmerg", tmerg, sim))
                        if v_n == "tcoll_gw":
                            raise ValueError("error")
                    data[sim][v_n] = val
                else:
                    data[sim][v_n] = o_par.get_par(v_n)
                if np.isnan(data[sim][v_n]):
                    complete = False
                    missing_data.append(sim)
        #
        Printcolor.green("> GW data collected")
        missing = []
        if not complete:
            for run in self.table:
                sim = run["name"]
                missing_v_ns = []
                for v_n in v_ns:
                    if np.isnan(data[sim][v_n]):
                        missing_v_ns.append(v_n)
                if len(v_ns) > 0:
                    missing.append(sim)
                    Printcolor.print_colored_string(["\t{}".format(sim), "missing:", missing_v_ns],
                                                    ["blue", "red", "red"])
        #
        for run in self.table:
            sim = run["name"]
            for v_n in v_ns:
                if not v_n in self.header:
                    raise NameError("v_n:{} is not in the file header".format(v_n))
                run[v_n] = data[sim][v_n]
        Printcolor.green("> GW data appended.")
        if len(missing) > 0:
            Printcolor.red("\t GW Data is missing for ({}) runs".format(len(missing)))

    def fill_mixed_data(self):

        v_ns = ["Mbi_Mb"]

        data = {}
        #
        # complete = True
        # missing_data = []
        #
        for run in self.table:
            sim = run["name"]
            o_par = ADD_METHODS_ALL_PAR(sim)
            o_init_data = LOAD_INIT_DATA(sim)
            #
            data[sim] = {}
            for v_n in v_ns:
                # rescaling time to postmerger

                if v_n == "Mbi_Mb":
                    mbi = float(o_init_data.get_par("Mb1") + o_init_data.get_par("Mb2"))
                    mass = o_par.get_total_mass()
                    mb0 = mass[0, 1]

                    val = (mbi / mb0)
                    if abs(val - 1.) > 0.5:
                        raise ValueError("Initial data appears to be totally wrong. "
                                         "{} Mbi:{} Mb_evol0:{}".format(sim, mbi, mb0))
                    data[sim][v_n] = val
                else:
                    raise NameError("Mixed dataMethod for v_n:{} not found".format(v_n))


        #
        # Printcolor.green("> Mixed data collected")
        # missing = []
        # if not complete:
        #     for run in self.table:
        #         sim = run["name"]
        #         missing_v_ns = []
        #         for v_n in v_ns:
        #             if np.isnan(data[sim][v_n]):
        #                 missing_v_ns.append(v_n)
        #         if len(v_ns) > 0:
        #             missing.append(sim)
        #             Printcolor.print_colored_string(["\t{}".format(sim), "missing:", missing_v_ns],
        #                                             ["blue", "red", "red"])
        # #
        for run in self.table:
            sim = run["name"]
            for v_n in v_ns:
                if not v_n in self.header:
                    raise NameError("v_n:{} is not in the file header".format(v_n))
                run[v_n] = data[sim][v_n]
        Printcolor.green("> Mixed data appended.")
        # if len(missing) > 0:
        #     Printcolor.red("\t GW Data is missing for ({}) runs".format(len(missing)))

    def fill_outflow_par(self, det=0, mask="geo"):

        if mask == "geo":
            v_ns = ["Mej_tot", "Ye_ave", "s_ave", "vel_inf_ave", "E_kin_ave", "theta_rms", "tend", "t98mass"]
        else:
            v_ns = ["Mej_tot", "Ye_ave", "s_ave", "vel_inf_ave", "E_kin_ave", "theta_rms"]


        # v_ns = ["Mej_tot", "Ye_ave", "s_ave", "vel_inf_ave", "E_kin_ave", "theta_rms", "tend", "t98mass"]
        # Mej_tot-geo,Ye_ave-geo,s_ave-geo,vel_inf_ave-geo,E_kin_ave-geo,theta_rms-geo,
        # Mej_tot-bern_geoend,Ye_ave-bern_geoend,s_ave-bern_geoend,vel_inf_ave-bern_geoend,E_kin_ave-bern_geoend,theta_rms-bern_geoend,
        data = {}
        #
        complete = True
        missing_sims = []
        missing_ppr_sims = []
        missing_data = []
        # Assesing data aveliablity
        for run in self.table:
            sim = run["name"]
            if not os.path.isdir(Paths.gw170817 + sim + '/'):
                missing_sims.append(sim)
            if not os.path.isdir(Paths.ppr_sims + sim + '/'):
                missing_ppr_sims.append(sim)
        #
        if len(missing_sims) > 0:
            Printcolor.yellow("Missing simulations from {} : \n\t{}"
                              .format(Paths.gw170817, missing_sims))
        if len(missing_ppr_sims) > 0:
            Printcolor.red("Missing ppr simulations from {} : \n\t{}"
                              .format(Paths.ppr_sims, missing_sims))
        # assessing table header compatibility
        missing_v_ns = []
        for v_n in v_ns:
            if not v_n + "-" + mask in self.header:
                missing_v_ns.append(v_n + "-" + mask)
                self.header.append(v_n + "-" + mask)
        if len(missing_v_ns) > 0:
            Printcolor.yellow("Missing v_ns from the header [corrected]: \n\t{}"
                              .format(missing_v_ns))
        #
        for run in self.table:
            if not self.set_tmerg_v_n in run.keys():
                raise ValueError("v_n:{} not found in run[{}]".format(self.set_tmerg_v_n, run['name']))
            sim = run["name"]
            if not sim in missing_sims:
                #
                if not self.set_tmerg_to_0:
                    tmerg = float(run[self.set_tmerg_v_n])
                else:
                    tmerg = 0
                #
                o_par = ADD_METHODS_ALL_PAR(sim)
                data[sim] = {}
                for v_n in v_ns:
                    try:
                        if v_n in ["tend", "t98mass"]:
                            data[sim][v_n + "-" + mask] = o_par.get_outflow_par(det, mask, v_n) - tmerg
                        else:
                            data[sim][v_n + "-" + mask] = o_par.get_outflow_par(det, mask, v_n)
                        if np.isnan(data[sim][v_n + "-" + mask]):
                            complete = False
                    except IOError:
                        missing_data.append(sim)
                        if not sim in missing_data:
                            missing_data.append(sim)
                        data[sim][v_n + "-" + mask] = np.nan
            else:
                Printcolor.yellow("skipping {}".format(sim))
        Printcolor.green("> OUTFLOW data collected")
        if len(missing_data) > 0:
            Printcolor.yellow("Missing data (files) for sims: \n\t{}"
                              .format(missing_data))
        Printcolor.yellow("")
        #
        for run in self.table:
            sim = run["name"]
            if sim not in missing_sims:
                if sim in data.keys():
                    for v_n in v_ns:
                        run[v_n + "-" + mask] = data[sim][v_n + "-" + mask]
        Printcolor.print_colored_string(["> OUTFLOW data appended", "det:", str(det), "mask", mask],
                                        ["green", "blue", "green", "blue", "green"])


        #
        # exit(1)
        #
        #
        #
        #
        #
        # for run in self.table:
        #     sim = run["name"]
        #     if sim in data.keys():
        #         try:
        #             o_par = ADD_METHODS_ALL_PAR(sim)
        #             #
        #             data[sim] = {}
        #             for v_n in v_ns:
        #                 # rescaling time to postmerger
        #                 if v_n in ["tend", "t98mass"]:
        #                     tmerg = o_par.get_par("tmerg")
        #                     data[sim][v_n + "-" + mask] = o_par.get_outflow_par(det, mask, v_n) - tmerg
        #                 else:
        #                     data[sim][v_n + "-" + mask] = o_par.get_outflow_par(det, mask, v_n)
        #                 if np.isnan(data[sim][v_n + "-" + mask]):
        #                     complete = False
        #                     missing_data.append(sim)
        #         except OSError:
        #             Printcolor.red("OSError sim:{} Might be missing simdir.".format(sim))
        #         except IOError:
        #             Printcolor.red("IOError sim:{} Might be missing data files".format(sim))
        #     else:
        #         Printcolor.red("KeyError sim:{} Might be simulation not being in table".format(sim))
        # #
        # Printcolor.green("> OUTFLOW data collected")
        # missing = []
        # if not complete:
        #     for run in self.table:
        #         sim = run["name"]
        #         if sim in data.keys():
        #             missing_v_ns = []
        #             for v_n in v_ns:
        #                 if np.isnan(data[sim][v_n + "-" + mask]):
        #                     missing_v_ns.append(v_n + "-" + mask)
        #             if len(v_ns) > 0:
        #                 missing.append(sim)
        #                 Printcolor.print_colored_string(["\t{}".format(sim), "missing:", missing_v_ns],
        #                                                 ["blue", "red", "red"])
        #
        # # ---
        # for run in self.table:
        #     sim = run["name"]
        #     if sim in data.keys():
        #         for v_n in v_ns:
        #             if not v_n + "-" + mask in self.header:
        #                 self.header.append(v_n + "-" + mask)
        #                 # run[]
        #                 # raise NameError("v_n:{} is not in the file header".format(v_n + "-" + mask))
        #
        #             run[v_n + "-" + mask] = data[sim][v_n + "-" + mask]
        # Printcolor.print_colored_string(["> OUTFLOW data appended", "det:", str(det), "mask", mask],
        #                                 ["green", "blue", "green", "blue", "green"])
        # if len(missing) > 0:
        #     Printcolor.red("\t OUTFLOW Data is missing for ({}) runs".format(len(missing)))

    def fill_parameter_manually(self, v_n, values_map, show_string=None, show_map=True):

        data = {}
        for run in self.table:
            sim = run["name"]
            data[sim] = {}
            if show_string != None:
                print(show_string.format(sim))
            if show_map:
                print(values_map)
            #
            value = raw_input("v_n:{} sim:{} ".format(v_n, sim))
            #
            if value in values_map.keys():
                value = values_map[value]
            else:
                value = ""
            data[sim][v_n] = value
        #
        for run in self.table:
            sim = run["name"]
            if not v_n in self.header:
                raise NameError("v_n:{} is not in the file header".format(v_n))
            run[v_n] = data[sim][v_n]
        #
        Printcolor.green("> Manual v_n:{} is appended.".format(v_n))

    def main(self):
        #
        self.check_sim_name()
        #
        self.fill_self_parameters()
        #
        self.fill_self_bern_phase_status()
        #
        self.fill_initial_data()
        #
        self.fill_comment()
        #
        self.fill_par_data()
        #
        self.fill_mixed_data()
        #
        self.fill_outflow_par(det=0, mask="geo")
        #
        self.fill_outflow_par(det=0, mask="bern_geoend")
        #
        self.fill_outflow_par(det=0, mask="geo_entropy_above_10")
        #
        self.fill_outflow_par(det=0, mask="geo_entropy_below_10")
        #
        self.save_table(self.header, self.table, self.set_outtable)
        #

    def load_table(self):
        #
        self.header, self.table = self.__load_table(self.set_intable)
        Printcolor.blue("\tInitial table is loaded")
        Printcolor.blue("------------------------------------")
        print(self.header)
        Printcolor.blue("------------------------------------")
        print([run["name"] for run in self.table])
        Printcolor.blue("------------------------------------")
        #


class GET_PAR_FROM_TABLE(ALL_SIMULATIONS_TABLE):

    def __init__(self):
        #
        ALL_SIMULATIONS_TABLE.__init__(self)
        #
        self.name='name'

    def compute_par2(self, data_dic, v_n, dtype=float):
        sim = data_dic[self.name]
        v_n = str(v_n)
        if v_n == "q":
            return float(data_dic["M1"]) / float(data_dic["M2"])
        if v_n == "q1mtot":
            mtot = float(data_dic["M1"]) + float(data_dic["M2"])
            q = float(data_dic["M1"]) / float(data_dic["M2"])
            return q / mtot
        if v_n == "q2":
            return (float(data_dic["M1"]) / float(data_dic["M2"])) ** 2
        if v_n == "mchirp":
            mtot = float(data_dic["M1"]) + float(data_dic["M2"])
            eta = float(data_dic["M1"]) * float(data_dic["M2"]) / mtot ** 2
            mchirp = mtot * eta ** (3./5.)

            # chirp = (float(data_dic["M1"]) * float(data_dic["M2"]) ** (3. / 5.)) / (
            #         (float(data_dic["M1"]) + float(data_dic["M2"])) ** (1. / 5.))
            return mchirp
        elif v_n == "qmchirp" or v_n == "mchirpq":

            chirp = (float(data_dic["M1"]) * float(data_dic["M2"]) ** (3. / 5.)) / (
                    (float(data_dic["M1"]) + float(data_dic["M2"])) ** (1. / 5.))
            q = float(data_dic["M1"]) / float(data_dic["M2"])
            return q * chirp
        elif v_n == "q1mchirp":
            chirp = (float(data_dic["M1"]) * float(data_dic["M2"]) ** (3. / 5.)) / (
                    (float(data_dic["M1"]) + float(data_dic["M2"])) ** (1. / 5.))
            q = float(data_dic["M1"]) / float(data_dic["M2"])
            return q / chirp
        elif v_n == "mchirp2":
            chirp = (float(data_dic["M1"]) * float(data_dic["M2"]) ** (3. / 5.)) / (
                    (float(data_dic["M1"]) + float(data_dic["M2"])) ** (1. / 5.))
            return chirp ** 2
        elif v_n == "mtot":
            mtot = float(data_dic["M1"]) + float(data_dic["M2"])
            return mtot
        elif v_n == "mtot2":
            mtot = float(data_dic["M1"]) + float(data_dic["M2"])
            return mtot ** 2
        elif v_n == "symq":
            q = float(data_dic["M1"]) * float(data_dic["M2"]) / (float(data_dic["M1"]) + float(data_dic["M2"])) ** 2
            return q
        elif v_n == "symqmchirp":
            q = float(data_dic["M1"]) * float(data_dic["M2"]) / (float(data_dic["M1"]) + float(data_dic["M2"])) ** 2
            chirp = (float(data_dic["M1"]) * float(data_dic["M2"]) ** (3. / 5.)) / (
                    (float(data_dic["M1"]) + float(data_dic["M2"])) ** (1. / 5.))
            return q * chirp
        elif v_n == "mtotsymqmchirp":
            mtot = float(data_dic["M1"]) + float(data_dic["M2"])
            q = float(data_dic["M1"]) * float(data_dic["M2"]) / (float(data_dic["M1"]) + float(data_dic["M2"])) ** 2
            chirp = (float(data_dic["M1"]) * float(data_dic["M2"]) ** (3. / 5.)) / (
                    (float(data_dic["M1"]) + float(data_dic["M2"])) ** (1. / 5.))
            return mtot * q * chirp
        elif v_n == "symq2":
            q = float(data_dic["M1"]) * float(data_dic["M2"]) / (float(data_dic["M1"]) + float(data_dic["M2"])) ** 2
            return q ** 2
        elif v_n == "qmtot" or v_n == "symq":
            mtot = float(data_dic["M1"]) + float(data_dic["M2"])
            q = float(data_dic["M1"]) / float(data_dic["M2"])
            return q * mtot
        elif v_n == "symqmtot" or v_n == "mtotsymq":
            mtot = float(data_dic["M1"]) + float(data_dic["M2"])
            q = float(data_dic["M1"]) * float(data_dic["M2"]) / (float(data_dic["M1"]) + float(data_dic["M2"])) ** 2
            return mtot * q
        #
        elif str(v_n).__contains__("_mult_"):
            v_n1 = v_n.split("_mult_")[0]
            v_n2 = v_n.split("_mult_")[-1]
            val1 = self.get_par(sim, v_n1, dtype=float)
            val2 = self.get_par(sim, v_n2, dtype=float)
            return val1 * val2
        #
        elif str(v_n).__contains__("_dev_"):
            v_n1 = v_n.split("_dev_")[0]
            v_n2 = v_n.split("_dev_")[-1]
            val1 = self.get_par(sim, v_n1, dtype=float)
            val2 = self.get_par(sim, v_n2, dtype=float)
            return val1 / val2

        else:
            raise NameError("unrecognized option to compute parameter: v_n:{} "
                            .format(v_n))

    def compute_par(self, data_dic, v_n, dtype=float):
        if dtype == float:
            if v_n == "q":
                return float(data_dic["M1"]) / float(data_dic["M2"])
            #
            elif v_n == "Mej_tot-geo_1":
                chirp = float(data_dic["M1"]) * float(data_dic["M2"]) / (
                            float(data_dic["M1"]) + float(data_dic["M2"])) ** (1. / 5.)
                return float(data_dic["Mej_tot-geo"]) / chirp
            elif v_n == "Mej1":
                chirp = float(data_dic["M1"]) * float(data_dic["M2"]) / (
                            float(data_dic["M1"]) + float(data_dic["M2"])) ** (1. / 5.)
                return float(data_dic["Mej"]) / chirp
            #
            elif v_n == "Mej_tot-geo_2":
                q = float(data_dic["M1"]) * float(data_dic["M2"]) / (float(data_dic["M1"]) + float(data_dic["M2"])) ** 2
                return float(data_dic["Mej_tot-geo"]) / q
            elif v_n == "Mej2":
                q = float(data_dic["M1"]) * float(data_dic["M2"]) / (float(data_dic["M1"]) + float(data_dic["M2"])) ** 2
                return float(data_dic["Mej"]) / q
            #
            elif v_n == "Mej_tot-geo_3":
                q = float(data_dic["M1"]) * float(data_dic["M2"]) / (float(data_dic["M1"]) + float(data_dic["M2"])) ** 2
                mtot = float(data_dic["M1"]) + float(data_dic["M2"])
                return float(data_dic["Mej_tot-geo"]) / (q * mtot)
            elif v_n == "Mej3":
                q = float(data_dic["M1"]) * float(data_dic["M2"]) / (float(data_dic["M1"]) + float(data_dic["M2"])) ** 2
                mtot = float(data_dic["M1"]) + float(data_dic["M2"])
                return float(data_dic["Mej"]) / (q * mtot)
            #
            elif v_n == "Mej_tot-geo_4":
                q = float(data_dic["M1"]) * float(data_dic["M2"]) / (float(data_dic["M1"]) + float(data_dic["M2"])) ** 2
                mtot = float(data_dic["M1"]) + float(data_dic["M2"])
                return float(data_dic["Mej_tot-geo"]) / (q * mtot ** 2)
            elif v_n == "Mej4":
                q = float(data_dic["M1"]) * float(data_dic["M2"]) / (float(data_dic["M1"]) + float(data_dic["M2"])) ** 2
                mtot = float(data_dic["M1"]) + float(data_dic["M2"])
                return float(data_dic["Mej"]) / (q * mtot ** 2)
            #
            elif v_n == "Mej_tot-geo_5":
                q = float(data_dic["M1"]) * float(data_dic["M2"]) / (float(data_dic["M1"]) + float(data_dic["M2"])) ** 2
                mtot = float(data_dic["M1"]) + float(data_dic["M2"])
                return float(data_dic["Mej_tot-geo"]) / (q ** 2)
            elif v_n == "Mej5":
                q = float(data_dic["M1"]) * float(data_dic["M2"]) / (float(data_dic["M1"]) + float(data_dic["M2"])) ** 2
                mtot = float(data_dic["M1"]) + float(data_dic["M2"])
                return float(data_dic["Mej"]) / (q ** 2)
            #
            elif v_n == "Mej_tot-geo_6":
                q = float(data_dic["M1"]) * float(data_dic["M2"]) / (float(data_dic["M1"]) + float(data_dic["M2"])) ** 2
                mtot = float(data_dic["M1"]) + float(data_dic["M2"])
                return float(data_dic["Mej_tot-geo"]) * (q ** 2)
            elif v_n == "Mej6":
                q = float(data_dic["M1"]) * float(data_dic["M2"]) / (float(data_dic["M1"]) + float(data_dic["M2"])) ** 2
                mtot = float(data_dic["M1"]) + float(data_dic["M2"])
                return float(data_dic["Mej"]) * (q ** 2)
            else:
                raise NameError("no methods found for v_n : {}".format(v_n))
        else:
            raise NameError("no methods for v_n:{} dtype:{}".format(v_n, dtype))

    def get_simdic(self, sim):

        sim_dic = {}
        for run in self.table:
            # run = {'name':blabla, 'q':...}
            __sim = run[self.name]
            if sim == __sim:
                sim_dic = run
                break
        # checking if the dictionary is found
        if len(sim_dic.keys()) == 0:
            raise NameError("sim:{} is not found in the table:{}"
                            .format(sim, [run[self.name] for run in self.table]))
        return sim_dic

    def get_par(self, sim, v_n, dtype=float):
        # checking if data is loaded
        if len(self.header) == 0:
            raise IOError("Input table is not loaded. Run 'load_table()' ")
        # locating data dictionary
        sim_dic = self.get_simdic(sim)
        # getting the needed value
        if v_n in self.header:
            # if the data is available in the header
            if dtype == float:
                try:
                    _val = float(sim_dic[v_n])
                except ValueError:
                    _val = np.nan
                    Printcolor.red("failed converting to float: sim:'{}' v_n:'{}' val:'{}'"
                                     .format(sim, v_n, sim_dic[v_n]))
                    # raise ValueError("failed converting to float: sim:'{}' v_n:'{}' val:'{}'"
                    #                  .format(sim, v_n, sim_dic[v_n]))
            elif dtype == str:
                try:
                    _val = str(sim_dic[v_n])
                except ValueError:
                    raise ValueError("failed converting to str: sim:{} v_n:{} val: {}"
                                     .format(sim, v_n, sim_dic[v_n]))
            else:
                raise TypeError("dtype: {} is not recognized".format(dtype))
        else:
            # if data is not directly available
            _val = self.compute_par2(sim_dic, v_n, dtype)
        return _val

    def get_par_with_error(self, sims, v_n, deferr = 0.2):

        if v_n == "nsims":
            return len(sims), len(sims), len(sims)

        if len(sims) == 0:
            raise ValueError("no simualtions passed")
        _resols, _values = [], []

        for sim in sims:
            _val = self.get_par(sim, v_n, dtype=float)
            # print(sim, _val)
            _res = "fuck"
            for res in resolutions.keys():
                if sim.__contains__(res):
                    _res = res
                    break
            if _res == "fuck":
                _res = "SR"
                #raise NameError("fuck")
            _resols.append(resolutions[_res])
            _values.append(_val)
        _resols = np.array(_resols)
        _values = np.array(_values)
        if len(sims) == 1:
            return _values[0], _values[0] - deferr * _values[0], _values[0] + deferr * _values[0]
        elif len(sims) == 2:
            delta = np.abs(_values[0] - _values[1])
            if _resols[0] < _resols[1]:
                return _values[0], _values[0] - delta, _values[0] + delta
            else:
                return _values[1], _values[1] - delta, _values[1] + delta
        elif len(sims) == 3:
            # print(_resols, _values)
            assert len(_resols) == len(_values)
            _resols_, _values_ = UTILS.x_y_z_sort(_resols, _values)  # 123, 185, 236
            delta1 = np.abs(_values_[0] - _values_[1])
            delta2 = np.abs(_values_[1] - _values_[2])
            # print(_values, _values_); exit(0)
            return _values_[1], _values_[1] - delta1, _values_[1] + delta2
        else:
            raise ValueError("Too many simulations: {}".format(sims))

    def get_is_prompt_coll(self, sims, delta_t = 3., v_n_tcoll="tcoll_gw", v_n_tmerg="tmerg"):

        isprompt = False
        isbh = False
        for sim in sims:
            tcoll = self.get_par(sim, v_n_tcoll)
            if np.isinf(tcoll):
                pass
            else:
                isbh = True
                # tmerg = self.get_par(sim, v_n_tmerg)
                # if not tcoll > tmerg:
                #     raise ValueError("{}:{} > {}:{} sim:{}".format(v_n_tcoll, tcoll, v_n_tmerg, tmerg, sim))
                if float(tcoll) < delta_t * 1e-3:
                    isprompt = True

        return isbh, isprompt

    def set_simdic(self, sim, dic):
        sim_dic = {}
        for run in self.table:
            # run = {'name':blabla, 'q':...}
            __sim = run[self.name]
            if sim == __sim:
                run = dic
                break

# class LOAD_ALL_SIMULATIONS_TABLE:
#
#     def __init__(self):
#         self.simulations = pd.read_csv(Paths.output + Files.models)
#         self.simulations = self.simulations.set_index("name")
#         # self.simulations["res"] = self.simulations["name"]
#         self.add_res()
#         self.add_q()
#         self.add_viscosity()
#         # self.add_dyn_phase_status()
#
#     def add_res(self):
#         """
#         because of the way how pandas load dictionary, one has to loop over
#         all the first row entries to select the needed resolution
#         then if the resolution SR, LR or HR is found in the simulation
#         name it is added, otherwise -- it complains and adds SR.
#         :return: Nothing
#         """
#         Printcolor.blue("...adding resolutions from sim. names")
#         resolutions = []
#         for sim_name in [sim[0] for sim in self.simulations.iterrows()]:
#             appended = False
#             for res in Lists.res:
#                 # print(sim_name)
#                 if str(sim_name).__contains__(str(res)):
#                     resolutions.append(res)
#                     appended = True
#             if not appended:
#                 Printcolor.yellow("Warning: No 'res' found in {} name. Using 'SR' instead".format(sim_name))
#                 resolutions.append('SR')
#                 # raise NameError("for sim:{} resolution not found".format(sim_name))
#         self.simulations["res"] = resolutions
#
#     def add_viscosity(self):
#         """
#         because of the way how pandas load dictionary, one has to loop over
#         all the first row entries to select the needed resolution
#         then if the resolution 'LK' is found in the simulation
#         name it is added, otherwise - it complains and adds '-' (no viscosity).
#         :return: Nothing
#         """
#         Printcolor.blue("...adding viscosity from sim. names")
#         viscosities = []
#         for sim_name in [sim[0] for sim in self.simulations.iterrows()]:
#             appended = False
#             for vis in Lists.visc:
#                 # print(sim_name)
#                 if str(sim_name).__contains__(str(vis)):
#                     viscosities.append(vis)
#                     appended = True
#             if not appended:
#                 print("Note: No 'visc' found in {} name. Using '-' instead".format(sim_name))
#                 viscosities.append('-')
#                 # raise NameError("for sim:{} resolution not found".format(sim_name))
#         self.simulations["visc"] = viscosities
#
#     def add_q(self):
#         Printcolor.blue("...adding q = M1/M2")
#         self.simulations["q"] = self.simulations["M1"] / self.simulations["M2"]
#
#     # def add_dyn_phase_status(self):
#     #     Printcolor.blue("...adding dynamical phase info from static list")
#     #     passed_not_passed = []
#     #
#     #     for name_sim, sim in self.simulations.iterrows():
#     #         if name_sim in Lists.dyn_not_pas:
#     #             passed_not_passed.append("passed")
#     #         else:
#     #             passed_not_passed.append("not passed")
#     #     self.simulations["dyn_phase"] = passed_not_passed
#
#     def get_all(self):
#         return self.simulations
#
#     def get_selected_models(self, lim_dic):
#         """
#         To use, provide a dictioany like :
#         {'EOS':['DD2'], 'res':['SR'], 'q':[1.], 'visc':['-'], 'maximum':'tend'}
#         Where the desired preperties are in the lists [], as they can be
#         selected in a several.
#         the last entry - 'maximum' will result in returning 1
#         simulation with the maximum (minimum) of this value
#
#         :param lim_dic:
#         :return:
#         """
#
#         cropped_sims = copy.deepcopy(self.simulations)
#         for v_n in lim_dic.keys():
#
#             if not v_n in cropped_sims.keys() and v_n != 'names':
#                 raise NameError("key: {} not in table.keys()\n{}"
#                                 .format(v_n, cropped_sims.keys()))
#             elif v_n == 'names':
#                 # .loc[[sim1, sim2 ...]] returns a dataframe with these simulations
#                 cropped_sims = cropped_sims.loc[lim_dic[v_n]]
#
#             # if v_n == 'name':
#             #     for value in lim_dic[v_n]:
#             #         cropped_sims = cropped_sims[cropped_sims[value] == value]
#
#             elif v_n == 'EOS':
#                 for value in lim_dic[v_n]:
#                     cropped_sims = cropped_sims[cropped_sims[v_n] == value]
#             elif v_n == 'res':
#                 for value in lim_dic[v_n]:
#                     cropped_sims = cropped_sims[cropped_sims[v_n] == value]
#             elif v_n == 'q':
#                 for value in lim_dic[v_n]:
#                     cropped_sims = cropped_sims[cropped_sims[v_n] == value]
#             elif v_n == 'visc':
#                 for value in lim_dic[v_n]:
#                     cropped_sims = cropped_sims[cropped_sims[v_n] == value]
#             #
#             # else:
#             #     for value in lim_dic[v_n]:
#             #         cropped_sims = cropped_sims[cropped_sims[v_n] == value]
#
#             # if v_n == 'name':
#             #     for value in lim_dic[v_n]:
#             #         cropped_sims = cropped_sims[cropped_sims[v_n] == value]
#             # else:
#             #     raise NameError("limit dic entry: {} is not reognized".format(v_n))
#
#         if 'maximum' in lim_dic.keys():
#             return cropped_sims[cropped_sims[lim_dic['maximum']] ==
#                                 cropped_sims[lim_dic['maximum']].max()]
#
#         if 'minimum' in lim_dic.keys():
#             return cropped_sims[cropped_sims[lim_dic['minimum']] ==
#                                 cropped_sims[lim_dic['minimum']].min()]
#
#         return cropped_sims
#
#     def save_new_table(self, new_simulations, fname="../output/summary.csv"):
#
#         header = []
#         table = []
#
#         for sim_name, sim in new_simulations.iterrows():
#             for v_n in sim.keys():
#                 header.append(v_n)
#
#         table = []
#         for sim_name, sim in new_simulations.iterrows():
#             run = {}
#             for v_n in sim.keys():
#                 value = sim[v_n]
#                 if value == None:
#                     run[str(v_n)] = ''
#                 else:
#                     run[str(v_n)] = value
#             table.append(run)
#
#                 # print(sim_name)
#                 # print(sim[v_n])
#                 # exit(1)
#
#         with open(fname, "w") as csvfile:
#             writer = csv.DictWriter(csvfile, header)
#             writer.writeheader()
#             for run in table:
#                 writer.writerow(run)
#
#     def get_value(self, sim, v_n):
#         return self.simulations.get_value(sim, v_n)




def __update_new_table_adding_more_data():

    o_tbl = ALL_SIMULATIONS_TABLE()
    o_tbl.load_table()
    o_tbl.set_intable = Paths.output + "models_tmp2.csv"
    o_tbl.set_outtable = Paths.output + "models3.csv"
    o_tbl.fill_input_table_with_all()
    Printcolor.green("done")

def __fill_radice_table_with_shocked_tidal():
    path = "/data01/numrel/vsevolod.nedora/postprocessed_radice/"
    fname = "/hist_entropy.dat"
    o_tbl = ALL_SIMULATIONS_TABLE()
    o_tbl.set_intable = "../output/radice2018_summary.csv"
    o_tbl.set_outtable = "../output/radice2018_summary2.csv"
    new_v_ns = ["Mej_shocked", "Mej_tidal"]
    o_tbl.load_table()
    # check if data for all sims is avialable
    for run in o_tbl.table:
        sim = run["name"]
        # check if the data is on the dir:
        if os.path.isdir(path + sim + '/'):
            Printcolor.green("Sim: {} is found".format(sim))
        else:
            Printcolor.red("Sim: {} is not found".format(sim))
    # load the histograms and extract the data
    for run in o_tbl.table:
        sim = run["name"]
        # check if the data is on the dir:
        if os.path.isfile(path + sim + '/' + fname):
            Printcolor.green("File for {} is found".format(sim))
        else:
            Printcolor.red("File for {} is not found".format(sim))
    #
    for run in o_tbl.table:
        sim = run["name"]
        if os.path.isfile(path + sim + '/' + fname):
            table = np.loadtxt(path + sim + '/' + fname, usecols=(0,1))
            s = table[:,0]
            m = table[:,1]
            m_shocked = np.sum(m[s > 10.])
            m_tidal = np.sum(m[s <= 10.])
            m_total = np.sum(m)
            mej_tbl = float(run["Mej"])
            delta = np.abs(mej_tbl - m_total)
            if delta > 1e-4:
                color="red"
            else:
                color="green"
            Printcolor.print_colored_string(["sim:",sim,"m_shocked","{:.2f}".format(m_shocked*1e2),
                                             "m_tidal", "{}".format(m_tidal*1e2), "delta:", "{}".format(delta)],
                                            ["blue","green","blue","green",
                                             "blue", "green", "blue", color])
            run["Mej_shocked"] = m_shocked
            run["Mej_tidal"] = m_tidal
        else:
            Printcolor.red("File for {} is not found".format(sim))
        #
    o_tbl.save_table(o_tbl.header, o_tbl.table, o_tbl.set_outtable)
    Printcolor.green("Done")
    exit(1)

def __fill_radice_table_with_shocked_tidal_from_outflowed():

    path = "/data01/numrel/vsevolod.nedora/postprocessed_radice2/"
    dirname_shocked = "outflow_0/geo_entropy_above_10/"
    dirname_tidal = "outflow_0/geo_entropy_below_10/"
    Paths.ppr_sims =  '/data01/numrel/vsevolod.nedora/postprocessed_radice2/'
    Paths.gw170817 = '/data1/numrel/WhiskyTHC/Backup/2017/'
    o_tbl = ALL_SIMULATIONS_TABLE()
    o_tbl.set_intable = Paths.output + "radice2018_summary.csv"
    o_tbl.set_outtable = Paths.output + "radice2018_summary3.csv"
    o_tbl.set_tmerg_to_0 = True
    o_tbl.load_table()
    # check if data for all sims is avialable
    for run in o_tbl.table:
        sim = run["name"]
        # check if the data is on the dir:
        if os.path.isdir(path + sim + '/'):
            Printcolor.green("Sim: {} is found. Subdirs:".format(sim), comma=True)
            if os.path.isdir(path + sim + '/' + dirname_shocked):
                Printcolor.green("shocked".format(sim), comma=True)
            else:
                Printcolor.red("shocked".format(sim), comma=True)
            if os.path.isdir(path + sim + '/' + dirname_tidal):
                Printcolor.green("tidal".format(sim))
            else:
                Printcolor.red("tidal".format(sim))
        else:
            Printcolor.red("Sim: {} is not found".format(sim))

    # load the histograms and extract the data

    o_tbl.fill_outflow_par(0, "geo")
    o_tbl.fill_outflow_par(0, "bern_geoend")
    o_tbl.fill_outflow_par(0, "geo_entropy_above_10")
    o_tbl.fill_outflow_par(0, "geo_entropy_below_10")

    #
    o_tbl.save_table(o_tbl.header, o_tbl.table, o_tbl.set_outtable)
    Printcolor.green("Done")
    #
    exit(1)

def __update_table():
    tbl = GET_PAR_FROM_TABLE()
    tbl.set_tmerg_v_n = "tmerg_r"
    tbl.set_intable = Paths.output + "models_tmp2.csv"
    tbl.set_outtable = Paths.output + "models3.csv"
    tbl.load_table()
    tbl.main()
    Printcolor.green("all done")
    exit(1)

def __set_manually_parameter():
    # v_n = "disk_phase"
    # values_map = {
    #     "1": "formed_ns_accretion",
    #     "2": "fromed_bh_accretion",
    #     "3": "not_formed",
    #     "4": "unclear",
    #     "-": ""
    # }
    values_map = {
        "1": "NS",
        "2": "BH",
        "3": "PC",
        "4": "unclear",
        "-": ""
    }
    v_n = "outcome"
    show_string = "\nfeh {}/waveforms/tmergtcoll.png "
    #
    tbl = GET_PAR_FROM_TABLE()
    tbl.set_tmerg_v_n = "tmerg_r"
    tbl.set_intable = Paths.output + "models_tmp2.csv"
    tbl.set_intable = Paths.output + "models_tmp2.csv"
    tbl.load_table()
    tbl.fill_parameter_manually(v_n, values_map, show_string)
    tbl.save_table(tbl.header, tbl.table, tbl.set_intable)
    exit(1)

''' --------------------------- GROUP --------------------------- '''

def __apply_mod(v_n, val, mod):
    # print(v_n, val, mod)
    if mod != None and mod != "":
        # exit(1)
        if mod[0] == '*':
            mult_by = float(mod.split("*")[-1])
            val = val * mult_by
            # print(val); exit(1)
        elif mod[0] == "/":
            dev_by = float(mod.split('/')[-1])
            val = val / dev_by
        else:
            raise NameError("v_n:{} mod:{} is invalid. Use mod '*float' or '/float' "
                            .format(v_n, mod))

    return val

def average_over_resolutions(val_sr, val_lr, val_hr, errormethod="st.div"):
    tmp = np.array([val_sr, val_lr, val_hr])
    if errormethod == "st.div":
        if len(tmp[~np.isnan(tmp)]) == 3:
            mean, stddiv = standard_div(tmp)
            return mean, stddiv
        elif np.isnan(val_lr) and ~np.isnan(val_hr) and ~np.isnan(val_hr):
            tmp = np.array([val_hr, val_sr])
            mean, stddiv = standard_div(tmp)
            return mean, 2 * stddiv
        elif np.isnan(val_hr) and ~np.isnan(val_sr) and ~np.isnan(val_lr):
            tmp = np.array([val_sr, val_lr])
            mean, stddiv = standard_div(tmp)
            return mean, 2 * stddiv
        elif np.isnan(val_sr) and ~np.isnan(val_hr) and ~np.isnan(val_lr):
            tmp = np.array([val_hr, val_lr])
            mean, stddiv = standard_div(tmp)
            return mean, 2 * stddiv
        elif len(tmp[~np.isnan(tmp)]) == 1:
            tmp = tmp[~np.isnan(tmp)]
            return float(tmp), np.nan
        elif len(tmp[~np.isnan(tmp)]) == 0:
            return np.nan, np.nan
        else:
            raise ValueError("something welt wring: {}".format(tmp))
    else:
        raise NameError("errormethod is not recognized: {}".format(errormethod))

def __average_value_for_group(group, v_n, method="st.div", deferr=0.2):
    """
    :param group: panda.dataframe of atmost 3 simualtions
    :param v_n: name of the variable in the dataframe
    :param method: None or st.div
    :return: mean, err1, err2
    """
    ress = sorted(list(group.resolution))
    if len(ress) > 3:
        raise ValueError("too many resolutions! {}".format(group.resolutons))
    #
    if v_n == "group":
        val = list(group.group)[0]
        return val
    #
    if v_n == "resolution" or v_n == "res":
        val = " ".join(ress)
        return val
    #
    if "SR" in ress and "LR" in ress and "HR" in ress:
        if method == None or method == "":
            sr = group[group.resolution == "SR"]
            val_sr = float(list(sr[v_n])[0])
            return val_sr, 0., 0.
        elif method == "st.div":
            hr = group[group.resolution == "HR"]
            sr = group[group.resolution == "SR"]
            lr = group[group.resolution == "LR"]

            val_sr = float(sr[v_n])
            val_lr = float(lr[v_n])
            val_hr = float(hr[v_n])

            mean, stddiv = standard_div([val_hr, val_sr, val_lr])
            return mean, stddiv, stddiv
        else:
            raise NameError("method:{} is not recognised. Use None of st.div".format(method))
    elif len(ress) == 2 and "SR" in ress and "LR" in ress:
        if method == None or method == "":
            sr = group[group.resolution == "SR"]
            val_sr = float(list(sr[v_n])[0])
            return val_sr, 0., 0.
        elif method == "st.div":
            # hr = group[group.resolution == "HR"]
            sr = group[group.resolution == "SR"]
            lr = group[group.resolution == "LR"]

            val_sr = float(sr[v_n])
            val_lr = float(lr[v_n])

            mean, stddiv = standard_div([val_sr, val_lr])
            return mean, stddiv, stddiv
        else:
            raise NameError("method:{} is not recognised. Use None of st.div".format(method))
            # return str("$" + val_sr + r"^{+" + str(del1) + r"} _{-" + str(del2)) + "} $"
    elif len(ress) == 2 and "SR" in ress and "HR" in ress:
        if method == None or method == "":
            sr = group[group.resolution == "SR"]
            val_sr = float(list(sr[v_n])[0])
            return val_sr, 0., 0.
        elif method == "st.div":
            hr = group[group.resolution == "HR"]
            sr = group[group.resolution == "SR"]

            val_sr = float(sr[v_n])
            val_hr = float(hr[v_n])

            mean, stddiv = standard_div([val_hr, val_sr])
            return mean, stddiv, stddiv
        else:
            raise NameError("method:{} is not recognised. Use None of st.div".format(method))
            # return str("$" + val_sr + r"^{+" + str(del1) + r"} _{-" + str(del2)) + "} $"
        pass
    elif len(ress) == 2 and "LR" in ress and "HR" in ress:
        # print("\t 3 ress:{} v_n:{} fmt:{} err:{}".format(ress, v_n, fmt, err))
        if method == None or method == "":
            lr = group[group.resolution == "LR"]
            val_lr = float(list(lr[v_n])[0])
            return val_lr, 0., 0.
        elif method == "st.div":
            hr = group[group.resolution == "HR"]
            lr = group[group.resolution == "LR"]

            val_lr = float(lr[v_n])
            val_hr = float(hr[v_n])

            mean, stddiv = standard_div([val_hr, val_lr])
            return mean, stddiv, stddiv
        else:
            raise NameError("method:{} is not recognised. Use None of st.div".format(method))
        pass
    elif len(ress) == 1 and "SR" in ress:
        # SR - X*SR, SR - X*SR
        if method == None or method == "":
            sr = group[group.resolution == "SR"]
            val_sr = float(list(sr[v_n])[0])
            return val_sr, 0., 0.
        elif method == "st.div":
            # hr = group[group.resolution == "HR"]
            sr = group[group.resolution == "SR"]

            val_sr = float(sr[v_n])

            del1 = del2 = val_sr * deferr

            return val_sr, del1, del2
        else:
            raise NameError("method:{} is not recognised. Use None of st.div".format(method))
    elif len(ress) == 1 and "LR" in ress:
        # LR - Y*LR, SR - Y*LR
        # print("\t 5 ress:{} v_n:{} fmt:{} err:{}".format(ress, v_n, fmt, err))
        if method == None or method == "":
            lr = group[group.resolution == "LR"]
            val_lr = float(list(lr[v_n])[0])
            return val_lr, 0., 0.
        elif method == "st.div":

            lr = group[group.resolution == "LR"]

            val_lr = float(lr[v_n])

            del1 = del2 = val_lr * deferr

            return val_lr, del1, del2
        else:
            raise NameError("method:{} is not recognised. Use None of st.div".format(method))
            # return str("$" + val_lr + "$")#r"^{" + str(del1) + r"} _{" + str(del2)) + "} $"
    elif len(ress) == 1 and "HR" in ress:
        # HR - Z*HR, SR - Z*HR
        # print("\t 6 ress:{} v_n:{} fmt:{} err:{}".format(ress, v_n, fmt, err))
        if method == None or method == "":
            hr = group[group.resolution == "HR"]
            val_hr = float(list(hr[v_n])[0])
            return val_hr, 0., 0.
        elif method == "st.div":
            hr = group[group.resolution == "HR"]

            val_hr = float(hr[v_n])

            del1 = del2 = val_hr * deferr

            return val_hr, del1, del2
        else:
            raise NameError("method:{} is not recognised. Use None of st.div".format(method))
            # return str("$" + val_hr + "$")#r"^{" + str(del1) + r"} _{" + str(del2)) + "} $"
    else:
        print(group)
        raise ValueError("\t 7. Unrecognized resoltion setup \n "
                         "ress:{} v_n:{} err:{}".format(ress, v_n, method))

def convert_models_to_groups_table(simulations):
    """
    :simulations: pandas.DataFrame with all the simualtions
    Loads simulation table, extract data for groups of simualtion (with different resolution)
    computes average quantites of selected v_ns and saves as a new .csv and returns dataframe
    :return: pandas.dataframe
    """
    import pandas

    ''' --------------| setup |---------- '''

    # taken from the highest resolution available [TABLE]
    const_v_ns = ["q", "viscosity", "EOS", "M1", "M2", "Mb1", "Mb2", "Mb", "Mg1", "Mg2", "R1", "R2", "C1",
                  "C2", "k21", "k22", "lam21", "lam22", "Lambda", "k2T", "MADM", "EGW", "JADM", "comment"]

    # taken from the end of every simulation [TABLE]
    finnish_v_ns = ["Mej_tot-geo", "vel_inf_ave-geo", "theta_rms-geo", "Mdisk3Dmax", "Mbi_Mb",  "Ye_ave-geo", "Mwinddot"]

    # taken at maximum common time [DATA]
    maxtime_v_ns = [("tdisk3D", "Mdisk3D"),
                    ("tend_wind", "Mej_tot-bern_geoend"),
                    ("tend_wind", "vel_inf_ave-bern_geoend"),
                    ("tend_wind", "Ye_ave-bern_geoend"),
                    ("tend_wind", "theta_rms-bern_geoend")]

    # for every res SR ^{HR} _{LR} [TABLE]
    for_res_v_ns = ["tcoll_gw", "tend", "Mbi_Mb", "outcome"]

    outfname = "../output/groups.csv"

    ''' --------------------------------- '''

    #
    new_data_frame = {}
    #
    # groups = sorted(list(set(simulations["group"])))
    eoss = sorted(list(set(simulations["EOS"])))
    #
    groups = []
    for eos in eoss:
        sel = simulations[simulations["EOS"] == eos]
        sel = sel.sort_values(by="q")
        i_groups = []
        for i, m in sel.iterrows():
            ii_group = m.group
            if not ii_group in i_groups:
                i_groups.append(ii_group)
        # print(sel["group"])
        # i_groups = list(set(sel["group"]))
        # print(i_groups); exit(1)
        # print('\t', i_groups)
        groups = groups + i_groups
        print("eos:{} {}".format(eos, len(i_groups)))
    new_data_frame["group"] = groups
    #
    # qs = [float(np.array(simulations[simulations["group"] == group].q)[0]) for group in groups]
    # print(qs); exit(1)
    # new_data_frame["group"] = groups
    new_data_frame["resolution"] = \
        [" ".join(simulations[simulations["group"] == group].resolution) for group in groups]

    assert len(new_data_frame["resolution"]) == len(new_data_frame["group"])

    #  taken from the highest resolution available [MODELS]
    for v_n in const_v_ns:
        values = []
        for group in groups:
            sims = simulations[simulations["group"] == group]
            assert len(sims) > 0 and len(sims) <= 3
            value = list(sims[v_n])[0]
            values.append(value)
        #
        new_data_frame[v_n] = values

    # taken from the end of every simulation
    for v_n in finnish_v_ns:
        values = []
        errors = []
        for group in groups:
            sims = simulations[simulations["group"] == group]
            assert len(sims) > 0 and len(sims) <= 3
            value, err1, err2 = __average_value_for_group(sims, v_n, "st.div", 0.2)
            values.append(value)
            errors.append(err1)
        new_data_frame[v_n] = values
        new_data_frame["err-"+v_n] = errors

    # for every res SR ^{HR} _{LR} [TABLE]
    for v_n in for_res_v_ns:
        values = []
        for group in groups:
            sims = simulations[simulations["group"] == group]
            vals = list(sims[v_n])
            #
            values.append(vals)
            #
        #
        new_data_frame[v_n] = values

    # taken at maximum common time [DATA]
    for tuple_ in maxtime_v_ns:
        v_n_time, v_n = tuple_[0], tuple_[1]
        time_values = []
        values = []
        errors = []
        for group in groups:
            sims = simulations[simulations["group"] == group]
            assert len(sims) > 0 and len(sims) <= 3
            o_data = AVERAGE_PAR(sims.index)
            #
            mmaxtime, value, err1, err2 = o_data.get_ave_val_last_common_time(v_n, method="st.div")
            # print(mmaxtime, value); # exit(1)
            time_values.append(mmaxtime)
            values.append(value)
            errors.append(err1)
        #
        new_data_frame[v_n_time] = time_values
        new_data_frame[v_n] = values
        new_data_frame["err-"+v_n] = errors

    # taken at maximum common time [DATA]
    # for v_n in maxtime_v_ns:
    #     values = []
    #     errors = []
    #     for group in groups:
    #         sims = simulations[simulations["group"] == group]
    #         assert len(sims) > 0 and len(sims) <= 3
    #         value, err1, err2 = __average_value_for_group(sims, v_n, "st.div", 0.2)
    #         values.append(value)
    #         errors.append(err1)
    #     new_data_frame[v_n] = values
    #     new_data_frame["err-"+v_n] = errors
    #
    df = pandas.DataFrame(new_data_frame, index=groups)
    df.set_index("group")
    df.to_csv(outfname)
    print("saved as {}".format(outfname))

    return df


if __name__ == '__main__':

    # __set_manually_parameter()


    parser = ArgumentParser(description="working with tables")
    # parser.add_argument("-s", dest="sim", required=True, help="task to perform")

    # parser.add_argument("-s", dest="sim", required=True, help="task to perform")
    parser.add_argument("-t", dest="task", required=False, nargs='+', default=[], help="tasks to perform")
    # parser.add_argument("--v_n", dest="v_ns", required=False, nargs='+', default=[], help="variable (or group) name")
    # parser.add_argument("--rl", dest="reflevels", required=False, nargs='+', default=[], help="reflevels")
    # parser.add_argument("--it", dest="iterations", required=False, nargs='+', default=[], help="iterations")
    # parser.add_argument('--time', dest="times", required=False, nargs='+', default=[], help='Timesteps')
    # #
    # parser.add_argument("-o", dest="outdir", required=False, default=Paths.ppr_sims, help="path for output dir")
    # parser.add_argument("-i", dest="simdir", required=False, default=Paths.gw170817, help="path to simulation dir")
    # parser.add_argument("--overwrite", dest="overwrite", required=False, default="no", help="overwrite if exists")
    # #
    # parser.add_argument("--sym", dest="symmetry", required=False, default=None, help="symmetry (like 'pi')")
    # Info/checks
    args = parser.parse_args()
    glob_task = args.task
    # glob_sim = args.sim
    # glob_simdir = args.simdir
    # glob_outdir = args.outdir
    # glob_v_ns = args.v_ns
    # glob_rls = args.reflevels
    # glob_its = args.iterations
    # glob_times = args.times
    # glob_symmetry = args.symmetry
    # glob_overwrite = args.overwrite
    # simdir = Paths.gw170817 + glob_sim + '/'
    # resdir = Paths.ppr_sims + glob_sim + '/'

    if "update_table" in glob_task:
        __update_table()
    elif "update_groups" in glob_task:
        from model_sets.models import simulations_nonblacklisted, mask_for_with_dynej, mask_for_non_ahfix
        sims = simulations_nonblacklisted[mask_for_with_dynej & mask_for_non_ahfix]
        convert_models_to_groups_table(sims)
        exit(1)
    else:
        raise NameError("task: {} is not recognized. ".format(glob_task))



    # __set_manually_parameter()

    from model_sets.models import simulations_nonblacklisted, mask_for_with_dynej, mask_for_non_ahfix
    sims = simulations_nonblacklisted[mask_for_with_dynej & mask_for_non_ahfix]
    convert_models_to_groups_table(sims)
    exit(1)

    __update_table()

    __fill_radice_table_with_shocked_tidal_from_outflowed()

    # __update_new_table_adding_more_data()
    # __fill_radice_table_with_shocked_tidal()
    #
    tbl = GET_PAR_FROM_TABLE()
    tbl.set_intable = Paths.output + "models3.csv"
    tbl.load_table()
    for run in tbl.table:
        sim = run[tbl.name]
        mtot = float(run["M1"]) + float(run["M2"])
        mgrav = float(run["Mg1"]) + float(run["Mg2"])
        Printcolor.blue("sim: {} mtot:{:.2f} mbtot:{:.2f}".format(sim, mtot, mgrav))
    exit(1)
    #
    tbl = GET_PAR_FROM_TABLE()
    tbl.set_intable = Paths.output + "models_tmp2.csv"
    tbl.set_outtable = Paths.output + "models2.csv"
    tbl.load_table()
    tbl.fill_input_table_with_all()
    exit(1)
    # testing

    tbl = GET_PAR_FROM_TABLE()
    tbl.set_intable = Paths.output + "models2.csv"
    tbl.load_table()
    print("{}".format(tbl.get_par("LS220_M14691268_M0_SR", "Lambda", dtype=float)))
    print(tbl.get_par_with_error(["BLh_M10201856_M0_LK_SR", "BLh_M10201856_M0_LK_LR", "BLh_M10201856_M0_LK_HR"],
                                 "Mej_tot-geo"))

