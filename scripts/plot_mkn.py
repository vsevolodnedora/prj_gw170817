from __future__ import division

import numpy as np
import pandas as pd
import sys
import os
import copy
import h5py
import math
import csv
from scipy import interpolate
from glob import glob
import matplotlib

#from legacy.prj_visc_ej import ax

matplotlib.use('agg')
import matplotlib.pyplot as plt
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')


from matplotlib.colors import LogNorm, Normalize

sys.path.append('/data01/numrel/vsevolod.nedora/bns_ppr_tools/')
# from preanalysis import LOAD_INIT_DATA
# from outflowed import EJECTA_PARS
# from preanalysis import LOAD_ITTIME
# from plotting_methods import PLOT_MANY_TASKS
# from profile import LOAD_PROFILE_XYXZ, LOAD_RES_CORR, LOAD_DENSITY_MODES, MAINMETHODS_STORE, MAINMETHODS_STORE_XYXZ
from utils import Paths, Lists, Labels, Constants, Printcolor, UTILS, Files, PHYSICS

from make_fit2 import Fitting_Coefficients, Fitting_Functions, Fit_Data

from model_sets import groups as md
from data import ADD_METHODS_ALL_PAR
from comparison import TWO_SIMS, THREE_SIMS
from settings import resolutions
from uutils import *

#from sys import path
sys.path.append(Paths.mkn)
try:
    from mkn_bayes import MKN
except ImportError:
    try:
        from mkn import MKN
    except ImportError:
        raise ImportError("Failed to import mkn from MKN (set path is: {} ".format(Paths.mkn))

__curdir__ = "/data01/numrel/vsevolod.nedora/prj_gw170817/scripts/"
__outplotdir__ = "../figs/all3/mkn_fit/"
if not os.path.isdir(__outplotdir__):
    os.mkdir(__outplotdir__)

__mkn__ = {"tasklist":["nrmkn", "plotmkn", "mkn", "print_table"],
           "geometries":["iso","aniso"],
           "components":["dynamics", "spiral", "wind", "secular"],
           "detectors":[0,1],
           "masks":["geo","bern_geoend"],
           "bands":["g", "z", "Ks"]}

''' --- computing lightcurve -- '''

class Vals:
    q = None
    Lambda = None

class COMPUTE_LIGHTCURVE():

    def __init__(self, sim, outdir=None):

        self.sim = sim
        self.path = "/data01/numrel/vsevolod.nedora/prj_gw170817/scripts/lightcurves/"
        self.output_fname = 'mkn_model.h5'
        if sim != None:
            self.o_data = ADD_METHODS_ALL_PAR(sim)
        else:
            self.o_data = None
        #
        if outdir == None:
            self.outdir = Paths.ppr_sims+self.sim+'/mkn/'
            self.outfpath = Paths.ppr_sims+self.sim+'/mkn/' + self.output_fname
        else:
            self.outdir = outdir
            self.outfpath = self.outdir + self.output_fname
        # if criteria == '' or criteria == 'geo':
        #     self.path_to_outflow_dir = LISTS.loc_of_sims + sim + '/outflow_{}/'.format(det)
        # elif criteria == 'bern' or criteria == ' bern':
        #     self.path_to_outflow_dir = LISTS.loc_of_sims + sim + '/outflow_{}_b/'.format(det)
        # elif criteria == 'bern dyn' or criteria == ' bern dyn':
        #     self.path_to_outflow_dir = LISTS.loc_of_sims + sim + '/outflow_{}_b_d/'.format(det)
        # elif criteria == 'bern wind' or criteria == ' bern wind':
        #     self.path_to_outflow_dir = LISTS.loc_of_sims + sim + '/outflow_{}_b_w/'.format(det)
        # else:
        #     raise NameError("Criteria '{}' is not recongnized".format(criteria))

        self.dyn_ejecta_profile_fpath = ""
        self.psdyn_ejecta_profile_fpath = ""

        if self.sim == None:
            self.set_use_dyn_NR = False
            self.set_use_bern_NR = False
        else:
            self.set_use_dyn_NR = True
            self.set_use_bern_NR = True
        self.set_dyn_iso_aniso       = None#"aniso"
        self.set_psdyn_iso_aniso     = None#"aniso"
        self.set_wind_iso_aniso      = None#"aniso"
        self.set_secular_iso_aniso   = None#"aniso"
        # --dyn aniso --spirla
        self.glob_params    = {}
        self.glob_vars      = {}
        self.ejecta_params  = {}
        self.ejecta_vars    = {}
        self.source_name    = {}

        # self.set_glob_par_var_source(True, dyn_ejecta_profile_fpath,
        #                              True, psdyn_ejecta_profile_fpath)
        # self.set_dyn_par_var(self.set_dyn_iso_aniso)
        # self.set_psdyn_par_var(self.set_psdyn_iso_aniso)
        # self.set_wind_par_war(self.set_wind_iso_aniso)
        # self.set_secular_par_war(self.set_secular_iso_aniso)

        # self.compute_save_lightcurve(write_output=True)

    ''' change parameters '''

    def set_glob_par_var_source(self, NR_data=True,NR2_data=True):

        self.glob_params = {'lc model'   : 'grossman',  # model for the lightcurve (grossman or villar)
                       #              'mkn model': 'aniso1comp',  # possible choices: iso1comp, iso2comp, iso3comp, aniso1comp, aniso2comp, aniso3comp
                       'omega frac':1.0,      # fraction of the solid angle filled by the ejecta
                       'rad shell': False,     # exclude the free streaming part
                       'v_min':     1.e-7,    # minimal velocity for the Grossman model
                       'n_v':       400,      # number of points for the Grossman model
                       'vscale':    'linear', # scale for the velocity in the Grossman model
                       'sigma0':    0.11,     # parameter for the nuclear heating rate
                       'alpha':     1.3,      # parameter for the nuclear heating rate
                       't0eps':     1.3,      # parameter for the nuclear heating rate
                       'cnst_eff':  0.3333,   # parameter for the constant heating efficiency
                       'n slices':  24,       # number for the number of slices along the polar angle [12,18,24,30]
                       'dist slices': 'cos_uniform',  # discretization law for the polar angle [uniform or cos_uniform]
                       'time min':  3600.,    # minimum time [s]
                       'time max':  2000000., # maximum time [s]
                       'n time':    200,      # integer number of bins in time
                       'scale for t': 'log',    # kind of spacing in time [log - linear - measures]
                       # my parameters
                       'save_profs': True,
                       'NR_data':   NR_data,     # use (True) or not use (False) NR profiles
                       'NR2_data':  NR2_data,
                       'NR_filename': self.dyn_ejecta_profile_fpath,
                       'NR2_filename': self.psdyn_ejecta_profile_fpath
                       # path of the NR profiles, necessary if NR_data is True
                       }

        self.source_name = 'AT2017gfo'
        # self.source_name = 'AT2017gfo view_angle=180/12.' # change the source properties

        if NR_data and NR2_data and self.set_wind_iso_aniso == None and self.set_secular_iso_aniso == None:
            mdisk = None
        else:
            if self.sim != None:
                mdisk = self.o_data.get_par("Mdisk3D")
                if np.isnan(mdisk):
                    raise ValueError("mass of the disk is not avilable (nan) for sim:{}".format(self.sim))
            else:
                print("\tUsing default disk mass")
                mdisk = 0.012

        self.glob_vars = {'m_disk':     mdisk, # mass of the disk [Msun], useful if the ejecta is expressed as a fraction of the disk mass
                         'eps0':        2e19, # prefactor of the nuclear heating rate [erg/s/g]
                         'view_angle':  180/12.,  # [deg]; if None, it uses the one in source properties
                         'source_distance': 40.,  # [pc] ; if None, it uses the one in source properties
                         'T_floor_LA':  1000., # floor temperature for Lanthanides [K]
                         'T_floor_Ni':  5000., # floor temperature for Nikel [K]
                         'a_eps_nuc':   0.5, # variation of the heating rate due to weak r-process heating: first parameter
                         'b_eps_nuc':   2.5, # variation of the heating rate due to weak r-process heating: second parameter
                         't_eps_nuc':   1.0} # variation of the heating rate due to weak r-process heating: time scale [days]

        # return glob_params, glob_vars, source_name

    def set_dyn_par_var(self, iso_or_aniso, det=0, mask="dyn"):

        if not self.sim == None and self.set_use_dyn_NR:
            mej = self.o_data.get_outflow_par(det,mask,"Mej_tot")
            if np.isnan(mej):
                raise ValueError("Ejecta mass for det:{} mask:{} is not avialble (nan)".format(det,mask))
        else:
            mej = 0.015

        if iso_or_aniso == 'iso':
            self.ejecta_params['dynamics'] = {'mass_dist':'uniform', 'vel_dist':'uniform', 'op_dist':'uniform',
                                     'therm_model':'BKWM', 'eps_ye_dep':'LR','v_law':'poly', 'entropy':10, 'tau':5}
            self.ejecta_vars['dynamics'] = {'xi_disk':          None,
                                           'm_ej':              0.003,
                                           'step_angle_mass':   None,
                                           'high_lat_flag':     None,
                                           'central_vel':       0.24,
                                           'high_lat_vel':      None,
                                           'low_lat_vel':       None,
                                           'step_angle_vel':    None,
                                           'central_op':        30.,
                                           'high_lat_op':       None,
                                           'low_lat_op':        None,
                                           'step_angle_op':     None,
                                           'T_floor':           1000}
        elif iso_or_aniso == 'aniso':
            self.ejecta_params['dynamics'] = {'mass_dist':'sin2', 'vel_dist':'uniform', 'op_dist':'step',
                                              'therm_model':'BKWM', 'eps_ye_dep':'PBR', 'entropy': 20., 'tau':5,
                                              'v_law':'poly', 'use_kappa_table':False}#, 'use_kappa_table':False}
            self.ejecta_vars['dynamics'] = {'xi_disk':          None,
                                           'm_ej':              mej, # 0.00169045, # - LS220 | 0.00263355 - DD2
                                           'step_angle_mass':   None,
                                           'high_lat_flag':     None,
                                           'central_vel':       0.30, # changed from 0.33
                                           'high_lat_vel':      None,
                                           'low_lat_vel':       None,
                                           'step_angle_vel':    None,
                                           'central_op':        None,
                                           'high_lat_op':       5.,  # F:1
                                           'low_lat_op':        20., # F:30    # does not work for NR
                                           'step_angle_op':     np.pi/4,
                                           'T_floor':           None} # F:30
        elif iso_or_aniso == "":
            pass
        else:
            raise NameError('only iso or aniso')

        # return dyn_ej_pars, dyn_ej_vars

    def set_spiral_par_var(self, iso_or_aniso, det=0, mask="bern_geoend"):

        if not self.sim == None and self.set_use_bern_NR:
            mej = self.o_data.get_outflow_par(det,mask,"Mej_tot")
            if np.isnan(mej):
                raise ValueError("Ejecta mass for det:{} mask:{} is not avialble (nan)".format(det,mask))
        else:
            mej = 0.002

        if iso_or_aniso == 'iso':
            self.ejecta_params['dynamics'] = {'mass_dist':'uniform', 'vel_dist':'uniform', 'op_dist':'uniform',
                                     'therm_model':'BKWM', 'eps_ye_dep':'LR','v_law':'poly', 'entropy':10, 'tau':5}
            self.ejecta_vars['dynamics'] = {'xi_disk':          None,
                                           'm_ej':              0.003,
                                           'step_angle_mass':   None,
                                           'high_lat_flag':     None,
                                           'central_vel':       0.24,
                                           'high_lat_vel':      None,
                                           'low_lat_vel':       None,
                                           'step_angle_vel':    None,
                                           'central_op':        30.,
                                           'high_lat_op':       None,
                                           'low_lat_op':        None,
                                           'step_angle_op':     None,
                                           'T_floor':           1000}
        elif iso_or_aniso == 'aniso':
            self.ejecta_params['spiral'] = {'mass_dist':'sin', 'vel_dist':'uniform', 'op_dist':'step'   ,
                                       'therm_model':'BKWM', 'eps_ye_dep':"PBR",'v_law':'poly', 'use_kappa_table':False,
                                              'entropy':20, 'tau':30}
            self.ejecta_vars['spiral'] = {'xi_disk':          None,
                                           'm_ej':              mej, # 0.00169045, # - LS220 | 0.00263355 - DD2
                                           'step_angle_mass':   None,
                                           'high_lat_flag':     None,
                                           'central_vel':       0.20, # changed from 0.33
                                           'high_lat_vel':      None,
                                           'low_lat_vel':       None,
                                           'step_angle_vel':    None,
                                           'central_op':        None,
                                           'high_lat_op':       1.,  # F:1
                                           'low_lat_op':        30., # F:30    # does not work for NR
                                           'override_m_ej':     False,  # for manual import
                                           'step_angle_op':     math.radians(15.),
                                           'T_floor':           None} # F:30
        elif iso_or_aniso == "":
            pass
        else:
            raise NameError('only iso or aniso')

        # return dyn_ej_pars, dyn_ej_vars

    def set_wind_par_war(self, iso_or_aniso):

        if iso_or_aniso == 'iso':
            self.ejecta_params['wind'] = {'mass_dist':'uniform', 'vel_dist':'uniform', 'op_dist':'uniform',
                                     'therm_model':'BKWM', 'eps_ye_dep':'LR','v_law':'poly', 'entropy':20, 'tau':33}
            self.ejecta_vars['wind'] = {'xi_disk' :         None,
                                        'm_ej':             0.02,
                                        'step_angle_mass':  None,
                                        'high_lat_flag':    True,
                                        'central_vel':      0.08,
                                        'high_lat_vel':     None,
                                        'low_lat_vel':      None,
                                        'step_angle_vel':   None,
                                        'central_op':       1.0,
                                        'high_lat_op':      None,
                                        'low_lat_op':       None,
                                        'step_angle_op':    None,
                                        'T_floor':          1000}
        elif iso_or_aniso == 'aniso':
            self.ejecta_params['wind'] = {'mass_dist':'step', 'vel_dist':'uniform', 'op_dist':'step',
                                          'therm_model':'BKWM', 'eps_ye_dep':'PBR', 'entropy': 10., 'tau':33, 'v_law':'poly'}
            self.ejecta_vars['wind'] = {
                         'xi_disk':         None, # 0.1 default
                         'm_ej':            0.004,
                         'step_angle_mass': np.pi/6.,
                         'high_lat_flag':   True,
                         'central_vel':     0.1, #  V:0.08
                         'high_lat_vel':    None,
                         'low_lat_vel':     None,
                         'step_angle_vel':  None,
                         'central_op':      None,
                         'high_lat_op':     1.0, # 0.1
                         'low_lat_op':      5.0, # F
                         'step_angle_op':   np.pi/6.,
                         'T_floor':         None} # F: 45 | might need # N:30
        elif iso_or_aniso == "":
            pass
        else:
            raise NameError("iso_or_aniso: {} is not recognized".format(iso_or_aniso))

    def set_secular_par_war(self, iso_or_aniso):

        if iso_or_aniso == 'iso':
            self.ejecta_params['secular'] =  {'mass_dist':'uniform', 'vel_dist':'uniform', 'op_dist':'uniform',
                                     'therm_model':'BKWM', 'eps_ye_dep':'LR','v_law':'poly', 'entropy':20, 'tau':33}
            self.ejecta_vars['secular'] = {
                            'xi_disk':          0.4,
                            'm_ej':             None,
                            'step_angle_mass':  None,
                            'high_lat_flag':    None,
                            'central_vel':      0.06,
                            'high_lat_vel':     None,
                            'low_lat_vel':      None,
                            'step_angle_vel':   None,
                            'central_op':       5.0,
                            'low_lat_op':       None,
                            'high_lat_op':      None,
                            'step_angle_op':    None,
                            'T_floor':           1000}
        elif iso_or_aniso == 'aniso':
            self.ejecta_params['secular'] = {'mass_dist':'sin2', 'vel_dist':'uniform', 'op_dist':'uniform',
                                             'therm_model':'BKWM', 'eps_ye_dep':'PBR', 'entropy': 10., 'tau':33, 'v_law':'poly'}
            self.ejecta_vars['secular'] = {
                            'xi_disk':          None, # default: 0.2
                            'm_ej':             0.03,
                            'step_angle_mass':  None,
                            'high_lat_flag':    None,
                            'central_vel':      0.08, # F: 0.04 def:0.06
                            'high_lat_vel':     None,
                            'low_lat_vel':      None,
                            'step_angle_vel':   None,
                            'central_op':       10.0, #
                            'low_lat_op':       None,
                            'high_lat_op':      None,
                            'step_angle_op':    None,
                            'T_floor':          None}
        elif iso_or_aniso == "":
            pass
        else:
            raise NameError("iso_or_aniso: {} is not recognized".format(iso_or_aniso))

    ''' set parameters '''

    def set_dyn_ej_nr(self, det, mask):
        fpath = Paths.ppr_sims+self.sim+'/'+"outflow_{:d}".format(det)+'/'+mask+'/'
        if not os.path.isdir(fpath):
            raise IOError("dir with outflow + mask is not found: {}".format(fpath))
        fname = "ejecta_profile.dat"
        fpath = fpath + fname
        if not os.path.isfile(fpath):
            raise IOError("file for mkn NR data is not found: {}".format(fpath))
        self.dyn_ejecta_profile_fpath = fpath

    def set_bern_ej_nr(self, det, mask):
        fpath = Paths.ppr_sims + self.sim + '/' + "outflow_{:d}".format(det) + '/' + mask + '/'
        if not os.path.isdir(fpath):
            raise IOError("dir with outflow + mask is not found: {}".format(fpath))
        fname = "ejecta_profile.dat"
        fpath = fpath + fname
        if not os.path.isfile(fpath):
            raise IOError("file for mkn NR data is not found: {}".format(fpath))
        self.psdyn_ejecta_profile_fpath = fpath

    # def set_par_war(self):
    #     #
    #     self.set_glob_par_var_source(self.set_use_dyn_NR, self.set_use_bern_NR)
    #     self.set_dyn_par_var(self.set_dyn_iso_aniso)
    #     self.set_spiral_par_var(self.set_psdyn_iso_aniso)
    #     self.set_wind_par_war(self.set_wind_iso_aniso)
    #     self.set_secular_par_war(self.set_secular_iso_aniso)

    ''' set parameters '''

    def modify_input(self, place, v_n, value):

        ''' Replaces the default value with the given '''

        if place == 'glob_params':
            if not v_n in self.glob_params.keys():
                raise NameError('v_n:{} is not in glob_params:{}'
                                .format(v_n, self.glob_params.keys()))
            self.glob_params[v_n] = value

        if place == 'glob_vars':
            if not v_n in self.glob_vars.keys():
                raise NameError('v_n:{} is not in glob_vars:{}'
                                .format(v_n, self.glob_vars.keys()))
            self.glob_vars[v_n] = value

        # ejecta_params[]
        if place == 'ejecta_params[dynamics]':
            if not v_n in self.ejecta_params['dynamics'].keys():
                raise NameError(
                    'v_n:{} is not in ejecta_params[dynamics]:{}'
                        .format(v_n, self.ejecta_params['dynamics'].keys()))
            self. ejecta_params['dynamics'][v_n] = value

        if place == 'ejecta_params[wind]':
            if not v_n in self.ejecta_params['wind'].keys():
                raise NameError('v_n:{} is not in ejecta_params[wind]:{}'
                                .format(v_n, self.ejecta_params['wind'].keys()))
            self.ejecta_params['wind'][v_n] = value

        if place == 'ejecta_params[secular]':
            if not v_n in self.ejecta_params['secular'].keys():
                raise NameError(
                    'v_n:{} is not in ejecta_params[secular]:{}'
                        .format(v_n, self.ejecta_params['secular'].keys()))
            self.ejecta_params['secular'][v_n] = value

        # shell_vars[]
        if place == 'shell_vars[dynamics]':
            if not v_n in self.ejecta_vars['dynamics'].keys():
                raise NameError('v_n:{} is not in shell_vars[dynamics]:{}'
                                .format(v_n, self.ejecta_vars['dynamics'].keys()))
            self.ejecta_vars['dynamics'][v_n] = value

        if place == 'shell_vars[wind]':
            if not v_n in self.ejecta_vars['wind'].keys():
                raise NameError('v_n:{} is not in shell_vars[wind]:{}'
                                .format(v_n, self.ejecta_vars['wind'].keys()))
            self.ejecta_vars['wind'][v_n] = value

        if place == 'shell_vars[secular]':
            if not v_n in self.ejecta_vars['secular'].keys():
                raise NameError('v_n:{} is not in shell_vars[wind]:{}'
                                .format(v_n, self.ejecta_vars['secular'].keys()))
            self.ejecta_vars['secular'][v_n] = value

    def compute_save_lightcurve(self, write_output = True ,fname = None):
        # glob_params, glob_vars, ejecta_params, shell_vars, source_name_d

        if len(self.glob_params.keys()) == 0:
            raise ValueError("parameters are not set. Use 'set_par_war()' for that")

        if not os.path.isdir(self.outdir):
            print("making directory {}".format(self.outdir))
            os.mkdir(self.outdir)

        print('I am initializing the model')
        # glob_params, glob_vars, ejecta_params, shell_vars, source_name = self.mkn_parameters()

        # go into the fold with all classes of mkn
        os.chdir(Paths.mkn)
        # from mkn import MKN

        # print(self.ejecta_vars['psdynamics']['m_ej'])
        model = MKN(self.glob_params, self.glob_vars, self.ejecta_params, self.ejecta_vars, self.source_name)

        print('I am computing the light curves')
        #    r_ph,L_bol,T_eff = model.lightcurve(ejecta_vars,glob_params['NR_data'],glob_params['NR_filename'])
        r_ph, L_bol, T_eff = model.E.lightcurve(model.angular_distribution,
                                                model.omega_distribution,
                                                model.time,
                                                model.ejecta_vars,
                                                model.ejecta_params,
                                                model.glob_vars,
                                                model.glob_params)
        # exit(1)
        print('I am computing the likelihood')
        logL = model.log_likelihood(r_ph, T_eff)

        if (write_output):
            print('I am printing out the output')
            model.write_output_h5(r_ph, T_eff, L_bol)
            model.write_filters_h5()

        # copy the result into sim folder and go back into the main script folder

        if (write_output):
            # from shutil import move
            from shutil import copyfile
            # move('./mkn_model.txt', self.path_to_outflow_dir + 'mkn_model.txt')
            if fname == None:
                copyfile('./mkn_model.h5', self.outfpath)
            else:
                copyfile('./mkn_model.h5',  self.outdir + fname)

        os.chdir(__curdir__)
        return logL

    # table methods
    def print_latex_table_of_glob_pars(self):

        '''
        \begin{table}
            \footnotesize
                \begin{tabular}[t]{|p{3.2cm}|c|}
                    \hline
                    bla   & 1\\ \hline
                    blubb & 2 \\ \hline
                    bla   & 1\\ \hline
                    blubb & 2 \\ \hline
                    bla   & 1\\ \hline
                    blubb & 2 \\ \hline
                    xxx   & x \\ \hline
                \end{tabular}
                % \hfill
                \begin{tabular}[t]{|p{3.2cm}|c|}
                    \hline
                    bla&1\\ \hline
                    blubb&2 \\ \hline
                    bla&1\\ \hline
                    blubb&2 \\ \hline
                    bla&1\\ \hline
                    blubb&2 \\ \hline
                \end{tabular}
            \hfill
            \caption{99 most frequent hashtags in the data set.}
        \end{table}
        :return:
        '''

        # glob_params, glob_vars, source_name = self.mkn_parameters_glob()

        print('\n')
        print('\\begin{table}[!ht]')
        print('\\footnotesize')

        # table of glob. parameters
        print('\\begin{tabular}[t]{ p{2.0cm} c }')
        print('\\hline')

        # printing rows
        for v_n, value in zip(self.glob_params.keys(), self.glob_params.values()):
            print(' {}  &  {} \\\\'.format(v_n.replace('_', '\\_'), value))
        print('\\hline')

        # table of glob. vars
        print('\\end{tabular}')
        print('\\begin{tabular}[t]{ p{2.0cm} c }')
        print('\\hline')

        for v_n, value in zip(self.glob_vars.keys(), self.glob_vars.values()):
            print(' {}  &  {} \\\\'.format(v_n.replace('_', '\\_'), value))

        print('\\hline')
        print('\\end{tabular}')
        print('\\caption{Global parameters (left) and global variables (right)}')
        print(r'\label{tbl:mkn_global}')
        print('\\end{table}')

    def print_latex_table_of_ejecta_pars(self, components):

        print('\n')
        print('\\begin{table}[!ht]')
        print('\\footnotesize')

        if "dynamics" in components:

            # dyn_ej_pars, dyn_ej_vars = self.mkn_parameters_dynamics()

            print('\\begin{tabular}[t]{ p{2.cm} c }')
            print('Dynamic & \\\\')
            print('\\hline')

            for v_n, value in zip(self.ejecta_params["dynamics"].keys(), self.ejecta_params["dynamics"].values()):
                if value == None:
                    print(' {}  &  {} \\\\'.format(v_n.replace('_', '\\_'), value))
                elif isinstance(value, float) or isinstance(value, int):
                    print(' {}  &  {:.2f} \\\\'.format(v_n.replace('_', '\\_'), value))
                elif isinstance(value, str):
                    print(' {}  &  {} \\\\'.format(v_n.replace('_', '\\_'), value))
                else:
                    raise ValueError("value:{} is niether float nor string".format(value))
            print('\\hline')

            print('\\hline')

            for v_n, value in zip(self.ejecta_vars["dynamics"].keys(), self.ejecta_vars["dynamics"].values()):
                if value == None:
                    print(' {}  &  {} \\\\'.format(v_n.replace('_', '\\_'), value))
                elif isinstance(value, float) or isinstance(value, int):
                    print(' {}  &  {:.2f} \\\\'.format(v_n.replace('_', '\\_'), value))
                elif isinstance(value, str):
                    print(' {}  &  {} \\\\'.format(v_n.replace('_', '\\_'), value))
                else:
                    raise ValueError("value:{} is niether float nor string".format(value))
            print('\\hline')

            print('\\end{tabular}')

        if "spiral" in components:

            # wind_pars, wind_vars = self.mkn_parameters_wind()

            print('\\begin{tabular}[t]{ p{2.cm} c }')
            print('Spiral & \\\\')
            print('\\hline')

            for v_n, value in zip(self.ejecta_params["spiral"].keys(), self.ejecta_params["spiral"].values()):
                if value == None:
                    print(' {}  &  {} \\\\'.format(v_n.replace('_', '\\_'), value))
                elif isinstance(value, float) or isinstance(value, int):
                    print(' {}  &  {:.2f} \\\\'.format(v_n.replace('_', '\\_'), value))
                elif isinstance(value, str):
                    print(' {}  &  {} \\\\'.format(v_n.replace('_', '\\_'), value))
                else:
                    raise ValueError("value:{} is niether float nor string".format(value))
            print('\\hline')

            print('\\hline')

            for v_n, value in zip(self.ejecta_vars["spiral"].keys(), self.ejecta_vars["spiral"].values()):
                if value == None:
                    print(' {}  &  {} \\\\'.format(v_n.replace('_', '\\_'), value))
                elif isinstance(value, float) or isinstance(value, int):
                    print(' {}  &  {:.2f} \\\\'.format(v_n.replace('_', '\\_'), value))
                elif isinstance(value, str):
                    print(' {}  &  {} \\\\'.format(v_n.replace('_', '\\_'), value))
                else:
                    raise ValueError("value:{} is niether float nor string".format(value))
            print('\\hline')

            print('\\end{tabular}')

        if "wind" in components:

            # wind_pars, wind_vars = self.mkn_parameters_wind()

            print('\\begin{tabular}[t]{ p{2.cm} c }')
            print('Wind & \\\\')
            print('\\hline')

            for v_n, value in zip(self.ejecta_params["wind"].keys(), self.ejecta_params["wind"].values()):
                if value == None:
                    print(' {}  &  {} \\\\'.format(v_n.replace('_', '\\_'), value))
                elif isinstance(value, float) or isinstance(value, int):
                    print(' {}  &  {:.2f} \\\\'.format(v_n.replace('_', '\\_'), value))
                elif isinstance(value, str):
                    print(' {}  &  {} \\\\'.format(v_n.replace('_', '\\_'), value))
                else:
                    raise ValueError("value:{} is niether float nor string".format(value))
            print('\\hline')

            print('\\hline')

            for v_n, value in zip(self.ejecta_vars["wind"].keys(), self.ejecta_vars["wind"].values()):
                if value == None:
                    print(' {}  &  {} \\\\'.format(v_n.replace('_', '\\_'), value))
                elif isinstance(value, float) or isinstance(value, int):
                    print(' {}  &  {:.2f} \\\\'.format(v_n.replace('_', '\\_'), value))
                elif isinstance(value, str):
                    print(' {}  &  {} \\\\'.format(v_n.replace('_', '\\_'), value))
                else:
                    raise ValueError("value:{} is niether float nor string".format(value))
            print('\\hline')

            print('\\end{tabular}')

        if "secular" in components:

            # secular_pars, secular_vars = self.mkn_parameters_secular()

            print('\\begin{tabular}[t]{ p{2.cm} c }')
            print('Secualr & \\\\')
            print('\\hline')

            for v_n, value in zip(self.ejecta_params["secular"].keys(), self.ejecta_params["secular"].values()):
                if value == None:
                    print(' {}  &  {} \\\\'.format(v_n.replace('_', '\\_'), value))
                elif isinstance(value, float) or isinstance(value, int):
                    print(' {}  &  {:.2f} \\\\'.format(v_n.replace('_', '\\_'), value))
                elif isinstance(value, str):
                    print(' {}  &  {} \\\\'.format(v_n.replace('_', '\\_'), value))
                else:
                    raise ValueError("value:{} is niether float nor string".format(value))
            print('\\hline')

            print('\\hline')

            for v_n, value in zip(self.ejecta_vars["secular"].keys(), self.ejecta_vars["secular"].values()):
                if value == None:
                    print(' {}  &  {} \\\\'.format(v_n.replace('_', '\\_'), value))
                elif isinstance(value, float) or isinstance(value, int):
                    print(' {}  &  {:.2f} \\\\'.format(v_n.replace('_', '\\_'), value))
                elif isinstance(value, str):
                    print(' {}  &  {} \\\\'.format(v_n.replace('_', '\\_'), value))
                else:
                    raise ValueError("value:{} is niether float nor string".format(value))
            print('\\hline')

            print('\\end{tabular}')

        print('\\caption{Ejecta parameters}')
        print(r'\label{tbl:mkn_components}')
        print('\\end{table}')


    def load_ej_profile_for_mkn(self, fpath):
        th, mass, vel, ye = np.loadtxt(fpath,
                                       unpack=True, usecols=(0, 1, 2, 3))
        return th, mass, vel, ye
    # tech func to check how the smoothing actually done
    def smooth_profile(self, mass):

        lmass = np.log10(mass)

        mass_smooth = []
        for i in range(len(mass)):
            if (i == 0):
                mass_smooth.append(10. ** ((lmass[0])))
            elif (i == 1):
                mass_smooth.append(10. ** ((lmass[i - 1] + lmass[i] + lmass[i + 1]) / 3.))
            elif (i == 2):
                mass_smooth.append(10. ** ((lmass[i - 2] + lmass[i - 1] + lmass[i] + lmass[i + 1] + lmass[i + 2]) / 5.))
            elif (i == len(mass) - 3):
                mass_smooth.append(10. ** ((lmass[i - 2] + lmass[i - 1] + lmass[i] + lmass[i + 1] + lmass[i + 2]) / 5.))
            elif (i == len(mass) - 2):
                mass_smooth.append(10. ** ((lmass[i - 1] + lmass[i] + lmass[i + 1]) / 3.))
            elif (i == len(mass) - 1):
                mass_smooth.append(10. ** ((lmass[i])))
            else:
                mass_smooth.append(10. ** ((lmass[i - 3] + lmass[i - 2] + lmass[i - 1] + lmass[i] + lmass[i + 1] +
                                            lmass[i + 2] + lmass[i + 3]) / 7.))
        mass_smooth = np.asarray(mass_smooth)
        # tmp1 = np.sum(mass)
        # tmp2 = np.sum(mass_smooth)
        # mass_smooth = tmp1 / tmp2 * mass_smooth

        return mass_smooth

class LOAD_LIGHTCURVE():

    def __init__(self, sim, indir=None):
        #
        self.sim = sim
        self.default_fname = "mkn_model.h5"
        #
        if indir != None:
            self.indir = indir
            fpaths = glob(indir + "mkn_model*.h5")
        else:
            self.models_dir = "mkn/"
            self.indir = Paths.ppr_sims + sim + "/" + self.models_dir
            fpaths = glob(self.indir + "mkn_model*.h5")
        #
        if len(fpaths) == 0: raise IOError("No mkn files found {}".format(self.indir + "mkn_model*.h5"))
        #
        self.filter_fpath = Paths.mkn + Files.filt_at2017gfo
        #
        #
        flist = []
        for file_ in fpaths:
            flist.append(file_.split('/')[-1])
        self.list_model_fnames = flist
        #
        #
        self.list_obs_filt_fnames = ["AT2017gfo.h5"]
        self.list_fnames = self.list_model_fnames + self.list_obs_filt_fnames
        #
        self.list_attrs = ["spiral", "dynamics", "wind", "secular"]
        self.attrs_matrix = [[{}
                              for z in range(len(self.list_attrs))]
                              for y in range(len(self.list_fnames))]
        #
        self.data_matrix = [{}
                             for y in range(len(self.list_fnames))]
        #
        self.filters = {}

    def check_fname(self, fname=''):
        if not fname in self.list_fnames:
            raise NameError("fname: {} not in list_fnames:\n{}"
                            .format(fname, self.list_fnames))

    def check_attr(self, attr):
        if not attr in self.list_attrs:
            raise NameError("attr:{} not in list of attrs:{}"
                            .format(attr, self.list_attrs))

    def i_attr(self, attr):
        return int(self.list_attrs.index(attr))

    def get_attr(self, attr, fname=''):

        self.check_fname(fname)
        self.check_attr(attr)
        self.is_mkn_file_loaded(fname)

        return self.attrs_matrix[self.i_fname(fname)][self.i_attr(attr)]

    def i_fname(self, fname=''):
        return int(self.list_fnames.index(fname))

    def load_mkn_model(self, fname=''):

        if fname == '': fname = self.default_fname
        model_fpath = self.indir + fname

        dict_model = {}

        model = h5py.File(model_fpath, "r")
        filters_model = []
        for it in model:
            if it in self.list_attrs:
                dic = {}
                for v_n in model[it].attrs:
                    dic[v_n] = model[it].attrs[v_n]
                self.attrs_matrix[self.i_fname(fname)][self.i_attr(it)] = dic
            else:
                filters_model.append(it)
                dict_model[str(it)] = np.array(model[it])

        # print('\t Following filters are available in mkn_model.h5: \n\t  {}'.format(filters_model))

        self.data_matrix[self.i_fname(fname)] = dict_model

    def load_obs_filters(self, fname=''):

        dict_obs_filters = {}

        obs_filters = h5py.File(self.filter_fpath, "r")

        filters_model = []
        for it in obs_filters:
            filters_model.append(it)
            arr = np.array(obs_filters[it])
            # print(arr.shape)
            dict_obs_filters[str(it)] = np.array(obs_filters[it])

        # print('\t Following filters are available in AT2017gfo.h5: \n\t  {}'.format(filters_model))

        self.filters = dict_obs_filters

    def is_filters_loaded(self, fname):

        if not bool(self.filters):
            self.load_obs_filters(fname)

    def is_mkn_file_loaded(self, fname=''):

        if not bool(self.data_matrix[self.i_fname(fname)]):
            self.load_mkn_model(fname)

    def get_mkn_model(self, fname=''):

        self.check_fname(fname)
        self.is_mkn_file_loaded(fname)

        return self.data_matrix[self.i_fname(fname)]

    def get_filters(self, fname):
        self.is_filters_loaded(fname)
        return self.filters

class EXTRACT_LIGHTCURVE(LOAD_LIGHTCURVE):

    def __init__(self, sim, indir=None):
        LOAD_LIGHTCURVE.__init__(self, sim, indir)
        self.list_bands = __mkn__["bands"]
        self.do_extract_parameters = True

        self.model_params = [[{"spiral":{}, "dynamics":{}, "wind":{}, "secular":{}}
                                  for y in range(len(self.list_bands))]
                                 for z in range(len(self.list_fnames))]

        self.model_mag_matrix = [[ []
                                     for y in range(len(self.list_bands))]
                                     for z in range(len(self.list_fnames))]
        self.obs_mag_matrix = [[ []
                                     for y in range(len(self.list_bands))]
                                     for z in range(len(self.list_fnames))]

    def check_band(self, band):
        if not band in self.list_bands:
            raise NameError("band:{} not in tha band list:{}"
                            .format(band, self.list_bands))

    def i_band(self, band):
        return int(self.list_bands.index(band))

    # ---

    def extract_lightcurve(self, band, fname=''):

        dict_model = self.get_mkn_model(fname)
        # arr = np.zeros(len(dict_model['time']))
        time_ = np.array(dict_model['time'])
        # if not band in dict_model.keys():
        #     raise NameError("band:{} is not in the loaded model:\n{}"
        #                     .format(band, dict_model.keys()))

        res = []
        for filter in dict_model.keys():
            if filter.split('_')[0] == band:
                # arr = np.vstack((arr, dict_model[filter]))
                res.append(np.vstack((time_, np.array(dict_model[filter]))).T)
        # times = arr[:, 0]
        # arr = np.delete(arr, 0, 0)

        if len(res) == 0:
            raise NameError("band:{} is not found in the loaded model:\n{}"
                                .format(band, dict_model.keys()))

        self.model_mag_matrix[self.i_fname(fname)][self.i_band(band)] = res

        # ''' extract parameters '''
        # if self.do_extract_parameters:
        #     if "psdynamics" in

    def extract_obs_data(self, band, fname):

        dict_obs_filters = self.get_filters(fname)
        # dict_model = self.get_mkn_model(fname)

        sub_bands = []
        for filter in dict_obs_filters.keys():
            if filter.split('_')[0] == band:# and filter in dict_obs_filters.keys():
                sub_bands.append(dict_obs_filters[filter])


        if len(sub_bands) == 0:
            raise NameError("band:{} is not found in the loaded obs filters:\n{}"
                            .format(band, dict_obs_filters.keys()))

        self.obs_mag_matrix[self.i_fname(fname)][self.i_band(band)] = sub_bands

    # ---

    def is_extracted(self, band, fname=''):

        data = self.model_mag_matrix[self.i_fname(fname)][self.i_band(band)]

        if len(data)  == 0 and fname in self.list_model_fnames:
            self.extract_lightcurve(band, fname)

        if len(data) == 0 and fname in self.list_obs_filt_fnames:
            self.extract_obs_data(band, fname)

    def get_obs_data(self, band, fname="AT2017gfo.h5"):
        """
        :param band:
        :param fname:
        :return:     list of [:times:, :magnitudes:, :errors:] 3D array for every subband in band
        """
        self.check_fname(fname)
        self.check_band(band)

        self.is_extracted(band, fname)


        return self.obs_mag_matrix[self.i_fname(fname)][self.i_band(band)]

    def get_model(self, band, fname="mkn_model.h5"):
        self.check_band(band)
        self.check_fname(fname)

        self.is_extracted(band, fname)

        return self.model_mag_matrix[self.i_fname(fname)][self.i_band(band)]

    def get_model_min_max(self, band, fname="mkn_model.h5"):

        band_list = self.get_model(band, fname)

        maxs = []
        mins = []
        times = []
        mags = []
        for i_band, band in enumerate(band_list):
            times = band[:, 0]
            mags = np.append(mags, band[:, 1])

        mags = np.reshape(mags, (len(band_list), len(times)))

        for i in range(len(times)):
            maxs.append(mags[:,i].max())
            mins.append(mags[:,i].min())

        return times, maxs, mins
        #
        #
        #
        #
        # time_ = arr[0, :]
        # # arr = np.delete(arr, 0, 0)
        #
        # print(arr.shape)
        # print(arr)
        #
        # maxs = []
        # for i in range(len(arr[0, :])):
        #     maxs = np.append(maxs, arr[1:,i].max())
        #
        # mins = []
        # for i in range(len(arr[0, :])):
        #     mins = np.append(mins, arr[1:,i].min())
        #
        # if len(time_) != len(mins):
        #     raise ValueError("len(time_) {} != {} len(mins)"
        #                      .format(len(time_) ,len(mins)))
        #
        #
        #
        # return time_, mins, maxs

    def get_model_median(self, band, fname="mkn_model.h5"):

        m_times, m_maxs, m_mins = self.get_model_min_max(band, fname)

        m_times = np.array(m_times)
        m_maxs = np.array(m_maxs)
        m_mins = np.array(m_mins)

        return m_times, m_mins + ((m_maxs - m_mins) / 2)

    def get_mismatch(self, band, fname="mkn_model.h5"):

        from scipy import interpolate

        m_times, m_maxs, m_mins = self.get_model_min_max(band, fname)
        obs_data = self.get_obs_data(band)


        all_obs_times = []
        all_obs_maxs = []
        all_obs_mins = []
        for sumbband in obs_data:

            obs_time = sumbband[:, 0]
            obs_maxs = sumbband[:, 1] + sumbband[:, 2] # data + error bar
            obs_mins = sumbband[:, 1] - sumbband[:, 2]  # data - error bar

            all_obs_times = np.append(all_obs_times, obs_time)
            all_obs_maxs = np.append(all_obs_maxs, obs_maxs)
            all_obs_mins = np.append(all_obs_mins, obs_mins)

        all_obs_times, all_obs_maxs, all_obs_mins = \
            UTILS.x_y_z_sort(all_obs_times, all_obs_maxs, all_obs_mins)

        # interpolate for observationa times

        int_m_times = all_obs_times
        if all_obs_times.max() > m_times.max():
            int_m_times = all_obs_times[all_obs_times < m_times.max()]
        int_m_maxs = interpolate.interp1d(m_times, m_maxs, kind='linear')(int_m_times)
        int_m_mins = interpolate.interp1d(m_times, m_mins, kind='linear')(int_m_times)

        min_mismatch = []
        max_mismatch = []

        for i in range(len(int_m_times)):
            m_max = int_m_maxs[i]
            m_min = int_m_mins[i]
            o_min = all_obs_mins[i]
            o_max = all_obs_maxs[i]

            if o_max > m_max and o_min < m_min:
                min_mismatch = np.append(min_mismatch, 0)
            elif o_min <= m_max and o_min >= m_min:
                min_mismatch = np.append(min_mismatch, 0)
            elif o_max <= m_max and o_max >= m_min:
                min_mismatch = np.append(min_mismatch, 0)
            elif (o_min > m_max):
                min_mismatch = np.append(min_mismatch, o_min - m_max)
            elif (o_max < m_min):
                min_mismatch = np.append(min_mismatch, o_max - m_min)
            else:
                raise ValueError("mismatched failed m_max:{} m_min:{} o_max:{} o_min:{}"
                                 .format(m_max, m_min, o_max, o_min))
            #
            #
            # min_mismatch = np.append(min_mismatch, min([o_min - m_min, o_min - m_max,
            #                                             m_max - m_min, o_max - m_max]))
            # max_mismatch = np.append(max_mismatch, max([o_min - m_min, o_min - m_max,
            #                                             m_max - m_min, o_max - m_max]))

        # print(min_mismatch)

        return int_m_times, min_mismatch, max_mismatch


        # print(obs_data)

    def get_model_peak(self, band, fname="mkn_model.h5"):
        t, mag = self.get_model_median(band, fname)
        idx = UTILS.find_nearest_index(mag, mag.min())
        return t[idx], mag[idx]

    def get_obs_peak(self, band, fname = "AT2017gfo.h5"):

        from scipy import interpolate

        obs_data = self.get_obs_data(band, fname)
        obs_times = []
        obs_mags = []

        for sumbband in obs_data:
            obs_times = np.append(obs_times, sumbband[:, 0])
            obs_mags = np.append(obs_mags, sumbband[:, 1])

        obs_times, obs_mags = UTILS.x_y_z_sort(obs_times, obs_mags)

        int_obs_times = np.mgrid[obs_times[0]:obs_times[-2]:100j]

        assert len(int_obs_times) == 100

        assert obs_times.min() <= int_obs_times.min()
        assert obs_times.max() >= int_obs_times.max()

        int_obs_mags = interpolate.interp1d(obs_times, obs_mags, kind='linear')(int_obs_times)
        print(int_obs_mags)
        idx = UTILS.find_nearest_index(int_obs_mags, int_obs_mags.min())
        return int_obs_times[idx], int_obs_mags[idx]


        # obs_data = self.get_obs_data(band)

        # all_obs_times = []
        # all_obs_maxs = []
        # all_obs_mins = []
        # for sumbband in obs_data:
        #     obs_time = sumbband[:, 0]
        #     obs_maxs = sumbband[:, 1] + sumbband[:, 2]  # data + error bar
        #     obs_mins = sumbband[:, 1] - sumbband[:, 2]  # data - error bar
        #
        #     all_obs_times = np.append(all_obs_times, obs_time)
        #     all_obs_maxs = np.append(all_obs_maxs, obs_maxs)
        #     all_obs_mins = np.append(all_obs_mins, obs_mins)
        #
        # all_obs_times, all_obs_maxs, all_obs_mins = \
        #     x_y_z_sort(all_obs_times, all_obs_maxs, all_obs_mins)
        #
        #
        # #
        # # print(m_times)
        # # print(all_obs_times)
        # #
        # # mask1 = (m_times < all_obs_times.max())
        # # mask2 = (m_times > all_obs_times.min())
        # # print(mask1)
        # # print(mask2)
        # # int_obs_times = m_times[mask1 & mask2]
        # int_obs_times = np.mgrid[all_obs_times.min():all_obs_times.max():100j]
        # print(np.log10(all_obs_times))
        # int_obs_maxs = interpolate.interp1d(all_obs_times, all_obs_maxs, kind='linear')(int_obs_times)
        # int_obs_mins = interpolate.interp1d(all_obs_times, all_obs_mins, kind='linear')(int_obs_times)
        #
        # idx = find_nearest_index(int_obs_maxs, int_obs_maxs.min())
        #
        # return int_obs_times[idx], int_obs_maxs[idx], int_obs_mins[idx]

        #
        #
        #
        #
        # # interpolate for observationa times
        #
        # int_m_times = all_obs_times
        # if all_obs_times.max() > m_times.max():
        #     int_m_times = all_obs_times[all_obs_times < m_times.max()]
        # int_m_maxs = interpolate.interp1d(all_obs_times, all_obs_maxs, kind='cubic')(int_m_times)
        # int_m_mins = interpolate.interp1d(all_obs_times, all_obs_mins, kind='cubic')(int_m_times)
        #
        # min_mismatch = []
        # max_mismatch = []
        #
        # for i in range(len(int_m_times)):
        #     m_max = int_m_maxs[i]
        #     m_min = int_m_mins[i]
        #     o_min = all_obs_mins[i]
        #     o_max = all_obs_maxs[i]
        #
        #     if o_max > m_max and o_min < m_min:
        #         min_mismatch = np.append(min_mismatch, 0)
        #     elif o_min <= m_max and o_min >= m_min:
        #         min_mismatch = np.append(min_mismatch, 0)
        #     elif o_max <= m_max and o_max >= m_min:
        #         min_mismatch = np.append(min_mismatch, 0)
        #     elif (o_min > m_max):
        #         min_mismatch = np.append(min_mismatch, o_min - m_max)
        #     elif (o_max < m_min):
        #         min_mismatch = np.append(min_mismatch, o_max - m_min)
        #     else:
        #         raise ValueError("mismatched failed m_max:{} m_min:{} o_max:{} o_min:{}"
        #                          .format(m_max, m_min, o_max, o_min))
        #     #
        #     #
        #     # min_mismatch = np.append(min_mismatch, min([o_min - m_min, o_min - m_max,
        #     #                                             m_max - m_min, o_max - m_max]))
        #     # max_mismatch = np.append(max_mismatch, max([o_min - m_min, o_min - m_max,
        #     #                                             m_max - m_min, o_max - m_max]))
        #
        # # print(min_mismatch)
        #
        # return int_m_times, min_mismatch, max_mismatch

    def get_obs_peak_duration(self, band, limit=1.,  fname = "AT2017gfo.h5"):

        from scipy import interpolate

        obs_data = self.get_obs_data(band,  fname)
        obs_times = []
        obs_mags = []

        for sumbband in obs_data:
            obs_times = np.append(obs_times, sumbband[:, 0])
            obs_mags = np.append(obs_mags, sumbband[:, 1])

        obs_times, obs_mags = UTILS.x_y_z_sort(obs_times, obs_mags)

        int_obs_times = np.mgrid[obs_times[0]:obs_times[-2]:100j]

        assert len(int_obs_times) == 100

        assert obs_times.min() <= int_obs_times.min()
        assert obs_times.max() >= int_obs_times.max()

        int_obs_mags = interpolate.interp1d(obs_times, obs_mags, kind='linear')(int_obs_times)
        print(int_obs_mags)
        idx = UTILS.find_nearest_index(int_obs_mags, int_obs_mags.min())

        peaktime = int_obs_times[idx]
        peakmag = int_obs_mags[idx]

        mask = (obs_times >= peaktime) & (obs_mags < peakmag + limit)
        assert len(mask) > 1
        post_peak_times = obs_times[mask]
        post_peak_mags = obs_mags[mask]

        assert len(post_peak_times) > 1


        return post_peak_times[-1] - peaktime, post_peak_mags[-1]

    def get_model_peak_duration(self, band, fname="mkn_model.h5", limit = 1.):

        t, mag = self.get_model_median(band, fname)
        idx = UTILS.find_nearest_index(mag, mag.min())
        tpeak = t[idx]
        magpeak = mag[idx]

        mask = (t >= tpeak) & (mag < magpeak + limit)
        assert len(mask) > 1
        post_peak_times = t[mask]
        post_peak_mags = mag[mask]

        assert len(post_peak_times) > 1

        return post_peak_times[-1] - tpeak, post_peak_mags[-1]

class COMBINE_LIGHTCURVES(EXTRACT_LIGHTCURVE):

    def __init__(self, sim, indir=None):

        EXTRACT_LIGHTCURVE.__init__(self, sim, indir)

    def get_model_peaks(self, band, files_name_gen=r"mkn_model2_m*.h5"):

        files = glob(self.indir + files_name_gen)
        # print(files)

        tpeaks = []
        mpeaks = []
        attrs = []

        for file_ in files:
            tpeak, mpeak = self.get_model_peak(band, file_.split('/')[-1])
            attr = self.get_attr("dynamics", file_.split('/')[-1])["m_ej"]

            tpeaks = np.append(tpeaks, tpeak)
            mpeaks = np.append(mpeaks, mpeak)
            attrs = np.append(attrs, attr)

        attrs, tpeaks, mpeaks = UTILS.x_y_z_sort(attrs, tpeaks, mpeaks)

        return attrs, tpeaks, mpeaks

    def get_model_peak_durations(self, band, files_name_gen=r"mkn_model2_m*.h5"):

        files = glob(Paths.ppr_sims + self.sim + '/' + self.models_dir + files_name_gen)
        # print(files)

        tdurs = []
        attrs = []

        for file_ in files:
            tdur, _ = self.get_model_peak_duration(band, file_.split('/')[-1], limit=1.)
            attr = self.get_attr("spiral", file_.split('/')[-1])["m_ej"]

            tdurs = np.append(tdurs, tdur)
            attrs = np.append(attrs, attr)

        attrs, tdurs = UTILS.x_y_z_sort(attrs, tdurs)

        return attrs, tdurs

    def get_table(self, band='g', files_name_gen=r"mkn_model2_m*.h5"):

        files = glob(Paths.ppr_sims+self.sim+'/' + self.models_dir+files_name_gen)
        # print(files)

        t_arr = []
        mag_arr = []
        attr_arr = []


        def get_atr(file_):
            return self.get_attr("spiral", file_.split('/')[-1])["m_ej"]

        files = sorted(files, key=get_atr)


        for file_ in files:

            m_time, m_mag = self.get_model_median(band, file_.split('/')[-1])
            attr = self.get_attr("spiral", file_.split('/')[-1])["m_ej"]

            print('\t processing {} atr: {}'.format(file_.split('/')[-1], attr))

            t_arr = m_time
            mag_arr = np.append(mag_arr, m_mag)
            attr_arr.append(attr)

        mag_table = np.reshape(mag_arr, (len(attr_arr), len(t_arr)))

        t_grid, attr_grid = np.meshgrid(t_arr, attr_arr)

        return  t_grid, attr_grid, mag_table

        #
        # dfile = h5py.File(files[0], "r")
        #
        #
        #
        #
        #
        # ejecta_type = "psdynamics"
        # print(dfile[ejecta_type])
        #
        # # dfile[ejecta_type].attrs[""]
        #
        # v_ns = []
        # values = []
        # for v_n in dfile[ejecta_type].attrs:
        #     v_ns.append(v_n)
        #     values.append(dfile[ejecta_type].attrs[v_n])
        #
        # print(v_ns, values)
        #
        # pass

''' --- '''

def predic_value_from_fitfuncs(data, tasks):

    # res = []
    for task in tasks:
        if task["v_n"] == "Mej_tot-geo":
            # fit Mej
            if "res" in task.keys() and len(task["res"])>0:
                res = task["res"]
            else:
                func = task["func"]
                coefs = task["coeffs"]
                res = func(coefs, data, v_n = task["v_n"])
        elif task["v_n"] == "vel_inf_ave-geo":
            # fit vel
            if "res" in task.keys() and len(task["res"])>0:
                res = task["res"]
            else:
                func = task["func"]
                coefs = task["coeffs"]
                res = func(coefs, data, v_n = task["v_n"])
        elif task["v_n"] == "Ye_ave-geo":
            # fit Ye
            if "res" in task.keys() and len(task["res"])>0:
                res = task["res"]
            else:
                func = task["func"]
                coefs = task["coeffs"]
                res = func(coefs, data, v_n = task["v_n"])
        elif task["v_n"] == "Mdisk3D":
            # fit Mdisk
            if "res" in task.keys() and len(task["res"])>0:
                res = task["res"]
            else:
                func = task["func"]
                coefs = task["coeffs"]
                res = func(coefs, data, v_n = task["v_n"])
        else:
            raise NameError("v_n:{} ".format(task["v_n"]))

        task["res"] = res



''' --- '''

def plot_lighcurves(plotdic, linedics):

    if "subplots" in plotdic:
        fig, axes = plt.subplots(**plotdic["subplots"])
    else:
        fig = plt.figure(figsize=plotdic["figsize"])
        ax = fig.add_subplot(111)
        axes = [ax]

    for ax in axes:
        for line in linedics:
            # print("linex: {}".format(line["x"]))
            # print("liney: {}".format(line["y"]))
            x, y  = line["x"], line["y"]
            del line["x"]
            del line["y"]
            ax.plot(x, y, **line)

    for ax in axes:
        if "xmin" in plotdic.keys() and "xmax" in plotdic.keys():
            ax.set_xlim(plotdic["xmin"], plotdic["xmax"])
        if "ymin" in plotdic.keys() and "ymax" in plotdic.keys():
            ax.set_ylim(plotdic["ymin"], plotdic["ymax"])
        if "xlabel" in plotdic.keys():
            ax.set_xlabel(plotdic["xlabel"], fontsize = plotdic["fontsize"])
        if "ylabel" in plotdic.keys():
            ax.set_ylabel(plotdic["ylabel"], fontsize = plotdic["fontsize"])
        if "tick_params" in plotdic.keys():
            ax.tick_params(**plotdic["tick_params"])
        if "text" in plotdic.keys():
            plotdic["text"]["transform"] = ax.transAxes
            ax.text(**plotdic["text"])
        if "legend" in plotdic.keys():
            ax.legend(**plotdic["legend"])

    print("plotted: \n")
    print(plotdic["figname"])
    if plotdic["tight_layout"]: plt.tight_layout()
    #
    print("curdir:",os.curdir)
    assert os.path.isdir(__outplotdir__)
    plt.savefig(plotdic["figname"], dpi=plotdic["dpi"])
    if plotdic["savepdf"]: plt.savefig(plotdic["figname"].replace(".png", ".pdf"))
    plt.close()
    print(plotdic["figname"])

def fitting_function_predict():

    #     qs = [1., 1.37]
    #     lams = [300, 110, 800]
    Vals.q = np.array([1., 1., 1., 1.37, 1.37, 1.37])
    Vals.Lambda = np.array([190, 300, 800, 190, 300, 800])

    mej_dic = {"v_n":"Mej_tot-geo", "func":None, "coeffs":None, "res":np.full(len(Vals.Lambda),3.442e-3)}
    vej_dic = {"v_n":"vel_inf_ave-geo", "func": Fitting_Functions.poly_2_qLambda, "coeffs": np.array([0.182, 0.159, -1.509e-04, -1.046e-01, 9.233e-05, -1.581e-08]), "res":[]}

    predic_value_from_fitfuncs(Vals, [mej_dic, vej_dic])

    line_dics = [
        {"color": "red", "ls": ':', "lw": 0.8, "label": r"({}, {})".format(Vals.q[0], Vals.Lambda[0])},
        {"color": "red", "ls": '-', "lw": 0.8, "label": r"({}, {})".format(Vals.q[1], Vals.Lambda[1])},
        {"color": "red", "ls": '-.', "lw": 0.8, "label": r"({}, {})".format(Vals.q[2], Vals.Lambda[2])},

        {"color": "orange", "ls": ':', "lw": 0.8, "label": r"({}, {})".format(Vals.q[3], Vals.Lambda[3])},
        {"color": "orange", "ls": '-', "lw": 0.8, "label": r"({}, {})".format(Vals.q[4], Vals.Lambda[4])},
        {"color": "orange", "ls": '-.', "lw": 0.8, "label": r"({}, {})".format(Vals.q[5], Vals.Lambda[5])},
    ]

    for mej, vej, linedic in zip(mej_dic["res"], vej_dic["res"], line_dics):
        # compute lightcurve
        o_mkn = COMPUTE_LIGHTCURVE(None, "/data01/numrel/vsevolod.nedora/prj_gw170817/scripts/lightcurves/")
        o_mkn.set_glob_par_var_source(False, False)
        o_mkn.set_dyn_iso_aniso = "aniso"
        o_mkn.set_dyn_par_var("aniso")
        o_mkn.ejecta_vars['dynamics']["mej"] = mej
        o_mkn.ejecta_vars["dynamics"]["central_vel"] = vej
        o_mkn.compute_save_lightcurve(write_output=True)

        load_mkn = COMBINE_LIGHTCURVES(None, "/data01/numrel/vsevolod.nedora/prj_gw170817/scripts/lightcurves/")
        times, mags = load_mkn.get_model_median("Ks")

        linedic["x"] = times
        linedic["y"] = mags

    ''' -- plotting --- '''

    plot_dic = {
        # "subplots":{"figsize": (6.0, 6.0), "ncols":1,"nrows":3, "sharex":True,"sharey":False},
        "figsize": (4.2, 3.6),
        'xmin': 3e-1, 'xmax': 3e1, 'xlabel': r"time [days]",
        'ymin': 23, 'ymax': 18, 'ylabel': r"AB magnitude at 40 Mpc",
        'fontsize':14,
        'labelsize':14,
        "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False, "tick1On": True, "tick2On": True,
                        "labelsize": 14,
                        "direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        "legend": {"fancybox": False, "loc": 'upper right',
                   # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 12, "columnspacing": 0.4,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        # "text": {'x': 0.85, 'y': 0.90, 's': r"Poly2", 'fontsize': 14, 'color': 'black',
        #          'horizontalalignment': 'center'},
        "savepdf": True,
        "figname": __outplotdir__ + "mkn_dyn_fit_target.png",
        "dpi":128,
        "tight_layout":True
        }

    plot_lighcurves(plot_dic, line_dics)

def get_times_mags_from_pars(pars, band = "Ks"):

    # compute MKN
    o_mkn = COMPUTE_LIGHTCURVE(None, "/data01/numrel/vsevolod.nedora/prj_gw170817/scripts/lightcurves/")
    o_mkn.set_glob_par_var_source(False, False)
    o_mkn.set_dyn_iso_aniso = "aniso"
    o_mkn.set_dyn_par_var("aniso")
    o_mkn.ejecta_vars['dynamics']["mej"] = pars["mej"]
    o_mkn.ejecta_vars["dynamics"]["central_vel"] = pars["vej"]
    o_mkn.compute_save_lightcurve(write_output=True)

    load_mkn = COMBINE_LIGHTCURVES(None, "/data01/numrel/vsevolod.nedora/prj_gw170817/scripts/lightcurves/")
    times, mags = load_mkn.get_model_median(band)

    return times, mags

''' --- plotting methods --- '''

''' --- tasks --- '''

def task_plot_lightcurve_synthetic_model():

    # model
    sim = "BLh_M13641364_M0_LK"
    model = md.groups[md.groups.index == sim]

    # fitted vals
    mej_mean = 5.220e-03 # Msun
    mej_poly22 = Fitting_Functions.poly_2_qLambda([2.549, 2.394, -3.005e-02, -3.376e+00, 0.038, -1.149e-05], model)
    mej_diet = Fitting_Functions.mej_dietrich16([-1.234, 3.089, -31.801, 17.526, -3.146], model) / 1e3
    mej_krug = Fitting_Functions.mej_kruger20([-0.981, 12.880, -35.148, 2.030], model) / 1e3

    vej_mean = 0.189 # \pm 0.049
    vej_poly22 = Fitting_Functions.poly_2_qLambda([0.182, 0.159, -1.509e-04, -1.046e-01, 9.233e-05, -1.581e-08], model)
    vej_diet = Fitting_Functions.vej_dietrich16([-0.422, 0.834, -1.510], model)

    yeej_poly22 = Fitting_Functions.poly_2_qLambda([-4.555e-01, 0.793, 7.509e-04, -3.139e-01, -1.899e-04, -4.460e-07], model)
    yeej_our = Fitting_Functions.yeej_like_vej([0.177, 0.452, -4.611], Vals)

    mdisk = Fitting_Functions.poly_2_qLambda([-8.951e-01, 1.195, 4.292e-04, -3.991e-01, 4.778e-05, -2.266e-07], model)
    mdisk = Fitting_Functions.mdisk_radice18([0.070, 0.101, 305.009, 189.952], model)
    mdisk = Fitting_Functions.mdisk_kruger20([-0.013, 1.000, 1325.652], model)

    print("\t-------------------")
    print("\t model: {}".format(sim))
    print("\t Mej Mean " + "{:.1f}".format(mej_mean * 1e3))
    print("\t Mej Eq.1~\cite{Dietrich:2016fpt}"  + "{:.1f}".format(mej_diet * 1e3) + "[10^{3} M_{\odot}]")
    print("\t Mej Eq.6~\cite{Kruger:2020gig} " + "{:.1f}".format(mej_krug * 1e3) + "[10^{3} M_{\odot}]")
    print("\t Mej Eq.P22 " + "{:.1f}".format(mej_poly22 * 1e3) + "[10^{3} M_{\odot}]")
    print("\t-------------------")
    print("\t vej Mean " + "{:.3f}".format(vej_mean))
    print("\t Mej Eq.5~\cite{Dietrich:2016fpt}" + "{:.1f}".format(vej_diet * 1e3) + "[c]")
    print("\t Mej Eq.P22 " + "{:.1f}".format(vej_diet * 1e3) + "[c]")


    # compute MKN
    pars = {"mej":mej, "vej":vej, "mdisk":mdisk}
    times, mags = get_times_mags_from_pars(pars, "Ks")
    linedic_poly22 = {"x":times,"y":mags, "color": "red", "ls": ':', "lw": 0.8, "label": r"(Poly22)"}


    plot_dic = {
        # "subplots":{"figsize": (6.0, 6.0), "ncols":1,"nrows":3, "sharex":True,"sharey":False},
        "figsize": (4.2, 3.6),
        'xmin': 3e-1, 'xmax': 3e1, 'xlabel': r"time [days]",
        'ymin': 23, 'ymax': 18, 'ylabel': r"AB magnitude at 40 Mpc",
        'fontsize': 14,
        'labelsize': 14,
        "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False, "tick1On": True, "tick2On": True,
                        "labelsize": 14,
                        "direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        "legend": {"fancybox": False, "loc": 'upper right',
                   # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 2, "fontsize": 12, "columnspacing": 0.4,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        # "text": {'x': 0.85, 'y': 0.90, 's': r"Poly2", 'fontsize': 14, 'color': 'black',
        #          'horizontalalignment': 'center'},
        "savepdf": True,
        "figname": __outplotdir__ + "mkn_dyn_fit_target.png",
        "dpi": 128,
        "tight_layout": True
    }

    plot_lighcurves(plot_dic, line_dics)


if __name__ == "__main__":

    # o_mkn = COMPUTE_LIGHTCURVE(None, "/data01/numrel/vsevolod.nedora/prj_gw170817/scripts/lightcurves/")
    # o_mkn.set_glob_par_var_source(False, False)
    # o_mkn.set_dyn_iso_aniso = "aniso"
    # o_mkn.set_dyn_par_var("aniso")
    # o_mkn.ejecta_vars['dynamics']["mej"] = 1e-3
    # o_mkn.ejecta_vars["dynamics"]["central_vel"] = 0.3
    # o_mkn.compute_save_lightcurve(write_output=True)
    #
    # load_mkn = COMBINE_LIGHTCURVES(None, "/data01/numrel/vsevolod.nedora/prj_gw170817/scripts/lightcurves/")
    #print(load_mkn.get_model_peaks("g", "mkn_model.h5"))

    fitting_function_predict()