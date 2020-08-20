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

import uutils

import click

#from legacy.prj_visc_ej import ax

matplotlib.use('agg')
import matplotlib.pyplot as plt
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')


from matplotlib.colors import LogNorm, Normalize

#sys.path.append('/data01/numrel/vsevolod.nedora/bns_ppr_tools/')
#sys.path.append('/data01/numrel/vsevolod.nedora/bns_ppr_tools/')


# from preanalysis import LOAD_INIT_DATA
# from outflowed import EJECTA_PARS
# from preanalysis import LOAD_ITTIME
# from plotting_methods import PLOT_MANY_TASKS
# from profile import LOAD_PROFILE_XYXZ, LOAD_RES_CORR, LOAD_DENSITY_MODES, MAINMETHODS_STORE, MAINMETHODS_STORE_XYXZ
#from utils import Paths, Lists, Labels, Constants, Printcolor, UTILS, Files, PHYSICS

from make_fit2 import Fitting_Coefficients, Fitting_Functions, Fit_Data

from model_sets import models as md
from data import ADD_METHODS_ALL_PAR
from comparison import TWO_SIMS, THREE_SIMS
from settings import resolutions
from uutils import *

from model_sets import models as ourmd

#from sys import path
mkn_code_path = "/home/vsevolod/GIT/bitbucket/mkn_framework/source/"
sys.path.append(mkn_code_path)
try:
    from mkn_bayes import MKN
except ImportError:
    try:
        from mkn import MKN
    except ImportError:
        raise ImportError("Failed to import mkn from MKN (set path is: {} ".format(mkn_code_path))

# __curdir__ = "/data01/numrel/vsevolod.nedora/prj_gw170817/scripts/"

__curdir__ = '/home/vsevolod/GIT/GitHub/prj_gw170817/scripts/'
__outplotdir__ = "../figs/all3/mkn_fit/"
__lightcurves_store__ = "/home/vsevolod/GIT/GitHub/prj_gw170817/data/"
__obs_filter_fname__ = "AT2017gfo.h5"
__nr_data__ = "/home/vsevolod/GIT/GitHub/prj_gw170817/data/"

if not os.path.isdir(__outplotdir__):
    os.mkdir(__outplotdir__)

from model_sets import models as md

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
        self.path = __lightcurves_store__
        self.output_fname = 'mkn_model.h5'
        # if sim != None:
        #     self.o_data = ADD_METHODS_ALL_PAR(sim)
        # else:
        #     self.o_data = None
        #
        self.outdir = __lightcurves_store__
        self.outfpath = self.outdir + self.output_fname

        # if outdir == None:
        #     self.outdir = Paths.ppr_sims+self.sim+'/mkn/'
        #     self.outfpath = Paths.ppr_sims+self.sim+'/mkn/' + self.output_fname
        # else:
        #     self.outdir = outdir
        #     self.outfpath = self.outdir + self.output_fname
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
                # mdisk = self.o_data.get_par("Mdisk3D")
                mdisk = float(md.simulations.loc[self.sim]["Mdisk3D"])
                if np.isnan(mdisk):
                    raise ValueError("mass of the disk is not avilable (nan) for sim:{}".format(self.sim))
            else:
                print("\tUsing default disk mass")
                mdisk = 0.012

        if NR2_data:
            # mdisk = self.o_data.get_par("Mdisk3D")
            mdisk = float(md.simulations.loc[self.sim]["Mdisk3D"])
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

        # if not self.sim == None and self.set_use_dyn_NR:
        #     mej = self.o_data.get_outflow_par(det,mask,"Mej_tot")
        #     if np.isnan(mej):
        #         raise ValueError("Ejecta mass for det:{} mask:{} is not avialble (nan)".format(det,mask))
        # else:
        #     mej = 0.015

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
                                           'm_ej':              0.0,#0.015, # 0.00169045, # - LS220 | 0.00263355 - DD2
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
                            'xi_disk':          0.2, # default: 0.2
                            'm_ej':             None,#0.03,
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
        fpath = __nr_data__ + self.sim + '/' + "outflow_{:d}".format(det) + '/' + mask + '/'
        if not os.path.isdir(fpath):
            raise IOError("dir with outflow + mask is not found: {}".format(fpath))
        fname = "ejecta_profile.dat"
        fpath = fpath + fname
        if not os.path.isfile(fpath):
            raise IOError("file for mkn NR data is not found: {}".format(fpath))
        self.dyn_ejecta_profile_fpath = fpath

    def set_bern_ej_nr(self, det, mask):
        fpath = __nr_data__ + self.sim + '/' + "outflow_{:d}".format(det) + '/' + mask + '/'
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
        os.chdir(mkn_code_path)
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
            self.indir = __nr_data__ + sim + "/" + self.models_dir
            fpaths = glob(self.indir + "mkn_model*.h5")
        #
        if len(fpaths) == 0:
            print("Error: No mkn files found {}".format(self.indir + "mkn_model*.h5"))
            # raise IOError("No mkn files found {}".format(self.indir + "mkn_model*.h5"))
        #
        self.filter_fpath = __lightcurves_store__ + __obs_filter_fname__
        #
        #
        flist = ["mkn_model.h5"]
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
            raise NameError("fname: {} not in list_fnames:\n{}".format(fname, self.list_fnames))

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
        if not os.path.isfile(model_fpath):
            raise IOError("File not found: {}".format(model_fpath))

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
            uutils.x_y_z_sort(all_obs_times, all_obs_maxs, all_obs_mins)

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
        idx = uutils.find_nearest_index(mag, mag.min())
        return t[idx], mag[idx]

    def get_obs_peak(self, band, fname = "AT2017gfo.h5"):

        from scipy import interpolate

        obs_data = self.get_obs_data(band, fname)
        obs_times_ = []
        obs_mags = []
        obs_errs = []

        for sumbband in obs_data:
            obs_times_ = np.append(obs_times_, sumbband[:, 0])
            obs_mags = np.append(obs_mags, sumbband[:, 1])
            obs_errs = np.append(obs_errs, sumbband[:, 2])

        # print(obs_times_)
        # print(obs_mags)
        # print(obs_errs)

        assert len(obs_times_) == len(obs_mags)
        assert len(obs_times_) == len(obs_errs)

        obs_times, obs_mags = uutils.x_y_z_sort(obs_times_, obs_mags)
        obs_times, obs_errs = uutils.x_y_z_sort(obs_times_, obs_errs)

        int_obs_times = np.mgrid[obs_times[0]:obs_times[-2]:100j]

        assert len(int_obs_times) == 100

        assert obs_times.min() <= int_obs_times.min()
        assert obs_times.max() >= int_obs_times.max()

        int_obs_mags = interpolate.interp1d(obs_times, obs_mags, kind='linear')(int_obs_times)
        int_obs_errs = interpolate.interp1d(obs_times, obs_errs, kind='linear')(int_obs_times)
        print(int_obs_mags, int_obs_errs)
        idx = uutils.find_nearest_index(int_obs_mags, int_obs_mags.min())
        return int_obs_times[idx], int_obs_mags[idx], int_obs_errs[idx]


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

        obs_times, obs_mags = uutils.x_y_z_sort(obs_times, obs_mags)

        int_obs_times = np.mgrid[obs_times[0]:obs_times[-2]:100j]

        assert len(int_obs_times) == 100

        assert obs_times.min() <= int_obs_times.min()
        assert obs_times.max() >= int_obs_times.max()

        int_obs_mags = interpolate.interp1d(obs_times, obs_mags, kind='linear')(int_obs_times)
        print(int_obs_mags)
        idx = uutils.find_nearest_index(int_obs_mags, int_obs_mags.min())

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
        idx = uutils.find_nearest_index(mag, mag.min())
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

        attrs, tpeaks, mpeaks = uutils.x_y_z_sort(attrs, tpeaks, mpeaks)

        return attrs, tpeaks, mpeaks

    def get_model_peak_durations(self, band, files_name_gen=r"mkn_model2_m*.h5"):

        files = glob(__nr_data__ + self.sim + '/' + self.models_dir + files_name_gen)
        # print(files)

        tdurs = []
        attrs = []

        for file_ in files:
            tdur, _ = self.get_model_peak_duration(band, file_.split('/')[-1], limit=1.)
            attr = self.get_attr("spiral", file_.split('/')[-1])["m_ej"]

            tdurs = np.append(tdurs, tdur)
            attrs = np.append(attrs, attr)

        attrs, tdurs = uutils.x_y_z_sort(attrs, tdurs)

        return attrs, tdurs

    def get_table(self, band='g', files_name_gen=r"mkn_model2_m*.h5"):

        files = glob(__nr_data__ + self.sim + '/' + self.models_dir + files_name_gen)
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
        if "add_lines" in plotdic.keys() and not "add_legend" in plotdic.keys(): # and not "add_legend" in plotdic.keys():
            for entry in plotdic["add_lines"]:
                ax.plot(**entry)

        elif "add_lines" in plotdic.keys() and "add_legend" in plotdic.keys():

            for entry in plotdic["add_lines"]:
                ax.plot([-1e5,-2e5],[-1e5,-2e5], **entry)

            n = len(plotdic["add_lines"])

            han, lab = ax.get_legend_handles_labels()
            print(len(han), len(lab))
            ax.add_artist(ax.legend(han[:-1 * n], lab[:-1 * n], **plotdic["legend"]))
            #
            ax.add_artist( ax.legend(han[len(han) - n:], lab[len(lab) - n:], **plotdic["add_legend"]))
        else:
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


def custom_plot_mkns_several_bands(plotdic, subplot_dics):
    bands = [subplot_dic["band"] for subplot_dic in subplot_dics]
    fig, axes = plt.subplots(figsize=plotdic["figsize"], ncols=len(bands), nrows=1, sharey=True)
    # ax = fig.add_subplot(111)
    i = 0
    for ax, band, subplot_dic in zip(axes, bands, subplot_dics):

        # lineset[band][simulation][label] = dict{linedic}
        band_liens = subplot_dic["lines"]
        for sel_model in band_liens.keys():
            for sel_parset in band_liens[sel_model].keys():
                dic = band_liens[sel_model][sel_parset]
                x, y = dic["x"], dic["y"]
                del dic["x"]
                del dic["y"]
                ax.plot(x, y, **dic)
        #
        ax.set_yscale(plotdic["yscale"])
        ax.set_xscale(plotdic["xscale"])
        #
        ax.set_xlabel(plotdic["xlabel"], fontsize=plotdic["fontsize"])  # , fontsize=11)
        if i == 0: ax.set_ylabel(plotdic["ylabel"], fontsize=plotdic["fontsize"])  # , fontsize=11)
        #
        if 'xmin' in subplot_dic.keys() and 'xmax' in subplot_dic.keys():
            ax.set_xlim(subplot_dic["xmin"], subplot_dic["xmax"])
        else:
            ax.set_xlim(plotdic["xmin"], plotdic["xmax"])

        if 'ymin' in subplot_dic.keys() and 'ymax' in subplot_dic.keys():
            ax.set_xlim(subplot_dic["ymin"], subplot_dic["ymax"])
        else:
            ax.set_ylim(plotdic["ymin"], plotdic["ymax"])
        #
        ax.tick_params(axis='both', which='both', labelleft=True,
                       labelright=False, tick1On=True, tick2On=True,
                       labelsize=plotdic["fontsize"],
                       direction='in',
                       bottom=True, top=True, left=True, right=True)
        ax.minorticks_on()
        # #

        if "add_lines" in subplot_dic.keys():
            for entry in subplot_dic["add_lines"]:
                ax.plot([-100, -100], [-200, -200], **entry)
        #
        if "title" in subplot_dic:
            ax.set_title(subplot_dic["title"])
        #
        if "title" in plotdic.keys():
            if plotdic["title"] == "#band":
                ax.set_title(band)
            else:
                ax.set_title(plotdic["title"])
        #
        if i > 0:
            ax.tick_params(labelleft=False)
            # ax.get_yaxis().set_ticks([])
            # ax.get_yaxis().set_visible(False)

        if "legend" in subplot_dic.keys():
            ax.legend(**subplot_dic["legend"])
        if len(plotdic["legend"].keys()) > 0: ax.legend(**plotdic["legend"])
        i = i + 1
    #
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.)
    #
    print("plotted: \n")
    print(plotdic["figname"])
    plt.savefig(plotdic["figname"], dpi=128)
    if "savepdf" in plotdic.keys() and plotdic["savepdf"]:
        plt.savefig(plotdic["figname"].replace(".png", ".pdf"))
    plt.close()
    # exit(1)
''' --- tasks --- '''

class Fit:

    def __init__(self, model):
        self.m = model
        # Best : mej_krug
        self.mej_mean = 6.223e-03  # Msun # sigma 9.762e-03 # 150 models # chi2 1489.864 dof
        self.mej_poly2 = float(Fitting_Functions.poly_2_Lambda([4.93e+00 , -3.64e-03 , 5.79e-06], model)) / 1e3 # 625.8 & 0.125
        self.mej_poly22 = float(Fitting_Functions.poly_2_qLambda([4.83e+01 , -6.89e+01 , -3.99e-02 , 2.47e+01 , 3.83e-02 , -1.03e-06], model)) / 1e3 # 81.7 & 0.755
        self.mej_diet = float(Fitting_Functions.mej_dietrich16([0.55630 , 1.030 , -5.820 , -5.560 , -4.465], model)) / 1e3 # 153.7 & 0.537
        self.mej_krug = float(Fitting_Functions.mej_kruger20([-11.07437 , 140.501 , -436.828 , 1.439], model)) / 1e3 # 53.4 & 0.582

        # Best vej_poly22
        self.vej_mean = 0.194  # \pm 0.052 # 128.000 models with chi2dof 6.6
        self.vej_poly2 = float(Fitting_Functions.poly_2_Lambda([2.28e-01 , -7.41e-05 , 3.04e-08], model)) # 6.5 & 0.029
        self.vej_poly22 = float(Fitting_Functions.poly_2_qLambda([4.69e-01 , -3.08e-01 , -9.19e-05 , 8.30e-02 , 1.71e-05 , 3.13e-08], model)) # 5.6 & 0.167
        self.vej_diet = float(Fitting_Functions.vej_dietrich16([-0.22154 , 0.581 , -0.934], model)) # 6.1 & 0.095

        # best yeej_poly22
        self.yeej_mean = 0.171  # \pm 0.054 # 88.000 models with chidof 28.7
        self.yeej_poly2 = float(Fitting_Functions.poly_2_Lambda([1.47e+01 , 3.37e-02 , -1.79e-05], model)) # 14.3 & 0.115
        self.yeej_poly22 = float(Fitting_Functions.poly_2_qLambda([-3.49e-01 , 7.65e-01 , 4.46e-04 , -2.94e-01 , -2.49e-04 , -1.28e-07], model)) # 17.8 & 0.394
        self.yeej_our = float(Fitting_Functions.yeej_like_vej([0.12819 , 0.349 , -4.161], model)) # 23.6 & 0.197

        # theta
        self.thetaej_mean = 27.555  # \pm 7.941 # 76.000 models with chidof 15.765
        self.thetaej_poly2 = float(Fitting_Functions.poly_2_Lambda([1.27e-01, 1.34e-04, -8.64e-08], model))  # 28.4 & 0.034
        self.thetaej_poly22 = float(Fitting_Functions.poly_2_qLambda([-1.04e+02 , 1.77e+02 , 1.10e-01 , -6.04e+01 , -6.51e-02 , -2.47e-05], model)) # 8.4 & 0.483

        # best mdisk_poly22
        self.mdisk_mean = 0.120  # \pm 0.086 with 114.000 model with chidof 2732.7
        self.mdisk_poly2 = float(Fitting_Functions.poly_2_Lambda([-6.73e-02 , 4.78e-04 , -2.11e-07], model)) #  643.4 & 0.406
        self.mdisk_poly22 = float(Fitting_Functions.poly_2_qLambda([-8.50e-01 , 1.12e+00 , 4.21e-04 , -3.71e-01 , 3.54e-05 , -2.13e-07], model)) # 202.7 & 0.655
        self.mdisk_radi = float(Fitting_Functions.mdisk_radice18([-59.49031 , 59.672 , -988.615 , 379.887], model)) # 630.7 & 0.437
        self.mdisk_krug = float(Fitting_Functions.mdisk_kruger20([-4.28527 , 0.844 , 1.354], model)) # 481.6 & 0.474

    def value(self, v_n, fitmodelname="mean"):

        #  = FitVals(model)
        if v_n == "Mej_tot-geo":
            if fitmodelname == None:        return float(self.m[v_n])
            elif fitmodelname == "mean":    return self.mej_mean
            elif fitmodelname == "poly2":   return self.mej_poly2
            elif fitmodelname == "poly22":  return self.mej_poly22
            elif fitmodelname == "diet16":  return self.mej_diet
            elif fitmodelname == "krug19":  return self.mej_krug
            else: raise NameError("v_n:{} fitmodelname:{} not recognized".format(v_n, fitmodelname))
        elif v_n == "vel_inf_ave-geo":
            if fitmodelname == None:        return float(self.m[v_n])
            elif fitmodelname == "mean":    return self.vej_mean
            elif fitmodelname == "poly22":  return self.vej_poly22
            elif fitmodelname == "diet16":  return self.vej_diet
            else: raise NameError("v_n:{} fitmodelname:{} not recognized".format(v_n, fitmodelname))
        elif v_n == "Ye_ave-geo":
            if fitmodelname == None:        return float(self.m[v_n])
            elif fitmodelname == "mean":    return self.yeej_mean
            elif fitmodelname == "poly22":  return self.yeej_poly22
            elif fitmodelname == "our":     return self.yeej_our
            else: raise NameError("v_n:{} fitmodelname:{} not recognized".format(v_n, fitmodelname))
        elif v_n == "theta_rms-geo":
            if fitmodelname == None:        return float(self.m[v_n])
            elif fitmodelname == "mean":    return self.thetaej_mean
            elif fitmodelname == "poly2":   return self.thetaej_poly2
            elif fitmodelname == "poly22":  return self.thetaej_poly22
        if v_n == "Mdisk3D":
            if fitmodelname == None:        return float(self.m[v_n])
            elif fitmodelname == "mean":    return self.mdisk_mean
            elif fitmodelname == "poly22":  return self.mdisk_poly22
            elif fitmodelname == "radi18":  return self.mdisk_radi
            elif fitmodelname == "krug19":  return self.mdisk_krug
            else: raise NameError("v_n:{} fitmodelname:{} not recognized".format(v_n, fitmodelname))
        else:
            raise NameError("v_n:{} fitmodelname:{} not recognized".format(v_n, fitmodelname))


def print_tex_table_fit_for_models_Mej():

    task = [
        {"sim": "DD2_M13641364_M0_LK_R04", "label": r"DD2 q=1.00 (SR)", "type": "long"},
        {"sim": "DD2_M15091235_M0_LK", "label": r"DD2 q=1.22 (SR)", "type": "long"},
        {"sim": "BLh_M13641364_M0_LK", "label": r"BLh q=1.00 (SR)", "type": "long"},
        {"sim": "BLh_M11461635_M0_LK", "label": r"BLh q=1.43 (SR)", "type": "long"},
        {"sim": "SLy4_M13641364_M0", "label": r"SLy4* q=1.00 (SR)", "type": "long"},
        {"sim": "SFHo_M13641364_M0", "label": r"SFHo* q=1.00 (SR)", "type": "long"},

    ]

    # task = [
    #     # 2 prompt collapse
    #     # {"sim": "BLh_M10201856_M0_LK_LR_AHfix", "label": r"BLh** q=1.82 (LR)", "color": "black", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long"},
    #     # {"sim": "BLh_M10201856_M0_LK_LR", "label": r"BLh* q=1.82 (LR)", "color": "black", "ls": "-", "lw": 1.0, "alpha": 1., "type": "long"},
    #     {"sim": "BLh_M10201856_M0", "label": r"BLh* q=1.82 (SR)", "type": "long"},
    #     {"sim": "BLh_M10651772_M0_LK", "label": r"BLh q=1.66 (LR)", "type": "long"},
    #     # {"sim": "BLh_M11841581_M0_LK_SR", "label": r"BLh q=1.34 (SR)", "color": "black", "ls": "-.", "lw": 1.0, "alpha": 1., "type": "long"},
    #     # 3 short lived
    #     {"sim": "LS220_M11461635_M0_LK", "label": r"LS220 q=1.43 (SR)", "type": "short"},
    #     {"sim": "SLy4_M13641364_M0", "label": r"SLy4* q=1.00 (SR)", "type": "short"},
    #     {"sim": "SFHo_M13641364_M0", "label": r"SFHo* q=1.00 (SR)", "type": "short"},
    #     # long
    #     {"sim": "BLh_M13641364_M0_LK", "label": r"BLh q=1.00 (SR)",  "type": "long"},
    #     # {"sim": "DD2_M13641364_M0_LK_SR_R04", "label": r"DD2* q=1.00 (SR)", "color": "blue", "ls": "--", "lw": 1.0, "alpha": 1., "type": "long", "t1": 40},
    # ]

    v_ns = [
        #"label", "q", "M1", "M2", "C1", "C2", "Lambda", "Mej_tot-geo", "Mej_tot-geo-poly22",
        'label', 'Mej_tot-geo', 'Mej_tot-geo-mean', 'Mej_tot-geo-poly22', "Mej_tot-geo-diet16", 'Mej_tot-geo-krug19'
    ]
    vlabels = [
        "Model", r"$\md$", r"$\bar{\amd}$", r"$P_2(q,\tilde\Lambda)$", r"Eq.~\eqref{eq:fit_Mej}", r"Eq.~\eqref{eq:fit_Mej_Kruger}"
    ]
    fmts = [
        "", ".1f", ".1f", ".1f", ".1f", ".1f"
    ]
    units = [
        "", "[$10^{3}M_{\odot}$]", "[$10^{3}M_{\odot}$]", "[$10^{3}M_{\odot}$]", "[$10^{3}M_{\odot}$]", "[$10^{3}M_{\odot}$]"
    ]
    mults = [
        #1., 1., 1., 1., 1., 1.,
        1e3, 1e3, 1e3, 1e3, 1e3, 1e3,
    ]


    cells = "c" * 11
    cells = "c" * len(v_ns) # coefs + chi2 + ch2dof + r2
    #


    models = md.groups

    lines = []
    for modeldic in task:
        print("\t processing: {}".format(modeldic["sim"]))
        line = ''
        model = models[models.group == modeldic["sim"]]
        print(model)
        fit = Fit(model)
        for v_n, fmt, mult in zip(v_ns, fmts, mults):
            if v_n == "label":
                value = modeldic["label"]
            elif v_n in model.keys():
                value = "%{}".format(fmt) % float(model[v_n] * mult)
            elif v_n.__contains__("-mean"):
                value = "%{}".format(fmt) % float(fit.value(v_n.replace("-mean", ""), "mean") * mult)
            elif v_n.__contains__("-poly22"):
                value = "%{}".format(fmt) % float(fit.value(v_n.replace("-poly22", ""), "poly22") * mult)
            elif v_n.__contains__("-diet16"):
                value = "%{}".format(fmt) % float(fit.value(v_n.replace("-diet16", ""), "diet16") * mult)
            elif v_n.__contains__("-our"):
                value = "%{}".format(fmt) % float(fit.value(v_n.replace("-our", ""), "our") * mult)
            elif v_n.__contains__("-krug19"):
                value = "%{}".format(fmt) % float(fit.value(v_n.replace("-krug19", ""), "krug19") * mult)
            elif v_n.__contains__("-radi18"):
                value = "%{}".format(fmt) % float(fit.value(v_n.replace("-radi18", ""), "radi18") * mult)
            else:
                raise NameError("name: {} is not recognzied".format(v_n))

            if v_n == v_ns[-1]:
                line = line + value + r" \\ "
            else:
                line = line + value + " & "
        lines.append(line)

    print("\n")
    print(r"\begin{table}")
    print(r"\caption{I am your little table}")
    print(r"\begin{tabular}{l|" + cells + "}")

    vline = ""
    for i, vlabel in enumerate(vlabels):
        if i == len(vlabels)-1: vline = vline + vlabel + r" \\ "
        else: vline = vline + vlabel + " & "
    print(vline)

    uline = ""
    for i, ulabel in enumerate(units):
        if i == len(units)-1: uline = uline + ulabel + r" \\ "
        else: uline = uline + ulabel + " & "
    print(uline)

    for line in lines: print(line)

    print(r"\end{tabular}")
    print(r"\end{table}")
def print_tex_table_fit_for_models_vej():
    task = [
        {"sim": "DD2_M13641364_M0_LK_R04", "label": r"DD2 q=1.00 (SR)", "type": "long"},
        {"sim": "DD2_M15091235_M0_LK", "label": r"DD2 q=1.22 (SR)", "type": "long"},
        {"sim": "BLh_M13641364_M0_LK", "label": r"BLh q=1.00 (SR)", "type": "long"},
        {"sim": "BLh_M11461635_M0_LK", "label": r"BLh q=1.43 (SR)", "type": "long"},
        {"sim": "SLy4_M13641364_M0", "label": r"SLy4* q=1.00 (SR)", "type": "long"},
        {"sim": "SFHo_M13641364_M0", "label": r"SFHo* q=1.00 (SR)", "type": "long"},

    ]

    v_ns = [
        #"label", "q", "M1", "M2", "C1", "C2", "Lambda", "Mej_tot-geo", "Mej_tot-geo-poly22",
        'label', 'vel_inf_ave-geo', 'vel_inf_ave-geo-mean', 'vel_inf_ave-geo-poly22', "vel_inf_ave-geo-diet16"#, 'vel_inf_ave-geo-krug19'
    ]
    vlabels = [
        "Model", r"$\vd$", r"$\avd$", r"$P_2(q,\tilde\Lambda)$", r"Eq.~\eqref{eq:fit_vej}"#, r"Eq.~\eqref{eq:fit_Mej_Kruger}"
    ]
    fmts = [
        "", ".3f", ".3f", ".3f", ".3f"#, ".1f"
    ]
    units = [
        "", "[c]", "[c]", "[c]", "[c]"#, "[$10^{3}M_{\odot}$]"
    ]
    mults = [
        1., 1., 1., 1., 1., 1.,
        #1e3, 1e3, 1e3, 1e3, 1e3, 1e3,
    ]


    #cells = "c" * 11
    cells = "c" * len(v_ns) # coefs + chi2 + ch2dof + r2
    #


    models = md.groups

    lines = []
    for modeldic in task:
        print("\t processing: {}".format(modeldic["sim"]))
        line = ''
        model = models[models.group == modeldic["sim"]]
        print(model)
        fit = Fit(model)
        for v_n, fmt, mult in zip(v_ns, fmts, mults):
            if v_n == "label":
                value = modeldic["label"]
            elif v_n in model.keys():
                value = "%{}".format(fmt) % float(model[v_n] * mult)
            elif v_n.__contains__("-mean"):
                value = "%{}".format(fmt) % float(fit.value(v_n.replace("-mean", ""), "mean") * mult)
            elif v_n.__contains__("-poly22"):
                value = "%{}".format(fmt) % float(fit.value(v_n.replace("-poly22", ""), "poly22") * mult)
            elif v_n.__contains__("-diet16"):
                value = "%{}".format(fmt) % float(fit.value(v_n.replace("-diet16", ""), "diet16") * mult)
            elif v_n.__contains__("-our"):
                value = "%{}".format(fmt) % float(fit.value(v_n.replace("-our", ""), "our") * mult)
            elif v_n.__contains__("-krug19"):
                value = "%{}".format(fmt) % float(fit.value(v_n.replace("-krug19", ""), "krug19") * mult)
            elif v_n.__contains__("-radi18"):
                value = "%{}".format(fmt) % float(fit.value(v_n.replace("-radi18", ""), "radi18") * mult)
            else:
                raise NameError("name: {} is not recognzied".format(v_n))

            if v_n == v_ns[-1]:
                line = line + value + r" \\ "
            else:
                line = line + value + " & "
        lines.append(line)

    print("\n")
    print(r"\begin{table}")
    print(r"\caption{I am your little table}")
    print(r"\begin{tabular}{l|" + cells + "}")

    vline = ""
    for i, vlabel in enumerate(vlabels):
        if i == len(vlabels)-1: vline = vline + vlabel + r" \\ "
        else: vline = vline + vlabel + " & "
    print(vline)

    uline = ""
    for i, ulabel in enumerate(units):
        if i == len(units)-1: uline = uline + ulabel + r" \\ "
        else: uline = uline + ulabel + " & "
    print(uline)

    for line in lines: print(line)

    print(r"\end{tabular}")
    print(r"\end{table}")
def print_tex_table_fit_for_models_yeej():
    task = [
        {"sim": "DD2_M13641364_M0_LK_R04", "label": r"DD2 q=1.00 (SR)", "type": "long"},
        {"sim": "DD2_M15091235_M0_LK", "label": r"DD2 q=1.22 (SR)", "type": "long"},
        {"sim": "BLh_M13641364_M0_LK", "label": r"BLh q=1.00 (SR)", "type": "long"},
        {"sim": "BLh_M11461635_M0_LK", "label": r"BLh q=1.43 (SR)", "type": "long"},
        {"sim": "SLy4_M13641364_M0", "label": r"SLy4* q=1.00 (SR)", "type": "long"},
        {"sim": "SFHo_M13641364_M0", "label": r"SFHo* q=1.00 (SR)", "type": "long"},

    ]

    v_ns = [
        #"label", "q", "M1", "M2", "C1", "C2", "Lambda", "Mej_tot-geo", "Mej_tot-geo-poly22",
        'label', 'Ye_ave-geo', 'Ye_ave-geo-mean', 'Ye_ave-geo-poly22', "Ye_ave-geo-our"#, 'vel_inf_ave-geo-krug19'
    ]
    vlabels = [
        "Model", r"$\yd$", r"$\ayd$", r"$P_2(q,\tilde\Lambda)$", r"Eq.~\eqref{eq:fit_Yeej}"#, r"Eq.~\eqref{eq:fit_Mej_Kruger}"
    ]
    fmts = [
        "", ".3f", ".3f", ".3f", ".3f"#, ".1f"
    ]
    units = [
        "", "", "", "", ""#, "[$10^{3}M_{\odot}$]"
    ]
    mults = [
        1., 1., 1., 1., 1., 1.,
        #1e3, 1e3, 1e3, 1e3, 1e3, 1e3,
    ]


    #cells = "c" * 11
    cells = "c" * len(v_ns) # coefs + chi2 + ch2dof + r2
    #


    models = md.groups

    lines = []
    for modeldic in task:
        print("\t processing: {}".format(modeldic["sim"]))
        line = ''
        model = models[models.group == modeldic["sim"]]
        print(model)
        fit = Fit(model)
        for v_n, fmt, mult in zip(v_ns, fmts, mults):
            if v_n == "label":
                value = modeldic["label"]
            elif v_n in model.keys():
                value = "%{}".format(fmt) % float(model[v_n] * mult)
            elif v_n.__contains__("-mean"):
                value = "%{}".format(fmt) % float(fit.value(v_n.replace("-mean", ""), "mean") * mult)
            elif v_n.__contains__("-poly22"):
                value = "%{}".format(fmt) % float(fit.value(v_n.replace("-poly22", ""), "poly22") * mult)
            elif v_n.__contains__("-diet16"):
                value = "%{}".format(fmt) % float(fit.value(v_n.replace("-diet16", ""), "diet16") * mult)
            elif v_n.__contains__("-our"):
                value = "%{}".format(fmt) % float(fit.value(v_n.replace("-our", ""), "our") * mult)
            elif v_n.__contains__("-krug19"):
                value = "%{}".format(fmt) % float(fit.value(v_n.replace("-krug19", ""), "krug19") * mult)
            elif v_n.__contains__("-radi18"):
                value = "%{}".format(fmt) % float(fit.value(v_n.replace("-radi18", ""), "radi18") * mult)
            else:
                raise NameError("name: {} is not recognzied".format(v_n))

            if v_n == v_ns[-1]:
                line = line + value + r" \\ "
            else:
                line = line + value + " & "
        lines.append(line)

    print("\n")
    print(r"\begin{table}")
    print(r"\caption{I am your little table}")
    print(r"\begin{tabular}{l|" + cells + "}")

    vline = ""
    for i, vlabel in enumerate(vlabels):
        if i == len(vlabels)-1: vline = vline + vlabel + r" \\ "
        else: vline = vline + vlabel + " & "
    print(vline)

    uline = ""
    for i, ulabel in enumerate(units):
        if i == len(units)-1: uline = uline + ulabel + r" \\ "
        else: uline = uline + ulabel + " & "
    print(uline)

    for line in lines: print(line)

    print(r"\end{tabular}")
    print(r"\end{table}")
def print_tex_table_fit_for_models_Mdisk():
    task = [
        {"sim": "DD2_M13641364_M0_LK_R04", "label": r"DD2 q=1.00 (SR)", "type": "long"},
        {"sim": "DD2_M15091235_M0_LK", "label": r"DD2 q=1.22 (SR)", "type": "long"},
        {"sim": "BLh_M13641364_M0_LK", "label": r"BLh q=1.00 (SR)", "type": "long"},
        {"sim": "BLh_M11461635_M0_LK", "label": r"BLh q=1.43 (SR)", "type": "long"},
        {"sim": "SLy4_M13641364_M0", "label": r"SLy4* q=1.00 (SR)", "type": "long"},
        {"sim": "SFHo_M13641364_M0", "label": r"SFHo* q=1.00 (SR)", "type": "long"},

    ]

    v_ns = [
        #"label", "q", "M1", "M2", "C1", "C2", "Lambda", "Mej_tot-geo", "Mej_tot-geo-poly22",
        'label', 'Mdisk3D', 'Mdisk3D-mean', 'Mdisk3D-poly22', "Mdisk3D-radi18", 'Mdisk3D-krug19'
    ]
    vlabels = [
        "Model", r"$M_{\rm disk}$", r"$\overline{M}_{\rm disk}$", r"$P_2(q,\tilde\Lambda)$", r"Eq.~\eqref{eq:fit_Mdisk}", r"Eq.~\eqref{eq:fit_Mdisk_Kruger}"
    ]
    fmts = [
        "", ".3f", ".3f", ".3f", ".3f", ".3f"
    ]
    units = [
        "", "[$M_{\odot}$]", "[$M_{\odot}$]", "[$M_{\odot}$]", "[$M_{\odot}$]", "[$M_{\odot}$]"
    ]
    mults = [
        1., 1., 1., 1., 1., 1.,
        #1e3, 1e3, 1e3, 1e3, 1e3, 1e3,
    ]


    cells = "c" * 11
    cells = "c" * len(v_ns) # coefs + chi2 + ch2dof + r2
    #


    models = md.groups

    lines = []
    for modeldic in task:
        print("\t processing: {}".format(modeldic["sim"]))
        line = ''
        model = models[models.group == modeldic["sim"]]
        print(model)
        fit = Fit(model)
        for v_n, fmt, mult in zip(v_ns, fmts, mults):
            if v_n == "label":
                value = modeldic["label"]
            elif v_n in model.keys():
                value = "%{}".format(fmt) % float(model[v_n] * mult)
            elif v_n.__contains__("-mean"):
                value = "%{}".format(fmt) % float(fit.value(v_n.replace("-mean", ""), "mean") * mult)
            elif v_n.__contains__("-poly22"):
                value = "%{}".format(fmt) % float(fit.value(v_n.replace("-poly22", ""), "poly22") * mult)
            elif v_n.__contains__("-diet16"):
                value = "%{}".format(fmt) % float(fit.value(v_n.replace("-diet16", ""), "diet16") * mult)
            elif v_n.__contains__("-our"):
                value = "%{}".format(fmt) % float(fit.value(v_n.replace("-our", ""), "our") * mult)
            elif v_n.__contains__("-krug19"):
                value = "%{}".format(fmt) % float(fit.value(v_n.replace("-krug19", ""), "krug19") * mult)
            elif v_n.__contains__("-radi18"):
                value = "%{}".format(fmt) % float(fit.value(v_n.replace("-radi18", ""), "radi18") * mult)
            else:
                raise NameError("name: {} is not recognzied".format(v_n))

            if v_n == v_ns[-1]:
                line = line + value + r" \\ "
            else:
                line = line + value + " & "
        lines.append(line)

    print("\n")
    print(r"\begin{table}")
    print(r"\caption{I am your little table}")
    print(r"\begin{tabular}{l|" + cells + "}")

    vline = ""
    for i, vlabel in enumerate(vlabels):
        if i == len(vlabels)-1: vline = vline + vlabel + r" \\ "
        else: vline = vline + vlabel + " & "
    print(vline)

    uline = ""
    for i, ulabel in enumerate(units):
        if i == len(units)-1: uline = uline + ulabel + r" \\ "
        else: uline = uline + ulabel + " & "
    print(uline)

    for line in lines: print(line)

    print(r"\end{tabular}")
    print(r"\end{table}")


# def task_plot_peak_times():
#
#     # task = [
#     #     # 2 prompt collapse
#     #     # {"sim": "BLh_M10201856_M0_LK_LR_AHfix", "label": r"BLh** q=1.82 (LR)", "color": "black", "ls": ":", "lw": 0.8, "alpha": 1., "type": "long"},
#     #     # {"sim": "BLh_M10201856_M0_LK_LR", "label": r"BLh* q=1.82 (LR)", "color": "black", "ls": "-", "lw": 1.0, "alpha": 1., "type": "long"},
#     #     {"sim": "BLh_M10201856_M0", "label": r"BLh* q=1.82 (SR)", "type": "long", "plot":{"color": "black", "ls": "-", "lw": 1.0, "alpha": 1.}},
#     #     {"sim": "BLh_M10651772_M0_LK", "label": r"BLh q=1.66 (LR)", "type": "long", "plot":{"color": "black", "ls": "-", "lw": 1.0, "alpha": 1.}},
#     #     # {"sim": "BLh_M11841581_M0_LK_SR", "label": r"BLh q=1.34 (SR)", "color": "black", "ls": "-.", "lw": 1.0, "alpha": 1., "type": "long"},
#     #     # 3 short lived
#     #     {"sim": "LS220_M11461635_M0_LK", "label": r"LS220 q=1.43 (SR)", "type": "short", "plot":{"color": "black", "ls": "-", "lw": 1.0, "alpha": 1.}},
#     #     {"sim": "SLy4_M13641364_M0", "label": r"SLy4* q=1.00 (SR)", "type": "short", "plot":{"color": "black", "ls": "-", "lw": 1.0, "alpha": 1.}},
#     #     {"sim": "SFHo_M13641364_M0", "label": r"SFHo* q=1.00 (SR)", "type": "short", "plot":{"color": "black", "ls": "-", "lw": 1.0, "alpha": 1.}},
#     #     # long
#     #     {"sim": "BLh_M13641364_M0_LK", "label": r"BLh q=1.00 (SR)",  "type": "long", "plot":{"color": "black", "ls": "-", "lw": 1.0, "alpha": 1.}},
#     #     # {"sim": "DD2_M13641364_M0_LK_SR_R04", "label": r"DD2* q=1.00 (SR)", "color": "blue", "ls": "--", "lw": 1.0, "alpha": 1., "type": "long", "t1": 40},
#     # ]
#
#     task = [
#         {"sim": "BLh_M13641364_M0_LK", "label": r"BLh q=1.00 (SR)", "plot":{}},
#         {"sim": "DD2_M13641364_M0_LK_R04", "label": r"DD2* q=1.00 (SR)", "plot":{}}
#         # {"sim": "LS220_M11461635_M0_LK", "label": r"LS220 q=1.43 (SR)",  "plot":{}}
#         # {"sim": "SFHo_M13641364_M0", "label": r"SFHo* q=1.00 (SR)",
#     ]
#
#     for t in task:
#         t["plot"]["color"] = md.sim_dic_color[t["sim"]]
#     #     t["plot"]["ls"] = md.sim_dic_ls[t["sim"]]
#     #     t["plot"]["lw"] = md.sim_dic_lw[t["sim"]]
#
#     bands = ["g", "z", "Ks"]
#     #components = ["dynamic", "dynamic secular"]
#
#     #
#     parameters = [
#         # {"label": r"$M_{\rm ej}", "ls":"-", r"v_{\rm ej}$ $P_2(q,\tilde\Lambda)$ ", "Mej_tot-geo": "poly22", "vel_inf_ave-geo": "poly22", "Mdisk3D": "poly22"},
#        {"label": r"$M_{\rm ej}, v_{\rm ej}$ Mean", "ls":":",
#         "Mej_tot-geo": "mean", "vel_inf_ave-geo": "mean", "Mdisk3D": "mean"},
#         {"label": r"$M_{\rm ej} v_{\rm ej}$ $P_2(q,\tilde\Lambda)$", "ls": "-",
#          "Mej_tot-geo": "poly22", "vel_inf_ave-geo": "poly22", "Mdisk3D": "poly22"},
#         {"label": r"$M_{\rm ej}$ Eq.(6) $v_{\rm ej}$ Eq.(9)", "ls": "--",
#          "Mej_tot-geo": "diet16", "vel_inf_ave-geo": "diet16", "Mdisk3D": "radi18"},
#         {"label": r"$M_{\rm ej}$ Eq.(7) $v_{\rm ej}$ Eq.(9)", "ls": "-.",
#          "Mej_tot-geo": "krug19", "vel_inf_ave-geo": "diet16", "Mdisk3D": "krug19"}
#     ]
#
#     # collect data
#     models = md.groups
#     lineset = {}
#     for band in bands:
#         lineset[band] = {}
#         #
#         for modeldic in task:
#             lineset[band][modeldic["sim"]] = {}
#             model = models[models.group == modeldic["sim"]]
#             ofit = Fit(model)
#             #
#             for pars in parameters:
#
#                 print("\t\t{} {} {}".format(band, modeldic["sim"], pars["label"]))
#
#                 lineset[band][modeldic["sim"]][pars["label"]] = {}
#
#                 mej = ofit.value("Mej_tot-geo", pars["Mej_tot-geo"])
#                 vej = ofit.value("vel_inf_ave-geo", pars["vel_inf_ave-geo"])
#                 mdisk = ofit.value("Mdisk3D", pars["Mdisk3D"])
#                 #
#                 vals = {"Mej_tot-geo": mej, "vel_inf_ave-geo": vej, "Mdisk3D": mdisk}
#                 #
#                 o_mkn = COMPUTE_LIGHTCURVE(None, "/data01/numrel/vsevolod.nedora/prj_gw170817/scripts/lightcurves/")
#                 o_mkn.set_glob_par_var_source(False, False)
#                 o_mkn.set_dyn_iso_aniso = "aniso"
#                 o_mkn.set_dyn_par_var("aniso")
#                 o_mkn.glob_vars['m_disk']                       = vals["Mdisk3D"]
#                 o_mkn.ejecta_vars['dynamics']["mej"]            = vals["Mej_tot-geo"]
#                 o_mkn.ejecta_vars["dynamics"]["central_vel"]    = vals["vel_inf_ave-geo"]
#                 o_mkn.compute_save_lightcurve(write_output=True)
#                 #
#                 load_mkn = COMBINE_LIGHTCURVES(None, "/data01/numrel/vsevolod.nedora/prj_gw170817/scripts/lightcurves/")
#                 times, mags = load_mkn.get_model_median(band)
#                 #
#                 linedic = {}
#                 linedic["x"] = times
#                 linedic["y"] = mags
#                 linedic["color"] = modeldic["plot"]["color"]
#                 linedic["ls"] = pars["ls"]
#                 linedic["alpha"] = 1.
#                 linedic["lw"] = 1.
#
#                 lineset[band][modeldic["sim"]][pars["label"]] = linedic
#
#                 #
#     print("data is collected")
#
#     subplot_dics = [
#         {
#             "band" : "g",
#             "data" : lineset["g"],
#             "add_lines": [
#                 {"ls": "-", "color": "blue", "label": "DD2 q=1.00 (SR)"},
#                 {"ls": "-", "color": "red", "label": "SFho q=1.00 (SR)"},
#                 {"ls": "-", "color": "green", "label": "BLh q=1.43 (SR)"},
#             ],
#             "legend": {"fancybox": False, "loc": 'upper right',
#                    # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
#                    "shadow": "False", "ncol": 1, "fontsize": 13, "columnspacing": 0.4,
#                    "framealpha": 0., "borderaxespad": 0., "frameon": False},
#         },
#         {
#             "band": "z",
#             "data": lineset["z"],
#             "add_lines": [
#                 {"ls": "-", "color": "blue", "label": "DD2 q=1.00 (SR)"},
#                 {"ls": "-", "color": "red", "label": "SFho q=1.00 (SR)"},
#                 {"ls": "-", "color": "green", "label": "BLh q=1.43 (SR)"},
#             ],
#             "legend": {"fancybox": False, "loc": 'upper right',
#                        # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
#                        "shadow": "False", "ncol": 1, "fontsize": 13, "columnspacing": 0.4,
#                        "framealpha": 0., "borderaxespad": 0., "frameon": False},
#         }
#
#     ]
#
#     #
#
#     plot_dic = {
#         "task_v_n" : "theta",
#         "type": "long",
#         "figsize": (16., 5.5),
#         'xmin': 3e-1, 'xmax': 3e1, 'xlabel': r"time [days]",
#         'ymin': 23, 'ymax': 18, 'ylabel': r"AB magnitude at 40 Mpc",
#         # "text": {'x': 0.25, 'y': 0.95, 's': r"Ks band", 'fontsize': 14, 'color': 'black',
#         #          'horizontalalignment': 'center'},
#         "xscale": "log", "yscale": "linear",
#         "legend": {"fancybox": False, "loc": 'upper right',
#                    # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
#                    "shadow": "False", "ncol": 1, "fontsize": 13, "columnspacing": 0.4,
#                    "framealpha": 0., "borderaxespad": 0., "frameon": False},
#         "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
#                         "labelright": False, "tick1On": True, "tick2On": True,
#                         "labelsize": 14,"direction": 'in',
#                         "bottom": True, "top": True, "left": True, "right": True},
#         "figname": __outplotdir__ + "dyn_fit_bands.png",
#         'fontsize': 14,
#         'labelsize': 14,
#         "savepdf": True,
#         #"title": "#band",
#         "add_lines": [
#             {"ls": "-", "color": "blue", "label": "DD2 q=1.00 (SR)"},
#             {"ls": "-", "color": "red", "label": "SFho q=1.00 (SR)"},
#             {"ls": "-", "color": "green", "label": "BLh q=1.43 (SR)"},
#         ],
#     }
#
#     # plot_dic = {
#     #     # "subplots":{"figsize": (6.0, 6.0), "ncols":1,"nrows":3, "sharex":True,"sharey":False},
#     #     "figsize": (6.0, 5.5),
#     #     'xmin': 3e-1, 'xmax': 3e1, 'xlabel': r"time [days]",
#     #     'ymin': 23, 'ymax': 18, 'ylabel': r"AB magnitude at 40 Mpc",
#     #     'fontsize': 14,
#     #     'labelsize': 14,
#     #     "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
#     #                     "labelright": False, "tick1On": True, "tick2On": True,
#     #                     "labelsize": 14,
#     #                     "direction": 'in',
#     #                     "bottom": True, "top": True, "left": True, "right": True},
#     #     "legend": {"fancybox": False, "loc": 'upper right',
#     #                # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
#     #                "shadow": "False", "ncol": 1, "fontsize": 13, "columnspacing": 0.4,
#     #                "framealpha": 0., "borderaxespad": 0., "frameon": False},
#     #     "text": {'x': 0.25, 'y': 0.95, 's': r"Ks band", 'fontsize': 14, 'color': 'black',
#     #              'horizontalalignment': 'center'},
#     #     # "add_legend"
#     #     "add_lines": [
#     #         {"ls": "-", "color": "blue", "label": "DD2 q=1.00 (SR)"},
#     #         {"ls": "-", "color": "red", "label": "SFho q=1.00 (SR)"},
#     #         {"ls": "-", "color": "green", "label": "BLh q=1.43 (SR)"},
#     #     ],
#     #     "add_legend": {"fancybox": False, "loc": 'lower left',
#     #                    # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
#     #                    "shadow": "False", "ncol": 1, "fontsize": 13, "columnspacing": 0.4,
#     #                    "framealpha": 0., "borderaxespad": 0., "frameon": False},
#     #     "savepdf": True,
#     #     "figname": __outplotdir__ + "Ks_dyn_fit.png",
#     #     "dpi": 128,
#     #     "tight_layout": True
#     # }
#
#     custom_plot_mkns_several_bands(plot_dic, bands, lineset)

def task_plot_fromfit_model_mkn_multiband():

    bands = ["g", "z", "Ks"]
    task = [
        {"sim": "BLh_M13641364_M0_LK", "plot": {"label": r"BLh q=1.00 (SR)", "color":"green"}},
        {"sim": "BLh_M11461635_M0_LK", "plot": {"label": r"BLh q=1.43 (SR)", "color": "red"}},
        #{"sim": "BLh_M11461635_M0_LK", "plot": {"label": r"BLh q=1.43 (SR)", "color": "red"}},
        # {"sim": "DD2_M13641364_M0_LK_R04", "plot": {"label": r"DD2* q=1.00 (SR)"}}
        # {"sim": "LS220_M11461635_M0_LK", "label": r"LS220 q=1.43 (SR)",  "plot":{}}
        # {"sim": "LS220_M13641364_M0_LK", "plot": {"label": r"LS220 q=1.00 (SR)", "color": "red"}},
        # {"sim": "SFHo_M13641364_M0", "plot": {"label": r"SFHo* q=1.00 (SR)", "color":"red"}},
    ]
    # for t in task:
    #     t["plot"]["color"] = md.sim_dic_color[t["sim"]]

    pardics = [
        {"Mej_tot-geo": None, "vel_inf_ave-geo": None, "Mdisk3D": None,             "plot": {"lw":2., "ls": "-", "color": "gray", "label": r"$M_{\rm ej}, v_{\rm ej}$ Model"}},
        {"Mej_tot-geo": "mean", "vel_inf_ave-geo": "mean", "Mdisk3D": "mean",       "plot": {"lw":1., "ls": ":", "color": "gray", "label": r"$M_{\rm ej}, v_{\rm ej}$ Mean"}},
        {"Mej_tot-geo": "poly22", "vel_inf_ave-geo": "poly22", "Mdisk3D": "poly22", "plot": {"lw":1., "ls": "-", "color": "gray", "label": r"$M_{\rm ej} v_{\rm ej}$ $P_2(q,\tilde\Lambda)$"}},
        {"Mej_tot-geo": "diet16", "vel_inf_ave-geo": "diet16", "Mdisk3D": "radi18", "plot": {"lw":1., "ls": "--", "color": "gray", "label": r"$M_{\rm ej}$ Eq.(6) $v_{\rm ej}$ Eq.(9)"}},
        {"Mej_tot-geo": "krug19", "vel_inf_ave-geo": "diet16", "Mdisk3D": "krug19", "plot": {"lw":1., "ls": "-.", "color": "gray", "label": r"$M_{\rm ej}$ Eq.(7) $v_{\rm ej}$ Eq.(9)"}}
    ]

    ''' --- collecting data --- '''

    models = md.groups
    lineset = {}
    for band in bands:
        lineset[band] = {}
        #
        for modeldic in task:
            lineset[band][modeldic["sim"]] = {}
            model = models[models.group == modeldic["sim"]]
            ofit = Fit(model)
            #
            for pars in pardics:
                print("\t\t{} {} {}".format(band, modeldic["sim"], pars["plot"]["label"]))

                lineset[band][modeldic["sim"]][pars["plot"]["label"]] = {}

                mej = ofit.value("Mej_tot-geo",     pars["Mej_tot-geo"])
                vej = ofit.value("vel_inf_ave-geo", pars["vel_inf_ave-geo"])
                mdisk = ofit.value("Mdisk3D",       pars["Mdisk3D"])
                #
                vals = {"Mej_tot-geo": mej, "vel_inf_ave-geo": vej, "Mdisk3D": mdisk}
                #
                o_mkn = COMPUTE_LIGHTCURVE(None, "/data01/numrel/vsevolod.nedora/prj_gw170817/scripts/lightcurves/")
                o_mkn.set_glob_par_var_source(False, False)
                o_mkn.set_dyn_iso_aniso = "aniso"
                o_mkn.set_secular_iso_aniso = "aniso"
                o_mkn.set_dyn_par_var("aniso")
                o_mkn.set_secular_par_war("aniso")
                o_mkn.glob_vars['m_disk']                       = vals["Mdisk3D"]
                o_mkn.ejecta_vars['dynamics']["m_ej"]           = vals["Mej_tot-geo"]
                o_mkn.ejecta_vars["dynamics"]["central_vel"]    = vals["vel_inf_ave-geo"]
                o_mkn.compute_save_lightcurve(write_output=True)
                #
                load_mkn = COMBINE_LIGHTCURVES(None, "/data01/numrel/vsevolod.nedora/prj_gw170817/scripts/lightcurves/")
                times, mags = load_mkn.get_model_median(band)
                #
                linedic = {}
                linedic["x"] = times
                linedic["y"] = mags
                linedic["color"] = modeldic["plot"]["color"]
                linedic["ls"] = pars["plot"]["ls"]
                linedic["alpha"] = 1.
                linedic["lw"] = pars["plot"]["lw"]

                lineset[band][modeldic["sim"]][pars["plot"]["label"]] = linedic

    ''' --- plot settings --- '''

    subplot_dics = [
        {
            "band": bands[0],
            "lines": lineset[bands[0]],
            "add_lines": [dic["plot"] for dic in pardics],
            "title": "#band",
            'xmin': 1e-1, 'xmax': 2e0, 'xlabel': r"time [days]",
        },
        {
            "band": bands[1],
            "lines": lineset[bands[1]],
            #"add_lines": [dic["plot"] for dic in pardics],
            "title": "#band",
            'xmin': 3e-1, 'xmax': 1e1, 'xlabel': r"time [days]",
        },
        {
            "band": bands[2],
            "lines": lineset[bands[2]],
            "add_lines": [dic["plot"] for dic in task],
            "title": "#band",
            'xmin': 3e-1, 'xmax': 3e1, 'xlabel': r"time [days]",
        }
    ]

    plot_dic = {
        "task_v_n" : "theta",
        "type": "long",
        "figsize": (16., 5.5),
        'xmin': 3e-1, 'xmax': 3e1, 'xlabel': r"time [days]",
        'ymin': 22, 'ymax': 17, 'ylabel': r"AB magnitude at 40 Mpc", # 23, 18 # sec: 22 17
        # "text": {'x': 0.25, 'y': 0.95, 's': r"Ks band", 'fontsize': 14, 'color': 'black',
        #          'horizontalalignment': 'center'},
        "xscale": "log", "yscale": "linear",
        "legend": {"fancybox": False, "loc": 'lower left',
                   # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 13, "columnspacing": 0.4,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False, "tick1On": True, "tick2On": True,
                        "labelsize": 14,"direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        "figname": __outplotdir__ + "dyn_sec_fit_bands.png",
        'fontsize': 14,
        'labelsize': 14,
        "savepdf": True,
        "title": "#band",
        "add_lines": [
            [dic["plot"] for dic in pardics]
            # {"ls": ":", "color": "gray", "label": r"$M_{\rm ej}, v_{\rm ej}$ Mean"},
            # {"ls": "-", "color": "gray", "label": "SFho q=1.00 (SR)"},
            # {"ls": "-", "color": "gray", "label": "BLh q=1.43 (SR)"},
            # {"ls": "-", "color": "gray", "label": "BLh q=1.43 (SR)"}
        ],
    }

    custom_plot_mkns_several_bands(plot_dic, subplot_dics)


def plot_custom_lightcurves(plotdic, subplot_dics, tasks):

    models = md.simulations[md.mask_for_with_dynej]

    bands = [subplot_dic["band"] for subplot_dic in subplot_dics]
    fig, axes = plt.subplots(figsize=plotdic["figsize"], ncols=len(bands), nrows=1, sharey=True)
    # ax = fig.add_subplot(111)
    i = 0
    for ax, band, subplot_dic in zip(axes, bands, subplot_dics):

        for task in tasks:

            if "obs" in task.keys():
                task_dic = task["obs"]
                if task_dic["method"] == "peak marker":
                    load_mkn = COMBINE_LIGHTCURVES(None, __lightcurves_store__)
                    tp, mp, emp = load_mkn.get_obs_peak(band)
                    # print(tp, mp, emp); exit(1)
                    # ax.scatter(tp, mp, **task_dic["plot"])
                    # ax.axvline(x=tp, linestyle="-.", color="gray")
                    # ax.axhline(y=mp, linestyle="-.", color="gray")
                    ax.errorbar(tp, mp, yerr=emp, xerr=None, fmt='',
                                fillstyle="none", ecolor="gray", capsize=4,
                                markersize=7, marker="o")

            if "model" in task.keys():

                sim = task["sim"]
                model = models[models.index == sim]
                ofit = Fit(model)

                # load the NR data, compute lightcurve, plot lightcurve

                # mdisk = ofit.value("Mdisk3D", task["Mdisk3D"])
                fname = "{}{}/outflow_0/geo/ejecta_profile.dat".format(__nr_data__, sim)
                assert os.path.isfile(fname)

                task_dic = task["model"]
                o_mkn = COMPUTE_LIGHTCURVE(sim, "/data01/numrel/vsevolod.nedora/prj_gw170817/scripts/lightcurves/")
                o_mkn.dyn_ejecta_profile_fpath = fname
                o_mkn.set_use_bern_NR = False
                o_mkn.set_use_dyn_NR = True
                o_mkn.set_glob_par_var_source(True, False)

                if "dyn" in task_dic["components"]: o_mkn.set_dyn_iso_aniso = "aniso"
                if "sec" in task_dic["components"]: o_mkn.set_secular_iso_aniso = "aniso"
                if "dyn" in task_dic["components"]: o_mkn.set_dyn_par_var("aniso")
                if "sec" in task_dic["components"]: o_mkn.set_secular_par_war("aniso")
                ### o_mkn.glob_vars['m_disk'] = mdisk

                o_mkn.compute_save_lightcurve(write_output=True)

                # "/data01/numrel/vsevolod.nedora/prj_gw170817/scripts/lightcurves/"
                load_mkn = COMBINE_LIGHTCURVES(None, __lightcurves_store__)
                times, mags = load_mkn.get_model_median(band)
                #
                ax.plot(times, mags, **task_dic["plot"])

            if "best" in task.keys() :

                sim = task["sim"]
                model = models[models.index == sim]
                ofit = Fit(model)

                # extract paramters from the dic, compute lightcurve
                task_dic = task["best"]

                mej = ofit.value("Mej_tot-geo",     task_dic["Mej_tot-geo"])
                vej = ofit.value("vel_inf_ave-geo", task_dic["vel_inf_ave-geo"])
                mdisk = ofit.value("Mdisk3D",       task_dic["Mdisk3D"])
                theta_ej = ofit.value("theta_rms-geo",       task_dic["theta_rms-geo"])

                print("\t[{}] Mej:{} vej:{} mdisk:{}".format(sim, mej, vej, mdisk))

                o_mkn = COMPUTE_LIGHTCURVE(None, "/data01/numrel/vsevolod.nedora/prj_gw170817/scripts/lightcurves/")
                o_mkn.set_glob_par_var_source(False, False)
                if "dyn" in task_dic["components"]: o_mkn.set_dyn_iso_aniso = "aniso"
                if "sec" in task_dic["components"]: o_mkn.set_secular_iso_aniso = "aniso"
                if "dyn" in task_dic["components"]: o_mkn.set_dyn_par_var("aniso")
                if "sec" in task_dic["components"]: o_mkn.set_secular_par_war("aniso")
                o_mkn.glob_vars['m_disk']                   = mdisk
                o_mkn.ejecta_vars['dynamics']["m_ej"]       = mej
                o_mkn.ejecta_vars["dynamics"]["central_vel"]= vej
                o_mkn.ejecta_vars["dynamics"]["step_angle_op"] = np.radians(theta_ej)
                o_mkn.compute_save_lightcurve(write_output=True)
                #
                load_mkn = COMBINE_LIGHTCURVES(None, __lightcurves_store__)
                times, mags = load_mkn.get_model_median(band)

                ax.plot(times, mags, **task_dic["plot"])

            if "worst" in task.keys():

                sim = task["sim"]
                model = models[models.index == sim]
                ofit = Fit(model)

                # extract paramters from the dic, compute lightcurve

                task_dic = task["worst"]


                mej = ofit.value("Mej_tot-geo",     task_dic["Mej_tot-geo"])
                vej = ofit.value("vel_inf_ave-geo", task_dic["vel_inf_ave-geo"])
                mdisk = ofit.value("Mdisk3D",       task_dic["Mdisk3D"])
                theta_ej = ofit.value("theta_rms-geo", task_dic["theta_rms-geo"])

                o_mkn = COMPUTE_LIGHTCURVE(None, __lightcurves_store__)
                o_mkn.set_glob_par_var_source(False, False)
                if "dyn" in task_dic["components"]: o_mkn.set_dyn_iso_aniso = "aniso"
                if "sec" in task_dic["components"]: o_mkn.set_secular_iso_aniso = "aniso"
                if "dyn" in task_dic["components"]: o_mkn.set_dyn_par_var("aniso")
                if "sec" in task_dic["components"]: o_mkn.set_secular_par_war("aniso")
                o_mkn.glob_vars['m_disk']                   = mdisk
                o_mkn.ejecta_vars['dynamics']["m_ej"]       = mej
                o_mkn.ejecta_vars["dynamics"]["central_vel"]= vej
                o_mkn.ejecta_vars["dynamics"]["step_angle_op"] = np.radians(theta_ej)
                o_mkn.compute_save_lightcurve(write_output=True)
                #
                load_mkn = COMBINE_LIGHTCURVES(None, __lightcurves_store__)
                times, mags = load_mkn.get_model_median(band)

                # print("mags", mags)
                # if click.confirm("is it ok?",True):
                #     pass

                ax.plot(times, mags, **task_dic["plot"])

            if "fits" in task.keys():

                sim = task["sim"]
                model = models[models.index == sim]
                ofit = Fit(model)

                # for ever parameter set in parameter dics compute lightcurve, plot a band they span
                fit_list = task["fits"]
                all_mags = []
                times = []
                for fit_par_dic in fit_list:

                    mej = ofit.value("Mej_tot-geo",     fit_par_dic["Mej_tot-geo"])
                    vej = ofit.value("vel_inf_ave-geo", fit_par_dic["vel_inf_ave-geo"])
                    mdisk = ofit.value("Mdisk3D",       fit_par_dic["Mdisk3D"])
                    theta_ej = ofit.value("theta_rms-geo", fit_par_dic["theta_rms-geo"])

                    #compute line
                    o_mkn = COMPUTE_LIGHTCURVE(None, __lightcurves_store__)
                    o_mkn.set_glob_par_var_source(False, False)
                    if "dyn" in fit_par_dic["components"]: o_mkn.set_dyn_iso_aniso = "aniso"
                    if "sec" in fit_par_dic["components"]: o_mkn.set_secular_iso_aniso = "aniso"
                    if "dyn" in fit_par_dic["components"]: o_mkn.set_dyn_par_var("aniso")
                    if "sec" in fit_par_dic["components"]: o_mkn.set_secular_par_war("aniso")
                    o_mkn.glob_vars['m_disk']                   = mdisk
                    o_mkn.ejecta_vars['dynamics']["m_ej"]       = mej
                    o_mkn.ejecta_vars["dynamics"]["central_vel"]= vej
                    o_mkn.ejecta_vars["dynamics"]["step_angle_op"] = np.radians(theta_ej)
                    o_mkn.compute_save_lightcurve(write_output=True)
                    #
                    load_mkn = COMBINE_LIGHTCURVES(None, __lightcurves_store__)
                    times, mags = load_mkn.get_model_median(band)
                    all_mags.append(mags)
                all_mags = np.reshape(np.array(all_mags), (len(fit_list), len(times))).T
                maxs = np.max(all_mags, axis=1)
                mins = np.min(all_mags, axis=1)
                # maxs = np.array([ np.max(all_mags[:, k]) for k in all_mags[:, k] ])
                # mins = np.array([np.min(all_mags[:, k]) for k in all_mags[:, k]])
                ax.fill_between(times, mins, maxs, **task["plotfits"])


        # end of tasks

        ax.set_yscale(plotdic["yscale"])
        ax.set_xscale(plotdic["xscale"])
        #
        ax.set_xlabel(plotdic["xlabel"], fontsize=plotdic["fontsize"])  # , fontsize=11)
        if i == 0: ax.set_ylabel(plotdic["ylabel"], fontsize=plotdic["fontsize"])  # , fontsize=11)
        #
        if 'xmin' in subplot_dic.keys() and 'xmax' in subplot_dic.keys():
            ax.set_xlim(subplot_dic["xmin"], subplot_dic["xmax"])
        else:
            ax.set_xlim(plotdic["xmin"], plotdic["xmax"])

        if 'ymin' in subplot_dic.keys() and 'ymax' in subplot_dic.keys():
            ax.set_xlim(subplot_dic["ymin"], subplot_dic["ymax"])
        else:
            ax.set_ylim(plotdic["ymin"], plotdic["ymax"])
        #
        ax.tick_params(axis='both', which='both', labelleft=True,
                       labelright=False, tick1On=True, tick2On=True,
                       labelsize=plotdic["fontsize"],
                       direction='in',
                       bottom=True, top=True, left=True, right=True)
        ax.minorticks_on()
        # #

        if "add_lines" in subplot_dic.keys():
            for entry in subplot_dic["add_lines"]:
                ax.plot([-100, -100], [-200, -200], **entry)
        #
        if "add_scatter" in subplot_dic.keys():
            for entry in subplot_dic["add_scatter"]:
                ax.scatter([-100, -100], [-200, -200], **entry)
        #
        if "add_bands" in subplot_dic.keys():
            for entry in subplot_dic["add_bands"]:
                ax.fill_between([-100, -100], [-200, -200], [-400, -400], **entry)
        #
        if "title" in subplot_dic:
            ax.set_title(subplot_dic["title"])
        #
        if "title" in plotdic.keys():
            if plotdic["title"] == "#band":
                ax.set_title(band)
            else:
                ax.set_title(plotdic["title"])
        #
        if i > 0:
            ax.tick_params(labelleft=False)
            # ax.get_yaxis().set_ticks([])
            # ax.get_yaxis().set_visible(False)

        if "legend" in subplot_dic.keys():
            ax.legend(**subplot_dic["legend"])

        i = i + 1

    if "legend" in plotdic.keys() and len(plotdic["legend"].keys()) > 0:
        plt.legend(**plotdic["legend"])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.)
    #
    print("plotted: \n")
    print(plotdic["figname"])
    plt.savefig(plotdic["figname"], dpi=128)
    if "savepdf" in plotdic.keys() and plotdic["savepdf"]:
        plt.savefig(plotdic["figname"].replace(".png", ".pdf"))
    plt.close()

def task_plot_fromfit_model_mkn_multiband_band():

    bands = ["g", "z", "Ks"]

    ''' Best : mej_krug | vej_poly22 | yeej_poly22 | mdisk_poly22 '''
    ''' Worst : "mean" , "mean", "mean", "mean"'''

    tasks = [
        {"sim": "BLh_M13641364_M0_LK_SR",
         "model": {"Mej_tot-geo": "model", "vel_inf_ave-geo": "model", "Mdisk3D": "model", "components":["dyn"],#"sec"],
                   "plot": {"lw": 1., "ls": "--", "color": "green"}},# "label": r"Model"}},
         "best":  {"Mej_tot-geo": "krug19", "vel_inf_ave-geo": "poly22", "Mdisk3D": "poly22",  "components":["dyn"],#"sec"],
                  "plot": {"lw": 1., "ls": "-", "color": "green"}},#, "label": r"Best Fit"}},
         "worst": {"Mej_tot-geo": "mean", "vel_inf_ave-geo": "mean", "Mdisk3D": "mean",  "components":["dyn"],#"sec"],
                   "plot": {"alpha":0.7, "lw": 0.9, "ls": ":", "color": "green"}},  # , "label": r"Best Fit"}},
         # "fits": [
         #     {"Mej_tot-geo": None, "vel_inf_ave-geo": None, "Mdisk3D": None},
         #     {"Mej_tot-geo": "poly22", "vel_inf_ave-geo": "poly22", "Mdisk3D": "poly22"},
         #     {"Mej_tot-geo": "diet16", "vel_inf_ave-geo": "diet16", "Mdisk3D": "radi18"},
         #     {"Mej_tot-geo": "krug19", "vel_inf_ave-geo": "diet16", "Mdisk3D": "krug19"}],
         # "plotfits":{"color":"green", "alpha": 0.3}#, "label": "Other Fits"}
         },
        # --=-=-==-=--=--=-=-=-=-=-=-=-=-=-==-=-=-=-==-=-=-==-==-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        {"sim": "DD2_M13641364_M0_LK_SR_R04",# "BLh_M11461635_M0_LK_SR",
         "model": {"Mej_tot-geo": "model", "vel_inf_ave-geo": "model", "Mdisk3D": "model",  "components":["dyn"],#"sec"],
                   "plot": {"lw": 1., "ls": "--", "color": "blue"}},  # "label": r"Model"}},
         "best": {"Mej_tot-geo": "poly2", "vel_inf_ave-geo": "poly22", "Mdisk3D": "poly22",  "components":["dyn"],#"sec"],
                  "plot": {"lw": 1., "ls": "-", "color": "blue"}},  # , "label": r"Best Fit"}},
         "worst": {"Mej_tot-geo": "krug19", "vel_inf_ave-geo": None, "Mdisk3D": None,  "components":["dyn"],#"sec"],
                  "plot": {"alpha":0.7, "lw": 0.9, "ls": ":", "color": "blue"}},  # , "label": r"Best Fit"}},
         # "fits": [
         #     {"Mej_tot-geo": None, "vel_inf_ave-geo": None, "Mdisk3D": None},
         #     {"Mej_tot-geo": "poly22", "vel_inf_ave-geo": "poly22", "Mdisk3D": "poly22"},
         #     {"Mej_tot-geo": "diet16", "vel_inf_ave-geo": "diet16", "Mdisk3D": "radi18"},
         #     {"Mej_tot-geo": "krug19", "vel_inf_ave-geo": "diet16", "Mdisk3D": "krug19"}],
         # "plotfits": {"color": "blue", "alpha": 0.3}  # , "label": "Other Fits"}
         },
        # --=-=-==-=--=--=-=-=-=-=-=-=-=-=-==-=-=-=-==-=-=-==-==-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        {"sim": "LS220_M13641364_M0_LK_SR_restart",
         "model": {"Mej_tot-geo": "model", "vel_inf_ave-geo": "model", "Mdisk3D": "model",  "components":["dyn"],#"sec"],
                   "plot": {"lw": 1., "ls": "--", "color": "red"}},  # "label": r"Model"}},
         "best": {"Mej_tot-geo": "poly2", "vel_inf_ave-geo": "poly22", "Mdisk3D": "poly22",  "components":["dyn"],#"sec"],
                  "plot": {"lw": 1., "ls": "-", "color": "red"}},  # , "label": r"Best Fit"}},
         "worst": {"Mej_tot-geo": "krug19", "vel_inf_ave-geo": None, "Mdisk3D": None,  "components":["dyn"],#"sec"],
                   "plot": {"alpha":0.7, "lw": 0.9, "ls": ":", "color": "red"}},  # , "label": r"Best Fit"}},
         # "fits": [
         #     {"Mej_tot-geo": None, "vel_inf_ave-geo": None, "Mdisk3D": None},
         #     {"Mej_tot-geo": "poly22", "vel_inf_ave-geo": "poly22", "Mdisk3D": "poly22"},
         #     {"Mej_tot-geo": "diet16", "vel_inf_ave-geo": "diet16", "Mdisk3D": "radi18"},
         #     {"Mej_tot-geo": "krug19", "vel_inf_ave-geo": "diet16", "Mdisk3D": "krug19"}],
         # "plotfits": {"color": "red", "alpha": 0.3}  # , "label": "Other Fits"}
         }
    ]

    # pardics = [
    #     {"Mej_tot-geo": "model", "vel_inf_ave-geo": "model", "Mdisk3D": "model",    "plot": {"lw": 2., "ls": "-", "color": "gray", "label": r"$M_{\rm ej}, v_{\rm ej}$ Model"}},
    #     {"Mej_tot-geo": None, "vel_inf_ave-geo": None, "Mdisk3D": None,             "plot": {"lw":2., "ls": "-", "color": "gray", "label": r"$M_{\rm ej}, v_{\rm ej}$ Model"}},
    #     {"Mej_tot-geo": "mean", "vel_inf_ave-geo": "mean", "Mdisk3D": "mean",       "plot": {"lw":1., "ls": ":", "color": "gray", "label": r"$M_{\rm ej}, v_{\rm ej}$ Mean"}},
    #     {"Mej_tot-geo": "poly22", "vel_inf_ave-geo": "poly22", "Mdisk3D": "poly22", "plot": {"lw":1., "ls": "-", "color": "gray", "label": r"$M_{\rm ej} v_{\rm ej}$ $P_2(q,\tilde\Lambda)$"}},
    #     {"Mej_tot-geo": "diet16", "vel_inf_ave-geo": "diet16", "Mdisk3D": "radi18", "plot": {"lw":1., "ls": "--", "color": "gray", "label": r"$M_{\rm ej}$ Eq.(6) $v_{\rm ej}$ Eq.(9)"}},
    #     {"Mej_tot-geo": "krug19", "vel_inf_ave-geo": "diet16", "Mdisk3D": "krug19", "plot": {"lw":1., "ls": "-.", "color": "gray", "label": r"$M_{\rm ej}$ Eq.(7) $v_{\rm ej}$ Eq.(9)"}}
    # ]

    ''' --- collecting data --- '''

    # models = md.groups
    # lineset = {}
    # for band in bands:
    #     for modeldic in task:







    # models = md.groups
    # lineset = {}
    # for band in bands:
    #     lineset[band] = {}
    #     #
    #     for modeldic in task:
    #         lineset[band][modeldic["sim"]] = {}
    #         model = models[models.group == modeldic["sim"]]
    #         ofit = Fit(model)
    #         #
    #         for pardic in pardics:
    #
    #             if pardic[""]
    #
    #             print("\t\t{} {} {}".format(band, modeldic["sim"], pars["plot"]["label"]))
    #
    #             lineset[band][modeldic["sim"]][pars["plot"]["label"]] = {}
    #
    #             mej = ofit.value("Mej_tot-geo",     pars["Mej_tot-geo"])
    #             vej = ofit.value("vel_inf_ave-geo", pars["vel_inf_ave-geo"])
    #             mdisk = ofit.value("Mdisk3D",       pars["Mdisk3D"])
    #             #
    #             vals = {"Mej_tot-geo": mej, "vel_inf_ave-geo": vej, "Mdisk3D": mdisk}
    #             #
    #             o_mkn = COMPUTE_LIGHTCURVE(None, "/data01/numrel/vsevolod.nedora/prj_gw170817/scripts/lightcurves/")
    #             o_mkn.set_glob_par_var_source(False, False)
    #             o_mkn.set_dyn_iso_aniso = "aniso"
    #             o_mkn.set_secular_iso_aniso = "aniso"
    #             o_mkn.set_dyn_par_var("aniso")
    #             o_mkn.set_secular_par_war("aniso")
    #             o_mkn.glob_vars['m_disk']                       = vals["Mdisk3D"]
    #             o_mkn.ejecta_vars['dynamics']["m_ej"]           = vals["Mej_tot-geo"]
    #             o_mkn.ejecta_vars["dynamics"]["central_vel"]    = vals["vel_inf_ave-geo"]
    #             o_mkn.compute_save_lightcurve(write_output=True)
    #             #
    #             load_mkn = COMBINE_LIGHTCURVES(None, "/data01/numrel/vsevolod.nedora/prj_gw170817/scripts/lightcurves/")
    #             times, mags = load_mkn.get_model_median(band)
    #             #
    #             linedic = {}
    #             linedic["x"] = times
    #             linedic["y"] = mags
    #             linedic["color"] = modeldic["plot"]["color"]
    #             linedic["ls"] = pars["plot"]["ls"]
    #             linedic["alpha"] = 1.
    #             linedic["lw"] = pars["plot"]["lw"]
    #
    #             lineset[band][modeldic["sim"]][pars["plot"]["label"]] = linedic

    ''' --- plot settings --- '''

    subplot_dics = [
        {
            "band": bands[0],
            # "add_lines": [dic["plot"] for dic in pardics],
            "add_lines": [
                {"lw": 1., "ls": "-", "color": "red", "label": r"LS220 q=1.00 (SR)"},
                {"lw": 1., "ls": "-", "color": "green", "label": r"BLh q=1.00 (SR)"},
                {"lw": 1., "ls": "-", "color": "blue", "label": r"DD2 q=1.00 (SR)"},#r"BLh q=1.43 (SR)"},
            ],
            "legend": {"fancybox": False, "loc": 'upper left',
                       # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                       "shadow": "False", "ncol": 1, "fontsize": 12, "columnspacing": 0.4,
                       "framealpha": 0., "borderaxespad": 0., "frameon": False},
            "title": "#band",
            'xmin': 1e-1, 'xmax': 2e0, 'xlabel': r"time [days]",

        },
        {
            "band": bands[1],
            #"add_lines": [dic["plot"] for dic in pardics],
            "title": "#band",
            'xmin': 1e-1, 'xmax': 5e0, 'xlabel': r"time [days]",
        },
        {
            "band": bands[2],
            # "lines": lineset[bands[2]],
            "add_lines": [
                {"lw": 1., "ls": "--", "color": "gray", "label": r"Model"},
                {"lw": 1., "ls": "-", "color": "gray", "label": r"Best Fit"},
                {"lw": 0.9, "ls": ":", "color": "gray", "label": r"Worst Fit"}
            ],
            "legend": {"fancybox": False, "loc": 'upper right',
                       # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                       "shadow": "False", "ncol": 1, "fontsize": 13, "columnspacing": 0.4,
                       "framealpha": 0., "borderaxespad": 0., "frameon": False},
            # "add_bands": [
            #     {"color": "gray", "alpha": 0.4, "label": "Other Fits"}
            # ],
            "title": "#band",
            'xmin': 3e-1, 'xmax': 1e1, 'xlabel': r"time [days]",
        }
    ]

    plot_dic = {
        "task_v_n" : "theta",
        "type": "long",
        "figsize": (6.4, 5.5),
        'xmin': 3e-1, 'xmax': 3e1, 'xlabel': r"time [days]",
        'ymin': 21, 'ymax': 18.5, 'ylabel': r"AB magnitude at 40 Mpc", # 23, 18 # sec: 22 17
        # "text": {'x': 0.25, 'y': 0.95, 's': r"Ks band", 'fontsize': 14, 'color': 'black',
        #          'horizontalalignment': 'center'},
        "xscale": "log", "yscale": "linear",
        "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False, "tick1On": True, "tick2On": True,
                        "labelsize": 14,"direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        "figname": __outplotdir__ + "dyn_fit_lines_3eos.png",
        'fontsize': 13,
        'labelsize': 13,
        "savepdf": True,
        "title": "#band",
        # "add_lines": [
        #     [dic["plot"] for dic in pardics]
        #     # {"ls": ":", "color": "gray", "label": r"$M_{\rm ej}, v_{\rm ej}$ Mean"},
        #     # {"ls": "-", "color": "gray", "label": "SFho q=1.00 (SR)"},
        #     # {"ls": "-", "color": "gray", "label": "BLh q=1.43 (SR)"},
        #     # {"ls": "-", "color": "gray", "label": "BLh q=1.43 (SR)"}
        # ],
    }

    plot_custom_lightcurves(plot_dic, subplot_dics, tasks)
def task_plot_fromfit_model_mkn_multiband_band2():

    bands = ["g", "z", "Ks"]

    ''' Best : mej_krug | vej_poly22 | yeej_poly22 | mdisk_poly22 '''
    ''' Worst : "mean" , "mean", "mean", "mean"'''

    tasks = [
        {"obs":
             {"method":"peak marker",
                "plot":{"label":"Obs", 'marker':'d', 'facecolors':"none", 'edgecolors':'gray'}}},

        {"sim": "BLh_M13641364_M0_LK_SR",
         "model": {"Mej_tot-geo": "model", "vel_inf_ave-geo": "model", "theta_rms-geo":"model", "Mdisk3D": "model", "components":["dyn", "sec"],#"sec"],
                   "plot": {"lw": 1., "ls": "--", "color": "green"}},# "label": r"Model"}},
         "best":  {"Mej_tot-geo": "krug19", "vel_inf_ave-geo": "poly22", "theta_rms-geo":"poly22", "Mdisk3D": "poly22",  "components":["dyn", "sec"],#"sec"],
                  "plot": {"lw": 1., "ls": "-", "color": "green"}},#, "label": r"Best Fit"}},
         "worst": {"Mej_tot-geo": "mean", "vel_inf_ave-geo": "mean",  "theta_rms-geo":"mean", "Mdisk3D": "mean",  "components":["dyn", "sec"],#"sec"],
                   "plot": {"alpha":0.7, "lw": 0.9, "ls": ":", "color": "green"}},  # , "label": r"Best Fit"}},
         # "fits": [
         #     {"Mej_tot-geo": None, "vel_inf_ave-geo": None, "Mdisk3D": None},
         #     {"Mej_tot-geo": "poly22", "vel_inf_ave-geo": "poly22", "Mdisk3D": "poly22"},
         #     {"Mej_tot-geo": "diet16", "vel_inf_ave-geo": "diet16", "Mdisk3D": "radi18"},
         #     {"Mej_tot-geo": "krug19", "vel_inf_ave-geo": "diet16", "Mdisk3D": "krug19"}],
         # "plotfits":{"color":"green", "alpha": 0.3}#, "label": "Other Fits"}
         },
        # --=-=-==-=--=--=-=-=-=-=-=-=-=-=-==-=-=-=-==-=-=-==-==-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        {"sim": "BLh_M11461635_M0_LK_SR",#"DD2_M13641364_M0_LK_SR_R04",# "BLh_M11461635_M0_LK_SR",
         "model": {"Mej_tot-geo": "model", "vel_inf_ave-geo": "model", "theta_rms-geo":"model", "Mdisk3D": "model",  "components":["dyn", "sec"],#"sec"],
                   "plot": {"lw": 1., "ls": "--", "color": "blue"}},  # "label": r"Model"}},
         "best": {"Mej_tot-geo": "poly2", "vel_inf_ave-geo": "poly22", "theta_rms-geo":"poly22", "Mdisk3D": "poly22",  "components":["dyn", "sec"],#"sec"],
                  "plot": {"lw": 1., "ls": "-", "color": "blue"}},  # , "label": r"Best Fit"}},
         "worst": {"Mej_tot-geo": "mean", "vel_inf_ave-geo": "mean",  "theta_rms-geo":"mean", "Mdisk3D": "mean",  "components":["dyn", "sec"],#"sec"],
                  "plot": {"alpha":0.7, "lw": 0.9, "ls": ":", "color": "blue"}},  # , "label": r"Best Fit"}},
         # "fits": [
         #     {"Mej_tot-geo": None, "vel_inf_ave-geo": None, "Mdisk3D": None},
         #     {"Mej_tot-geo": "poly22", "vel_inf_ave-geo": "poly22", "Mdisk3D": "poly22"},
         #     {"Mej_tot-geo": "diet16", "vel_inf_ave-geo": "diet16", "Mdisk3D": "radi18"},
         #     {"Mej_tot-geo": "krug19", "vel_inf_ave-geo": "diet16", "Mdisk3D": "krug19"}],
         # "plotfits": {"color": "blue", "alpha": 0.3}  # , "label": "Other Fits"}
         },
        # --=-=-==-=--=--=-=-=-=-=-=-=-=-=-==-=-=-=-==-=-=-==-==-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        {"sim": "LS220_M13641364_M0_LK_SR_restart",
         "model": {"Mej_tot-geo": "model", "vel_inf_ave-geo": "model", "theta_rms-geo":"model", "Mdisk3D": "model",  "components":["dyn", "sec"],#"sec"],
                   "plot": {"lw": 1., "ls": "--", "color": "red"}},  # "label": r"Model"}},
         "best": {"Mej_tot-geo": "poly2", "vel_inf_ave-geo": "poly22", "theta_rms-geo":"poly22", "Mdisk3D": "poly22",  "components":["dyn", "sec"],#"sec"],
                  "plot": {"lw": 1., "ls": "-", "color": "red"}},  # , "label": r"Best Fit"}},
         "worst": {"Mej_tot-geo": "mean", "vel_inf_ave-geo": "mean",  "theta_rms-geo":"mean", "Mdisk3D": "mean",  "components":["dyn", "sec"],#"sec"],
                   "plot": {"alpha":0.7, "lw": 0.9, "ls": ":", "color": "red"}},  # , "label": r"Best Fit"}},
         # "fits": [
         #     {"Mej_tot-geo": None, "vel_inf_ave-geo": None, "Mdisk3D": None},
         #     {"Mej_tot-geo": "poly22", "vel_inf_ave-geo": "poly22", "Mdisk3D": "poly22"},
         #     {"Mej_tot-geo": "diet16", "vel_inf_ave-geo": "diet16", "Mdisk3D": "radi18"},
         #     {"Mej_tot-geo": "krug19", "vel_inf_ave-geo": "diet16", "Mdisk3D": "krug19"}],
         # "plotfits": {"color": "red", "alpha": 0.3}  # , "label": "Other Fits"}
         }
    ]

    # pardics = [
    #     {"Mej_tot-geo": "model", "vel_inf_ave-geo": "model", "Mdisk3D": "model",    "plot": {"lw": 2., "ls": "-", "color": "gray", "label": r"$M_{\rm ej}, v_{\rm ej}$ Model"}},
    #     {"Mej_tot-geo": None, "vel_inf_ave-geo": None, "Mdisk3D": None,             "plot": {"lw":2., "ls": "-", "color": "gray", "label": r"$M_{\rm ej}, v_{\rm ej}$ Model"}},
    #     {"Mej_tot-geo": "mean", "vel_inf_ave-geo": "mean", "Mdisk3D": "mean",       "plot": {"lw":1., "ls": ":", "color": "gray", "label": r"$M_{\rm ej}, v_{\rm ej}$ Mean"}},
    #     {"Mej_tot-geo": "poly22", "vel_inf_ave-geo": "poly22", "Mdisk3D": "poly22", "plot": {"lw":1., "ls": "-", "color": "gray", "label": r"$M_{\rm ej} v_{\rm ej}$ $P_2(q,\tilde\Lambda)$"}},
    #     {"Mej_tot-geo": "diet16", "vel_inf_ave-geo": "diet16", "Mdisk3D": "radi18", "plot": {"lw":1., "ls": "--", "color": "gray", "label": r"$M_{\rm ej}$ Eq.(6) $v_{\rm ej}$ Eq.(9)"}},
    #     {"Mej_tot-geo": "krug19", "vel_inf_ave-geo": "diet16", "Mdisk3D": "krug19", "plot": {"lw":1., "ls": "-.", "color": "gray", "label": r"$M_{\rm ej}$ Eq.(7) $v_{\rm ej}$ Eq.(9)"}}
    # ]

    ''' --- collecting data --- '''

    # models = md.groups
    # lineset = {}
    # for band in bands:
    #     for modeldic in task:







    # models = md.groups
    # lineset = {}
    # for band in bands:
    #     lineset[band] = {}
    #     #
    #     for modeldic in task:
    #         lineset[band][modeldic["sim"]] = {}
    #         model = models[models.group == modeldic["sim"]]
    #         ofit = Fit(model)
    #         #
    #         for pardic in pardics:
    #
    #             if pardic[""]
    #
    #             print("\t\t{} {} {}".format(band, modeldic["sim"], pars["plot"]["label"]))
    #
    #             lineset[band][modeldic["sim"]][pars["plot"]["label"]] = {}
    #
    #             mej = ofit.value("Mej_tot-geo",     pars["Mej_tot-geo"])
    #             vej = ofit.value("vel_inf_ave-geo", pars["vel_inf_ave-geo"])
    #             mdisk = ofit.value("Mdisk3D",       pars["Mdisk3D"])
    #             #
    #             vals = {"Mej_tot-geo": mej, "vel_inf_ave-geo": vej, "Mdisk3D": mdisk}
    #             #
    #             o_mkn = COMPUTE_LIGHTCURVE(None, "/data01/numrel/vsevolod.nedora/prj_gw170817/scripts/lightcurves/")
    #             o_mkn.set_glob_par_var_source(False, False)
    #             o_mkn.set_dyn_iso_aniso = "aniso"
    #             o_mkn.set_secular_iso_aniso = "aniso"
    #             o_mkn.set_dyn_par_var("aniso")
    #             o_mkn.set_secular_par_war("aniso")
    #             o_mkn.glob_vars['m_disk']                       = vals["Mdisk3D"]
    #             o_mkn.ejecta_vars['dynamics']["m_ej"]           = vals["Mej_tot-geo"]
    #             o_mkn.ejecta_vars["dynamics"]["central_vel"]    = vals["vel_inf_ave-geo"]
    #             o_mkn.compute_save_lightcurve(write_output=True)
    #             #
    #             load_mkn = COMBINE_LIGHTCURVES(None, "/data01/numrel/vsevolod.nedora/prj_gw170817/scripts/lightcurves/")
    #             times, mags = load_mkn.get_model_median(band)
    #             #
    #             linedic = {}
    #             linedic["x"] = times
    #             linedic["y"] = mags
    #             linedic["color"] = modeldic["plot"]["color"]
    #             linedic["ls"] = pars["plot"]["ls"]
    #             linedic["alpha"] = 1.
    #             linedic["lw"] = pars["plot"]["lw"]
    #
    #             lineset[band][modeldic["sim"]][pars["plot"]["label"]] = linedic

    ''' --- plot settings --- '''

    subplot_dics = [
        {
            "band": bands[0],
            # "add_lines": [dic["plot"] for dic in pardics],
            "add_lines": [
                {"lw": 1., "ls": "-", "color": "red", "label": r"LS220 q=1.00 (SR)"},
                {"lw": 1., "ls": "-", "color": "green", "label": r"BLh q=1.00 (SR)"},
                {"lw": 1., "ls": "-", "color": "blue", "label": "BLh q=1.43 (SR)"}#r"DD2 q=1.00 (SR)"},#r"BLh q=1.43 (SR)"},
            ],
            "add_scatter":[
                {"marker":'o',"s":25, "edgecolor":"gray","facecolor":"none", "label":"AT2017gfo"}
            ],
            "legend": {"fancybox": False, "loc": 'center left',
                       # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                       "shadow": "False", "ncol": 1, "fontsize": 12, "columnspacing": 0.4,
                       "framealpha": 0., "borderaxespad": 0., "frameon": False},
            "title": "#band",
            'xmin': 1e-1, 'xmax': 2e0, 'xlabel': r"time [days]",

        },
        {
            "band": bands[1],
            #"add_lines": [dic["plot"] for dic in pardics],
            "title": "#band",
            "add_lines": [
                {"lw": 1., "ls": "--", "color": "gray", "label": r"Model"},
                {"lw": 1., "ls": "-", "color": "gray", "label": r"Best Fit"},
                {"lw": 0.9, "ls": ":", "color": "gray", "label": r"Worst Fit"}
            ],
            "legend": {"fancybox": False, "loc": 'lower center',
                       # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                       "shadow": "False", "ncol": 1, "fontsize": 13, "columnspacing": 0.4,
                       "framealpha": 0., "borderaxespad": 0., "frameon": False},
            'xmin': 1e-1, 'xmax': 5e0, 'xlabel': r"time [days]",
        },
        {
            "band": bands[2],
            # "lines": lineset[bands[2]],
            # "add_bands": [
            #     {"color": "gray", "alpha": 0.4, "label": "Other Fits"}
            # ],
            "title": "#band",
            'xmin': 3e-1, 'xmax': 2e1, 'xlabel': r"time [days]",
        }
    ]

    plot_dic = {
        "task_v_n" : "theta",
        "type": "long",
        "figsize": (6.4, 5.5),
        'xmin': 3e-1, 'xmax': 3e1, 'xlabel': r"time [days]",
        'ymin': 19.5, 'ymax': 17., 'ylabel': r"AB magnitude at 40 Mpc", # 23, 18 # sec: 22 17
        # "text": {'x': 0.25, 'y': 0.95, 's': r"Ks band", 'fontsize': 14, 'color': 'black',
        #          'horizontalalignment': 'center'},
        "xscale": "log", "yscale": "linear",
        "tick_params": {"axis": 'both', "which": 'both', "labelleft": True,
                        "labelright": False, "tick1On": True, "tick2On": True,
                        "labelsize": 14,"direction": 'in',
                        "bottom": True, "top": True, "left": True, "right": True},
        "figname": __outplotdir__ + "dyn_sec_fit_lines_3eos.png",
        'fontsize': 13,
        'labelsize': 13,
        "savepdf": True,
        "title": "#band",
        # "add_lines": [
        #     [dic["plot"] for dic in pardics]
        #     # {"ls": ":", "color": "gray", "label": r"$M_{\rm ej}, v_{\rm ej}$ Mean"},
        #     # {"ls": "-", "color": "gray", "label": "SFho q=1.00 (SR)"},
        #     # {"ls": "-", "color": "gray", "label": "BLh q=1.43 (SR)"},
        #     # {"ls": "-", "color": "gray", "label": "BLh q=1.43 (SR)"}
        # ],
    }

    plot_custom_lightcurves(plot_dic, subplot_dics, tasks)

def task_plot_lightcurve_synthetic_model():



    line_dics = []
    ''' --- mode1 --- '''

    # model
    sim = "DD2_M13641364_M0_LK_R04"
    models = md.groups


    #models.drop(params.blacklist, axis=0)
    # mdoels = models.set_index("group")
    #print(models); exit(0)
    model = models[models.group == sim]
    print(model)
    #model = md.groups[md.groups.index == sim]

    # fitted vals
    mej_mean = 5.220e-03 # Msun
    mej_poly22 = float(Fitting_Functions.poly_2_qLambda([2.549, 2.394, -3.005e-02, -3.376e+00, 0.038, -1.149e-05], model)) / 1e3
    mej_diet = float(Fitting_Functions.mej_dietrich16([-1.234, 3.089, -31.801, 17.526, -3.146], model)) / 1e3
    mej_krug = float(Fitting_Functions.mej_kruger20([-0.981, 12.880, -35.148, 2.030], model)) / 1e3

    vej_mean = 0.189 # \pm 0.049
    vej_poly22 = float(Fitting_Functions.poly_2_qLambda([0.182, 0.159, -1.509e-04, -1.046e-01, 9.233e-05, -1.581e-08], model))
    vej_diet = float(Fitting_Functions.vej_dietrich16([-0.422, 0.834, -1.510], model))

    yeej_mean = 0.167 # \pm 0.057
    yeej_poly22 = float(Fitting_Functions.poly_2_qLambda([-4.555e-01, 0.793, 7.509e-04, -3.139e-01, -1.899e-04, -4.460e-07], model))
    yeej_our = float(Fitting_Functions.yeej_like_vej([0.177, 0.452, -4.611], model))

    mdisk_mean = 0.122 # \pm 0.088
    mdisk_poly22 = float(Fitting_Functions.poly_2_qLambda([-8.951e-01, 1.195, 4.292e-04, -3.991e-01, 4.778e-05, -2.266e-07], model))
    mdisk_radi = float(Fitting_Functions.mdisk_radice18([0.070, 0.101, 305.009, 189.952], model))
    mdisk_krug = float(Fitting_Functions.mdisk_kruger20([-0.013, 1.000, 1325.652], model))

    print("\t--------------------")
    print("\t model: {}".format(sim))
    print("\t--------------------")
    print("\t Mej Model                        " + "{:.1f}".format(float(model["Mej_tot-geo"]) * 1e3))
    print("\t Mej Mean                         " + "{:.1f}".format(mej_mean * 1e3))
    print("\t Mej Eq.1~\cite{Dietrich:2016fpt} "  + "{:.1f}".format(mej_diet * 1e3) + " [10^{3} M_{\odot}]")
    print("\t Mej Eq.6~\cite{Kruger:2020gig}   " + "{:.1f}".format(mej_krug * 1e3) + " [10^{3} M_{\odot}]")
    print("\t Mej Eq.P22                       " + "{:.1f}".format(mej_poly22 * 1e3) + " [10^{3} M_{\odot}]")
    print("\t-------------------")
    print("\t vej Model                        " + "{:.3f}".format(float(model["vel_inf_ave-geo"])) + " [c]")
    print("\t vej Mean                         " + "{:.3f}".format(vej_mean) + " [c]")
    print("\t Mej Eq.5~\cite{Dietrich:2016fpt} " + "{:.1f}".format(vej_diet) + " [c]")
    print("\t Mej Eq.P22                       " + "{:.1f}".format(vej_poly22) + " [c]")
    print("\t-------------------")
    print("\t Ye Model                         " + "{:.3f}".format(float(model["Ye_ave-geo"])))
    print("\t Ye Mean                          " + "{:.3f}".format(yeej_mean))
    print("\t Mej Eq.~OUR                      " + "{:.1f}".format(yeej_our))
    print("\t Mej Eq.P22                       " + "{:.1f}".format(yeej_poly22))
    print("\t-------------------")
    print("\t Mdisk Model                      " + "{:.3f}".format(float(model["Mdisk3D"])) + " [M_{\odot}]")
    print("\t Mdisk mean                       " + "{:.3f}".format(mdisk_mean) + " [M_{\odot}]")
    print("\t Mej Eq.25~\cite{Radice:2018pdn}  " + "{:.3f}".format(mdisk_radi) + " [M_{\odot}]")
    print("\t Mej Eq.4~\cite{Kruger:2020gig}   " + "{:.3f}".format(mdisk_krug) + " [M_{\odot}]")
    print("\t Mej Eq.P22                       " + "{:.1f}".format(mdisk_poly22) + " [M_{\odot}]")

    #exit(1)

    # define MKN parameters

    pars = {"mej": mej_mean, "vej": vej_mean, "mdisk": mdisk_mean}
    times, mags = get_times_mags_from_pars(pars, "Ks")
    line_dics.append({"x": times, "y": mags, "color": "blue", "ls": '-', "lw": 0.8, "label": r"$M_{\rm ej}$ Mean $v_{\rm ej}$ Mean"})

    pars = {"mej": mej_poly22, "vej": vej_poly22, "mdisk": mdisk_poly22}
    times, mags = get_times_mags_from_pars(pars, "Ks")
    line_dics.append({"x": times, "y": mags, "color": "blue", "ls": '--', "lw": 0.8, "label": r"$M_{\rm ej}$ P22 $v_{\rm ej}$ P22"})

    pars = {"mej": mej_diet, "vej": vej_diet, "mdisk": mdisk_radi}
    times, mags = get_times_mags_from_pars(pars, "Ks")
    line_dics.append({"x": times, "y": mags, "color": "blue", "ls": '-.', "lw": 0.8, "label": r"$M_{\rm ej}$ D+16 $v_{\rm ej}$ D+16"})

    pars = {"mej": mej_krug, "vej": vej_poly22, "mdisk": mdisk_krug}
    times, mags = get_times_mags_from_pars(pars, "Ks")
    line_dics.append({"x": times, "y": mags, "color": "blue", "ls": ':', "lw": 0.8, "label": r"$M_{\rm ej}$ K+19 $v_{\rm ej}$ P22"})


    ''' --- mode2 --- '''

    sim = "SFHo_M13641364_M0_LK_p2019"
    models = md.groups

    model = models[models.group == sim]
    print(model)

    mej_mean = 5.220e-03 # Msun
    mej_poly22 = float(Fitting_Functions.poly_2_qLambda([2.549, 2.394, -3.005e-02, -3.376e+00, 0.038, -1.149e-05], model)) / 1e3
    mej_diet = float(Fitting_Functions.mej_dietrich16([-1.234, 3.089, -31.801, 17.526, -3.146], model)) / 1e3
    mej_krug = float(Fitting_Functions.mej_kruger20([-0.981, 12.880, -35.148, 2.030], model)) / 1e3

    vej_mean = 0.189 # \pm 0.049
    vej_poly22 = float(Fitting_Functions.poly_2_qLambda([0.182, 0.159, -1.509e-04, -1.046e-01, 9.233e-05, -1.581e-08], model))
    vej_diet = float(Fitting_Functions.vej_dietrich16([-0.422, 0.834, -1.510], model))

    yeej_mean = 0.167 # \pm 0.057
    yeej_poly22 = float(Fitting_Functions.poly_2_qLambda([-4.555e-01, 0.793, 7.509e-04, -3.139e-01, -1.899e-04, -4.460e-07], model))
    yeej_our = float(Fitting_Functions.yeej_like_vej([0.177, 0.452, -4.611], model))

    mdisk_mean = 0.122 # \pm 0.088
    mdisk_poly22 = float(Fitting_Functions.poly_2_qLambda([-8.951e-01, 1.195, 4.292e-04, -3.991e-01, 4.778e-05, -2.266e-07], model))
    mdisk_radi = float(Fitting_Functions.mdisk_radice18([0.070, 0.101, 305.009, 189.952], model))
    mdisk_krug = float(Fitting_Functions.mdisk_kruger20([-0.013, 1.000, 1325.652], model))

    print("\t--------------------")
    print("\t model: {}".format(sim))
    print("\t--------------------")
    print("\t Mej Model                        " + "{:.1f}".format(float(model["Mej_tot-geo"]) * 1e3))
    print("\t Mej Mean                         " + "{:.1f}".format(mej_mean * 1e3))
    print("\t Mej Eq.1~\cite{Dietrich:2016fpt} "  + "{:.1f}".format(mej_diet * 1e3) + " [10^{3} M_{\odot}]")
    print("\t Mej Eq.6~\cite{Kruger:2020gig}   " + "{:.1f}".format(mej_krug * 1e3) + " [10^{3} M_{\odot}]")
    print("\t Mej Eq.P22                       " + "{:.1f}".format(mej_poly22 * 1e3) + " [10^{3} M_{\odot}]")
    print("\t-------------------")
    print("\t vej Model                        " + "{:.3f}".format(float(model["vel_inf_ave-geo"])) + " [c]")
    print("\t vej Mean                         " + "{:.3f}".format(vej_mean) + " [c]")
    print("\t Mej Eq.5~\cite{Dietrich:2016fpt} " + "{:.1f}".format(vej_diet) + " [c]")
    print("\t Mej Eq.P22                       " + "{:.1f}".format(vej_poly22) + " [c]")
    print("\t-------------------")
    print("\t Ye Model                         " + "{:.3f}".format(float(model["Ye_ave-geo"])))
    print("\t Ye Mean                          " + "{:.3f}".format(yeej_mean))
    print("\t Mej Eq.~OUR                      " + "{:.1f}".format(yeej_our))
    print("\t Mej Eq.P22                       " + "{:.1f}".format(yeej_poly22))
    print("\t-------------------")
    print("\t Mdisk Model                      " + "{:.3f}".format(float(model["Mdisk3D"])) + " [M_{\odot}]")
    print("\t Mdisk mean                       " + "{:.3f}".format(mdisk_mean) + " [M_{\odot}]")
    print("\t Mej Eq.25~\cite{Radice:2018pdn}  " + "{:.3f}".format(mdisk_radi) + " [M_{\odot}]")
    print("\t Mej Eq.4~\cite{Kruger:2020gig}   " + "{:.3f}".format(mdisk_krug) + " [M_{\odot}]")
    print("\t Mej Eq.P22                       " + "{:.1f}".format(mdisk_poly22) + " [M_{\odot}]")

    #exit(1)


    # define MKN parameters

    pars = {"mej": mej_mean, "vej": vej_mean, "mdisk": mdisk_mean}
    times, mags = get_times_mags_from_pars(pars, "Ks")
    line_dics.append({"x": times, "y": mags, "color": "red", "ls": '-', "lw": 0.8})#, "label": r"$M_{\rm ej}$ Mean $v_{\rm rj}$ Mean"})

    pars = {"mej": mej_poly22, "vej": vej_poly22, "mdisk": mdisk_poly22}
    times, mags = get_times_mags_from_pars(pars, "Ks")
    line_dics.append({"x": times, "y": mags, "color": "red", "ls": '--', "lw": 0.8})#, "label": r"$M_{\rm ej}$ P22 $v_{\rm rj}$ P22"})

    pars = {"mej": mej_diet, "vej": vej_diet, "mdisk": mdisk_radi}
    times, mags = get_times_mags_from_pars(pars, "Ks")
    line_dics.append({"x": times, "y": mags, "color": "red", "ls": '-.', "lw": 0.8})#, "label": r"$M_{\rm ej}$ D+16 $v_{\rm rj}$ D+16"})

    pars = {"mej": mej_krug, "vej": vej_poly22, "mdisk": mdisk_krug}
    times, mags = get_times_mags_from_pars(pars, "Ks")
    line_dics.append({"x": times, "y": mags, "color": "red", "ls": ':', "lw": 0.8})#, "label": r"$M_{\rm ej}$ K+19 $v_{\rm rj}$ P22"})

    ''' --- mode3 --- '''

    sim = "BLh_M11461635_M0_LK"
    models = md.groups

    model = models[models.group == sim]
    print(model)

    mej_mean = 5.220e-03  # Msun
    mej_poly22 = float(
        Fitting_Functions.poly_2_qLambda([2.549, 2.394, -3.005e-02, -3.376e+00, 0.038, -1.149e-05], model)) / 1e3
    mej_diet = float(Fitting_Functions.mej_dietrich16([-1.234, 3.089, -31.801, 17.526, -3.146], model)) / 1e3
    mej_krug = float(Fitting_Functions.mej_kruger20([-0.981, 12.880, -35.148, 2.030], model)) / 1e3

    vej_mean = 0.189  # \pm 0.049
    vej_poly22 = float(
        Fitting_Functions.poly_2_qLambda([0.182, 0.159, -1.509e-04, -1.046e-01, 9.233e-05, -1.581e-08], model))
    vej_diet = float(Fitting_Functions.vej_dietrich16([-0.422, 0.834, -1.510], model))

    yeej_mean = 0.167  # \pm 0.057
    yeej_poly22 = float(
        Fitting_Functions.poly_2_qLambda([-4.555e-01, 0.793, 7.509e-04, -3.139e-01, -1.899e-04, -4.460e-07], model))
    yeej_our = float(Fitting_Functions.yeej_like_vej([0.177, 0.452, -4.611], model))

    mdisk_mean = 0.122  # \pm 0.088
    mdisk_poly22 = float(
        Fitting_Functions.poly_2_qLambda([-8.951e-01, 1.195, 4.292e-04, -3.991e-01, 4.778e-05, -2.266e-07], model))
    mdisk_radi = float(Fitting_Functions.mdisk_radice18([0.070, 0.101, 305.009, 189.952], model))
    mdisk_krug = float(Fitting_Functions.mdisk_kruger20([-0.013, 1.000, 1325.652], model))

    print("\t--------------------")
    print("\t model: {}".format(sim))
    print("\t--------------------")
    print("\t Mej Model                        " + "{:.1f}".format(float(model["Mej_tot-geo"]) * 1e3))
    print("\t Mej Mean                         " + "{:.1f}".format(mej_mean * 1e3))
    print("\t Mej Eq.1~\cite{Dietrich:2016fpt} " + "{:.1f}".format(mej_diet * 1e3) + " [10^{3} M_{\odot}]")
    print("\t Mej Eq.6~\cite{Kruger:2020gig}   " + "{:.1f}".format(mej_krug * 1e3) + " [10^{3} M_{\odot}]")
    print("\t Mej Eq.P22                       " + "{:.1f}".format(mej_poly22 * 1e3) + " [10^{3} M_{\odot}]")
    print("\t-------------------")
    print("\t vej Model                        " + "{:.3f}".format(float(model["vel_inf_ave-geo"])) + " [c]")
    print("\t vej Mean                         " + "{:.3f}".format(vej_mean) + " [c]")
    print("\t Mej Eq.5~\cite{Dietrich:2016fpt} " + "{:.1f}".format(vej_diet) + " [c]")
    print("\t Mej Eq.P22                       " + "{:.1f}".format(vej_poly22) + " [c]")
    print("\t-------------------")
    print("\t Ye Model                         " + "{:.3f}".format(float(model["Ye_ave-geo"])))
    print("\t Ye Mean                          " + "{:.3f}".format(yeej_mean))
    print("\t Mej Eq.~OUR                      " + "{:.1f}".format(yeej_our))
    print("\t Mej Eq.P22                       " + "{:.1f}".format(yeej_poly22))
    print("\t-------------------")
    print("\t Mdisk Model                      " + "{:.3f}".format(float(model["Mdisk3D"])) + " [M_{\odot}]")
    print("\t Mdisk mean                       " + "{:.3f}".format(mdisk_mean) + " [M_{\odot}]")
    print("\t Mej Eq.25~\cite{Radice:2018pdn}  " + "{:.3f}".format(mdisk_radi) + " [M_{\odot}]")
    print("\t Mej Eq.4~\cite{Kruger:2020gig}   " + "{:.3f}".format(mdisk_krug) + " [M_{\odot}]")
    print("\t Mej Eq.P22                       " + "{:.1f}".format(mdisk_poly22) + " [M_{\odot}]")

    # exit(1)

    # define MKN parameters

    pars = {"mej": mej_mean, "vej": vej_mean, "mdisk": mdisk_mean}
    times, mags = get_times_mags_from_pars(pars, "Ks")
    line_dics.append({"x": times, "y": mags, "color": "green", "ls": '-',
                      "lw": 0.8})  # , "label": r"$M_{\rm ej}$ Mean $v_{\rm rj}$ Mean"})

    pars = {"mej": mej_poly22, "vej": vej_poly22, "mdisk": mdisk_poly22}
    times, mags = get_times_mags_from_pars(pars, "Ks")
    line_dics.append({"x": times, "y": mags, "color": "green", "ls": '--',
                      "lw": 0.8})  # , "label": r"$M_{\rm ej}$ P22 $v_{\rm rj}$ P22"})

    pars = {"mej": mej_diet, "vej": vej_diet, "mdisk": mdisk_radi}
    times, mags = get_times_mags_from_pars(pars, "Ks")
    line_dics.append({"x": times, "y": mags, "color": "green", "ls": '-.',
                      "lw": 0.8})  # , "label": r"$M_{\rm ej}$ D+16 $v_{\rm rj}$ D+16"})

    pars = {"mej": mej_krug, "vej": vej_poly22, "mdisk": mdisk_krug}
    times, mags = get_times_mags_from_pars(pars, "Ks")
    line_dics.append({"x": times, "y": mags, "color": "green", "ls": ':',
                      "lw": 0.8})  # , "label": r"$M_{\rm ej}$ K+19 $v_{\rm rj}$ P22"})

    plot_dic = {
        # "subplots":{"figsize": (6.0, 6.0), "ncols":1,"nrows":3, "sharex":True,"sharey":False},
        "figsize": (6.0, 5.5),
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
                   "shadow": "False", "ncol": 1, "fontsize": 13, "columnspacing": 0.4,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "text": {'x': 0.25, 'y': 0.95, 's': r"Ks band", 'fontsize': 14, 'color': 'black',
                 'horizontalalignment': 'center'},
        # "add_legend"
        "add_lines":[
            {"ls" : "-", "color": "blue", "label": "DD2 q=1.00 (SR)"},
            {"ls" : "-", "color": "red", "label": "SFho q=1.00 (SR)"},
            {"ls":  "-", "color": "green", "label": "BLh q=1.43 (SR)"},
        ],
        "add_legend": {"fancybox": False, "loc": 'lower left',
                   # "bbox_to_anchor": (0.5, 1.2),  # loc=(0.0, 0.6),  # (1.0, 0.3), # <-> |
                   "shadow": "False", "ncol": 1, "fontsize": 13, "columnspacing": 0.4,
                   "framealpha": 0., "borderaxespad": 0., "frameon": False},
        "savepdf": True,
        "figname": __outplotdir__ + "Ks_dyn_fit.png",
        "dpi": 128,
        "tight_layout": True
    }

    plot_lighcurves(plot_dic, line_dics)


if __name__ == "__main__":
    # print_tex_table_fit_for_models_Mej()
    # print_tex_table_fit_for_models_vej()
    # print_tex_table_fit_for_models_yeej()
    # print_tex_table_fit_for_models_Mdisk()

    # task_plot_fromfit_model_mkn_multiband_band()
    task_plot_fromfit_model_mkn_multiband_band2()

    # task_plot_fromfit_model_mkn_multiband()
    # print_tex_table_fit_for_models_Mdisk()

    exit(1)

    o_mkn = COMPUTE_LIGHTCURVE(None, "/data01/numrel/vsevolod.nedora/prj_gw170817/scripts/lightcurves/")
    o_mkn.set_glob_par_var_source(False, False)
    o_mkn.set_dyn_iso_aniso = "aniso"
    o_mkn.set_secular_iso_aniso = "aniso"
    o_mkn.set_dyn_par_var("aniso")
    o_mkn.set_secular_par_war("aniso")
    # o_mkn.print_latex_table_of_glob_pars()
    # o_mkn.print_latex_table_of_ejecta_pars(["dynamics"])
    o_mkn.print_latex_table_of_ejecta_pars(["secular"])
    #o_mkn.compute_save_lightcurve(write_output=True)


    #task_plot_lightcurve_synthetic_model()

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

    #fitting_function_predict()