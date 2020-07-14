from __future__ import division
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import numpy as np
import re
import sys
from scivis import units as ut
import copy
import csv
from scipy import interpolate
import scipy.optimize as opt # for least square method

__outplotdir__ = "/data01/numrel/vsevolod.nedora/figs/all3/tov_sequences/"

class Paths:
    to_tovs = "../Data/TOVs/"

M_of_M0 = {} # function
# seq = {}

def load_tov_get_M_of_M0(eos):
    rhoc, M, M0 = np.loadtxt(Paths.to_tovs+"{}_sequence.txt".format(eos),
                             usecols=(0,1,2), unpack=True)
    imax = rhoc.argmax()
    M = M[:imax]
    M0 = M0[:imax]

    M_of_M0[eos] = interpolate.interp1d(M0, M, bounds_error=False)

# def plot_tov(eos):
#
#
#
#     fig, ax = plt.subplots()
#     ax.plot(seq[eos].rho_c, seq[eos].M[0], label=eos,
#                 color="black")
#
#     # ax.set_xlim(9, 19)
#     ax.set_ylim(0, 3)
#
#     ax.set_xlabel(r"$\rho_c\ [10^{15}\, {\rm g}\, {\rm cm}^{-3}]$")
#     ax.set_ylabel(r"$M\ [M_\odot]$")
#
#     ax.legend(ncol=2, loc="lower right")
#
#
#     print(__outplotdir__ + "tov_rhoc_{}.png".format(eos.lower()))
#     plt.savefig(__outplotdir__ + "tov_rhoc_{}.png".format(eos.lower()), dpi=256)
#     plt.close()

if __name__ == "__main__":
    # plot_tov()
    pass