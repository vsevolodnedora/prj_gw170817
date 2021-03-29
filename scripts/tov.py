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

__outplotdir__ = "../data/TOVs/"# "../figs/all3/tov_sequences/"

class Paths:
    to_tovs = "../data/TOVs/"



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

def load_plot_TOVS_MR():

    G = 6.67408 * 1e-11
    c = 299792458.
    Msun = 1.989 * 1e30
    GMsun_c2 = G * Msun / c ** 2

    eoss = ["DD2", "LS220", "BLh", "SFHo", "SLy4"]
    colors = ["blue", "red", "green", "orange", "magenta"]
    lss = ["-", "-", "-", "-", "-"]

    fig, ax = plt.subplots(nrows=1, ncols=1)

    for eos, color, ls in zip(eoss, colors, lss):
        # rhoc              M             Mb              R              C             kl            lam
        rhoc, M, M0, R, C, kl, lam = np.loadtxt(Paths.to_tovs + "{}_sequence.txt".format(eos), unpack=True)
        # imax = rhoc.argmax()
        # M = M[:imax]
        # M0 = M0[:imax]

        ax.plot(R*GMsun_c2/1e3, M, c=color, ls = ls, label=eos)

    # models
    from model_sets import groups
    simulations = groups.groups
    for eos, color in zip(eoss, colors):
        sel = simulations[simulations["EOS"] == eos]
        qs = np.array(list(set(list(sel["q"]))))#.round(2)
        print("eos:{} models:{} q:{}".format(eos, len(sel), qs))
        for q in qs:
            ssel = sel[sel["q"] == q]
            img = float(np.array(ssel["Mg1"])[0])
            tmr = float(np.array(ssel["R1"])[0]) * GMsun_c2/1e3
            # print("\t{} {} {}".format(q, tmr, img))
            ax.scatter(tmr, img, marker="d", s=60, edgecolors=color, facecolors="none")
            # ax.annotate('{:.2f}'.format(q),
            #             xy=(tmr, img),  # theta, radius
            #             #xytext=(0.05, 0.05),  # fraction, fraction
            #             #textcoords='figure fraction',
            #             #arrowprops=dict(facecolor='black', shrink=0.05),
            #             horizontalalignment='left',
            #             verticalalignment='bottom',
            #             )

    # ax.axhline(y=2.01, color='k', ls='--', label='J0348+0432')
    # ax.fill_between([6, 16], y1=2.01 - 0.04, y2=2.01 + 0.04, color='grey', alpha=0.5)
    # ax.axhline(y=2.14, color='k', ls='-.', label='J0740+6620')
    # ax.fill_between([6, 16], y1=2.14 - 0.09, y2=2.14 + 0.1, color='grey', alpha=0.5)
    ax.set_ylabel(r'Gravitational Mass $M[M_\odot]$', fontsize=14)
    ax.set_xlabel(r'Radius $R[{\rm km}]$', fontsize=14)
    ax.set_xticks(np.arange(5, 11, 0.5), minor=True)
    ax.set_xlim(10, 16)
    ax.legend(loc='best', fontsize=16, shadow= "False", ncol=1, columnspacing= 0.4,
                       framealpha= 0., borderaxespad= 0., frameon= False)
    # axes[0].set_ylim(1.,3.)
    ax.tick_params(axis='both', which='both',
                   labelleft=True, labelright=False,
                   tick1On=True, tick2On=True,
                   direction='in', labelsize=14)
    ax.minorticks_on()

    print("plotted: {}".format(__outplotdir__ + 'tov_mr.pdf'))
    plt.savefig(__outplotdir__ + 'tov_mr.pdf')
    plt.show()
    plt.close()

if __name__ == "__main__":
    # plot_tov()
    load_plot_TOVS_MR()
