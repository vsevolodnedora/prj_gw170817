
import os
import pandas as pd
import click
from uutils import *
# import mayavi
# from mayavi import mlab
# mlab.options.offscreen = True
#mlab.engine.current_scene.scene.off_screen_rendering = True

# def task_copy_collated_gw_data():
#
#     dirs_tp_copy = ["waveforms"] # "collated",
#
#     indir = "/data01/numrel/vsevolod.nedora/postprocessed4/"
#     outdir = "/data1/numrel/WhiskyTHC/Backup/2018/GW170817/"
#
#     in_sims = os.listdir(indir)
#     out_sims = os.listdir(outdir)
#
#     print(in_sims)
#     print(out_sims)
#
#     for insim in in_sims:
#         if insim in out_sims:
#             #
#             content = os.listdir(outdir + insim + '/')
#             for reqdir in dirs_tp_copy:
#                 if not reqdir in content:
#                     print("cp -r {}{}/{} {}{}/".format(indir,insim,reqdir, outdir,insim))
#                     os.system("cp -r {}{}/{} {}{}".format(indir,insim,reqdir, outdir,insim))
#                 else:
#                     print("\tDir: {} already in {}".format(reqdir, insim))
# #
# def test_myavi():
#
#     figpath = "../figs/all3/test_myavi/"
#
#     X = np.linspace(-10, 10, 100)
#     x, y, z = np.meshgrid(X, X, X)
#     f = np.cos(0.66 * np.pi * (x) / np.sqrt(x ** 2 + y ** 2 + z ** 2))
#
#     fig = mlab.figure()
#     mlab.contour3d(f, contours=6, transparent=True, figure=fig)
#     print("sved: {}".format(figpath+"test1.png"))
#     #mlab.show()
#     mlab.savefig(figpath+"test1.png")
#     #mlab.clf(fig)
   # mlab.close()


from model_sets import models
simmulations = models.simulations


# rename old initial data:
__simpath__ = "/data01/numrel/vsevolod.nedora/postprocessed4/"
__zenodopat__ = ""
__pprscriptpath__ = "/data01/numrel/vsevolod.nedora/bns_ppr_tools/"
def rename_old_initcsv():
    simpath = __simpath__
    fname = "init_data.csv"
    missing = []
    for sim, dic in simmulations.iterrows():
        if os.path.isfile("{}{}/{}".format(simpath, sim, fname)):
            cmd = "mv {}{}/{} {}{}/old{}".format(simpath, sim, fname, simpath, sim, fname)
            print(cmd)
            os.system(cmd)
        else:
            missing.append(sim)

    print("No initial data for :")
    for sin in missing:
        print(sim)

# run preanalysis to update the initial data
def run_preanalysis():
    scriptfpath = __pprscriptpath__
    os.chdir(scriptfpath)
    for sim, md in simmulations.iterrows():
        if True:# "#click.confirm('Update initiad data for sim: {}'.format(sim)):
            os.system("python preanalysis.py -s {} -t init_data --overwrite yes".format(sim))
            # click.echo('Well done!')
    # PROBLEMS:
    # BLh_M13641364_M0_LR BLh_M12591482_M0_LR BLh_M11841581_M0_LR
    # BLh_M11041699_M0_LR BLh_M10201856_M0_LR SLy4_M18461025_M0_LK_SR_R05_AHfix
    #

# compy run data
def copy_run_from_old_to_new():

    oldfname = "oldinit_data.csv"
    newfname = "init_data.csv"
    simpath = __simpath__

    for sim, md in simmulations.iterrows():
        if True:#click.confirm('Update run data for sim: {}'.format(sim)):
            try:
                old_file_object = open(simpath+sim+"/"+oldfname, 'r')
                run = ""
                for line in old_file_object.readlines():
                    if line.__contains__("run"):
                        run = line
                        break
                if run != "":
                    print("run info: {}".format(run))
                    new_file_object = open(simpath+sim+"/"+newfname, 'a')
                    new_file_object.writelines(run)
                else:
                    print("\tNo run info found")
            except IOError :
                print("IOError FAILED for {}".format(sim))

# copy ejecta data
def copy_files_to_zenodo():
    pass


def get_xmin_xmax_ymin_ymax_zmin_zmax(rl):
    if rl == 6:
        xmin, xmax = -14, 14
        ymin, ymax = -14, 14
        zmin, zmax = 0, 14
    elif rl == 5:
        xmin, xmax = -28, 28
        ymin, ymax = -28, 28
        zmin, zmax = 0, 28
    elif rl == 4:
        xmin, xmax = -48, 48
        ymin, ymax = -48, +48
        zmin, zmax = 0, 48
    elif rl == 3:
        xmin, xmax = -88, 88
        ymin, ymax = -88, 88
        zmin, zmax = 0, 88
    elif rl == 2:
        xmin, xmax = -178, 178
        ymin, ymax = -178, +178
        zmin, zmax = 0, 178
    elif rl == 1:
        xmin, xmax = -354, 354
        ymin, ymax = -354, +354
        zmin, zmax = 0, 354
    elif rl == 0:
        xmin, xmax = -1044, 1044
        ymin, ymax = -1044, 1044
        zmin, zmax = 0, 1044
    else:
        # pass
        raise IOError("Set limits for rl:{}".format(rl))

    conv_to_km = lambda val: units.conv_length(units.cactus, units.cgs, val)/1e5

    return conv_to_km(xmin), conv_to_km(xmax), conv_to_km(ymin), \
           conv_to_km(ymax), conv_to_km(zmin), conv_to_km(zmax)

def update_simlog():

    path_to_simlog = "/home/vsevolod/GIT/bitbucket/runs-thc-aa/GW170817/simulations.json"
    simlog = pd.read_json(path_to_simlog, orient="records")
    simlog = simlog["GW170817"]
    # print(simlog)
    missing_from_log = []
    missing_from_sims = []

    print('-'*40)

    for e in simlog:
        if not e["simfolder"] in simmulations.index:
            missing_from_sims.append(e["simfolder"])
    for sim, simdic in simmulations.iterrows():
        found = False
        for e in simlog:
            if e["simfolder"] == sim:
                found = True
        if not found:
            missing_from_log.append(sim)
            #
    print("Missing from log: ({})".format(len(missing_from_log)))
    print(missing_from_log)

    # print("Missing from sims: ({})".format(len(missing_from_sims)))
    # print(missing_from_sims)
            # print("mssing: {}".format(e["simfolder"]))
        # print(entry) # -- dict for a given run
        # print(e["simfolder"], e["res"], e["M_ADM"])


    # for key, val in simlog.iterrows():
    #     print(val)

if __name__ == '__main__':
    update_simlog()

    # from model_sets.groups import groups
    # print(groups.keys())
    # print(groups[groups.index=="BLh_M10651772_M0_LK"][["tend_wind"]])
    # groups = groups[(groups.tend_wind > 0.030)]# & (groups.index != "BLh_M10651772_M0_LK")]
    # print(groups[["q", "resolution"]])
    ##### rename_old_initcsv()
    # run_preanalysis()
    # copy_run_from_old_to_new()
    # pass
    # test_myavi()
    # print(15*constant_length)
    #task_copy_collated_gw_data()

    '''
    BLh_M12591482_M0_LR
    BLh_M11841581_M0_LR
    BLh_M11041699_M0_LR
    BLh_M10201856_M0_LR
    '''


    #  from scidata import units
    # print(units.conv_length(units.cactus, units.cgs, 1)/1e5) # Msun -> km
    # print(units.conv_length(units.cgs, units.cactus, 1e5))

    # print(get_xmin_xmax_ymin_ymax_zmin_zmax(1))

    # reflevel 0

    # print('='*60)
    # from model_sets import combined as md
    # sims = md.simulations[md.mask_none]

    # print(sims)

    # print("q:     ({}, {})".format(min(sims["q"]),max(sims["q"])))
    # print("lambda:({}, {})".format(min(sims["Lambda"]), max(sims["Lambda"])))
    # print("Mchirp:({}, {})".format(min(sims["Mchirp"]), max(sims["Mchirp"])))
    # print("Mtot:  ({}, {})".format(min(sims["Mtot"]), max(sims["Mtot"])))
    # print(sims[md.mask_refset][["q"]])


    # from model_sets.models_lehner2016 import simulations as tst
    # print(tst)