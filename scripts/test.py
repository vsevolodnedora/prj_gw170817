
import os
from uutils import *
import mayavi
from mayavi import mlab
mlab.options.offscreen = True
#mlab.engine.current_scene.scene.off_screen_rendering = True

def task_copy_collated_gw_data():

    dirs_tp_copy = ["waveforms"] # "collated",

    indir = "/data01/numrel/vsevolod.nedora/postprocessed4/"
    outdir = "/data1/numrel/WhiskyTHC/Backup/2018/GW170817/"

    in_sims = os.listdir(indir)
    out_sims = os.listdir(outdir)

    print(in_sims)
    print(out_sims)

    for insim in in_sims:
        if insim in out_sims:
            #
            content = os.listdir(outdir + insim + '/')
            for reqdir in dirs_tp_copy:
                if not reqdir in content:
                    print("cp -r {}{}/{} {}{}/".format(indir,insim,reqdir, outdir,insim))
                    os.system("cp -r {}{}/{} {}{}".format(indir,insim,reqdir, outdir,insim))
                else:
                    print("\tDir: {} already in {}".format(reqdir, insim))

def test_myavi():

    figpath = "../figs/all3/test_myavi/"

    X = np.linspace(-10, 10, 100)
    x, y, z = np.meshgrid(X, X, X)
    f = np.cos(0.66 * np.pi * (x) / np.sqrt(x ** 2 + y ** 2 + z ** 2))

    fig = mlab.figure()
    mlab.contour3d(f, contours=6, transparent=True, figure=fig)
    print("sved: {}".format(figpath+"test1.png"))
    #mlab.show()
    mlab.savefig(figpath+"test1.png")
    #mlab.clf(fig)
   # mlab.close()


if __name__ == '__main__':
    test_myavi()
    # print(15*constant_length)
    #task_copy_collated_gw_data()
