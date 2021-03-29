"""

"""

import numpy as np
import matplotlib.pyplot as plt

from model_sets import models as md

print(md.simulations.loc["BLh_M13641364_M0_LK_LR"]["Mej_tot-geo"])

m1 = md.simulations.loc["BLh_M13641364_M0_LK_LR"]["Mej_tot-geo"]
m2 = md.simulations.loc["BLh_M13641364_M0_LK_SR"]["Mej_tot-geo"]
m3 = md.simulations.loc["BLh_M13641364_M0_LK_HR"]["Mej_tot-geo"]

print("{:.2f} pm {:.2f}".format(np.mean([m1,m2,m3])*1e2 , np.std([m1,m2,m3])*1e2))

m1 = md.simulations.loc["BLh_M10201856_M0_LK_LR"]["Mej_tot-geo"]
m2 = md.simulations.loc["BLh_M10201856_M0_LK_SR"]["Mej_tot-geo"]
m3 = md.simulations.loc["BLh_M10201856_M0_LK_HR"]["Mej_tot-geo"]
print(m1,m2,m3)
print("{:.2f} pm {:.2f}".format(np.mean([m1,m2,m3])*1e2 , np.std([m1,m2,m3])*1e2))