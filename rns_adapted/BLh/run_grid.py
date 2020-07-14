''' 
This program generates a grid in rho_c parameter space.
Then feeds them to ./kepler one by one, to generate sequences
of rigidly rotating stars with varying J.
Then merges all the individual outfiles into one and deletes them.

  rho_c: central density of star
'''

import os
from multiprocessing import Pool, cpu_count
from math import log10
import numpy as np

EOS="BLh"

def run_ith_set(i):
        s = "../RNS_code/kepler -q tab -f ../eos/{0} -e {1:g} -d 0.001 -o data/{2}.out".format(EOS, rho_c_values[i], i)
        print('{:03d}/{:03d}: {}'.format(i+1, rho_c_values.shape[0], s))
        os.system(s)

rho_c_values = np.linspace (1e14, 5e15, num=300) # start, stop, num_points 

# Run the ./kepler code with each set of parameters, in parallel
p = Pool(cpu_count())
p.map(run_ith_set, range(rho_c_values.shape[0]))

# Merge all individual .out files into output.dat and remove them.
# Use '>' for deleting previous output.dat, '>>' for appending.
os.system('cat data/*.out > data/output.dat && rm data/*.out')
