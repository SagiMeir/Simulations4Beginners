import numpy as np
import pandas as pd
import os
os.chdir("c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners")
from sim import Simulation 

params = { "omega":  7.596E13 }

R = np.zeros((1,3,1))

R[0,:,0] = [5,0,0]

PIMD = Simulation(gamma = 1E11, VVtype = "",dt = 0.833E-15, R = R * 1E-10, Nsteps= 10000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, xyzname = "c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/sim.xyz", outname = "c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/sim.log")
PIMD.run(**params)