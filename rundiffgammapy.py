import numpy as np
import pandas as pd
import os
os.chdir("c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners")
from sim import Simulation

params = { "omega":  7.596E13 }




for i in np.linspace(11, 15, 9):
     gammaSim = Simulation(gamma = 10**i, VVtype = "_NVT",dt = 0.833E-15, R= np.array([[5,0,0]]) * 1E-10, Nsteps= 400000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, xyzname = "c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/sim.xyz", outname = "c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/sim"+str(i)+".log")
     gammaSim.run(**params)

print("DONE")
#gammaSim = Simulation(gamma = 1E11, VVtype = "_NVT",dt = 0.833E-15, R= np.array([[5,0,0]]) * 1E-10, Nsteps= 400000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, xyzname = "c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/sim.xyz", outname = "c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/sim.log")
#gammaSim.run(**params)
