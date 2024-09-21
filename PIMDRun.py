import numpy as np
import pandas as pd
import os
os.chdir("c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners")
from sim import Simulation 

params = { "omega":  7.596E13 }
beads = 8
R = np.zeros((1,3,beads))


# # # R[0,0,:]=np.array([0 for i in range(beads)])
#R[0,:,0] = np.array([[0,0,0]])

PIMD = Simulation(temp = 96.7 ,beads = beads,gamma = 1E14, VVtype = "_NVT",dt = 0.833E-15, R = R *1E-10, Nsteps=300000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, xyzname = "c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/sim.xyz", outname = "c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/sim.log")
PIMD.sampleMB()
PIMD.run(**params)

#for i in range(2,8):
#    beads = 2**i
#    R = np.zeros((1,3,beads))

#    PIMD = Simulation(temp = 96.7 ,beads = beads,gamma = 1E11, VVtype = "_NVT",dt =  0.1E-16, R = R *1E-10, Nsteps= 200000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, xyzname = "c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/sim.xyz", outname = "C:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/estimator2(better)/sim"+str(beads)+".log")
#    PIMD.sampleMB()
#    PIMD.run(**params)