import numpy as np

mySim = Simulation(dt = 0.1e-15, Nsteps = 10000, printfreq = 250, step = 0)
mySim.R = np.array([[5.0, 0.0, 0.0]])
mySim.kind = ['Ar']
mySim.Natoms = 1
mySim.mass = 6.6335209e-26

mySim.p = np.random.normal(0, 1, size=(1, 3)) * np.sqrt(mySim.mass * 1.38e-23 * mySim.temp)

params = {"omega": 7.586017233041558e13}
mySim.evalForce(**params)

mySim.CalcKinE()
mySim.U = float(mySim.U)
mySim.K = float(mySim.K)
mySim.E = mySim.K + mySim.U

mySim.dumpXYZ()

mySim.run(**params)

from google.colab import files
files.download("sim.xyz")
files.download("sim.log")
