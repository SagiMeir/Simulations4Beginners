import numpy as np
import pandas as pd
from sim import Simulation
import os

seeds = []
for i in range(1):
  seeds.append(937142 + i)

for i, seedNum in enumerate(seeds):
  for gaussianPosition in range(100000 ,1000000, 100000):
    url = f"new research/otherGamma/seed-{seedNum}/gaussianPosition-{gaussianPosition}"
    os.makedirs(url)
    url = url + "/specificLocation"
    sim = Simulation(dt = 0.833E-15, R= np.array([[0.8,0,0]]) * 1E-10, Nsteps= 1000000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, xyzname = f"{url}.xyz", outname = f"{url}.log", momentname=f"{url}_p.log", forcenamme=f"{url}_F.log", mtype="NVT", ftype="DoubleWell", gamma= 7.596E13, startingStep=120000, withMetaD=True, w=5E-20, sigma=0.5e-12, MetaDfreq=gaussianPosition, seed=seedNum, gaussiansname=f"{url}_gaussianPos.log", oneGaussian=True)
    sim.run()



print('''
  _____     ____    _   _   ______ 
 |  __ \   / __ \  | \ | | |  ____|
 | |  | | | |  | | |  \| | | |__   
 | |  | | | |  | | | . ` | |  __|  
 | |__| | | |__| | | |\  | | |____ 
 |_____/   \____/  |_| \_| |______|
                                                                     
''')