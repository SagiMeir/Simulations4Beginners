import numpy as np
import pandas as pd
from sim import Simulation

"""
HERE, TO RUN THE SIMULATION, YOU WILL NEED TO DO THE FOLLOWING THINGS:
    
    1. CREATE AN OBJECT OF THE SIMULATION CLASS. A MINIMAL EXAMPLE IS
        >>> mysim = Simulation( dt=0.1E-15 )
    
    2. DEFINE THE PARAMETERS FOR THE POTENTIAL. USE A DICTIONARY, FOR EXAMPLE
    FOR THE LJ MODEL OF ARGON, IN SI UNITS:
        >>> params = { "omega":  1.656778224E9 }
    
THEN, CALLING THE METHODS YOU IMPLEMENTED IN sim.py, YOU NEED TO
    3. READ THE INITIAL XYZ FILE PROVIDED OR SAMPLE INITIAL COORDINATES.
    4. SAMPLE INITIAL MOMENTA FROM MB DISTRIBUTION (if needed)
    5. REMOVE COM MOTION (if needed).
    6. RUN THE SIMULATION, INCLUDING PRINTING XYZ AND ENERGIES TO FILES.

THE SPECIFIC SIMULATIONS YOU NEED TO RUN, AND THE QUESTIONS YOU NEED TO ANSWER,
ARE DEFINED IN THE JUPYTER NOTEBOOK.
    
NOTE THAT TO CALL A METHOD OF A CLASS FOR THE OBJECT mysim, THE SYNTAX IS
    >>> mysim.funcName( args )
    
FINALLY, YOU SHOULD
    7. ANALYZE YOUR RESULTS. THE SPECIFIC GRAPHS TO PLOT ARE EXPLAINED IN THE
    JUPYTER NOTEBOOK.

NOTE: YOUR OUTPUT XYZ FILE SHOULD BE PRINTED IN ANGSTROM, BUT YOU CAN USE
ANY UNITS YOU WANT IN BETWEEN.

"""



mysim = Simulation(dt = 0.1E-17, R = np.array([[5/1E10,0,0]]), Nsteps = 100000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10,printfreq=1000, xyzname = "c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/sim.xyz",  outname = "c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/sim.log")
params ={"omega": 7.596E13}
mysim.run(**params)





"""

    >>>simulations<<<
for i,dt in enumerate(np.linspace(0.1E-15, 0.1E-14, 5)):#0.875
    mysim = Simulation(dt = dt, R = np.array([[5/1E10,0,0]]), Nsteps = 100000, mass = 6.6E-26, kind = ["Ar"], fac = 1E10, xyzname = "c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/sim" + str(i) + ".xyz",  outname = "c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/sim"+str(i)+".log")
    params ={"omega": 7.56E13}
    mysim.run(**params)



"""



'''
            >>>the optimal value for dt fig1<<<


dts = np.linspace(0.1E-15, 8.530612244897958e-16)

# print(dts[i])
# mysim = Simulation(dt = dts[0], R = np.array([[5/1E10,0,0]]), Nsteps = 100000, mass = 6.6E-26, kind = ["Ar"], fac = 1E10, xyzname = "c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/sim.xyz",  outname = "c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/sim1.log")
# mysim.run(**params)

df = pd.read_csv("c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/sim1.log",sep='\s+')
deltaper = abs((df["E"].values - df["E"].values[0]))/df["E"].values *100
for dt in dts:
    
    mysim = Simulation(dt = dt, R = np.array([[5/1E10,0,0]]), Nsteps = 100000, mass = 6.6E-26, kind = ["Ar"], fac = 1E10, xyzname = "c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/sim.xyz",  outname = "c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/sim.log")
    mysim.run(**params)
    df = pd.read_csv("c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/sim.log",sep='\s+')
    deltaper = (abs(df["E"].values - df["E"].values[0]))/df["E"].values *100

    if(max(deltaper) >=0.1):
        break

print(dt)

'''