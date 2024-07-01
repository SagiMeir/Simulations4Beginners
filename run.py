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

################################################################
####################### YOUR CODE GOES HERE ####################
################################################################

params = { "omega":  7.596E13 }

#fig1
# for i, dt in enumerate(np.linspace(0.1E-15, 1E-15, 5)):
#     SimForFig1 = Simulation(dt, R= np.array([[5,0,0]]) * 1E-10, Nsteps= 100000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, xyzname = "simForFig1_" + str(i) + ".xyz", outname = "simForFig1_" + str(i) + ".log")
#     SimForFig1.run(**params)

#figDt
# for i, dt in enumerate(np.linspace(0.5E-15, 1E-15, 7)):
#     SimForFigDt = Simulation(dt, R= np.array([[5,0,0]]) * 1E-10, Nsteps= 100000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, xyzname = "SimForFigDt_" + str(i) + ".xyz", outname = "SimForFigDt_" + str(i) + ".log")
#     SimForFigDt.run(**params)


#fig2
SimForFig2 = Simulation(dt = 0.833E-15, R= np.array([[5,0,0]]) * 1E-10, Nsteps= 100000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, xyzname = "simForFig2.xyz", outname = "simForFig2.log")
SimForFig2.run(**params)



print('''
  _____     ____    _   _   ______ 
 |  __ \   / __ \  | \ | | |  ____|
 | |  | | | |  | | |  \| | | |__   
 | |  | | | |  | | | . ` | |  __|  
 | |__| | | |__| | | |\  | | |____ 
 |_____/   \____/  |_| \_| |______|
                                                                     
''')