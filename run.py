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
ExSim = Simulation(dt = 0.1E-18, R= np.array([[5,0,0]]) * 1E-10, p = np.array([[0,0,0]]), Nsteps= 10000, mass = 6.633E-26, kind = ["Ar"], fac=1E10)
params = { "omega":  1.209E16 }
ExSim.run(**params)