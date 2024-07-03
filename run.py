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


#fig 2
# SimForFig2 = Simulation(dt = 0.833E-15, R= np.array([[5,0,0]]) * 1E-10, Nsteps= 100000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, xyzname = "simForFig2.xyz", outname = "simForFig2.log")
# SimForFig2.run(**params)

# fig 3
# SimForFig3 = Simulation(dt = 0.833E-15, R= np.array([[5,0,0]]) * 1E-10, Nsteps= 100000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, xyzname = "simForFig3.xyz", outname = "simForFig3.log")
# SimForFig3.run(**params)

#fig 4
# SimForFig4 = Simulation(dt = 0.833E-15, R= np.array([[5,0,0]]) * 1E-10, Nsteps= 100000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, xyzname = "simForFig4.xyz", outname = "simForFig4.log")
# SimForFig4.run(**params)

#fig 5
# SimForFig5 = Simulation(dt = 0.833E-15, R= np.array([[5,0,0]]) * 1E-10, Nsteps= 100000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, xyzname = "simForFig5.xyz", outname = "simForFig5.log", momentname= "simForFig5_p.log")
# SimForFig5.run(**params)

# for pos vs time
# SimForPos_NVT = Simulation(dt = 0.833E-15, R= np.array([[5,0,0]]) * 1E-10, Nsteps= 300000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, xyzname = "simForPos_NVT.xyz", outname = "simForPos_NVT.log", momentname= "simForPos_NVT_p.log", mtype="NVT")
# SimForPos_NVT.run(**params)

# for temp vs time
# SimForTemp_NVT = Simulation(dt = 0.833E-15, R= np.array([[5,0,0]]) * 1E-10, Nsteps= 300000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, xyzname = "simForTemp_NVT.xyz", outname = "simForTemp_NVT.log", momentname= "simForTemp_NVT_p.log", mtype="NVT")
# SimForTemp_NVT.run(**params)

# after remove the start
# SimForTempAndPosAfterTheStart_NVT = Simulation(dt = 0.833E-15, R= np.array([[5,0,0]]) * 1E-10, Nsteps= 300000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, xyzname = "SimForTempAndPosAfterTheStart_NVT.xyz", outname = "SimForTempAndPosAfterTheStart_NVT.log", momentname= "SimForTempAndPosAfterTheStart_NVT_p.log", mtype="NVT", startingStep=120000)
# SimForTempAndPosAfterTheStart_NVT.run(**params)

# for best gamma
# for i, gamma in enumerate(np.linspace(1E11, 1E15, 5)):
#     simForGamma = Simulation(dt = 0.833E-15, R= np.array([[5,0,0]]) * 1E-10, Nsteps= 500000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, xyzname = "simForGamma" + str(i) + ".xyz", outname = "simForGamma" + str(i) + ".log", momentname= "simForGamma_p" + str(i) + ".log", mtype="NVT", startingStep=120000, gamma= gamma)
#     simForGamma.run(**params)

# hist in the best gamma
# simForHist = Simulation(dt = 0.833E-15, R= np.array([[5,0,0]]) * 1E-10, Nsteps= 500000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, xyzname = "simForHist.xyz", outname = "simForHist.log", momentname= "simForHist_p.log", mtype="NVT", startingStep=120000, gamma= 7.596E13)
# simForHist.run(**params)


#############################################################################
################################METADYNAMICS#################################
#############################################################################


# to find new best dt
# for i, dt in enumerate(np.linspace(1E-16, 2E-15, 7)):
#     SimFor2WDt = Simulation(dt, R= np.array([[2,0,0]]) * 1E-10, Nsteps= 300000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, xyzname = "SimFor2WDt" + str(i) + ".xyz", outname = "SimFor2WDt" + str(i) + ".log", momentname= "SimFor2WDt" + str(i) + "_p.log", ftype="DoubleWell")
#     SimFor2WDt.run()


#  without MD in 2 well
simFor2Well = Simulation(dt = 0.833E-15, R= np.array([[0.1,0,0]]) * 1E-10, Nsteps= 1000000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, xyzname = "simFor2Well.xyz", outname = "simFor2Well.log", momentname= "simFor2Well_p.log", mtype="NVE", ftype="DoubleWell")
simFor2Well.run()

#  without MD in 2 well
simFor2Well_NVT = Simulation(dt = 0.833E-15, R= np.array([[0.1,0,0]]) * 1E-10, Nsteps= 1000000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, xyzname = "simFor2Well_NVT.xyz", outname = "simFor2Well_NVT.log", momentname= "simFor2Well_NVT_p.log", mtype="NVT", ftype="DoubleWell", gamma= 7.596E13, startingStep=120000)
simFor2Well_NVT.run()





print('''
  _____     ____    _   _   ______ 
 |  __ \   / __ \  | \ | | |  ____|
 | |  | | | |  | | |  \| | | |__   
 | |  | | | |  | | | . ` | |  __|  
 | |__| | | |__| | | |\  | | |____ 
 |_____/   \____/  |_| \_| |______|
                                                                     
''')