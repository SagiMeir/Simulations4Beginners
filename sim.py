from typing import Optional
import numpy as np
import pandas as pd
from scipy.constants import Boltzmann as BOLTZMANN
from scipy.constants import hbar

class Simulation:
    
    def __init__( self, 
                 dt:float, 
                 temp:float = 298, 
                 Nsteps:int = 0, 
                 R:Optional[np.ndarray] = None, 
                 mass:Optional[float] = None, 
                 kind:Optional[list] = None, 
                 p:Optional[np.ndarray] = None, 
                 F:Optional[np.ndarray] = None, 
                 U:Optional[float] = None, 
                 K:Optional[float] = None, 
                 seed:Optional[int] = 937142, 
                 ftype:str = "Harm", 
                 step:int = 0, 
                 printfreq:int = 1000, 
                 xyzname:str = "sim.xyz", 
                 fac:float = 1.0,  
                 outname:str = "sim.log"
                 ) -> None:
        """
        Parameters
        ----------
        dt : float
            Simulation time step.
      
        temp: float
            The temperature.
            
        Nsteps : int, optional
            Number of steps to take. The default is 0.
            
        R : numpy.ndarray, optional
            Particles' positions, Natoms x 3 array. The default is None.
            
        mass : float, optional
            Particles' masses, Natoms x 1 array. The default is None.
            
        kind : list of str, optional
            Natoms x 1 list with atom type for printing. The default is None.
            
        p : numpy.ndarray, optional
            Particles' momenta, Natoms x 3 array. The default is None.
            
        F : numpy.ndarray, optional
            Particles' forces, Natoms x 3 array. The default is None.
            
        U : float, optional
            Potential energy . The default is None.
            
        K : float, optional
            Kinetic energy. The default is None.
            
        seed : int, optional
            Big number for reproducible random numbers. The default is 937142.
            
        ftype : str, optional
            String to call the force evaluation method. The default is None.
            
        step : INT, optional
            Current simulation step. The default is 0.
            
        printfreq : int, optional
            PRINT EVERY printfreq TIME STEPS. The default is 1000.
            
        xyzname : TYPE, optional
            DESCRIPTION. The default is "sim.xyz".
            
        fac : float, optional
            Factor to multiply the positions for printing. The default is 1.0.
        
        thermo_type: str, optional
            String to call the thermostating evaluation method. The default is None.
            
        outname : TYPE, optional
            DESCRIPTION. The default is "sim.log".

        Returns
        -------
        None.
        """
        
        #general        
        self.printfreq = printfreq 
        self.xyzfile = open( xyzname, 'w' ) 
        self.outfile = open( outname, 'w' ) 
        
        #simulation
        self.temp=temp
        self.Nsteps = Nsteps 
        self.dt = dt 
        self.seed = seed 
        self.step = step         
        self.fac = fac
        
        #system        
        if R is not None:
            self.R = R        
            self.mass = mass
            self.kind = kind
            self.Natoms = self.R.shape[0]
            # self.mass_matrix = np.array( [self.mass,]*3 ).transpose()
        else:
            self.R = np.zeros( (1,3) )
            self.mass = 1.6735575E-27 # H mass in kg as default
            self.kind = ["H"]
            self.Natoms = self.R.shape[0]
            # self.mass_matrix = np.array( [self.mass,]*3 ).transpose()
        if p is not None:
            self.p = p
            self.K = K
        else:
            self.p = np.zeros( (self.Natoms,3) )
            self.K = 0.0
        
        if F is not None:
            self.F = F
            self.U = U
        else:
            self.F = np.zeros( (self.Natoms,3) )
            self.U = 0.0
        
        
        #set seed
        np.random.seed( self.seed )
        
        #check force type
        if ( ftype == "Harm" or ftype == "Anharm"):
            self.ftype = "eval" + ftype
        else:
            raise ValueError("Wrong ftype value - use Harm or Anharm.")
        

    def __del__( self ) -> None:
        """
        THIS IS THE DESCTRUCTOR. NOT USUALLY NEEDED IN PYTHON. 
        JUST HERE TO CLOSE THE FILES.
        Returns
        -------
        None.
        """
        self.xyzfile.close()
        self.outfile.close()
    
    def evalForce( self, **kwargs ) -> None:
        """
        THIS FUNCTION CALLS THE FORCE EVALUATION METHOD, BASED ON THE VALUE
        OF FTYPE, AND PASSES ALL OF THE ARGUMENTS (WHATEVER THEY ARE).
        Returns
        -------
        None. Calls the correct method based on self.ftype.
        """
        getattr(self, self.ftype)(**kwargs)
    
    def dumpThermo( self ) -> None:
        """
        THIS FUNCTION DUMPS THE ENERGY OF THE SYSTEM TO FILE.
        Parameters
        ----------
        R : numpy.ndarray
            Natoms x 3 array with particle coordinates.
        step : int
            Simulation time step.
        file : _io.TextIOWrapper 
            File object to write to. Created using open() in main.py
        Returns
        -------
        None.
        """
        if( self.step == 0 ):
            self.outfile.write( "step K U E \n" )
        
        self.outfile.write( str(self.step) + " " \
                          + "{:.6e}".format(self.K) + " " \
                          + "{:.6e}".format(self.U) + " " \
                          + "{:.6e}".format(self.E) + "\n" )
        
        self.outfile.flush()
                
    def dumpXYZ( self ) -> None:
        """
        THIS FUNCTION DUMP THE COORDINATES OF THE SYSTEM IN XYZ FORMAT TO FILE.
        Parameters
        ----------
        R : numpy.ndarray
            Natoms x 3 array with particle coordinates.
        step : int
            Simulation time step.
        file : _io.TextIOWrapper 
            File object to write to. Created using open() in main.py
        Returns
        -------
        None.
        """
            
        self.xyzfile.write( str( self.Natoms ) + "\n")
        self.xyzfile.write( "Step " + str( self.step ) + "\n" )
        
        for i in range( self.Natoms ):
            self.xyzfile.write( self.kind[i] + " " + \
                              "{:.6e}".format( self.R[i,0]*self.fac ) + " " + \
                              "{:.6e}".format( self.R[i,1]*self.fac ) + " " + \
                              "{:.6e}".format( self.R[i,2]*self.fac ) + "\n" )
        
        self.xyzfile.flush()
    
    def readXYZ( self, inpname:str ) -> None:
        """
        THIS FUNCTION READS THE INITIAL COORDINATES IN XYZ FORMAT.
        Parameters
        ----------
        inpname : str
                xyz input file to read

        Returns
        -------
        None.
        """
           
        df = pd.read_csv( inpname, sep="\s+", skiprows=2, header=None )
        
        self.kind = df[ 0 ]
        self.R = df[ [1,2,3] ].to_numpy()
        self.Natoms = self.R.shape[0]
        
        
################################################################
################## NO EDITING ABOVE THIS LINE ##################
################################################################

    def CalcKinE( self ):
        """
        THIS FUNCTIONS EVALUATES THE KINETIC ENERGY OF THE SYSTEM.
        Returns
        -------
        None. Sets the value of self.K.
        """
        ################################################################
        ##################### YOUR CODE GOES HERE ######################
        ################################################################
        pass

    def sampleMB( self, removeCM=True ):
        """
        THIS FUNCTIONS SAMPLES INITIAL MOMENTA FROM THE MB DISTRIBUTION.
        IT ALSO REMOVES THE COM MOMENTA, IF REQUESTED.
        Parameters
        ----------
        removeCM : bool, optional
            Remove COM velocity or not. The default is True.
        Returns
        -------
        None. Sets the value of self.p.
        """
        ################################################################
        ##################### YOUR CODE GOES HERE ######################
        ################################################################
        pass


    def evalHarm( self, omega:float ):
        """
        THIS FUNCTION EVALUATES THE POTENTIAL AND FORCE FOR A HARMONIC TRAP.
        Parameters
        ----------
        omega : float
            The frequency of the trap.
        Returns
        -------
        None. Sets the value of self.F and self.U 
        """
        
        ################################################################
        ####################### YOUR CODE GOES HERE ####################
        ################################################################
        pass

    def VVstep( self, **kwargs ):
        """
        THIS FUNCTIONS PERFORMS ONE VELOCITY VERLET STEP.
        Returns
        -------
        None. Sets self.R, self.p.
        """
        ################################################################
        ####################### YOUR CODE GOES HERE ####################
        ################################################################
        pass

    def run( self, **kwargs ):
        """
        THIS FUNCTION DEFINES WHAT THE SIMULATION DOES, GIVEN AN INSTANCE OF 
        THE SIMULATION CLASS. YOU WILL NEED TO:
            1. EVALUATE THE FORCES (USE evaluateForce() AND PASS A DICTIONARY
                                    WITH ALL THE PARAMETERS).
            2. PROPAGATE FOR N TIME STEPS USING THE VELOCITY VERLET ALGORITHM.
            3. CALCULATE THE KINETIC, POTENTIAL AND TOTAL ENERGY AT EACH TIME
            STEP. 
            4. YOU WILL ALSO NEED TO PRINT THE COORDINATES AND ENERGIES EVERY 
        PRINTFREQ TIME STEPS TO THEIR RESPECTIVE FILES, xyzfile AND outfile.
        Returns
        -------
        None.
        """      
        
        ################################################################
        ####################### YOUR CODE GOES HERE ####################
        ################################################################ 

