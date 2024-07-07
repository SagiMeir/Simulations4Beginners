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
                 outname:str = "sim.log",
                 momentaname:str = "momenta.log",
                 gamma:float = 1E11,
                 VVtype:str = "",
                 beads = 1
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
            
        mass : numpy.ndarray, optional
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
        #self.xyzfile = open( xyzname, 'w' ) 
        self.outfile = open( outname, 'w' ) 
        #self.momentafile = open(momentaname, 'w')

        # self.openfiles = []
        # self.filenames =["sim" + str(i) + ".xyz" for i in range(beads)]
        # # for file in self.filenames:
        # #        self.openfiles.append( open( file, 'w' ) )


        
        #simulation
        self.temp=temp
        self.Nsteps = Nsteps 
        self.dt = dt 
        self.seed = seed 
        self.step = step         
        self.fac = fac
        self.VVtype = VVtype
        self.beads = beads
        self.gamma = gamma
        self.beadomega = np.sqrt(self.beads)*BOLTZMANN*self.temp/hbar

        
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
            self.p = np.zeros( (self.Natoms,3,self.beads) )
            self.K = 0.0
        
        if F is not None:
            self.F = F
            self.U = U
        else:
            self.F = np.zeros( (self.Natoms,3,self.beads) )
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
        # self.momentafile.close()
        # self.xyzfile.close()
        self.outfile.close()
        # for file in self.openfiles:
        #    file.close()

    
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
            self.outfile.write( "step K U E T QE SpringEnergy\n" )
        
        self.outfile.write(str(self.step) + " " \
                          + "{:.6e}".format(self.K) + " " \
                          + "{:.6e}".format(self.U) + " " \
                          + "{:.6e}".format(self.E) + " " \
                          + "{:.6e}".format(self.systemp)+" " \
                          + "{:.6e}".format(self.QE)+" " \
                          + "{:.6e}".format(self.totspringE) + "\n" )
        
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
        # self.xyzname = "sim"+str(i)+".xyz"    
        


        for k in range( self.beads ):
            for i in range(self.Natoms):
                self.openfiles[k].write( str( self.Natoms ) + "\n")
                self.openfiles[k].write( "Step " + str( self.step ) + "\n" )
                self.openfiles[k].write( self.kind[i] + " " + \
                        "{:.6e}".format( self.R[i,0,k]*self.fac ) + " " + \
                        "{:.6e}".format( self.R[i,1,k]*self.fac ) + " " + \
                        "{:.6e}".format( self.R[i,2,k]*self.fac ) + "\n" )
        
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
    def calcQE(self):
        self.QE = self.beads*BOLTZMANN*self.temp/2

        for i in range(self.Natoms):
            for k in range(self.beads - 1):
                self.QE -= (self.mass*self.beads*(BOLTZMANN*self.temp)**2/(2*hbar**2))  *  (self.R[i,0,k+1] - self.R[i,0,k])**2
            self.QE -= (self.mass*self.beads*(BOLTZMANN*self.temp)**2/(2*hbar**2))  *  (self.R[i,0,self.beads - 1] - self.R[i,0,0])**2

        for i in range(self.Natoms):
            self.QE += self.U/self.beads


    def CalcSpringEF(self):
        self.totspringE = 0
        
        for i in range(self.Natoms):
            for k in range(self.beads-1):
                self.totspringE += 0.5 * self.mass * self.beadomega ** 2 * (self.R[i,0,k+1] - self.R[i,0,k])**2
                self.F[i,:,k] = -1 * self.mass * self.beadomega ** 2 * (2*self.R[i,:,k] - self.R[i,:,k+1] - self.R[i,:,k-1])
            self.totspringE += 0.5 * self.mass * self.beadomega**2*(self.R[i,0,self.beads-1] - self.R[i,0,0])**2
            self.F[i,:,self.beads-1] = -1 * self.mass * self.beadomega ** 2 * (2*self.R[i,:,self.beads-1] - self.R[i,:,0] - self.R[i,:,self.beads-2])

    def evalVVstep( self, **kwargs ) -> None:
        
        getattr(self, "VVstep" + self.VVtype)(**kwargs)

    def VVstep_NVT(self,**kwargs):
        xi = np.random.randn(1,3,self.beads)
        xi[0,1:3,:] = np.array([0 for i in range(self.beads)])
        self.p = np.exp(-1*self.gamma*self.dt/2)*self.p + np.sqrt(BOLTZMANN*self.mass*self.temp)*np.sqrt(1-np.exp(-1*self.gamma*self.dt))*xi

        self.VVstep(**kwargs)
        # print("h")

        xi = np.random.randn(1,3,self.beads)
        xi[0,1:3,:] = np.array([0 for i in range(self.beads)])
        self.p = np.exp(-1*self.gamma*self.dt/2)*self.p + np.sqrt(BOLTZMANN*self.mass*self.temp)*np.sqrt(1-np.exp(-1*self.gamma*self.dt))*xi


    def CalcKinE( self ):
        """
        THIS FUNCTIONS EVALUATES THE KINETIC ENERGY OF THE SYSTEM.
        Returns
        -------
        None. Sets the value of self.K.
        """

        self.K = (self.p ** 2).sum() / (2 * self.mass)
        self.systemp =  self.K / BOLTZMANN / self.beads /self.Natoms * 2

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
        for i in range(self.Natoms):
            for k in range(self.beads):
                self.p[i,:,k] = np.random.randn(3)*(np.sqrt(BOLTZMANN*self.temp/self.mass))*np.array([[1,0,0]])*self.mass

    def evalHarm( self, omega:float ):
        """
        THIS FUNCTION EVALUATES THE POTENTIAL AND FORCE FOR A HARMONIC TRAP.
        Parameters
        ----------
        omega : float
            The frequency of the trap.
        Returns
        -------
        None. Sets the value of self.F, self.U and self.K
        """
        
        self.potF = -self.mass * omega ** 2* self.R      
        self.U = 0.5 * self.mass * ((omega * self.R) ** 2).sum()  

    def VVstep( self, **kwargs ):
        """
        THIS FUNCTIONS PERFORMS ONE VELOCITY VERLET STEP.
        Returns
        -------
        None. Sets self.R, self.p.
        """

        self.p = (self.p).copy() + 0.5 * (self.F+self.potF).copy() * self.dt


        self.R = (self.R).copy() + (self.p).copy() * self.dt / self.mass 

        self.evalForce(**kwargs)
        self.CalcSpringEF()

        self.p = (self.p).copy() + 0.5 * (self.F+self.potF).copy() * self.dt

    def dumpMomnta( self ) -> None:
        if( self.step == 0 ):
            self.momentafile.write( "MOMENTA_X MOMENTA_Y MOMENTA_Z" +"\n" )

        for i in range( self.Natoms ):
            for k in range(self.beads):
                self.momentafile.write("bid " + str(k)+"\n")
                self.momentafile.write( "{:.6e}".format( self.p[i,0,k]*self.fac ) + " " + \
                                        "{:.6e}".format( self.p[i,1,k]*self.fac ) + " " + \
                                        "{:.6e}".format( self.p[i,2,k]*self.fac ) + "\n")
        
        self.momentafile.flush()

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

        self.evalForce(**kwargs)
        if(self.beadomega >= self.gamma):
            self.gamma = self.beadomega
        print(self.Natoms)
        for self.step in range(self.Nsteps):
            self.evalVVstep(**kwargs)
            self.CalcKinE()
            self.calcQE()
            self.E =  self.K + self.U + self.totspringE

            if(self.step % self.printfreq == 0):

                self.dumpThermo()
                #self.dumpMomnta()
                #self.dumpXYZ()
        
