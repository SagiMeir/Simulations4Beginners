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
                 Vbias:Optional[np.ndarray] = np.zeros((1,3)), 
                 seed:Optional[int] = 937142, 
                 ftype:str = "Harm",
                 mtype:str = "NVE", 
                 step:int = 0, 
                 printfreq:int = 1000,
                 MetaDfreq:int = 100, 
                 xyzname:str = "sim.xyz", 
                 fac:float = 1.0,  
                 outname:str = "sim.log",
                 momentname:str = "moment.log",
                 forcenamme:str = "force.log",
                 gaussiansname:str = "gaussiansPos.log",
                 gamma:float = 2.5E13,
                 numOfDim:int = 1,
                 startingStep:int = 0,
                 w:int = 1,
                 sigma:int = 1,
                 withMetaD:bool = False,
                 notUseMB:bool = False,
                 withPoissonDist:bool = False,
                 isDeep:bool = False,
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
        self.xyzfile = open( xyzname, 'w' ) 
        self.outfile = open( outname, 'w' ) 
        self.momentfile = open( momentname, 'w')
        self.forcefile = open( forcenamme, 'w')
        
        #simulation
        self.temp=temp
        self.Nsteps = Nsteps 
        self.dt = dt 
        self.seed = seed 
        self.step = step         
        self.fac = fac
        self.gamma = gamma
        self.numOfDim = numOfDim
        self.startingStep = startingStep
        self.w = w
        self.sigma = sigma
        self.Vbias = Vbias
        self.withMedaD = withMetaD
        self.gaussiansPos = []
        self.MetaDfreq  = MetaDfreq
        self.notUseMB = notUseMB
        self.withPoissonDist = withPoissonDist
        self.numOfGaussians = 0
        self.stepsPos = []
        self.isDeep = isDeep


        if(self.withMedaD):
            self.gaussiansfile = open( gaussiansname, 'w')
        
        #system        
        if R is not None:
            self.R = R        
            self.mass = mass
            self.kind = kind
            self.Natoms = R.shape[0]
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

        self.imgF = np.zeros((self.Natoms ,3))
        
        
        #set seed
        np.random.seed( self.seed )
        
        #check force type
        if ( ftype == "Harm" or ftype == "DoubleWell"):
            self.ftype = "eval" + ftype
        else:
            raise ValueError("Wrong ftype value - use Right type")
        
        if(mtype == "NVT" or "NVE"):
            self.mtype = "VVstep_" + mtype
        else:
            raise ValueError("Wrong mtype value - use NVT or NVE")
        
        match self.numOfDim:
            case 2:
                self.dim = np.array([1,1,0])
            case 3:
                self.dim = np.array([1,1,1])
            case _:
                self.dim = np.array([1,0,0])
        

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
        self.momentfile.close()
        self.forcefile.close()
        self.gaussiansfile.close()
    
    def evalForce( self, **kwargs ) -> None:
        """
        THIS FUNCTION CALLS THE FORCE EVALUATION METHOD, BASED ON THE VALUE
        OF FTYPE, AND PASSES ALL OF THE ARGUMENTS (WHATEVER THEY ARE).
        Returns
        -------
        None. Calls the correct method based on self.ftype.
        """
        getattr(self, self.ftype)(**kwargs)

    def evalMethod(self, **kwargs) -> None:
        getattr(self, self.mtype)(**kwargs)
    
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
        if( self.step == self.startingStep ):
            self.outfile.write( "step K U E temp\n" )
        
        self.outfile.write( str(self.step - self.startingStep) + " " \
                          + "{:.6e}".format(self.K) + " " \
                          + "{:.6e}".format(self.U) + " " \
                          + "{:.6e}".format(self.E) + " " \
                          + "{:.6e}".format(self.systemTemp) + "\n")
        
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

    def dumpMoment(self) -> None:
        if(self.step == self.startingStep):
            self.momentfile.write( "pX pY pZ \n" )
        
        for i in range( self.Natoms ):
            self.momentfile.write(
                        "{:.6e}".format( self.p[i,0]*self.fac ) + " " + \
                        "{:.6e}".format( self.p[i,1]*self.fac ) + " " + \
                        "{:.6e}".format( self.p[i,2]*self.fac ) + "\n" )
            
        self.momentfile.flush()
            
    
    def dumpForce(self) -> None:
        if(self.step == self.startingStep):
            self.forcefile.write( "FX FY FZ \n" )
        
        for i in range( self.Natoms ):
            self.forcefile.write(
                        "{:.6e}".format( self.F[i,0] ) + " " + \
                        "{:.6e}".format( self.F[i,1] ) + " " + \
                        "{:.6e}".format( self.F[i,2] ) + "\n" )   
        self.forcefile.flush()  

    def dumpGaussiansPos(self) -> None:  
        if(self.step == self.startingStep + self.MetaDfreq):
            self.gaussiansfile.write("Step X Y Z \n")

        for i in range(self.Natoms):
            self.gaussiansfile.write( 
                            str(self.step) + " " + \
                            "{:.6e}".format( self.R[i,0]*self.fac ) + " " + \
                            "{:.6e}".format( self.R[i,1]*self.fac ) + " " + \
                            "{:.6e}".format( self.R[i,2]*self.fac ) + "\n" )  

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
        
    def CalcKinE( self ):
        """
        THIS FUNCTIONS EVALUATES THE KINETIC ENERGY OF THE SYSTEM.
        Returns
        -------
        None. Sets the value of self.K.
        """
        self.K = (self.p ** 2).sum() / (2 * self.mass)

    def CalcTemp(self):
        self.systemTemp = 2 * self.K / (self.Natoms * BOLTZMANN)


    def sampleMB( self ):
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
        self.p = np.random.normal(0,self.mass * np.sqrt(BOLTZMANN * self.temp) * 1E10, size=(self.Natoms, self.numOfDim)) * self.dim
        print(self.p * self.fac)


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
        
        self.F = -1 * self.mass * omega ** 2 * self.R
        self.U = (0.5 * self.mass * (omega * self.R) ** 2).sum()

    def evalDoubleWell(self):
        if(not self.isDeep):
            A = 4.11E20
            B = 8.22
        else:
            A = 4.11E22
            B = 822
        self.F = (-4 * A * self.R ** 3) + (2 * B * self.R) 
        self.U = (A * self.R ** 4).sum() - (B * self.R ** 2).sum()

    def updateMetaD(self):
        if(self.withPoissonDist and self.step == self.stepsPos[self.numOfGaussians]):
            self.gaussiansPos.append(self.R)
            self.numOfGaussians += 1
        if(not self.withPoissonDist and self.step % self.MetaDfreq == 0):
            self.gaussiansPos.append(self.R)

        diff = self.R[np.newaxis,:,:] - np.array(self.gaussiansPos)
        self.Vbias = self.w * (np.exp(-diff ** 2 / (2 * self.sigma ** 2))).sum(0)
        self.imgF = self.w / self.sigma ** 2 * (diff * (np.exp(-diff ** 2 / (2 * self.sigma ** 2)))).sum(0)

    def VVstep_NVE( self, **kwargs ):
        """
        THIS FUNCTIONS PERFORMS ONE VELOCITY VERLET STEP.
        Returns
        -------
        None. Sets self.R, self.p.
        """

        self.p = (self.p + 0.5 * (self.F + self.imgF) * self.dt) * self.dim
        self.R = (self.R + self.p * self.dt / self.mass) * self.dim
        self.evalForce(**kwargs)
        if(self.withMedaD and self.step >= self.startingStep + self.MetaDfreq):
            self.updateMetaD()
            self.dumpGaussiansPos()
        self.p = (self.p + 0.5 * (self.F + self.imgF) * self.dt) * self.dim

    def VVstep_NVT(self, **kwargs):
        Xi1 = np.random.randn(3)
        Xi2 = np.random.randn(3)
        self.p = np.exp(-self.gamma * self.dt /2) * self.p + np.sqrt(BOLTZMANN * self.temp * self.mass) * np.sqrt(1 - np.exp(-self.gamma * self.dt)) * Xi1 * self.dim
        self.VVstep_NVE(**kwargs)
        self.p = np.exp(-self.gamma * self.dt /2) * self.p + np.sqrt(BOLTZMANN * self.temp * self.mass) * np.sqrt(1 - np.exp(-self.gamma * self.dt)) * Xi2 * self.dim


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
        
        stepsLenght = []
        for i in range(int(self.Nsteps * 1.5 / self.MetaDfreq)):
            roll= np.ceil(np.random.exponential(self.MetaDfreq))
            stepsLenght.append(roll)

        self.stepsPos = [0] * len(stepsLenght)
        
        for i in range(len(stepsLenght)):
            if(i == 0):
                self.stepsPos[0] = self.startingStep + self.MetaDfreq
            else:
                self.stepsPos[i] = self.stepsPos[i-1] + stepsLenght[i]


        if(not self.notUseMB):
            self.sampleMB()
        self.evalForce(**kwargs)

        for self.step in range(self.Nsteps):
            self.evalMethod(**kwargs)
            self.CalcKinE()
            self.CalcTemp()
            self.E =  self.K + self.U + (self.Vbias).sum()

            if(self.step % self.printfreq == 0 and self.step >= self.startingStep):
                self.dumpXYZ()
                self.dumpThermo()
                self.dumpMoment()
                self.dumpForce()
                print(self.step)