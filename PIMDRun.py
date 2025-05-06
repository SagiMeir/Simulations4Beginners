import os
os.chdir("c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners")
from sim import Simulation
import numpy as np
from scipy.constants import Boltzmann as BOLTZMANN
from scipy.constants import hbar

params = { "omega":  7.596E13 }
omega = 7.596E13
beads = 32
R = np.zeros((1,3,beads))
temp = 96.7
beadomega = np.sqrt(beads)*BOLTZMANN*temp/hbar
gamma = max(beadomega, omega)

# R[0,0,:]=np.array([0 for i in range(beads)])
#R[0,0,:] = np.array([[5,5,5,5,5,5,5,5,5,5]])

def one_sim():
    PIMD = Simulation(temp = temp ,beads =  beads, gamma = gamma, VVtype = "",dt = 0.1E-16, R = R *1E-10, Nsteps=300000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, xyzname = "c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/sim.xyz", outname = "c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/sim.log")
    PIMD.sampleMB()
    PIMD.run(**params)

def estim_7stops():
    for i in range(2,7):
       beads = 2**i
       R = np.zeros((1,3,beads))

       PIMD = Simulation(temp = 96.7 ,beads = beads,gamma = 1E14, VVtype = "_NVT",dt =  0.1E-16, R = R *1E-10, Nsteps= 1000000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, xyzname = "c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/sim.xyz", outname = "C:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/estimator2(better)/sim"+str(beads)+".log")
       PIMD.sampleMB()
       PIMD.run(**params)

def estim_diif_seed():
    for i in range(2,7):
       for seed in range (1,6):
            beads = 2**i
            R = np.zeros((1,3,beads))
            PIMD = Simulation(seed =seed,temp = 96.7 ,beads = beads,gamma = 1E14, VVtype = "_NVT",dt =  0.1E-16, R = R *1E-10, Nsteps= 1000000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, outname = "C:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/estimatorr in different seed/sim"+str(beads)+str(seed)+".log")
            PIMD.sampleMB()
            PIMD.run(**params)

def diff_temp_64beads():
    for seed in range (1,6):
        for i in range(1,7):
            temp = hbar*7.596E13/BOLTZMANN/i
            R = np.zeros((1,3,beads))
            PIMD = Simulation(seed =seed,temp = temp ,beads = 64,gamma = 1E14, VVtype = "_NVT",dt =  0.1E-16, R = R *1E-10, Nsteps= 1000000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, outname = "C:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/64beadsdifftemp/sim"+str(seed)+str(i)+".log")
            PIMD.sampleMB()
            PIMD.run(**params)

def run_Morse_try():
    mass = 6.633E-26
    a =1e-2
    i = 6
    # PIMD = Simulation(a = a, De = (omega**2*mass)/a**2/2,ftype = "Harm" ,temp = temp ,beads =  beads, gamma = gamma, VVtype = "_NVT",dt = 0.1E-16, R = R *1E-15, Nsteps=1000000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, xyzname = "c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/sim.xyz", outname = f"c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/morse potential/try/sim_harm.log")
    # PIMD.sampleMB()
    # PIMD.run(**params)
    # # j=1

    PIMD = Simulation(a = a, De = (omega**2*mass)/a**2/2,ftype = "Morse" ,temp = hbar*7.596E13/BOLTZMANN/i ,beads =  beads, gamma = gamma, VVtype = "_NVT",dt = 0.1E-16, R = R *1E-15, Nsteps=250000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, xyzname = "c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/sim.xyz", outname = f"c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/morse potential/try/sim{i}_long.log")
    PIMD.sampleMB()
    PIMD.run(**params)

def Morse_vs_harm():
    R[0,0,:] = np.array([np.array(1)]) 
    PIMD = Simulation(ftype = "Morse" ,temp = temp ,beads =  beads, gamma = gamma, VVtype = "",dt = 0.1E-16, R = R *1E-15, Nsteps=100000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, xyzname = "c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/sim.xyz", outname = "c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/morse potential/sim_morse.log")
    PIMD.run(**params)

    PIMD = Simulation(ftype = "Harm" ,temp = temp ,beads =  beads, gamma = gamma, VVtype = "",dt = 0.1E-16, R = R *1E-15, Nsteps=100000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, xyzname = "c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/sim.xyz", outname = "c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/morse potential/sim_harm.log")
    PIMD.run(**params)

def Morse_diff_De_diff_temp():
    mass = 6.633E-26
    diff_a = np.linspace(1E12, 1E14,10)

    i = 1
    for a in ([1]):
        for j in range(1,2):
            a=1E12
            temp = hbar*7.596E13/BOLTZMANN/1
            print(f"current sim number {i}, in temp {temp}")
            PIMD = Simulation(a =a , De = (omega**2*mass)/a**2/2, ftype = "Morse" ,temp = temp ,beads =  beads, gamma = gamma, VVtype = "_NVT",dt = 0.1E-16, R = R *1E-15, Nsteps = 400000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, xyzname = "c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/sim.xyz", outname = f"c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/morse potential/sim_Morse_{i}_{j}.log")
            PIMD.sampleMB()
            PIMD.run(**params)
        i+=1
        
def Morse_diff_De_diff_temp_diff_seed():#final a = linspace(1E-2, 1E6,5)
    mass = 6.633E-26
    diff_a = np.linspace(1E-2, 1E6,5)
    print(f"{a:.3e}" for a in diff_a)
    i = 1
    for a in diff_a:
        for j in range(1,7):
            temp = hbar*7.596E13/BOLTZMANN/j
            for seed in range(5):
                print(f"current sim number {i}, in temp {temp}")
                PIMD = Simulation(seed = seed,a = a , De = (omega**2*mass)/a**2/2, ftype = "Morse" ,temp = temp ,beads =  beads, gamma = gamma, VVtype = "_NVT",dt = 0.1E-16, R = R *1E-15, Nsteps = 500000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, xyzname = "c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/sim.xyz", outname = f"c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/morse potential/final/sim_Morse_{i}_{j}_{seed}_{a:.2e}_5_seeds_long.log")
                PIMD.sampleMB()
                PIMD.run(**params)

        i+=1

def final_diff_a(): #final a = np.linspace(1e7, 1e14,5)
    mass = 6.633E-26
    diff_a = np.linspace(1e7, 1e14,5)
    print(f"{a:.3e}" for a in diff_a)
    i = 1
    for a in diff_a:
        for j in range(1,7):
            temp = hbar*7.596E13/BOLTZMANN/j
            for seed in range(5):
                print(f"current sim number {i}, in temp {temp}")
                PIMD = Simulation(seed = seed,a = a , De = (omega**2*mass)/a**2/2, ftype = "Morse" ,temp = temp ,beads =  beads, gamma = gamma, VVtype = "_NVT",dt = 0.1E-16, R = R *1E-15, Nsteps = 500000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, xyzname = "c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/sim.xyz", outname = f"c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/morse potential/final_large_params/sim_Morse_{i}_{j}_{seed}_{a:.2e}_5_seeds_long.log")
                PIMD.sampleMB()
                PIMD.run(**params)

        i+=1

def final_array_of_a(): #final a = [1e-2,1e9,1e11,1e13,1e15]
    mass = 6.633E-26
    diff_a = [1e-2,1e9,1e11,1e13,1e15]
    print(f"{a:.3e}" for a in diff_a)
    i = 1
    for a in diff_a:
        for j in range(1,7):
            temp = hbar*7.596E13/BOLTZMANN/j
            for seed in range(5):
                print(f"current sim number {i}, in temp {temp}")
                PIMD = Simulation(seed = seed,a = a , De = (omega**2*mass)/a**2/2, ftype = "Morse" ,temp = temp ,beads =  beads, gamma = gamma, VVtype = "_NVT",dt = 0.1E-16, R = R *1E-15, Nsteps = 500000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, xyzname = "c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/sim.xyz", outname = f"c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/morse potential/final_1e-2_1e9_1e11_1e13_1e15/sim_Morse_{i}_{j}_{seed}_{a:.2e}_5_seeds_long.log")
                PIMD.sampleMB()
                PIMD.run(**params)

        i+=1

def last_one():
    mass = 6.633E-26
    diff_a = np.linspace(1e11,1e12,5)
    print(f"{a:.3e}" for a in diff_a)
    i = 1
    for a in diff_a:
        for j in range(1,7):
            temp = hbar*7.596E13/BOLTZMANN/j
            for seed in range(5):
                print(f"current sim number {i}, in temp {temp}")
                PIMD = Simulation(seed = seed,a = a , De = (omega**2*mass)/a**2/2, ftype = "Morse" ,temp = temp ,beads =  beads, gamma = gamma, VVtype = "_NVT",dt = 0.1E-16, R = R *1E-15, Nsteps = 500000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, xyzname = "c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/sim.xyz", outname = f"c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/morse potential/last_one_1e11to1e12/sim_Morse_{i}_{j}_{seed}_{a:.2e}_5_seeds_long.log")
                PIMD.sampleMB()
                PIMD.run(**params)

        i+=1

def smalla1to10():
    mass = 6.633E-26
    diff_a = np.linspace(1,10,5)
    print(f"{a:.3e}" for a in diff_a)
    i = 1
    for a in diff_a:
        for j in range(1,7):
            temp = hbar*7.596E13/BOLTZMANN/j
            for seed in range(5):
                print(f"current sim number {i}, in temp {temp}, a = {a}")
                PIMD = Simulation(seed = seed,a = a , De = (omega**2*mass)/(a**2/2), ftype = "Morse" ,temp = temp ,beads =  beads, gamma = gamma, VVtype = "_NVT",dt = 0.1E-16, R = R *1E-15, Nsteps = 500000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, xyzname = "c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/sim.xyz", outname = f"c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/morse potential/smalla_1to10/sim_Morse_{i}_{j}_{seed}_{a:.2e}_5_seeds_long.log")
                PIMD.sampleMB()
                PIMD.run(**params)

        i+=1

def diffa():
    mass = 6.633E-26
    diff_a = [1e-2,1e-1,1,10,100]
    print(f"{a:.3e}" for a in diff_a)
    i = 1
    for a in diff_a:
        for j in range(1,7):
            temp = hbar*7.596E13/BOLTZMANN/j
            for seed in range(5):
                De = (omega**2*mass)/(2*a**2)
                print(f"current sim number {i}, in temp {temp}, a = {a}")
                print(De*a**2)
                PIMD = Simulation(seed = seed,a = a ,De = De , ftype = "Morse" ,temp = temp ,beads =  beads, gamma = gamma, VVtype = "_NVT",dt = 0.1E-16, R = R *1E-15, Nsteps = 500000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, xyzname = "c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/sim.xyz", outname = f"c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/morse potential/diffa/sim_Morse_{i}_{j}_{seed}_{a:.2e}_5_seeds_long.log")
                PIMD.sampleMB()
                PIMD.run(**params)

        i+=1

def a_1e10to1e11():
    mass = 6.633E-26
    diff_a = np.linspace(1e10,1e11,5)
    print(f"{a:.3e}" for a in diff_a)
    i = 1
    for a in diff_a:
        for j in range(1,7):
            temp = hbar*7.596E13/BOLTZMANN/j
            for seed in range(5):
                De = (omega**2*mass)/(2*a**2)
                print(f"current sim number {i}, in temp {temp}, a = {a}")
                print(De*a**2)
                PIMD = Simulation(seed = seed,a = a ,De = De , ftype = "Morse" ,temp = temp ,beads =  beads, gamma = gamma, VVtype = "_NVT",dt = 0.1E-16, R = R *1E-15, Nsteps = 500000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, xyzname = "c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/sim.xyz", outname = f"c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/morse potential/diff_1e10to1e11/sim_Morse_{i}_{j}_{seed}_{a:.2e}_5_seeds_long.log")
                PIMD.sampleMB()
                PIMD.run(**params)

        i+=1

def a_1e10to05e11():
    mass = 6.633E-26
    diff_a = np.linspace(1e10,0.5e11,5)
    print(f"{a:.3e}" for a in diff_a)
    i = 1
    for a in diff_a:
        for j in range(1,2):
            temp = hbar*7.596E13/BOLTZMANN/j
            for seed in range(1):
                De = (omega**2*mass)/(2*a**2)
                print(f"current sim number {i}, in temp {temp}, a = {a}")
                print(De)
                #PIMD = Simulation(seed = seed,a = a ,De = De , ftype = "Morse" ,temp = temp ,beads =  beads, gamma = gamma, VVtype = "_NVT",dt = 0.1E-16, R = R *1E-15, Nsteps = 500000, mass = 6.633E-26, kind = ["Ar"], fac = 1E10, xyzname = "c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/sim.xyz", outname = f"c:/Users/elira/OneDrive/Documents/GitHub/Simulations4Beginners/morse potential/diff_a_1e10to0.5e11/sim_Morse_{i}_{j}_{seed}_{a:.2e}_5_seeds_long.log")
                # PIMD.sampleMB()
                # PIMD.run(**params)

        i+=1



a_1e10to05e11()
print("----------------------------DONE----------------------------")
