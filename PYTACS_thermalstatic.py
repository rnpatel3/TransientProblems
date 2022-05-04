"""
The nominal case is a heat conduction problem of a
1m radius plate with a Dirichilet boundary condition applied at the edges,
such that:
    T(theta) = T0 + dT * sin(2*theta)
    T0 = 70 C
    dT = 30 C
The problem is then to solve for the temperature within the boundaries of the plate.
The problem basically boils down to Laplaces problem:
    grad**2 T = 0

This example runs to problem cases:
    1. A transient problem where the bc's are applied time t=0
    2. A static problem that represents the steady state solution of the above at t=infinty
"""
# ==============================================================================
# Standard Python modules
# ==============================================================================
from __future__ import print_function
import os

# ==============================================================================
# External Python modules
# ==============================================================================
from pprint import pprint

import numpy as np
from mpi4py import MPI

# ==============================================================================
# Extension modules
# ==============================================================================
from tacs import functions, constitutive, elements, TACS, pyTACS, problems

comm = MPI.COMM_WORLD

# Instantiate FEAAssembler
structOptions = {
    'printtimings':True,
    # Specify what type of elements we want in the f5
    #'writeSolution':True,
    #'outputElement': TACS.PLANE_STRESS_ELEMENT,
}

bdfFile = os.path.join(os.path.dirname(__file__), 'stiffPanel7.dat')
FEAAssembler = pyTACS(bdfFile, comm, options=structOptions)

def elemCallBack(dvNum, compID, compDescript, elemDescripts, globalDVs, **kwargs):
    # Material properties
    rho = 2500.0        # density kg/m^3
    kappa = 230.0       # Thermal conductivity W/(m⋅K)
    specificHeat = 921.0 # Specific heat J/(kg⋅K)
    alpha = 8.6e-6       #CTE Titanium 6Al-4V
    E = 70e9            # Young's modulus (Pa)
    nu = 0.3            # Poisson's ratio
    ys = 464.0e6        # yield stress

    # Plate geometry
    tplate = 0.005    # 1 mm
    tMin = 0.0001    # 0.1 mm
    tMax = 0.05     # 5 cm

    # Setup property and constitutive objects
    prop = constitutive.MaterialProperties(rho=rho, kappa=kappa, specific_heat=specificHeat, E=E, nu=nu, ys=ys)
    # Set one thickness dv for every component
    con = constitutive.IsoShellConstitutive(prop, t=tplate, tNum=dvNum, tlb=tMin, tub=tMax)
    transform = None

    # For each element type in this component,
    # pass back the appropriate tacs element object
    elemList = []
    #model = elements.HeatConduction3D(con)
    for elemDescript in elemDescripts:
        if elemDescript in ['CQUAD4', 'CQUADR']:
            #basis = elements.LinearQuadBasis()
            elem = elements.Quad4NonlinearThermalShell(transform, con)
        elif elemDescript in ['CTRIA3', 'CTRIAR']:
            basis = elements.LinearTriangleBasis()
        else:
            print("Uh oh, '%s' not recognized" % (elemDescript))
        #elem = elements.Quad4ThermalShell(model, basis)
        elemList.append(elem)

    # Add scale for thickness dv
    scale = [100.0]
    return elemList, scale

# Set up constitutive objects and elements
FEAAssembler.initialize(elemCallBack)
transientOptions = {'printlevel':1}
# Setup problems
# Create a transient problem that will represent time varying convection
#transientProb = FEAAssembler.createTransientProblem('TransientThermCoarse', tInit=0.0, tFinal=5.0, numSteps=51, options=transientOptions)
# Create a static problem that will represent the steady state solution
staticProb = FEAAssembler.createStaticProblem(name='NonlinThermalStatic_20imp')
# Add both problems to a list
allProblems = []



# Add functions to each problem
staticProb.addFunction('mass', functions.StructuralMass)
staticProb.addFunction('ks_temp', functions.KSTemperature,
                        ksWeight=100.0)


bdfInfo = FEAAssembler.getBDFInfo()
# cross-reference bdf object to use some of pynastrans advanced features
bdfInfo.cross_reference()
# Loop through each element and get the spatial variation of the pressure
Pxy = []
eIDs = []
nIDs = []
fhz = 1.0
L = 3.0

displ_mag = []
for nID in bdfInfo.nodes:
    #[x,y,z] = bdfInfo.nodes[nID]
    #displ = 2 * np.sin(2* np.pi*z/L)
    #displ_mag.append(displ)
    nIDs.append(nID)


#Trying to add a slight initial geom imperfection

X = staticProb.getNodes()
Xpts = X
Lz = 3.0
Xpts[1::3] += 0.2*np.sin(np.pi*Xpts[2::3]/Lz)

staticProb.setNodes(X)


F = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.5e-2])

staticProb.addLoadToNodes(nIDs, F, nastranOrdering=True)

allProblems.append(staticProb)

# Solve state for each problem, evaluate functions and sensitivities
funcs = {}
funcsSens = {}
for problem in allProblems:
    problem.solve()
    #problem.evalFunctions(funcs)
    #problem.evalFunctionsSens(funcsSens)
    problem.writeSolution()

#Don't care about functions, sens for now
#if comm.rank == 0:
#    pprint(funcs)
#    pprint(funcsSens) Don't need sensitivities for now

""" if comm.rank == 0:
    #Create X and Y Vectors
    x = np.zeros((994, 20))
    y = np.zeros((994, 20))

    #Start placing values into x and y vecs
    for j in range(20):
        if j > 0:
            y[:,j] = q
        t, q, qdot, qddot = problem.getVariables(j) #Returns numpy array
        x[:,j] = q

    #Start performing SVD
    u,s,v = np.linalg.svd(x)
    u = u[:,:20] #Trim U to account for nonsquare matrices SVD
    sigma = np.diag(s)
    pprint(s)
    #Compute pseudoinverse of SIGMA
    sigma_inv = np.zeros((20,20))
    for k in range(20):
        if s[k] > 0:
            sigma_inv[k,k] = s[k]
    #sigma_inv = np.linalg.inv(sigma)

    #A_approx = np.dot(np.transpose(u), np.dot(y, np.dot(np.transpose(v), sigma_inv)))
    A_approx = np.transpose(u)@y@v@sigma_inv
    # pprint(u.shape)
    # pprint(y.shape)
    # pprint(v.shape)
    # pprint(sigma_inv.shape)
    lam, w = np.linalg.eigh(A_approx)
    phi = np.dot(u,w)
    pprint(lam)
    exit(0)
     """
