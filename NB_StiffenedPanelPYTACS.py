#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 12:09:07 2021

@author: rpatel449
"""

import numpy as np
import sys
import os
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pylab as plt
from mpi4py import MPI
from tacs import TACS, elements, constitutive, functions, pyTACS

# Load structural mesh from BDF file
tacs_comm = MPI.COMM_WORLD

#struct_mesh = TACS.MeshLoader(tacs_comm)
bdfFile = os.path.join(os.path.dirname(__file__), 'stiffPanel4.dat')
struct_mesh = pyTACS(bdfFile, tacs_comm)
#struct_mesh.scanBDFFile("axial_stiffened_panel.bdf")


# structOptions = {
#     'printtimings':True,
#     # Specify what type of elements we want in the f5
#     'writeSolution':True,
#     'outputElement': TACS.PLANE_STRESS_ELEMENT,
# }

# bdfFile = os.path.join(os.path.dirname(__file__), 'circ-plate-dirichlet-bcs.bdf')
# struct_mesh = pyTACS(bdfFile, tacs_comm, options=structOptions)

def elemCallBack(dvNum, compID, compDescript, elemDescripts, specialDVs, **kwargs):
    # Material properties
    rho = 2500.0        # density kg/m^3
    E = 70e9            # Young's modulus (Pa)
    nu = 0.3            # Poisson's ratio
    ys = 464.0e6        # yield stress

    # Plate geometry
    tplate = 0.005    # 1 mm
    tMin = 0.0001    # 0.1 mm
    tMax = 0.05     # 5 cm

    # Set up property model
    prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)
    # Set up constitutive model
    con = constitutive.IsoShellConstitutive(prop, t=tplate, tNum=dvNum, tlb=tMin, tub=tMax)
    transform = None
    # Set up element
    elem = elements.Quad4Shell(transform, con)
    scale = [100.0]
    return elem, scale

assembler = struct_mesh.createTACSAssembler(elemCallBack)

# Loop over components, creating stiffness and element object for each
num_components = struct_mesh.getNumComponents()

'''
Previous method of importing geometry and setting up stiffness/elements
'''
# for i in range(num_components):
#     descriptor = struct_mesh.getElementDescript(i)
#     print(descriptor)
#     # Setup (isotropic) property and constitutive objects
#     prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)
#     # Set one thickness dv for every component
#     stiff = constitutive.IsoShellConstitutive(prop, t=thickness, tMin=min_thickness, tMax=max_thickness, tNum=i)

#     element = None
#     transform = None
#     if descriptor in ["CQUAD", "CQUADR", "CQUAD4"]:
#         element = elements.Quad4Shell(transform, stiff)
#     elif descriptor in ["CQUAD9"]:
#         element = elements.Quad9Shell(transform, stiff)
#     struct_mesh.setElement(i, element)

# # Create tacs assembler object from mesh loader
# assembler = struct_mesh.createTACS(6)


# Get the design variable values
x = assembler.createDesignVec()
x_array = x.getArray()
assembler.getDesignVars(x)

# Get the node locations
X = assembler.createNodeVec()
assembler.getNodes(X)
assembler.setNodes(X)

# # Create the forces
forces = assembler.createVec()
force_array = forces.getArray() 
force_array[2::6] += 100.0 # uniform load in z direction
assembler.applyBCs(forces)


class Newmark(pyTACS.TransientProblem):
    def __init__(self, t, x0, xdot0, u, m1, c1, k1, beta = 0.25, gamma = 0.5) :
        self.t = t
        self.M = m1
        self.C = c1
        self.K = k1
        self.u = u
        self.x0 = x0
        self.xdot0 = xdot0
        self.N = len(t)
        self.beta = beta
        self.gamma = gamma
        #self.x = np.zeros((len(t),3))
        self.x = []
        self.xdot = []
        self.xddot = []
        for i in range(len(t)):
            self.x.append(assembler.createVec())
            self.xdot.append(assembler.createVec())
            self.xddot.append(assembler.createVec())

        #self.xdot = np.zeros((len(t),3))
        #self.xddot = np.zeros((len(t),3))
        self.newton_iters = 100
        self.ntol = 1e-9
        #Need to apply ICs, but since they're 0, can ignore for now
        
        
        return
    

    def forward_integration(self):
        # dt =  self.t[1] - self.t[0]
        dt = 0.1 #Fixed val for now
        # res = np.zeros((3,1))
        # J = np.zeros((3, 3))
        res = assembler.createVec()
        J = assembler.createSchurMat()
        
        # forces = assembler.createVec()
        # forces_array = forces.getArray()
        # # print(forces_array)
        # # forces_array[2::5] = 100
        # forces_array[1::6] = -10
        # #forces_array[1202:1322:6] = 100
        # print(forces_array)
        # assembler.applyBCs(forces)        
        t = self.t
        
        # Create the force vector
        forces = assembler.createVec()
        temp = assembler.createVec()
        
        # Set the compressive force
        # forces_array = forces.getArray()
        # forces_array[1::6] = -10
        # assembler.applyBCs(forces)

        for i in range(0, self.N-1):
            #u = np.ones((1,3)) #Some estimate of u[i+1]
            
            # if i < 2:
            #         # Initial Perturbation force out of plane
            #     forces_array = forces.getArray()
            #     forces_array[1::6] = -10.0
            #     forces_array[1202:1322:6] = 1000.0
            #     assembler.applyBCs(forces)
            # else:
            #     forces_array = forces.getArray()
            #     forces_array[1::6] = -10.0
            #     forces_array[1202:1322:6] = 0.0
            #     assembler.applyBCs(forces)
            
            # force_arr = forces.getArray()
            # print(force_arr)
            
            u = assembler.createVec()
            udot = assembler.createVec()
            uddot = assembler.createVec()
            
            # uddot = (u - self.x[i])/(self.beta*dt**2) - self.xdot[i]/(self.beta*dt) - (1/(2*self.beta) - 1)* self.xddot[i] #Eq 5.42 based on this estimate
            u.axpy(-1.0, self.x[i])
            u.scale(1.0/self.beta*dt**2)
            u.axpy(-1.0/(self.beta*dt), self.xdot[i])
            u.axpy(-(1.0/(2*self.beta) - 1.0), self.xddot[i])
            uddot.axpy(1,u)
            
            u = assembler.createVec() # Reset u for next calculation
            
            #udot = self.gamma*(u - self.x[i,:])/(self.beta*dt) + self.xdot[i,:]*(1-self.gamma/self.beta) + dt*self.xddot[i,:]*(1-self.gamma/(2*self.beta)) #Eq. 5.43 based on estimate
            u.axpy(-1.0, self.x[i])
            u.scale(self.gamma/(self.beta*dt))
            u.axpy((1.0-self.gamma/self.beta), self.xdot[i])
            u.axpy((dt*(1-self.gamma/(2*self.beta))), self.xddot[i])
                   
            u = assembler.createVec() #Reset u for the last time
            #force = np.reshape(self.u[:,i+1], (1,3)) #Need to handle forces somehow into the "residual" again

            for j in range(self.newton_iters):
                update = assembler.createVec()
                tacs_alpha = 1.0
                tacs_beta = self.gamma/(self.beta*dt) # Eq 5.43
                tacs_gamma = 1.0/(self.beta*dt**2) # Eq 5.42    
                
                # assembler.assembleJacobian(tacs_alpha, tacs_beta, tacs_gamma,  force, res, J)
                # rnorm = np.sqrt(np.dot(res.flatten(),res.flatten()))
                assembler.assembleJacobian(tacs_alpha, tacs_beta, tacs_gamma, res, J)
                pc = TACS.Pc(J)
                pc.factor()
                gmres_iters = 5
                nrestart = 2
                is_flexible = 1
                gmres = TACS.KSM(J, pc, gmres_iters, nrestart, is_flexible)
                
                if i < 5:
                    res.axpy(-t[i], forces)
                else:
                    res.axpy(-0.5, forces)
                
                if res.norm() < self.ntol:
                    break

                
                #gmres.setMonitor(comm, freq=1)
                gmres.solve(res, update)
                
                #Apply updates here
                u.axpy(-1.0, update)
                udot.axpy(-tacs_beta, update)
                uddot.axpy(-tacs_gamma, update)
                assembler.setVariables(u,udot,uddot)

            #Store update to u, udot, uddot

            self.x[i+1] = u
            self.xdot[i+1] = udot
            self.xddot[i+1] = uddot


        return self.x

    def visualize_disp(self):
        # fig, ax = plt.subplots(1)
        # #Edit displacement visualization
        # ax.plot(t, self.x[1].getArray())
        # plt.xlabel('time (s)')
        # plt.ylabel('displacement (m)')
        # plt.show()
        
        for i in range(len(self.t)):
            assembler.setVariables(self.x[i], self.xdot[i], self.xddot[i])
            f5.writeToFile('stiffened_panel_pytacs%d.f5'%(i))
        return
    

t = np.linspace(0,1.0,11)
#x0 = np.array([0, 0, 0])
#xdot0 = np.array([0, 0, 0])
x0 =  assembler.createVec()
xdot0 = assembler.createVec()

#Create space to set force time-history
u = []
for i in range(len(t)):
    u.append(assembler.createVec())
#u = np.zeros((3,len(t)))
pulse  =  np.zeros(len(t))
omega = np.pi/.29
#u[2,:] = 50*np.sin(omega*t)
for j in range(0,len(t)):
    pulse[j] = np.sin(omega*t[j])
    if t[j] > np.pi/omega:
        pulse[j] = 0

# u[2,:] = 50*pulse #Need to find a way to add loads

#Create Stiffness, Mass, and Damping Matrices
m1 = assembler.createMat()
assembler.assembleMatType(TACS.MASS_MATRIX, m1)
k1 = assembler.createMat()
assembler.assembleMatType(TACS.STIFFNESS_MATRIX, k1)
c1 = assembler.createMat()
c1.scale(3e-2)

# m1 = np.array([[10, 0, 0],[0, 20, 0],[0, 0, 30]])
# k1 = 1e3* np.array([[45, -20, -15],[-20, 45, -25],[-15, -25, 40]])
# c1 = 3e-2*k1

newmark = Newmark(t, x0, xdot0, u, m1, c1, k1)
newmark.solve()
flag = (TACS.OUTPUT_CONNECTIVITY |
        TACS.OUTPUT_NODES |
        TACS.OUTPUT_DISPLACEMENTS |
        TACS.OUTPUT_STRAINS)
f5 = TACS.ToFH5(assembler, TACS.BEAM_OR_SHELL_ELEMENT, flag)
# f5.writeToFile('shell_dyn.f5')

newmark.visualize_disp()
