#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 12:09:07 2021

@author: rpatel449
"""

import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pylab as plt

from mpi4py import MPI
from tacs import TACS, elements, constitutive, functions

'''
Create TACS Assembler
'''
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size
nx = 20  # number of elements in x direction
ny = 20  # number of elements in y direction
Lx = 1.0
Ly = 1.0
varsPerNode = 6
    
nodesPerProc = int((nx+1)*(ny+1)/size)
elemsPerProc = int(nx*ny/size)
numOwnedNodes = int(nodesPerProc)
numElements = int(elemsPerProc)
numDependentNodes = 0

# Adjust for the last processor
if (rank == size-1):
    numOwnedNodes = (nx+1)*(ny+1) - nodesPerProc*(size-1)
    numElements = nx*ny - elemsPerProc*(size-1)

assembler = TACS.Assembler.create(comm, varsPerNode,
                                  numOwnedNodes, numElements,
                                  numDependentNodes)

'''
Setup geometry, mesh, material and element
'''

# Set up partition
firstElem = rank*elemsPerProc
firstNode = rank*nodesPerProc
lastElem = (rank+1)*elemsPerProc
lastNode = (rank+1)*nodesPerProc
if (rank == size-1):
    lastElem = nx*ny
    lastNode = (nx+1)*(ny+1)

# Populate connectivity
ptr = np.zeros(numElements + 1, dtype=np.int32)
conn = np.zeros(4*numElements, dtype=np.int32)
ptr[0] = 0
k = 0
for elem in range(firstElem, lastElem):
    i = elem % nx
    j = elem // nx
    conn[4*k] =   i   + j*(nx+1)
    conn[4*k+1] = i+1 + j*(nx+1)
    conn[4*k+2] = i   + (j+1)*(nx+1)
    conn[4*k+3] = i+1 + (j+1)*(nx+1)
    ptr[k+1] = 4*(k+1)
    k += 1

# Set the connectivity
assembler.setElementConnectivity(ptr, conn)

# Create the isotropic material class
props = constitutive.MaterialProperties(rho=2700.0, E=70e9, nu=0.3, ys=270.0)

# Create basis, constitutive, element, etc
linear_basis = elements.LinearQuadBasis()
stiff = constitutive.PlaneStressConstitutive(props)
elements_list = []
plate_elements = []
for elem in range(firstElem, lastElem):
    stiff = constitutive.PlaneStressConstitutive(props, 1.0, elem)
    model = elements.LinearElasticity2D(stiff);
    elements_list.append(elements.Element2D(model, linear_basis))

    plate_stiff = constitutive.IsoShellConstitutive(props)
    plate_model = elements.PlateModel(plate_stiff)
    plate_elements.append(elements.Quad4Shell(None, plate_stiff))

# Set elements into the mesh
assembler.setElements(plate_elements)

# Set boundary conditions
for i in range(0, nx + 1):
    # Here nodal indexing is global
    nodes = np.array([i], dtype=np.int32)
    dof = np.array([0, 1, 2], dtype=np.int32)
    values = np.array([0.0, 0.0, 0.0])
    assembler.addBCs(nodes, dof, values)
    
for i in range(0, nx + 1):
    # Here nodal indexing is global
    nodes = np.array([i+ny*(nx+1)], dtype=np.int32)
    dof = np.array([2], dtype=np.int32)
    values = np.array([0.0])
    assembler.addBCs(nodes, dof, values)



# Done adding elements
assembler.initialize()

# Create the node location vector
X = assembler.createNodeVec()
Xpts = X.getArray()

# Get nodal locations
k = 0
coeff = 0.05
for node in range(firstNode, lastNode):
    i = node % (nx + 1)
    j = node // (nx + 1)
    Xpts[k] = i*Lx/nx
    Xpts[k+1] = j*Ly/ny
    Xpts[k+2] = np.sin(Xpts[k+1]/Lx*np.pi)*coeff*Lx
    k += 3

assembler.reorderVec(X)  # Might not needed since we don't reorder the matrix
assembler.setNodes(X)

# comm = MPI.COMM_WORLD

# # Create the stiffness object
# props = constitutive.MaterialProperties(rho=2570.0, E=70e9, nu=0.3, ys=350e6)
# stiff = constitutive.IsoShellConstitutive(props)

# # Set up the basis function
# model = elements.PlateModel(stiff)
# basis = elements.LinearQuadBasis()
# elem = elements.Quad4Shell(None, stiff)

# # Allocate the TACSCreator object
# #varsPerNode = model.getVarsPerNode()
# varsPerNode = 6
# creator = TACS.Creator(comm, varsPerNode)

# if comm.rank == 0:
#     # Create the elements
#     nx = 10
#     ny = 10
    
#     # Set the nodes
#     nnodes = (2*nx+1)*(2*ny+1)
#     nelems = 2*nx*ny
#     nodes = np.arange(nnodes).reshape((2*nx+1, 2*ny+1))
    
#     conn = []
#     for j in range(ny):
#         for i in range(nx):
#             # Append the first set of nodes
#             conn.append([nodes[2*i, 2*j],
#                          nodes[2*i+2, 2*j],
#                          nodes[2*i+2, 2*j+2],
#                          nodes[2*i+1, 2*j],
#                          nodes[2*i+2, 2*j+1],
#                          nodes[2*i+1, 2*j+1]])
            
#             # Append the second set of nodes
#             conn.append([nodes[2*i, 2*j+2],
#                          nodes[2*i, 2*j],
#                          nodes[2*i+2, 2*j+2],
#                          nodes[2*i, 2*j+1],
#                          nodes[2*i+1, 2*j+1],
#                          nodes[2*i+1, 2*j+2]])

#     # Set the node pointers
#     conn = np.array(conn, dtype=np.intc).flatten()
#     ptr = np.arange(0, 6*nelems+1, 6, dtype=np.intc)
#     elem_ids = np.zeros(nelems, dtype=np.intc)
#     creator.setGlobalConnectivity(nnodes, ptr, conn, elem_ids)

#     # Set up the boundary conditions
#     bcnodes = np.array(nodes[0,:], dtype=np.intc)

#     # Set the boundary condition variables
#     nbcs = 2*bcnodes.shape[0]
#     bcvars = np.zeros(nbcs, dtype=np.intc)
#     bcvars[:nbcs:2] = 0
#     bcvars[1:nbcs:2] = 1

#     # Set the boundary condition pointers
#     bcptr = np.arange(0, nbcs+1, 2, dtype=np.intc)
#     creator.setBoundaryConditions(bcnodes, bcvars, bcptr)

#     # Set the node locations
#     Xpts = np.zeros(3*nnodes)
#     x = np.linspace(0, 10, 2*nx+1)
#     y = np.linspace(0, 10, 2*nx+1)
#     for j in range(2*ny+1):
#         for i in range(2*nx+1):
#             Xpts[3*nodes[i,j]] = x[i]
#             Xpts[3*nodes[i,j]+1] = y[j]
            
#     # Set the node locations
#     creator.setNodes(Xpts)

# # Set the elements
# elements = [ elem ]
# creator.setElements(elements)

# # Create the tacs assembler object
# assembler = creator.createTACS()

# res = assembler.createVec()
# ans = assembler.createVec()
# # mat = assembler.createSchurMat()
# mat = assembler.createMat()


class Newmark():
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
        #assembler = Assembler(self.M, self.C, self.K)
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
        
        # Set the compressive force
        forces_array = forces.getArray()
        #forces_array[1::6] = -10
        forces_array[2281::6] = -250000000.0 #Add one more zero for complete collapse of structure
        assembler.applyBCs(forces)

        for i in range(0, self.N-1):
            #u = np.ones((1,3)) #Some estimate of u[i+1]
            
            # if i < 3:
            #         # Initial Perturbation force out of plane
            #     forces_array = forces.getArray()
            #     #forces_array[1::6] = -10000.0
            #     forces_array[2281::6] = -2500000.0
            #     forces_array[1202:1322:6] = 10000.0
            #     assembler.applyBCs(forces)
            # else:
            #     forces_array = forces.getArray()
            #     #forces_array[1::6] = -10000.0
            #     forces_array[2281::6] = -2500000.0
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
                
                res.axpy(-t[i], forces)
                
                # if rnorm.all() < self.ntol:
                #     break
            
                if res.norm() < self.ntol:
                    break

                
                #gmres.setMonitor(comm, freq=1)
                gmres.solve(res, update)
                
                #Apply updates here
                u.axpy(-1.0, update)
                udot.axpy(-tacs_beta, update)
                uddot.axpy(-tacs_gamma, update)
                assembler.setVariables(u,udot,uddot)

                # u -= np.transpose(update)
                # udot -= tacs_beta*np.transpose(update)
                # uddot -= tacs_gamma*np.transpose(update)

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
            f5.writeToFile('panel_test_dyn_initialDef_lessF%d.f5'%(i))
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
newmark.forward_integration()
flag = (TACS.OUTPUT_CONNECTIVITY |
        TACS.OUTPUT_NODES |
        TACS.OUTPUT_DISPLACEMENTS |
        TACS.OUTPUT_STRAINS)
f5 = TACS.ToFH5(assembler, TACS.PLANE_STRESS_ELEMENT, flag)


newmark.visualize_disp()
