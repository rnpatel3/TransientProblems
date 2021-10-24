#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 12:09:07 2021

@author: rpatel449
"""

import numpy as np
import matplotlib.pylab as plt

class Assembler():
    def __init__(self, M, C, K):
        self.u = np.zeros((3,1))
        self.udot = np.zeros((3,1))
        self.uddot = np.zeros((3,1))
        self.M = M
        self.C = C
        self.K = K

    def setVariables(self, u, udot, uddot):
        self.u[:] = np.transpose(u)
        self.udot[:] = np.transpose(udot)
        self.uddot[:] = np.transpose(uddot)
        return
    
    def assembleJacobian(self, alpha, beta, gamma, force, res, mat):
        res[:] = np.dot(self.K, self.u) + np.dot(self.C, self.udot) + np.dot(self.M, self.uddot) - np.transpose(force)
        mat[:] = alpha*self.K + beta*self.C + gamma*self.M
        return 




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
        self.x = np.zeros((len(t),3))
        self.xdot = np.zeros((len(t),3))
        self.xddot = np.zeros((len(t),3))
        self.newton_iters = 100
        self.ntol = 1e-9
        #Need to apply ICs, but since they're 0, can ignore for now
        return
    
#    def forward_integration_orig(self):
#         dt = self.t[1] - self.t[0]
#        
#         A = (1/(self.beta*(dt**2)))*self.M + (self.gamma/(self.beta*dt))*self.C + self.K
#         invA = np.linalg.inv(A)
#
#         self.xddot[0,:] = np.dot(np.linalg.inv(self.M),(self.u[:,0]-np.dot(self.C,self.xdot[0,:])-np.dot(self.K,self.x[0,:])))
#         for i in range(0,self.N-1):
#             B = self.u[:,i+1] + self.M @ ((1/(self.beta*dt**2))*self.x[i,:] + (1/(self.beta*dt))*self.xdot[i,:] + (1/(2*self.beta)-1)*self.xddot[i,:]) \
#                                         + self.C @ ((self.gamma/(self.beta*dt))*self.x[i,:] \
#                                         + (self.gamma/self.beta-1) * self.xdot[i,:] + (self.gamma/self.beta-2) * (dt/2) * self.xddot[i,:])
#             self.x[i+1,:] = invA @ B
#            
#             self.xddot[i+1,:] = (1/(self.beta*dt**2)) * (self.x[i+1,:]-self.x[i,:]) - (1/(self.beta*dt)) * self.xdot[i,:] - ((1/(2*self.beta))-1) * self.xddot[i]
#             self.xdot[i+1,:] = self.xdot[i,:] + (1-self.gamma) * dt * self.xddot[i,:] + self.gamma * dt * self.xddot[i+1,:]
#            
#         return self.x

    def forward_integration(self):
        dt =  self.t[1] - self.t[0]
        assembler = Assembler(self.M, self.C, self.K)
        res = np.zeros((3,1))
        J = np.zeros((3, 3))

        for i in range(0, self.N-1):
            u = np.ones((1,3)) #Some estimate of u[i+1]
            uddot = (u - self.x[i,:])/(self.beta*dt**2) - self.xdot[i,:]/(self.beta*dt) - (1/(2*self.beta) - 1)* self.xddot[i,:] #Eq 5.42 based on this estimate
            udot = self.gamma*(u - self.x[i,:])/(self.beta*dt) + self.xdot[i,:]*(1-self.gamma/self.beta) + dt*self.xddot[i,:]*(1-self.gamma/(2*self.beta)) #Eq. 5.43 based on estimate
            force = np.reshape(self.u[:,i+1], (1,3))

            for j in range(self.newton_iters):
                assembler.setVariables(u, udot, uddot)

                tacs_alpha = 1.0
                tacs_beta = self.gamma/(self.beta*dt) # Eq 5.43
                tacs_gamma = 1.0/(self.beta*dt**2) # Eq 5.42    

                assembler.assembleJacobian(tacs_alpha, tacs_beta, tacs_gamma,  force, res, J)
                rnorm = np.sqrt(np.dot(res.flatten(),res.flatten()))
                
                if rnorm.all() < self.ntol:
                    break

                update = np.linalg.solve(J, res)

                u -= np.transpose(update)
                udot -= tacs_beta*np.transpose(update)
                uddot -= tacs_gamma*np.transpose(update)

            #Store update to u, udot, uddot

            self.x[i+1,:] = u
            self.xdot[i+1,:] = udot
            self.xddot[i+1,:] = uddot

        return self.x

    def visualize_disp(self):
        fig, ax = plt.subplots(1)
        ax.plot(t, self.x[:,:])
        plt.xlabel('time (s)')
        plt.ylabel('displacement (m)')
        plt.show()
        
        return
    

t = np.linspace(0,3.0,601)
x0 = np.array([0, 0, 0])
xdot0 = np.array([0, 0, 0])

u = np.zeros((3,len(t)))
pulse  =  np.zeros(len(t))
omega = np.pi/.29
#u[2,:] = 50*np.sin(omega*t)
for j in range(0,len(t)):
    pulse[j] = np.sin(omega*t[j])
    if t[j] > np.pi/omega:
        pulse[j] = 0

u[2,:] = 50*pulse

m1 = np.array([[10, 0, 0],[0, 20, 0],[0, 0, 30]])
k1 = 1e3* np.array([[45, -20, -15],[-20, 45, -25],[-15, -25, 40]])
c1 = 3e-2*k1

newmark = Newmark(t, x0, xdot0, u, m1, c1, k1)  
xout = np.zeros((len(t),3))
newmark.forward_integration()
newmark.visualize_disp()
