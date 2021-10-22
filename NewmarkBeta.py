#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 12:09:07 2021

@author: rpatel449
"""
import numpy as np
import matplotlib.pylab as plt

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
        #Need to apply ICs, but since they're 0, can ignore for now
        return
    
    def forward_integration(self):
        dt = self.t[1] - self.t[0]
        
        A = (1/(self.beta*(dt**2)))*self.M + (self.gamma/(self.beta*dt))*self.C + self.K
        invA = np.linalg.inv(A)

        self.xddot[0,:] = np.dot(np.linalg.inv(self.M),(self.u[:,0]-np.dot(self.C,self.xdot[0,:])-np.dot(self.K,self.x[0,:])))
        for i in range(0,self.N-1):
            B = self.u[:,i+1] + self.M @ ((1/(self.beta*dt**2))*self.x[i,:] + (1/(self.beta*dt))*self.xdot[i,:] + (1/(2*self.beta)-1)*self.xddot[i,:]) \
                                        + self.C @ ((self.gamma/(self.beta*dt))*self.x[i,:] \
                                        + (self.gamma/self.beta-1) * self.xdot[i,:] + (self.gamma/self.beta-2) * (dt/2) * self.xddot[i,:])
            self.x[i+1,:] = invA @ B
            
            self.xddot[i+1,:] = (1/(self.beta*dt**2)) * (self.x[i+1,:]-self.x[i,:]) - (1/(self.beta*dt)) * self.xdot[i,:] - ((1/(2*self.beta))-1) * self.xddot[i]
            self.xdot[i+1,:] = self.xdot[i,:] + (1-self.gamma) * dt * self.xddot[i,:] + self.gamma * dt * self.xddot[i+1,:]
            
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

u[2,:] = 50*pulse;

m1 = np.array([[10, 0, 0],[0, 20, 0],[0, 0, 30]])
k1 = 1e3* np.array([[45, -20, -15],[-20, 45, -25],[-15, -25, 40]])
c1 = 3e-2*k1

newmark = Newmark(t, x0, xdot0, u, m1, c1, k1)  

newmark.forward_integration()
newmark.visualize_disp()
