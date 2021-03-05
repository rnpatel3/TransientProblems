# -*- coding: utf-8 -*-
"""
@author: rohan
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
def func(y, t):
    # m*dot(y[1]) = F = -mg + delta
    # dot(y[0]) = y[1]
    
    k = 1e6
    m = 2.0
    delta_vy = 0.0
    delta_vx = 0.0
    radius = 0.1
    mu = 0.05 #Need to find a good value for this parameter
    g = 9.81
    if y[0] <= radius:
        a = k*(1-0.8**2)/(0.8*(-np.abs(y[1]))) #0.8 Represents a "coeff of restitution"
        delta_vy = k*(radius - y[0]) + a*(radius - y[0])*y[1]
        delta_vx = (mu*delta_vy*y[3])
        
    ydot = np.zeros(4, dtype=y.dtype)
    #xdot = np.zeros(2, dtype=y.dtype)
    
    ydot[1] = -g + delta_vy/m      #Vertical accel
    ydot[0] = y[1]              #Vertical vel
    ydot[3] = -delta_vx      #Horizontal accel
    ydot[2] = y[3]      #Horizontal vel
    
#    xdot[1] = 
#    xdot[0] = 
    
    return ydot

def getForce(v,dt):
    deltaV = np.diff(v)
    force = np.divide(deltaV,dt)
    force = np.insert(force,0,0)
    return force
    
t0 = 0.0
y0 = np.array([2.0, -1.75, 0 ,0.5])
#x0 = np.array([0, 0.5])
t1 = 5
#iters = [1000,2000,5000,10000,20000,50000,100000,5000000,9000000]
iters = [5000000]
Forces = []

for N in iters:
    dt = (t1-t0)/N
    t = np.linspace(t0,t1,N)
    ysol = odeint(func,y0,t)
    yy = ysol[:,0]
    xx = ysol[:,2]
    force = getForce(ysol[:,1],dt)
    FirstBounce = np.max(force)
    Forces.append(FirstBounce)
    if N == 5000000:
        plt.figure(1)
        plt.plot(t,yy)
        plt.figure(3)
        plt.plot(t,ysol[:,1])
        plt.figure(4)
        plt.plot(t,force)
        plt.figure(5)
        plt.plot(xx,yy)
        plt.figure(6)
        plt.plot(t,ysol[:,3])



#plt.figure(2)
#plt.plot(np.divide(t1,iters),Forces)
#plt.xlabel("Time Step (s)")
#plt.ylabel("Force")
#plt.yscale('log')
#plt.xscale('log')