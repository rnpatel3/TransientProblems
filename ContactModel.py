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
    
    k = 1e8
    m = 2.0
    delta = 0.0
    radius = 0.1
    g = 9.81
    if y[0] <= radius:
        a = k*(1-0.8**2)/(0.8*(-np.abs(y[1]))) #0.8 Represents a "coeff of restitution"
        delta = k*(radius - y[0]) + a*(radius - y[0])*y[1]

    ydot = np.zeros(2, dtype=y.dtype)

    ydot[1] = -g + delta/m
    ydot[0] = y[1]
    
    return ydot

def getForce(v,dt):
    deltaV = np.diff(v)
    force = np.divide(deltaV,dt)
    force = np.insert(force,0,0)
    return force
    
t0 = 0.0
y0 = np.array([2.0, -1.75])
t1 = 2.5
iters = [1000,2000,5000,10000,20000,50000,100000,5000000,9000000]
Forces = []

for N in iters:
    dt = (t1-t0)/N
    t = np.linspace(t0,t1,N)
    ysol = odeint(func,y0,t)
    yy = ysol[:,0]
    force = getForce(ysol[:,1],dt)
    FirstBounce = np.max(force)
    Forces.append(FirstBounce)
    if N == 5000000:
        plt.figure(1)
        plt.plot(t,yy)
        plt.figure(3)
        plt.plot(t,ysol[:,1])



plt.figure(2)
plt.plot(np.divide(t1,iters),Forces)
plt.xlabel("Time Step (s)")
plt.ylabel("Force")
plt.yscale('log')
plt.xscale('log')