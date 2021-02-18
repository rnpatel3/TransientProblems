# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 21:36:05 2021

@author: rohan
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
def func(y, t):
    # m*dot(y[1]) = F = -mg + delta
    # dot(y[0]) = y[1]
    
    k = 1e8
    m = 1.0
    delta = 0.0
    radius = 0.05
    g = 9.81
    if y[0] <= radius:
        a = k*(1-0.9**2)/(0.9*(-np.abs(1.75)))
        delta = k*(radius - y[0]) + a*(radius - y[0])*y[1]
        
    
    ydot = np.zeros(2, dtype=y.dtype)
    ydot[1] = -g + delta/m
    ydot[0] = y[1]
    
    return ydot
t0 = 0.0
y0 = np.array([5.0, -1.75])
t1 = 10.0
dt = 0.01
#r = ode(func).set_integrator('zvode', method='bdf')
#r.set_initial_value(y0, t0)
#while r.successful() and r.t < t1:
#    print(r.t+dt, r.integrate(r.t+dt))

#sol1 = solve_ivp(func,[0,t1],y0)
#plt.plot([0,t1],sol1,'g-')

t = np.linspace(t0,t1,5000000)

ysol = odeint(func,y0,t)
yy = ysol[:,0]
plt.plot(t,yy)