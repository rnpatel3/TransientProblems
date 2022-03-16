#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 18:03:17 2020

@author: rohan
"""

from sympy import symbols, sin, cos, sqrt
import numpy as np
import sympy as sp

# LandingGearODE
l1, l2, lsp1, lsp2, lsp1_init, lsp2_init = symbols('l1, l2, lsp1, lsp2, lsp1_init, lsp2_init')
x, v, xdd, u1, u2, g, k1, k2, m, mL, c1, c2 = symbols('x, v, xdd, u1, u2, g, k1, k2, m, mL, c1, c2')
lsp1 = sqrt(l1**2 + l2**2 + 2*l2*x)
lsp2 = sqrt(l1**2 + l2**2 - 2*l2*x)
dlsp1dx = sp.diff(lsp1, x)
dlsp2dx = sp.diff(lsp2, x)

c = (x**2)/((x**2-l1**2))
gx = (-2*x*l1**2)/((x**2-l1**2)**2)

rtside = u1*dlsp1dx + u2*dlsp2dx - c1*((dlsp1dx)**2)*v - c2*(dlsp2dx**2)*v

lftside = m*xdd + 0.5*mL*gx*v**3 + mL*c*xdd + k1*dlsp1dx + k2*dlsp2dx - m*g

intans1 = rtside-lftside
a = sp.nonlinsolve([intans1],[xdd])
print(a)


xdot = v
vdot = ((-c1*l1**6*l2**2*v*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - c1*l1**4*l2**4*v*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 2*c1*l1**4*l2**3*v*x*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 2*c1*l1**4*l2**2*v*x**2*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 2*c1*l1**2*l2**4*v*x**2*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 4*c1*l1**2*l2**3*v*x**3*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - c1*l1**2*l2**2*v*x**4*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - c1*l2**4*v*x**4*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 2*c1*l2**3*v*x**5*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - c2*l1**6*l2**2*v*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - c2*l1**4*l2**4*v*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 2*c2*l1**4*l2**3*v*x*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 2*c2*l1**4*l2**2*v*x**2*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 2*c2*l1**2*l2**4*v*x**2*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 4*c2*l1**2*l2**3*v*x**3*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - c2*l1**2*l2**2*v*x**4*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - c2*l2**4*v*x**4*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 2*c2*l2**3*v*x**5*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + g*l1**8*m*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 2*g*l1**6*l2**2*m*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 2*g*l1**6*m*x**2*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + g*l1**4*l2**4*m*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 8*g*l1**4*l2**2*m*x**2*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + g*l1**4*m*x**4*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 2*g*l1**2*l2**4*m*x**2*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 10*g*l1**2*l2**2*m*x**4*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + g*l2**4*m*x**4*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 4*g*l2**2*m*x**6*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - k1*l1**8*l2*sqrt(l1**2 + l2**2 - 2*l2*x) - 2*k1*l1**6*l2**3*sqrt(l1**2 + l2**2 - 2*l2*x) + 2*k1*l1**6*l2*x**2*sqrt(l1**2 + l2**2 - 2*l2*x) - k1*l1**4*l2**5*sqrt(l1**2 + l2**2 - 2*l2*x) + 8*k1*l1**4*l2**3*x**2*sqrt(l1**2 + l2**2 - 2*l2*x) - k1*l1**4*l2*x**4*sqrt(l1**2 + l2**2 - 2*l2*x) + 2*k1*l1**2*l2**5*x**2*sqrt(l1**2 + l2**2 - 2*l2*x) - 10*k1*l1**2*l2**3*x**4*sqrt(l1**2 + l2**2 - 2*l2*x) - k1*l2**5*x**4*sqrt(l1**2 + l2**2 - 2*l2*x) + 4*k1*l2**3*x**6*sqrt(l1**2 + l2**2 - 2*l2*x) + k2*l1**8*l2*sqrt(l1**2 + l2**2 + 2*l2*x) + 2*k2*l1**6*l2**3*sqrt(l1**2 + l2**2 + 2*l2*x) - 2*k2*l1**6*l2*x**2*sqrt(l1**2 + l2**2 + 2*l2*x) + k2*l1**4*l2**5*sqrt(l1**2 + l2**2 + 2*l2*x) - 8*k2*l1**4*l2**3*x**2*sqrt(l1**2 + l2**2 + 2*l2*x) + k2*l1**4*l2*x**4*sqrt(l1**2 + l2**2 + 2*l2*x) - 2*k2*l1**2*l2**5*x**2*sqrt(l1**2 + l2**2 + 2*l2*x) + 10*k2*l1**2*l2**3*x**4*sqrt(l1**2 + l2**2 + 2*l2*x) + k2*l2**5*x**4*sqrt(l1**2 + l2**2 + 2*l2*x) - 4*k2*l2**3*x**6*sqrt(l1**2 + l2**2 + 2*l2*x) + l1**8*l2*u1*sqrt(l1**2 + l2**2 - 2*l2*x) - l1**8*l2*u2*sqrt(l1**2 + l2**2 + 2*l2*x) + 2*l1**6*l2**3*u1*sqrt(l1**2 + l2**2 - 2*l2*x) - 2*l1**6*l2**3*u2*sqrt(l1**2 + l2**2 + 2*l2*x) - 2*l1**6*l2*u1*x**2*sqrt(l1**2 + l2**2 - 2*l2*x) + 2*l1**6*l2*u2*x**2*sqrt(l1**2 + l2**2 + 2*l2*x) + 1.0*l1**6*mL*v**3*x*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + l1**4*l2**5*u1*sqrt(l1**2 + l2**2 - 2*l2*x) - l1**4*l2**5*u2*sqrt(l1**2 + l2**2 + 2*l2*x) - 8*l1**4*l2**3*u1*x**2*sqrt(l1**2 + l2**2 - 2*l2*x) + 8*l1**4*l2**3*u2*x**2*sqrt(l1**2 + l2**2 + 2*l2*x) + 2.0*l1**4*l2**2*mL*v**3*x*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + l1**4*l2*u1*x**4*sqrt(l1**2 + l2**2 - 2*l2*x) - l1**4*l2*u2*x**4*sqrt(l1**2 + l2**2 + 2*l2*x) - 2*l1**2*l2**5*u1*x**2*sqrt(l1**2 + l2**2 - 2*l2*x) + 2*l1**2*l2**5*u2*x**2*sqrt(l1**2 + l2**2 + 2*l2*x) + 1.0*l1**2*l2**4*mL*v**3*x*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 10*l1**2*l2**3*u1*x**4*sqrt(l1**2 + l2**2 - 2*l2*x) - 10*l1**2*l2**3*u2*x**4*sqrt(l1**2 + l2**2 + 2*l2*x) - 4.0*l1**2*l2**2*mL*v**3*x**3*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + l2**5*u1*x**4*sqrt(l1**2 + l2**2 - 2*l2*x) - l2**5*u2*x**4*sqrt(l1**2 + l2**2 + 2*l2*x) - 4*l2**3*u1*x**6*sqrt(l1**2 + l2**2 - 2*l2*x) + 4*l2**3*u2*x**6*sqrt(l1**2 + l2**2 + 2*l2*x))/(sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x)*(l1**8*m + 2*l1**6*l2**2*m - 2*l1**6*m*x**2 - l1**6*mL*x**2 + l1**4*l2**4*m - 8*l1**4*l2**2*m*x**2 - 2*l1**4*l2**2*mL*x**2 + l1**4*m*x**4 + l1**4*mL*x**4 - 2*l1**2*l2**4*m*x**2 - l1**2*l2**4*mL*x**2 + 10*l1**2*l2**2*m*x**4 + 6*l1**2*l2**2*mL*x**4 + l2**4*m*x**4 + l2**4*mL*x**4 - 4*l2**2*m*x**6 - 4*l2**2*mL*x**6)))
print('--------------------------------------------')
print('xdot:',xdot)
print('vdot:',vdot)
print('partials[xdot, x]:', sp.diff(xdot, x))
print('partials[xdot, v]:', sp.diff(xdot, v))
print('partials[vdot, x]:', sp.diff(vdot, x))
print('partials[vdot, v]:', sp.diff(vdot, v))
print('partials[vdot, u1]:', sp.diff(vdot, u1))
print('partials[vdot, u2]:', sp.diff(vdot, u2))