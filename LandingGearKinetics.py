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

phi = (-l1**2+l2**2+lsp2**2)/(2*l2*lsp2)

rtside =  - k1 * (lsp1 - lsp1_init) * dlsp1dx \
       - k2 * (lsp2 - lsp2_init) * dlsp2dx \
       + g - c1*dlsp1dx**2*v - c2*dlsp2dx**2*v \
       + u1*dlsp1dx + u2*dlsp2dx

lftside = m*xdd + 0.5*mL*((2*(v+dlsp2dx*v*(phi))*(xdd+dlsp2dx*xdd*(phi)+dlsp2dx*v \
            *((2*l2*lsp2*(2*lsp2*(dlsp2dx*v))-(-l1**2+l2**2+lsp2**2)*(2*l2*(dlsp2dx*v)))/(2*l2*lsp2)**2))) \
       +2*(dlsp2dx*v*sin(sp.acos(phi)))*(xdd*dlsp2dx*sin(sp.acos(phi)))*(v*phi*((2*l2*lsp2*(2*lsp2*(dlsp2dx*v))\
            -(-l1**2+l2**2+lsp2**2)*(2*l2*(dlsp2dx*v)))/(2*l2*lsp2)**2)))

intans1 = rtside-lftside
a = sp.nonlinsolve([intans1],[xdd])
#print(a)


xdot = v
vdot = ((-c1*l1**8*l2**2*v*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) \
               - 4*c1*l1**6*l2**4*v*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) \
               + 8*c1*l1**6*l2**3*v*x*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) \
               - 6*c1*l1**4*l2**6*v*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) \
               + 24*c1*l1**4*l2**5*v*x*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) \
               - 24*c1*l1**4*l2**4*v*x**2*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) \
               - 4*c1*l1**2*l2**8*v*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x)\
               + 24*c1*l1**2*l2**7*v*x*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) \
               - 48*c1*l1**2*l2**6*v*x**2*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) \
               + 32*c1*l1**2*l2**5*v*x**3*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) \
               - c1*l2**10*v*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 8*c1*l2**9*v*x*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 24*c1*l2**8*v*x**2*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 32*c1*l2**7*v*x**3*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 16*c1*l2**6*v*x**4*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - c2*l1**8*l2**2*v*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 4*c2*l1**6*l2**4*v*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 4*c2*l1**6*l2**3*v*x*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 6*c2*l1**4*l2**6*v*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 12*c2*l1**4*l2**5*v*x*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 4*c2*l1**2*l2**8*v*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 12*c2*l1**2*l2**7*v*x*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 16*c2*l1**2*l2**5*v*x**3*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - c2*l2**10*v*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 4*c2*l2**9*v*x*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 16*c2*l2**7*v*x**3*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 16*c2*l2**6*v*x**4*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + g*l1**10*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 5*g*l1**8*l2**2*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 6*g*l1**8*l2*x*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 10*g*l1**6*l2**4*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 24*g*l1**6*l2**3*x*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 8*g*l1**6*l2**2*x**2*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 10*g*l1**4*l2**6*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 36*g*l1**4*l2**5*x*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 24*g*l1**4*l2**4*x**2*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 16*g*l1**4*l2**3*x**3*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 5*g*l1**2*l2**8*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 24*g*l1**2*l2**7*x*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 24*g*l1**2*l2**6*x**2*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 32*g*l1**2*l2**5*x**3*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 48*g*l1**2*l2**4*x**4*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + g*l2**10*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 6*g*l2**9*x*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 8*g*l2**8*x**2*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 16*g*l2**7*x**3*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 48*g*l2**6*x**4*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 32*g*l2**5*x**5*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + k1*l1**10*l2*lsp1_init*sqrt(l1**2 + l2**2 - 2*l2*x) - k1*l1**10*l2*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 5*k1*l1**8*l2**3*lsp1_init*sqrt(l1**2 + l2**2 - 2*l2*x) - 5*k1*l1**8*l2**3*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 6*k1*l1**8*l2**2*lsp1_init*x*sqrt(l1**2 + l2**2 - 2*l2*x) + 6*k1*l1**8*l2**2*x*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 10*k1*l1**6*l2**5*lsp1_init*sqrt(l1**2 + l2**2 - 2*l2*x) - 10*k1*l1**6*l2**5*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 24*k1*l1**6*l2**4*lsp1_init*x*sqrt(l1**2 + l2**2 - 2*l2*x) + 24*k1*l1**6*l2**4*x*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 8*k1*l1**6*l2**3*lsp1_init*x**2*sqrt(l1**2 + l2**2 - 2*l2*x) - 8*k1*l1**6*l2**3*x**2*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 10*k1*l1**4*l2**7*lsp1_init*sqrt(l1**2 + l2**2 - 2*l2*x) - 10*k1*l1**4*l2**7*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 36*k1*l1**4*l2**6*lsp1_init*x*sqrt(l1**2 + l2**2 - 2*l2*x) + 36*k1*l1**4*l2**6*x*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 24*k1*l1**4*l2**5*lsp1_init*x**2*sqrt(l1**2 + l2**2 - 2*l2*x) - 24*k1*l1**4*l2**5*x**2*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 16*k1*l1**4*l2**4*lsp1_init*x**3*sqrt(l1**2 + l2**2 - 2*l2*x) - 16*k1*l1**4*l2**4*x**3*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 5*k1*l1**2*l2**9*lsp1_init*sqrt(l1**2 + l2**2 - 2*l2*x) - 5*k1*l1**2*l2**9*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 24*k1*l1**2*l2**8*lsp1_init*x*sqrt(l1**2 + l2**2 - 2*l2*x) + 24*k1*l1**2*l2**8*x*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 24*k1*l1**2*l2**7*lsp1_init*x**2*sqrt(l1**2 + l2**2 - 2*l2*x) - 24*k1*l1**2*l2**7*x**2*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 32*k1*l1**2*l2**6*lsp1_init*x**3*sqrt(l1**2 + l2**2 - 2*l2*x) - 32*k1*l1**2*l2**6*x**3*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 48*k1*l1**2*l2**5*lsp1_init*x**4*sqrt(l1**2 + l2**2 - 2*l2*x) + 48*k1*l1**2*l2**5*x**4*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + k1*l2**11*lsp1_init*sqrt(l1**2 + l2**2 - 2*l2*x) - k1*l2**11*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 6*k1*l2**10*lsp1_init*x*sqrt(l1**2 + l2**2 - 2*l2*x) + 6*k1*l2**10*x*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 8*k1*l2**9*lsp1_init*x**2*sqrt(l1**2 + l2**2 - 2*l2*x) - 8*k1*l2**9*x**2*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 16*k1*l2**8*lsp1_init*x**3*sqrt(l1**2 + l2**2 - 2*l2*x) - 16*k1*l2**8*x**3*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 48*k1*l2**7*lsp1_init*x**4*sqrt(l1**2 + l2**2 - 2*l2*x) + 48*k1*l2**7*x**4*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 32*k1*l2**6*lsp1_init*x**5*sqrt(l1**2 + l2**2 - 2*l2*x) - 32*k1*l2**6*x**5*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - k2*l1**10*l2*lsp2_init*sqrt(l1**2 + l2**2 + 2*l2*x) + k2*l1**10*l2*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 5*k2*l1**8*l2**3*lsp2_init*sqrt(l1**2 + l2**2 + 2*l2*x) + 5*k2*l1**8*l2**3*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 6*k2*l1**8*l2**2*lsp2_init*x*sqrt(l1**2 + l2**2 + 2*l2*x) - 6*k2*l1**8*l2**2*x*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 10*k2*l1**6*l2**5*lsp2_init*sqrt(l1**2 + l2**2 + 2*l2*x) + 10*k2*l1**6*l2**5*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 24*k2*l1**6*l2**4*lsp2_init*x*sqrt(l1**2 + l2**2 + 2*l2*x) - 24*k2*l1**6*l2**4*x*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 8*k2*l1**6*l2**3*lsp2_init*x**2*sqrt(l1**2 + l2**2 + 2*l2*x) + 8*k2*l1**6*l2**3*x**2*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 10*k2*l1**4*l2**7*lsp2_init*sqrt(l1**2 + l2**2 + 2*l2*x) + 10*k2*l1**4*l2**7*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 36*k2*l1**4*l2**6*lsp2_init*x*sqrt(l1**2 + l2**2 + 2*l2*x) - 36*k2*l1**4*l2**6*x*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 24*k2*l1**4*l2**5*lsp2_init*x**2*sqrt(l1**2 + l2**2 + 2*l2*x) + 24*k2*l1**4*l2**5*x**2*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 16*k2*l1**4*l2**4*lsp2_init*x**3*sqrt(l1**2 + l2**2 + 2*l2*x) + 16*k2*l1**4*l2**4*x**3*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 5*k2*l1**2*l2**9*lsp2_init*sqrt(l1**2 + l2**2 + 2*l2*x) + 5*k2*l1**2*l2**9*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 24*k2*l1**2*l2**8*lsp2_init*x*sqrt(l1**2 + l2**2 + 2*l2*x) - 24*k2*l1**2*l2**8*x*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 24*k2*l1**2*l2**7*lsp2_init*x**2*sqrt(l1**2 + l2**2 + 2*l2*x) + 24*k2*l1**2*l2**7*x**2*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 32*k2*l1**2*l2**6*lsp2_init*x**3*sqrt(l1**2 + l2**2 + 2*l2*x) + 32*k2*l1**2*l2**6*x**3*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 48*k2*l1**2*l2**5*lsp2_init*x**4*sqrt(l1**2 + l2**2 + 2*l2*x) - 48*k2*l1**2*l2**5*x**4*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - k2*l2**11*lsp2_init*sqrt(l1**2 + l2**2 + 2*l2*x) + k2*l2**11*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 6*k2*l2**10*lsp2_init*x*sqrt(l1**2 + l2**2 + 2*l2*x) - 6*k2*l2**10*x*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 8*k2*l2**9*lsp2_init*x**2*sqrt(l1**2 + l2**2 + 2*l2*x) + 8*k2*l2**9*x**2*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 16*k2*l2**8*lsp2_init*x**3*sqrt(l1**2 + l2**2 + 2*l2*x) + 16*k2*l2**8*x**3*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 48*k2*l2**7*lsp2_init*x**4*sqrt(l1**2 + l2**2 + 2*l2*x) - 48*k2*l2**7*x**4*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 32*k2*l2**6*lsp2_init*x**5*sqrt(l1**2 + l2**2 + 2*l2*x) + 32*k2*l2**6*x**5*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + l1**10*l2*u1*sqrt(l1**2 + l2**2 - 2*l2*x) - l1**10*l2*u2*sqrt(l1**2 + l2**2 + 2*l2*x) + 5*l1**8*l2**3*u1*sqrt(l1**2 + l2**2 - 2*l2*x) - 5*l1**8*l2**3*u2*sqrt(l1**2 + l2**2 + 2*l2*x) - 6*l1**8*l2**2*u1*x*sqrt(l1**2 + l2**2 - 2*l2*x) + 6*l1**8*l2**2*u2*x*sqrt(l1**2 + l2**2 + 2*l2*x) - 1.0*l1**8*l2*mL*v**3*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 10*l1**6*l2**5*u1*sqrt(l1**2 + l2**2 - 2*l2*x) - 10*l1**6*l2**5*u2*sqrt(l1**2 + l2**2 + 2*l2*x) - 24*l1**6*l2**4*u1*x*sqrt(l1**2 + l2**2 - 2*l2*x) + 24*l1**6*l2**4*u2*x*sqrt(l1**2 + l2**2 + 2*l2*x) - 2.0*l1**6*l2**3*mL*v**3*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 8*l1**6*l2**3*u1*x**2*sqrt(l1**2 + l2**2 - 2*l2*x) - 8*l1**6*l2**3*u2*x**2*sqrt(l1**2 + l2**2 + 2*l2*x) + 2.0*l1**6*l2**2*mL*v**3*x*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 10*l1**4*l2**7*u1*sqrt(l1**2 + l2**2 - 2*l2*x) - 10*l1**4*l2**7*u2*sqrt(l1**2 + l2**2 + 2*l2*x) - 36*l1**4*l2**6*u1*x*sqrt(l1**2 + l2**2 - 2*l2*x) + 36*l1**4*l2**6*u2*x*sqrt(l1**2 + l2**2 + 2*l2*x) - 1.0*l1**4*l2**5*mL*v**3*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 24*l1**4*l2**5*u1*x**2*sqrt(l1**2 + l2**2 - 2*l2*x) - 24*l1**4*l2**5*u2*x**2*sqrt(l1**2 + l2**2 + 2*l2*x) + 4.0*l1**4*l2**4*mL*v**3*x*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 16*l1**4*l2**4*u1*x**3*sqrt(l1**2 + l2**2 - 2*l2*x) - 16*l1**4*l2**4*u2*x**3*sqrt(l1**2 + l2**2 + 2*l2*x) + 3.0*l1**4*l2**3*mL*v**3*x**2*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 5*l1**2*l2**9*u1*sqrt(l1**2 + l2**2 - 2*l2*x) - 5*l1**2*l2**9*u2*sqrt(l1**2 + l2**2 + 2*l2*x) - 24*l1**2*l2**8*u1*x*sqrt(l1**2 + l2**2 - 2*l2*x) + 24*l1**2*l2**8*u2*x*sqrt(l1**2 + l2**2 + 2*l2*x) + 24*l1**2*l2**7*u1*x**2*sqrt(l1**2 + l2**2 - 2*l2*x) - 24*l1**2*l2**7*u2*x**2*sqrt(l1**2 + l2**2 + 2*l2*x) + 2.0*l1**2*l2**6*mL*v**3*x*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + 32*l1**2*l2**6*u1*x**3*sqrt(l1**2 + l2**2 - 2*l2*x) - 32*l1**2*l2**6*u2*x**3*sqrt(l1**2 + l2**2 + 2*l2*x) - 2.0*l1**2*l2**5*mL*v**3*x**2*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 48*l1**2*l2**5*u1*x**4*sqrt(l1**2 + l2**2 - 2*l2*x) + 48*l1**2*l2**5*u2*x**4*sqrt(l1**2 + l2**2 + 2*l2*x) - 8.0*l1**2*l2**4*mL*v**3*x**3*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) + l2**11*u1*sqrt(l1**2 + l2**2 - 2*l2*x) - l2**11*u2*sqrt(l1**2 + l2**2 + 2*l2*x) - 6*l2**10*u1*x*sqrt(l1**2 + l2**2 - 2*l2*x) + 6*l2**10*u2*x*sqrt(l1**2 + l2**2 + 2*l2*x) + 8*l2**9*u1*x**2*sqrt(l1**2 + l2**2 - 2*l2*x) - 8*l2**9*u2*x**2*sqrt(l1**2 + l2**2 + 2*l2*x) + 16*l2**8*u1*x**3*sqrt(l1**2 + l2**2 - 2*l2*x) - 16*l2**8*u2*x**3*sqrt(l1**2 + l2**2 + 2*l2*x) - 1.0*l2**7*mL*v**3*x**2*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x) - 48*l2**7*u1*x**4*sqrt(l1**2 + l2**2 - 2*l2*x) + 48*l2**7*u2*x**4*sqrt(l1**2 + l2**2 + 2*l2*x) + 32*l2**6*u1*x**5*sqrt(l1**2 + l2**2 - 2*l2*x) - 32*l2**6*u2*x**5*sqrt(l1**2 + l2**2 + 2*l2*x) + 4.0*l2**5*mL*v**3*x**4*sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x))/(sqrt(l1**2 + l2**2 - 2*l2*x)*sqrt(l1**2 + l2**2 + 2*l2*x)*(l1**10*m + 1.0*l1**10*mL*v + 5*l1**8*l2**2*m + 3.0*l1**8*l2**2*mL*v - 6*l1**8*l2*m*x - 4.0*l1**8*l2*mL*v*x + 10*l1**6*l2**4*m + 3.0*l1**6*l2**4*mL*v - 24*l1**6*l2**3*m*x - 1.0*l1**6*l2**3*mL*v**3 - 10.0*l1**6*l2**3*mL*v*x + 8*l1**6*l2**2*m*x**2 + 1.0*l1**6*l2**2*mL*v**3*x + 1.0*l1**6*l2**2*mL*v*x**2 + 10*l1**4*l2**6*m + 1.0*l1**4*l2**6*mL*v - 36*l1**4*l2**5*m*x - 1.0*l1**4*l2**5*mL*v**3 - 8.0*l1**4*l2**5*mL*v*x + 24*l1**4*l2**4*m*x**2 + 7.0*l1**4*l2**4*mL*v*x**2 + 16*l1**4*l2**3*m*x**3 + 2.0*l1**4*l2**3*mL*v**3*x**2 + 14.0*l1**4*l2**3*mL*v*x**3 - 1.0*l1**4*l2**2*mL*v**3*x**3 + 5*l1**2*l2**8*m - 24*l1**2*l2**7*m*x - 2.0*l1**2*l2**7*mL*v*x + 24*l1**2*l2**6*m*x**2 + 1.0*l1**2*l2**6*mL*v**3*x + 7.0*l1**2*l2**6*mL*v*x**2 + 32*l1**2*l2**5*m*x**3 + 2.0*l1**2*l2**5*mL*v**3*x**2 + 4.0*l1**2*l2**5*mL*v*x**3 - 48*l1**2*l2**4*m*x**4 - 2.0*l1**2*l2**4*mL*v**3*x**3 - 20.0*l1**2*l2**4*mL*v*x**4 - 1.0*l1**2*l2**3*mL*v**3*x**4 + l2**10*m - 6*l2**9*m*x + 8*l2**8*m*x**2 + 1.0*l2**8*mL*v*x**2 + 16*l2**7*m*x**3 - 2.0*l2**7*mL*v*x**3 - 48*l2**6*m*x**4 - 1.0*l2**6*mL*v**3*x**3 - 4.0*l2**6*mL*v*x**4 + 32*l2**5*m*x**5 - 1.0*l2**5*mL*v**3*x**4 + 8.0*l2**5*mL*v*x**5 + 2.0*l2**4*mL*v**3*x**5)))

print('--------------------------------------------')
print('xdot:',xdot)
print('vdot:',vdot)
print('partials[xdot, x]:', sp.diff(xdot, x))
print('partials[xdot, v]:', sp.diff(xdot, v))
print('partials[vdot, x]:', sp.diff(vdot, x))
print('partials[vdot, v]:', sp.diff(vdot, v))
print('partials[vdot, u1]:', sp.diff(vdot, u1))
print('partials[vdot, u2]:', sp.diff(vdot, u2))