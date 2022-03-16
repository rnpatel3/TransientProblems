"""
This script generates a 2D plane stress finite element analysis
problem, and stores the problem object to a python pickle file.

The structure of the problem object is as follows:

prob_pkl: dict
    |---prob_name:    problem name,            str
    |---nelems:       number of elements,      int
    |---nnodes:       number of mesh nodes,    int
    |---ndof:         number of nodal dof,     int
    |---C:            constitutive matrix,     3*3 ndarray
    |---conn:         connectivity,            nelems*4 ndarray
    |---X:            nodal position,          nnodes*2 ndarray
    |---dof:          nodal degree of freedom, nnodes*2 ndarray
    |---force:        nodal forces,            ndof*1 ndarray
    |---r0:           filter radius,           float
    |---density:      material density,        float
    |---qval:         SIMP penalty,            float
    |---x:            nodal design variable,   None or nnodes*1 ndarray
    |---opt_settings: optimization consts,     None or dict
"""

import numpy as np
import pickle
import argparse
import os

# Set up parser
p = argparse.ArgumentParser()
p.add_argument('--nx', type=int, default=64)
p.add_argument('--ny', type=int, default=80)
p.add_argument('--lx', type=float, default=1.0)
p.add_argument('--ly', type=float, default=1.25)
p.add_argument('--qval', type=float, default=5.0)
p.add_argument('--nr0', type=int, default=32, 
        help='r0 = ly divided by nr0')
p.add_argument('--outdir', type=str, default='',
    help='directory for pkl output')
p.add_argument('--type', type=str, default='cantilever',
        choices=['cantilever', 'michell'])
args = p.parse_args()

# nelems and nnodes
nx = args.nx
ny = args.ny
lx = args.lx
ly = args.ly
nelems = nx*ny
nnodes = (nx+1)*(ny+1)

# prob_name
prob_type = 'cantilever'
if args.type == 'michell':
    prob_type = 'michell'
prob_name = '{:s}-nx{:d}-ny{:d}-lx{:.1f}-ly{:.1f}V3'.format(prob_type, nx, ny, lx, ly)

# r0
r0 = ly / args.nr0

# density
density = 2700.0

# qval
qval = args.qval

# C
C = np.zeros((3, 3))
E = 70e3
nu = 0.3
C[0, 0] = E/(1.0 - nu**2)
C[0, 1] = nu*E/(1.0 - nu**2)
C[1, 0] = C[0, 1]
C[1, 1] = C[0, 0]
C[2, 2] = 0.5*E/(1.0 + nu)

# ndof, conn, X, dof
conn = np.zeros((nelems, 4), dtype=np.intc)
dof = -np.ones((nnodes, 2), dtype=np.intc)
X = np.zeros((nnodes, 2))
for j in range(ny):
    for i in range(nx):
        conn[i + j*nx, 0] = i + (nx+1)*j
        conn[i + j*nx, 1] = i+1 + (nx+1)*j
        conn[i + j*nx, 2] = i + (nx+1)*(j+1)
        conn[i + j*nx, 3] = i+1 + (nx+1)*(j+1)
ndof = 0
for j in range(ny+1):
    for i in range(nx+1):
        X[i + j*(nx+1), 0] = lx*i/nx
        X[i + j*(nx+1), 1] = ly*j/ny
        if i > 0 or j<48 or j>64:
            dof[i + j*(nx+1), 0] = ndof
            ndof += 1
            dof[i + j*(nx+1), 1] = ndof
            ndof += 1

# force
forceval_y = 5400.0/12/18 #Divide load into the appropriate elements
forceval_x = -1000.0/12/18
force = np.zeros(ndof)
if args.type == 'cantilever':
    j = 0
    for i in range(int(0.8*nx),nx+1):
        force[dof[i + j*(nx+1), 1]] = forceval_y
        force[dof[i + j*(nx+1), 0]] = forceval_x
else:
    if ny % 2 == 0:
        i = nx
        j = ny // 2
        force[dof[i + j*(nx+1), 1]] = -forceval_y
    else:
        i = nx
        j = ny // 2
        force[dof[i + j*(nx+1), 1]] = -forceval_y / 2
        j = ny // 2 + 1
        force[dof[i + j*(nx+1), 1]] = -forceval_y / 2

# Generate pickle file
prob_pkl = dict()
prob_pkl['prob_name'] = prob_name
prob_pkl['nelems'] = nelems
prob_pkl['nnodes'] = nnodes
prob_pkl['ndof'] = ndof
prob_pkl['C'] = C
prob_pkl['conn'] = conn
prob_pkl['X'] = X
prob_pkl['dof'] = dof
prob_pkl['force'] = force
prob_pkl['r0'] = r0
prob_pkl['density'] = density
prob_pkl['qval'] = qval
prob_pkl['x'] = None
prob_pkl['opt_settings'] = None

outname = prob_pkl['prob_name']+'.pkl'
if args.outdir != '':
    try:
        os.mkdir(args.outdir)
    except:
        pass
    outname = args.outdir + '/' + outname
with open(outname, 'wb') as pklfile:
    pickle.dump(prob_pkl, pklfile)
