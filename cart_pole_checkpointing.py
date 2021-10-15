"""
Cart-pole trajectory optimization problem from "An Introduction to
Trajectory Optimization" by Matthew Kelly from SIAM Review, Vol 59, No 4.

The initial conditions are that the pole be oriented in the downward
configuration and at rest, and at the final configuration, the pole be
vertical and at rest.
"""

import os
import numpy as np
import mpi4py.MPI as MPI
from paropt import ParOpt
import copy


def generate_jacobian():
    """
    Generate the expressions for the Jacobian using sympy
    """
    import sympy as sym

    # Create the symbols
    q = []
    qdot = []
    res = []
    for i in range(4):
        q.append(sym.Symbol('q[%d]'%(i)))
        qdot.append(sym.Symbol('qdot[%d]'%(i)))
        res.append(sym.Symbol('res[%d]'%(i)))
    u = sym.Symbol('u')

    # Create the constants
    m1 = sym.Symbol('self.m1')
    m2 = sym.Symbol('self.m2')
    L = sym.Symbol('self.L')
    g = sym.Symbol('self.g')
    alpha = sym.Symbol('alpha')
    beta = sym.Symbol('beta')

    res[0] = q[2] - qdot[0]
    res[1] = q[3] - qdot[1]
    res[2] = ((m1 + m2*(1.0 - sym.cos(q[1])**2))*qdot[2] -
            (L*m2*sym.sin(q[1])*q[3]**2 + u +
                m2*g*sym.cos(q[1])*sym.sin(q[1])))
    res[3] = (L*(m1 + m2*(1.0 - sym.cos(q[1])**2))*qdot[3] +
            (L*m2*sym.cos(q[1])*sym.sin(q[1])*q[3]**2 +
                u*sym.cos(q[1]) +
                (m1 + m2)*g*sym.sin(q[1])))

    for i in range(4):
        for j in range(4):
            expr = (beta*sym.simplify(sym.diff(res[i], qdot[j])) +
                    alpha*sym.simplify(sym.diff(res[i], q[j])))
            print('J[%d,%d] = '%(i, j) + str(expr))

    return

class CartPole(ParOpt.Problem):
    def __init__(self, t, m1=1.0, m2=0.3, L=0.5):
        self.m1 = m1
        self.m2 = m2
        self.L = L
        self.g = 9.81
        self.t = t
        #self.numCheckpoints = 40
        self.checkpoints = np.array([0,10,20,30])
        self.checkpoint_states = np.array([[]])
        self.checkpoint_prev_states = np.array([[]])
        self.iter_counter = 0

        # Compute the weights for the objective function
        self.h = np.zeros(t.shape[0] - 1)
        self.h = self.t[1:] - self.t[:-1]

        self.max_newton_iters = 100
        self.newton_tol = 1e-8
        self.fobj_scale = 0.01
        self.con_scale = 10.0

        # Set the number of design variables and constraints
        nvars = len(t)-1
        ncon = 4

        # Initialize the base class
        nineq = 0
        super(CartPole, self).__init__(MPI.COMM_WORLD, nvars, ncon, nineq)

        return

    def getVarsAndBounds(self, x, lb, ub):
        """Set the values of the bounds"""
        x[:] = 1.0
        lb[:] = -20.0
        ub[:] = 20.0
        return

    def evalObjCon(self, x):
        """Evaluate the objective and constraint"""
        # Evaluate the objective and constraints

        fail = 0
        con = np.zeros(4, dtype=ParOpt.dtype)

        # Compute the sum square of the weights
        fobj = self.fobj_scale*np.sum(self.h*x[:]**2)

        # Compute the full trajectory based on the input forces
        #print('control force x = ', x[:])
        self.q = self.computeTrajectory(self.t, x[:], 1, len(self.t))
        #print("checkpoint states (objcon): ", self.checkpoint_states)
        # Compute the constraints
        #print("q = ", self.q)
        con[0] = self.q[-1, 0] - 1.0 # q1 = 1.0
        con[1] = self.q[-1, 1] - np.pi # q2 = np.pi
        con[2] = self.q[-1, 2] # qdot(q1) = 0
        con[3] = self.q[-1, 3] # qdot(q2) = 0
        con[:] *= self.con_scale

        print('iter: %d   obj: %15g  infeas: %15g'%
              (self.iter_counter, fobj, np.sqrt(np.dot(con, con))))
        self.iter_counter += 1

        return fail, fobj, con

    def evalObjConGradient(self, x, g, A):
        """Evaluate the objective and constraint gradient"""
        fail = 0

        # The objective gradient
        g[:] = 2.0*self.fobj_scale*self.h*x[:]

        # Create a vector to store the derivative of the constraints
        dfdx = np.zeros(self.t.shape, dtype=ParOpt.dtype)
        for state in range(4):
            self.computeAdjointDeriv(self.t, self.q, x[:], state, dfdx)
            A[state][:] = self.con_scale*dfdx

        return fail

    def computeResidual(self, q, qdot, u, res):
        """
        Compute the residual of the system dynamics.
        """
        # q = [q1, q2, q1dot, q2dot]
        #print("q = ", q)
        res[0] = q[0][2] - qdot[0][0]
        res[1] = q[0][3] - qdot[0][1]

        # Compute the residual for the first equation of motion
        res[2] = ((self.m1 + self.m2*(1.0 - np.cos(q[0][1])**2))*qdot[0][2] -
            (self.L*self.m2*np.sin(q[0][1])*q[0][3]**2 + u +
            self.m2*self.g*np.cos(q[0][1])*np.sin(q[0][1])))

        # Compute the residual for the second equation of motion
        res[3] = (self.L*(self.m1 + self.m2*(1.0 - np.cos(q[0][1])**2))*qdot[0][3] +
            (self.L*self.m2*np.cos(q[0][1])*np.sin(q[0][1])*q[0][3]**2 +
            u*np.cos(q[0][1]) +
            (self.m1 + self.m2)*self.g*np.sin(q[0][1])))
        return

    def computeJacobian(self, alpha, beta, q, qdot, u, J):
        """
        Compute the Jacobian of the system dynamics.
        """

        J[:,:] = 0.0
        J[0,0] = -beta
        J[0,2] = alpha
        J[1,1] = -beta
        J[1,3] = alpha
        J[2,1] = alpha*(self.m2*(-q[0][3]**2*self.L*np.cos(q[0][1]) +
                                 qdot[0][2]*np.sin(2*q[0][1]) - self.g*np.cos(2*q[0][1])))
        J[2,2] = beta*(1.0*self.m1 + 1.0*self.m2*np.sin(q[0][1])**2)
        J[2,3] = -2*alpha*q[0][3]*self.L*self.m2*np.sin(q[0][1])
        J[3,1] = alpha*((q[0][3]**2*self.L*self.m2*np.cos(2*q[0][1]) +
                         qdot[0][3]*self.L*self.m2*np.sin(2*q[0][1]) +
                         self.g*self.m1*np.cos(q[0][1]) +
                         self.g*self.m2*np.cos(q[0][1]) - u*np.sin(q[0][1])))
        J[3,3] = (alpha*(q[0][3]*self.L*self.m2*np.sin(2*q[0][1])) +
                  beta*self.L*(self.m1 + self.m2*np.sin(q[0][1])**2))

        return

    def verifyJacobian(self, h=1e-30):
        """
        Use complex-step to verify the Jacobian matrix
        """
        q = np.random.uniform(size=(1,4))
        qdot = np.random.uniform(size=(1,4))
        p = np.random.uniform(size=(4))
        u = -1.72

        alpha = 0.154
        beta = 0.721
        J = np.zeros((4, 4))
        self.computeJacobian(alpha, beta, q, qdot, u, J)
        Jp = np.dot(J, p)

        res = np.zeros(4, dtype=np.complex)
        self.computeResidual(q + h*alpha*1j*p, qdot + h*beta*1j*p, u, res)
        Jpc = res.imag/h

        print('Relative error: ', (Jp - Jpc)/Jpc)
        return

    def computeTrajectory(self, t, u, start, end, store=False):
        """
        Given the input control force u[i] for t[i] = 0, to t final,
        compute the trajectory.
        """

        # Allocate space for the state variables
        #q = np.zeros((len(t), 4), dtype=ParOpt.dtype)

        # Set the initial conditions.
        #q[0,:] = 0.0
        q_i_prev = np.zeros((1,4), dtype=ParOpt.dtype)
        #checkpoints = np.arange(0,40)
                                
        
        # Compute the residual and Jacobian
        res = np.zeros(4, dtype=ParOpt.dtype)
        J = np.zeros((4, 4), dtype=ParOpt.dtype)

        # Integrate forward in time
        for i in range(start, end):
            # Copy the starting point for the first iteration, loading checkpoint states
            #print("i =", i)
            if i == 1 or (i==0 and store):
                q_i_temp = copy.copy(q_i_prev)
                #self.verifyJacobian()
            elif i==0 and i==start and i==end-1:
                q_i_prev = copy.copy(self.checkpoint_prev_states[np.where(self.checkpoints==start)[0].item()][:])
                q_i_prev = np.reshape(q_i_prev, (1,4))
                return q_i_prev
            elif i==start:
                # print("checkpoint states (traj): ", self.checkpoint_states)
                q_i_temp = copy.copy(self.checkpoint_states[np.where(self.checkpoints==start)[0].item()][:])
                q_i_temp = np.reshape(q_i_temp, (1,4))
                q_i_prev = copy.copy(self.checkpoint_prev_states[np.where(self.checkpoints==start)[0].item()][:])
                q_i_prev = np.reshape(q_i_prev, (1,4))
                #print("checkpoint states: ", self.checkpoint_states)
                #print("q_i_temp, chk: ", q_i_temp)
                #print("q_i_prev, chk: ", q_i_prev)

                
            else:
                #print("checkpoint states (traj): ", self.checkpoint_states)
                #print("qi_prev: ", q_i_prev)
                q_i_temp = copy.copy(q_i_prev)

            # Solve the nonlinear equations for q[i]
            #print("state (i) = ", i)
            for j in range(self.max_newton_iters):
                # Compute the approximate value of the velocities
                alpha = 0.5
                qi = alpha*(q_i_temp + q_i_prev)
                beta = 1.0/(t[i] - t[i-1])
                qdot = beta*(q_i_temp - q_i_prev)
                #print("q = ", qi)    
                self.computeResidual(qi, qdot, u[i-1], res)
                self.computeJacobian(alpha, beta, qi, qdot, u[i-1], J)
                
                update = np.linalg.solve(J, res)
                #print(update)
                q_i_temp -= update
                #print("checkpoints", self.checkpoints) Properly loads checkpoint states here
                rnorm = np.sqrt(np.dot(res, res))
                #print("rnorm: ", rnorm) 
                if rnorm < self.newton_tol:
                    #print("checkpoints", self.checkpoints)
                    if i in self.checkpoints and store:
                        
                        #print("Adding to stored states")
                        #print("checkpoint states: ", self.checkpoint_states) works
                        #print("States to add: ", q_i_temp) works
                        if i==0:
                            q_i_prev = np.zeros((1,4),dtype=ParOpt.dtype())
                            self.checkpoint_states = np.reshape(q_i_prev,(1,4))
                            self.checkpoint_prev_states = np.reshape(q_i_prev, (1,4))
                        else:
                            self.checkpoint_states = np.append(self.checkpoint_states,np.reshape(q_i_temp, (1,4)),axis=0)
                            self.checkpoint_prev_states = np.append(self.checkpoint_prev_states,np.reshape(q_i_prev, (1,4)), axis=0) #Store previous state to use in the next forward integration
                        #print("new checkpoint states: ", self.checkpoint_states) All of these work below
                        #print("Previous checkpoint state to add: ", q_i_prev)
                        #print("Prev checkpoint states: ", self.checkpoint_prev_states)

                    if i == 0:
                        q_i_prev = np.zeros((1,4),dtype=ParOpt.dtype)
                    else:
                        q_i_prev = np.reshape(q_i_temp[:],(1,4))
                        #rint("q_i_prev: ", q_i_prev)
                    break
            #print("q_i_prev = ", q_i_prev)
        return q_i_temp #Returns q{i-1} (same as q_i_final)

    def computeAdjointDeriv(self, t, q, u, state, dfdx):
        """
        Compute the derivative of the final state index with respect to
        the control.

        System dynamics:
        R(q, dot{q}, u) = 0

        adjoint[i]*R[i] =
            adjoint[i]*R(0.5*(q[i] + q[i-1]), (q[i] - q[i-1])/(t[i] - t[i-1])) +
            adjoint[i+1]*R(0.5*(q[i+1] + q[i]), (q[i+1] - q[i])/(t[i+1] - t[i]))

        J[i,j] = dR[i]/dq[j]
        J[i,i] = J(alpha, beta)
        J[i,i-1] = J(alpha, -beta)

        dL/dq[i] = df/dq[i] + adjoint[i]*J[i,i] + adjoint[i+1]*J[i+1,i] = 0
        """
        # Zero-out the contributions to the state variables
        dfdx[:] = 0.0
        #print("checkpoint states = ", self.checkpoint_states)

        # Compute the residual and Jacobian
        res = np.zeros(4, dtype=ParOpt.dtype)
        res[state] = 1.0 # df/du
        J = np.zeros((4, 4), dtype=ParOpt.dtype)
        #q_i_prev = np.zeros((1,4), dtype=ParOpt.dtype)
        q_i_prev = self.computeTrajectory(t, u, 0, len(t), True)
        # Integrate the adjoint in reverse
        for i in range(len(t)-1, 0, -1):
            #Get q[i]
            if i == len(t)-1:
                q_i = copy.copy(q_i_prev)        #At the last step, get the final states from the previous run of computeTrajectory
            else:
                q_i = copy.copy(q_i_prev)
            #Get q[i-1]
            lastChkpnt = self.checkpoints[self.checkpoints < i].max() #Find previous checkpoint state
            q_i_prev = self.computeTrajectory(t, u,lastChkpnt,i, False)
            
            # Set alpha and the qdot values
            alpha = 0.5
            qi = alpha*(q_i + q_i_prev)
            beta = 1.0/(t[i] - t[i-1])
            qdot = beta*(q_i - q_i_prev)
            #print("Adjoint state (i): ", i)
            #print("qi: ", q_i)
            #print("qi prev: ", q_i_prev)
            # Compute the Jacobian matrix
            self.computeJacobian(alpha, beta, qi, qdot, u[i-1], J)
            #print("jac: ", J) Error in jacobian and res
            # Compute the adjoint variables
            adjoint = -np.linalg.solve(J.T, res)
            #print("adjoint: ", adjoint) error in adjoint vars

            # Compute the total derivative
            dfdx[i-1] += -adjoint[2] + adjoint[3]*np.cos(qi[0][1])

            # Compute the right-hand-side for the next adjoint
            self.computeJacobian(alpha, -beta, qi, qdot, u[i-1], J)
            res = np.dot(J.T, adjoint)

        return

    def visualize(self, x, skip=10):
        """
        Visualize the output from a simulation
        """

        import matplotlib.pylab as plt
        from matplotlib.collections import LineCollection
        import matplotlib.cm as cm

        # Set the control input
        u = x[:]

        # Set the values of the states
        q = self.q[::skip]

        # Create the time-lapse visualization
        fig = plt.figure()
        plt.axis('equal')
        plt.axis('off')

        values = np.linspace(0, 1.0, q.shape[0])
        cmap = cm.get_cmap('viridis')

        x = []
        y = []
        for i in range(q.shape[0]):
            color = cmap(values[i])

            x1 = q[i,0]
            y1 = 0.0
            x2 = q[i,0] + self.L*np.sin(q[i,1])
            y2 = -self.L*np.cos(q[i,1])

            x.append(x2)
            y.append(y2)

            plt.plot([x1], [y1], color=color, marker='o')
            plt.plot([x1, x2], [y1, y2], linewidth=2, color=color)

        # Create the line segments
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a continuous norm to map from data points to colors
        lc = LineCollection(segments, cmap=cmap)

        # Set the values used for colormapping
        lc.set_array(values)
        fig.axes[0].add_collection(lc)

        fig, ax = plt.subplots(3, 1)

        ax[0].plot(self.t, self.q[:,0])
        ax[1].plot(self.t, self.q[:,1])
        ax[2].plot(0.5*(self.t[1:] + self.t[:-1]), u)

        plt.show()

# Create the
n = 40
t = np.linspace(0, 2.0, n)
problem = CartPole(t)

problem.checkGradients(1e-5)

filename = 'paropt.out'
options = {
    'algorithm': 'tr',
    'output_level': 0,
    'tr_l1_tol': 1e-30,
    'tr_linfty_tol': 1e-30,
    'norm_type': 'infinity',
    'max_major_iters': 50,
    # 'barrier_strategy': 'mehrotra',
    # 'starting_point_strategy': 'affine_step',
    'qn_type': 'bfgs',
    'qn_update_type': 'damped_update',
    'output_file': filename,
    'tr_min_size': 1e-6,
    'tr_max_size': 1e3,
    'tr_max_iterations': 500,
    'tr_adaptive_gamma_update': False,
    'tr_accept_step_strategy': 'penalty_method',
    'tr_use_soc': False,
    'tr_soc_use_quad_model': True,
    'tr_max_iterations': 200
    }

opt = ParOpt.Optimizer(problem, options)

# Set a new starting point
opt.optimize()
x, z, zw, zl, zu = opt.getOptimizedPoint()

# Visualize the final design
problem.visualize(x, skip=1)
