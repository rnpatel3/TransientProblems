import numpy as np
#from autograd import grad
import scipy
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class CartPole:
    def __init__(self, t, m1=1.0, m2=0.3, L=0.5):
        self.m1 = m1
        self.m2 = m2
        self.L = L
        self.g = 9.81
        self.t = t
        self.iter_counter = 0

        # Compute the weights for the objective function
        self.h = np.zeros(t.shape[0] - 1)
        self.h = self.t[1:] - self.t[:-1]
        self.max_newton_iters = 10
        self.newton_tol = 1e-8

    def computeResidual(self, q, qdot, x):
        """
        Compute the residual of the system dynamics.
        """
        # q = [q1, q2, q1dot, q2dot]
        res = np.array([
            q[2] - qdot[0],
            q[3] - qdot[1],

        # Compute the residual for the first equation of motion
            ((self.m1 + self.m2*(1.0 - np.cos(q[1])**2))*qdot[2] -
                  (self.L*self.m2*np.sin(q[1])*q[3]**2 + x +
                  self.m2*self.g*np.cos(q[1])*np.sin(q[1]))),

        # Compute the residual for the second equation of motion
            (self.L*(self.m1 + self.m2*(1.0 - np.cos(q[1])**2))*qdot[3] +
                  (self.L*self.m2*np.cos(q[1])*np.sin(q[1])*q[3]**2 +
                  x*np.cos(q[1]) +
                  (self.m1 + self.m2)*self.g*np.sin(q[1])))])

        return res

    def computeResidualdx(self, q, qdot, x):
        return np.array([0.0, 0.0, -1.0, np.cos(q[1])])

    def computeJacobian(self, alpha, beta, q, qdot, x):
        """
        Compute the Jacobian of the system dynamics.
        """

        J = np.array([
          [-beta, 0.0, alpha, 0.0],
          [0.0, -beta, 0.0, alpha],
          [0.0,
          alpha*(self.m2*(-q[3]**2*self.L*np.cos(q[1]) +
                 qdot[2]*np.sin(2*q[1]) - self.g*np.cos(2*q[1]))),
          beta*(1.0*self.m1 + 1.0*self.m2*np.sin(q[1])**2),
          -2*alpha*q[3]*self.L*self.m2*np.sin(q[1])],
          [0.0, alpha*((q[3]**2*self.L*self.m2*np.cos(2*q[1]) +
                                 qdot[3]*self.L*self.m2*np.sin(2*q[1]) +
                                 self.g*self.m1*np.cos(q[1]) +
                                 self.g*self.m2*np.cos(q[1]) - x*np.sin(q[1]))),
                                 0.0,
          (alpha*(q[3]*self.L*self.m2*np.sin(2*q[1])) +
                          beta*self.L*(self.m1 + self.m2*np.sin(q[1])**2))]])

        return J

    def computeTrajectory(self, x):
        """
        Given the input control force x[i] for t[i] = 0, to t final,
        compute the trajectory.
        """
        # Allocate space for the state variables
        # u = np.zeros((len(self.t), 4)) #  dtype=x.dtype)

        # Set the initial conditions.

        u = [np.array([0.0, 0.0, 0.0, 0.0])]

        # Integrate forward in time
        for i in range(1, len(self.t)):
            # Copy the starting point for the first iteration
            # u[i,:] = u[i-1,:]
            v = np.array(u[-1])

            # Solve the nonlinear equations for q[i]
            for j in range(self.max_newton_iters):
                # Compute the approximate value of the velocities
                alpha = 0.5
                qi = alpha*(v + u[i-1])
                beta = 1.0/(t[i] - t[i-1])
                qdot = beta*(v - u[i-1])
                res = self.computeResidual(qi, qdot, x[i-1])
                J = self.computeJacobian(alpha, beta, qi, qdot, x[i-1])
                update = np.linalg.solve(J, res)

                v = v - update
                rnorm = np.sqrt(np.dot(res, res))
                if rnorm < self.newton_tol:
                    break

            u.append(v)

        return np.array(u)

    def computeStateJacobian(self, x):
        """
        Given the design variables x, compute the full dR/du matrix
        """

        n = len(self.t)
        dRdu = np.zeros((4*n, 4*n))
        dRdx = np.zeros((4*n, n-1))

        u = self.computeTrajectory(x)

        # Set the Jacobian entries for d(u0)/d(u0) = I
        for i in range(4):
            dRdu[i, i] = 1.0

        res = np.zeros(4)
        J = np.zeros((4, 4))

        for i in range(1, len(self.t)):
            alpha = 0.5
            qi = alpha*(u[i,:] + u[i-1,:])
            beta = 1.0/(t[i] - t[i-1])
            qdot = beta*(u[i,:] - u[i-1,:])

            J = self.computeJacobian(alpha, beta, qi, qdot, x[i-1])
            for ii in range(4):
                for jj in range(4):
                    dRdu[4*i + ii, 4*i + jj] = J[ii, jj]

            J = self.computeJacobian(alpha, -beta, qi, qdot, x[i-1])
            for ii in range(4):
                for jj in range(4):
                    dRdu[4*i + ii, 4*(i-1) + jj] = J[ii, jj]

            res = self.computeResidualdx(qi, qdot, x[i-1])
            for ii in range(4):
                dRdx[4*i + ii, i-1] = res[ii]

        return dRdu, dRdx

    def computeConstraints(self, x):
        #Computing the constraint violation
        c = np.zeros(4)

        u = self.computeTrajectory(x)

        c = np.array([
            u[-1, 0] - 1.0,
            u[-1, 1] - np.pi,
            u[-1, 2],
            u[-1, 3]])

        return c

#    def computeC1(self, x):
#        u = self.computeTrajectory(x)
#        return u[-1, 0]
#
#    def computeC2(self, x):
#        u = self.computeTrajectory(x)
#        return u[-1, 1]

    def computeConstraintGradient(self, x):
        """
        Compute the gradient using the direct method
        """

        dRdu, dRdx = self.computeStateJacobian(x)

        # Solve for the total derivative of u w.r.t. x
        phi = - np.linalg.solve(dRdu, dRdx)

        n = len(self.t)
        A = np.zeros((4, n-1))

        A[0, :] = phi[4*(n-1), :]
        A[1, :] = phi[4*(n-1)+1, :]
        A[2, :] = phi[4*(n-1)+2, :]
        A[3, :] = phi[4*(n-1)+3, :]

        return A
    
    def computeTotalForces(self, x):
        total_force = 0
        for i in range(len(x)):
           total_force += x[i]**2 
        return total_force
    
    def computeObjectiveGradient(self, x):
        return 2*x
    
    def visualize(self, x, q, skip=10):
        """
        Visualize the output from a simulation
        """

        import matplotlib.pylab as plt
        from matplotlib.collections import LineCollection
        import matplotlib.cm as cm

        # Set the control input
        u = x[:]

        # Set the values of the states
        q = q[::skip]

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

        ax[0].plot(self.t, q[:,0])
        ax[1].plot(self.t, q[:,1])
        ax[2].plot(0.5*(self.t[1:] + self.t[:-1]), u)

        plt.show()

n = 40
t = np.linspace(0, 2.0, n)
cart = CartPole(t)
x = np.linspace(-1, 1, n-1)
q = cart.computeTrajectory(x)

p = np.random.uniform(size=(n-1))
h = 1e-6

c = cart.computeConstraints(x)
c1 = cart.computeConstraints(x + h*p)

#con_grad1 = grad(cart.computeC1)

A = cart.computeConstraintGradient(x)

final_states = scipy.optimize.NonlinearConstraint(cart.computeConstraints,0,0,
                                                  jac=cart.computeConstraintGradient)

result = minimize(cart.computeTotalForces,x,method = 'SLSQP',
                  jac = cart.computeObjectiveGradient,
                  constraints = final_states, options = {'maxiter' : 5000})
print(np.asarray(result))

q = cart.computeTrajectory(result.x)
# plt.plot(t,q[:,0])
# plt.plot(t,q[:,1])
# plt.show()
cart.visualize(result.x, q, skip=1)


#A1 = con_grad(x)

#print((c1 - c)/h - np.dot(A, p))

#print((c1[0] - c[0])/h - np.dot(A1, p))
