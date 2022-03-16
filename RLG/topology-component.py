import numpy as np
import openmdao.api as om
from scipy import sparse
from scipy.sparse import linalg
import matplotlib.pyplot as plt
from paropt.paropt_driver import ParOptDriver
import argparse

'''
Component for the topology optimization analysis which computes the
effective mass and stiffness scalar quantities
'''

class TopoAnalysis(om.ExplicitComponent):
    def __init__(self, nxelems, nyelems, Lx, Ly, r0=1.5, p=3.0,
                 E0=1.0, nu=0.3, draw_figure=False):
        super().__init__()

        self.nxelems = nxelems
        self.nyelems = nyelems
        self.nelems = self.nxelems*self.nyelems
        self.xfilter = None
        self.Lx = Lx
        self.Ly = Ly
        self.r0 = r0
        self.p = p
        self.E0 = E0
        self.nu = nu
        self.draw_figure = draw_figure
        self.iter_count = 0

        # Set the element variables and boundary conditions
        self.nvars = 2*( self.nxelems+1)*(self.nyelems+1)
        self.uvars = np.arange(0, self.nvars, 2, dtype=np.int).reshape(self.nyelems+1, -1)
        self.vvars = np.arange(1, self.nvars, 2, dtype=np.int).reshape(self.nyelems+1, -1)

        # Set the element variable values
        self.elem_vars = np.zeros((self.nelems, 8), dtype=np.int)

        for j in range(self.nyelems):
            for i in range(self.nxelems):
                elem = i + j*self.nxelems
                self.elem_vars[elem, 0] = self.uvars[j,i]
                self.elem_vars[elem, 1] = self.vvars[j,i]
                self.elem_vars[elem, 2] = self.uvars[j,i+1]
                self.elem_vars[elem, 3] = self.vvars[j,i+1]
                self.elem_vars[elem, 4] = self.uvars[j+1,i]
                self.elem_vars[elem, 5] = self.vvars[j+1,i]
                self.elem_vars[elem, 6] = self.uvars[j+1,i+1]
                self.elem_vars[elem, 7] = self.vvars[j+1,i+1]

        # Set the boundary conditions
        self.bcs = np.hstack((self.uvars[:,0], self.vvars[:,0]))

        # Set the force vector
        self.unit_force = 4
        self.f = np.zeros(self.nvars)
        self.f[self.uvars[0, self.nxelems]] = self.unit_force
        self.f[self.bcs] = 0.0

        # Now, compute the filter weights and store them as a sparse
        # matrix
        F = sparse.lil_matrix((self.nxelems*self.nyelems,
                               self.nxelems*self.nyelems))

        # Compute the inter corresponding to the filter radius
        ri = int(np.ceil(self.r0))

        for j in range(self.nyelems):
            for i in range(self.nxelems):
                w = []
                wvars = []

                # Compute the filtered design variable: xfilter
                for jj in range(max(0, j-ri), min(self.nyelems, j+ri+1)):
                    for ii in range(max(0, i-ri), min(self.nxelems, i+ri+1)):
                        r = np.sqrt((i - ii)**2 + (j - jj)**2)
                        if r < self.r0:
                            w.append((self.r0 - r)/self.r0)
                            wvars.append(ii + jj*self.nxelems)

                # Normalize the weights
                w = np.array(w)
                w /= np.sum(w)

                # Set the weights into the filter matrix W
                F[i + j*self.nxelems, wvars] = w

        # Covert the matrix to a CSR data format
        self.F = F.tocsr()

        return

    def setup(self):

        # Define the inputs
        self.add_input('x', 0.95*np.ones(self.nelems),
                       desc='topology design variables',
                       shape=(self.nxelems*self.nyelems))

        # Define the outputs
        self.add_output('m', desc='actual beam mass')
        self.add_output('m1', desc='equivalent mass')
        self.add_output('k1', desc='beam equivalent stiffness')

        # Define the partials
        self.declare_partials(of=['m', 'm1', 'k1'], wrt='x')

        return

    def mass(self, x):
        """
        Compute the mass of the structure
        """

        area = (self.Lx/self.nxelems)*(self.Ly/self.nyelems)

        return area*np.sum(x)

    def mass_grad(self, x):
        """
        Compute the derivative of the mass
        """

        area = (self.Lx/self.nxelems)*(self.Ly/self.nyelems)
        dmdx = area*np.ones(x.shape)

        return dmdx

    def compliance(self, x):
        """
        Compute the structural compliance
        """

        # Compute the filtered compliance. Note that 'dot' is scipy
        # matrix-vector multiplicataion
        xfilter = self.F.dot(x)

        # Compute the Young's modulus in each element
        E = self.E0*xfilter**self.p
        self.analyze_structure(E)

        # Return the compliance
        return 0.5*np.dot(self.f, self.u)

    def compliance_grad(self, x):
        """
        Compute the gradient of the compliance using the adjoint
        method.

        Since the governing equations are self-adjoint, and the
        function itself takes a special form:

        K*psi = 0.5*f => psi = 0.5*u

        So we can skip the adjoint computation itself since we have
        the displacement vector u from the solution.

        d(compliance)/dx = - 0.5*u^{T}*d(K*u - f)/dx = - 0.5*u^{T}*dK/dx*u
        """

        # Compute the filtered variables
        self.xfilter = self.F.dot(x)

        # First compute the derivative with respect to the filtered
        # variables
        dcdxf = np.zeros(x.shape)

        # Sum up the contributions from each
        kelem = self.compute_element_stiffness()

        for i in range(self.nelems):
            evars = self.u[self.elem_vars[i, :]]
            dxfdE = self.E0*self.p*self.xfilter[i]**(self.p - 1.0)
            dcdxf[i] = -0.5*np.dot(evars, np.dot(kelem, evars))*dxfdE

        # Now evaluate the effect of the filter
        dcdx = (self.F.transpose()).dot(dcdxf)

        return dcdx

    def analyze_structure(self, E):
        """
        Given the elastic modulus variable values, perform the
        analysis and update the state variables.

        This function sets up and solves the linear finite-element
        problem with the given set of elastic moduli. Note that E > 0
        (component wise).

        Args:
           E: An array of the elastic modulus for every element in the
              plane stress domain
        """

        # Compute the finite-element stiffness matrix
        kelem = self.compute_element_stiffness()

        # Set all the values, (duplicate entries are added together)
        data = np.zeros((self.nelems, 8, 8))
        i = np.zeros((self.nelems, 8, 8), dtype=np.int)
        j = np.zeros((self.nelems, 8, 8), dtype=np.int)
        for k in range(self.nelems):
            data[k] = E[k]*kelem
            for kk in range(8):
                i[k,:,kk] = self.elem_vars[k, :]
                j[k,kk,:] = self.elem_vars[k, :]

        # Assemble things as a COO format
        K = sparse.coo_matrix((data.flatten(), (i.flatten(), j.flatten())),
                              shape=(self.nvars, self.nvars))

        # Convert to list-of-lists to apply BCS
        K = K.tolil()
        K[:, self.bcs] = 0.0
        K[self.bcs, :] = 0.0
        K[self.bcs, self.bcs] = 1.0

        # Convert to csc format for factorization
        self.K = K.tocsc()

        # Solve the sparse linear system for the load vector
        self.LU = linalg.dsolve.factorized(self.K)

        # Compute the solution to the linear system K*u = f
        self.u = self.LU(self.f)

        return

    def compute_element_stiffness(self):
        """
        Compute the element stiffness matrix using a Gauss quadrature
        scheme.

        Note that this code assumes that all elements are uniformly
        rectangular and so the same element stiffness matrix can be
        used for every element.
        """

        # Compute the element stiffness matrix
        gauss_pts = [-1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0)]

        # Create the 8 x 8 element stiffness matrix
        kelem = np.zeros((8, 8))
        B = np.zeros((3, 8))

        # Compute the constitutivve matrix
        C = np.array([[1.0, self.nu, 0.0],
                      [self.nu, 1.0, 0.0],
                      [0.0, 0.0, 0.5*(1.0 - self.nu)]])
        C = 1.0/(1.0 - self.nu**2)*C

        # Set the terms for the area-dependences
        xi = 2.0*self.nxelems/self.Lx
        eta = 2.0*self.nyelems/self.Ly
        area = 1.0/(xi*eta)

        for x in gauss_pts:
            for y in gauss_pts:
                # Evaluate the derivative of the shape functions with
                # respect to the x/y directions
                Nx = 0.25*xi*np.array([y - 1.0, 1.0 - y, -1.0 - y, 1.0 + y])
                Ny = 0.25*eta*np.array([x - 1.0, -1.0 - x, 1.0 - x, 1.0 + x])

                # Evaluate the B matrix
                B = np.array(
                    [[ Nx[0], 0.0, Nx[1], 0.0, Nx[2], 0.0, Nx[3], 0.0 ],
                     [ 0.0, Ny[0], 0.0, Ny[1], 0.0, Ny[2], 0.0, Ny[3] ],
                     [ Ny[0], Nx[0], Ny[1], Nx[1], Ny[2], Nx[2], Ny[3], Nx[3] ]])

                # Add the contribution to the stiffness matrix
                kelem += area*np.dot(B.transpose(), np.dot(C, B))

        return kelem

    def equivalent_stiffness(self, x):

        c = self.compliance(x)
        k1 = self.unit_force/c

        return k1

    def equivalent_stiffness_grad(self, x):

        c = self.compliance(x)
        dcdx = self.compliance_grad(x)
        dk1dx = (-self.unit_force/(c**2))*dcdx

        return dk1dx

    def compute(self, inputs, outputs):

        x = inputs['x']

        # Compute the static deflection with a unit load
        self.analyze_structure(x)

        # Compute the outputs
        m = self.mass(x)
        # m1 = self.equivalent_mass(x)
        k1 = self.equivalent_stiffness(x)
        outputs['m'] = m
        outputs['m1'] = m
        outputs['k1'] = k1*1e3
        # outputs['k1'] = k1

        self.iter_count += 1
        print('------[%3d]------' % self.iter_count)
        print("x avg:  %15.10f" % (np.sum(x) / len(x)))
        print("x min:  %15.10f" % np.min(x))
        print("x max:  %15.10f" % np.max(x))
        print("topo-k1:%15.3e" % outputs['k1'])

        return

    def compute_partials(self, inputs, partials):

        x = inputs['x']

        # Compute the partials
        dmdx = self.mass_grad(x)
        # dm1dx = self.equivalent_mass_grad(x)
        dk1dx = self.equivalent_stiffness_grad(x)
        partials['m', 'x'] = dmdx
        partials['m1', 'x'] = dmdx
        partials['k1', 'x'] = dk1dx*1e3
        # partials['k1', 'x'] = dk1dx

        self.write_output(x[:])

        return

    def write_output(self, x):
        """
        Write out something to the screen
        """

        if self.draw_figure:
            if not hasattr(self, 'fig'):
                plt.ion()
                self.fig, self.ax = plt.subplots()
                plt.draw()

            xfilter = self.F.dot(x)

            # Prepare a pixel visualization of the design vars
            image = np.zeros((self.nyelems, self.nxelems))
            for j in range(self.nyelems):
                for i in range(self.nxelems):
                    image[j, i] = xfilter[i + j*self.nxelems]

            x = np.linspace(0, self.Lx, self.nxelems)
            y = np.linspace(0, self.Ly, self.nyelems)

            self.ax.contourf(x, y, image)
            self.ax.set_aspect('equal', 'box')
            plt.draw()
            plt.pause(0.001)
            plt.savefig('topo.png')

        return

if __name__ == '__main__':
    '''
    Run the beam and spring-mass system optimization
    '''

    # Add options
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', default='ParOpt',
                    choices=['ParOpt', 'SLSQP', 'SNOPT', 'IPOPT'],
                    help='optimizer')
    args = parser.parse_args()
    optimizer = args.optimizer

    # Set the problem geometry, material properties, and
    # topology optimization parameters
    nxelems = 150
    nyelems = 50
    ndvs = nxelems*nyelems
    ndofs = 2*ndvs
    Lx = 3.0
    Ly = 1.0
    r0 = 3
    p = 3.0
    E = 1.0
    nu = 0.3
    mf = 0.3*Lx*Ly

    # Create the problem object
    prob = om.Problem()

    # Create the independent variable component
    indeps = prob.model.add_subsystem('indeps', om.IndepVarComp())
    indeps.add_output('x', 0.95*np.ones(ndvs))
    indeps.add_output('mf', mf)

    # Add the topology analysis subsystem and make connections
    prob.model.add_subsystem('topo',
                             TopoAnalysis(nxelems, nyelems,
                                          Lx, Ly, r0=r0,
                                          p=p, E0=E, nu=nu,
                                          draw_figure=True))
    prob.model.connect('indeps.x', 'topo.x')

    # Set up the subsystem for the mass constraint
    class Con(om.ExplicitComponent):
        def setup(self):
            self.add_input('m', val=1.0)
            self.add_input('mf', val=1.0)
            self.add_output('c', shape=(1,))
            self.declare_partials(of='c', wrt='mf')
            self.declare_partials(of='c', wrt='m')
        def compute(self, inputs, outputs):
            outputs['c'] = inputs['mf'] - inputs['m']
        def compute_partials(self, inputs, partials):
            partials['c', 'mf'] = 1.0
            partials['c', 'm'] = -1.0
    prob.model.add_subsystem('con', Con())
    prob.model.connect('indeps.mf', 'con.mf')
    prob.model.connect('topo.m', 'con.m')

    # Add another component to flip objective
    prob.model.add_subsystem('obj', om.ExecComp('kobj = -k1'))
    prob.model.connect('topo.k1', 'obj.k1')


    # Define the optimization problem
    prob.model.add_design_var('indeps.x', lower=0.0, upper=1.0)
    prob.model.add_objective('obj.kobj', scaler=1.0)
    prob.model.add_constraint('con.c', lower=0.0)

    # Create the ParOpt driver
    if optimizer != 'paropt':
        prob.driver = om.pyOptSparseDriver()
        prob.driver.options['optimizer'] = optimizer
    else:
        prob.driver = ParOptDriver()

        # Set options for the paropt driver
        options = {
            'algorithm': 'tr',
            'tr_init_size': 0.05,
            'tr_min_size': 1e-6,
            'tr_max_size': 10.0,
            'tr_eta': 0.25,
            'tr_infeas_tol': 1e-6,
            'tr_l1_tol': 1e-3,
            'tr_linfty_tol': 0.0,
            'tr_adaptive_gamma_update': True,
            'tr_max_iterations': 200,
            'penalty_gamma': 10.0,
            'qn_subspace_size': 10,
            'qn_type': 'bfgs',
            'qn_diag_type': 'yts_over_sts',
            'abs_res_tol': 1e-8,
            'starting_point_strategy': 'affine_step',
            'barrier_strategy': 'mehrotra_predictor_corrector',
            'tr_steering_barrier_strategy':
            'mehrotra_predictor_corrector',
            'tr_steering_starting_point_strategy': 'affine_step',
            'use_line_search': False,
            'gradient_verification_frequency': -1}

        for key in options:
            prob.driver.options[key] = options[key]

    # Run the problem
    prob.setup()
    prob.run_driver()
