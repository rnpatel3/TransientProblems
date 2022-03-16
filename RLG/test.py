
class Assembler:

    def setVariables(self, u, udot, uddot):
        self.u[:] = u[:]
        self.udot[:] = udot[:]
        self.uddot[:] = uddot[:]
        return

    def assembleJacobian(self, alpha, beta, gamma, res, mat):
        res[:] = np.dot(self.K, self.u) + np.dot(self.C, self.udot) + np.dot(self.M, self.uddot)
        mat[:] = alpha*self.K + beta*self.C + gamma*self.M
        return 

# Acceleration, velocity and displacement at n+1
uddot = (5.42)
udot = (5.43)
u = # Estimate

for i in range(newton_iters):
    assembler.setVariables(u, udot, uddot)

    tacs_alpha = 1.0
    tacs_beta = gamma/(beta*delta_t) # Eq 5.43
    tacs_gamma = 1.0/(beta*delta_t**2) # Eq 5.42
    
    assembler.assembleJacobian(tacs_alpha, tacs_beta, tacs_gamma,
                               res, mat)

    if res.norm() < tol:
        break

    pc.factor()
    gmres.solve(res, update)

    u.axpy(-1.0, update)
    udot.axpy(-1.0, tacs_beta, update)
    uddot.axpy(-1.0, tacs_gamma, update)

    update = np.linalg.solve(mat, res)
    

# Now 
