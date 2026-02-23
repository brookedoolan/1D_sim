import numpy as np
from correlations.correlations import haaland, gnielinski

class RegenSolver:

    def __init__(self, geometry, coolant, gas):
        self.geom = geometry
        self.coolant = coolant
        self.gas = gas

        N = geometry.n_nodes

        # Solution arrays
        self.T_c = np.zeros(N)
        self.P_c = np.zeros(N)
        self.q = np.zeros(N)
        self.T_wg = np.zeros(N)
        self.T_wl = np.zeros(N)

    def solve(self, T_in, P_in):

        self.T_c[0] = T_in
        self.P_c[0] = P_in

        dh = self.geom.hydraulic_diameter()
        A_flow = self.geom.total_flow_area()
        dx = self.geom.dx

        for i in range(self.geom.n_nodes - 1):
            
            # COOLANT SIDE
            # Coolant properties
            rho, mu, k, cp = self.coolant.properties(
                self.T_c[i], self.P_c[i]
            )

            G = self.coolant.mdot/A_flow # Mass flux
            u = G/rho # Velocity

            Re = G*dh/mu # Reynolds
            Pr = cp*mu/k # Prandtl

            f = haaland(Re, dh, eps=1e-5)

            Nu = gnielinski(Re, Pr, f)

            h_c = Nu*k/dh # Coolant side heat transfer coeff

            # GAS SIDE (simplified constant)
            r = self.geom.r[i]
            A = np.pi*r**2

            mu_g = 3e-5
            cp_g = 3000
            Pr_g = 0.7

            h_g = self.gas.heat_transfer_coefficient(
                A, mu_g, cp_g, Pr_g
            )

            # Recovery temperature approx
            T_aw = self.gas.Tc

            # Heat flux
            R_total = 1/h_g + self.geom.t_wall/400 + 1/h_c
            q = (T_aw-self.T_c[i])/R_total
            self.q[i] = q

            # Wall temps
            self.T_wg[i] = T_aw - q/h_g
            self.T_wl[i] = self.T_c[i] + q/h_c

            # Energy update
            S_g = 2*np.pi*r*dx

            self.T_c[i+1] = self.T_c[i] + q*S_g / (self.coolant.mdot * cp)

            # Pressure drop
            dP = f*dx/dh*rho*u**2/2
            self.P_c[i+1] = self.P_c[i] - dP