import numpy as np
from correlations.correlations import haaland, gnielinski

class RegenSolver:

    def __init__(self, geometry, coolant, gas, material):
        
        self.geom = geometry
        self.coolant = coolant
        self.gas = gas
        self.material = material 

        N = geometry.n_nodes

        # Coolant 
        self.T_c = np.zeros(N) # Coolant temp
        self.P_c = np.zeros(N) # Coolant pressure

        # Gas/Flow
        self.M = np.zeros(N) # Mach number
        self.gamma_g = np.zeros(N) # Gamma (CC gas)
        self.Pr_g = np.zeros(N) # Prandtil (CC gas)
        self.T_g = np.zeros(N) # Hot CC gas temp
        self.mu_g = np.zeros(N) # Viscocity (CC gas)
        self.cp_g = np.zeros(N) # Specific heat (CC gas)

        # Thermal
        self.q = np.zeros(N) # Heat flux
        self.T_wg = np.zeros(N) # Gas side wall temp
        self.T_wl = np.zeros(N) # Coolant side wall temp

    def solve(self, T_in, P_in):

        N = self.geom.n_nodes 
        throat_index = self.geom.throat_index()

        self.T_c[0] = T_in
        self.P_c[0] = P_in

        dh = self.geom.hydraulic_diameter()
        A_flow = self.geom.total_flow_area()
        
        # PHASE 1: PRECOMPUTE GAS AND FLOW
        for i in range(N):

            r = self.geom.r[i]
            A = np.pi*self.geom.r[i]**2
            
            # Gas properties from CEA
            gamma, cp, mu, Pr, T_static = self.gas.properties(A)

            self.gamma_g[i] = gamma
            self.cp_g[i] = cp
            self.mu_g[i] = mu
            self.Pr_g[i] = Pr
            self.T_g[i] = T_static

            # Mach marching
            if i == 0:
                self.M[i] = 0.05
            else:
                self.M[i] = self.gas.mach_from_area(A, gamma, self.M[i-1])

        # PHASE 2: THERMAL MARCHING
        for i in range(N - 1):
            
            dx = self.geom.dx[i]
            r = self.geom.r[i]
            A = np.pi*r**2

            gamma = self.gamma_g[i]
            cp_g = self.cp_g[i]
            mu_g = self.mu_g[i]
            Pr_g = self.Pr_g[i]
            Tg = self.T_g[i]
            M = self.M[i]

            # ------ COOLANT SIDE ------
            # Coolant properties
            rho_c, mu_c, k_c, cp_c = self.coolant.properties(
                self.T_c[i], self.P_c[i]
            )

            G = self.coolant.mdot/A_flow # Mass flux
            u = G/rho_c # Velocity

            Re = G*dh/mu_c # Reynolds
            Pr_c = cp_c*mu_c/k_c # Prandtl

            f = haaland(Re, dh, eps=1e-5)
            Nu = gnielinski(Re, Pr_c, f)

            h_c = Nu*k_c/dh # Coolant side heat transfer coeff

            # ------ GAS SIDE ---------
            # Recovery temp
            r_factor = Pr_g**(1/3) # Recovery factor for turbulent flow (approx.)
            T_aw = Tg*(1+r_factor*(gamma-1)/2*M**2) # Tg is static gas temp

            # Base Bartz coeff
            h_base = self.gas.bartz_base(A, mu_g, cp_g, Pr_g)

            # Wall material conductivity 
            T_wall_avg = (self.T_c[i]+T_aw)/2 # Assume average wall temp
            k_w = self.material.thermal_conductivity(T_wall_avg)

            # ------ SIGMA ITERATION -------
            sigma = 1.0 # Initial guess, usually around 0.6-1.0
            eps = 1e-4 # Convergence criterion
            
            for _ in range(20):
                
                h_g = h_base*sigma

                # Total thermal resistance (CC conv -> wall cond -> cool conv)
                R_total = 1/h_g + self.geom.t_wall/k_w + 1/h_c
                
                # Heat flux through wall
                q = (T_aw - self.T_c[i])/R_total

                T_wg = T_aw - q/h_g

                sigma_new = (
                    (0.5*(T_wg/T_aw)*(1+(gamma-1)/2*M**2)+0.5)
                    **(-0.68)*(1+(gamma-1)/2*M**2)**(-0.12)
                )

                if abs(sigma_new-sigma)<eps:
                    break
                sigma = sigma_new
            
            # Final converged values
            h_g = h_base*sigma 
            R_total = 1/h_g + self.geom.t_wall/k_w + 1/h_c
            q = (T_aw - self.T_c[i])/R_total
            
            self.q[i] = q
            self.T_wg[i] = T_aw - q/h_g
            self.T_wl[i] = self.T_c[i] + q/h_c

            # ----- ENERGY UPDATE ------
            S_g = 2*np.pi*r*dx

            self.T_c[i+1] = self.T_c[i] + q*S_g/(self.coolant.mdot*cp_c)

            # Pressure drop
            dP = f*dx/dh*rho_c*u**2/2 # This assumes velocity is constant, but changes
            self.P_c[i+1] = self.P_c[i] - dP

        
        # Remove zeros at end (never computed)
        self.M[-1] = self.M[-2]
        self.T_wg[-1] = self.T_wg[-2]
        self.q[-1] = self.q[-2]
        self.T_wl[-1] = self.T_wl[-2]