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

            # Mach marching - determine flow branch relative to throat
            if i < self.gas.throat_index:
                branch = "subsonic"
            elif i == self.gas.throat_index:
                self.M[i] = 1.0
                continue
            else:
                branch = "supersonic"

            self.M[i] = self.gas.mach_from_area(A, gamma, branch)

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

            # ----- GAS SIDE SIGMA ITERATION -------
            T_wg = self.T_c[i] + 50 # Initial guess
            tol = 1e-4 # Convergence criterion
            
            for _ in range(20):
                # Reco temp
                # Stagnation temperature
                T0 = Tg*(1+(gamma-1)/2*M**2)

                # Bartz recovery factor - added dependancy on wall temp
                r_factor = (Pr_g**(1/3))*(T_wg/T0)**0.25

                # Adiabatic wall temp
                T_aw = T0*((1+r_factor*(gamma-1)/2*M**2)
                        /(1+(gamma-1)/2*M**2)
                )

                # Film temp
                #T_film = 0.5*(T_wg+T0)
                #mu_film = self.gas.viscosity_from_T(T_film, T_ref=Tg, mu_ref=mu_g)

                # Instead using Bartz reference temp
                T_star = T0*(0.5+0.5*(T_wg/T0)+0.22*r_factor*M**2)
                mu_star = self.gas.viscosity_from_T(T_star, T_ref=Tg, mu_ref=mu_g)

                # Base bartz coeff
                h_base = self.gas.bartz_base(A, mu_star, cp_g, Pr_g)

                # Compute sigma using current wall temp
                sigma = ((0.5*(T_wg/T0)+0.5)**(-0.68)*(1+(gamma-1)/2*M**2)**(-0.12))

                # Gas side HTC
                h_g = h_base*sigma

                # Total thermal resistance (CC conv -> wall cond -> cool conv)
                k_w = self.material.thermal_conductivity(T_wg)
                R_total = 1/h_g + self.geom.t_wall/k_w + 1/h_c
                
                # Heat flux through wall
                q = (T_aw - self.T_c[i])/R_total

                T_wg_new = T_aw - q/h_g
                if abs(T_wg_new-T_wg)<tol:
                    break
                T_wg = T_wg_new
                    
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