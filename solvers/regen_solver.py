import numpy as np
from correlations.correlations import haaland, gnielinski

class RegenSolver:

    def __init__(self, geometry, coolant, gas, material, flow_direction):

        self.geom = geometry
        self.coolant = coolant
        self.gas = gas
        self.material = material

        N = geometry.n_nodes

        # Coolant flow direction flag
        self.flow_direction = flow_direction

        # Coolant
        self.T_c = np.zeros(N) # Coolant temp
        self.P_c = np.zeros(N) # Coolant pressure
        self.rho_c = np.zeros(N) # Coolant density
        self.u_c = np.zeros(N) # Coolant velocity

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

        # Adjust flow direction
        if self.flow_direction == "injector_to_nozzle":
            march_nodes = np.arange(N)
        elif self.flow_direction == "nozzle_to_injector":
            march_nodes = np.arange(N)[::-1]
        else:
            print(f"No flow direction specified. Assuming Reveser flow: Nozzle to Injector.")
            march_nodes = np.arange(N)[::-1]

        if self.flow_direction == "injector_to_nozzle":
            self.T_c[0] = T_in
            self.P_c[0] = P_in
        elif self.flow_direction == "nozzle_to_injector":
            self.T_c[-1] = T_in
            self.P_c[-1] = P_in
        else:
            print(f"No flow direction specified. Assuming Reveser flow: Nozzle to Injector.")
            self.T_c[-1] = T_in
            self.P_c[-1] = P_in

        dh = self.geom.hydraulic_diameter()      # array, indexed per node
        A_flow = self.geom.total_flow_area()     # array, indexed per node

        mdot = self.coolant.mdot

        # PHASE 1: PRECOMPUTE GAS AND FLOW
        for i in range(N):

            r = self.geom.r[i]
            A = np.pi*self.geom.r[i]**2

            # Gas properties from CEA
            gamma, cp, mu, Pr = self.gas.properties(A)

            self.gamma_g[i] = gamma
            self.cp_g[i] = cp
            self.mu_g[i] = mu
            self.Pr_g[i] = Pr

            # Mach from area-Mach relation
            if i < self.gas.throat_index:
                branch = "subsonic"
            elif i == self.gas.throat_index:
                self.M[i] = 1.0
                self.T_g[i] = self.gas.T0 * 2/(gamma+1)
                continue
            else:
                branch = "supersonic"

            self.M[i] = self.gas.mach_from_area(A, gamma, branch)

            # T_static from isentropic relation — correct for all nodes including chamber
            self.T_g[i] = self.gas.T0 / (1 + (gamma-1)/2 * self.M[i]**2)

        # PHASE 2: THERMAL MARCHING
        for k in range(N-1):
            i = march_nodes[k]
            j = march_nodes[k+1]

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

            G = mdot/A_flow[i] # Mass flux
            u = G/rho_c # Velocity
            self.rho_c[i] = rho_c
            self.u_c[i] = u

            Re = G*dh[i]/mu_c # Reynolds
            Pr_c = cp_c*mu_c/k_c # Prandtl

            f = haaland(Re, dh[i], eps=1e-5)

            # ----- GAS SIDE SIGMA ITERATION -------
            T_wg = self.T_c[i] + 50 # Initial guess
            T_wl = self.T_c[i] + 10 # Initial guess for coolant-side wall temp
            tol = 1e-4 # Convergence criterion

            T0 = self.gas.T0  # Stagnation temperature (isentropic, constant along nozzle)

            for _ in range(20):

                # Sieder-Tate wall viscosity correction on Gnielinski
                mu_w = self.coolant.properties(T_wl, self.P_c[i])[1]
                Nu = gnielinski(Re, Pr_c, f) * (mu_c / mu_w) ** 0.11
                h_c = Nu*k_c/dh[i]

                # Bartz recovery factor (classical turbulent reco)
                r_factor = (Pr_g)**(1/3)
                # Bartz recovery factor - added dependancy on wall temp??
                #r_factor = (Pr_g**(1/3))*(T_wg/T0)**0.25

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
                h_g = h_base * sigma

                k_w = self.material.thermal_conductivity(T_wg)

                # Total thermal resistance (CC conv -> inner wall cond -> cool conv)
                # Web/side walls assumed negligible heat transfer (small temp gradient)
                R_total = 1/h_g + self.geom.t_wall/k_w + 1/h_c

                # Heat flux through wall
                q = (T_aw - self.T_c[i])/R_total

                T_wg_new = T_aw - q/h_g
                T_wl = self.T_c[i] + q/h_c  # update coolant-side wall temp for next iteration

                if abs(T_wg_new-T_wg)<tol:
                    break
                T_wg = T_wg_new

            self.q[i] = q
            self.T_wg[i] = T_aw - q/h_g
            self.T_wl[i] = self.T_c[i] + q/h_c

            # ----- ENERGY UPDATE ------
            #S_g = 2*np.pi*r*dx
            # changed so only accounts for area of channel interfacing with chamber wall (i.e. not full channel)
            S_g = self.geom.N * self.geom.a[i] * dx

            self.T_c[j] = self.T_c[i] + q*S_g/(mdot*cp_c)

            # Pressure drop
            dP = f*dx/dh[i]*rho_c*u**2/2 # This assumes velocity is constant, but changes
            self.P_c[j] = self.P_c[i] - dP


        # Fill boundary node never reached by thermal march
        if self.flow_direction == "nozzle_to_injector":
            self.M[0] = self.M[1]
            self.T_wg[0] = self.T_wg[1]
            self.q[0] = self.q[1]
            self.T_wl[0] = self.T_wl[1]
            self.rho_c[0] = self.rho_c[1]
            self.u_c[0] = self.u_c[1]
        else:
            self.M[-1] = self.M[-2]
            self.T_wg[-1] = self.T_wg[-2]
            self.q[-1] = self.q[-2]
            self.T_wl[-1] = self.T_wl[-2]
            self.rho_c[-1] = self.rho_c[-2]
            self.u_c[-1] = self.u_c[-2]