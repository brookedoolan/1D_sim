import numpy as np
from correlations.correlations import (
    colebrook, dittus_boelter, fin_efficiency, regen_thermal_resistance,
    # haaland, gnielinski, sieder_tate  # available alternatives
)

class RegenSolver:

    def __init__(self, geometry, coolant, gas, material, flow_direction, film_cooling=None):

        self.geom = geometry
        self.coolant = coolant
        self.gas = gas
        self.material = material
        self.film_cooling = film_cooling  # optional FilmCooling instance

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
        self.q = np.zeros(N)     # Total heat flux (conv + rad)
        self.q_rad = np.zeros(N) # Radiative component
        self.T_wg = np.zeros(N)  # Gas side wall temp
        self.T_wl = np.zeros(N)  # Coolant side wall temp
        self.h_g = np.zeros(N)   # Gas-side HTC (Bartz)
        self.T_aw = np.zeros(N)  # Adiabatic wall temp (before film correction)

        # Film cooling
        self.eta_film = np.zeros(N)  # Film effectiveness at each node (0 if no film cooling)
        self.T_aw_eff = np.zeros(N)  # Effective adiabatic wall temp (modified by film)


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

            # Determine branch first — needed so properties() uses the correct
            # CEA eps for gamma (subsonic chamber ≠ supersonic nozzle at same A/At)
            if i < self.gas.throat_index:
                branch = "subsonic"
            elif i == self.gas.throat_index:
                branch = "throat"
            else:
                branch = "supersonic"

            # Gas properties from CEA (transport always at throat reference)
            gamma, cp, mu, Pr = self.gas.properties(A, branch=branch)

            self.gamma_g[i] = gamma
            self.cp_g[i] = cp
            self.mu_g[i] = mu
            self.Pr_g[i] = Pr

            # Mach from area-Mach relation
            if branch == "throat":
                self.M[i] = 1.0
                self.T_g[i] = self.gas.T0 * 2/(gamma+1)
                continue

            self.M[i] = self.gas.mach_from_area(A, gamma, branch)

            # T_static from isentropic relation — correct for all nodes including chamber
            self.T_g[i] = self.gas.T0 / (1 + (gamma-1)/2 * self.M[i]**2)

        # PHASE 1b: PRECOMPUTE FILM COOLING EFFECTIVENESS (if active)
        # Uses reference chamber diameter for Gater-L'Ecuyer s/D_ref scaling
        if self.film_cooling is not None:
            D_ref = 2 * self.geom.r[0]  # chamber diameter at injector face
            self.eta_film = self.film_cooling.effectiveness(
                self.geom.x,
                self.cp_g,
                self.gas.mdot,
                D_ref
            )

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

            # f = haaland(Re, dh[i], eps=self.geom.roughness)  # OLD Haaland friction factor
            f = colebrook(Re, dh[i], eps=self.geom.roughness)

            # ----- GAS SIDE SIGMA ITERATION -------
            T_wg = self.T_c[i] + 50 # Initial guess
            T_wl = self.T_c[i] + 10 # Initial guess for coolant-side wall temp
            tol = 1e-4 # Convergence criterion

            T0 = self.gas.T0  # Stagnation temperature (isentropic, constant along nozzle)

            for _ in range(20):

                # Dittus-Boelter (no wall-viscosity correction)
                Nu = dittus_boelter(Re, Pr_c, heating=True)
                h_c = Nu * k_c / dh[i]

                # # Sieder-Tate
                # T_wl_safe = max(T_wl, self.T_c[i])  # guard against unphysical values during iteration
                # mu_w = self.coolant.properties(T_wl_safe, self.P_c[i])[1]
                # # Nu = gnielinski(Re, Pr_c, f) * (mu_c / mu_w) ** 0.11  # old: Gnielinski + ST correction
                # Nu = sieder_tate(Re, Pr_c, mu_c, mu_w)
                # h_c = Nu * k_c / dh[i]

                # Bartz recovery factor (classical turbulent reco)
                r_factor = (Pr_g)**(1/3)
                # Adiabatic wall temp (no film cooling)
                T_aw = T0*((1+r_factor*(gamma-1)/2*M**2)
                        /(1+(gamma-1)/2*M**2)
                )

                # Film cooling: reduce effective T_aw
                if self.film_cooling is not None:
                    eta = self.eta_film[i]
                    T_aw_use = self.film_cooling.effective_T_aw(T_aw, eta)
                else:
                    T_aw_use = T_aw

                # Eckert reference temperature: T* = 0.5*(T_wg + T_static) + 0.22*(T_aw - T_static)
                # Tg is T_static here; previous form incorrectly used T0 instead of T_static
                T_star = 0.5*T_wg + Tg*(0.5 + 0.22*r_factor*(gamma-1)/2*M**2)
                mu_star = self.gas.viscosity_from_T(T_star, T_ref=Tg, mu_ref=mu_g)

                # Base bartz coeff
                h_base = self.gas.bartz_base(A, mu_star, cp_g, Pr_g)

                # Bartz sigma — correct form includes Mach correction on T_wg/T0 term
                sigma = ((0.5*(T_wg/T0)*(1+(gamma-1)/2*M**2)+0.5)**(-0.68)*(1+(gamma-1)/2*M**2)**(-0.12))

                # Gas side HTC
                h_g = h_base * sigma

                k_w = self.material.thermal_conductivity(T_wg)

                # Radiative heat flux — grey gas: q_rad = ε·σ·(T_g⁴ - T_wg⁴)
                SIGMA_SB = 5.67e-8  # Stefan-Boltzmann coeff (W/m²/K⁴)
                q_rad = self.gas.emissivity * SIGMA_SB * (Tg**4 - T_wg**4)

                # Convective heat flux from gas side (uses film-corrected T_aw)
                q_conv = h_g * (T_aw_use - T_wg)

                # Total heat into wall
                q = q_conv + q_rad

                # Fin efficiency for channel side walls (webs)
                b = max(self.geom.web_width()[i], 1e-7)  # web width at this station
                eta_f = fin_efficiency(h_c, k_w, self.geom.H[i], b)

                # Cylindrical wall + fin-corrected coolant resistance
                # R_rest_flat = self.geom.t_wall/k_w + 1/h_c  # old flat-wall form
                R_rest, R_cool = regen_thermal_resistance(
                    h_c, k_w, r, self.geom.t_wall,
                    self.geom.N, self.geom.a[i], self.geom.H[i], eta_f,
                    path_factor=self.geom.path_factor
                )

                # T_wg from coolant-side balance: q = (T_wg - T_c) / R_rest
                T_wg_new = self.T_c[i] + q * R_rest
                T_wl = self.T_c[i] + q * R_cool

                if abs(T_wg_new-T_wg)<tol:
                    break
                T_wg = T_wg_new

            self.q[i] = q
            self.q_rad[i] = q_rad
            self.T_wg[i] = T_wg
            self.T_wl[i] = T_wl
            self.h_g[i] = h_g
            self.T_aw[i] = T_aw
            self.T_aw_eff[i] = T_aw_use  # store for plotting

            # ----- PRESSURE DROP (computed first — P[j] needed for enthalpy inversion) ------
            # path_factor accounts for helical channel (ds = dx/cos(α))
            dP = f * (dx * self.geom.path_factor) / dh[i] * rho_c * u**2 / 2
            self.P_c[j] = self.P_c[i] - dP

            # ----- ENERGY UPDATE ------
            S_g = 2*np.pi*r*dx
            # changed so only accounts for area of channel interfacing with chamber wall (i.e. not full channel)
            # S_g = self.geom.N * self.geom.a[i] * dx

            # --- OLD: incremental cp*ΔT (approximation, inaccurate near phase change) ---
            # self.T_c[j] = self.T_c[i] + q*S_g/(mdot*cp_c)

            # --- NEW: enthalpy-based (Cardiff REDS approach, exact through CoolProp) ---
            # P[j] already computed above so enthalpy inversion uses correct pressure
            h_i = self.coolant.enthalpy(self.T_c[i], self.P_c[i])
            h_j = h_i + q * S_g / mdot
            self.T_c[j] = self.coolant.T_from_enthalpy(h_j, self.P_c[j])


        # Fill boundary node never reached by thermal march
        if self.flow_direction == "nozzle_to_injector":
            self.M[0] = self.M[1]
            self.T_wg[0] = self.T_wg[1]
            self.q[0] = self.q[1]
            self.q_rad[0] = self.q_rad[1]
            self.T_wl[0] = self.T_wl[1]
            self.h_g[0] = self.h_g[1]
            self.T_aw[0] = self.T_aw[1]
            self.rho_c[0] = self.rho_c[1]
            self.u_c[0] = self.u_c[1]
            self.T_aw_eff[0] = self.T_aw_eff[1]
        else:
            self.M[-1] = self.M[-2]
            self.T_wg[-1] = self.T_wg[-2]
            self.q[-1] = self.q[-2]
            self.q_rad[-1] = self.q_rad[-2]
            self.T_wl[-1] = self.T_wl[-2]
            self.h_g[-1] = self.h_g[-2]
            self.T_aw[-1] = self.T_aw[-2]
            self.rho_c[-1] = self.rho_c[-2]
            self.u_c[-1] = self.u_c[-2]
            self.T_aw_eff[-1] = self.T_aw_eff[-2]
