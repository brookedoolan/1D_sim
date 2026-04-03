import numpy as np

class ChamberStress:

    def __init__(self, geom, material, Pc_bar):

        self.geom = geom
        self.mat = material
        self.Pc = Pc_bar*1e5

    def compute(self, solver):

        N = len(solver.T_c)

        sigma_t = np.zeros(N)
        sigma_t_th = np.zeros(N)
        sigma_t_global = np.zeros(N)
        sigma_l = np.zeros(N)
        sigma_vm = np.zeros(N)
        safety = np.zeros(N)

        tw = self.geom.t_wall

        for i in range(N):
            w = self.geom.a[i]
            r = self.geom.r[i]
            H = self.geom.H[i]
            T_wg = solver.T_wg[i]
            T_wl = solver.T_wl[i]
            q = solver.q[i]
            Pco = solver.P_c[i]
            Pc = self.Pc

            # CuCr1Zr properties
            E = self.mat.youngs_modulus(T_wg)*1e9 # GPa -> Pa
            k = self.mat.thermal_conductivity(T_wg)
            nu = self.mat.poisson_ratio()
            aval = self.mat.thermal_expansion()
            sy = self.mat.yield_strength(T_wg)*1e6 # MPa -> Pa

            # Stresses
            # Local pressure hoop — inner wall as flat plate spanning channel width
            sigma_t[i] = ((Pco - Pc)/2)*(w/tw)**2

            # Thermal tangential
            sigma_t_th[i] = E*aval*q*tw/(2*(1-nu)*k)

            # Global hoop — thin-wall pressure vessel, full structural thickness
            # t_total = inner wall + channel height + outer wall (assumed = t_wall)
            t_total = 2*tw + H
            sigma_t_global[i] = Pc * r / t_total

            # Longitudinal — thin-wall thermal gradient, biaxially constrained
            sigma_l[i] = E*aval*(T_wg - T_wl)

            # Von Mises (biaxial plane stress)
            # All three tangential terms act in same direction
            sigma_hoop = sigma_t[i] + sigma_t_th[i] + sigma_t_global[i]
            sigma_vm[i] = np.sqrt(
                sigma_hoop**2 - sigma_hoop*sigma_l[i] + sigma_l[i]**2
            )

            safety[i] = sy / sigma_vm[i]

        return sigma_vm, sigma_t, sigma_t_th, sigma_t_global, sigma_l, safety