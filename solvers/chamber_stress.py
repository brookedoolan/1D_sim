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
        sigma_l = np.zeros(N)
        sigma_vm = np.zeros(N)
        safety = np.zeros(N)

        w = self.geom.a
        tw = self.geom.t_wall

        for i in range(N):
            T_wg = solver.T_wg[i]
            T_wl = solver.T_wl[i]
            q = solver.q[i]
            Pco = solver.P_c[i]
            Pc = self.Pc

            # CuCr1Zr properties
            E = self.mat.youngs_modulus(T_wg)*1e9 # GPa -> Pa
            k = self.mat.thermal_conductivity(T_wg)
            nu = self.mat.poisson_ratio()
            a = self.mat.thermal_expansion() 
            sy = self.mat.yield_strength(T_wg)*1e6 # MPa -> Pa

            # Stresses
            # pressure hoop
            sigma_t[i] = ((Pco - Pc)/2)*(w/tw)**2

            # thermal tangential
            sigma_t_th[i] = E*a*q*tw/(2*(1-nu)*k)

            # longitudinal
            sigma_l[i] = E*a*(T_wg - T_wl)

            # von mises
            sigma_vm[i] = np.sqrt(
                sigma_t[i]**2 +
                sigma_t_th[i]**2 +
                sigma_l[i]**2
            )

            safety[i] = sy / sigma_vm[i]

        return sigma_vm, sigma_t, sigma_t_th, sigma_l, safety