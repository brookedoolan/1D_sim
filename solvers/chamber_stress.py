import numpy as np

def _thick_wall_hoop(P_a: float, P_b: float, a: float, b: float, r: float) -> float:
    """Lamé hoop stress for a thick-walled cylinder evaluated at radius r.
    P_a : pressure at inner surface (r = a)
    P_b : pressure at outer surface (r = b)
    """
    denom = b**2 - a**2
    return (P_a * a**2 - P_b * b**2) / denom + (a**2 * b**2 * (P_a - P_b)) / (denom * r**2)

class ChamberStress:

    def __init__(self, geom, material, Pc_bar):

        self.geom = geom
        self.mat = material
        self.Pc = Pc_bar * 1e5  # retained as fallback reference pressure

    def compute(self, solver):

        N = len(solver.T_c)

        sigma_t = np.zeros(N)
        sigma_t_th = np.zeros(N)
        sigma_t_global = np.zeros(N)
        sigma_l = np.zeros(N)
        sigma_vm = np.zeros(N)
        safety = np.zeros(N)

        tw = self.geom.t_wall
        P0 = solver.gas.Pc_bar * 1e5  # stagnation pressure [Pa]

        for i in range(N):
            w    = self.geom.a[i]
            r    = self.geom.r[i]
            H    = self.geom.H[i]
            T_wg = solver.T_wg[i]
            T_wl = solver.T_wl[i]
            q    = solver.q[i]

            gamma = solver.gamma_g[i]
            M     = solver.M[i]

            # Local gas static pressure from isentropic relation
            P_gas     = P0 / (1.0 + (gamma - 1.0) / 2.0 * M**2) ** (gamma / (gamma - 1.0))
            P_coolant = solver.P_c[i]

            # CuCr1Zr properties
            E    = self.mat.youngs_modulus(T_wg) * 1e9   # GPa -> Pa
            k    = self.mat.thermal_conductivity(T_wg)
            nu   = self.mat.poisson_ratio()
            aval = self.mat.thermal_expansion()
            sy   = self.mat.yield_strength(T_wg) * 1e6   # MPa -> Pa

            # 1. Local pressure hoop — inner wall as flat plate spanning channel width
            sigma_t[i] = ((P_coolant - P_gas) / 2.0) * (w / tw) ** 2

            # 2. Thermal tangential
            sigma_t_th[i] = E * aval * q * tw / (2.0 * (1.0 - nu) * k)

            # 3. Global hoop — Lamé thick-wall, inner=gas, outer=coolant
            t_total = 2.0 * tw + H
            b_r = r + t_total
            sigma_t_global[i] = _thick_wall_hoop(P_gas, P_coolant, r, b_r, r)

            # 4. Longitudinal — biaxially constrained thermal gradient
            sigma_l[i] = E * aval * (T_wg - T_wl)

            # 5. Von Mises (biaxial plane stress)
            sigma_hoop = sigma_t[i] + sigma_t_th[i] + sigma_t_global[i]
            sigma_vm[i] = np.sqrt(
                sigma_hoop**2 - sigma_hoop * sigma_l[i] + sigma_l[i]**2
            )

            safety[i] = sy / sigma_vm[i] if sigma_vm[i] > 0 else np.inf

        return sigma_vm, sigma_t, sigma_t_th, sigma_t_global, sigma_l, safety