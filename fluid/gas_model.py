import numpy as np

# Bartz 
# WITHOUT sigma correlation
# INITIALLY gas properties constant but get from NASA CEA/COOLPROP

class GasModel:
    def __init__(self, Pc, Tc, gamma, c_star, At):
        self.Pc = Pc
        self.Tc = Tc
        self.gamma = gamma
        self.c_star = c_star
        self.At = At

    def mach_from_area(self, A):
        gamma = self.gamma
        At = self.At

        # Solve area-Mach relation numerically
        from scipy.optimize import fsolve

        def func(M):
            return (1/M)*\
                   ((2/(gamma+1)) *
                    (1 + (gamma-1)/2*M**2))**((gamma+1)/(2*(gamma-1))) \
                   - A/At

        M_guess = 0.1 if A > At else 2.0
        M = fsolve(func, M_guess)[0]
        return M
    
    def heat_transfer_coefficient(self, A, mu_g, cp_g, Pr_g):
        C = 0.026
        Pc = self.Pc
        c_star = self.c_star
        At = self.At

        hg = C * mu_g**0.2 * cp_g * Pr_g**(-0.6) * \
             (Pc/c_star)**0.8 * \
             (At/A)**0.9

        return hg