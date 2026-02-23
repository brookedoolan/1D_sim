from CoolProp.CoolProp import PropsSI

class CoolantModel:
    def __init__(self, mdot, fluid_name):
        self.mdot = mdot
        self.fluid = fluid_name
    
    def properties(self, T, P):
        rho = PropsSI("D", "T", T, "P", P, self.fluid)
        mu = PropsSI("V", "T", T, "P", P, self.fluid)
        k = PropsSI("L", "T", T, "P", P, self.fluid)
        cp = PropsSI("C", "T", T, "P", P, self.fluid)

        return rho, mu, k, cp