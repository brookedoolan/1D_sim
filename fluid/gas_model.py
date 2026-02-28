import numpy as np
from rocketcea.cea_obj_w_units import CEA_Obj
from scipy.optimize import brentq, fsolve

# Bartz 

class GasModel:
    
    def __init__(self, Pc_bar, MR, geometry, ox_name, fuel_name, mdot=None, cstar=None):
        
        self.Pc_bar = Pc_bar
        self.MR = MR

        self.geometry = geometry
        self.At = geometry.throat_area()
        self.Rt = geometry.throat_radius_of_curvature()

        self.Dt = np.sqrt(4*self.At/np.pi)

        self.cea = CEA_Obj(
            oxName=ox_name,
            fuelName=fuel_name,
            pressure_units='bar',
            temperature_units='K',
            specific_heat_units='J/kg-K',
            enthalpy_units='J/kg',
            density_units='kg/m^3',
            sonic_velocity_units='m/s',
            cstar_units='m/s',
            viscosity_units='millipoise',
            thermal_cond_units='W/cm-degC' # Need to convert to W/m-K, 1 W/cm-K = 100 W/m-K
        )

        # Either provided by RPA or computed from CEA
        if cstar is None:
            self.c_star = self.cea.get_Cstar(Pc=self.Pc_bar, MR=self.MR)
        else:
            self.c_star = cstar

        # Enforce choked mass flow
        if mdot is None:
            self.mdot = self.Pc_bar*1e5*self.At/self.c_star
        else:
            self.mdot = mdot 

        self.throat_index = np.argmin(self.geometry.r) # may be redundant
    
    def properties(self, A):
        area_ratio = A/self.At

        # Gamma
        mw, gamma = self.cea.get_exit_MolWt_gamma(
            Pc=self.Pc_bar,
            MR=self.MR,
            eps=area_ratio,
            frozen=0
        )

        # Static temperature
        T_static = self.cea.get_Temperatures(
            Pc=self.Pc_bar,
            MR=self.MR,
            eps=area_ratio,
            frozen=0
        )[2]

        # Heat capacity
        cp = self.cea.get_HeatCapacities(
            Pc=self.Pc_bar,
            MR=self.MR,
            eps=area_ratio,
            frozen=0
        )[2] # Index 2 for EXIT (but essentially where area_ratio is)

        # Transport properties
        cp_tr, mu_millipoise, k_cm, Pr = self.cea.get_Exit_Transport(
            Pc=self.Pc_bar,
            MR=self.MR,
            eps=area_ratio,
            frozen=1
        )

        mu = mu_millipoise*1e-4 # Millipoise -> Pa-s
        k = k_cm*100 # W/cm-K -> W/m-K
        
        return float(gamma), float(cp), float(mu), float(Pr), float(T_static)

    def mach_from_area(self, A, gamma, branch): 
        area_ratio = A/self.At

        # Solve area-Mach relation numerically
        def area_mach(M):
            return (1/M)*\
                   ((2/(gamma+1)) *
                    (1 + (gamma-1)/2*M**2))**((gamma+1)/(2*(gamma-1))) \
                   - area_ratio

        if abs(area_ratio-1.0)<1e-6:
            return 1.0

        if branch == "subsonic":
            return brentq(area_mach, 1e-6, 0.9999)
        elif branch == "supersonic":
            return brentq(area_mach, 1.00001, 10.0)
        else:
            raise ValueError("Branch must be 'subsonic' or 'supersonic'")
    
    def viscosity_from_T(self, T, T_ref, mu_ref, exponent=0.7):
        return mu_ref*(T/T_ref)**exponent
    
    def bartz_base(self, A, mu_g, cp_g, Pr_g):
        
        # Full Bartz Equation
        # Pc in Pa
        # c* in m/s
        # mu in Pa-s
        # cp in J/kg-K
        # A in m^2
        

        C = 0.026 
        Pc = self.Pc_bar*1e5 # Convert bar to Pa 

        c_star = self.c_star
        At = self.At
        Dt = self.Dt
        Rt = self.Rt

        hg = (
            C
            *(mu_g**0.2)
            *cp_g
            *(Pr_g**(-0.6))
            *(Pc/c_star)**0.8
            *(Dt/Rt)**0.1
            *(self.At/A)**0.9
            *(Dt**(-0.2))
        )
        return hg
    

    # Modern Bartz - trying something newwww
    def bartz_base_modern(self, A, mu_g, cp_g, Pr_g):

        """
        Modern SI Bartz using local mass flux instead of Pc/c*
        """

        C = 0.026

        At = self.At
        Dt = self.Dt
        Rt = self.Rt

        # local mass flux (kg/m^2/s)
        G = self.mdot/A

        hg = (
            C
            *cp_g
            *(mu_g**0.2)
            *(Pr_g**(-0.6))
            *(G**0.8)
            *(Dt/Rt)**0.1
            *(At/A)**0.9
        )

        return hg

    def static_temperature(self, T0, gamma, M):
        return T0/(1+(gamma-1)/2*M**2)

    def static_pressure(self, P0, gamma, M):
        return P0/(1+(gamma-1)/2*M**2)**(gamma/(gamma-1))

    def density(self, P, T, R):
        return P/(R*T)