import numpy as np
from rocketcea.cea_obj_w_units import CEA_Obj

# Bartz 

class GasModel:
    
    def __init__(self, Pc_bar, MR, geometry, ox_name, fuel_name):
        
        self.Pc_bar = Pc_bar
        self.MR = MR

        self.geometry = geometry
        self.At = geometry.throat_area()

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

        # FOR NOW ASSUMING 3 REGIONS. Use NASA CEA properly in future to get more accurate contour
        # RocketCEA does NOT give continuous gas properties along contour
        # 1. Chamer section, A > At upstream. Assume constant properties throughout.
        # 2. Throat region, A = At
        # 3. Diverging/nozzle region, A > At downstream. Use exit properties with equilibium modelling.
        
        # Note RocketCEA returns [chamber, throat, exit] with index 0 = chamber, index 1 = throat, index 2 = exit
        
        # Full interpolation would require parsing get_full_cea_output()
        
        # get_X_transport() returns list of heat capacity, viscosity, thermal cond, Pr


        # Chamber constants
        self.Tc = self.cea.get_Tcomb(Pc=self.Pc_bar, MR=self.MR)
        self.c_star = self.cea.get_Cstar(Pc=self.Pc_bar, MR=self.MR)

        # Pre-store chamber values
        self.gamma_ch = self.cea.get_Chamber_MolWt_gamma(self.Pc_bar, self.MR)[1]
        self.cp_ch = self.cea.get_Chamber_Cp(self.Pc_bar, self.MR)
        self.transport_ch = self.cea.get_Chamber_Transport(self.Pc_bar, self.MR)

        # Pre-store throat values
        self.gamma_th = self.cea.get_Throat_MolWt_gamma(self.Pc_bar, self.MR)[1]
        self.transport_th = self.cea.get_Throat_Transport(self.Pc_bar, self.MR)

    
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
            frozen=0
        )

        mu = mu_millipoise*1e-4 # Millipoise -> Pa-s
        k = k_cm*100 # W/cm-K -> W/m-K
        
        return float(gamma), float(cp), float(mu), float(Pr), float(T_static)

    def mach_from_area(self, A, gamma, M_prev):
        
        At = self.At
        area_ratio = A/At

        # Solve area-Mach relation numerically
        from scipy.optimize import fsolve

        def area_mach(M):
            return (1/M)*\
                   ((2/(gamma+1)) *
                    (1 + (gamma-1)/2*M**2))**((gamma+1)/(2*(gamma-1))) \
                   - area_ratio

        # Use previous Mach as initial guess
        M = fsolve(area_mach, M_prev)[0]
        return M
    
    def bartz_base(self, A, mu_g, cp_g, Pr_g):
        """
        Bartz Equation
        Pc in Pa
        c* in m/s
        mu in Pa-s
        cp in J/kg-K
        A in m^2
        """
        C = 0.026
        Pc = self.Pc_bar*1e5 # Convert bar to Pa 
        c_star = self.c_star
        At = self.At

        hg = C*mu_g**0.2*cp_g*Pr_g**(-0.6)*(Pc/c_star)**0.8*(At/A)**0.9

        return hg
