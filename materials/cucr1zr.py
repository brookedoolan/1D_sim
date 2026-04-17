import numpy as np

# MATERIAL PROPERTIES FOR CUCR1ZR

class CuCr1Zr:
    """
    CuCr1Zr matieral properties based on Nikon SLM data sheet
    Takes worst case non-heat-treated material properties, all testing at room temperature.
    Baseline room temp values extrapolated according to UKAEA relations.
    SLM 280 Prime, CW106C
    All temperatures in K
    Young's modulus in GPa
    UTS and Yield strength in MPa
    """
    
    def __init__(self):
        self.alpha = 16.35e-6 # 1/K
        self.nu = 0.38
        

    def thermal_conductivity(self, T):
        # Typical CuCr1Zr: ~320 W/m·K
        return 320
    
    def youngs_modulus(self,T):
        """"
        Based on UKAEA relation but adjusted to fit given data
        """
        E_SAA = -1.9234*1e-4*T**2 - 2.1233*1e-2*T + 124.91
        return E_SAA
    
    def yield_strength(self,T):
        """
        Based on UKAEA relation
        Min Non-heat-treated yield strength:
        - Horizontal: 170 MPa
        - Vertical: 165 MPa
        """

        Y_SAA = -2.2847E-4*T**2 - 0.13931*T + 292.19
        Y_SAA_Nikon = Y_SAA - 65.39
        return Y_SAA_Nikon
    
    def ultimate_strength(self,T):
        """
        Based on UKAEA relation
        Min non-heat-trated UTS:
        - Horizontal: 250 MPa
        - Vertical: 215 MPa
        """

        UTS_SAA = -0.42631*T + 413.45
        UTS_SAA_Nikon = UTS_SAA - 72.41
        return UTS_SAA_Nikon
    
    def poisson_ratio(self, T=None):
        return self.nu
    
    def thermal_expansion(self, T=None):
        return self.alpha

    