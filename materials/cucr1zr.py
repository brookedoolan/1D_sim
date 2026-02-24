import numpy as np

# MATERIAL PROPERTIES FOR CUCR1ZR
# Temperature dependent 
# Thermal conductivity k, Youngs modulus, Yield strength

class CuCr1Zr:
    
    def __init__(self):
        pass

    def thermal_conductivity(self, T):
        # Replace with actual data
        # Typical CuCr1Zr
        # ~330 W/mk at 300 K
        # ~280 W/mK at 700 K
        return 320