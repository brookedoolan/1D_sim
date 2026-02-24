import numpy as np

class EngineGeometry:
    """
    Combines:
    - Engine contour (x, r)
    - Cooling channel geometry: straight, rectangular channels
    """

    def __init__(self, x, r, a, H, N_channels, t_wall):

        self.x = x
        self.r = r
        self.n_nodes = len(x)

        # CURRENTLY CONSTANT BUT ADD PIECEWISE
        self.a = a # channel width
        self.H = H  # channel height
        self.N = N_channels # number of channels
        self.t_wall = t_wall

        # Non-uniform spacing supported
        self.dx = np.gradient(self.x)

    def hydraulic_diameter(self):
        return 2*self.a*self.H/(self.a+self.H)
    
    def total_flow_area(self):
        return self.N*self.a*self.H
    
    def throat_index(self):
        return np.argmin(self.r)
    
    def area(self):
        return np.pi*self.r**2
    
    def throat_area(self):
        return np.min(self.area())