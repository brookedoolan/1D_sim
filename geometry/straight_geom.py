import numpy as np

# Defines engine contour and cooling channel geom
# Assumes STRAIGHT cooling channels

class EngineGeometry:
    def __init__(self, length, n_nodes, r_func, a, H, N_channels, t_wall):

        self.length = length
        self.n_nodes = n_nodes
        self.x = np.linspace(0, length, n_nodes)

        # Radius as function of x
        self.r = r_func(self.x)

        # CURRENTLY CONSTANT BUT ADD PIECEWISE
        self.a = a # channel width
        self.H = H # channel height
        self.N = N_channels # number of channels
        self.t_wall = t_wall

        self.dx = self.x[1] - self.x[0]

    def hydraulic_diameter(self):
        return 2*self.a*self.H/(self.a+self.H)
    
    def total_flow_area(self):
        return self.N*self.a*self.H