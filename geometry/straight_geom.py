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
    
    def throat_radius_of_curvature(self, window=10):
        """
        Computes throat radius of curvature Rt from contour.
        Uses circle fit. Window = number of points on each side of throat used in fit.
        Required for Bartz.
        """
        x = self.x
        r = self.r

        i0 = np.argmin(r)

        # ensure stay in bounds
        i1 = max(0, i0-window)
        i2 = min(len(x), i0+window+1)

        x_fit = x[i1:i2]
        y_fit = r[i1:i2]

        # shift origin to improve conditioning
        x_m = np.mean(x_fit)
        y_m = np.mean(y_fit)

        u = x_fit - x_m
        v = y_fit - y_m

        # circle least squares (Kasa method)
        Suu = np.sum(u*u)
        Svv = np.sum(v*v)
        Suv = np.sum(u*v)
        Suuu = np.sum(u*u*u)
        Svvv = np.sum(v*v*v)
        Suvv = np.sum(u*v*v)
        Svuu = np.sum(v*u*u)

        A = np.array([[Suu, Suv],
                    [Suv, Svv]])

        B = np.array([
            0.5*(Suuu + Suvv),
            0.5*(Svvv + Svuu)
        ])

        uc, vc = np.linalg.solve(A, B)

        R = np.sqrt(uc**2 + vc**2 + (Suu+Svv)/len(u))

        return R