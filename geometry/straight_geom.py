import numpy as np

class EngineGeometry:
    """
    Combines:
    - Engine contour (x, r)
    - Cooling channel geometry: straight, rectangular channels
    """

    def __init__(self, x, r, a1, a_min, a2, H1, H_min, H2, N_channels, t_wall):

        self.x = x
        self.r = r
        self.n_nodes = len(x)
        self.N = N_channels
        self.t_wall = t_wall

        # Piecewise linear channel width and height:
        # injector (1) -> throat (min) -> nozzle exit (2)
        throat_idx = np.argmin(r)
        x_throat = x[throat_idx]

        self.a = np.where(
            x <= x_throat,
            np.interp(x, [x[0], x_throat], [a1, a_min]),
            np.interp(x, [x_throat, x[-1]], [a_min, a2])
        )
        self.H = np.where(
            x <= x_throat,
            np.interp(x, [x[0], x_throat], [H1, H_min]),
            np.interp(x, [x_throat, x[-1]], [H_min, H2])
        )

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