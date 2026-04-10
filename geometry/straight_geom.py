import numpy as np


def piecewise_channel(x, r, a1, a_min, a2, H1, H_min, H2):
    """
    RPA-style piecewise linear channel geometry.
    Linearly interpolates injector (1) -> throat (min) -> nozzle exit (2).
    Returns (a, H) arrays of length len(x).
    """
    throat_idx = np.argmin(r)
    x_throat = x[throat_idx]

    a = np.where(
        x <= x_throat,
        np.interp(x, [x[0], x_throat], [a1, a_min]),
        np.interp(x, [x_throat, x[-1]], [a_min, a2])
    )
    H = np.where(
        x <= x_throat,
        np.interp(x, [x[0], x_throat], [H1, H_min]),
        np.interp(x, [x_throat, x[-1]], [H_min, H2])
    )
    return a, H


class EngineGeometry:
    """
    Combines:
    - Engine contour (x, r)
    - Cooling channel geometry: straight, rectangular channels
    """

    def __init__(self, x, r, a, H, N_channels, t_wall, roughness=1e-5, helix_angle=0.0):
        """
        a, H can be scalars (constant channel) or arrays (varying channel).
        roughness    : absolute wall roughness in metres (default 10 µm).
                       e.g. 5e-6 (5 µm) for machined, 50e-6 (50 µm) for rough/printed.
        helix_angle  : channel helix angle in degrees measured from the axial direction
                       (0 = straight axial channels, typical range 10–30°).
                       Increases effective channel path length and pressure drop by 1/cos(α).
        """
        self.x = x
        self.r = r
        self.n_nodes = len(x)
        self.N = N_channels
        self.t_wall = t_wall
        self.roughness = roughness

        import numpy as np
        self.helix_angle = np.radians(helix_angle)  # store in radians
        self.path_factor = 1.0 / np.cos(self.helix_angle)  # ds/dx — >1 for helical

        N = len(x)
        self.a = np.broadcast_to(np.asarray(a, dtype=float), (N,)).copy()
        self.H = np.broadcast_to(np.asarray(H, dtype=float), (N,)).copy()

        # Non-uniform spacing supported
        self.dx = np.gradient(self.x)

    def web_width(self):
        """Web (rib) width b(x) = circumferential pitch minus channel width. Varies with radius."""
        return 2*np.pi*self.r / self.N - self.a

    def gas_side_area_per_dx(self):
        return self.N*self.a # m2/m - multiply by dx in solver

    def hydraulic_diameter(self):
        return 2*self.a*self.H/(self.a+self.H)

    def total_flow_area(self):
        return self.N*self.a*self.H

    def channel_length(self):
        """Total helical channel length [m] = axial engine length × path_factor."""
        axial_length = self.x[-1] - self.x[0]
        return axial_length * self.path_factor

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