import numpy as np


def haaland(Re, dh, eps):
    """Haaland explicit friction factor. Falls back to Filonenko for smooth wall."""
    if eps == 0:
        f = (1.82*np.log10(Re) - 1.64)**(-2)  # Filonenko smooth pipe
    else:
        term = (eps/dh/3.7)**1.11 + 6.9/Re
        f = (-1.8*np.log10(term))**(-2)
    return f


def colebrook(Re, dh, eps):
    """
    Colebrook-White friction factor (Darcy) solved by fixed-point iteration.

    1/sqrt(f) = -2*log10( eps/(3.71*dh) + 2.52/(Re*sqrt(f)) )

    For eps=0 uses Filonenko smooth-pipe formula directly (no iteration needed).
    Initial guess from Haaland; converges in <20 iterations to 1e-10.
    """
    if eps == 0:
        return (1.82*np.log10(Re) - 1.64)**(-2)  # Filonenko

    f = haaland(Re, dh, eps)  # initial guess
    for _ in range(50):
        x = -2.0 * np.log10(eps / (3.71*dh) + 2.52 / (Re * np.sqrt(f)))
        f_new = 1.0 / x**2
        if abs(f_new - f) < 1e-10:
            break
        f = f_new
    return f


def gnielinski(Re, Pr, f):
    """Gnielinski Nusselt number correlation."""
    Nu = ((f/8)*(Re - 1000)*Pr) / (1 + 12.7*np.sqrt(f/8)*(Pr**(2/3) - 1))
    return Nu


def dittus_boelter(Re, Pr, heating=True):
    """
    Dittus-Boelter correlation.  n=0.4 for heating (fluid is being heated), 0.3 for cooling.
    No wall-viscosity correction — appropriate when the fluid may be near-critical at the wall.
    """
    n = 0.4 if heating else 0.3
    return 0.023 * Re**0.8 * Pr**n


def sieder_tate(Re, Pr, mu_bulk, mu_wall):
    """
    Full Sieder-Tate Nusselt number

    Valid for Re > 10 000, 0.7 < Pr < 16 700, L/D > 10.
    """
    return 0.027 * Re**0.8 * Pr**(1/3) * (mu_bulk / mu_wall)**0.14


def fin_efficiency(h_c, k_w, H, b):
    """
    Rectangular fin efficiency for cooling channel side walls (webs).

    m   = sqrt(h_c / (k_w * b))
    η_f = tanh(m * H) / (m * H)

    Parameters
    ----------
    h_c : coolant-side HTC [W/m²/K]
    k_w : wall thermal conductivity [W/m/K]
    H   : channel height (fin height) [m]
    b   : web width (fin thickness) [m]
    """
    if b <= 0:
        return 1.0  # no fin, full efficiency
    m  = np.sqrt(h_c / (k_w * b))
    mH = m * H
    if mH < 1e-6:
        return 1.0  # limit: tanh(x)/x → 1 as x → 0
    return float(np.tanh(mH) / mH)


def regen_thermal_resistance(h_c, k_w, r, t_wall, N, a, H, eta_f, path_factor=1.0):
    """
    Effective thermal resistance per unit gas-side area [m²·K/W]. Uses cylindrical wall conduction.

        T_wg = T_c + q * R_total
        T_wl = T_c + q * R_cool

    Components
    ----------
    R_cyl  : cylindrical wall conduction
             r * ln(1 + t_wall/r) / k_w

    R_cool : fin-corrected coolant-side convection (all N channels in parallel)
             2*π*r / (h_c * N * (2*η_f*H + a))

    Parameters
    ----------
    h_c    : coolant HTC [W/m²/K]
    k_w    : wall thermal conductivity [W/m/K]
    r      : inner chamber radius at station [m]
    t_wall : inner wall thickness [m]
    N      : number of channels
    a      : channel width [m]
    H      : channel height [m]
    eta_f  : fin efficiency [-]

    Returns
    -------
    R_total : R_cyl + R_cool [m²·K/W]
    R_cool  : coolant-side resistance only [m²·K/W]  (used for T_wl)
    """
    R_cyl  = r * np.log(1.0 + t_wall / r) / k_w
    #R_cyl = t_wall / k_w # flat wall approx instead
    R_cool = 2 * np.pi * r / (h_c * N * (2 * eta_f * H + a) * path_factor)
    return R_cyl + R_cool, R_cool