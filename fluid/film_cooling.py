import numpy as np


class FilmCooling:
    """
    Film cooling effectiveness using the Gater-L'Ecuyer slot injection
    correlation, matching the RPA implementation.

    η(s) = 1 / (1 + A * (ṁ_g * cp_g) / (ṁ_f * cp_f) * s / D_c)

    s   : arc length along the inner wall surface from the injection point
    D_c : chamber inner diameter at injector face

    Reference: Gater & L'Ecuyer (1970); Huzel & Huang Ch. 4.

    Parameters
    ----------
    mdot_film    : float  — film coolant mass flow rate [kg/s]
    T_film       : float  — film temperature at injection [K]
    cp_film      : float  — film specific heat [J/kg·K]
    injection_x  : float  — axial position of injection [m]
    A_coeff      : float  — correlation constant (0.329 slot, ~0.5 drilled holes)
    """

    def __init__(self, mdot_film, T_film, cp_film, injection_x, A_coeff=0.329):
        self.mdot_film   = mdot_film
        self.T_film      = T_film
        self.cp_film     = cp_film
        self.injection_x = injection_x
        self.A_coeff     = A_coeff

    def effectiveness(self, x, r, cp_g_arr, mdot_gas, D_ref):
        N   = len(x)
        eta = np.zeros(N)

        arc = np.zeros(N)
        for i in range(1, N):
            arc[i] = arc[i-1] + np.sqrt((x[i]-x[i-1])**2 + (r[i]-r[i-1])**2)

        inj_arc = np.interp(self.injection_x, x, arc)

        for i in range(N):
            s = arc[i] - inj_arc
            if s <= 0.0:
                continue
            B = (mdot_gas * cp_g_arr[i]) / (self.mdot_film * self.cp_film)
            eta[i] = 1.0 / (1.0 + self.A_coeff * B * s / D_ref)

        return eta

    def effective_T_aw(self, T_aw, eta):
        """T_aw_eff = T_aw - η·(T_aw - T_film)"""
        return T_aw - eta * (T_aw - self.T_film)


class ZucrowSellersFilm:
    """
    Liquid film cooling model — Huzel & Huang Eq. 4-33 (Zucrow & Sellers).

    decay_mode="none"  : pure Z-S (uniform eta, no axial decay)
    decay_mode="gl"    : hybrid — G-L decay from eta=1 at injection,
                         flooring at the Z-S energy-balance value:
                         eta(x) = eta_ZS + (1 - eta_ZS) / (1 + A*B*s/D_ref)

    Parameters
    ----------
    mdot_film   : float  — film mass flow rate [kg/s]
    T_film      : float  — film injection temperature [K]
    C_plc       : float  — liquid-phase cp [J/kg·K]
    C_pvc       : float  — vapour-phase cp [J/kg·K]
    dH_vc       : float  — latent heat [J/kg]
    mdot_gas    : float  — total gas mass flow rate [kg/s]
    injection_x : float  — axial injection position [m]
    eta_c       : float  — film efficiency (0.3–0.7)
    f_friction  : float  — two-phase friction factor
    Vg_Vd       : float  — centreline-to-BL velocity ratio
    decay_mode  : str    — "none" or "gl"
    A_coeff     : float  — G-L constant (only used when decay_mode="gl")
    x_geom      : array  — axial positions [m] (required for decay_mode="gl")
    r_geom      : array  — radii [m]           (required for decay_mode="gl")
    """

    def __init__(self, mdot_film, T_film, C_plc, C_pvc, dH_vc, mdot_gas,
                 injection_x, eta_c=0.5, f_friction=0.035, Vg_Vd=1.2,
                 decay_mode="none", A_coeff=0.06, x_geom=None, r_geom=None):
        self.mdot_film   = mdot_film
        self.T_film      = T_film
        self.C_plc       = C_plc
        self.C_pvc       = C_pvc
        self.dH_vc       = dH_vc
        self.injection_x = injection_x
        self.eta_c       = eta_c
        self.decay_mode  = decay_mode
        self.A_coeff     = A_coeff
        self.mdot_gas    = mdot_gas

        self.a    = 2.0 * Vg_Vd / f_friction
        self.b    = Vg_Vd - 1.0
        self.GcGg = mdot_film / mdot_gas

        # Precompute arc lengths and reference diameter for G-L decay
        if decay_mode == "gl":
            if x_geom is None or r_geom is None:
                raise ValueError("decay_mode='gl' requires x_geom and r_geom")
            arc = np.zeros(len(x_geom))
            for i in range(1, len(x_geom)):
                arc[i] = arc[i-1] + np.sqrt((x_geom[i]-x_geom[i-1])**2 + (r_geom[i]-r_geom[i-1])**2)
            self._arc     = arc
            self._x_geom  = x_geom
            self._inj_arc = float(np.interp(injection_x, x_geom, arc))
            self._D_ref   = 2.0 * r_geom[0]

    def _arc_from_x(self, x):
        return float(np.interp(x, self._x_geom, self._arc))

    def T_wg_protected(self, T_aw, T_co, cp_g):
        """Wall temperature the Z-S energy balance can maintain."""
        b_term  = (self.b ** (self.C_pvc / max(cp_g, 1.0))) if self.b > 0 else 0.0
        H_avail = self.GcGg * self.eta_c * self.a * (1.0 + b_term)
        num = self.C_pvc * T_aw + H_avail * (self.C_plc * T_co - self.dH_vc)
        den = self.C_pvc + H_avail * self.C_plc
        return num / den

    def local_eta(self, T_aw, T_co, cp_g, x=None):
        """
        Film effectiveness at this station.
        decay_mode="none" : pure Z-S (uniform)
        decay_mode="gl"   : hybrid, decays from 1 at injection down to eta_ZS
        """
        denom = T_aw - self.T_film
        if denom < 1.0:
            return 0.0

        T_wg_zs = self.T_wg_protected(T_aw, T_co, cp_g)
        eta_zs  = float(np.clip((T_aw - T_wg_zs) / denom, 0.0, 1.0))

        if self.decay_mode == "none" or x is None:
            return eta_zs

        # Hybrid: eta = eta_ZS + (1 - eta_ZS) / (1 + A*B*s/D_ref)
        s       = max(self._arc_from_x(x) - self._inj_arc, 0.0)
        B       = (self.mdot_gas * cp_g) / (self.mdot_film * self.C_plc)
        gl_part = 1.0 / (1.0 + self.A_coeff * B * s / self._D_ref)
        return float(np.clip(eta_zs + (1.0 - eta_zs) * gl_part, 0.0, 1.0))

    def effective_T_aw(self, T_aw, T_co, cp_g, x=None):
        """T_aw reduced by film effectiveness."""
        eta = self.local_eta(T_aw, T_co, cp_g, x=x)
        return T_aw - eta * (T_aw - self.T_film)
