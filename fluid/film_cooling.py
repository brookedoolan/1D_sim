import numpy as np


class FilmCooling:
    """
    Film cooling effectiveness model using the Gater-L'Ecuyer slot injection correlation.

    η(s) = 1 / (1 + A * (cp_g/cp_f) * (mdot_g/mdot_f) * s/D_ref)

    where s is the distance along the wall from the injection point.

    Reference: Gater & L'Ecuyer (1970), also used in Huzel & Huang Ch. 4.

    Parameters
    ----------
    mdot_film : float
        Total film coolant mass flow rate [kg/s].
    T_film : float
        Temperature of film coolant at injection point [K].
    cp_film : float
        Specific heat of film coolant [J/kg-K].
    injection_x : float
        Axial position of film injection [m]. Typically x[0] (injector face).
    A_coeff : float
        Gater-L'Ecuyer correlation constant. Default 0.329 for slot injection.
        Use ~0.5 for angled-hole injection.
    """

    def __init__(self, mdot_film, T_film, cp_film, injection_x, A_coeff=0.329):
        self.mdot_film = mdot_film
        self.T_film = T_film
        self.cp_film = cp_film
        self.injection_x = injection_x
        self.A_coeff = A_coeff

    def effectiveness(self, x, cp_g_arr, mdot_gas, D_ref):
        """
        Compute film cooling effectiveness η at each axial station.

        Parameters
        ----------
        x : array [m]
        cp_g_arr : array [J/kg-K]  frozen gas cp at each node
        mdot_gas : float [kg/s]
        D_ref : float [m]  reference diameter (chamber diameter)

        Returns
        -------
        eta : array, values in [0, 1], zero upstream of injection point.
        """
        eta = np.zeros(len(x))

        for i, xi in enumerate(x):
            s = xi - self.injection_x
            if s < 0:
                continue

            cp_g = cp_g_arr[i]
            heat_cap_ratio = cp_g / self.cp_film
            mass_flow_ratio = mdot_gas / self.mdot_film

            eta[i] = 1.0 / (1.0 + self.A_coeff * heat_cap_ratio * mass_flow_ratio * s / D_ref)

        return eta

    def effective_T_aw(self, T_aw, eta):
        """
        Reduce adiabatic wall temperature by film cooling effectiveness.

        T_aw_eff = T_aw - η * (T_aw - T_film)
        """
        return T_aw - eta * (T_aw - self.T_film)
