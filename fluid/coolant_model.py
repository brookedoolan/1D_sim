from CoolProp.CoolProp import PropsSI

def _safe_PropsSI(output, in1, val1, in2, val2, fluid, nudge=0.05):
    """
    Call PropsSI, retrying with a small temperature nudge if CoolProp raises
    a ValueError about being on the saturation curve.

    nudge : K to add to T on each retry (tries +nudge then -nudge then +2*nudge)
    """
    # Identify which input is temperature so we can perturb it
    try:
        return PropsSI(output, in1, val1, in2, val2, fluid)
    except ValueError as e:
        if "Saturation" not in str(e) and "saturation" not in str(e):
            raise

    # Try a small positive nudge on whatever the temperature input is
    for sign in (+1, -1, +2, -2):
        try:
            if in1 == "T":
                return PropsSI(output, in1, val1 + sign * nudge, in2, val2, fluid)
            elif in2 == "T":
                return PropsSI(output, in1, val1, in2, val2 + sign * nudge, fluid)
            else:
                raise  # can't nudge — re-raise original
        except ValueError:
            continue

    # If all retries fail, raise with a clear message
    raise RuntimeError(
        f"CoolProp saturation error for {fluid}: {output}({in1}={val1}, {in2}={val2}). "
        "All temperature nudges failed."
    )


class CoolantModel:
    def __init__(self, mdot, fluid_name):
        self.mdot = mdot
        self.fluid = fluid_name

    def properties(self, T, P):
        rho = _safe_PropsSI("D", "T", T, "P", P, self.fluid)
        mu  = _safe_PropsSI("V", "T", T, "P", P, self.fluid)
        k   = _safe_PropsSI("L", "T", T, "P", P, self.fluid)
        cp  = _safe_PropsSI("C", "T", T, "P", P, self.fluid)
        return rho, mu, k, cp

    def enthalpy(self, T, P):
        """Specific enthalpy [J/kg] at given T [K] and P [Pa]."""
        return _safe_PropsSI("H", "T", T, "P", P, self.fluid)

    def T_from_enthalpy(self, h, P):
        """Invert enthalpy → temperature [K] at given P [Pa] using CoolProp."""
        try:
            return PropsSI("T", "H", h, "P", P, self.fluid)
        except ValueError as e:
            if "Saturation" not in str(e) and "saturation" not in str(e):
                raise
            # Nudge enthalpy slightly above the saturation boundary
            for sign in (+1, -1, +2, -2):
                try:
                    return PropsSI("T", "H", h + sign * 500.0, "P", P, self.fluid)
                except ValueError:
                    continue
            raise RuntimeError(
                f"CoolProp saturation error for {self.fluid}: T(H={h}, P={P}). "
                "All enthalpy nudges failed."
            )
