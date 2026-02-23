import numpy as np
import matplotlib.pyplot as plt

from geometry.straight_geom import EngineGeometry
from fluid.coolant_model import CoolantModel
from fluid.gas_model import GasModel
from solvers.regen_solver import RegenSolver


def radius_function(x):
    # simple cylindrical chamber for now
    return 0.05 * np.ones_like(x)

length = 0.5
n_nodes = 300

geom = EngineGeometry(
    length=length,
    n_nodes=n_nodes,
    r_func=radius_function,
    a=2e-3,
    H=3e-3,
    N_channels=40,
    t_wall=2e-3
)

coolant = CoolantModel(
    mdot=0.5,
    fluid_name="Isopropanol"  # CoolProp fluid string
)

gas = GasModel(
    Pc=3e6,
    Tc=3300,
    gamma=1.2,
    c_star=1500,
    At=np.pi*(0.02)**2
)

solver = RegenSolver(geom, coolant, gas)
solver.solve(T_in=300, P_in=4e6)

x = geom.x

plt.figure()
plt.plot(x, solver.T_c)
plt.xlabel("Axial position (m)")
plt.ylabel("Coolant Temperature (K)")
plt.title("Coolant Temperature")
plt.show()

plt.figure()
plt.plot(x, solver.P_c/1e6)
plt.xlabel("Axial position (m)")
plt.ylabel("Coolant Pressure (MPa)")
plt.title("Coolant Pressure")
plt.show()

plt.figure()
plt.plot(x, solver.T_wg)
plt.xlabel("Axial position (m)")
plt.ylabel("Gas-side Wall Temperature (K)")
plt.title("Gas Side Wall Temperature")
plt.show()

plt.figure()
plt.plot(x, solver.q/1e6)
plt.xlabel("Axial position (m)")
plt.ylabel("Heat Flux (MW/m^2)")
plt.title("Heat Flux")
plt.show()