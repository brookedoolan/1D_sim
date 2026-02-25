import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from geometry.straight_geom import EngineGeometry
from fluid.coolant_model import CoolantModel
from fluid.gas_model import GasModel
from solvers.regen_solver import RegenSolver
from materials.cucr1zr import CuCr1Zr
from geometry.rpa_loader import load_rpa_contour
from geometry.straight_geom import EngineGeometry

BASE_DIR = Path(__file__).resolve().parent # Project root (folder containing main.py)
contour_path = BASE_DIR / "geometry" / "rpa_contours" / "contour.txt"
x, r = load_rpa_contour(contour_path) # Import RPA contour

geom = EngineGeometry(
    x=x,
    r=r,
    a=1.5e-3,
    H=1.5e-3,
    N_channels=40,
    t_wall=1.5e-3
)

coolant = CoolantModel(
    mdot=1.0,
    fluid_name="Ethanol"  # CoolProp fluid string
)

gas = GasModel(
    Pc_bar = 35,
    MR = 1.5,
    geometry = geom,
    ox_name = 'LOX',
    fuel_name = 'Ethanol', # RocketCEA does NOT have IPA
    mdot = 2.04543,
    cstar = 1684.44
)

material = CuCr1Zr()

solver = RegenSolver(geom, coolant, gas, material)
solver.solve(T_in=298, P_in=4.5e6)

x = geom.x
r = geom.r

fig, axs = plt.subplots(2, 3, figsize=(16, 9))

# Coolant Temp
axs[0, 0].plot(x, solver.T_c, label="Coolant T", color="tab:blue")
axs[0, 0].set_xlabel("Axial position (m)")
axs[0, 0].set_ylabel("Temperature (K)")
axs[0, 0].set_title("Coolant Temperature")
axs[0, 0].grid(True)

# Coolant Pressure
axs[0, 1].plot(x, solver.P_c/1e6, label="Coolant P", color="tab:green")
axs[0, 1].set_xlabel("Axial position (m)")
axs[0, 1].set_ylabel("Pressure (MPa)")
axs[0, 1].set_title("Coolant Pressure")
axs[0, 1].grid(True)

# Mach & Radius
axM = axs[0, 2]
axM.plot(x, solver.M, color="tab:blue")
axM.set_xlabel("Axial position (m)")
axM.set_ylabel("Mach", color="tab:blue")
axM.tick_params(axis='y', labelcolor='tab:blue')
axM.set_title("Mach + Radius")
axM.grid(True)

axM2 = axM.twinx()
axM2.plot(x, r, color="tab:red")
axM2.set_ylabel("Radius (m)", color="tab:red")
axM2.tick_params(axis='y', labelcolor='tab:red')

# Gas wall temp & radius
axTw = axs[1, 0]
axTw.plot(x, solver.T_wg, color="tab:orange")
axTw.set_xlabel("Axial position (m)")
axTw.set_ylabel("Wall Temp (K)", color="tab:orange")
axTw.set_title("Gas-side Wall Temperature")
axTw.grid(True)

axTw2 = axTw.twinx()
axTw2.plot(x, r, color="tab:red")
axTw2.set_ylabel("Radius (m)", color="tab:red")
axTw2.tick_params(axis='y', labelcolor='tab:red')

# Heat flux & radius
axq = axs[1, 1]
axq.plot(x, solver.q/1e6, color="tab:purple")
axq.set_xlabel("Axial position (m)")
axq.set_ylabel("Heat Flux (MW/m²)", color="tab:purple")
axq.set_title("Heat Flux")
axq.grid(True)

axq2 = axq.twinx()
axq2.plot(x, r, color="tab:red")
axq2.set_ylabel("Radius (m)", color="tab:red")
axq2.tick_params(axis='y', labelcolor='tab:red')

# EMPTY PANEL for future
axs[1, 2].axis("off")

plt.tight_layout()
plt.show()