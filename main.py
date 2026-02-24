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
    mdot=0.5,
    fluid_name="Ethanol"  # CoolProp fluid string
)

gas = GasModel(
    Pc_bar = 30,
    MR = 1.5,
    geometry = geom,
    ox_name = 'LOX',
    fuel_name = 'Ethanol' # RocketCEA does NOT have IPA
)

material = CuCr1Zr()

solver = RegenSolver(geom, coolant, gas, material)
solver.solve(T_in=298, P_in=4.5e6)

x = geom.x

import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# --- Coolant Temperature ---
axs[0, 0].plot(x, solver.T_c)
axs[0, 0].set_xlabel("Axial position (m)")
axs[0, 0].set_ylabel("Coolant Temperature (K)")
axs[0, 0].set_title("Coolant Temperature")

# --- Coolant Pressure ---
axs[0, 1].plot(x, solver.P_c / 1e6)
axs[0, 1].set_xlabel("Axial position (m)")
axs[0, 1].set_ylabel("Coolant Pressure (MPa)")
axs[0, 1].set_title("Coolant Pressure")

# --- Gas-side Wall Temperature ---
axs[1, 0].plot(x, solver.T_wg)
axs[1, 0].set_xlabel("Axial position (m)")
axs[1, 0].set_ylabel("Gas-side Wall Temperature (K)")
axs[1, 0].set_title("Gas-side Wall Temperature")

# --- Heat Flux ---
axs[1, 1].plot(x, solver.q / 1e6)
axs[1, 1].set_xlabel("Axial position (m)")
axs[1, 1].set_ylabel("Heat Flux (MW/m²)")
axs[1, 1].set_title("Heat Flux")

plt.tight_layout()
plt.show()

# --- Mach Number --- 
fig1, ax1 = plt.subplots()

# First axis for M
ax1.plot(x, solver.M, label="Mach", color="blue")
ax1.set_xlabel("Axial position (m)")
ax1.set_ylabel("Mach number", color="blue")
ax1.tick_params(axis='y', labelcolor='blue')

# Second axis for radius
ax2 = ax1.twinx()
ax2.plot(x, geom.r, label="Radius", color="red")
ax2.set_ylabel("Radius (m)", color="red")
ax2.tick_params(axis='y', labelcolor='red')

plt.title("Mach and Radius Profile")
plt.grid(True)
plt.show()