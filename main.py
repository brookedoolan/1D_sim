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
from solvers.chamber_stress import ChamberStress

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

# Thermal analysis
regen_solver = RegenSolver(geom, coolant, gas, material, "nozzle_to_injector")
regen_solver.solve(T_in=298, P_in=4.5e6)

# Stress analysis
stress_model = ChamberStress(geom, material, Pc_bar=35)
sigma_vm, sigma_t, sigma_t_th, sigma_l, safety = stress_model.compute(regen_solver)

x = geom.x
r = geom.r

# ------------ THERMAL PLOTS -----------------------
fig, axs = plt.subplots(3, 3, figsize=(18, 13))

# Coolant Temp
axs[0, 0].plot(x, regen_solver.T_c, label="Coolant T", color="tab:blue")
axs[0, 0].set_xlabel("Axial position (m)")
axs[0, 0].set_ylabel("Temperature (K)")
axs[0, 0].set_title("Coolant Temperature")
axs[0, 0].grid(True)

# Coolant Pressure
axs[0, 1].plot(x, regen_solver.P_c/1e6, label="Coolant P", color="tab:green")
axs[0, 1].set_xlabel("Axial position (m)")
axs[0, 1].set_ylabel("Pressure (MPa)")
axs[0, 1].set_title("Coolant Pressure")
axs[0, 1].grid(True)

# Mach & Radius
axM = axs[0, 2]
axM.plot(x, regen_solver.M, color="tab:blue")
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
axTw.plot(x, regen_solver.T_wg, color="tab:orange")
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
axq.plot(x, regen_solver.q/1e6, color="tab:purple")
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

# ----------- STRESS PLOTS ----------------------
# Stress summary: longitudinal, von mises, pressure tangential, thermal tangential 
axs[2,0].plot(x, sigma_vm/1e6, label="Von Mises")
axs[2,0].plot(x, sigma_l/1e6, label="Longitudinal")
axs[2,0].plot(x, sigma_t/1e6, label="Pressure Tangential")
axs[2,0].plot(x, sigma_t_th/1e6, label="Thermal Tangential")
axs[2,0].set_title("Stress Components")
axs[2,0].set_ylabel("MPa")
axs[2,0].legend()
axs[2,0].grid()

# Strength and Safety factors
ax = axs[2,1]

yield_strength = np.array([material.yield_strength(T)/1e6 for T in regen_solver.T_wg])
uts = np.array([material.ultimate_strength(T)/1e6 for T in regen_solver.T_wg])

ax.plot(x, sigma_vm/1e6, label="Von Mises")
ax.plot(x, yield_strength, "--", label="Yield Strength")
ax.plot(x, uts, "--", label="UTS")
ax.set_ylabel("MPa")
ax.set_title("Strength vs Stress")
ax.legend()
ax.grid()

ax2 = ax.twinx()
ax2.plot(x, safety, "k-.", label="Safety Factor")
ax2.set_ylabel("Safety Factor")

axs[2,2].axis("off")  # spare

plt.tight_layout()
plt.show()

# -------- MATERIAL PROPERTY CURVES --------
T = np.linspace(250,800,300)

E = [material.youngs_modulus(t) for t in T]
YS = [material.yield_strength(t) for t in T]
UTS = [material.ultimate_strength(t) for t in T]

fig3, ax1 = plt.subplots(figsize=(8,6))

ax1.plot(T,E,label="Youngs Modulus (GPa)")
ax1.set_xlabel("Temperature (K)")
ax1.set_ylabel("E (GPa)")
ax1.grid()

ax2 = ax1.twinx()
ax2.plot(T,YS,label="Yield Strength",linestyle="--")
ax2.plot(T,UTS,label="UTS",linestyle=":")
ax2.set_ylabel("Strength (MPa)")

lines,labels = ax1.get_legend_handles_labels()
lines2,labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines+lines2,labels+labels2,loc="upper right")

plt.title("CuCr1Zr Material Properties vs Temperature")
plt.tight_layout()
plt.show()