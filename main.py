import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from geometry.straight_geom import EngineGeometry
from fluid.coolant_model import CoolantModel
from fluid.gas_model import GasModel
from fluid.film_cooling import FilmCooling
from solvers.regen_solver import RegenSolver
from materials.cucr1zr import CuCr1Zr
from geometry.rpa_loader import load_rpa_contour
from solvers.chamber_stress import ChamberStress

BASE_DIR = Path(__file__).resolve().parent # Project root (folder containing main.py)
contour_path = BASE_DIR / "geometry" / "rpa_contours" / "contour2.txt"
x, r = load_rpa_contour(contour_path, n_points=300) # Import RPA contour (n_points controls resolution)

CHANNEL_MODE = "rpa"  # "bruv" or "rpa"

if CHANNEL_MODE == "bruv":
    # Channel width follows contour radius: a(x) = r(x) * Ltheta_chnl
    no_web = 40
    Ltheta_web = 0.03       # rad, angular web width
    th_web = 2.0e-3     # m, channel height (radial depth)
    th_iw = 1.5e-3     # m, inner wall thickness
    Ltheta_chnl = 2*np.pi/no_web - Ltheta_web
    a_channel = r * Ltheta_chnl
    H_channel = th_web

elif CHANNEL_MODE == "rpa":
    # Fixed channel width and height — web width b(x) varies naturally with contour radius
    a_channel = 1.5e-3   # m, channel width (fixed)
    H_channel = 1.5e-3   # m, channel height (fixed)
    th_iw  = 1.5e-3
    no_web = 40

geom = EngineGeometry(
    x=x,
    r=r,
    a=a_channel,
    H=H_channel,
    N_channels=no_web,
    t_wall=th_iw,
    roughness=0  # m, absolute wall roughness (30 µm default)
)

coolant = CoolantModel(
    mdot=0.8182, # computing mdot = N*rho*V*A_channel from RPA
    fluid_name="Ethanol"  # CoolProp fluid string
)

gas = GasModel(
    Pc_bar = 35, # chamber pressure in bar
    MR = 1.5, # O/F mass ratio
    geometry = geom,
    ox_name = 'LOX',
    fuel_name = 'Ethanol', # RocketCEA does NOT have IPA
    mdot = 2.00660, # total MFR kg/s
    cstar = 1688.88, # estimated delivered performance, reduced efficiency, ideal c* = 1727.32 m/s
    emissivity = 0.14  # effective grey-gas emissivity for CO2/H2O combustion products (~0.13-0.15)
)

material = CuCr1Zr()

# Film cooling (set FILM_COOLING = True to enable)
FILM_COOLING = True

# Pass 1: run without film to get actual coolant outlet conditions at injector face (x[0])
_solver_p1 = RegenSolver(geom, coolant, gas, material, "nozzle_to_injector", film_cooling=None)
_solver_p1.solve(T_in=298, P_in=4.5e6)

film = None
if FILM_COOLING:
    # Coolant exits regen channels at x[0] (injector face) — use those conditions for film
    T_film = _solver_p1.T_c[0]           # K, actual coolant outlet temp from regen channels
    P_film = _solver_p1.P_c[0]           # Pa, actual coolant outlet pressure
    _, _, _, cp_film = coolant.properties(T_film, P_film)  # CoolProp cp at outlet conditions

    mdot_film_frac = 0.1                 # fraction of total propellant mdot used as film
    mdot_film = gas.mdot * mdot_film_frac / (1 + gas.MR)  # fuel fraction of total mdot

    film = FilmCooling(
        mdot_film=mdot_film,
        T_film=T_film,
        cp_film=cp_film,
        injection_x=geom.x[0],           # injector face
        A_coeff=0.329                     # Gater-L'Ecuyer slot injection constant
    )
    print(f"Film: T_film={T_film:.1f} K, cp_film={cp_film:.0f} J/kg-K, mdot_film={mdot_film:.4f} kg/s")

# Pass 2: full solve with film (or reuse pass-1 if film is off)
if FILM_COOLING:
    regen_solver = RegenSolver(geom, coolant, gas, material, "nozzle_to_injector", film_cooling=film)
    regen_solver.solve(T_in=298, P_in=4.5e6)
else:
    regen_solver = _solver_p1

# Stress analysis
stress_model = ChamberStress(geom, material, Pc_bar=35)
sigma_vm, sigma_t, sigma_t_th, sigma_t_global, sigma_l, safety = stress_model.compute(regen_solver)

x = geom.x
r = geom.r

# ----------- EXPORT DATA -------------
df = pd.DataFrame({
    "x_m": x,
    "heat_flux_W_m2": regen_solver.q,
    "heat_flux_conv_W_m2": regen_solver.q - regen_solver.q_rad,
    "heat_flux_rad_W_m2": regen_solver.q_rad,
    "T_wg_K": regen_solver.T_wg,
    "T_wl_K": regen_solver.T_wl,
    "T_c_K": regen_solver.T_c
})
df.to_csv(BASE_DIR / "results" / "solver_outputs.csv", index=False)

# ============================================================
# FIGURE 1 — THERMAL / FLOW
# ============================================================
fig1, axs1 = plt.subplots(2, 3, figsize=(20, 11))

# Coolant Temp
axs1[0, 0].plot(x, regen_solver.T_c, color="tab:blue")
axs1[0, 0].set_xlabel("Axial position (m)")
axs1[0, 0].set_ylabel("Temperature (K)")
axs1[0, 0].set_title("Coolant Temperature")
axs1[0, 0].grid(True)

# Coolant Pressure
axs1[0, 1].plot(x, regen_solver.P_c/1e6, color="tab:green")
axs1[0, 1].set_xlabel("Axial position (m)")
axs1[0, 1].set_ylabel("Pressure (MPa)")
axs1[0, 1].set_title("Coolant Pressure")
axs1[0, 1].grid(True)

# Mach & Radius
axM = axs1[0, 2]
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

# Wall temps & radius
axTw = axs1[1, 0]
axTw.plot(x, regen_solver.T_wg, color="tab:orange", label="T_wg")
axTw.plot(x, regen_solver.T_wl, color="tab:blue", label="T_wl")
axTw.set_xlabel("Axial position (m)")
axTw.set_ylabel("Wall Temp (K)")
axTw.set_title("Gas-side Wall Temperature")
axTw.legend(fontsize=8)
axTw.grid(True)
axTw2 = axTw.twinx()
axTw2.plot(x, r, color="tab:red")
axTw2.set_ylabel("Radius (m)", color="tab:red")
axTw2.tick_params(axis='y', labelcolor='tab:red')

# Heat flux & radius
axq = axs1[1, 1]
axq.plot(x, regen_solver.q/1e6, color="tab:purple", label="Total")
axq.plot(x, (regen_solver.q - regen_solver.q_rad)/1e6, color="tab:blue", linestyle="--", label="Convective")
axq.plot(x, regen_solver.q_rad/1e6, color="tab:orange", linestyle="--", label="Radiative")
axq.set_xlabel("Axial position (m)")
axq.set_ylabel("Heat Flux (MW/m²)")
axq.set_title("Heat Flux")
axq.legend(fontsize=8)
axq.grid(True)
axq2 = axq.twinx()
axq2.plot(x, r, color="tab:red")
axq2.set_ylabel("Radius (m)", color="tab:red")
axq2.tick_params(axis='y', labelcolor='tab:red')

# Coolant velocity & density
axV = axs1[1, 2]
axV.plot(x, regen_solver.u_c, color="tab:blue")
axV.set_xlabel("Axial position (m)")
axV.set_ylabel("Velocity (m/s)", color="tab:blue")
axV.tick_params(axis='y', labelcolor='tab:blue')
axV.set_title("Coolant Velocity & Density")
axV.grid(True)
axRho = axV.twinx()
axRho.plot(x, regen_solver.rho_c, color="tab:orange")
axRho.set_ylabel("Density (kg/m³)", color="tab:orange")
axRho.tick_params(axis='y', labelcolor='tab:orange')

fig1.tight_layout(pad=3.0)
plt.show()

# ============================================================
# FIGURE 2 — STRESS
# ============================================================
yield_strength = np.array([material.yield_strength(T)/1e6 for T in regen_solver.T_wg])
uts = np.array([material.ultimate_strength(T)/1e6 for T in regen_solver.T_wg])

fig2, axs2 = plt.subplots(1, 3, figsize=(20, 6))

# Stress components
axs2[0].plot(x, sigma_vm/1e6, label="Von Mises")
axs2[0].plot(x, sigma_l/1e6, label="Longitudinal")
axs2[0].plot(x, sigma_t/1e6, label="Local Pressure Hoop")
axs2[0].plot(x, sigma_t_th/1e6, label="Thermal Tangential")
axs2[0].plot(x, sigma_t_global/1e6, label="Global Hoop")
axs2[0].set_xlabel("Axial position (m)")
axs2[0].set_ylabel("MPa")
axs2[0].set_title("Stress Components")
axs2[0].legend()
axs2[0].grid()

# Strength vs stress & safety factor
axs2[1].plot(x, sigma_vm/1e6, label="Von Mises")
axs2[1].plot(x, yield_strength, "--", label="Yield Strength")
axs2[1].plot(x, uts, "--", label="UTS")
axs2[1].set_xlabel("Axial position (m)")
axs2[1].set_ylabel("MPa")
axs2[1].set_title("Strength vs Stress")
axs2[1].legend()
axs2[1].grid()
ax2b = axs2[1].twinx()
ax2b.plot(x, safety, "k-.", label="Safety Factor")
ax2b.set_ylabel("Safety Factor")

# Channel geometry
axs2[2].plot(x, geom.a*1e3, color="tab:blue", label="Width (a)")
axs2[2].plot(x, geom.H*1e3, color="tab:orange", label="Height (H)")
axs2[2].plot(x, geom.web_width()*1e3, color="tab:green", label="Web width (b)")
axs2[2].set_xlabel("Axial position (m)")
axs2[2].set_ylabel("Channel dimension (mm)")
axs2[2].set_title("Cooling Channel Geometry")
axs2[2].legend()
axs2[2].grid(True)

fig2.tight_layout(pad=3.0)
plt.show()

# ============================================================
# FIGURE 3 — FILM COOLING DIAGNOSTICS
# ============================================================
T_aw_base = gas.T0 * (
    (1 + regen_solver.Pr_g**(1/3) * (regen_solver.gamma_g-1)/2 * regen_solver.M**2)
    / (1 + (regen_solver.gamma_g-1)/2 * regen_solver.M**2)
)

fig3, axs3 = plt.subplots(1, 3, figsize=(20, 6))

# Film effectiveness
axs3[0].plot(x, regen_solver.eta_film, color="tab:cyan")
axs3[0].set_xlabel("Axial position (m)")
axs3[0].set_ylabel("Effectiveness η")
axs3[0].set_title("Film Cooling Effectiveness")
axs3[0].set_ylim(0, 1)
axs3[0].grid(True)
if not FILM_COOLING:
    axs3[0].text(0.5, 0.5, "Film cooling OFF", transform=axs3[0].transAxes,
                 ha='center', va='center', fontsize=12, color='gray')

# T_aw comparison
axs3[1].plot(x, T_aw_base, color="tab:red", linestyle="--", label="T_aw (no film)")
axs3[1].plot(x, regen_solver.T_aw_eff, color="tab:blue", label="T_aw_eff (with film)")
axs3[1].plot(x, regen_solver.T_wg, color="tab:orange", label="T_wg")
axs3[1].set_xlabel("Axial position (m)")
axs3[1].set_ylabel("Temperature (K)")
axs3[1].set_title("Adiabatic Wall Temp: Film Effect")
axs3[1].legend(fontsize=8)
axs3[1].grid(True)

# Heat flux with contour
axs3[2].plot(x, regen_solver.q/1e6, color="tab:purple", label="Heat flux")
axs3[2].set_xlabel("Axial position (m)")
axs3[2].set_ylabel("Heat Flux (MW/m²)")
axs3[2].set_title("Heat Flux (Film Cooling)")
axs3[2].legend(fontsize=8)
axs3[2].grid(True)
axq3b = axs3[2].twinx()
axq3b.plot(x, r, color="tab:red", linewidth=0.8)
axq3b.set_ylabel("Radius (m)", color="tab:red")
axq3b.tick_params(axis='y', labelcolor='tab:red')

fig3.tight_layout(pad=3.0)
plt.show()

# ============================================================
# FIGURE 4 — FILM COOLING COMPARISON (no film vs film)
# ============================================================
fig4, (ax_T, ax_q) = plt.subplots(1, 2, figsize=(14, 5))

# --- Wall temperatures ---
ax_T.plot(x, _solver_p1.T_wg, color="tab:orange", label="T_wg  (no film)")
ax_T.plot(x, _solver_p1.T_wl, color="tab:blue",   label="T_wl  (no film)")
if FILM_COOLING:
    ax_T.plot(x, regen_solver.T_wg, color="tab:orange", linestyle="--", label="T_wg  (film)")
    ax_T.plot(x, regen_solver.T_wl, color="tab:blue",   linestyle="--", label="T_wl  (film)")
ax_T.set_xlabel("Axial position (m)")
ax_T.set_ylabel("Temperature (K)")
ax_T.set_title("Wall Temperatures: Film vs No Film")
ax_T.legend()
ax_T.grid(True)

ax_T2 = ax_T.twinx()
ax_T2.plot(x, r, color="tab:red", linewidth=0.8, alpha=0.4)
ax_T2.set_ylabel("Radius (m)", color="tab:red")
ax_T2.tick_params(axis='y', labelcolor='tab:red')

# --- Heat flux ---
ax_q.plot(x, _solver_p1.q/1e6, color="tab:purple", label="Total  (no film)")
ax_q.plot(x, (_solver_p1.q - _solver_p1.q_rad)/1e6, color="tab:blue",
          label="Conv   (no film)")
if FILM_COOLING:
    ax_q.plot(x, regen_solver.q/1e6, color="tab:purple", linestyle="--", label="Total  (film)")
    ax_q.plot(x, (regen_solver.q - regen_solver.q_rad)/1e6, color="tab:blue",
              linestyle="--", label="Conv   (film)")
ax_q.set_xlabel("Axial position (m)")
ax_q.set_ylabel("Heat Flux (MW/m²)")
ax_q.set_title("Heat Flux: Film vs No Film")
ax_q.legend()
ax_q.grid(True)

ax_q2 = ax_q.twinx()
ax_q2.plot(x, r, color="tab:red", linewidth=0.8, alpha=0.4)
ax_q2.set_ylabel("Radius (m)", color="tab:red")
ax_q2.tick_params(axis='y', labelcolor='tab:red')

fig4.tight_layout(pad=2.0)
plt.show()

"""
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
"""