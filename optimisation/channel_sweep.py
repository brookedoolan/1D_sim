"""
channel_sweep.py
----------------
Parametric sweep over piecewise cooling channel geometry to minimise peak
gas-side wall temperature (T_wg).

Fixed:  t_wall = 1.5 mm, helix = 30 deg, no film cooling
        nozzle-exit values: a2 = 2.0 mm, H2 = 2.0 mm

Swept:  a1    [mm]  — channel width  at chamber
        a_min [mm]  — channel width  at throat   (tightest constraint)
        H1    [mm]  — channel height at chamber
        H_min [mm]  — channel height at throat
        N           — number of channels

Constraints (print minimum):
  - web width at throat  >= MIN_WEB
  - a_min, H_min        >= 1 mm  (already enforced by sweep ranges)
  - max coolant temp     < MAX_TC

Outputs:
  results/channel_sweep.csv   full results table, sorted by peak T_wg
  results/channel_sweep.png   scatter + top-15 bar chart
"""

import sys
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from geometry.straight_geom import EngineGeometry, piecewise_channel
from geometry.rpa_loader    import load_rpa_contour
from fluid.coolant_model    import CoolantModel
from fluid.gas_model        import GasModel
from solvers.regen_solver   import RegenSolver
from materials.cucr1zr      import CuCr1Zr

# ── Fixed parameters ──────────────────────────────────────────────────────────
CONTOUR   = BASE_DIR / "geometry" / "rpa_contours" / "rpa_35bar_newcontour.txt"
T_WALL    = 1.5e-3   # m, inner wall thickness
HELIX     = 30.0     # degrees
T_IN      = 298.0    # K,  coolant inlet temperature (at nozzle exit)
P_IN      = 4.5e6    # Pa, coolant inlet pressure

A2_FIXED  = 1.5e-3   # m, nozzle-exit channel width  (fixed)
H2_FIXED  = 1.5e-3   # m, nozzle-exit channel height (fixed)

# ── Sweep ranges ──────────────────────────────────────────────────────────────
A1_VALUES    = [1.5, 2.0, 2.5]       # chamber width,  mm
A_MIN_VALUES = [1.0, 1.5]            # throat  width,  mm
H1_VALUES    = [1.5, 2.0, 2.5]       # chamber height, mm
H_MIN_VALUES = [1.0, 1.5]            # throat  height, mm
N_VALUES     = [30, 40, 50, 60]      # number of channels

MIN_WEB   = 0.8e-3   # m — minimum web/land width (print constraint)
MAX_TC    = 550.0    # K — coolant thermal limit

# ── Load contour + shared models (built once) ─────────────────────────────────
print("Loading contour and shared models...")
x, r = load_rpa_contour(CONTOUR, n_points=300)
r_throat = r.min()

coolant  = CoolantModel(mdot=0.8182, fluid_name="Ethanol")
material = CuCr1Zr()

_a_ref, _H_ref = piecewise_channel(x, r, 2.5e-3, 1.0e-3, 2.0e-3, 3.0e-3, 1.0e-3, 2.0e-3)
_geom_ref = EngineGeometry(x=x, r=r, a=_a_ref, H=_H_ref, N_channels=40,
                            t_wall=T_WALL, roughness=0, helix_angle=HELIX)
gas = GasModel(
    Pc_bar=35, MR=1.5, geometry=_geom_ref,
    ox_name='LOX', fuel_name='Ethanol',
    mdot=2.00660, cstar=1688.88, emissivity=0.175,
)

# ── Sweep ─────────────────────────────────────────────────────────────────────
combos  = list(itertools.product(A1_VALUES, A_MIN_VALUES, H1_VALUES, H_MIN_VALUES, N_VALUES))
results = []
skipped_web = 0
skipped_tc  = 0
skipped_err = 0

print(f"Total combinations: {len(combos)}  |  throat radius: {r_throat*1e3:.1f} mm\n")

for idx, (a1_mm, a_min_mm, H1_mm, H_min_mm, N) in enumerate(combos):
    a1    = a1_mm    * 1e-3
    a_min = a_min_mm * 1e-3
    H1    = H1_mm    * 1e-3
    H_min = H_min_mm * 1e-3

    # ── Constraint: web width at throat (tightest point) ─────────────────────
    web_throat = 2 * np.pi * r_throat / N - a_min
    if web_throat < MIN_WEB:
        skipped_web += 1
        continue

    # ── Build piecewise arrays and solve ─────────────────────────────────────
    try:
        a_arr, H_arr = piecewise_channel(
            x, r,
            a1=a1, a_min=a_min, a2=A2_FIXED,
            H1=H1, H_min=H_min, H2=H2_FIXED,
        )
        geom = EngineGeometry(x=x, r=r, a=a_arr, H=H_arr, N_channels=N,
                              t_wall=T_WALL, roughness=0, helix_angle=HELIX)
        solver = RegenSolver(geom, coolant, gas, material,
                             "nozzle_to_injector", film_cooling=None)
        solver.solve(T_in=T_IN, P_in=P_IN)
    except Exception as e:
        skipped_err += 1
        continue

    # ── Extract metrics ───────────────────────────────────────────────────────
    max_Twg  = float(solver.T_wg.max())
    max_Tc   = float(solver.T_c.max())
    delta_P  = float(P_IN - solver.P_c[0]) / 1e6
    Tc_out   = float(solver.T_c[0])

    # ── Constraint: coolant temperature ──────────────────────────────────────
    if max_Tc > MAX_TC:
        skipped_tc += 1
        continue

    results.append({
        "a1_mm":       a1_mm,
        "a_min_mm":    a_min_mm,
        "a2_mm":       A2_FIXED * 1e3,
        "H1_mm":       H1_mm,
        "H_min_mm":    H_min_mm,
        "H2_mm":       H2_FIXED * 1e3,
        "N":           N,
        "web_mm":      web_throat * 1e3,
        "max_Twg_K":   max_Twg,
        "max_Tc_K":    max_Tc,
        "Tc_out_K":    Tc_out,
        "delta_P_MPa": delta_P,
    })

    print(f"[{idx+1:>3}/{len(combos)}] a={a1_mm:.1f}/{a_min_mm:.1f}  H={H1_mm:.1f}/{H_min_mm:.1f}  N={N:2d}"
          f" | T_wg={max_Twg:.0f} K  ΔP={delta_P:.2f} MPa"
          f"  web={web_throat*1e3:.1f} mm  T_c={max_Tc:.0f} K", flush=True)

# ── Results ───────────────────────────────────────────────────────────────────
print(f"\nSkipped — web constraint: {skipped_web}")
print(f"Skipped — coolant temp:   {skipped_tc}")
print(f"Skipped — solver error:   {skipped_err}")
print(f"Valid results:            {len(results)}")

if not results:
    print("\nNo valid configurations found. Try relaxing MIN_WEB or MAX_TC.")
    sys.exit(0)

df = pd.DataFrame(results).sort_values("max_Twg_K").reset_index(drop=True)

out_dir = BASE_DIR / "results"
out_dir.mkdir(exist_ok=True)
df.to_csv(out_dir / "channel_sweep.csv", index=False)

print(f"\n{'='*70}")
print(f"Valid configurations: {len(df)}")
print(f"\nTop 10 by lowest peak T_wg:")
print(df.head(10).to_string(index=False))
print(f"{'='*70}")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Piecewise Channel Sweep — Objective: Minimum Peak T_wg\n"
             f"Fixed: t_wall=1.5 mm, helix=30°, a2=2.0 mm, H2=2.0 mm, no film cooling",
             fontsize=12, fontweight="bold")

# Left: T_wg vs ΔP scatter, coloured by N
sc = axes[0].scatter(
    df["delta_P_MPa"], df["max_Twg_K"],
    c=df["N"], cmap="viridis", s=50, alpha=0.75,
)
plt.colorbar(sc, ax=axes[0], label="N channels")
top5 = df.head(5)
axes[0].scatter(
    top5["delta_P_MPa"], top5["max_Twg_K"],
    edgecolors="red", facecolors="none", s=140, linewidths=2, label="Top 5",
)
for _, row in top5.iterrows():
    axes[0].annotate(
        f"a={row.a1_mm:.1f}/{row.a_min_mm:.1f} H={row.H1_mm:.1f}/{row.H_min_mm:.1f} N={int(row.N)}",
        (row.delta_P_MPa, row.max_Twg_K),
        fontsize=7, xytext=(5, 3), textcoords="offset points",
    )
axes[0].set_xlabel("Coolant ΔP [MPa]")
axes[0].set_ylabel("Peak T_wg [K]")
axes[0].set_title("T_wg vs Pressure Drop")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right: horizontal bar chart of top 15
top15 = df.head(15).copy()
top15["label"] = top15.apply(
    lambda row: f"a={row.a1_mm:.1f}/{row.a_min_mm:.1f}  H={row.H1_mm:.1f}/{row.H_min_mm:.1f}  N={int(row.N):2d}",
    axis=1,
)
bars = axes[1].barh(
    top15["label"][::-1], top15["max_Twg_K"][::-1],
    color="steelblue", edgecolor="white",
)
for bar, (_, row) in zip(bars, top15[::-1].iterrows()):
    axes[1].text(
        bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
        f"ΔP={row.delta_P_MPa:.2f} MPa",
        va="center", fontsize=7,
    )
axes[1].set_xlabel("Peak T_wg [K]")
axes[1].set_title("Top 15 Configurations")
axes[1].grid(True, alpha=0.3, axis="x")

plt.tight_layout()
fig.savefig(out_dir / "channel_sweep.png", dpi=150, bbox_inches="tight")
print(f"\nSaved → results/channel_sweep.csv")
print(f"Saved → results/channel_sweep.png")
plt.show()