"""
RPA vs 1D Sim comparison script.

Usage:
    python validation/compare_rpa.py
    python validation/compare_rpa.py --rpa rpa_results/my_file.txt --sim results/solver_outputs.csv
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_rpa(filepath):
    """
    Parse RPA heat transfer text output into a DataFrame.
    Returns columns: x_mm, r_mm, h_g, q_conv, q_rad, q_total, T_wg, T_wi, T_wc, T_c, P_c, w_c, rho_c
    """
    cols = [
        "x_mm", "r_mm", "h_g_kW_m2K", "q_conv_kW_m2", "q_rad_kW_m2",
        "q_total_kW_m2", "T_wg_K", "T_wi_K", "T_wc_K", "T_c_K",
        "P_c_MPa", "w_c_ms", "rho_c_kgm3",
    ]
    rows = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            # Extract only numeric tokens (skip the trailing comment column)
            vals = []
            for p in parts:
                token = p.strip().split()[0] if p.strip() else ""
                try:
                    vals.append(float(token))
                except (ValueError, IndexError):
                    pass  # skip non-numeric (e.g. "regenerative cooling..." comment)
            if len(vals) >= 13:
                rows.append(vals[:13])

    df = pd.DataFrame(rows, columns=cols)
    # Convert to SI
    df["x_m"]       = df["x_mm"] * 1e-3
    df["r_m"]       = df["r_mm"] * 1e-3
    df["q_conv_W"]  = df["q_conv_kW_m2"] * 1e3
    df["q_rad_W"]   = df["q_rad_kW_m2"]  * 1e3
    df["q_total_W"] = df["q_total_kW_m2"] * 1e3
    df["P_c_Pa"]    = df["P_c_MPa"] * 1e6
    return df


def parse_sim(filepath):
    """Load solver_outputs.csv."""
    df = pd.read_csv(filepath)
    df = df.rename(columns={
        "heat_flux_W_m2":      "q_total_W",
        "heat_flux_conv_W_m2": "q_conv_W",
        "heat_flux_rad_W_m2":  "q_rad_W",
    })
    return df


# ── Interpolation ─────────────────────────────────────────────────────────────

def interp_onto(source_x, source_y, target_x):
    """1-D linear interpolation, clipped to source range."""
    return np.interp(target_x, source_x, source_y)


# ── Numeric summary ───────────────────────────────────────────────────────────

def percent_error(sim, rpa):
    return (sim - rpa) / rpa * 100.0


def print_comparison_table(rpa_df, sim_df):
    """
    Print % error at key stations: chamber, throat (max q), and nozzle exit.
    """
    # Common grid = RPA x positions
    x_grid = rpa_df["x_m"].values

    channels = {
        "q_total [kW/m²]": ("q_total_W",  1e3),
        "q_conv  [kW/m²]": ("q_conv_W",   1e3),
        "q_rad   [kW/m²]": ("q_rad_W",    1e3),
        "h_g     [W/m²K]": ("h_g_W_m2K",  1.0),
        "T_aw    [K]"    : ("T_aw_K",      1.0),
        "T_wg    [K]"    : ("T_wg_K",      1.0),
        "T_wl/wc [K]"    : ("T_wl_K",      1.0),
        "T_c     [K]"    : ("T_c_K",       1.0),
    }

    rpa_map = {
        "q_total_W":  "q_total_W",
        "q_conv_W":   "q_conv_W",
        "q_rad_W":    "q_rad_W",
        "h_g_W_m2K":  "h_g_kW_m2K",  # RPA column, will scale by 1e3
        "T_aw_K":     None,            # RPA doesn't output T_aw directly
        "T_wg_K":     "T_wg_K",
        "T_wl_K":     "T_wc_K",
        "T_c_K":      "T_c_K",
    }

    # Station indices on RPA grid
    throat_idx = int(np.argmax(rpa_df["q_total_W"].values))
    stations   = {
        "Chamber (x=0)": 0,
        "Throat (max q)": throat_idx,
        "Nozzle exit": len(x_grid) - 1,
    }

    print("\n" + "="*80)
    print(f"{'Station':<22} {'Variable':<18} {'RPA':>12} {'1D Sim':>12} {'Error %':>10}")
    print("="*80)

    # Extra unit scale for columns whose RPA value needs an additional conversion
    rpa_extra_scale = {"h_g_W_m2K": 1e3}   # RPA h_g is in kW/m²K → multiply by 1e3

    for station_name, idx in stations.items():
        x_q = x_grid[idx]
        for label, (sim_col, scale) in channels.items():
            rpa_col = rpa_map[sim_col]

            if rpa_col is None:
                # No RPA equivalent — just print our value
                if sim_col in sim_df.columns:
                    sim_val = interp_onto(sim_df["x_m"].values, sim_df[sim_col].values, x_q) / scale
                    print(f"  {station_name:<20} {label:<18} {'N/A':>12} {sim_val:>12.2f} {'—':>10}")
                continue

            extra = rpa_extra_scale.get(sim_col, 1.0)
            rpa_val = rpa_df[rpa_col].iloc[idx] * extra / scale

            if sim_col in sim_df.columns:
                sim_val_interp = interp_onto(sim_df["x_m"].values,
                                             sim_df[sim_col].values, x_q) / scale
                err = percent_error(sim_val_interp, rpa_val)
                print(f"  {station_name:<20} {label:<18} {rpa_val:>12.2f} {sim_val_interp:>12.2f} {err:>+10.1f}%")
            else:
                print(f"  {station_name:<20} {label:<18} {rpa_val:>12.2f} {'N/A':>12} {'—':>10}")
        print()

    print("="*80)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_comparison(rpa_df, sim_df, save_dir=None):
    x_rpa = rpa_df["x_m"].values * 1e3   # → mm for readability
    x_sim = sim_df["x_m"].values  * 1e3

    pairs = [
        ("Total heat flux [kW/m²]", rpa_df["q_total_W"]/1e3,    sim_df["q_total_W"]/1e3),
        ("Conv. heat flux [kW/m²]", rpa_df["q_conv_W"]/1e3,     sim_df["q_conv_W"]/1e3),
        ("Rad. heat flux [kW/m²]",  rpa_df["q_rad_W"]/1e3,      sim_df["q_rad_W"]/1e3),
        ("T_wg [K]",                rpa_df["T_wg_K"],            sim_df["T_wg_K"]),
        ("T_wl/wc [K]",             rpa_df["T_wc_K"],            sim_df["T_wl_K"]),
        ("T_coolant [K]",           rpa_df["T_c_K"],             sim_df["T_c_K"]),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("RPA vs 1D Sim Comparison", fontsize=14, fontweight="bold")

    for ax, (title, y_rpa, y_sim) in zip(axes.flat, pairs):
        ax.plot(x_rpa, y_rpa, "k-",  lw=2,   label="RPA")
        ax.plot(x_sim, y_sim, "r--", lw=1.5, label="1D Sim")
        ax.set_title(title)
        ax.set_xlabel("x [mm]")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir:
        out = Path(save_dir) / "rpa_comparison.png"
        plt.savefig(out, dpi=150)
        print(f"\nFigure saved -> {out}")
    plt.show()


def plot_percent_error(rpa_df, sim_df, save_dir=None):
    x_rpa = rpa_df["x_m"].values

    channels = {
        "q_total": ("q_total_W", "q_total_W"),
        "q_conv":  ("q_conv_W",  "q_conv_W"),
        "T_wg":    ("T_wg_K",    "T_wg_K"),
        "T_wl/wc": ("T_wc_K",   "T_wl_K"),
        "T_c":     ("T_c_K",    "T_c_K"),
    }

    fig, ax = plt.subplots(figsize=(12, 5))
    for label, (rpa_col, sim_col) in channels.items():
        sim_interp = interp_onto(sim_df["x_m"].values, sim_df[sim_col].values, x_rpa)
        err = percent_error(sim_interp, rpa_df[rpa_col].values)
        ax.plot(x_rpa * 1e3, err, label=label, lw=1.5)

    ax.axhline(+5, color="gray", ls="--", lw=1, label="±5% target")
    ax.axhline(-5, color="gray", ls="--", lw=1)
    ax.axhline(0,  color="black", lw=0.8)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("% error  (sim − RPA) / RPA × 100")
    ax.set_title("Percent Error vs RPA")
    ax.legend(ncol=3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir:
        out = Path(save_dir) / "rpa_percent_error.png"
        plt.savefig(out, dpi=150)
        print(f"Figure saved -> {out}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpa", default="rpa_results/rpa_35bar_output1.txt")
    parser.add_argument("--sim", default="results/solver_outputs.csv")
    parser.add_argument("--save", default="results", help="Directory to save figures")
    args = parser.parse_args()

    rpa_df = parse_rpa(args.rpa)
    sim_df = parse_sim(args.sim)

    print(f"RPA:    {len(rpa_df)} stations, x = {rpa_df['x_mm'].min():.1f}–{rpa_df['x_mm'].max():.1f} mm")
    print(f"1D Sim: {len(sim_df)} nodes,    x = {sim_df['x_m'].min()*1e3:.1f}–{sim_df['x_m'].max()*1e3:.1f} mm")

    print_comparison_table(rpa_df, sim_df)
    plot_comparison(rpa_df, sim_df, save_dir=args.save)
    plot_percent_error(rpa_df, sim_df, save_dir=args.save)


if __name__ == "__main__":
    main()
