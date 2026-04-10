"""
1D Regenerative Cooling Simulator — PyQt6 GUI
Run from project root: python app.py
"""
import sys
import traceback
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QHBoxLayout, QVBoxLayout, QFormLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QCheckBox,
    QTabWidget, QFileDialog, QScrollArea,
    QDoubleSpinBox, QSpinBox, QComboBox,
    QMessageBox, QProgressBar, QSizePolicy,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

# Project imports (relative to project root)
sys.path.insert(0, str(Path(__file__).parent))
from geometry.straight_geom import EngineGeometry
from geometry.rpa_loader import load_rpa_contour
from fluid.coolant_model import CoolantModel
from fluid.gas_model import GasModel
from fluid.film_cooling import FilmCooling
from solvers.regen_solver import RegenSolver
from materials.cucr1zr import CuCr1Zr
from solvers.chamber_stress import ChamberStress


# ── Matplotlib canvas ────────────────────────────────────────────────────────

class MplCanvas(FigureCanvas):
    def __init__(self, nrows=2, ncols=3, figsize=(12, 7)):
        self.fig = Figure(figsize=figsize, tight_layout=True)
        self.axes = self.fig.subplots(nrows, ncols)
        if nrows == 1:
            self.axes = self.axes.reshape(1, ncols)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def clear(self):
        self.fig.clear()

    def redraw(self):
        self.fig.tight_layout(pad=2.5)
        self.draw()


# ── Solver worker thread ─────────────────────────────────────────────────────

class SolverWorker(QThread):
    result_ready = pyqtSignal(dict)
    error        = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            p = self.params

            x, r = load_rpa_contour(p['contour_file'], n_points=p['n_points'])

            geom = EngineGeometry(
                x=x, r=r,
                a=p['a'], H=p['H'],
                N_channels=p['N_channels'],
                t_wall=p['t_wall'],
                roughness=p['roughness'],
                helix_angle=p['helix_angle'],
            )

            material = CuCr1Zr()
            coolant  = CoolantModel(mdot=p['mdot_coolant'], fluid_name=p['coolant_fluid'])
            gas      = GasModel(
                Pc_bar    = p['Pc_bar'],
                MR        = p['MR'],
                geometry  = geom,
                ox_name   = p['ox_name'],
                fuel_name = p['fuel_name'],
                mdot      = p['mdot_gas'],
                cstar     = p['cstar'],
                emissivity= p['emissivity'],
            )

            # Pass 1 — no film
            solver_p1 = RegenSolver(geom, coolant, gas, material, "nozzle_to_injector")
            solver_p1.solve(T_in=p['T_in'], P_in=p['P_in'])

            # Pass 2 — with film (optional)
            film = None
            if p['film_cooling']:
                T_film = solver_p1.T_c[0]
                P_film = solver_p1.P_c[0]
                _, _, _, cp_film = coolant.properties(T_film, P_film)
                mdot_film = p['mdot_gas'] * p['film_frac'] / (1 + p['MR'])
                film = FilmCooling(
                    mdot_film  = mdot_film,
                    T_film     = T_film,
                    cp_film    = cp_film,
                    injection_x= geom.x[0],
                    A_coeff    = p['film_A_coeff'],
                )
                solver = RegenSolver(geom, coolant, gas, material,
                                     "nozzle_to_injector", film_cooling=film)
                solver.solve(T_in=p['T_in'], P_in=p['P_in'])
            else:
                solver = solver_p1

            # Stress
            stress = ChamberStress(geom, material, Pc_bar=p['Pc_bar'])
            sigma_vm, sigma_t, sigma_t_th, sigma_t_global, sigma_l, safety = stress.compute(solver)

            self.result_ready.emit({
                'x': x, 'r': r,
                'geom': geom, 'material': material, 'gas': gas,
                'solver_p1': solver_p1, 'solver': solver,
                'film_cooling': p['film_cooling'],
                'sigma_vm': sigma_vm, 'sigma_t': sigma_t,
                'sigma_t_th': sigma_t_th, 'sigma_t_global': sigma_t_global,
                'sigma_l': sigma_l, 'safety': safety,
            })
        except Exception:
            self.error.emit(traceback.format_exc())


# ── Helper: labelled spin box ─────────────────────────────────────────────────

def dbl_spin(val, lo, hi, step, decimals=3, suffix=''):
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setDecimals(decimals)
    w.setValue(val)
    w.setSuffix(suffix)
    return w

def int_spin(val, lo, hi):
    w = QSpinBox()
    w.setRange(lo, hi)
    w.setValue(val)
    return w


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("1D Regen Cooling Simulator")
        self.resize(1400, 820)
        self._worker = None
        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter)

        # Left: scrollable input panel
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(320)
        scroll.setMaximumWidth(400)
        scroll_inner = QWidget()
        scroll.setWidget(scroll_inner)
        left = QVBoxLayout(scroll_inner)
        left.setSpacing(6)
        splitter.addWidget(scroll)

        # Right: plot tabs
        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(1, 3)

        # ── Input groups ──────────────────────────────────────────────────

        # Contour file
        grp_contour = QGroupBox("Contour")
        f_contour   = QFormLayout(grp_contour)
        self.contour_path = QLineEdit()
        self.contour_path.setPlaceholderText("Select .txt contour file…")
        default_contour = Path(__file__).parent / "geometry" / "rpa_contours" / "contour2.txt"
        if default_contour.exists():
            self.contour_path.setText(str(default_contour))
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse_contour)
        self.n_points = int_spin(300, 50, 2000)
        f_contour.addRow(QLabel("File"), self.contour_path)
        f_contour.addRow("", btn_browse)
        f_contour.addRow("Mesh points", self.n_points)
        left.addWidget(grp_contour)

        # Engine / CEA
        grp_eng = QGroupBox("Engine / CEA")
        f_eng   = QFormLayout(grp_eng)
        self.Pc_bar    = dbl_spin(35.0,   1,   500,  1.0,  1, " bar")
        self.MR        = dbl_spin(1.5,    0.5,  10,  0.1,  2)
        self.cstar     = dbl_spin(1688.9, 100, 4000, 10.0,  1, " m/s")
        self.mdot_gas  = dbl_spin(2.007,  0.01, 200, 0.01,  4, " kg/s")
        self.emissivity= dbl_spin(0.14,   0.0,  1.0, 0.01,  2)
        self.fuel_cb   = QComboBox(); self.fuel_cb.addItems(["Ethanol", "RP1", "CH4", "H2"])
        self.ox_cb     = QComboBox(); self.ox_cb.addItems(["LOX", "N2O4", "N2O"])
        f_eng.addRow("Pc",          self.Pc_bar)
        f_eng.addRow("O/F ratio",   self.MR)
        f_eng.addRow("c*",          self.cstar)
        f_eng.addRow("ṁ total",     self.mdot_gas)
        f_eng.addRow("Emissivity",  self.emissivity)
        f_eng.addRow("Fuel",        self.fuel_cb)
        f_eng.addRow("Oxidiser",    self.ox_cb)
        left.addWidget(grp_eng)

        # Coolant
        grp_cool = QGroupBox("Coolant")
        f_cool   = QFormLayout(grp_cool)
        self.mdot_coolant  = dbl_spin(0.8182, 0.001, 100, 0.01, 4, " kg/s")
        self.T_in          = dbl_spin(298.0,  200,   500, 1.0,  1, " K")
        self.P_in          = dbl_spin(4.5,    0.5,   50,  0.1,  2, " MPa")
        self.coolant_cb    = QComboBox(); self.coolant_cb.addItems(["Ethanol", "Water", "Methanol", "IsoButane"])
        f_cool.addRow("ṁ coolant",  self.mdot_coolant)
        f_cool.addRow("T inlet",    self.T_in)
        f_cool.addRow("P inlet",    self.P_in)
        f_cool.addRow("Fluid",      self.coolant_cb)
        left.addWidget(grp_cool)

        # Channels
        grp_ch = QGroupBox("Cooling Channels")
        f_ch   = QFormLayout(grp_ch)
        self.N_ch      = int_spin(40,   4, 500)
        self.a_ch      = dbl_spin(1.5,  0.1, 20, 0.1, 2, " mm")
        self.H_ch      = dbl_spin(1.5,  0.1, 20, 0.1, 2, " mm")
        self.t_wall    = dbl_spin(1.5,  0.1, 10, 0.1, 2, " mm")
        self.roughness = dbl_spin(0.0,  0.0, 500, 1.0, 1, " µm")
        self.helix_ang = dbl_spin(0.0,  0.0, 60, 1.0, 1, " °")
        f_ch.addRow("N channels",   self.N_ch)
        f_ch.addRow("Width  a",     self.a_ch)
        f_ch.addRow("Height H",     self.H_ch)
        f_ch.addRow("Wall thick.",  self.t_wall)
        f_ch.addRow("Roughness",    self.roughness)
        f_ch.addRow("Helix angle",  self.helix_ang)
        left.addWidget(grp_ch)

        # Film cooling
        grp_film = QGroupBox("Film Cooling")
        f_film   = QFormLayout(grp_film)
        self.film_enable  = QCheckBox("Enable film cooling")
        self.film_frac    = dbl_spin(5.0,  0.1, 50,  0.5, 1, " %")
        self.film_A_coeff = dbl_spin(0.329, 0.1, 2.0, 0.01, 3)
        f_film.addRow(self.film_enable)
        f_film.addRow("Film fraction", self.film_frac)
        f_film.addRow("A coeff (G-L)", self.film_A_coeff)
        left.addWidget(grp_film)

        # Run button + status
        self.btn_run = QPushButton("▶  Run Simulation")
        self.btn_run.setFixedHeight(40)
        self.btn_run.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.btn_run.clicked.connect(self._run)
        left.addWidget(self.btn_run)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)   # indeterminate
        self.progress.setVisible(False)
        left.addWidget(self.progress)

        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        left.addWidget(self.status_lbl)
        left.addStretch()

        # ── Plot tabs (empty until first run) ────────────────────────────

        self.canvas_thermal = MplCanvas(nrows=2, ncols=3, figsize=(13, 7))
        self.canvas_stress  = MplCanvas(nrows=1, ncols=3, figsize=(13, 4))
        self.canvas_film    = MplCanvas(nrows=1, ncols=3, figsize=(13, 4))
        self.canvas_compare = MplCanvas(nrows=1, ncols=2, figsize=(12, 4))

        self.tabs.addTab(self.canvas_thermal, "Thermal")
        self.tabs.addTab(self.canvas_stress,  "Stress")
        self.tabs.addTab(self.canvas_film,    "Film Cooling")
        self.tabs.addTab(self.canvas_compare, "Film Comparison")

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _browse_contour(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select contour file", "", "Text files (*.txt);;All files (*)")
        if path:
            self.contour_path.setText(path)

    def _run(self):
        contour = self.contour_path.text().strip()
        if not contour or not Path(contour).exists():
            QMessageBox.warning(self, "Missing contour", "Please select a valid contour file.")
            return

        params = {
            'contour_file': contour,
            'n_points':     self.n_points.value(),
            'Pc_bar':       self.Pc_bar.value(),
            'MR':           self.MR.value(),
            'cstar':        self.cstar.value(),
            'mdot_gas':     self.mdot_gas.value(),
            'emissivity':   self.emissivity.value(),
            'fuel_name':    self.fuel_cb.currentText(),
            'ox_name':      self.ox_cb.currentText(),
            'mdot_coolant': self.mdot_coolant.value(),
            'T_in':         self.T_in.value(),
            'P_in':         self.P_in.value() * 1e6,   # MPa → Pa
            'coolant_fluid':self.coolant_cb.currentText(),
            'N_channels':   self.N_ch.value(),
            'a':            self.a_ch.value()    * 1e-3,   # mm → m
            'H':            self.H_ch.value()    * 1e-3,
            't_wall':       self.t_wall.value()  * 1e-3,
            'roughness':    self.roughness.value()* 1e-6,  # µm → m
            'helix_angle':  self.helix_ang.value(),
            'film_cooling': self.film_enable.isChecked(),
            'film_frac':    self.film_frac.value() / 100.0,
            'film_A_coeff': self.film_A_coeff.value(),
        }

        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)
        self.status_lbl.setText("Running solver…")

        self._worker = SolverWorker(params)
        self._worker.result_ready.connect(self._on_result)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_error(self, msg):
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)
        self.status_lbl.setText("Error — see details.")
        QMessageBox.critical(self, "Solver error", msg)

    def _on_result(self, res):
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)
        self.status_lbl.setText(
            f"Done.  q_throat = {res['solver'].q[res['geom'].throat_index()]/1e6:.2f} MW/m²  |  "
            f"T_wg_throat = {res['solver'].T_wg[res['geom'].throat_index()]:.0f} K"
        )
        self._plot_thermal(res)
        self._plot_stress(res)
        self._plot_film(res)
        self._plot_compare(res)
        self.tabs.setCurrentIndex(0)

    # ── Plotting ──────────────────────────────────────────────────────────

    def _plot_thermal(self, res):
        cv = self.canvas_thermal
        cv.fig.clear()
        axs = cv.fig.subplots(2, 3)
        x, r, s = res['x'], res['r'], res['solver']

        axs[0,0].plot(x, s.T_c, color='tab:blue')
        axs[0,0].set_title("Coolant Temperature"); axs[0,0].set_ylabel("K"); axs[0,0].grid(True)

        axs[0,1].plot(x, s.P_c/1e6, color='tab:green')
        axs[0,1].set_title("Coolant Pressure"); axs[0,1].set_ylabel("MPa"); axs[0,1].grid(True)

        ax = axs[0,2]
        ax.plot(x, s.M, color='tab:blue'); ax.set_ylabel("Mach", color='tab:blue')
        ax.set_title("Mach + Radius"); ax.grid(True)
        ax2 = ax.twinx(); ax2.plot(x, r, color='tab:red', lw=0.8)
        ax2.set_ylabel("Radius (m)", color='tab:red')

        axs[1,0].plot(x, s.T_wg, color='tab:orange', label='T_wg')
        axs[1,0].plot(x, s.T_wl, color='tab:blue',   label='T_wl')
        axs[1,0].set_title("Wall Temperatures"); axs[1,0].set_ylabel("K")
        axs[1,0].legend(fontsize=8); axs[1,0].grid(True)
        ax2 = axs[1,0].twinx(); ax2.plot(x, r, color='tab:red', lw=0.8)
        ax2.set_ylabel("Radius (m)", color='tab:red')

        axs[1,1].plot(x, s.q/1e6,                    color='tab:purple', label='Total')
        axs[1,1].plot(x, (s.q-s.q_rad)/1e6,          color='tab:blue',   label='Conv', ls='--')
        axs[1,1].plot(x, s.q_rad/1e6,                color='tab:orange', label='Rad',  ls='--')
        axs[1,1].set_title("Heat Flux"); axs[1,1].set_ylabel("MW/m²")
        axs[1,1].legend(fontsize=8); axs[1,1].grid(True)
        ax2 = axs[1,1].twinx(); ax2.plot(x, r, color='tab:red', lw=0.8)
        ax2.set_ylabel("Radius (m)", color='tab:red')

        ax = axs[1,2]
        ax.plot(x, s.u_c, color='tab:blue'); ax.set_ylabel("Velocity (m/s)", color='tab:blue')
        ax.set_title("Coolant Velocity & Density"); ax.grid(True)
        ax2 = ax.twinx(); ax2.plot(x, s.rho_c, color='tab:orange')
        ax2.set_ylabel("Density (kg/m³)", color='tab:orange')

        for ax in axs.flat:
            ax.set_xlabel("Axial position (m)")
        cv.redraw()

    def _plot_stress(self, res):
        cv = self.canvas_stress
        cv.fig.clear()
        axs = cv.fig.subplots(1, 3)
        x, r, s = res['x'], res['r'], res['solver']
        mat = res['material']
        sv, st, stth, stg, sl, sf = (res['sigma_vm'], res['sigma_t'],
            res['sigma_t_th'], res['sigma_t_global'], res['sigma_l'], res['safety'])

        axs[0].plot(x, sv/1e6,   label='Von Mises')
        axs[0].plot(x, sl/1e6,   label='Longitudinal')
        axs[0].plot(x, st/1e6,   label='Local Hoop')
        axs[0].plot(x, stth/1e6, label='Thermal Tang.')
        axs[0].plot(x, stg/1e6,  label='Global Hoop')
        axs[0].set_title("Stress Components"); axs[0].set_ylabel("MPa")
        axs[0].legend(fontsize=7); axs[0].grid(True)

        ys  = np.array([mat.yield_strength(T)/1e6 for T in s.T_wg])
        uts = np.array([mat.ultimate_strength(T)/1e6 for T in s.T_wg])
        axs[1].plot(x, sv/1e6,  label='Von Mises')
        axs[1].plot(x, ys,  '--', label='Yield')
        axs[1].plot(x, uts, '--', label='UTS')
        axs[1].set_title("Strength vs Stress"); axs[1].set_ylabel("MPa")
        axs[1].legend(fontsize=7); axs[1].grid(True)
        ax2 = axs[1].twinx(); ax2.plot(x, sf, 'k-.', lw=0.8)
        ax2.set_ylabel("Safety Factor")

        geom = res['geom']
        axs[2].plot(x, geom.a*1e3,            color='tab:blue',   label='Width a')
        axs[2].plot(x, geom.H*1e3,            color='tab:orange', label='Height H')
        axs[2].plot(x, geom.web_width()*1e3,  color='tab:green',  label='Web b')
        axs[2].set_title("Channel Geometry"); axs[2].set_ylabel("mm")
        axs[2].legend(fontsize=7); axs[2].grid(True)

        for ax in axs:
            ax.set_xlabel("Axial position (m)")
        cv.redraw()

    def _plot_film(self, res):
        cv = self.canvas_film
        cv.fig.clear()
        axs = cv.fig.subplots(1, 3)
        x, r, s, gas = res['x'], res['r'], res['solver'], res['gas']

        T_aw_base = gas.T0 * (
            (1 + s.Pr_g**(1/3) * (s.gamma_g-1)/2 * s.M**2)
            / (1 + (s.gamma_g-1)/2 * s.M**2)
        )

        axs[0].plot(x, s.eta_film, color='tab:cyan')
        axs[0].set_title("Film Effectiveness η"); axs[0].set_ylabel("η"); axs[0].set_ylim(0, 1)
        if not res['film_cooling']:
            axs[0].text(0.5, 0.5, "Film cooling OFF",
                        transform=axs[0].transAxes, ha='center', va='center',
                        fontsize=11, color='gray')
        axs[0].grid(True)

        axs[1].plot(x, T_aw_base,  color='tab:red',    ls='--', label='T_aw (no film)')
        axs[1].plot(x, s.T_aw_eff, color='tab:blue',           label='T_aw_eff')
        axs[1].plot(x, s.T_wg,     color='tab:orange',          label='T_wg')
        axs[1].set_title("Adiabatic Wall Temp"); axs[1].set_ylabel("K")
        axs[1].legend(fontsize=8); axs[1].grid(True)

        axs[2].plot(x, s.q/1e6, color='tab:purple')
        axs[2].set_title("Heat Flux"); axs[2].set_ylabel("MW/m²"); axs[2].grid(True)
        ax2 = axs[2].twinx(); ax2.plot(x, r, color='tab:red', lw=0.8)
        ax2.set_ylabel("Radius (m)", color='tab:red')

        for ax in axs:
            ax.set_xlabel("Axial position (m)")
        cv.redraw()

    def _plot_compare(self, res):
        cv = self.canvas_compare
        cv.fig.clear()
        axs = cv.fig.subplots(1, 2)
        x, r = res['x'], res['r']
        p1, s = res['solver_p1'], res['solver']

        axs[0].plot(x, p1.T_wg, color='tab:orange', label='T_wg (no film)')
        axs[0].plot(x, p1.T_wl, color='tab:blue',   label='T_wl (no film)')
        if res['film_cooling']:
            axs[0].plot(x, s.T_wg, color='tab:orange', ls='--', label='T_wg (film)')
            axs[0].plot(x, s.T_wl, color='tab:blue',   ls='--', label='T_wl (film)')
        axs[0].set_title("Wall Temps: Film vs No Film"); axs[0].set_ylabel("K")
        axs[0].legend(fontsize=8); axs[0].grid(True)
        ax2 = axs[0].twinx(); ax2.plot(x, r, color='tab:red', lw=0.8, alpha=0.4)
        ax2.set_ylabel("Radius (m)", color='tab:red')

        axs[1].plot(x, p1.q/1e6,           color='tab:purple', label='Total (no film)')
        axs[1].plot(x, (p1.q-p1.q_rad)/1e6,color='tab:blue',   label='Conv  (no film)')
        if res['film_cooling']:
            axs[1].plot(x, s.q/1e6,            color='tab:purple', ls='--', label='Total (film)')
            axs[1].plot(x, (s.q-s.q_rad)/1e6,  color='tab:blue',   ls='--', label='Conv  (film)')
        axs[1].set_title("Heat Flux: Film vs No Film"); axs[1].set_ylabel("MW/m²")
        axs[1].legend(fontsize=8); axs[1].grid(True)
        ax2 = axs[1].twinx(); ax2.plot(x, r, color='tab:red', lw=0.8, alpha=0.4)
        ax2.set_ylabel("Radius (m)", color='tab:red')

        for ax in axs:
            ax.set_xlabel("Axial position (m)")
        cv.redraw()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
