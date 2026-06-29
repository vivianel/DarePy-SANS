# -*- coding: utf-8 -*-
"""
Script 2: Flexible Beam Center Determination (Draggable & Resizable Circle Mode)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import Button

current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

codes_dir = os.path.join(parent_dir, "darepy/codes")
if codes_dir not in sys.path:
    sys.path.insert(0, codes_dir)

from utils import load_hdf, find_hdf_filename, load_config, load_instrument_registry

# ==========================================
# %% Configuration & Path Resolution
# ==========================================
config = load_config()
plt.close('all')

if len(sys.argv) > 1 and sys.argv[1].endswith('.yaml'):
    yaml_filepath = os.path.abspath(sys.argv[1])
else:
    from utils import CURRENT_DIR
    pointer_file = os.path.join(CURRENT_DIR, ".active_experiment.txt")
    if os.path.exists(pointer_file):
        with open(pointer_file, 'r') as f:
            exp_folder = f.read().strip()
            yaml_filepath = os.path.join(exp_folder, "config_experiment.yaml")
    else:
        yaml_filepath = os.path.join(config['analysis_paths']['scripts_dir'], 'config_experiment.yaml')

path_hdf_raw = config['analysis_paths']['raw_data']
instrument = config['instrument_setup']['which_instrument']
scanNr = config['beam_center_mask']['scan_nr']

config_clim = config['beam_center_mask'].get('clim', None)
plot_scale = config['beam_center_mask'].get('plot_scale', 'log')

# ==========================================
# %% Load HDF & Instrument Definitions
# ==========================================
name_hdf = find_hdf_filename(path_hdf_raw, scanNr)
if not name_hdf:
    print(f"[ERROR] HDF file entry not found.")
    sys.exit(1)

inst_reg = load_instrument_registry()
pixel1 = inst_reg[instrument]['pixel_size']
pixel2 = inst_reg[instrument]['pixel_size']

try:
    img = load_hdf(path_hdf_raw, name_hdf, 'counts')
    Detector_distance = load_hdf(path_hdf_raw, name_hdf, 'detx')
    wl = load_hdf(path_hdf_raw, name_hdf, 'wl')
except Exception as e:
    print(f"Error loading HDF data: {e}")
    sys.exit(1)

if img.ndim != 2:
    img = np.mean(img, axis=0)

dist_key = float(Detector_distance)

# ----------------------------------------------------------------------
# %% Center Finder Click Handler (Pure Interactive Circle Mode)
class CenterDeterminationManager:
    try:
        matplotlib.use('Qt5Agg')
    except:
        matplotlib.use('TkAgg')

    def __init__(self, raw_img, clim, scale_mode, initial_fig, initial_ax):
        self.raw_img = raw_img
        self.clim = clim
        self.scale_mode = scale_mode

        self.fig = initial_fig
        self.ax = initial_ax
        self.clicks = []

        self.center_x = None
        self.center_y = None
        self.circle_radius = None

        self.overlay_patch = None
        self.center_marker = None
        self.colorbar = None

        # Draggable state variables
        self._is_dragging = False
        self._is_resizing = False
        self._press_xy = None
        self._initial_center = None
        self._initial_radius = None

    def connect(self):
        # Adjusted layout slightly to make clear room for buttons and the colorbar axis
        self.fig.subplots_adjust(left=0.22, right=0.85)
        self.redraw_axes()

        # Keep a dedicated clean button on the left sidebar panel
        ax_clear = self.fig.add_axes([0.02, 0.65, 0.15, 0.05])
        self.btn_clear = Button(ax_clear, 'Reset Circle')
        self.btn_clear.on_clicked(self.clear_clicks)

        # Connect core mouse events for placement and dragging
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('close_event', self.handle_close)

    def clear_clicks(self, event):
        self.clicks = []
        self.center_x = None
        self.center_y = None
        self.circle_radius = None
        self.overlay_patch = None
        self.center_marker = None
        self.redraw_axes()

    def redraw_axes(self):
        """Re-renders canvas frame updates smoothly."""
        self.ax.clear()

        # Prevent log(0) warnings by clipping zeros natively
        safe_img = np.where(self.raw_img <= 0, 1e-4, self.raw_img)

        if self.scale_mode == 'lin':
            self.im = self.ax.imshow(self.raw_img, origin='lower', cmap='jet', clim=self.clim)
        elif self.scale_mode == 'log':
            self.im = self.ax.imshow(np.log(safe_img), origin='lower', cmap='jet', clim=self.clim)

        # --- COLORBAR ADDED HERE ---
        # Generate it on the first draw, update it seamlessly on redraws to prevent glitches
        if self.colorbar is None:
            cax = self.fig.add_axes([0.87, 0.11, 0.03, 0.77])  # Fixed position in the right margin
            self.colorbar = self.fig.colorbar(self.im, cax=cax)
            self.colorbar.set_label('Intensity (Counts)', rotation=270, labelpad=15)
        else:
            self.colorbar.update_normal(self.im)
        # ---------------------------

        # Safely format titles when center is not yet defined
        cx_str = f"{self.center_x:.2f}" if self.center_x is not None else "N/A"
        cy_str = f"{self.center_y:.2f}" if self.center_y is not None else "N/A"

        self.ax.set_title(f"Interactive Beam Center Alignment\nDistance: {dist_key}m | Position: [{cx_str}, {cy_str}]")
        self.ax.set_xlabel("Detector Matrix Width (X)")
        self.ax.set_ylabel("Detector Matrix Height (Y)")

        # Re-verify and stitch patch layers if they are active
        if self.center_x is not None and self.circle_radius is not None:
            self.ax.axhline(self.center_y, color='magenta', linestyle=':', alpha=0.6)
            self.ax.axvline(self.center_x, color='magenta', linestyle=':', alpha=0.6)
            self.update_circle_patch()
        elif self.center_x is not None:
            self.center_marker = self.ax.plot(self.center_x, self.center_y, 'ro', markersize=6)

        self.fig.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes != self.ax:
            return

        # --- DRAG / RESIZE LOGIC FOR EXISTING CIRCLE ---
        if self.center_x is not None and self.circle_radius is not None:
            dx = event.xdata - self.center_x
            dy = event.ydata - self.center_y
            dist_from_center = np.sqrt(dx**2 + dy**2)

            # Hit-test window calculation near edge ring boundary
            edge_tolerance = max(2.0, self.circle_radius * 0.15)

            if abs(dist_from_center - self.circle_radius) < edge_tolerance:
                self._is_resizing = True
                self._press_xy = (event.xdata, event.ydata)
                self._initial_radius = self.circle_radius
                return
            elif dist_from_center < self.circle_radius:
                self._is_dragging = True
                self._press_xy = (event.xdata, event.ydata)
                self._initial_center = (self.center_x, self.center_y)
                return

        # --- INITIAL SELECTION GENERATION ---
        if len(self.clicks) < 2:
            self.clicks.append([event.xdata, event.ydata])

            if len(self.clicks) == 1:
                self.center_x, self.center_y = event.xdata, event.ydata
                self.center_marker = self.ax.plot(self.center_x, self.center_y, 'ro', markersize=6)
                print(f"Center anchor set at: X={np.round(self.center_x,2)}, Y={np.round(self.center_y,2)}")
            elif len(self.clicks) == 2:
                dx = event.xdata - self.center_x
                dy = event.ydata - self.center_y
                self.circle_radius = np.sqrt(dx**2 + dy**2)
                self.redraw_axes() # Renders crosshairs and patches properly

        self.fig.canvas.draw_idle()

    def on_motion(self, event):
        if event.inaxes != self.ax or not (self._is_dragging or self._is_resizing):
            return

        if self._is_dragging:
            dx = event.xdata - self._press_xy[0]
            dy = event.ydata - self._press_xy[1]
            self.center_x = self._initial_center[0] + dx
            self.center_y = self._initial_center[1] + dy

        elif self._is_resizing:
            dx = event.xdata - self.center_x
            dy = event.ydata - self.center_y
            self.circle_radius = np.sqrt(dx**2 + dy**2)

        self.update_circle_patch()
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        if self._is_dragging or self._is_resizing:
            print(f"Adjusted Circle Center: X={np.round(self.center_x,2)}, Y={np.round(self.center_y,2)} | Radius={np.round(self.circle_radius, 2)}")
        self._is_dragging = False
        self._is_resizing = False
        self._press_xy = None

    def update_circle_patch(self):
        if self.overlay_patch:
            self.overlay_patch.remove()

        self.overlay_patch = plt.Circle((self.center_x, self.center_y), self.circle_radius,
                                        color='white', fill=False, linestyle='--', linewidth=2)
        self.ax.add_patch(self.overlay_patch)

        if self.center_marker and len(self.center_marker) > 0:
            self.center_marker[0].set_data([self.center_x], [self.center_y])

        # Update title text safely dynamically
        self.ax.set_title(f"Interactive Beam Center Alignment\nDistance: {dist_key}m | Position: [{self.center_x:.2f}, {self.center_y:.2f}]")

    def on_key(self, event):
        if event.key == 'enter':
            if self.circle_radius is None:
                print("[WARNING] Please click a second point to define a radius.")
                return

            plt.close(self.fig)
            self._integrate_and_display_final()
            save_center_to_yaml(yaml_filepath, Detector_distance, self)


    def handle_close(self, event):
        self.fig.canvas.stop_event_loop()

    def _integrate_and_display_final(self):
        fig_results, axes_results = plt.subplots(1, 1, figsize=(8, 7))
        fig_results.suptitle('SANS Beam Center Analysis Results', fontsize=16)

        safe_img = np.where(self.raw_img <= 0, 1e-4, self.raw_img)
        if self.scale_mode == 'lin':
            im = axes_results.imshow(self.raw_img, origin='lower', cmap='jet', clim=self.clim)
        elif self.scale_mode == 'log':
            im = axes_results.imshow(np.log(safe_img), origin='lower', cmap='jet', clim=self.clim)

        axes_results.plot(self.center_x, self.center_y, 'wx', markersize=15, mew=2)
        axes_results.add_patch(plt.Circle((self.center_x, self.center_y), self.circle_radius, color='white', fill=False, linestyle='--', linewidth=2))
        axes_results.set_title(f'Calculated Center (CIRCLE): X={np.round(self.center_x,2)}, Y={np.round(self.center_y,2)}')

        # Attach matching scale colorbar to the final calculations window too
        cbar_results = fig_results.colorbar(im, ax=axes_results, fraction=0.046, pad=0.04)
        cbar_results.set_label('Intensity (Counts)', rotation=270, labelpad=15)

        plt.pause(0.1)

# ==========================================
# %% Overwrite Center in Configuration File
# ==========================================
def save_center_to_yaml(yaml_path, distance_val, manager):
    try:
        from ruamel.yaml import YAML
        yaml_parser = 'ruamel'
    except ImportError:
        import yaml
        yaml_parser = 'standard'

    if not os.path.exists(yaml_path) or manager.center_x is None:
        return

    dist_key = float(distance_val)
    new_center = [float(np.round(manager.center_x, 2)), float(np.round(manager.center_y, 2))]

    with open(yaml_path, 'r') as f:
        if yaml_parser == 'ruamel':
            ryaml = YAML()
            ryaml.preserve_quotes = True
            config_dict = ryaml.load(f) or {}
        else:
            config_dict = yaml.safe_load(f) or {}

    choice = input(f"Overwrite Center configuration for distance {dist_key}m with {new_center}? (y/n): ").strip().lower()
    if choice in ['y', 'yes']:
        if 'detector_geometry' not in config_dict:
            config_dict['detector_geometry'] = {}
        if 'beam_center_guess' not in config_dict['detector_geometry']:
            config_dict['detector_geometry']['beam_center_guess'] = {}

        config_dict['detector_geometry']['beam_center_guess'][dist_key] = new_center
        with open(yaml_path, 'w') as f:
            if yaml_parser == 'ruamel':
                ryaml.dump(config_dict, f)
            else:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        print("[SUCCESS] Center saved successfully.")

# ==========================================
# %% Main Execution Loop
# ==========================================
if not matplotlib.get_backend().startswith('Tk') and not 'IPython' in sys.modules:
    matplotlib.use('TkAgg')

interactive_fig, interactive_ax = plt.subplots(figsize=(9, 7))
center_manager = CenterDeterminationManager(img, config_clim, plot_scale, interactive_fig, interactive_ax)
center_manager.connect()

print("\n[INFO] Beam Center Window is active. Modify your alignment, then press [ENTER] to save coordinates.")

while True:
    try:
        if len(plt.get_fignums()) > 0:
            plt.pause(0.2)
        else:
            break
    except KeyboardInterrupt:
        break
