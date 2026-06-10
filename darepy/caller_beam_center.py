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

# %% Apply Scaling Modes Dynamically (Fixed with Native Matplotlib Norms)
img_display = img.copy()
finite_vals = img_display[np.isfinite(img_display) & (img_display > 0)]

if config_clim is not None:
    vmin, vmax = float(config_clim[0]), float(config_clim[1])
else:
    vmin = np.percentile(finite_vals, 1) if finite_vals.size > 0 else 0.1
    vmax = np.percentile(finite_vals, 99.5) if finite_vals.size > 0 else 100.0

if plot_scale == 'log':
    print(f"Plotting in LOGARITHMIC scale. Range: [{vmin} to {vmax}]")
    vmin = max(1e-6, vmin)
    plot_norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
else:
    print(f"Plotting in LINEAR scale. Range: [{vmin} to {vmax}]")
    plot_norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

dist_key = float(Detector_distance)

# ----------------------------------------------------------------------
# %% Center Finder Click Handler (Pure Interactive Circle Mode)
class CenterDeterminationManager:
    try:
        matplotlib.use('Qt5Agg')
    except:
        matplotlib.use('TkAgg')

    def __init__(self, img_data, plot_norm, initial_fig, initial_ax):
        self.img_data = img_data
        self.norm = plot_norm

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
        if self.overlay_patch:
            self.overlay_patch.remove()
            self.overlay_patch = None
        if self.center_marker:
            for marker in self.center_marker:
                marker.remove()
            self.center_marker = None
        self.redraw_axes()

    def redraw_axes(self):
        self.ax.clear()
        self.fig.suptitle('SANS Center Calibration', fontsize=14, color='darkgreen')

        msg = '1st Click: Set Center | 2nd Click: Set Radius\nAfterward: Drag inside to move, drag outer ring to resize!\nPress [ENTER] to confirm.'
        self.ax.set_title(msg, fontsize=9)

        # Display main matrix mapping
        im = self.ax.imshow(self.img_data, origin='lower', cmap='jet', norm=self.norm)
        self.ax.grid(True, color='w', linestyle='--')

        # Draw colorbar only if it doesn't already exist to prevent duplicate stacking
        if self.colorbar is None:
            self.colorbar = self.fig.colorbar(im, ax=self.ax, fraction=0.046, pad=0.04)
            self.colorbar.set_label('Intensity (Counts)', rotation=270, labelpad=15)

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
                self.update_circle_patch()

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

        if self.center_marker:
            self.center_marker[0].set_data([self.center_x], [self.center_y])

    def on_key(self, event):
        if event.key == 'enter':
            if self.circle_radius is None:
                print("[WARNING] Please click a second point to define a radius.")
                return

            plt.close(self.fig)
            self.process_beam_center()

    def handle_close(self, event):
        self.fig.canvas.stop_event_loop()

    def _refine_center_of_mass(self, guess_x, guess_y, search_mask):
        y_grid, x_grid = np.indices(self.img_data.shape)
        valid_pixels = np.isfinite(img) & (img > 0) & search_mask
        total_intensity = np.sum(img[valid_pixels])
        if total_intensity > 0:
            return np.sum(x_grid[valid_pixels] * img[valid_pixels]) / total_intensity, np.sum(y_grid[valid_pixels] * img[valid_pixels]) / total_intensity
        return guess_x, guess_y

    def process_beam_center(self):
        y_grid, x_grid = np.indices(self.img_data.shape)
        print(f"Refining profile centers using center-of-mass limits...")

        search_width = 15
        r_grid = np.sqrt((x_grid - self.center_x)**2 + (y_grid - self.center_y)**2)
        search_mask = (r_grid >= (self.circle_radius - search_width)) & (r_grid <= (self.circle_radius + search_width))
        guess_x, guess_y = self.center_x, self.center_y

        self.center_x, self.center_y = self._refine_center_of_mass(guess_x, guess_y, search_mask)
        self._integrate_and_display_final()
        save_center_to_yaml(yaml_filepath, Detector_distance, self)

    def _integrate_and_display_final(self):
        fig_results, axes_results = plt.subplots(1, 1, figsize=(8, 7))
        fig_results.suptitle('SANS Beam Center Analysis Results', fontsize=16)

        im = axes_results.imshow(self.img_data, origin='lower', cmap='jet', norm=self.norm)
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
center_manager = CenterDeterminationManager(img_display, plot_norm, interactive_fig, interactive_ax)
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
