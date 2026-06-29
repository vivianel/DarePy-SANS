# -*- coding: utf-8 -*-
"""
Script 1: Masking & Transmission Area Selection
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Path configurations to locate darepy modules
current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

codes_dir = os.path.join(parent_dir, "darepy/codes")
if codes_dir not in sys.path:
    sys.path.insert(0, codes_dir)

from utils import load_hdf, find_hdf_filename, load_config

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
beamstop = config.get('transmission_setup', {}).get('beamstop', 'standard')

# --- FETCH CLIM & SCALE FROM CONFIG ---
config_clim = config['beam_center_mask'].get('clim', None)
plot_scale = config['beam_center_mask'].get('plot_scale', 'log')

# ==========================================
# %% Load Data
# ==========================================
name_hdf = find_hdf_filename(path_hdf_raw, scanNr)
if not name_hdf:
    print(f"[ERROR] Could not find any HDF file ending in {scanNr}.hdf in the raw_data folder.")
    sys.exit(1)

try:
    img = load_hdf(path_hdf_raw, name_hdf, 'counts')
    Detector_distance = load_hdf(path_hdf_raw, name_hdf, 'detx')
except Exception as e:
    print(f"Error loading HDF data: {e}")
    sys.exit(1)

if img.ndim != 2:
    img = np.mean(img, axis=0)

# %% Parse Limits Natively
finite_vals = img[np.isfinite(img)]
if config_clim is not None:
    clim = [float(config_clim[0]), float(config_clim[1])]
else:
    if finite_vals.size > 0:
        clim = [float(np.percentile(finite_vals, 1)), float(np.percentile(finite_vals, 99.5))]
    else:
        clim = [1e-2, 100.0] if plot_scale == 'log' else [0.0, 100.0]

if plot_scale == 'log':
    if clim[0] <= 0:
        clim[0] = 1e-2
    if clim[1] <= clim[0]:
        clim[1] = clim[0] + 1.0

print(f"Plotting mode: {plot_scale.upper()}")
print(f"Colorbar limits (linear reference space) set to: {clim}")

# ==========================================
# %% Interactive Masking Class
# ==========================================
class MaskTransmissionManager:
    try:
        matplotlib.use('Qt5Agg')
    except:
        matplotlib.use('TkAgg')

    def __init__(self, img_data, clim_limits, scale_mode, initial_fig, initial_ax, inst_name):
        self.img_data = img_data.copy()
        self.clim = clim_limits
        self.scale_mode = scale_mode
        self.instrument = inst_name

        self.fig = initial_fig
        self.ax = initial_ax
        self.cbar = None
        self.clicks = []

        self.cid_click = None
        self.cid_key = None
        self.cid_close = None

        self.current_state = 'bs_collect'
        self.current_bs_count = 0
        self.pending_coords = None

        self.bs_masks_dict = {}
        self.img_masked_bs = self.img_data.copy()
        self.saved_trans = None

        self.switching_step = False
        self.user_aborted = False  # Soft abort tracking flag

    def connect(self):
        self.disconnect_listeners()
        self.clicks = []
        self.pending_coords = None
        self.switching_step = False
        self.ax.clear()

        if self.current_state == 'bs_collect':
            self.fig.suptitle('SANS Masking Setup: Step 1 - Define Beam Stops', fontsize=14, color='darkblue')
            title = f'Area {self.current_bs_count + 1}. Click 4 corners to instantly mask. Press ENTER to finish Step 1.'
            img_to_display = self.img_masked_bs
        elif self.current_state == 'transmission':
            self.fig.suptitle('SANS Masking Setup: Step 2 - Select Transmission Area', fontsize=14, color='darkorange')
            title = 'Click 4 corners + ENTER to save and view Final Results.'
            img_to_display = self.img_data
        else:
            return

        self.ax.set_title(title, fontsize=10)

        # --- UPDATED COLOR SCALING LOGIC (MATCHING SCRIPT 2) ---
        with np.errstate(invalid='ignore', divide='ignore'):
            if self.scale_mode == 'lin':
                im = self.ax.imshow(img_to_display, origin='lower', cmap='jet', clim=self.clim)
            elif self.scale_mode == 'log':
                # Safely clip zeros while preserving active NaNs
                safe_img = np.where(img_to_display <= 0, 1e-4, img_to_display)
                im = self.ax.imshow(np.log(safe_img), origin='lower', cmap='jet', clim=self.clim)

        # --- PERSISTENT COLORBAR AXIS ENGINE ---
        # Ensures the colorbar axis is only generated once per unique figure window
        if not hasattr(self, 'cax_bar') or self.cax_bar not in self.fig.axes:
            divider = make_axes_locatable(self.ax)
            self.cax_bar = divider.append_axes("right", size="5%", pad=0.1)
        else:
            self.cax_bar.clear()

        self.cbar = self.fig.colorbar(im, cax=self.cax_bar)

        self.ax.grid(which='major', color='w', linestyle='--', linewidth=1)
        self.fig.canvas.draw_idle()

        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', lambda event: self.on_click(event, 4))
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.cid_close = self.fig.canvas.mpl_connect('close_event', self.on_close)

    def disconnect_listeners(self):
        if self.cid_click is not None:
            self.fig.canvas.mpl_disconnect(self.cid_click)
            self.cid_click = None
        if self.cid_key is not None:
            self.fig.canvas.mpl_disconnect(self.cid_key)
            self.cid_key = None
        if self.cid_close is not None:
            self.fig.canvas.mpl_disconnect(self.cid_close)
            self.cid_close = None

    def on_click(self, event, click_limit):
        if event.inaxes == self.ax:
            if len(self.clicks) < click_limit:
                self.clicks.append([event.xdata, event.ydata])
                print(f'Clicked: x = {np.round(event.xdata, 2)}, y = {np.round(event.ydata, 2)}')

                if len(self.clicks) == click_limit:
                    self.pending_coords = self._calculate_area_coords()

                    if self.current_state == 'bs_collect':
                        self.confirm_beamstop()
                    elif self.current_state == 'transmission':
                        Ymin, Ymax, Xmin, Xmax = self.pending_coords
                        self.ax.add_patch(plt.Rectangle((Xmin, Ymin), Xmax - Xmin, Ymax - Ymin, linewidth=2, edgecolor='red', facecolor='none'))
                        self.ax.set_title("Transmission area selected. Press ENTER to complete.", fontsize=10, color='red')
                        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == 'enter':
            if self.current_state == 'bs_collect':
                print("\n[ENTER] Finished Step 1 via keyboard.")
                self.switching_step = True
                self.disconnect_listeners()
                plt.close(self.fig)
            elif self.current_state == 'transmission':
                if self.pending_coords is not None:
                    self.switching_step = True
                    self.confirm_transmission()
                else:
                    print("Cannot finalize transmission yet. You must click 4 corners first.")

    def on_close(self, event):
        """Triggers a soft abort tracking flag without taking down the interactive kernel."""
        if not self.switching_step:
            print("\n[INFO] Interface window closed manually. Stopping setup sequence.")
            self.user_aborted = True

    def _calculate_area_coords(self):
        all_x = [click[0] for click in self.clicks]
        all_y = [click[1] for click in self.clicks]
        rows, cols = self.img_data.shape
        min_y = max(0, int(np.floor(np.min(all_y))))
        max_y = min(rows, int(np.ceil(np.max(all_y))))
        min_x = max(0, int(np.floor(np.min(all_x))))
        max_x = min(cols, int(np.ceil(np.max(all_x))))
        return [min_y, max_y, min_x, max_x]

    def confirm_beamstop(self):
        coords = self.pending_coords
        self.current_bs_count += 1
        key = f'bs_{self.current_bs_count}'
        self.bs_masks_dict[key] = coords
        print(f"[INSTANT MASK] Stored {key}: [Ymin, Ymax, Xmin, Xmax] = {coords}")

        Ymin, Ymax, Xmin, Xmax = coords
        self.img_masked_bs[Ymin:Ymax, Xmin:Xmax] = np.nan
        self.connect()

    def confirm_transmission(self):
        self.saved_trans = self.pending_coords
        print(f'[CONFIRMED] Transmission area configured: {self.saved_trans}')
        self.disconnect_listeners()
        plt.close(self.fig)
        self.current_state = 'done'
        self._plot_final_results()

    def _plot_final_results(self):
        fig_results, axes_results = plt.subplots(1, 1, figsize=(7, 7))

        with np.errstate(invalid='ignore', divide='ignore'):
            if self.scale_mode == 'lin':
                im = axes_results.imshow(self.img_masked_bs, origin='lower', cmap='jet', clim=self.clim)
            elif self.scale_mode == 'log':
                safe_img = np.where(self.img_masked_bs <= 0, 1e-4, self.img_masked_bs)
                im = axes_results.imshow(np.log(safe_img), origin='lower', cmap='jet', clim=self.clim)

        divider = make_axes_locatable(axes_results)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig_results.colorbar(im, cax=cax)

        if self.saved_trans:
            axes_results.add_patch(plt.Rectangle((self.saved_trans[2], self.saved_trans[0]), self.saved_trans[3]-self.saved_trans[2], self.saved_trans[1]-self.saved_trans[0], linewidth=2, edgecolor='red', facecolor='none'))
        axes_results.set_title('Configured Calibration Masks')
        plt.show()

# ==========================================
# %% YAML Exporter
# ==========================================
def save_masks_to_yaml(yaml_path, distance_val, manager_obj):
    try:
        from ruamel.yaml import YAML
        yaml_parser = 'ruamel'
    except ImportError:
        import yaml
        yaml_parser = 'standard'

    if not os.path.exists(yaml_path):
        return

    dist_key = float(distance_val)
    new_bs = {f'bs{i}': coords for i, (_, coords) in enumerate(manager_obj.bs_masks_dict.items())}

    with open(yaml_path, 'r') as f:
        if yaml_parser == 'ruamel':
            ryaml = YAML()
            ryaml.preserve_quotes = True
            config_dict = ryaml.load(f) or {}
        else:
            config_dict = yaml.safe_load(f) or {}

    if 'detector_geometry' not in config_dict:
        config_dict['detector_geometry'] = {}
    for sec in ['beamstopper_coordinates', 'transmission_coordinates']:
        if sec not in config_dict['detector_geometry']:
            config_dict['detector_geometry'][sec] = {}

    while True:
        choice = input(f"Save these updated mask coordinates to config file for {dist_key}m? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            config_dict['detector_geometry']['beamstopper_coordinates'][dist_key] = new_bs
            if manager_obj.saved_trans:
                config_dict['detector_geometry']['transmission_coordinates'][dist_key] = manager_obj.saved_trans
            with open(yaml_path, 'w') as f:
                if yaml_parser == 'ruamel':
                    ryaml.dump(config_dict, f)
                else:
                    yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            print("[SUCCESS] Mask rules successfully saved.")
            break
        elif choice in ['n', 'no']:
            break

# ==========================================
# %% Main Execution Loop
# ==========================================
interactive_fig, interactive_ax = plt.subplots(figsize=(8, 7))
manager = MaskTransmissionManager(img, clim, plot_scale, interactive_fig, interactive_ax, instrument)
manager.connect()

while manager.current_state != 'done':
    try:
        # Check if user clicked close window natively
        if manager.user_aborted:
            break

        if plt.fignum_exists(interactive_fig.number):
            plt.pause(0.1)
        else:
            if manager.current_state == 'bs_collect':
                if beamstop == 'semitransparent':
                    print("\nMoving to Step 2: Transmission Area Selection.")
                    interactive_fig, interactive_ax = plt.subplots(figsize=(8, 7))
                    manager.fig, manager.ax = interactive_fig, interactive_ax
                    manager.current_state = 'transmission'
                    manager.connect()
                else:
                    print(f"Skipping transmission step because beamstop setup is '{beamstop}'.")
                    break
            else:
                break
    except KeyboardInterrupt:
        break

# Only proceed to the saving engine if the user completed the process smoothly
if not manager.user_aborted:
    save_masks_to_yaml(yaml_filepath, Detector_distance, manager)
else:
    print("[INFO] Script stopped without altering configuration.")
