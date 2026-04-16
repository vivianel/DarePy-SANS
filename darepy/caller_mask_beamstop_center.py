# -*- coding: utf-8 -*-
"""
Created on Wed Dec 6 10:52:03 2023

@author: lutzbueno_v
"""

import sys
import os

# 1. Get the directory of the current script (darepy/codes/)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Go up one level to find utils.py (in darepy/)
parent_dir = os.path.dirname(current_script_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


# ==========================================
# %% STANDARD IMPORTS
# ==========================================
import numpy as np
from ruamel.yaml import YAML
yaml = YAML()
yaml.preserve_quotes = True

import matplotlib.pyplot as plt
import matplotlib
import pyFAI.azimuthalIntegrator as pyFAI_ai
from utils import load_hdf, find_hdf_filename, load_config, load_instrument_registry

# ==========================================
# %% Configuration
# ==========================================
config = load_config()
# --- FIX: Retrieve the absolute path used by the loader ---
import sys
if len(sys.argv) > 1 and sys.argv[1].endswith('.yaml'):
    yaml_filepath = os.path.abspath(sys.argv[1])
else:
    # Fallback to the pointer logic if running manually in Spyder
    from utils import CURRENT_DIR
    pointer_file = os.path.join(CURRENT_DIR, ".active_experiment.txt")
    if os.path.exists(pointer_file):
        with open(pointer_file, 'r') as f:
            exp_folder = f.read().strip()
            yaml_filepath = os.path.join(exp_folder, "config_experiment.yaml")
    else:
        # Emergency fallback to current logic
        yaml_filepath = os.path.join(config['analysis_paths']['scripts_dir'], 'config_experiment.yaml')

print(f"Target YAML for saving: {yaml_filepath}")

# Pull paths dynamically from YAML
path_hdf_raw = config['analysis_paths']['raw_data']

instrument = config['instrument_setup']['which_instrument']

# Assuming you move this down to where config = load_config() is called:
scanNr = config['beam_center_mask']['scan_nr']


# ==========================================
# %% Load Data
# ==========================================

# --- 1. AUTOMATIC FILENAME DISCOVERY ---
name_hdf = find_hdf_filename(path_hdf_raw, scanNr)

if name_hdf:
    print(f"Automatically found: {name_hdf}")
else:
    print(f"[ERROR] Could not find any HDF file ending in {scanNr}.hdf in the raw_data folder.")
    import sys; sys.exit(1) #to stop the script if the file is missing

# Load the central instrument registry
inst_reg = load_instrument_registry()

# Pull pixel size dynamically!
pixel1 = inst_reg[instrument]['pixel_size']
pixel2 = inst_reg[instrument]['pixel_size']


plt.close('all')
try:
    img = load_hdf(path_hdf_raw, name_hdf, 'counts')
    Detector_distance = load_hdf(path_hdf_raw, name_hdf, 'detx')
    wl = load_hdf(path_hdf_raw, name_hdf, 'wl')
except Exception as e:
    print(f"Error loading HDF data: {e}")

# %% Safe Log Calculation
# We use np.clip to ensure the minimum value is slightly above 0 to avoid -inf
def safe_log(data):
    return np.log(np.clip(data, a_min=1e-6, a_max=None))

try:
    if img.ndim == 2:
        img_log = safe_log(img)
    else:
        img_mean = np.mean(img, axis=0)
        img_log = safe_log(img_mean)
        img = img_mean
except Exception as e:
    print(f"Error processing dimensions: {e}")

# %% Automate Color bar limits for image display (clim)
finite_log_values = img_log[np.isfinite(img_log)]

if finite_log_values.size > 0:
    p_low = 1
    p_high = 99.5
    auto_clim_min = np.percentile(finite_log_values, p_low)
    auto_clim_max = np.percentile(finite_log_values, p_high)
    clim = [max(0.0, auto_clim_min), auto_clim_max]
    print(f"Automated clim set to: {clim}")
else:
    print("Warning: No finite log values found. Using fallback [0, 7].")
    clim = [0, 7]

# ----------------------------------------------------------------------
# %% Function to plot the final results in a new figure
def plot_final_results_figure(
    original_img_log,
    masked_img_bs,
    center_x,
    center_y,
    circle_radius,
    sector_integrations,
    clim_limits,
    trans_area_coords
):
    fig_results, axes_results = plt.subplots(1, 3, figsize=(20, 7))
    fig_results.suptitle('SANS Beam Center Analysis Results', fontsize=16)

    # Left Column
    ax_left = axes_results[0]
    ax_left.imshow(masked_img_bs, origin='lower', cmap='jet', clim=clim_limits)
    ax_left.set_title('Masked Beamstop & Transmission Area')
    ax_left.set_xlabel('Pixels (X)')
    ax_left.set_ylabel('Pixels (Y)')
    ax_left.grid(True)

    if trans_area_coords:
        Ymin, Ymax, Xmin, Xmax = trans_area_coords
        rect_width = Xmax - Xmin
        rect_height = Ymax - Ymin
        transmission_box = plt.Rectangle((Xmin, Ymin), rect_width, rect_height,
                                         linewidth=2, edgecolor='red', facecolor='none', linestyle='-')
        ax_left.add_patch(transmission_box)
        ax_left.text(Xmin, Ymax, 'Trans. Area', color='red', fontsize=10, verticalalignment='bottom')

    # Middle Column
    ax_middle = axes_results[1]
    ax_middle.imshow(original_img_log, origin='lower', cmap='jet', clim=clim_limits)
    ax_middle.set_title(f'Beam Center: X={np.round(center_x,2)}, Y={np.round(center_y,2)}')
    ax_middle.set_xlabel('Pixels (X)')
    ax_middle.set_ylabel('Pixels (Y)')
    ax_middle.grid(True)
    ax_middle.plot(center_x, center_y, 'wx', markersize=15, mew=2, label='Beam Center')
    circle = plt.Circle((center_x, center_y), circle_radius, color='white', fill=False, linestyle='--', linewidth=2)
    ax_middle.add_patch(circle)
    ax_middle.legend()

    # Right Column
    ax_right = axes_results[2]
    ax_right.set_title('Azimuthal Sector Integration')
    ax_right.set_xlabel('$q\ (\\AA^{-1})$')
    ax_right.set_ylabel('Intensity (log scale)')
    ax_right.set_xscale('log')
    ax_right.set_yscale('log')
    ax_right.grid(True, which="both", ls="-")

    all_intensities = []
    for q, I, _ in sector_integrations:
        all_intensities.append(I)

    if all_intensities:
        all_intensities_flat = np.concatenate(all_intensities)
        valid_intensities = all_intensities_flat[np.isfinite(all_intensities_flat) & (all_intensities_flat > 0)]
        if valid_intensities.size > 0:
            I_min = np.min(valid_intensities)
            I_max = np.max(valid_intensities)
            ax_right.set_ylim(I_min * 0.5, I_max * 2.0)

    for q, I, azim_range in sector_integrations:
        ax_right.loglog(q, I, label=f'{int(azim_range[0])}-{int(azim_range[1])}°')
    ax_right.legend(loc='best', fontsize='small', ncol=2)

    fig_results.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# ----------------------------------------------------------------------
# %% Interactive Click Handler
class ClickManager:
    # Force an interactive backend BEFORE importing pyplot
    try:
        matplotlib.use('Qt5Agg')
    except:
        matplotlib.use('TkAgg')

    plt.ion() # Turn on interactive mode for immediate plot updates
    def __init__(self, img_data, clim, detector_distance, wavelength, pixel_size_x, pixel_size_y, initial_fig, initial_ax, instrument):
        self.img_data = img_data
        self.clim = clim
        self.detector_distance = detector_distance
        self.wavelength = wavelength
        self.pixel_size_x = pixel_size_x
        self.pixel_size_y = pixel_size_y
        self.instrument = instrument

        self.fig = initial_fig
        self.ax = initial_ax
        self.clicks = []
        self.cid = None

        self.current_state = 'bs_collect'
        self.current_bs_count = 0

        self.bs_masks_dict = {}
        self.img_masked_bs = self.img_data.copy()
        self.center_x = None
        self.center_y = None
        self.circle_radius = None
        self.sector_integrations_data = []
        self.trans_area_coords = None

    def connect(self):
        self.clicks = []
        self.ax.clear()

        if self.current_state == 'bs_collect':
            self.fig.suptitle('SANS Setup: Step 1 - Define Beam Stops', fontsize=14, color='darkblue')
            title = f'Area {self.current_bs_count + 1}. Click 4 corners (left, right, bottom, top). CLOSE WINDOW to finish step.'
            img_to_display = self.img_masked_bs
            click_limit = 4
        elif self.current_state == 'beam_center':
            self.fig.suptitle('SANS Setup: Step 2 - Determine Beam Center', fontsize=14, color='darkgreen')
            title = 'Click 4 points on the scattering ring (left, right, bottom, top). It will move automatically to the next step.'
            img_to_display = self.img_data
            click_limit = 4
        elif self.current_state == 'transmission':
            self.fig.suptitle('SANS Setup: Step 3 - Select Transmission Area', fontsize=14, color='darkorange')
            title = 'Click 4 corners (left, right, bottom, top). This is the final selection step.'
            img_to_display = self.img_data
            click_limit = 4
        else:
            return

        self.ax.set_title(title, fontsize=10)
        self.ax.imshow(img_to_display, origin='lower', cmap='jet', clim=self.clim)
        self.ax.grid(which='major', color='w', linestyle='--', linewidth=1)
        self.fig.canvas.draw_idle()

        print(f"\n--- {title} ---")
        self.cid = self.fig.canvas.mpl_connect('button_press_event', lambda event: self.on_click(event, click_limit))

        if self.current_state == 'bs_collect' and self.current_bs_count > 0:
            print(f"You have defined {self.current_bs_count} beam stop areas. Click 4 more for Area {self.current_bs_count + 1}, or **CLOSE THE WINDOW** to finish and proceed to Step 2.")

    def on_click(self, event, click_limit):
        if event.inaxes == self.ax:
            self.clicks.append([event.xdata, event.ydata])
            print(f'Clicked: x = {np.round(event.xdata, 2)}, y = {np.round(event.ydata, 2)}')

            if len(self.clicks) == click_limit:
                self.fig.canvas.mpl_disconnect(self.cid)
                if self.current_state == 'bs_collect':
                    self.process_beamstop()
                elif self.current_state == 'beam_center':
                    self.process_beam_center()
                elif self.current_state == 'transmission':
                    self.process_transmission()

    def _calculate_area_coords(self):
        all_x = [click[0] for click in self.clicks]
        all_y = [click[1] for click in self.clicks]

        x_min_float = np.min(all_x)
        x_max_float = np.max(all_x)
        y_min_float = np.min(all_y)
        y_max_float = np.max(all_y)

        rows, cols = self.img_data.shape

        min_y = max(0, int(np.floor(y_min_float)))
        max_y = min(rows, int(np.ceil(y_max_float)))
        min_x = max(0, int(np.floor(x_min_float)))
        max_x = min(cols, int(np.ceil(x_max_float)))

        return [min_y, max_y, min_x, max_x]

    def process_beamstop(self):
        print("Processing Beamstop Coordinates...")
        coords = self._calculate_area_coords()

        self.current_bs_count += 1
        key = f'bs_{self.current_bs_count}'
        self.bs_masks_dict[key] = coords
        print(f"Stored {key}: [Ymin, Ymax, Xmin, Xmax] = {coords}")

        Ymin, Ymax, Xmin, Xmax = coords
        self.img_masked_bs[Ymin:Ymax, Xmin:Xmax] = np.nan

        self.ax.clear()
        self.ax.imshow(self.img_masked_bs, origin='lower', cmap='jet', clim=self.clim)

        rect_width = Xmax - Xmin
        rect_height = Ymax - Ymin
        mask_box = plt.Rectangle((Xmin, Ymin), rect_width, rect_height,
                                 linewidth=2, edgecolor='yellow', facecolor='none', linestyle='-')
        self.ax.add_patch(mask_box)
        self.fig.canvas.draw_idle()
        self.connect()

    def _refine_center_of_mass(self, guess_x, guess_y, guess_r, search_width=15):
        """
        Helper method: Refines the beam center by finding the intensity-weighted
        center of mass within a narrow annulus (donut) around the guessed ring.
        """
        import numpy as np

        # Create an X and Y coordinate grid for the whole image
        y_grid, x_grid = np.indices(self.img_data.shape)

        # Calculate how far every pixel is from the guessed center
        r_grid = np.sqrt((x_grid - guess_x)**2 + (y_grid - guess_y)**2)

        # Create a donut mask: only look at pixels near the guessed radius
        annulus_mask = (r_grid >= (guess_r - search_width)) & (r_grid <= (guess_r + search_width))

        # Only use valid, positive intensity pixels inside the donut
        valid_pixels = np.isfinite(self.img_data) & (self.img_data > 0) & annulus_mask

        # Calculate the Center of Mass (weighted by actual neutron counts)
        total_intensity = np.sum(self.img_data[valid_pixels])

        if total_intensity > 0:
            refined_x = np.sum(x_grid[valid_pixels] * self.img_data[valid_pixels]) / total_intensity
            refined_y = np.sum(y_grid[valid_pixels] * self.img_data[valid_pixels]) / total_intensity
            return refined_x, refined_y
        else:
            print("  [Warning] Annulus empty or invalid. Falling back to manual clicks.")
            return guess_x, guess_y

    def process_beam_center(self):
        print("Processing Beam Center Coordinates...")
        all_x_clicks = [click[0] for click in self.clicks]
        all_y_clicks = [click[1] for click in self.clicks]

        # 1. Calculate the Rough Guess from human clicks
        guess_x = np.mean(all_x_clicks)
        guess_y = np.mean(all_y_clicks)

        span_x_clicks = np.max(all_x_clicks) - np.min(all_x_clicks)
        span_y_clicks = np.max(all_y_clicks) - np.min(all_y_clicks)
        guess_r = np.mean([span_x_clicks, span_y_clicks]) / 2

        print(f'  Initial Manual Guess: X={np.round(guess_x, 2)}, Y={np.round(guess_y, 2)}, R={np.round(guess_r, 2)}')

        # 2. Apply Mathematical Refinement
        print("  Refining center using intensity-weighted annulus...")
        self.center_x, self.center_y = self._refine_center_of_mass(guess_x, guess_y, guess_r, search_width=15)
        self.circle_radius = guess_r # Keep the visual circle the same size as the clicks

        print(f'  Refined Beam Center: X={np.round(self.center_x, 2)}, Y={np.round(self.center_y, 2)}')

        # DECISION POINT based on Instrument
        if self.instrument == 'SANS-LLB':
            self.current_state = 'transmission'
            plt.close(self.fig)
        else:
            print(f"Skipping transmission step for {self.instrument}.")
            self.trans_area_coords = None
            self._finish_and_plot()

    def process_transmission(self):
        print("Processing Transmission Area Coordinates...")
        min_y, max_y, min_x, max_x = self._calculate_area_coords()
        self.trans_area_coords = [min_y, max_y, min_x, max_x]
        print(f'Transmission area coordinates (Ymin, Ymax, Xmin, Xmax): {self.trans_area_coords}')
        self._finish_and_plot()

    def _finish_and_plot(self):
        print("\n--- Final PyFAI Integration and Plotting ---")
        poni2 = self.center_x * self.pixel_size_x
        poni1 = self.center_y * self.pixel_size_y

        self.ai_final = pyFAI_ai.AzimuthalIntegrator(dist=self.detector_distance, poni1=poni1, poni2=poni2, rot1=0,
                                                     rot2=0, rot3=0, pixel1=self.pixel_size_x, pixel2=self.pixel_size_y,
                                                     splinefile=None, detector=None, wavelength=self.wavelength)
        self.ai_final.setChiDiscAtZero()

        sectors_nr = 16
        npt_azim_ranges = []
        for rr in range(0, sectors_nr):
            azim_start = rr * (360 / sectors_nr)
            azim_end = (rr + 1) * (360 / sectors_nr)
            npt_azim_ranges.append([azim_start, azim_end])

        img_for_integration = img.copy()

        for _, coords in self.bs_masks_dict.items():
            Ymin, Ymax, Xmin, Xmax = coords
            img_for_integration[Ymin:Ymax, Xmin:Xmax] = np.nan

        for azim_range in npt_azim_ranges:
            q, I, sigma = self.ai_final.integrate1d(img_for_integration, 100,
                                                    correctSolidAngle = True,
                                                    method = 'nosplit_csr',
                                                    unit = 'q_A^-1',
                                                    safe = True,
                                                    error_model = "azimuthal",
                                                    azimuth_range = azim_range,
                                                    flat = None,
                                                    dark = None)
            self.sector_integrations_data.append((q, I, azim_range))

        plt.close(self.fig)
        self.current_state = 'done'

        plot_final_results_figure(
            self.img_data,
            self.img_masked_bs,
            self.center_x,
            self.center_y,
            self.circle_radius,
            self.sector_integrations_data,
            self.clim,
            self.trans_area_coords
        )

# ==========================================
# %% AUTOMATED YAML UPDATE WITH CONFIRMATION
# ==========================================
def confirm_and_save_to_yaml(yaml_path, distance_val, manager):
    """
    Reads the YAML, compares old vs new coordinates for the specific distance,
    and asks for user confirmation before overwriting inside 'detector_geometry'.
    Preserves all comments using ruamel.yaml.
    """
    import os
    import numpy as np

    # Try to import ruamel.yaml to preserve comments
    try:
        from ruamel.yaml import YAML
    except ImportError:
        print("\n[ERROR] ruamel.yaml is not installed. Please run 'pip install ruamel.yaml'.")
        print("Falling back to standard yaml (WARNING: Comments will be lost!)")
        import yaml
        yaml_parser = 'standard'
    else:
        yaml_parser = 'ruamel'

    if not os.path.exists(yaml_path):
        print(f"\n[ERROR] '{yaml_path}' not found in the current directory.")
        return

    dist_key = float(distance_val)

    new_bs = {}
    for i, (key, coords) in enumerate(manager.bs_masks_dict.items()):
        new_bs[f'bs{i}'] = coords

    new_trans = manager.trans_area_coords
    new_center = [float(np.round(manager.center_x, 2)), float(np.round(manager.center_y, 2))]

    # Load the YAML file
    with open(yaml_path, 'r') as f:
        if yaml_parser == 'ruamel':
            ryaml = YAML()
            ryaml.preserve_quotes = True
            config = ryaml.load(f) or {}
        else:
            config = yaml.safe_load(f) or {}

    if 'detector_geometry' not in config:
        config['detector_geometry'] = {}

    for section in ['beamstopper_coordinates', 'transmission_coordinates', 'beam_center_guess']:
        if section not in config['detector_geometry']:
            config['detector_geometry'][section] = {}

    old_bs = config['detector_geometry']['beamstopper_coordinates'].get(dist_key, "Not Set")
    old_trans = config['detector_geometry']['transmission_coordinates'].get(dist_key, "Not Set")
    old_center = config['detector_geometry']['beam_center_guess'].get(dist_key, "Not Set")

    print("\n" + "="*60)
    print(f"YAML UPDATE CONFIRMATION (Detector Distance: {dist_key}m)")
    print("="*60)

    print(f"\n--- Beamstopper Coordinates ---")
    print(f"  OLD: {old_bs}")
    print(f"  NEW: {new_bs}")

    print(f"\n--- Transmission Area [Ymin, Ymax, Xmin, Xmax] ---")
    print(f"  OLD: {old_trans}")
    if new_trans is not None:
        print(f"  NEW: {new_trans}")
    else:
        print(f"  NEW: Skipped (Not required for {manager.instrument})")

    print(f"\n--- Beam Center [X, Y] ---")
    print(f"  OLD: {old_center}")
    print(f"  NEW: {new_center}\n")

    while True:
        choice = input("Overwrite these values in config_experiment.yaml? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:

            # Update the configuration dictionary
            config['detector_geometry']['beamstopper_coordinates'][dist_key] = new_bs
            if new_trans is not None:
                config['detector_geometry']['transmission_coordinates'][dist_key] = new_trans
            config['detector_geometry']['beam_center_guess'][dist_key] = new_center

            # Save it back out
            with open(yaml_path, 'w') as f:
                if yaml_parser == 'ruamel':
                    ryaml.dump(config, f)
                else:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            print("\n[SUCCESS] YAML file successfully updated with comments preserved!")
            break
        elif choice in ['n', 'no']:
            print("\n[CANCELLED] Update aborted. YAML file remains unchanged.")
            break
        else:
            print("Invalid input. Please type 'y' or 'n'.")

# ==========================================
# %% Main Execution
# ==========================================
interactive_fig, interactive_ax = plt.subplots(figsize=(8, 7))

click_manager = ClickManager(
    img_log,
    clim,
    Detector_distance,
    wl,
    pixel1,
    pixel2,
    interactive_fig,
    interactive_ax,
    instrument # Passing instrument into the class here
)

click_manager.connect()

print("\nInteractive mode active. If the window does not appear, check your Python backend settings.")

# Main loop to keep the script running while interactive figures are open
while click_manager.current_state != 'done':
    try:
        if plt.fignum_exists(interactive_fig.number):
            plt.pause(0.1)
        else:
            if click_manager.current_state == 'bs_collect':
                print("\nMoving to Step 2: Beam Center.")
                interactive_fig, interactive_ax = plt.subplots(figsize=(8, 7))
                click_manager.fig, click_manager.ax = interactive_fig, interactive_ax
                click_manager.current_state = 'beam_center'
                click_manager.connect()
            elif click_manager.current_state == 'transmission':
                 interactive_fig, interactive_ax = plt.subplots(figsize=(8, 7))
                 click_manager.fig, click_manager.ax = interactive_fig, interactive_ax
                 click_manager.connect()
            else:
                break
    except Exception as e:
        print(f"Interactive loop error: {e}")
        break

print("\nAll interactive analysis steps completed.")

# Trigger the confirmation prompt using the path defined at the top
confirm_and_save_to_yaml(yaml_filepath, Detector_distance, click_manager)
