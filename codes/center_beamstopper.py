# -*- coding: utf-8 -*-
"""
Created on Wed Dec 6 10:52:03 2023

@author: lutzbueno_v
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import load_hdf # Assuming utils.py is in the same directory or accessible
# Using the preferred import path for PyFAI to avoid deprecation warnings
import pyFAI.azimuthalIntegrator as pyFAI_ai
import pyFAI

plt.ion() # Turn on interactive mode for immediate plot updates

# %% Configuration (Assuming these variables are correctly loaded/defined)
# where are the hdf files saved
path_hdf_raw = "C:/Users/lutzbueno_v/Documents/Analysis/data/SANS-LLB/2024_SANS-LLB/DarePy-SANS/raw_data/"
# number of the AgBE scan
scanNr = 1328
instrument = 'SANS-LLB'

# for SANS-1
if instrument == 'SANS-I':
    name_hdf = 'sans2023n0' + str(scanNr) + '.hdf'
    # Pixel size
    pixel1 = 7.5e-3 # mm
    pixel2 = 7.5e-3 # mm
elif instrument == 'SANS-LLB':
    # Note: Using f-string for better formatting, assuming '00' is intended
    name_hdf = f'sans-llb2025n00{scanNr}.hdf'
    # Pixel size
    pixel1 = 5e-3 # mm
    pixel2 = 5e-3 # mm


print(f"Attempting to load: {name_hdf}")

# %% Load Data (Assuming load_hdf is defined and works)
plt.close('all')
try:
    img = load_hdf(path_hdf_raw, name_hdf, 'counts')
    Detector_distance = load_hdf(path_hdf_raw, name_hdf, 'detx')
    wl = load_hdf(path_hdf_raw, name_hdf, 'wl')
except Exception as e:
    print(f"Error loading HDF data: {e}")
    # Placeholder data for testing if necessary, but keep original intent
    # raise e

# Get detector dimensions
try:
    detector_rows, detector_cols = img.shape
    # RuntimeWarning occurs here for log(0) or log(negative). This is expected.
    img_log = np.log(img) # In log scale
    #img_log = img

except:
    frames, detector_rows, detector_cols = img.shape
    img_mean = np.mean(img, axis=0)
    img_log = np.log(img_mean) # In log scale
    img = img_mean # Use mean image for integration

# %% Automate Color bar limits for image display (clim)
# Filter out non-finite values (NaN, inf, -inf) which result from log(0)
finite_log_values = img_log[np.isfinite(img_log)]

if finite_log_values.size > 0:
    p_low = 1
    p_high = 99.5

    auto_clim_min = np.percentile(finite_log_values, p_low)
    auto_clim_max = np.percentile(finite_log_values, p_high)

    clim = [max(0.0, auto_clim_min), auto_clim_max]
    print(f"Automated clim set to: {clim}")
else:
    print("Warning: No finite log values found to auto-determine clim. Using fallback [0, 7].")
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
    # Create a new figure with 1 row and 3 columns
    fig_results, axes_results = plt.subplots(1, 3, figsize=(20, 7))
    fig_results.suptitle('SANS Beam Center Analysis Results', fontsize=16)

    # Left Column: Masked beamstop overlayed with SANS pattern
    ax_left = axes_results[0]
    ax_left.imshow(masked_img_bs, origin='lower', cmap='jet', clim=clim_limits)
    ax_left.set_title('Masked Beamstop & Transmission Area')
    ax_left.set_xlabel('Pixels (X)')
    ax_left.set_ylabel('Pixels (Y)')
    ax_left.grid(True)

    # Highlight the transmission area on the left plot
    if trans_area_coords:
        # coords are [Ymin, Ymax, Xmin, Xmax]
        Ymin, Ymax, Xmin, Xmax = trans_area_coords
        rect_width = Xmax - Xmin
        rect_height = Ymax - Ymin

        # Create a rectangle patch
        transmission_box = plt.Rectangle((Xmin, Ymin), rect_width, rect_height,
                                         linewidth=2, edgecolor='red', facecolor='none', linestyle='-')
        ax_left.add_patch(transmission_box)
        ax_left.text(Xmin, Ymax, 'Trans. Area', color='red', fontsize=10, verticalalignment='bottom')


    # Middle Column: SANS pattern with beam center defined as a circle with an x
    ax_middle = axes_results[1]
    ax_middle.imshow(original_img_log, origin='lower', cmap='jet', clim=clim_limits)
    ax_middle.set_title(f'Beam Center: X={np.round(center_x,2)}, Y={np.round(center_y,2)}')
    ax_middle.set_xlabel('Pixels (X)')
    ax_middle.set_ylabel('Pixels (Y)')
    ax_middle.grid(True)
    # Plot beam center as larger 'x' in white
    ax_middle.plot(center_x, center_y, 'wx', markersize=15, mew=2, label='Beam Center')
    # Plot a circle centered on it with the calculated radius in white
    circle = plt.Circle((center_x, center_y), circle_radius, color='white', fill=False, linestyle='--', linewidth=2)
    ax_middle.add_patch(circle)
    ax_middle.legend()

    # Right Column: Data integrated in sectors
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
        else:
            print("Warning: No valid positive intensities found for Y-axis scaling.")


    for q, I, azim_range in sector_integrations:
        ax_right.loglog(q, I, label=f'{int(azim_range[0])}-{int(azim_range[1])}°')
    ax_right.legend(loc='best', fontsize='small', ncol=2)

    fig_results.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# ----------------------------------------------------------------------


# %% Interactive Click Handler
class ClickManager:
    def __init__(self, img_data, clim, detector_distance, wavelength, pixel_size_x, pixel_size_y, initial_fig, initial_ax):
        self.img_data = img_data
        self.clim = clim
        self.detector_distance = detector_distance
        self.wavelength = wavelength
        self.pixel_size_x = pixel_size_x
        self.pixel_size_y = pixel_size_y

        self.fig = initial_fig
        self.ax = initial_ax
        self.clicks = []
        self.cid = None

        # --- NEW STATE MANAGEMENT ---
        self.current_state = 'bs_collect'
        self.current_bs_count = 0
        # ----------------------------

        # Data to be stored
        self.bs_masks_dict = {}
        self.img_masked_bs = self.img_data.copy()
        self.center_x = None
        self.center_y = None
        self.circle_radius = None
        self.sector_integrations_data = []
        self.trans_area_coords = None


    def connect(self):
        """Sets up the plot and connects the click handler based on the current state."""
        self.clicks = [] # Reset clicks for each area selection
        self.ax.clear()

        # Determine the image and titles based on the state
        if self.current_state == 'bs_collect':
            # Set the figure super title (the friendly title)
            self.fig.suptitle('SANS Setup: Step 1 - Define Beam Stops', fontsize=14, color='darkblue')
            # Set the axis title (the instruction)
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
            print("Error: Invalid state.")
            return

        # Set the axis title and display the image
        self.ax.set_title(title, fontsize=10)
        self.ax.imshow(img_to_display, origin='lower', cmap='jet', clim=self.clim)
        self.ax.grid(which='major', color='w', linestyle='--', linewidth=1)
        self.fig.canvas.draw_idle()

        # Print detailed instructions to the console
        print(f"\n--- {title} ---")

        # Use lambda function to pass click_limit to on_click
        self.cid = self.fig.canvas.mpl_connect('button_press_event', lambda event: self.on_click(event, click_limit))

        if self.current_state == 'bs_collect' and self.current_bs_count > 0:
            print(f"You have defined {self.current_bs_count} beam stop areas. Click 4 more for Area {self.current_bs_count + 1}, or **CLOSE THE WINDOW** to finish and proceed to Step 2.")


    def on_click(self, event, click_limit):
        if event.inaxes == self.ax:
            self.clicks.append([event.xdata, event.ydata])
            print(f'Clicked: x = {np.round(event.xdata, 2)}, y = {np.round(event.ydata, 2)}')

            if len(self.clicks) == click_limit:
                # Disconnect the click handler after 4 clicks
                self.fig.canvas.mpl_disconnect(self.cid)
                if self.current_state == 'bs_collect':
                    self.process_beamstop()
                elif self.current_state == 'beam_center':
                    self.process_beam_center()
                elif self.current_state == 'transmission':
                    self.process_transmission()

    def _calculate_area_coords(self):
        """Helper to calculate min/max integer indices from 4 clicks (Ymin, Ymax, Xmin, Xmax)."""

        # Use the four clicks collected in self.clicks
        all_x = [click[0] for click in self.clicks]
        all_y = [click[1] for click in self.clicks]

        # Determine min/max coordinates
        x_min_float = np.min(all_x)
        x_max_float = np.max(all_x)
        y_min_float = np.min(all_y)
        y_max_float = np.max(all_y)

        # Convert to integer indices, ensuring they are within bounds [0, max_dim]
        # X: cols, Y: rows
        rows, cols = self.img_data.shape

        min_y = max(0, int(np.floor(y_min_float)))
        max_y = min(rows, int(np.ceil(y_max_float)))
        min_x = max(0, int(np.floor(x_min_float)))
        max_x = min(cols, int(np.ceil(x_max_float)))

        # Return [Ymin, Ymax, Xmin, Xmax]
        return [min_y, max_y, min_x, max_x]

    def process_beamstop(self):
        """Collects one beam stop area, applies the mask, and prepares for the next."""
        print("Processing Beamstop Coordinates...")

        # Calculate mask coordinates [Ymin, Ymax, Xmin, Xmax]
        coords = self._calculate_area_coords()

        # Store in dictionary
        self.current_bs_count += 1
        key = f'bs_{self.current_bs_count}'
        self.bs_masks_dict[key] = coords

        print(f"Stored {key}: [Ymin, Ymax, Xmin, Xmax] = {coords}")

        # Apply mask to the image copy
        Ymin, Ymax, Xmin, Xmax = coords
        # Use np.nan for log-scale masking for better visual effect and PyFAI handling
        self.img_masked_bs[Ymin:Ymax, Xmin:Xmax] = np.nan

        # Clear the figure and show the updated image with the new mask
        self.ax.clear()

        # Redraw the image with the new mask applied
        self.ax.imshow(self.img_masked_bs, origin='lower', cmap='jet', clim=self.clim)

        # Highlight the new mask area
        rect_width = Xmax - Xmin
        rect_height = Ymax - Ymin
        mask_box = plt.Rectangle((Xmin, Ymin), rect_width, rect_height,
                                 linewidth=2, edgecolor='yellow', facecolor='none', linestyle='-')
        self.ax.add_patch(mask_box)

        self.fig.canvas.draw_idle()

        # Reconnect for the next beam stop
        self.connect()


    def process_beam_center(self):
        print("Processing Beam Center Coordinates...")

        # Robust beam center calculation: centroid of all four clicks
        all_x_clicks = [click[0] for click in self.clicks]
        all_y_clicks = [click[1] for click in self.clicks]

        self.center_x = np.mean(all_x_clicks)
        self.center_y = np.mean(all_y_clicks)

        print(f'The beamcenter along X is: {np.round(self.center_x, 2)}')
        print(f'The beamcenter along Y is: {np.round(self.center_y, 2)}')

        # Calculate radius for the circle on the beam center plot
        span_x_clicks = np.max(all_x_clicks) - np.min(all_x_clicks)
        span_y_clicks = np.max(all_y_clicks) - np.min(all_y_clicks)
        self.circle_radius = np.mean([span_x_clicks, span_y_clicks]) / 2

        self.current_state = 'transmission'

        # Close the current figure to trigger the next step's figure creation in the main loop
        plt.close(self.fig)


    def process_transmission(self):
        print("Processing Transmission Area Coordinates...")

        # Calculate transmission area coordinates [Ymin, Ymax, Xmin, Xmax]
        min_y, max_y, min_x, max_x = self._calculate_area_coords()
        self.trans_area_coords = [min_y, max_y, min_x, max_x]

        print(f'Transmission area coordinates (Ymin, Ymax, Xmin, Xmax): {self.trans_area_coords}')

        # Now that all interactive data is collected, proceed with integration and plotting
        self._finish_and_plot()


    def _finish_and_plot(self):
        print("\n--- Final PyFAI Integration and Plotting ---")

        # Convert pixel coordinates to poni (position of normal incidence) in mm
        poni2 = self.center_x * self.pixel_size_x # X coordinate in mm
        poni1 = self.center_y * self.pixel_size_y # Y coordinate in mm

        # PyFAI Azimuthal Integrator setup
        self.ai_final = pyFAI_ai.AzimuthalIntegrator(dist=self.detector_distance, poni1=poni1, poni2=poni2, rot1=0,
                                                     rot2=0, rot3=0, pixel1=self.pixel_size_x, pixel2=self.pixel_size_y,
                                                     splineFile=None, detector=None, wavelength=self.wavelength)
        self.ai_final.setChiDiscAtZero()

        # Define the number of sectors for azimuthal integration
        sectors_nr = 16
        npt_azim_ranges = []
        for rr in range(0, sectors_nr):
            azim_start = rr * (360 / sectors_nr)
            azim_end = (rr + 1) * (360 / sectors_nr)
            npt_azim_ranges.append([azim_start, azim_end])

        # Perform 1D integration for each sector
        img_for_integration = img.copy() # Use the original counts image (not log) for integration

        # Apply ALL beam stop masks for integration
        for _, coords in self.bs_masks_dict.items():
            Ymin, Ymax, Xmin, Xmax = coords
            # Mask out beamstop for integration using a NaN mask (pyFAI handles NaN)
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

        # Close the interactive figure
        plt.close(self.fig)
        self.current_state = 'done' # Set final state to terminate the main loop

        # Call the external function to plot the final results
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

        # --- Print all coordinates in the EXACT requested dictionary format ---
        output_bs_dict = {}
        for i, (key, coords) in enumerate(self.bs_masks_dict.items()):
            # Use i as the index for the 'bsX' keys
            output_bs_dict[f'bs{i}'] = coords

        print('\n_______________COORDINATES________________')
        print('__________________________________________')
        # Print the beam stop coordinates as a single dictionary assignment
        print(f'beamstopper_coordinates => {self.detector_distance}:{output_bs_dict}')

        # Print transmission and beam center coordinates using the existing format
        print(f'transmission_coordinates => {self.detector_distance}: [{self.trans_area_coords[0]}, {self.trans_area_coords[1]}, {self.trans_area_coords[2]}, {self.trans_area_coords[3]}]')
        print(f'beam_center_guess => {self.detector_distance}: [{np.round(self.center_x, 2)}, {np.round(self.center_y, 2)}]')
        print('__________________________________________')
        print('__________________________________________')
        # -----------------------------------------------------------

# %% Main Execution
# Create the initial single figure for interactive clicks
interactive_fig, interactive_ax = plt.subplots(figsize=(8, 7))

# Initialize the ClickManager
click_manager = ClickManager(
    img_log, # Pass log image for visual clicks
    clim,
    Detector_distance,
    wl,
    pixel1,
    pixel2,
    interactive_fig,
    interactive_ax
)

# Start the interactive process
click_manager.connect()

print("\nInteractive mode active. Please follow the instructions in the figure title and console.")

# The main loop monitors the figure and manages state transitions when the figure is closed.
while plt.fignum_exists(interactive_fig.number) or click_manager.current_state != 'done':
    plt.pause(0.1)

    # --- State Transition Logic: Beam Stop Collection -> Beam Center ---
    # Triggered when the user closes the figure in the 'bs_collect' state.
    if click_manager.current_state == 'bs_collect' and not plt.fignum_exists(interactive_fig.number):
        if not click_manager.bs_masks_dict:
            print("\nWARNING: No beam stop area was defined. The image will be used without a mask.")

        print("\nBeam stop collection finished (Figure closed). Moving to Step 2: Beam Center selection.")

        # 1. Re-create the figure for the next step
        interactive_fig, interactive_ax = plt.subplots(figsize=(8, 7))
        click_manager.fig = interactive_fig
        click_manager.ax = interactive_ax

        # 2. Transition state and connect new click handler
        click_manager.current_state = 'beam_center'
        click_manager.connect()

    # --- State Transition Logic: Beam Center -> Transmission ---
    # Triggered when the user completes 4 clicks in 'beam_center' and process_beam_center() closes the figure.
    elif click_manager.current_state == 'transmission' and not plt.fignum_exists(interactive_fig.number):
        # We need to explicitly check if we have received the 4 clicks before proceeding,
        # but since process_beam_center() sets the state and closes the window,
        # reaching this point means we are ready for the next step.
        print("\nBeam center defined (Figure closed). Moving to Step 3: Transmission Area selection.")

        # 1. Re-create the figure for the next step
        interactive_fig, interactive_ax = plt.subplots(figsize=(8, 7))
        click_manager.fig = interactive_fig
        click_manager.ax = interactive_ax

        # 2. Re-connect to start the transmission step
        click_manager.connect()

    # Exit condition for the main loop
    if click_manager.current_state == 'done':
        break

print("\nAll interactive analysis steps completed.")
