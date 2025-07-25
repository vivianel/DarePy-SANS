# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 10:52:03 2023

@author: lutzbueno_v
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import load_hdf # Assuming utils.py is in the same directory or accessible
import pyFAI

plt.ion() # Turn on interactive mode for immediate plot updates

# %% Configuration
# where are the hdf files saved
path_hdf_raw = 'C:/Users/lutzbueno_v/Documents/Analysis/DarePy-SANS/raw_data'
# number of the AgBE scan
scanNr = 23111

# As requested, keep the name_hdf fixed with 'sans2025n0' + scanNr
name_hdf = 'sans2025n0' + str(scanNr) + '.hdf'

print(f"Attempting to load: {name_hdf}") # Added for debugging/clarity


# Pixel size (common for many SANS detectors)
pixel1 = 7.5e-3 # mm
pixel2 = 7.5e-3 # mm

# %% Load Data
img = load_hdf(path_hdf_raw, name_hdf, 'counts')
Detector_distance = load_hdf(path_hdf_raw, name_hdf, 'detx')
wl = load_hdf(path_hdf_raw, name_hdf, 'wl')
img_log = np.log(img) # In log scale

# Get detector dimensions (still useful for general info)
detector_rows, detector_cols = img.shape

# %% Automate Color bar limits for image display (clim)
# Filter out non-finite values (NaN, inf, -inf) which can result from log(0) or other issues
finite_log_values = img_log[np.isfinite(img_log)]

if finite_log_values.size > 0:
    # Use percentiles for robust min/max. Adjust as needed.
    # 1st percentile to capture most low-intensity signal
    p_low = 1
    # 99.5th percentile to capture high-intensity signal but avoid single saturated pixels
    p_high = 99.5

    auto_clim_min = np.percentile(finite_log_values, p_low)
    auto_clim_max = np.percentile(finite_log_values, p_high)

    # Optionally, force the minimum to be non-negative if desired, similar to original [0, 7]
    # This clips log(counts) values that are negative (i.e., counts between 0 and 1) to 0.
    clim = [max(0.0, auto_clim_min), auto_clim_max] # Use 0.0 to ensure float type
    print(f"Automated clim set to: {clim}")
else:
    # Fallback if no finite log values (e.g., empty image or all invalid)
    print("Warning: No finite log values found to auto-determine clim. Using fallback [0, 7].")
    clim = [0, 7]


# %% Interactive Click Handler
class ClickManager:
    def __init__(self, img_data, clim, detector_distance, wavelength, pixel_size_x, pixel_size_y, initial_fig, initial_ax):
        self.img_data = img_data # Original log image
        self.clim = clim
        self.detector_distance = detector_distance
        self.wavelength = wavelength
        self.pixel_size_x = pixel_size_x
        self.pixel_size_y = pixel_size_y

        self.fig = initial_fig # The single figure for interactive clicks
        self.ax = initial_ax # The single axis for interactive clicks
        self.clicks = []
        self.cid = None
        self.current_step = 0 # 0 for beamstop, 1 for beam center

        # Data to be stored for final plots
        self.bs_mask = None # Store beamstop mask for printing later
        self.img_masked_bs = None
        self.center_x = None
        self.center_y = None
        self.circle_radius = None # Store the calculated radius for plotting
        self.sector_integrations_data = [] # List of (q, I) tuples for each sector

    def connect(self, title):
        self.clicks = [] # Reset clicks for each step for the new phase
        self.ax.clear()
        self.ax.imshow(self.img_data, origin='lower', cmap='jet', clim=self.clim)
        self.ax.set_title(title)
        self.ax.grid(which='major', color='w', linestyle='--', linewidth=1)
        self.fig.canvas.draw_idle()
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        print(title)

    def on_click(self, event):
        if event.inaxes == self.ax: # Ensure click is within our active axis
            self.clicks.append([event.xdata, event.ydata])
            print(f'Clicked: x = {np.round(event.xdata, 2)}, y = {np.round(event.ydata, 2)}')
            if len(self.clicks) == 4:
                self.fig.canvas.mpl_disconnect(self.cid) # Disconnect clicks after 4
                if self.current_step == 0:
                    self.process_beamstop()
                elif self.current_step == 1:
                    self.process_beam_center()
                # Do not increment self.current_step here, it's handled by process functions
                # to allow sequential calling

    def process_beamstop(self):
        print("\nProcessing Beamstop Coordinates...")
        # Get coordinates from the clicks (assuming left, right, bottom, top order)
        # Note: These are specific for beamstopper dimensions, not beam center
        x_left_bs, _ = self.clicks[0]
        x_right_bs, _ = self.clicks[1]
        _, y_bottom_bs = self.clicks[2]
        _, y_top_bs = self.clicks[3]

        # Calculate sizes based on the specified clicks
        size_x = np.abs(x_right_bs - x_left_bs) # Simple horizontal distance
        print(f'The horizontal beamstopper size is: x = {np.round(size_x, 2)} pixels')

        size_y = np.abs(y_top_bs - y_bottom_bs) # Simple vertical distance
        print(f'The vertical beamstopper size is: y = {np.round(size_y, 2)} pixels')

        # Calculate beamstopper mask coordinates
        min_y = max(0, int(np.floor(y_bottom_bs)))
        max_y = min(self.img_data.shape[0], int(np.ceil(y_top_bs)))
        min_x = max(0, int(np.floor(x_left_bs)))
        max_x = min(self.img_data.shape[1], int(np.ceil(x_right_bs)))

        self.bs_mask = [min_y, max_y, min_x, max_x]

        # Apply mask for display on the same interactive figure
        self.img_masked_bs = self.img_data.copy()
        self.img_masked_bs[self.bs_mask[0]:self.bs_mask[1], self.bs_mask[2]:self.bs_mask[3]] = 0 # Mask out beamstop


        self.ax.clear() # Clear the single axis
        self.ax.imshow(self.img_masked_bs, origin='lower', cmap='jet', clim=self.clim)
        self.ax.set_title('Beamstopper Mask Applied (Click on Ring for Beam Center)')
        self.ax.grid(which='major', color='w', linestyle='--', linewidth=1)
        self.fig.canvas.draw_idle() # Redraw the single figure

        # Do NOT print beamstopper coordinates here, will print all together later
        self.current_step = 1 # Set step to beam center
        self.connect('Click on the ring: left, right, bottom and top to define the beam center')


    def process_beam_center(self):
        print("\nProcessing Beam Center Coordinates...")

        # Robust beam center calculation: centroid of all four clicks
        all_x_clicks = [click[0] for click in self.clicks]
        all_y_clicks = [click[1] for click in self.clicks]

        self.center_x = np.mean(all_x_clicks)
        self.center_y = np.mean(all_y_clicks)

        print(f'The beamcenter along X is: {np.round(self.center_x, 2)}')
        print(f'The beamcenter along Y is: {np.round(self.center_y, 2)}')

        # Calculate radius for the circle on the beam center plot
        # Use half of the average of the horizontal and vertical spans of the clicked points
        span_x_clicks = np.max(all_x_clicks) - np.min(all_x_clicks)
        span_y_clicks = np.max(all_y_clicks) - np.min(all_y_clicks)
        self.circle_radius = np.mean([span_x_clicks, span_y_clicks]) / 2


        # Convert pixel coordinates to poni (position of normal incidence) in mm
        poni2 = self.center_x * self.pixel_size_x # X coordinate in mm
        poni1 = self.center_y * self.pixel_size_y # Y coordinate in mm

        # Create the radial integrator
        self.ai_final = pyFAI.AzimuthalIntegrator(dist=self.detector_distance, poni1=poni1, poni2=poni2, rot1=0,
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
        img_for_integration = self.img_data.copy()
        if self.bs_mask:
            # Mask out beamstop for integration using a NaN mask (pyFAI handles NaN)
            img_for_integration[self.bs_mask[0]:self.bs_mask[1], self.bs_mask[2]:self.bs_mask[3]] = np.nan

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



        # All data collected, close the interactive figure
        plt.close(self.fig)

        # Call the external function to plot the final results
        plot_final_results_figure(
            self.img_data,
            self.img_masked_bs,
            self.center_x,
            self.center_y,
            self.circle_radius, # Pass the calculated radius
            self.sector_integrations_data,
            self.clim # Pass the auto-calculated clim
        )

        # --- Print all coordinates together after all calculations ---
        print('\n_______________COORDINATES________________')
        print('__________________________________________')
        print(f'beamstopper_coordinates -> {self.detector_distance}:[{self.bs_mask[0]}, {self.bs_mask[1]}, {self.bs_mask[2]}, {self.bs_mask[3]}]')
        print(f'beam_center_guess -> {self.detector_distance}:[{np.round(self.center_x, 2)}, {np.round(self.center_y, 2)}]')
        print('__________________________________________')
        print('__________________________________________')
        # -----------------------------------------------------------

# %% Function to plot the final results in a new figure
def plot_final_results_figure(
    original_img_log,
    masked_img_bs,
    center_x,
    center_y,
    circle_radius, # New parameter for the circle radius
    sector_integrations,
    clim_limits
):
    # Create a new figure with 1 row and 3 columns
    fig_results, axes_results = plt.subplots(1, 3, figsize=(20, 7))
    fig_results.suptitle('SANS Beam Center Analysis Results', fontsize=16)

    # Left Column: Masked beamstop overlayed with SANS pattern
    ax_left = axes_results[0]
    ax_left.imshow(masked_img_bs, origin='lower', cmap='jet', clim=clim_limits)
    ax_left.set_title('Masked Beamstop & SANS Pattern')
    ax_left.set_xlabel('Pixels (X)')
    ax_left.set_ylabel('Pixels (Y)')
    ax_left.grid(True)

    # Middle Column: SANS pattern with beam center defined as a circle with an x
    ax_middle = axes_results[1]
    ax_middle.imshow(original_img_log, origin='lower', cmap='jet', clim=clim_limits)
    ax_middle.set_title(f'Beam Center: X={np.round(center_x,2)}, Y={np.round(center_y,2)}')
    ax_middle.set_xlabel('Pixels (X)')
    ax_middle.set_ylabel('Pixels (Y)')
    ax_middle.grid(True)
    # Plot beam center as larger 'x' in white
    ax_middle.plot(center_x, center_y, 'wx', markersize=15, mew=2, label='Beam Center') # 'wx' for white x
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

    # Collect all valid intensities to determine focused Y-axis limits
    all_intensities = []
    for q, I, _ in sector_integrations:
        all_intensities.append(I)

    if all_intensities:
        # Concatenate and filter for positive, finite values
        all_intensities_flat = np.concatenate(all_intensities)
        valid_intensities = all_intensities_flat[np.isfinite(all_intensities_flat) & (all_intensities_flat > 0)]

        if valid_intensities.size > 0:
            I_min = np.min(valid_intensities)
            I_max = np.max(valid_intensities)
            # Set Y-axis limits with a small buffer on log scale
            ax_right.set_ylim(I_min * 0.5, I_max * 2.0) # Adjust buffer as needed (e.g., 0.1 and 10 for wider)
        else:
            print("Warning: No valid positive intensities found for Y-axis scaling.")


    for q, I, azim_range in sector_integrations:
        ax_right.loglog(q, I, label=f'{int(azim_range[0])}-{int(azim_range[1])}°')
    ax_right.legend(loc='best', fontsize='small', ncol=2)

    fig_results.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
    plt.show() # Display the final results figure

# %% Main Execution
# Create the initial single figure for interactive clicks
interactive_fig, interactive_ax = plt.subplots(figsize=(8, 7))

# Initialize the ClickManager with the single interactive figure/axis
click_manager = ClickManager(
    img_log,
    clim, # Pass the auto-calculated clim
    Detector_distance,
    wl,
    pixel1,
    pixel2,
    interactive_fig,
    interactive_ax
)

# Start by setting up for beamstop clicks on the single figure
click_manager.connect('Click on the beamstopper edges: left, right, bottom and top')

# The script will pause here until the interactive_fig is closed, then plot_final_results_figure takes over.
