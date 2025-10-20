import os
import numpy as np
import matplotlib.pyplot as plt
from utils import create_analysis_folder # Used for getting analysis folder path
import integration as integ # Used for make_file_name
from scipy import optimize
import re # Used for extracting scan and frame numbers

"""
This module contains functions for generating 2D and 1D plots of integrated
Small-Angle Neutron Scattering (SANS) data. It visualizes the raw 2D scattering
patterns, radial intensity profiles, and azimuthal anisotropy information.
"""

def plot_integ_radial(config, result, ScanNr, Frame, img_2D, data_azimuth):
    """
    Generates a multi-panel figure displaying various aspects of radial integration:
    - 2D scattering pattern (linear and logarithmic scale).
    - 1D radial intensity profile (I vs q).
    - 2D cake plot (intensity vs q and azimuthal angle).
    - Radial intensity profiles for different azimuthal sectors.
    - Azimuthal intensity profile (sum of intensity vs azimuthal angle) with a cosine fit.

    The generated figure is saved to the detector-specific figures folder.

    Args:
        config (dict): The main configuration dictionary, containing analysis
                       and instrument parameters (e.g., 'show_plots', 'detector_size', etc.).
        result (dict): The results dictionary, containing overview data,
                       integration setup (ai, mask, beam center), and integration results.
        ScanNr (int): The scan number of the measurement to plot.
        Frame (int): The frame number of the measurement (for multi-frame data).
    """
    path_analysis = create_analysis_folder(config)
    class_all = result['overview']['all_files']

    # Find the detector distance and sample name for the given scan number
    det_float = None
    sample_name = "N/A"
    attenuator_setting = -1 # Default
    try:
        # Iterate through the list to find the matching scan number
        idx = -1
        for i, scan in enumerate(class_all['scan']):
            if scan == ScanNr:
                idx = i
                break

        if idx != -1:
            det_float = class_all['detx_m'][idx] # Detector distance in meters (float)
            sample_name = class_all['sample_name'][idx]
            attenuator_setting = class_all['att'][idx]
        else:
            print(f"Warning: Scan number {ScanNr} not found in 'all_files' overview. Cannot retrieve detector distance or sample info for plotting.")
            return # Exit if scan info is missing
    except KeyError as e:
        print(f"Error accessing overview data for plotting (missing key: {e}). Skipping radial plot for Scan {ScanNr}.")
        return
    except IndexError:
        print(f"Error: Scan number {ScanNr} index out of bounds in 'all_files'. Skipping radial plot.")
        return

    if det_float is None:
        print(f"Warning: Detector distance not found for Scan {ScanNr}. Skipping radial plot.")
        return

    det_str = str(det_float).replace('.', 'p') # Convert float to string format like '1p6' for paths

    # Define folder to load the radial and azimuthal integration files
    path_integ = os.path.join(path_analysis, f'det_{det_str}', 'integration/')

    # Create folder to save the figures from radial integration if plotting is enabled
    # The condition `config['analysis']['plot_radial'] == 1 or config['analysis']['plot_azimuthal'] == 1`
    # means this folder is created if *any* plotting is enabled.
    path_figures = os.path.join(path_analysis, f'det_{det_str}', 'figures/')
    if not os.path.exists(path_figures):
        try:
            os.mkdir(path_figures)
            print(f"Created figures folder: {path_figures}")
        except OSError as e:
            print(f"Error creating figures folder {path_figures}: {e}. Skipping plots for Scan {ScanNr}.")
            return

    # %% Load the files from previous integration steps
    # Load the 2D pattern data
    img1=img_2D

    # Load the radial integration data (q, I, sigma)
    prefix_radial = 'radial_integ'
    sufix_dat = 'dat'
    file_name_radial = integ.make_file_name(path_integ, prefix_radial, sufix_dat, sample_name, det_str, ScanNr, Frame)
    try:
        # Usecols 0 for q, 1 for I, 2 for sigma
        q = np.genfromtxt(file_name_radial, delimiter=',', usecols=0, comments='#')
        I = np.genfromtxt(file_name_radial, delimiter=',', usecols=1, comments='#')
        sigma = np.genfromtxt(file_name_radial, delimiter=',', usecols=2, comments='#')
    except (IOError, ValueError) as e:
        print(f"Error loading radial integrated data from {file_name_radial}: {e}. Skipping radial plot for Scan {ScanNr}.")
        return
    if q.size == 0 or I.size == 0:
        print(f"Warning: Radial integrated data for Scan {ScanNr}, Frame {Frame} is empty. Skipping radial plot.")
        return

    # Load parameters for azimuthal plotting from result dictionary
    sectors_nr = result['integration'].get('sectors_nr')
    pixel_range_azim = result['integration'].get('pixel_range_azim')

    if sectors_nr is None or pixel_range_azim is None:
        print(f"Warning: Azimuthal integration parameters (sectors_nr, pixel_range_azim) not found in result. Azimuthal parts of plot will be skipped.")
        # We can still proceed with radial parts if needed, but flag this.
        plot_azim_parts = False
    else:
        plot_azim_parts = True
        # npt_azim defines the start angles of the sectors for plotting labels
        # np.linspace is used in integration.py, but range is used here. For plotting angles, range is fine for integer steps.
        npt_azim = np.linspace(0, 360, sectors_nr + 1) # npt_azim = range(0, 360, int(360/sectors_nr))


    # Load azimuthal integration data
    try:
        data_azim = data_azimuth
        # First column is q, subsequent columns are I for each sector
        q_a = data_azim[:,0]
        # I_a contains intensity for each sector. Columns 1 to sectors_nr.
        # sigma columns are in the second half of the data if present.
        I_a = data_azim[:, 1 : sectors_nr + 1] # (Corrected slice for sectors_nr columns)
        # Note: Original code calculated sigma_a by splitting `data.shape[1]-1)/2`, but azimuthal_integ saves q, I_all, sigma_all
        # So sigma_a would be data_azim[:, sectors_nr + 1:]
    except (IOError, ValueError) as e:
        print(f"Warning: Error loading azimuthal integrated data from {file_name_azim}: {e}. Azimuthal parts of plot will be skipped.")
        plot_azim_parts = False # Disable azimuthal plotting if data cannot be loaded
    if I_a is None or I_a.size == 0:
        print(f"Warning: Azimuthal integrated data for Scan {ScanNr}, Frame {Frame} is empty. Azimuthal parts of plot will be skipped.")
        plot_azim_parts = False


    # %% Control plotting display (ioff/ion)
    # This function is called for each plot, so we manage its interactive state locally.
    # The overall `plt.ion()`/`plt.ioff()` is managed in `integration.py`
    plt.ioff() # Turn off interactive plot display

    # Retrieve pyFAI integrator for detector parameters (beam center, pixel size, wavelength)
    ai = result['integration'].get('ai')
    if ai is None:
        print(f"Error: pyFAI AzimuthalIntegrator not found in 'result' for Scan {ScanNr}. Cannot calculate q-extent for 2D plots.")
        # Proceed with plotting without q-extent or exit if critical
        return

    bc_x = result['integration']['beam_center_x']
    bc_y = result['integration']['beam_center_y']
    mask = result['integration']['int_mask']
    integration_points = result['integration']['integration_points']

    # %% Define the figure axis (3 rows, 2 columns)
    fig1, ([[axs0, axs1], [axs2, axs3], [axs4, axs5]])  = plt.subplots(3, 2,  figsize=(12, 17))

    # Invert the mask for plotting (1 means masked, 0 means unmasked, but imshow uses values)
    # The mask contains 1 for beamstop/bad regions and 0 for valid.
    # We want to show valid regions and hide bad ones.
    # mask_inv will be 0 for masked, 1 for valid regions.
    mask_bool = mask.astype(bool) # Convert to boolean mask
    # For plotting, where mask_inv is 1 for valid regions and 0 for beamstop/bad areas.
    # This allows multiplying img1 by mask_inv to zero out masked regions.

    if attenuator_setting == 0: # If attenuator is 0, it means no attenuator (direct beam), so beamstop might be present
        # Assume beamstop is present if attenuator is 0.
        # This means the mask should be applied for visualization.
        # We also need to set masked areas to NaN for proper cmap.set_bad behavior.
        pass # img1 is already modified below for masked areas
    else: # If attenuator > 0, it means an attenuator is in, possibly no beamstop (if it was removed)
        # Based on original logic, if attenuator > 0, the beamstop might be out.
        # However, it's safer to always apply the int_mask unless specific reasons.
        # The original code's logic here was simplified, so let's stick to the consistent mask.
        pass # The mask `int_mask` should already include beamstop if present, regardless of attenuator.

    # Set masked areas in img1 to NaN for consistent plotting with cmap.set_bad
    img1_plot = img1.copy()
    img1_plot[mask_bool] = np.nan # Set masked regions to NaN
    
    img1_plot[img1_plot == 0] = np.nan # Set zeros to nan to improve plots

    # Define the extent of the image in q-space (A^-1)
    # x2q function converts pixel coordinates (relative to beam center) to q-values
    def x2q(x_pixels, wl_A, dist_m, pixelsize_m):
        """Converts pixel coordinate offset from beam center to q-value."""
        # Wavelength in Angstroms, distance in meters, pixelsize in meters
        # Convert dist to Angstroms for consistent units: 1m = 1e10 A
        # theta = arctan(pixelsize_m * x_pixels / dist_m)
        # q = (4 * pi / wavelength_A) * sin(theta / 2)
        dist_A = dist_m * 1e10
        theta = np.arctan((pixelsize_m / 1e-10) * x_pixels / dist_A) # All in Angstroms
        return (4 * np.pi / wl_A) * np.sin(theta / 2)

    # Get wavelength, distance, pixel size from pyFAI integrator
    current_wl_A = ai.wavelength * 1e10 # Convert from meters (pyFAI) to Angstroms
    current_dist_m = ai.dist # Distance in meters
    current_pixel1_m = ai.pixel1 # Pixel size in meters

    qx = x2q(np.arange(img1_plot.shape[1]) - bc_x, current_wl_A, current_dist_m, current_pixel1_m)
    qy = x2q(np.arange(img1_plot.shape[0]) - bc_y, current_wl_A, current_dist_m, current_pixel1_m)
    # Extent for imshow: [left, right, bottom, top]
    extent_q = [qx.min(), qx.max(), qy.min(), qy.max()] # Original code used np.divide(extent, 1e9) to convert, but x2q already returns A^-1

    # Set color for "bad values" (masked regions)
    cmap_mask = plt.get_cmap('jet')
    cmap_mask.set_bad(color='black') # Pixels set to NaN will be black

    # AXS0: Plot the scattering pattern in 2D (linear scale)
    im0 = axs0.imshow(img1_plot, origin='lower', aspect = 'equal', cmap = cmap_mask, extent = extent_q)
    fig1.colorbar(im0, ax = axs0, orientation = 'horizontal', shrink = 0.75).set_label(r'I [cm$^{-1}$]')
    axs0.grid(color = 'white', linestyle = '--', linewidth = 0.25)
    axs0.set(ylabel = r'q$_{y}$ [$\AA$$^{-1}$]', xlabel = r'q$_{x}$ [$\AA$$^{-1}$]')
    axs0.set_title(f'2D Scattering Pattern (Linear Scale)\nScan: {ScanNr}, Frame: {Frame}, Det: {det_float}m')

    # AXS1: Plot the integrated radial integration (I vs q)
    axs1.plot(q, I, 'ok', label = 'total', markersize=6, alpha = 0.8)
    axs1.set(xlabel = r'Scattering vector q [$\AA^{-1}$]', ylabel = r'Intensity I [cm$^{-1}$]', xscale = 'log',
                yscale = 'log', title = 'Sample: '+ str(sample_name))
    axs1.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    axs1.errorbar(q, I, yerr = sigma, color = 'black', lw = 1, markersize=2, capsize=3)

    # AXS2: Plot the scattering pattern in 2D (logarithmic scale)
    # Replace non-positive values with a small number before taking log to avoid log(0) or log(negative)
    img1_log_plot = img1.copy()
    # img1_log_plot[img1_log_plot <= 1e-5] = 1e-5 # Original threshold
    img1_log_plot[mask_bool] = np.nan # Apply mask to the log image as well
    img1_log_plot[img1_log_plot == 0] = np.nan
    img2_log = np.log(img1_log_plot) # Take logarithm

    im2 = axs2.imshow(img2_log, origin='lower', aspect = 'equal', cmap = cmap_mask, extent = extent_q)
    fig1.colorbar(im2, ax = axs2, orientation = 'horizontal', shrink = 0.75).set_label(r'log(I) [cm$^{-1}$]')
    axs2.grid(color = 'white', linestyle = '--', linewidth = 0.25)
    axs2.set(ylabel = r'q$_{y}$ [$\AA$$^{-1}$]', xlabel = r'q$_{x}$ [$\AA$$^{-1}$]')
    axs2.set_title('2D Scattering Pattern (Log Scale)')

    # AXS3: Plot the 2D cake plot (q vs. azimuthal angle)
    # The integrate2d method returns (intensity, 2theta/q, chi)
    try:
        res2d = ai.integrate2d(img1_log_plot, integration_points, 360, method = 'BBox', unit = 'q_A^-1', mask=mask_bool) # Use img1_log_plot here
        I_c, tth_q, chi = res2d

        # Set masked/zero intensity areas in cake plot to NaN for cmap.set_bad
        I_c_plot = I_c.copy()
        I_c_plot[I_c_plot == 0] = np.nan # Mark zero intensity regions as NaN

        img3 = axs3.imshow(I_c_plot, origin="lower", extent=[tth_q.min(), tth_q.max(), chi.min(), chi.max()], aspect="auto", cmap = cmap_mask)
        fig1.colorbar(img3, ax = axs3, orientation = 'horizontal', shrink = 0.75).set_label(r'log(I) [cm$^{-1}$]')
        axs3.set(ylabel = r'Azimuthal angle $\chi$ [degrees]', xlabel = r'q [$\AA^{-1}$]')
        axs3.grid(color='w', linestyle='--', linewidth=1)
        axs3.set_title('2D Integration (Cake Plot)')
    except Exception as e:
        print(f"Error generating 2D cake plot for Scan {ScanNr}, Frame {Frame}: {e}. Skipping cake plot.")
        axs3.text(0.5, 0.5, 'Error generating cake plot', horizontalalignment='center', verticalalignment='center', transform=axs3.transAxes)


    # AXS4: Plot the integrated radial integration in the sectors
    if plot_azim_parts and I_a is not None:
        I_select = I_a[pixel_range_azim, :] # Select intensity data for the specified pixel range
        colors = plt.cm.plasma(np.linspace(0, 1 , I_select.shape[1])) # Generate distinct colors for each sector
        q_a_selected = q_a[pixel_range_azim] # Corresponding q values for the selected range

        for ii in range(I_select.shape[1]):
            # Filter out NaN values for plotting
            valid_indices = ~np.isnan(I_select[:, ii])
            if np.any(valid_indices):
                axs4.loglog(q_a_selected[valid_indices], I_select[valid_indices, ii], marker = 'o', color = colors[ii], markersize=3, alpha=0.5, label=f'{int(npt_azim[ii]):d}-{int(npt_azim[ii+1]):d}°')
            else:
                print(f"Warning: No valid data for azimuthal sector {ii} in selected range. Skipping plot for this sector in AXS4.")
        axs4.set_xlabel(r'Scattering vector q [$\AA^{-1}$]')
        axs4.set_ylabel(r'Intensity I [cm$^{-1}$]')
        axs4.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
        axs4.set_title('Radial Integration by Azimuthal Sector')
        axs4.legend(title='Sector Angles', fontsize='small', ncol=2)
    else:
        axs4.text(0.5, 0.5, 'Azimuthal data not loaded or configured', horizontalalignment='center', verticalalignment='center', transform=axs4.transAxes)

    # AXS 5: Plot the azimuthal plot: I vs angle
    if plot_azim_parts and I_a is not None:
        range_angle_midpoints = [(npt_azim[rr] + npt_azim[rr+1]) / 2 for rr in range(sectors_nr)] # Midpoints for plotting
        sum_point = []
        for ii in range(I_select.shape[1]):
            # Calculate mean intensity over the selected q-range for each sector, ignoring NaNs
            valid_vals = I_select[:, ii][~np.isnan(I_select[:, ii])]
            if valid_vals.size > 0:
                sum_point_tmp = np.mean(valid_vals) # Using mean as in original (nansum/count_nonzero)
            else:
                sum_point_tmp = np.nan # If no valid points, result is NaN
            sum_point.append(sum_point_tmp)

        # Remove NaN values from sum_point and corresponding angles for fitting
        valid_fit_indices = ~np.isnan(sum_point)
        if np.any(valid_fit_indices):
            sum_point_valid = np.array(sum_point)[valid_fit_indices]
            range_angle_valid = np.array(range_angle_midpoints)[valid_fit_indices]

            # Fit a cosine function to the azimuthal intensity
            def cosine_form(theta, I_0, theta0, offset):
                """Cosine squared function for fitting azimuthal anisotropy."""
                return I_0 * np.cos(np.radians(theta - theta0)) ** 2 + offset

            try:
                # Provide initial guess for parameters: [Amplitude, Phase, Offset]
                # Guess for I_0 (amplitude) could be max(sum_point_valid) - min(sum_point_valid)
                # Guess for theta0 (phase) could be 90 (common for oriented samples perpendicular to beam)
                # Guess for offset could be min(sum_point_valid)
                initial_guess = [np.max(sum_point_valid) - np.min(sum_point_valid) + 1e-6, 90, np.min(sum_point_valid) - 1e-6]
                param, cov = optimize.curve_fit(cosine_form, range_angle_valid, sum_point_valid, p0=initial_guess)

                I_0_fit, theta0_fit, offset_fit = param

                # Normalize theta0 to be within 0-180 degrees
                theta0_norm = theta0_fit % 180
                if theta0_norm < 0:
                    theta0_norm += 180

                # Calculate Anisotropy Factor (Af) = Offset / Amplitude
                # Ensure I_0_fit is not zero for division
                if I_0_fit != 0:
                    Af = np.abs(offset_fit / I_0_fit) #
                    Af_rounded = np.round(Af, 4)
                else:
                    Af_rounded = np.nan # Anisotropy undefined if amplitude is zero

                axs5.semilogy(range_angle_valid, cosine_form(range_angle_valid, *param), '--', color = colors[0], label = f'Fit (Angle: {theta0_norm:.1f}°)')
                axs5.set(title = f'Azimuthal Anisotropy (Af = {Af_rounded})')
            except RuntimeError as e:
                print(f"Warning: Could not fit cosine function for azimuthal data for Scan {ScanNr}, Frame {Frame}: {e}.")
                axs5.text(0.5, 0.8, 'Fit failed', horizontalalignment='center', verticalalignment='center', transform=axs5.transAxes)
                axs5.set(title = 'Azimuthal Anisotropy (Fit Failed)')
            except Exception as e:
                print(f"An unexpected error occurred during azimuthal fit for Scan {ScanNr}, Frame {Frame}: {e}.")
                axs5.text(0.5, 0.8, 'Error during fit', horizontalalignment='center', verticalalignment='center', transform=axs5.transAxes)
                axs5.set(title = 'Azimuthal Anisotropy (Error)')


            # Plot individual sector points
            for ii in range(len(range_angle_midpoints)): # Iterate over all sector midpoints
                if not np.isnan(sum_point[ii]):
                    axs5.semilogy(range_angle_midpoints[ii], sum_point[ii], marker = 'o', markersize=10, color=colors[ii], label=f'{int(npt_azim[ii]):d}-{int(npt_azim[ii+1]):d}°')

            axs5.set_xlabel(r'Azimuthal angle $\chi$ [$^o$]')
            axs5.set_ylabel(r'Mean Intensity I($\chi$) [cm$^{-1}$]')
            axs5.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
            axs5.legend(fontsize='small', ncol=2)
        else:
            axs5.text(0.5, 0.5, 'No valid azimuthal data for fitting', horizontalalignment='center', verticalalignment='center', transform=axs5.transAxes)
            axs5.set(title = 'Azimuthal Anisotropy (No Data)')
    else:
        axs5.text(0.5, 0.5, 'Azimuthal data not loaded or configured', horizontalalignment='center', verticalalignment='center', transform=axs5.transAxes)


    # Adjust layout to prevent overlap
    fig1.tight_layout()

    # Save the plots
    prefix_fig = 'radial_integ_plot' # More descriptive name for the figure file
    sufix_jpeg = 'jpeg'
    file_name_fig = integ.make_file_name(path_figures, prefix_fig, sufix_jpeg, sample_name, det_str, ScanNr, Frame)
    try:
        plt.savefig(file_name_fig, dpi=150, bbox_inches='tight') # Save with high DPI and tight bounding box
        print(f"Saved radial integration plot: {file_name_fig}")
    except Exception as e:
        print(f"Error saving radial integration plot for Scan {ScanNr}, Frame {Frame}: {e}.")

    plt.close(fig1) # Close the figure to free memory

    # Revert to previous interactive state if it was off before
    plt.ion() # Restore interactive mode if it was turned off by this function


def plot_integ_azimuthal(config, result, ScanNr, Frame):
    """
    Generates a figure displaying azimuthal integration results, typically
    showing radial profiles for different azimuthal sectors and the sum
    intensity vs. azimuthal angle.

    Args:
        config (dict): The main configuration dictionary.
        result (dict): The results dictionary.
        ScanNr (int): The scan number of the measurement to plot.
        Frame (int): The frame number of the measurement.
    """
    path_analysis = create_analysis_folder(config)

    # Retrieve q_range and npt_azim for plotting setup
    # Note: `pixel_range_azim` is used in plot_integ_radial (AXS4).
    # `q_range` is used here. Ensure consistency in naming.
    # The original code used `pixel_range` but it was not defined in the provided `caller_radial_integration.py` or `integration.py`.
    # Let's assume `pixel_range_azim` from config is meant for the radial range to plot for azimuthal data.
    q_range_for_plot = result['integration'].get('pixel_range_azim')
    sectors_nr = result['integration'].get('sectors_nr')

    if q_range_for_plot is None or sectors_nr is None:
        print(f"Warning: Azimuthal plotting parameters (q_range_for_plot, sectors_nr) not found in result for Scan {ScanNr}. Skipping azimuthal plot.")
        return

    # Determine detector distance and sample name
    class_all = result['overview']['all_files']
    det_float = None
    sample_name = "N/A"
    try:
        idx = -1
        for i, scan in enumerate(class_all['scan']):
            if scan == ScanNr:
                idx = i
                break
        if idx != -1:
            det_float = class_all['detx_m'][idx]
            sample_name = class_all['sample_name'][idx]
        else:
            print(f"Warning: Scan number {ScanNr} not found in 'all_files' overview. Cannot retrieve detector distance or sample info for azimuthal plotting.")
            return
    except KeyError as e:
        print(f"Error accessing overview data for azimuthal plotting (missing key: {e}). Skipping azimuthal plot for Scan {ScanNr}.")
        return

    if det_float is None:
        print(f"Warning: Detector distance not found for Scan {ScanNr} for azimuthal plotting. Skipping.")
        return
    det_str = str(det_float).replace('.', 'p')

    path_figures = os.path.join(path_analysis, f'det_{det_str}', 'figures/')
    if not os.path.exists(path_figures):
        try:
            os.mkdir(path_figures)
        except OSError as e:
            print(f"Error creating figures folder {path_figures}: {e}. Skipping azimuthal plot for Scan {ScanNr}.")
            return

    path_integ = os.path.join(path_analysis, f'det_{det_str}', 'integration/')

    # Redefine npt_azim for consistency with integration step if needed for axis labels
    npt_azim_plot = np.linspace(0, 360, sectors_nr + 1) #
    # Original code had npt_azim from result, but it's generated dynamically in integration.py
    # and not explicitly saved to result for plot_integration.py's direct use.
    # So recalculate here or ensure it's saved. For now, recalculate.

    # Load azimuthal files
    prefix_azim = 'azim_integ'
    sufix_dat = 'dat'
    file_name_azim = integ.make_file_name(path_integ, prefix_azim, sufix_dat, sample_name, det_str, ScanNr, Frame)

    data_azim = None
    try:
        data_azim = np.genfromtxt(file_name_azim, delimiter=',', comments='#')
    except (IOError, ValueError) as e:
        print(f"Error loading azimuthal integrated data from {file_name_azim}: {e}. Skipping azimuthal plot for Scan {ScanNr}.")
        return

    if data_azim is None or data_azim.size == 0:
        print(f"Warning: Azimuthal data file {file_name_azim} is empty or unreadable. Skipping azimuthal plot for Scan {ScanNr}.")
        return

    q = data_azim[:,0]
    # I contains intensity for each sector (columns 1 to sectors_nr)
    I = data_azim[:, 1 : sectors_nr + 1] # Corrected slice
    # sigma = data_azim[:, sectors_nr + 1:] # If sigma columns are present after I columns

    # Select data for the specified q-range
    # Ensure q_range_for_plot (pixel_range_azim) is valid for indexing `q`
    if not (isinstance(q_range_for_plot, range) or isinstance(q_range_for_plot, list) or isinstance(q_range_for_plot, np.ndarray)):
        print(f"Warning: `pixel_range_azim` is not a valid range type for indexing. Skipping azimuthal plot for Scan {ScanNr}.")
        return

    # Check if the q_range indices are within the bounds of `q`
    if max(q_range_for_plot) >= len(q):
        print(f"Warning: `pixel_range_azim` ({q_range_for_plot}) extends beyond the available q-points ({len(q)}). Adjusting range for plotting.")
        q_range_for_plot = range(min(q_range_for_plot), len(q))
        if not q_range_for_plot: # Check if range is now empty
            print("Adjusted q_range_for_plot is empty. Skipping azimuthal plot.")
            return

    I_select = I[q_range_for_plot, :]
    # Filter out rows that are entirely NaN or contain NaNs to prevent issues in sum
    # Or keep NaNs and use nanmean/nansum where appropriate.
    # For plotting, it's often best to handle NaNs by masking or skipping.

    # For plotting, turn off interactive mode if desired
    plt.ioff() #

    colors = plt.cm.viridis(np.linspace(0, 1 , I_select.shape[1])) # Colors for sectors
    fig2, ((axs0, axs1))  = plt.subplots(1, 2,  figsize=(15, 5))

    # AXS0: Plot radial intensity profiles for each azimuthal sector
    for ii in range(I_select.shape[1]):
        # Filter out NaN values for plotting
        valid_indices = ~np.isnan(I_select[:, ii])
        if np.any(valid_indices):
            axs0.loglog(q[q_range_for_plot][valid_indices], I_select[valid_indices, ii], marker = 'o', color = colors[ii], label=f'{int(npt_azim_plot[ii]):d}-{int(npt_azim_plot[ii+1]):d}°')
    axs0.set_xlabel(r'Scattering vector q [$\AA^{-1}$]')
    axs0.set_ylabel(r'Intensity I [cm$^{-1}$]')
    # axs0.set_ylim([1e-2, 1e1]) # Original commented line, depends on data range
    axs0.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    axs0.set_title(f'Azimuthal Sectors (Scan: {ScanNr}, Frame: {Frame})')
    axs0.legend(title='Sector Angles', fontsize='small', ncol=2)


    # AXS1: Plot sum intensity vs. azimuthal angle
    # I_sum: sum over the selected q-range for each azimuthal sector
    # Use np.nansum to ignore NaN values that might result from masked regions or errors
    I_sum = np.nansum(I_select, axis=0)

    # Avoid log of zero or negative values for plotting: add a small offset
    min_I_sum = np.min(I_sum[I_sum > 0]) if np.any(I_sum > 0) else 1e-8 # Find min of positive values, else a small number
    offset_for_plot = min_I_sum * 0.99 # Use 0.99 as in original code, or a fixed small value

    range_angle_midpoints = [(npt_azim_plot[rr] + npt_azim_plot[rr+1]) / 2 for rr in range(sectors_nr)] # Midpoints for plotting

    axs1.semilogy(range_angle_midpoints, I_sum - offset_for_plot, color=colors[0], linestyle = '--', label='Sum over q-range')
    for ii in range(I_select.shape[1]):
        if I_sum[ii] > 0: # Only plot if the sum is positive
            axs1.semilogy(range_angle_midpoints[ii], I_sum[ii] - offset_for_plot, marker = 'o', markersize=10, color=colors[ii])

    axs1.set_xlabel(r'Azimuthal angle $\chi$ [$^o$]')
    axs1.set_ylabel(r'Sum intensity I($\chi$) [cm$^{-1}$]')
    # axs1.set_ylim([1e-2, 1e3]) # Original commented line
    axs1.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    axs1.set_title(f'Integrated Azimuthal Profile (Sample: {sample_name})')
    axs1.legend()

    fig2.tight_layout() # Adjust layout

    # Save the plots
    prefix_fig = 'azim_integ_plot' # More descriptive name for the figure file
    sufix_jpeg = 'jpeg'
    file_name_fig = integ.make_file_name(path_figures, prefix_fig, sufix_jpeg, sample_name, det_str, ScanNr, Frame)
    try:
        fig2.savefig(file_name_fig, dpi=150, bbox_inches='tight')
        print(f"Saved azimuthal integration plot: {file_name_fig}")
    except Exception as e:
        print(f"Error saving azimuthal integration plot for Scan {ScanNr}, Frame {Frame}: {e}.")

    plt.close(fig2) # Close the figure

    # Revert to previous interactive state if it was off before
    plt.ion() # Restore interactive mode if it was turned off by this function
