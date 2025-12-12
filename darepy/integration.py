"""
This module performs the core data integration and preliminary plotting for SANS data.
It orchestrates the process of radial and azimuthal integration of 2D detector images
into 1D scattering curves, applies corrections, and prepares the output files and plots.
It relies on pre-processed configuration and calibration data.

Created on Thu Jul 27 21:47:30 2023

@author: lutzbueno_v
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import re
import sys # Import sys for controlled exits
from .utils import load_hdf # Used for loading HDF5 data
from .utils import create_analysis_folder # Used for creating analysis directories
from .utils import save_results # Used for saving intermediate results
from .correction import prepare_corrections # Used for setting up pyFAI and masks
from .correction import load_standards # Used for loading and correcting standard samples
from .correction import load_and_normalize # Used for loading and applying full normalizations
from .correction import correct_EC # Used for empty cell subtraction
from .calibration import absolute_calibration_2D # Used for absolute intensity calibration
from . import normalization as norm
from .plotting import integration as plot_integ


def set_integration(config, result, det_str):
    """
    Orchestrates the integration process for a specific detector distance.

    This function sets up the integration environment, prepares corrections,
    loads calibration standards, and then initiates the processing loop for
    all relevant sample files for the given detector distance. It also manages
    the `force_reintegrate` flag for efficient reprocessing.

    Args:
        config (dict): The main configuration dictionary with all experiment
                       and analysis parameters.
        result (dict): The results dictionary, updated throughout the pipeline
                       with overview, transmission, and integration data.
        det_str (str): The string identifier for the current detector distance
                       (e.g., '1p6', '6p0', '18p0').

    Returns:
        dict: The updated results dictionary after processing the detector distance.
    """
    path_dir_an = create_analysis_folder(config) # Get the base analysis directory
    perform_abs_calib = config['analysis']['perform_abs_calib'] # Flag to perform absolute calibration

    # Construct the path to the detector-specific analysis folder
    det_folder_name = 'det_' + det_str
    path_det = os.path.join(path_dir_an, det_folder_name)

    # Ensure the detector-specific analysis folder exists
    if not os.path.exists(path_det):
        print(f"Error: Detector folder '{path_det}' not found. Ensure 'prepare_input.select_detector_distances' has run for {det_str.replace('p', '.')}m.")
        sys.exit(1) # Critical error, cannot proceed

    # Create the 'integration' subfolder within the detector's analysis folder
    path_rad_int = os.path.join(path_det, 'integration/')
    if not os.path.exists(path_rad_int):
        os.mkdir(path_rad_int)
        print(f"Created integration folder: {path_rad_int}")

    # Get the file overview specific to this detector distance
    class_file_key = 'det_files_' + det_str
    if class_file_key not in result['overview']:
        print(f"Error: No overview data found for detector {det_str.replace('p', '.')}m in 'result'. Skipping integration.")
        return result # Return current result if no data to process

    class_file = result['overview'][class_file_key] #

    if not class_file['scan']: # Check if there are any scans for this detector distance
        print(f"No files found for detector distance {det_str.replace('p', '.')}m to integrate.")
        return result

    # Prepare pyFAI integrator and masks for this detector distance
    # This must be done even if some scans are skipped later, as standards are loaded
    # and AI object is needed for calibration.
    print(f"Setting up corrections and standards for {det_str.replace('p', '.')}m detector.")
    prepare_corrections(config, result, det_str) #

    # If absolute calibration is desired, load and correct standard samples
    if perform_abs_calib == 1:
        result = load_standards(config, result, det_str) #

    # Perform the actual integration of all files for this detector distance.
    # The `integrate` function will now handle per-file skipping based on `force_reintegrate`.
    result = integrate(config, result, det_str, path_rad_int)

    return result


def make_file_name(path, prefix, sufix, sample_name, det_str, scanNr, frame):
    """
    Constructs a standardized file name for integrated data or 2D patterns.

    Args:
        path (str): The base directory path for the file.
        prefix (str): A prefix for the file name (e.g., 'radial_integ', 'pattern2D').
        sufix (str): The file extension (e.g., 'dat', 'jpeg').
        sample_name (str): The name of the sample.
        det_str (str): The detector distance string (e.g., '1p6', '6p0').
        scanNr (int): The scan number.
        frame (int): The frame number (for multi-frame acquisitions).

    Returns:
        str: The full constructed file path and name.
    """
    # Formats scan number to 7 digits with leading zeros (e.g., 0000123)
    # Formats frame number to 5 digits with leading zeros (e.g., 00000)
    file_n = (f"{path}{prefix}_"
              f"{scanNr:07d}_"
              f"{frame:05d}_"
              f"{sample_name}_"
              f"det{det_str}m.{sufix}")
    return file_n


def integrate(config, result, det_str, path_rad_int):
    """
    Iterates through all sample images for a given detector distance, applies
    corrections, performs radial and azimuthal integration, and saves results.
    This function handles per-scan skipping based on the 'force_reintegrate' flag.

    Args:
        config (dict): The main configuration dictionary.
        result (dict): The results dictionary, which is updated with integration data.
        det_str (str): The detector distance string (e.g., '1p6').
        path_rad_int (str): The path to save integrated radial/azimuthal data and 2D patterns.

    Returns:
        dict: The updated results dictionary.
    """
    # Control overall interactive plotting state for this integration run
    # If BOTH radial and azimuthal plotting are OFF, disable interactive plots.
    # Otherwise, leave interactive plots ON (default behavior of matplotlib)
    plotting_enabled = config['analysis']['plot_radial'] == 1 or \
                       config['analysis']['plot_azimuthal'] == 1

    if not plotting_enabled:
        plt.ioff() # Turn off interactive plotting for the entire integration process
        plotting_was_off = True
    else:
        # If any plotting is enabled, we assume interactive mode is desired
        # No need to explicitly call plt.ion() here, as it's typically default or handled by user env.
        plotting_was_off = False # Keep track of original state if we turned it off

    path_hdf_raw = config['analysis']['path_hdf_raw'] # Path to raw HDF5 files
    class_file = result['overview']['det_files_' + det_str] # File overview for current detector
    perform_abs_calib = config['analysis']['perform_abs_calib'] # Flag for absolute calibration
    force_reintegrate = config['analysis']['force_reintegrate'] # Flag to force re-integration

    # Define common file suffixes needed within this function
    sufix_dat = 'dat'

    print(f'Processing {det_str.replace("p", ".")}m detector data...')

    # Loop through each measurement (scan) for the current detector distance
    for ii in range(0, len(class_file['sample_name'])):
        sample_name = class_file['sample_name'][ii]
        scanNr = class_file['scan'][ii]
        hdf_name = class_file['name_hdf'][ii] # Original HDF5 filename
        frame_nr_total = class_file['frame_nr'][ii] # Total number of frames in this HDF5 file

        # Skip integration if the sample is a calibration standard (standards are handled separately)
        #if sample_name in calibration_names.values():
        #    print(f"Skipping '{sample_name}' (Scan: {scanNr}) as it is a calibration standard.")
        #    continue # Move to the next file (scan)

        # NEW LOGIC: check if the present s complete either for monitors or time
        flux_monit = load_hdf(path_hdf_raw, hdf_name, 'flux_monit')
        time_s = load_hdf(path_hdf_raw, hdf_name, 'time')
        preset = load_hdf(path_hdf_raw, hdf_name, 'moni') # needs to be fixed: moni is either time in s or monitors in e4

        if force_reintegrate == 0:
            if flux_monit == preset*1e4 or time_s == preset:
                print(f"Scan {scanNr} ('{sample_name}') already integrated. Skipping due to force_reintegrate = 0.")
                continue # Skip to the next scan if already processed and not forcing re-integration


        # Loop through each frame within a multi-frame HDF5 file (if applicable)
        for ff in range(0, frame_nr_total):
            img1 = None # Initialize img1 for scope

            if perform_abs_calib == 1:
                # Load and apply all normalizations (time, deadtime, flux, attenuator, transmission, thickness)
                # `load_and_normalize` will use hdf_name_raw internally to get data.
                img_raw_normalized, img_variance = load_and_normalize(config, result, hdf_name, return_variance=True) #

                # Get correction images from 'result' (loaded by load_standards)
                empty_cell_img = result['integration'].get('empty_cell')
                empty_cell_variance = result['integration'].get('empty_cell_variance')

                if empty_cell_img is None:
                    print(f"Error: Dark or empty cell images not found in 'result' for {hdf_name}, Frame {ff}. Skipping corrections.")
                    # If this happens, it means load_standards failed critically.
                    # We continue with raw normalized image for this frame.
                    img_corr = img_raw_normalized[ff, :, :] if frame_nr_total > 1 else img_raw_normalized
                    img_corr_variance = img_variance[ff, :, :] if frame_nr_total > 1 else img_variance
                else:
                    # Apply empty cell, thickness, flat field corrections
                    # and absolute scaling
                    if frame_nr_total > 1:
                        # If multi-frame, select the specific frame for correction
                        # img_corrected_dark = correct_dark(img_raw_normalized[ff,:,:], dark_img) #
                        img1 = correct_EC(img_raw_normalized[ff,:,:], empty_cell_img)
                        img1_variance = img_variance[ff,:,:] + empty_cell_variance
                        img2 = norm.normalize_thickness(config, hdf_name, result, img1)
                        img2_variance = np.square(norm.normalize_thickness(config, hdf_name, result, np.sqrt(img1_variance)))
                        img_corr = absolute_calibration_2D(config, result, scanNr, img2, result['integration'].get('water'))
                        img_corr_variance = np.square(absolute_calibration_2D(config, result, scanNr, np.sqrt(img2_variance), result['integration'].get('water')))
                    else:
                        # Single-frame image
                        # img_corrected_dark = correct_dark(img_raw_normalized, dark_img) #
                        img1 = correct_EC(img_raw_normalized, empty_cell_img)
                        img1_variance = img_variance + empty_cell_variance
                        img2 = norm.normalize_thickness(config, hdf_name, result, img1)#
                        img2_variance = np.square(norm.normalize_thickness(config, hdf_name, result, np.sqrt(img1_variance)))
                        img_corr = absolute_calibration_2D(config, result, scanNr, img2, result['integration'].get('water'))
                        img_corr_variance = np.square(absolute_calibration_2D(config, result, scanNr, np.sqrt(img2_variance), result['integration'].get('water')))

                print(f'Corrected scan {scanNr}, Frame: {ff}')
            else:
                # If absolute calibration is not performed, load raw counts directly
                img_raw = load_hdf(path_hdf_raw, hdf_name, 'counts') #
                if img_raw is None:
                    print(f"Error: Could not load raw counts for {hdf_name}, Frame {ff}. Skipping integration for this frame.")
                    continue # Skip to next frame if counts cannot be loaded

                if frame_nr_total > 1:
                    img_corr = img_raw[ff,:,:]
                    img_corr_variance = img_raw[ff,:,:].copy()
                else:
                    img_corr = img_raw
                    img_corr_variance = img_raw.copy()
                print(f'NOT corrected scan {scanNr}, Frame: {ff}')

           # Ensure corrected img is 2D (remove singleton dimensions)
            img_corr = np.squeeze(img_corr)
            img_corr_variance = np.squeeze(img_corr_variance)

            # --- Save the 2D pattern (Conditional) ---
            if config['analysis'].get('save_2d_patterns', 0) == 1: # Default to 0 if not defined
                prefix_pattern2D = 'pattern2D'
                prefix_variance2D = 'variance2D'
                file_name_pattern2D = make_file_name(path_rad_int, prefix_pattern2D, sufix_dat,
                                                      sample_name, det_str, scanNr, ff)
                file_name_variance2D = make_file_name(path_rad_int, prefix_variance2D, sufix_dat,
                                                      sample_name, det_str, scanNr, ff)
                try:
                    np.savetxt(file_name_pattern2D, img_corr, delimiter=',')
                    np.savetxt(file_name_variance2D, img_corr_variance, delimiter=',')
                except Exception as e:
                    print(f"Error saving 2D pattern to {file_name_pattern2D}: {e}. Skipping further processing for this frame.")
                    continue # Skip to next frame if saving fails

            # --- Perform Radial Integration ---
            prefix_radial = 'radial_integ'
            file_name_radial = make_file_name(path_rad_int, prefix_radial, sufix_dat,
                                                sample_name, det_str, scanNr, ff)
            # Revert: Do NOT pass hdf_name_raw here.
            radial_integ(config, result, img_corr, file_name_radial, img1_variance=img_corr_variance)

            # --- Perform Azimuthal Integration (Conditional) ---
            prefix_azim = 'azim_integ'
            file_name_azim = make_file_name(path_rad_int, prefix_azim, sufix_dat,
                                                 sample_name, det_str, scanNr, ff)
            data_azimuth = azimuthal_integ(config, result, img_corr, file_name_azim, img1_variance=img_corr_variance)


            # --- Plot Radial and Azimuthal Integration Results (Conditional) ---
            # Plotting functions will manage their own specific plotting enablement
            if config['analysis']['plot_radial'] == 1:
                #try:
                plot_integ.plot_integ_radial(config, result, scanNr, ff, img_corr, data_azimuth) #
                #except Exception as e:
                #    print(f"Error plotting radial integration for Scan {scanNr}, Frame {ff}: {e}.")

            if config['analysis']['plot_azimuthal'] == 1:
                #try:
                plot_integ.plot_integ_azimuthal(config, result, scanNr, ff) #
                #except Exception as e:
                #    print(f"Error plotting azimuthal integration for Scan {scanNr}, Frame {ff}: {e}.")


    # Restore overall interactive state if it was turned off by this function
    if plotting_was_off: # If we explicitly turned it off at the beginning
        plt.ion() # Restore interactive mode

    return result


def radial_integ(config, result, img1, file_name, img1_variance=None): # MODIFIED: removed hdf_name_raw
    """
    Performs 1D radial (azimuthally averaged) integration of a 2D detector image.
    The integrated data (q, I, sigma) is saved to a CSV file.
    If absolute calibration is enabled, it applies the calibration.

    Args:
        config (dict): The main configuration dictionary.
        result (dict): The results dictionary, containing the pyFAI integrator ('ai'),
                       mask ('int_mask'), and integration points.
        img1 (np.ndarray): The 2D corrected detector image to integrate.
        file_name (str): The full path and name for the output CSV file.
        # Removed: hdf_name_raw (str): The original raw HDF5 filename from which the data originated.
    """
    ai = result['integration'].get('ai')
    mask = result['integration'].get('int_mask')
    integration_points = result['integration'].get('integration_points') # Number of q-bins

    # Basic validation for essential integration components
    if ai is None or mask is None or integration_points is None:
        print(f"Error: Missing pyFAI integrator (ai), mask, or integration points in 'result' for {file_name}. Radial integration skipped.")
        return

    # Ensure mask has appropriate dtype for pyFAI
    if mask.dtype != bool:
        mask = mask.astype(bool)

    if img1_variance is None:
        error_model = "azimuthal"
    else:
        error_model = None

    # Perform azimuthal integration (1D radial profile)
    try:
        q, I, sigma = ai.integrate1d(img1, integration_points,
                                     correctSolidAngle = True, # Correct for solid angle effects
                                     variance = img1_variance,
                                     mask = mask,
                                     method = 'nosplit_csr', # Fast integration method
                                     unit = 'q_A^-1', # Output q-units in inverse Angstroms
                                     safe = True, # Perform checks for NaNs/Infs
                                     error_model = error_model, # Use azimuthal error model if variance is unknown
                                     flat = None, # Flat field already applied as 2D correction
                                     dark = None)
    except Exception as e:
        print(f"Error during radial integration for {file_name}: {e}. Skipping further radial processing for this file.")
        return

    # Save the integrated files to a CSV
    data_save = np.column_stack((q, I, sigma))
    header_text = 'q (A-1), absolute intensity  I (1/cm), standard deviation'
    try:
        np.savetxt(file_name, data_save, delimiter=',', header=header_text, comments='# ') # Add comments char
    except Exception as e:
        print(f"Error saving radial integrated data to {file_name}: {e}.")

    # Save the overall result dictionary state (e.g., after updating absolute calibration)
    path_dir_an = create_analysis_folder(config) #
    save_results(path_dir_an, result) #


def azimuthal_integ(config, result, img1, file_name, img1_variance=None): # MODIFIED: removed hdf_name_raw
    """
    Performs 1D azimuthal integration (intensity vs. angle) for a specific q-range.
    The integrated data (q, I_sectors, sigma_sectors) is saved to a CSV file.
    If absolute calibration is enabled, it applies the calibration for each sector.

    Args:
        config (dict): The main configuration dictionary.
        result (dict): The results dictionary, containing the pyFAI integrator ('ai'),
                       mask ('int_mask'), integration points, and number of sectors.
        img1 (np.ndarray): The 2D corrected detector image to integrate.
        file_name (str): The full path and name for the output CSV file.
        # Removed: hdf_name_raw (str): The original raw HDF5 filename from which the data originated.
    """
    ai = result['integration'].get('ai')
    mask = result['integration'].get('int_mask')
    integration_points = result['integration'].get('integration_points') # Number of q-bins
    sectors_nr = result['integration'].get('sectors_nr') # Number of azimuthal sectors

    # Basic validation for essential integration components
    if ai is None or mask is None or integration_points is None or sectors_nr is None:
        print(f"Error: Missing pyFAI integrator (ai), mask, integration points, or sectors_nr in 'result' for {file_name}. Azimuthal integration skipped.")
        return

    # Ensure mask has appropriate dtype for pyFAI
    if mask.dtype != bool:
        mask = mask.astype(bool)

    # Define the azimuthal sectors
    npt_azim = np.linspace(0, 360, sectors_nr + 1) # Generates `sectors_nr + 1` points for `sectors_nr` intervals

    I_all = None
    sigma_all = None
    q_all_sectors = None # To store q for the sectors, should be consistent

    if img1_variance is None:
        error_model = "azimuthal"
    else:
        error_model = None


    I_all, q_all_sectors, angles_all, sigma_all = ai.integrate2d_ng(img1,
                                 integration_points, npt_azim=sectors_nr,
                                 correctSolidAngle = True, variance=img1_variance,
                                 mask = mask,
                                 method = ('full', 'csr', 'cython'), unit = 'q_A^-1',
                                 safe = True, error_model = error_model,
                                 flat = None, dark = None)


    # Check if any integration was successful (e.g., first sector produced valid q)
    if q_all_sectors is None or I_all is None or sigma_all is None or q_all_sectors.size == 0:
        print(f"Warning: No valid azimuthal integration data generated for {file_name}. Skipping file save.")
        return



    # The control to save is handled by 'plot_azimuthal' flag that triggers the call to this function.
    # If plot_azimuthal is 0, this function isn't called, so data isn't saved.
    # If plot_azimuthal is 1, this function IS called, and data IS saved.
    data_save = np.column_stack((q_all_sectors, I_all.transpose(), sigma_all.transpose()))
    if config['analysis'].get('save_azimuthal', 0) == 1:
        header_text = (f'q (A-1), {sectors_nr} columns for absolute intensity I (1/cm) '
                       f'(sectors from {npt_azim[0]:.1f} to {npt_azim[-1]:.1f} deg), '
                       f'{sectors_nr} columns for standard deviation\n'
                       f'Angles {angles_all}')
        try:
            np.savetxt(file_name, data_save, delimiter=',', header=header_text, comments='# ') # Add comments char
        except Exception as e:
            print(f"Error saving azimuthal integrated data to {file_name}: {e}.")

        # Save the overall result dictionary state (e.g., after updating absolute calibration)
        path_dir_an = create_analysis_folder(config) #
        save_results(path_dir_an, result) #
    return(data_save)


def plot_radial_integ(config, result, file_name):
    """
    Triggers the plotting functions for radial and azimuthal integration results.
    This function acts as a dispatcher to `plot_integration.py`.

    Note: The name `plot_radial_integ` might be misleading as it triggers both
    radial and azimuthal plots. Consider renaming or clarifying purpose.

    Args:
        config (dict): The main configuration dictionary, containing analysis
                       plotting flags ('plot_azimuthal', 'plot_radial').
        result (dict): The results dictionary.
        file_name (str): The full path and name of the integrated file, used
                         to extract scan and frame numbers.
    """
    # Extract ScanNr and Frame from the file_name using regex
    # Regex explanation:
    # \D: non-digit character (to find delimiters around numbers)
    # (\d{7}): capture group for 7 digits (ScanNr)
    # (\d{5}): capture group for 5 digits (Frame)
    scan_match = re.findall(r"\D(\d{7})\D", file_name)
    frame_match = re.findall(r"\D(\d{5})\D", file_name)

    ScanNr = int(scan_match[0]) if scan_match else None
    Frame = int(frame_match[0]) if frame_match else None

    if ScanNr is None or Frame is None:
        print(f"Warning: Could not extract ScanNr or Frame from file_name: {file_name}. Skipping plotting.")
        return

    # Call plotting functions based on configuration flags
    # These functions will manage their own specific plotting enablement (ioff/ion based on caller).
    if config['analysis']['plot_azimuthal'] == 1:
        #try:
        plot_integ.plot_integ_azimuthal(config, result, ScanNr, Frame) #
#        except Exception as e:
 #           print(f"Error plotting azimuthal integration for Scan {ScanNr}, Frame {Frame}: {e}.")

    if config['analysis']['plot_radial'] == 1:
        #try:
        plot_integ.plot_integ_radial(config, result, ScanNr, Frame) #
        #except Exception as e:
        #    print(f"Error plotting radial integration for Scan {ScanNr}, Frame {Frame}: {e}.")
