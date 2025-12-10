"""
This module handles various corrections and preparatory steps for SANS data
integration. It includes defining the integration mask and beam center,
loading and applying standard sample corrections (dark field, empty cell,
flat field), and orchestrating the initial normalization steps before
radial and azimuthal integration.

Created on Wed Aug 16 10:59:14 2023

@author: lutzbueno_v
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys # Imported for sys.exit
from utils import create_analysis_folder # Imported from utils
from utils import save_results # Imported from utils
from utils import load_hdf # Imported from utils
import normalization as norm # Imported for normalization functions
import pyFAI

def prepare_corrections(config, result, det):
    """
    Prepares the necessary components for SANS data corrections and integration,
    including setting up the pyFAI AzimuthalIntegrator, defining the integration
    mask, and determining beam center coordinates.

    This function should be called once per detector distance to initialize
    the integration environment.

    Args:
        config (dict): The main configuration dictionary, containing instrument
                       and analysis parameters.
        result (dict): The results dictionary, where integration setup parameters
                       (e.g., beam center, mask, pyFAI integrator) will be stored.
        det (str): The detector distance string (e.g., '1p6', '6p0', '18p0')
                   for which to prepare the corrections.

    Returns:
        tuple: A tuple containing:
            - ai (pyFAI.AzimuthalIntegrator): The configured pyFAI integrator object.
            - mask (np.ndarray): The 2D boolean mask array for the detector.
            - result (dict): The updated results dictionary.
    """
    path_dir_an = create_analysis_folder(config) # Get the main analysis folder path
    path_det_hdf_raw = os.path.join(path_dir_an, 'det_' + det, 'hdf_raw/')

    # Check if the detector-specific HDF raw data folder is empty
    list_hdf_files = os.listdir(path_det_hdf_raw)
    if not list_hdf_files:
        print(f"Error: No HDF5 files found in {path_det_hdf_raw}. Cannot prepare corrections.")
        sys.exit(1) # Critical error, cannot proceed without raw data.

    # Pick the first HDF file to extract common parameters like distance and wavelength
    first_hdf_name = list_hdf_files[0]

    # Load detector distance from the HDF file
    dist = load_hdf(path_det_hdf_raw, first_hdf_name, 'detx') # Distance in meters
    if dist is None or not isinstance(dist, (int, float)):
        print(f"Error: Could not load valid detector distance for {first_hdf_name}. Check HDF5 file or 'detx' property.")
        sys.exit(1)

    # Define pixel size (assumed to be square pixels)
    pixel1 = config['instrument']['pixel_size'] # Pixel size in meters
    pixel2 = pixel1

    # Define the wavelength
    wl_input = config['experiment']['wl_input']
    if wl_input == 'auto':
        wl = load_hdf(path_det_hdf_raw, first_hdf_name, 'wl') # Wavelength in Angstroms
        if wl is None or not isinstance(wl, (int, float)):
             print(f"Error: Could not load valid wavelength for {first_hdf_name} with 'auto' setting. Check HDF5 file or 'wl' property.")
             sys.exit(1)
        wl *= 1e-10  # Convert from Angstroms to meters for pyFAI
    else:
        wl = wl_input * 1e-10  # Convert input Angstroms to meters

    # Define the beam center coordinates in pixels
    beam_center_guess = config['analysis']['beam_center_guess'] # Dictionary of guesses
    det_float_key = float(det.replace('p', '.')) # Convert '1p6' to 1.6 for dictionary lookup

    if det_float_key not in beam_center_guess:
        print(f"Error: Beam center guess not provided for detector distance {det_float_key}m.")
        sys.exit(1)

    bc_x = beam_center_guess[det_float_key][0] # X-coordinate (horizontal)
    bc_y = beam_center_guess[det_float_key][1] # Y-coordinate (vertical)

    # Convert beam center from pixels to meters for pyFAI (poni coordinates)
    poni2 = bc_x * pixel1 # Perpendicular to the detector surface, along horizontal detector axis
    poni1 = bc_y * pixel2 # Perpendicular to the detector surface, along vertical detector axis

    # Store beam center in results for later plotting
    result['integration']['beam_center_x'] = bc_x
    result['integration']['beam_center_y'] = bc_y

    # Define the detector mask to exclude beam stop and edges
    detector_size = config['instrument']['detector_size'] # Detector dimensions in pixels (e.g., 128)
    mask = np.zeros([detector_size, detector_size], dtype=int) # Initialize mask with zeros (all valid)

    # Apply beam stopper mask based on configured coordinates
    beamstopper_coordinates = config['analysis']['beamstopper_coordinates'] # Dictionary of beam stopper coordinates
    if det_float_key in beamstopper_coordinates:
        bs_all = beamstopper_coordinates[det_float_key]
        try:
            for jj in range(0, len(bs_all)):
                bs_select = 'bs' + str(jj)
                bs_coords = bs_all[bs_select]
                if len(bs_coords) == 4:
                    y_n, y_p, x_n, x_p = bs_coords # Unpack y_min, y_max, x_min, x_max
                    mask[y_n:y_p, x_n:x_p] = 1 # Mark beam stopper region as masked (1)
                else:
                    print(f"Warning: Beamstopper coordinates for {det_float_key}m are not in [y_min, y_max, x_min, x_max] format. Beamstopper mask not applied.")
        except:
            if len(bs_all) == 4:
                y_n, y_p, x_n, x_p = bs_all # Unpack y_min, y_max, x_min, x_max
                mask[y_n:y_p, x_n:x_p] = 1 # Mark beam stopper region as masked (1)
            else:
                print(f"Warning: Beamstopper coordinates for {det_float_key}m are not in [y_min, y_max, x_min, x_max] format. Beamstopper mask not applied.")
    else:
        print(f"Warning: No beamstopper coordinates defined for detector distance {det_float_key}m. Ensure no beamstopper interferes or define its coordinates if present.")

    if config['instrument']['name'] == 'SANS-I':
        # Remove the edge lines around the detector (common for most detectors)
        lines = 2 # Number of pixels to mask from the edges
        mask[:, 0:lines] = 1 # Left edge
        mask[:, detector_size - lines : detector_size] = 1 # Right edge (corrected slice end)
        mask[0:lines, :] = 1 # Bottom edge
        mask[detector_size - lines : detector_size, :] = 1 # Top edge (corrected slice end)

        # Remove the last thick line - only for SANS-I (specific to instrument setup)
        if dist > 15: # Assuming this applies to 18m distance
            lines = 6 # Larger number of pixels to mask
            mask[:, detector_size - lines : detector_size] = 1 # Mask thicker section on the right edge

        # Remove the corners (to avoid noisy regions)
        corner = 10 # Size of the square corner to mask
        mask[0:corner, 0:corner] = 1 # Bottom-left
        mask[detector_size - corner:detector_size, 0:corner] = 1 # Top-left (corrected slice start)
        mask[detector_size - corner:detector_size, detector_size - corner:detector_size] = 1 # Top-right (corrected slice starts)
        mask[0:corner, detector_size - corner:detector_size] = 1 # Bottom-right (corrected slice starts)
    elif config['instrument']['name'] == 'SANS-LLB':
        # Remove the edge lines around the detector (common for most detectors)
        lines = 2 # Number of pixels to mask from the edges
        mask[:, 0:lines] = 1 # Left edge
        mask[:, detector_size - 6 : detector_size] = 1 # Right edge (corrected slice end)
        mask[0:lines, :] = 1 # Bottom edge
        mask[detector_size - lines : detector_size, :] = 1 # Top edge (corrected slice end)

        if dist < 2: # Assuming this applies to short distance
            lines = 6
            mask[:, 0:lines] = 1 # Left edge
            lines = 10
            mask[0:lines, :] = 1 # Bottom edge
            lines = 6 # Larger number of pixels to mask
            mask[:, detector_size - lines : detector_size] = 1 # Mask thicker section on the right edge



    result['integration']['int_mask'] = mask # Store the final mask in results

    # Create the pyFAI AzimuthalIntegrator object
    ai = pyFAI.AzimuthalIntegrator(dist=dist, poni1=poni1, poni2=poni2,
                                   rot1=0, rot2=0, rot3=0, # Rotation angles, typically 0 for SANS
                                   pixel1=pixel1, pixel2=pixel2,
                                   splineFile=None, detector=None, wavelength=wl)
    ai.setChiDiscAtZero() # Sets the azimuthal angle chi=0 to be along the positive x-axis

    result['integration']['ai'] = ai # Store the pyFAI integrator in results
    save_results(path_dir_an, result) # Save the updated results dictionary

    # Plot the mask and beam center for visual inspection
    plt.ioff() # Turn off interactive plotting

    plt.figure(figsize=(6, 6)) # Create a new figure
    plt.imshow(mask, origin='lower', aspect = 'equal', clim=[0, 1], cmap='gray') # Display the mask
    plt.plot(bc_x, bc_y, 'r+', markersize=10, markeredgewidth=2) # Plot beam center as a red cross
    plt.title(f'Detector Mask for {dist:.1f}m\nBeam Center: ({bc_x:.2f}, {bc_y:.2f}) pixels')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.colorbar(orientation = 'vertical', shrink = 0.5, ticks=[0, 1]).set_label('Masked (1) / Valid (0)')
    file_name = os.path.join(path_dir_an, f'beamcenter_mask_det{det}.jpg') # Use det string for filename
    plt.savefig(file_name, dpi=300, bbox_inches='tight') # Save the figure
    plt.close('all') # Close all figures to free memory
    # Revert to previous interactive state if it was off before
    plt.ion() # Restore interactive mode if it was off

    return (ai, mask, result) # Return the integrator, mask, and updated result

def load_standards(config, result, det):
    """
    Loads and preprocesses calibration standard measurements (cadmium, water, empty cell).
    These standards are used for dark field and flat field corrections later.

    Args:
        config (dict): The main configuration dictionary.
        result (dict): The results dictionary, where loaded and corrected standard
                       data will be stored under 'integration' key.
        det (str): The detector distance string (e.g., '1p6', '6p0') for which
                   to load the standards.

    Returns:
        dict: The updated results dictionary containing the processed standard data.

    Raises:
        SystemExit: If a required calibration measurement is missing,
                    the script will exit.
    """
    # Define path_dir_an at the beginning of the function
    path_dir_an = create_analysis_folder(config) #

    calibration_names = config['experiment']['calibration'] # Get mapping of generic names to actual sample names
    class_file_key = 'det_files_' + det # Key for the specific detector's file list

    if class_file_key not in result['overview']:
        print(f"Error: No file overview for detector {det.replace('p', '.')}m in 'result'. Cannot load standards.")
        sys.exit(1)

    class_det_files = result['overview'][class_file_key] # Get the file list for the current detector distance

    # List of required standards to check for
    required_standards = ['cadmium', 'water', 'water_cell', 'empty_cell']

    # Iterate through required standards and load their data
    for standard_key in required_standards:
        standard_sample_name = calibration_names.get(standard_key)

        if standard_sample_name not in class_det_files['sample_name']:
            print('###########################################################')
            print(f"ERROR: Missing calibration measurement for '{standard_key}' ('{standard_sample_name}') at {det.replace('p', '.')}m.")
            print('Please ensure all required calibration files are present for this detector distance.')
            print('###########################################################')
            sys.exit('Critical: Missing calibration data.')

        # Find the index of the standard measurement in the current detector's file list
        idx = class_det_files['sample_name'].index(standard_sample_name)
        hdf_name = class_det_files['name_hdf'][idx]

        print(f"Loading and normalizing standard: {standard_sample_name} for {det.replace('p', '.')}m")
        # Load and normalize the standard measurement (time, deadtime, flux, attenuator, thickness)
        # Transmission is NOT applied here, as standards often have 100% transmission or are not measured with it.
        # This function implicitly skips transmission for standards based on 'trans_dist' in caller_radial_integration.py
        # and its logic in normalization.py.
        if standard_key == 'cadmium':
            path_hdf_raw = config['analysis']['path_hdf_raw']
            counts = load_hdf(path_hdf_raw, hdf_name, 'counts') # Load raw counts

            if counts is None:
                print(f"Error: Could not load counts data for {hdf_name}. Skipping normalizations.")
                img = np.array([[]]) # Return an empty/invalid array if counts cannot be loaded

            else:
                counts = norm.normalize_deadtime(config, hdf_name, counts)
                counts_per_sec = norm.normalize_time(config, hdf_name, counts)
                img = counts_per_sec.copy()
        else:
            img, img_variance = load_and_normalize(config, result, hdf_name, return_variance=True)
            result['integration'][standard_key + '_variance'] = img_variance
        if standard_key == 'water':
            water_hdf_name = hdf_name
        result['integration'][standard_key] = img


    # Subtract empty cell from water to get pure water scattering for flat field
    img_h2o = result['integration']['water']
    img_h2o_variance = result['integration']['water_variance']
    img_cell = result['integration']['water_cell']
    img_cell_variance = result['integration']['water_cell_variance']
    img_h2o = correct_EC(img_h2o, img_cell)
    img_h2o_variance = img_h2o_variance + img_cell_variance
    img_h2o = norm.normalize_thickness(config, water_hdf_name, result, img_h2o)
    img_h2o_variance = np.square(norm.normalize_thickness(config, water_hdf_name, result, np.sqrt(img_h2o_variance)))
    result['integration']['water'] = img_h2o # Store the corrected water
    result['integration']['water_variance'] = img_h2o_variance

    # Avoid negative numbers and zeros in the final water image, set to small positive for log scale etc.
    # This is critical for subsequent flat-fielding or log plots.
    result['integration']['water'][result['integration']['water'] <= 0] = 0
    result['integration']['water_variance'][result['integration']['water_variance'] <= 0] = 0

    save_results(path_dir_an, result) # Save updated results including standards
    return result

def load_and_normalize(config, result, hdf_name, return_variance=False):
    """
    Loads raw counts data from an HDF5 file and applies a sequence of essential
    normalizations: time, deadtime, flux, attenuator, transmission, and thickness.

    This function is a wrapper for calling multiple normalization steps.

    Args:
        config (dict): The main configuration dictionary.
        result (dict): The results dictionary, required for transmission and thickness
                       normalizations.
        hdf_name (str): The filename of the HDF5 file to load and normalize.

    Returns:
        np.ndarray: The fully normalized 2D detector counts array.
    """
    path_hdf_raw = config['analysis']['path_hdf_raw']
    counts = load_hdf(path_hdf_raw, hdf_name, 'counts') # Load raw counts
    counts_deviation = np.sqrt(counts.copy())

    if counts is None:
        print(f"Error: Could not load counts data for {hdf_name}. Skipping normalizations.")
        return np.array([[]]) # Return an empty/invalid array if counts cannot be loaded

    dark_per_sec_img = (result['integration'].get('cadmium')).copy()
    meas_time = load_hdf(path_hdf_raw, hdf_name, 'time')

    dark_img = dark_per_sec_img * meas_time

    # Transmission normalization is conditional based on 'trans_dist' setting for SANS-I
    # for SANS-LLB it happens for all samples and must be performed first
    if config['experiment']['trans_dist'] > 0:
        counts = norm.normalize_transmission(config, hdf_name, result, counts)
        counts_deviation = norm.normalize_transmission(config, hdf_name, result, counts_deviation)
    else:
        print(f"Error: trans_dist is < 0. Skipping transmission normalization for {hdf_name}.")

    if return_variance:
        return counts, np.square(counts_deviation)
    else:
        return counts

    # Apply normalizations in sequence
    counts = norm.normalize_deadtime(config, hdf_name, counts)
    counts = correct_dark(counts, dark_img)
    counts = norm.normalize_flux(config, hdf_name, counts)
    counts = norm.normalize_attenuator(config, hdf_name, counts)

    counts_deviation = norm.normalize_deadtime(config, hdf_name, counts_deviation)
    counts_deviation = np.sqrt(dark_img + counts_deviation**2)
    counts_deviation = norm.normalize_flux(config, hdf_name, counts_deviation)
    counts_deviation = norm.normalize_attenuator(config, hdf_name, counts_deviation)



def correct_dark(img, dark):
    """
    Performs dark field correction by subtracting a dark current image from the data.

    Any negative values resulting from the subtraction are clipped to zero.

    Args:
        img (np.ndarray): The 2D detector image array to be corrected.
        dark (np.ndarray): The 2D dark current image array (e.g., from cadmium).

    Returns:
        np.ndarray: The corrected 2D detector image.
    """
    if img.shape != dark.shape:
        print(f"Warning: Image and dark field dimensions mismatch ({img.shape} vs {dark.shape}). Dark correction skipped.")
        return img # Return original if dimensions don't match

    corrected_img = np.subtract(img, dark)
    # Clip any negative values to zero, as intensity cannot be negative
    corrected_img[corrected_img < 0] = 0
    return corrected_img

def correct_EC(img, EC):
    """
    Performs empty cell correction by subtracting an empty cell image from the data.

    This removes scattering contributions from the sample holder or container.
    Any negative values resulting from the subtraction are clipped to zero.

    Args:
        img (np.ndarray): The 2D detector image array to be corrected.
        EC (np.ndarray): The 2D empty cell image array.

    Returns:
        np.ndarray: The corrected 2D detector image.
    """
    if img.shape != EC.shape:
        print(f"Warning: Image and empty cell dimensions mismatch ({img.shape} vs {EC.shape}). Empty cell correction skipped.")
        return img # Return original if dimensions don't match

    corrected_img = np.subtract(img, EC)
    # Clip any negative values to zero
    corrected_img[corrected_img < 0] = 0
    return corrected_img
