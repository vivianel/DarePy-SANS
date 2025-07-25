# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 13:31:39 2023

@author: lutzbueno_v
"""
"""
This is the main script for the DarePy-SANS data reduction pipeline.
It orchestrates the entire process from loading raw HDF5 files to performing
radial and azimuthal integration, applying various corrections (dark field,
empty cell, flat field, transmission), and optionally performing absolute
intensity calibration.

Users can configure experimental parameters, instrument-specific settings,
and detailed analysis options. The pipeline supports processing for all
detected detector distances or a selection of specific distances.
Results, including integrated data and plots, are saved to a designated
analysis directory.
"""
# %% EXPERIMENTAL PARAMETERS
# This section defines parameters related to the SANS experiment and instrument setup.

# Select the SANS instrument used for data collection.
# This choice determines which instrument-specific configurations (e.g., deadtime,
# attenuator values, absolute calibration constants) are applied.
# Current options: 'SANS-I'
instrument = 'SANS-I'

# Calibration standards required for data reduction.
# These sample names MUST EXACTLY MATCH the 'sample_name' metadata in your
# HDF5 raw data files. The script will look for these names in your raw data.
# If any of these measurements are missing for a given detector distance,
# the script will issue a critical error and stop.

# Optional measurement of the empty beam for transmission calculation.
# This sample represents the direct beam intensity without any sample in the beam path.
# Set to the sample name used for your empty beam measurement (e.g., 'EB', 'EMPTY').
empty_beam = 'EB'

# Cadmium measurement: Used for dark field (background) correction.
# This measurement typically blocks all neutrons, providing a baseline of detector
# noise and ambient background.
# Set to the sample name used for your cadmium measurement (e.g., 'Cd', 'CADMIUM').
cadmium = 'Cd'

# Water measurement: Used for flat field correction and absolute intensity calibration.
# Water has a well-known incoherent scattering cross-section, serving as a standard reference.
# Set to the sample name used for your water measurement (e.g., 'H2O', 'WATER').
water = 'H2O'

# Water cell measurement: Represents the scattering from the empty container used for the water sample.
# This is subtracted from the 'water' measurement to isolate the scattering from water itself.
# Set to the sample name used for your empty water cell (e.g., 'EC', 'WATER_EC').
water_cell = 'EC'

# Empty cell measurement: Represents the scattering from the empty container used for your samples.
# This is crucial for subtracting contributions from sample holders.
# Set to the sample name used for your empty sample cell (e.g., 'EC', 'SAMPLE_EC').
empty_cell = 'EC'

# Sample thickness information.
# Provide a dictionary where keys are sample names (matching HDF5 metadata) and
# values are their thicknesses in centimeters (cm).
# Example for specific samples: {'MySample1': 0.1, 'AnotherSample': 0.25}
# If all your samples have the same thickness, you can specify it using the
# special key 'all'. This value will be used for any sample not explicitly listed.
# Example for universal thickness: {'all': 0.1} (assumes all samples are 0.1 cm thick)
# If a sample name is not found here and 'all' is not specified, a default of 0.1 cm will be used.
sample_thickness = {'all':0.1}

# Transmission measurement distance.
# This specifies the detector distance (in meters) at which transmission measurements
# were performed. Transmission correction is crucial for absolute intensity scaling
# and accounting for sample absorption.
# - A positive value (e.g., 18.0) indicates that transmission measurements
#   were taken at this distance, and transmission correction will be applied.
# - Set to a non-positive value (e.g., 0, -1) if transmission correction is
#   not needed for your experiment or if transmission data is not available.
trans_dist = 18.0

# Detector distance to use for flat field correction at large distances.
# For very large detector distances (e.g., 18.0m), water flat field measurements
# can be noisy due to low neutron counts. This parameter allows you to use
# a water measurement from a shorter, typically less noisy, distance (e.g., 4.5m)
# as a more reliable flat field. The script will automatically scale this
# replacement water to match the 18m data.
# - Provide a detector distance (e.g., 4.5, 6.0) if a replacement is desired.
# - Set to 0 or a negative value if no replacement is desired.
replace_18m = 4.5

# Wavelength of the incident neutrons.
# This value is crucial for converting scattering angles to the scattering vector 'q'.
# - Set to 'auto' (string) to automatically load the wavelength from the HDF5 raw data files.
#   This is the recommended setting if wavelength is recorded in your HDF5 metadata.
# - Alternatively, provide a specific wavelength value in Angstroms (e.g., 6.0).
#   Use this if 'auto' fails or if you need to override the recorded value.
wl = 'auto' # Options: 'auto' (read from HDF5) or a float value in Angstroms (e.g., 6.0)

# Beam stopper coordinates.
# This is a dictionary defining the rectangular region of the beam stopper on the detector.
# These coordinates are used to create a mask that excludes the direct beam and beam stopper
# from the integration, preventing artifacts.
# Format: {detector_distance_in_meters (float): [y_min_pixel, y_max_pixel, x_min_pixel, x_max_pixel]}
# - (y_min_pixel, y_max_pixel): Vertical pixel range of the beam stopper.
# - (x_min_pixel, x_max_pixel): Horizontal pixel range of the beam stopper.
beamstopper_coordinates = {
    1.6: [56, 77, 50, 69],
    4.5: [58, 71, 93, 106],
    18.0: [56, 71, 93, 106]
}

# Beam center guess.
# A dictionary providing initial guesses for the beam center coordinates (in pixels)
# for different detector distances. These values are critical for accurate radial
# and azimuthal integration as they define the origin for q-space conversion.
# Format: {detector_distance_in_meters (float): [center_x_pixel, center_y_pixel]}
beam_center_guess = {
    1.6: [60.58, 67.63],
    4.5: [99.83, 64.57],
    18.0: [99.73, 64.24]
}

# Target detector distances for processing.
# This parameter controls which detector distances will be processed by the analysis pipeline.
# You can choose to process all available data or focus on specific distances to save time.
# - Set to 'all' (string) to process every unique detector distance found in your raw data.
# - Provide a list of specific detector distances (in meters, as floats) to process only those.
#   Example: target_detector_distances = [6.0]  # Processes only data from 6.0m detector distance.
#   Example: target_detector_distances = [1.6, 6.0] # Processes data from 1.6m and 6.0m.
target_detector_distances = 'all' # Options: 'all' or a list of floats (e.g., [6.0], [1.6, 4.5])


# %% ANALYSIS PARAMETERS
# This section defines parameters related to the data reduction and analysis pipeline's execution.

# Path where the raw HDF5 data files are located.
# Ensure this path points directly to the directory containing your `.hdf` files.
path_hdf_raw = 'C:/Users/lutzbueno_v/Documents/Analysis/DarePy-SANS/raw_data/'

# Path to the working directory where all analysis results (integrated data, plots, logs) will be saved.
# A main analysis folder (e.g., 'analysis/') or a uniquely identified subfolder
# (e.g., 'analysis_batch1/') will be created within this directory.
path_dir = 'C:/Users/lutzbueno_v/Documents/Analysis/DarePy-SANS/'

# Identifier for the analysis output folder.
# This string will be appended to the default 'analysis/' folder name.
# - An empty string (e.g., '') will create a folder named 'analysis/'.
# - Providing a string (e.g., 'batch1') will create 'analysis_batch1/'.
# This helps in organizing results from multiple analysis runs.
add_id = ''

# List of scan numbers to be excluded from the analysis pipeline.
# Provide a list of integer scan numbers that should be completely ignored
# during the processing. This is useful for problematic or irrelevant scans.
# Example: [23177, 23178, 23180]
# You can also use list(range(start, end)) for a sequence of scans.
# Keep as an empty list [] if no files need to be excluded.
exclude_files = []

# Control radial integration and plotting.
# Radial integration produces 1D scattering curves (Intensity vs. q).
# - Set to 1 to perform radial integration and generate corresponding plots
#   for visualization and quality checks.
# - Set to 0 to skip radial integration and plotting. Raw 2D data will still
#   be processed through correction steps if absolute calibration is enabled,
#   but no 1D radial data will be generated or saved.
plot_radial = 1

# Control azimuthal integration and plotting.
# Azimuthal integration analyzes scattering intensity as a function of azimuthal angle,
# useful for studying anisotropy.
# - Set to 1 to perform azimuthal integration and generate corresponding plots.
# - Set to 0 to skip azimuthal integration and plotting. No azimuthal data
#   will be generated or saved.
plot_azimuthal = 0

# Control saving of azimuthal integration data files (.dat).
# This flag independently controls whether the calculated 1D azimuthal data
# (q vs. I for each sector) is saved to disk.
# - Set to 1 to save azimuthal data files (e.g., 'azim_integ_*.dat').
# - Set to 0 to skip saving these data files, even if 'plot_azimuthal' is 1
#   (the plots will still be generated if plot_azimuthal is 1).
#   Note: Setting 'plot_azimuthal' to 0 will implicitly skip saving data as well.
save_azimuthal = 0

# Control saving of raw 2D detector patterns (.dat).
# These files are direct representations of the corrected 2D detector images.
# They can be very large and consume significant disk space.
# - Set to 1 if you need to save these raw 2D patterns for every frame.
# - Set to 0 to skip saving these 2D pattern files. This can significantly
#   speed up the process by reducing disk I/O.
save_2d_patterns = 0

# Absolute calibration toggle.
# This step converts scattering intensities from arbitrary detector counts
# into absolute units (cm^-1), allowing for quantitative comparison with models
# or other experiments.
# - Set to 1 to perform absolute intensity calibration. This requires the
#   'water' standard measurement and its associated configuration.
# - Set to 0 to skip absolute calibration. Intensities will remain in
#   arbitrary units, suitable for relative comparisons.
perform_abs_calib = 1

# Force re-integration of data.
# This flag controls whether previously integrated files are re-processed.
# - Set to 1 to force the radial and azimuthal integration process to re-run
#   for all files, even if corresponding integrated `.dat` files already exist.
#   This will overwrite existing results. Use this if you change analysis parameters.
# - Set to 0 to only integrate new or un-integrated files. This significantly
#   speeds up re-runs if only a few new files are added or parameters are unchanged.
force_reintegrate = 1

# Number of integration points for radial integration.
# This determines the number of 'q' (scattering vector) bins in the final 1D
# radial scattering curve. A higher number provides more detailed curves but
# may increase processing time and file size.
integration_points = 120

# Number of angular sectors for azimuthal integration.
# This divides the 2D detector image into this many angular slices (bins) for
# azimuthal anisotropy analysis.
sectors_nr = 16

# Pixel range for azimuthal integration.
# Defines the radial pixel range (distance from the beam center) over which
# azimuthal intensity will be averaged. This helps focus azimuthal analysis on
# a specific 'q' region of interest.
# Example: range(5, 40) means pixels from 5 to 39 (inclusive) from the beam center.
pixel_range_azim = range(5,40)


# %% PIPELINE EXECUTION (DO NOT MODIFY BELOW THIS LINE)
# This section contains the core logic for running the data reduction pipeline.
# It initializes the configuration, orchestrates the various processing steps,
# and manages module imports.

import os
import sys # Import sys for controlled exits
# Change the current working directory to the 'codes' subfolder within your path_dir.
# This is crucial for importing other Python modules (like prepare_input, transmission, integration).
# Ensure your main script and all other necessary Python modules are located in
# a 'codes' folder (e.g., 'C:/Users/lutzbueno_v/Documents/Analysis/DarePy-SANS/codes/').
try:
    os.chdir(path_dir + '/codes/')
except FileNotFoundError:
    print(f"Error: The 'codes' directory expected at '{path_dir}/codes/' was not found.")
    print("Please ensure your Python scripts are in this location or update 'path_dir' accordingly.")
    sys.exit(1) # Exit if the code directory is not found

# Import necessary modules from your project
import prepare_input as org
from transmission import trans_calc
import integration as ri



# Prepare a dictionary that maps generic calibration names (used in the code)
# to their specific sample names (as recorded in your HDF5 raw data).
calibration = {'cadmium':cadmium, 'water':water, 'water_cell': water_cell, 'empty_cell':empty_cell}

# Initialize the 'result' dictionary. This dictionary will serve as a central
# repository to store all intermediate data, processing flags, and final
# analysis outputs throughout the pipeline. It is passed between functions.
result = {'transmission':{},
       'overview':{},
       'integration':{
           'pixel_range_azim':pixel_range_azim,
           'integration_points':integration_points,
           'sectors_nr': sectors_nr}}


# Create the complete configuration dictionary for the selected instrument.
# This consolidates all parameters for easy access throughout the pipeline.
configuration = {'SANS-I':{
    'instrument': {'deadtime': 6.6e-7, # Detector deadtime in seconds.
                   # Attenuator transmission factors: A dictionary where keys are
                   # attenuator settings (e.g., '0', '1') as strings, and values
                   # are their corresponding transmission factors (e.g., 1 for no att., 1/485 for att. 1).
                   'list_attenuation': {'0':1, '1':1/485,'2':1/88,'3':1/8, '4':1/3.5,'5':1/8.3},
                   'pixel_size':7.5e-3, # Physical size of a single detector pixel in meters.
                   'detector_size': 128, # Number of pixels along one side of the (assumed square) detector.
                   # List of beam stopper sizes (if applicable, often not used directly in this pipeline).
                   'list_bs': {'1':40, '2':70,'3':85,'4':100},
                   # Absolute calibration cross-sections for different wavelengths.
                   # A dictionary where keys are rounded wavelengths in Angstroms (as strings)
                   # and values are calibration factors (e.g., from water calibration)
                   # to convert to cm^-1 units.
                   'list_abs_calib': {'5':0.909, '6':0.989, '8':1.090, '10':1.241, '12':1.452}},
    'experiment': {'trans_dist': trans_dist, # Transmission measurement distance (from above).
                   'calibration':calibration, # Calibration sample names (from above).
                   'sample_thickness':sample_thickness, # Sample thickness info (from above).
                   'wl_input': wl}, # Wavelength input setting (from above).
    'analysis': {'path_dir': path_dir, # Working directory path (from above).
                 'path_hdf_raw':path_hdf_raw, # Raw HDF5 data path (from above).
                 'exclude_files':exclude_files, # List of scans to exclude (from above).
                 'perform_abs_calib':perform_abs_calib, # Absolute calibration toggle (from above).
                 'force_reintegrate': force_reintegrate, # Force re-integration toggle (from above).
                 'replace_18m':replace_18m, # 18m replacement distance (from above).
                 "plot_azimuthal":plot_azimuthal, # Azimuthal plot toggle (from above).
                 "plot_radial":plot_radial, # Radial plot toggle (from above).
                 'add_id':add_id, # Analysis folder ID (from above).
                 'save_azimuthal': save_azimuthal, # Save azimuthal data files (from above).
                 'save_2d_patterns': save_2d_patterns, # Save 2D pattern files (from above).
                 'empty_beam':empty_beam, # Empty beam sample name (from above).
                 'beam_center_guess': beam_center_guess, # Beam center guesses (from above).
                 'beamstopper_coordinates': beamstopper_coordinates, # Beam stopper coords (from above).
                 'target_detector_distances': target_detector_distances # Target detector distances (from above).
                 }},
                  'SANS-LLB':{ # Placeholder for another instrument configuration (SANS-LLB).
    'instrument': {'deadtime':1e5}, # Example instrument-specific parameter for LLB.
                    'experiment': {}, # Placeholder for experiment parameters specific to LLB.
                    'analysis': {}}} # Placeholder for analysis parameters specific to LLB.

# %% STEP 1: Load all HDF5 files and create an overview.
# This step scans the raw data directory, extracts relevant metadata (scan number,
# sample name, detector distance, etc.) from each HDF5 file, and compiles it
# into a structured dictionary ('class_files'). This overview is then saved
# to the 'result' dictionary.
config = configuration[instrument] # Select the configuration set for the chosen instrument.
class_files = org.list_files(config, result)

# %% STEP 2: Calculate transmission for samples (if applicable).
# This step determines the transmission of each sample by comparing its direct
# beam intensity to that of an empty beam measurement. Transmission correction
# is crucial for absolute intensity calibration and accounting for neutron
# absorption by the sample. The actual correction is applied later during normalization.
if trans_dist > 0: # Only run if a positive transmission measurement distance is specified.
    trans_calc(config, class_files, result)
else:
    print('No transmission measurements specified (trans_dist <= 0). Transmission correction will be skipped.')

# %% STEP 3: Organize files by detector distance.
# This step categorizes the loaded files based on their unique detector distances.
# It also copies relevant raw HDF5 files into detector-specific subfolders within
# the analysis directory for easier access during integration. Only detector
# distances specified in 'target_detector_distances' will be processed here.
result = org.select_detector_distances(config, class_files, result)


# %% STEP 4: Perform radial and azimuthal integration.
# This is the core data reduction step. It converts 2D detector images into
# 1D scattering curves (Intensity vs. scattering vector 'q').
# This step applies various corrections including dark field subtraction, empty
# cell subtraction, flat field correction, transmission correction, and optionally
# absolute intensity calibration.
# It also generates and saves integrated data files and plots based on user settings.

# Determine which detector distances to integrate based on the
# 'target_detector_distances' setting in the configuration.
if config['analysis']['target_detector_distances'] == 'all':
    # If 'all' is selected, iterate through all detector distances that were
    # successfully processed and organized in the 'result' dictionary during
    # the 'select_detector_distances' step (Step 3).
    processed_det_distances = []
    for key in result['overview'].keys():
        if key.startswith('det_files_'):
            # Extract the detector distance string (e.g., '1p6', '4p5', '18p0')
            # from the 'result' dictionary keys.
            processed_det_distances.append(key.replace('det_files_', ''))
else:
    # If specific distances are listed (e.g., [4.5, 18.0]), convert them to the
    # string format (e.g., '4.5' -> '4p5') to match the internal folder
    # and key naming conventions used by the pipeline.
    processed_det_distances = [str(d).replace('.', 'p') for d in config['analysis']['target_detector_distances']]

# Iterate through each selected/processed detector distance and perform integration.
# This loop ensures that all data for each specified detector distance is processed
# sequentially and correctly.
for det_str in processed_det_distances:
    # Print a clear message indicating which detector distance is currently being processed.
    print(f"\n--- Processing Detector Distance: {det_str.replace('p', '.')}m ---")
    # Call the 'set_integration' function from the 'integration' module.
    # The 'det_str' argument ensures that processing is confined to this specific detector distance.
    result = ri.set_integration(config, result, det_str)
