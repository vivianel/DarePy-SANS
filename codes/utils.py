# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 09:23:28 2023

@author: lutzbueno_v
"""
"""
This module provides essential utility functions for the SANS data reduction
pipeline. It handles loading data from HDF5 files, managing the analysis
folder structure, saving results, and performing basic data manipulation
like smoothing.
"""

import numpy as np
import h5py
import os
import pickle
import sys # Added for sys.exit in case of critical errors

# ==========================================
# GLOBAL CACHE (Prevents severe pipeline slowdowns)
# ==========================================
_CACHED_CONFIG = None
_CACHED_REGISTRY = None

def load_hdf(path_hdf_raw, hdf_name, which_property):
    """
    Loads a specific property from an HDF5 file.
    Automatically pulls the instrument name and monitor path from the YAML
    registries without requiring them as function arguments.
    """
    global _CACHED_CONFIG, _CACHED_REGISTRY

    # 1. LAZY LOADING: Only read the YAML files from the hard drive ONCE.
    if _CACHED_CONFIG is None or _CACHED_REGISTRY is None:
        from utils import load_config, load_instrument_registry # Ensure imports match your structure
        _CACHED_CONFIG = load_config()
        _CACHED_REGISTRY = load_instrument_registry()

    # 2. DYNAMICALLY PULL FROM YAML
    instrument = _CACHED_CONFIG['instrument_setup']['which_instrument']

    try:
        registry_monitor_path = _CACHED_REGISTRY[instrument]['monitor_path']
    except KeyError:
        print(f"\n[ERROR] 'monitor_path' is missing for {instrument} in instrument_registry.yaml!")
        sys.exit(1)

    full_hdf_path = os.path.join(path_hdf_raw, hdf_name)

    # 3. HDF5 PATH DICTIONARIES
    PATHS = {
        'SANS-I': {
            'beamstop_y':  'entry1/SANS/beam_stop/y_position',
            'att':         'entry1/SANS/attenuator/selection',
            'coll':        '/entry1/SANS/collimator/length',
            'detx':        '/entry1/SANS/detector/x_position',
            'dety':        '/entry1/SANS/detector/y_position',
            'wl':          '/entry1/SANS/Dornier-VS/lambda',
            'abs_time':    '/entry1/control/absolute_time',
            'spos':        '/entry1/sample/position',
            'beam_stop':   '/entry1/SANS/beam_stop/out_flag',
            'sample_name': '/entry1/sample/name',
            'time':        '/entry1/SANS/detector/counting_time',
            'moni':        '/entry1/SANS/detector/preset',
            'temp':        '/entry1/sample/temperature',
            'counts':      'entry1/SANS/detector/counts',
            'flux_monit':  registry_monitor_path  # read from instrument_registry
        },
        'SANS-LLB': {
            'beamstop_y':  'entry0/SANS-LLB/beam_stop/y',
            'att':         'entry0/SANS-LLB/attenuator/selection',
            'coll':        '/entry0/SANS-LLB/collimator/geometry/size',
            'detx':        '/entry0/SANS-LLB/central_detector/distance',
            'dety':        '/entry0/SANS-LLB/central_detector/x',
            'wl':          '/entry0/SANS-LLB/velocity_selector/wavelength',
            'abs_time':    '/entry0/control/absolute_time',
            'spos':        '/entry0/sample/position',
            'beam_stop':   '/entry0/SANS-LLB/beam_stop/out_flag',
            'sample_name': '/entry0/sample/name',
            'time':        '/entry0/control/count_time',
            'moni':        '/entry0/control/preset',
            'temp':        '/entry0/sample/temperature',
            'counts':      'entry0/central_data/data',
            'counts_left': 'entry0/left_data/data',
            'counts_bottom':'entry0/bottom_data/data',
            'flux_monit':  registry_monitor_path  # read from instrument_registry
        }
    }

    try:
        with h5py.File(full_hdf_path, 'r') as file_hdf:

            if which_property not in PATHS[instrument]:
                return None

            hdf_internal_path = PATHS[instrument][which_property]

            if which_property == 'temp' and hdf_internal_path not in file_hdf:
                return ''

            raw_data = file_hdf[hdf_internal_path]

            if which_property in ['time', 'moni', 'temp', 'counts', 'counts_left', 'counts_bottom']:
                prop = np.asarray(raw_data)
            else:
                prop = raw_data[0]

            # 4. MATH & CONVERSIONS
            if which_property == 'temp' and np.isnan(prop).any():
                return ''

            if which_property == 'sample_name':
                if isinstance(prop, bytes):
                    return prop.decode('utf-8')
                return str(prop)

            prop = check_dimension(prop)

            if which_property == 'coll' and instrument == 'SANS-LLB':
                prop = prop / 1000.0
            elif which_property in ['detx', 'dety']:
                prop = round(prop / 1000.0, 2)
            elif which_property == 'wl':
                prop = prop * 10.0
            elif which_property == 'moni':
                prop = prop / 1e4

            return prop

    except FileNotFoundError:
        print(f"\n[ERROR] HDF5 file not found: {full_hdf_path}")
        sys.exit(1)
    except KeyError as e:
        print(f"\n[ERROR] Path missing in {hdf_name}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error reading '{which_property}' in {hdf_name}: {e}")
        sys.exit(1)


def check_dimension(prop):
    """
    Helper function to standardize data types and dimensions loaded from HDF5.
    Safely extracts 0D scalars, averages 1D arrays, and casts 2D+ arrays to float32
    WITHOUT forcing arbitrary rounding.
    """
    # 1. Handle Byte Strings
    if isinstance(prop, (bytes, np.bytes_)):
        prop_str = prop.decode('utf-8').strip()
        if prop_str == '':
            return ''
        # If it's a number hiding in a string, convert it, but don't round!
        try:
            return float(prop_str)
        except ValueError:
            return prop_str

    # 2. Extract standard Python floats from 0D numpy types (np.float64, np.int32, etc.)
    elif isinstance(prop, np.generic) and not isinstance(prop, np.ndarray):
        return float(prop)

    # 3. Handle NumPy Arrays
    elif isinstance(prop, np.ndarray):
        if prop.ndim == 0:
            # It's an array with a single item (e.g., array(6.0))
            return float(prop)

        elif prop.ndim == 1:
            # For 1D numeric arrays (like temperature logs), return the exact mean
            if prop.dtype.kind in 'biufc':
                return float(np.mean(prop))
            return prop

        elif prop.ndim >= 2:
            # Ensure 2D detector images are strictly float32 for memory efficiency
            return np.float32(prop)

    # 4. Fallback for standard Python ints/floats
    return prop

def create_analysis_folder(config):
    """
    Creates a unique analysis folder based on the configured path and an optional ID.

    This ensures that results from different analysis runs can be kept separate.

    Args:
        config (dict): The configuration dictionary containing 'analysis' settings,
                       specifically 'path_dir' and 'add_id'.

    Returns:
        str: The absolute path to the newly created or existing analysis folder.
    """
    add_id = config['analysis']['add_id']
    path_dir = config['analysis']['path_dir']

    if add_id: # If add_id is not an empty string
        path_dir_an = os.path.join(path_dir, f'analysis_{add_id}/')
    else:
        path_dir_an = os.path.join(path_dir, 'analysis/')

    # Create the directory if it does not exist. makedirs creates intermediate dirs too.
    if not os.path.exists(path_dir_an):
        try:
            os.makedirs(path_dir_an)
            print(f"Created analysis folder: {path_dir_an}")
        except OSError as e:
            print(f"Error creating analysis folder {path_dir_an}: {e}")
            sys.exit(1) # Critical error, exit
    else:
        # Comment out or remove this line to suppress the warning
        # print(f"Analysis folder already exists: {path_dir_an}")
        pass # Or add a pass statement

    return path_dir_an

def save_results(path_save, result):
    """
    Saves the 'result' dictionary, containing all analysis outputs, to a .npy file
    using Python's pickle module.

    This allows the entire state of the analysis results to be persistently stored
    and reloaded.

    Args:
        path_save (str): The directory path where the result file should be saved.
                         Typically the main analysis folder.
        result (dict): The dictionary containing all aggregated analysis results.

    Returns:
        dict: The updated result dictionary (same as input).

    Raises:
        IOError: If there's an issue writing the file to disk.
    """
    save_file_path = os.path.join(path_save, 'result.npy')
    try:
        # 'wb' mode for writing in binary
        with open(save_file_path, 'wb') as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # print(f"Results saved to: {save_file_path}") # Optional: uncomment for verbose output
    except IOError as e:
        print(f"Error saving results to {save_file_path}: {e}")
        sys.exit(1) # Critical error, exit
    return result

def smooth(y, box_pts):
    """
    Applies a simple moving average (boxcar) smoothing to a 1D array.

    Args:
        y (np.ndarray): The 1D NumPy array to be smoothed.
        box_pts (int): The number of points to use in the smoothing window.
                       Must be a positive integer.

    Returns:
        np.ndarray: The smoothed 1D NumPy array.
    """
    if not isinstance(y, np.ndarray) or y.ndim != 1:
        print("Warning: 'y' must be a 1D NumPy array for smoothing.")
        return y # Return original if input is not valid
    if not isinstance(box_pts, int) or box_pts <= 0:
        print("Warning: 'box_pts' must be a positive integer for smoothing. No smoothing applied.")
        return y # Return original if box_pts is not valid

    box = np.ones(box_pts) / box_pts
    # 'same' mode pads the array so that the output has the same length as y
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def get_flexible_value(config_block, sample_name, default_fallback=None):
    """
    Look up values in a case-insensitive way.
    Handles: {'CTAB': 0.1, 'default': 0.1}
    """
    if not isinstance(config_block, dict):
        return config_block if config_block is not None else default_fallback

    # Clean the input sample name
    target = str(sample_name).strip().lower()

    # Create a lower-case version of the dictionary keys for matching
    lower_dict = {str(k).lower(): v for k, v in config_block.items()}

    if target in lower_dict:
        return lower_dict[target]

    return config_block.get('default', default_fallback)

def find_hdf_by_identifier(identifier, class_all_files):
    """
    Finds an HDF5 filename whether the identifier is a
    sample_name (str) or a scan_number (int).
    Includes automatic whitespace cleaning for robust string matching.
    """
    if identifier is None:
        return None

    # CASE 1: The identifier is a Scan Number (e.g., 83905)
    if isinstance(identifier, int):
        if identifier in class_all_files['scan']:
            idx = class_all_files['scan'].index(identifier)
            return class_all_files['name_hdf'][idx]

    # CASE 2: The identifier is a Sample Name (e.g., 'CTAB' or 'EC')
    else:
        clean_target = str(identifier).strip()

        # Clean the list from the metadata to ensure a perfect match
        clean_sample_list = [str(name).strip() for name in class_all_files['sample_name']]

        if clean_target in clean_sample_list:
            idx = clean_sample_list.index(clean_target)
            return class_all_files['name_hdf'][idx]

    # If not found in either column
    return None



import re

def find_hdf_filename(path_raw_dir, scan_number):
    """
    Automatically finds the HDF filename for a given scan number,
    regardless of the instrument prefix, year, or zero-padding.

    Args:
        path_raw_dir (str): Path to the raw_data folder.
        scan_number (int): The integer scan number to look for.

    Returns:
        str: The exact filename (e.g., 'sans2022n083907.hdf') or None if not found.
    """
    if not os.path.exists(path_raw_dir):
        print(f"[ERROR] Directory not found: {path_raw_dir}")
        return None

    # Regex explanation:
    # (\d+) captures the consecutive digits.
    # \.hdf$ ensures those digits are immediately followed by .hdf at the end of the line.
    pattern = re.compile(r'(\d+)\.hdf$', re.IGNORECASE)

    for filename in os.listdir(path_raw_dir):
        match = pattern.search(filename)
        if match:
            # Convert the matched string (e.g., "083907") to an integer (83907)
            file_scan_nr = int(match.group(1))

            if file_scan_nr == scan_number:
                return filename

    return None


import yaml

# ==========================================
# MASTER CONFIGURATION PATHS
# Automatically dynamically detects the current directory.
# ==========================================

# 1. Find the exact folder where utils.py is physically located
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
#Go back one folder above it (e.g., /.../2022_2358_beamtime)
CURRENT_DIR = os.path.dirname(CURRENT_DIR)

# 2. Attach the YAML filenames to that folder path
DEFAULT_CONFIG_PATH = os.path.join(CURRENT_DIR, "config_experiment.yaml")
DEFAULT_REGISTRY_PATH = os.path.join(CURRENT_DIR, "instrument_registry.yaml")


def load_config(filepath=DEFAULT_CONFIG_PATH):
    """Loads the main experiment YAML configuration file."""
    try:
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"\n[ERROR] config_experiment.yaml not found at:\n{filepath}")
        print("Please ensure the YAML file is in the same directory as utils.py")
        sys.exit(1)

def load_instrument_registry(filepath=DEFAULT_REGISTRY_PATH):
    """Loads the instrument hardware registry YAML."""
    try:
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"\n[ERROR] instrument_registry.yaml not found at:\n{filepath}")
        print("Please ensure the YAML file is in the same directory as utils.py")
        sys.exit(1)
