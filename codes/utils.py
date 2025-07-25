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
import math
import pickle
import sys # Added for sys.exit in case of critical errors

# functions to load various values from hdf files

def load_hdf(path_hdf_raw, hdf_name, which_property):
    """
    Loads a specific property (metadata or counts data) from an HDF5 file.

    This function is designed to extract various experimental parameters and
    detector counts from the standard HDF5 file structure used at PSI SANS.
    It includes error handling for file access and data extraction.

    Args:
        path_hdf_raw (str): The base directory where the raw HDF5 files are located.
        hdf_name (str): The filename of the HDF5 file (e.g., 'scan_000123.hdf').
        which_property (str): The name of the property to load.
                              Supported values: 'beamstop_y', 'att', 'coll',
                              'detx', 'dety', 'wl', 'abs_time', 'spos',
                              'flux_monit', 'beam_stop', 'sample_name',
                              'time', 'moni', 'temp', 'counts'.

    Returns:
        Union[float, str, np.ndarray]: The loaded property value.
            - Floats for scalar properties like positions, attenuator, wavelength.
            - String for 'sample_name'.
            - NumPy array for 'counts', 'time', 'moni', 'temp'.
        Returns an empty string or 0 if a property is not found or is NaN (for 'temp').

    Raises:
        FileNotFoundError: If the specified HDF5 file does not exist.
        KeyError: If the specified 'which_property' path does not exist in the HDF5 file.
        Exception: For other unforeseen HDF5 reading errors.
    """
    full_hdf_path = os.path.join(path_hdf_raw, hdf_name)
    res = None # Initialize res to None

    try:
        # Open the hdf file in read mode
        with h5py.File(full_hdf_path, 'r') as file_hdf:
            # Handle scalar properties first
            if which_property == 'beamstop_y':
                prop = file_hdf['entry1/SANS/beam_stop/y_position'][0]
                res = check_dimension(prop)
            elif which_property == 'att':
                prop = file_hdf['entry1/SANS/attenuator/selection'][0]
                res = check_dimension(prop)
            elif which_property == 'coll':
                prop = file_hdf['/entry1/SANS/collimator/length'][0]
                res = check_dimension(prop) # in m
            elif which_property == 'detx':
                prop = file_hdf['/entry1/SANS/detector/x_position'][0]
                res = round(check_dimension(prop)/1000, 2) # convert from mm to m and round
            elif which_property == 'dety':
                prop = file_hdf['/entry1/SANS/detector/y_position'][0]
                res = round(check_dimension(prop)/1000, 2) # convert from mm to m and round
            elif which_property == 'wl':
                prop = file_hdf['/entry1/SANS/Dornier-VS/lambda'][0]
                res = check_dimension(prop)*10 # convert from nm to A
            elif which_property == 'abs_time':
                prop = file_hdf['/entry1/control/absolute_time'][0]
                res = check_dimension(prop)
            elif which_property == 'spos':
                prop = file_hdf['/entry1/sample/position'][0]
                res = check_dimension(prop)
            elif which_property == 'flux_monit':
                prop = file_hdf['/entry1/SANS/monitor2/counts'][0]
                res = check_dimension(prop)
            elif which_property == 'beam_stop': # This specific property returns a boolean/flag
                res = file_hdf['/entry1/SANS/beam_stop/out_flag'][0]
            elif which_property == 'sample_name':
                prop = file_hdf['/entry1/sample/name'][0]
                res = check_dimension(prop) # Will handle decoding bytes if necessary
            # Handle array properties
            elif which_property == 'time':
                prop = np.asarray(file_hdf['/entry1/SANS/detector/counting_time'])
                res = check_dimension(prop)  # in s
            elif which_property == 'moni':
                prop = np.asarray(file_hdf['/entry1/SANS/detector/preset'])
                res = check_dimension(prop)/1e4 # To have monitors as 1e4, as per existing logic
            elif which_property == 'temp': # Read in Celsius
                try:
                    prop = np.asarray(file_hdf['/entry1/sample/temperature'])
                    if np.isnan(prop).any(): # Check if any element is NaN
                        res = '' # Return empty string for NaN temperature
                    else:
                        res = check_dimension(prop)
                except KeyError:
                    res = '' # Return empty string if temperature path doesn't exist
                except Exception as e:
                    print(f"Warning: Could not load temperature for {hdf_name}: {e}")
                    res = ''
            elif which_property == 'counts':
                prop = np.array(file_hdf['entry1/SANS/detector/counts'])
                res = check_dimension(prop)
                # Correction to avoid negative values in counts data
                res[res < 0] = 0
            else:
                print(f"Warning: Unknown property '{which_property}' requested for file '{hdf_name}'.")
                res = None # Explicitly set to None for unsupported property

    except FileNotFoundError:
        print(f"Error: HDF5 file not found: {full_hdf_path}")
        sys.exit(1) # Exit the script as this is a critical error
    except KeyError as e:
        print(f"Error: Property path '{e}' not found in HDF5 file: {full_hdf_path} for '{which_property}'.")
        print("This might indicate an inconsistency in the HDF5 file structure or a typo in 'which_property'.")
        sys.exit(1) # Exit the script as this is a critical error
    except Exception as e:
        print(f"An unexpected error occurred while loading '{which_property}' from '{full_hdf_path}': {e}")
        sys.exit(1) # Exit the script for other critical errors

    return res

def check_dimension(prop):
    """
    Helper function to standardize data types and dimensions loaded from HDF5.

    It handles conversion from numpy byte strings to Python strings,
    rounding of floats, and ensures NumPy arrays are returned as float32.

    Args:
        prop (Union[np.bytes_, np.int32, np.float64, np.float32, np.ndarray]):
            The raw property value loaded from the HDF5 file.

    Returns:
        Union[float, str, np.ndarray]: The processed property value.
            - Decoded string if prop is a numpy bytes object.
            - Rounded float for scalar numeric types.
            - np.float32 array for 2D or higher dimension arrays.
            - Mean of values for 1D arrays (if numeric).
    """
    if isinstance(prop, np.bytes_):
        # Decode numpy byte string to standard Python string
        prop_str = prop.decode('utf-8')
        if prop_str == '':
            return ''
        # Attempt to convert to float if all characters are digits (or period for decimal)
        # This handles cases where numeric values might be stored as strings
        try:
            return round(float(prop_str), 2)
        except ValueError:
            return prop_str # Return as string if not purely numeric
    elif isinstance(prop, (np.int32, np.float64, np.float32)):
        # Round scalar numeric types to 2 decimal places
        return round(float(prop), 2)
    elif isinstance(prop, np.ndarray):
        if prop.ndim == 1:
            # For 1D arrays, return the mean if numeric, otherwise handle as string or pass through
            if prop.dtype.kind in 'biufc': # Check if numeric (bool, int, uint, float, complex)
                return round(float(np.mean(prop)), 2)
            else:
                return prop # Return the array if not numeric (e.g., array of strings)
        elif prop.ndim >= 2:
            # Ensure 2D or higher arrays are float32 for consistency
            return np.float32(prop)
    # If none of the above types, return as is (e.g., Python int, float, or other object)
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
