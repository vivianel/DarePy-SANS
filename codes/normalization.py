# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 12:38:56 2022

@author: lutzbueno_v
"""
"""
This module contains functions for normalizing SANS detector counts by various
experimental parameters. Normalizations include corrections for measurement time,
detector deadtime, incident flux, attenuator transmission, sample transmission,
and sample thickness, allowing for comparison of scattering intensities.
"""

import numpy as np
from utils import load_hdf # Importing load_hdf from the utils module

def normalize_time(config, hdf_name, counts):
    """
    Normalizes detector counts by the measurement (counting) time.

    This corrects for variations in the exposure duration of each measurement.

    Args:
        config (dict): The configuration dictionary, containing 'analysis' settings
                       like 'path_hdf_raw'.
        hdf_name (str): The filename of the HDF5 file from which 'counts' originated.
        counts (np.ndarray): The raw or partially normalized 2D detector counts array.

    Returns:
        np.ndarray: The counts array normalized by measurement time.
    """
    path_hdf_raw = config['analysis']['path_hdf_raw']
    # Load the counting time for the given HDF5 file
    meas_time = load_hdf(path_hdf_raw, hdf_name, 'time')

    # Ensure measurement time is valid to prevent division by zero
    if meas_time is None or meas_time <= 0:
        print(f"Warning: Invalid or zero measurement time ({meas_time}) for {hdf_name}. Time normalization skipped.")
        return counts # Return original counts if time is invalid

    # Normalize counts by measurement time
    normalized_counts = (counts / meas_time)
    return normalized_counts


def normalize_deadtime(config, hdf_name, counts):
    """
    Corrects detector counts for detector deadtime effects.

    Deadtime correction accounts for the period after a detection event
    during which the detector cannot register another event, leading to
    under-counting at high count rates.

    Args:
        config (dict): The configuration dictionary, containing 'instrument' settings
                       like 'deadtime' and 'analysis' settings like 'path_hdf_raw'.
        hdf_name (str): The filename of the HDF5 file from which 'counts' originated.
        counts (np.ndarray): The raw or partially normalized 2D detector counts array.

    Returns:
        np.ndarray: The counts array corrected for detector deadtime.
    """
    detector_deadtime = config['instrument']['deadtime'] # Deadtime in seconds
    # Sum of all counts on the detector for this frame/image
    total_counts = np.sum(counts)
    path_hdf_raw = config['analysis']['path_hdf_raw']
    meas_time = load_hdf(path_hdf_raw, hdf_name, 'time') # Measurement time in seconds

    if meas_time is None or meas_time <= 0:
        print(f"Warning: Invalid or zero measurement time ({meas_time}) for {hdf_name}. Deadtime normalization skipped.")
        return counts

    # Calculate the deadtime correction factor: 1 / (1 - (deadtime / measurement_time) * total_counts)
    # This formula assumes a non-paralyzable detector model often used for SANS.
    deadtime_factor = (detector_deadtime / meas_time) * total_counts
    if deadtime_factor >= 1.0: # Check for division by zero or negative correction
        print(f"Warning: Deadtime correction factor ({deadtime_factor}) is too high for {hdf_name}. Deadtime normalization skipped. Check deadtime or counts/time.")
        return counts # Return original counts if correction is problematic

    normalized_counts = counts / (1 - deadtime_factor)
    return normalized_counts

def normalize_flux(config, hdf_name, counts):
    """
    Normalizes detector counts by the incident neutron flux (monitor counts).

    This corrects for variations in the incoming beam intensity during the experiment.

    Args:
        config (dict): The configuration dictionary, containing 'analysis' settings
                       like 'path_hdf_raw'.
        hdf_name (str): The filename of the HDF5 file from which 'counts' originated.
        counts (np.ndarray): The raw or partially normalized 2D detector counts array.

    Returns:
        np.ndarray: The counts array normalized by the incident flux.
    """
    path_hdf_raw = config['analysis']['path_hdf_raw']
    # Load the flux monitor counts (e.g., from monitor2)
    flux_mon = load_hdf(path_hdf_raw, hdf_name, 'flux_monit')

    # Ensure flux monitor value is valid to prevent division by zero
    if flux_mon is None or flux_mon <= 0:
        print(f"Warning: Invalid or zero flux monitor ({flux_mon}) for {hdf_name}. Flux normalization skipped.")
        return counts # Return original counts if flux is invalid

    # Normalize counts by the flux monitor
    normalized_counts = (counts / flux_mon)
    return normalized_counts

def normalize_attenuator(config, hdf_name, counts):
    """
    Corrects detector counts for the transmission factor of the attenuator used.

    Different attenuators reduce the beam intensity by specific factors,
    which need to be accounted for to get true scattering intensity.

    Args:
        config (dict): The configuration dictionary, containing 'instrument' settings
                       like 'list_attenuation' and 'analysis' settings like 'path_hdf_raw'.
        hdf_name (str): The filename of the HDF5 file from which 'counts' originated.
        counts (np.ndarray): The raw or partially normalized 2D detector counts array.

    Returns:
        np.ndarray: The counts array corrected by the attenuator's transmission factor.
    """
    path_hdf_raw = config['analysis']['path_hdf_raw']
    # Load the attenuator setting (e.g., '0', '1', '2' etc.)
    attenuator_setting = int(load_hdf(path_hdf_raw, hdf_name, 'att'))

    # Load the list of attenuation factors from the config
    list_attenuation = config['instrument']['list_attenuation']

    # Get the corresponding attenuation factor from the dictionary
    # The keys in list_attenuation are strings '0', '1', etc., so convert attenuator_setting to string
    attenuator_key = str(attenuator_setting)
    if attenuator_key not in list_attenuation:
        print(f"Warning: Attenuator setting '{attenuator_key}' not found in configuration for {hdf_name}. Attenuator correction skipped.")
        return counts # Return original counts if attenuator setting is not defined

    attenuation_factor = float(list_attenuation[attenuator_key])

    if attenuation_factor <= 0:
        print(f"Warning: Attenuation factor for '{attenuator_key}' is zero or negative for {hdf_name}. Attenuator correction skipped.")
        return counts

    # Correct counts by dividing by the attenuation factor
    normalized_counts = counts / attenuation_factor
    return normalized_counts


def normalize_transmission(config, hdf_name, result, counts):
    """
    Corrects detector counts for the transmission of the sample itself.

    This is important for quantitative analysis, as samples absorb neutrons
    differently.

    Args:
        config (dict): The configuration dictionary.
        hdf_name (str): The filename of the HDF5 file for the current sample.
        result (dict): The results dictionary, which contains 'overview' with
                       'all_files' and 'trans_files' that hold transmission data.
        counts (np.ndarray): The partially normalized 2D detector counts array.

    Returns:
        np.ndarray: The counts array corrected by the sample's transmission.
    """
    # Access the overall list of all files, which should contain transmission info
    # 'class_trans' here refers to 'all_files' that now includes transmission
    class_all_files = result['overview']['all_files']

    if hdf_name in class_all_files['name_hdf']:
        # Find the index of the current HDF file in the list of all files
        try:
            idx_file = list(class_all_files['name_hdf']).index(str(hdf_name))
        except ValueError:
            print(f"Warning: HDF file '{hdf_name}' not found in 'all_files' overview. Transmission correction skipped.")
            return counts

        trans_value = class_all_files['transmission'][idx_file]
        sample_name = class_all_files['sample_name'][idx_file]

        # Check if transmission value is a valid float and greater than zero
        if isinstance(trans_value, (float, np.float32, np.float64)) and trans_value > 0:
            normalized_counts = counts / trans_value
            print(f"Sample '{sample_name}' (Scan: {class_all_files['scan'][idx_file]}) corrected by Transmission = {trans_value:.3f}")
            return normalized_counts
        else:
            print(f"Warning: Invalid or missing transmission value ('{trans_value}') for sample '{sample_name}' (Scan: {class_all_files['scan'][idx_file]}). Transmission correction skipped.")
            return counts # Return original counts if transmission is invalid
    else:
        print(f"Warning: HDF file '{hdf_name}' not found in main overview. Transmission correction skipped.")
        return counts


def normalize_thickness(config, hdf_name, result, counts):
    """
    Normalizes detector counts by the sample's thickness.

    This is essential for obtaining scattering intensities in absolute units
    (e.g., cm^-1), enabling direct comparison between samples of different thicknesses.

    Args:
        config (dict): The configuration dictionary, containing 'experiment' settings
                       like 'sample_thickness'.
        hdf_name (str): The filename of the HDF5 file for the current sample.
        result (dict): The results dictionary, which contains 'overview' with 'all_files'.
        counts (np.ndarray): The partially normalized 2D detector counts array.

    Returns:
        np.ndarray: The counts array normalized by the sample's thickness.
    """
    class_all_files = result['overview']['all_files']
    sample_name = "Unknown Sample" # Default for logging if not found
    thickness = 0.1 # Default thickness as per caller_radial_integration.py config

    if hdf_name in class_all_files['name_hdf']:
        try:
            idx_file = list(class_all_files['name_hdf']).index(str(hdf_name))
        except ValueError:
            print(f"Warning: HDF file '{hdf_name}' not found in 'all_files' overview during thickness lookup. Using default thickness.")
        
        thickness = class_all_files['thickness_cm'][idx_file]
        sample_name = class_all_files['sample_name'][idx_file]
            
            # else: thickness remains 0.1 (default set above)
    else:
        print(f"Warning: HDF file '{hdf_name}' not found in main overview for thickness normalization. Using default thickness.")

    # Validate thickness to prevent division by zero
    if thickness is None or thickness <= 0:
        print(f"Warning: Invalid or zero thickness ({thickness:.3f} cm) for sample '{sample_name}'. Thickness normalization skipped.")
        return counts # Return original counts if thickness is invalid

    normalized_counts = counts / thickness
    # print(f"Sample '{sample_name}' corrected by thickness = {thickness:.3f} cm") # Original commented print
    return normalized_counts
