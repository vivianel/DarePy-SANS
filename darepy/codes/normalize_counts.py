# -*- coding: utf-8 -*-
"""
DarePy-SANS: Normalization Module
Handles scalar and array-based normalizations for SANS data.
"""

import numpy as np
from utils import load_hdf


def normalize_time(config, hdf_name, counts):
    """Normalizes detector counts by measurement time."""
    path_hdf_raw = config['analysis']['path_hdf_raw']
    meas_time = load_hdf(path_hdf_raw, hdf_name, 'time')
    if meas_time is None or meas_time <= 0:
        return counts
    return counts / meas_time

def normalize_deadtime(config, hdf_name, counts):
    """Corrects counts for detector deadtime (Non-paralyzable model)."""
    detector_deadtime = config['instrument']['deadtime']
    path_hdf_raw = config['analysis']['path_hdf_raw']
    meas_time = load_hdf(path_hdf_raw, hdf_name, 'time')

    if meas_time is None or meas_time <= 0:
        return counts

    total_counts = np.sum(counts)
    count_rate = total_counts / meas_time
    deadtime_factor = count_rate * detector_deadtime

    if deadtime_factor >= 1.0:
        print(f"[WARNING] Deadtime saturation for {hdf_name}. Correction skipped.")
        return counts

    counts_corrected = counts / (1 - deadtime_factor)
    return counts_corrected

def normalize_flux(config, hdf_name, counts):
    """Normalizes counts by incident monitor flux."""
    path_hdf_raw = config['analysis']['path_hdf_raw']
    flux_mon = load_hdf(path_hdf_raw, hdf_name, 'flux_monit')

    if flux_mon is None or flux_mon <= 0:
        return counts

    return counts / flux_mon

def normalize_attenuator(config, hdf_name, counts):
    """Corrects for attenuator transmission factor."""
    path_hdf_raw = config['analysis']['path_hdf_raw']
    att_setting = int(load_hdf(path_hdf_raw, hdf_name, 'att'))
    list_attenuation = config['instrument']['list_attenuation']

    att_key = str(att_setting)
    if att_key not in list_attenuation:
        return counts

    factor = float(list_attenuation[att_key])
    return counts / factor if factor > 0 else counts

def normalize_transmission(config, hdf_name, result, counts):
    """
    Corrects for sample-specific neutron transmission.
    - SANS-I: Uses pre-calculated values from metadata (lookup).
    - SANS-LLB: Performs on-the-fly calculation if trans_dist < 0.
    """
    class_all = result['overview']['all_files']

    if hdf_name not in class_all['name_hdf']:
        return counts

    idx = list(class_all['name_hdf']).index(str(hdf_name))
    trans_value = class_all['transmission'][idx]

    # Check if we have a valid numerical transmission value
    if isinstance(trans_value, (float, int, np.float64)) and trans_value > 0:
        # Intensity scales by 1/T
        return counts / trans_value

    return counts



def normalize_thickness(config, hdf_name, result, counts):
    """Normalizes counts by sample thickness (cm)."""
    class_all = result['overview']['all_files']

    idx = list(class_all['name_hdf']).index(str(hdf_name))
    thickness = class_all['thickness_cm'][idx]

    if thickness is None or thickness <= 0:
        return counts

    return counts / thickness
