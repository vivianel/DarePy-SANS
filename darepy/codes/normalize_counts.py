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
    instrument = config['instrument']['name']
    path_hdf_raw = config['analysis']['path_hdf_raw']

    if hdf_name not in class_all['name_hdf']:
        return counts

    idx = list(class_all['name_hdf']).index(str(hdf_name))
    sample_name = class_all['sample_name'][idx]

    # --- CASE 1: SANS-I (Standard Lookup) ---
    # Also used for LLB if a positive transmission distance was specified
    if instrument == 'SANS-I' or (instrument == 'SANS-LLB' and config['experiment']['trans_dist'] > 0):
        trans_value = class_all['transmission'][idx]

    # --- CASE 2: SANS-LLB (Runtime Calculation) ---
    elif instrument == 'SANS-LLB' and config['experiment']['trans_dist'] == False:
        import transmission  # Local import to avoid circular dependency

        # Load detector distance to find the correct reference EB
        det_dist = load_hdf(path_hdf_raw, hdf_name, 'detx')

        try:
            # Look up the specific mask and EB reference for this distance
            mask = result['transmission'][f'mask_{det_dist}']
            EB_ref = result['transmission'][f'mean_EB_{det_dist}']

            # Normalize the raw counts for transmission math (Deadtime/Flux/Atten)
            img_norm = transmission.normalize_trans(config, result, hdf_name, counts.copy())

            # Calculate T = Sum(Sample * Mask) / Sum(EB * Mask)
            sum_counts = float(np.sum(np.multiply(img_norm, mask)))
            trans_value = round(sum_counts / EB_ref, 3)

            # Update the global record so it appears in final reports
            class_all['transmission'][idx] = trans_value

        except KeyError:
            print(f"  [ERROR] No EB reference found for distance {det_dist}m. Skipping T-correction for {sample_name}.")
            return counts

    # --- FINAL APPLICATION ---
    # Check if we have a valid numerical transmission value
    if isinstance(trans_value, (float, int, np.float64)) and trans_value > 0:
        # Intensity scales by 1/T
        return counts / trans_value

    return counts

def normalize_thickness(config, hdf_name, result, counts):
    """Normalizes counts by sample thickness (cm)."""
    class_all = result['overview']['all_files']

    if hdf_name not in class_all['name_hdf']:
        return counts

    idx = list(class_all['name_hdf']).index(str(hdf_name))
    thickness = class_all['thickness_cm'][idx]

    if thickness is None or thickness <= 0:
        return counts

    return counts / thickness
