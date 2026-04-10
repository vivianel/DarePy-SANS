# -*- coding: utf-8 -*-
"""
DarePy-SANS: Corrections Module
Handles integration masks, pyFAI setup, and base instrumental normalizations.
Modernized for Virtual Referencing (no file copying) and Pluggable Physics.
"""
import numpy as np
from pathlib import Path
import sys
from utils import create_analysis_folder, save_results, load_hdf
import normalize_counts as norm


def load_standards(config, result, det):
    path_dir_an = create_analysis_folder(config)
    class_det_files = result['overview']['det_files_' + det]
    physics = config.get('physics_corrections', {})

    # --- THE SINGLE SOURCE OF TRUTH ---
    calib_map = result['overview'].get(f'calibration_map_{det}')

    if not calib_map:
        print(f"\n  [FATAL ERROR] Calibration map for {det}m is missing! Run prepare_input first.")
        sys.exit(1)

    standard_requirements = {
        'dark_current': physics.get('subtract_dark_current', False),
        'empty_cell': physics.get('subtract_empty_cell', False),
        'water': physics.get('perform_absolute_scaling', False),
        'water_cell': physics.get('perform_absolute_scaling', False)
    }

    for standard_key, is_required in standard_requirements.items():
        if not is_required:
            continue

        # --- LOAD DIRECTLY FROM THE MAP ---
        mapped_data = calib_map.get(standard_key)

        # Handle dictionaries (e.g., if you have multiple Empty Cells mapped)
        if isinstance(mapped_data, dict):
            identifier = config['experiment']['calibration'].get(standard_key)
            default_id = identifier.get('default') if isinstance(identifier, dict) else identifier
            hdf_name = mapped_data.get(default_id)
        else:
            hdf_name = mapped_data

        if hdf_name:
            idx = class_det_files['name_hdf'].index(hdf_name)
            scan_nr = class_det_files['scan'][idx]

            print(f"  -> Loading Standard: {standard_key} | Scan: {scan_nr} | Det: {det}m")

            is_dark_flag = (standard_key == 'dark_current')
            img, img_variance = load_and_normalize(config, result, hdf_name, is_dark=is_dark_flag, return_variance=True)

            result['integration'][standard_key] = img
            result['integration'][standard_key + '_variance'] = img_variance

            result['integration'][standard_key + '_hdf'] = hdf_name
            result['integration'][standard_key + '_scan'] = scan_nr

        else:
            print(f"\n  [FATAL ERROR] '{standard_key}' is required but NOT MAPPED at the {det}m distance!")
            sys.exit(1)

    save_results(path_dir_an, result)
    return result

def load_and_normalize(config, result, hdf_name, is_dark=False, return_variance=False):
    path_hdf_raw = config['analysis']['path_hdf_raw']
    counts = load_hdf(path_hdf_raw, hdf_name, 'counts')

    if counts is None:
        return (np.array([[]]), np.array([[]])) if return_variance else np.array([[]])

    counts_var = counts.copy()

    # 1. Deadtime applies to everything (electronics recovery)
    counts = norm.normalize_deadtime(config, hdf_name, counts)
    counts_var = norm.normalize_deadtime(config, hdf_name, counts_var)

    if is_dark:
        # DARK FILES: Electronic noise scales strictly with time.
        # This converts the dark image into a "Dark Rate" (counts per second).
        counts = norm.normalize_time(config, hdf_name, counts)
        counts_var = np.square(norm.normalize_time(config, hdf_name, np.sqrt(counts_var)))
    else:
        # NORMAL SAMPLES: Normalize by physical beam conditions
        counts = norm.normalize_attenuator(config, hdf_name, counts)
        counts_var = np.square(norm.normalize_attenuator(config, hdf_name, np.sqrt(counts_var)))

        if config.get('physics_corrections', {}).get('normalize_to_monitor', True):
            counts = norm.normalize_flux(config, hdf_name, counts)
            counts_var = np.square(norm.normalize_flux(config, hdf_name, np.sqrt(counts_var)))

    if return_variance:
        return counts, counts_var
    return counts

def correct_dark(img, dark):
    if img.shape != dark.shape:
        print(f"Warning: Dim mismatch ({img.shape} vs {dark.shape}). Dark correction skipped.")
        return img
    corrected_img = np.subtract(img, dark)
    return corrected_img

def correct_EC(img, EC):
    if img.shape != EC.shape:
        print(f"Warning: Dim mismatch ({img.shape} vs {EC.shape}). EC correction skipped.")
        return img
    corrected_img = np.subtract(img, EC)
    return corrected_img

def correct_flat_field(config, I):
    eff_file_name = config['instrument'].get('efficiency_map')
    if not eff_file_name:
        return I

    base_path = Path(config['analysis']['scripts_dir']).resolve()
    full_path = base_path / "codes" / eff_file_name

    if not full_path.exists():
        print(f"[DEBUG] Looking for Flat Field here: {full_path}")
        print(f"[WARNING] Flat field file missing. Skipping correction.")
        return I

    try:
        detector_eff = np.loadtxt(full_path, delimiter='\t')
        detector_eff[detector_eff <= 0] = 1.0
        return I / detector_eff
    except Exception as e:
        print(f"[ERROR] Flat field application failed: {e}")
        return I
