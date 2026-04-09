# -*- coding: utf-8 -*-
"""
DarePy-SANS: Corrections Module
Handles integration masks, pyFAI setup, and base instrumental normalizations.
Modernized for Virtual Referencing (no file copying) and Pluggable Physics.
"""
import numpy as np
from pathlib import Path
import sys
from utils import create_analysis_folder, save_results, load_hdf, find_hdf_by_identifier
import normalize_counts as norm


def load_standards(config, result, det):
    path_dir_an = create_analysis_folder(config)
    calibration_names = config['experiment']['calibration']

    class_det_files = result['overview']['det_files_' + det]
    class_all_files = result['overview']['all_files']
    physics = config.get('physics_corrections', {})

    standard_requirements = {
        'dark_current': physics.get('subtract_dark_current', False),
        'empty_cell': physics.get('subtract_empty_cell', False),
        'water': physics.get('perform_absolute_scaling', False),
        'water_cell': physics.get('perform_absolute_scaling', False)
    }

    for standard_key, is_required in standard_requirements.items():
        if not is_required:
            continue

        raw_id = calibration_names.get(standard_key)

        if isinstance(raw_id, dict):
            identifier = raw_id.get('default')
            if identifier is None:
                print(f"  [INFO] No default {standard_key} defined. Will look up per sample.")
                continue
        else:
            identifier = raw_id

        hdf_name = find_hdf_by_identifier(identifier, class_det_files)
        if hdf_name is None:
            hdf_name = find_hdf_by_identifier(identifier, class_all_files)

        if hdf_name:
            print(f"  -> Loading Standard: {identifier} ({standard_key})")
            img, img_variance = load_and_normalize(config, result, hdf_name, return_variance=True)

            result['integration'][standard_key] = img
            result['integration'][standard_key + '_variance'] = img_variance

            # [NEW FIX] Save the HDF name so we can look up its specific transmission later
            result['integration'][standard_key + '_hdf'] = hdf_name

        else:
            print(f"  [ERROR] '{standard_key}' ({identifier}) is required but not found in any scan!")
            sys.exit('Critical: Missing calibration data.')

    # ==============================================================
    # WATER CALIBRATION BUILDER
    # ==============================================================
    if physics.get('perform_absolute_scaling', False):
        img_w = result['integration']['water']
        var_w = result['integration']['water_variance']
        hdf_w = result['integration']['water_hdf']

        img_wc = result['integration']['water_cell']
        var_wc = result['integration']['water_cell_variance']
        hdf_wc = result['integration']['water_cell_hdf']

        # Apply Transmission to Water and its Empty Cell BEFORE subtraction
        if physics.get('apply_transmission', False):
            img_w = norm.normalize_transmission(config, hdf_w, result, img_w)
            var_w = np.square(norm.normalize_transmission(config, hdf_w, result, np.sqrt(var_w)))

            img_wc = norm.normalize_transmission(config, hdf_wc, result, img_wc)
            var_wc = np.square(norm.normalize_transmission(config, hdf_wc, result, np.sqrt(var_wc)))

        # 1. Subtract Empty Cell
        img_h2o = correct_EC(img_w, img_wc)
        img_h2o_var = var_w + var_wc

        # 2. Normalize by Thickness
        img_h2o = norm.normalize_thickness(config, hdf_w, result, img_h2o)
        img_h2o_var = np.square(norm.normalize_thickness(config, hdf_w, result, np.sqrt(img_h2o_var)))

        # 3. Safeguard against negative pixels before absolute scaling
        img_h2o[img_h2o <= 0] = 1e-6

        result['integration']['water'] = img_h2o
        result['integration']['water_variance'] = img_h2o_var

    save_results(path_dir_an, result)
    return result

def load_and_normalize(config, result, hdf_name, return_variance=False):
    path_hdf_raw = config['analysis']['path_hdf_raw']
    counts = load_hdf(path_hdf_raw, hdf_name, 'counts')

    if counts is None:
        return (np.array([[]]), np.array([[]])) if return_variance else np.array([[]])

    counts_var = counts.copy()

    counts = norm.normalize_deadtime(config, hdf_name, counts)
    counts_var = norm.normalize_deadtime(config, hdf_name, counts_var)

    dark_id = config.get('calibration_samples', {}).get('dark_current')
    dark_hdf = find_hdf_by_identifier(dark_id, result['overview']['all_files'])
    is_dark_file = (hdf_name == dark_hdf)

    if not is_dark_file and config.get('physics_corrections', {}).get('subtract_dark_current', False):
        dark_rate_img = result['integration'].get('dark_current')
        dark_rate_var = result['integration'].get('dark_current_variance')

        if dark_rate_img is not None:
            time_s = load_hdf(path_hdf_raw, hdf_name, 'time')
            scaled_dark = dark_rate_img * time_s
            scaled_dark_var = dark_rate_var * (time_s**2)

            counts = correct_dark(counts, scaled_dark)
            counts_var = counts_var + scaled_dark_var

    if is_dark_file:
        counts = norm.normalize_time(config, hdf_name, counts)
        counts_var = np.square(norm.normalize_time(config, hdf_name, np.sqrt(counts_var)))
    else:
        if config.get('physics_corrections', {}).get('normalize_to_monitor', True):
            counts = norm.normalize_flux(config, hdf_name, counts)
            counts_var = np.square(norm.normalize_flux(config, hdf_name, np.sqrt(counts_var)))

        counts = norm.normalize_attenuator(config, hdf_name, counts)
        counts_var = np.square(norm.normalize_attenuator(config, hdf_name, np.sqrt(counts_var)))

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
