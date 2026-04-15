# -*- coding: utf-8 -*-
"""
DarePy-SANS: Prepare Input Module
Handles HDF5 parsing, metadata extraction, and data organization.
Modernized for Pluggable Physics Pipeline and Live-Mode efficiency.
"""
import os
import re
import sys
import csv  # Replaced pandas with native csv
import numpy as np
import pickle
from tabulate import tabulate
from utils import find_hdf_by_identifier, parse_scan_list, load_hdf, create_analysis_folder, save_results

def list_files(config, result):
    # Initialize dictionary for storing metadata
    class_files = {
        'name_hdf': [], 'scan': [], 'sample_name': [], 'att': [],
        'beamstop_y': [], 'coll_m': [], 'wl_A': [], 'detx_m': [],
        'dety_m': [],  'moni_e4': [], 'time_s': [], 'thickness_cm': [],
        'frame_nr': [], 'temp_C': []
    }

    path_hdf_raw = config['analysis']['path_hdf_raw']
    raw_exclude = config['analysis'].get('exclude_files', [])
    exclude_files = parse_scan_list(raw_exclude)

    files_dict = {}

    for r, d, f in os.walk(path_hdf_raw):
        f.sort()
        for file in f:
            if file.endswith('.hdf'):
                scan_match = re.findall(r"\D(\d{6})\D", file)
                if not scan_match:
                    continue

                scan_nr = int(scan_match[0])

                if scan_nr in files_dict:
                    if '_mod' in file:
                        files_dict[scan_nr] = file
                else:
                    files_dict[scan_nr] = file

    files = sorted(list(files_dict.values()))

    for file in files:
        scan_nr = int(re.findall(r"\D(\d{6})\D", file)[0])

        if scan_nr not in exclude_files:
            class_files['name_hdf'].append(file)
            class_files['scan'].append(scan_nr)
            class_files['att'].append(load_hdf(path_hdf_raw, file, 'att'))
            class_files['beamstop_y'].append(load_hdf(path_hdf_raw, file, 'beamstop_y'))
            class_files['coll_m'].append(load_hdf(path_hdf_raw, file, 'coll'))
            class_files['time_s'].append(load_hdf(path_hdf_raw, file, 'time'))
            class_files['moni_e4'].append(load_hdf(path_hdf_raw, file, 'moni'))
            class_files['temp_C'].append(load_hdf(path_hdf_raw, file, 'temp'))
            class_files['detx_m'].append(load_hdf(path_hdf_raw, file, 'detx'))
            class_files['dety_m'].append(load_hdf(path_hdf_raw, file, 'dety'))
            class_files['wl_A'].append(load_hdf(path_hdf_raw, file, 'wl'))

            sample_name = load_hdf(path_hdf_raw, file, 'sample_name')
            class_files['sample_name'].append(sample_name)

            res = load_hdf(path_hdf_raw, file, 'counts')
            class_files['frame_nr'].append(res.shape[0] if (res is not None and res.ndim > 2) else 1)

            list_thickness = config['experiment'].get('sample_thickness', {})
            class_files['thickness_cm'].append(list_thickness.get(sample_name, list_thickness.get('all', 0.1)))

    path_dir_an = create_analysis_folder(config)
    save_list_files(path_dir_an, path_dir_an, class_files, 'all_files', result)

    save_file = os.path.join(path_dir_an, 'config.npy')
    with open(save_file, 'wb') as handle:
        pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return class_files

def save_list_files(path_save, path_dir_an, class_files, name, result):
    """Saves metadata to CSV and prints a table WITHOUT pandas."""

    save_file = os.path.join(path_save, f"{name}.csv")

    # Get keys for the header
    keys = class_files.keys()
    # Transpose dictionary of lists into a list of rows
    rows = zip(*[class_files[k] for k in keys])

    # 1. Write the FULL table to CSV using native 'csv' module
    with open(save_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(rows)

    # 2. Print TRUNCATED table for the console
    # Re-generating rows for tabulate (it only needs the first 20)
    truncated_rows = list(zip(*[class_files[k] for k in keys]))[:20]
    data_console = tabulate(truncated_rows, headers=keys, tablefmt='psql')

    print(f"\n[INFO] Saving file list: {name}.csv")
    print(data_console)

    if len(class_files['name_hdf']) > 20:
        print(f"... (Showing first 20 of {len(class_files['name_hdf'])} files. Check {name}.csv!)")

    print("-" * 20)
    result['overview'][name] = class_files
    save_results(path_dir_an, result)

def select_detector_distances(config, class_files, result):
    """Groups files by detector distance by creating metadata lists (no copying)."""
    calibration = config['experiment']['calibration']
    path_dir_an = create_analysis_folder(config)

    unique_det = np.unique(class_files['detx_m'])

    for jj in unique_det:
        string = str(jj).replace('.', 'p')
        # We still create the detector folder, but ONLY to store the RESULTS/PLOTS
        path_det = os.path.join(path_dir_an, f"det_{string}/")

        list_det = list(class_files.keys())
        class_det = {key: [] for key in list_det}

        if not os.path.exists(path_det):
            os.mkdir(path_det)

        for ii in range(len(class_files['detx_m'])):
            # Filter criteria for valid measurements
            if (class_files['detx_m'][ii] == jj and
                class_files['beamstop_y'][ii] > -30 and
                class_files['time_s'][ii] > 0):

                # We simply append the file metadata to our virtual list
                for iii in list_det:
                    class_det[iii].append(class_files[iii][ii])

        print('\n' + '%' * 50)
        print(f'   For sample-detector distance: {string}m')
        print('%' * 50)

        # --- NEW: Build the Calibration Map for this distance ---
        calib_map = {}

        print('\n  --- Checking Calibration Dependencies ---')
        for calib_key, sample_id in calibration.items():
            # The checker will now return the exact HDF filename it found
            mapped_hdf = check_calibration_dependency(calib_key, sample_id, class_det, config, string)

            # If a file was successfully mapped, store it!
            if mapped_hdf is not None:
                calib_map[calib_key] = mapped_hdf

        # Save this definitive map directly into the results dictionary
        result['overview'][f'calibration_map_{string}'] = calib_map

        # This CSV file and the 'result' dictionary now act as our Single Source of Truth
        save_list_files(path_det, path_dir_an, class_det, f'det_files_{string}', result)

    return result

def check_calibration_dependency(calib_key, file_id, class_det, config, det_string):
    """
    Upgraded Pre-flight check:
    1. Searches STRICTLY within the current detector distance.
    2. Triggers an immediate fatal error if a required 2D standard is missing.
    3. RETURNS the exact HDF filename to build the Calibration Map.
    """
    physics = config.get('physics_corrections', {})

    dependency_map = {
        'dark_current': physics.get('subtract_dark_current', False),
        'empty_cell': physics.get('subtract_empty_cell', False),
        'water': physics.get('perform_absolute_scaling', False),
        'water_cell': physics.get('perform_absolute_scaling', False)
    }

    if not dependency_map.get(calib_key, False):
        if calib_key == 'thickness':
             print(f"  [OK] Calibration '{calib_key}' (Mathematical Scalars) loaded.")
        return None

    # Handle both simple strings ('Cd') and nested dictionaries (for multiple empty cells)
    targets = list(set(file_id.values())) if isinstance(file_id, dict) else [file_id]

    # We will store the exact files we find
    found_files = {}

    for target in targets:
        if target is None: continue

        # --- STRICT LOCAL SEARCH ---
        found_hdf = find_hdf_by_identifier(target, class_det)

        if found_hdf:
            print(f"  [OK] '{calib_key}' mapped to '{found_hdf}' at {det_string.replace('p', '.')}m.")
            found_files[target] = found_hdf
        else:
            print(f"\n  [FATAL ERROR] Calibration '{calib_key}' -> '{target}' NOT FOUND at {det_string.replace('p', '.')}m!")
            print(f"  The pipeline cannot continue. Please measure this standard, turn off the correction in your YAML, or fix the file name.")
            sys.exit(1)

    # Return the mapped file (or a dictionary if there were multiple targets like advanced ECs)
    if isinstance(file_id, str):
        return found_files.get(file_id)
    else:
        return found_files
