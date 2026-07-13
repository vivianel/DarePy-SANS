# -*- coding: utf-8 -*-
"""
DarePy-SANS: Transmission Calculation
Calculates transmission scalars using the Virtual Referencing architecture.
Saves logs directly to the analysis root.
"""
import numpy as np
import os
import sys
import csv
from tabulate import tabulate

# Importing unified flexible logic directly from utils
from utils import load_hdf, create_analysis_folder, save_results, get_flexible_value, find_hdf_by_identifier
from prepare_input import save_list_files
import normalize_counts as norm


def trans_calc(config, class_files, result):
    """Master entry point for transmission calculations."""
    # ==========================================
    # 1. MASTER KILL SWITCH
    # ==========================================
    run_flag = config.get('pipeline_control', {}).get('run_transmission', True)
    if str(run_flag).lower() == 'false' or run_flag is False:
        print("\n[INFO] Skipping transmission calculation (run_transmission set to False in YAML).")
        return result

    # ==========================================
    # 2. PROCEED WITH PROCESSING
    # ==========================================
    beamstop = config.get('transmission_setup', {}).get('beamstop', 'standard')

    if beamstop == 'standard':
        result = select_transmission(config, class_files, result)
        result = trans_calc_reference(config, result, class_files)
        result = trans_calc_sample(config, result)
    elif beamstop == 'semitransparent':
        result = trans_calc_reference(config, result, class_files)
        result = trans_calc_sample(config, result)

    return result


def trans_calc_reference(config, result, class_files):
    """Calculates Empty Beam (EB) reference intensities based on beamstop type."""
    path_dir_an = create_analysis_folder(config)
    path_hdf_raw = config['analysis']['path_hdf_raw']
    coordinates = config['analysis']['transmission_coordinates']
    beamstop = config.get('transmission_setup', {}).get('beamstop', 'standard')

    eb_block = config.get('transmission_setup', {}).get('empty_beam', 'EB')
    eb_id = get_flexible_value(eb_block, 'default', default_fallback='EB')

    # -------------------------------------------------------------------------
    # CASE 1: Standard Beamstop Reference (Auto-Masking via Image Moments)
    # -------------------------------------------------------------------------
    if beamstop == 'standard':
        trans_dist = config.get('physics_corrections', {}).get('dist_trans_measurements', 0)
        if not trans_dist or float(trans_dist) <= 0:
            print("\n[INFO] Skipping transmission reference calculation (No valid dist_trans_measurements).")
            return result

        class_trans = result['overview']['trans_files']
        eb_hdf = find_hdf_by_identifier(eb_id, class_trans)

        if eb_hdf is not None:
            idx = class_trans['name_hdf'].index(eb_hdf)
            if class_trans['detx_m'][idx] == trans_dist:
                counts = load_hdf(path_hdf_raw, eb_hdf, 'counts')
                img_eb = normalize_trans(config, result, eb_hdf, counts)

                # Shape-Based Elliptical Masking
                print(f"\n[AUTO-MASK] Calculating shape-based elliptical mask for {trans_dist}m...")
                noise_floor = np.median(img_eb)
                beam_only = np.clip(img_eb - noise_floor, 0, None)

                shape_threshold = beam_only.max() * 0.10
                core_beam = np.where(beam_only > shape_threshold, beam_only, 0)
                total_intensity = core_beam.sum()
                mask = np.zeros_like(img_eb)

                if total_intensity > 0:
                    y_indices, x_indices = np.indices(core_beam.shape)
                    center_y = (y_indices * core_beam).sum() / total_intensity
                    center_x = (x_indices * core_beam).sum() / total_intensity

                    var_y = (((y_indices - center_y)**2) * core_beam).sum() / total_intensity
                    var_x = (((x_indices - center_x)**2) * core_beam).sum() / total_intensity

                    rx = max(4.0 * np.sqrt(var_x), 1.0)
                    ry = max(4.0 * np.sqrt(var_y), 1.0)

                    ellipse_eq = ((x_indices - center_x)**2) / (rx**2) + ((y_indices - center_y)**2) / (ry**2)
                    mask[ellipse_eq <= 1.0] = 1

                    EB_ref = float(np.sum(img_eb * mask))
                    result['transmission']['mask'] = mask
                    result['transmission']['mean_EB'] = EB_ref
                    result['transmission']['EB_counts'] = img_eb
                else:
                    sys.exit("\n[ERROR] Empty Beam has zero intensity! Cannot calculate mask.")
        else:
            sys.exit(f'For standard beamstop, measure an empty beam ({eb_id}) at distance {trans_dist}m.')

    # -------------------------------------------------------------------------
    # CASE 2: Semi-transparent Beamstop Reference (Coordinate-Based Region Box)
    # -------------------------------------------------------------------------
    elif beamstop == 'semitransparent':
        det_distances = list(set(class_files['detx_m']))
        for dist in det_distances:
            class_dist = filter_by_distance(class_files, dist)
            eb_hdf = find_hdf_by_identifier(eb_id, class_dist)

            if eb_hdf is not None:
                counts = load_hdf(path_hdf_raw, eb_hdf, 'counts')
                img = normalize_trans(config, result, eb_hdf, counts)

                mask = np.zeros_like(img)

                # --- FIX: Type-safe flexible lookup for coordinate dict keys ---
                matched_key = None
                for k in coordinates.keys():
                    try:
                        # Safely evaluate if keys match numerically (handles float 5.0 vs int 5 vs string '5')
                        if float(k) == float(dist):
                            matched_key = k
                            break
                    except ValueError:
                        if str(k) == str(dist):
                            matched_key = k
                            break

                if matched_key is None:
                    sys.exit(f'[ERROR] Missing region specifications inside transmission_coordinates for distance {dist}m.')

                c = coordinates[matched_key]
                # Apply the bounding box coordinates to select the area
                mask[c[0]:c[1], c[2]:c[3]] = 1
                EB_ref = float(np.sum(img * mask))

                result['transmission'][f'mask_{dist}'] = mask
                result['transmission'][f'mean_EB_{dist}'] = EB_ref
                result['transmission'][f'counts_EB_{dist}'] = img
    save_results(path_dir_an, result)
    return result


def trans_calc_sample(config, result):
    """Processes all master scans, calculates transmission values, and maps thickness values cleanly."""
    path_dir_an = create_analysis_folder(config)
    path_hdf_raw = config['analysis']['path_hdf_raw']
    beamstop = config.get('transmission_setup', {}).get('beamstop', 'standard')
    eb_block = config.get('transmission_setup', {}).get('empty_beam', 'EB')

    # Extract Master File Records
    class_all = result['overview']['all_files']
    class_trans = result['overview'].get('trans_files', {})
    trans_hdf_names = class_trans.get('name_hdf', [])

    # Extract Thickness Rules
    thickness_map = config['experiment'].get('sample_thickness', {})
    default_t = thickness_map.get('default', 0.1)
    trans_dist = config.get('physics_corrections', {}).get('dist_trans_measurements', 0)

    # Output Trackers
    list_trans_all = []
    list_counts_all = []
    list_thick_all = []
    trans_log = []
    log_headers = ["Scan", "Sample", "Det_m", "EB_Scan", "Trans_Counts", "Transmission", "Thickness(cm)"]

    # ==========================================
    # UNIFIED FILE PROCESSING LOOP
    # ==========================================
    for ii in range(len(class_all['sample_name'])):
        hdf_name = class_all['name_hdf'][ii]
        sample_name = class_all['sample_name'][ii]
        scan_nr = class_all['scan'][ii]
        det_m = class_all['detx_m'][ii]

        # 1. Hierarchical Thickness Check (Metadata -> Config Map -> Default Fallback)
        thickness = resolve_thickness(path_hdf_raw, hdf_name, sample_name, thickness_map, default_t)
        list_thick_all.append(thickness)

        # Base Placeholders
        trans_val = "--"
        sum_counts_str = "--"
        eb_scan_str = "N/A"

        # 2. Process Standard Beamstop Track
        if beamstop == 'standard':
            if hdf_name in trans_hdf_names and det_m == trans_dist:
                mask = result['transmission'].get('mask')
                EB_ref = result['transmission'].get('mean_EB', 0)

                eb_id = get_flexible_value(eb_block, sample_name, default_fallback='EB')
                eb_hdf = find_hdf_by_identifier(eb_id, class_trans)
                eb_scan_str = get_scan_number_str(eb_hdf, class_all)

                if mask is not None and EB_ref > 0:
                    counts = load_hdf(path_hdf_raw, hdf_name, 'counts')
                    img = normalize_trans(config, result, hdf_name, counts)
                    sum_counts = float(np.sum(img * mask))
                    sum_counts_str = f"{sum_counts:.2e}"

                    trans_val = 1.000 if hdf_name == eb_hdf else round(sum_counts / EB_ref, 3)
            else:
                trans_val = "--"  # Bypassed scan for standard transmission setup

        # 3. Process Semi-transparent Beamstop Track
        elif beamstop == 'semitransparent':
            mask_key = f'mask_{det_m}'
            eb_ref_key = f'mean_EB_{det_m}'

            eb_id = get_flexible_value(eb_block, sample_name, default_fallback='EB')
            class_dist = filter_by_distance(class_all, det_m)
            eb_hdf = find_hdf_by_identifier(eb_id, class_dist)
            eb_scan_str = get_scan_number_str(eb_hdf, class_all)

            if mask_key in result['transmission'] and eb_ref_key in result['transmission']:
                mask = result['transmission'][mask_key]
                EB_ref = result['transmission'][eb_ref_key]

                counts = load_hdf(path_hdf_raw, hdf_name, 'counts')
                img = normalize_trans(config, result, hdf_name, counts)
                sum_counts = float(np.sum(img * mask))
                sum_counts_str = f"{sum_counts:.2e}"

                if hdf_name == eb_hdf:
                    trans_val = 1.000
                else:
                    trans_val = round(sum_counts / EB_ref, 3) if EB_ref > 0 else 0.000
            else:
                trans_val = "NO_REF"

        # Collect results
        list_trans_all.append(trans_val)
        list_counts_all.append(sum_counts_str)

        # Append row formatted cleanly for logging
        trans_str = f"{trans_val:.3f}" if isinstance(trans_val, (int, float)) else str(trans_val)
        trans_log.append([scan_nr, sample_name, det_m, eb_scan_str, sum_counts_str, trans_str, f"{thickness:.3f}"])

    # ==========================================
    # SAVE & WRITE LOG OUTCOMES
    # ==========================================
    result['transmission']['calc'] = list_trans_all
    class_all['transmission'] = list_trans_all
    class_all['thickness_cm'] = list_thick_all

    # Print to console
    log_title = f"TRANSMISSION LOG ({beamstop.upper()} BEAMSTOP)"
    print(f"\n{bcolors.HEADER}{'='*80}\n{log_title}\n{'='*80}{bcolors.ENDC}")
    print(tabulate(trans_log, headers=log_headers, tablefmt="grid"))

    # Save to CSV
    log_filename = "transmission_log.csv" if beamstop == 'standard' else "transmission_log_multidist.csv"
    log_file_path = os.path.join(path_dir_an, log_filename)
    with open(log_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(log_headers)
        writer.writerows(trans_log)
    print(f"\n[INFO] Transmission log saved directly to: {log_file_path}")

    save_list_files(path_dir_an, path_dir_an, class_all, 'all_files', result)
    return result


# =============================================================================
# HELPER UTILITIES
# =============================================================================

def resolve_thickness(path_hdf_raw, hdf_name, sample_name, thickness_map, default_t):
    """Resolves thickness with priority order: Metadata -> Config Named Sample -> Config Default."""
    # 1. Try reading directly from file metadata
    try:
        t_meta = load_hdf(path_hdf_raw, hdf_name, 'thickness')
        if t_meta not in (None, 0, 0.0, "None", "nan", "NaN"):
            val = float(t_meta)
            if not np.isnan(val) and val > 0:
                return val
    except Exception:
        pass

    # 2. Look for the explicit sample match in the configuration YAML
    if sample_name in thickness_map:
        return float(thickness_map[sample_name])

    # 3. Fallback to global configuration baseline
    return float(default_t)


def filter_by_distance(class_files, target_dist):
    """Returns a subset dictionary of files matching a precise detector distance."""
    filtered = {k: [] for k in class_files.keys()}
    for idx, dist in enumerate(class_files['detx_m']):
        if dist == target_dist:
            for k in class_files.keys():
                filtered[k].append(class_files[k][idx])
    return filtered


def get_scan_number_str(eb_hdf, class_all):
    """Finds and extracts a matching file scan number for console logs."""
    if eb_hdf is not None and eb_hdf in class_all['name_hdf']:
        idx = class_all['name_hdf'].index(eb_hdf)
        return str(class_all['scan'][idx])
    return "MISSING"


def normalize_trans(config, result, hdf_name, counts):
    """Executes consecutive standard SANS pipeline normalizations."""
    counts = norm.normalize_deadtime(config, hdf_name, counts)
    counts = norm.normalize_flux(config, hdf_name, counts)
    counts = norm.normalize_attenuator(config, hdf_name, counts)
    return counts


def select_transmission(config, class_files, result):
    """Isolates active transmission scanning configurations for standard runs."""
    trans_dist = config['physics_corrections']['dist_trans_measurements']
    if not trans_dist or not isinstance(trans_dist, (int, float)):
        return result

    list_keys = list(class_files.keys())
    class_trans = {key: [] for key in list_keys}

    for ii in range(len(class_files['att'])):
        if (class_files['att'][ii] > 0 and
            class_files['detx_m'][ii] == trans_dist and
            class_files['time_s'][ii] > 0 and
            (class_files['beamstop_y'][ii] < -30 or class_files['beamstop_y'][ii] < 0)):
            for k in list_keys:
                class_trans[k].append(class_files[k][ii])

    result['overview']['trans_files'] = class_trans
    return result


class bcolors:
    HEADER = '\033[95m'
    ENDC = '\033[0m'
