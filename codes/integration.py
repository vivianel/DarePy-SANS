import numpy as np
import os
import sys
import csv
from tabulate import tabulate
import matplotlib.pyplot as plt
from utils import load_hdf, create_analysis_folder, get_flexible_value
from image_corrections import (load_standards, load_and_normalize,
                        correct_EC, correct_flat_field, correct_dark, process_empty_cell)
from setup_integrator import(generate_beamstop_mask, setup_integration)
from absolute_scaling import calculate_1D_absolute_scalar, apply_absolute_scaling, process_water_standard
import normalize_counts as norm
import plot_integration as plot_integ

def _parse_pixel_ranges(raw_ranges):
    """Helper function to cleanly parse legacy single ranges or nested lists."""
    if raw_ranges is None:
        return []
    if isinstance(raw_ranges, range):
        return [[raw_ranges.start, raw_ranges.stop]]
    elif isinstance(raw_ranges, list) and len(raw_ranges) > 0:
        if isinstance(raw_ranges[0], int):
            return [raw_ranges]
        elif isinstance(raw_ranges[0], list):
            return raw_ranges
    return [] # Default to empty if nothing provided


def set_integration(config, result, det_str):
    """Orchestrates the integration process for a specific detector distance."""
    path_dir_an = create_analysis_folder(config)
    det_folder_name = 'det_' + det_str
    path_det = os.path.join(path_dir_an, det_folder_name)

    if not os.path.exists(path_det):
        print(f"Error: Detector folder '{path_det}' not found.")
        sys.exit(1)

    path_rad_int = os.path.join(path_det, 'integration/')
    if not os.path.exists(path_rad_int):
        os.makedirs(path_rad_int)

    class_file_key = 'det_files_' + det_str
    if class_file_key not in result['overview']:
        return result

    class_file = result['overview'][class_file_key]
    if not class_file['scan']:
        return result

    print(f"Setting up pyFAI geometry for {det_str.replace('p', '.')}m detector.")

    generate_beamstop_mask(config, result, det_str)
    setup_integration(config, result, det_str)
    result = load_standards(config, result, det_str)

    # ==============================================================
    # 1D ABSOLUTE CALIBRATION
    # Calculates the scaling factor independently per detector distance
    # ==============================================================
    if config.get('physics_corrections', {}).get('perform_absolute_scaling', False):

        water_img, water_var = process_water_standard(config, result)
        permanent_mask = result['integration'].get('int_mask')

        if water_img is not None and permanent_mask is not None:
            water_img_sq = np.squeeze(water_img)
            water_var_sq = np.squeeze(water_var) if water_var is not None else None

            print(f"  [CALIBRATION] Deriving Absolute Scalar for {det_str}m...")
            scalar = calculate_1D_absolute_scalar(config, result, det_str, water_img_sq, water_var_sq)
            result['integration']['absolute_scalar'] = scalar
        else:
            print(f"  [WARNING] Missing water standard for {det_str}m. Scaling set to 1.0.")
            result['integration']['absolute_scalar'] = 1.0

    # Execute the core integration loop
    result = integrate(config, result, det_str, path_rad_int, path_det)

    # Ensure the dictionary is returned! ---
    return result

def make_file_name(path, prefix, sufix, sample_name, det_str, scanNr, frame):
    """Constructs a standardized file name."""
    return f"{path}{prefix}_{scanNr:07d}_{frame:05d}_{sample_name}_det{det_str}m.{sufix}"

def integrate(config, result, det_str, path_rad_int, path_det):
    """The Core Math Engine with Tabulated Logging and Modular Functions."""
    # Handle Plotting State
    plotting_enabled = config['analysis'].get('save_plot_radial', 0) == 1 or config['analysis'].get('save_plot_azimuthal', 0) == 1
    if not plotting_enabled:
        plt.ioff()
        plotting_was_off = True
    else:
        plotting_was_off = False

    path_hdf_raw = config['analysis']['path_hdf_raw']
    class_file = result['overview']['det_files_' + det_str]
    force_reintegrate = config['analysis'].get('force_reintegrate', False)
    physics = config.get('physics_corrections', {})
    calib_map = result['overview'].get(f'calibration_map_{det_str}', {})

    # Log Book Setup
    reduction_log = []
    log_headers = ["Scan", "Sample", "Frame", "Monitor", "Dark_Curr", "Trans", "EC_Used", "Thick_cm", "Abs_Scal"]

    print(f'\n[PROCESSING] Initiating Reduction for {det_str.replace("p", ".")}m detector...')

    for ii in range(len(class_file['sample_name'])):
        sample_name = class_file['sample_name'][ii]
        scanNr = class_file['scan'][ii]
        hdf_name = class_file['name_hdf'][ii]
        frame_nr_total = class_file['frame_nr'][ii]

        # Check for skip logic
        flux_monit = load_hdf(path_hdf_raw, hdf_name, 'flux_monit')
        time_s = load_hdf(path_hdf_raw, hdf_name, 'time')
        preset = load_hdf(path_hdf_raw, hdf_name, 'moni')

        if force_reintegrate == 0:
            if flux_monit == preset * 1e4 or time_s == preset:
                print(f"  -> Skip: Scan {scanNr} ('{sample_name}') already integrated.")
                continue

        print(f"\n--- Reducing Scan {scanNr} ({sample_name}) ---")

        for ff in range(frame_nr_total):
            current_log = [scanNr, sample_name, ff]

            # 1: Base Loading
            if physics.get('normalize_to_monitor', True):
                img_base, var_base = load_and_normalize(config, result, hdf_name, return_variance=True)
                img = img_base[ff] if frame_nr_total > 1 else img_base
                var = var_base[ff] if frame_nr_total > 1 else var_base
                current_log.append(f"{flux_monit:.2e}")
            else:
                img_raw = load_hdf(path_hdf_raw, hdf_name, 'counts')
                img = img_raw[ff] if frame_nr_total > 1 else img_raw
                var = img.copy()
                current_log.append("None")

            img = np.squeeze(img)
            var = np.squeeze(var)
            clean_name = str(sample_name).strip()

            # 2: Dark Current
            if physics.get('subtract_dark_current', False):
                dark_block = config['experiment']['calibration']['dark_current']
                dark_id = get_flexible_value(dark_block, clean_name, default_fallback='MISSING')

                # --- PULL DIRECTLY FROM MAP ---
                mapped_dark = calib_map.get('dark_current')
                dark_hdf = mapped_dark.get(dark_id) if isinstance(mapped_dark, dict) else mapped_dark

                if dark_hdf:
                    dark_img, dark_var = load_and_normalize(config, result, dark_hdf, return_variance=True)
                    dark_img = np.squeeze(dark_img)
                    dark_var = np.squeeze(dark_var)

                    img = correct_dark(img, dark_img)
                    var = var + dark_var

                    # Log the exact scan number using the index!
                    idx = class_file['name_hdf'].index(dark_hdf)
                    current_log.append(str(class_file['scan'][idx]))
                else:
                    print(f"  [WARNING] Dark Current '{dark_id}' NOT MAPPED for {det_str}m!")
                    current_log.append("MISSING")
            else:
                current_log.append("OFF")

            # 3: Transmission (Sample)
            if physics.get('apply_transmission', False):
                idx_all = list(result['overview']['all_files']['name_hdf']).index(hdf_name)
                trans = result['overview']['all_files']['transmission'][idx_all]

                if isinstance(trans, (float, int, np.float64)) and trans > 0:
                    img = norm.normalize_transmission(config, hdf_name, result, img)
                    var = np.square(norm.normalize_transmission(config, hdf_name, result, np.sqrt(var)))
                    current_log.append(f"{trans:.3f}")
                else:
                    current_log.append("INVALID")
            else:
                current_log.append("1.000")

            # 4: Empty Cell
            if physics.get('subtract_empty_cell', False):

                # Let the Builder handle all the messy prep!
                img_ec, var_ec, ec_scan = process_empty_cell(config, result, calib_map, class_file, clean_name)

                if img_ec is not None:
                    # Pure mathematical subtraction
                    img = correct_EC(img, img_ec)
                    var = var + var_ec

                    current_log.append(str(ec_scan))
                else:
                    print(f"  [WARNING] Empty Cell NOT MAPPED for {det_str}m! Subtraction skipped.")
                    current_log.append("MISSING")
            else:
                current_log.append("OFF")

            # 5: Thickness
            thick_block = config['experiment']['calibration']['thickness']
            thick_val_for_log = get_flexible_value(thick_block, clean_name, default_fallback=0.1)

            if physics.get('normalize_to_thickness', False):
                img = norm.normalize_thickness(config, hdf_name, result, img)
                var = np.square(norm.normalize_thickness(config, hdf_name, result, np.sqrt(var)))
                current_log.append(f"{thick_val_for_log:.3f}")
            else:
                current_log.append("1.000")

            # 6: Flat Field
            if physics.get('apply_flat_field', False):
                img = correct_flat_field(config, img)
                var = np.square(correct_flat_field(config, np.sqrt(var)))

            # ==========================================================
            # 7: Absolute Scaling (Applies the 1D Scalar to the 2D Image)
            # ==========================================================
            if physics.get('perform_absolute_scaling', False):
                img, var = apply_absolute_scaling(config, result, scanNr, img, var)
                current_log.append(f"{result['integration'].get('absolute_scalar', 1.0):.3e}")
            else:
                current_log.append("OFF")

            # --- INTEGRATIONS & PLOTTING ---
            f_rad = make_file_name(path_rad_int, 'radial_integ', 'dat', sample_name, det_str, scanNr, ff)
            radial_integ(config, result, img, f_rad, img1_variance=var)

            f_azim = make_file_name(path_rad_int, 'azim_integ', 'dat', sample_name, det_str, scanNr, ff)
            data_azimuth = azimuthal_integ(config, result, img, f_azim, img1_variance=var)

            if config['analysis'].get('save_2d_patterns', 0) == 1:
                f_pat = make_file_name(path_rad_int, 'pattern2D', 'dat', sample_name, det_str, scanNr, ff)
                f_var = make_file_name(path_rad_int, 'variance2D', 'dat', sample_name, det_str, scanNr, ff)
                np.savetxt(f_pat, img, delimiter=',')
                np.savetxt(f_var, var, delimiter=',')

            if config['analysis'].get('save_plot_radial', 0) == 1:
                plot_integ.plot_integ_radial(config, result, scanNr, ff, img, data_azimuth)

            if config['analysis'].get('save_plot_azimuthal', 0) == 1:
                plot_integ.plot_integ_azimuthal(config, result, scanNr, ff)

            reduction_log.append(current_log)
            print(f"  -> Success: Frame {ff} processed.")

    print("\n" + "="*80)
    print(f"REDUCTION LOG: {det_str.replace('p', '.')}m")
    print("="*80)
    print(tabulate(reduction_log, headers=log_headers, tablefmt="grid"))

    log_file = os.path.join(path_det, f"reduction_log_det{det_str}.csv")
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(log_headers)
        writer.writerows(reduction_log)
    print(f"\n[INFO] Log book saved to: {log_file}")

    if plotting_was_off:
        plt.ion()

    return result

def radial_integ(config, result, img1, file_name, img1_variance=None):
    """Performs 1D radial integration using a dynamic mask for non-positive pixels."""
    ai = result['integration'].get('ai')
    permanent_mask = result['integration'].get('int_mask')
    integration_points = result['integration'].get('integration_points')

    if ai is None or permanent_mask is None or integration_points is None:
        return

    dynamic_mask = (permanent_mask == 1) | (img1 <= 0)
    mask = dynamic_mask.astype(bool)
    error_model = "azimuthal" if img1_variance is None else None

    try:
        q, I, sigma = ai.integrate1d(
            img1, integration_points, correctSolidAngle=True, variance=img1_variance,
            mask=mask, method='nosplit_csr', unit='q_A^-1', safe=True,
            error_model=error_model, flat=None, dark=None
        )
        data_save = np.column_stack((q, I, sigma))
        header_text = 'q (A-1), absolute intensity  I (1/cm), standard deviation'
        np.savetxt(file_name, data_save, delimiter=',', header=header_text, comments='# ')
    except Exception as e:
        print(f"  [ERROR] Radial integration failed: {e}")

def azimuthal_integ(config, result, img1, file_name, img1_variance=None):
    """Performs 2D azimuthal integration, applies dynamic masking, and optionally saves 1D profiles."""
    ai = result['integration'].get('ai')
    permanent_mask = result['integration'].get('int_mask')
    integration_points = result['integration'].get('integration_points')
    sectors_nr = result['integration'].get('sectors_nr')

    if ai is None or permanent_mask is None or integration_points is None or sectors_nr is None:
        return None

    dynamic_mask = (permanent_mask == 1) | (img1 <= 0)
    mask = dynamic_mask.astype(bool)
    error_model = "azimuthal" if img1_variance is None else None

    try:
        I_all, q_all_sectors, angles_all, sigma_all = ai.integrate2d_ng(
            img1, integration_points, npt_azim=sectors_nr, correctSolidAngle=True,
            variance=img1_variance, mask=mask, method=('full', 'csr', 'cython'),
            unit='q_A^-1', safe=True, error_model=error_model, flat=None, dark=None
        )

        if q_all_sectors is not None:
            data_save = np.column_stack((q_all_sectors, I_all.transpose(), sigma_all.transpose()))

            # Check the save flag (Handles 1, True, or 'true' from YAML safely)
            save_flag = config['analysis'].get('save_data_azimuthal', 0)
            if save_flag in [1, True, 'true', 'True']:

                # 1. SAVE THE MASTER 2D CAKE PLOT
                header_text = f'q (A-1), {sectors_nr} columns for absolute intensity\nAngles {angles_all}'
                np.savetxt(file_name, data_save, delimiter=',', header=header_text, comments='# ')

                # 2. EXTRACT AND SAVE INDIVIDUAL 1D AZIMUTHAL PROFILES
                raw_ranges = result['integration'].get('pixel_range_azim')
                ranges_to_save = _parse_pixel_ranges(raw_ranges)

                q = q_all_sectors
                I = I_all.transpose()

                # If no ranges were explicitly provided, DEFAULT TO FULL AVAILABLE q-RANGE!
                if not ranges_to_save:
                    ranges_to_save = [[0, len(q)]]

                npt_azim_plot = np.linspace(0, 360, sectors_nr + 1)
                range_angle_midpoints = [(npt_azim_plot[rr] + npt_azim_plot[rr+1]) / 2 for rr in range(sectors_nr)]

                for i, q_bnds in enumerate(ranges_to_save):
                    start_idx = max(0, min(q_bnds[0], len(q) - 1))
                    end_idx = max(1, min(q_bnds[1], len(q)))

                    if start_idx >= end_idx:
                        continue

                    q_range = range(start_idx, end_idx)
                    I_select = I[q_range, :]
                    I_sum = np.nansum(I_select, axis=0)

                    # Package the data for this specific range
                    export_data = np.column_stack((range_angle_midpoints, I_sum))
                    hdr_string = f"Chi_deg, I_Sum_q{q_bnds[0]}-{q_bnds[1]}"

                    # Dynamically inject the specific q-range into the filename!
                    file_name_1d = file_name.replace('azim_integ', f'azim_integ_q{q_bnds[0]}-{q_bnds[1]}')

                    np.savetxt(file_name_1d, export_data, delimiter=',', header=hdr_string, comments='# ')

            return data_save

    except Exception as e:
        print(f"  [ERROR] Azimuthal integration failed: {e}")

    return None
