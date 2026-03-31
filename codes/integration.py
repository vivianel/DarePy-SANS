import numpy as np
import os
import sys
import csv
from tabulate import tabulate
import matplotlib.pyplot as plt
from utils import load_hdf, create_analysis_folder, get_flexible_value, find_hdf_by_identifier
from image_corrections import (load_standards, load_and_normalize,
                        correct_EC, correct_flat_field)
from setup_integrator import(generate_beamstop_mask, setup_integration)
from absolute_scaling import absolute_calibration_2D
import normalize_counts as norm
import plot_integration as plot_integ

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

    result = integrate(config, result, det_str, path_rad_int, path_det)
    return result

def make_file_name(path, prefix, sufix, sample_name, det_str, scanNr, frame):
    """Constructs a standardized file name."""
    return f"{path}{prefix}_{scanNr:07d}_{frame:05d}_{sample_name}_det{det_str}m.{sufix}"

def integrate(config, result, det_str, path_rad_int, path_det):
    """The Core Math Engine with Tabulated Logging and Modular Functions."""
    # Handle Plotting State
    plotting_enabled = config['analysis'].get('plot_radial', 0) == 1 or config['analysis'].get('plot_azimuthal', 0) == 1
    if not plotting_enabled:
        plt.ioff()
        plotting_was_off = True
    else:
        plotting_was_off = False

    path_hdf_raw = config['analysis']['path_hdf_raw']
    class_file = result['overview']['det_files_' + det_str]
    force_reintegrate = config['analysis'].get('force_reintegrate', False)
    physics = config.get('physics_corrections', {})

    # Log Book Setup - Headers reordered to match the new physics sequence!
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
            # Temporary log storage for this frame
            current_log = [scanNr, sample_name, ff]

            # ==========================================
            # STEP 1: Base Loading & Flux Normalization
            # ==========================================
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

            # Clean the name from HDF5 first so all steps can use it
            clean_name = str(sample_name).strip()

            # ==========================================
            # STEP 2: LOG DARK CURRENT
            # ==========================================
            if physics.get('subtract_dark_current', False):
                dark_block = config['experiment']['calibration']['dark_current']
                dark_id = get_flexible_value(dark_block, clean_name, default_fallback='MISSING')
                current_log.append(str(dark_id))
            else:
                current_log.append("OFF")

            # ==========================================
            # STEP 3: Transmission (Sample)
            # ==========================================
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

            # ==========================================
            # STEP 4: Empty Cell Subtraction
            # ==========================================
            ec_block = config['experiment']['calibration']['empty_cell']
            ec_id = get_flexible_value(ec_block, clean_name, default_fallback='EC')

            if physics.get('subtract_empty_cell', False):
                ec_hdf = find_hdf_by_identifier(ec_id, result['overview']['all_files'])
                if ec_hdf:
                    # 1. Load the Dark-Corrected and Flux-Normalized Empty Cell
                    img_ec, var_ec = load_and_normalize(config, result, ec_hdf, return_variance=True)
                    img_ec = np.squeeze(img_ec)
                    var_ec = np.squeeze(var_ec)

                    # 2. **CRITICAL PHYSICS FIX**: Apply the Empty Cell's specific transmission
                    if physics.get('apply_transmission', False):
                        img_ec = norm.normalize_transmission(config, ec_hdf, result, img_ec)
                        var_ec = np.square(norm.normalize_transmission(config, ec_hdf, result, np.sqrt(var_ec)))

                    # 3. Subtract from the already-transmitted Sample
                    img = correct_EC(img, img_ec)
                    var = var + var_ec

                    current_log.append(str(ec_id))
                else:
                    current_log.append("MISSING")
            else:
                current_log.append("OFF")

            # ==========================================
            # STEP 5: Thickness Normalization
            # ==========================================
            thick_block = config['experiment']['calibration']['thickness']
            thick_val_for_log = get_flexible_value(thick_block, clean_name, default_fallback=0.1)

            if physics.get('normalize_to_thickness', False):
                img = norm.normalize_thickness(config, hdf_name, result, img)
                var = np.square(norm.normalize_thickness(config, hdf_name, result, np.sqrt(var)))
                current_log.append(f"{thick_val_for_log:.3f}")
            else:
                current_log.append("1.000")

            # ==========================================
            # STEP 6: Flat Field
            # ==========================================
            if physics.get('apply_flat_field', False):
                img = correct_flat_field(config, img)
                var = np.square(correct_flat_field(config, np.sqrt(var)))

            # ==========================================
            # STEP 7: Absolute Scaling
            # ==========================================
            if physics.get('perform_absolute_scaling', False):
                water_std = result['integration'].get('water')
                if water_std is not None:
                    img = absolute_calibration_2D(config, result, scanNr, img, water_std)
                    var = np.square(absolute_calibration_2D(config, result, scanNr, np.sqrt(var), water_std))
                    current_log.append("YES")
                else:
                    current_log.append("FAIL")
            else:
                current_log.append("OFF")

            # --- INTEGRATIONS & PLOTTING ---
            f_rad = make_file_name(path_rad_int, 'radial_integ', 'dat', sample_name, det_str, scanNr, ff)
            radial_integ(config, result, img, f_rad, img1_variance=var)

            f_azim = make_file_name(path_rad_int, 'azim_integ', 'dat', sample_name, det_str, scanNr, ff)
            data_azimuth = azimuthal_integ(config, result, img, f_azim, img1_variance=var)

            # Save 2D Pattern
            if config['analysis'].get('save_2d_patterns', 0) == 1:
                f_pat = make_file_name(path_rad_int, 'pattern2D', 'dat', sample_name, det_str, scanNr, ff)
                f_var = make_file_name(path_rad_int, 'variance2D', 'dat', sample_name, det_str, scanNr, ff)
                np.savetxt(f_pat, img, delimiter=',')
                np.savetxt(f_var, var, delimiter=',')

            if config['analysis'].get('plot_radial', 0) == 1:
                plot_integ.plot_integ_radial(config, result, scanNr, ff, img, data_azimuth)

            if config['analysis'].get('plot_azimuthal', 0) == 1:
                plot_integ.plot_integ_azimuthal(config, result, scanNr, ff)

            reduction_log.append(current_log)
            print(f"  -> Success: Frame {ff} processed.")

    # Print Log Table
    print("\n" + "="*80)
    print(f"REDUCTION LOG: {det_str.replace('p', '.')}m")
    print("="*80)
    print(tabulate(reduction_log, headers=log_headers, tablefmt="grid"))

    # Save Log to CSV
    log_file = os.path.join(path_det, f"reduction_log_det{det_str}.csv")
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(log_headers)
        writer.writerows(reduction_log)
    print(f"\n[INFO] Log book saved to: {log_file}")

    if plotting_was_off:
        plt.ion()

    return result

# ---------------------------------------------------------
# Keep your existing radial_integ and azimuthal_integ
# exactly as they were here at the bottom.
# ---------------------------------------------------------
def radial_integ(config, result, img1, file_name, img1_variance=None):
    """Performs 1D radial integration using a dynamic mask for non-positive pixels."""
    ai = result['integration'].get('ai')
    permanent_mask = result['integration'].get('int_mask')
    integration_points = result['integration'].get('integration_points')

    if ai is None or permanent_mask is None or integration_points is None:
        return

    # --- RIGOROUS MASKING LOGIC ---
    # Create a dynamic mask:
    # Mask if (Permanent Mask is 1) OR (Intensity is 0 or negative)
    dynamic_mask = (permanent_mask == 1) | (img1 <= 0)

    # Ensure it is boolean for pyFAI
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
    """Performs 2D azimuthal integration using a dynamic mask."""
    ai = result['integration'].get('ai')
    permanent_mask = result['integration'].get('int_mask')
    integration_points = result['integration'].get('integration_points')
    sectors_nr = result['integration'].get('sectors_nr')

    if ai is None or permanent_mask is None or integration_points is None or sectors_nr is None:
        return None

    # --- RIGOROUS MASKING LOGIC ---
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

            if config['analysis'].get('save_azimuthal', 0) == 1:
                header_text = f'q (A-1), {sectors_nr} columns for absolute intensity\nAngles {angles_all}'
                np.savetxt(file_name, data_save, delimiter=',', header=header_text, comments='# ')

            return data_save

    except Exception as e:
        print(f"  [ERROR] Azimuthal integration failed: {e}")

    return None
