# -*- coding: utf-8 -*-
"""
DarePy-SANS: Calibration Module
Handles absolute intensity scaling (cm^-1) using 1D integration of Water standards.
"""
import numpy as np
import normalize_counts as norm
import sys
from utils import load_hdf
from image_corrections import correct_EC, correct_flat_field, correct_dark

def process_water_standard(config, result):
    """
    Processes the raw water and empty cell standards to produce the fully
    corrected water scattering image for absolute scaling.
    """
    physics = config.get('physics_corrections', {})
    path_hdf_raw = config['analysis']['path_hdf_raw']

    img_w = result['integration'].get('water')
    var_w = result['integration'].get('water_variance')
    hdf_w = result['integration'].get('water_hdf')

    img_wc = result['integration'].get('water_cell')
    var_wc = result['integration'].get('water_cell_variance')
    hdf_wc = result['integration'].get('water_cell_hdf')

    if img_w is None or img_wc is None:
        return None, None

    # Copy the arrays to avoid permanently mutating the loaded baseline
    img_w = img_w.copy()
    var_w = var_w.copy() if var_w is not None else None
    img_wc = img_wc.copy()
    var_wc = var_wc.copy() if var_wc is not None else None

    # ==========================================================
    # 0. Subtract Dark Current
    # ==========================================================
    if physics.get('subtract_dark_current', False):
        dark_rate = result['integration'].get('dark_current')
        dark_rate_var = result['integration'].get('dark_current_variance')

        if dark_rate is not None:
            def apply_dark(img, var, hdf_name):
                # 1. Scale rate to sample time
                sample_time = load_hdf(path_hdf_raw, hdf_name, 'time')
                scaled_dark = dark_rate * sample_time
                scaled_dark_var = dark_rate_var * (sample_time ** 2)

                # 2. Balance Denominators
                att_setting = str(int(load_hdf(path_hdf_raw, hdf_name, 'att')))
                list_attenuation = config['instrument']['list_attenuation']
                if att_setting in list_attenuation:
                    att_factor = float(list_attenuation[att_setting])
                    if att_factor > 0:
                        scaled_dark = scaled_dark / att_factor
                        scaled_dark_var = scaled_dark_var / (att_factor ** 2)

                if physics.get('normalize_to_monitor', True):
                    sample_moni = load_hdf(path_hdf_raw, hdf_name, 'flux_monit')
                    if sample_moni > 0:
                        scaled_dark = scaled_dark / sample_moni
                        scaled_dark_var = scaled_dark_var / (sample_moni ** 2)

                # 3. Apply math
                img_out = correct_dark(img, scaled_dark)
                var_out = var + scaled_dark_var
                return img_out, var_out

            # Apply identical dark correction to both standard runs
            img_w, var_w = apply_dark(img_w, var_w, hdf_w)
            img_wc, var_wc = apply_dark(img_wc, var_wc, hdf_wc)


    # ==========================================================
    # 1. Apply Transmission to Water and its Empty Cell BEFORE subtraction
    # ==========================================================
    if physics.get('apply_transmission', False):
        img_w = norm.normalize_transmission(config, hdf_w, result, img_w)
        var_w = np.square(norm.normalize_transmission(config, hdf_w, result, np.sqrt(var_w)))

        img_wc = norm.normalize_transmission(config, hdf_wc, result, img_wc)
        var_wc = np.square(norm.normalize_transmission(config, hdf_wc, result, np.sqrt(var_wc)))

    # 2. Subtract Empty Cell
    img_h2o = correct_EC(img_w, img_wc)
    img_h2o_var = var_w + var_wc

    # 3. Normalize by Thickness
    if physics.get('normalize_to_thickness', False):
        img_h2o = norm.normalize_thickness(config, hdf_w, result, img_h2o)
        img_h2o_var = np.square(norm.normalize_thickness(config, hdf_w, result, np.sqrt(img_h2o_var)))

    # 4. Apply Flat Field Correction
    if physics.get('apply_flat_field', False):
        img_h2o = correct_flat_field(config, img_h2o)
        img_h2o_var = np.square(correct_flat_field(config, np.sqrt(img_h2o_var)))

    # 5. Safeguard against negative pixels before absolute scaling integration
    img_h2o[img_h2o <= 0] = 1e-6

    return img_h2o, img_h2o_var

def calculate_1D_absolute_scalar(config, result, det_str, water_img, water_var=None):
    """
    Integrates the Water standard to 1D based on geometric detector radius,
    finds the plateau mean, and returns the absolute scaling factor.
    """
    ai = result['integration'].get('ai')
    permanent_mask = result['integration'].get('int_mask')
    integration_points = result['integration'].get('integration_points')

    if ai is None or permanent_mask is None:
        print("  [ERROR] pyFAI geometry missing. Scaling set to 1.0.")
        return 1.0

    # Defensively squeeze everything to 2D so pyFAI doesn't crash!
    water_img = np.squeeze(water_img)
    if water_var is not None:
        water_var = np.squeeze(water_var)
    mask = np.squeeze(permanent_mask).astype(bool)

    error_model = "azimuthal" if water_var is None else None

    try:
        # 1. Integrate the water standard to 1D using RADIUS (r_mm) instead of q
        r_mm, I, sigma = ai.integrate1d(
            water_img, integration_points, correctSolidAngle=True, variance=water_var,
            mask=mask, method='nosplit_csr', unit='r_mm', safe=True,
            error_model=error_model, flat=None, dark=None
        )

        # ---------------------------------------------------------
        # Define the flat plateau in pixels
        # ---------------------------------------------------------
        pixel_min = 20  # Safely outside the beamstop shadow
        pixel_max = 80  # Safely inside the detector edges

        # ai.pixel1 is the pixel size in meters. Convert to mm.
        pixel_size_mm = ai.pixel1 * 1000

        r_min_mm = pixel_min * pixel_size_mm
        r_max_mm = pixel_max * pixel_size_mm

        # 2. Find the valid geometric radius indices
        valid_idx = (r_mm >= r_min_mm) & (r_mm <= r_max_mm)

        if not np.any(valid_idx):
            valid_idx = (r_mm >= np.min(r_mm)) # fallback if data is outside range

        mean_water_I = np.nanmedian(I[valid_idx]) # probably best with median

        if mean_water_I <= 0 or np.isnan(mean_water_I):
            print(f"  [ERROR] Water 1D mean is invalid ({mean_water_I}). Scaling failed.")
            sys.exit(1)

        # --- CRITICAL FIX: The Noise Gate ---
        if mean_water_I < 1e-10:
            print(f"  [WARNING] Water signal is essentially zero ({mean_water_I:.6e}).")
            print(f"  [WARNING] This standard is pure noise. Setting scalar to 1.0.")
            print(f"  [WARNING] Rely on caller_merging.py to scale the {det_str}m overlap.")
            return 1.0

        if np.isnan(mean_water_I):
            print(f"  [ERROR] Water 1D mean is NaN. Scaling failed.")
            sys.exit(1)

        # 3. Lookup Wavelength and g(lambda) safely
        list_cs = config['instrument'].get('list_abs_calib', {})
        wl_str = "Unknown"
        try:
            water_hdf = result['integration'].get('water_hdf')
            if water_hdf:
                # Force list cast so .index() works on numpy arrays
                idx = list(result['overview']['all_files']['name_hdf']).index(water_hdf)
                wl_str = str(int(result['overview']['all_files']['wl_A'][idx]))
        except Exception as e:
            pass

        g_lambda = float(list_cs.get(wl_str, 1.0))
        if wl_str not in list_cs:
            print(f"  [WARNING] Wavelength {wl_str}A not found in calibration table. Using 1.0.")

        # 4. Calculate Final Scalar
        scaling_factor = g_lambda / mean_water_I
        total_scaling = scaling_factor / result['integration'].get('scaling_factor', 1.0)

        print(f"  [CALIBRATION] 1D Water Mean (Radius {pixel_min}-{pixel_max} px): {mean_water_I:.4f} cm^-1")
        print(f"  [CALIBRATION] Final Absolute Scalar calculated: {total_scaling:.4f}")

        return total_scaling

    except Exception as e:
        print(f"  [ERROR] 1D Calibration failed: {e}")
        return 1.0


def apply_absolute_scaling(config, result, scan_nr, img, var):
    """
    Applies the pre-calculated 1D scalar to the 2D sample image and variance.
    Returns the 2D corrected pattern and variance for pyFAI integration.
    """
    scalar = result['integration'].get('absolute_scalar', 1.0)

    if scalar <= 0:
        print(f"  [WARNING] Invalid absolute scalar ({scalar}) for Scan {scan_nr}. Returning unscaled data.")
        return img, var

    img_scaled = img * scalar

    # Rigorous Error Propagation: Variance scales by the square of the multiplier
    if var is not None:
        var_scaled = var * (scalar ** 2)
    else:
        var_scaled = None

    return img_scaled, var_scaled
