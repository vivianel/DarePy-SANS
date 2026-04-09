# -*- coding: utf-8 -*-
"""
DarePy-SANS: Calibration Module
Handles absolute intensity scaling (cm^-1) using 1D integration of Water standards.
"""
import numpy as np

def calculate_1D_absolute_scalar(config, result, det_str, water_img, water_var=None):
    """
    Integrates the Water standard to 1D, finds the plateau mean,
    and returns the absolute scaling factor.
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
        # 1. Integrate the water standard to 1D
        q, I, sigma = ai.integrate1d(
            water_img, integration_points, correctSolidAngle=True, variance=water_var,
            mask=mask, method='nosplit_csr', unit='q_A^-1', safe=True,
            error_model=error_model, flat=None, dark=None
        )

        # 2. Find the flat plateau (e.g., q = 0.05 to 0.15 A^-1)
        valid_idx = (q >= 0.05) & (q <= 0.15)
        if not np.any(valid_idx):
            valid_idx = (q >= np.min(q)) # fallback if data is outside range

        mean_water_I = np.nanmean(I[valid_idx])

        if mean_water_I <= 0 or np.isnan(mean_water_I):
            print(f"  [ERROR] Water 1D mean is invalid ({mean_water_I}). Scaling failed.")
            return 1.0

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

        print(f"  [CALIBRATION] 1D Water Mean (q=0.05-0.15): {mean_water_I:.4f} cm^-1")
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
