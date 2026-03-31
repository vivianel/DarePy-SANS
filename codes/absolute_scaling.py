# -*- coding: utf-8 -*-
"""
DarePy-SANS: Calibration Module
Handles absolute intensity scaling (cm^-1) using Water standards.
"""
import numpy as np

def absolute_calibration_2D(config, result, scan_nr, I, I_water):
    """
    Scales 2D detector data to absolute units (cm^-1).
    """
    # 1. Handle Water Mean (The denominator for scaling)
    # We use a masked mean to avoid beamstop/dead pixels
    mask = result['integration'].get('int_mask')
    masked_water = np.ma.MaskedArray(data=I_water, mask=mask)
    water_mean = masked_water.mean()

    if water_mean <= 0:
        print(f"[ERROR] Water mean is non-positive for Scan {scan_nr}. Absolute scaling failed.")
        return I

    # 2. Lookup the Wavelength Calibration Constant g(lambda)
    # This is usually provided by the instrument scientist
    list_cs = config['instrument'].get('list_abs_calib', {})

    # Get wavelength from the current scan's metadata
    # We look for the wavelength in the all_files overview
    try:
        # Find index of current scan
        idx = result['overview']['all_files']['scan'].index(scan_nr)
        wl_val = int(result['overview']['all_files']['wl_A'][idx])
        wl_str = str(wl_val)
    except:
        wl_str = "Unknown"

    if wl_str in list_cs:
        g_lambda = float(list_cs[wl_str])
    else:
        g_lambda = 1.0
        print(f"[WARNING] Wavelength {wl_str}A not found in calibration table. Using 1.0.")

    # 3. Apply the Scaling Factor
    # Scaling = g(lambda) / <I_water>
    scaling_factor = g_lambda / water_mean

    # Handle the 18m specific geometry scaling if present
    # (Optional: only if your specific instrument requires it)
    total_scaling = scaling_factor / result['integration'].get('scaling_factor', 1.0)

    return I * total_scaling
