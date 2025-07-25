# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 11:13:29 2023

@author: lutzbueno_v
"""
"""
This module provides the function for performing absolute intensity calibration
of Small-Angle Neutron Scattering (SANS) data. It scales the scattering
intensity (I) to absolute units (cm^-1) using a flat field standard (typically water)
and propagates the associated errors.
"""

import numpy as np
import sys # Import sys for potential exit on critical errors

def absolute_calibration(config, result, file_name, I, sigma, I_flat, sigma_flat):
    """
    Applies absolute intensity calibration to the integrated SANS data.

    This involves:
    1. Applying a specific scaling factor for 18m detector distance if configured.
    2. Dividing the sample intensity by the flat field (e.g., water) intensity.
    3. Scaling to absolute units (cm^-1) using a wavelength-dependent calibration constant.
    4. Propagating errors through these operations.

    Args:
        config (dict): The main configuration dictionary, containing instrument
                       and analysis parameters (e.g., 'list_abs_calib').
        result (dict): The results dictionary, containing 'overview' data
                       (e.g., 'all_files' for wavelength) and 'integration'
                       (e.g., 'scaling_factor').
        file_name (str): The name of the file being processed, used to check
                         for 18m detector distance.
        I (np.ndarray): The 1D array of integrated intensity values for the sample.
        sigma (np.ndarray): The 1D array of standard deviations for the sample intensity.
        I_flat (np.ndarray): The 1D array of integrated intensity values for the flat field standard (e.g., water).
        sigma_flat (np.ndarray): The 1D array of standard deviations for the flat field intensity.

    Returns:
        tuple: A tuple containing:
            - I_corr (np.ndarray): The absolutely calibrated intensity values in cm^-1.
            - sigma_corr (np.ndarray): The propagated errors for the calibrated intensity.
    """
    # 1. Apply a scaling factor if the data is from 18m and a replacement water was used.
    # This factor corrects for potential differences in flux when using a water standard
    # from a different detector distance (e.g., 6m for 18m data).
    scaling_factor = result['integration']['scaling_factor']
    if '18p0' in file_name: # Check if the current file corresponds to 18m detector distance
        if scaling_factor is None or scaling_factor <= 0:
            print(f"Warning: Invalid or zero scaling factor ({scaling_factor}) for 18m data '{file_name}'. Skipping scaling correction.")
        else:
            I = I / scaling_factor
            # Error propagation for scaling: sigma_scaled = sigma_original / scaling_factor
            sigma = sigma / scaling_factor # (Implied from I scaling)

    # 2. Prepare I_flat for division: avoid division by zero.
    # Replace any non-positive values in I_flat with a very small positive number (epsilon).
    # This is crucial because I_flat is in the denominator.
    # np.finfo(I_flat.dtype).eps gets the smallest representable positive number for the array's data type.
    I_flat_safe = I_flat.copy() # Work on a copy to avoid modifying the original array in 'result'
    if I_flat_safe.size == 0 or np.all(I_flat_safe <= 0):
        print(f"Error: Flat field intensity (I_flat) is empty or all non-positive for '{file_name}'. Cannot perform absolute calibration.")
        # Return original I and sigma, or raise an error, depending on desired robustness.
        # For now, return original, indicating an issue without crashing the whole pipeline.
        return I, sigma

    I_flat_safe[I_flat_safe <= 0] = np.finfo(I_flat_safe.dtype).eps

    # Ensure sample intensity also doesn't have zeros for log plotting or ratios,
    # though I_flat is the more critical denominator.
    # The original code sets I[I <= 0] = np.median(np.abs(I[I>0]))
    # This can lead to issues if I[I>0] is empty. A safer approach for clipping might be needed.
    # For now, if I is 0 or negative, it will become very small positive.
    if I.size == 0:
        print(f"Warning: Sample intensity (I) is empty for '{file_name}'. Cannot perform absolute calibration.")
        return I, sigma # Return original if I is empty

    I_safe = I.copy()
    if np.all(I_safe <= 0):
        print(f"Warning: Sample intensity (I) is all non-positive for '{file_name}'. Replacing with median of non-zero positive values or epsilon.")
        # Fallback if no positive values: use epsilon, otherwise median of positive
        if np.any(I[I > 0]):
            I_safe[I_safe <= 0] = np.median(I[I > 0]) #
        else:
            I_safe[I_safe <= 0] = np.finfo(I_safe.dtype).eps # Use epsilon if no positive values exist

    # 3. Divide by flat field standard.
    # Ensure shapes match for element-wise division.
    if I_safe.shape != I_flat_safe.shape:
        print(f"Error: Sample intensity shape ({I_safe.shape}) and flat field intensity shape ({I_flat_safe.shape}) mismatch for '{file_name}'. Cannot perform absolute calibration.")
        return I, sigma # Return original if shapes don't match

    I_corr = np.divide(I_safe, I_flat_safe)

    # 4. Scale to absolute units (cm^-1)
    list_cs = config['instrument']['list_abs_calib']
    wl = str(int(result['overview']['all_files']['wl_A'][1]))
    if wl in list_cs.keys():
        correction = float(list_cs[str(wl)])
    else:
        correction = 1
        print('Wavelength has not been calibrated.')
    I_corr = I_corr*correction

    # 5. Error propagation.
    # Relative error squared for sample: (sigma/I)^2
    # Relative error squared for flat: (sigma_flat/I_flat)^2
    # Total relative error squared for I_corr: (sigma/I)^2 + (sigma_flat/I_flat)^2
    # sigma_corr = I_corr * sqrt((sigma/I)^2 + (sigma_flat/I_flat)^2)

    # Ensure sigma_flat_safe and sigma_safe also handle zeros or small values in their denominators
    sigma_flat_safe = sigma_flat.copy()
    sigma_safe = sigma.copy()

    # Avoid division by zero when calculating relative errors.
    # If a data point's intensity is 0 or very small, its relative error becomes undefined/very large.
    # For relative error calculation (sigma/I), where I is in the denominator for sigma,
    # if I_safe is small, sigma_safe / I_safe can become very large.
    # We should ensure that I_safe and I_flat_safe are not zero for relative error calculation.
    # Use the same safety values as for I_flat_safe and I_safe.
    I_flat_for_err = I_flat_safe # Already handled for zeros above
    I_for_err = I_safe # Already handled for zeros above

    # If I or I_flat were originally zero and replaced by epsilon,
    # their corresponding sigma_flat/sigma values might also need careful handling
    # to avoid NaNs or Infs if sigma itself is also zero.
    # Here, we assume sigma is reasonable if I is reasonable.

    # Calculate relative errors squared safely
    with np.errstate(divide='ignore', invalid='ignore'): # Suppress warnings for division by zero/invalid operations
        relative_error_I_sq = np.square(sigma_safe / I_for_err)
        relative_error_I_flat_sq = np.square(sigma_flat_safe / I_flat_for_err)

    # Replace NaNs or Infs that might arise from division by zero in relative_error_sq
    # This might happen if sigma and I were both 0, leading to 0/0 (NaN).
    relative_error_I_sq[np.isnan(relative_error_I_sq) | np.isinf(relative_error_I_sq)] = 0
    relative_error_I_flat_sq[np.isnan(relative_error_I_flat_sq) | np.isinf(relative_error_I_flat_sq)] = 0

    # Total relative error squared
    total_relative_error_sq = relative_error_I_sq + relative_error_I_flat_sq

    # Propagate the absolute calibration factor into the error
    sigma_corr = np.sqrt(total_relative_error_sq) * np.abs(I_corr)

    return (I_corr, sigma_corr)
