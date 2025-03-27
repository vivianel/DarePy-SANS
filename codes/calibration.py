# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 11:13:29 2023

@author: lutzbueno_v
"""

import numpy as np

def absolute_calibration(config, result, file_name, I, sigma, I_flat, sigma_flat):
    #scale the data at 18 m due to reduction of flux. Use a water measurement
    scaling_factor = result['integration']['scaling_factor']
    if '18p0' in file_name:
        I = I/scaling_factor
    # avoid getting errors from the division by zero: here we have the 1D scattering pattern
    I_flat[I_flat <= 0] = np.median(np.abs(I_flat))
    I[I <= 0] = np.median(np.abs(I[I>0]))
    # divide by flat sample - water
    I_corr = np.divide(I, I_flat)
    # scale to absolute units (cm -1)
    list_cs = config['instrument']['list_abs_calib']
    wl = str(int(result['overview']['all_files']['wl_A'][1]))
    if wl in list_cs.keys():
        correction = float(list_cs[str(wl)])
    else:
        correction = 1
        print('Wavelength has not been calibrated.')
    I_corr = I_corr*correction
    # Error propagation
    sigma_corr = np.sqrt(np.square(sigma/I) + np.square(sigma_flat/I_flat))
    sigma_corr = np.multiply(sigma_corr, I_corr)
    return (I_corr, sigma_corr)
