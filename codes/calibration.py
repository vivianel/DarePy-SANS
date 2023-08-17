# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 11:13:29 2023

@author: lutzbueno_v
"""

import numpy as np

def absolute_calibration(config, result, file_name, I, sigma, I_flat):
    #scale the data at 18 m due to reduction of flux. Use a water measurement
    scaling_factor = result['integration']['scaling_factor']
    if '18p0' in file_name:
        I = I/scaling_factor
    # divide by flat sample - water
    I_flat[I_flat <= 0] = 1
    I = np.divide(I, I_flat)
    sigma = np.divide(sigma, I_flat)
    # scale to absolute units (cm -1)
    list_cs = config['instrument']['list_abs_calib']
    wl = str(int(result['overview']['all_files']['wl_A'][1]))
    if wl in list_cs.keys():
        correction = float(list_cs[str(wl)])
    else:
        correction = 1
        print('Wavelength has not been calibrated.')
    I = I*correction
    return (I, sigma)
