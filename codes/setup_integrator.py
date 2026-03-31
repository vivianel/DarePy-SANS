# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 11:05:51 2026

@author: lutzbueno_v
"""

import matplotlib.pyplot as plt
import os
import sys
import pyFAI


from utils import create_analysis_folder, save_results, load_hdf


# ==========================================
# STEP 1: MASK GENERATION & BEAM CENTER
# ==========================================
def generate_beamstop_mask(config, result, det):
    """
    Creates the 2D mask array based on beamstop coordinates and detector edges.
    Extracts the beam center and saves both to the 'result' dictionary.
    """
    import sys
    import numpy as np

    detector_size = config['instrument']['detector_size']
    mask = np.zeros([detector_size, detector_size], dtype=int)
    det_float_key = float(det.replace('p', '.'))

    # --- 1. MASK LOGIC ---
    beamstopper_coordinates = config['analysis']['beamstopper_coordinates']

    if det_float_key in beamstopper_coordinates:
        bs_all = beamstopper_coordinates[det_float_key]
        try:
            for jj in range(len(bs_all)):
                bs_select = 'bs' + str(jj)
                if bs_select in bs_all:
                    y_n, y_p, x_n, x_p = bs_all[bs_select]
                    mask[y_n:y_p, x_n:x_p] = 1
        except:
            if len(bs_all) == 4:
                y_n, y_p, x_n, x_p = bs_all
                mask[y_n:y_p, x_n:x_p] = 1

    # Edge masking
    lines = 1
    mask[:, 0:lines] = 1
    mask[:, detector_size - lines:detector_size] = 1
    mask[0:lines, :] = 1
    mask[detector_size - lines:detector_size, :] = 1

    # --- 2. BEAM CENTER LOGIC ---
    beam_center_guess = config['analysis']['beam_center_guess']
    if det_float_key not in beam_center_guess:
        print(f"Error: Beam center guess not provided for detector distance {det_float_key}m.")
        sys.exit(1)

    bc_x = beam_center_guess[det_float_key][0]
    bc_y = beam_center_guess[det_float_key][1]

    # --- 3. SAVE TO RESULT DICTIONARY ---
    if 'integration' not in result:
        result['integration'] = {}

    result['integration']['int_mask'] = mask
    result['integration']['beam_center_x'] = bc_x
    result['integration']['beam_center_y'] = bc_y

    return mask, result

# ==========================================
# STEP 2: INTEGRATOR SETUP
# ==========================================
def setup_integration(config, result, det):
    """
    Sets up the pyFAI azimuthal integrator, plots the mask, and saves results.
    """
    path_dir_an = create_analysis_folder(config)
    path_hdf_raw = config['analysis']['path_hdf_raw']
    mask =  result['integration']['int_mask']

    class_file_key = 'det_files_' + det
    if class_file_key not in result['overview'] or not result['overview'][class_file_key]['name_hdf']:
        print(f"Error: No valid HDF5 files found in metadata for {det}m. Cannot prepare pyFAI.")
        sys.exit(1)

    first_hdf_name = result['overview'][class_file_key]['name_hdf'][0]

    # Load physical parameters
    dist = load_hdf(path_hdf_raw, first_hdf_name, 'detx')
    if dist is None or not isinstance(dist, (int, float)):
        print(f"Error: Could not load valid detector distance for {first_hdf_name}.")
        sys.exit(1)

    pixel1 = config['instrument']['pixel_size']
    pixel2 = pixel1

    wl_input = config['experiment']['wl_input']
    if wl_input == 'auto':
        wl = load_hdf(path_hdf_raw, first_hdf_name, 'wl')
        if wl is None:
             print(f"Error: Could not load valid wavelength for {first_hdf_name}.")
             sys.exit(1)
        wl *= 1e-10
    else:
        wl = wl_input * 1e-10

    # Beam center logic
    beam_center_guess = config['analysis']['beam_center_guess']
    det_float_key = float(det.replace('p', '.'))

    if det_float_key not in beam_center_guess:
        print(f"Error: Beam center guess not provided for detector distance {det_float_key}m.")
        sys.exit(1)

    bc_x = beam_center_guess[det_float_key][0]
    bc_y = beam_center_guess[det_float_key][1]

    poni2 = bc_x * pixel1
    poni1 = bc_y * pixel2

    # Save metadata to dictionary
    result['integration']['beam_center_x'] = bc_x
    result['integration']['beam_center_y'] = bc_y
    result['integration']['int_mask'] = mask

    # Create pyFAI Integrator
    ai = pyFAI.integrator.azimuthal.AzimuthalIntegrator(dist=dist, poni1=poni1, poni2=poni2,
                                   rot1=0, rot2=0, rot3=0,
                                   pixel1=pixel1, pixel2=pixel2,
                                   splineFile=None, detector=None, wavelength=wl)
    ai.setChiDiscAtZero()
    result['integration']['ai'] = ai

    # Save visualization
    plt.ioff()
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, origin='lower', aspect='equal', clim=[0, 1], cmap='gray')
    plt.plot(bc_x, bc_y, 'r+', markersize=10, markeredgewidth=2)
    plt.title(f'Detector Mask for {dist:.1f}m\nBeam Center: ({bc_x:.2f}, {bc_y:.2f})')
    file_name = os.path.join(path_dir_an, f'beamcenter_mask_det{det}.jpg')
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close('all')
    plt.ion()

    save_results(path_dir_an, result)
    return (ai, result)
