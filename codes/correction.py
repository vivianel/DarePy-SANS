# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 10:59:14 2023

@author: lutzbueno_v
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from utils import create_analysis_folder
from utils import save_results
from utils import load_hdf
import normalization as norm
import pyFAI
import cv2


def prepare_corrections(config, result, det):
    path_dir_an = create_analysis_folder(config)
    path_det = os.path.join(path_dir_an, 'det' + det)
    # prepare radial integration files
    path_hdf_raw = os.path.join(path_det, 'hdf_raw/')
    beam_center = config['experiment']['calibration']['beam_center']
    class_trans = result['overview']['all_files']
    array = np.array(class_trans['sample_name'])
    indices = np.where(array == beam_center)[0]
    # prepare ai
    for ll in indices:
        if class_trans['detx_m'][ll] == float(det.replace('p', '.')) and class_trans['att'][ll] > 0:
            name_hdf = class_trans['name_hdf'][ll]
            ai = prepare_ai(config, beam_center, name_hdf, result)
    if 'ai' not in locals():
        print('###########################################################')
        print('An Empty beam is needed for this configuration: ' + det + ' m')
        print('###########################################################')
        sys.exit('Please measure an empty beam (EB).')


def prepare_ai(config, beam_center, name_hdf, result):
    pixel1 = config['instrument']['pixel_size']
    pixel2 = pixel1
    path_hdf_raw = config['analysis']['path_hdf_raw']
    path_dir_an = create_analysis_folder(config)
    dist = load_hdf(path_hdf_raw, name_hdf, 'detx')
    # calculate the beam center
    counts = load_hdf(path_hdf_raw, name_hdf, 'counts')
    bc_x, bc_y = calculate_beam_center(config, counts, name_hdf)
    poni2 = bc_x*pixel1
    poni1 = bc_y*pixel2
    result['integration']['beam_center_x'] = bc_x
    result['integration']['beam_center_y'] = bc_y
    # create a mask
    detector_size = config['instrument']['detector_size']
    mask = np.zeros([detector_size, detector_size])
    # find the size of the beam stopper, based on the list
    beam_stop = load_hdf(path_hdf_raw, name_hdf, 'beam_stop')
    list_bs = config['instrument']['list_bs']
    beam_stopper = list_bs[str(int(beam_stop))]
    beam_stopper = (beam_stopper/(pixel1*1000)/2)
    # load manual offsets for the mask
    [d_bc_x,d_bc_y,d_x_p,d_x_n,d_y_p,d_y_n] = config['analysis']['mask_offsets']

    # remove those pixels around the beam stopper
    y_p = int(bc_y + d_bc_y + beam_stopper) + d_y_p
    y_n = int(bc_y + d_bc_y - beam_stopper) - d_y_n
    x_p = int(bc_x + d_bc_x + beam_stopper) + d_x_p
    x_n = int(bc_x + d_bc_x - beam_stopper) - d_x_n

    mask[y_n:y_p, x_n:x_p] = 1
    # remove the lines around the detector
    lines = 2
    mask[:, 0:lines] = 1
    mask[:, detector_size - lines : detector_size + 1] = 1
    mask[0:lines, :] = 1
    mask[detector_size - lines:detector_size + 1, :] = 1
    # remove the last thick line - only for SANS-I
    if dist > 15:
        lines = 6
        mask[:, detector_size - lines : detector_size + 1] = 1
    # remove the corners
    corner = 10
    mask[0:corner, 0:corner] = 1
    mask[-corner:-1, 0:corner] = 1
    mask[-corner:-1, -corner:-1] = 1
    mask[0:corner, -corner:-1] = 1
    result['integration']['int_mask'] = mask
    wl = load_hdf(path_hdf_raw, name_hdf, 'wl')*1e-10  # from A to m
    # create the radial integrator
    ai = pyFAI.AzimuthalIntegrator(dist=dist, poni1=poni1, poni2=poni2,rot1=0,
                                   rot2=0, rot3=0, pixel1=pixel1, pixel2=pixel2,
                                   splineFile=None,  detector=None, wavelength=wl)
    ai.setChiDiscAtZero()
    result['integration']['ai'] = ai
    save_results(path_dir_an, result)
    return (ai, mask, result)


def calculate_beam_center(config, counts0, name_hdf):
    path_hdf_raw = config['analysis']['path_hdf_raw']
    dist = load_hdf(path_hdf_raw, name_hdf, 'detx')
    interpolation_factor = 2
    # reshape the image
    sizeX = counts0.shape[0]*interpolation_factor
    sizeY = counts0.shape[1]*interpolation_factor
    counts = cv2.resize(counts0, dsize=(sizeX, sizeY), interpolation=cv2.INTER_NEAREST)
    counts = np.where(counts <= 0, 1e-8, counts)
    cutoff = counts[counts > 0].max()/1.3
    counts = np.where(counts < cutoff, 0, counts)
    im = np.where(counts >= cutoff, 1, counts)
    # Find coordinates of thresholded image
    y, x = np.nonzero(im)
    # Find average
    xmean = x.mean()
    ymean = y.mean()
    bc_x = xmean/interpolation_factor
    bc_y = ymean/interpolation_factor
    # this seems to be needed for smaller detector distances
    if dist < 10:
        bc_x = bc_x + 1
        bc_y = bc_y + 1
    # turn off the interactivity
    plt.ioff()
    # Plot on figure
    plt.figure()
    #plt.imshow(np.dstack([im, im, im]))
    plt.imshow(counts0)
    plt.plot(bc_x, bc_y, 'r+', markersize=10)
    plt.title('x_center = ' + str(round(bc_x, 2)) + ', y_center =' + str(round(bc_y, 2)) + ' pixels')
    # Show image and make sure axis is removed
    plt.axis('off')
    path_dir_an = create_analysis_folder(config)
    file_name = path_dir_an + 'beamcenter_' + str(dist).replace('.', 'p') + 'm.jpg'
    plt.savefig(file_name)
    plt.close('all')
    return bc_x, bc_y


def load_standards(config, result, det):
    calibration = config['experiment']['calibration']
    class_file = result['overview']['det_files_' + det]
    path_dir_an = create_analysis_folder(config)
    for key, value in calibration.items():
        idx = class_file['sample_name'].index(value)
        name_hdf = class_file['name_hdf'][idx]
        if value in class_file['sample_name']:
            img = load_and_normalize(config, result, name_hdf)
            result['integration'][key] = img
        else:
            print('###########################################################')
            print('There is no ' + value + ' measurement for this configuration: ' + det + ' m')
            print('###########################################################')
            sys.exit('Please load a ' + value + ' measurement.')
    # subtract cadmium
    for key, value in calibration.items():
        if key != 'cadmium':
            result['integration'][key] = correct_dark(result['integration'][key], result['integration']['cadmium'])

    # subtract empty cell
    img_h2o = result['integration']['water']
    img_cell = result['integration']['water_cell']
    img_h2o = correct_EC(img_h2o, img_cell)

    # determine the scaling factor to replace water at 18 m
    ai = result['integration']['ai']
    mask = result['integration']['int_mask']
    # used for the correction factor
    q_h2o, I_h2o, sigma_h2o = ai.integrate1d(img_h2o,  200,
                                             correctSolidAngle=True,
                                             mask=mask,
                                             method = 'nosplit_csr',
                                             unit = 'q_A^-1',
                                             safe=True,
                                             error_model="azimuthal",
                                             flat = None,
                                             dark = None)
    # replace the water at 18 m
    replace_18m = config['analysis']['replace_18m']
    if det == '18p0' and replace_18m > 0:
        replace_18m = round(replace_18m, 1)
        det_m = str(replace_18m).replace('.', 'p')
        class_file = result['overview']['det_files_' + det_m]
        idx = class_file['sample_name'].index(calibration.get('water'))
        name_hdf = class_file['name_hdf'][idx]
        img_h2o_corr = load_and_normalize(config, result, name_hdf)
        idx = class_file['sample_name'].index(calibration.get('water_cell'))
        name_hdf = class_file['name_hdf'][idx]
        img_cell_corr = load_and_normalize(config, result, name_hdf)
        img_h2o_corr = np.subtract(img_h2o_corr, result['integration']['cadmium'])
        img_h2o_corr = np.subtract(img_h2o_corr,img_cell_corr)
        # get correction factor
        q_h2o_corr, I_h2o_corr, sigma_h2o_corr = ai.integrate1d(img_h2o_corr,  200,
                                                                correctSolidAngle=True,
                                                                mask=mask,
                                                                method = 'nosplit_csr',
                                                                unit = 'q_A^-1',
                                                                safe=True,
                                                                error_model="azimuthal",
                                                                flat = None,
                                                                dark = None)
        scaling_factor = (I_h2o[50:-10]/I_h2o_corr[50:-10]).mean()
        img_h2o = img_h2o_corr
        result['integration']['scaling_factor']= scaling_factor
    else:
        result['integration']['scaling_factor'] = 1
    # avoid negative numbers and zeros
    img_h2o[img_h2o <= 0] = 1e-8
    result['integration']['water'] = img_h2o

    save_results(path_dir_an, result)
    return result

def load_and_normalize(config, result, name_hdf):
    path_hdf_raw = config['analysis']['path_hdf_raw']
    counts = load_hdf(path_hdf_raw, name_hdf, 'counts')
    # counts = norm.normalize_time(config, name_hdf, counts)
    counts = norm.normalize_deadtime(config, name_hdf, counts)
    counts = norm.normalize_flux(config, name_hdf, counts)
    counts = norm.normalize_attenuator(config, name_hdf, counts)
    if config['experiment']['trans_dist'] > 0:
        counts = norm.normalize_transmission(config, name_hdf, result, counts)
    counts = norm.normalize_thickness(config, name_hdf, result, counts)
    return counts

def correct_dark(img, dark):
    img = np.subtract(img, dark)
    return img

def correct_EC(img, EC):
    img = np.subtract(img, EC)
    return img
