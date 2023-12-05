# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 21:47:30 2023

@author: lutzbueno_v
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from utils import load_hdf
import re
import plot_integration as plot_integ
from utils import create_analysis_folder
from utils import save_results
from correction import prepare_corrections
from correction import load_standards
from correction import load_and_normalize
from correction import correct_dark
from correction import correct_EC
from calibration import absolute_calibration


def set_integration(config, result):
    # find all files in the folder
    path_dir_an = create_analysis_folder(config)
    list_dir = list(os.listdir(path_dir_an))
    force_reintegrate = config['analysis']['force_reintegrate']
    perform_abs_calib = config['analysis']['perform_abs_calib']
    for folder_name in list_dir:
        if folder_name[0:3] == 'det':
            det = folder_name[4:]
            path_det = os.path.join(path_dir_an, str(folder_name))
            #  create poni and masks
            path_rad_int = os.path.join(path_det, 'integration/')
            if not os.path.exists(path_rad_int):
                os.mkdir(path_rad_int)
            # name the sample
            path = path_rad_int
            prefix = 'radial_integ'
            class_file = result['overview']['det_files_' + det]
            scanNr = class_file['scan'][-1]
            sample_name = class_file['sample_name'][-1]
            frame = 0
            sufix = 'dat'
            last_file = make_file_name(path, prefix, sufix, sample_name, det, scanNr, frame)
            # check if we want to integrate
            if os.path.exists(last_file) and force_reintegrate == 0:
                print('All files are already integrated at ' + det + 'm')
            else:
                prepare_corrections(config, result, det)
                if perform_abs_calib == 1:
                    result = load_standards(config, result, det)
                result = integrate(config, result, det, path_rad_int)
    return result

def make_file_name(path, prefix, sufix, sample_name, det, scanNr, frame):
    file_n = path + prefix + '_' + f"{scanNr:07d}" + '_'+ f"{frame:05d}_"  + sample_name + '_' +'det' + det + 'm'+ '.' + sufix
    return file_n

def integrate(config, result, det, path_rad_int):
    plt.ioff()
    path_hdf_raw = config['analysis']['path_hdf_raw']
    class_file = result['overview']['det_files_'+ det]
    # correct to absolute scale
    perform_abs_calib = config['analysis']['perform_abs_calib']
    perform_azimuthal = config['analysis']['perform_azimuthal']
    perform_radial = config['analysis']['perform_radial']
    class_file = result['overview']['det_files_'+ det]

    # pixel range defines how many q the final curve will contain
    pixel_range = range(0, 100)
    result['integration']['pixel_range'] = pixel_range
    # execute the corrections for all
    print('DOING ' + str(det) + 'm')
    for ii in range(0, len(class_file['sample_name'])):
        name_hdf = class_file['name_hdf'][ii]
        sample_name = class_file['sample_name'][ii]
        scanNr = class_file['scan'][ii]
        # do radial integration for each frame
        for ff in range(0, class_file['frame_nr'][ii]):
            if perform_abs_calib == 1:
                dark =  result['integration']['cadmium']
                img = load_and_normalize(config, result, name_hdf)
                # Subtract empty cell and Cadmium
                img_cell = result['integration']['empty_cell']
                # subraction of empty cell
                if class_file['frame_nr'][ii] > 1:
                    img1 =  correct_dark(img[ff,:,:], dark)
                    img1 =  correct_EC(img1, img_cell)
                else:
                    img1 =  correct_dark(img, dark)
                    img1 =  correct_EC(img1, img_cell)
                print('Corrected scan ' + class_file['name_hdf'][ii] + ', Frame: ' + str(ff) )
            else:
                img = load_hdf(path_hdf_raw, name_hdf, 'counts')
                if class_file['frame_nr'][ii] > 1:
                    img1 = img[ff,:,:]
                else:
                    img1=img
                print('NOT corrected scan ' + class_file['name_hdf'][ii] + ', Frame: ' + str(ff) )
            img1= np.squeeze(img1)
            # get the frame number
            frame = ff
            # azimuthal integration
            if perform_radial == 1:
                # name the sample
                prefix = 'pattern2D'
                sufix = 'dat'
                file_name = make_file_name(path_rad_int, prefix, sufix, sample_name, det, scanNr, frame)
                np.savetxt(file_name, img1, delimiter=',')
                # name the sample
                prefix = 'radial_integ'
                sufix = 'dat'
                file_name = make_file_name(path_rad_int, prefix, sufix, sample_name, det, scanNr, frame)
                radial_integ(config, result, img1, file_name)
            if perform_azimuthal == 1:
                # name the sample
                prefix = 'azim_integ'
                sufix = 'dat'
                file_name = make_file_name(path_rad_int, prefix, sufix,  sample_name, det, scanNr, frame)
                azimuthal_integ(config, result, img1, file_name)
            plot_radial_integ(config, result, file_name)
    return result


def radial_integ(config, result, img1, file_name):
    ai = result['integration']['ai']
    mask = result['integration']['int_mask']
    pixel_range = result['integration']['pixel_range']
    perform_abs_calib = config['analysis']['perform_abs_calib']
    # integrate for radial plots
    q, I, sigma = ai.integrate1d(img1, len(pixel_range),
                                 correctSolidAngle = True,
                                 mask = mask,
                                 method = 'nosplit_csr',
                                 unit = 'q_A^-1',
                                 safe = True,
                                 error_model="azimuthal",
                                 flat = None,
                                 dark = None)
    if perform_abs_calib == 1:
        # correct for the number of pixels
        flat =     flat = result['integration']['water']
        q_flat, I_flat, sigma_flat = ai.integrate1d(flat,  len(pixel_range),
                                                 correctSolidAngle = True,
                                                 mask = mask,
                                                 method = 'nosplit_csr',
                                                 unit = 'q_A^-1',
                                                 safe = True,
                                                 error_model="azimuthal",
                                                 flat = None,
                                                 dark = None)
        I, sigma = absolute_calibration(config, result, file_name, I, sigma, I_flat)
    # save the integrated files
    data_save = np.column_stack((q, I, sigma))
    header_text = 'q (A-1), absolute intensity  I (1/cm), standard deviation'
    np.savetxt(file_name, data_save, delimiter=',' , header = header_text)
    # save result
    path_dir_an = create_analysis_folder(config)
    save_results(path_dir_an, result)


def azimuthal_integ(config, result, img1, file_name):
    ai = result['integration']['ai']
    mask = result['integration']['int_mask']
    pixel_range = result['integration']['pixel_range']
    perform_abs_calib = config['analysis']['perform_abs_calib']
    # define the number of sectors
    sectors_nr = 16
    # integrate for azimuthal plots
    npt_azim = range(0, 370, int(360/sectors_nr))
    result['integration']['sectors_nr'] = sectors_nr
    result['integration']['npt_azim'] = npt_azim
    for rr in range(0, len(npt_azim)-1):
        azim_start = npt_azim[rr]
        azim_end = npt_azim[rr+1]
        q, I, sigma = ai.integrate1d(img1, len(pixel_range),
                                     correctSolidAngle = True,
                                     mask = mask,
                                     method = 'nosplit_csr',
                                     unit = 'q_A^-1',
                                     safe = True,
                                     error_model = "azimuthal",
                                     azimuth_range = [azim_start, azim_end],
                                     flat = None,
                                     dark = None)
        if perform_abs_calib == 1:
            # correct for the number of pixels
            flat = result['integration']['water']
            q_flat, I_flat, sigma_flat = ai.integrate1d(flat,  len(pixel_range),
                                                     correctSolidAngle = True,
                                                     mask = mask,
                                                     method = 'nosplit_csr',
                                                     unit = 'q_A^-1',
                                                     safe = True,
                                                     error_model = "azimuthal",
                                                     azimuth_range = [azim_start, azim_end],
                                                     flat = None,
                                                     dark = None)
            I, sigma = absolute_calibration(config, result, file_name, I, sigma, I_flat)
        if rr == 0:
            I_all = I
            sigma_all = sigma
        else:
           I_all = np.column_stack((I_all,I))
           sigma_all = np.column_stack((sigma_all, sigma))
    #save the integrated data
    data_save = np.column_stack((q, I_all, sigma_all))
    header_text = 'q (A-1), ' + str(sectors_nr) + ' columns for absolute intensity  I (1/cm), '+ str(sectors_nr) + ' columns for standard deviation'
    np.savetxt(file_name, data_save, delimiter=',' , header = header_text)
    # save result
    path_dir_an = create_analysis_folder(config)
    save_results(path_dir_an, result)

def plot_radial_integ(config, result, file_name):
    # plot and save the results
    if config['analysis']['plot_azimuthal'] ==1:
        ScanNr = int(re.findall(r"\D(\d{7})\D", file_name)[0])
        Frame = int(re.findall(r"\D(\d{5})\D", file_name)[0])
        plot_integ.plot_integ_azimuthal(config, result, ScanNr, Frame)

    if config['analysis']['plot_radial'] ==1:
        ScanNr = int(re.findall(r"\D(\d{7})\D", file_name)[0])
        Frame = int(re.findall(r"\D(\d{5})\D", file_name)[0])
        plot_integ.plot_integ_radial(config, result, ScanNr, Frame)
