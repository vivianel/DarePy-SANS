# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 21:47:30 2023

@author: lutzbueno_v
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import organize_hdf_files as org
from load_hdf import load_hdf
import corrections as corr
import pyFAI
import plot_integration as plot_integ
import cv2


def set_radial_int(config, result):
    # find all files in the folder
    path_dir_an = org.create_analysis_folder(config)
    list_dir = list(os.listdir(path_dir_an))
    force_reintegrate = config['analysis']['force_reintegrate']
    perform_abs_calib = config['analysis']['perform_abs_calib']
    for kk in list_dir:
        if kk[0:3] == 'det':
            det = kk[4:]
            path_det = os.path.join(path_dir_an, str(kk))
            class_file = result['overview']['det_files_' + det]
            #  create poni and masks
            path_rad_int = os.path.join(path_det, 'integration/')
            if not os.path.exists(path_rad_int):
                os.mkdir(path_rad_int)
            # check if we want to integrate
            last_file = path_rad_int + 'rad_integ_' + str(class_file['scan'][-1]) + '_' + class_file['sample_name'][-1] + '_' + kk[4:] + 'm.dat'
            if os.path.exists(last_file) and force_reintegrate == 0:
                print('All files are already integrated at ' + det + 'm')
            else:
                prepare_corrections(config, result, det)
                if perform_abs_calib == 1:
                    result = load_standards(config, result, det)
                result = radial_integration(config, result, det, path_rad_int)
    return result


def prepare_corrections(config, result, det):
    path_dir_an = org.create_analysis_folder(config)
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
    path_dir_an = org.create_analysis_folder(config)
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
    # find the size of the beam stopper
    beam_stop = load_hdf(path_hdf_raw, name_hdf, 'beam_stop')
    list_bs = config['instrument']['list_bs']
    beam_stopper = list_bs[str(int(beam_stop))]
    beam_stopper = (beam_stopper/(pixel1*1000)/2)+1
    # increase the size for large detector distances distances
    if dist > 10:
        beam_stopper = beam_stopper + 2
    # remove those pixels around the beam stopper
    mask[int(bc_y-beam_stopper):int(bc_y+beam_stopper), int(bc_x-beam_stopper):int(bc_x+beam_stopper)] = 1

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
    org.save_results(path_dir_an, result)
    return (ai, mask, result)


def calculate_beam_center(config, counts0, name_hdf):
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
    path_hdf_raw = config['analysis']['path_hdf_raw']
    dist = load_hdf(path_hdf_raw, name_hdf, 'detx')
    path_dir_an = org.create_analysis_folder(config)
    file_name = path_dir_an + 'beamcenter_' + str(dist).replace('.', 'p') + 'm.jpg'
    plt.savefig(file_name)
    plt.close('all')
    return bc_x, bc_y


def load_and_correct(config, result, name_hdf):
    path_hdf_raw = config['analysis']['path_hdf_raw']
    counts = load_hdf(path_hdf_raw, name_hdf, 'counts')
    counts = corr.normalize_time(config, name_hdf, counts)
    counts = corr.deadtime_corrections(config, name_hdf, counts)
    counts = corr.normalize_flux(config, name_hdf, counts)
    counts = corr.correct_attenuator(config, name_hdf, counts)
    counts = corr.correct_transmission(config, name_hdf, result, counts)
    return counts


def load_standards(config, result, det):
    calibration = config['experiment']['calibration']
    class_file = result['overview']['det_files_' + det]
    for key, value in calibration.items():
        idx = class_file['sample_name'].index(value)
        name_hdf = class_file['name_hdf'][idx]
        if value in class_file['sample_name']:
            img = load_and_correct(config, result, name_hdf)
            result['integration'][key] = img
        else:
            print('###########################################################')
            print('There is no ' + value + ' measurement for this configuration: ' + det + ' m')
            print('###########################################################')
            sys.exit('Please load a ' + value + ' measurement.')
    # subtract cadmium
    for key, value in calibration.items():
        if key != 'cadmium':
            result['integration'][key] = np.subtract(result['integration'][key], result['integration']['cadmium'])

    # subtract empty cell
    img_h2o = result['integration']['water']
    img_cell = result['integration']['water_cell']
    img_h2o = np.subtract(img_h2o, img_cell)

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
    replace_h2o_18m = config['analysis']['replace_h2o_18m']
    if det == '18p0' and replace_h2o_18m > 0:
        det_m = str(replace_h2o_18m).replace('.', 'p')
        class_file = result['overview']['det_files_' + det_m]
        idx = class_file['sample_name'].index(calibration.get('water'))
        name_hdf = class_file['name_hdf'][idx]
        img_h2o_corr = load_and_correct(config, result, name_hdf)
        idx = class_file['sample_name'].index(calibration.get('water_cell'))
        name_hdf = class_file['name_hdf'][idx]
        img_cell_corr = load_and_correct(config, result, name_hdf)
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
    path_dir_an = org.create_analysis_folder(config)
    org.save_results(path_dir_an, result)
    return result


def normalize_water(config, result, ii, det, img):
    scaling_factor = result['integration']['scaling_factor']
    list_cs = config['instrument']['list_abs_calib']
    class_file = result['overview']['det_files_'+ det]
    wl = class_file['wl_A'][ii]/10 # from A into nm
    correction = float(list_cs[str(wl)])
    if det == '18p0':
        img = img/scaling_factor
    img1 = np.divide(img, result['integration']['water']) * correction
    return img1

def radial_integration(config, result, det, path_rad_int):
    plt.ioff()
    path_hdf_raw = config['analysis']['path_hdf_raw']
    class_file = result['overview']['det_files_'+ det]
    # correct to absolute scale
    perform_abs_calib = config['analysis']['perform_abs_calib']
    perform_azimuthal = config['analysis']['perform_azimuthal']
    perform_radial = config['analysis']['perform_radial']
    ai = result['integration']['ai']
    mask = result['integration']['int_mask']
    dark =  result['integration']['cadmium']
    class_file = result['overview']['det_files_'+ det]
    pixel_range = range(0, 200)
    result['integration']['pixel_range'] = pixel_range
    # execute the corrections for all
    print('DOING ' + str(det) + 'm')
    for ii in range(0, len(class_file['sample_name'])):
        name_hdf = class_file['name_hdf'][ii]
        sample_name = class_file['sample_name'][ii]
        scan_nr = class_file['scan'][ii]
        scanNr = f"{scan_nr:07d}"
        # do radial integration for each frame
        for ff in range(0, class_file['frame_nr'][ii]):
            if perform_abs_calib == 1:
                img = load_and_correct(config, result, name_hdf)
                # Subtract empty cell and Cadmium
                img_cell = result['integration']['empty_cell']
                # subraction of empty cell
                if class_file['frame_nr'][ii] > 1:
                    img1 =  np.subtract(img[ff,:,:], dark)
                    img1 =  np.subtract(img1, img_cell)
                else:
                    img1 =  np.subtract(img, dark)
                    img1 =  np.subtract(img1, img_cell)
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
            frame_name = f"{ff:04d}"
            frame = ff
            if perform_abs_calib == 1:
                img1 = normalize_water(config, result, ii, det, img1)
            # save the 2D image
            file_name = path_rad_int + 'pattern2D_'  + sample_name + '_' +'det' + det + 'm_'+ scanNr + '_'+ frame_name +'.dat'
            np.savetxt(file_name, img1, delimiter=',')

            # azimuthal integration
            if perform_azimuthal == 1:
                # define the number of sectors
                sectors_nr = 12
                # integrate for azimuthal plots
                npt_azim = range(0, 370, int(360/sectors_nr))
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
                        test = np.ones(img1.shape)
                        q_test, I_test, sigma_test = ai.integrate1d(test,  len(pixel_range),
                                                                 correctSolidAngle = True,
                                                                 mask = mask,
                                                                 method = 'nosplit_csr',
                                                                 unit = 'q_A^-1',
                                                                 safe = True,
                                                                 error_model = "azimuthal",
                                                                 azimuth_range = [azim_start, azim_end],
                                                                 flat = None,
                                                                 dark = None)
                        I_test[I_test <= 0] = 1
                        I = np.divide(I,I_test)
                    if rr == 0:
                        I_all = I
                        sigma_all = sigma
                    else:
                       I_all = np.column_stack((I_all,I))
                       sigma_all = np.column_stack((sigma_all, sigma))
                #save the data
                data_save = np.column_stack((q, I_all, sigma_all))
                file_name = path_rad_int + 'azim_integ_'  + sample_name + '_' +'det' + det + 'm_'+ scanNr + '_'+ frame_name +'.dat'
                header_text = 'q (A-1), ' + str(sectors_nr) + ' columns for absolute intensity  I (1/cm), '+ str(sectors_nr) + ' columns for standard deviation'
                np.savetxt(file_name, data_save, delimiter=',' , header = header_text)
                result['integration']['npt_azim'] = npt_azim
                plot_integ.plot_integ_azimuthal(config, result, scan_nr, frame)
                # save result
                path_dir_an = org.create_analysis_folder(config)
                org.save_results(path_dir_an, result)

            # for the radial integration
            if perform_radial == 1:
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
                    test = np.ones(img1.shape)
                    q_test, I_test, sigma_test = ai.integrate1d(test,  len(pixel_range),
                                                             correctSolidAngle = True,
                                                             mask = mask,
                                                             method = 'nosplit_csr',
                                                             unit = 'q_A^-1',
                                                             safe = True,
                                                             error_model="azimuthal",
                                                             flat = None,
                                                             dark = None)
                    I_test[I_test == 0] = 1
                    I = I/I_test
                data_save = np.column_stack((q, I, sigma))
                file_name = path_rad_int + 'radial_integ_'  + sample_name + '_' +'det' + det + 'm_'+ scanNr + '_'+ frame_name +'.dat'
                header_text = 'q (A-1), absolute intensity  I (1/cm), standard deviation'
                np.savetxt(file_name, data_save, delimiter=',' , header = header_text)
                # plot and save the results
                path_dir_an = org.create_analysis_folder(config)
                plot_integ.plot_integ_radial(config, result, scan_nr, frame)
                # save result
                path_dir_an = org.create_analysis_folder(config)
                org.save_results(path_dir_an, result)
    return result
