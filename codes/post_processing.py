# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 20:50:57 2022

@author: lutzbueno_v

This function automatically marges the data collected in different detector distances
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import integration as integ
from scipy.interpolate import interp1d
import scipy.optimize
import os

# %% plot_all_data

def plot_all_data(path_dir_an):
    # create all paths for the merged data
    path_merged = os.path.join(path_dir_an, 'merged/')
    if not os.path.exists(path_merged):
        os.mkdir(path_merged)
    path_merged_fig = os.path.join(path_merged, 'figures/')
    if not os.path.exists(path_merged_fig):
        os.mkdir(path_merged_fig)
    path_merged_txt = os.path.join(path_merged, 'data_txt/')
    if not os.path.exists(path_merged_txt):
        os.mkdir(path_merged_txt)

    file_results = os.path.join(path_dir_an, 'result.npy')
    with open(file_results, 'rb') as handle:
        result = pickle.load(handle)
    file_config = os.path.join(path_dir_an, 'config.npy')
    with open(file_config, 'rb') as handle:
        config = pickle.load(handle)

    calibration = config['experiment']['calibration']

    list_class_files = result['overview']

    merged_files = {}

    # create a dictionary with all intensities and q for all detectors
    for keys in list_class_files:
        if 'det' in keys:
            total_samples = len(list_class_files[keys]['scan'])
            for ii in range(total_samples):
                sample_name = list_class_files[keys]['sample_name'][ii]
                if sample_name not in calibration.values():
                    # load the radial integration
                    prefix = 'radial_integ'
                    sufix = 'dat'
                    ScanNr = list_class_files[keys]['scan'][ii]
                    det = list_class_files[keys]['detx_m'][ii]
                    Frame = 0
                    path_integ = path_dir_an + '/det_' + str(det).replace('.','p') + '/integration/'
                    file_name = integ.make_file_name(path_integ, prefix, sufix, sample_name, str(det).replace('.','p'), ScanNr, Frame)
                    q = np.genfromtxt(file_name,
                                         dtype = None,
                                         delimiter = ',',
                                         usecols = 0)
                    I = np.genfromtxt(file_name,
                                         dtype = None,
                                         delimiter = ',',
                                         usecols = 1)
                    e = np.genfromtxt(file_name,
                                         dtype = None,
                                         delimiter = ',',
                                         usecols = 2)
                    if sample_name in merged_files:
                        temp = merged_files[sample_name]['I']
                        merged_files[sample_name]['I'] =  np.vstack((temp, I))
                        temp1 = merged_files[sample_name]['q']
                        merged_files[sample_name]['q'] =  np.vstack((temp1, q))
                        temp2 = merged_files[sample_name]['error']
                        merged_files[sample_name]['error'] =  np.vstack((temp2, e))
                    else:
                        merged_files[sample_name] = {}
                        merged_files[sample_name]['I'] = I
                        merged_files[sample_name]['q'] = q
                        merged_files[sample_name]['error'] = e

    # plot the files
    for keys in merged_files:
        plt.close('all')
        plt.ioff()
        if merged_files[keys]['q'].ndim > 1:
            dd = merged_files[keys]['q'].shape[0]
            for ii in range(dd):
                q = merged_files[keys]['q'][ii, :]
                I = merged_files[keys]['I'][ii,:]
                e = merged_files[keys]['error'][ii,:]
                plt.errorbar(q, I, e, lw = 0.3, marker = 'o',  ms = 2)

            plt.xlabel(r'Scattering vector q [$\AA^{-1}$]')
            plt.xscale('log')
            plt.yscale('log')
            plt.ylabel(r'Intensity I [cm$^{-1}$]')
            plt.title('Sample: '+  keys)
            file_name = path_merged_fig + keys + '_all_det_dist' + '.jpeg'
            plt.savefig(file_name)

        else:
            print('_____________________________')
            print('Single detector distance measurement: ' + keys)
            print('_____________________________')
    return merged_files

# %% merging_data
def merging_data(path_dir_an, merged_files, skip_start, skip_end, interp_type, interp_points):
    #  merge the files
    # low to high q
    path_merged = os.path.join(path_dir_an, 'merged/')
    path_merged_fig = os.path.join(path_merged, 'figures/')
    if not os.path.exists(path_merged_fig):
        os.mkdir(path_merged_fig)
    path_merged_txt = os.path.join(path_merged, 'data_txt/')
    interp_points = 100

    for keys in merged_files:
        # initiate variables to join the values
        I_all = []
        q_all = []
        e_all = []
        plt.close('all')
        plt.ioff()


        if merged_files[keys]['q'].ndim > 1:
            # define to order for the detector distances
            range_det = merged_files[keys]['q'].shape[0]
            first_q = []
            for kk in range(range_det):
                first_q.append(merged_files[keys]['q'][kk][0])
            idx_det = np.argsort(first_q)
            for ii in idx_det:
                # get the values skipping the start and end points defined in the function
                q = merged_files[keys]['q'][ii, :]
                q = q[skip_start[str(ii)]:len(q)-skip_end[str(ii)]]
                I = merged_files[keys]['I'][ii,:]
                I = I[skip_start[str(ii)]:len(I)-skip_end[str(ii)]]
                e = merged_files[keys]['error'][ii,:]
                e = e[skip_start[str(ii)]:len(e)-skip_end[str(ii)]]
                q_all = np.concatenate((q_all, q), axis = None)
                I_all = np.concatenate((I_all, I), axis = None)
                e_all = np.concatenate((e_all, e), axis = None)


                # checking if there is the need to shift slighly the plot
                if ii == 0:
                    # define where the q0 fits the qmax
                    start_pt = np.where(np.round(q_all, 2) == np.round(q[0], 2))
                    start_pt = start_pt[0][-1]
                    end_pt = np.where(np.round(q, 2) == np.round(q_all[-1], 2))
                    end_pt = end_pt[0][0]

                    # if there is the need for an adjustmet in the patterns
                    scaling = np.median(I_all[start_pt:])/np.median(I[:end_pt])

                    if np.isnan(scaling):
                        scaling = 1
                    I = np.multiply(I, scaling)

                    # save after the multiplication
                    q_all = np.concatenate((q_all, q), axis = None)
                    I_all = np.concatenate((I_all, I), axis = None)
                    e_all = np.concatenate((e_all, e), axis = None)
                else:
                    q_all = np.concatenate((q_all, q), axis = None)
                    I_all = np.concatenate((I_all, I), axis = None)
                    e_all = np.concatenate((e_all, e), axis = None)
                # for indexing the points

        if merged_files[keys]['q'].ndim > 1:
            idx = np.argsort(q_all)
            q_all = q_all[idx]
            I_all = I_all[idx]
            e_all = e_all[idx]
            # plot original data
            plt.errorbar(q_all, I_all, e_all, lw = 0, marker = 'o',  ms = 7, color = 'blue', alpha = 0.1, label = 'merged, scale = ' + str(np.round(scaling, 4)))
            if interp_type == 'log':
                # Interpolate it to new time points
                min_pt = np.round(np.log10(np.min(q_all))/1.01, 3)
                max_pt = np.round(np.log10(np.max(q_all))*1.01, 3)
                interpolation_pts = np.logspace(min_pt, max_pt, interp_points)
                linear_interp = interp1d(q_all, I_all)
                linear_results = linear_interp(interpolation_pts)
                interpolation_pts = np.append(q_all[:2], interpolation_pts)
                linear_results = np.append(I_all[:2], linear_results)
                plt.loglog(interpolation_pts, linear_results, lw = 0.3,
                           marker = 'o',  ms = 4, color = 'red', label = 'interpolated')
            if interp_type == 'linear':
                # Interpolate it to new time points
                interpolation_pts = np.linspace(np.min(q_all), np.max(q_all), interp_points)
                linear_interp = interp1d(q_all, I_all)
                linear_results = linear_interp(interpolation_pts)
                interpolation_pts = np.append(q_all[:2], interpolation_pts)
                linear_results = np.append(I_all[:2], linear_results)
                plt.loglog(interpolation_pts, linear_results, lw = 0.3,
                           marker = 'o',  ms = 4, color = 'red', label = 'interpolated')

            plt.xlabel(r'Scattering vector q [$\AA^{-1}$]')
            plt.xscale('log')
            plt.yscale('log')
            plt.ylabel(r'Intensity I [cm$^{-1}$]')
            plt.title('Sample: '+  keys)
            plt.legend()

            file_name = path_merged_fig + keys + '_merged' + '.jpeg'
            plt.savefig(file_name)

            header_text = 'q (A-1), I (1/cm), error'
            file_name = path_merged_txt +  keys + '_merged' + '.dat'
            data_save = np.column_stack((q_all, I_all, e_all))
            np.savetxt(file_name, data_save, delimiter=',', header=header_text)

            if interp_type == 'linear' or interp_type == 'log':
                file_name = path_merged_txt + keys +  '_interpolated' + '.dat'
                data_save = np.column_stack((interpolation_pts, linear_results))
                np.savetxt(file_name, data_save, delimiter=',', header=header_text)

# %%
def subtract_incoherent(path_dir_an, fitting_range):
    path_merged = path_dir_an + '/merged'
    path_merged_txt = path_merged +  '/data_txt/'

    file_name = os.path.join(path_dir_an, 'config.npy')
    with open(file_name, 'rb') as handle:
        config = pickle.load(handle)

    calibration = config['experiment']['calibration']

    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path_merged_txt):
        if 'interpolated' in f:
            for file in f:
                if 'interpolated' in file:
                    files.append(os.path.join(file))
                    # start of the name
                    idx = 17
        else:
            for file in f:
                if 'merged' in file:
                    files.append(os.path.join(file))
                    # start of the name
                    idx = 11


    for ii in range(0, len(files)):
        if files[ii][:-idx] not in calibration.values():
            plt.close('all')
            #plt.ioff()
            file_name =  path_merged_txt + files[ii]
            print(file_name)
            I = np.genfromtxt(file_name,
                                 dtype = None,
                                 delimiter = ',',
                                 usecols = 1)
            q = np.genfromtxt(file_name,
                                 dtype = None,
                                 delimiter = ',',
                                 usecols = 0)


            off_set = len(I) - fitting_range
            # skip the first points
            fitting_I = I[off_set:]
            fitting_q = q[off_set:]

            def porod(q, coef, slope, incoherent):
                return (coef * q**(slope-4) + incoherent)

            base = np.polyfit(fitting_q, fitting_I, 0)
            base = np.float64(base)
            # perform the fit
            params, cv = scipy.optimize.curve_fit(porod, fitting_q, fitting_I, bounds = ((0, -10, 0),(np.inf, 1, base )))
            m, t, b = params
            coeff = m
            slope = t
            incoherent = b

            # determine quality of the fit
            squaredDiffs = np.square(fitting_I - porod(fitting_q, m, t, b))
            squaredDiffsFromMean = np.square(fitting_I - np.mean(fitting_I))
            rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
            rSquared = round(rSquared, 2)


            # plot the results
            plt.loglog(q, I, '.', label="original data", color = 'blue', alpha = 0.2)
            plt.loglog(fitting_q, porod(fitting_q, m, t, b), '--', color = 'blue', label= f"fit - R² = {rSquared}")
            plt.title("Fitted Porod Approximation")
            plt.loglog(q, np.ones(len(I))*incoherent, '--', color = 'red', label = 'incoherent = ' + str(round(incoherent, 4)))
            plt.loglog(q, coeff * q**(slope-4), '--', color = 'black', label = 'slope = ' + str(round(slope-4, 2)))
            plt.loglog(q, I-incoherent, '.', color = 'black', label = 'subtracted data')
            plt.legend()

            path_merged_fig = path_merged +  '/figures/'
            file_name = path_merged_fig + files[ii][:-idx] + '_subtracted' + '.jpeg'
            plt.savefig(file_name)

            header_text = 'q (A-1), I (1/cm)'
            file_name = path_merged_txt +  files[ii][:-idx] + '_subtracted' + '.dat'
            data_save = np.column_stack((q, I-incoherent))
            np.savetxt(file_name, data_save, delimiter=',', header=header_text)
