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

    file_name = os.path.join(path_dir_an, 'result.npy')
    with open(file_name, 'rb') as handle:
        result = pickle.load(handle)
    file_name = os.path.join(path_dir_an, 'config.npy')
    with open(file_name, 'rb') as handle:
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
                    I = np.genfromtxt(file_name,
                                         dtype = None,
                                         delimiter = ',',
                                         usecols = 1)
                    if sample_name in merged_files:
                        temp = merged_files[sample_name]
                        merged_files[sample_name] =  np.vstack((temp, I))
                    else:
                        merged_files[sample_name] = I
                q = np.genfromtxt(file_name,
                                     dtype = None,
                                     delimiter = ',',
                                     usecols = 0)
            if 'q' in merged_files:
                temp = merged_files['q']
                merged_files['q'] = np.vstack((temp, q))
            else:
                merged_files['q'] = q

    # plot the files
    for keys in merged_files:
        plt.close('all')
        plt.ioff()
        if merged_files[keys].ndim > 1 and merged_files[keys].shape[0] == merged_files['q'].shape[0]:
            for ii in range(merged_files['q'].shape[0]):
                q = merged_files['q'][ii, :]
                I = merged_files[keys][ii,:]
                plt.loglog(q, I, lw = 0.3, marker = 'o',  ms = 2)
            if keys != 'q':
                plt.xlabel(r'Scattering vector q [$\AA^{-1}$]')
                plt.xscale('log')
                plt.yscale('log')
                plt.ylabel(r'Intensity I [cm$^{-1}$]')
                plt.title('Sample: '+  keys)
                file_name = path_merged_fig + keys + '_all_det_dist' + '.jpeg'
                plt.savefig(file_name)
    return merged_files

# %% merging_data
def merging_data(path_dir_an, merged_files, skip_start, skip_end):
    #  merge the files
    # low to high q
    path_merged = os.path.join(path_dir_an, 'merged/')
    path_merged_fig = os.path.join(path_merged, 'figures/')
    if not os.path.exists(path_merged_fig):
        os.mkdir(path_merged_fig)
    path_merged_txt = os.path.join(path_merged, 'data_txt/')
    interp_points = 100
    interp_type = 'log' # 'log' or 'linear'

    for keys in merged_files:
        I_all = []
        q_all = []
        plt.close('all')
        plt.ioff()
        range_det = (range(merged_files['q'].shape[0]-1, -1, -1))
        count = 0
        if merged_files[keys].ndim > 1 and merged_files[keys].shape[0] == merged_files['q'].shape[0]:
            for ii in range_det:
                q = merged_files['q'][ii, :]
                q = q[skip_start[count]:len(q)-skip_end[count]]
                I = merged_files[keys][ii,:]
                I = I[skip_start[count]:len(I)-skip_end[count]]
                if ii == range_det[0]:
                    q_all = np.concatenate((q_all, q), axis = None)
                    I_all = np.concatenate((I_all, I), axis = None)
                else:
                    start_pt = np.where(np.round(q_all, 2) == np.round(q[0], 2))
                    start_pt = start_pt[0][-1]
                    end_pt = np.where(np.round(q, 2) == np.round(q_all[-1], 2))
                    end_pt = end_pt[0][0]
                    scaling = np.median(I_all[start_pt:])/np.median(I[:end_pt])
                    if np.isnan(scaling):
                        scaling = 1
                    I = np.multiply(I, scaling)
                    q_all = np.concatenate((q_all, q), axis = None)
                    I_all = np.concatenate((I_all, I), axis = None)
                count = count + 1
            if keys != 'q':
                if merged_files[keys].ndim > 1:
                    idx = np.argsort(q_all)
                    q_all = q_all[idx]
                    I_all = I_all[idx]
                    plt.loglog(q_all, I_all, lw = 0, marker = 'o',  ms = 10, color = 'black', alpha = 0.2, label = 'merged')
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

                    header_text = 'q (A-1), I (1/cm)'
                    file_name = path_merged_txt + 'merged_'  + keys + '.dat'
                    data_save = np.column_stack((q_all, I_all))
                    np.savetxt(file_name, data_save, delimiter=',', header=header_text)

                    file_name = path_merged_txt + 'interpolated_'  + keys + '.dat'
                    data_save = np.column_stack((interpolation_pts, linear_results))
                    np.savetxt(file_name, data_save, delimiter=',', header=header_text)

# %%
def subtract_incoherent(path_dir_an, var_offset, fitting_range):
    path_merged = path_dir_an + '/merged'
    path_merged_txt = path_merged +  '/data_txt/'

    file_name = os.path.join(path_dir_an, 'config.npy')
    with open(file_name, 'rb') as handle:
        config = pickle.load(handle)

    calibration = config['experiment']['calibration']

    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path_merged_txt):
        for file in f:
            if 'interpolated' in file:
                files.append(os.path.join(file))

    for ii in range(0, len(files)):
        if files[ii][13:-4] not in calibration.values():
            plt.close('all')
            plt.ioff()
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

            diff_I = np.diff(np.subtract(I, np.min(I)))

            if var_offset > 0:
                var = []
                range_bins = range(0, len(diff_I), 10)
                for jj in range(len(range_bins)-1):
                    init = range_bins[jj]
                    end = range_bins[jj+1]
                    var.append(np.var(np.abs(diff_I[init:end])))
                idx = np.where(np.array(var) > var_offset)
                off_set = range_bins[idx[0][-2]]
            else:
                off_set = len(I)-fitting_range

            # slip the first points
            fitting_I = I[off_set:-3]
            fitting_q = q[off_set:-3]

            def porod(q, coef, slope, incoherent):
                return (coef * q**(slope-4) + incoherent)

            base = np.polyfit(fitting_q, fitting_I, 0)
            base = np.float64(base)
            # perform the fit
            params, cv = scipy.optimize.curve_fit(porod, fitting_q, fitting_I, p0 = [0.01, -5, 0])#,
                                                  #bounds = ((-np.inf, -5, 1e-5),(np.inf, 5, base )))
            m, t, b = params
            incoherent = b * 0.95
            slope = t
            coeff = m

            # determine quality of the fit
            squaredDiffs = np.square(fitting_I - porod(fitting_q, m, t, b))
            squaredDiffsFromMean = np.square(fitting_I - np.mean(fitting_I))
            rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
            rSquared = round(rSquared, 2)


            # plot the results
            plt.loglog(q, I, '.', label="original data", color = 'blue', alpha = 0.2)
            plt.loglog(fitting_q, porod(fitting_q, m, t, b), '--', color = 'blue', label= f"fit - R² = {rSquared}")
            plt.title("Fitted Porod Approximation")
            plt.loglog(q, np.ones(len(I))*incoherent, '--', color = 'red', label = 'incoherent = ' + str(round(incoherent, 2)))
            plt.loglog(q, coeff * q**(slope-4), '--', color = 'black', label = 'slope = ' + str(round(slope-4, 2)))
            plt.loglog(q, I-incoherent, '.', color = 'black', label = 'subtracted data')
            plt.legend()

            path_merged_fig = path_merged +  '/figures/'
            file_name = path_merged_fig + files[ii][13:-4] + '_subtracted' + '.jpeg'
            plt.savefig(file_name)

            header_text = 'q (A-1), I (1/cm)'
            file_name = path_merged_txt + 'subtracted_'  + files[ii][13:-4] + '.dat'
            data_save = np.column_stack((q, I-incoherent))
            np.savetxt(file_name, data_save, delimiter=',', header=header_text)
