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
from utils import smooth
from scipy.stats import linregress # To check for linearity/constancy

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
                    # smooth data
                    #I = smooth(I, 3)
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

# %% plot the sectors data
def plot_all_data_sectors(path_dir_an):
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
    sectors_nr = result['integration']['sectors_nr']
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
                    prefix = 'azim_integ'
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
                                             usecols = range(1,sectors_nr+1))
                    # smooth data
                    #I = smooth(I, 3)
                    #e = np.genfromtxt(file_name, dtype = None, delimiter = ',', usecols = 2)
                    if sample_name in merged_files:
                        temp = merged_files[sample_name]['I']
                        merged_files[sample_name]['I'] =  np.vstack((temp, I))
                        temp1 = merged_files[sample_name]['q']
                        merged_files[sample_name]['q'] =  np.vstack((temp1, q))
#                        temp2 = merged_files[sample_name]['error']
                        #merged_files[sample_name]['error'] =  np.vstack((temp2, e))
                    else:
                        merged_files[sample_name] = {}
                        merged_files[sample_name]['I'] = I
                        merged_files[sample_name]['q'] = q
                        #merged_files[sample_name]['error'] = e

    # plot the files
    for keys in merged_files:
        plt.close('all')
        plt.ioff()
        if merged_files[keys]['q'].ndim > 1:
            dd = merged_files[keys]['q'].shape[0]
            for ii in range(dd):
                for jj in range(1, sectors_nr):
                    q = merged_files[keys]['q'][:, ii]
                    I_s = merged_files[keys]['I'][:,jj]
                    #                e = merged_files[keys]['error'][ii,:]
                    plt.loglog(q, I_s, lw = 0.3, marker = 'o',  ms = 2)

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
# %% merging_data
def merging_data(path_dir_an, merged_files, skip_start, skip_end, interp_type, interp_points, smooth_window):
    #  merge the files
    # low to high q
    path_merged = os.path.join(path_dir_an, 'merged/')
    path_merged_fig = os.path.join(path_merged, 'figures/')
    if not os.path.exists(path_merged_fig):
        os.mkdir(path_merged_fig)
    path_merged_txt = os.path.join(path_merged, 'data_txt/')

    for keys in merged_files:
        # initiate variables to join the values
        I_all = []
        q_all = []
        e_all = []
        plt.close('all')
        plt.ioff()


        if merged_files[keys]['q'].ndim > 1:
            # --- IMPROVED SELECTION OF ORDER OF FILES START ---
            # Define the order for the detector distances by sorting their first q-value.
            # This ensures merging proceeds from lowest q-range to highest q-range.
            first_q_values = [merged_files[keys]['q'][kk][0] for kk in range(merged_files[keys]['q'].shape[0])]
            idx_det = np.argsort(first_q_values)
            # --- IMPROVED SELECTION OF ORDER OF FILES END ---

            for ii in idx_det: # Loop through the detector indices in the sorted order
                # get the values skipping the start and end points defined in the function
                q = merged_files[keys]['q'][ii, :]

                # Use .get() with a default value to safely retrieve skip points,
                # in case a specific detector index (as a string) is not in the skip_start/end dicts.
                skip_start_val = skip_start.get(str(ii), 0)
                skip_end_val = skip_end.get(str(ii), 0)

                q = q[skip_start_val:len(q)-skip_end_val]
                I = merged_files[keys]['I'][ii,:]
                I = I[skip_start_val:len(I)-skip_end_val]
                e = merged_files[keys]['error'][ii,:]
                e = e[skip_start_val:len(e)-skip_end_val]

                # Concatenate only if the trimmed array is not empty
                if q.size > 0:
                    q_all = np.concatenate((q_all, q), axis = None)
                    I_all = np.concatenate((I_all, I), axis = None)
                    e_all = np.concatenate((e_all, e), axis = None)
                else:
                    # Optionally, add a print statement here if you want to know when a segment is skipped
                    # print(f"Warning: Detector {ii} data is empty after skipping points. Skipping concatenation.")
                    pass


                # checking if there is the need to shift slighly the plot
                # Note: This original scaling logic has potential issues if q_all is empty
                # at ii=0 or if overlap regions are not well-defined or contain NaNs/zeros.
                # It is left unchanged as per your request.
                if ii == 0:
                    # define where the q0 fits the qmax
                    start_pt = np.where(np.round(q_all, 2) == np.round(q[0], 2))
                    # Ensure start_pt is not empty
                    if start_pt[0].size > 0:
                        start_pt = start_pt[0][-1]
                    else:
                        start_pt = 0 # Default to 0 if no match found

                    end_pt = np.where(np.round(q, 2) == np.round(q_all[-1], 2))
                    # Ensure end_pt is not empty
                    if end_pt[0].size > 0:
                        end_pt = end_pt[0][0]
                    else:
                        end_pt = len(I) # Default to end of current I if no match found

                    # if there is the need for an adjustmet in the patterns
                    # Ensure slices are valid before taking median
                    median_q_all = np.median(I_all[start_pt:]) if len(I_all[start_pt:]) > 0 else np.nan
                    median_I_current = np.median(I[:end_pt]) if len(I[:end_pt]) > 0 else np.nan

                    if median_I_current != 0 and not np.isnan(median_q_all) and not np.isnan(median_I_current):
                        scaling = median_q_all / median_I_current
                    else:
                        scaling = 1 # Default if division by zero, or NaN present

                    if np.isnan(scaling):
                         scaling = 1
                    I = np.multiply(I, scaling)

                    # save after the multiplication
                    # This block implies `q_all`, `I_all`, `e_all` were already modified
                    # by concatenation before this scaling. This structure might lead
                    # to `q_all` having duplicated points if the scaled `I` is concatenated again.
                    # As per instruction, this original logic is retained.
                    q_all = np.concatenate((q_all, q), axis = None)
                    I_all = np.concatenate((I_all, I), axis = None)
                    e_all = np.concatenate((e_all, e), axis = None)
                else:
                    # This 'else' branch is only reached if ii != 0 AND the data was already concatenated
                    # in the first part of the loop. If the above `if ii == 0` block re-concatenates,
                    # this 'else' is problematic. Based on strict "original code" retention, it stays.
                    q_all = np.concatenate((q_all, q), axis = None)
                    I_all = np.concatenate((I_all, I), axis = None)
                    e_all = np.concatenate((e_all, e), axis = None)
                # for indexing the points

        # The handling for ndim = 1 data should be outside the if merged_files[keys]['q'].ndim > 1: block
        # to ensure it's processed if it's a single detector file.
        # This part of the original code structure is kept, assuming it correctly handles single-dim data
        # not being part of the 'idx_det' loop.
        if merged_files[keys]['q'].ndim == 1:
            q_all = merged_files[keys]['q'] # Directly assign if it's single dim
            I_all = merged_files[keys]['I']
            e_all = merged_files[keys]['error']
            scaling = 1 # No scaling for single detector data


        if merged_files[keys]['q'].ndim > 1: # This condition might need adjustment if it should apply to ndim=1 too
            idx = np.argsort(q_all)
            q_all = q_all[idx]
            I_all = I_all[idx]
            e_all = e_all[idx]
            # plot original data
            plt.errorbar(q_all, I_all, e_all, lw = 1, marker = 'o',  ms = 10, color = 'black', alpha = 0.05, label = 'merged, scale = ' + str(np.round(scaling, 4)))
            if interp_type == 'log':
                # Interpolate it to new time points
                min_pt = np.round(np.log10(np.min(q_all))/1.01, 3)
                max_pt = np.round(np.log10(np.max(q_all))*1.01, 3)
                interpolation_pts = np.logspace(min_pt, max_pt, interp_points)
                linear_interp = interp1d(q_all, I_all)
                linear_results = linear_interp(interpolation_pts)
                interpolation_pts = np.append(q_all[:2], interpolation_pts)
                linear_results = np.append(I_all[:2], linear_results)
                # smooth
                # The 'smooth' function needs to be defined elsewhere in your code.
                linear_results = smooth(linear_results, smooth_window)
                plt.loglog(interpolation_pts, linear_results, lw = 0.3,
                           marker = 'o',  ms = 4, color = 'red', label = 'interpolated')
            if interp_type == 'linear':
                # Interpolate it to new time points
                interpolation_pts = np.linspace(np.min(q_all), np.max(q_all), interp_points)
                linear_interp = interp1d(q_all, I_all)
                linear_results = linear_interp(interpolation_pts)
                interpolation_pts = np.append(q_all[:2], interpolation_pts)
                linear_results = np.append(I_all[:2], linear_results)
                # smooth
                # The 'smooth' function needs to be defined elsewhere in your code.
                linear_results = smooth(linear_results, smooth_window)
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
            file_name = path_merged_txt +  keys +'_merged' + '.dat'
            data_save = np.column_stack((q_all, I_all, e_all))
            np.savetxt(file_name, data_save, delimiter=',', header=header_text)

            if interp_type == 'linear' or interp_type == 'log':
                file_name = path_merged_txt + keys + '_interp'  '.dat'
                data_save = np.column_stack((interpolation_pts, linear_results))
                np.savetxt(file_name, data_save, delimiter=',', header=header_text)

# %% subtract incoherent
def subtract_incoherent(path_dir_an, initial_last_points_fit=50, constancy_threshold=0.05):
    """
    Subtracts incoherent scattering from SAXS/WAXS data using Porod approximation.

    This version dynamically determines if the tail of the scattering curve is constant
    and adapts the fitting range accordingly.

    Args:
        path_dir_an (str): Path to the analysis directory containing 'merged' and 'config.npy'.
        initial_last_points_fit (int): Initial number of points from the end to consider
                                       for checking constancy and as a starting point for fitting.
        constancy_threshold (float): A threshold (e.g., as a percentage of the mean intensity)
                                     to determine if the tail is "constant".
                                     Smaller values mean stricter constancy.
    """
    path_merged = os.path.join(path_dir_an, 'merged')
    path_merged_txt = os.path.join(path_merged, 'data_txt')
    path_merged_fig = os.path.join(path_merged, 'figures')

    # Create directories if they don't exist
    os.makedirs(path_merged_txt, exist_ok=True)
    os.makedirs(path_merged_fig, exist_ok=True)

    file_name_config = os.path.join(path_dir_an, 'config.npy')
    try:
        with open(file_name_config, 'rb') as handle:
            config = pickle.load(handle)
    except FileNotFoundError:
        print(f"Error: config.npy not found at {file_name_config}. Please ensure the file exists.")
        return
    except Exception as e:
        print(f"Error loading config.npy: {e}")
        return

    calibration = config.get('experiment', {}).get('calibration', {})
    if not calibration:
        print("Warning: 'calibration' not found in config.npy or is empty. This might affect file processing.")

    files_to_process = []
    # os.walk iterates through directories. We need files directly in path_merged_txt
    for file in os.listdir(path_merged_txt):
        if file.endswith('.dat') and ('interpolated' in file or 'merged' in file):
            files_to_process.append(file)

    if not files_to_process:
        print(f"No '.dat' files (interpolated or merged) found in {path_merged_txt}. Exiting.")
        return

    def porod(q_val, coef, slope, incoherent_val):
        """Porod scattering model with incoherent background."""
        return (coef * q_val**(slope - 4) + incoherent_val)

    for file_short_name in files_to_process:
        plt.close('all') # Close previous plots to prevent them from piling up in memory

        # More robust way to get the base file name
        base_name = file_short_name.replace('merged_', '').replace('.dat', '')

        if base_name in calibration.values():
            print(f"Skipping {file_short_name} as it's a calibration file.")
            continue

        file_path_data = os.path.join(path_merged_txt, file_short_name)
        print(f"\nProcessing: {file_path_data}")

        try:
            I = np.genfromtxt(file_path_data, dtype=float, delimiter=',', usecols=1)
            q = np.genfromtxt(file_path_data, dtype=float, delimiter=',', usecols=0)
        except Exception as e:
            print(f"Error reading data from {file_path_data}: {e}. Skipping this file.")
            continue

        # Basic data validation
        if len(I) == 0 or len(q) == 0 or len(I) != len(q):
            print(f"Warning: Data arrays are empty or mismatched for {file_short_name}. Skipping.")
            continue
        if np.any(np.isnan(I)) or np.any(np.isnan(q)) or np.any(np.isinf(I)) or np.any(np.isinf(q)):
            print(f"Warning: Data contains NaN or Inf values for {file_short_name}. Skipping.")
            continue
        # Filter out zero or negative intensities for log plot and Porod fit
        positive_mask = I > 0
        if not np.any(positive_mask):
            print(f"Warning: All intensities are zero or negative for {file_short_name}. Cannot perform log plot/fit. Skipping.")
            continue
        q_pos = q[positive_mask]
        I_pos = I[positive_mask]

        if len(I_pos) < initial_last_points_fit + 5: # Ensure enough points for initial analysis + fit
            print(f"Warning: Not enough data points ({len(I_pos)}) for initial analysis and fitting in {file_short_name}. Skipping.")
            continue

        # --- Dynamic Constancy Check and Adaptable Fitting Range ---
        determined_fitting_range = None

        # Analyze the tail for constancy
        # Start by looking at a larger segment (e.g., initial_last_points_fit * 2 or up to a fixed large number)
        # but ensure we don't go beyond the start of the data.
        tail_start_idx = max(0, len(I_pos) - initial_last_points_fit * 2) # Look at twice the initial guess or less if not available
        q_tail_analysis = q_pos[tail_start_idx:]
        I_tail_analysis = I_pos[tail_start_idx:]

        if len(I_tail_analysis) < 5: # Need at least a few points to assess constancy
            print(f"Warning: Tail analysis range too small for {file_short_name}. Cannot assess constancy. Skipping.")
            continue

        # Option 1: Check standard deviation relative to mean for constancy
        # A smaller value for `std_threshold` means stricter constancy.
        std_threshold = constancy_threshold # e.g., 0.05 means std must be <= 5% of mean
        if np.mean(I_tail_analysis) > 0: # Avoid division by zero
            relative_std = np.std(I_tail_analysis) / np.mean(I_tail_analysis)
        else:
            relative_std = np.inf # If mean is zero, it's not constant/positive

        # Option 2: Check slope of a linear fit to the tail (on linear scale for I)
        slope, intercept, r_value, p_value, std_err = linregress(q_tail_analysis, I_tail_analysis)

        # Determine if constant based on relative_std and slope magnitude
        # We also want the R-squared of the linear fit to be low for "constant"
        is_constant_tail = (relative_std < std_threshold) and (abs(slope) < (constancy_threshold * np.mean(I_tail_analysis) / (q_tail_analysis[-1] - q_tail_analysis[0] + 1e-9)))

        if not is_constant_tail:
            print(f"Warning: Tail of data for {file_short_name} is not sufficiently constant (relative std: {relative_std:.3f}, slope: {slope:.3e}). Skipping Porod fit.")
            continue

        # If constant, determine the fitting range.
        # We can iterate backwards from the end to find the start of the "constant" region
        # or simply use a fixed number of points once constancy is confirmed.
        # For simplicity, let's use a dynamic range that is at least initial_last_points_fit,
        # but extends further into the constant region if possible.

        # A simple strategy: use `initial_last_points_fit` points if tail is constant.
        # More advanced: iteratively find the largest range from the end that is constant.

        # Let's use `initial_last_points_fit` points, ensuring it doesn't exceed the constant tail,
        # but if the constant tail is shorter, we cap it there.
        determined_fitting_range = min(initial_last_points_fit, len(I_tail_analysis))

        # Ensure we have enough points for a robust fit
        min_points_for_fit = 10 # You might need more depending on your data and model complexity
        if determined_fitting_range < min_points_for_fit:
            print(f"Warning: Determined fitting range ({determined_fitting_range} points) is too small for a robust fit for {file_short_name}. Skipping.")
            continue

        off_set = len(I_pos) - determined_fitting_range
        fitting_I = I_pos[off_set:]
        fitting_q = q_pos[off_set:]

        print(f"Using {len(fitting_I)} points for fitting (q_range: {fitting_q[0]:.3e} to {fitting_q[-1]:.3e}).")

        # --- Improved Initial Guesses for curve_fit ---
        # 1. Incoherent background (b): Estimate from the average of the last few points of fitting_I
        initial_incoherent = np.mean(fitting_I[-min(5, len(fitting_I)):]) if len(fitting_I) > 0 else 1e-3
        if initial_incoherent <= 0: # Ensure positive guess for log-scale
            initial_incoherent = 1e-3 # Small positive value

        # 2. Coefficient (m): Estimate by assuming incoherent is removed and using Porod approx at first point
        # I = coef * q**(slope-4) + incoherent => coef ~ (I - incoherent) / q**(slope-4)
        # Assuming slope ~ 0 (for -4 power)
        initial_coef = (fitting_I[0] - initial_incoherent) * fitting_q[0]**4 if fitting_q[0] > 0 else 1.0
        if initial_coef < 0: initial_coef = 1.0 # Ensure positive

        # 3. Slope (t): For Porod, the power is -4, so t should be near 0.
        initial_slope = 0.0 # This is (exponent + 4)

        p0 = [initial_coef, initial_slope, initial_incoherent]

        # Bounds for parameters:
        # coef: must be positive
        # slope: can vary around 0 (e.g., -10 to 1 for (slope-4))
        # incoherent: must be non-negative, but give it some upper flexibility.
        # It shouldn't be higher than the maximum intensity in the fitting range.
        lower_bounds = [0, -10, 0]
        upper_bounds = [np.inf, 1, np.max(fitting_I) * 2] # Allow incoherent to be up to twice max fitted I

        try:
            params, cv = scipy.optimize.curve_fit(
                porod, fitting_q, fitting_I,
                p0=p0,
                bounds=(lower_bounds, upper_bounds),
                max_nfev=2000 # Increase max function evaluations if it frequently fails
            )
            m, t, b = params
            coeff = m
            slope = t
            incoherent = b

            # Determine quality of the fit
            squaredDiffs = np.square(fitting_I - porod(fitting_q, m, t, b))
            squaredDiffsFromMean = np.square(fitting_I - np.mean(fitting_I))

            if np.sum(squaredDiffsFromMean) == 0:
                rSquared = 1.0
            else:
                rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
            rSquared = round(rSquared, 2)

            # Plot the results
            plt.figure(figsize=(10, 7))
            plt.loglog(q_pos, I_pos, '.', label="Original data", color='blue', alpha=0.3)
            plt.loglog(fitting_q, porod(fitting_q, m, t, b), '--', color='blue', linewidth=3, label=f"Fit - R² = {rSquared}")
            plt.title(f"Fitted Porod Approximation for {base_name}")
            plt.loglog(q_pos, np.ones(len(q_pos))*incoherent, '--', color='red', label=f"Incoherent = {incoherent:.4f}")

            # Ensure I_pos - incoherent is positive for loglog plot
            subtracted_I = I_pos - incoherent
            valid_subtracted_mask = subtracted_I > 0
            plt.loglog(q_pos[valid_subtracted_mask], subtracted_I[valid_subtracted_mask], '.', color='black', label='Subtracted data', markersize=3)

            plt.xlabel("q (A$^{-1}$)") # Use LaTeX for Angstrom symbol
            plt.ylabel("I (1/cm)")
            plt.legend()
            plt.grid(True, which="both", ls="-", alpha=0.2)

            file_name_fig = os.path.join(path_merged_fig, f"{base_name}_subtracted.jpeg")
            plt.savefig(file_name_fig, bbox_inches='tight', dpi=150)

            # Save the processed data
            header_text = 'q (A-1), I (1/cm)'
            file_name_data_out = os.path.join(path_merged_txt, f"{base_name}_subtracted.dat")

            # Save all original q, and the subtracted I (handling negative if needed, though logplot handles by mask)
            # For saving, it's often better to keep all data, even if it goes negative after subtraction,
            # unless there's a specific reason to clip or filter.
            np.savetxt(file_name_data_out, np.column_stack((q, I - incoherent)),
                       delimiter=',', header=header_text, comments='# ')

        except (RuntimeError, ValueError) as e:
            print(f"Error during curve fitting or plotting for {file_short_name}: {e}. Skipping this file.")
            # Save original data if fit failed to ensure data is not lost or corrupted
            header_text = 'q (A-1), I (1/cm) - Fit failed, data not subtracted'
            file_name_data_out = os.path.join(path_merged_txt, f"{base_name}_original_fit_failed.dat")
            np.savetxt(file_name_data_out, np.column_stack((q, I)),
                       delimiter=',', header=header_text, comments='# ')
            continue
        except Exception as e:
            print(f"An unexpected error occurred for {file_short_name}: {e}. Skipping this file.")
            continue

    print("\nIncoherent subtraction process completed.")
