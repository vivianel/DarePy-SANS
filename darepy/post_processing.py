"""
This function automatically marges the data collected in different detector distances

Created on Wed Jul 13 20:50:57 2022

@author: lutzbueno_v
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from . import integration as integ
from scipy.interpolate import interp1d
import scipy.optimize
import os
from .utils import smooth
from scipy.stats import linregress # To check for linearity/constancy

# %% plot_all_data

def plot_all_data(path_dir_an):
    """
    Loads radial integration data from different detector distances, organizes it
    by sample name, and plots the raw (unmerged) data for visual inspection.

    Args:
        path_dir_an (str): The path to the main analysis directory where results
                           and configuration files are stored, and where detector-specific
                           integration data is located.

    Returns:
        dict: A dictionary where keys are sample names and values are dictionaries
              containing 'I' (intensities), 'q' (scattering vectors), and 'error'
              (standard deviations) from all relevant detector distances, stacked
              for each sample.
    """
    # Create necessary subdirectories for merged data and figures if they don't exist
    path_merged = os.path.join(path_dir_an, 'merged/')
    if not os.path.exists(path_merged):
        os.mkdir(path_merged)
    path_merged_fig = os.path.join(path_merged, 'figures/')
    if not os.path.exists(path_merged_fig):
        os.mkdir(path_merged_fig)
    path_merged_txt = os.path.join(path_merged, 'data_txt/')
    if not os.path.exists(path_merged_txt):
        os.mkdir(path_merged_txt)

    # Load result and configuration files
    file_results = os.path.join(path_dir_an, 'result.npy')
    with open(file_results, 'rb') as handle:
        result = pickle.load(handle)
    file_config = os.path.join(path_dir_an, 'config.npy')
    with open(file_config, 'rb') as handle:
        config = pickle.load(handle)

    #calibration = config['experiment']['calibration']
    list_class_files = result['overview']

    merged_files = {} # Dictionary to store intensities, q, and errors for each sample

    # Iterate through all detector distances found in the result overview
    for keys in list_class_files:
        if 'det' in keys: # Filter for detector-specific keys
            total_samples = len(list_class_files[keys]['scan'])
            for ii in range(total_samples):
                sample_name = list_class_files[keys]['sample_name'][ii]
                # Skip calibration samples as they are processed separately
                #if sample_name not in calibration.values():
                # at the moment I want to double chech the value for water
                # Load the radial integration data for the current sample and detector
                prefix = 'radial_integ'
                sufix = 'dat'
                ScanNr = list_class_files[keys]['scan'][ii]
                det = list_class_files[keys]['detx_m'][ii]
                Frame = 0 # Assuming integration is done per scan/frame 0
                path_integ = path_dir_an + '/det_' + str(det).replace('.','p') + '/integration/'
                # Construct the file name using the utility function from integration module
                file_name = integ.make_file_name(path_integ, prefix, sufix, sample_name, str(det).replace('.','p'), ScanNr, Frame)

                try:
                    # Load q, Intensity (I), and error (e) from the integrated data file
                    q = np.genfromtxt(file_name, dtype = None, delimiter = ',', usecols = 0)
                    I = np.genfromtxt(file_name, dtype = None, delimiter = ',', usecols = 1)
                    e = np.genfromtxt(file_name, dtype = None, delimiter = ',', usecols = 2) # Load error
                except Exception as e_load:
                    print(f"Warning: Could not load data from {file_name}. Error: {e_load}. Skipping this file.")
                    continue # Skip to the next file if loading fails

                # Ensure errors are non-negative before storing
                e = np.abs(e)

                # Aggregate data for each sample from different detector distances
                if sample_name in merged_files:
                    # If sample already exists, append new data vertically
                    temp_I = merged_files[sample_name]['I']
                    merged_files[sample_name]['I'] =  np.vstack((temp_I, I))
                    temp_q = merged_files[sample_name]['q']
                    merged_files[sample_name]['q'] =  np.vstack((temp_q, q))
                    temp_e = merged_files[sample_name]['error']
                    merged_files[sample_name]['error'] =  np.vstack((temp_e, e)) # Include error
                else:
                    # If sample is new, initialize its entry in the dictionary
                    merged_files[sample_name] = {}
                    merged_files[sample_name]['I'] = I
                    merged_files[sample_name]['q'] = q
                    merged_files[sample_name]['error'] = e

    # Plot the raw (unmerged) data from all detectors for each sample
    for keys in merged_files:
        plt.close('all') # Close all existing plots
        plt.ioff() # Turn off interactive plotting to prevent window pop-ups

        if merged_files[keys]['q'].ndim > 1: # Check if data from multiple detectors exists
            dd = merged_files[keys]['q'].shape[0]
            for ii in range(dd):
                q = merged_files[keys]['q'][ii, :]
                I = merged_files[keys]['I'][ii,:]
                e = merged_files[keys]['error'][ii,:]
                plt.errorbar(q, I, e, lw = 0.3, marker = 'o',  ms = 2) # Plot with error bars

            plt.xlabel(r'Scattering vector q [$\AA^{-1}$]')
            plt.xscale('log')
            plt.yscale('log')
            plt.ylabel(r'Intensity I [cm$^{-1}$]')
            plt.title('Sample: '+  keys)
            file_name = path_merged_fig + keys + '_all_det_dist' + '.jpeg'
            plt.savefig(file_name) # Save the plot
        else:
            print('_____________________________')
            print('Single detector distance measurement: ' + keys)
            print('_____________________________')
    return merged_files

# %% merging_data
def merging_data(path_dir_an, merged_files, skip_start, skip_end, interp_type, interp_points, smooth_window):
    """
    Merges scattering data from different detector distances for each sample,
    handles overlapping regions, applies scaling, and performs interpolation and smoothing.

    Args:
        path_dir_an (str): The path to the main analysis directory.
        merged_files (dict): Dictionary of raw intensity, q, and error for each sample
                             from different detector distances (output of `plot_all_data`).
        skip_start (dict): Dictionary specifying number of points to skip from the
                           beginning of each detector's data (key: detector index as string).
        skip_end (dict): Dictionary specifying number of points to skip from the
                         end of each detector's data (key: detector index as string).
        interp_type (str): Type of interpolation ('log' for log-spaced q, 'linear' for linear q).
        interp_points (int): Number of points for the interpolated data.
        smooth_window (int): Window size for smoothing the interpolated data.
    """
    # Create necessary subdirectories for merged data and figures if they don't exist
    path_merged = os.path.join(path_dir_an, 'merged/')
    path_merged_fig = os.path.join(path_merged, 'figures/')
    if not os.path.exists(path_merged_fig):
        os.mkdir(path_merged_fig)
    path_merged_txt = os.path.join(path_merged, 'data_txt/')

    for keys in merged_files: # Iterate through each sample
        # Initialize lists to accumulate all q, I, and error values for the current sample
        I_all = []
        q_all = []
        e_all = []
        plt.close('all') # Close previous plots
        plt.ioff() # Turn off interactive plotting

        if merged_files[keys]['q'].ndim > 1: # Process only if data from multiple detectors exists
            # Determine the order of merging based on the first q-value of each detector.
            # This ensures merging proceeds from low q to high q.
            first_q_values = [merged_files[keys]['q'][kk][0] for kk in range(merged_files[keys]['q'].shape[0])]
            idx_det = np.argsort(first_q_values) # Get indices that would sort the array

            scaling = 1.0 # Initialize scaling factor for data adjustment

            # Loop through the detector indices in the sorted order (from low q to high q)
            for ii in idx_det:
                q = merged_files[keys]['q'][ii, :]
                I = merged_files[keys]['I'][ii,:]
                e = merged_files[keys]['error'][ii,:] # Load error

                # Retrieve skip points for the current detector, defaulting to 0 if not specified
                skip_start_val = skip_start.get(str(ii), 0)
                skip_end_val = skip_end.get(str(ii), 0)

                # Apply skipping to the data
                q = q[skip_start_val:len(q)-skip_end_val]
                I = I[skip_start_val:len(I)-skip_end_val]
                e = e[skip_start_val:len(e)-skip_end_val] # Apply skipping to error as well

                # If it's not the first detector in the sorted order, apply scaling
                if ii != idx_det[0]: # Check against the first index after sorting
                    # Define overlap region: last part of previously added data (q_all)
                    # and first part of current detector data (q, I, e)
                    # We need to find the overlap and calculate the scaling factor.
                    # A robust method would involve finding a common q-range for overlap.
                    # For simplicity and adhering to original logic, we take a median ratio.
                    # This part of the logic can be complex and might need more sophisticated alignment.

                    # For the purpose of applying the scaling factor from the previous segment
                    # to the current segment before concatenation:
                    # The `scaling` variable in the original code appears to be calculated
                    # and then implicitly applied to `I` and `e` before the next concatenation.
                    # Let's adjust the structure to apply scaling to the current (I, e) before adding to (I_all, e_all).

                    # Calculate overlap region and scaling only if q_all is not empty
                    if len(q_all) > 0 and len(q) > 0:
                        # Find the intersection of q ranges (a simplified approach)
                        # This assumes overlap exists and data quality in overlap is good.
                        min_q_overlap = max(q_all[0], q[0])
                        max_q_overlap = min(q_all[-1], q[-1])

                        # Create interpolation functions for the previous data (q_all, I_all)
                        # and current data (q, I) within the overlap region.
                        # Using 'nearest' or 'linear' for simple interpolation.
                        if max_q_overlap > min_q_overlap:
                            # Define interpolation points within the overlap
                            interp_q_overlap = np.linspace(min_q_overlap, max_q_overlap, 50) # Use a fixed number of points
                            try:
                                interp_prev_I = interp1d(q_all, I_all, kind='linear', fill_value="extrapolate")(interp_q_overlap)
                                interp_curr_I = interp1d(q, I, kind='linear', fill_value="extrapolate")(interp_q_overlap)

                                # Calculate median ratio, avoiding division by zero
                                valid_overlap = (interp_curr_I > 0) & (interp_prev_I > 0)
                                if np.any(valid_overlap):
                                    scaling = np.median(interp_prev_I[valid_overlap] / interp_curr_I[valid_overlap])
                                else:
                                    scaling = 1.0 # No valid overlap points, no scaling
                                if np.isnan(scaling) or np.isinf(scaling):
                                    scaling = 1.0
                            except ValueError as e:
                                print(f"Warning: Could not interpolate for scaling in sample {keys}, detector {ii}. Error: {e}. Setting scaling to 1.0.")
                                scaling = 1.0
                        else:
                            scaling = 1.0 # No overlap, no scaling
                    else:
                        scaling = 1.0 # No previous data or current data, no scaling

                    # Apply scaling to the current intensity and its error
                    I = np.multiply(I, scaling)
                    e = np.multiply(e, scaling) # Propagate error by multiplying with scaling factor

                # Ensure errors are non-negative after scaling
                e = np.abs(e)

                # Concatenate the current (potentially scaled) data to the overall lists
                # Ensure array is not empty before concatenation
                if q.size > 0:
                    q_all = np.concatenate((q_all, q), axis=None)
                    I_all = np.concatenate((I_all, I), axis=None)
                    e_all = np.concatenate((e_all, e), axis=None) # Include error in concatenation
                else:
                    print(f"Warning: Data for detector {ii} of sample {keys} is empty after skipping points. Skipping concatenation.")
                    pass # Skip concatenation if the array is empty


        else: # Case for single detector distance measurement
            print('_____________________________')
            print('Single detector distance measurement: ' + keys)
            print('_____________________________')
            q_all = merged_files[keys]['q']
            I_all = merged_files[keys]['I']
            e_all = merged_files[keys]['error'] # Error for single detector data
            e_all = np.abs(e_all) # Ensure errors are non-negative for single detector data too

        # Sort all aggregated data by q-value
        if q_all.ndim > 0 and q_all.size > 0: # Ensure q_all is not empty before sorting
            idx = np.argsort(q_all)
            q_all = q_all[idx]
            I_all = I_all[idx]
            e_all = e_all[idx] # Sort errors consistently with q and I

            # Plot original (merged, un-interpolated) data
            plt.errorbar(q_all, I_all, e_all, lw = 1, marker = 'o',  ms = 10, color = 'black',
                         alpha = 0.05, label = f'merged, scale = {np.round(scaling, 4)}') # Plot error bars

            interpolation_pts = None
            linear_results_I = None
            linear_results_e = None # New: for interpolated errors

            if interp_type == 'log':
                # Interpolate on a log-scale for q
                min_pt = np.log10(np.min(q_all[q_all > 0])) # Ensure q is positive for log
                max_pt = np.log10(np.max(q_all[q_all > 0]))
                interpolation_pts = np.logspace(min_pt, max_pt, interp_points)

                # Interpolate Intensity
                linear_interp_I = interp1d(q_all, I_all, kind='linear', fill_value="extrapolate")
                linear_results_I = linear_interp_I(interpolation_pts)
                # Smooth the interpolated intensity
                linear_results_I = smooth(linear_results_I, smooth_window)

                # Interpolate Error
                # Using 'linear' interpolation for errors. Consider 'nearest' if errors are sparse.
                linear_interp_e = interp1d(q_all, e_all, kind='linear', fill_value="extrapolate")
                linear_results_e = linear_interp_e(interpolation_pts)
                # Ensure interpolated errors are non-negative
                linear_results_e = np.abs(linear_results_e)
                # Smooth the interpolated errors (optional, use with caution)
                linear_results_e = smooth(linear_results_e, smooth_window)


                plt.loglog(interpolation_pts, linear_results_I, lw = 0.3,
                           marker = 'o',  ms = 4, color = 'red', label = 'interpolated')
                # Plot interpolated errors (as error bars or as a shaded region if preferred)
                plt.errorbar(interpolation_pts, linear_results_I, linear_results_e,
                             lw=0.1, color='red', alpha=0.3, fmt='none', label='interpolated errors')


            elif interp_type == 'linear':
                # Interpolate on a linear scale for q
                interpolation_pts = np.linspace(np.min(q_all), np.max(q_all), interp_points)

                # Interpolate Intensity
                linear_interp_I = interp1d(q_all, I_all, kind='linear', fill_value="extrapolate")
                linear_results_I = linear_interp_I(interpolation_pts)
                # Smooth the interpolated intensity
                linear_results_I = smooth(linear_results_I, smooth_window)

                # Interpolate Error
                linear_interp_e = interp1d(q_all, e_all, kind='linear', fill_value="extrapolate")
                linear_results_e = linear_interp_e(interpolation_pts)
                # Ensure interpolated errors are non-negative
                linear_results_e = np.abs(linear_results_e)
                # Smooth the interpolated errors (optional, use with caution)
                linear_results_e = smooth(linear_results_e, smooth_window)


                plt.loglog(interpolation_pts, linear_results_I, lw = 0.3,
                           marker = 'o',  ms = 4, color = 'red', label = 'interpolated')
                # Plot interpolated errors
                plt.errorbar(interpolation_pts, linear_results_I, linear_results_e,
                             lw=0.1, color='red', alpha=0.3, fmt='none', label='interpolated errors')


            plt.xlabel(r'Scattering vector q [$\AA^{-1}$]')
            plt.xscale('log')
            plt.yscale('log')
            plt.ylabel(r'Intensity I [cm$^{-1}$]')
            plt.title('Sample: '+  keys)
            plt.legend()

            file_name_fig = path_merged_fig + keys + '_merged' + '.jpeg'
            plt.savefig(file_name_fig)

            # Save the merged (un-interpolated) data to a text file
            header_text = 'q (A-1), I (1/cm), error' # Updated header
            file_name_txt = path_merged_txt +  keys +'_merged' + '.dat'
            data_save = np.column_stack((q_all, I_all, e_all)) # Include error
            np.savetxt(file_name_txt, data_save, delimiter=',', header=header_text)

            # Save the interpolated data if interpolation was performed
            if interp_type == 'linear' or interp_type == 'log':
                if interpolation_pts is not None and linear_results_I is not None and linear_results_e is not None:
                    file_name_interp_txt = path_merged_txt + keys + '_interp'  '.dat'
                    # Save interpolated q, I, and e
                    data_save_interp = np.column_stack((interpolation_pts, linear_results_I, linear_results_e))
                    np.savetxt(file_name_interp_txt, data_save_interp, delimiter=',', header=header_text)
        else:
            print(f"Warning: No valid data points to process for sample: {keys}. Skipping merging and saving.")


# %% subtract incoherent
def subtract_incoherent(path_dir_an, initial_last_points_fit=50, constancy_threshold=0.05):
    """
    Subtracts an incoherent flat background from SAXS/WAXS data.

    This function identifies a constant background in the high-q region of the
    scattering curve and subtracts it from the original data. It assumes that
    at high-q, the scattering from structural features has decayed sufficiently,
    leaving primarily a flat incoherent contribution.

    Args:
        path_dir_an (str): Path to the analysis directory containing 'merged' and 'config.npy'.
        initial_last_points_fit (int): Number of points from the end of the data to consider
                                       for fitting the flat background.
        constancy_threshold (float): A threshold (e.g., as a percentage of the mean intensity)
                                     to determine if the tail is "constant" enough for fitting.
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
    # Collect all relevant .dat files (merged or interpolated) from the merged data directory
    for file in os.listdir(path_merged_txt):
        if file.endswith('.dat') and ('interp' in file or 'merged' in file): # Look for 'interp' or 'merged' in filename
            files_to_process.append(file)

    if not files_to_process:
        print(f"No '.dat' files (interpolated or merged) found in {path_merged_txt}. Exiting.")
        return

    # MODIFIED: Simplified fitting model to only a flat background
    def flat_background_model(q_val, incoherent_val):
        """
        Scattering model for high-q region: only a flat incoherent background.
        I(q) = incoherent_val
        """
        return np.full_like(q_val, incoherent_val) # Returns an array of incoherent_val matching q_val's shape

    # Process each identified file
    for file_short_name in files_to_process:
        plt.close('all') # Close previous plots to prevent them from piling up in memory

        # Extract base name for output files and titles
        base_name = file_short_name.replace('_merged.dat', '').replace('_interp.dat', '')

        # Skip calibration files if they are in the list
        if base_name in calibration.values():
            print(f"Skipping {file_short_name} as it's a calibration file.")
            continue

        file_path_data = os.path.join(path_merged_txt, file_short_name)
        print(f"\nProcessing: {file_path_data}")

        try:
            # Load intensity (I), q-vector (q), and error (e) from the data file
            q = np.genfromtxt(file_path_data, dtype=float, delimiter=',', usecols=0)
            I = np.genfromtxt(file_path_data, dtype=float, delimiter=',', usecols=1)
            e = np.genfromtxt(file_path_data, dtype=float, delimiter=',', usecols=2) # Load errors
        except Exception as e:
            print(f"Error reading data from {file_path_data}: {e}. Skipping this file.")
            continue

        # Basic data validation
        if len(I) == 0 or len(q) == 0 or len(I) != len(q) or len(I) != len(e):
            print(f"Warning: Data arrays are empty or mismatched for {file_short_name}. Skipping.")
            continue
        if np.any(np.isnan(I)) or np.any(np.isnan(q)) or np.any(np.isinf(I)) or np.any(np.isinf(q)):
            print(f"Warning: Data contains NaN or Inf values for {file_short_name}. Skipping.")
            continue
        # Filter out zero or negative intensities for log plot and fitting, as they are problematic for log scale.
        # Also, filter errors to correspond to valid data.
        positive_mask = I > 0
        if not np.any(positive_mask):
            print(f"Warning: All intensities are zero or negative for {file_short_name}. Cannot perform log plot/fit. Skipping.")
            continue
        q_pos = q[positive_mask]
        I_pos = I[positive_mask]
        e_pos = e[positive_mask] # Filter errors consistently

        # Ensure errors are non-negative, as they represent standard deviations
        e_pos = np.abs(e_pos)
        # Avoid zero errors for weighted fitting, replace with a very small value if any are zero
        e_pos[e_pos == 0] = 1e-12


        if len(I_pos) < initial_last_points_fit + 5: # Ensure enough points for initial analysis + fit
            print(f"Warning: Not enough data points ({len(I_pos)}) for initial analysis and fitting in {file_short_name}. Skipping.")
            continue

        # --- Dynamic Constancy Check and Adaptable Fitting Range ---
        # Analyze the tail of the scattering curve to determine a stable region for fitting.
        # This involves checking for constancy in intensity and a minimal slope.
        # We look for a region that is genuinely flat for the incoherent background.
        tail_start_idx = max(0, len(I_pos) - initial_last_points_fit * 2) # Consider a broader tail region for analysis
        q_tail_analysis = q_pos[tail_start_idx:]
        I_tail_analysis = I_pos[tail_start_idx:]

        if len(I_tail_analysis) < 5: # Need at least a few points to assess constancy reliably
            print(f"Warning: Tail analysis range too small for {file_short_name}. Cannot assess constancy. Skipping.")
            # MODIFIED: Removed 'continue' here. The fit will now always be attempted.
            # This allows the user to decide if the fit is acceptable even if the automated check warns.

        # Check standard deviation relative to mean for intensity constancy
        # A smaller value for `constancy_threshold` means stricter constancy.
        if np.mean(I_tail_analysis) > 0:
            relative_std = np.std(I_tail_analysis) / np.mean(I_tail_analysis)
        else:
            relative_std = np.inf # If mean is zero, it's not constant/positive in a meaningful way

        # Check the slope of a linear fit to the tail on a linear scale.
        # This is crucial for verifying if the region is truly 'flat'.
        slope_lin, intercept_lin, r_value_lin, p_value_lin, std_err_lin = linregress(q_tail_analysis, I_tail_analysis)

        # Criteria for a "constant" tail: low relative standard deviation AND a very flat slope
        is_constant_tail = (relative_std < constancy_threshold) and \
                           (abs(slope_lin) < (constancy_threshold * np.mean(I_tail_analysis) / (q_tail_analysis[-1] - q_tail_analysis[0] + 1e-9)))

        if not is_constant_tail:
            # MODIFIED: Changed this from skipping the file to just printing a warning.
            print(f"Warning: Tail of data for {file_short_name} is not sufficiently constant (relative std: {relative_std:.3f}, slope: {slope_lin:.3e}). Proceeding with fit, but manual review recommended.")
        else:
            print(f"Tail of data for {file_short_name} appears constant (relative std: {relative_std:.3f}). Proceeding with fit.")


        # If the tail is considered constant, define the fitting range for the model.
        # We use `initial_last_points_fit` points for the fitting.
        determined_fitting_range = min(initial_last_points_fit, len(I_pos)) # Ensure we don't exceed available points

        # Ensure we have a minimum number of points for a robust fit (only 1 parameter to fit: incoherent_val)
        min_points_for_fit = 1 # Need at least 1 point to fit a constant, but more for robust average
        if determined_fitting_range < min_points_for_fit:
            print(f"Error: Determined fitting range ({determined_fitting_range} points) is too small for a robust fit for {file_short_name}. Skipping.")
            continue # This is a critical error, so we skip the file.

        # Select the data points for fitting the model
        off_set = len(I_pos) - determined_fitting_range
        fitting_I = I_pos[off_set:]
        fitting_q = q_pos[off_set:]
        fitting_e = e_pos[off_set:] # Corresponding errors for fitting

        print(f"Using {len(fitting_I)} points for fitting (q_range: {fitting_q[0]:.3e} to {fitting_q[-1]:.3e}).")

        # --- Initial Guess and Bounds for curve_fit ---
        # Only fitting `incoherent_val` now.
        initial_incoherent = np.mean(fitting_I) # Simple average of the fitting range
        if initial_incoherent <= 0:
            initial_incoherent = 1e-6 # Ensure a small positive guess

        p0 = [initial_incoherent] # Initial guess for the single parameter

        # Define bounds for the single parameter (incoherent_val)
        lower_bounds = [0] # Incoherent background must be non-negative
        upper_bounds = [np.max(fitting_I) * 2] # Should not be arbitrarily large

        try:
            # Perform the non-linear least squares fit using `scipy.optimize.curve_fit`
            # `sigma=fitting_e` is crucial for weighted least squares, giving more importance to points with smaller errors.
            params, cv = scipy.optimize.curve_fit(
                flat_background_model, fitting_q, fitting_I,
                p0=p0,
                sigma=fitting_e, # Use errors for weighted fitting
                bounds=(lower_bounds, upper_bounds),
                max_nfev=1000 # Max function evaluations
            )
            incoherent_fit = params[0] # Unpack the single fitted parameter

            # Calculate R-squared to assess the goodness of the fit
            residuals = fitting_I - flat_background_model(fitting_q, incoherent_fit)
            # to decrease a bit the level of the incohenet background
            incoherent_fit = incoherent_fit*0.99
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((fitting_I - np.mean(fitting_I))**2)
            rSquared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
            rSquared = round(rSquared, 2)

            # --- Plotting the fitted background and subtracted data ---
            # Create a single figure for the log-log plot
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 7)) # Only one subplot now

            # Main log-log plot (ax1)
            ax1.errorbar(q_pos, I_pos, e_pos, fmt='.', lw=0.3, color='blue', alpha=0.5, label="Original data")

            # Plot the fitted incoherent background (flat line) across the full q range
            q_full_range = q_pos
            ax1.loglog(q_full_range, np.ones(len(q_full_range)) * incoherent_fit, '-', color='red', linewidth=2,
                       label=f"Fitted Incoherent Background = {incoherent_fit:.4f} (R²={rSquared})")

            # Plot the specific fitting region for clarity
            ax1.plot(fitting_q, fitting_I, 'o', color='orange', mfc='none', markersize=6, label='Data used for fit')

            # Subtracted data - now plotted on the SAME log-log axes.
            # Only positive values will be shown due to log scale.
            subtracted_I = I_pos - incoherent_fit
            subtracted_e = e_pos # Error remains the same if incoherent is treated as a perfect constant subtraction

            # Mask for positive subtracted data for log plotting
            valid_subtracted_mask = subtracted_I > 0
            ax1.errorbar(q_pos[valid_subtracted_mask], subtracted_I[valid_subtracted_mask],
                         subtracted_e[valid_subtracted_mask], fmt='o', ms=3, lw=0.5, color='black', label='Subtracted data (positive only)')


            ax1.set_title(f"Flat Background Subtraction for {base_name}")
            ax1.set_xlabel(r'Scattering vector q [$\AA^{-1}$]') # X-label only on main plot
            ax1.set_ylabel(r'Intensity I [cm$^{-1}$]')
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.legend(loc='best') # Place legend automatically
            ax1.grid(True, which="both", ls="-", alpha=0.2)

            plt.tight_layout() # Adjust layout
            file_name_fig = os.path.join(path_merged_fig, f"{base_name}_flat_background_subtraction.jpeg")
            plt.savefig(file_name_fig, bbox_inches='tight', dpi=150)
            plt.close(fig) # Close the figure to free memory

            # Save the processed data (q, I_subtracted, error)
            header_text = 'q (A-1), I_subtracted (1/cm), error' # Updated header
            file_name_data_out = os.path.join(path_merged_txt, f"{base_name}_subtracted.dat")

            np.savetxt(file_name_data_out, np.column_stack((q_pos, subtracted_I, subtracted_e)),
                       delimiter=',', header=header_text, comments='# ')

        except (RuntimeError, ValueError) as e:
            print(f"Error during curve fitting or plotting for {file_short_name}: {e}. Skipping this file.")
            # Save original data if fit failed to ensure data is not lost or corrupted
            header_text = 'q (A-1), I (1/cm), error - Fit failed, data not subtracted'
            file_name_data_out = os.path.join(path_merged_txt, f"{base_name}_original_fit_failed.dat")
            np.savetxt(file_name_data_out, np.column_stack((q, I, e)), # Save original I and e
                       delimiter=',', header=header_text, comments='# ')
            continue
        except Exception as e:
            print(f"An unexpected error occurred for {file_short_name}: {e}. Skipping this file.")
            continue

    print("\nFlat background subtraction process completed.")
