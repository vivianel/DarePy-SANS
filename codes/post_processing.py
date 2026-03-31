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
def plot_all_data(path_dir_an, skip_start, skip_end):
    """
    Loads radial integration data and highlights the points defined in YAML skips.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import pickle
    import integration as integ

    # 1. Setup Folder Structure
    path_merged = os.path.join(path_dir_an, 'merged/')
    path_merged_fig = os.path.join(path_merged, 'figures/')
    for p in [path_merged, path_merged_fig]:
        os.makedirs(p, exist_ok=True)

    # 2. Load Metadata
    file_results = os.path.join(path_dir_an, 'result.npy')
    with open(file_results, 'rb') as handle:
        result = pickle.load(handle)
    list_class_files = result['overview']

    merged_files = {}

    # 3. Aggregate Data
    for key in list_class_files:
        if 'det' in key:
            det_dist_str = key.replace('det_files_', '').replace('p', '.')
            # Determine the index (0, 1, 2) based on the order detector distances are found
            # This logic assumes unique_det ordering matches your skip keys
            total_samples = len(list_class_files[key]['scan'])

            for ii in range(total_samples):
                sample_name = list_class_files[key]['sample_name'][ii]
                scan_nr = list_class_files[key]['scan'][ii]
                det_val = list_class_files[key]['detx_m'][ii]

                path_integ = os.path.join(path_dir_an, f'det_{str(det_val).replace(".","p")}', 'integration/')
                file_name = integ.make_file_name(path_integ, 'radial_integ', 'dat', sample_name,
                                               str(det_val).replace('.','p'), scan_nr, 0)

                try:
                    data = np.genfromtxt(file_name, delimiter=',', skip_header=1)
                    q, I, e = data[:, 0], data[:, 1], np.abs(data[:, 2])
                except: continue

                if sample_name not in merged_files:
                    merged_files[sample_name] = {'I': [], 'q': [], 'error': [], 'det': []}

                merged_files[sample_name]['I'].append(I)
                merged_files[sample_name]['q'].append(q)
                merged_files[sample_name]['error'].append(e)
                merged_files[sample_name]['det'].append(det_dist_str)

    # 4. Plot with YAML-defined Exclusions
    for name in merged_files:
        plt.figure(figsize=(10, 7))
        plt.ioff()

        num_segments = len(merged_files[name]['q'])
        for i in range(num_segments):
            q_seg = merged_files[name]['q'][i]
            I_seg = merged_files[name]['I'][i]
            e_seg = merged_files[name]['error'][i]
            d_label = merged_files[name]['det'][i] # This is the string '1.6', '6.0', etc.

            # --- IDENTIFY YAML SKIPS BY DISTANCE ---
            # We now use the d_label (distance) as the key instead of str(i)
            s_start = skip_start.get(d_label, 0)
            s_end = skip_end.get(d_label, 0)

            exclude_mask = np.zeros(len(q_seg), dtype=bool)
            if s_start > 0: exclude_mask[:s_start] = True
            if s_end > 0: exclude_mask[-s_end:] = True

            # Plot "Good" points (Original Segment Color)
            plt.errorbar(q_seg[~exclude_mask], I_seg[~exclude_mask], yerr=e_seg[~exclude_mask],
                         fmt='o', ms=3, lw=0.6, label=f"Det {d_label}m")

            # Plot "Excluded" points (RED)
            if np.any(exclude_mask):
                plt.plot(q_seg[exclude_mask], I_seg[exclude_mask], 'rx', ms=4, alpha=0.5,
                         label="Exclusion Defined in YAML" if i == 0 else "")

        plt.xscale('log'); plt.yscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.xlabel(r'$q$ [$\AA^{-1}$]'); plt.ylabel(r'$I(q)$ [cm$^{-1}$]')
        plt.title(f'Overlay Check: {name} (Current YAML Trimming)')
        plt.legend(fontsize='small', loc='best')

        file_path = os.path.join(path_merged_fig, f'{name}_yaml_trim_check.jpeg')
        plt.savefig(file_path, dpi=200, bbox_inches='tight')
        plt.close()

    return merged_files

# %% merging_data
def merging_data(path_dir_an, merged_files, skip_start, skip_end):
    """
    Stitches scattering data from different detector distances for each sample.
    Handles data stored as lists to accommodate varying segment lengths.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d

    # Setup directories
    path_merged = os.path.join(path_dir_an, 'merged/')
    path_merged_fig = os.path.join(path_merged, 'figures/')
    path_merged_txt = os.path.join(path_merged, 'data_txt/')
    os.makedirs(path_merged_fig, exist_ok=True)
    os.makedirs(path_merged_txt, exist_ok=True)

    for keys in merged_files:
        I_all, q_all, e_all = [], [], []
        plt.close('all')
        plt.ioff()

        # Check if we have multiple segments (stored as a list)
        num_segments = len(merged_files[keys]['q'])

        if num_segments > 1:
            # Sort detectors from low-q to high-q based on the first q-point of each list item
            first_q_vals = [merged_files[keys]['q'][kk][0] for kk in range(num_segments)]
            idx_det = np.argsort(first_q_vals)
            scaling = 1.0

            for ii in idx_det:
                q = np.array(merged_files[keys]['q'][ii])
                I = np.array(merged_files[keys]['I'][ii])
                e = np.array(merged_files[keys]['error'][ii])
                d_label = merged_files[keys]['det'][ii] # Get distance string

                # 1. Clean NaNs and Zeros
                mask = (~np.isnan(q)) & (I > 0)
                q, I, e = q[mask], I[mask], e[mask]

                # 2. Apply YAML Skips BY DISTANCE
                s_start = skip_start.get(d_label, 0)
                s_end = skip_end.get(d_label, 0)

                end_idx = max(0, len(q) - s_end)
                q, I, e = q[s_start:end_idx], I[s_start:end_idx], e[s_start:end_idx]

                # 3. Calculate Scaling based on Overlap
                if ii != idx_det[0] and len(q_all) > 0:
                    min_overlap = max(np.min(q_all), np.min(q))
                    max_overlap = min(np.max(q_all), np.max(q))

                    if max_overlap > min_overlap:
                        interp_q = np.linspace(min_overlap, max_overlap, 50)
                        try:
                            # Interpolate existing data vs new segment
                            i_prev = interp1d(q_all, I_all, kind='linear', fill_value="extrapolate")(interp_q)
                            i_curr = interp1d(q, I, kind='linear', fill_value="extrapolate")(interp_q)

                            valid = (i_curr > 0) & (i_prev > 0)
                            if np.any(valid):
                                scaling = np.median(i_prev[valid] / i_curr[valid])
                        except:
                            scaling = 1.0

                    I = np.multiply(I, scaling)
                    e = np.multiply(e, scaling)

                # Concatenate the trimmed and scaled segment to the master lists
                q_all = np.concatenate((q_all, q))
                I_all = np.concatenate((I_all, I))
                e_all = np.concatenate((e_all, e))
        else:
            # Handle single detector measurement (convert list element to array)
            q_all = np.array(merged_files[keys]['q'][0])
            I_all = np.array(merged_files[keys]['I'][0])
            e_all = np.array(merged_files[keys]['error'][0])

        # 4. Final Sort and Save
        if len(q_all) > 0:
            idx = np.argsort(q_all)
            q_final, I_final, e_final = q_all[idx], I_all[idx], e_all[idx]

            # Save Text File (Name + _merged.dat)
            file_txt = os.path.join(path_merged_txt, f"{keys}_merged.dat")
            header = 'q (A-1), I (1/cm), error'
            np.savetxt(file_txt, np.column_stack((q_final, I_final, e_final)), delimiter=',', header=header)
            print(f"  [SAVED] Merged raw data: {keys}_merged.dat")

            # 5. Save Plot
            plt.figure(figsize=(8, 6))
            plt.errorbar(q_final, I_final, yerr=e_final, fmt='o', ms=2, lw=0.4, color='black', alpha=0.6)
            plt.xscale('log'); plt.yscale('log')
            plt.xlabel(r'$q$ [$\AA^{-1}$]'); plt.ylabel(r'$I(q)$ [cm$^{-1}$]')
            plt.title(f'Merged Stitched Data: {keys}')
            file_fig = os.path.join(path_merged_fig, f"{keys}_merged.jpeg")
            plt.savefig(file_fig, dpi=150)
            plt.close()
        else:
            print(f"  [SKIP] {keys}: No data points remaining after trimming.")

    return True

# %% interpolate_data
def interpolate_data(path_dir_an, interp_type='log', interp_points=150, smooth_window=1):
    """
    Optional Step 3: Reads merged raw files and performs rebinning/interpolation.
    Uses log-binning to physically crush high-q error bars.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d
    from utils import smooth

    # 1. Setup paths
    path_merged = os.path.join(path_dir_an, 'merged/')
    path_merged_txt = os.path.join(path_merged, 'data_txt/')
    path_merged_fig = os.path.join(path_merged, 'figures/')

    # 2. Identify merged files to process
    files_to_process = [f for f in os.listdir(path_merged_txt) if f.endswith('_merged.dat')]

    if not files_to_process:
        print("[WARNING] No '_merged.dat' files found. Run Step 2 (merging_data) first.")
        return

    print(f"\n[STEP 3] Interpolating {len(files_to_process)} files (Type: {interp_type})...")

    for file_name in files_to_process:
        sample_name = file_name.replace('_merged.dat', '')
        file_path = os.path.join(path_merged_txt, file_name)

        # Load raw stitched data
        data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
        q_raw, I_raw, e_raw = data[:, 0], data[:, 1], data[:, 2]

        # --- LOGARITHMIC REBINNING (Recommended for SANS) ---
        if interp_type == 'log':
            # Create log-spaced bins
            q_min, q_max = np.min(q_raw[q_raw > 0]), np.max(q_raw)
            q_bins = np.logspace(np.log10(q_min), np.log10(q_max), interp_points + 1)

            q_int, I_int, e_int = [], [], []
            for i in range(interp_points):
                mask = (q_raw >= q_bins[i]) & (q_raw < q_bins[i+1])
                if np.any(mask):
                    q_int.append(np.mean(q_raw[mask]))
                    # Rigorous Inverse-Variance Weighting for Noise Reduction
                    weights = 1.0 / (e_raw[mask]**2)
                    I_int.append(np.average(I_raw[mask], weights=weights))
                    e_int.append(np.sqrt(1.0 / np.sum(weights)))

            q_final, I_final, e_final = np.array(q_int), np.array(I_int), np.array(e_int)

        # --- LINEAR INTERPOLATION ---
        elif interp_type == 'linear':
            q_final = np.linspace(np.min(q_raw), np.max(q_raw), interp_points)
            I_final = interp1d(q_raw, I_raw, kind='linear')(q_final)
            e_final = interp1d(q_raw, e_raw, kind='linear')(q_final)
            if smooth_window > 1:
                I_final = smooth(I_final, smooth_window)

        else:
            print(f"Skipping interpolation for {sample_name} (type set to 'none').")
            continue

        # 3. Save Interpolated Text File
        file_out = os.path.join(path_merged_txt, f"{sample_name}_interp.dat")
        header = f'q (A-1), I (1/cm), error (Interpolated: {interp_type})'
        np.savetxt(file_out, np.column_stack((q_final, I_final, e_final)), delimiter=',', header=header)

        # 4. Plot Comparison
        plt.figure(figsize=(8, 6))
        plt.ioff()
        plt.errorbar(q_raw, I_raw, yerr=e_raw, fmt='o', ms=2, color='gray', alpha=0.2, label='Raw Stitched')
        plt.errorbar(q_final, I_final, yerr=e_final, fmt='o', ms=4, color='red', label=f'Interpolated ({interp_type})')
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel(r'$q$ [$\AA^{-1}$]'); plt.ylabel(r'$I(q)$ [cm$^{-1}$]')
        plt.title(f'Interpolation Check: {sample_name}')
        plt.legend()
        plt.savefig(os.path.join(path_merged_fig, f"{sample_name}_interp.jpeg"), dpi=150)
        plt.close()

    print("Step 3 Complete. Check the 'data_txt' folder for '_interp.dat' files.")
    return True

# %% subtract incoherent
# %% subtract_incoherent
def subtract_incoherent(path_dir_an, initial_last_points_fit=50, constancy_threshold=0.05):
    """
    Step 4: Subtracts incoherent flat background.
    Automatically prioritizes interpolated data for better fitting accuracy.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle
    import scipy.optimize
    from scipy.stats import linregress

    path_merged = os.path.join(path_dir_an, 'merged')
    path_merged_txt = os.path.join(path_merged, 'data_txt')
    path_merged_fig = os.path.join(path_merged, 'figures')

    # 1. Decide which files to process (Prioritize _interp over _merged)
    all_files = os.listdir(path_merged_txt)
    interp_files = [f for f in all_files if f.endswith('_interp.dat')]
    raw_merged_files = [f for f in all_files if f.endswith('_merged.dat')]

    # If interp files exist, we use them. Otherwise, we use the raw merged ones.
    files_to_process = interp_files if interp_files else raw_merged_files

    if not files_to_process:
        print("[ERROR] No data files found to perform background subtraction.")
        return

    def flat_background_model(q_val, incoherent_val):
        return np.full_like(q_val, incoherent_val)

    for file_short_name in files_to_process:
        plt.close('all')
        base_name = file_short_name.replace('_merged.dat', '').replace('_interp.dat', '')
        file_path_data = os.path.join(path_merged_txt, file_short_name)

        print(f"\n[STEP 4] Subtracting background from: {file_short_name}")

        try:
            data = np.genfromtxt(file_path_data, delimiter=',', skip_header=1)
            q, I, e = data[:, 0], data[:, 1], data[:, 2]
        except: continue

        # Filter for positive intensities
        mask = I > 0
        q_pos, I_pos, e_pos = q[mask], I[mask], e[mask]

        # --- DYNAMIC FITTING ---
        # Use the end of the curve to find the plateau
        fit_range = min(initial_last_points_fit, len(I_pos))
        off_set = len(I_pos) - fit_range
        f_q, f_I, f_e = q_pos[off_set:], I_pos[off_set:], e_pos[off_set:]

        # Avoid zero-division in weighting
        f_e[f_e <= 0] = 1e-12

        try:
            p0 = [np.mean(f_I)]
            params, _ = scipy.optimize.curve_fit(
                flat_background_model, f_q, f_I, p0=p0, sigma=f_e,
                bounds=(0, np.max(f_I)*1.5)
            )
            # Apply slight 3% correction factor as per your preference
            incoherent_fit = params[0] * 0.97

            subtracted_I = I_pos - incoherent_fit

            # --- SAVE RESULTS ---
            header = f'q (A-1), I_subtracted (1/cm), error (BG Subtracted: {incoherent_fit:.5f})'
            suffix = "_subtracted.dat"
            file_out = os.path.join(path_merged_txt, f"{base_name}{suffix}")
            np.savetxt(file_out, np.column_stack((q_pos, subtracted_I, e_pos)), delimiter=',', header=header)

            # --- PLOT RESULTS ---
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.errorbar(q_pos, I_pos, yerr=e_pos, fmt='.', color='blue', alpha=0.3, label="Original")
            ax.axhline(incoherent_fit, color='red', lw=2, label=f"Fit: {incoherent_fit:.4f}")

            # Plot positive subtracted results
            sub_mask = subtracted_I > 0
            ax.errorbar(q_pos[sub_mask], subtracted_I[sub_mask], yerr=e_pos[sub_mask],
                        fmt='o', ms=3, color='black', label="Subtracted")

            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlabel(r'$q$ [$\AA^{-1}$]'); ax.set_ylabel(r'$I(q)$ [cm$^{-1}$]')
            ax.set_title(f"Background Subtraction: {base_name}")
            ax.legend()
            plt.savefig(os.path.join(path_merged_fig, f"{base_name}_subtracted.jpeg"), dpi=150)

        except Exception as err:
            print(f"  [ERROR] Fit failed for {base_name}: {err}")

    print("Background subtraction process completed.")
