# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 20:50:57 2022

@author: lutzbueno_v

This function automatically marges the data collected in different detector distances
"""



# %% plot_all_data
def plot_all_data(path_dir_an, skip_start, skip_end, force_replot=False):
    """
    Loads radial integration data and highlights the points defined in YAML skips.
    Skips plotting if the file already exists and force_replot is False.
    """
    import numpy as np
    import matplotlib
    try:
        matplotlib.use('Qt5Agg')
    except:
        matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import os
    import pickle
    import integration as integ

    print("\n[INFO] Starting YAML trim check and overlay plotting...")

    # 1. Setup Folder Structure
    path_merged = os.path.join(path_dir_an, 'merged/')
    path_merged_fig = os.path.join(path_merged, 'figures/')
    for p in [path_merged, path_merged_fig]:
        os.makedirs(p, exist_ok=True)

    print(f" -> Output directory verified: {path_merged_fig}")

    # 2. Load Metadata
    file_results = os.path.join(path_dir_an, 'result.npy')
    try:
        with open(file_results, 'rb') as handle:
            result = pickle.load(handle)
        list_class_files = result['overview']
    except FileNotFoundError:
        print(f"[ERROR] Could not find metadata file: {file_results}")
        return {}

    merged_files = {}

    # 3. Aggregate Data
    print(" -> Aggregating data across detectors...")
    for key in list_class_files:
        if 'det' in key:
            det_dist_str = key.replace('det_files_', '').replace('p', '.')
            det_dist_val = float(det_dist_str)  # <-- Convert to numeric float!
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
                    q, I, e, dq = data[:, 0], data[:, 1], np.abs(data[:, 2]), np.abs(data[:, 3])
                except:
                    continue

                if sample_name not in merged_files:
                    merged_files[sample_name] = {'I': [], 'q': [], 'error': [], 'dq': [], 'det': []}

                merged_files[sample_name]['I'].append(I)
                merged_files[sample_name]['q'].append(q)
                merged_files[sample_name]['error'].append(e)
                merged_files[sample_name]['dq'].append(dq)
                merged_files[sample_name]['det'].append(det_dist_val)

    print(f" -> Found {len(merged_files)} unique samples to plot.")

    # 4. Plot with YAML-defined Exclusions
    print("\n[INFO] Generating and saving plots...")
    for name in merged_files:

        # Define the file path FIRST so we can check if it exists
        file_path = os.path.join(path_merged_fig, f'{name}_yaml_trim_check.jpeg')

        # ==========================================
        # NEW: Skip logic
        # ==========================================
        if not force_replot and os.path.exists(file_path):
            print(f"  -> Skip: Plot for {name} already exists.")
            continue  # Jump straight to the next sample in the loop

        print(f"  -> Processing sample: {name}")
        plt.figure(figsize=(10, 7))
        plt.ioff()

        num_segments = len(merged_files[name]['q'])
        for i in range(num_segments):
            q_seg = merged_files[name]['q'][i]
            I_seg = merged_files[name]['I'][i]
            e_seg = merged_files[name]['error'][i]
            #dq_seg = merged_files[name]['dq'][i]
            d_label = merged_files[name]['det'][i]

            # --- IDENTIFY YAML SKIPS BY DISTANCE ---
            s_start = skip_start.get(d_label, 0)
            s_end = skip_end.get(d_label, 0)

            exclude_mask = np.zeros(len(q_seg), dtype=bool)
            if s_start > 0: exclude_mask[:s_start] = True
            if s_end > 0: exclude_mask[-s_end:] = True

            # Plot "Good" points
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

        try:
            plt.savefig(file_path, dpi=200, bbox_inches='tight')
            print(f"     Saved: {file_path}")
        except Exception as err:
            print(f"     [ERROR] Failed to save {file_path}: {err}")

        plt.close()

    print("\n[SUCCESS] All overlay checks processed!")
    return merged_files


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
    path_merged_txt = os.path.join(path_merged, 'data_merged/')
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

    print("Step 3 Complete. Check the 'data_merged' folder for '_interp.dat' files.")
    return True


def merging_data(
        path_dir_an,
        merged_files,
        skip_start,
        skip_end,
        slope_tol=0.30,
        min_overlap_points=8,
        min_logq_span=0.05,
        max_offset_fraction=0.98,
        required_improvement=0.30,
        slope_passes=2
    ):
    """
    Stitch scattering data from different detector distances.

    Procedure
    ---------
    1. Clean and trim each detector curve.
    2. Sort detector curves from high-q to low-q.
    3. Compare log-log slopes in overlapping q-regions.
    4. If one curve is significantly flatter, try subtracting a
       constant background from that curve.
    5. Accept the subtraction only if the slope agreement improves.
    6. Multiplicatively scale adjacent curves.
    7. Merge, sort and save.

    Parameters
    ----------
    slope_tol : float
        Minimum absolute difference between log-log slopes before
        considering an additive-background correction.

    min_overlap_points : int
        Minimum number of points required in each curve in the overlap.

    min_logq_span : float
        Minimum overlap width in decades of log10(q).
        0.05 corresponds to roughly a 12% q-range.

    max_offset_fraction : float
        Maximum constant that can be subtracted, expressed as a fraction
        of the minimum intensity in the overlap.

    required_improvement : float
        Minimum fractional improvement in slope mismatch required for
        accepting a correction.
        0.30 means the mismatch must improve by at least 30%.

    slope_passes : int
        Number of passes through adjacent detector pairs.
        Two passes help when the middle detector is corrected.
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt

    from scipy.interpolate import interp1d
    from scipy.optimize import minimize_scalar


    # ================================================================
    # Helper: log-log slope
    # ================================================================
    def loglog_slope(q, I, qmin, qmax):

        mask = (
            np.isfinite(q) &
            np.isfinite(I) &
            (q > 0) &
            (I > 0) &
            (q >= qmin) &
            (q <= qmax)
        )

        if np.count_nonzero(mask) < min_overlap_points:
            return None

        q_fit = q[mask]
        I_fit = I[mask]

        logq = np.log10(q_fit)

        # Do not determine slopes from an extremely narrow q interval
        if np.ptp(logq) < min_logq_span:
            return None

        logI = np.log10(I_fit)

        slope, intercept = np.polyfit(logq, logI, 1)

        return slope


    # ================================================================
    # Helper: determine constant background
    # ================================================================
    def find_constant_for_target_slope(
            q,
            I,
            qmin,
            qmax,
            target_slope
        ):

        mask = (
            np.isfinite(q) &
            np.isfinite(I) &
            (q > 0) &
            (I > 0) &
            (q >= qmin) &
            (q <= qmax)
        )

        q_fit = q[mask]
        I_fit = I[mask]

        if len(q_fit) < min_overlap_points:
            return None, None

        # Maximum subtraction allowed.
        #
        # It must remain below the minimum intensity in the overlap.
        C_max = max_offset_fraction * np.min(I_fit)

        if C_max <= 0:
            return None, None

        logq = np.log10(q_fit)

        def objective(C):

            corrected = I_fit - C

            if np.any(corrected <= 0):
                return 1e20

            slope = np.polyfit(
                logq,
                np.log10(corrected),
                1
            )[0]

            return (slope - target_slope)**2

        result = minimize_scalar(
            objective,
            bounds=(0.0, C_max),
            method='bounded',
            options={
                'xatol': max(C_max * 1e-8, 1e-15)
            }
        )

        if not result.success:
            return None, None

        C = result.x

        corrected = I_fit - C

        new_slope = np.polyfit(
            logq,
            np.log10(corrected),
            1
        )[0]

        return C, new_slope


    # ================================================================
    # Directories
    # ================================================================
    path_merged = os.path.join(path_dir_an, 'merged/')
    path_merged_fig = os.path.join(path_merged, 'figures/')
    path_merged_txt = os.path.join(path_merged, 'data_merged/')

    os.makedirs(path_merged_fig, exist_ok=True)
    os.makedirs(path_merged_txt, exist_ok=True)


    # ================================================================
    # Loop over samples
    # ================================================================
    for keys in merged_files:

        plt.close('all')
        plt.ioff()

        num_segments = len(merged_files[keys]['q'])

        # ============================================================
        # Read, clean and trim detector segments
        # ============================================================
        segments = []

        for ii in range(num_segments):

            q = np.asarray(
                merged_files[keys]['q'][ii],
                dtype=float
            )

            I = np.asarray(
                merged_files[keys]['I'][ii],
                dtype=float
            )

            e = np.asarray(
                merged_files[keys]['error'][ii],
                dtype=float
            )

            dq = np.asarray(
                merged_files[keys]['dq'][ii],
                dtype=float
            )

            d_label = merged_files[keys]['det'][ii]

            # --------------------------------------------------------
            # Clean
            # --------------------------------------------------------
            mask = (
                np.isfinite(q) &
                np.isfinite(I) &
                np.isfinite(e) &
                (q > 0) &
                (I > 0)
            )

            q = q[mask]
            I = I[mask]
            e = e[mask]
            dq = dq[mask]

            # --------------------------------------------------------
            # YAML trimming
            # --------------------------------------------------------
            s_start = skip_start.get(d_label, 0)
            s_end = skip_end.get(d_label, 0)

            if s_end > 0:
                end_idx = len(q) - s_end
            else:
                end_idx = len(q)

            end_idx = max(0, end_idx)

            q = q[s_start:end_idx]
            I = I[s_start:end_idx]
            e = e[s_start:end_idx]
            dq = dq[s_start:end_idx]

            if len(q) == 0:

                print(
                    f"  [WARNING] Det {d_label} was "
                    f"completely trimmed/empty."
                )

                continue

            # Sort internally by q
            idx = np.argsort(q)

            segments.append({
                'q': q[idx],
                'I': I[idx],
                'e': e[idx],
                'dq': dq[idx],
                'det': d_label
            })


        if len(segments) == 0:

            print(
                f"  [SKIP] {keys}: no usable data."
            )

            continue


        # ============================================================
        # Sort HIGH-q -> LOW-q
        # ============================================================
        segments.sort(
            key=lambda s: np.max(s['q']),
            reverse=True
        )


        print()
        print("=" * 70)
        print(f"Sample: {keys}")
        print("=" * 70)

        print(
            "Detector order:",
            " -> ".join(
                str(s['det']) for s in segments
            )
        )


        # ============================================================
        # STEP 1
        #
        # ADDITIVE BACKGROUND CORRECTION
        #
        # This is done BEFORE multiplicative scaling because scaling
        # does not change the log-log slope, while an additive constant
        # does.
        # ============================================================

        for slope_pass in range(slope_passes):

            print(
                f"\nSlope correction pass "
                f"{slope_pass + 1}/{slope_passes}"
            )

            correction_made = False

            for j in range(1, len(segments)):

                seg_hi = segments[j - 1]
                seg_lo = segments[j]

                qmin = max(
                    np.min(seg_hi['q']),
                    np.min(seg_lo['q'])
                )

                qmax = min(
                    np.max(seg_hi['q']),
                    np.max(seg_lo['q'])
                )

                if qmax <= qmin:

                    print(
                        f"  No overlap: "
                        f"{seg_hi['det']} / {seg_lo['det']}"
                    )

                    continue


                slope_hi = loglog_slope(
                    seg_hi['q'],
                    seg_hi['I'],
                    qmin,
                    qmax
                )

                slope_lo = loglog_slope(
                    seg_lo['q'],
                    seg_lo['I'],
                    qmin,
                    qmax
                )

                if slope_hi is None or slope_lo is None:

                    print(
                        f"  Insufficient overlap for slope: "
                        f"{seg_hi['det']} / {seg_lo['det']}"
                    )

                    continue


                delta_slope = abs(
                    slope_hi - slope_lo
                )

                print(
                    f"  {seg_hi['det']} vs {seg_lo['det']} "
                    f"| q = {qmin:.4g}-{qmax:.4g}"
                )

                print(
                    f"      slopes: "
                    f"{slope_hi:.3f}  /  "
                    f"{slope_lo:.3f}"
                )

                print(
                    f"      Δslope = "
                    f"{delta_slope:.3f}"
                )


                # ----------------------------------------------------
                # Slopes already sufficiently close
                # ----------------------------------------------------
                if delta_slope <= slope_tol:

                    print(
                        "      -> slopes compatible"
                    )

                    continue


                # ----------------------------------------------------
                # A positive constant background makes a decreasing
                # curve flatter.
                #
                # Therefore this automatic correction is only sensible
                # when both curves are decreasing.
                # ----------------------------------------------------
                if slope_hi >= 0 or slope_lo >= 0:

                    print(
                        "      -> WARNING: slope sign is not "
                        "consistent with simple background flattening."
                    )

                    print(
                        "         No automatic subtraction."
                    )

                    continue


                # ----------------------------------------------------
                # Find flatter curve
                #
                # Example:
                #
                # -0.5 is flatter than -3.0
                # ----------------------------------------------------
                if abs(slope_hi) < abs(slope_lo):

                    flat_seg = seg_hi
                    flat_slope = slope_hi
                    target_slope = slope_lo
                    flat_name = seg_hi['det']

                else:

                    flat_seg = seg_lo
                    flat_slope = slope_lo
                    target_slope = slope_hi
                    flat_name = seg_lo['det']


                # ----------------------------------------------------
                # Find constant C such that slope(I-C)
                # approaches the reference slope
                # ----------------------------------------------------
                C, corrected_slope = (
                    find_constant_for_target_slope(
                        flat_seg['q'],
                        flat_seg['I'],
                        qmin,
                        qmax,
                        target_slope
                    )
                )

                if C is None:

                    print(
                        "      -> could not determine "
                        "stable additive correction"
                    )

                    continue


                mismatch_before = abs(
                    flat_slope - target_slope
                )

                mismatch_after = abs(
                    corrected_slope - target_slope
                )


                if mismatch_before > 0:

                    improvement = (
                        1.0 -
                        mismatch_after /
                        mismatch_before
                    )

                else:
                    improvement = 0


                print(
                    f"      flatter curve: Det {flat_name}"
                )

                print(
                    f"      candidate background C = "
                    f"{C:.6g} cm^-1"
                )

                print(
                    f"      corrected slope = "
                    f"{corrected_slope:.3f}"
                )

                print(
                    f"      improvement = "
                    f"{100 * improvement:.1f}%"
                )


                # ----------------------------------------------------
                # Accept only meaningful corrections
                # ----------------------------------------------------
                if improvement >= required_improvement:

                    flat_seg['I'] = (
                        flat_seg['I'] - C
                    )

                    # Error remains unchanged here.
                    #
                    # This assumes C is a deterministic correction.
                    # If C has an uncertainty, it should be propagated
                    # separately.

                    # Remove points becoming <= 0
                    valid = (
                        np.isfinite(flat_seg['I']) &
                        (flat_seg['I'] > 0)
                    )

                    removed = (
                        len(flat_seg['I']) -
                        np.count_nonzero(valid)
                    )

                    flat_seg['q'] = (
                        flat_seg['q'][valid]
                    )

                    flat_seg['I'] = (
                        flat_seg['I'][valid]
                    )

                    flat_seg['e'] = (
                        flat_seg['e'][valid]
                    )

                    flat_seg['dq'] = (
                        flat_seg['dq'][valid]
                    )

                    correction_made = True

                    print(
                        f"      -> ACCEPTED: "
                        f"subtracting {C:.6g} cm^-1 "
                        f"from Det {flat_name}"
                    )

                    if removed:
                        print(
                            f"         removed {removed} "
                            f"non-positive points"
                        )

                else:

                    print(
                        "      -> rejected: additive constant "
                        "does not sufficiently improve the match"
                    )


            if not correction_made:
                break


        # ============================================================
        # STEP 2
        #
        # MULTIPLICATIVE SCALING
        # ============================================================

        print("\nMultiplicative stitching:")

        for j in range(len(segments)):

            if j == 0:

                print(
                    f"  Det {segments[j]['det']}: "
                    f"reference scale = 1"
                )

                continue


            prev = segments[j - 1]
            curr = segments[j]

            qmin = max(
                np.min(prev['q']),
                np.min(curr['q'])
            )

            qmax = min(
                np.max(prev['q']),
                np.max(curr['q'])
            )

            scaling = 1.0


            if qmax > qmin:

                # Logarithmic sampling is preferable because q itself
                # is logarithmically distributed in most SAS analyses.
                interp_q = np.geomspace(
                    qmin,
                    qmax,
                    100
                )

                f_prev = interp1d(
                    prev['q'],
                    prev['I'],
                    kind='linear',
                    bounds_error=False,
                    fill_value=np.nan
                )

                f_curr = interp1d(
                    curr['q'],
                    curr['I'],
                    kind='linear',
                    bounds_error=False,
                    fill_value=np.nan
                )

                i_prev = f_prev(interp_q)
                i_curr = f_curr(interp_q)

                valid = (
                    np.isfinite(i_prev) &
                    np.isfinite(i_curr) &
                    (i_prev > 0) &
                    (i_curr > 0)
                )

                if np.count_nonzero(valid) >= min_overlap_points:

                    ratios = (
                        i_prev[valid] /
                        i_curr[valid]
                    )

                    scaling = np.median(ratios)

                    curr['I'] *= scaling
                    curr['e'] *= scaling

                    print(
                        f"  Det {curr['det']} -> "
                        f"Det {prev['det']}: "
                        f"scale = {scaling:.6g}"
                    )

                else:

                    print(
                        f"  [WARNING] Not enough valid points "
                        f"for scaling {curr['det']} -> "
                        f"{prev['det']}"
                    )

            else:

                print(
                    f"  [WARNING] No overlap for scaling "
                    f"{curr['det']} -> {prev['det']}"
                )


        # ============================================================
        # STEP 3
        #
        # Merge segments
        # ============================================================

        q_all = np.concatenate([
            s['q'] for s in segments
        ])

        I_all = np.concatenate([
            s['I'] for s in segments
        ])

        e_all = np.concatenate([
            s['e'] for s in segments
        ])

        dq_all = np.concatenate([
            s['dq'] for s in segments
        ])


        # ============================================================
        # Final sort
        # ============================================================

        idx = np.argsort(q_all)

        q_final = q_all[idx]
        I_final = I_all[idx]
        e_final = e_all[idx]
        dq_final = dq_all[idx]


        # ============================================================
        # Save data
        # ============================================================

        file_txt = os.path.join(
            path_merged_txt,
            f"{keys}_merged.dat"
        )

        header = (
            'q (A-1), I (1/cm), error, dq(A-1)'
        )

        np.savetxt(
            file_txt,
            np.column_stack((
                q_final,
                I_final,
                e_final,
                dq_final
            )),
            delimiter=',',
            header=header
        )

        print(
            f"\n  [SAVED] "
            f"{keys}_merged.dat"
        )


        # ============================================================
        # Save merged plot
        # ============================================================

        plt.figure(figsize=(8, 6))

        plt.errorbar(
            q_final,
            I_final,
            yerr=e_final,
            fmt='o',
            ms=2,
            lw=0.4,
            alpha=0.6
        )

        plt.xscale('log')
        plt.yscale('log')

        plt.xlabel(
            r'$q$ [$\AA^{-1}$]'
        )

        plt.ylabel(
            r'$I(q)$ [cm$^{-1}$]'
        )

        plt.title(
            f'Merged Stitched Data: {keys}'
        )

        plt.tight_layout()

        file_fig = os.path.join(
            path_merged_fig,
            f"{keys}_merged.jpeg"
        )

        plt.savefig(
            file_fig,
            dpi=150
        )

        plt.close()


    return True

# %% subtract_incoherent
def subtract_incoherent(path_dir_an, scale_subtraction, initial_last_points_fit=50, constancy_threshold=0.05):
    """
    Step 4: Subtracts incoherent flat background.
    Automatically prioritizes interpolated data for better fitting accuracy.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.optimize


    path_merged = os.path.join(path_dir_an, 'merged')
    path_merged_txt = os.path.join(path_merged, 'data_merged')
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
            incoherent_fit = params[0] * scale_subtraction

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
