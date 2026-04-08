# -*- coding: utf-8 -*-
"""
DarePy-SANS: Plotting Module
Generates 2D and 1D plots of integrated SANS data.
Optimized with the headless 'Agg' backend for ultra-fast loop processing.
Now acts purely as a visualization tool (math and saving is handled by integration.py).
"""
import os
import gc  # Added for strict memory management
import numpy as np
from scipy import optimize

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import create_analysis_folder
import integration as integ


def plot_integ_radial(config, result, ScanNr, Frame, img_2D, data_azimuth):
    """Generates the multi-panel radial integration figure with overlaid q-ranges."""
    path_analysis = create_analysis_folder(config)
    class_all = result['overview']['all_files']

    det_float = None
    sample_name = "N/A"

    try:
        idx = class_all['scan'].index(ScanNr)
        det_float = class_all['detx_m'][idx]
        sample_name = class_all['sample_name'][idx]
    except ValueError:
        return

    if det_float is None:
        return

    det_str = str(det_float).replace('.', 'p')
    path_integ = os.path.join(path_analysis, f'det_{det_str}', 'integration/')
    path_figures = os.path.join(path_analysis, f'det_{det_str}', 'figures/')
    os.makedirs(path_figures, exist_ok=True)

    # LOAD 1D RADIAL DATA ---
    prefix_radial = 'radial_integ'
    file_name_radial = integ.make_file_name(path_integ, prefix_radial, 'dat', sample_name, det_str, ScanNr, Frame)
    try:
        data_rad = np.genfromtxt(file_name_radial, delimiter=',', comments='#')
        if data_rad.size == 0: return
        q = data_rad[:, 0]
        I = data_rad[:, 1]
        sigma = data_rad[:, 2]
    except (IOError, ValueError):
        return

    sectors_nr = result['integration'].get('sectors_nr')
    raw_ranges = result['integration'].get('pixel_range_azim')
    ranges_to_plot = integ._parse_pixel_ranges(raw_ranges)

    plot_azim_parts = sectors_nr is not None and data_azimuth is not None

    if plot_azim_parts:
        npt_azim = np.linspace(0, 360, sectors_nr + 1)
        try:
            q_a = data_azimuth[:, 0]
            I_a = data_azimuth[:, 1 : sectors_nr + 1]
        except (IndexError, TypeError):
            plot_azim_parts = False

    ai = result['integration'].get('ai')
    if ai is None: return

    bc_x = result['integration']['beam_center_x']
    bc_y = result['integration']['beam_center_y']
    mask = result['integration']['int_mask']
    integration_points = result['integration']['integration_points']

    # ==============================================================
    # START FIGURE GENERATION (DONE ONCE)
    # ==============================================================
    plt.ioff()
    fig1, ([[axs0, axs1], [axs2, axs3], [axs4, axs5]]) = plt.subplots(3, 2, figsize=(12, 17))

    mask_bool = mask.astype(bool)
    img1_plot = img_2D.copy()
    img1_plot[mask_bool] = np.nan
    img1_plot[img1_plot == 0] = np.nan

    def x2q(x_pixels, wl_A, dist_m, pixelsize_m):
        dist_A = dist_m * 1e10
        theta = np.arctan((pixelsize_m / 1e-10) * x_pixels / dist_A)
        return (4 * np.pi / wl_A) * np.sin(theta / 2)

    qx = x2q(np.arange(img1_plot.shape[1]) - bc_x, ai.wavelength * 1e10, ai.dist, ai.pixel1)
    qy = x2q(np.arange(img1_plot.shape[0]) - bc_y, ai.wavelength * 1e10, ai.dist, ai.pixel1)
    extent_q = [qx.min(), qx.max(), qy.min(), qy.max()]

    cmap_mask = plt.get_cmap('jet').copy()
    cmap_mask.set_bad(color='black')

    img1_log_plot = img_2D.copy()
    img1_log_plot[img1_log_plot <= 1e-5] = 1e-5
    img1_log_plot[mask_bool] = np.nan
    img1_log_plot[img1_log_plot == 0] = np.nan
    img2_log = np.log(img1_log_plot)

    # Panel 0: Linear 2D
    im0 = axs0.imshow(img1_plot, origin='lower', aspect='equal', cmap=cmap_mask, extent=extent_q, interpolation='nearest')
    fig1.colorbar(im0, ax=axs0, orientation='horizontal', shrink=0.75).set_label(r'I [cm$^{-1}$]')
    axs0.grid(color='white', linestyle='--', linewidth=0.25)
    axs0.set(ylabel=r'q$_{y}$ [$\AA$$^{-1}$]', xlabel=r'q$_{x}$ [$\AA$$^{-1}$]')
    axs0.set_title(f'2D Scattering Pattern (Linear)\nScan: {ScanNr}, Frame: {Frame}, Det: {det_float}m')

    # Panel 1: 1D Total Radial
    axs1.plot(q, I, 'ok', label='total', markersize=6, alpha=0.8)
    axs1.set(xlabel=r'Scattering vector q [$\AA^{-1}$]', ylabel=r'Intensity I [cm$^{-1}$]', xscale='log', yscale='log', title='Sample: '+ str(sample_name))
    axs1.grid(color='gray', linestyle='--', linewidth=0.5)
    axs1.errorbar(q, I, yerr=sigma, color='black', lw=1, markersize=2, capsize=3)

    # Panel 2: Log 2D
    im2 = axs2.imshow(img2_log, origin='lower', aspect='equal', cmap=cmap_mask, extent=extent_q, interpolation='nearest')
    fig1.colorbar(im2, ax=axs2, orientation='horizontal', shrink=0.75).set_label(r'log(I) [cm$^{-1}$]')
    axs2.grid(color='white', linestyle='--', linewidth=0.25)
    axs2.set(ylabel=r'q$_{y}$ [$\AA$$^{-1}$]', xlabel=r'q$_{x}$ [$\AA$$^{-1}$]')
    axs2.set_title('2D Scattering Pattern (Log Scale)')

    # Panel 3: Cake Plot
    try:
        res2d = ai.integrate2d(img1_log_plot, integration_points, 360, method='BBox', unit='q_A^-1', mask=mask_bool)
        I_c, tth_q, chi = res2d
        I_c_plot = I_c.copy()
        I_c_plot[I_c_plot == 0] = np.nan
        img3 = axs3.imshow(I_c_plot, origin="lower", extent=[tth_q.min(), tth_q.max(), chi.min(), chi.max()], aspect="auto", cmap=cmap_mask, interpolation='nearest')
        fig1.colorbar(img3, ax=axs3, orientation='horizontal', shrink=0.75).set_label(r'log(I) [cm$^{-1}$]')
        axs3.set(ylabel=r'Azimuthal angle $\chi$ [degrees]', xlabel=r'q [$\AA^{-1}$]')
        axs3.grid(color='w', linestyle='--', linewidth=1)
        axs3.set_title('2D Integration (Cake Plot)')
    except Exception:
        axs3.text(0.5, 0.5, 'Error generating cake plot', horizontalalignment='center', verticalalignment='center', transform=axs3.transAxes)

    # ==============================================================
    # OVERLAY LOOP FOR AZIMUTHAL PANELS (4 & 5)
    # ==============================================================
    axs4.set_xlabel(r'Scattering vector q [$\AA^{-1}$]')
    axs4.set_ylabel(r'Intensity I [cm$^{-1}$]')
    axs4.grid(color='gray', linestyle='--', linewidth=0.5)
    axs4.set_title('Radial Integration by Azimuthal Sector')

    axs5.set_xlabel(r'Azimuthal angle $\chi$ [$^o$]')
    axs5.set_ylabel(r'Mean Intensity I($\chi$) [cm$^{-1}$]')
    axs5.grid(color='gray', linestyle='--', linewidth=0.5)
    axs5.set_title('Azimuthal Anisotropy (Overlaid)')

    cmaps = ['plasma', 'viridis', 'inferno', 'cividis', 'magma']

    if plot_azim_parts and 'I_a' in locals():
        for i, q_bnds in enumerate(ranges_to_plot):
            start_idx = max(0, q_bnds[0])
            end_idx = min(len(q_a), q_bnds[1])
            pixel_range_azim = range(start_idx, end_idx)

            if len(pixel_range_azim) > 0:
                I_select = I_a[pixel_range_azim, :]

                current_cmap = plt.get_cmap(cmaps[i % len(cmaps)])
                colors = current_cmap(np.linspace(0.1, 0.9, I_select.shape[1]))
                q_a_selected = q_a[pixel_range_azim]

                # Plot Panel 4
                for ii in range(I_select.shape[1]):
                    valid_indices = ~np.isnan(I_select[:, ii])
                    if np.any(valid_indices):
                        lbl = f'{int(npt_azim[ii])}° (R{i+1})' if i == 0 else "_nolegend_"
                        axs4.loglog(q_a_selected[valid_indices], I_select[valid_indices, ii], marker='o', color=colors[ii], markersize=3, alpha=0.5, label=lbl)

                # Plot Panel 5
                range_angle_midpoints = [(npt_azim[rr] + npt_azim[rr+1]) / 2 for rr in range(sectors_nr)]
                sum_point = []
                for ii in range(I_select.shape[1]):
                    valid_vals = I_select[:, ii][~np.isnan(I_select[:, ii])]
                    sum_point.append(np.mean(valid_vals) if valid_vals.size > 0 else np.nan)

                valid_fit_indices = ~np.isnan(sum_point)
                if np.any(valid_fit_indices):
                    sum_point_valid = np.array(sum_point)[valid_fit_indices]
                    range_angle_valid = np.array(range_angle_midpoints)[valid_fit_indices]

                    def cosine_form(theta, I_0, theta0, offset):
                        return I_0 * np.cos(np.radians(theta - theta0)) ** 2 + offset

                    try:
                        base_guess = np.min(sum_point_valid)
                        amp_guess = np.max(sum_point_valid) - base_guess
                        phase_guess = range_angle_valid[np.argmax(sum_point_valid)]
                        param, _ = optimize.curve_fit(
                            cosine_form, range_angle_valid, sum_point_valid,
                            p0=[amp_guess, phase_guess, base_guess], bounds=([0, 0, 0], [np.inf, 180, np.inf])
                        )
                        I_aniso, phi, I_iso = param

                        chi_smooth = np.linspace(0, 360, 100)
                        axs5.semilogy(chi_smooth, cosine_form(chi_smooth, *param), '--', color=colors[0], label=f'R{i+1} Fit ({phi:.1f}°)')
                    except Exception:
                        pass

                    for ii in range(len(range_angle_midpoints)):
                        if not np.isnan(sum_point[ii]):
                            lbl = f'Range {i+1} [{q_bnds[0]}-{q_bnds[1]}]' if ii == 0 else "_nolegend_"
                            axs5.semilogy(range_angle_midpoints[ii], sum_point[ii], marker='o', markersize=8, color=colors[ii], label=lbl)

        axs4.legend(title='Sector Angles', fontsize='x-small', ncol=3)
        axs5.legend(fontsize='small', ncol=2)
    else:
        axs4.text(0.5, 0.5, 'Azimuthal data not loaded', horizontalalignment='center', verticalalignment='center', transform=axs4.transAxes)
        axs5.text(0.5, 0.5, 'Azimuthal data not loaded', horizontalalignment='center', verticalalignment='center', transform=axs5.transAxes)

    fig1.tight_layout()

    # Save the combined Figure
    file_name_fig = integ.make_file_name(path_figures, 'radial_integ_plot_combined', 'jpeg', sample_name, det_str, ScanNr, Frame)
    try:
        plt.savefig(file_name_fig, dpi=150, bbox_inches='tight')
    except Exception as e:
        print(f"Error saving plot for Scan {ScanNr}: {e}")

    fig1.clf(); plt.close(fig1); gc.collect()


def plot_integ_azimuthal(config, result, ScanNr, Frame):
    """Generates the azimuthal integration figure. Only draws visuals, NO file saving!"""
    path_analysis = create_analysis_folder(config)
    sectors_nr = result['integration'].get('sectors_nr')
    raw_ranges = result['integration'].get('pixel_range_azim')
    ranges_to_plot = integ._parse_pixel_ranges(raw_ranges)

    if sectors_nr is None or not ranges_to_plot:
        return

    class_all = result['overview']['all_files']

    try:
        idx = class_all['scan'].index(ScanNr)
        det_float = class_all['detx_m'][idx]
        sample_name = class_all['sample_name'][idx]
    except ValueError:
        return

    if det_float is None: return

    det_str = str(det_float).replace('.', 'p')
    path_figures = os.path.join(path_analysis, f'det_{det_str}', 'figures/')
    os.makedirs(path_figures, exist_ok=True)
    path_integ = os.path.join(path_analysis, f'det_{det_str}', 'integration/')

    file_name_azim = integ.make_file_name(path_integ, 'azim_integ', 'dat', sample_name, det_str, ScanNr, Frame)

    try:
        data_azim = np.genfromtxt(file_name_azim, delimiter=',', comments='#')
    except (IOError, ValueError):
        return

    if data_azim.size == 0: return

    q = data_azim[:, 0]
    I = data_azim[:, 1 : sectors_nr + 1]
    npt_azim_plot = np.linspace(0, 360, sectors_nr + 1)
    range_angle_midpoints = [(npt_azim_plot[rr] + npt_azim_plot[rr+1]) / 2 for rr in range(sectors_nr)]

    plt.ioff()
    fig2, ((axs0, axs1)) = plt.subplots(1, 2, figsize=(15, 5))
    cmaps = ['plasma', 'viridis', 'inferno', 'cividis', 'magma']

    axs0.set_xlabel(r'Scattering vector q [$\AA^{-1}$]')
    axs0.set_ylabel(r'Intensity I [cm$^{-1}$]')
    axs0.grid(color='gray', linestyle='--', linewidth=0.5)
    axs0.set_title(f'Azimuthal Sectors (Scan: {ScanNr})')

    axs1.set_xlabel(r'Azimuthal angle $\chi$ [$^o$]')
    axs1.set_ylabel(r'Sum intensity I($\chi$) [cm$^{-1}$]')
    axs1.grid(color='gray', linestyle='--', linewidth=0.5)
    axs1.set_title(f'Integrated Azimuthal Profile (Sample: {sample_name})')

    for i, q_bnds in enumerate(ranges_to_plot):
        start_idx = max(0, min(q_bnds[0], len(q) - 1))
        end_idx = max(1, min(q_bnds[1], len(q)))

        if start_idx >= end_idx: continue

        q_range_for_plot = range(start_idx, end_idx)
        I_select = I[q_range_for_plot, :]

        current_cmap = plt.get_cmap(cmaps[i % len(cmaps)])
        colors = current_cmap(np.linspace(0.1, 0.9, I_select.shape[1]))

        # Plot Panel 1 (Sectors vs Q)
        for ii in range(I_select.shape[1]):
            valid_indices = ~np.isnan(I_select[:, ii])
            if np.any(valid_indices):
                lbl = f'Sector (R{i+1})' if ii == 0 else "_nolegend_"
                axs0.loglog(q[q_range_for_plot][valid_indices], I_select[valid_indices, ii], marker='o', color=colors[ii], alpha=0.5, label=lbl)

        # Plot Panel 2 (Sum vs Chi)
        I_sum = np.nansum(I_select, axis=0)

        min_I_sum = np.min(I_sum[I_sum > 0]) if np.any(I_sum > 0) else 1e-8
        offset_for_plot = min_I_sum * 0.99

        axs1.semilogy(range_angle_midpoints, I_sum - offset_for_plot, color=colors[0], linestyle='--', label=f'Range {i+1} [{q_bnds[0]}-{q_bnds[1]}]')
        for ii in range(I_select.shape[1]):
            if I_sum[ii] > 0:
                axs1.semilogy(range_angle_midpoints[ii], I_sum[ii] - offset_for_plot, marker='o', markersize=8, color=colors[ii])

    axs0.legend(fontsize='small')
    axs1.legend()
    fig2.tight_layout()

    # Save Combined Plot
    file_name_fig = integ.make_file_name(path_figures, f'azim_integ_plot_combined', 'jpeg', sample_name, det_str, ScanNr, Frame)
    try:
        fig2.savefig(file_name_fig, dpi=150, bbox_inches='tight')
    except Exception as e:
        print(f"Error saving azimuthal plot: {e}")

    fig2.clf(); plt.close(fig2); gc.collect()
