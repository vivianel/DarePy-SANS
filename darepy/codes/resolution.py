import numpy as np

def calculate_q_resolution(q, config, det_dist_m, wl_A, L1_m, chi_deg=None):
    """
    Calculates the q-dependent resolution (dq). If chi_deg is provided,
    it dynamically calculates the anisotropic smearing for that specific sector angle.
    """
    inst_name = config.get('instrument_setup', {}).get('which_instrument', 'SANS-I')
    registry = config.get(inst_name, {})

    dl_l_fwhm = registry.get('wavelength_spread', 0.10)
    pixel_size_m = registry.get('pixel_size', 0.0075)
    L2_m = float(det_dist_m)
    res_settings = config.get('resolution_settings', {})

    # -------------------------------------------------------------
    # 1. DYNAMIC SOURCE VARIANCE (A1)
    # -------------------------------------------------------------
    if res_settings.get('source_shape', 'circular') == 'rectangular':
        W1_m = res_settings.get('source_slit_width', 0.050)
        H1_m = res_settings.get('source_slit_height', 0.050)
        var_W1, var_H1 = (W1_m**2) / 12, (H1_m**2) / 12

        if chi_deg is not None:
            chi_rad = np.radians(chi_deg)
            var_A1 = var_W1 * (np.cos(chi_rad)**2) + var_H1 * (np.sin(chi_rad)**2)
        else:
            var_A1 = (var_W1 + var_H1) / 2 # Isotropic average
    else:
        var_A1 = (res_settings.get('source_aperture_radius', 0.009)**2) / 4

    # -------------------------------------------------------------
    # 2. DYNAMIC SAMPLE VARIANCE (A2 - Slit)
    # -------------------------------------------------------------
    if res_settings.get('aperture_shape', 'circular') == 'rectangular':
        W2_m = res_settings.get('sample_slit_width', 0.0002)
        H2_m = res_settings.get('sample_slit_height', 0.002)
        var_W2, var_H2 = (W2_m**2) / 12, (H2_m**2) / 12

        if chi_deg is not None:
            chi_rad = np.radians(chi_deg)
            var_A2 = var_W2 * (np.cos(chi_rad)**2) + var_H2 * (np.sin(chi_rad)**2)
        else:
            var_A2 = (var_W2 + var_H2) / 2 # Isotropic average
    else:
        var_A2 = (res_settings.get('sample_aperture_radius', 0.004)**2) / 4

    # -------------------------------------------------------------
    # 3. RESOLUTION MATH
    # -------------------------------------------------------------
    var_q_lambda = (q * (dl_l_fwhm / np.sqrt(8 * np.log(2))))**2
    var_geom_m2 = var_A1 * (L2_m / L1_m)**2 + var_A2 * ((L1_m + L2_m) / L1_m)**2
    conv_factor = (2 * np.pi) / (wl_A * L2_m)

    var_q_geom = (conv_factor**2) * var_geom_m2
    var_q_det = (conv_factor**2) * ((pixel_size_m**2) / 12)

    return np.sqrt(var_q_lambda + var_q_geom + var_q_det)
