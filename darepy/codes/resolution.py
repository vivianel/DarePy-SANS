import numpy as np
from utils import load_instrument_registry, load_hdf

def get_spatial_variance(width, height, shape):
    """Calculates independent horizontal (X) and vertical (Y) spatial variances."""
    if shape in ['rectangular', 'square']:
        return (width**2) / 12, (height**2) / 12
    else:
        # Default to circular/open approximation
        return (width**2) / 4, (width**2) / 4

def calculate_geometric_variance(wl_A, L1_m, L2_m, source_dims, sample_dims):
    """
    Computes Mildner-Carpenter geometric variance components along X and Y axes (Å⁻²).
    Dimensions must be supplied as tuples: (width, height, shape)
    """
    w1, h1, shape1 = source_dims
    w2, h2, shape2 = sample_dims

    var_W1, var_H1 = get_spatial_variance(w1, h1, shape1)
    var_W2, var_H2 = get_spatial_variance(w2, h2, shape2)

    # Spatial baseline variances at the detector plane (m²)
    var_geom_m2_X = var_W1 * (L2_m / L1_m)**2 + var_W2 * ((L1_m + L2_m) / L1_m)**2
    var_geom_m2_Y = var_H1 * (L2_m / L1_m)**2 + var_H2 * ((L1_m + L2_m) / L1_m)**2

    # Convert from meters to momentum transfer (Å⁻²)
    conv_factor = (2 * np.pi) / (wl_A * L2_m)
    return (conv_factor**2) * var_geom_m2_X, (conv_factor**2) * var_geom_m2_Y

def calculate_wavelength_variance(q_A_1, dl_l_fwhm):
    """Computes the radial wavelength dispersion variance contribution (Å⁻²)."""
    return (q_A_1 * (dl_l_fwhm / np.sqrt(8 * np.log(2))))**2

def calculate_detector_variance(wl_A, L2_m, pixel_size_m):
    """Computes the detector pixel resolution variance contribution (Å⁻²)."""
    conv_factor = (2 * np.pi) / (wl_A * L2_m)
    return (conv_factor**2) * ((pixel_size_m**2) / 12)


def calculate_q_resolution(q_A_1, config, det_dist_m, wl_A, L1_m, hdf_name, chi_deg=None):
    """
    Orchestrates and combines independent resolution contributions into the
    total momentum resolution (dq) in Å⁻¹ for a targeted sector.
    """
    # -------------------------------------------------------------
    # LOAD INSTRUMENT PARAMETERS
    # -------------------------------------------------------------
    registry = load_instrument_registry()
    inst_name = config['instrument']['name']
    registry = registry[inst_name]
    path_hdf_raw = config['analysis']['path_hdf_raw']

    dl_l_fwhm = registry.get('wavelength_spread', 0.10)
    pixel_size_m = registry.get('pixel_size', 0.0075)
    L2_m = float(det_dist_m)

    if config['experiment']['resolution_settings']['source_slit_shape'] == 'auto':
        # Source slit (A1) dimensions in meters
        source_dims = (
            registry['source_slit_x'],
            registry['source_slit_y'],
            registry['source_slit_shape']       )
    else:
        source_dims = (
            config['experiment']['resolution_settings']['source_slit_x'],
            config['experiment']['resolution_settings']['source_slit_y'],
            config['experiment']['resolution_settings']['source_slit_shape'])

    if config['experiment']['resolution_settings']['sample_slit_shape'] == 'auto':
    # Sample slit (A2) dimensions in meters
        sample_dims = (
            load_hdf(path_hdf_raw, hdf_name, 'sample_slit_x'),
            load_hdf(path_hdf_raw, hdf_name, 'sample_slit_y'),
            load_hdf(path_hdf_raw, hdf_name, 'sample_slit_shape')        )
    else:
        sample_dims = (
            config['experiment']['resolution_settings']['sample_slit_x'],
            config['experiment']['resolution_settings']['sample_slit_y'],
            config['experiment']['resolution_settings']['sample_slit_shape'])
    # -------------------------------------------------------------
    # CALCULATE INDEPENDENT CONTRIBUTIONS (As Variances)
    # -------------------------------------------------------------
    # 1. Geometry components
    var_q_geom_X, var_q_geom_Y = calculate_geometric_variance(wl_A, L1_m, L2_m, source_dims, sample_dims)

    # 2. Wavelength spreading
    var_q_lambda = calculate_wavelength_variance(q_A_1, dl_l_fwhm)

    # 3. Detector pixel smearing
    var_q_det = calculate_detector_variance(wl_A, L2_m, pixel_size_m)

    # -------------------------------------------------------------
    # DIRECTIONAL PROJECTION AND TOTAL SUMMATION
    # -------------------------------------------------------------
    if chi_deg is not None:
        chi_rad = np.radians(chi_deg)
        cos2 = np.cos(chi_rad)**2
        sin2 = np.sin(chi_rad)**2

        # Project geometric variance onto the parallel (radial) axis for the chosen sector
        var_q_geom_parallel = (var_q_geom_X * cos2) + (var_q_geom_Y * sin2)
    else:
        # Isotropic average fallback
        var_q_geom_parallel = (var_q_geom_X + var_q_geom_Y) / 2

    # Total variance along the radial scattering direction of the selected sector
    var_total = var_q_geom_parallel + var_q_lambda + var_q_det

    return np.sqrt(var_total)
