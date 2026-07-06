import numpy as np
from utils import load_instrument_registry, load_hdf

def calculate_q_resolution(q_A_1, config, det_dist_m, wl_A, L1_m, hdf_name, chi_deg=None):
    """
    Calculates the total q-resolution (dq) in Å⁻¹ combining geometric collimation
    and wavelength dispersion.
    """
    registry = load_instrument_registry()
    inst_name = config['instrument']['name']
    registry = registry[inst_name]
    path_hdf_raw = config['analysis']['path_hdf_raw']

    dl_l_fwhm = registry.get('wavelength_spread', 0.10)
    L2_m = float(det_dist_m)

    # Source (A1) Settings - e.g., the neutron guide
    source_shape = registry['source_shape']
    source_slit_width = registry['source_guide_width']   # W1 in meters
    source_slit_height = registry['source_guide_height'] # H1 in meters

    # Sample (A2) Settings - e.g., the microfluidic channel or slit
    aperture_shape = load_hdf(path_hdf_raw, hdf_name, 'aperture_shape')
    sample_slit_width = load_hdf(path_hdf_raw, hdf_name, 'aperture_x')    # W2 in meters
    sample_slit_height = load_hdf(path_hdf_raw, hdf_name, 'aperture_y')   # H2 in meters

    # -------------------------------------------------------------
    # 1. SPATIAL VARIANCES (Horizontal and Vertical)
    # -------------------------------------------------------------
    if source_shape in ['rectangular', 'square']:
        var_W1 = (source_slit_width**2) / 12
        var_H1 = (source_slit_height**2) / 12
    else:  # Default to circular/open approximation
        var_W1 = var_H1 = (source_slit_width**2) / 4

    if aperture_shape in ['rectangular', 'square']:
        var_W2 = (sample_slit_width**2) / 12
        var_H2 = (sample_slit_height**2) / 12
    else:
        var_W2 = var_H2 = (sample_slit_width**2) / 4

    # -------------------------------------------------------------
    # 2. MILDNER-CARPENTER GEOMETRIC COMPONENTS
    # -------------------------------------------------------------
    # Calculate geometric spatial variance independently for X and Y (in m²)
    var_geom_m2_X = var_W1 * (L2_m / L1_m)**2 + var_W2 * ((L1_m + L2_m) / L1_m)**2
    var_geom_m2_Y = var_H1 * (L2_m / L1_m)**2 + var_H2 * ((L1_m + L2_m) / L1_m)**2

    # Conversion factor from spatial variance (m²) to momentum variance (Å⁻²)
    conv_factor = (2 * np.pi) / (wl_A * L2_m)

    var_q_geom_X = (conv_factor**2) * var_geom_m2_X
    var_q_geom_Y = (conv_factor**2) * var_geom_m2_Y

    # -------------------------------------------------------------
    # 3. WAVELENGTH DISPERSION (Radial smearing)
    # -------------------------------------------------------------
    # Converts FWHM of Delta_lambda / lambda to Gaussian standard deviation sigma
    var_q_lambda = (q_A_1 * (dl_l_fwhm / np.sqrt(8 * np.log(2))))**2

    # -------------------------------------------------------------
    # 4. DIRECTIONAL PROJECTION AND TOTAL RESOLUTION
    # -------------------------------------------------------------
    if chi_deg is not None:
        chi_rad = np.radians(chi_deg)
        cos2 = np.cos(chi_rad)**2
        sin2 = np.sin(chi_rad)**2

        # Project geometric variance onto the parallel (radial) axis
        var_q_geom_parallel = (var_q_geom_X * cos2) + (var_q_geom_Y * sin2)

        # Wavelength smearing acts strictly along the radial (parallel) vector
        var_total = var_q_geom_parallel + var_q_lambda
    else:
        # Isotropic average fallback
        var_q_geom_avg = (var_q_geom_X + var_q_geom_Y) / 2
        var_total = var_q_geom_avg + var_q_lambda

    return np.sqrt(var_total)
