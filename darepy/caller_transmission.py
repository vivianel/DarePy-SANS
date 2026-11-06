# -*- coding: utf-8 -*-
import copy
import sys
from pathlib import Path
import os
import pickle

# 1. Get the directory of the current script (darepy/codes/)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Go up one level to find utils.py (in darepy/)
parent_dir = os.path.dirname(current_script_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 3. Point directly to the 'codes' subdirectory where utils.py and backends live
codes_dir = os.path.join(parent_dir, "darepy/codes")

if codes_dir not in sys.path:
    sys.path.insert(0, codes_dir)

# 4. Import utilities and backend logic
from utils import load_config, load_instrument_registry, create_analysis_folder, save_results
import prepare_input as org
from transmission import trans_calc


def _normalise_beamstop(configuration):
    """Return a normalized beamstop name used by the transmission selector."""
    beamstop = configuration.get('transmission_setup', {}).get(
        'beamstop',
        configuration.get('experiment', {}).get('beamstop', 'semitransparent')
    )
    return str(beamstop).strip().lower().replace('-', '').replace('_', '').replace(' ', '')


def _distance_from_overview_key(key):
    """Convert e.g. det_files_17p7 -> 17.7. Return None for invalid keys."""
    if not key.startswith('det_files_'):
        return None
    try:
        return float(key[len('det_files_'):].replace('p', '.'))
    except (TypeError, ValueError):
        return None


def _requested_detector_distances(target_dist):
    """Normalize target_detector_distances to floats, or None for 'all'."""
    if target_dist is None:
        return None

    if isinstance(target_dist, str):
        value = target_dist.strip()
        if value == '' or value.lower() == 'all':
            return None
        target_dist = [value]
    elif isinstance(target_dist, (int, float)):
        target_dist = [target_dist]

    try:
        return [float(value) for value in target_dist]
    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid target_detector_distances value: {target_dist!r}"
        )


def _selected_detector_distances(configuration, result):
    """
    Apply prepare_input's selector, then explicitly retain only the distances
    requested by target_detector_distances. The explicit filtering matters
    because result['overview'] can still contain det_files_* entries from
    other detector distances.
    """
    result = org.select_detector_distances(configuration, result)

    target_dist = configuration.get('analysis', {}).get(
        'target_detector_distances', 'all'
    )
    requested = _requested_detector_distances(target_dist)

    available_distances = []
    selected_distances = []

    for key in result.get('overview', {}):
        distance = _distance_from_overview_key(key)
        if distance is None:
            continue

        available_distances.append(distance)
        if requested is None or any(abs(distance - target) < 1e-6 for target in requested):
            selected_distances.append(distance)

    # Keep stable order while removing duplicates.
    available_distances = list(dict.fromkeys(available_distances))
    selected_distances = list(dict.fromkeys(selected_distances))

    print(f"Requested detector distance(s): {target_dist}")
    print(
        "Available detector distance(s): "
        + (", ".join(f"{d:g}" for d in available_distances)
           if available_distances else "NONE")
    )
    print(
        "Selected detector distance(s): "
        + (", ".join(f"{d:g}" for d in selected_distances)
           if selected_distances else "NONE")
    )

    return result, selected_distances


def _configuration_for_transmission_distance(configuration, distance):
    """
    Return an isolated configuration for one transmission distance.

    The backend historically reads the distance from transmission_setup /
    physics_corrections. experiment['trans_dist'] is synchronized as well so
    both caller styles carry the same value.
    """
    local_configuration = copy.deepcopy(configuration)

    local_configuration.setdefault('transmission_setup', {})[
        'dist_trans_measurements'
    ] = distance
    local_configuration.setdefault('physics_corrections', {})[
        'dist_trans_measurements'
    ] = distance
    local_configuration.setdefault('experiment', {})['trans_dist'] = distance

    return local_configuration


def run_transmission(configuration, class_files, result, ctrl):
    """
    Run transmission with beamstop-dependent distance selection.

    standard beamstop:
        use only transmission_setup.dist_trans_measurements (trans_dist)

    semitransparent beamstop:
        use exactly the detector distances selected by
        analysis.target_detector_distances / select_detector_distances()
    """
    run_trans = ctrl.get('run_transmission', True)
    if not run_trans:
        print("\n--- Transmission Calculation Skipped (Disabled in YAML) ---")
        return result

    beamstop = _normalise_beamstop(configuration)

    if beamstop == 'standard':
        t_dist = configuration.get('transmission_setup', {}).get(
            'dist_trans_measurements', 0
        )

        if not isinstance(t_dist, (int, float)) or t_dist == 0:
            print(
                f"\n--- Transmission skipped. No valid dedicated "
                f"transmission distance provided ({t_dist}m). ---"
            )
            return result

        print("\n" + "*" * 50)
        print("🚀 STARTING: CALCULATING TRANSMISSIONS")
        print("Beamstop: standard")
        print(f"Transmission distance: {t_dist}m")
        print("*" * 50 + "\n")

        local_configuration = _configuration_for_transmission_distance(
            configuration, t_dist
        )
        return trans_calc(local_configuration, class_files, result)

    if beamstop == 'semitransparent':
        # This is the critical behavior change: use the exact same selector as
        # radial integration instead of the dedicated trans_dist value.
        result, selected_distances = _selected_detector_distances(
            configuration, result
        )

        if not selected_distances:
            print(
                "\n--- Transmission skipped. No detector distances remain "
                "after target_detector_distances selection. ---"
            )
            return result

        print("\n" + "*" * 50)
        print("🚀 STARTING: CALCULATING TRANSMISSIONS")
        print("Beamstop: semitransparent")
        print(
            "Selected detector distance(s): "
            + ", ".join(f"{distance:g}m" for distance in selected_distances)
        )
        print("*" * 50 + "\n")

        # A semitransparent transmission is selection-dependent. Rebuild this
        # section so stale transmissions from previously selected distances do
        # not remain active in the result file.
        result['transmission'] = {}

        for distance in selected_distances:
            print(f" -> Transmission at selected detector distance: {distance:g}m")
            local_configuration = _configuration_for_transmission_distance(
                configuration, distance
            )
            result = trans_calc(local_configuration, class_files, result)

        return result

    raise ValueError(
        "Unsupported beamstop setting "
        f"'{configuration.get('transmission_setup', {}).get('beamstop')}'. "
        "Expected 'standard' or 'semitransparent'."
    )


if __name__ == "__main__":
    # --- STEP 1: LOAD CONFIGURATIONS ---
    ext_cfg = load_config()
    INSTRUMENT_REGISTRY = load_instrument_registry()
    selected_inst = ext_cfg['instrument_setup']['which_instrument']
    sample_environment = ext_cfg.get('sample_environment', {})

    transmission_setup = ext_cfg.get('transmission_setup', {})
    t_dist = transmission_setup.get('dist_trans_measurements', 18)

    # --- STEP 2: CONSTRUCT CONFIGURATION OBJECT ---
    configuration = {
        'instrument': INSTRUMENT_REGISTRY[selected_inst],
        'transmission_setup': transmission_setup,
        'experiment': {
            'sample_environment': sample_environment,
            'calibration': ext_cfg.get('calibration_samples', {}),
            'wl_input': ext_cfg['pipeline_control'].get('wavelength', 'auto'),
            'sample_thickness': ext_cfg.get('calibration_samples', {}).get('thickness', {}),
            'beamstop': transmission_setup.get('beamstop', 'semitransparent'),
            'trans_dist': t_dist,
        },
        'physics_corrections': {
            **ext_cfg.get('physics_corrections', {}),
            'dist_trans_measurements': t_dist
        },
        'analysis': {
            'path_dir': str(Path(ext_cfg['analysis_paths']['project_base'])),
            'path_hdf_raw': ext_cfg['analysis_paths']['raw_data'],
            'scripts_dir': ext_cfg['analysis_paths']['scripts_dir'],
            'add_id': ext_cfg['analysis_flags'].get('add_id', ''),
            'exclude_files': ext_cfg['pipeline_control'].get('exclude_files', []),
            'transmission_coordinates': ext_cfg['detector_geometry'].get('transmission_coordinates', {}),
            'force_reintegrate': ext_cfg['analysis_flags'].get('force_reintegrate', False),
            'target_detector_distances': ext_cfg['pipeline_control'].get(
                'target_detector_distances', 'all'
            ),
        }
    }

    # Resolve the destination analysis folder beforehand
    analysis_folder = create_analysis_folder(configuration)
    result_file = os.path.join(analysis_folder, 'result.npy')

    if os.path.exists(result_file):
        print(f"📦 Existing results container found! Loading from: {result_file}")
        with open(result_file, 'rb') as f:
            result = pickle.load(f)

        result.setdefault('transmission', {})
        result.setdefault('overview', {})
        result.setdefault('integration', {})

        # Dynamically synchronize active integration configurations on-load
        result['integration']['pixel_range_azim'] = ext_cfg['analysis_flags']['pixel_range_azim']
        result['integration']['integration_points'] = ext_cfg['analysis_flags'].get('integration_points', 120)
        result['integration']['sectors_nr'] = ext_cfg['analysis_flags']['sectors_nr']
    else:
        print("ℹ️ No existing results file found. Initializing blank results container.")
        result = {
            'transmission': {},
            'overview': {},
            'integration': {
                'pixel_range_azim': ext_cfg['analysis_flags']['pixel_range_azim'],
                'integration_points': ext_cfg['analysis_flags'].get('integration_points', 120),
                'sectors_nr': ext_cfg['analysis_flags']['sectors_nr']
            }
        }

    # --- STEP 3: SINGLE EXECUTION SEQUENCE ---
    print("Step 0: Indexing files (Required for Transmission)...")
    class_files = org.list_files(configuration, result)

    if class_files:
        ctrl = ext_cfg.get('pipeline_control', {})
        result = run_transmission(configuration, class_files, result, ctrl)

        # --- STEP 4: PERSIST RESULTS FOR RADIAL INTEGRATION ---
        save_results(analysis_folder, result)

        print("\n✅ SUCCESS: Transmission calculation complete.")
        print(f"📂 Results saved to: {analysis_folder}")
    else:
        print("❌ ERROR: No valid HDF5 files found in the raw data directory.")
