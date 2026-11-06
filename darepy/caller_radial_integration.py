# -*- coding: utf-8 -*-
"""
DarePy-SANS: Main Orchestration Script (Integration Only)

Transmission selection rules:
- standard beamstop -> dedicated transmission distance (trans_dist)
- semitransparent beamstop -> same target_detector_distances selection as reduction
"""
import sys
import time
import os
import pickle
import subprocess


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


# ==========================================
# STEP 0 & 1: LOAD MASTER CONFIGURATIONS
# ==========================================
from utils import load_config, load_instrument_registry, create_analysis_folder, save_results
import prepare_input as org
import integration as ri

ext_cfg = load_config()
INSTRUMENT_REGISTRY = load_instrument_registry()
sample_environment = ext_cfg.get('sample_environment', {})
selected_inst = ext_cfg['instrument_setup']['which_instrument']
transmission_setup = ext_cfg.get('transmission_setup', {})
ctrl = ext_cfg.get('pipeline_control', {})
live_monitoring = bool(ctrl.get('live_monitoring', False))
stop_file = os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), '.darepy_stop_radial') if len(sys.argv) > 1 and sys.argv[1].lower().endswith(('.yaml', '.yml')) else os.path.join(os.getcwd(), '.darepy_stop_radial')


# ==========================================
# STEP 2: CONSTRUCT CONFIGURATION OBJECT
# ==========================================
configuration = {
    'instrument': INSTRUMENT_REGISTRY[selected_inst],
    'transmission_setup': transmission_setup,
    'experiment': {
        'calibration': ext_cfg['calibration_samples'],
        'wl_input': ext_cfg['pipeline_control']['wavelength'],
        'sample_thickness': ext_cfg.get('calibration_samples', {}).get('thickness', {}),
        'beamstop': transmission_setup.get('beamstop', 'semitransparent'),
        'trans_dist': transmission_setup.get('dist_trans_measurements', 18),
        'sample_environment': sample_environment,
        'resolution_settings': ext_cfg['resolution_settings']
    },
    'physics_corrections': ext_cfg.get('physics_corrections', {}),
    'analysis': {
        'path_dir': ext_cfg['analysis_paths']['project_base'],
        'path_hdf_raw': ext_cfg['analysis_paths']['raw_data'],
        'scripts_dir': ext_cfg['analysis_paths']['scripts_dir'],
        'add_id': ext_cfg['analysis_flags'].get('add_id', ''),
        'exclude_files': ext_cfg['pipeline_control'].get('exclude_files', []),
        'force_reintegrate': ext_cfg['pipeline_control']['force_reintegrate'],
        'live_monitoring': live_monitoring,
        'add_plot_azimuthal': ext_cfg['analysis_flags']['add_plot_azimuthal'],
        'save_plot_radial': ext_cfg['analysis_flags']['save_plot_radial'],
        'save_data_azimuthal': ext_cfg['analysis_flags']['save_data_azimuthal'],
        'save_2d_patterns': ext_cfg['analysis_flags']['save_2d_patterns'],
        'save_format_2d': ext_cfg['analysis_flags']['save_format_2d'],
        'sectors_nr': ext_cfg['analysis_flags']['sectors_nr'],
        'beam_center_guess': {
            # Beam centers MUST be floats for sub-pixel accuracy in pyFAI
            k: [float(x) for x in str(v).split(',')] if isinstance(v, str) else v
            for k, v in ext_cfg['detector_geometry']['beam_center_guess'].items()
        },
        'beamstopper_coordinates': {
            # Mask coordinates MUST be integers for numpy array slicing
            k: {
                sk: (
                    [int(float(x)) for x in str(sv).split(',')]
                    if isinstance(sv, str)
                    else [int(x) for x in sv]
                )
                for sk, sv in (
                    v.items() if isinstance(v, dict) else [('default', v)]
                )
            }
            for k, v in ext_cfg['detector_geometry']['beamstopper_coordinates'].items()
        },
        'transmission_coordinates': ext_cfg['detector_geometry'].get('transmission_coordinates', {}),
        'target_detector_distances': ext_cfg['pipeline_control']['target_detector_distances']
    }
}


def _normalise_beamstop(value):
    """Normalize supported YAML spellings for beamstop selection."""
    return str(value).strip().lower().replace('-', '').replace('_', '').replace(' ', '')


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


def _selected_overview_distance_keys(configuration, result):
    """Return only det_files_* keys requested by target_detector_distances."""
    requested = _requested_detector_distances(
        configuration.get('analysis', {}).get('target_detector_distances', 'all')
    )

    selected = []
    for key in result.get('overview', {}):
        distance = _distance_from_overview_key(key)
        if distance is None:
            continue

        if requested is None or any(abs(distance - target) < 1e-6 for target in requested):
            selected.append(key)

    return selected


def _empty_result_structure():
    """Create the minimum shared result structure used by both pipelines."""
    return {
        'transmission': {},
        'overview': {},
        'integration': {
            'pixel_range_azim': ext_cfg['analysis_flags']['pixel_range_azim'],
            'integration_points': ext_cfg['analysis_flags'].get('integration_points', 120),
            'sectors_nr': ext_cfg['analysis_flags']['sectors_nr']
        }
    }


# Resolve target folder and extract toggle configuration
analysis_folder = create_analysis_folder(configuration)
apply_transmission = configuration['physics_corrections'].get('apply_transmission', False)
result_file = os.path.join(analysis_folder, 'result.npy')
beamstop = _normalise_beamstop(configuration['experiment']['beamstop'])

if beamstop not in {'standard', 'semitransparent'}:
    raise ValueError(
        f"Unsupported beamstop setting '{configuration['experiment']['beamstop']}'. "
        "Expected 'standard' or 'semitransparent'."
    )

should_run_transmission = False


# ==========================================
# STEP 3: MANAGE FLOW & INITIALIZE WORKSPACE
# ==========================================
if os.path.exists(result_file):
    print(f"📦 Loading previously calculated results from: {result_file}")
    with open(result_file, 'rb') as f:
        result = pickle.load(f)

    result.setdefault('transmission', {})
    result.setdefault('overview', {})
    result.setdefault('integration', {})
else:
    # Always create/index the workspace before deciding whether transmission
    # runs. This gives both reduction and semitransparent transmission the same
    # detector-distance inventory.
    print("ℹ️ No existing results file found. Building initial workspace index.")
    result = _empty_result_structure()

    print("Step 0: Indexing files to guarantee the correct results structure...")
    class_files = org.list_files(configuration, result)
    if not class_files and not live_monitoring:
        print("❌ ERROR: No valid HDF5 files found in the raw data directory.")
        sys.exit(1)

    save_results(analysis_folder, result)
    print(f"📂 Saved initial workspace listing structure to: {analysis_folder}")


if not apply_transmission:
    print("🧹 'Apply Transmission' is disabled. Clearing cached transmission data.")
    result['transmission'] = {}

elif beamstop == 'semitransparent':
    # Semitransparent transmission depends directly on the current
    # target_detector_distances selection. Always let caller_transmission
    # rebuild it for the active selection so data from an earlier selection
    # cannot silently be reused.
    print(
        "ℹ️ Semitransparent beamstop: transmission will follow "
        "target_detector_distances."
    )
    should_run_transmission = True

elif not result.get('transmission'):
    # Standard beamstop uses a dedicated trans_dist, so an existing cached
    # transmission result remains valid independently of reduction distance.
    print(
        "⚠️ Standard beamstop: no cached transmission found. "
        "Scheduling calculation at trans_dist."
    )
    should_run_transmission = True


# For semitransparent data, apply the reduction selector before launching the
# transmission subprocess. caller_transmission applies the same selector again
# after re-indexing, making the behavior robust when called independently too.
if apply_transmission and beamstop == 'semitransparent':
    result = org.select_detector_distances(configuration, result)
    save_results(analysis_folder, result)


# ==========================================================
# PRE-REQUISITE: AUTOMATICALLY RUN TRANSMISSION SCRIPT
# ==========================================================
if should_run_transmission and result.get('overview', {}).get('all_files', {}).get('name_hdf'):
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    transmission_script = os.path.join(current_script_dir, "caller_transmission.py")

    if os.path.exists(transmission_script):
        print("\n🚀 [AUTOMATION] Running Transmission pipeline before Radial Integration...")
        try:
            # Build command and pass the .yaml config argument forward if it was provided
            cmd = [sys.executable, transmission_script]
            if len(sys.argv) > 1 and sys.argv[1].endswith('.yaml'):
                cmd.append(os.path.abspath(sys.argv[1]))

            subprocess.run(cmd, check=True)
            print("✅ [AUTOMATION] Transmission completed successfully. Continuing to integration.\n")

            # Load the newly created result.npy containing calculated transmissions
            if os.path.exists(result_file):
                print(f"📦 Loading newly calculated results (with transmissions) from: {result_file}")
                with open(result_file, 'rb') as f:
                    result = pickle.load(f)
            else:
                print(
                    f"❌ [CRITICAL ERROR] Transmission ran successfully, "
                    f"but '{result_file}' was not found!"
                )
                sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(
                f"❌ [CRITICAL ERROR] Transmission run failed "
                f"(Exit Code: {e.returncode}). Radial Integration aborted."
            )
            sys.exit(1)
    else:
        print(
            f"⚠️ [WARNING] '{transmission_script}' not found. "
            "Falling back to existing cached results if possible."
        )
        if result is None:
            print("❌ [CRITICAL ERROR] No results exist and transmission script is missing. Exiting.")
            sys.exit(1)
else:
    if apply_transmission:
        print("\nℹ️ [INFO] Valid standard-beamstop transmission data already exists. Skipping calculation.")
    else:
        print("\nℹ️ [INFO] 'Apply Transmission' is disabled. Skipping Transmission calculation pipeline.")


# ==========================================
# STEP 4: EXECUTION PIPELINE
# ==========================================
iteration = 1
last_raw_signature = None


def _raw_folder_signature():
    """Quickly detect added or still-changing HDF files without opening them."""
    signature = []
    raw_folder = configuration['analysis']['path_hdf_raw']
    for root, _dirs, files in os.walk(raw_folder):
        for filename in files:
            if filename.lower().endswith('.hdf'):
                path = os.path.join(root, filename)
                try:
                    stat = os.stat(path)
                    signature.append((os.path.relpath(path, raw_folder), stat.st_size, stat.st_mtime_ns))
                except OSError:
                    # A file may be in the middle of being created. The next
                    # poll will see it once the filesystem reports it normally.
                    continue
    return tuple(sorted(signature))


def _stop_requested():
    return os.path.exists(stop_file)


def _clear_stop_request():
    try:
        os.remove(stop_file)
    except OSError:
        pass

while True:
    if _stop_requested():
        _clear_stop_request()
        print("⏹ Radial integration stopped by request.")
        break

    pipeline_start_time = time.time()
    print("\n" + "=" * 50)
    print(f"   STARTING RADIAL INTEGRATION (Iteration {iteration})")
    print("=" * 50)

    inventory_changed = False
    if live_monitoring:
        raw_signature = _raw_folder_signature()
        inventory_changed = raw_signature != last_raw_signature
        if inventory_changed:
            last_raw_signature = raw_signature
            print("[LIVE] Raw-data folder changed; refreshing the file index.")
            previous_all_files = result.get('overview', {}).get('all_files', {})
            previous_names = list(previous_all_files.get('name_hdf', []))
            previous_transmissions = previous_all_files.get('transmission')
            class_files = org.list_files(configuration, result)
            new_names = list(class_files.get('name_hdf', []))
            if class_files and apply_transmission and (
                set(new_names) != set(previous_names) or previous_transmissions is None
            ):
                # Use the existing transmission pipeline so new scans receive
                # the same transmission treatment as in a regular run.
                transmission_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "caller_transmission.py")
                cmd = [sys.executable, transmission_script]
                if len(sys.argv) > 1 and sys.argv[1].lower().endswith(('.yaml', '.yml')):
                    cmd.append(os.path.abspath(sys.argv[1]))
                subprocess.run(cmd, check=True)
                with open(result_file, 'rb') as f:
                    result = pickle.load(f)
            elif class_files and apply_transmission and previous_transmissions is not None:
                # Counts can change while a scan is active. Keep the previously
                # calculated transmission values for the same scans; only a
                # changed file inventory needs the transmission stage rerun.
                old_transmission_by_name = dict(zip(previous_names, previous_transmissions))
                class_files['transmission'] = [
                    old_transmission_by_name.get(name, '--') for name in new_names
                ]
                result['overview']['all_files'] = class_files
            elif class_files:
                result = org.select_detector_distances(configuration, result)

        if not result.get('overview', {}).get('all_files', {}).get('name_hdf'):
            print("[LIVE] No valid HDF files yet; waiting for acquisition.")
            time.sleep(max(1, float(ctrl.get('monitor_interval', 60))))
            iteration += 1
            continue

    if ctrl.get('run_reduction', True):
        # Let prepare_input update the result structure, then explicitly filter
        # the overview keys. select_detector_distances() does not necessarily
        # delete keys belonging to detector distances that were not requested.
        result = org.select_detector_distances(configuration, result)

        selected_keys = _selected_overview_distance_keys(configuration, result)
        processed_det_distances = [
            key.replace('det_files_', '') for key in selected_keys
        ]

        requested = configuration['analysis']['target_detector_distances']
        print(f"Requested detector distance(s): {requested}")
        print(
            "Selected detector distance(s): "
            + (", ".join(d.replace('p', '.') for d in processed_det_distances)
               if processed_det_distances else "NONE")
        )

        if not processed_det_distances:
            raise ValueError(
                f"No detector files matched target_detector_distances={requested!r}. "
                f"Available keys: {[k for k in result.get('overview', {}) if k.startswith('det_files_')]}"
            )

        for det_str in processed_det_distances:
            print(f" -> Processing Distance: {det_str.replace('p', '.')}m")
            result = ri.set_integration(configuration, result, det_str)

    # --- STEP 5: PERSIST RESULTS FOR RADIAL INTEGRATION ---
    analysis_folder = create_analysis_folder(configuration)
    save_results(analysis_folder, result)

    # A configured forced run is honored for the initial live pass. Later
    # passes only pick up missing outputs and revisit the newest scan.
    if live_monitoring and configuration['analysis'].get('force_reintegrate', False):
        configuration['analysis']['force_reintegrate'] = False
        print("[LIVE] Initial forced pass complete; later passes will process new scans and revisit the newest scan only.")

    if _stop_requested():
        _clear_stop_request()
        print("⏹ Radial integration stopped by request after the current processing pass.")
        break

    elapsed_time = time.time() - pipeline_start_time
    print(f"\nIteration {iteration} finished in {elapsed_time:.2f} seconds.")

    if not live_monitoring:
        break

    time.sleep(max(1, float(ctrl.get('monitor_interval', 60))))
    iteration += 1

print("\nPipeline Complete.")
