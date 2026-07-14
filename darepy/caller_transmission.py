# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import os
import pickle  # Added to load/deserialize previously saved results

# 1. Get the directory of the current script (darepy/codes/)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Go up one level to find utils.py (in darepy/)
parent_dir = os.path.dirname(current_script_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


# 2. Point directly to the 'codes' subdirectory where utils.py and backends live
codes_dir = os.path.join(parent_dir, "darepy/codes")

if codes_dir not in sys.path:
    sys.path.insert(0, codes_dir)

# 3. Import utilities and backend logic
from utils import load_config, load_instrument_registry, create_analysis_folder, save_results
import prepare_input as org
from transmission import trans_calc

def run_transmission(configuration, class_files, result, ctrl):
    """Refined execution logic following your original pipeline."""
    # Check the master toggle in pipeline_control
    run_trans = ctrl.get('run_transmission', True)
    if not run_trans:
        print("\n--- Transmission Calculation Skipped (Disabled in YAML) ---")
        return result

    # Get the distance from the transmission_setup
    t_dist = configuration['transmission_setup'].get('dist_trans_measurements', 0)

    # Validate distance and run calculation
    if isinstance(t_dist, (int, float)) and t_dist != 0:
        print("\n" + "*"*50)
        print(f"🚀 STARTING: CALCULATING TRANSMISSIONS")
        print(f"Target Distance: {t_dist}m")
        print("*"*50 + "\n")

        # Capture the updated result dictionary from the backend
        result = trans_calc(configuration, class_files, result)
    else:
        print(f"\n--- Step 3: Skipped. No valid distance provided ({t_dist}m). ---")

    return result

if __name__ == "__main__":
    # --- STEP 1: LOAD CONFIGURATIONS ---
    ext_cfg = load_config()
    INSTRUMENT_REGISTRY = load_instrument_registry()
    selected_inst = ext_cfg['instrument_setup']['which_instrument']
    sample_environment = ext_cfg.get('sample_environment', {})

    # Extract the distance for transmission measurement
    t_dist = ext_cfg.get('transmission_setup', {}).get('dist_trans_measurements', 18)

    # --- STEP 2: CONSTRUCT CONFIGURATION OBJECT ---
    configuration = {
        'instrument': INSTRUMENT_REGISTRY[selected_inst],
        'transmission_setup': ext_cfg.get('transmission_setup', {}),
        'experiment': {
            'sample_environment': sample_environment,
            'calibration': ext_cfg.get('calibration_samples', {}),
            'wl_input': ext_cfg['pipeline_control'].get('wavelength', 'auto'),
            'sample_thickness': ext_cfg.get('calibration_samples', {}).get('thickness', {})
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
            'force_reintegrate': ext_cfg['analysis_flags'].get('force_reintegrate', False)
        }
    }

    # Resolve the destination analysis folder beforehand
    analysis_folder = create_analysis_folder(configuration)

    # --- NEW: Check if results structure already exists and load it ---
    result_file = os.path.join(analysis_folder, 'result.npy')

    if os.path.exists(result_file):
        print(f"📦 Existing results container found! Loading from: {result_file}")
        with open(result_file, 'rb') as f:
            result = pickle.load(f)

        # Ensure crucial structural top-level keys exist in the loaded dict
        if 'transmission' not in result:
            result['transmission'] = {}
        if 'overview' not in result:
            result['overview'] = {}
        if 'integration' not in result:
            result['integration'] = {}

        # Dynamically synchronize active integration configurations on-load
        result['integration']['pixel_range_azim'] = ext_cfg['analysis_flags']['pixel_range_azim']
        result['integration']['integration_points'] = ext_cfg['analysis_flags'].get('integration_points', 120)
        result['integration']['sectors_nr'] = ext_cfg['analysis_flags'].get('sectors_nr', 1)
    else:
        print("ℹ️ No existing results file found. Initializing blank results container.")
        # Fallback empty structure
        result = {
            'transmission': {},
            'overview': {},
            'integration': {
                'pixel_range_azim': ext_cfg['analysis_flags']['pixel_range_azim'],
                'integration_points': ext_cfg['analysis_flags'].get('integration_points', 120),
                'sectors_nr': ext_cfg['analysis_flags'].get('sectors_nr', 1)
            }}

    # --- STEP 3: SINGLE EXECUTION SEQUENCE ---
    print("Step 0: Indexing files (Required for Transmission)...")
    class_files = org.list_files(configuration, result)
    if class_files:
        ctrl = ext_cfg.get('pipeline_control', {})

        # Run calculation and update the result object
        result = run_transmission(configuration, class_files, result, ctrl)

        # --- STEP 4: PERSIST RESULTS FOR RADIAL INTEGRATION ---
        # Save results back to disk utilizing the pre-resolved folder path
        save_results(analysis_folder, result)

        print(f"\n✅ SUCCESS: Transmission calculation complete.")
        print(f"📂 Results saved to: {analysis_folder}")
    else:
        print("❌ ERROR: No valid HDF5 files found in the raw data directory.")
