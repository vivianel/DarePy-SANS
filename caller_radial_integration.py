# -*- coding: utf-8 -*-
"""
DarePy-SANS: Main Orchestration Script
Updated for YAML-based configuration, improved path management, and timing.
"""
import sys
import time
import os
from pathlib import Path

# ==========================================
# %% DYNAMIC PATH INJECTION
# ==========================================
# Safely find the 'codes' folder even when running in Spyder
try:
    current_dir = Path(__file__).resolve().parent
except NameError:
    current_dir = Path(os.getcwd()).resolve()

code_dir = current_dir / 'codes'

if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

# ==========================================
# STEP 0 & 1: LOAD MASTER CONFIGURATIONS
# ==========================================
from utils import load_config, load_instrument_registry

ext_cfg = load_config()
INSTRUMENT_REGISTRY = load_instrument_registry()

# Extract the critical variables needed for validation and imports
selected_inst = ext_cfg['instrument_setup']['which_instrument']
path_dir = Path(ext_cfg['analysis_paths']['project_base'])
# ==========================================
# STEP 2: DRY RUN / PATH VALIDATION
# ==========================================
if ext_cfg.get('pipeline_control', {}).get('dry_run', False):
    print("\n" + "="*40)
    print("      DAREPY-SANS: DRY-RUN MODE")
    print("="*40)

    validation_errors = 0

    # 1. Validate Code Directory
    if not code_dir.exists():
        print(f"[ERROR] Codes directory NOT FOUND: {code_dir}")
        validation_errors += 1
    else:
        print(f"[OK] Codes directory found: {code_dir}")

    # 2. Validate Raw Data Path
    p_raw = Path(ext_cfg['analysis_paths']['raw_data'])
    if not p_raw.exists():
        print(f"[ERROR] Raw data path NOT FOUND: {p_raw}")
        validation_errors += 1
    else:
        print(f"[OK] Raw data path found: {p_raw}")

    # 3. Validate Efficiency Map
    map_name = INSTRUMENT_REGISTRY[selected_inst]['efficiency_map']
    eff_map = code_dir / map_name

    if not eff_map.exists():
        print(f"[ERROR] Efficiency map '{map_name}' NOT FOUND in: {code_dir}")
        validation_errors += 1
    else:
        print(f"[OK] Efficiency map found: {map_name}")

    print("="*40)
    if validation_errors > 0:
        print(f"FAILED: {validation_errors} path/file error(s) found.")
        print("Please check your 'config_experiment.yaml' and file locations.")
        sys.exit(1)
    else:
        print("SUCCESS: All paths and critical files are valid.")
        print("Set 'dry_run: false' in your YAML to begin processing.")
        sys.exit(0)

# ==========================================
# STEP 3: ORCHESTRATION & IMPORTS
# ==========================================
if not code_dir.exists():
    print(f"Error: Codes directory not found at {code_dir}")
    sys.exit(1)

# Add the codes directory to Python's path so it can find your modules
sys.path.append(str(code_dir))

# Now we can import your modules safely
import prepare_input as org
from transmission import trans_calc
import integration as ri

# ==========================================
# STEP 4: CONSTRUCT CONFIGURATION OBJECT
# ==========================================
configuration = {
    'instrument': INSTRUMENT_REGISTRY[selected_inst],
    'experiment': {
        'calibration': ext_cfg['calibration_samples'],
        'wl_input': ext_cfg['instrument_setup']['wavelength']
    },
    'physics_corrections': ext_cfg.get('physics_corrections', {}), # <-- New Pluggable Physics
    'analysis': {
        'path_dir': str(path_dir),
        'path_hdf_raw': ext_cfg['analysis_paths']['raw_data'],
        'scripts_dir': ext_cfg['analysis_paths']['scripts_dir'],
        'exclude_files': ext_cfg['analysis_flags'].get('exclude_files', []),
        'force_reintegrate': ext_cfg['analysis_flags']['force_reintegrate'],
        'plot_azimuthal': ext_cfg['analysis_flags']['plot_azimuthal'],
        'plot_radial': ext_cfg['analysis_flags']['plot_radial'],
        'add_id': ext_cfg['analysis_flags'].get('add_id', ''),
        'save_azimuthal': ext_cfg['analysis_flags']['save_azimuthal'],
        'save_2d_patterns': ext_cfg['analysis_flags']['save_2d_patterns'],
        'beam_center_guess': ext_cfg['detector_geometry']['beam_center_guess'],
        'beamstopper_coordinates': ext_cfg['detector_geometry']['beamstopper_coordinates'],
        'transmission_coordinates': ext_cfg['detector_geometry']['transmission_coordinates'],
        'target_detector_distances': ext_cfg['instrument_setup']['target_detector_distances']
    }
}

# Initialize result container
result = {
    'transmission': {},
    'overview': {},
    'integration': {
        'pixel_range_azim': range(*ext_cfg['analysis_flags']['pixel_range_azim']),
        'integration_points': ext_cfg['analysis_flags']['integration_points'],
        'sectors_nr': ext_cfg['analysis_flags']['sectors_nr']
    }
}

# ==========================================
# STEP 5: EXECUTION PIPELINE (WITH LIVE MONITORING)
# ==========================================
ctrl = ext_cfg.get('pipeline_control', {})
live_mode = ctrl.get('live_monitoring', False)
monitor_interval = ctrl.get('monitor_interval', 60)

# We define a loop counter to track the iterations
iteration = 1

while True:
    # --- START TIMING ---
    pipeline_start_time = time.time()
    print("\n" + "="*50)
    print(f"   STARTING DAREPY-SANS PIPELINE (Iteration {iteration})")
    print("="*50)

    # 1. Load files
    if ctrl.get('run_file_listing', True):
        print("\n--- Step 1: Listing and Parsing HDF5 Files ---")
        class_files = org.list_files(configuration, result)
    else:
        print("\nSkipping File Listing. (Warning: Subsequent steps may fail)")
        class_files = {}

    # 2. Transmission
    trans_dist_val = configuration['physics_corrections']['transmission_dist']

    # Check if it's a number and greater than 0
    if isinstance(trans_dist_val, (int, float)) and trans_dist_val > 0:
        print(f"\n--- Step 2: Calculating Transmissions (Distance: {trans_dist_val}m) ---")
        result = trans_calc(configuration, class_files, result)
    elif isinstance(trans_dist_val, (int, float)) and trans_dist_val < 0:
        print(f"\n--- Step 2: SANS-LLB Mode (Transmission will be calculated on-the-fly) ---")
        # For LLB, we still might need to prepare the EB references
        result = trans_calc(configuration, class_files, result)
    else:
        print("\n--- Step 2: Transmission Calculation Skipped ---")

    # 3: Data reduction (organize and integrate)
    if ctrl.get('run_reduction', True):
        print("\n--- Step 3a: Organizing Files by Detector Distance ---")
        result = org.select_detector_distances(configuration, class_files, result)

        print("\n--- Step 3b: Running Radial/Azimuthal Integration ---")
        target_dist = configuration['analysis']['target_detector_distances']

        if target_dist == 'all':
            processed_det_distances = [
                k.replace('det_files_', '') for k in result['overview'].keys() if k.startswith('det_files_')
            ]
        else:
            processed_det_distances = [str(d).replace('.', 'p') for d in target_dist]

        if not processed_det_distances:
            print("[WARNING] No detector distances were found to process! Check your raw data.")
        else:
            for det_str in processed_det_distances:
                print(f"\n--- Processing Detector Distance: {det_str.replace('p', '.')}m ---")
                result = ri.set_integration(configuration, result, det_str)
    else:
        print("\n--- Step 3: Data Reduction Skipped by User in YAML ---")

    # --- END TIMING & REPORT ---
    pipeline_end_time = time.time()
    elapsed_time = pipeline_end_time - pipeline_start_time
    print("\n" + "-"*50)
    print(f" Iteration {iteration} finished in {elapsed_time:.2f} seconds.")
    print("-"*50)

    # ==========================================
    # LIVE MONITORING LOGIC
    # ==========================================
    if not live_mode:
        print("Live monitoring is disabled. Pipeline complete.")
        break  # Exit the loop

    print(f"\n[LIVE MODE] Waiting {monitor_interval} seconds for new data...")
    time.sleep(monitor_interval)

    # PREPARE FOR NEXT ITERATION
    # 1. Turn off 'force_reintegrate' so we only process new files
    configuration['analysis']['force_reintegrate'] = 0

    # 2. Identify the most recent HDF5 file in the raw directory
    p_raw = Path(configuration['analysis']['path_hdf_raw'])
    all_raw_files = list(p_raw.glob('*.hdf*')) # Catch .hdf or .hdf5

    if all_raw_files:
        # Sort files by modification time (newest last)
        all_raw_files.sort(key=lambda x: x.stat().st_mtime)
        last_file = all_raw_files[-1]

        print(f"[LIVE MODE] Forcing re-integration of the last active file:\n -> {last_file.name}")

        # We need to pass this specific file to `integration.py` so it knows to override
        # the `force_reintegrate = 0` rule for this file ONLY.
        # We store it in the configuration so Step 3b can access it.
        configuration['analysis']['force_last_file'] = last_file.name

    iteration += 1

# ==========================================
# PIPELINE COMPLETE & TIMING REPORT
# ==========================================
# --- END TIMING ---
pipeline_end_time = time.time()
elapsed_time = pipeline_end_time - pipeline_start_time

print("\n" + "="*40)
print(" DarePy-SANS run finished successfully.")
print(f" Total Execution Time: {elapsed_time:.2f} seconds")
print("="*40 + "\n")
