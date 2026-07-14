# -*- coding: utf-8 -*-
"""
DarePy-SANS: Main Orchestration Script (Integration Only)
Fixed: Loads previous results and maps transmission distance correctly.
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

# 2. Point directly to the 'codes' subdirectory where utils.py and backends live
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


# ==========================================
# STEP 2: CONSTRUCT CONFIGURATION OBJECT
# ==========================================
configuration = {
    'instrument': INSTRUMENT_REGISTRY[selected_inst],
    'experiment': {
        'calibration': ext_cfg['calibration_samples'],
        'wl_input': ext_cfg['pipeline_control']['wavelength'],
        'sample_thickness': ext_cfg.get('calibration_samples', {}).get('thickness', {}),
        'beamstop': ext_cfg.get('transmission_setup', {}).get('beamstop', 'semitransparent'),
        'trans_dist': ext_cfg.get('transmission_setup', {}).get('dist_trans_measurements', 18),
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
        'add_plot_azimuthal': ext_cfg['analysis_flags']['add_plot_azimuthal'],
        'save_plot_radial': ext_cfg['analysis_flags']['save_plot_radial'],
        'save_data_azimuthal': ext_cfg['analysis_flags']['save_data_azimuthal'],
        'save_2d_patterns': ext_cfg['analysis_flags']['save_2d_patterns'],
        'beam_center_guess': {
            # Beam centers MUST be floats for sub-pixel accuracy in pyFAI
            k: [float(x) for x in str(v).split(',')] if isinstance(v, str) else v
            for k, v in ext_cfg['detector_geometry']['beam_center_guess'].items()
        },
        'beamstopper_coordinates': {
            # Mask coordinates MUST be integers for numpy array slicing
            k: {sk: ([int(float(x)) for x in str(sv).split(',')] if isinstance(sv, str) else [int(x) for x in sv])
                for sk, sv in (v.items() if isinstance(v, dict) else [('default', v)])}
            for k, v in ext_cfg['detector_geometry']['beamstopper_coordinates'].items()
        },
        'transmission_coordinates': ext_cfg['detector_geometry'].get('transmission_coordinates', {}),
        'target_detector_distances': ext_cfg['pipeline_control']['target_detector_distances']
    }
}

# Resolve target folder and extract toggle configuration
analysis_folder = create_analysis_folder(configuration)
apply_transmission = configuration['physics_corrections'].get('apply_transmission', False)
result_file = os.path.join(analysis_folder, 'result.npy')

# Flag to determine whether we need to trigger transmission calculations
should_run_transmission = False
result = None


# ==========================================
# STEP 3: MANAGE flow & INITIALIZE WORKSPACE
# ==========================================
if os.path.exists(result_file):
    print(f"📦 Loading previously calculated results from: {result_file}")
    with open(result_file, 'rb') as f:
        result = pickle.load(f)

    # If results exist, but do not have results['transmission'] populated, trigger calculation in case apply_transmission is True
    if apply_transmission and (not result.get('transmission') or len(result.get('transmission', {})) == 0):
        print("⚠️ Results exist, but do not contain 'transmission' data. Triggering automated calculation.")
        should_run_transmission = True
else:
    if apply_transmission:
        print("ℹ️ No existing results file found. Transmission is requested; scheduling transmission calculation first.")
        should_run_transmission = True
    else:
        # Transmission is not requested, and results do not exist. Create list structure directly.
        print("ℹ️ No existing results file found and Transmission is disabled. Building empty structure directly via index listing.")
        result = {
            'transmission': {},
            'overview': {},
            'integration': {
                'pixel_range_azim': ext_cfg['analysis_flags']['pixel_range_azim'],
                'integration_points': ext_cfg['analysis_flags'].get('integration_points', 120),
                'sectors_nr': ext_cfg['analysis_flags'].get('sectors_nr', 1)
            }
        }
        print("Step 0: Indexing files to guarantee the correct results structure...")
        class_files = org.list_files(configuration, result)
        if class_files:
            # Save this structural foundation to disk so next steps can safely access it
            save_results(analysis_folder, result)
            print(f"📂 Saved initial workspace listing structure to: {analysis_folder}")
        else:
            print("❌ ERROR: No valid HDF5 files found in the raw data directory.")
            sys.exit(1)


# ==========================================================
# %% Pre-requisite: Automatically run the Transmission script
# ==========================================================
if should_run_transmission:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    transmission_script = os.path.join(current_script_dir, "caller_transmission.py")

    if os.path.exists(transmission_script):
        print("\n🚀 [AUTOMATION] Running Transmission pipeline before Radial Integration...")
        try:
            # Build command and pass the .yaml config argument forward if it was provided
            cmd = [sys.executable, transmission_script]
            if len(sys.argv) > 1 and sys.argv[1].endswith('.yaml'):
                cmd.append(os.path.abspath(sys.argv[1]))

            # check=True will halt execution here if caller_transmission crashes
            subprocess.run(cmd, check=True)
            print("✅ [AUTOMATION] Transmission completed successfully. Continuing to integration.\n")

            # Now load the newly created result.npy file containing the calculated transmissions
            if os.path.exists(result_file):
                print(f"📦 Loading newly calculated results (with transmissions) from: {result_file}")
                with open(result_file, 'rb') as f:
                    result = pickle.load(f)
            else:
                print(f"❌ [CRITICAL ERROR] Transmission ran successfully, but '{result_file}' was not found!")
                sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"❌ [CRITICAL ERROR] Transmission run failed (Exit Code: {e.returncode}). Radial Integration aborted.")
            sys.exit(1)
    else:
        print(f"⚠️ [WARNING] '{transmission_script}' not found. Falling back to existing cached results if possible.")
        if result is None:
            print("❌ [CRITICAL ERROR] No results exist and transmission script is missing. Exiting.")
            sys.exit(1)
else:
    if apply_transmission:
        print("\nℹ️ [INFO] Transmission data already exists. Skipping automated transmission calculation.")
    else:
        print("\nℹ️ [INFO] 'Apply Transmission' is disabled. Skipping Transmission calculation pipeline.")


# ==========================================
# STEP 4: EXECUTION PIPELINE
# ==========================================
ctrl = ext_cfg.get('pipeline_control', {})
iteration = 1

while True:
    pipeline_start_time = time.time()
    print("\n" + "="*50)
    print(f"   STARTING RADIAL INTEGRATION (Iteration {iteration})")
    print("="*50)


    # 2. Integration Loop
    if ctrl.get('run_reduction', True):
        result = org.select_detector_distances(configuration, result)
        target_dist = configuration['analysis']['target_detector_distances']

        if target_dist == 'all' or target_dist == '':
            processed_det_distances = [
                k.replace('det_files_', '') for k in result['overview'].keys() if k.startswith('det_files_')
            ]
        elif isinstance(target_dist, int) or isinstance(target_dist, float):
            target_str = str(float(target_dist)).replace('.', 'p')
            processed_det_distances = [k.replace('det_files_', '') for k in result['overview'].keys()
                                       if k.startswith('det_files_') and k.endswith(target_str)]
        else:
            target_strings = [str(float(d)) for d in target_dist]
            processed_det_distances = [
                k.replace('det_files_', '') for k in result['overview'].keys()
                if k.startswith('det_files_') and str(float(k.replace('det_files_', '').replace('p', '.'))) in target_strings
            ]

        for det_str in processed_det_distances:
            print(f" -> Processing Distance: {det_str.replace('p', '.')}m")
            result = ri.set_integration(configuration, result, det_str)

    # --- STEP 5: PERSIST RESULTS FOR RADIAL INTEGRATION ---
    analysis_folder = create_analysis_folder(configuration)
    save_results(analysis_folder, result)

    elapsed_time = time.time() - pipeline_start_time
    print(f"\nIteration {iteration} finished in {elapsed_time:.2f} seconds.")

    if not ext_cfg.get('pipeline_control', {}).get('live_monitoring', False):
        break
    time.sleep(ext_cfg.get('pipeline_control', {}).get('monitor_interval', 60))
    iteration += 1

print("\nPipeline Complete.")
