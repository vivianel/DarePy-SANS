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
# Identify the analysis folder to find saved results
analysis_folder = create_analysis_folder({
    'analysis': {
        'add_id': ext_cfg['analysis_flags'].get('add_id', ''),
        'path_dir': ext_cfg['analysis_paths']['project_base']
    }
})

# ==========================================================
# %% Pre-requisite: Automatically run the Transmission script
# ==========================================================
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
    except subprocess.CalledProcessError as e:
        print(f"❌ [CRITICAL ERROR] Transmission run failed (Exit Code: {e.returncode}). Radial Integration aborted.")
        sys.exit(1)
else:
    print(f"⚠️ [WARNING] '{transmission_script}' not found. Falling back to existing cached results.")

# ==========================================
# STEP 2: LOAD PREVIOUS RESULTS (Step 3 Output)
# ==========================================
result_file = os.path.join(analysis_folder, 'result.npy')
if os.path.exists(result_file):
    print(f"📦 Loading previously calculated results (Transmissions) from: {result_file}")
    with open(result_file, 'rb') as f:
        result = pickle.load(f)
else:
    print("⚠️ [ERROR] No existing results found. Initializing blank results container.")
    sys.exit(1)


# ==========================================
# STEP 3: CONSTRUCT CONFIGURATION OBJECT
# ==========================================
configuration = {
    'instrument': INSTRUMENT_REGISTRY[selected_inst],
    'experiment': {
        'calibration': ext_cfg['calibration_samples'],
        'wl_input': ext_cfg['pipeline_control']['wavelength'],
        'sample_thickness': ext_cfg.get('calibration_samples', {}).get('thickness', {}),
        # FIX: Map the transmission distance to where the backend expects it
        'trans_dist': ext_cfg.get('transmission_setup', {}).get('dist_trans_measurements', 18),
        'sample_environment': sample_environment,
        'resolution_settings':ext_cfg['resolution_settings']
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

    # 1. Silent File Discovery
    class_files = org.list_files(configuration, result)

    # --- THE CRITICAL FIX: INJECT TRANSMISSIONS INTO METADATA ---
    # --- METADATA INJECTION ---
    if os.path.exists(result_file):
        if 'transmission' in result:
            with open(result_file, 'rb') as f:
                saved_data = pickle.load(f)
                saved_trans = saved_data.get('transmission', {})
                result['transmission'].update(saved_trans)

                if class_files and saved_trans:
                    trans_column = [saved_trans.get(hdf, 1.0) for hdf in class_files['name_hdf']]
                    class_files['transmission'] = trans_column
                    result['overview']['transmission'] = trans_column
                    print(f"🔗 Linked saved transmissions to file list.")

    # --- THE NEW VALIDATION GUARD ---
    apply_t = configuration['physics_corrections'].get('apply_transmission', False)
    has_t_data = 'transmission' in class_files if class_files else False

    if apply_t and not has_t_data:
        print("\n" + "!"*60)
        print("🚨 ERROR: 'Apply Transmission' is ENABLED but no data was found!")
        print("   -> You MUST run 'caller_transmission' first.")
        print("!"*60 + "\n")
        sys.exit(1)

    if not class_files:
        break

    # 2. Integration Loop
    if ctrl.get('run_reduction', True):
        result = org.select_detector_distances(configuration, class_files, result)
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
            processed_det_distances = [ k.replace('det_files_', '') for k in result['overview'].keys()
    if k.startswith('det_files_') and str(float(k.replace('det_files_', '').replace('p', '.'))) in target_strings]


        for det_str in processed_det_distances:
            print(f" -> Processing Distance: {det_str.replace('p', '.')}m")
            # This will now find the loaded transmission values in 'result'
            result = ri.set_integration(configuration, result, det_str)

    # --- STEP 4: PERSIST RESULTS FOR RADIAL INTEGRATION ---
    # This creates the 'analysis' folder and saves result.npy
    analysis_folder = create_analysis_folder(configuration)
    save_results(analysis_folder, result)

    elapsed_time = time.time() - pipeline_start_time
    print(f"\nIteration {iteration} finished in {elapsed_time:.2f} seconds.")

    if not ext_cfg.get('pipeline_control', {}).get('live_monitoring', False):
        break
    time.sleep(ext_cfg.get('pipeline_control', {}).get('monitor_interval', 60))
    iteration += 1

print("\nPipeline Complete.")
