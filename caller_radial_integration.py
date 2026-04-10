# -*- coding: utf-8 -*-
"""
DarePy-SANS: Main Orchestration Script (Integration Only)
Fixed: Loads previous results and maps transmission distance correctly.
"""
import sys
import time
import os
import pickle
from pathlib import Path

# ==========================================
# %% DYNAMIC PATH INJECTION
# ==========================================
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
from utils import load_config, load_instrument_registry, create_analysis_folder
import prepare_input as org
import integration as ri

ext_cfg = load_config()
INSTRUMENT_REGISTRY = load_instrument_registry()

selected_inst = ext_cfg['instrument_setup']['which_instrument']
# Identify the analysis folder to find saved results
analysis_folder = create_analysis_folder({
    'analysis': {
        'add_id': ext_cfg['analysis_flags'].get('add_id', ''),
        'path_dir': ext_cfg['analysis_paths']['project_base']
    }
})

# ==========================================
# STEP 2: LOAD PREVIOUS RESULTS (Step 3 Output)
# ==========================================
result_file = os.path.join(analysis_folder, 'result.npy')

if os.path.exists(result_file):
    print(f"📦 Loading previously calculated results (Transmissions) from: {result_file}")
    with open(result_file, 'rb') as f:
        result = pickle.load(f)
else:
    print("⚠️ No existing results found. Initializing blank results container.")
    result = {
        'transmission': {},
        'overview': {},
        'integration': {
            'pixel_range_azim': ext_cfg['analysis_flags']['pixel_range_azim'],
            'integration_points': ext_cfg['analysis_flags'].get('integration_points', 120),
            'sectors_nr': ext_cfg['analysis_flags'].get('sectors_nr', 1)
        }
    }

# ==========================================
# STEP 3: CONSTRUCT CONFIGURATION OBJECT
# ==========================================
configuration = {
    'instrument': INSTRUMENT_REGISTRY[selected_inst],
    'experiment': {
        'calibration': ext_cfg['calibration_samples'],
        'wl_input': ext_cfg['physics_corrections']['wavelength'],
        'sample_thickness': ext_cfg.get('calibration_samples', {}).get('thickness', {}),
        # FIX: Map the transmission distance to where the backend expects it
        'trans_dist': ext_cfg.get('transmission_setup', {}).get('transmission_dist', 18)
    },
    'physics_corrections': ext_cfg.get('physics_corrections', {}),
    'analysis': {
        'path_dir': ext_cfg['analysis_paths']['project_base'],
        'path_hdf_raw': ext_cfg['analysis_paths']['raw_data'],
        'scripts_dir': ext_cfg['analysis_paths']['scripts_dir'],
        'add_id': ext_cfg['analysis_flags'].get('add_id', ''),
        'exclude_files': ext_cfg['analysis_flags'].get('exclude_files', []),
        'force_reintegrate': ext_cfg['analysis_flags']['force_reintegrate'],
        'save_plot_azimuthal': ext_cfg['analysis_flags']['save_plot_azimuthal'],
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
                for sk, sv in v.items()}
            for k, v in ext_cfg['detector_geometry']['beamstopper_coordinates'].items()
        },
        'transmission_coordinates': ext_cfg['detector_geometry'].get('transmission_coordinates', {}),
        'target_detector_distances': ext_cfg['physics_corrections']['target_detector_distances']
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

        if target_dist == 'all':
            processed_det_distances = [
                k.replace('det_files_', '') for k in result['overview'].keys() if k.startswith('det_files_')
            ]
        else:
            processed_det_distances = [str(d).replace('.', 'p') for d in target_dist]

        for det_str in processed_det_distances:
            print(f" -> Processing Distance: {det_str.replace('p', '.')}m")
            # This will now find the loaded transmission values in 'result'
            result = ri.set_integration(configuration, result, det_str)

    elapsed_time = time.time() - pipeline_start_time
    print(f"\nIteration {iteration} finished in {elapsed_time:.2f} seconds.")

    if not ext_cfg.get('pipeline_control', {}).get('live_monitoring', False):
        break
    time.sleep(ext_cfg.get('pipeline_control', {}).get('monitor_interval', 60))
    iteration += 1

print("\nPipeline Complete.")
