# -*- coding: utf-8 -*-
import sys
from pathlib import Path

# ==========================================
# %% DYNAMIC PATH INJECTION
# ==========================================
current_dir = Path(__file__).resolve().parent
code_dir = current_dir / 'codes'

if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from utils import load_config, load_instrument_registry
import prepare_input as org
from transmission import trans_calc

def run_transmission(configuration, class_files, result, ctrl):
    """Refined execution logic following your original pipeline."""
    # 1. Check the master toggle
    run_trans = ctrl.get('run_transmission', True)
    if not run_trans:
        print("\n--- Transmission Calculation Skipped (Disabled in YAML) ---")
        return result

    # 2. Get the distance from the new 'transmission_setup'
    t_dist = configuration['transmission_setup'].get('transmission_dist', 0)

    # 3. Your original logic: Check if dist > 0 or dist < 0 (LLB mode)
    if isinstance(t_dist, (int, float)) and t_dist != 0:
        print("\n" + "*"*50)
        print(f"🚀 STARTING: CALCULATING TRANSMISSIONS")
        print(f"Target Distance: {t_dist}m")
        print("*"*50 + "\n")

        # This is the actual calculation call
        result = trans_calc(configuration, class_files, result)
    else:
        print(f"\n--- Step 3: Skipped. No valid distance provided ({t_dist}m). ---")

    return result

if __name__ == "__main__":
    ext_cfg = load_config()
    INSTRUMENT_REGISTRY = load_instrument_registry()
    selected_inst = ext_cfg['instrument_setup']['which_instrument']

    # Extract the distance from your new YAML field
    t_dist = ext_cfg.get('transmission_setup', {}).get('transmission_dist', 18)

    # --- THE ROBUST CONFIGURATION ---
    configuration = {
        'instrument': INSTRUMENT_REGISTRY[selected_inst],
        'transmission_setup': ext_cfg.get('transmission_setup', {}),
        'experiment': {
            'calibration': ext_cfg.get('calibration_samples', {}),
            'wl_input': ext_cfg['physics_corrections'].get('wavelength', 'auto'),
            'sample_thickness': ext_cfg.get('calibration_samples', {}).get('thickness', {})
        },
        'physics_corrections': {
            # We put it here too so the backend trans_calc() finds it!
            **ext_cfg.get('physics_corrections', {}),
            'transmission_dist': t_dist
        },
        'analysis': {
            'path_dir': str(Path(ext_cfg['analysis_paths']['project_base'])),
            'path_hdf_raw': ext_cfg['analysis_paths']['raw_data'],
            'scripts_dir': ext_cfg['analysis_paths']['scripts_dir'],
            'add_id': ext_cfg['analysis_flags'].get('add_id', ''),
            'exclude_files': ext_cfg['analysis_flags'].get('exclude_files', []),
            'transmission_coordinates': ext_cfg['detector_geometry'].get('transmission_coordinates', {}),
            'force_reintegrate': ext_cfg['analysis_flags'].get('force_reintegrate', False)
        }
    }

    result = {
        'overview': {},
        'transmission': {},
        'integration': {'integration_points': 120, 'sectors_nr': 1}
    }

    print("Step 0: Indexing files (Required for Transmission)...")
    class_files = org.list_files(configuration, result)

    if class_files:
        run_transmission(configuration, class_files, result, ext_cfg.get('pipeline_control', {}))

    # 4. Step 1: Index files
    print("Step 0: Indexing files...")
    class_files = org.list_files(configuration, result)
    ctrl = ext_cfg.get('pipeline_control', {})
    # 5. Step 3: Run the calculation
    if class_files:
        result = run_transmission(configuration, class_files, result, ctrl)

        # ENSURE DATA IS PERSISTED
        from utils import create_analysis_folder, save_results
        analysis_folder = create_analysis_folder(configuration)
        save_results(analysis_folder, result)
        print(f"✅ Transmission calculation complete. Results saved to {analysis_folder}")
