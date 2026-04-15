# -*- coding: utf-8 -*-
import sys
import os
from pathlib import Path


# 1. Get the directory of the current script (darepy/codes/)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Go up one level to find utils.py (in darepy/)
parent_dir = os.path.dirname(current_script_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 3. Now you can safely import utils
from utils import load_config, find_strict_calibration_file
import prepare_input as org

def run_calibration_check(configuration, class_files):
    """
    Validates calibration standards for EVERY detector distance in the dataset.
    Uses utils.find_strict_calibration_file for metadata-aware matching.
    """
    print("\n" + "="*60)
    print("--- Multi-Distance Calibration Validation ---")
    print("="*60)

    cal_setup = configuration['experiment']['calibration']

    # Identify the standards we need to find per distance
    required_ids = {
        'Dark Current (Cd)': cal_setup.get('dark_current'),
        'Water Standard (H2O)': cal_setup.get('water'),
        'Water Cell (EC)': cal_setup.get('water_cell'),
        'Empty Beam (EB)': cal_setup.get('empty_beam')
    }

    # Identify all unique detector distances present in the loaded data
    unique_distances = sorted(list(set(class_files['detx_m'])))

    overall_success = True

    for dist in unique_distances:
        print(f"\n📡 Checking Configuration: {dist}m")
        print("-" * 30)

        # Find the first file index that uses this distance to get target WL and Coll
        sample_idx = class_files['detx_m'].index(dist)
        target_wl = class_files['wl_A'][sample_idx]
        target_coll = class_files['coll_m'][sample_idx]

        print(f"   (Target Profile: {target_wl}Å, {target_coll}m Collimation)")

        dist_missing = []
        for label, identifier in required_ids.items():
            if not identifier:
                continue

            # USE YOUR UTILS FUNCTION: find_strict_calibration_file
            # This checks name, distance, wavelength, and collimation
            filename = find_strict_calibration_file(identifier, sample_idx, class_files)

            if filename:
                print(f"   ✅ {label:22}: Found in {filename}")
            else:
                print(f"   ❌ {label:22}: MISSING for this setup")
                dist_missing.append(identifier)
                overall_success = False

    print("\n" + "="*60)
    if overall_success:
        print("✨ SUCCESS: All configurations have complete calibration sets.")
    else:
        print("⚠️  WARNING: Some distances are missing required calibration files.")
    print("="*60)

    return overall_success

if __name__ == "__main__":
    ext_cfg = load_config()
    p_base = ext_cfg['analysis_paths']['project_base']

    configuration = {
        'experiment': {
            'calibration': ext_cfg.get('calibration_samples', {}),
            'sample_thickness': ext_cfg.get('calibration_samples', {}).get('thickness', {})
        },
        'analysis': {
            'path_dir': str(Path(p_base)),
            'path_hdf_raw': ext_cfg['analysis_paths']['raw_data'],
            'scripts_dir': ext_cfg['analysis_paths']['scripts_dir'],
            'add_id': ext_cfg['analysis_flags'].get('add_id', ''),
            'exclude_files': ext_cfg['analysis_flags'].get('exclude_files', []),
        }
    }

    result = {'overview': {}}
    print("Scanning raw data folder...")
    class_files = org.list_files(configuration, result)

    if class_files:
        run_calibration_check(configuration, class_files)
