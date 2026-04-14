# -*- coding: utf-8 -*-
import sys
import os
from pathlib import Path

# ==========================================
# %% DYNAMIC PATH INJECTION
# ==========================================
current_dir = Path(__file__).resolve().parent
code_dir = current_dir / 'codes'

if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

# Now we can import your tools
from utils import load_config, load_instrument_registry
import prepare_input as org

def run_path_validation(configuration, ext_cfg):
    """
    Performs the Dry Run validation logic to ensure all paths
    and critical files defined in Global Setup exist.
    """
    print("\n" + "="*40)
    print("      DAREPY-SANS: DRY-RUN VALIDATION")
    print("="*40)

    errors = 0
    # 1. Validate Scripts/Codes Directory
    c_dir = Path(configuration['analysis']['scripts_dir'])
    if not c_dir.exists():
        print(f"❌ [ERROR] Scripts directory NOT FOUND: {c_dir}")
        errors += 1
    else:
        print(f"✅ [OK] Scripts directory found.")

    # 2. Validate Raw Data Path
    p_raw = Path(configuration['analysis']['path_hdf_raw'])
    if not p_raw.exists():
        print(f"❌ [ERROR] Raw data path NOT FOUND: {p_raw}")
        errors += 1
    else:
        print(f"✅ [OK] Raw data path found.")

    # 3. Validate Efficiency Map from Registry
    inst = ext_cfg['instrument_setup']['which_instrument']
    registry = load_instrument_registry()

    # Get the filename from the registry and check the codes folder
    map_name = registry.get(inst, {}).get('efficiency_map', 'none')
    eff_path = c_dir / 'codes' / map_name

    if not eff_path.exists():
        print(f"❌ [ERROR] Efficiency map '{map_name}' NOT FOUND in codes folder.")
        errors += 1
    else:
        print(f"✅ [OK] Efficiency map verified: {map_name}")

    print("="*40)
    return errors == 0

def run_listing(configuration, result):
    print("\n" + "="*40)
    print("--- Listing and Parsing HDF5 Files ---")
    print("="*40)

    class_files = org.list_files(configuration, result)

    if not class_files or not class_files.get('name_hdf'):
        print("\n[FATAL ERROR] No valid HDF5 files found.")
        return None

    print(f"✅ SUCCESS: Indexed {len(class_files.get('name_hdf', []))} files.")
    return class_files

if __name__ == "__main__":
    # Load the latest GUI settings
    ext_cfg = load_config()

    # Define the project path
    p_base = ext_cfg['analysis_paths']['project_base']
    path_dir = Path(p_base)

    # --- THE ROBUST CONFIGURATION ---
    configuration = {
        'experiment': {
            'sample_thickness': ext_cfg.get('calibration_samples', {}).get('thickness', {})
        },
        'analysis': {
            'path_dir': str(path_dir),
            'path_hdf_raw': ext_cfg['analysis_paths']['raw_data'],
            'scripts_dir': ext_cfg['analysis_paths']['scripts_dir'],
            'add_id': ext_cfg['analysis_flags'].get('add_id', ''),
            'exclude_files': ext_cfg['analysis_flags'].get('exclude_files', []),
            'force_reintegrate': ext_cfg['analysis_flags'].get('force_reintegrate', 0),
        }
    }

    # --- DRY RUN LOGIC ---
    # Lookup dry_run from analysis_paths
    is_dry_run = ext_cfg.get('analysis_paths', {}).get('dry_run', False)

    if is_dry_run:
        success = run_path_validation(configuration, ext_cfg)
        if not success:
            # STOP if paths are wrong
            print("\n🛑 Dry Run Failed. Please fix your paths in Tab 1 and try again.")
            sys.exit(1)
        else:
            # KEEP GOING if paths are correct
            print("\n✨ Dry Run Successful! Path validation passed. Proceeding to scan...")

    # Initialize result container
    result = {'overview': {}}

    # This will now run regardless of whether Dry Run was checked,
    # as long as the validation didn't fail.
    run_listing(configuration, result)
