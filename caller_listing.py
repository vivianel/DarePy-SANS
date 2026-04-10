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

# Now we can import your tools
from utils import load_config
import prepare_input as org

def run_listing(configuration, result):
    print("\n" + "="*40)
    print("--- Listing and Parsing HDF5 Files ---")
    print("="*40)

    # This calls the library function that was crashing
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
    # We include all keys that prepare_input and utils might look for
    configuration = {
        'experiment': {
            'sample_thickness': ext_cfg.get('calibration_samples', {}).get('thickness', {})
        },
        'analysis': {
            # Critical paths
            'path_dir': str(path_dir),
            'path_hdf_raw': ext_cfg['analysis_paths']['raw_data'],
            'scripts_dir': ext_cfg['analysis_paths']['scripts_dir'],

            # Flags and IDs (The missing 'add_id' is here)
            'add_id': ext_cfg['analysis_flags'].get('add_id', ''),
            'exclude_files': ext_cfg['analysis_flags'].get('exclude_files', []),

            # Integration defaults (Just in case listing looks for them)
            'force_reintegrate': ext_cfg['analysis_flags'].get('force_reintegrate', 0),
        }
    }

    # Initialize result container
    result = {'overview': {}}

    # Run the listing logic
    run_listing(configuration, result)
