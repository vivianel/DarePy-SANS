# -*- coding: utf-8 -*-
import h5py
import numpy as np
import os
import sys
import math
import shutil
from pathlib import Path

# ==========================================
# %% 1. PATH INJECTION
# ==========================================
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent # Points to /darepy/

if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from utils import load_config, parse_scan_list

# Load settings from the YAML path provided by the GUI
config = load_config()

# %% USER INPUT PARAMETERS
path_hdf_raw = config['analysis_paths']['raw_data']
cfg_rename = config['rename_samples']

files_change = parse_scan_list(cfg_rename['files_change'])
subscript = cfg_rename['subscript']
replace_with = cfg_rename['replace_with']
#copy_suffix = cfg_rename['copy_suffix']

# Copy Suffix (NEW)
#    This text will be added to the end of the newly created HDF5 file.
#    Example: 'sans2025n083877.hdf' becomes 'sans2025n083877_mod.hdf'
copy_suffix = '_mod'


# %% SCRIPT LOGIC (No User Modification Needed Below This Line)

files = []
try:
    for r, d, f in os.walk(path_hdf_raw):
        for file in f:
            # Skip files that are ALREADY modified copies to prevent infinite loops
            if file.endswith('.hdf') and copy_suffix not in file:
                files.append(file)

    if not files:
        print(f"Warning: No valid original .hdf files found in '{path_hdf_raw}'. Exiting.")
        sys.exit(0)
except FileNotFoundError:
    print(f"Error: Raw data path '{path_hdf_raw}' not found. Please check the path.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred while listing files: {e}")
    sys.exit(1)


def _load_hdf_string(hdf_file_object, dataset_path):
    """Helper to load string data from HDF5."""
    try:
        data = hdf_file_object[dataset_path]
        if isinstance(data, h5py.Dataset):
            prop = data[()]
            if isinstance(prop, np.bytes_):
                return prop.decode('utf-8').strip()
            elif isinstance(prop, (np.ndarray, str)):
                # If it's an array, flatten it and get the first item
                if isinstance(prop, np.ndarray) and prop.size > 0:
                    prop = prop.flat[0]

                # Decode if it's a byte string inside the array
                if isinstance(prop, (bytes, np.bytes_)):
                    return prop.decode('utf-8').strip()
                return str(prop).strip()
            else:
                return str(prop).strip()
        else:
            return str(data).strip()
    except KeyError:
        return None
    except Exception as e:
        print(f"Warning: Could not load string from '{dataset_path}': {e}")
        return None


for ii in range(0, len(files)):
    file_name = files[ii]
    scan_nr_str = file_name[9:-4]

    try:
        current_scan_nr = int(scan_nr_str)
    except ValueError:
        print(f"Warning: Could not extract integer scan number from file '{file_name}'. Skipping.")
        continue

    process_this_file = False
    if not files_change:
        if replace_with != 0:
            process_this_file = True
    elif current_scan_nr in files_change:
        process_this_file = True

    if process_this_file:
        full_hdf_path = os.path.join(path_hdf_raw, file_name)

        # --- NEW COPY LOGIC ---
        new_file_name = file_name.replace('.hdf', f'{copy_suffix}.hdf')
        new_hdf_path = os.path.join(path_hdf_raw, new_file_name)

        print(f"Copying {file_name} -> {new_file_name}...")
        try:
            shutil.copy2(full_hdf_path, new_hdf_path) # copy2 preserves original creation/modification times
        except Exception as e:
            print(f"Error copying file {file_name}: {e}. Skipping modifications.")
            continue
        # ----------------------

        file_hdf = None

        try:
            # Open the NEW COPY in read/write mode
            file_hdf = h5py.File(new_hdf_path, 'r+')

            current_sample_name = _load_hdf_string(file_hdf, '/entry1/sample/name')
            if current_sample_name is None:
                print(f"Warning: Could not retrieve original sample name for scan {current_scan_nr}.")
                continue

            new_sample_name = current_sample_name

            if subscript == 'temp':
                try:
                    temp_data = file_hdf['/entry1/sample/temperature'][0]
                    if not math.isnan(temp_data):
                        temp_int = int(np.round(temp_data, 0))
                        new_sample_name = f"{new_sample_name}_{temp_int}"
                    else:
                        print(f"Warning: Temperature is NaN for scan {current_scan_nr}.")
                except Exception as e:
                    print(f"Warning: Error reading temperature for scan {current_scan_nr}: {e}.")
            elif isinstance(subscript, str) and subscript != '':
                new_sample_name = f"{new_sample_name}{subscript}"

            if isinstance(replace_with, str) and replace_with != '':
                new_sample_name = replace_with
            elif isinstance(replace_with, int) and replace_with != 0:
                new_sample_name = str(replace_with)

            # --- OVERWRITE THE ORIGINAL NAME FIELD ---
            dataset_path = '/entry1/sample/name'
            if dataset_path in file_hdf:
                del file_hdf[dataset_path] # Delete the old one safely

            # Create the exact same path but format it as a 1D array containing a byte string!
            file_hdf.create_dataset(dataset_path, data=np.array([np.bytes_(new_sample_name)]))

            print(f"  -> Success: '{current_sample_name}' renamed to '{new_sample_name}' in copied file.")

        except FileNotFoundError:
            print(f"Error: HDF5 file '{new_hdf_path}' not accessible.")
        except KeyError as e:
            print(f"Error: Missing HDF5 dataset '{e}' in '{new_hdf_path}'.")
        except Exception as e:
            print(f"An unexpected error occurred processing '{new_hdf_path}': {e}.")
        finally:
            if file_hdf:
                file_hdf.close()
