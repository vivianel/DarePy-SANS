"""
Rename sample within SANS HDF5 raw data files.

It allows users to batch-rename samples based on scan numbers, append a temperature
subscript, or replace the sample name entirely. This is particularly useful for
correcting metadata directly in the HDF5 files.

Created on Mon Jul 18 12:37:26 2022

@author: lutzbueno_v
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

class Action:
    priority = 50

    @classmethod
    def get_parser(cls) -> ArgumentParser:
        parser = ArgumentParser(prog='darepy bsc', formatter_class=ArgumentDefaultsHelpFormatter,
                                description='Determing beamstopper center and masks interactively')
        parser.add_argument('replace_with',
                            help='Sample name to put into the data files')
        parser.add_argument('file_index', type=int, default=[], nargs='*',
                            help='Files to Change (Scan Numbers), only these scan files are modify.'
                                 'If empty, all files will be processed')
        parser.add_argument('-s', '--subscript', dest='subscript', default='',
                            help='Append a suffix to all sample names based on a measurement parameter'
                                 'provided here')

        return parser

    def __init__(self, arguments):
        self.arguments = arguments

    def run(self):
        from ..config import cfg

        import h5py
        import numpy as np
        import os
        import sys # Added for controlled exit in case of critical errors
        import math # Added for math.isnan

        path_hdf_raw = cfg.experiment.path_hdf_raw

        files_change = self.arguments.file_index

        subscript = self.arguments.subscript

        replace_with = self.arguments.replace_with

        # %% SCRIPT LOGIC (No User Modification Needed Below This Line)

        # List all HDF5 files in the specified raw data directory
        files = []
        try:
            for r, d, f in os.walk(path_hdf_raw):
                for file in f:
                    if file.endswith('.hdf'): # Use endswith for more robustness
                        files.append(os.path.join(file))
            if not files:
                print(f"Warning: No .hdf files found in '{path_hdf_raw}'. Exiting.")
                sys.exit(0) # Exit gracefully if no files found
        except FileNotFoundError:
            print(f"Error: Raw data path '{path_hdf_raw}' not found. Please check the path.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred while listing files in '{path_hdf_raw}': {e}")
            sys.exit(1)


        # Function to robustly load string properties (like sample name) from HDF5
        def _load_hdf_string(hdf_file_object, dataset_path):
            """
            Helper to load string data from HDF5, handling bytes and scalar numpy arrays.
            """
            try:
                data = hdf_file_object[dataset_path]
                if isinstance(data, h5py.Dataset):
                    prop = data[()] # Read the actual data from the dataset
                    if isinstance(prop, np.bytes_):
                        return prop.decode('utf-8').strip()
                    elif isinstance(prop, (np.ndarray, str)): # Handle case where it's already a string or 0-D array
                        return str(prop).strip()
                    else:
                        return str(prop).strip()
                else:
                    return str(data).strip() # Fallback for non-dataset objects
            except KeyError:
                return None # Dataset not found
            except Exception as e:
                print(f"Warning: Could not load string from '{dataset_path}': {e}")
                return None


        # Iterate through each identified HDF5 file
        for file_name in files:
            scan_nr_str = file_name.rsplit('n',1)[1].split('.',1)[0] # Assumes 'InstrYYYYnXXXXXX.hdf' format based on error message

            # Validate scan number extraction
            try:
                current_scan_nr = int(scan_nr_str)
            except ValueError:
                print(f"Warning: Could not extract integer scan number from file '{file_name}'. Skipping this file.")
                continue # Skip to the next file

            # Determine if the current file should be processed based on 'files_change' list
            process_this_file = False
            if not files_change: # If files_change is empty, process all files (if replace_with is not 0)
                if replace_with != 0: # Check the value, not just if it's empty
                    process_this_file = True
            elif current_scan_nr in files_change: # If files_change is not empty, process only specified scans
                process_this_file = True

            if process_this_file:
                print(file_name)

                full_hdf_path = os.path.join(path_hdf_raw, file_name)
                file_hdf = None # Initialize file_hdf outside try block

                try:
                    # Open the HDF5 file in read/write mode ('r+')
                    file_hdf = h5py.File(full_hdf_path, 'r+')

                    # Get the existing sample name from the HDF5 file using the helper function
                    current_sample_name = _load_hdf_string(file_hdf, '/entry1/sample/name')
                    if current_sample_name is None:
                        print(f"Warning: Could not retrieve original sample name for scan {current_scan_nr}. Skipping this file.")
                        continue # Skip if original name is not found

                    new_sample_name = current_sample_name # Start with current name

                    # Apply subscript if specified
                    if subscript == 'temp':
                        try:
                            temp_data = file_hdf['/entry1/sample/temperature'][0]
                            # Check for NaN temperature
                            if not math.isnan(temp_data):
                                temp_int = int(np.round(temp_data, 0))
                                new_sample_name = f"{new_sample_name}_{temp_int}"
                            else:
                                print(f"Warning: Temperature is NaN for scan {current_scan_nr}. Skipping temperature subscript.")
                        except KeyError:
                            print(f"Warning: Temperature dataset not found for scan {current_scan_nr}. Skipping temperature subscript.")
                        except Exception as e:
                            print(f"Warning: Error reading temperature for scan {current_scan_nr}: {e}. Skipping temperature subscript.")
                    elif isinstance(subscript, str) and subscript != '': # If subscript is a non-empty string, append it directly
                        new_sample_name = f"{new_sample_name}{subscript}"

                    # Replace with a new name if 'replace_with' is specified and not 0
                    if isinstance(replace_with, str) and replace_with != '': # Ensure replace_with is a meaningful string
                        new_sample_name = replace_with # Use directly, no need to cast from 0
                    elif isinstance(replace_with, int) and replace_with != 0: # Handle case where replace_with might be a non-zero int
                        new_sample_name = str(replace_with)


                    # Write the new sample name to '/entry1/sample/name_new'
                    dataset_path = '/entry1/sample/name_new'
                    if dataset_path in file_hdf:
                        del file_hdf[dataset_path] # Delete existing dataset to ensure clean overwrite

                    # Create the new dataset. Store as numpy bytes string.
                    file_hdf.create_dataset(dataset_path, data=np.bytes_(new_sample_name))
                    print(f"Scan {current_scan_nr}: Sample name changed from '{current_sample_name}' to '{new_sample_name}'")

                except FileNotFoundError: # Catches if file_hdf could not be opened
                    print(f"Error: HDF5 file '{full_hdf_path}' not found or accessible. Skipping.")
                except KeyError as e: # Catches if '/entry1/sample/name' or '/entry1/sample/temperature' is missing
                    print(f"Error: Missing HDF5 dataset '{e}' in '{full_hdf_path}'. Skipping this file.")
                except Exception as e: # Catch any other unexpected errors during processing
                    print(f"An unexpected error occurred processing '{full_hdf_path}': {e}. Skipping this file.")
                finally:
                    # Ensure the HDF5 file is closed even if errors occur
                    if file_hdf:
                        file_hdf.close()

        # The redundant file_hdf.close() at the very end of the script has been removed.
