"""
Perform radial integration on datafiles

It orchestrates the entire process from loading raw HDF5 files to performing
radial and azimuthal integration, applying various corrections (dark field,
empty cell, flat field, transmission), and optionally performing absolute
intensity calibration.

Users can configure experimental parameters, instrument-specific settings,
and detailed analysis options. The pipeline supports processing for all
detected detector distances or a selection of specific distances.
Results, including integrated data and plots, are saved to a designated
analysis directory.

Created on Wed Jul 26 13:31:39 2023

@author: lutzbueno_v
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from dataclasses import asdict


class Action:
    priority = 50

    @classmethod
    def get_parser(cls) -> ArgumentParser:
        parser = ArgumentParser(prog='darepy radint', formatter_class=ArgumentDefaultsHelpFormatter,
                                description='Perform radial integration on datafiles')
        parser.add_argument('target_detector_distances', type=int, default=[], nargs='*',
                            help='If present, select target detector distances for processing.\n'
                            'This parameter controls which detector distances will be processed by the analysis pipeline.'
                            'You can choose to process all available data (no entry) or focus on specific distances to save time.')

        parser.add_argument('-w', '--wavelength', type=float,
                            help='Wavelength of the instrument to use for q-calculations, if absent will be read from HDF')
        parser.add_argument('-s', '--analysis-suffix', type=str, default='',
                            help='A string appended to the default analysis directory to determine the folder to save to')
        return parser

    def __init__(self, arguments):
        self.arguments = arguments

    def set_instrument(self):
        # load first file in raw_data directory to determine which instrument configuration to use
        from ..config import cfg
        import h5py
        import os
        from glob import glob
        hdf = h5py.File(glob(os.path.join(cfg.experiment.path_hdf_raw, '*.hdf'))[0], 'r')
        inst_str = hdf.attrs['instrument'].decode('utf-8')
        if inst_str.lower().startswith('sans-llb'):
            cfg.instrument = cfg.sansllb
        else:
            cfg.instrument = cfg.sans1

    def run(self):
        from ..config import cfg

        # %% EXPERIMENTAL PARAMETERS
        # This section defines parameters related to the SANS experiment and instrument setup.

        which_instrument = cfg.experiment.instrument

        empty_beam = cfg.experiment.empty_beam
        cadmium = cfg.experiment.cadmium
        water = cfg.experiment.water
        water_cell = cfg.experiment.water_cell
        empty_cell = cfg.experiment.empty_cell
        sample_thickness = cfg.experiment.sample_thickness
        trans_dist = 18.3
        wl = getattr(self.arguments, 'wavelength', 'auto')
        beamstopper_coordinates = cfg.experiment.beamstopper_coordinates
        transmission_coordinates = cfg.experiment.transmission_coordinates
        beam_center_guess = cfg.experiment.beam_center_guesses
        target_detector_distances = self.arguments.target_detector_distances or 'all'


        # %% ANALYSIS PARAMETERS
        # This section defines parameters related to the data reduction and analysis pipeline's execution.
        path_hdf_raw =cfg.experiment.path_hdf_raw
        add_id = self.arguments.analysis_suffix
        exclude_files = cfg.reduction.exclude_files
        plot_radial = cfg.reduction.plot_radial
        plot_azimuthal = cfg.reduction.plot_azimuthal
        save_azimuthal = cfg.reduction.save_azimuthal
        save_2d_patterns = cfg.reduction.save_2d_patterns
        perform_abs_calib = cfg.reduction.perform_abs_calib
        force_reintegrate = cfg.reduction.force_reintegrate
        integration_points = cfg.reduction.integration_points
        sectors_nr = cfg.reduction.sectors_nr
        pixel_range_azim = range(*cfg.reduction.pixel_range_azim)


        # %% PIPELINE EXECUTION (DO NOT MODIFY BELOW THIS LINE)
        # This section contains the core logic for running the data reduction pipeline.
        # It initializes the configuration, orchestrates the various processing steps,
        # and manages module imports.

        # Import necessary modules for reduction
        import darepy.prepare_input as org
        from darepy.transmission import trans_calc
        import darepy.integration as ri



        # Prepare a dictionary that maps generic calibration names (used in the code)
        # to their specific sample names (as recorded in your HDF5 raw data).
        calibration = {'cadmium':cadmium, 'water':water, 'water_cell': water_cell, 'empty_cell':empty_cell}

        # Initialize the 'result' dictionary. This dictionary will serve as a central
        # repository to store all intermediate data, processing flags, and final
        # analysis outputs throughout the pipeline. It is passed between functions.
        result = {'transmission':{},
               'overview':{},
               'integration':{
                   'pixel_range_azim':pixel_range_azim,
                   'integration_points':integration_points,
                   'sectors_nr': sectors_nr}}



        # select the corrcet settings
        self.set_instrument()
        configuration = {'instrument': asdict(cfg.instrument),
            'experiment': {'trans_dist': trans_dist, # Transmission measurement distance (from above).
                           'calibration':calibration, # Calibration sample names (from above).
                           'sample_thickness':sample_thickness, # Sample thickness info (from above).
                           'wl_input': wl}, # Wavelength input setting (from above).
            'analysis': {'path_dir': cfg.experiment.output_dir, # Working directory path (from above).
                         'path_hdf_raw':path_hdf_raw, # Raw HDF5 data path (from above).
                         'exclude_files':exclude_files, # List of scans to exclude (from above).
                         'perform_abs_calib':perform_abs_calib, # Absolute calibration toggle (from above).
                         'force_reintegrate': force_reintegrate, # Force re-integration toggle (from above).
                         "plot_azimuthal":plot_azimuthal, # Azimuthal plot toggle (from above).
                         "plot_radial":plot_radial, # Radial plot toggle (from above).
                         'add_id':add_id, # Analysis folder ID (from above).
                         'save_azimuthal': save_azimuthal, # Save azimuthal data files (from above).
                         'save_2d_patterns': save_2d_patterns, # Save 2D pattern files (from above).
                         'empty_beam':empty_beam, # Empty beam sample name (from above).
                         'beam_center_guess': beam_center_guess, # Beam center guesses (from above).
                         'beamstopper_coordinates': beamstopper_coordinates, # Beam stopper coords (from above).
                         'transmission_coordinates': transmission_coordinates, # transmission coords (from above).
                         'target_detector_distances': target_detector_distances # Target detector distances (from above).
                         }}

        # %% STEP 1: Load all HDF5 files and create an overview.
        # This step scans the raw data directory, extracts relevant metadata (scan number,
        # sample name, detector distance, etc.) from each HDF5 file, and compiles it
        # into a structured dictionary ('class_files'). This overview is then saved
        # to the 'result' dictionary.
        config = configuration # Select the configuration set for the chosen instrument.
        class_files = org.list_files(config, result)

        # %% STEP 2: Calculate transmission for samples (if applicable).
        # This step determines the transmission of each sample by comparing its direct
        # beam intensity to that of an empty beam measurement. Transmission correction
        # is crucial for absolute intensity calibration and accounting for neutron
        # absorption by the sample. The actual correction is applied later during normalization.
        if trans_dist > 0: # Only run if a positive transmission measurement distance is specified.
            trans_calc(config, class_files, result)
        else:
            print('No transmission measurements specified (trans_dist <= 0). Transmission correction will be skipped.')

        # %% STEP 3: Organize files by detector distance.
        # This step categorizes the loaded files based on their unique detector distances.
        # It also copies relevant raw HDF5 files into detector-specific subfolders within
        # the analysis directory for easier access during integration. Only detector
        # distances specified in 'target_detector_distances' will be processed here.
        result = org.select_detector_distances(config, class_files, result)


        # %% STEP 4: Perform radial and azimuthal integration.
        # This is the core data reduction step. It converts 2D detector images into
        # 1D scattering curves (Intensity vs. scattering vector 'q').
        # This step applies various corrections including dark field subtraction, empty
        # cell subtraction, flat field correction, transmission correction, and optionally
        # absolute intensity calibration.
        # It also generates and saves integrated data files and plots based on user settings.

        # Determine which detector distances to integrate based on the
        # 'target_detector_distances' setting in the configuration.
        if config['analysis']['target_detector_distances'] == 'all':
            # If 'all' is selected, iterate through all detector distances that were
            # successfully processed and organized in the 'result' dictionary during
            # the 'select_detector_distances' step (Step 3).
            processed_det_distances = []
            for key in result['overview'].keys():
                if key.startswith('det_files_'):
                    # Extract the detector distance string (e.g., '1p6', '4p5', '18p0')
                    # from the 'result' dictionary keys.
                    processed_det_distances.append(key.replace('det_files_', ''))
        else:
            # If specific distances are listed (e.g., [4.5, 18.0]), convert them to the
            # string format (e.g., '4.5' -> '4p5') to match the internal folder
            # and key naming conventions used by the pipeline.
            processed_det_distances = [str(d).replace('.', 'p') for d in config['analysis']['target_detector_distances']]

        # Iterate through each selected/processed detector distance and perform integration.
        # This loop ensures that all data for each specified detector distance is processed
        # sequentially and correctly.
        for det_str in processed_det_distances:
            # Print a clear message indicating which detector distance is currently being processed.
            print(f"\n--- Processing Detector Distance: {det_str.replace('p', '.')}m ---")
            # Call the 'set_integration' function from the 'integration' module.
            # The 'det_str' argument ensures that processing is confined to this specific detector distance.
            result = ri.set_integration(config, result, det_str)
