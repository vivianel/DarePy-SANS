"""
Combine analyzed files together

Created on Thu Aug 17 16:33:06 2023

@author: lutzbueno_v
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from dataclasses import asdict


class Action:
    priority = 60

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

    def run(self):
        from ..config import cfg

        import os
        # %% Plot the detector distances in the same graphic
        path_dir_an = os.path.join(cfg.experiment.output_dir, 'analysis')
        import darepy.post_processing as pp

        # %% STEP 1: PLOT DATA TOGETHER

        #merged_files = pp.plot_all_data_sectors(path_dir_an)

        merged_files = pp.plot_all_data(path_dir_an)

        # %% STEP 2: REMOVE POINTS AND MERGE
        skip_start = cfg.merging.skip_start
        skip_end = cfg.merging.skip_end

        # For the interpolation and in which scale
        interp_type = cfg.merging.interp_type
        interp_points = cfg.merging.interp_points
        smooth_window = cfg.merging.smooth_window

        pp.merging_data(path_dir_an, merged_files, skip_start, skip_end, interp_type, interp_points, smooth_window)



        # %% FIT THE POROD LINE AND REMOVE INCOHERENT
        Last_points_fit = 2
        pp.subtract_incoherent(path_dir_an, Last_points_fit)
