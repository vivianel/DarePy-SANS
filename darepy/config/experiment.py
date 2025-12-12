"""
Configuration for the experiment, defines data folders and non-fixed instrument settings.
"""
from dataclasses import dataclass
from .base import ConfigObject, cf
from typing import Dict, List

@dataclass
class Experiment(ConfigObject):
    """
    Parameters for the experiment.
    """
    path_hdf_raw: str = cf('raw_data', 'The (relative) path to the raw hdf data files')
    output_dir: str = cf('.', 'The (relative) path where resulting files will be stored')

    instrument: str = cf('SANS-LLB', 'The instrument used') # remove in future

    empty_beam: str = cf('EB', 'HDF file sample name for empty beam measurement')
    cadmium: str = cf('Cd', 'HDF file sample name for cadmium measurement used for background')
    water: str = cf('H2O', 'HDF file sample name for water measurement used as flat field')
    water_cell: str = cf('EC', 'HDF file sample name for empty cell measurement for water')
    empty_cell: str = cf('EC', 'HDF file sample name for empty cell measurement for samples')

    sample_thickness: Dict[str,float] = cf(default_factory=lambda: {'all': 0.1},
                                           doc='Sample thickness information [cm]. '
                                               'The key is matched with the sample name from the datafile.')

    beamstopper_coordinates: Dict[float, list] = cf(default_factory=lambda: {},
                          doc='Mapping of detector distance to beamstopper '
                              'masked areas determined using bsc action')
    transmission_coordinates: Dict[float, list] = cf(default_factory=lambda: {},
                          doc='Mapping of detector distance to transmitted beam coordinates (SANS-LLB)')
    beam_center_guesses: Dict[float, list] = cf(default_factory=lambda: {},
                          doc='Mapping of detector distance to beam center in pixels')
