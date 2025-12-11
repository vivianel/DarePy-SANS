"""
Configuration for the experiment, defines data folders and non-fixed instrument settings.
"""
from pathlib import Path
from dataclasses import dataclass
from .base import ConfigObject, cf

@dataclass
class Experiment(ConfigObject):
    """
    Parameters for the experiment.
    """
    path_hdf_raw: str = cf('raw_data', 'The (relative) path to the raw hdf data files')
