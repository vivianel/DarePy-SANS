"""
Configuration for the instruments.
"""
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

from .base import ConfigObject, cf

@dataclass
class SANS1Cfg(ConfigObject):
    """
    Parameters for the SANS-I instrument.
    """
    name:str = "SANS-I"
    deadtime:float = 6.6e-7
    list_attenuation: Dict[str,float] = cf(default_factory=lambda:
            {'0':1, '1':1/485,'2':1/88,'3':1/8, '4':1/3.5,'5':1/8.3},
            doc='Mapping of attenuator index to transmitted intensity fraction')
    pixel_size:float = 7.5e-3
    detector_size:int = 128
    list_bs: Dict[str,float] = cf(default_factory=lambda:
        {'1': 40., '2': 70., '3': 85., '4': 100.},
            doc='List of beam stopper sizes (if applicable, often not used directly in this pipeline).')
    efficiency_map:str = "flat_field_SANS-I.txt"
    list_abs_calib: Dict[str,float] = cf(default_factory=lambda:
        {'5':0.909, '6':0.989, '8':1.090, '10':1.241, '12':1.452},
            doc="""Absolute calibration cross-sections for different wavelengths.
          A dictionary where keys are rounded wavelengths in Angstroms (as strings)
          and values are calibration factors (e.g., from water calibration)
          to convert to cm^-1 units.""")

@dataclass
class SANSLLBCfg(ConfigObject):
    """
    Parameters for the SANS-LLB instrument.
    """
    name:str = "SANS-LLB"
    deadtime:float = 3.5e-6
    list_attenuation: Dict[str,float] = cf(default_factory=lambda:
            {'0':1, '1':1/485,'2':1/88,'3':1/8, '4':1/3.5,'5':1/8.3},
            doc='Mapping of attenuator index to transmitted intensity fraction')
    pixel_size:float = 5e-3
    detector_size:int = 128
    list_bs: Dict[str,float] = cf(default_factory=lambda:
        {'1': 40., '2': 70., '3': 85., '4': 100.},
            doc='List of beam stopper sizes (if applicable, often not used directly in this pipeline).')
    efficiency_map:str = "flat_field_SANS-LLB.txt"
    list_abs_calib: Dict[str,float] = cf(default_factory=lambda:
        {'5':0.909, '6':0.989, '8':1.090, '10':1.241, '12':1.452},
            doc="""Absolute calibration cross-sections for different wavelengths.
          A dictionary where keys are rounded wavelengths in Angstroms (as strings)
          and values are calibration factors (e.g., from water calibration)
          to convert to cm^-1 units.""")
