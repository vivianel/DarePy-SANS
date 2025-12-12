"""
Configuration for the experiment, defines data folders and non-fixed instrument settings.
"""
from dataclasses import dataclass
from typing import Dict, List

from .base import ConfigObject, cf

@dataclass
class Merging(ConfigObject):
    """
    Optons used when merging analyzed files.
    """
    skip_start:Dict[str,int] = cf(default_factory=lambda :{'2':7,'1':30 ,'0':8},
          doc="""Skip the points at the start of the radial integration
        for measurements with 3 detector distances: [X, Y, Z ] points""")
    skip_end:Dict[str,int] = cf(default_factory=lambda :{'2':30,'1':5 ,'0':1},
          doc="""Skip the points at the end of the radial integration
        for measurements with 3 detector distances: [X, Y, Z ] points""")
    interp_type:str = cf('log', "'log' or 'linear' or 'none' for avoiding the interpolation")
    interp_points:int = 100
    smooth_window:int = 1
    last_points_fit:int = cf(2, 'define the range of the incoherent part to fit')

@dataclass
class Reduction(ConfigObject):
    """
    Parameters for the data reduction like binning in radial integration.
    """
    plot_radial:bool = cf(True, """Control radial integration and plotting.
        Radial integration produces 1D scattering curves (Intensity vs. q).
        - Set to true to perform radial integration and generate corresponding plots
          for visualization and quality checks.
        - Set to false to skip radial integration and plotting. Raw 2D data will still
          be processed through correction steps if absolute calibration is enabled,
          but no 1D radial data will be generated or saved.""")
    plot_azimuthal:bool = cf(False, """Control azimuthal integration and plotting.
        Azimuthal integration analyzes scattering intensity as a function of azimuthal angle,
        useful for studying anisotropy.
        - Set to 1 to perform azimuthal integration and generate corresponding plots.
        - Set to 0 to skip azimuthal integration and plotting. No azimuthal data
          will be generated or saved.""")
    save_azimuthal:bool = cf(False, """Control saving of azimuthal integration data files (.dat).
        This flag independently controls whether the calculated 1D azimuthal data
        (q vs. I for each sector) is saved to disk.
        - Set to 1 to save azimuthal data files (e.g., 'azim_integ_*.dat').
        - Set to 0 to skip saving these data files, even if 'plot_azimuthal' is 1
          (the plots will still be generated if plot_azimuthal is 1).
          Note: Setting 'plot_azimuthal' to 0 will implicitly skip saving data as well.""")
    save_2d_patterns:bool = cf(False, """Control saving of raw 2D detector patterns (.dat).
        These files are direct representations of the corrected 2D detector images.
        They can be very large and consume significant disk space.
        - Set to 1 if you need to save these raw 2D patterns for every frame.
        - Set to 0 to skip saving these 2D pattern files. This can significantly
          speed up the process by reducing disk I/O.""")
    perform_abs_calib:bool = cf(True, """Absolute calibration toggle.
        This step converts scattering intensities from arbitrary detector counts
        into absolute units (cm^-1), allowing for quantitative comparison with models
        or other experiments.
        - Set to 1 to perform absolute intensity calibration. This requires the
          'water' standard measurement and its associated configuration.
        - Set to 0 to skip absolute calibration. Intensities will remain in
          arbitrary units, suitable for relative comparisons.""")
    force_reintegrate:bool = cf(True, """Force re-integration of data.
        This flag controls whether previously integrated files are re-processed.
        - Set to 1 to force the radial and azimuthal integration process to re-run
          for all files, even if corresponding integrated `.dat` files already exist.
          This will overwrite existing results. Use this if you change analysis parameters.
        - Set to 0 to only integrate new or un-integrated files. This significantly
          speeds up re-runs if only a few new files are added or parameters are unchanged.""")


    integration_points:int = cf(120, """Number of integration points for radial integration.
        This determines the number of 'q' (scattering vector) bins in the final 1D
        radial scattering curve. A higher number provides more detailed curves but
        may increase processing time and file size.""")
    sectors_nr:int = cf(16, """Number of angular sectors for azimuthal integration.
        This divides the 2D detector image into this many angular slices (bins) for
        azimuthal anisotropy analysis.""")

    pixel_range_azim:tuple = cf((5,100), """Pixel range for azimuthal integration.
        Defines the radial pixel range (distance from the beam center) over which
        azimuthal intensity will be averaged. This helps focus azimuthal analysis on
        a specific 'q' region of interest.
        Example: (5, 40) means pixels from 5 to 39 (inclusive) from the beam center.""")
    exclude_files: List[int] = cf(default_factory=lambda: [],
                                  doc='List of scan numbers to be excluded from the analysis pipeline')