# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 14:20:44 2022

@author: Sinquser
"""

# path where the raw data is saved: usually AFS
path_dir_raw = 'C:/Users/lutzbueno_v/Documents/Analysis/data/2022_2581_Andrea/raw_data'
# path where to save the data: working directory
path_dir = 'C:/Users/lutzbueno_v/Documents/Analysis/data/2022_2581_Andrea/'
# path where codes are located
path_codes = 'C:/Users/lutzbueno_v/Documents/Analysis/data/2022_2581_Andrea/codes/'


# distance where transmission has been calculated
# use trans_dist < 0 to avoid correction by transmission
trans_dist = 18 # in m

# sample to consider for the creation of the transmission mask
mask_measurement = 'EB'

file_beam_center = 'EB' #usually we use the empty beam (EB)

perform_abs_calib = 1
perform_anisotropy = 0
force_reintegrate = 1

replace_18m_detector = 4.5 # if 0 it does not replace


# files to exclude from measurements (scan numbers)
exclude_files = []

########################################################################################
import pylab
pylab.ioff()
import sys
del sys.path[8:]
sys.path.append(path_codes)

from list_files import list_files
list_files(path_dir_raw, path_dir, trans_dist, exclude_files)

from transmission_calc import trans_calc
trans_calc(path_dir, mask_measurement, trans_dist)

from radial_integration import radial_integ
radial_integ(path_dir_raw, path_dir, file_beam_center, perform_abs_calib, 
              trans_dist, perform_anisotropy, force_reintegrate, replace_18m_detector)

pylab.ion()