# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:33:06 2023

@author: lutzbueno_v
"""

import post_processing as pp

path_dir_an = 'C:/Users/lutzbueno_v/Documents/Analysis/data/2023_SANS_Ashley/DarePy-SANS/analysis/'


# %% Plot the detector distances in the same graphic

merged_files = pp.plot_all_data(path_dir_an)

# %%

# skip the points at the start of the radial integration
# for measurements with 3 detector distances: [X, Y, Z ] points
skip_start = [1,10 , 2]

# skip the points at the end of the radial integration
# for measurements with 3 detector distances: [X, Y, Z ] points
skip_end = [2, 2, 0]

# For the interpolation and in which scale
interp_type = 'log' # 'log' or 'linear'

pp.merging_data(path_dir_an, merged_files, skip_start, skip_end, interp_type)

# %% Fitting the Porod scale and the incoherent background

# how much the value can vary
var_offset = 0

# define the points to fit at the end
fitting_range = 20

pp.subtract_incoherent(path_dir_an, var_offset, fitting_range)
