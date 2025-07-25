# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:33:06 2023

@author: lutzbueno_v
"""


# %% Plot the detector distances in the same graphic
path_dir = 'C:/Users/lutzbueno_v/Documents/Analysis/DarePy-SANS/'


import os
os.chdir(path_dir + '/codes/')
path_dir_an = path_dir + '/analysis/'
import post_processing as pp

# %% STEP 1: PLOT DATA TOGETHER

#merged_files = pp.plot_all_data_sectors(path_dir_an)

merged_files = pp.plot_all_data(path_dir_an)

# %% STEP 2: REMOVE POINTS AND MERGE
# skip the points at the start of the radial integration
# for measurements with 3 detector distances: [X, Y, Z ] points
skip_start = {'2':0,'1':5 ,'0':10}
# for measurements with 2 detector distances: [X, Y, Z ] points
#skip_start = {'1':0 ,'0':0}


# skip the points at the end of the radial integration
# for measurements with 3 detector distances: [X, Y, Z ] points
skip_end = {'2':30,'1':10 ,'0':1}
# for measurements with 3 detector distances: [X, Y, Z ] points
#skip_end = {'1':0 ,'0':0}


# For the interpolation and in which scale
interp_type = 'log' # 'log' or 'linear' or 'none' for avoiding the interpolation
interp_points = 100
smooth_window = 1

pp.merging_data(path_dir_an, merged_files, skip_start, skip_end, interp_type, interp_points, smooth_window)



# %% FIT THE POROD LINE AND REMOVE INCOHERENT

# define the range of the inoherent part to fit
Last_points_fit = 50
pp.subtract_incoherent(path_dir_an, Last_points_fit)
