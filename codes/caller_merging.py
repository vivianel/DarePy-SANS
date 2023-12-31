# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:33:06 2023

@author: lutzbueno_v
"""

import post_processing as pp

path_dir_an = 'C:/Users/lutzbueno_v/Documents/Analysis/data/2022_0915_Combet/DarePy-SANS/analysis/'



# %% STEP 1: PLOT DATA TOGETHER

merged_files = pp.plot_all_data(path_dir_an)

# %% STEP 2: REMOVE POINTS AND MERGE
skip_start = {'2':1,'1':5 ,'0':5}
skip_end = {'2':40,'1':5 ,'0':2}


interp_type = 'none' # 'log' or 'linear' or 'none' for no interpolation
interp_points = 100

pp.merging_data(path_dir_an, merged_files, skip_start, skip_end, interp_type, interp_points)



# %% FIT THE POROD LINE AND REMOVE INCOHERENT

# define the range of the inoherent part to fit
Last_points_fit = 30

pp.subtract_incoherent(path_dir_an, Last_points_fit)
