# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:33:06 2023

@author: lutzbueno_v
"""

import post_processing as pp

path_dir_an = 'C:/Users/lutzbueno_v/Documents/Analysis/data/2022_2581_Viviane/DarePy-SANS/analysis/'

merged_files = pp.plot_all_data(path_dir_an)

# %%
skip_start = [1,10 , 2]
skip_end = [2, 2, 0]
interp_type = 'log' # 'log' or 'linear'

pp.merging_data(path_dir_an, merged_files, skip_start, skip_end, interp_type)


# %%

# how much the value can vary
var_offset = 1e-3

# define the points to fit at the end
fitting_range = 5

pp.subtract_incoherent(path_dir_an, var_offset, fitting_range)
