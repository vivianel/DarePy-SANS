# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:33:06 2023

@author: lutzbueno_v
"""

import post_processing as pp

path_dir_an = 'C:/Users/lutzbueno_v/Documents/Analysis/data/2022_2581_Viviane/analysis/'

merged_files = pp.plot_all_data(path_dir_an)

# %%
skip_start = [0, 15, 20]
skip_end = [80, 20, 5]

pp.merging_data(path_dir_an, merged_files, skip_start, skip_end)


# %%
var_offset = 1e-3
pp.subtract_incoherent(path_dir_an, var_offset)
