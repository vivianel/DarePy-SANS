# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:33:06 2023

@author: lutzbueno_v
"""

import post_processing as pp

path_dir_an = 'C:/Users/lutzbueno_v/Documents/Analysis/data/2022_2581_Andrea/analysis/'

pp.merging_data(path_dir_an)
pp.subtract_incoherent(path_dir_an)
