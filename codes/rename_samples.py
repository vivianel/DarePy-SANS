# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 12:37:26 2022

@author: lutzbueno_v
"""

import h5py
import numpy as np
import os

# path where the raw data is saved: usually AFS
path_hdf_raw = 'C:/Users/lutzbueno_v/Documents/Analysis/data/2023_0546_RheoSANS/DarePy-SANStest/raw_data/'

# for a single file you need to add the quotes
files_change = [57218]

# if you want to replace with temparature call the subscript 'temp', otherwise keep it empty ''
subscript = ''
# name to be replaced
replace_with = 'GA_SiC_fibres'

files = []
for r, d, f in os.walk(path_hdf_raw):
    for file in f:
        if '.hdf' in file:
            files.append(os.path.join(file))

for ii in range(0, len(files)):
    if int(files[ii][9:-4])  in files_change:

        name_hdf = os.path.join(path_hdf_raw, files[ii])
        file_hdf = h5py.File(name_hdf, 'r+')

        name = str(np.asarray(file_hdf['/entry1/sample/name']))
        name = name[3:-2]
        if subscript == 'temp':
            temp = int(np.round(file_hdf['/entry1/sample/temperature'][0], 0))
            name = name + '_' + str(temp)
        else:
            name = name + str(subscript)
        if replace_with != 0:
            name = replace_with
        try:
            del file_hdf['/entry1/sample/name_new']
            file_hdf.create_dataset('/entry1/sample/name_new', data = np.bytes_(name))
        except:
            file_hdf.create_dataset('/entry1/sample/name_new', data = np.bytes_(name))

        file_hdf.close()

file_hdf.close()
