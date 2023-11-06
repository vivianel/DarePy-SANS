# -*- coding: utf-8 -*-
"""
Spyder Editor

conda activate spyder-env

This is a temporary script file.
"""
list_scan = [65339]#list(range(45319,45328))#[45220, 45291]#list(range(45219,45221))


path_dir_raw = 'C:/Users/lutzbueno_v/Documents/Analysis/data/2023_SANS_Ashley/DarePy-SANS/raw_data/'

import h5py
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
plt.ion()

for jj in range(0, len(list_scan)):
    name_hdf = path_dir_raw + '/sans2023n0' + str(list_scan[jj]) +'.hdf'
    file_hdf = h5py.File(name_hdf, 'r')
    img = np.array(file_hdf['entry1/SANS/detector/counts'])
    img1 = np.where(img==0, 1e-4, img)
    plt.figure()
    Int = 1
    img2 = img1
    clim1 = (0, Int*img2[img2>0].std())

    imgplot = plt.imshow(np.log(img1), clim=[0, 7], origin='lower')
    imgplot.set_cmap('jet')
    plt.colorbar()
    plt.title((str(np.asarray(file_hdf['/entry1/sample/name'])))+ str(list_scan[jj]))
    file_hdf.close()
