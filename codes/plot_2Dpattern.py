# -*- coding: utf-8 -*-
"""
Spyder Editor

conda activate spyder-env

This is a temporary script file.
"""
list_scan = list(range(22029,22038))#[45220, 45291]#list(range(45219,45221))


path_hdf_raw = 'C:/Users/lutzbueno_v/Documents/Analysis/data/GA_data/2022_2581_GA_dilution/DarePy-SANS/raw_data/'

import numpy as np
import matplotlib.pyplot as plt
from utils import load_hdf

plt.close('all')
plt.ion()

# load background
scanNr = 22038
name_hdf = 'sans2023n0' + str(scanNr) +'.hdf'
img = load_hdf(path_hdf_raw, name_hdf, 'counts')
back = np.where(img==0, 1e-4, img)



for jj in range(0, len(list_scan)):
    scanNr = list_scan[jj]
    name_hdf = 'sans2023n0' + str(scanNr) +'.hdf'
    img = load_hdf(path_hdf_raw, name_hdf, 'counts')
    img1 = np.where(img==0, 1e-4, img)
    img1 = img1 - back
    plt.figure()
    Int = 1
    img2 = img1
    clim1 = (0, Int*img2[img2>0].std())

    imgplot = plt.imshow((img1), clim = [0, 100], origin='lower')
    imgplot.set_cmap('jet')
    plt.colorbar()
    sample_name = load_hdf(path_hdf_raw, name_hdf, 'sample_name')
    plt.title(sample_name + ', #'+ str(list_scan[jj]))

    folder_merged = 'C:/Users/lutzbueno_v/Documents/Analysis/data/GA_data/2022_2581_GA_dilution/DarePy-SANS/analysis/merged/'
    file_name = folder_merged + sample_name + '.jpeg'
    plt.savefig(file_name)
