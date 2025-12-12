"""
Created on Sun Dec  7 12:04:30 2025

@author: lutzbueno_v
"""

from darepy.utils import load_hdf
from darepy import normalization
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

path_dir = 'C:/Users/lutzbueno_v/Documents/Analysis/data/SANS-LLB/2024_SANS-LLB/DarePy-SANS'

# number of the AgBE scan
scanNr = 883
instrument = 'SANS-LLB'
hdf_name = f'sans-llb2025n000{scanNr}.hdf'

# %%
print(f"Attempting to load: {hdf_name}")


path_hdf_raw = os.path.join(path_dir, 'raw_data')
path_dir_an =  os.path.join(path_dir, 'analysis')
save_directory =  os.path.join(path_dir, 'codes')

file_name_config = os.path.join(path_dir_an, 'config.npy')
with open(file_name_config, 'rb') as handle:
        config = pickle.load(handle)


#  Load Data
counts = load_hdf(path_hdf_raw, hdf_name, 'counts')
#counts = normalization.normalize_deadtime(config, hdf_name, counts)
counts = normalization.normalize_time(config, hdf_name, counts)

#plt.imshow(np.log(counts))
# this is a "quick flat field correction
# we need a better correction
c =  [55, 71, 55, 71]
counts[c[0]:c[1], c[2]:c[3]]  = np.median(counts)
counts = counts/np.mean(counts)
plt.imshow(counts)
plt.show()

file_name = "../darepy/data/flat_field_SANS-LLB.txt"
full_path = os.path.join(save_directory, file_name)
np.savetxt(full_path, counts, fmt='%.5f', delimiter='\t')
