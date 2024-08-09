# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 16:06:38 2024

@author: lutzbueno_v
"""

# which instrument
instrument = "sans-llb" # or "sans"

# path to the hdf files
path_hdf_raw = 'C:/Users/lutzbueno_v/Documents/Analysis/data/2024_SANS-LLB/DarePy-SANS/raw_data/'

# path to save th eimages
save_figures = 'C:/Users/lutzbueno_v/Documents/Analysis/data/2024_SANS-LLB/DarePy-SANS/analysis/'

# range of scans to plot
list_scan = list(range(24,51))


##########################################
import numpy as np
import matplotlib.pyplot as plt
from utils import load_hdf

plt.close('all')
plt.ion()

vs_rpm = []
vs_vibration = []
vs_wl = []

for jj in range(0, len(list_scan)):
    scanNr = list_scan[jj]
    # if you get an error double check the year of the file
    name_hdf = instrument + '2024n' +f"{scanNr:06}" +'.hdf'
    vs_vibration_add = load_hdf(path_hdf_raw, name_hdf, 'vs_vibration', instrument)
    vs_vibration.append(vs_vibration_add)
    vs_rpm_add = load_hdf(path_hdf_raw, name_hdf, 'vs_rpm', instrument)
    vs_rpm.append(vs_rpm_add)
    vs_wl_add = load_hdf(path_hdf_raw, name_hdf, 'wl', instrument)
    vs_wl.append(vs_wl_add)

# %% plot



fig, ax1 = plt.subplots()
ax1.plot(vs_rpm, vs_vibration, 'ro', markersize = 10)
#ax1.semilogy(vs_rpm, vs_wl, 'bo', markersize = 10)
ax2 = ax1.twiny()
range_rpm = np.arange(5000, 26000, step=1000)
ax1.set_xticks(range_rpm)

wl = 89237*(1/np.array(range_rpm)) + 0.12
wl = np.round(wl, 1)

ax2.set_xticklabels(wl)

ax1.set_xlabel('Speed (rpm)')
ax2.set_xlabel('Wavelength (A)')

ax1.set_ylabel('Vibration (mm/s)')
plt.xlim([5000, 26000])
plt.xticks(np.arange(5000, 26000, step=1000))
plt.grid()
