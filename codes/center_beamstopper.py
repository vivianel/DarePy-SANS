# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 10:52:03 2023

@author: lutzbueno_v
"""

# %% STEP 1: load the AgBE file
# where are the hdf files saved
path_hdf_raw = 'C:/Users/lutzbueno_v/Documents/Analysis/data/GA_data/2022_2581_GA_dilution/DarePy-SANS/raw_data_GA/'
# number of the AgBE scan
scanNr = 22014

# load hdf
import numpy as np
import matplotlib.pyplot as plt
from utils import load_hdf
import pyFAI
plt.ion()

# NOTE: this name has to be updated every year
name_hdf = 'sans2023n0' + str(scanNr) +'.hdf'
img = load_hdf(path_hdf_raw, name_hdf, 'counts')
Detector_distance = load_hdf(path_hdf_raw, name_hdf, 'detx')
wl = load_hdf(path_hdf_raw, name_hdf, 'wl')
# in log scale
img1 = np.log(img)

#color bar limits

clim = [0,7]

# %% STEP 2: define the mask coordinates by clicking on the 4 edges, and then close the figure

def set_Clicks(event):
    print('press on the ring: left, right, bottom and top')
    global coord0
    coord0.append([event.xdata, event.ydata])
    print('you pressed: x =', np.round(event.xdata, 2), ', and y = ', np.round(event.ydata, 2))
    if len(coord0) == 4:
        fig.canvas.mpl_disconnect(cid0)
    return coord0

coord0 = []
fig, ax = plt.subplots()
ax.imshow(img1, origin='lower', cmap='jet',  clim = clim)
plt.title('press on the beamstopper edges: left, right, bottom and top')
ax.grid(which='major', color='w', linestyle='--', linewidth=1)
cid0 = fig.canvas.mpl_connect('button_press_event', set_Clicks)


#%% STEP 3: calculate and plot the mask for the beamstopper

x0 = coord0[0][0]
y0 = coord0[0][1]
x1 = coord0[1][0]
y1 = coord0[1][1]
size_x = np.sqrt(((x0 - x1)**2)+((y0 - y1)**2))
print('the horizontal beamstopper size is: x =', np.round(size_x,2), 'pixels' )


x2 = coord0[2][0]
y2 = coord0[2][1]
x3 = coord0[3][0]
y3 = coord0[3][1]
size_y = np.sqrt(((x2 - x3)**2)+((y2 - y3)**2))
print('the vertical beamstopper size is: y = ', np.round(size_y,2), 'pixels' )

img2 = img1
mask_beamstopper = [int(np.floor(y2)), int(np.ceil(y3)), int(np.floor(x0)), int(np.ceil(x1))]
bs = mask_beamstopper
img2[bs[0]:bs[1], bs[2]:bs[3]] = 0


# plot the mask
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(img2, origin='lower',  clim = clim)

print('_______________COPY________________')
print('___________________________________')
print(f'beamstopper_coordinates -> \'{Detector_distance}\':[{bs[0]}, {bs[1]}, {bs[2]}, {bs[3]}]')
print('___________________________________')
print('___________________________________')


# %% STEP 4: click on 4 positions of the ring to define the beamcenter, and close the figure afterwards
plt.close('all')

coord0 = []
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(img1, origin='lower', cmap='jet',  clim = clim)
plt.title('press on the ring: left, right, bottom and top')
ax.grid(which='major', color='w', linestyle='--', linewidth=1)
cid0 = fig.canvas.mpl_connect('button_press_event', set_Clicks)


#%% STEP 5: calculate, plot and save the beamcenter from the coordinates
x0 = coord0[0][0]
y0 = coord0[0][1]
x1 = coord0[1][0]
y1 = coord0[1][1]
diameter_x = np.abs(np.sqrt(((x0 - x1)**2)+((y0 - y1)**2)))
center_x = np.round(x0 + diameter_x/2, 2)
print('the horizontal diameter is:', np.round(diameter_x,2), 'pixels' )
print('the beamcenter along X is:', center_x)


x2 = coord0[2][0]
y2 = coord0[2][1]
x3 = coord0[3][0]
y3 = coord0[3][1]
diameter_y = np.abs(np.sqrt(((x2 - x3)**2)+((y2 - y3)**2)))
center_y = np.round(y2 + diameter_y/2, 2)
print('the vertical diameter is:', np.round(diameter_y,2), 'pixels' )
print('the beamcenter along y is:', center_y)


center_x = 62
center_y = 64.5

pixel1 = 7.5e-3
pixel2 = pixel1
poni2 = center_x*pixel1
poni1 = center_y*pixel2

# create the radial integrator
ai = pyFAI.AzimuthalIntegrator(dist = Detector_distance, poni1=poni1, poni2=poni2,rot1=0,
                               rot2=0, rot3=0, pixel1=pixel1, pixel2=pixel2,
                               splineFile=None,  detector=None, wavelength=wl)
ai.setChiDiscAtZero()
# define the number of sectors
sectors_nr = 16
# integrate for azimuthal plots
npt_azim = range(0, 360, int(360/sectors_nr))

for rr in range(0, len(npt_azim)-1):
    azim_start = npt_azim[rr]
    azim_end = npt_azim[rr+1]
    q, I, sigma = ai.integrate1d(img2, 100,
                                 correctSolidAngle = True,
                                 method = 'nosplit_csr',
                                 unit = 'q_A^-1',
                                 safe = True,
                                 error_model = "azimuthal", # "poisson" or "azimuthal",
                                 azimuth_range = [azim_start, azim_end],
                                 flat = None,
                                 dark = None)
    plt.loglog(q, I)



print('_______________COPY________________')
print('___________________________________')
print(f'beam_center_guess -> \'{Detector_distance}\':[{center_x}, {center_y}]')
print('___________________________________')
print('___________________________________')
