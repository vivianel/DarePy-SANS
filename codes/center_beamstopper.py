# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 10:52:03 2023

@author: lutzbueno_v
"""

# %% STEP 1: load the AgBE file
# where are the hdf files saved
path_hdf_raw = 'C:/Users/lutzbueno_v/Documents/Analysis/data/Connor/SANS_2022_2581/DarePy-SANS/raw_data/'
# number of the AgBE scan
scanNr = 22028

# load hdf
import numpy as np
import matplotlib.pyplot as plt
from utils import load_hdf

# NOTE: this name has to be updated every year
name_hdf = 'sans2023n0' + str(scanNr) +'.hdf'
img = load_hdf(path_hdf_raw, name_hdf, 'counts')
Detector_distance = load_hdf(path_hdf_raw, name_hdf, 'detx')
img1 = np.where(img==0, 1e-4, img)

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
ax.imshow(np.log(img1), origin='lower', cmap='jet', clim=[0, 7])
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

img2 = img
mask_beamstopper = [int(np.floor(y2)), int(np.ceil(y3)), int(np.floor(x0)), int(np.ceil(x1))]
bs = mask_beamstopper
img2[bs[0]:bs[1], bs[2]:bs[3]] = 0


# plot the mask
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(np.log(img2), origin='lower')

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
ax.imshow(img1, origin='lower', cmap='jet')
plt.title('press on the ring: left, right, bottom and top')
ax.grid(which='major', color='w', linestyle='--', linewidth=1)
cid0 = fig.canvas.mpl_connect('button_press_event', set_Clicks)


#%% STEP 5: calaculate, plot and save the beamcenter from the coordinates
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




fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(img1, origin='lower', cmap='jet')
radius = (diameter_x/2 + diameter_y/2)/2
circ = plt.Circle((center_x,center_y), radius = radius, facecolor = 'None', edgecolor = 'white', linestyle = '--')
ax.add_patch(circ)


print('_______________COPY________________')
print('___________________________________')
print(f'beam_center_guess -> \'{Detector_distance}\':[{center_x}, {center_y}]')
print('___________________________________')
print('___________________________________')
