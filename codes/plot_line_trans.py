# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 19:10:22 2022

@author: Sinquser
"""

#plot the transmission line of the images
list_scan = list(range(45268, 45278))
ref = 45268
#step = 0.5 # mm
along_axis = 'x'# 'x' or 'z'


path_dir_raw = 'C:/Users/Sinquser/Documents/Python_Scripts/20212857/raw_data'


# function
import h5py
import numpy as np
import matplotlib.pyplot as plt
from normalize_time import normalize_time

plt.close('all')
# find references
name_hdf = path_dir_raw + '/sans2022n0' + str(ref) +'.hdf'
file_hdf = h5py.File(name_hdf, 'r')
img = np.array(file_hdf['entry1/data1/counts'])
# 1e-4 is chosen as a small value
img1 = normalize_time(file_hdf, img)
# calculation of the cuttoff value for the mask
cutoff = img1[img1>0].mean()
plt.figure()
plt.imshow(img1, clim=[0, round(cutoff)], cmap='jet', origin='lower')
plt.colorbar(orientation = 'vertical', shrink = 0.5).set_label('log(Intensity)')
im_title = 'Empty_beam'
plt.title(im_title)
#im_title = str(path_transmission + im_title + '.jpg')
#plt.savefig(im_title)
img2 = np.where(img1<cutoff, 0, img1)
mask = np.where(img2>=cutoff, 1, img2)
plt.figure()
plt.imshow(mask, cmap='gray', origin='lower')
plt.colorbar(orientation = 'vertical', shrink = 0.5, ticks=[0, 1]).set_label('Binary')
im_title = 'Transmission_Mask'
EB_ref = int(np.sum(np.multiply(img,mask)))
#print('###########################################################')
#print('This is the EB transmission:' + str(EB_ref))
#print('###########################################################')
plt.title(im_title +  ', Total counts = ' + str(EB_ref))
#im_title = str(path_transmission + im_title + '.jpg')
#plt.savefig(im_title)

list_trans = []
list_counts = []
range_mm = []
# calculate all     

   
for jj in range(0, len(list_scan)):     
    name_hdf = path_dir_raw + '/sans2022n0' + str(list_scan[jj]) +'.hdf'
    file_hdf = h5py.File(name_hdf, 'r')
    img = np.array(file_hdf['entry1/data1/counts'])
    counts = int(np.sum(np.multiply(img,mask)))
    if along_axis == 'z':
        step = (int(np.asarray(file_hdf['/entry1/sample/z_position'])))
    if along_axis == 'x':
        step = (int(np.asarray(file_hdf['/entry1/sample/position'])))
    list_counts.append(counts)
    #print('This is sample '+ class_trans['name'][ii] + ' transmission:' + str(counts))
    trans = np.divide(counts,1)#EB_ref)
    trans = round(trans, 3)
    list_trans.append(trans)
    range_mm.append(step)
    
plt.figure()   
plt.plot(range_mm, list_trans)
plt.figure()
plt.plot(range_mm, list_counts)
           