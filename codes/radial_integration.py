# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 08:49:22 2024

@author: lutzbueno_v
"""

# which instrument
instrument = "sans-llb" # or "sans"

# path to the hdf files
path_hdf_raw = 'C:/Users/lutzbueno_v/Documents/Analysis/data/2024_SANS-LLB/DarePy-SANS/raw_data/'

# path to save th eimages
save_figures = 'C:/Users/lutzbueno_v/Documents/Analysis/data/2024_SANS-LLB/DarePy-SANS/analysis/'

# minimum value for the colorbar in log
vmin = 0

# maximum value for the colorbar in log
vmax = 7

# range of scans to plot
list_scan = list(range(24, 51))


##########################################
import numpy as np
import matplotlib.pyplot as plt
from utils import load_hdf
import cv2
import pyFAI

plt.close('all')
plt.ion()


for jj in range(0, len(list_scan)):
    scanNr = list_scan[jj]
    # if you get an error double check the year of the file
    name_hdf = instrument + '2024n' +f"{scanNr:06}" +'.hdf'

    img_main = load_hdf(path_hdf_raw, name_hdf, 'counts_main', instrument)
    h, w = img_main.shape
    img_main = img_main[:, int(h/2):int(w-(h/2))]
    h, w = img_main.shape
    data = cv2.resize(img_main, (h, w//3), interpolation = cv2.INTER_NEAREST);
    data = np.where(data <= 0, 1e-4, data)


    # calculate wavelength for the title
    vs_rpm = load_hdf(path_hdf_raw, name_hdf, 'vs_rpm', instrument)
    wl = 89237*(1/vs_rpm) + 0.12
    sample_name = load_hdf(path_hdf_raw, name_hdf, 'sample_name', instrument)
    title_c = ('#'+ str(list_scan[jj]) + ', ' + f"{wl:.2}" + 'A, ' + sample_name )


    fig = plt.figure(figsize=(12, 7))
    ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)


    # beam_center_guess -> '4':[173.94, 190.37]
    #beamstopper_coordinates -> '4':[166, 212, 153, 196]
    beamstopper_coordinates = [55, 71, 30, 44]
    pixelx = 5e-3 # m
    pixely = 5e-3 #m
    # position of the beam center
    bc_x = 36
    bc_y =  63

    ponix = bc_x*pixelx
    poniy = bc_y*pixely
    dist = 3.85 #m

    ai = pyFAI.AzimuthalIntegrator(dist=dist, poni1=poniy, poni2=ponix,rot1=0,
                                   rot2=0, rot3=0, pixel1=pixely, pixel2=pixelx,
                                   splineFile=None,  detector=None, wavelength=wl*1e-10)  # from A to m

    # set a mask
    # create a mask
    detector_size_y = data.shape[0]
    detector_size_x = data.shape[1]
    mask = np.zeros([detector_size_y, detector_size_x])
    y_n = beamstopper_coordinates[0]
    y_p = beamstopper_coordinates[1]
    x_n = beamstopper_coordinates[2]
    x_p = beamstopper_coordinates[3]
    mask[y_n:y_p, x_n:x_p] = 1


    # remove the edge lines around the detector
    #left edge
    mask[:, 0:4] = 1
    # right edge
    mask[:, -3:detector_size_x+1] = 1
    # bottom
    mask[0:1, :] = 1
    # top
    mask[detector_size_y-1:detector_size_y+1, :] = 1

    masked_data = np.multiply(data, np.logical_not(mask).astype(int))
    ax1.imshow(np.log(masked_data), interpolation='none', cmap='jet', origin='lower')
    ax1.plot(int(bc_x), int(bc_y), 'rx', markersize = 10 )

    ai.setChiDiscAtZero()

    q, I, sigma = ai.integrate1d(data, 100,
                                 correctSolidAngle = False,
                                 mask = mask,
                                 method = 'nosplit_csr',
                                 unit = 'q_A^-1',
                                 safe = True,
                                 error_model="azimuthal", # "poisson" or "azimuthal",
                                 flat = None,
                                 dark = None)
    ax2.loglog(q, I)
    ax2.set_ylabel('Intensity I [a.u.]')
    ax2.set_xlabel('Scattering vector q [Angstrom]')
    ax2.grid()
    ax2.title.set_text(title_c)

    plt.savefig(save_figures + str(list_scan[jj]) + '_' + sample_name + '.jpg')
    plt.close('all')
