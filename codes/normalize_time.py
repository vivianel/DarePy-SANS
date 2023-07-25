# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 12:38:56 2022

@author: lutzbueno_v
"""

def normalize_time(file_hdf, img1):
    import numpy as np
    from deadtime_corrections import deadtime_corrections
    
    # normalize by detector deadtime
    meas_time = round(float(np.asarray(file_hdf['/entry1/SANS/detector/counting_time'])), 2)
    img1 = np.where(img1<=0, 1e-8, img1)
    
    
    #import scipy.ndimage as ndimage
    #img1 = ndimage.gaussian_filter(img1, sigma=0.5, mode = 'nearest')
    
    img2 = deadtime_corrections(img1,meas_time)
    
    # normalize by detector time
    img2 = (img2/meas_time)
    
     # normalize by monitor2, it was tested that it depends on the wavelength
    #flux_mon = round(float(np.asarray(file_hdf['/entry1/SANS/monitor2/counts'])), 2)
    #img2 = (img2/flux_mon)
    
    # correct by attenuation
    attenuator = (int(np.asarray(file_hdf['/entry1/SANS/attenuator/selection'])))
    #load params for attenuation
    list_attenuation = {'0':1, '1':1/485,'2':1/88,'3':1/8, '4':1/3.5,'5':1/8.3}
    # search for attenuation
    idx_att = list(list_attenuation).index(str(attenuator))
    #correct
    img2 = img2 / float(list_attenuation[str(idx_att)])
    
   

    return (img2) 