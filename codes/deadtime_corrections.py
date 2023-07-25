# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 13:46:38 2022

@author: lutzbueno_v
"""
def deadtime_corrections(I_detector, measurement_time):
    import numpy as np
    
    #I_detector = 0
    detector_deadtime_sansI = 6.6e-7 # in s
    total_counts = np.sum(I_detector) 
    #measurement_time = 0
    
    I_det_corr = (I_detector/(1-(detector_deadtime_sansI/measurement_time)*total_counts))
    
    return I_det_corr