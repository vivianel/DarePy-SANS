# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 11:15:45 2022

@author: lutzbueno_v
"""
def list_calib(file_id, class_det):
    import numpy as np
    
    if file_id in class_det['name']:
        #open the file for the beam center and radial integrator
        array = np.array(class_det['name'])
        indices = np.where(array == file_id)[0]
        scan_nr = []
        for jj in indices:
            scan_nr.append(int(class_det['scan'][jj]))
        print('Calibration: ' + file_id + ', Scan: ' + str(scan_nr))
        return(scan_nr)
    else:
        print('Calibration: ' + file_id + ', Scan: MISSING')
        return