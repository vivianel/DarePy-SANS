# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 15:30:07 2022

@author: lutzbueno_v
"""

def apply_corrections(file_hdf, img, img_cd,class_file, class_all, idx):
    from correct_transmission import correct_transmission
    from normalize_time import normalize_time 
    import numpy as np
    
    
    img = normalize_time(file_hdf, img)
    img = np.subtract(img,img_cd)
    img, trans = correct_transmission(img, class_file, class_all, idx)
    return(img, trans)