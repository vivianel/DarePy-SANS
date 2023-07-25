# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 12:29:42 2022

@author: lutzbueno_v
"""
def correct_transmission(det_img, class_file, class_all, ii):
    # correct transmission
    if class_file['scan'][ii] in class_all['scan']: 
        idx_trans = list(class_all['scan']).index(str(class_file['scan'][ii]))
        trans = class_all['transmission'][idx_trans]
    else:
        trans = 0
    if (type(trans) == int or type(trans) == float):
        if trans > 0:
            print('Sample ' + class_file['name'][ii]+ ' is corrected by Transmission = ' + str(trans))
            det_img = det_img/trans
    return(det_img, trans)