# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 10:26:35 2025

@author: gruene_e
"""

import numpy as np

def make_thickness_dict(csv_file) :
    data_file = np.loadtxt(csv_file, delimiter=',')
    thickness_dict = {'scanno':[], 'thickness_cm':[]}
    
    for i, (scan, thickness) in enumerate(zip(data_file[:,0], data_file[:,2])):
        thickness_dict['scanno'].append(int(scan))
        thickness_dict['thickness_cm'].append(thickness)
        
    for i, (scan, thickness) in enumerate(zip(data_file[:,1], data_file[:,2])):
        if int(scan) not in thickness_dict['scanno']:
            thickness_dict['scanno'].append(int(scan))
            thickness_dict['thickness_cm'].append(thickness)
            
    return thickness_dict

def make_trans_dict(csv_file):
    data_file = np.loadtxt(csv_file, delimiter=',')
    trans_dict = {'scat_scan':[], 'trans_scan':[], 'thickness_cm':[]}
    for i, (scat_scan, trans_scan, thickness) in enumerate(zip(data_file[:,0], data_file[:,1], data_file[:,2])):
        trans_dict['scat_scan'].append(int(scat_scan))
        trans_dict['trans_scan'].append(int(trans_scan))
        trans_dict['thickness_cm'].append(thickness)
        
    return trans_dict
    