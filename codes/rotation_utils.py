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

def make_EC_dict(csv_file):
    data_file = np.loadtxt(csv_file, delimiter=',')
    EC_dict = {}
    for (scat_scan, EC_scat) in zip(data_file[:,0], data_file[:,2]):
        EC_dict[int(scat_scan)] = int(EC_scat)
        
    return EC_dict
    
def find_empty_cell(scat_scan, config, result):
    EC_dict = make_EC_dict(config['analysis']['empty_cell_table'])
    
    if scat_scan in EC_dict :
        empty_cell_scan = EC_dict[scat_scan]
        if empty_cell_scan in result['integration']['other_empty_cells']:
            empty_cell_img = result['integration']['other_empty_cells'][empty_cell_scan]
            print(f'Scan {scat_scan} will be corrected by empty cell {empty_cell_scan}.')
        else:
            empty_cell_img = result['integration']['empty_cell']
            print(f'Scan {scat_scan} will be corrected by default empty cell.')
    else :
        empty_cell_img = result['integration']['empty_cell']
        print(f'Scan {scat_scan} will be corrected by default empty cell.')
    
    return empty_cell_img
    
