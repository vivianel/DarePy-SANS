# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 12:38:56 2022

@author: lutzbueno_v
"""

import numpy as np
from utils import load_hdf


def normalize_time(config, hdf_name, counts):
    # normalize by detector deadtime
    path_hdf_raw = config['analysis']['path_hdf_raw']
    meas_time = load_hdf(path_hdf_raw, hdf_name, 'time')
    #counts = np.where(counts <=0, 1e-8, counts)
    # normalize by detector time
    counts = (counts/meas_time)
    return counts


def normalize_deadtime(config, hdf_name, counts):
    detector_deadtime = config['instrument']['deadtime'] # in s
    total_counts = np.sum(counts)
    path_hdf_raw = config['analysis']['path_hdf_raw']
    meas_time = load_hdf(path_hdf_raw, hdf_name, 'time')
    counts = (counts/(1-(detector_deadtime/meas_time)*total_counts))
    return counts

def normalize_flux(config, hdf_name, counts):
    # normalize by monitor2, it was tested that it depends on the wavelength
    path_hdf_raw = config['analysis']['path_hdf_raw']
    flux_mon = load_hdf(path_hdf_raw, hdf_name, 'flux_monit')
    counts = (counts/flux_mon)
    return counts

def normalize_attenuator(config, hdf_name, counts):
    # correct by attenuation
    path_hdf_raw = config['analysis']['path_hdf_raw']
    attenuator = int(load_hdf(path_hdf_raw, hdf_name, 'att'))
    #load params for attenuation
    list_attenuation = config['instrument']['list_attenuation']
    # search for attenuation
    idx_att = list(list_attenuation).index(str(attenuator))
    #correct
    counts = counts / float(list_attenuation[str(idx_att)])
    return counts


def normalize_transmission(config, hdf_name, result, counts):
    # correct transmission
    class_trans = result['overview']['all_files']
    if hdf_name in class_trans['name_hdf']:
        idx_trans = list(class_trans['name_hdf']).index(str(hdf_name))
        trans = class_trans['transmission'][idx_trans]
        if isinstance(trans, float) and trans > 0:
            counts = counts/trans
        print('%%')
        print('Sample ' + class_trans['sample_name'][idx_trans]+ ' is corrected by Transmission = ' + str(trans))
    else:
        print('No transmission found, and data not corrected!')
    return counts


def normalize_thickness(config, hdf_name, result, counts):
    # correct transmission
    class_all = result['overview']['all_files']
    if hdf_name in class_all['name_hdf']:
        idx = list(class_all['name_hdf']).index(str(hdf_name))
        sample_name = class_all['sample_name'][idx]
    list_thickness = config['experiment']['sample_thickness']
    if sample_name in list_thickness:
        thickness = list_thickness[sample_name]
        counts = counts/thickness
    elif 'all' in list_thickness:
        thickness = list_thickness['all']
        counts = counts/thickness
    else:
        thickness = 0.1
        counts = counts/thickness
    #print('Sample'+ sample_name +' corrected by thickness = ' + str(thickness) + ' mm')
    return counts
