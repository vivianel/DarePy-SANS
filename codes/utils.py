# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 09:23:28 2023

@author: lutzbueno_v
"""


import numpy as np
import h5py
import os
import math
import pickle



# functions to load various values from hdf files

def load_hdf(path_hdf_raw, hdf_name, which_property):
    name_hdf = os.path.join(path_hdf_raw, hdf_name)
    # open the hdf files
    file_hdf = h5py.File(name_hdf, 'r')
    # those are only scalars
    if which_property == 'beamstop_y':
        prop = file_hdf['entry1/SANS/beam_stop/y_position'][0]
        res = check_dimension(prop)
    if which_property == 'att':
        prop = file_hdf['entry1/SANS/attenuator/selection'][0]
        res = check_dimension(prop)
    elif which_property == 'coll':
        prop = file_hdf['/entry1/SANS/collimator/length'][0]
        res = check_dimension(prop) # in m
    elif which_property == 'detx':
        prop = file_hdf['/entry1/SANS/detector/x_position'][0]
        res = check_dimension(prop)/1000 # convert from mm to m
        res = round(res, 2)
    elif which_property == 'dety':
        prop = file_hdf['/entry1/SANS/detector/y_position'][0]
        res = check_dimension(prop)/1000 # convert from mm to m
        res = round(res, 2)
    elif which_property == 'wl':
        prop = file_hdf['/entry1/SANS/Dornier-VS/lambda'][0]
        res = check_dimension(prop)*10 # convert from nm to A
    elif which_property == 'abs_time':
        prop = file_hdf['/entry1/control/absolute_time'][0]
        res = check_dimension(prop)
    elif which_property == 'spos':
        prop = file_hdf['/entry1/sample/position'][0]
        res = check_dimension(prop)
    elif which_property == 'flux_monit':
        prop = file_hdf['/entry1/SANS/monitor2/counts'][0]
        res = check_dimension(prop)
    elif which_property == 'beam_stop':
        res = file_hdf['/entry1/SANS/beam_stop/out_flag'][0]
    elif which_property == 'sample_name':
        try:
            prop = file_hdf['/entry1/sample/name_new']
            res = prop.asstr()[()]
        except:
            prop = file_hdf['/entry1/sample/name'][0]
            res = check_dimension(prop)
    # those values can be arrays
    if which_property == 'time':
        prop = np.asarray(file_hdf['/entry1/SANS/detector/counting_time'])
        res = check_dimension(prop)  # in s
    elif which_property == 'moni':
        prop = np.asarray(file_hdf['/entry1/SANS/detector/preset'])
        res = check_dimension(prop)/1e4 #to have monitors as 1e4
    elif which_property == 'temp': # read in C
        try:
            prop = np.asarray(file_hdf['/entry1/sample/temperature'])
            if math.isnan(prop):
                res = ''
            else:
                res = check_dimension(prop)# in s
        except:
            res = ''
    #load the data
    if  which_property == 'counts':
        prop = np.array(file_hdf['entry1/SANS/detector/counts'])
        res = check_dimension(prop)
        # correction to avoid zeros and negative values
        #res[res <= 0] = np.median(res)
    file_hdf.close()
    return res

def check_dimension(prop):
    if isinstance(prop,np.bytes_):
        prop = prop.decode()
        if prop == '':
            prop = ''
        elif all(i.isdigit() for i in prop):
            prop = round(float(prop), 2)
        return prop
    elif isinstance (prop,np.int32):
        return round(float(prop), 2)
    elif isinstance (prop,np.float64):
        return round(prop, 2)
    elif isinstance (prop,np.float32):
        return round(prop, 2)
    elif isinstance (prop,np.ndarray):
        if prop.ndim == 1:
            return round(float(np.mean(prop)),2)
        elif prop.ndim == 2:
            return np.float32(prop)
        elif prop.ndim > 2:
            return np.float32(prop)

def create_analysis_folder(config):
    add_id = config['analysis']['add_id']
    # create the analysis folder to save the results
    path_dir = config['analysis']['path_dir']
    if add_id != '':
        path_dir_an = os.path.join(path_dir, 'analysis_%s/' % add_id)
    else:
        path_dir_an = os.path.join(path_dir, 'analysis/')
    if not os.path.exists(path_dir_an):
        os.mkdir(path_dir_an)
    return path_dir_an

def save_results(path_save, result):
    # also save as json
    save_file = os.path.join(path_save, 'result.npy')
    # Store data (serialize)
    with open(save_file, 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return result

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
