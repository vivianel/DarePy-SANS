# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 10:21:49 2023

@author: lutzbueno_v
"""
import os
import re
import numpy as np
from tabulate import tabulate
from contextlib import redirect_stdout
import shutil
import pickle
from utils import load_hdf
from utils import create_analysis_folder
from utils import save_results


def list_files(config, result):
    # create a list for containing all the measurements
    # classify measurements with the following info
    class_files = {'name_hdf':[ ], 'scan':[], 'sample_name':[],'att':[], 'beamstop_y':[], 'coll_m':[], 'wl_A':[],
                    'detx_m':[], 'dety_m':[],  'moni_e4':[], 'time_s':[], 'thickness_cm':[], 'frame_nr':[], 'temp_C':[]}
    path_hdf_raw = config['analysis']['path_hdf_raw']
    exclude_files = config['analysis']['exclude_files']
    # find all hdf files in the path_hdf_raw folder
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path_hdf_raw):
        f.sort()
        for file in f:
            if '.hdf' in file:
                files.append(os.path.join(file))
    # go through all measured data, check wheter it is a scan or not
    for ii in range(0, len(files)):
        # exclude some of the files if needed
        scan_nr = re.findall(r"\D(\d{6})\D", files[ii])
        scan_nr = int(scan_nr[0])
        if int(scan_nr) not in exclude_files:
            print('Loading scan ' + str(scan_nr) + ',  Status: ' + str(round(ii/len(files)*100, 2)) + ' %')
            class_files['name_hdf'].append(files[ii])
            class_files['scan'].append(scan_nr)
            class_files['att'].append(load_hdf(path_hdf_raw, files[ii], 'att'))
            class_files['beamstop_y'].append(load_hdf(path_hdf_raw, files[ii], 'beamstop_y'))
            class_files['coll_m'].append(load_hdf(path_hdf_raw, files[ii], 'coll'))
            class_files['time_s'].append(load_hdf(path_hdf_raw, files[ii], 'time'))
            class_files['moni_e4'].append(load_hdf(path_hdf_raw, files[ii], 'moni'))
            class_files['temp_C'].append(load_hdf(path_hdf_raw, files[ii], 'temp'))
            class_files['detx_m'].append(load_hdf(path_hdf_raw, files[ii], 'detx'))
            class_files['dety_m'].append(load_hdf(path_hdf_raw, files[ii], 'dety'))
            class_files['wl_A'].append(load_hdf(path_hdf_raw, files[ii], 'wl'))
            class_files['sample_name'].append(load_hdf(path_hdf_raw, files[ii], 'sample_name'))
            res = load_hdf(path_hdf_raw, files[ii], 'counts')
            if res.ndim > 2:
                class_files['frame_nr'].append(res.shape[0])
            else:
                class_files['frame_nr'].append(1)
            # save sample thickness
            sample_name = class_files['sample_name'][-1]
            list_thickness = config['experiment']['sample_thickness']
            if sample_name in list_thickness:
                thickness = list_thickness[sample_name]
                class_files['thickness_cm'].append(thickness)
            elif 'all' in list_thickness:
                thickness = list_thickness['all']
                class_files['thickness_cm'].append(thickness)
            else:
                thickness = 0.1
                class_files['thickness_cm'].append(thickness)
    # save list of files as text
    path_dir_an = create_analysis_folder(config)
    save_list_files(path_dir_an, path_dir_an, class_files, 'all_files', result)
    save_file = os.path.join(path_dir_an, 'config.npy')
    # Store data (serialize)
    with open(save_file, 'wb') as handle:
        pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return class_files


# print the list and save files
# name is'trans_files.json'
def save_list_files(path_save, path_dir_an, class_files, name, result):
    data = tabulate(class_files, headers='keys', tablefmt='psql')
    save_file = os.path.join(path_save, name + '.txt')
    with open(save_file, 'w') as f:
        with redirect_stdout(f):
            print(data)
    result['overview'][name] = class_files
    save_results(path_dir_an, result)


def  select_detector_distances(config, class_files, result):
    #select the different detector distances measurements
    calibration = config['experiment']['calibration']
    path_hdf_raw = config['analysis']['path_hdf_raw']
    # select the unique detector distance values
    unique_det = np.unique(class_files['detx_m'])
    #generate the analysis folder
    path_dir_an = create_analysis_folder(config)

    # select for the detector distances
    for jj in unique_det:
        string = str(jj)
        string = string.replace('.', 'p')
        path_det = os.path.join(path_dir_an, 'det_' + string +'/')
        list_det = list(class_files.keys())
        class_det = {key: [] for key in list_det}
        for ii in range(0, len(class_files['detx_m'])):
            if not os.path.exists(path_det):
                os.mkdir(path_det)
            source = os.path.join(path_hdf_raw, class_files['name_hdf'][ii])
            destination = os.path.join(path_det, 'hdf_raw/')
            if not os.path.exists(destination):
                os.mkdir(destination)
            # I change to select the files based on the position of the BS
            if (class_files['detx_m'][ii] == jj and class_files['beamstop_y'][ii] > -30 and class_files['time_s'][ii] > 0):
                shutil.copyfile(source, destination+class_files['name_hdf'][ii])
                for iii in list_det:
                    class_det[iii].append(class_files[iii][ii])
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('For sample-detector distance: ' + string + 'm')
        # print the calibration files
        for kk in calibration.values():
            list_calib(kk, class_det)
        save_list_files(path_det, path_dir_an, class_det, 'det_files_' + string, result)
    return result

def list_calib(file_id, class_det):
    if file_id in class_det['sample_name']:
        #open the file for the beam center and radial integrator
        array = np.array(class_det['sample_name'])
        indices = np.where(array == file_id)[0]
        scan_nr = []
        for jj in indices:
            scan_nr.append(int(class_det['scan'][jj]))
        print('     Calibration: ' + file_id + ', Scan: ' + str(scan_nr))
        return(scan_nr)
    else:
        print('     Calibration: ' + file_id + ', Scan: MISSING')
        return
