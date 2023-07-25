# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 09:11:44 2022

@author: lutzbueno_v
"""

#issues:
#what if I dont have transmission?
#check for missing measurements
#check for different parameters

# path where the raw data is saved: usually AFS
#path_dir_raw = 'C:/Users/Sinquser/Documents/Python_Scripts/VLB/pefi/raw_data/'
# path where to save the data: working directory
#path_dir = 'C:/Users/Sinquser/Documents/Python_Scripts/VLB/pefi/'
# distance where transmission has been calculated
#trans_dist = 18 # in m
# files to exclude from measurements (scan numbers)
#exclude_files =  list(range(41663, 41688, 1)) + [41645, 41840]

def list_files(path_dir_raw, path_dir, trans_dist, exclude_files):
    
    #########################################################################
    import os
    import h5py
    import numpy as np
    import shutil
    import pandas as pd
    from tabulate import tabulate
    from contextlib import redirect_stdout
    import json
    from list_calibrations import list_calib
    
    # find all files in the folder
    files = []
    path_dir_an = os.path.join(path_dir, 'analysis/')
    if not os.path.exists(path_dir_an):
        os.mkdir(path_dir_an)
    path_hdf_raw = path_dir_raw #os.path.join(path_dir, 'raw_data/')
    
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path_hdf_raw):
        for file in f:
            if '.hdf' in file:
                files.append(os.path.join(file))
    
    #print the list
    #for f in files:
    #    print(f)
    
    # classify measurements
    class_files = {'order':[],'scan':[], 'name':[],'att':[], 'coll':[], 'wavelength':[],
                   'det':[], 'det_y':[],  'moni':[], 'time':[], 'name_hdf':[ ]}    #'temp':[],           
    for ii in range(0, len(files)):
    #for ii in range(0,10):
        if int(files[ii][9:-4]) not in exclude_files:
            name_hdf = os.path.join(path_hdf_raw, files[ii])
            file_hdf = h5py.File(name_hdf, 'r')
            class_files['order'].append(ii)
            class_files['name_hdf'].append(files[ii])
            class_files['scan'].append(files[ii][9:-4])
            print(files[ii][9:-4])
            class_files['att'].append(float(np.asarray(file_hdf['entry1/SANS/attenuator/selection'])))
            class_files['coll'].append(round(float(np.asarray(file_hdf['/entry1/SANS/collimator/length'])), 1))
            class_files['time'].append(round(float(np.asarray(file_hdf['/entry1/SANS/detector/counting_time'])), 2))
            class_files['moni'].append(round(float(np.asarray(file_hdf['/entry1/SANS/detector/preset'])), 2))
            #class_files['temp'].append(round(float(np.asarray(file_hdf['/entry1/sample/temperature'])), 2))
            class_files['det'].append(round(float(np.asarray(file_hdf['/entry1/SANS/detector/x_position']))/1000, 1)) # in m
            class_files['det_y'].append(round(float(np.asarray(file_hdf['/entry1/SANS/detector/y_position']))/1000, 2)) # in m
            class_files['wavelength'].append((float(np.asarray(file_hdf['/entry1/SANS/Dornier-VS/lambda'])))/1e8)
            try:
                name = str(np.asarray(file_hdf['/entry1/sample/name_new']))
                class_files['name'].append(name[2:-1])
            except:
                name = str(np.asarray(file_hdf['/entry1/sample/name']))
                class_files['name'].append(name[3:-2])
            
    # save list of files    
    df = pd.DataFrame(class_files)
    data = tabulate(df, headers='keys', tablefmt='psql')
    print(data)
    save_file = os.path.join(path_dir_an, 'all_files.txt')
    with open(save_file, 'w') as f:
        with redirect_stdout(f):
            print(data)
    save_file = os.path.join(path_dir_an, 'all_files.json')
    a_file = open(save_file, "w")
    json.dump(class_files, a_file)
    a_file.close()
    # %%
    #select the transmission measurements
    if trans_dist > 0:
        path_transmission = os.path.join(path_dir_an, 'transmission/')
        list_trans = list(class_files.keys())
        class_trans = {key: [] for key in list_trans}
        for ii in range(0, len(class_files['att'])):
             if class_files['scan'][ii] not in exclude_files:
                if (class_files['att'][ii] > 0 and class_files['det'][ii] == trans_dist and class_files['time'][ii] > 0) or (class_files['att'][ii] > 0 and class_files['name'][ii] == 'EB' and class_files['time'][ii] > 0):
                    if not os.path.exists(path_transmission):
                        os.mkdir(path_transmission)
                    source = os.path.join(path_hdf_raw, class_files['name_hdf'][ii])
                    destination = os.path.join(path_transmission, 'hdf_raw/')
                    if not os.path.exists(destination):
                        os.mkdir(destination)
                    shutil.copyfile(source, destination+class_files['name_hdf'][ii])
                    for iii in list_trans:
                        class_trans[iii].append(class_files[iii][ii])
        
        # print the list of the transmission files
        df_trans = pd.DataFrame(class_trans)
        data = tabulate(df_trans, headers='keys', tablefmt='psql')
        #print(data)
        save_file = os.path.join(path_transmission, 'trans_files.txt')
        with open(save_file, 'w') as f:
            with redirect_stdout(f):
                print(data)
        save_file = os.path.join(path_transmission, 'trans_files.json')
        a_file = open(save_file, "w")
        json.dump(class_trans, a_file)
        a_file.close()
                
    # %%
    #select the different detector distances measurements
    
    # select the unique detector distance values
    unique_det = np.unique(class_files['det'])
    
    for jj in unique_det:
        string = str(jj)
        string = string.replace('.', 'p')
        path_det = os.path.join(path_dir_an, 'det_' + string +'/')
        list_det = list(class_files.keys())
        class_det = {key: [] for key in list_det}
        for ii in range(0, len(class_files['det'])):
            if class_files['scan'][ii] not in exclude_files:
                if ((class_files['det'][ii] == jj) and class_files['att'][ii] <= 1 and class_files['time'][ii] > 0) or ((class_files['det'][ii] == jj) and class_files['name'][ii] == 'EB' and class_files['time'][ii] > 0) :
                    if not os.path.exists(path_det):
                        os.mkdir(path_det)
                    source = os.path.join(path_hdf_raw, class_files['name_hdf'][ii])
                    destination = os.path.join(path_det, 'hdf_raw/')
                    if not os.path.exists(destination):
                        os.mkdir(destination)
                    shutil.copyfile(source, destination+class_files['name_hdf'][ii])
                    for iii in list_det:
                        class_det[iii].append(class_files[iii][ii])
        print('%')
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('For sample-detector distance: ' + string + 'm')
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        # print the calibration files
        file_id = 'Cd'
        list_calib(file_id, class_det)
        list_calib('EC', class_det)
        list_calib('EB', class_det)
        list_calib('H2O', class_det)
        print('(Optional:)')
        list_calib('Mask', class_det)
        list_calib('D2O', class_det)
        
            
        
        # print the list of detector files
        df_det = pd.DataFrame(class_det)
        data = tabulate(df_det, headers='keys', tablefmt='psql')
        #print(data)
        save_file = os.path.join(path_det, 'det_files_'+ string + 'm.txt')
        with open(save_file, 'w') as f:
            with redirect_stdout(f):
                print(data)
        save_file = os.path.join(path_det, 'det_files_'+ string + 'm.json')
        a_file = open(save_file, "w")
        json.dump(class_det, a_file)
        a_file.close()
                
