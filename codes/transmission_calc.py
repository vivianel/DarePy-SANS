# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 14:50:44 2022

@author: lutzbueno_v
"""
#path_dir = 'C:/Users/Sinquser/Documents/Python_Scripts/VLB/pefi/'

# sample to consider for the creation of the transmission mask
#mask_measurement = 'EB'


def trans_calc(path_dir, mask_measurement, trans_dist):
    import h5py
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import pandas as pd
    from tabulate import tabulate
    from contextlib import redirect_stdout
    import json
    from normalize_time import normalize_time 
    
    plt.close('all')
    path_dir_an = os.path.join(path_dir, 'analysis/')
    
    # open transmission file
    path_transmission = os.path.join(path_dir_an, 'transmission/')
    class_trans = open(os.path.join(path_transmission, 'trans_files.json'))
    class_trans = eval(class_trans.read())
    
    # open the list for all
    class_all  = open(os.path.join(path_dir_an, 'all_files.json'))  
    class_all = eval(class_all.read())
    #%% find the reference value
    for ii in range(0, len(class_trans['name'])):
        if class_trans['name'][ii] == mask_measurement and class_trans['det'][ii] == trans_dist:
            path_hdf_raw = os.path.join(path_transmission, 'hdf_raw/')
            name_hdf = path_hdf_raw + '/'+ class_trans['name_hdf'][ii]
            file_hdf = h5py.File(name_hdf, 'r')
            img = np.array(file_hdf['entry1/SANS/detector/counts'])
            
            #img1 = normalize_time(file_hdf, img)
            img1 = img
            # calculation of the cuttoff value for the mask
            cutoff = 1.5*img1[img1>0].mean()
            plt.figure()
            plt.imshow(img1, clim=[0, round(cutoff)], cmap='jet', origin='lower')
            plt.colorbar(orientation = 'vertical', shrink = 0.5).set_label('log(Intensity)')
            im_title = 'Empty_beam'
            plt.title(im_title)
            im_title = str(path_transmission + im_title + '.jpg')
            plt.savefig(im_title)
            img2 = np.where(img1<cutoff, 0, img1)
            mask = np.where(img2>=cutoff, 1, img2)
            plt.figure()
            plt.imshow(mask, cmap='gray', origin='lower')
            plt.colorbar(orientation = 'vertical', shrink = 0.5, ticks=[0, 1]).set_label('Binary')
            im_title = 'Transmission_Mask'
            EB_ref = int(np.sum(np.multiply(img1,mask)))
            #print('###########################################################')
            #print('This is the EB transmission:' + str(EB_ref))
            #print('###########################################################')
            plt.title(im_title +  ', Total counts = ' + str(EB_ref))
            im_title = str(path_transmission + im_title + '.jpg')
            plt.savefig(im_title)
            
            
            
            
    #%% calculate all transmissions
    list_trans = []
    list_counts = []
    path_hdf_raw = os.path.join(path_transmission, 'hdf_raw/')
    for ii in range(0, len(class_trans['name'])):
            name_hdf = path_hdf_raw + '/'+ class_trans['name_hdf'][ii]
            file_hdf = h5py.File(name_hdf, 'r')
            img = np.array(file_hdf['entry1/SANS/detector/counts'])
            
            #img = normalize_time(file_hdf, img)
            
            counts = int(np.sum(np.multiply(img,mask)))
            list_counts.append(counts)
            if class_trans['det'][ii] == trans_dist:
                #print('This is sample '+ class_trans['name'][ii] + ' transmission:' + str(counts))
                trans = np.divide(counts,EB_ref)
                trans = round(trans, 3)
                list_trans.append(trans)
                #print(trans)
            else:
                trans = 1
                list_trans.append(trans)
                
    class_trans['transmission'] = list_trans
    class_trans['counts'] = list_counts
    # print the updated list of the transmission files
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
    
    
    # save the transmission of the samples in the all_file
    list_trans = []
    for ii in range(0, len(class_all['name'])):
        if class_all['name'][ii] in class_trans['name']: 
            idx_trans = list(class_trans['name']).index(str(class_all['name'][ii]))
            list_trans.append(class_trans['transmission'][idx_trans])
        else:
            list_trans.append('--')    
    class_all['transmission'] = list_trans
    # print the updated list of the transmission files
    df_trans = pd.DataFrame(class_all)
    data = tabulate(df_trans, headers='keys', tablefmt='psql')
    #print(data)
    save_file = os.path.join(path_dir_an, 'all_files.txt')
    with open(save_file, 'w') as f:
        with redirect_stdout(f):
            print(data)
    save_file = os.path.join(path_dir_an, 'all_files.json')
    a_file = open(save_file, "w")
    json.dump(class_all, a_file)
    a_file.close()
    plt.close('all')
   