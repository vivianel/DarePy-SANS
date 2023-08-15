import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import organize_hdf_files as org
from load_hdf import load_hdf
import corrections as corr

#def select_transmission(path_dir, path_hdf_raw, class_files, trans_dist, empty_beam):
def select_transmission(config, class_files, result):
    #generate the analysis folder
    path_hdf_raw = config['analysis']['path_hdf_raw']
    trans_dist = config['experiment']['trans_dist']
    empty_beam = config['experiment']['calibration']['empty_beam']
    path_dir_an = org.create_analysis_folder(config)

    #create a folder
    path_transmission = os.path.join(path_dir_an, 'transmission/')
    if not os.path.exists(path_transmission):
        os.mkdir(path_transmission)

    list_trans = list(class_files.keys())
    class_trans = {key: [] for key in list_trans}

    for ii in range(0, len(class_files['att'])):
         if (class_files['att'][ii] > 0 and class_files['detx_m'][ii] == trans_dist and class_files['time_s'][ii] > 0) or (class_files['att'][ii] > 0 and class_files['sample_name'][ii] == empty_beam and class_files['time_s'][ii] > 0):
             for iii in list_trans:
                 class_trans[iii].append(class_files[iii][ii])
             #save a copy of the transmission files
             source = os.path.join(path_hdf_raw, class_files['name_hdf'][ii])
             destination = os.path.join(path_transmission, 'hdf_raw/')
             if not os.path.exists(destination):
                 os.mkdir(destination)
             shutil.copyfile(source, destination + class_files['name_hdf'][ii])
    org.save_list_files(path_transmission, path_dir_an, class_trans, 'trans_files', result)
    return class_trans

def correct_input_trans(config, result, hdf_name, counts):
    counts = corr.normalize_time(config, hdf_name, counts)
    counts = corr.deadtime_corrections(config, hdf_name, counts)
    #counts = corr.normalize_flux(config, hdf_name, counts)
    counts = corr.correct_attenuator(config, hdf_name, counts)
    counts, sigma = corr.normalize_thickness(config, hdf_name, result, counts, 0)
    return counts

def trans_calc_reference(config, result):
    path_dir_an = org.create_analysis_folder(config)

    #find transmission folder
    path_transmission = os.path.join(path_dir_an, 'transmission/')

    # open the list for all
    class_trans  = result['overview']['trans_files']
    empty_beam = config['experiment']['calibration']['empty_beam']
    trans_dist = config['experiment']['trans_dist']

    # calculate the number of counts of the EB measurement
    for ii in range(0, len(class_trans['sample_name'])):
        if class_trans['sample_name'][ii] == empty_beam and class_trans['detx_m'][ii] == trans_dist:
            path_hdf_raw = os.path.join(path_transmission, 'hdf_raw/')
            scan_nr = class_trans['scan'][ii]
            hdf_name = class_trans['name_hdf'][ii]
            counts = load_hdf(path_hdf_raw, hdf_name, 'counts')
            img = correct_input_trans(config, result, hdf_name, counts)


            # calculation of the cuttoff value for the mask
            cutoff = 1.5*img[img>0].mean()
            plt.figure()
            plt.imshow(img, clim=[0, round(cutoff)], cmap='jet', origin='lower')
            plt.colorbar(orientation = 'vertical', shrink = 0.5).set_label('log(Intensity)')
            im_title = 'Empty_beam, Scan: ' + str(scan_nr)
            plt.title(im_title)
            im_title = str(path_transmission + 'Empty_beam.jpg')
            plt.savefig(im_title)

            # prepare the mask
            img1 = np.where(img<cutoff, 0, img)
            mask = np.where(img1>=cutoff, 1, img1)
            plt.figure()
            plt.imshow(mask, cmap='gray', origin='lower')
            plt.colorbar(orientation = 'vertical', shrink = 0.5, ticks=[0, 1]).set_label('Binary')
            im_title = 'Transmission_Mask, Scan: ' + str(scan_nr)

            # define the reference value
            EB_ref = int(np.sum(np.multiply(img,mask)))
            print(EB_ref)

            plt.title(im_title +  ', Total counts = ' + str(EB_ref))
            im_title = str(path_transmission + 'mask.jpg')
            plt.savefig(im_title)
    plt.close('all')
    result['transmission']['mask'] = mask
    result['transmission']['mean_EB'] = EB_ref
    result['transmission']['EB_counts'] = img
    org.save_results(path_dir_an, result)
    return result



def trans_calc(config, result):

    path_dir_an = org.create_analysis_folder(config)
    #find transmission folder
    path_transmission = os.path.join(path_dir_an, 'transmission/')
    #%% calculate all transmissions
    list_trans = []
    list_counts = []
    mask =  result['transmission']['mask']
    EB_ref = result['transmission']['mean_EB']
    class_trans = result['overview']['trans_files']
    trans_dist = config['experiment']['trans_dist']

    path_hdf_raw = os.path.join(path_transmission, 'hdf_raw/')
    for ii in range(0, len(class_trans['sample_name'])):
        hdf_name = class_trans['name_hdf'][ii]
        counts = load_hdf(path_hdf_raw, hdf_name , 'counts')
        img = correct_input_trans(config, result, hdf_name, counts)

        sum_counts = int(np.sum(np.multiply(img, mask)))
        list_counts.append(sum_counts)

        if class_trans['detx_m'][ii] == trans_dist:
            #print('This is sample '+ class_trans['name'][ii] + ' transmission:' + str(counts))
            trans = np.divide(sum_counts,EB_ref)
            trans = round(trans, 3)
            list_trans.append(trans)
            print(trans)
        else:
            trans = 1
            list_trans.append(trans)

    class_trans['transmission'] = list_trans
    class_trans['counts'] = list_counts

    result['overview']['trans_files']['transmission'] = list_trans
    result['overview']['trans_files'] ['counts'] = list_counts
    # print the updated list of the transmission files
    org.save_list_files(path_transmission, path_dir_an, class_trans, 'trans_files', result)


    # save the transmission of the samples in the all_file
    list_trans = []
    class_all = result['overview']['all_files']
    for ii in range(0, len(class_all['sample_name'])):
        if class_all['sample_name'][ii] in class_trans['sample_name']:
            idx_trans = list(class_trans['sample_name']).index(str(class_all['sample_name'][ii]))
            list_trans.append(class_trans['transmission'][idx_trans])
        else:
            list_trans.append('--')
    class_all['transmission'] = list_trans
    # print the updated list of the transmission files
    org.save_list_files(path_dir_an, path_dir_an, class_all, 'all_files', result)
    return result
