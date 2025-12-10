import numpy as np
import os
import shutil
from utils import load_hdf
from utils import create_analysis_folder
from utils import save_results
from prepare_input import save_list_files
import normalization as norm
import sys

def trans_calc(config, class_files, result):
    instrument = config['instrument']['name']
    if instrument == 'SANS-I':
        result = select_transmission(config, class_files, result)
        result = trans_calc_reference(config, result, class_files)
        result = trans_calc_sample(config, result)
    elif instrument == 'SANS-LLB':
        result = trans_calc_reference(config, result, class_files)
    return result


def trans_calc_reference(config, result, class_files):
    path_dir_an = create_analysis_folder(config)
    #find transmission folder
    instrument = config['instrument']['name']
    empty_beam = config['analysis']['empty_beam']
    coordinates = config['analysis']['transmission_coordinates']
    if instrument == 'SANS-I':
        path_transmission = os.path.join(path_dir_an, 'transmission/')
        # open the list for all
        class_trans  = result['overview']['trans_files']
        trans_dist = config['experiment']['trans_dist']
        # calculate the number of counts of the EB measurement
        array = np.array(class_trans['sample_name'])
        index = np.where(array == empty_beam)[0]
        if len(index) > 0 and class_trans['detx_m'][index[0]] == trans_dist:
            path_hdf_raw = os.path.join(path_transmission, 'hdf_raw/')
            hdf_name = class_trans['name_hdf'][index[0]]
            counts = load_hdf(path_hdf_raw, hdf_name, 'counts')
            img = normalize_trans(config, result, hdf_name, counts)
            # calculation of the cuttoff value for the mask
            cutoff = img[img > 0].mean()
            # prepare the mask for selecting the direct beam
            img1 = np.where(img < cutoff, 0, img)
            mask = np.where(img1 >= cutoff, 1, img1)
            # define the reference value from the empty beam
            # when we get a reference for another beamtime, we might need to adjust the transmission
            # knowing that water with 1 mm should have a transmission of around 0.5
            # factor to correct the transmission (if needed)
            Factor_correction = 1
            EB_ref = float(np.sum(np.multiply(img,mask)))*Factor_correction
            # save the reference value
            result['transmission']['mask'] = mask
            result['transmission']['mean_EB'] = EB_ref
            result['transmission']['EB_counts'] = img
        else:
            sys.exit('Please measure an empty beam (EB) for the same detector distance, for calculating the relative transmission. \n Or change the trans_dist < 0 in "caller_radial_integration" for not correcting by transmission.')
    elif instrument == 'SANS-LLB':
        # list all files in the
        det_dist = list(set(class_files['detx_m']))
        for jj in det_dist:
            for ii in range(0, len(class_files['detx_m'])):
                if class_files['detx_m'][ii] == jj and class_files['sample_name'][ii] == empty_beam:
                    path_hdf_raw = config['analysis']['path_hdf_raw']
                    hdf_name = class_files['name_hdf'][ii]
                    counts = load_hdf(path_hdf_raw, hdf_name, 'counts')
                    img = normalize_trans(config, result, hdf_name, counts)
                    #img = counts
                    mask = np.zeros_like(img)
                    try:
                        c = coordinates[jj]
                    except:
                        sys.exit('Please measure an empty beam (EB) for the same detector distance, for calculating the relative transmission. \n Or change the trans_dist < 0 in "caller_radial_integration" for not correcting by transmission.')
                    mask[c[0]:c[1], c[2]:c[3]]  = 1
                    # define the reference value from the empty beam
                    # when we get a reference for another beamtime, we might need to adjust the transmission
                    # knowing that water with 1 mm should have a transmission of around 0.5
                    # factor to correct the transmission (if needed)
                    Factor_correction = 1
                    EB_ref = float(np.sum(np.multiply(img,mask)))*Factor_correction
                    name_m = 'mask_' + str(jj)
                    # save the reference value
                    result['transmission'][name_m] = mask
                    name_EB = 'mean_EB_' + str(jj)
                    result['transmission'][name_EB] = EB_ref
                    name_EB = 'counts_EB_' + str(jj)
                    result['transmission'][name_EB] = img

    save_results(path_dir_an, result)


    # #some plotting for checking the results
    # plt.figure()
    # plt.imshow(img, cmap='jet', origin='lower')
    # plt.colorbar(orientation = 'vertical', shrink = 0.5).set_label('Intensity (counts)')
    # #im_title = 'Empty_beam_Intensity, cutoff = ' + str(round(cutoff)) + ', Scan: ' + str(scan_nr)
    # #plt.title(im_title)
    # im_title = str(path_transmission + 'Empty_beam.jpg')
    # plt.savefig(im_title)

    # plt.figure()
    # plt.imshow(mask, cmap='gray', origin='lower')
    # plt.colorbar(orientation = 'vertical', shrink = 0.5, ticks=[0, 1]).set_label('Binary')
    # im_title = 'Transmission_Mask, Scan: ' + str(scan_nr)
    # plt.title(im_title +  ', Total counts = ' + str(EB_ref))
    # im_title = str(path_transmission + 'transmission_mask.jpg')
    # plt.savefig(im_title)
    # plt.close('all')
    return result




def trans_calc_sample(config, result):
    path_dir_an = create_analysis_folder(config)
    #find transmission folder
    path_transmission = os.path.join(path_dir_an, 'transmission/')
    # calculate all transmissions
    list_trans = []
    list_counts = []
    mask =  result['transmission']['mask']
    EB_ref = result['transmission']['mean_EB']
    empty_beam = config['analysis']['empty_beam']
    class_trans = result['overview']['trans_files']
    trans_dist = config['experiment']['trans_dist']
    path_hdf_raw = os.path.join(path_transmission, 'hdf_raw/')
    for ii in range(0, len(class_trans['sample_name'])):
        hdf_name = class_trans['name_hdf'][ii]
        counts = load_hdf(path_hdf_raw, hdf_name , 'counts')
        img = normalize_trans(config, result, hdf_name, counts)
        sum_counts = float(np.sum(np.multiply(img, mask)))
        list_counts.append(sum_counts)
        if class_trans['detx_m'][ii] == trans_dist and class_trans['sample_name'][ii] != empty_beam:
            trans = np.divide(sum_counts,EB_ref)
            trans = round(trans, 3)
            list_trans.append(trans)
            #print('This is sample '+ class_trans['name'][ii] + ' transmission:' + str(trans))
        else:
            trans = 1
            list_trans.append(trans)
    class_trans['transmission'] = list_trans
    class_trans['counts'] = list_counts
    result['overview']['trans_files']['transmission'] = list_trans
    result['overview']['trans_files'] ['counts'] = list_counts
    # print the updated list of the transmission files
    save_list_files(path_transmission, path_dir_an, class_trans, 'trans_files', result)
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
    save_list_files(path_dir_an, path_dir_an, class_all, 'all_files', result)
    return result


def normalize_trans(config, result, hdf_name, counts):
    counts = norm.normalize_deadtime(config, hdf_name, counts)
    counts = norm.normalize_flux(config, hdf_name, counts)
    counts = norm.normalize_attenuator(config, hdf_name, counts)
    return counts

def select_transmission(config, class_files, result):
    #generate the analysis folder
    path_hdf_raw = config['analysis']['path_hdf_raw']
    path_dir_an = create_analysis_folder(config)
    trans_dist = config['experiment']['trans_dist']
    #create a folder to save the files of the tranmission - SANS-I
    path_transmission = os.path.join(path_dir_an, 'transmission/')
    if not os.path.exists(path_transmission):
        os.mkdir(path_transmission)
    # list all files in the
    list_trans = list(class_files.keys())
    class_trans = {key: [] for key in list_trans}
    for ii in range(0, len(class_files['att'])):
         if (class_files['att'][ii] > 0 and class_files['detx_m'][ii] == trans_dist and class_files['time_s'][ii] > 0) and class_files['beamstop_y'][ii] < -30:
             for iii in list_trans:
                 class_trans[iii].append(class_files[iii][ii])
             #save a copy of the transmission files
             source = os.path.join(path_hdf_raw, class_files['name_hdf'][ii])
             destination = os.path.join(path_transmission, 'hdf_raw/')
             if not os.path.exists(destination):
                 os.mkdir(destination)
             shutil.copyfile(source, destination + class_files['name_hdf'][ii])
    save_list_files(path_transmission, path_dir_an, class_trans, 'trans_files', result)

    return result
