# -*- coding: utf-8 -*-
"""
DarePy-SANS: Transmission Calculation
Calculates transmission scalars using the Virtual Referencing architecture.
Saves logs directly to the analysis root.
"""
import numpy as np
import os
import sys
import csv
from tabulate import tabulate

# Importing unified flexible logic directly from utils
from utils import load_hdf, create_analysis_folder, save_results, get_flexible_value, find_hdf_by_identifier
from prepare_input import save_list_files
import normalize_counts as norm

def trans_calc(config, class_files, result):
    instrument = config['instrument']['name']
    if instrument == 'SANS-I' or (instrument == 'SANS-LLB' and config['experiment']['trans_dist'] > 0):
        result = select_transmission(config, class_files, result)
        result = trans_calc_reference(config, result, class_files)
        result = trans_calc_sample(config, result)
    elif instrument == 'SANS-LLB' and config['experiment']['trans_dist'] < 0:
        result = trans_calc_reference(config, result, class_files)
    return result

def trans_calc_reference(config, result, class_files):
    path_dir_an = create_analysis_folder(config)
    path_hdf_raw = config['analysis']['path_hdf_raw']

    instrument = config['instrument']['name']
    coordinates = config['analysis']['transmission_coordinates']

    eb_block = config.get('calibration_samples', {}).get('empty_beam', 'EB')
    eb_id = get_flexible_value(eb_block, 'default', default_fallback='EB')

    if instrument == 'SANS-I' or (instrument == 'SANS-LLB' and config['experiment']['trans_dist'] > 0):
        class_trans = result['overview']['trans_files']
        trans_dist = config['experiment']['trans_dist']

        eb_hdf = find_hdf_by_identifier(eb_id, class_trans)

        if eb_hdf is not None:
            idx = class_trans['name_hdf'].index(eb_hdf)
            if class_trans['detx_m'][idx] == trans_dist:
                counts = load_hdf(path_hdf_raw, eb_hdf, 'counts')
                img = normalize_trans(config, result, eb_hdf, counts)

                cutoff = img[img > 0].mean()
                img1 = np.where(img < cutoff, 0, img)
                mask = np.where(img1 >= cutoff, 1, img1)

                Factor_correction = 1
                EB_ref = float(np.sum(np.multiply(img,mask))) * Factor_correction

                result['transmission']['mask'] = mask
                result['transmission']['mean_EB'] = EB_ref
                result['transmission']['EB_counts'] = img
        else:
            sys.exit(f'Please measure an empty beam ({eb_id}) for the same detector distance ({trans_dist}m).')

    elif instrument == 'SANS-LLB' and config['experiment']['trans_dist'] < 0:
        det_dist = list(set(class_files['detx_m']))
        for jj in det_dist:
            class_dist = {k: [] for k in class_files.keys()}
            for ii in range(len(class_files['detx_m'])):
                if class_files['detx_m'][ii] == jj:
                    for k in class_files.keys():
                        class_dist[k].append(class_files[k][ii])

            eb_hdf = find_hdf_by_identifier(eb_id, class_dist)

            if eb_hdf is not None:
                counts = load_hdf(path_hdf_raw, eb_hdf, 'counts')
                img = normalize_trans(config, result, eb_hdf, counts)

                mask = np.zeros_like(img)
                try:
                    c = coordinates[jj]
                except KeyError:
                    sys.exit(f'Check coordinates for EB ({eb_id}) at distance {jj}m.')

                mask[c[0]:c[1], c[2]:c[3]] = 1
                Factor_correction = 1
                EB_ref = float(np.sum(np.multiply(img,mask))) * Factor_correction

                result['transmission']['mask_' + str(jj)] = mask
                result['transmission']['mean_EB_' + str(jj)] = EB_ref
                result['transmission']['counts_EB_' + str(jj)] = img

    save_results(path_dir_an, result)
    return result

def trans_calc_sample(config, result):
    path_dir_an = create_analysis_folder(config)
    path_hdf_raw = config['analysis']['path_hdf_raw']

    list_trans = []
    list_counts = []
    mask = result['transmission']['mask']
    EB_ref = result['transmission']['mean_EB']

    class_trans = result['overview']['trans_files']
    trans_dist = config['experiment']['trans_dist']
    eb_block = config.get('calibration_samples', {}).get('empty_beam', 'EB')

    # --- LOG BOOK SETUP ---
    trans_log = []
    log_headers = ["Scan", "Sample", "Det_m", "EB_Used", "Trans_Counts", "Transmission"]

    for ii in range(len(class_trans['sample_name'])):
        hdf_name = class_trans['name_hdf'][ii]
        sample_name = class_trans['sample_name'][ii]
        scanNr = class_trans['scan'][ii]
        det_m = class_trans['detx_m'][ii]

        counts = load_hdf(path_hdf_raw, hdf_name , 'counts')
        img = normalize_trans(config, result, hdf_name, counts)

        sum_counts = float(np.sum(np.multiply(img, mask)))
        list_counts.append(sum_counts)

        eb_id = get_flexible_value(eb_block, sample_name, default_fallback='EB')
        eb_hdf = find_hdf_by_identifier(eb_id, class_trans)

        current_log = [scanNr, sample_name, det_m, eb_id, f"{sum_counts:.2e}"]

        if class_trans['detx_m'][ii] == trans_dist and hdf_name != eb_hdf:
            trans = np.divide(sum_counts, EB_ref)
            list_trans.append(round(trans, 3))
            current_log.append(f"{trans:.3f}")
        else:
            list_trans.append(1)
            current_log.append("REF (1.000)" if hdf_name == eb_hdf else "1.000")

        trans_log.append(current_log)

    class_trans['transmission'] = list_trans
    class_trans['counts'] = list_counts
    result['overview']['trans_files']['transmission'] = list_trans
    result['overview']['trans_files']['counts'] = list_counts

    # --- PRINT AND SAVE LOG DIRECTLY IN ANALYSIS FOLDER ---
    print("\n" + "="*80)
    print("TRANSMISSION LOG")
    print("="*80)
    print(tabulate(trans_log, headers=log_headers, tablefmt="grid"))

    # SAVED DIRECTLY IN path_dir_an
    log_file = os.path.join(path_dir_an, "transmission_log.csv")
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(log_headers)
        writer.writerows(trans_log)
    print(f"\n[INFO] Transmission log saved to: {log_file}")

    # Map back to all_files
    list_trans_all = []
    class_all = result['overview']['all_files']
    for ii in range(len(class_all['sample_name'])):
        if class_all['sample_name'][ii] in class_trans['sample_name']:
            idx_trans = list(class_trans['sample_name']).index(str(class_all['sample_name'][ii]))
            list_trans_all.append(class_trans['transmission'][idx_trans])
        else:
            list_trans_all.append('--')

    class_all['transmission'] = list_trans_all
    save_list_files(path_dir_an, path_dir_an, class_all, 'all_files', result)
    return result

def normalize_trans(config, result, hdf_name, counts):
    counts = norm.normalize_deadtime(config, hdf_name, counts)
    counts = norm.normalize_flux(config, hdf_name, counts)
    counts = norm.normalize_attenuator(config, hdf_name, counts)
    return counts

def select_transmission(config, class_files, result):
    path_dir_an = create_analysis_folder(config)
    trans_dist = config['experiment']['trans_dist']

    if not trans_dist or not isinstance(trans_dist, (int, float)):
        return result

    list_keys = list(class_files.keys())
    class_trans = {key: [] for key in list_keys}

    for ii in range(len(class_files['att'])):
         if (class_files['att'][ii] > 0 and
             class_files['detx_m'][ii] == trans_dist and
             class_files['time_s'][ii] > 0 and
             class_files['beamstop_y'][ii] < -30):

             for k in list_keys:
                 class_trans[k].append(class_files[k][ii])

    result['overview']['trans_files'] = class_trans
    return result
