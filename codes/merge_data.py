# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 20:50:57 2022

@author: lutzbueno_v

This function automatically marges the data collected in different detector distances
"""



# this path should be updated to where the data is being analyzed located in your computer
path_dir = 'C:/Users/lutzbueno_v/Documents/Analysis/SANS-darep/'

# %% function


import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

import pylab
pylab.ioff()


path_dir_an = os.path.join(path_dir, 'analysis/')
list_dir = list(os.listdir(path_dir_an))

path_merged = os.path.join(path_dir_an, 'merged/')
if not os.path.exists(path_merged):
    os.mkdir(path_merged)

path_merged_fig = os.path.join(path_merged, 'figures/')
if not os.path.exists(path_merged_fig):
    os.mkdir(path_merged_fig)

path_merged_txt = os.path.join(path_merged, 'data_txt/')
if not os.path.exists(path_merged_txt):
    os.mkdir(path_merged_txt)


files = {'files':[], 'det':[],'idx':[], 'sample_name':[], 'q':[], 'I':[], 'qc':[], 'Ic':[], 'sigma':[], 'sigmac':[], 'a':[], 'b':[]}
for kk in list_dir:
    if kk[0:3] == 'det':
        det = kk[4:]
        # load results
        file_name = os.path.join(path_dir_an, 'result.npy')
        with open(file_name, 'rb') as handle:
            result = pickle.load(handle)
        class_file = result['overview']['det_files' + str(kk[3:])]
        files['files'].append(class_file)
        files['det'].append(det)

det_ls = []
for kk in range(0, len(files['det'])):
    det = files['det'][kk]
    det_ls.append(float(det.replace('p','.')))
order = np.argsort(det_ls)

for kk in  range(0, len(files['det'])):
    files['a'].append(files['det'][order[kk]])
    files['b'].append(files['files'][order[kk]])
for kk in  range(0, len(files['det'])):
    files['det'][kk] = files['a'][kk]
    files['files'][kk] = files['b'][kk]

del files['a']
del files['b']

# %%
# define the longest name
large = 0
for kk in range(0, len(files['files'])):
    temp = len(files['files'][kk]['sample_name'])
    if temp > large:
        large = kk



for jj in range(0, len(files['files'][large]['sample_name'])):
    name_0 = str(files['files'][large]['sample_name'][jj])
    idx = []
    name_list = []
    for kk in range(0, len(files['det'])):
        if name_0 in files['files'][kk]['sample_name']:
            name_find = files['files'][kk]['sample_name'].index(name_0)
            idx.append(name_find)
            name_list.append((files['files'][kk]['sample_name'][name_find]))
        else:
            idx.append('--')
            name_list.append('--')
    files['idx'].append(idx)
    files['sample_name'].append(name_list)

# %%
for kk in range(0, len(files['idx'])):
    I = []
    q = []
    sigma = []
    plt.close('all')
    for jj in range(0, len(files['idx'][kk])):
        path_det = os.path.join(path_dir_an, 'det_'+ files['det'][jj])
        path_rad_int = os.path.join(path_det, 'integration/')

        ii = files['idx'][kk][jj]
        sample_name = files['sample_name'][kk][jj]
        ScanNr = files['files'][jj]['scan'][ii]
        Frame = 0

        if ii == '--':
            q.append([])
            I.append([])
            sigma.append([])
        else:
            file_name = path_rad_int + 'radial_integ_' + sample_name + '_det' + files['det'][jj] + f"m_{ScanNr:07d}" + '_' + f"{Frame:04d}.dat"
            #print(file_name)
            file_dat = np.genfromtxt(file_name, delimiter=',')
            q.append(file_dat[:,0])
            I.append(file_dat[:,1])
            sigma.append(file_dat[:,2])
            plt.errorbar(file_dat[:,0], file_dat[:,1], yerr =file_dat[:,2], lw = 0.3, marker = 'o',  ms = 2, label = str(files['files'][jj]['scan'][ii]) + '  ' + files['files'][jj]['sample_name'][ii] + ' ' + files['det'][jj])
    plt.xlabel(r'Scattering vector q [$\AA^{-1}$]')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'Intensity I [cm$^{-1}$]')
    plt.title('Sample: '+  files['sample_name'][kk][jj])
    plt.legend()
    file_name = path_merged_fig + sample_name + '_all_det_dist' + '.jpeg'
    plt.savefig(file_name)
    files['q'].append(q)
    files['I'].append(I)
    files['sigma'].append(sigma)



# %%

files['Ic'] = files['I']
files['qc'] = files['q']
files['sigmac'] = files['sigma']

scale = 0

if scale == 1:
    for kk in range(0, len(files['idx'])):
        min_q = []
        max_q = []
        plt.close('all')

        for jj in range(0, len(files['q'][kk])):
            if len(files['q'][kk][jj]) != 0:
                min_q.append(np.amin(files['q'][kk][jj]))
                max_q.append(np.amax(files['q'][kk][jj]))

            else:
                min_q.append(0)
                max_q.append(0)


        if len(files['q'][kk][0]) != 0 and len(files['q'][kk][1]) != 0:
            if len(files['q'][kk][0]) != 0:
                plt.loglog(files['q'][kk][0], files['I'][kk][0], 'ok', label = 'base', ms = 2)
                files['Ic'][kk][0] = files['I'][kk][0]
            idx0 = np.where(np.round(files['q'][kk][1], 2) == (round(min_q[0], 2)))
            idx0 = round(idx0[0][0] + (idx0[0][-1]-idx0[0][0])/2, 1)
            idx0 = 5#int(idx0)
            idx1 = np.where(np.round(files['q'][kk][0], 2) == (round(max_q[1], 2)))
            idx1 = round(idx1[0][0] + (idx1[0][-1]-idx1[0][0])/2, 1)
            idx1 = 5#int(idx1)
            mean_01 = np.median(files['I'][kk][0][0:idx1])
            mean_10 = np.median(files['I'][kk][1][idx0:])
            plt.loglog(files['q'][kk][1], files['I'][kk][1], 'r+', label = 'original', ms = 1)
            files['Ic'][kk][1] = files['I'][kk][1]*mean_01/mean_10
            files['sigmac'][kk][1] = files['sigma'][kk][1]*mean_01/mean_10
            plt.loglog(files['qc'][kk][1], files['Ic'][kk][1], 'or', label = 'x ' + str(round(mean_01/mean_10,4)), ms = 2)

        try:
            if len(files['q'][kk][1]) != 0 and len(files['q'][kk][2]) != 0:
                if len(files['q'][kk][0]) == 0 and len(files['q'][kk][1]) != 0:
                    plt.loglog(files['q'][kk][1], files['I'][kk][1], 'ok', label = 'base', ms = 2)
                    files['Ic'][kk][1] = files['I'][kk][1]
                idx2 = np.where(np.round(files['q'][kk][2], 2) == (round(min_q[1], 2)))
                idx2 = round(idx2[0][0] + (idx2[0][-1]-idx2[0][0])/2, 1)
                idx2 = 30#int(idx2)
                idx3 = np.where(np.round(files['q'][kk][1], 2) == (round(max_q[2], 2)))
                idx3 = round(idx3[0][0] + (idx3[0][-1]-idx3[0][0])/2, 1)
                idx3 = 5#int(idx3)
                mean_12 = (files['Ic'][kk][1][0] - files['Ic'][kk][1][idx3])/2
                mean_21 = np.median(files['I'][kk][2][idx2:])
                mean_21 = (files['I'][kk][2][idx2] - files['I'][kk][2][-5])/2
                plt.loglog(files['q'][kk][2], files['I'][kk][2], 'b+', label = 'original', ms = 1)
                files['Ic'][kk][2] = files['I'][kk][2]*mean_12/mean_21
                files['sigmac'][kk][2] = files['sigma'][kk][2]*mean_12/mean_21
                plt.loglog(files['qc'][kk][2], files['Ic'][kk][2], 'ob', label = 'x ' + str(round((mean_12/mean_21)/(mean_01/mean_10) ,4)), ms = 2)
        except:
             print('')
        for ii in files['name'][kk]:
            if ii != '--':
                name = ii
        plt.xlabel(r'Scattering vector q [$\AA^{-1}$]')
        plt.ylabel(r'Intensity I [cm$^{-1}$]')
        plt.title('Sample: '+  name)
        plt.legend()
        file_name = path_merged_fig + name + '_merged'+ '.jpeg'
        plt.savefig(file_name)


# %%
for kk in range(0, len(files['idx'])):
    q_all = []
    I_all = []
    sigma_all = []
    plt.close('all')
    for jj in range(0, len(files['I'][kk])):
        try:
            if jj ==2 and len(files['qc'][kk][jj]) > 0:
                rg= range(0,len(files['qc'][kk][jj])-10)
            elif jj < 2 and len(files['qc'][kk][jj]) > 0:
                rg = range(10, len(files['qc'][kk][jj]))

            q_all = np.concatenate((q_all, files['qc'][kk][jj][rg]), axis=None)
            I_all = np.concatenate((I_all, files['Ic'][kk][jj][rg]), axis=None)
            sigma_all = np.concatenate((sigma_all, files['sigma'][kk][jj][rg]), axis=None)
        except:
            q_all = np.concatenate((q_all, files['qc'][kk][jj][0:-1]), axis=None)
            I_all = np.concatenate((I_all, files['Ic'][kk][jj][0:-1]), axis=None)
            sigma_all = np.concatenate((sigma_all, files['sigma'][kk][jj][0:-1]), axis=None)
    idx = np.argsort(q_all)
    q = np.sort(q_all)
    I = I_all[idx]
    sigma = sigma_all[idx]
    plt.errorbar(q, I, yerr = sigma, color = 'red', lw = 0.3, label = 'merged', ms = 2, marker = 'o')
    plt.xscale('log')
    plt.yscale('log')
    baseline = I[-60:-1]
    baseline = sorted(baseline)
    baseline = np.array(baseline[0:2])
    baseline = np.mean(baseline[baseline > 0])
    # to correct the lack of measurements at 2m
    if kk >=45:
        baseline = baseline
    baseline_plt = np.ones([len(I)])*baseline
    plt.loglog(q, baseline_plt, '--r', label = 'incoherent = ' + str(round(baseline, 4)))

    I_corr = I-baseline*0.9
    sigma_corr = np.sqrt(np.square(np.divide(sigma, baseline)))
    sigma_corr = sigma_corr/100
    plt.errorbar(q, I_corr, yerr = sigma_corr, color = 'black', lw = 0.3, label = 'subtracted', ms = 4, marker = 'o')
    plt.legend()
    plt.xlabel(r'Scattering vector q [$\AA^{-1}$]')
    plt.ylabel(r'Intensity I [cm$^{-1}$]')
    for ii in files['sample_name'][kk]:
        if ii != '--':
            name = ii
    plt.title('Sample: '+  name)
    file_name = path_merged_fig +  name + '_subtracted'+ '.jpeg'
    plt.savefig(file_name)

    file_name = path_merged_txt + 'subtracted_'  + name + '.dat'
    data_save = np.column_stack((q, I_corr, sigma_corr))
    header_text = 'This file contains: q in Angstroms, absolute intensity with incoherent backgroun subtrated in 1/cm, and corrected standard deviation.'
    np.savetxt(file_name, data_save, delimiter=',', header=header_text)

    file_name = path_merged_txt + 'subtracted_noSigma_'  + name + '.dat'
    data_save = np.column_stack((q, I_corr))
    header_text = 'This file contains: q in Angstroms, and absolute intensity with incoherent backgroun subtrated in 1/cm.'
    np.savetxt(file_name, data_save, delimiter=',', header=header_text)

    file_name = path_merged_txt + 'corrected_'  + name + '.dat'
    data_save = np.column_stack((q, I, sigma))
    header_text = 'This file contains: q in Angstroms, absolute intensity in 1/cm, and standard deviation.'
    np.savetxt(file_name, data_save, delimiter=',', header=header_text)

    file_name = path_merged_txt + 'corrected_noSigma_'  + name + '.dat'
    data_save = np.column_stack((q, I))
    header_text = 'This file contains: q in Angstroms, and absolute intensity in 1/cm.'
    np.savetxt(file_name, data_save, delimiter=',', header=header_text)



pylab.ion()
