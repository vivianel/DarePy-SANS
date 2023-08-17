# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 14:06:55 2023

@author: lutzbueno_v
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import os
import pickle

path_dir_an = 'C:/Users/lutzbueno_v/Documents/Analysis/SANS-darep/analysis_test/'
path_merged = 'C:/Users/lutzbueno_v/Documents/Analysis/SANS-darep/analysis_test/merged'
path_merged_txt = path_merged +  '/data_txt/'

file_name = os.path.join(path_dir_an, 'config.npy')
with open(file_name, 'rb') as handle:
    config = pickle.load(handle)
calibration = config['experiment']['calibration']


files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path_merged_txt):
    for file in f:
        if 'interpolated' in file:
            files.append(os.path.join(file))

for ii in range(0, len(files)):
    if files[ii][13:-4] not in calibration.values():
        plt.close('all')
        plt.ioff()
        print (files[ii][13:-4])
        file_name =  path_merged_txt + files[ii]

        I = np.genfromtxt(file_name,
                             dtype = None,
                             delimiter = ',',
                             usecols = 1)
        q = np.genfromtxt(file_name,
                             dtype = None,
                             delimiter = ',',
                             usecols = 0)

        diff_I = np.diff(I)
        off_set = np.median(np.abs(diff_I))/5
        print(off_set)
        select_diff = np.where(np.abs(diff_I) < off_set )
        # slip the first points
        select_diff = select_diff[0]
        fitting_I = I[select_diff[0]:-2]
        fitting_q = q[select_diff[0]:-2]

        def porod(q, coef, slope, incoherent):
            return (coef * q**(slope-4) + incoherent)

        # perform the fit
        params, cv = scipy.optimize.curve_fit(porod, fitting_q, fitting_I)
        m, t, b = params
        incoherent = b
        slope = t
        coeff = m

        # determine quality of the fit
        squaredDiffs = np.square(fitting_I - porod(fitting_q, m, t, b))
        squaredDiffsFromMean = np.square(fitting_I - np.mean(fitting_I))
        rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
        rSquared = round(rSquared, 2)


        # plot the results
        plt.loglog(q, I, '.', label="original data", color = 'blue', alpha = 0.2)
        plt.loglog(fitting_q, porod(fitting_q, m, t, b), '--', color = 'blue', label= f"fit - R² = {rSquared}")
        plt.title("Fitted Porod Approximation")
        plt.loglog(q, np.ones(len(I))*incoherent, '--', color = 'red', label = 'incoherent = ' + str(round(incoherent, 2)))
        plt.loglog(q, coeff * q**(slope-4), '--', color = 'black', label = 'slope = ' + str(round(slope-4, 2)))
        plt.loglog(q, I-incoherent, '.', color = 'black', label = 'subtracted data')
        plt.legend()

        path_merged_fig = path_merged +  '/figures/'
        file_name = path_merged_fig + files[ii][13:-4] + '_subtracted' + '.jpeg'
        plt.savefig(file_name)

        header_text = 'q (A-1), I (1/cm)'
        file_name = path_merged_txt + 'subtracted_'  + files[ii][13:-4] + '.dat'
        data_save = np.column_stack((q, I-incoherent))
        np.savetxt(file_name, data_save, delimiter=',', header=header_text)
