# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 13:31:39 2023

@author: lutzbueno_v
"""
# %% EXPERIMENTAL PARAMETERS
# select the instrument
instrument = 'SANS-I'
#these calibrants are neeeded to run the data reduction
# optional measurement of empty beam for transmission
empty_beam = 'EB'
# the following standards are required for data reduction
# cadmium is used for the dark field correction
cadmium = 'Cd'
# water and the empty container (EC) where it was measured is used for the flat field correction
water = 'H2O'
water_cell = 'EC'
# the empty cell container of the sample
empty_cell = 'EC'
# provide a dictionary with the list of the sample name and thickness in cm. Otherwise,
# it is assumed that the thickness is 0.1 cm. If all cells differ than 0.1 cm,
# but are equal to X, please indicate with {'all':X}, where x is the thickness.
sample_thickness = {'all':0.1}
# indicate the distance in meters where the transmission has been measured.
# if transmission correction is not needed, provide a negative value, such as -1
trans_dist = 18
# for the case of flat field correction at large detector distances, indicate which
# detctor distance to use instead in m, it should be "6.0" one
replace_18m = 6.0
# used wavelength
wl = 'auto' # in Angstrons or 'auto'
# add the size of the beamstopper as a dictionary: {'detector_distance':[size_x, size_y]} in pixels
beamstopper_coordinates = {'1.6':[56, 77, 50, 69], '6.0':[57, 72, 53, 66], '18.0':[57, 70, 53, 65]}
# add the guess for beamcenter as a dictionary: {'detector_distance':[center_x, center_y]}
beam_center_guess = {'1.6':[60.58, 67.63], '6.0':[56.41, 75.62], '18.0':[59.69, 62.8]}


# ANALYSIS PARAMETERS
# path where the raw hdf files are saved
path_hdf_raw = 'C:/Users/lutzbueno_v/Documents/Analysis/data/microfluidics/2022_2358_MF_1mm_rest/raw_data/'
# path to the working directory (where the analysis will be saved)
path_dir = 'C:/Users/lutzbueno_v/Documents/Analysis/data/microfluidics/2022_2358_MF_1mm_rest/DarePy-SANS/'
show_plots = 0
# id to the analysis folder. Use '' to avoid it
add_id = ''
# Scan numbers to be excluded from the analysis pipeline. They should be lists,
# such as: list(range(23177, 28000). If not needed keep it to empty [].
exclude_files = []
# perform_radial and plot_radial = 1 to integrate, plot, and save the results.
plot_radial = 1
# perform_azimuthal and plot_azimuthal = 1 to integrate, plot, and save the results.
plot_azimuthal = 0
# perform_abs_calib = 1 to perform the absolute calibration of the data
# perform_abs_calib = 0 to deactivate
perform_abs_calib = 1
# force_reintegrate = 1 the radial integration will run again for all files
# if force_reintegrate = 0, only the new files will be integrated
force_reintegrate = 1
# integration_points defines how many q values the final curve will contain
integration_points = 100
# number of sectors for azimuthal integration
sectors_nr = 16
pixel_range_azim = range(5,40)



# run for starting the data analysis pipeline
import os
os.chdir(path_dir + '/codes/')

import prepare_input as org
from transmission import trans_calc
import integration as ri


# prepare a dictionary with the names of the samples for calibration
calibration = {'cadmium':cadmium, 'water':water, 'water_cell': water_cell, 'empty_cell':empty_cell}

result = {'transmission':{},
       'overview':{},
       'integration':{
           'pixel_range_azim':pixel_range_azim,
           'integration_points':integration_points,
           'sectors_nr': sectors_nr}}
           #'npt_azim':npt_azim}


configuration = {'SANS-I':{
    'instrument': {'deadtime': 6.6e-7,
                   'list_attenuation': {'0':1, '1':1/485,'2':1/88,'3':1/8, '4':1/3.5,'5':1/8.3},
                   'pixel_size':7.5e-3,
                   'detector_size': 128,
                   'list_bs': {'1':40, '2':70,'3':85,'4':100},
                   'list_abs_calib': {'5':0.909, '6':0.989, '8':1.090, '10':1.241, '12':1.452}},
    'experiment': {'trans_dist': trans_dist,
                   'calibration':calibration,
                   'sample_thickness':sample_thickness,
                   'wl_input': wl},
    'analysis': {'path_dir': path_dir,
                 'path_hdf_raw':path_hdf_raw,
                 'show_plots':show_plots,
                 'exclude_files':exclude_files,
                 'perform_abs_calib':perform_abs_calib,
                 'force_reintegrate': force_reintegrate,
                 'replace_18m':replace_18m,
                 "plot_azimuthal":plot_azimuthal,
                 "plot_radial":plot_radial,
                 'add_id':add_id,
                 'empty_beam':empty_beam,
                 'beam_center_guess': beam_center_guess,
                 'beamstopper_coordinates': beamstopper_coordinates,
                 }},
                  'SANS-LLB':{
    'instrument': {'deadtime':1e5},
                    'experiment': {},
                    'analysis': {}}}

# %% STEP 1: load all files
config = configuration[instrument]
class_files = org.list_files(config, result)

# %% STEP 2: calculate transmission

#select the transmission measurements if they are present. Otherwise, keep it with -1
if trans_dist > 0:
    trans_calc(config, class_files, result)
else:
    print('No transmission has been measured.')

# %% STEP 3: divide into different detector distances
result = org.select_detector_distances(config, class_files, result)


# %% STEP 4: radial integration
result = ri.set_integration(config, result)
