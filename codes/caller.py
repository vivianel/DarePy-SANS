# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 13:31:39 2023

@author: lutzbueno_v
"""
import organize_hdf_files as org
import calculate_transmission as trans
import radial_integration as ri


path_hdf_raw = 'C:/Users/lutzbueno_v/Documents/Analysis/SANS-darep/raw_data/'
path_dir = 'C:/Users/lutzbueno_v/Documents/Analysis/SANS-darep/'


instrument = 'SANS-I'
#these calibrants are neeeded
cadmium = 'Cd'
water = 'H2O'
water_cell = 'EC'
empty_cell = 'EC'
empty_beam = 'EB'
beam_center = 'EB'

exclude_files = [ ] #+ list(range(23177, 28000)) # we need to solve the transit position problem, they should be all lists

perform_abs_calib = 1
perform_azimuthal = 1
trans_dist = 18 #use negative values to deactivate
perform_radial = 1
force_reintegrate = 1
replace_h2o_18m = 4.5 # it needs the .0
plot_azimuthal = 1
plot_radial = 1
save_plots = 1

calibration = {'cadmium':cadmium, 'water':water, 'water_cell': water_cell, 'empty_cell':empty_cell, 'empty_beam':empty_beam, 'beam_center':beam_center}

result = {'transmission':{},
       'overview':{},
       'integration':{}
       }


configuration = {'SANS-I':{'instrument': {'deadtime': 6.6e-7, 'list_attenuation': {'0':1, '1':1/485,'2':1/88,'3':1/8, '4':1/3.5,'5':1/8.3}, 'pixel_size':7.5e-3, 'detector_size': 128, 'list_bs': {'1':40, '2':70,'3':85,'4':100}, 'list_abs_calib': {'0.5':0.909, '0.6':0.989, '0.8':1.090, '1.0':1.241, '1.2':1.452}},
                           'experiment': {'trans_dist': trans_dist, 'calibration':calibration},
                           'analysis': {'path_dir': path_dir, 'path_hdf_raw':path_hdf_raw, 'exclude_files':exclude_files, 'perform_abs_calib':perform_abs_calib, 'perform_azimuthal':perform_azimuthal, 'perform_radial':perform_radial, 'force_reintegrate': force_reintegrate, 'replace_h2o_18m':replace_h2o_18m, "plot_azimuthal":plot_azimuthal, "plot_radial":plot_radial, "save_plots":save_plots }},
          'SANS-LLB':{'instrument': {'deadtime':1e5},
                    'experiment': {},
                    'analysis': {}}}




class_files = org.list_files(configuration[instrument], result)

#select the transmission measurements if they are present. Otherwise, keep it with -1
if trans_dist > 0:
   res = trans.select_transmission(configuration[instrument], class_files, result)
   result = trans.trans_calc_reference(configuration[instrument], result)
   result = trans.trans_calc(configuration[instrument], result)
else:
    print('No transmission has been measured.')

result = org.select_detector_distances(configuration['SANS-I'], class_files, result)

result = ri.set_radial_int(configuration['SANS-I'], result)
