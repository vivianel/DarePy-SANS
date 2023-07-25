# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 15:09:54 2022

@author: lutzbueno_v
"""

# empty beam
def prepare_ai(file_beam_center, class_file, path_hdf_raw, path_rad_int_fig, det, idx):
    import h5py
    import numpy as np
    import pyFAI
    from calculate_beam_center import calculate_center
    
   
    #open the file for the beam center and radial integrator
    name_hdf = path_hdf_raw + '/'+ class_file['name_hdf'][idx]
    file_hdf = h5py.File(name_hdf, 'r')
    
# we measured until now with the wrong detector offset (5 mm)              
    dist = (float(np.asarray(file_hdf['/entry1/SANS/detector/x_position'])))/1000
    pixel1 = 7.5e-3 # detector size is 7.5 x 7.5 mm²
    pixel2 = 7.5e-3 # detector size is 7.5 x 7.5 mm²
    wavelength = (float(np.asarray(file_hdf['/entry1/SANS/Dornier-VS/lambda'])))/1e9 # in m from nm
    
   # if file_beam_center in class_file['name']:
        # calculate the beam center
    img = np.array(file_hdf['entry1/SANS/detector/counts'])
    file_name = path_rad_int_fig + 'beam_center_' + class_file['scan'][idx] + '_' + class_file['name'][idx] + '_' + det + 'm.jpeg' 
    bc_x, bc_y = calculate_center(img, file_name)
    
        
    poni2 = bc_x*pixel1
    poni1 = bc_y*pixel2

    # create a m
    mask = np.zeros([128, 128])
    # find the size of the beam stopper
    beam_stop = (float(np.asarray(file_hdf['/entry1/SANS/beam_stop/out_flag'])))
    list_bs = {'1':40, '2':70,'3':85,'4':100} # in cm
    beam_stopper = round(((list_bs[str(int(beam_stop))])/(pixel1*1000))/2)
    beam_stopper = beam_stopper + 0.5
    if float(det.replace('p','.')) < 2:
         beam_stopper = beam_stopper + 2
    # remove those pixels around the beam stopper
    mask[round(bc_y-beam_stopper):round(bc_y+beam_stopper), round(bc_x-beam_stopper):round(bc_x+beam_stopper)] = 1
    # remove the pixels around the detector
    # to cover reflection
    #if det == '18p0':
    #    mask[round(bc_y-beam_stopper):round(bc_y+beam_stopper), round(bc_x):round(bc_x+20)] = 1
    mask[:,0:3] = 1
    mask[:,125:129] = 1
    mask[0:3,:] = 1
    mask[125:129,:] = 1
    # remove the edges
    corner = 10
    mask[0:corner,0:corner] = 1
    mask[-corner:-1,0:corner] = 1
    mask[-corner:-1,-corner:-1] = 1
    mask[0:corner,-corner:-1] = 1
    
    #create the radial integrator
    ai = pyFAI.AzimuthalIntegrator(dist=dist, poni1=poni1, poni2=poni2,
     rot1=0, rot2=0, rot3=0,
     pixel1=pixel1, pixel2=pixel2,
     splineFile=None, detector=None, wavelength=wavelength)
    #print("\nIntegrator: \n", ai)

    return(ai, mask)
    

        


