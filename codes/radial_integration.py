# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 09:11:44 2022

@author: lutzbueno_v
"""

#path_dir = 'C:/Users/Sinquser/Documents/Python_Scripts/VLB/pefi/'
#file_beam_center = 'EB' #usually we use the empty beam

def radial_integ(path_dir_raw, path_dir, file_beam_center, perform_abs_calib, trans_dist, perform_anisotropy,  force_reintegrate, use_detector):
    #load the data
    import h5py
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import sys
    from prepare_ai import prepare_ai
    from apply_corrections import apply_corrections
    from normalize_time import normalize_time 
    
    
    plt.close('all')
    # find all files in the folder
    path_dir_an = os.path.join(path_dir, 'analysis/')
    list_dir = list(os.listdir(path_dir_an))
    
    # open the list for all
    class_all  = open(os.path.join(path_dir_an, 'all_files.json'))  
    class_all = eval(class_all.read())
    
    if trans_dist > 0:
        path_transmission = os.path.join(path_dir_an, 'transmission/')
        class_trans = open(os.path.join(path_transmission, 'trans_files.json'))
        class_trans = eval(class_trans.read())
        
    counter = 0
    for kk in list_dir:
        if kk[0:3] == 'det':
            det = kk[4:]
            path_det = os.path.join(path_dir_an, str(kk))
            file_name = os.path.join(path_det, 'det_files_'+ det + 'm.json')
            class_file = open(file_name)
            class_file = eval(class_file.read())
            class_file
            #  create poni and masks
            path_rad_int = os.path.join(path_det, 'integration/')
            if not os.path.exists(path_rad_int):
                os.mkdir(path_rad_int)
            # integrate
            path_rad_int_fig = os.path.join(path_det, 'figures/')
            if not os.path.exists(path_rad_int_fig):
                os.mkdir(path_rad_int_fig)    
            
            # check if we want to integrate
            last_file = path_rad_int + 'rad_integ_' + class_file['scan'][-1] + '_' + class_file['name'][-1] + '_' +kk[4:] + 'm.dat'  
            
            if os.path.exists(last_file) and force_reintegrate == 0:
                print('All files are already integrated at ' + det +'m')
            else:
                # prepare radial integration files
                path_hdf_raw = os.path.join(path_det, 'hdf_raw/')
                path_trans = os.path.join(path_transmission, 'hdf_raw/')
                array = np.array(class_trans['name'])
                indices = np.where(array == file_beam_center)[0]
                for ll in indices:
                    if class_trans['det'][ll] == float(det.replace('p','.')) and class_trans['att'][ll] >0:
                       (ai, mask) = prepare_ai(file_beam_center, class_trans, path_trans, path_transmission, det, ll)
                if 'ai' not in locals():
                    print('###########################################################')
                    print('An Empty beam is needed for this configuration: ' + det + ' m' )
                    print('###########################################################')
                    sys.exit('Please measure an empty beam (EB).')
                
                    
                counter = counter +1
                print(counter, det)
                   
                             
                if perform_abs_calib == 1:
                    # open cadmium
                    file_cd = 'Cd'
                    if file_cd in class_file['name']:
                        #open the file for the beam center and radial integrator
                        idx = class_file['name'].index(file_cd)
                        name_hdf = path_hdf_raw + '/'+ class_file['name_hdf'][idx]
                        file_hdf = h5py.File(name_hdf, 'r')
                        # get the image
                        img_cd = np.array(file_hdf['entry1/SANS/detector/counts'])
                        img_cd = normalize_time(file_hdf, img_cd)
                       
                    else:
                        print('###########################################################')
                        print('There is no cadmium measurement for this configuration: ' + det + ' m' )
                        print('###########################################################')
                        sys.exit('Please load a cadmium (Cd) measurement.')
                        
                                    
                    # open empty cell
                    file_ec = 'EC'
                    if file_ec in class_file['name']:
                          #open the file for the beam center and radial integrator
                          idx = class_file['name'].index(file_ec)
                          name_hdf = path_hdf_raw + '/'+ class_file['name_hdf'][idx]
                          file_hdf = h5py.File(name_hdf, 'r')
                          # get the image
                          img_ec = np.array(file_hdf['entry1/SANS/detector/counts'])
                          img_ec, trans = apply_corrections(file_hdf, img_ec, img_cd,class_file, class_all, idx)
                        
                          
                    else:
                          print('###########################################################')
                          print('There is no empty cell measurement for this configuration: ' + det + ' m' )
                          print('###########################################################')
                          sys.exit('Please load a empty cell (EC) measurement.')
                          
                   
                    # open water
                    def read_water(path_hdf_raw_w, class_file_w, file_h2o):
                        idx = class_file_w['name'].index(file_h2o)
                        name_hdf = path_hdf_raw_w + '/'+ class_file_w['name_hdf'][idx]
                        file_hdf = h5py.File(name_hdf, 'r')
                        # get the image
                        img_h2o = np.array(file_hdf['entry1/SANS/detector/counts'])
                        img_h2o, trans= apply_corrections(file_hdf, img_h2o, img_cd,class_file_w, class_all, idx)
                        return img_h2o
                    
                    file_h2o = 'H2O'
                    if file_h2o in class_file['name']:
                        class_file_w = class_file
                        path_hdf_raw_w  = path_hdf_raw
                        img_h2o = read_water(path_hdf_raw_w, class_file_w, file_h2o)
                        img_cell = img_ec
                        img_h2o = np.subtract(img_h2o,img_cell)
                        # get correction factor
                        q_h2o, I_h2o, sigma_h2o = ai.integrate1d(img_h2o,  120, correctSolidAngle=True, mask=mask,
                                              method = 'nosplit_csr', unit = 'q_A^-1', safe=True, error_model="azimuthal")
                        
                        if det == '18p0' and use_detector > 0:
                              det_m = str(use_detector).replace('.','p')
                              path_det_1 = os.path.join(path_dir_an, 'det_' + det_m)
                              file_name = os.path.join(path_det_1, 'det_files_'+ det_m + 'm.json')
                              class_file_w = open(file_name)
                              class_file_w = eval(class_file_w.read())
                              path_hdf_raw_w = os.path.join(path_dir_an, 'det_' + det_m + '/hdf_raw/')
                              img_h2o_corr = read_water(path_hdf_raw_w, class_file_w, file_h2o)     
                              img_cell = img_ec
                              img_h2o_corr = np.subtract(img_h2o_corr,img_cell)
                              # get correction factor
                              q_h2o_corr, I_h2o_corr, sigma_h2o_corr = ai.integrate1d(img_h2o_corr,  120, correctSolidAngle=True, mask=mask,
                                                  method = 'nosplit_csr', unit = 'q_A^-1', safe=True, error_model="azimuthal")                     
                              scaling_factor = (I_h2o[50:-10]/I_h2o_corr[50:-10]).mean()
                              I_h2o = I_h2o_corr
                              sigma_h2o = sigma_h2o_corr
                        #open the file for the beam center and radial integrator
                        
                        
                        # Subtract empty cell
                        
                        wl = round(float(np.asarray(file_hdf['/entry1/SANS/Dornier-VS/lambda'])), 1) # in nm
                      
                        list_cs = {'0.5':0.909, '0.6':0.989, '0.8':1.090, '1.0':1.241, '1.2':1.452} # in cm-1
                        correction = float(list_cs[str(wl)])
                    else:
                            print('###########################################################')
                            print('There is no water measurement for this configuration: ' + det + ' m' )
                            print('###########################################################')
                            sys.exit('Please load a water (h2o) measurement.')
                
                else:
                    print('Intensity is not calibrated')
                    img_cd = np.zeros([128, 128])
                    img_ec = np.zeros([128, 128])
                    correction = 1
                
                plt.close('all')    
                # execute the corrections for all
                for ii in range(0, len(class_file['name'])):
                    name_hdf = path_hdf_raw + '/'+ class_file['name_hdf'][ii]
                    file_hdf = h5py.File(name_hdf, 'r')
                    # get the image
                    img = np.array(file_hdf['entry1/SANS/detector/counts'])
                    img1, trans = apply_corrections(file_hdf, img, img_cd,class_file, class_all, ii)
                    
                    if perform_abs_calib == 1:
                        # Subtract empty cell
                        img_cell = img_ec
                        # subraction of empty cell     
                        img2 =  np.subtract(img1, img_cell)
                    else:
                        img2 = img1
             
                    
                    print('Corrected scan ' + class_file['name_hdf'][ii])
                    
                  # perform radial integration
                    # radial integration
                    file_name = path_rad_int + 'rad_integ_' + class_file['scan'][ii] + '_' + class_file['name'][ii] + '_' +kk[4:] + 'm.dat'           
                    q, I, sigma = ai.integrate1d(img2,  120, correctSolidAngle=True, mask=mask,
                                            method = 'nosplit_csr', unit = 'q_A^-1', safe=True, error_model="azimuthal")
                    def std_calc(sigmaA, A, sigmaB, B):
                        A = np.divide(sigmaA, A)
                        B = np.divide(sigmaB, B)
                        sigma = np.sqrt(np.square(A) + np.square(B))
                        return sigma
                    
                    if perform_abs_calib == 1:
                        if det == '18p0':
                            I = I/scaling_factor
                            sigma = sigma/scaling_factor
                        I = (I/I_h2o) * correction
                        sigma = std_calc(sigma, I, sigma_h2o, I_h2o)
                        sigma = sigma * correction
                    
                    data_save = np.column_stack((q, I, sigma))
                    header_text = 'This file contains: q in Angstroms, absolute intensity in 1/cm, and standard deviation.'
                    np.savetxt(file_name, data_save, delimiter=',', header = header_text) 
                    
                    bc_x= ai.poni2/ai.pixel1;
                    bc_y = ai.poni1/ai.pixel2;
                    
                    if perform_anisotropy == 1:
                        theta_range = [0, 360]
                        ai.setChiDiscAtZero()
                        # perform the cake graphic
                        res2d = ai.integrate2d(img2, 120, theta_range[-1], mask=mask,   method = 'BBox', unit = 'q_A^-1')
                        Ia, tth, chi = res2d
                        sum_I = np.sum(Ia[:,10:30], 1)
                        n_zeros = []
                        for l in range(theta_range[0], theta_range[-1]):
                            n_zeros.append(np.count_nonzero(Ia[l,10:30]==0))
                        sum_I = np.divide(sum_I, np.subtract(120,n_zeros))
                        
                        step = 4
                        chi_range = np.arange(theta_range[0],theta_range[-1])
                        chi_range = np.arange(0, len(chi_range), step) # take only every second point
     
                        
                        chi_v = list(range(60,120, step)) + list(range(240, 300, step))
                        chi_h = list(range(0,30, step)) + list(range(150,210, step)) + list(range(330,360, step))
                        
                        file_name = path_rad_int + 'rad_integ_vert_' + class_file['scan'][ii] + '_' + class_file['name'][ii] + '_' +kk[4:] + 'm.dat'           
                        q_v, I_v, sigma_v = ai.integrate1d(img2,  120, correctSolidAngle=True, mask=mask, 
                                            method = 'nosplit_csr', unit = 'q_A^-1', safe=True, error_model="azimuthal",
                                            azimuth_range = [chi_v[0], chi_v[-1]] )
                        
                        if perform_abs_calib == 1:
                            if det == '18p0':
                                I_v = I_v/scaling_factor
                                sigma_v = sigma_v/scaling_factor
                            I_v = (I_v/I_h2o) * correction
                            sigma_v = std_calc(sigma_v, I_v, sigma_h2o, I_h2o)
                            sigma_v = sigma_v * correction
                                
                        data_save = np.column_stack((q_v, I_v, sigma_v))
                        header_text = 'This file contains: q in Angstroms, absolute intensity in 1/cm, and standard deviation.'
                        np.savetxt(file_name, data_save, delimiter=',', header = header_text) 
                        
                        chi_v_sum = np.sum(sum_I[chi_v])
                        file_name = path_rad_int + 'rad_integ_horiz_' + class_file['scan'][ii] + '_' + class_file['name'][ii] + '_' +kk[4:] + 'm.dat'           
                        q_h, I_h, sigma_h = ai.integrate1d(img2,  120, correctSolidAngle=True, mask=mask, 
                                            method = 'nosplit_csr', unit = 'q_A^-1', safe=True, error_model="azimuthal",
                                            azimuth_range = [chi_h[0], chi_h[-1]])

                        if perform_abs_calib == 1:
                            if det == '18p0':
                                I_h = I_h/scaling_factor
                                sigma_h = sigma_h/scaling_factor
                            I_h = (I_h/I_h2o) * correction
                            sigma_h = std_calc(sigma_h, I_h, sigma_h2o, I_h2o)
                            sigma_h = sigma_h * correction

                        data_save = np.column_stack((q_h, I_h, sigma_h))
                        header_text = 'This file contains: q in Angstroms, absolute intensity in 1/cm, and standard deviation.'
                        np.savetxt(file_name, data_save, delimiter=',', header = header_text) 
                        
                        chi_h_sum = np.sum(sum_I[chi_h])
                        
                                       
                    # plots
                    # Convert pixel coordinates starting at the beam center to coordinates in the inverse space (unit: nm ^ -1)
                    def x2q(x, wv, dist, pixelsize): 
                        return 4*np.pi/ai.wavelength*np.sin(np.arctan(pixelsize*x/dist)/2)
                    
                                        
                    qx = x2q(np.arange(img.shape[1])-bc_x, ai.wavelength, ai.dist, ai.pixel1)
                    qy = x2q(np.arange(img.shape[0])-bc_y, ai.wavelength, ai.dist, ai.pixel2)
                    extent = [qx.min(), qx.max(), qy.min(), qy.max()]
                    
                    
                    # define the figure axis
                    if perform_anisotropy == 1:
                        fig, ((axs0, axs1, axs5), (axs2, axs3, axs4))  = plt.subplots(2, 3,  figsize=(18, 12)) # , gridspec_kw={'width_ratios':[2, 3]}
                    else:
                        fig, ((axs0, axs1))  = plt.subplots(1, 2,  figsize=(10, 6)) # , gridspec_kw={'width_ratios':[2, 3]}
                    
                        
                    attenuator = (int(np.asarray(file_hdf['/entry1/SANS/attenuator/selection'])))
                    if attenuator == 0:
                        mask_inv =  (mask == 0).astype(int)
                    else:
                        mask_inv = np.ones([128, 128])
                        
                        
                    if img2.mean() > 0:
                        mean_img =  (img2*mask_inv).mean()
                        clim1 = (0, 2*(mean_img))
                    else:
                        clim1 = (0, 1)
                        
                    im1 = axs0.imshow(img2*mask_inv, origin='lower', aspect = 'equal', clim =clim1, cmap = 'turbo', extent = np.divide(extent,1e9)) # to have in A
                    fig.colorbar(im1, ax = axs0, orientation = 'horizontal', shrink = 0.75).set_label(r'Intensity I [cm$^{-1}$]')
                    axs0.grid(color = 'white', linestyle = '--', linewidth = 0.25)
                    axs0.set(ylabel = r'q$_{y}$ [$\AA$$^{-1}$]', xlabel = r'q$_{x}$ [$\AA$$^{-1}$]')
                           
                    axs1.plot(q, I, 'ok', label = 'total')
                    axs1.set(xlabel = r'Scattering vector q [$\AA^{-1}$]', ylabel = r'Intensity I [cm$^{-1}$]', xscale = 'log', 
                                yscale = 'log', title = 'Sample: '+  class_file['name'][ii])
                    axs1.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
                    axs1.errorbar(q, I, yerr = sigma, color = 'black', lw = 1)
                    
                    if perform_anisotropy == 1:
                        axs1.plot(q_v, I_v, '*b', label = 'vertical')
                        axs1.plot(q_h, I_h, '*r', label = 'horizontal')
                        axs1.legend(loc='lower left')
                        
                        # divided
                        axs5.plot(q_v, np.divide(I_v,I_h) , '*b', label = 'division')
                        axs5.set(xlabel = r'Scattering vector q [$\AA^{-1}$]', ylabel = r'Intensity $I_v/I_h$', xscale = 'log', 
                                yscale = 'linear', title = 'Sample: '+  class_file['name'][ii])
                        axs5.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
                        axs5.set_ylim(0, 2)
                                                
                        
                        im2=axs2.imshow(Ia, origin="lower", extent=[tth.min(), tth.max(), chi.min(), chi.max()], aspect="auto",  cmap='turbo', clim = clim1)
                        fig.colorbar(im2, ax = axs2, orientation = 'horizontal', shrink = 0.75).set_label(r'Intensity I [cm$^{-1}$]')
                        axs2.set(ylabel = r'Azimuthal angle $\chi$ [degrees]', xlabel = r'q [$\AA^{-1}$]')
                        axs2.grid(color='w', linestyle='--', linewidth=1)
                        axs2.set_title('2D integration')
                       
                        
                        axs3.plot(range(theta_range[0], theta_range[-1]), sum_I, 'ob', label=r"Sum_I")
                        axs3.set_xlabel(r'Azimuthal angle $\chi$ [degrees]')
                        axs3.set_ylabel(r'Normalized Intensity I [a.u.]')
                        axs3.set_title("Radial integration")
                        axs3.legend(loc='upper right')
     
                        axs3.set_xticks([0,90,180,270, 360])
                        axs3.set_xlim(chi_range[0], chi_range[-1]+2)
                        #axs3.set_ylim(25000, 35000)
                        #axs3.set_yticks([])
                        
                        axs4.bar(['horizontal', 'vertical'], [chi_h_sum, chi_v_sum])
                        #axs4.bar('v', chi_v, 'ob', label=r"counts_vertical")
                        axs4.set_xlabel(r'Azimuthal sector $\chi$')
                        axs4.set_ylabel(r'Sum of intensity [cm$^{-1}$]')
                        axs4.set_title("Anisotropy factor 1 - v/h = " + str(round(1 - chi_v_sum/chi_h_sum, 2)))
                        #axs4.legend(loc='upper left')
                                        
                    file_name = path_rad_int_fig + 'integ_' + class_file['scan'][ii] + '_' + class_file['name'][ii] + '_' +kk[4:] + 'm.jpeg' 
                    plt.savefig(file_name)
                    plt.close('all')    
                    
   
    



