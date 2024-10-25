import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils import create_analysis_folder
import integration as integ


def plot_integ_radial(config, result, ScanNr, Frame):
    # open the files
    path_analysis = create_analysis_folder(config)
    class_all = result['overview']['all_files']
    if ScanNr in class_all['scan']:
        idx = list(class_all['scan']).index(ScanNr)
        det = class_all['detx_m'][idx]
        det = str(det).replace('.', 'p')
        sample_name = class_all['sample_name'][idx]
        attenuator = class_all['att'][idx]
    # create folder to save the figures
    path_rad_int_fig = path_analysis + 'det_' + det + '/figures/'
    if not os.path.exists(path_rad_int_fig):
        os.mkdir(path_rad_int_fig)
    # create folder to save the radial integration
    path_integ = path_analysis + 'det_' + det + '/integration/'

    # load the 2D pattern
    prefix = 'pattern2D'
    sufix = 'dat'
    # create a file name
    file_name = integ.make_file_name(path_integ, prefix, sufix, sample_name, det, ScanNr, Frame)
    img1 = np.genfromtxt(file_name,
                         dtype = None,
                         delimiter=',')
    # to avoid zeros in the log representation
    img1[img1 <= 0] = 1e-10

    # load the radial integration
    prefix = 'radial_integ'
    sufix = 'dat'
    file_name = integ.make_file_name(path_integ, prefix, sufix, sample_name, det, ScanNr, Frame)
    q = np.genfromtxt(file_name,
                         dtype = None,
                         delimiter = ',',
                         usecols = 0)
    I = np.genfromtxt(file_name,
                         dtype = None,
                         delimiter = ',',
                         usecols = 1)
    sigma = np.genfromtxt(file_name,
                         dtype = None,
                         delimiter = ',',
                         usecols = 2)

    npt_azim = result['integration']['npt_azim']
    sectors_nr = result['integration']['sectors_nr']
    q_range = result['integration']['pixel_range']
    # load azimuthal files
    prefix = 'azim_integ'
    sufix = 'dat'
    file_name = integ.make_file_name(path_integ, prefix, sufix, sample_name, det, ScanNr, Frame)
    data = np.genfromtxt(file_name,
                         dtype=None,
                         delimiter=',')
    q_a = data[:,0]
    I_a = data[:, 1:len(npt_azim)]


    # plots
    # Convert pixel coordinates starting at the beam center to coordinates in the inverse space (unit: nm ^ -1)
    ai = result['integration']['ai']
    bc_x = result['integration']['beam_center_x']
    bc_y = result['integration']['beam_center_y']
    mask = result['integration']['int_mask']
    pixel_range = len(result['integration']['pixel_range'])
    if attenuator == 0:
        mask_inv =  (mask == 0).astype(int)
    else:
        detector_size = config['instrument']['detector_size']
        mask_inv = np.ones([detector_size, detector_size])

    # define the extent of the image in q
    def x2q(x, wl, dist, pixelsize):
       return 4*np.pi/wl*np.sin(np.arctan(pixelsize*x/dist)/2)

    # caked images
    res2d = ai.integrate2d(img1*mask_inv, pixel_range, 360,   method = 'BBox', unit = 'q_A^-1')
    I_c, tth, chi = res2d

    # reduce the size of the image for plotting
    #img1 = img1[35:-40, 35:-40]
    #mask_inv = mask_inv[35:-40, 35:-40]

    #calculate the extent of the image in q
    qx = x2q(np.arange(img1.shape[1])-bc_x, ai.wavelength, ai.dist, ai.pixel1)
    qy = x2q(np.arange(img1.shape[0])-bc_y, ai.wavelength, ai.dist, ai.pixel2)
    extent = [qx.min(), qx.max(), qy.min(), qy.max()]

    # for the plotting not to pop
    plt.ioff()

    # define the figure axis
    fig1, ([[axs0, axs1], [axs2, axs3], [axs4, axs5]])  = plt.subplots(3, 2,  figsize=(12, 17))


    scale = 'log'
    if scale == 'log':
        # set color for "bad values"
        bool_mask = mask.astype('bool')
        img1[bool_mask] = np.nan
        clim = (np.min(np.log(img1[~bool_mask]))/2, np.max(np.log(img1[~bool_mask])))
        cmap_mask = mpl.colormaps.get_cmap('jet')
        cmap_mask.set_bad(color='black')
        im1 = axs0.imshow(np.log(img1), origin='lower', clim = clim, aspect = 'equal', cmap = cmap_mask, extent = np.divide(extent,1e9)) # to have in A
        fig1.colorbar(im1, ax = axs0, orientation = 'horizontal', shrink = 0.75).set_label(r'log(I) [cm$^{-1}$]')
        im2 = axs2.imshow(np.log(I_c), origin="lower", extent=[tth.min(), tth.max(), chi.min(), chi.max()], aspect="auto", cmap = cmap_mask, clim = clim,)
    else:
        clim = (0, np.max(img1)/2)
        im1 = axs0.imshow(img1*mask_inv, origin='lower', aspect = 'equal', clim = clim, cmap = 'jet', extent = np.divide(extent, 1e9)) # to have in A
        fig1.colorbar(im1, ax = axs0, orientation = 'horizontal', shrink = 0.75).set_label(r'intensity (I) [cm$^{-1}$]')
        im2 = axs2.imshow(I_c, origin="lower", extent=[tth.min(), tth.max(), chi.min(), chi.max()], aspect="auto",  cmap='jet', clim = clim)

    axs0.grid(color = 'white', linestyle = '--', linewidth = 0.25)
    axs0.set(ylabel = r'q$_{y}$ [$\AA$$^{-1}$]', xlabel = r'q$_{x}$ [$\AA$$^{-1}$]')
    axs1.plot(q, I, 'ok', label = 'total', markersize=6, alpha = 0.8)
    axs1.set(xlabel = r'Scattering vector q [$\AA^{-1}$]', ylabel = r'Intensity I [cm$^{-1}$]', xscale = 'log',
                yscale = 'log', title = 'Sample: '+ sample_name)
    axs1.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    axs1.errorbar(q, I, yerr = sigma, color = 'black', lw = 1, markersize=2)

    # add caked imges to the files
    fig1.colorbar(im2, ax = axs2, orientation = 'horizontal', shrink = 0.75).set_label(r'Intensity I [cm$^{-1}$]')
    axs2.set(ylabel = r'Azimuthal angle $\chi$ [degrees]', xlabel = r'q [$\AA^{-1}$]')
    axs2.grid(color='w', linestyle='--', linewidth=1)
    axs2.set_title('2D integration')

    #
    sectors = sectors_nr
    range_sectors = int(sectors/4)
    I_h_range = (0,sectors-1, int(2*range_sectors), int((2*range_sectors)-1))
    I_h = np.sum(I_a[:,I_h_range], axis = 1)/2
    I_v_range = (int(range_sectors), int(range_sectors)-1, int(3*range_sectors), int(3*range_sectors)-1)
    I_v = np.sum(I_a[:,(I_v_range)], axis = 1)/2

    axs3.plot(q_a, I_h, 'ob', label = 'horizontal, 0 and 180', markersize=2)

    axs3.plot(q_a, I_v, 'or', label = 'vertical, 90 and 270', markersize=2)
    axs3.set(xlabel = r'Scattering vector q [$\AA^{-1}$]', ylabel = r'Intensity I [cm$^{-1}$]', xscale = 'log',
                yscale = 'log')
    axs3.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    axs3.set_title('Anisotropy I_v/I_h = ' + str(np.round((np.sum(I_v)/np.sum(I_h)), 2)))
    axs3.legend()

    I_select = I_a[q_range, :]
    colors = plt.cm.plasma(np.linspace(0, 1 , I_select.shape[1]))
    for ii in range(I_select.shape[1]):
        axs4.loglog(q_a[q_range], I_select[:, ii], marker = 'o', color = colors[ii], markersize=3, alpha=0.5)
    axs4.set_xlabel(r'Scattering vector q [$\AA^{-1}$]')
    axs4.set_ylabel(r'Intensity I [cm$^{-1}$]')
    axs4.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    #axs0.set_ylim([1e-2, 1e1])
    I_sum = np.sum(I_select, 0)
    range_angle = list(npt_azim[1:])
    axs5.semilogy(range_angle, I_sum - np.min(I_sum)*0.99, color=colors[0], linestyle = '--')
    for ii in range(I_select.shape[1]):
        axs5.semilogy(range_angle[ii], I_sum[ii]-np.min(I_sum)*0.99, marker = 'o', markersize=10, color=colors[ii])
    axs5.set_xlabel(r'Azimuthal angle $\chi$ [$^o$]')
    axs5.set_ylabel(r'Sum intensity I($\chi$) [cm$^{-1}$]')
    axs5.grid(color = 'gray', linestyle = '--', linewidth = 0.5)

    # save the plots
    prefix = 'radial_integ'
    sufix = 'jpeg'
    file_name = integ.make_file_name(path_rad_int_fig, prefix, sufix, sample_name, det, ScanNr, Frame)
    plt.savefig(file_name)
    plt.close(fig1)
    # to return into plotting
    plt.ion()


def plot_integ_azimuthal(config, result, ScanNr, Frame):
    path_analysis = create_analysis_folder(config)
    q_range = result['integration']['pixel_range']
    class_all = result['overview']['all_files']
    if ScanNr in class_all['scan']:
        idx = list(class_all['scan']).index(ScanNr)
        det = class_all['detx_m'][idx]
        det = str(det).replace('.', 'p')
        sample_name = class_all['sample_name'][idx]
    path_rad_int_fig = path_analysis + 'det_' + det + '/figures/'
    if not os.path.exists(path_rad_int_fig):
        os.mkdir(path_rad_int_fig)
    path_integ = path_analysis + 'det_' + det + '/integration/'
    npt_azim = result['integration']['npt_azim']
    # load azimuthal files
    prefix = 'azim_integ'
    sufix = 'dat'
    file_name = integ.make_file_name(path_integ, prefix, sufix, sample_name, det, ScanNr, Frame)
    data = np.genfromtxt(file_name,
                         dtype=None,
                         delimiter=',')
    q = data[:,0]
    I = data[:, 1:len(npt_azim)]
    #sigma = data[:,int((data.shape[1]-1)/2):]
    I_select = I[q_range, :]

    # to avoid plots poping up
    plt.ioff()
    colors = plt.cm.viridis(np.linspace(0, 1 , I_select.shape[1]))
    fig2, ((axs0, axs1))  = plt.subplots(1, 2,  figsize=(15, 5))
    for ii in range(I_select.shape[1]):
        axs0.loglog(q[q_range], I_select[:, ii], marker = 'o', color = colors[ii])
    axs0.set_xlabel(r'Scattering vector q [$\AA^{-1}$]')
    axs0.set_ylabel(r'Intensity I [cm$^{-1}$]')
    #axs0.set_ylim([1e-2, 1e1])
    I_sum = np.sum(I_select, 0)
    range_angle = list(npt_azim[1:])
    axs1.semilogy(range_angle, I_sum - np.min(I_sum)*0.99, color=colors[0], linestyle = '--')
    for ii in range(I_select.shape[1]):
        axs1.semilogy(range_angle[ii], I_sum[ii]-np.min(I_sum)*0.99, marker = 'o', markersize=10, color=colors[ii])
    axs1.set_xlabel(r'Azimuthal angle $\chi$ [$^o$]')
    axs1.set_ylabel(r'Sum intensity I($\chi$) [cm$^{-1}$]')
    #axs1.set_ylim([1e-2, 1e3])
    # save the plots
    prefix = 'azim_integ'
    sufix = 'jpeg'
    file_name = integ.make_file_name(path_rad_int_fig, prefix, sufix, sample_name, det, ScanNr, Frame)
    fig2.savefig(file_name)
    plt.close(fig2)
    # to return into plotting
    plt.ion()
