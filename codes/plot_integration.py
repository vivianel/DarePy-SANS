import os
import numpy as np
import matplotlib.pyplot as plt
#import pickle
import organize_hdf_files as org


def plot_integ_radial(config, result, ScanNr, Frame):
    path_analysis = org.create_analysis_folder(config)
    save_fig = config['analysis']['save_plots']

    class_all = result['overview']['all_files']
    if ScanNr in class_all['scan']:
        idx = list(class_all['scan']).index(ScanNr)
        det = class_all['detx_m'][idx]
        det = str(det).replace('.', 'p')
        sample_name = class_all['sample_name'][idx]
        attenuator = class_all['att'][idx]

    path_rad_int_fig = path_analysis + 'det_' + det + '/figures/'
    if not os.path.exists(path_rad_int_fig):
        os.mkdir(path_rad_int_fig)
    path_integ = path_analysis + 'det_' + det + '/integration/'


    file_name = path_integ + 'pattern2D_' + sample_name + '_det' + det + f"m_{ScanNr:07d}" + '_' + f"{Frame:04d}.dat"
    img1 = np.genfromtxt(file_name,
                         dtype=None,
                         delimiter=',')

    file_name = path_integ + 'radial_integ_' + sample_name + '_det' + det + f"m_{ScanNr:07d}" + '_' + f"{Frame:04d}.dat"
    q = np.genfromtxt(file_name,
                         dtype=None,
                         delimiter=',',
                         usecols = 0)
    I = np.genfromtxt(file_name,
                         dtype=None,
                         delimiter=',',
                         usecols = 1)
    sigma = np.genfromtxt(file_name,
                         dtype=None,
                         delimiter=',',
                         usecols = 2)

    # plots
    # Convert pixel coordinates starting at the beam center to coordinates in the inverse space (unit: nm ^ -1)
    ai = result['integration']['ai']
    bc_x = result['integration']['beam_center_x']
    bc_y = result['integration']['beam_center_y']
    mask = result['integration']['int_mask']


    if attenuator == 0:
        mask_inv =  (mask == 0).astype(int)
    else:
        detector_size = config['instrument']['detector_size']
        mask_inv = np.ones([detector_size, detector_size])


    if img1.mean() > 0:
        mean_img =  (img1*mask_inv).mean()
        clim1 = (0, 3*(mean_img))
    else:
        clim1 = (0, 1)

    # define the extent of the image in q
    def x2q(x, wl, dist, pixelsize):
       return 4*np.pi/wl*np.sin(np.arctan(pixelsize*x/dist)/2)


    #calculate the extent of the image in q
    qx = x2q(np.arange(img1.shape[1])-bc_x, ai.wavelength, ai.dist, ai.pixel1)
    qy = x2q(np.arange(img1.shape[0])-bc_y, ai.wavelength, ai.dist, ai.pixel2)
    extent = [qx.min(), qx.max(), qy.min(), qy.max()]

    plt.ioff()
    # define the figure axis
    fig1, ((axs0, axs1))  = plt.subplots(1, 2,  figsize=(10, 6))

    im1 = axs0.imshow(img1*mask_inv, origin='lower', aspect = 'equal', clim = clim1, cmap = 'turbo', extent = np.divide(extent,1e9)) # to have in A
    fig1.colorbar(im1, ax = axs0, orientation = 'horizontal', shrink = 0.75).set_label(r'Intensity I [cm$^{-1}$]')
    axs0.grid(color = 'white', linestyle = '--', linewidth = 0.25)
    axs0.set(ylabel = r'q$_{y}$ [$\AA$$^{-1}$]', xlabel = r'q$_{x}$ [$\AA$$^{-1}$]')
    axs1.plot(q, I, 'ok', label = 'total')
    axs1.set(xlabel = r'Scattering vector q [$\AA^{-1}$]', ylabel = r'Intensity I [cm$^{-1}$]', xscale = 'log',
                yscale = 'log', title = 'Sample: '+ sample_name)
    axs1.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    axs1.errorbar(q, I, yerr = sigma, color = 'black', lw = 1)

    if save_fig == 1:
        file_name = path_rad_int_fig + 'radial_integ_' + sample_name + '_det' + det + f"m_{ScanNr:07d}" + '_' + f"{Frame:04d}.jpeg"
        plt.savefig(file_name)
        plt.close(fig1)


def plot_integ_azimuthal(config, result, ScanNr, Frame):
    path_analysis = org.create_analysis_folder(config)
    save_fig = config['analysis']['save_plots']
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
    file_name = path_integ + 'azim_integ_' + sample_name + '_det' + det + f"m_{ScanNr:07d}" + '_' + f"{Frame:04d}.dat"

    data = np.genfromtxt(file_name,
                         dtype=None,
                         delimiter=',')
    q = data[:,0]
    I = data[:, 1:len(npt_azim)]
    #sigma = data[:,int((data.shape[1]-1)/2):]

    I_select = I[q_range, :]

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


    if save_fig == 1:
        file_name = path_rad_int_fig + 'azim_integ_' + sample_name + '_det' + det + f"m_{ScanNr:07d}" + '_' + f"{Frame:04d}.jpeg"
        fig2.savefig(file_name)
        plt.close(fig2)
    return result
