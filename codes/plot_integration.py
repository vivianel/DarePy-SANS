import os
import numpy as np
import matplotlib.pyplot as plt
from utils import create_analysis_folder
import integration as integ
from scipy import optimize


def plot_integ_radial(config, result, ScanNr, Frame):
    # open the files
    path_analysis = create_analysis_folder(config)
    class_all = result['overview']['all_files']

    # loop over all scan numbers that are in the folder
    if ScanNr in class_all['scan']:
        idx = list(class_all['scan']).index(ScanNr)
        det = class_all['detx_m'][idx]
        det = str(det).replace('.', 'p')
        sample_name = class_all['sample_name'][idx]
        attenuator = class_all['att'][idx]

    # define folder to load the radial and azimuthal integration
    path_integ = path_analysis + 'det_' + det + '/integration/'

    # create folder to save the figures from radial integration
    if config['analysis']['plot_radial'] == 1 or config['analysis']['plot_azimuthal'] == 1:
        path_rad_int_fig = path_analysis + 'det_' + det + '/figures/'
        if not os.path.exists(path_rad_int_fig):
            os.mkdir(path_rad_int_fig)

    # %% Load the files from previous
    # load the 2D pattern already integrated from previous script
    prefix = 'pattern2D'
    sufix = 'dat'
    # create a file name
    file_name = integ.make_file_name(path_integ, prefix, sufix, sample_name, det, ScanNr, Frame)
    # load the 2D
    img1 = np.genfromtxt(file_name, dtype = None, delimiter=',')
    # smoothen the data for display
    img1[img1 <= 0] = np.median(img1)

    # load the radial integration from the integrated files
    prefix = 'radial_integ'
    sufix = 'dat'
    file_name = integ.make_file_name(path_integ, prefix, sufix, sample_name, det, ScanNr, Frame)
    q = np.genfromtxt(file_name, dtype = None, delimiter = ',', usecols = 0)
    I = np.genfromtxt(file_name, dtype = None, delimiter = ',', usecols = 1)
    sigma = np.genfromtxt(file_name, dtype = None, delimiter = ',', usecols = 2)

    # Load the definitions for the plotting - same used for the radial integration
    sectors_nr = result['integration']['sectors_nr']
    # integrate for azimuthal plots
    npt_azim = range(0, 360, int(360/sectors_nr))
    # pixel range for integration in azimuthal
    pixel_range_azim = result['integration']['pixel_range_azim']

    # load azimuthal files
    prefix = 'azim_integ'
    sufix = 'dat'
    file_name = integ.make_file_name(path_integ, prefix, sufix, sample_name, det, ScanNr, Frame)
    data = np.genfromtxt(file_name, dtype=None, delimiter=',')
    q_a = data[:,0]
    I_a = data[:, 1:len(npt_azim)]


    # %% plotting the results
    # for the plotting not to open while the code is running
    if config['analysis']['show_plots'] == 0:
        plt.ioff()
    if config['analysis']['show_plots'] == 1:
        plt.ion()

    # Convert pixel coordinates starting at the beam center to coordinates in the inverse space (unit: nm ^ -1)
    ai = result['integration']['ai']
    bc_x = result['integration']['beam_center_x']
    bc_y = result['integration']['beam_center_y']
    mask = result['integration']['int_mask']
    integration_points = result['integration']['integration_points']


    # %%define the figure axis (3 rows, 2 columns)
    fig1, ([[axs0, axs1], [axs2, axs3], [axs4, axs5]])  = plt.subplots(3, 2,  figsize=(12, 17))

    # invert the mask for the cases when the direct beam is plotted
    if attenuator == 0: # with beamstop
        mask_inv =  (mask == 0).astype(int)
    else: # without beamstop
        detector_size = config['instrument']['detector_size']
        mask_inv = np.ones([detector_size, detector_size])

    # define the extent of the image in q: from pixels to scattering vector
    def x2q(x, wl, dist, pixelsize):
        return 4*np.pi/wl*np.sin(np.arctan(pixelsize*x/dist)/2)
    #calculate the extent of the image in q
    qx = x2q(np.arange(img1.shape[1]) - bc_x, ai.wavelength, ai.dist, ai.pixel1)
    qy = x2q(np.arange(img1.shape[0]) - bc_y, ai.wavelength, ai.dist, ai.pixel2)
    extent = [qx.min(), qx.max(), qy.min(), qy.max()]

    # set color for "bad values"
    bool_mask = mask.astype('bool')
    img1[bool_mask] = np.nan
    cmap_mask = plt.get_cmap('jet')
    cmap_mask.set_bad(color='black')


    # AXS0: plot the scattering pattern in 2D in axs0
    im1 = axs0.imshow(img1*mask_inv, origin='lower', aspect = 'equal', cmap = cmap_mask, extent = np.divide(extent,1e9)) # to have in A
    fig1.colorbar(im1, ax = axs0, orientation = 'horizontal', shrink = 0.75).set_label(r'I [cm$^{-1}$]')
    axs0.grid(color = 'white', linestyle = '--', linewidth = 0.25)
    axs0.set(ylabel = r'q$_{y}$ [$\AA$$^{-1}$]', xlabel = r'q$_{x}$ [$\AA$$^{-1}$]')

    # AXS1: plot the integrated radial integration in axs1
    axs1.plot(q, I, 'ok', label = 'total', markersize=6, alpha = 0.8)
    axs1.set(xlabel = r'Scattering vector q [$\AA^{-1}$]', ylabel = r'Intensity I [cm$^{-1}$]', xscale = 'log',
                yscale = 'log', title = 'Sample: '+ sample_name)
    axs1.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    axs1.errorbar(q, I, yerr = sigma, color = 'black', lw = 1, markersize=2)

    # AXS2: plot the scattering pattern in 2D in axs2 (log scale)
    img2 = np.log(img1)
    #clim = [np.mean(img2), np.mean(img2)+3]
    im1 = axs2.imshow(img2*mask_inv, origin='lower', aspect = 'equal', cmap = cmap_mask, extent = np.divide(extent,1e9)) # to have in A
    fig1.colorbar(im1, ax = axs2, orientation = 'horizontal', shrink = 0.75).set_label(r'log(I) [cm$^{-1}$]')
    axs2.grid(color = 'white', linestyle = '--', linewidth = 0.25)
    axs2.set(ylabel = r'q$_{y}$ [$\AA$$^{-1}$]', xlabel = r'q$_{x}$ [$\AA$$^{-1}$]')

    # AXS3: plot the cake plot in axs2
    res2d = ai.integrate2d(img2*mask_inv, integration_points, 360, method = 'BBox', unit = 'q_A^-1')
    I_c, tth, chi = res2d

    # set color for "bad values"
    I_c_mask =  (I_c == 0).astype(int)
    bool_mask = I_c_mask.astype('bool')
    I_c[bool_mask] = np.nan
    cmap_mask = plt.get_cmap('jet')
    cmap_mask.set_bad(color='black')
    img3 = axs3.imshow(I_c, origin="lower", extent=[tth.min(), tth.max(), chi.min(), chi.max()], aspect="auto", cmap = cmap_mask)
    fig1.colorbar(img3, ax = axs3, orientation = 'horizontal', shrink = 0.75).set_label(r'log(I) [cm$^{-1}$]')
    axs3.set(ylabel = r'Azimuthal angle $\chi$ [degrees]', xlabel = r'q [$\AA^{-1}$]')
    axs3.grid(color='w', linestyle='--', linewidth=1)
    axs3.set_title('2D integration')


    # AXS4: plot the integrated radial integration in the sectors in axs4
    I_select = I_a[pixel_range_azim, :]
    colors = plt.cm.plasma(np.linspace(0, 1 , I_select.shape[1]))
    for ii in range(I_select.shape[1]):
        axs4.loglog(q_a[pixel_range_azim], I_select[:, ii], marker = 'o', color = colors[ii], markersize=3, alpha=0.5)
    axs4.set_xlabel(r'Scattering vector q [$\AA^{-1}$]')
    axs4.set_ylabel(r'Intensity I [cm$^{-1}$]')
    axs4.grid(color = 'gray', linestyle = '--', linewidth = 0.5)

    # AXS 5: plot the azimuthal plot: I vs angle
    range_angle = list(npt_azim[1:])
    sum_point = []
    for ii in range(I_select.shape[1]):
        sum_point_tmp = np.divide(np.nansum(I_select[:, ii]), np.count_nonzero(~np.isnan(I_select[:, ii])))
        axs5.semilogy(range_angle[ii], sum_point_tmp, marker = 'o', markersize=10, color=colors[ii])
        sum_point.append(sum_point_tmp)

    # fit a cosine
    def form(theta, I_0, theta0, offset):
        return I_0 * np.cos(np.radians(theta - theta0)) ** 2 + offset

    param, cov = optimize.curve_fit(form, range_angle, sum_point, [3e-4, 90, 0])
    offset = param[1]
    offset = offset - np.floor(offset/360)*360
    if offset >= 180:
        offset = offset - 180

    Af = param[2]/param[0]
    Af = np.abs(np.round(Af, 4))

    axs5.semilogy(range_angle, form(range_angle, *param), '--', color = colors[0], label = str(round(offset)))
    axs5.set_xlabel(r'Azimuthal angle $\chi$ [$^o$]')
    axs5.set_ylabel(r'Sum intensity I($\chi$) [cm$^{-1}$]')
    axs5.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    axs5.legend()
    axs5.set(title = 'Anisotropy = ' + str(Af))

    # save the plots
    prefix = 'radial_integ'
    sufix = 'jpeg'
    file_name = integ.make_file_name(path_rad_int_fig, prefix, sufix, sample_name, det, ScanNr, Frame)
    plt.savefig(file_name)
    plt.close(fig1)

    if config['analysis']['show_plots'] == 0:
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
