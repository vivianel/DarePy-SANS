# which instrument
instrument = "sans-llb" # or "sans"
save_figures = 1 # if you don't want to save then change to 0

# path to the hdf files
path_hdf_raw = 'C:/Users/lutzbueno_v/Documents/Analysis/data/2024_SANS-LLB/DarePy-SANS/raw_data/'

# path to save th eimages
path_save_figures = 'C:/Users/lutzbueno_v/Documents/Analysis/data/2024_SANS-LLB/DarePy-SANS/analysis/'

# minimum value for the colorbar in log scale
vmin = 0

# maximum value for the colorbar in log scale
vmax = 7

# range of scans to plot
list_scan = [24] # for a range of scans: list(range(start, end)), for a single scan: [scan]


##########################################
import numpy as np
import matplotlib.pyplot as plt
from utils import load_hdf

plt.close('all')
plt.ion()


for jj in range(0, len(list_scan)):
    scanNr = list_scan[jj]
    # if you get an error double check the year of the file
    name_hdf = instrument + '2024n' +f"{scanNr:06}" +'.hdf'

    if instrument == 'sans-llb':
        vs_rpm = load_hdf(path_hdf_raw, name_hdf, 'vs_rpm', instrument)
        wl = 89237*(1/vs_rpm) + 0.12

        img_main = load_hdf(path_hdf_raw, name_hdf, 'counts_main', instrument)
        img_main = np.log(img_main)
        img_main[img_main < 0] = 1e-20

        img_left = load_hdf(path_hdf_raw, name_hdf, 'counts_left', instrument)
        img_left = np.log(img_left)
        img_left[img_left < 0] = 1e-20

        img_bottom = load_hdf(path_hdf_raw, name_hdf, 'counts_bottom', instrument)
        img_bottom = np.log(img_bottom)
        img_bottom[img_bottom < 0] = 1e-20


        sample_name = load_hdf(path_hdf_raw, name_hdf, 'sample_name', instrument)
        title_c = ('#'+ str(list_scan[jj]) + ', ' + f"{wl:.2}" + 'A, ' + sample_name )

        fig = plt.figure()

        ax1 = plt.subplot2grid((3, 3), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((3, 3), (0, 1), colspan=2, rowspan=2)
        ax3 = plt.subplot2grid((3, 3), (2, 1), colspan=2 )
        plt.tight_layout()


        ax1.imshow((img_left), aspect=0.1, origin = 'lower', cmap = 'jet', vmin=vmin, vmax=vmax)
        main = ax2.imshow((img_main), aspect=2.7, origin = 'lower', cmap = 'jet', vmin=vmin, vmax=vmax)
        ax2.title.set_text(title_c)
        cbar = plt.colorbar(main, ax=ax2)
        cbar.set_label('log(I)', rotation=270)
        ax3.imshow((img_bottom), aspect=7, origin = 'lower', cmap = 'jet', vmin=vmin, vmax=vmax)

    if instrument == 'sans':
        img = load_hdf(path_hdf_raw, name_hdf, 'counts')
        img1 = np.where(img==0, 1e-4, img)
        plt.figure()
        Int = 1
        img2 = img1
        clim1 = (0, Int*img2[img2>0].std())

        imgplot = plt.imshow(np.log(img1), clim=[0, 7], origin='lower')
        imgplot.set_cmap('jet')
        plt.colorbar()
        sample_name = load_hdf(path_hdf_raw, name_hdf, 'sample_name')
        plt.title(sample_name + ', #'+ str(list_scan[jj]))

    if save_figures == 1:
        # save the graphics
        plt.savefig(path_save_figures + str(list_scan[jj]) + '_' + sample_name + '.png')
        plt.close('all')
