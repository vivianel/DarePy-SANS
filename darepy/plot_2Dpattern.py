# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import matplotlib
from pathlib import Path

# ==========================================
# %% 1. PATH INJECTION (Finds utils.py one level up)
# ==========================================
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent  # This is the /darepy/ folder

if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from utils import load_hdf, find_hdf_filename, load_config, parse_scan_list

# Catch the YAML path sent by the GUI (sys.argv[1])
config = load_config()

# --- REST OF THE IMPORTS ---
try:
    matplotlib.use('Qt5Agg')
except:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image


cfg_2d = config['plot_2d']

LIST_SCAN = parse_scan_list(config['plot_2d']['list_scan'])
clim = cfg_2d['clim']
plot_scale = cfg_2d['plot_scale']
ENABLE_BACKGROUND_SUBTRACTION = cfg_2d['enable_bg_subtraction']
BACKGROUND_SCAN_NR = cfg_2d['background_scan_nr']
OUTPUT_MODE = cfg_2d['output_mode']

# --- Animation Options (Only used if OUTPUT_MODE is 'gif') ---
ANIMATION_FPS = 5            # Frames per second
ANIMATION_LOOP = 0           # 0 = loop infinitely, 1 = play once
ANIMATION_QUALITY_DPI = 150  # DPI resolution for the animation

# --- Frame Options (Only used if OUTPUT_MODE is 'frames') ---
FRAME_OUTPUT_FORMAT = 'png'  # Options: 'png', 'jpeg', etc.
FRAME_QUALITY_DPI = 150      # DPI resolution for individual images
# --- Functions and Definitions ---

# ==========================================
# %% 2. DYNAMIC PATHS (Loaded from YAML)
# ==========================================


# Pull paths dynamically
PROJECT_BASE = config['analysis_paths']['project_base']
PATH_HDF_RAW = config['analysis_paths']['raw_data']

# Derived Paths (Automatically places output in your project folder)
OUTPUT_FOLDER = os.path.join(PROJECT_BASE, 'extra_results')


def load_and_process_scan(scan_number, path_raw_dir, background_img=None, enable_bg_sub=False):
    """
    Loads an HDF scan, performs background subtraction if enabled, and prepares
    the image data. (Dynamically finds the filename).
    """

    # --- NEW AUTOMATIC LOOKUP ---
    name_hdf = find_hdf_filename(path_raw_dir, scan_number)

    if not name_hdf:
        print(f"Error: Could not automatically find HDF file for scan {scan_number} in '{path_raw_dir}'.")
        return None, None, None

    full_hdf_path = os.path.join(path_raw_dir, name_hdf)
    print(f"Attempting to load: {full_hdf_path}")

    try:
        img = load_hdf(path_raw_dir, name_hdf, 'counts')
    except Exception as e:
        print(f"An error occurred loading {name_hdf} from {path_raw_dir}: {e}. Skipping scan {scan_number}.")
        return None, None, None

    # Convert the img data into a list of 2D images
    if img.ndim == 3:
        # Assuming the first dimension is the number of images
        images = [img[i, :, :] for i in range(img.shape[0])]
    else:
        # It's a single 2D image
        images = [img]

    processed_images = []
    for current_img in images:
        img_processed = np.where(current_img == 0, 1e-4, current_img)

        if enable_bg_sub and background_img is not None:
            # Check if background image has the same shape as the current image
            if background_img.shape == img_processed.shape:
                img_processed = img_processed - background_img
                img_processed[img_processed < 0] = 0.0001
            else:
                print("Warning: Background image shape does not match current image. Skipping subtraction.")

        processed_images.append(img_processed)

    sample_name = load_hdf(path_raw_dir, name_hdf, 'sample_name')
    return processed_images, sample_name, name_hdf

def save_individual_frames(processed_data, output_folder, output_format, dpi, clim):
    """
    Saves each processed image as a separate file directly into the specified output_folder.
    Figures are kept open after saving.
    """
    print(f"Saving individual frames to: {output_folder}")

    for idx, (img_data, plot_title) in enumerate(processed_data):
        fig_frame, ax_frame = plt.subplots(figsize=(8, 6))
        if plot_scale == 'lin':
            imgplot_frame = ax_frame.imshow(img_data, origin='lower', cmap='jet', clim = clim)
        elif plot_scale == 'log':
            imgplot_frame = ax_frame.imshow(np.log(img_data), origin='lower', cmap='jet', clim = clim)
        fig_frame.colorbar(imgplot_frame, ax=ax_frame, fraction=0.046, pad=0.04)
        ax_frame.set_title(plot_title)

        # Use a consistent filename pattern for individual frames
        clean_plot_title = plot_title.replace(', ', '_').replace('#', '').replace(' ', '_').replace('/', '_')
        frame_file_name = f"{clean_plot_title}_frame_{idx:03d}.{output_format}"
        frame_path = os.path.join(output_folder, frame_file_name)

        plt.savefig(frame_path, dpi=dpi, bbox_inches='tight')
        # DO NOT close fig_frame here. It will remain open for interaction.
        print(f"Saved frame: {frame_file_name}")

    print("Individual frames saved successfully.")
    # plt.show() will be called once at the end of the main block to display all figures.


def create_and_save_gif_animation(processed_data, raw_image_data_for_clim, output_folder,
                                  list_scan_info, fps, loop, dpi, clim):
    """
    Handles the creation and saving of the GIF animation.
    This function will still close its internal figure after saving,
    as it's designed for headless rendering for reliable animation saving.
    """
    if not processed_data:
        print("No valid scan data processed to create an animation. Skipping GIF creation.")
        return

    # When creating GIF, we still want to use a non-interactive approach for stability.
    # This means the figure created here will be for rendering to GIF, not for display.
    # If the user has an interactive backend active, this might still interact with it.

    # We explicitly create a figure for the animation, which FuncAnimation manages.
    # This figure is intended for rendering to file, not for interactive display.
    fig, ax = plt.subplots(figsize=(8, 6))


    initial_img_data, initial_plot_title = processed_data[0]
    if plot_scale == 'lin':
        img_plot = ax.imshow(initial_img_data, origin='lower', cmap='jet', clim = clim)
    elif plot_scale == 'log':
        img_plot = ax.imshow(np.log(initial_img_data), origin='lower', cmap='jet', clim = clim)
    title_obj = ax.set_title(initial_plot_title)

    def update_frame(frame_index):
        img_data, plot_title = processed_data[frame_index]
        img_plot.set_array(img_data)
        title_obj.set_text(plot_title)
        return [img_plot, title_obj]

    name_file = 'scan_' + str(list_scan_info[0]) + '_' + str(list_scan_info[-1]) + '.gif'
    animation_file_name = os.path.join(output_folder, name_file)

    try:
        writer = 'pillow'
        print(f"Attempting to save GIF animation to {animation_file_name} using Matplotlib's Pillow writer...")

        ani = FuncAnimation(fig, update_frame, frames=len(processed_data),
                            interval=1000/fps, blit=False, # blit=False is safer for saving
                            repeat=True if loop == 0 else False, repeat_delay=1000)

        ani.save(animation_file_name, writer=writer, dpi=dpi)
        print(f"GIF animation saved successfully to: {animation_file_name}")

    except Exception as e:
        print(f"\n--- ERROR DURING MATPLOTLIB ANIMATION SAVE ---")
        print(f"Error: {e}")
        print(f"This often indicates an issue with Matplotlib's animation backend or configuration in an interactive environment.")
        print("Falling back to manual GIF creation via Pillow (more robust for troubleshooting).")
        print(f"--- END ERROR ---\n")

        # Fallback: Manual GIF creation with PIL (Pillow) by saving frames
        temp_frames_dir = os.path.join(output_folder, 'temp_animation_frames_for_gif')
        os.makedirs(temp_frames_dir, exist_ok=True)

        print("Saving individual frames to temporary directory for manual GIF creation...")
        frames_for_pil = []
        for idx, (img_data, plot_title) in enumerate(processed_data):
            fig_frame, ax_frame = plt.subplots(figsize=(8, 6))
            if plot_scale == 'lin':
                imgplot_frame = ax_frame.imshow(img_data, origin='lower', cmap='jet', clim = clim)
            elif plot_scale == 'log':
                imgplot_frame = ax_frame.imshow(np.log(img_data), origin='lower', cmap='jet', clim = clim)
            fig_frame.colorbar(imgplot_frame, ax=ax_frame, fraction=0.046, pad=0.04)
            ax_frame.set_title(plot_title)

            frame_path = os.path.join(temp_frames_dir, f'frame_{idx:03d}.png')
            plt.savefig(frame_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig_frame) # Close these temp figures
            frames_for_pil.append(Image.open(frame_path))

        if frames_for_pil:
            print(f"Combining {len(frames_for_pil)} frames into GIF using Pillow directly...")
            frame_duration_ms = int(1000 / fps)
            frames_for_pil[0].save(
                animation_file_name,
                format='GIF',
                append_images=frames_for_pil[1:],
                save_all=True,
                duration=frame_duration_ms,
                loop=loop
            )
            print(f"Manual GIF animation saved to: {animation_file_name}")
        else:
            print("No frames were generated for manual GIF creation.")

        if os.path.exists(temp_frames_dir):
            for f in os.listdir(temp_frames_dir):
                os.remove(os.path.join(temp_frames_dir, f))
            os.rmdir(temp_frames_dir)
            print("Temporary frames cleaned up.")

    plt.close(fig) # Always close the animation figure to avoid memory leaks.


# --- Main Script Execution ---
if __name__ == '__main__':
    # Initial setup for interactive plotting
    plt.close('all') # Close any existing plots from previous runs

    # Enable interactive mode so figures created with plt.subplots() stay open.
    plt.ion()

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    background_image = None
    background_file_year = None

    if ENABLE_BACKGROUND_SUBTRACTION:
        print(f"Determining year for background scan: {BACKGROUND_SCAN_NR}")
        if background_file_year:
            print(f"Loading background scan: sans{background_file_year}n0{BACKGROUND_SCAN_NR}.hdf from {PATH_HDF_RAW}")
            # Load background, which may also be a list of images. We'll use the first one.
            bg_imgs, _, _ = load_and_process_scan(BACKGROUND_SCAN_NR, PATH_HDF_RAW)
            if bg_imgs is not None and len(bg_imgs) > 0:
                background_image = np.where(bg_imgs[0] == 0, 1e-4, bg_imgs[0])
                print("Background loaded successfully.")
            else:
                print("Could not load background image. Background subtraction will be skipped for all scans.")
                ENABLE_BACKGROUND_SUBTRACTION = False
        else:
            print("Could not determine year for background scan. Background subtraction will be skipped for all scans.")
            ENABLE_BACKGROUND_SUBTRACTION = False

    processed_scan_data = []
    raw_image_data_for_clim = []

    for i, scan_nr in enumerate(LIST_SCAN):
        print(f"Processing scan: {scan_nr}")
        images_data, sample_name, hdf_name = load_and_process_scan(
            scan_nr, PATH_HDF_RAW, background_image, ENABLE_BACKGROUND_SUBTRACTION)

        if images_data is None:
            continue

        # Iterate through all images returned for the current scan
        for sub_index, img_data in enumerate(images_data):
            # Create a more specific plot title for each image
            plot_title = f"{sample_name}, #{scan_nr}"
            if len(images_data) > 1:
                 plot_title += f" (Frame {sub_index+1}/{len(images_data)})"

            processed_scan_data.append((img_data, plot_title))
            raw_image_data_for_clim.append(img_data)

    if not processed_scan_data:
        print("No valid scan data processed. Nothing to save/animate. Exiting.")
    elif OUTPUT_MODE == 'gif':
        # For 'gif' output, we create a figure for the animation, which is then saved.
        create_and_save_gif_animation(processed_scan_data, raw_image_data_for_clim, OUTPUT_FOLDER,
                                      LIST_SCAN, ANIMATION_FPS, ANIMATION_LOOP, ANIMATION_QUALITY_DPI)
        # Even after saving, if other interactive figures were opened, plt.show()
        # at the end ensures they remain responsive (though for GIF, usually none are left).

    elif OUTPUT_MODE == 'frames':
        # Calculate global clim once for consistency across individual frames

        save_individual_frames(processed_scan_data, OUTPUT_FOLDER,
                               FRAME_OUTPUT_FORMAT, FRAME_QUALITY_DPI, clim)

        # After saving all frames and leaving their figures open, call plt.show()
        # to ensure they are actually displayed and interactive in your environment.
        plt.show()

    else:
        print(f"Error: Invalid OUTPUT_MODE '{OUTPUT_MODE}'. Please choose 'gif' or 'frames'.")

    print("Script finished.")
