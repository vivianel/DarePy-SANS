# -*- coding: utf-8 -*-
"""
Spyder Editor

conda activate spyder-env

This script processes SANS HDF5 data to create a GIF animation or save individual frames.
It automatically detects the year in filenames and supports background subtraction.
"""

import numpy as np
import matplotlib.pyplot as plt # Matplotlib imported for interactive use
import os
import re
from matplotlib.animation import FuncAnimation
from PIL import Image # For saving GIFs from individual frames (fallback method)

# Assuming utils.py is in the same directory or accessible in your Python path
from utils import load_hdf

# --- User Options ---
# Main base path for SANS analysis data.
# Raw data will be looked for in 'MAIN_BASE_PATH/raw_data'
# Results will be saved in 'MAIN_BASE_PATH/extra_results'
MAIN_BASE_PATH = 'C:/Users/lutzbueno_v/Documents/Analysis/data/microfluidics/2024_0212_Wade/DarePy-SANS/'

# List of scan numbers to process
LIST_SCAN = list(range(37107,37109))

# file year saved
file_year = '2024'

# Background scan number (if background subtraction is enabled)
BACKGROUND_SCAN_NR = 23088

# Enable or disable background subtraction
ENABLE_BACKGROUND_SUBTRACTION = False

# --- Output Mode Selection ---
# Choose 'gif' to save a single GIF animation.
# Choose 'frames' to save each processed image as a separate PNG/JPEG file AND keep them open.
OUTPUT_MODE = 'frames' # Options: 'gif', 'frames'

# --- Animation Specific Options (if OUTPUT_MODE is 'gif') ---
ANIMATION_FPS = 5 # Frames per second for the GIF animation
ANIMATION_LOOP = 0 # 0 means loop infinitely for GIF, 1 means play once
ANIMATION_QUALITY_DPI = 150 # DPI for the animation frames (affects resolution)

# --- Individual Frame Specific Options (if OUTPUT_MODE is 'frames') ---
FRAME_OUTPUT_FORMAT = 'png' # File format for individual frames (e.g., 'png', 'jpeg')
FRAME_QUALITY_DPI = 150 # DPI for individual frame images


# --- Derived Paths (Do not modify directly) ---
PATH_HDF_RAW = os.path.join(MAIN_BASE_PATH, 'raw_data')
OUTPUT_FOLDER = os.path.join(MAIN_BASE_PATH, 'extra_results')


# --- Functions and Definitions ---


def load_and_process_scan(scan_number, path_raw_dir, background_img=None, enable_bg_sub=False, file_year=None):
    """
    Loads an HDF scan, performs background subtraction if enabled, and prepares
    the image data.

    Args:
        scan_number (int): The scan number to load.
        path_raw_dir (str): The specific raw data directory where the HDF files are located.
        background_img (numpy.ndarray, optional): The background image to subtract.
                                                   Required if enable_bg_sub is True. Defaults to None.
        enable_bg_sub (bool, optional): Whether to perform background subtraction. Defaults to False.
        file_year (str, optional): The four-digit year to use in the filename. If None, it will be determined.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The processed image data.
            - str: The sample name.
            - str: The full name of the HDF file.
            Returns (None, None, None) if the file cannot be loaded.
    """
    current_year = file_year

    name_hdf = f'sans{current_year}n0{scan_number}.hdf'
    full_hdf_path = os.path.join(path_raw_dir, name_hdf)
    print(f"Attempting to load: {full_hdf_path}")

    try:
        img = load_hdf(path_raw_dir, name_hdf, 'counts')
    except FileNotFoundError:
        print(f"Error: File '{name_hdf}' not found in '{path_raw_dir}'. Skipping scan {scan_number}.")
        return None, None, None
    except Exception as e:
        print(f"An error occurred loading {name_hdf} from {path_raw_dir}: {e}. Skipping scan {scan_number}.")
        return None, None, None

    img_processed = np.where(img == 0, 1e-4, img)

    if enable_bg_sub and background_img is not None:
        img_processed = img_processed - background_img
        img_processed[img_processed < 0] = 0.0001 # Set negative to a small positive value

    sample_name = load_hdf(path_raw_dir, name_hdf, 'sample_name')
    return img_processed, sample_name, name_hdf

def get_plot_clim(image_data_list):
    """
    Determines a reasonable global color limit (clim) for plotting based on a
    collection of image data. This is crucial for consistent animation scaling.

    Args:
        image_data_list (list): A list of numpy arrays, where each array is an image frame.

    Returns:
        tuple: (vmin, vmax) for color scaling.
    """
    valid_data_frames = [arr for arr in image_data_list if arr is not None]

    if not valid_data_frames:
        print("Warning: No valid image data to determine global CLIM. Defaulting to (0, 100).")
        return (0, 100)

    all_vals = np.concatenate([arr.flatten() for arr in valid_data_frames])
    positive_vals = all_vals[all_vals > 0]

    if positive_vals.size > 0:
        clim_vmax = np.mean(positive_vals) + 0.5 * np.std(positive_vals)
        if clim_vmax <= 0 or np.isinf(clim_vmax) or np.isnan(clim_vmax):
             clim_vmax = np.max(positive_vals)
             if clim_vmax <= 0:
                 clim_vmax = 100
    else:
        clim_vmax = 100

    return (0, max(1, clim_vmax))


def save_individual_frames(processed_data, output_folder, global_clim, output_format, dpi):
    """
    Saves each processed image as a separate file directly into the specified output_folder.
    Figures are kept open after saving.
    """
    print(f"Saving individual frames to: {output_folder}")

    for idx, (img_data, plot_title) in enumerate(processed_data):
        fig_frame, ax_frame = plt.subplots(figsize=(8, 6))
        imgplot_frame = ax_frame.imshow(img_data, clim=global_clim, origin='lower', cmap='jet')
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
                                  list_scan_info, fps, loop, dpi):
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

    global_clim = get_plot_clim(raw_image_data_for_clim)
    print(f"Calculated global color limits (clim): {global_clim}")

    initial_img_data, initial_plot_title = processed_data[0]
    img_plot = ax.imshow(initial_img_data, clim=global_clim, origin='lower', cmap='jet')
    cbar = fig.colorbar(img_plot, ax=ax, fraction=0.046, pad=0.04)
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
            imgplot_frame = ax_frame.imshow(img_data, clim=global_clim, origin='lower', cmap='jet')
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
        background_file_year = file_year
        if background_file_year:
            print(f"Loading background scan: sans{background_file_year}n0{BACKGROUND_SCAN_NR}.hdf from {PATH_HDF_RAW}")
            bg_img_raw, _, _ = load_and_process_scan(BACKGROUND_SCAN_NR, PATH_HDF_RAW, file_year=background_file_year)
            if bg_img_raw is not None:
                background_image = np.where(bg_img_raw == 0, 1e-4, bg_img_raw)
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
        img_data, sample_name, hdf_name = load_and_process_scan(
            scan_nr, PATH_HDF_RAW, background_image, ENABLE_BACKGROUND_SUBTRACTION, file_year=file_year
        )

        if img_data is None:
            continue

        plot_title = f"{sample_name}, #{scan_nr}"
        processed_scan_data.append((img_data, plot_title))
        raw_image_data_for_clim.append(img_data)

    if not processed_scan_data:
        print("No valid scan data processed. Nothing to save/animate. Exiting.")
    elif OUTPUT_MODE == 'gif':
        # For 'gif' output, we create a figure for the animation, which is then saved.
        # This figure will likely not be interactively displayed if the 'Agg' backend
        # is chosen internally by FuncAnimation for saving.
        # The crucial point is that we allow Matplotlib to operate in interactive mode (plt.ion())
        # for overall control, and rely on the robust saving logic in the function.
        create_and_save_gif_animation(processed_scan_data, raw_image_data_for_clim, OUTPUT_FOLDER,
                                      LIST_SCAN, ANIMATION_FPS, ANIMATION_LOOP, ANIMATION_QUALITY_DPI)
        # Even after saving, if other interactive figures were opened, plt.show()
        # at the end ensures they remain responsive (though for GIF, usually none are left).

    elif OUTPUT_MODE == 'frames':
        # Calculate global clim once for consistency across individual frames
        global_clim_for_frames = get_plot_clim(raw_image_data_for_clim)

        save_individual_frames(processed_scan_data, OUTPUT_FOLDER, global_clim_for_frames,
                               FRAME_OUTPUT_FORMAT, FRAME_QUALITY_DPI)

        # After saving all frames and leaving their figures open, call plt.show()
        # to ensure they are actually displayed and interactive in your environment.
        plt.show()

    else:
        print(f"Error: Invalid OUTPUT_MODE '{OUTPUT_MODE}'. Please choose 'gif' or 'frames'.")

    print("Script finished.")
