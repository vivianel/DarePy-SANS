# -*- coding: utf-8 -*-
"""
DarePy-SANS: Post-Processing & Merging Caller
Orchestrates a 4-step modular pipeline:
1. Overlay/Noise Analysis, 2. Stitched Merging, 3. Interpolation, 4. Background Subtraction
"""

import sys
import os

# 1. Get the directory of the current script (darepy/codes/)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Go up one level to find utils.py (in darepy/)
parent_dir = os.path.dirname(current_script_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 3. Now you can safely import utils
from utils import load_config

# ==========================================
# %% STANDARD IMPORTS
# ==========================================
import post_processing as pp


# ==========================================
# STEP 0: LOAD CONFIGURATION
# ==========================================
# This one line replaces the entire try/except block!
ext_cfg = load_config()

project_base = ext_cfg['analysis_paths']['project_base']
scripts_dir = ext_cfg['analysis_paths']['scripts_dir']
path_dir_an = os.path.join(project_base, 'analysis')

# Ensure the codes directory is in the system path so Python can find 'post_processing.py'
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)


m_set = ext_cfg.get('merging_settings', {})

print("\n" + "="*60)
print("DAREPY-SANS: POST-PROCESSING & MERGING")
print("="*60)

# ==========================================
# CLEAN UP YAML DICTIONARIES (Force numeric keys)
# ==========================================
raw_skip_start = m_set.get('skip_start', {})
raw_skip_end = m_set.get('skip_end', {})

# This gracefully handles if the YAML provided '1.6' (str), 1.6 (float), or 6 (int)
skip_start = {float(k): int(v) for k, v in raw_skip_start.items()}
skip_end = {float(k): int(v) for k, v in raw_skip_end.items()}

# ==========================================
# FUNCTION 1: INITIAL OVERLAY & NOISE CHECK
# ==========================================
run_plotting = m_set.get('run_step_1_plotting', True)

if run_plotting:
    print(f"\n[STEP 1] Generating plots with current YAML skip settings (Overwriting old files)...")
    merged_files = pp.plot_all_data(path_dir_an, skip_start, skip_end, force_replot=True)
else:
    print(f"\n[SKIP] Step 1: Noise analysis plots disabled (Loading data only).")
    merged_files = pp.plot_all_data(path_dir_an, skip_start, skip_end, force_replot=False)

# ==========================================
# FUNCTION 2: SCALING & STITCHED MERGING (RAW)
# ==========================================
if m_set.get('run_step_2_merging', True):
    print(f"\n[STEP 2] Stitching raw segments (Applying Skips)...")
    pp.merging_data(path_dir_an, merged_files, skip_start, skip_end)
else:
    print(f"\n[SKIP] Step 2: Raw merging disabled.")
# ==========================================
# FUNCTION 3: OPTIONAL INTERPOLATION / REBINNING
# ==========================================
if m_set.get('run_step_3_interpolation', False):
    i_type = m_set.get('interp_type', 'log')
    i_pts = m_set.get('interp_points', 150)
    s_win = m_set.get('smooth_window', 1)

    print(f"\n[STEP 3] Interpolating data (Type: {i_type}, Points: {i_pts})...")
    pp.interpolate_data(path_dir_an, interp_type=i_type, interp_points=i_pts, smooth_window=s_win)
else:
    print(f"\n[SKIP] Step 3: Interpolation/Rebinning disabled.")

# ==========================================
# FUNCTION 4: INCOHERENT BACKGROUND SUBTRACTION
# ==========================================
if m_set.get('run_step_4_incoherent', False):
    last_points = m_set.get('last_points_to_fit', 50)
    print(f"\n[STEP 4] Subtracting incoherent background (Last {last_points} pts)...")
    pp.subtract_incoherent(path_dir_an, initial_last_points_fit=last_points)
else:
    print(f"\n[SKIP] Step 4: Background subtraction disabled.")

print("\n" + "="*60)
print("PROCESSING COMPLETE. Check the 'merged' folder.")
print("="*60)
