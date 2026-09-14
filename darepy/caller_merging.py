# -*- coding: utf-8 -*-
"""
DarePy-SANS: Post-Processing & Merging Caller
Orchestrates a 4-step modular pipeline:
1. Overlay/Noise Analysis, 2. Stitched Merging,
3. Scaled Sample-Background Subtraction, 4. Incoherent Subtraction
"""

import sys
import os

# 1. Get the directory of the current script (darepy/codes/)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Go up one level to find utils.py (in darepy/)
parent_dir = os.path.dirname(current_script_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


# 2. Point directly to the 'codes' subdirectory where utils.py and backends live
codes_dir = os.path.join(parent_dir, "darepy/codes")

if codes_dir not in sys.path:
    sys.path.insert(0, codes_dir)

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
# FUNCTION 3: AUTOMATIC MERGED-SAMPLE BACKGROUND SUBTRACTION
# ==========================================
sample_background_applied = False
step_3_requested = m_set.get('run_step_3_sample_background', False)

if step_3_requested:
    background_sample = m_set.get('background_sample', None)
    background_scale_region = str(m_set.get('background_scale_region', 'low_q')).strip().lower()
    background_scale_points = int(m_set.get('background_scale_points', 10))

    if not background_sample:
        print("\n[ERROR] Step 3 enabled but 'background_sample' is not defined in merging_settings.")
    else:
        print(
            f"\n[STEP 3] Subtracting merged background '{background_sample}' "
            f"with automatic per-sample scaling from the {background_scale_region} "
            f"region ({background_scale_points} points)..."
        )
        sample_background_applied = pp.subtract_merged_background(
            path_dir_an,
            background_sample=background_sample,
            background_scale_region=background_scale_region,
            background_scale_points=background_scale_points,
        )
else:
    print("\n[SKIP] Step 3: Sample-background subtraction disabled.")

# ==========================================
# FUNCTION 4: RESIDUAL INCOHERENT SUBTRACTION
# ==========================================
if m_set.get('run_step_4_incoherent', False):
    if step_3_requested and not sample_background_applied:
        print(
            "\n[SKIP] Step 4: Step 3 was requested but did not complete successfully. "
            "Residual incoherent subtraction was not run on uncorrected data."
        )
    else:
        last_points = int(m_set.get('last_points_to_fit', 10))
        scale_subtraction = float(m_set.get('scale_subtraction', 1.0))
        print(f"\n[STEP 4] Subtracting residual incoherent background (Last {last_points} pts)...")
        pp.subtract_incoherent(
            path_dir_an,
            scale_subtraction,
            initial_last_points_fit=last_points,
            use_sample_background=sample_background_applied,
        )
else:
    print("\n[SKIP] Step 4: Incoherent subtraction disabled.")

print("\n" + "="*60)
print("PROCESSING COMPLETE. Check the 'merged' folder.")
print("="*60)
