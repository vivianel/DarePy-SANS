# -*- coding: utf-8 -*-
"""
DarePy-SANS Master Pipeline Orchestrator (Spyder Optimized)
"""


# ==========================================
# PIPELINE EXECUTION FLAGS (ON / OFF)
# ==========================================
RUN_RENAME_SAMPLES       = False
RUN_PLOT_2D              = False
RUN_MASK_BEAMSTOP        = False
RUN_RADIAL_INTEGRATION   = True
RUN_MERGING              = False

# ==========================================
# PATH MANAGEMENT (SPYDER FRIENDLY)
# ==========================================

import sys
import os
import runpy
from pathlib import Path

# Safely find the 'codes' folder even when running in Spyder
try:
    current_dir = Path(__file__).resolve().parent
except NameError:
    current_dir = Path(os.getcwd()).resolve()

codes_dir = current_dir / 'codes'

# Inject codes_dir globally so no child script ever gets lost
if str(codes_dir) not in sys.path:
    sys.path.insert(0, str(codes_dir))

def run_pipeline_step(script_name):
    print("\n" + "="*50)
    print(f" RUNNING: {script_name}")
    print("="*50)
    try:
        # runpy executes the script exactly like Spyder's %runfile
        runpy.run_path(script_name, run_name="__main__")
    except SystemExit as e:
        # Catch sys.exit(0) so it doesn't accidentally kill the Master Script
        if e.code not in [0, None]:
            print(f"\n[ERROR] {script_name} halted with error code: {e.code}")
            sys.exit(e.code)
    except Exception as e:
        print(f"\n[ERROR] Pipeline halted. {script_name} crashed: {e}")
        sys.exit(1)

print("="*50)
print(" Starting DAREPY-SANS Master Pipeline...")
print("="*50)
print(f"System Path globally updated to include:\n -> {codes_dir}\n")

# Execute the pipeline steps
if RUN_RENAME_SAMPLES: run_pipeline_step("rename_samples.py")
else: print("[SKIPPED] rename_samples.py")

if RUN_PLOT_2D: run_pipeline_step("plot_2Dpattern.py")
else: print("[SKIPPED] plot_2Dpattern.py")

if RUN_MASK_BEAMSTOP: run_pipeline_step("mask_beamstop_center.py")
else: print("[SKIPPED] mask_beamstop_center.py")

if RUN_RADIAL_INTEGRATION: run_pipeline_step("caller_radial_integration.py")
else: print("[SKIPPED] caller_radial_integration.py")

if RUN_MERGING: run_pipeline_step("caller_merging.py")
else: print("[SKIPPED] caller_merging.py")

print("\n" + "="*50)
print(" PIPELINE SUCCESSFULLY COMPLETED!")
print("="*50)
