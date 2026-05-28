# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 17:45:41 2026

@author: lutzbueno_v
"""

import os

# 1. Paste the path to your current experiment folder here
EXPERIMENT_FOLDER = r"C:\Users\lutzbueno_v\Documents\Analysis\DarePy-SANS_code\DarePy-SANS\experiments\2025_2012_Maria"

# 2. Save it to a hidden pointer file
pointer_file = os.path.join(os.path.dirname(__file__), ".active_experiment.txt")
with open(pointer_file, "w") as f:
    f.write(EXPERIMENT_FOLDER.strip())

print(f"✅ Active experiment set to: {EXPERIMENT_FOLDER}")
