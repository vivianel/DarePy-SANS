# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 13:28:02 2026

@author: lutzbueno_v
"""

import os

# 1. Locate the GUI manager relative to this folder
# Assuming: experiments/default_exp/launch_gui.py -> ../../darepy/gui_manager.py
current_dir = os.path.dirname(os.path.abspath(__file__))
gui_path = os.path.abspath(os.path.join(current_dir, "../../darepy/gui_manager.py"))
config_path = os.path.join(current_dir, "config_experiment.yaml")

# 2. Execute the GUI manager using Spyder's %run magic
from IPython import get_ipython
ipy = get_ipython()
if ipy:
    # This runs the GUI and passes the local config as an argument
    ipy.run_line_magic('run', f'"{gui_path}" "{config_path}"')
else:
    # Fallback for standard python
    os.system(f'python "{gui_path}" "{config_path}"')
