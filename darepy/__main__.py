"""
Executabel for the darepy package. Used to dispatch calls from the command line.
"""
from .config.experiment import Experiment

def create_default_config():
    # create the default config file
    Experiment().save_config_file()

if __name__ == "__main__":
    import sys
    if '-c' in sys.argv:
        create_default_config()
