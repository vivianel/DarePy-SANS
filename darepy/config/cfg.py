"""
Stores all relevant configuration objects. On first import it loads / generates the
different config objects needed for any action.

For practicle purposes this behavior makes the module behave similar to a Singleton object.
"""

from .experiment import Experiment

experiment = Experiment.load()

