"""
Stores all relevant configuration objects. On first import it loads / generates the
different config objects needed for any action.

For practicle purposes this behavior makes the module behave similar to a Singleton object.
"""

from .experiment import Experiment
from .instruments import SANS1Cfg, SANSLLBCfg
from .reduction import Reduction, Merging

experiment = Experiment.load()
reduction = Reduction.load()
merging = Merging.load()

sans1 = SANS1Cfg.load()
sansllb = SANSLLBCfg.load()

instrument=sansllb