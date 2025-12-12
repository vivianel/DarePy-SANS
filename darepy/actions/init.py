"""
Create a new configuration for this data reduction.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from ..config import instruments, experiment

try:
    from enum import StrEnum
except ImportError:
    try:
        # python <3.11 try to use backports
        from backports.strenum import StrEnum
    except ImportError:
        # python <3.10 use Enum instead
        from enum import Enum as StrEnum

class InstrumentChoices(StrEnum):
    SANS_1 = 'SANS-I'
    SANS_LLB = 'SANS-LLB'

class Action:
    priority = 0

    @classmethod
    def get_parser(cls) -> ArgumentParser:
        parser = ArgumentParser(prog='darepy init', formatter_class=ArgumentDefaultsHelpFormatter,
                                description='DarePy init action to create a new configuration for this data reduction.')
        parser.add_argument('-r', '--raw-data-path', dest='raw_path',
                            default=experiment.Experiment.path_hdf_raw,
                            help='Path to the raw hdf data files')
        parser.add_argument('-i', '--instrument-config', dest='instrument_config', required=False,
                            choices=list(InstrumentChoices),
                            help='Add Instrument configuration to config file, otherwise default will be used.')
        return parser

    def __init__(self, arguments):
        self.arguments = arguments

    def run(self):
        # create the default config file
        e = experiment.Experiment()
        e.path_hdf_raw = self.arguments.raw_path
        e.save()

        # optionally add instrument configuration options
        if self.arguments.instrument_config == 'SANS-I':
            i = instruments.SANS1Cfg()
            i.save()
        elif self.arguments.instrument_config == 'SANS-LLB':
            i = instruments.SANSLLBCfg()
            i.save()
