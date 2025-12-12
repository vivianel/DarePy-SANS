"""
List datafiles with extracted information
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import os
from h5py import File
from tabulate import tabulate
from glob import glob

class Action:
    priority = 60

    mapping = [
        # key , hdf path
        ('sample', 'entry0/sample/name'),
        ('coll', 'entry0/SANS-LLB/collimator/length'),
        ('detz', 'entry0/SANS-LLB/central_detector/distance'),
        ]

    @classmethod
    def get_parser(cls) -> ArgumentParser:
        parser = ArgumentParser(prog='darepy list', formatter_class=ArgumentDefaultsHelpFormatter,
                                description='List hdf files with relevant measurement information.'
                                            'By default the columns shown are sample,coll and detz')
        parser.add_argument('-a', '--add-column', action='append', dest='add_column', default=[],
                            nargs=2, help='Show additional column, must supply hdf path as second argument')
        return parser

    def __init__(self, arguments):
        self.arguments = arguments
        self.mapping = self.mapping + arguments.add_column

    def run(self):
        from ..config import cfg

        data = dict([('filename', [])]+[(key, []) for key,_ in self.mapping])
        for fn in sorted(glob(os.path.join(cfg.experiment.path_hdf_raw, '*.hdf'))):
            hdf = File(fn)
            data['filename'].append(os.path.basename(fn))
            for key, hdf_path in self.mapping:
                data[key].append(self.read_key(hdf, hdf_path))

        print(tabulate(data, headers='keys', tablefmt='psql'))

    def read_key(self, hdf:File, hdfpath:str):
        element = hdf[hdfpath]
        if element.dtype.char=='S':
            return element[0].decode('utf-8')
        else:
            return element[0]