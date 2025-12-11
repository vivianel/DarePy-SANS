import os
from h5py import File
from tabulate import tabulate
from glob import glob

path_hdf_raw = "C:/Users/glavic_a/DarePy-SANS/raw_data"

data = {'file': [], 'sample': [], 'coll': []}
for fn in sorted(glob(path_hdf_raw + '/*.hdf')):
    hdf = File(fn)
    data['file'].append(os.path.basename(fn))
    data['sample'].append(hdf['entry0/sample/name'][0].decode('utf-8'))
    data['coll'].append(hdf['entry0/SANS-LLB/collimator/length'][:])

print(tabulate(data, headers='keys', tablefmt='psql'))
