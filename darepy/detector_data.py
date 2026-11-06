"""Read-only, bounded HDF frame access for the interactive detector browser."""
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import re

import h5py
import numpy as np

COUNTS_PATHS = {'SANS-I': '/entry1/SANS/detector/counts',
                'SANS-LLB': '/entry0/central_data/data'}


def _metadata_text(value):
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(_metadata_text(item) for item in value) + ']'
    return str(value)


def find_files(folder):
    root = Path(folder)
    if not root.is_dir():
        raise ValueError(f'Raw data directory not found: {root}')
    return sorted((p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in
                   ('.hdf', '.h5', '.hdf5')), key=lambda p: str(p.relative_to(root)).lower())


def scan_number(path):
    match = re.search(r'(\d+)(?:_mod)?\.(?:hdf|h5|hdf5)$', Path(path).name, re.I)
    return int(match.group(1)) if match else None


def validate_limits(low, high):
    low, high = float(low), float(high)
    if not np.isfinite([low, high]).all() or low >= high:
        raise ValueError('Color limits must be finite numbers with minimum < maximum.')
    return low, high


def display_image(image, scale='lin', background=None, log_base='10'):
    data = np.array(image, dtype=np.float32, copy=True)
    if data.ndim != 2:
        raise ValueError('The detector preview requires a 2D frame.')
    data[data == 0] = 1e-4
    if background is not None:
        if background.shape != data.shape:
            raise ValueError(f'Background shape {background.shape} does not match detector shape {data.shape}.')
        data -= np.where(background == 0, 1e-4, background)
        data[data < 0] = 1e-4
    if scale == 'log':
        with np.errstate(invalid='ignore', divide='ignore'):
            if str(log_base).lower() == 'e':
                transform = np.log
            elif str(log_base) in ('10', '10.0'):
                transform = np.log10
            else:
                raise ValueError('Supported logarithm bases are 10 and e.')
            data = transform(np.where(data > 0, data, 1e-4))
    elif scale != 'lin':
        raise ValueError('Select lin or log.')
    return np.ma.masked_invalid(data)


@dataclass
class Frame:
    path: Path
    image: np.ndarray
    frame: int
    frames: int
    dataset: str
    metadata: tuple


class DetectorReader:
    """Use from one worker thread; never keep HDF files open between requests."""
    def __init__(self, cache_size=8):
        self.cache_size = cache_size
        self.frames = OrderedDict()
        self.metadata = OrderedDict()

    def _remember(self, cache, key, value):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self.cache_size:
            cache.popitem(last=False)
        return value

    def read(self, path, instrument, frame=0):
        path = Path(path).resolve()
        stat = path.stat()
        identity = (str(path), stat.st_mtime_ns, stat.st_size, instrument)
        key = identity + (int(frame),)
        if key in self.frames:
            self.frames.move_to_end(key)
            return self.frames[key]
        with h5py.File(path, 'r') as source:
            dataset_path = COUNTS_PATHS.get(instrument)
            if dataset_path is None:
                raise ValueError(f'Unsupported instrument: {instrument}')
            if dataset_path not in source:
                raise ValueError(f'Main detector dataset missing: {dataset_path}')
            dataset = source[dataset_path]
            if dataset.ndim not in (2, 3):
                raise ValueError(f'Expected a 2D detector or a 3D frame stack, found {dataset.shape}.')
            count = dataset.shape[0] if dataset.ndim == 3 else 1
            if not 0 <= frame < count:
                raise ValueError(f'Frame must be between 1 and {count}.')
            image = np.asarray(dataset[frame, :, :] if dataset.ndim == 3 else dataset[...], dtype=np.float32)
            image.setflags(write=False)
            if identity not in self.metadata:
                values = [('File', str(path)), ('Main detector', dataset_path),
                          ('Detector shape', str(dataset.shape)), ('Frames', str(count))]

                def visit(name, obj):
                    if len(values) >= 2000:
                        return 'metadata limit'
                    if not isinstance(obj, h5py.Dataset) or obj.name == dataset_path:
                        return None
                    try:
                        if obj.size <= 16:
                            value = obj[()]
                            if isinstance(value, np.ndarray):
                                value = value.tolist()
                            value = _metadata_text(value)[:2000]
                        else:
                            value = f'array {obj.shape} ({obj.dtype}); values not loaded'
                    except (OSError, ValueError) as exc:
                        value = f'unavailable: {exc}'
                    values.append(('/' + name, value))
                    return None

                source.visititems(visit)
                self._remember(self.metadata, identity, tuple(values))
            metadata = self.metadata[identity]
        result = Frame(path, image, frame, count, dataset_path, metadata)
        return self._remember(self.frames, key, result)


def viewer_limits(config):
    """Convert unmarked legacy ln limits for the log10 viewer, without mutation."""
    low, high = validate_limits(*config.get('clim', [0, 10]))
    if config.get('plot_scale', 'log') == 'log':
        base = str(config.get('log_base', 'e')).lower()
        if base == 'e':
            low, high = low / np.log(10), high / np.log(10)
        elif base not in ('10', '10.0'):
            raise ValueError('Supported logarithm bases are 10 and e.')
    return low, high


class SampleNameReader:
    """Read one small sample-name dataset; cache labels without detector reads."""
    def __init__(self, cache_size=4096):
        self.cache_size = cache_size
        self.cache = OrderedDict()

    def read(self, path, instrument):
        path = Path(path)
        stat = path.stat()
        identity = (str(path), stat.st_mtime_ns, stat.st_size, instrument)
        if identity in self.cache:
            self.cache.move_to_end(identity)
            return self.cache[identity]
        dataset_path = {'SANS-I': '/entry1/sample/name',
                        'SANS-LLB': '/entry0/sample/name'}.get(instrument)
        if dataset_path is None:
            raise ValueError(f'Unsupported instrument: {instrument}')
        with h5py.File(path, 'r') as source:
            if dataset_path not in source or source[dataset_path].size == 0:
                name = '(unnamed)'
            else:
                dataset = source[dataset_path]
                value = dataset[tuple(0 for _ in dataset.shape)] if dataset.shape else dataset[()]
                name = _metadata_text(value).strip() or '(unnamed)'
        self.cache[identity] = name
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return name
