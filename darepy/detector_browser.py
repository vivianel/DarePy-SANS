"""Tk detector browser with asynchronous HDF reads and an embedded Matplotlib canvas."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
from queue import SimpleQueue, Empty
from threading import Event
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from darepy.detector_data import DetectorReader, display_image, find_files, scan_number, validate_limits, viewer_limits, SampleNameReader


class _LimitsEntry:
    def __init__(self, viewer):
        self.viewer = viewer

    def get(self):
        low, high = validate_limits(self.viewer.low.get(), self.viewer.high.get())
        return f'{low}, {high}'


class DetectorBrowser(ttk.Frame):
    def __init__(self, parent, host):
        super().__init__(parent, padding=8)
        self.host = host
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='DarePy-preview')
        self.reader = DetectorReader()
        self.name_reader = SampleNameReader()
        self.name_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='DarePy-sample-names')
        self.name_stop = Event()
        self.name_generation = 0
        self.sample_names = {}
        self.pending = None
        self.generation = 0
        self.closed = False
        self.after_ids = set()
        self.render_after = None
        self.frame_after = None
        self.raw = self.background = self.current = None
        self.files = []
        self.selected_path = None
        self._updating = False
        cfg = host.config_dict['plot_2d']
        self.folder = self._resolve(host.config_dict['analysis_paths']['raw_data'])
        self.instrument = host.config_dict['instrument_setup']['which_instrument']
        limits = viewer_limits(cfg)
        self.low = tk.StringVar(self, str(limits[0]))
        self.high = tk.StringVar(self, str(limits[1]))
        self.scale = tk.StringVar(self, cfg.get('plot_scale', 'log'))
        self.log_base = tk.StringVar(self, '10')
        self.frame = tk.StringVar(self, '1')
        self.filter = tk.StringVar(self)
        self.meta_filter = tk.StringVar(self)
        self.bg_enabled = tk.BooleanVar(self, cfg.get('enable_bg_subtraction', False))
        self.bg_scan = tk.StringVar(self, str(cfg.get('background_scan_nr', '')))
        self.scans = tk.StringVar(self, self._text(cfg.get('list_scan', '')))
        self.output = tk.StringVar(self, {'frames': 'individual_frames', 'gif': 'gif_animation'}.get(cfg.get('output_mode'), cfg.get('output_mode', 'individual_frames')))
        self.fps = tk.StringVar(self, str(cfg.get('gif_fps', 2)))
        self.play_once = tk.BooleanVar(self, cfg.get('play_once', False))
        self.status = tk.StringVar(self, 'Choose a file to preview the main detector.')
        self._build()
        self._register_entries()
        self.low.trace_add('write', self._schedule_render)
        self.high.trace_add('write', self._schedule_render)
        self.scale.trace_add('write', self._scale_changed)
        self.filter.trace_add('write', lambda *_: self._filter_files())
        self.meta_filter.trace_add('write', lambda *_: self._show_metadata())
        self.frame.trace_add('write', self._schedule_frame)
        self.bind('<Destroy>', self._destroyed, add='+')
        self.refresh_files()

    @staticmethod
    def _text(value):
        return ', '.join(map(str, value)) if isinstance(value, (list, tuple)) else str(value or '')

    def _resolve(self, value):
        path = Path(value).expanduser()
        return path if path.is_absolute() else Path(self.host.config_file).resolve().parent / path

    def _later(self, delay, callback):
        if self.closed:
            return None
        token = None

        def run():
            self.after_ids.discard(token)
            if not self.closed:
                callback()
        token = self.after(delay, run)
        self.after_ids.add(token)
        return token

    def _cancel(self, token):
        if token:
            self.after_cancel(token)
            self.after_ids.discard(token)

    def _button(self, parent, **options):
        return tk.Button(parent, bg='#2196F3', fg='white', activebackground='#1976D2',
                         activeforeground='white', font=('Arial', 10, 'bold'),
                         padx=8, pady=4, **options)

    def _build(self):
        panes = ttk.Panedwindow(self, orient='horizontal')
        panes.pack(fill='both', expand=True)
        left = ttk.Frame(panes, width=350)
        panes.add(left, weight=1)
        right = ttk.Frame(panes)
        panes.add(right, weight=4)
        ttk.Label(left, text='Raw HDF files', font=('Arial', 11, 'bold')).pack(anchor='w')
        ttk.Label(left, text=str(self.folder), wraplength=240).pack(anchor='w', pady=4)
        ttk.Entry(left, textvariable=self.filter).pack(fill='x', pady=4)
        file_frame = ttk.Frame(left)
        file_frame.pack(fill='both', expand=True)
        self.file_tree = ttk.Treeview(file_frame, columns=('sample',), show='tree headings', selectmode='browse')
        self.file_tree.heading('#0', text='HDF file')
        self.file_tree.heading('sample', text='Sample name')
        self.file_tree.column('#0', width=190, minwidth=110)
        self.file_tree.column('sample', width=150, minwidth=100)
        self.file_tree.pack(side='left', fill='both', expand=True)
        scrollbar = ttk.Scrollbar(file_frame, command=self.file_tree.yview)
        scrollbar.pack(side='right', fill='y')
        self.file_tree.configure(yscrollcommand=scrollbar.set)
        horizontal = ttk.Scrollbar(left, orient='horizontal', command=self.file_tree.xview)
        horizontal.pack(fill='x')
        self.file_tree.configure(xscrollcommand=horizontal.set)
        self.file_tree.bind('<<TreeviewSelect>>', self._selected)
        self._button(left, text='Refresh file list', command=self.refresh_files).pack(fill='x', pady=4)
        controls = ttk.Frame(right)
        controls.pack(fill='x')
        ttk.Label(controls, text='Scale').grid(row=0, column=0)
        ttk.Combobox(controls, textvariable=self.scale, values=('lin', 'log'), state='readonly', width=6).grid(row=0, column=1, padx=4)
        ttk.Label(controls, text='Frame').grid(row=0, column=2)
        self.frame_spin = ttk.Spinbox(controls, from_=1, to=1, textvariable=self.frame, width=7)
        self.frame_spin.grid(row=0, column=3, padx=4)
        self._button(controls, text='Auto limits', command=self.auto_limits).grid(row=0, column=4, padx=6)
        ttk.Label(controls, text='Limits: counts (lin) or log10(counts) (log)').grid(row=0, column=5, sticky='w')
        self.sliders = []
        for index, (label, var) in enumerate((('Min', self.low), ('Max', self.high)), 1):
            ttk.Label(controls, text=label).grid(row=index, column=0)
            ttk.Entry(controls, textvariable=var, width=12).grid(row=index, column=1, columnspan=2, sticky='ew', padx=4)
            slider = ttk.Scale(controls, from_=-10, to=20, command=lambda value, v=var: v.set(f'{float(value):.5g}'))
            slider.grid(row=index, column=3, columnspan=3, sticky='ew', padx=4)
            self.sliders.append(slider)
        controls.columnconfigure(5, weight=1)
        split = ttk.Panedwindow(right, orient='vertical')
        split.pack(fill='both', expand=True)
        plot = ttk.Frame(split)
        split.add(plot, weight=4)
        self.figure = Figure(figsize=(6, 4), dpi=90, constrained_layout=True)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_xlabel('Detector x (pixel)')
        self.axes.set_ylabel('Detector y (pixel)')
        self.axes.set_title('Select a raw HDF file')
        self.artist = None
        self.colorbar = None
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, plot, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill='x')
        metadata = ttk.Frame(split)
        split.add(metadata, weight=2)
        meta_header = ttk.Frame(metadata)
        meta_header.pack(fill='x')
        ttk.Label(meta_header, text='Metadata', font=('Arial', 10, 'bold')).pack(side='left')
        ttk.Entry(meta_header, textvariable=self.meta_filter, width=30).pack(side='right', padx=4)
        meta_frame = ttk.Frame(metadata)
        meta_frame.pack(fill='both', expand=True)
        self.meta_tree = ttk.Treeview(meta_frame, columns=('field', 'value'), show='headings', height=5)
        self.meta_tree.heading('field', text='HDF field / path')
        self.meta_tree.heading('value', text='Value')
        self.meta_tree.column('field', width=320)
        self.meta_tree.column('value', width=350)
        self.meta_tree.pack(side='left', fill='both', expand=True)
        scrollbar = ttk.Scrollbar(meta_frame, command=self.meta_tree.yview)
        scrollbar.pack(side='right', fill='y')
        self.meta_tree.configure(yscrollcommand=scrollbar.set)
        background = ttk.Frame(self)
        background.pack(fill='x', pady=5)
        ttk.Checkbutton(background, text='Subtract background', variable=self.bg_enabled, command=self.request_frame).pack(side='left')
        ttk.Label(background, text='Background scan').pack(side='left', padx=(10, 4))
        bg_entry = ttk.Entry(background, textvariable=self.bg_scan, width=10)
        bg_entry.pack(side='left')
        bg_entry.bind('<Return>', lambda _: self.request_frame())
        bg_entry.bind('<FocusOut>', lambda _: self.request_frame() if self.bg_enabled.get() else None)
        options = ttk.LabelFrame(self, text='Batch export — existing 2D options', padding=6)
        options.pack(fill='x')
        ttk.Label(options, text='Scans / ranges').grid(row=0, column=0)
        ttk.Entry(options, textvariable=self.scans, width=30).grid(row=0, column=1, sticky='ew', padx=4)
        self._button(options, text='Use selected scan', command=self.use_selected_scan).grid(row=0, column=2, padx=4)
        ttk.Combobox(options, textvariable=self.output, values=('individual_frames', 'gif_animation'), state='readonly', width=19).grid(row=0, column=3, padx=4)
        ttk.Label(options, text='GIF fps').grid(row=1, column=0)
        ttk.Spinbox(options, from_=0.1, to=60, increment=0.5, textvariable=self.fps, width=8).grid(row=1, column=1, sticky='w', padx=4)
        ttk.Checkbutton(options, text='Play once', variable=self.play_once).grid(row=1, column=2, sticky='w')
        self._button(options, text='Export scan list (PNG/GIF)', command=lambda: self.host.run_script('caller_plot_2Dpattern.py')).grid(row=1, column=3, padx=4)
        options.columnconfigure(1, weight=1)
        footer = ttk.Frame(self)
        footer.pack(fill='x', pady=(6, 0))
        self._button(footer, text='Save settings', command=self.host.save_data).pack(side='left')
        self._button(footer, text='Save displayed image…', command=self.save_image).pack(side='left', padx=6)
        ttk.Label(footer, textvariable=self.status, wraplength=650).pack(side='left', padx=8)

    def _register_entries(self):
        self.host.entries['plot_2d'] = {
            ('clim',): _LimitsEntry(self), ('plot_scale',): self.scale, ('log_base',): self.log_base,
            ('list_scan',): self.scans, ('enable_bg_subtraction',): self.bg_enabled,
            ('background_scan_nr',): self.bg_scan, ('output_mode',): self.output,
            ('gif_fps',): self.fps, ('play_once',): self.play_once,
        }

    def validate_settings(self):
        try:
            validate_limits(self.low.get(), self.high.get())
            fps = float(self.fps.get())
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError('GIF frames per second must be positive.')
            if self.bg_enabled.get():
                int(self.bg_scan.get())
            return True
        except (ValueError, TypeError) as exc:
            self.status.set(str(exc))
            return False

    def _submit(self, job, callback):
        self.generation += 1
        generation = self.generation
        if self.pending:
            self.pending.cancel()
        future = self.executor.submit(job)
        self.pending = future

        def poll():
            if self.closed or generation != self.generation:
                return
            if not future.done():
                self._later(35, poll)
                return
            try:
                value = future.result()
            except Exception as exc:
                self.status.set(f'Cannot preview: {exc}')
                return
            callback(value)
        self._later(35, poll)

    def refresh_files(self):
        self.folder = self._resolve(self.host.config_dict['analysis_paths']['raw_data'])
        self.status.set('Scanning raw_data…')

        def loaded(files):
            self.files = files
            self._filter_files()
            self._index_sample_names(files)
            self.status.set(f'{len(files)} HDF files. Select a file to preview.')
        self._submit(lambda: find_files(self.folder), loaded)

    def _index_sample_names(self, files):
        self.name_stop.set()
        self.name_stop = Event()
        stop = self.name_stop
        self.name_generation += 1
        generation = self.name_generation
        updates = SimpleQueue()
        instrument = self.host.config_dict['instrument_setup']['which_instrument']

        def read_names():
            for index, path in enumerate(files):
                if stop.is_set():
                    return
                try:
                    name = self.name_reader.read(path, instrument)
                except (OSError, ValueError, TypeError):
                    name = '(unavailable)'
                updates.put((index, path, name))

        future = self.name_executor.submit(read_names)

        def poll():
            if self.closed or generation != self.name_generation:
                return
            changed = False
            for _ in range(128):
                try:
                    index, path, name = updates.get_nowait()
                except Empty:
                    break
                self.sample_names[path] = name
                if self.file_tree.exists(str(index)):
                    self.file_tree.set(str(index), 'sample', name)
                changed = True
            if changed and self.filter.get().strip():
                self._filter_files()
            if not future.done() or not updates.empty():
                self._later(50, poll)
        self._later(50, poll)

    def _filter_files(self):
        self.file_tree.delete(*self.file_tree.get_children())
        query = self.filter.get().lower().strip()
        for index, path in enumerate(self.files):
            label = str(path.relative_to(self.folder))
            sample = self.sample_names.get(path, 'Reading…')
            if query in (label + ' ' + sample).lower():
                self.file_tree.insert('', 'end', iid=str(index), text=label, values=(sample,))
                if path == self.selected_path:
                    self.file_tree.selection_set(str(index))

    def _selected(self, _event=None):
        selection = self.file_tree.selection()
        if not selection:
            return
        path = self.files[int(selection[0])]
        if path == self.selected_path:
            return
        self.selected_path = path
        self.raw = self.current = self.background = None
        self.meta_tree.delete(*self.meta_tree.get_children())
        if self.artist is not None:
            self.artist.set_visible(False)
            self.canvas.draw_idle()
        self._updating = True
        self.frame.set('1')
        self._updating = False
        self.request_frame()

    def _schedule_frame(self, *_):
        if self._updating:
            return
        self._cancel(self.frame_after)
        self.frame_after = self._later(120, self.request_frame)

    def request_frame(self):
        if self.selected_path is None:
            return
        try:
            frame = int(self.frame.get()) - 1
            bg_scan = int(self.bg_scan.get()) if self.bg_enabled.get() else None
            path = self.selected_path
            instrument = self.host.config_dict['instrument_setup']['which_instrument']
            matches = [p for p in self.files if scan_number(p) == bg_scan] if bg_scan is not None else []
            if bg_scan is not None and len(matches) != 1:
                raise ValueError(f'Background scan {bg_scan}: found {len(matches)} matching files; choose an unambiguous scan.')
            background_path = matches[0] if matches else None
        except ValueError as exc:
            self.status.set(str(exc))
            return
        self.status.set(f'Loading {path.name}, frame {frame + 1}…')

        def load():
            result = self.reader.read(path, instrument, frame)
            # Keep the established export behavior: first frame of background.
            background = self.reader.read(background_path, instrument, 0).image if background_path else None
            if background is not None and background.shape != result.image.shape:
                raise ValueError('Background and main detector shapes differ; subtraction was not applied.')
            return result, background

        def loaded(value):
            changed_file = self.current is None or self.current.path != value[0].path
            self.current, self.background = value
            self.raw = self.current.image
            self.frame_spin.configure(to=self.current.frames, state='normal' if self.current.frames > 1 else 'disabled')
            self._show_metadata()
            if changed_file:
                self.auto_limits()
            else:
                self.render(ranges=True)
        self._submit(load, loaded)

    def _show_metadata(self):
        self.meta_tree.delete(*self.meta_tree.get_children())
        if not self.current:
            return
        query = self.meta_filter.get().lower().strip()
        for field, value in self.current.metadata:
            if query in (field + ' ' + value).lower():
                self.meta_tree.insert('', 'end', values=(field, value))

    def _scale_changed(self, *_):
        if self.raw is not None:
            self.auto_limits()
        else:
            self._schedule_render(ranges=True)

    def _schedule_render(self, *_, ranges=False):
        if self._updating:
            return
        self._cancel(self.render_after)
        self.render_after = self._later(35, lambda: self.render(ranges=ranges))

    def render(self, ranges=False):
        if self.raw is None:
            return
        try:
            low, high = validate_limits(self.low.get(), self.high.get())
            image = display_image(self.raw, self.scale.get(), self.background)
            if self.artist is None:
                self.artist = self.axes.imshow(image, origin='lower', cmap='jet', vmin=low, vmax=high)
                self.colorbar = self.figure.colorbar(self.artist, ax=self.axes, fraction=0.046, pad=0.04, panchor=False)
            else:
                self.artist.set_visible(True)
                self.artist.set_data(image)
                self.artist.set_extent((-0.5, image.shape[1] - 0.5, -0.5, image.shape[0] - 0.5))
                self.artist.set_clim(low, high)
            self.axes.set_title(f'{self.current.path.name} · frame {self.current.frame + 1}/{self.current.frames}')
            self.colorbar.set_label('log10(counts)' if self.scale.get() == 'log' else 'Counts')
            self.axes.set_xlim(-0.5, image.shape[1] - 0.5)
            self.axes.set_ylim(-0.5, image.shape[0] - 0.5)
            if ranges:
                finite = image.compressed()
                finite = finite[np.isfinite(finite)]
                lower = min(float(finite.min()), low) if finite.size else low
                upper = max(float(finite.max()), high) if finite.size else high
                if lower == upper:
                    upper = lower + 1
                for slider, value in zip(self.sliders, (low, high)):
                    slider.configure(from_=lower, to=upper)
                    slider.set(value)
            self.canvas.draw_idle()
            self.status.set(f'{self.current.path.name} · {image.shape[1]} × {image.shape[0]} pixels' + (' · background subtracted' if self.background is not None else ''))
        except (ValueError, TypeError) as exc:
            self.status.set(str(exc))

    def auto_limits(self):
        if self.raw is None:
            return
        image = display_image(self.raw, self.scale.get(), self.background).compressed()
        if not image.size:
            return
        low, high = np.percentile(image, [1, 99.7])
        if high <= low:
            high = low + 1
        self._updating = True
        self.low.set(f'{low:.6g}')
        self.high.set(f'{high:.6g}')
        self._updating = False
        self.render(ranges=True)

    def use_selected_scan(self):
        if self.selected_path is not None:
            scan = scan_number(self.selected_path)
            if scan is not None:
                self.scans.set(str(scan))

    def save_image(self):
        if self.artist is None or not self.validate_settings():
            self.status.set('Select a valid detector image and color limits first.')
            return
        folder = self._resolve(self.host.config_dict['analysis_paths']['project_base']) / 'extra_results'
        folder.mkdir(parents=True, exist_ok=True)
        name = re.sub(r'[^\w.-]+', '_', self.current.path.stem)
        path = filedialog.asksaveasfilename(parent=self, initialdir=str(folder),
                    initialfile=f'{name}_frame_{self.current.frame + 1:03d}.png',
                    defaultextension='.png', filetypes=[('PNG image', '*.png'), ('PDF', '*.pdf')])
        if not path:
            return
        try:
            self.render()
            self.figure.savefig(path, dpi=150)
            if self.host.save_data():
                self.status.set(f'Saved image and last settings: {path}')
            else:
                self.status.set(f'Image saved, but settings could not be saved: {path}')
        except (OSError, ValueError) as exc:
            messagebox.showerror('Cannot save image', str(exc), parent=self)

    def _destroyed(self, event):
        if event.widget is not self or self.closed:
            return
        self.closed = True
        self.name_stop.set()
        self.name_executor.shutdown(wait=False, cancel_futures=True)
        for token in tuple(self.after_ids):
            self.after_cancel(token)
        self.after_ids.clear()
        if self.pending:
            self.pending.cancel()
        self.executor.shutdown(wait=False, cancel_futures=True)
        if self.canvas._idle_draw_id:
            self.canvas._tkcanvas.after_cancel(self.canvas._idle_draw_id)
        # The toolbar widgets have already been destroyed by Tk at this point.
        self.canvas.toolbar = None
        self.figure.clear()
