"""Add missing DarePy settings without replacing saved values (Python 3.9+)."""
import argparse
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
import os
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.request import urlopen

try:
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap
except ImportError:
    raise SystemExit('Install the dependency first: python -m pip install ruamel.yaml')

GITHUB_URL = ('https://raw.githubusercontent.com/vivianel/DarePy-SANS/main/'
              'experiments/default_exp/config_experiment.yaml')
REFERENCE = Path(__file__).resolve().parents[1] / 'experiments' / 'default_exp' / 'config_experiment.yaml'
# These mappings hold experiment data, rather than a fixed set of setting names.
DYNAMIC = {
    'beam_center_mask.scans', 'detector_geometry.beam_center_guess',
    'detector_geometry.beamstopper_coordinates',
    'detector_geometry.transmission_coordinates', 'calibration_samples.thickness',
    'calibration_samples.empty_cell', 'merging_settings.skip_start',
    'merging_settings.skip_end',
}
# Explicit, reviewed moves/renames; never infer migrations from similar names.
ALIASES = {
    'pipeline_control.force_reintegrate': 'analysis_flags.force_reintegrate',
    'pipeline_control.exclude_files': 'analysis_flags.exclude_files',
    'pipeline_control.wavelength': 'physics_corrections.wavelength',
    'pipeline_control.target_detector_distances': 'physics_corrections.target_detector_distances',
    'transmission_setup.dist_trans_measurements': 'transmission_setup.transmission_dist',
    'transmission_setup.empty_beam': 'calibration_samples.empty_beam',
    'rename_samples.add_subscript': 'rename_samples.subscript',
    'analysis_flags.add_plot_azimuthal': 'analysis_flags.save_plot_azimuthal',
}
MISSING = object()


def yaml_engine():
    y = YAML()
    y.preserve_quotes = True
    y.allow_duplicate_keys = False
    y.width = 120
    return y


def parse(text):
    data = yaml_engine().load(text)
    if not isinstance(data, Mapping):
        raise ValueError('Configuration must contain a YAML mapping at the top level.')
    return data


def dump(data):
    stream = StringIO()
    yaml_engine().dump(data, stream)
    return stream.getvalue()


def get(data, path):
    for key in path.split('.'):
        if not isinstance(data, Mapping) or key not in data:
            return MISSING
        data = data[key]
    return data


@dataclass
class Addition:
    path: str
    value: object
    source: str


@dataclass
class Plan:
    data: object
    additions: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)


def complete_config(original, reference):
    """Return an independent plan. Existing keys, including nulls/lists, are kept."""
    plan = Plan(deepcopy(original))

    def merge(target, template, prefix=''):
        for key, default in template.items():
            path = f'{prefix}.{key}' if prefix else str(key)
            if path in DYNAMIC:
                if key not in target:
                    target[key] = CommentedMap()
                    plan.additions.append(Addition(path, {}, 'empty experiment-specific mapping'))
                    plan.warnings.append(f'{path}: no experiment data copied; populate if required.')
                elif not isinstance(target[key], Mapping):
                    plan.warnings.append(f'{path}: saved non-mapping value preserved; review manually.')
                # A named default is a setting; sample names/distances are not.
                if isinstance(target[key], Mapping) and isinstance(default, Mapping):
                    if 'default' in default and 'default' not in target[key]:
                        target[key]['default'] = deepcopy(default['default'])
                        plan.additions.append(Addition(path + '.default', deepcopy(default['default']), 'reference; review experiment value'))
                        plan.warnings.append(f'{path}.default: reference experiment value needs review.')
                continue
            if key not in target:
                old_path = ALIASES.get(path)
                old = get(original, old_path) if old_path else MISSING
                if old is not MISSING:
                    target[key] = deepcopy(old)
                    plan.additions.append(Addition(path, deepcopy(old), f'saved value from {old_path}'))
                elif isinstance(default, Mapping):
                    target[key] = CommentedMap()
                    merge(target[key], default, path)
                else:
                    target[key] = deepcopy(default)
                    plan.additions.append(Addition(path, deepcopy(default), 'reference'))
                if hasattr(template, 'ca') and key in template.ca.items:
                    target.ca.items[key] = deepcopy(template.ca.items[key])
            elif isinstance(default, Mapping):
                if isinstance(target[key], Mapping):
                    merge(target[key], default, path)
                else:
                    plan.errors.append(f'{path}: expected a section, but a saved value exists. Preserved; cannot fill its children.')

    merge(plan.data, reference)
    for new, old in ALIASES.items():
        a, b = get(original, new), get(original, old)
        if a is not MISSING and b is not MISSING and a != b:
            plan.warnings.append(f'{new} and legacy {old} differ; both preserved, current code uses {new}.')
    mode = get(original, 'plot_2d.output_mode')
    if isinstance(mode, str) and mode in ('frames', 'gif'):
        replacement = {'frames': 'individual_frames', 'gif': 'gif_animation'}[mode]
        plan.warnings.append(f'plot_2d.output_mode={mode!r} preserved; current reference uses {replacement!r}. Change manually if needed.')
    for item in plan.additions:
        if item.path.startswith(('resolution_settings.', 'physics_corrections.', 'transmission_setup.', 'merging_settings.')) and not item.path.endswith('._desc'):
            plan.warnings.append(f'{item.path}: review the added scientific/processing default before reduction.')
        if item.path in ('analysis_paths.project_base', 'analysis_paths.raw_data', 'analysis_paths.scripts_dir', 'instrument_setup.which_instrument'):
            plan.warnings.append(f'{item.path}: reference value is experiment-specific; adjust before use.')
    return plan


def assert_preserved(original, updated, path=''):
    """Verify every original key/value recursively; sequences are indivisible."""
    if isinstance(original, Mapping):
        if not isinstance(updated, Mapping):
            raise ValueError(f'Saved section changed: {path}')
        for key, value in original.items():
            if key not in updated:
                raise ValueError(f'Saved key lost: {path}.{key}')
            assert_preserved(value, updated[key], f'{path}.{key}')
    elif original != updated and not (isinstance(original, float) and isinstance(updated, float) and original != original and updated != updated):
        raise ValueError(f'Saved value changed: {path}')


def save_plan(plan, original_text, source, destination):
    if plan.errors:
        raise ValueError('Resolve the conflicting sections before saving: ' + '; '.join(plan.errors))
    source, destination = Path(source), Path(destination)
    if source.read_text(encoding='utf-8-sig') != original_text:
        raise ValueError('Source changed since preview. Reload and compare again.')
    original = parse(original_text)
    # Re-read serialized data to also verify that YAML output preserves values.
    text = dump(plan.data) if plan.additions else original_text
    assert_preserved(original, parse(text))
    in_place = source.resolve() == destination.resolve()
    if destination.exists() and not in_place:
        raise ValueError('Output already exists. Choose a new filename.')
    if in_place:
        backup = source.with_name(source.name + '.' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.bak')
        shutil.copy2(source, backup)
    fd, temp = tempfile.mkstemp(prefix='.config_', suffix='.tmp', dir=str(destination.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as stream:
            stream.write(text)
        if in_place:
            os.replace(temp, destination)
        else:
            # Exclusive creation prevents overwriting another output accidentally.
            with open(destination, 'x', encoding='utf-8', newline='') as stream:
                stream.write(text)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return destination


def load_reference(path=None, github=False):
    if github:
        with urlopen(GITHUB_URL, timeout=20) as response:
            return parse(response.read().decode('utf-8-sig'))
    return parse(Path(path or REFERENCE).read_text(encoding='utf-8-sig'))


def load_config_file(filepath, reference=None, reporter=print):
    """Complete a config in memory using the checked-out reference, without writing.

    Return the plan so a GUI can highlight additions. Scripts use plan.data.
    Explicit nulls and malformed sections are never silently replaced.
    """
    original = parse(Path(filepath).read_text(encoding='utf-8-sig'))
    plan = complete_config(original, load_reference(reference))
    if plan.errors:
        raise ValueError('Configuration cannot be completed: ' + '; '.join(plan.errors))
    assert_preserved(original, plan.data)
    if reporter is not None:
        for item in plan.additions:
            reporter(f'[CONFIG ADDED] {item.path} = {item.value!r} [{item.source}]')
        for warning in plan.warnings:
            reporter(f'[CONFIG REVIEW] {warning}')
    return plan


def backup_config(filepath):
    """Keep the on-disk config before the GUI first persists completed defaults."""
    source = Path(filepath)
    backup = source.with_name(source.name + '.' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.bak')
    shutil.copy2(source, backup)
    return backup


def launch_gui(initial=None, reference=None, github=False):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    root = tk.Tk()
    root.title('DarePy — Complete configuration')
    root.geometry('1120x760')
    root.minsize(800, 550)
    source_var = tk.StringVar(value=str(initial or ''))
    ref_var = tk.StringVar(value=str(reference or REFERENCE))
    online = tk.BooleanVar(value=github)
    state = {}
    top = ttk.Frame(root, padding=12)
    top.pack(fill='x')

    def pick(var):
        name = filedialog.askopenfilename(filetypes=[('YAML configuration', '*.yaml *.yml'), ('All files', '*.*')])
        if name:
            var.set(name)
    for label, var in [('Your configuration', source_var), ('Reference configuration', ref_var)]:
        row = ttk.Frame(top)
        row.pack(fill='x', pady=3)
        ttk.Label(row, text=label, width=24).pack(side='left')
        ttk.Entry(row, textvariable=var).pack(side='left', fill='x', expand=True)
        ttk.Button(row, text='Browse…', command=lambda v=var: pick(v)).pack(side='left', padx=6)
    ttk.Checkbutton(top, text='Use latest reference from GitHub (requires internet)', variable=online).pack(anchor='w')
    ttk.Label(top, text='Saved values are preserved. Red rows are additions. Review defaults before using the completed file.').pack(anchor='w', pady=8)
    tree_frame = ttk.Frame(root, padding=(12, 0))
    tree_frame.pack(fill='both', expand=True)
    tree = ttk.Treeview(tree_frame, columns=('path', 'value', 'source'), show='headings')
    for col, title, width in [('path', 'Added setting', 400), ('value', 'Value', 250), ('source', 'Source', 340)]:
        tree.heading(col, text=title)
        tree.column(col, width=width)
    tree.tag_configure('added', foreground='#b00020', background='#fff0f1')
    tree.pack(side='left', fill='both', expand=True)
    scroll = ttk.Scrollbar(tree_frame, orient='vertical', command=tree.yview)
    scroll.pack(side='right', fill='y')
    tree.configure(yscrollcommand=scroll.set)
    notes = tk.Text(root, height=9, wrap='word', padx=10, pady=8)
    notes.pack(fill='x', padx=12, pady=8)
    notes.configure(state='disabled')
    status = tk.StringVar(value='Select your file and choose Preview additions.')
    ttk.Label(root, textvariable=status, padding=12).pack(side='bottom', fill='x')
    buttons = ttk.Frame(root, padding=12)
    buttons.pack(fill='x')

    def show_plan():
        plan = state['plan']
        tree.delete(*tree.get_children())
        for i, item in enumerate(plan.additions):
            tree.insert('', 'end', iid=str(i), values=(item.path, repr(item.value), item.source), tags=('added',))
        notes.configure(state='normal')
        notes.delete('1.0', 'end')
        notes.insert('end', '\n'.join(['ERROR: ' + x for x in plan.errors] + ['REVIEW: ' + x for x in plan.warnings]) or 'No warnings.')
        notes.configure(state='disabled')
        status.set(f'{len(plan.additions)} additions; {len(plan.warnings)} review notes; {len(plan.errors)} blocking conflicts. Double-click an added value to edit it.')

    def preview():
        try:
            source = Path(source_var.get())
            text = source.read_text(encoding='utf-8-sig')
            state.update(source=source, text=text, plan=complete_config(parse(text), load_reference(ref_var.get(), online.get())))
            show_plan()
        except Exception as exc:
            state.clear()
            tree.delete(*tree.get_children())
            status.set('Preview failed. Select valid files and try again.')
            messagebox.showerror('Cannot preview', str(exc))

    def edit(event):
        row = tree.identify_row(event.y)
        if not row or 'plan' not in state:
            return
        item = state['plan'].additions[int(row)]
        dialog = tk.Toplevel(root)
        dialog.title('Edit added setting')
        dialog.transient(root)
        ttk.Label(dialog, text=item.path, padding=10).pack(anchor='w')
        editor = tk.Text(dialog, width=70, height=8)
        editor.pack(padx=10)
        editor.insert('1.0', dump(item.value))

        def apply():
            try:
                value = yaml_engine().load(editor.get('1.0', 'end'))
                target = state['plan'].data
                parts = item.path.split('.')
                for part in parts[:-1]:
                    target = target[part]
                target[parts[-1]] = value
                item.value, item.source = value, 'edited in preview'
                show_plan()
                dialog.destroy()
            except Exception as exc:
                messagebox.showerror('Invalid value', str(exc), parent=dialog)
        ttk.Button(dialog, text='Apply value', command=apply).pack(pady=10)

    def save():
        if 'plan' not in state:
            messagebox.showinfo('Preview first', 'Preview the additions before saving.')
            return
        if Path(source_var.get()).resolve() != state['source'].resolve():
            messagebox.showerror('File changed', 'Preview the selected configuration again.')
            return
        source = state['source']
        name = filedialog.asksaveasfilename(initialdir=str(source.parent), initialfile=source.stem + '_completed.yaml', defaultextension='.yaml', filetypes=[('YAML', '*.yaml')])
        if not name:
            return
        try:
            output = save_plan(state['plan'], state['text'], source, name)
            status.set(f'Saved: {output}')
            messagebox.showinfo('Configuration saved', f'Saved to:\n{output}\n\nReview notes remain relevant before reduction.')
        except Exception as exc:
            messagebox.showerror('Cannot save', str(exc))

    tree.bind('<Double-1>', edit)
    ttk.Button(buttons, text='Preview additions', command=preview).pack(side='left')
    ttk.Button(buttons, text='Save completed configuration…', command=save).pack(side='right')
    if initial:
        root.after(100, preview)
    root.mainloop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?', type=Path)
    parser.add_argument('--gui', action='store_true', help='Open the preview window; also the default with no arguments')
    refs = parser.add_mutually_exclusive_group()
    refs.add_argument('--reference', type=Path, help='Local reference YAML; defaults to experiments/default_exp/config_experiment.yaml')
    refs.add_argument('--github', action='store_true', help='Fetch current reference from GitHub')
    writes = parser.add_mutually_exclusive_group()
    writes.add_argument('--output', type=Path, help='Write a new file (existing output files are refused)')
    writes.add_argument('--in-place', action='store_true', help='Update source with a timestamped backup')
    args = parser.parse_args(argv)
    if args.gui and (args.output or args.in_place):
        parser.error('Use the GUI Save button, or omit --gui for command-line writing.')
    if args.gui or args.config is None:
        if args.output or args.in_place:
            parser.error('A configuration path is required when writing.')
        launch_gui(args.config, args.reference, args.github)
        return 0
    text = args.config.read_text(encoding='utf-8-sig')
    plan = complete_config(parse(text), load_reference(args.reference, args.github))
    for item in plan.additions:
        print(f'ADD {item.path} = {item.value!r} [{item.source}]')
    for warning in plan.warnings:
        print(f'REVIEW {warning}')
    for error in plan.errors:
        print(f'ERROR {error}', file=sys.stderr)
    print(f'{len(plan.additions)} additions, {len(plan.warnings)} review notes, {len(plan.errors)} blocking conflicts.')
    if plan.errors:
        return 2
    if args.output or args.in_place:
        print('Saved:', save_plan(plan, text, args.config, args.config if args.in_place else args.output))
    else:
        print('Preview only. Use --output FILE or --in-place to save.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(2)
