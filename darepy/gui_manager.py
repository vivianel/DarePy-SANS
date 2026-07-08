import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os, ast, shutil
from datetime import datetime
from ruamel.yaml import YAML
import sys

# --- DYNAMIC PATH SETUP ---
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

yaml = YAML()
yaml.preserve_quotes = True

class DarePyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DarePy-SANS Control Panel")
        self.root.geometry("1100x1000") # Slightly widened to comfortably hold dual buttons

        self.config_file = None

        # 1. Check for command line argument (passed by Spyder %run)
        if len(sys.argv) > 1 and sys.argv[1].endswith('.yaml'):
            self.config_file = sys.argv[1]

        # 2. File Selector if no argument found
        if not self.config_file:
            self.root.withdraw()

            # --- SPYDER FREEZE FIX ---
            self.root.attributes('-topmost', True)
            self.root.update()

            initial_dir = os.path.join(root_dir, "experiments")
            self.config_file = filedialog.askopenfilename(
                parent=self.root,
                initialdir=initial_dir,
                title="Select config_experiment.yaml",
                filetypes=(("YAML files", "*.yaml"), ("All files", "*.*"))
            )

            self.root.attributes('-topmost', False)
            # -------------------------

            if not self.config_file:
                sys.exit()

            self.root.deiconify()

        self.setup_styles()
        self.entries = {}

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)

        self.setup_console()

        if self.load_data():
            self.fill_notebook()
        else:
            self.log_to_console("CRITICAL: Application started with empty/broken config.")

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TNotebook", background="#f0f0f0", borderwidth=0)
        style.configure("TNotebook.Tab", padding=[15, 5], font=('Arial', 9, 'bold'),
                        background="#dcdcdc", foreground="#333333", borderwidth=1)
        style.map("TNotebook.Tab",
                  background=[("selected", "#2196F3"), ("active", "#cce5ff")],
                  foreground=[("selected", "white"), ("active", "black")])

    def setup_console(self):
        c_frame = tk.LabelFrame(self.root, text="Console Output", padx=5, pady=5)
        c_frame.pack(fill="x", side="bottom", padx=10, pady=5)
        self.console = tk.Text(c_frame, height=8, bg="#1e1e1e", fg="#4CAF50",
                               font=("Consolas", 9), wrap="word")
        self.console.pack(side="left", fill="x", expand=True)
        scrollbar = ttk.Scrollbar(c_frame, command=self.console.yview)
        scrollbar.pack(side="right", fill="y")
        self.console.config(yscrollcommand=scrollbar.set)
        self.log_to_console(f"System Ready. Loaded: {os.path.basename(self.config_file)}")

    def log_to_console(self, msg):
        self.console.insert(tk.END, f"{msg}\n")
        self.console.see(tk.END)
        self.root.update()

    def load_data(self):
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config_dict = yaml.load(f)
            return True
        except Exception as e:
            self.log_to_console(f"[YAML LOAD ERROR] {e}")
            return False

    def save_data(self):
        try:
            # Update the dictionary from UI entries
            for s_key, keys in self.entries.items():
                for path, entry in keys.items():
                    raw = entry.get()
                    actual = raw if isinstance(raw, bool) else self._parse_string(raw)
                    target = self.config_dict[s_key]
                    for p in path[:-1]:
                        if p not in target: target[p] = {}
                        target = target[p]

                    last_key = path[-1]
                    # FIX: Update sequences and mappings in-place to preserve ruamel.yaml types/formatting
                    if isinstance(actual, list) and isinstance(target.get(last_key), list):
                        target[last_key][:] = actual
                    elif isinstance(actual, dict) and isinstance(target.get(last_key), dict):
                        target[last_key].update(actual)
                    else:
                        target[last_key] = actual

            # Write the current file
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_dict, f)

            self.log_to_console(f"✅ Settings saved to: {os.path.basename(self.config_file)}")

            # Trigger the smart archiving
            self.archive_config()

            return True
        except Exception as e:
            self.log_to_console(f"❌ SAVE ERROR: {e}")
            return False

    def _parse_string(self, raw_val):
        v = str(raw_val).strip()
        if v == "": return ""
        if v.lower() == 'true': return True
        if v.lower() == 'false': return False

        # FIX: If the string contains a colon, it's a range shorthand (e.g., '10:30').
        # Keep it completely intact as a raw string so it doesn't get corrupted.
        if ":" in v:
            return v

        if "," in v and not v.startswith('['):
            try: return [self._parse_string(x) for x in v.split(",")]
            except: return v
        if v.startswith(('[', '{')):
            try: return ast.literal_eval(v)
            except: return v
        try: return float(v) if '.' in v else int(v)
        except: return v

    def archive_config(self):
        """Archives the config file only if changes are detected compared to the latest version."""
        import glob
        try:
            # 1. Setup paths
            experiment_dir = os.path.dirname(os.path.abspath(self.config_file))
            history_dir = os.path.join(experiment_dir, "config_history")
            project_name = os.path.basename(experiment_dir)

            if not os.path.exists(history_dir):
                os.makedirs(history_dir)

            # 2. Find the most recent history file for comparison
            history_files = glob.glob(os.path.join(history_dir, f"config_{project_name}_*.yaml"))
            history_files.sort(key=os.path.getmtime, reverse=True)

            should_archive = True
            if history_files:
                latest_history = history_files[0]
                # Compare current file content with the latest archived version
                with open(self.config_file, 'r', encoding='utf-8') as current_f, \
                     open(latest_history, 'r', encoding='utf-8') as history_f:
                    if current_f.read() == history_f.read():
                        should_archive = False

            # 3. Perform the archive if changes exist
            if should_archive:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                archive_path = os.path.join(history_dir, f"config_{project_name}_{timestamp}.yaml")
                shutil.copy2(self.config_file, archive_path)
                self.log_to_console(f"📦 Archive created: config_{project_name}_{timestamp}.yaml")
            else:
                self.log_to_console("ℹ️ No changes detected; skipping archive.")

        except Exception as e:
            self.log_to_console(f"⚠️ Archiving warning: {e}")

    def refresh_ui(self):
        try: main_idx = self.notebook.index("current")
        except: main_idx = 0
        sub_idx = 0
        if hasattr(self, 'sub_nb'):
            try: sub_idx = self.sub_nb.index("current")
            except: sub_idx = 0

        if self.load_data():
            self.entries = {}
            for child in self.notebook.winfo_children(): child.destroy()
            self.fill_notebook()
            self.root.update_idletasks()
            try:
                self.notebook.select(main_idx)
                self.root.update_idletasks()
                if main_idx == 5 and hasattr(self, 'sub_nb'): self.sub_nb.select(sub_idx)
            except: pass

    def create_scrollable_tab(self, notebook, text):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text=text)
        bg = self.root.cget('bg')
        footer = tk.Frame(tab, bg=bg)
        footer.pack(side="bottom", fill="x", padx=20, pady=15)
        container = tk.Frame(tab, bg=bg)
        container.pack(side="top", fill="both", expand=True)
        sb = ttk.Scrollbar(container, orient="vertical")
        sb.pack(side="right", fill="y")
        cv = tk.Canvas(container, highlightthickness=0, bg=bg)
        cv.pack(side="left", fill="both", expand=True)
        sf = tk.Frame(cv, bg=bg)
        cw = cv.create_window((0, 0), window=sf, anchor="nw")
        cv.configure(yscrollcommand=sb.set)
        sb.config(command=cv.yview)
        sf.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.bind("<Configure>", lambda e: cv.itemconfig(cw, width=e.width))
        cv.bind("<Enter>", lambda e: cv.bind_all("<MouseWheel>", lambda ev: cv.yview_scroll(int(-1*(ev.delta/120)), "units")))
        cv.bind("<Leave>", lambda e: cv.unbind_all("<MouseWheel>"))
        return sf, footer

    def build_config_area(self, parent, title, config_key):
        bg = parent.cget('bg')
        group = tk.LabelFrame(parent, text=title, padx=10, pady=10, bg=bg)
        group.pack(fill="x", padx=10, pady=5, anchor="w")
        if config_key not in self.entries: self.entries[config_key] = {}
        self._build_nested_ui(group, self.config_dict.get(config_key, {}), config_key, path=())

    def _build_nested_ui(self, parent, data, config_key, path):
        if not hasattr(data, "items"): return
        bg = parent.cget('bg')

        if "_desc" in data:
            desc_text = str(data["_desc"]).strip()
            desc_box = tk.Label(parent, text=desc_text, font=("Arial", 12, "italic"),
                                fg="#444", bg="#eef2f7", justify="left",
                                anchor="w", padx=12, pady=8, wraplength=800)
            desc_box.pack(fill="x", pady=(0, 10), anchor="w")

        for k, v in data.items():
            if str(k).startswith("_"): continue

            inline_comment = ""
            if hasattr(data, 'ca') and k in data.ca.items:
                comment_token = data.ca.items[k][2]
                if comment_token:
                    inline_comment = comment_token.value.strip().lstrip('#').strip()

            current_path = path + (k,)

            if isinstance(v, (dict, type(data))):
                sub = tk.LabelFrame(parent, text=str(k), padx=10, pady=5,
                                    font=("Arial", 12, "bold"), fg="#e67e22", bg=bg)
                sub.pack(fill="x", padx=5, pady=5, anchor="w")
                self._build_nested_ui(sub, v, config_key, current_path)
            else:
                self._create_field(parent, str(k), v, config_key, current_path, inline_comment)

    def _create_field(self, parent, label, value, config_key, path, comment=""):
        bg = parent.cget('bg')
        f = tk.Frame(parent, bg=bg)
        f.pack(fill="x", pady=4, anchor="w")

        label_frame = tk.Frame(f, bg=bg)
        label_frame.pack(fill="x", anchor="w")

        # If it is a boolean value, pack the checkbox inside the label row FIRST
        if isinstance(value, bool):
            var = tk.BooleanVar(master=self.root, value=value)
            tk.Checkbutton(label_frame, text="", variable=var, bg=bg).pack(side="left", padx=(0, 5))
            self.entries[config_key][path] = var

        # Pack the word/label next to the checkbox
        tk.Label(label_frame, text=label, font=("Arial", 9, "bold"), bg=bg).pack(side="left")

        # Pack the comment safely next to the label
        if comment:
            tk.Label(label_frame, text=f"  # {comment}", font=("Arial", 8, "italic"),
                     fg="#888", bg=bg).pack(side="left")

        # Handle options or entries only if it is NOT a boolean setup
        if not isinstance(value, bool):
            if label in ["which_instrument", "integration_direction", "beamstop", "plot_scale","interp_type", 'output_mode', "source_slit_shape", "sample_slit_shape"]:
                if label == "which_instrument":
                    opts = ["SANS-I", "SANS-LLB"]
                elif label == "integration_direction":
                    opts = ["horizontal", "vertical", "azimuthal"]
                elif label == "beamstop":
                    opts = ["semitransparent", "standard"]
                elif label in ["plot_scale", "interp_type"]:
                    opts = ["lin", "log"]
                elif label in ["sample_slit_shape", "source_slit_shape"]:
                    opts = ["auto", "square", "rectangular", "circular"]
                elif label in ['output_mode']:
                    opts = ['individual_frames', 'gif_animation']

                w = ttk.Combobox(f, values=opts, state="readonly")
                w.set(value)
                w.pack(anchor="w", ipady=2)
                self.entries[config_key][path] = w
            else:
                w = tk.Entry(f)
                disp = ", ".join(map(str, value)) if isinstance(value, list) else str(value)
                w.insert(0, disp)
                w.pack(fill="x", padx=(0, 40), ipady=3)
                self.entries[config_key][path] = w

    def fill_notebook(self):
        bg = self.root.cget('bg')

        # --- TAB 1: GLOBAL SETUP ---
        s1, f1 = self.create_scrollable_tab(self.notebook, "1. Global Setup")
        tk.Button(f1, text="SAVE SETTINGS", bg="#4CAF50", fg="white", font=("Arial", 10, "bold"),
                  command=self.save_data).pack(side="left", fill="x", expand=True, padx=2)
        tk.Button(f1, text="LIST ALL MEASUREMENTS (Metadata)", bg="#2196F3", fg="white", font=("Arial", 10, "bold"),
                  command=lambda: self.run_script("caller_listing.py")).pack(side="left", fill="x", expand=True, padx=2)
        self.build_config_area(s1, "Analysis Paths", "analysis_paths")
        self.build_config_area(s1, "Instrument", "instrument_setup")
        self.build_config_area(s1, "Sample Environment", "sample_environment")

        s2, f2 = self.create_scrollable_tab(self.notebook, "2. Pipeline control")
        self.build_config_area(s2, "Pipeline Control", "pipeline_control")

        # --- TAB 2: RENAME SAMPLES ---
        s3, f3 = self.create_scrollable_tab(self.notebook, "2. Rename Samples")
        tk.Button(f3, text="RENAME SAMPLES", bg="#2196F3", fg="white", font=("Arial", 10, "bold"),
                  command=lambda: self.run_script("caller_rename_samples.py")).pack(fill="x")
        self.build_config_area(s3, "Rename Settings", "rename_samples")

        # --- TAB 3: 2D VISUALIZATION ---
        s4, f4 = self.create_scrollable_tab(self.notebook, "3. 2D Visualization")
        tk.Button(f4, text="GENERATE 2D PLOTS/GIF", bg="#2196F3", fg="white", font=("Arial", 10, "bold"),
                  command=lambda: self.run_script("caller_plot_2Dpattern.py")).pack(fill="x")
        self.build_config_area(s4, "Plot 2D Settings", "plot_2d")

        # --- TAB 4: MASK & CENTER ---
        s5, f5 = self.create_scrollable_tab(self.notebook, "4. Mask & Center")

        # Middle Layout Setup Box: Distance list loops & Semitransparent Toggle
        act_f = tk.LabelFrame(s5, text="Distance-Specific Masking & Alignment", padx=10, pady=10, bg=bg, fg="#2196F3", font=("Arial", 10, "bold"))
        act_f.pack(fill="x", padx=10, pady=5)

        add_r = tk.Frame(act_f, bg=bg)
        add_r.pack(fill="x", pady=(0, 10))
        tk.Label(add_r, text="New Dist:", font=("Arial", 8, "italic"), bg=bg).pack(side="left")
        self.new_dist_entry = tk.Entry(add_r, width=8)
        self.new_dist_entry.pack(side="left", padx=5)
        tk.Button(add_r, text=" + ", bg="#4CAF50", fg="white", font=("Arial", 8, "bold"), command=self.add_new_distance).pack(side="left")

        # Initialize tracking sub-dictionary registry if not present
        if 'beam_center_mask' not in self.entries:
            self.entries['beam_center_mask'] = {}

        # semitransparent Boolean Checkbutton Setup
        semi_val = self.config_dict.get('beam_center_mask', {}).get('semitransparent', True)
        self.semi_var = tk.BooleanVar(value=semi_val)

        semi_chk = tk.Checkbutton(act_f, text="Run Semitransparent Beamstop Masking",
                                  variable=self.semi_var, font=("Arial", 9, "bold"),
                                  bg=bg, activebackground=bg, anchor="w")
        semi_chk.pack(anchor="w", pady=(5, 10))
        self.entries['beam_center_mask'][('semitransparent',)] = self.semi_var

        scans_dict = self.config_dict.get('beam_center_mask', {}).get('scans', {})
        if isinstance(scans_dict, dict):
            for d in sorted(scans_dict.keys(), key=float):
                r = tk.Frame(act_f, bg=bg)
                r.pack(fill="x", pady=2)
                tk.Label(r, text=f"{d} m:", width=8, font=("Arial", 9, "bold"), bg=bg, anchor="w").pack(side="left")
                e = tk.Entry(r, width=12)
                e.insert(0, str(scans_dict[d]))
                e.pack(side="left", padx=5)
                self.entries['beam_center_mask'][('scans', d)] = e

                tk.Button(r, text="MASKING", bg="#2196F3", fg="white", font=("Arial", 8, "bold"), command=lambda dist=d: self.run_mask_trans_for_dist(dist)).pack(side="left", fill="x", expand=True, padx=2)
                tk.Button(r, text="BEAM CENTER", bg="#4CAF50", fg="white", font=("Arial", 8, "bold"), command=lambda dist=d: self.run_beam_center_for_dist(dist)).pack(side="left", fill="x", expand=True, padx=2)
                tk.Button(r, text=" 🗑️ ", bg="#f44336", fg="white", font=("Arial", 8, "bold"), command=lambda dist=d: self.remove_distance(dist)).pack(side="right", padx=(5, 0))

        # Bottom Additional Box: clim & plot_scale
        options_f = tk.LabelFrame(s5, text="Display & Intensity Options", padx=10, pady=10, bg=bg)
        options_f.pack(fill="x", padx=10, pady=5)

        # clim Field Setup
        tk.Label(options_f, text="clim", font=("Arial", 9, "bold"), bg=bg).pack(anchor="w")
        clim_val = self.config_dict.get('beam_center_mask', {}).get('clim', [0, 100])
        clim_disp = ", ".join(map(str, clim_val)) if isinstance(clim_val, list) else str(clim_val)
        clim_ent = tk.Entry(options_f)
        clim_ent.insert(0, clim_disp)
        clim_ent.pack(fill="x", ipady=3, pady=(0, 10))
        self.entries['beam_center_mask'][('clim',)] = clim_ent

        # plot_scale Dropdown Selection Menu Setup
        tk.Label(options_f, text="plot_scale", font=("Arial", 9, "bold"), bg=bg).pack(anchor="w")
        scale_val = self.config_dict.get('beam_center_mask', {}).get('plot_scale', 'lin')
        scale_cmb = ttk.Combobox(options_f, values=["lin", "log"], state="readonly")
        scale_cmb.set(scale_val)
        scale_cmb.pack(anchor="w", ipady=2)
        self.entries['beam_center_mask'][('plot_scale',)] = scale_cmb

        # Detector Geometry configuration subsection area binding
        self.build_config_area(s5, "Detector Geometry", "detector_geometry")
        tk.Button(f5, text="REFRESH FROM YAML", bg="#FF9800", fg="white", font=("Arial", 10, "bold"), pady=8, command=self.refresh_ui).pack(fill="x")

        # --- TAB 5: TRANSMISSION ---
        s6, f6 = self.create_scrollable_tab(self.notebook, "5. Transmission")
        tk.Button(f6, text="RUN TRANSMISSION CALCULATION", bg="#03A9F4", fg="white",
                  font=("Arial", 10, "bold"), pady=12,
                  command=lambda: self.run_script("caller_transmission.py")).pack(fill="x")
        self.build_config_area(s6, "Transmission Physics", "transmission_setup")
        self._build_dict_editor(s6, "Sample Thickness (cm)", "calibration_samples", "thickness")

        # --- TAB 6: RADIAL INTEGRATION ---
        t7_main = ttk.Frame(self.notebook); self.notebook.add(t7_main, text="6. Radial Integration")
        f7 = tk.Frame(t7_main, bg=bg); f7.pack(side="bottom", fill="x", padx=20, pady=15)
        tk.Button(f7, text="RUN FULL INTEGRATION PIPELINE", bg="#2196F3", fg="white", font=("Arial", 10, "bold"),
                  pady=10, command=lambda: self.run_script("caller_radial_integration.py")).pack(fill="x")

        self.sub_nb = ttk.Notebook(t7_main); self.sub_nb.pack(expand=True, fill="both", padx=10, pady=5)

        sc, _ = self.create_scrollable_tab(self.sub_nb, "Calibration Samples")
        if 'calibration_samples' not in self.entries: self.entries['calibration_samples'] = {}


        cal_data = self.config_dict.get('calibration_samples', {})

        if "_desc" in cal_data:
             tk.Label(sc, text=str(cal_data["_desc"]).strip(), font=("Arial", 12, "italic"),
                      fg="#444", bg="#eef2f7", justify="left", anchor="w",
                      padx=10, pady=8, wraplength=800).pack(fill="x", pady=(0, 10))

        def get_cal_comment(key):
            try:
                ca_items = getattr(getattr(cal_data, 'ca', None), 'items', None)
                if ca_items and key in ca_items:
                    for token in ca_items[key]:
                        if token and hasattr(token, 'value'):
                            val = token.value.strip().lstrip('#').strip()
                            if val: return f"  # {val}"
            except Exception: pass
            return ""

        cal_f = tk.LabelFrame(sc, text="Primary Standards", padx=10, pady=10, bg=bg, fg="#e67e22", font=("Arial", 10, "bold"))
        cal_f.pack(fill="x", padx=10, pady=5)

        for field in ['dark_current', 'water', 'water_cell']:
            row_f = tk.Frame(cal_f, bg=bg)
            row_f.pack(fill="x", pady=2, anchor="w")

            lbl_f = tk.Frame(row_f, bg=bg)
            lbl_f.pack(fill="x", anchor="w")

            tk.Label(lbl_f, text=field, font=("Arial", 9, "bold"), bg=bg).pack(side="left")

            cmt = get_cal_comment(field)
            if cmt:
                tk.Label(lbl_f, text=cmt, font=("Arial", 8, "italic"), fg="#888", bg=bg).pack(side="left")

            e = tk.Entry(row_f)
            e.insert(0, str(cal_data.get(field, "")))
            e.pack(fill="x", pady=(2, 5))
            self.entries['calibration_samples'][(field,)] = e

        tk.Button(sc, text="🔍 CHECK EXISTENCE OF ALL CALIBRATION FILES", bg="#2196F3", fg="white",
                  font=("Arial", 9, "bold"), pady=8, command=lambda: self.run_script("caller_check_calibration.py")).pack(fill="x", pady=10)

        self._build_dict_editor(sc, "Empty Cell Mapping", "calibration_samples", "empty_cell")
        self._build_dict_editor(sc, "Sample Thickness (cm)", "calibration_samples", "thickness")



        s_combined, _ = self.create_scrollable_tab(self.sub_nb, "Physics Settings")
        self.build_config_area(s_combined, "Physics Corrections", "physics_corrections")
        #self.build_config_area(s_combined, "Pipeline Control", "pipeline_control")

        sd, _ = self.create_scrollable_tab(self.sub_nb, "Analysis Flags")
        self.build_config_area(sd, "Flags", "analysis_flags")

        sr, _ = self.create_scrollable_tab(self.sub_nb, "Resolution Settings")
        self.build_config_area(sr, "Resolution Geometry (dq)", "resolution_settings")


        # --- TAB 7: MERGING CURVES ---
        s8, f8 = self.create_scrollable_tab(self.notebook, "7. Merging Curves")
        tk.Button(f8, text="SAVE SETTINGS", bg="#4CAF50", fg="white", font=("Arial", 10, "bold"),
                  command=self.save_data).pack(side="left", fill="x", expand=True, padx=2)
        tk.Button(f8, text="RUN DATA MERGING", bg="#2196F3", fg="white", font=("Arial", 10, "bold"),
                  command=lambda: self.run_script("caller_merging.py")).pack(side="left", fill="x", expand=True, padx=2)

        top_f = tk.LabelFrame(s8, text="Merging Pipeline Controls", padx=10, pady=10, bg=bg)
        top_f.pack(fill="x", padx=10, pady=5)

        m_data = self.config_dict.get('merging_settings', {})
        if 'merging_settings' not in self.entries: self.entries['merging_settings'] = {}

        def get_comment(key):
            try:
                ca_items = getattr(getattr(m_data, 'ca', None), 'items', None)
                if ca_items and key in ca_items:
                    for token in ca_items[key]:
                        if token and hasattr(token, 'value'):
                            return token.value.strip().lstrip('#').strip()
            except Exception: pass
            return ""

        if "_desc" in m_data:
             tk.Label(top_f, text=str(m_data["_desc"]).strip(), font=("Arial", 12, "italic"),
                      fg="#444", bg="#eef2f7", justify="left", anchor="w",
                      padx=10, pady=8, wraplength=800).pack(fill="x", pady=(0, 10))

        # Helper function to generate a consistent checkbox row for a step
        def create_step_checkbox(parent_frame, step_key):
            val = m_data.get(step_key, False)
            var = tk.BooleanVar(master=self.root, value=val)

            row_f = tk.Frame(parent_frame, bg=bg)
            row_f.pack(fill="x", anchor="w", pady=(8, 4))

            tk.Checkbutton(row_f, text=step_key, variable=var, bg=bg, font=("Arial", 9, "bold")).pack(side="left")

            cmt = get_comment(step_key)
            if cmt:
                tk.Label(row_f, text=f"   # {cmt}", font=("Arial", 8, "italic"), fg="#888", bg=bg).pack(side="left")

            self.entries['merging_settings'][(step_key,)] = var

        # ==========================================
        # STAGE 1: PLOTTING & CLIPPING
        # ==========================================
        create_step_checkbox(top_f, 'run_step_1_plotting')

        # Build the data clipping skip table right below Step 1
        self._build_merging_table(top_f, "Data Clipping (Skip Points)", "merging_settings")

        # ==========================================
        # STAGE 2: MERGING
        # ==========================================
        create_step_checkbox(top_f, 'run_step_2_merging')

        # ==========================================
        # STAGE 3: INTERPOLATION & RESAMPLING
        # ==========================================
        create_step_checkbox(top_f, 'run_step_3_interpolation')

        # Subsection frame for interpolation options
        interp_options_f = tk.Frame(top_f, bg=bg, padx=15)
        interp_options_f.pack(fill="x", pady=(2, 5))

        for param_key in ['interp_type', 'interp_points']:
            if param_key in m_data:
                cmt = get_comment(param_key)
                self._create_field(
                    parent=interp_options_f,
                    label=param_key,
                    value=m_data[param_key],
                    config_key='merging_settings',
                    path=(param_key,),
                    comment=cmt
                )

        # ==========================================
        # STAGE 4: INCOHERENT BACKGROUND FIT
        # ==========================================
        create_step_checkbox(top_f, 'run_step_4_incoherent')

        # Subsection frame for background parameters
        incoherent_options_f = tk.Frame(top_f, bg=bg, padx=15)
        incoherent_options_f.pack(fill="x", pady=(2, 5))

        if 'last_points_to_fit' in m_data:
            cmt = get_comment('last_points_to_fit')
            self._create_field(
                parent=incoherent_options_f,
                label='last_points_to_fit',
                value=m_data['last_points_to_fit'],
                config_key='merging_settings',
                path=('last_points_to_fit',),
                comment=cmt
            )


    def _build_merging_table(self, parent, title, m_key):
        bg = self.root.cget('bg')
        if m_key not in self.entries: self.entries[m_key] = {}

        group = tk.LabelFrame(parent, text=title, padx=10, pady=10, bg=bg, fg="#e67e22", font=("Arial", 10, "bold"))
        group.pack(fill="x", padx=10, pady=5)

        m_data = self.config_dict.get(m_key, {})

        if isinstance(m_data, dict) and "_desc" in m_data:
             tk.Label(group, text=str(m_data["_desc"]).strip(), font=("Arial", 12, "italic"),
                      fg="#444", bg="#eef2f7", justify="left", anchor="w",
                      padx=10, pady=8, wraplength=800).pack(fill="x", pady=(0, 10))

        # --- FIX: ADD DISTANCE CONTROLS ADDED FOR THE MERGING TAB ---
        add_r = tk.Frame(group, bg=bg)
        add_r.pack(fill="x", pady=(0, 10))
        tk.Label(add_r, text="New Dist:", font=("Arial", 8, "italic"), bg=bg).pack(side="left")
        self.merge_dist_entry = tk.Entry(add_r, width=8)
        self.merge_dist_entry.pack(side="left", padx=5)
        tk.Button(add_r, text=" + ", bg="#4CAF50", fg="white", font=("Arial", 8, "bold"),
                  command=lambda: self.add_distance_global(self.merge_dist_entry)).pack(side="left")

        c_start, c_end = "", ""
        try:
            ca_items = getattr(getattr(m_data, 'ca', None), 'items', None)
            if ca_items:
                def get_comment(key):
                    for token in ca_items.get(key, []):
                        if token and hasattr(token, 'value'):
                            val = token.value.strip().lstrip('#').strip()
                            if val: return val
                    return ""

                c_start = get_comment('skip_start')
                c_end = get_comment('skip_end')
        except Exception:
            pass

        if c_start or c_end:
            info_f = tk.Frame(group, bg=bg); info_f.pack(fill="x", pady=(0, 5))
            if c_start: tk.Label(info_f, text=f"  # Skip Start: {c_start}", font=("Arial", 8, "italic"), fg="#777", bg=bg).pack(anchor="w")
            if c_end: tk.Label(info_f, text=f"  # Skip End: {c_end}", font=("Arial", 8, "italic"), fg="#777", bg=bg).pack(anchor="w")

        h_f = tk.Frame(group, bg=bg); h_f.pack(fill="x", pady=(0,5))
        tk.Label(h_f, text="Distance (m)", width=12, font=("Arial", 8, "bold"), bg=bg, anchor="w").pack(side="left")
        tk.Label(h_f, text="Skip Start", width=15, font=("Arial", 8, "bold"), bg=bg).pack(side="left")
        tk.Label(h_f, text="Skip End", width=15, font=("Arial", 8, "bold"), bg=bg).pack(side="left")

        starts = m_data.get('skip_start', {})
        ends = m_data.get('skip_end', {})

        for d in sorted(set(list(starts.keys()) + list(ends.keys())), key=float):
            r = tk.Frame(group, bg=bg); r.pack(fill="x", pady=2)
            tk.Label(r, text=f"{d} m", width=12, bg=bg, anchor="w").pack(side="left")
            s_ent = tk.Entry(r, width=15); s_ent.insert(0, str(starts.get(d, 0))); s_ent.pack(side="left", padx=2)
            self.entries[m_key][('skip_start', d)] = s_ent
            e_ent = tk.Entry(r, width=15); e_ent.insert(0, str(ends.get(d, 0))); e_ent.pack(side="left", padx=2)
            self.entries[m_key][('skip_end', d)] = e_ent

            # --- FIX: ADDED REMOVAL TRASH BUTTON FOR EACH ROW ---
            tk.Button(r, text=" 🗑️ ", bg="#f44336", fg="white", font=("Arial", 8, "bold"),
                      command=lambda dist=d: self.remove_distance(dist)).pack(side="left", padx=5)

    def _build_dict_editor(self, parent, title, m_key, s_key):
        bg = self.root.cget('bg')
        if m_key not in self.entries: self.entries[m_key] = {}
        g = tk.LabelFrame(parent, text=title, padx=10, pady=10, bg=bg, fg="#e67e22", font=("Arial", 10, "bold"))
        g.pack(fill="x", padx=10, pady=5)

        d_dict = self.config_dict.get(m_key, {}).get(s_key, {})

        if isinstance(d_dict, dict) and "_desc" in d_dict:
            tk.Label(g, text=str(d_dict["_desc"]).strip(), font=("Arial", 12, "italic"),
                     fg="#444", bg="#eef2f7", justify="left", anchor="w",
                     padx=10, pady=8, wraplength=800).pack(fill="x", pady=(0, 10))

        af = tk.Frame(g, bg=bg); af.pack(fill="x", pady=(0,10))
        nn = tk.Entry(af, width=25); nn.insert(0, "New Sample...")
        nn.pack(side="left", padx=2)
        nv = tk.Entry(af, width=15); nv.insert(0, "0.1")
        nv.pack(side="left", padx=2)
        tk.Button(af, text=" + ", bg="#4CAF50", fg="white", font=("Arial", 9, "bold"),
                  command=lambda: self.add_dict_item(m_key, s_key, nn, nv)).pack(side="left", padx=5)

        for k, v in d_dict.items():
            if str(k).startswith("_"): continue
            r = tk.Frame(g, bg=bg); r.pack(fill="x", pady=2)
            tk.Label(r, text=k, width=30, anchor="w", bg=bg).pack(side="left")
            e = tk.Entry(r, width=20); e.insert(0, str(v)); e.pack(side="left")
            self.entries[m_key][(s_key, k)] = e
            tk.Button(r, text=" 🗑️ ", bg="#f44336", fg="white", font=("Arial", 7),
                      command=lambda mk=m_key, sk=s_key, item=k: self.remove_dict_item(mk, sk, item)).pack(side="right")

    # --- ACTION METHODS ---
    def run_mask_trans_for_dist(self, dist):
        """Saves current state and executes the Mask and Transmission setup script."""
        if self.save_data():
            scan = self.entries['beam_center_mask'][('scans', dist)].get()
            self.config_dict['beam_center_mask']['scan_nr'] = int(scan)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_dict, f)
            self.run_script("caller_mask_beamstop.py")
            self.refresh_ui()

    def run_beam_center_for_dist(self, dist):
        """Saves current state and executes the Beam Center refinement script."""
        if self.save_data():
            scan = self.entries['beam_center_mask'][('scans', dist)].get()
            self.config_dict['beam_center_mask']['scan_nr'] = int(scan)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_dict, f)
            self.run_script("caller_beam_center.py")
            self.refresh_ui()

    # --- FIX: FULLY SYNCHRONIZED DISTANCE REMOVAL PIPELINE ---
    def remove_distance(self, dist):
        if messagebox.askyesno("Confirm", f"Remove {dist}m from all settings?"):
            # 1. Purge from Memory Buffers
            if 'beam_center_mask' in self.entries:
                self.entries['beam_center_mask'].pop(('scans', dist), None)
            if 'merging_settings' in self.entries:
                self.entries['merging_settings'].pop(('skip_start', dist), None)
                self.entries['merging_settings'].pop(('skip_end', dist), None)

            bm = self.config_dict.get('beam_center_mask', {})
            dg = self.config_dict.get('detector_geometry', {})
            ms = self.config_dict.get('merging_settings', {})

            # Setup variations of data type matching what ruamel might parse
            keys_to_purge = [dist, str(dist)]
            try: keys_to_purge.append(float(dist))
            except: pass
            try: keys_to_purge.append(int(dist))
            except: pass

            # Unique clean types tracking
            seen = set()
            keys_to_purge = [x for x in keys_to_purge if not (x in seen or seen.add(x))]

            # 2. Symmetrical cascade deletion from ALL blocks
            for k in keys_to_purge:
                if 'scans' in bm: bm['scans'].pop(k, None)
                if 'beam_center_guess' in dg: dg['beam_center_guess'].pop(k, None)
                if 'beamstopper_coordinates' in dg: dg['beamstopper_coordinates'].pop(k, None)
                if 'transmission_coordinates' in dg: dg['transmission_coordinates'].pop(k, None)
                if 'skip_start' in ms: ms['skip_start'].pop(k, None)
                if 'skip_end' in ms: ms['skip_end'].pop(k, None)

            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_dict, f)

            self.log_to_console(f"🗑️ Purged distance {dist}m from all records.")
            self.refresh_ui()

    # --- FIX: FULLY SYNCHRONIZED GLOBAL DISTANCE ADDITION PIPELINE ---
    def add_distance_global(self, entry_widget):
        v = entry_widget.get().strip()
        if v:
            try:
                d = float(v)
                if d.is_integer(): d = int(d)

                bm = self.config_dict.setdefault('beam_center_mask', {})
                dg = self.config_dict.setdefault('detector_geometry', {})
                ms = self.config_dict.setdefault('merging_settings', {})

                # Symmetrical initialization parameters for safety
                scans = bm.setdefault('scans', {})
                if d not in scans:
                    scans[d] = 0

                dg.setdefault('beam_center_guess', {}).setdefault(d, [0.0, 0.0])
                dg.setdefault('beamstopper_coordinates', {}).setdefault(d, [0.0, 0.0, 0.0, 0.0])
                dg.setdefault('transmission_coordinates', {}).setdefault(d, [0.0, 0.0, 0.0, 0.0])

                ms.setdefault('skip_start', {}).setdefault(d, 0)
                ms.setdefault('skip_end', {}).setdefault(d, 0)

                self.save_data()
                self.refresh_ui()
            except Exception as e:
                self.log_to_console(f"⚠️ Add distance error: {e}")

    def add_new_distance(self):
        self.add_distance_global(self.new_dist_entry)

    def add_dict_item(self, m_key, s_key, n_ent, v_ent):
        n, v = n_ent.get().strip(), v_ent.get().strip()
        if n and v and "New Sample" not in n:
            d = self.config_dict.setdefault(m_key, {}).setdefault(s_key, {})
            d[n] = self._parse_string(v)
            self.save_data(); self.refresh_ui()

    def remove_dict_item(self, m_key, s_key, item_name):
        if messagebox.askyesno("Confirm", f"Remove '{item_name}' from {s_key}?"):
            if m_key in self.entries:
                self.entries[m_key].pop((s_key, item_name), None)
            if s_key in self.config_dict.get(m_key, {}):
                self.config_dict[m_key][s_key].pop(item_name, None)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_dict, f)
            self.log_to_console(f"🗑️ REMOVED from {s_key}: {item_name}")
            self.refresh_ui()

    def run_script(self, name):
        """Runs scripts safely using Spyder's native workspace context runner or an observed background subprocess."""
        if self.save_data():
            script_path = os.path.normpath(os.path.join(script_dir, "codes", name))
            if not os.path.exists(script_path):
                script_path = os.path.normpath(os.path.join(script_dir, name))

            if os.path.exists(script_path):
                self.log_to_console(f"🚀 RUNNING: {name}")
                try:
                    from IPython import get_ipython
                    ipy = get_ipython()
                    if ipy:
                        # Runs inside Spyder's console context cleanly (Blocking execution)
                        ipy.run_line_magic('run', f'\"{script_path}\" \"{self.config_file}\"')
                        self.refresh_ui()  # Refreshes right after execution finishes
                        return
                except:
                    pass

                # Asynchronous subprocess context execution loop
                import subprocess
                experiment_dir = os.path.dirname(os.path.abspath(self.config_file))
                
                # Capture the process object instead of discarding it
                process = subprocess.Popen(
                    [sys.executable, script_path, self.config_file],
                    cwd=experiment_dir,
                    env=os.environ.copy()
                )
                
                # Start non-blocking polling sequence to see when the window closes
                self.check_process_status(process, name)
            else:
                messagebox.showerror("Error", f"Target execution path module not found:\n{script_path}")
                self.refresh_ui()

    def check_process_status(self, process, script_name):
        """Periodically checks if the background process has closed, then refreshes the interface."""
        # poll() returns None if process is running, or return code if completed
        if process.poll() is None:
            # Not finished yet; re-check again in 500ms (keeps main UI responsive)
            self.root.after(500, lambda: self.check_process_status(process, script_name))
        else:
            # Process finished! Trigger layout reload engine
            self.log_to_console(f"🏁 FINISHED: {script_name}. Reloading parameters...")
            self.refresh_ui()

if __name__ == "__main__":
    root = tk.Tk()
    app = DarePyGUI(root)

    # 1. Define a clear exit routine
    def safe_exit():
        root.quit()     # This explicitly breaks root.mainloop() so the script stops
        root.destroy()  # This safely clears the window components out of memory

    # 2. Bind the window closing event to our clean routine
    root.protocol("WM_DELETE_WINDOW", safe_exit)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass