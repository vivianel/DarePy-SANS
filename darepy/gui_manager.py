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

global CONFIG_FILE
CONFIG_FILE = None

if len(sys.argv) > 1:
    CONFIG_FILE = sys.argv[1]

yaml = YAML()
yaml.preserve_quotes = True

class DarePyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DarePy-SANS Control Panel")
        self.root.geometry("1000x850") # Slightly shorter for better fit

        global CONFIG_FILE
        if not CONFIG_FILE:
            self.root.withdraw()
            initial_dir = os.path.join(root_dir, "experiments")
            CONFIG_FILE = filedialog.askopenfilename(
                initialdir=initial_dir,
                title="Select config_experiment.yaml",
                filetypes=(("YAML files", "*.yaml"), ("All files", "*.*"))
            )
            if not CONFIG_FILE:
                sys.exit()
            self.root.deiconify()

        self.setup_styles()
        self.entries = {}

        # Main Notebook
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
                  foreground=[("selected", "white"), ("active", "black")],
                  lightcolor=[("selected", "#2196F3")],
                  borderwidth=[("selected", 0)])

    def setup_console(self):
        c_frame = tk.LabelFrame(self.root, text="Console Output", padx=5, pady=5)
        c_frame.pack(fill="x", side="bottom", padx=10, pady=5)
        self.console = tk.Text(c_frame, height=8, bg="#1e1e1e", fg="#4CAF50",
                               font=("Consolas", 9), wrap="word")
        self.console.pack(side="left", fill="x", expand=True)
        scrollbar = ttk.Scrollbar(c_frame, command=self.console.yview)
        scrollbar.pack(side="right", fill="y")
        self.console.config(yscrollcommand=scrollbar.set)
        self.log_to_console("System Ready.")

    def load_data(self):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                self.config_dict = yaml.load(f)
            return True
        except Exception as e:
            self.log_to_console(f"[YAML LOAD ERROR] {e}")
            return False

    def save_data(self):
        try:
            for s_key, keys in self.entries.items():
                for path, entry in keys.items():
                    raw = entry.get()
                    actual = raw if isinstance(raw, bool) else self._parse_string(raw)
                    target = self.config_dict[s_key]
                    for p in path[:-1]:
                        if p not in target: target[p] = {}
                        target = target[p]
                    target[path[-1]] = actual

            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_dict, f)

            # Archiving logic
            p_path = self.config_dict.get('analysis_paths', {}).get('project_base', 'Unknown')
            project_name = os.path.basename(os.path.normpath(p_path))
            history_dir = os.path.join(os.path.dirname(__file__), "config_history")
            if not os.path.exists(history_dir): os.makedirs(history_dir)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            archive_path = os.path.join(history_dir, f"config_experiment_{project_name}_{timestamp}.yaml")
            shutil.copy2(CONFIG_FILE, archive_path)

            self.log_to_console(f"✅ Saved & Archived: {project_name}_{timestamp}")
            return True
        except Exception as e:
            self.log_to_console(f"❌ SAVE ERROR: {e}")
            return False

    def _parse_string(self, raw_val):
        """Standardizes input and handles comma-separated lists automatically."""
        v = str(raw_val).strip()
        if v == "": return ""
        if v.lower() == 'true': return True
        if v.lower() == 'false': return False

        # NEW: Handle comma-separated numbers without needing brackets
        if "," in v and not v.startswith('['):
            try:
                return [self._parse_string(x) for x in v.split(",")]
            except:
                return v

        if v.startswith(('[', '{')):
            try: return ast.literal_eval(v)
            except: return v

        try: return float(v) if '.' in v else int(v)
        except: return v

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

    def build_config_area(self, parent, title, config_key):
        """Entry point to build a section. Now pulls help text from the YAML."""
        bg = parent.cget('bg')
        group = tk.LabelFrame(parent, text=title, padx=10, pady=10, bg=bg)
        # Snap the frame to the left
        group.pack(fill="x", padx=10, pady=5, anchor="w")

        if config_key not in self.entries:
            self.entries[config_key] = {}

        # Call the recursive builder to handle nested dicts and descriptions
        self._build_nested_ui(group, self.config_dict.get(config_key, {}), config_key, path=())

    def _build_nested_ui(self, parent, data, config_key, path):
        """Recursively builds UI and extracts inline comments from ruamel.yaml."""
        if not hasattr(data, "items"):
            return

        bg = parent.cget('bg')

        # 1. RENDER BLOCK DESCRIPTION (_desc)
        if "_desc" in data:
            desc_text = str(data["_desc"]).strip()
            desc_box = tk.Message(parent, text=desc_text, font=("Arial", 9, "italic"),
                                  fg="#555", bg="#f8f9fa", width=800,
                                  justify="left", padx=10, pady=5)
            desc_box.pack(fill="x", pady=(0, 10), anchor="w")

        # 2. RENDER ACTUAL FIELDS
        for k, v in data.items():
            if str(k).startswith("_"): continue

            # --- EXTRACT INLINE COMMENT ---
            inline_comment = ""
            if hasattr(data, 'ca') and k in data.ca.items:
                # ruamel stores EOL comments at index 2 of the items list
                comment_token = data.ca.items[k][2]
                if comment_token:
                    inline_comment = comment_token.value.strip().lstrip('#').strip()

            current_path = path + (k,)
            if isinstance(v, (dict, type(data))):
                sub = tk.LabelFrame(parent, text=str(k), padx=10, pady=5,
                                    font=("Arial", 9, "bold"), fg="#e67e22", bg=bg)
                sub.pack(fill="x", padx=5, pady=5, anchor="w")
                self._build_nested_ui(sub, v, config_key, current_path)
            else:
                # Pass the extracted comment to the field creator
                self._create_field(parent, str(k), v, config_key, current_path, inline_comment)

    def _create_field(self, parent, label, value, config_key, path, comment=""):
        """Creates widgets with help text extracted from YAML comments."""
        bg = parent.cget('bg')
        f = tk.Frame(parent, bg=bg)
        f.pack(fill="x", pady=4, anchor="w")

        # 1. Label and Comment Row
        label_frame = tk.Frame(f, bg=bg)
        label_frame.pack(fill="x", anchor="w")

        tk.Label(label_frame, text=label, font=("Arial", 9, "bold"), bg=bg).pack(side="left")

        if comment:
            # Display the YAML comment in a smaller, gray font next to the label
            tk.Label(label_frame, text=f"  # {comment}", font=("Arial", 8, "italic"),
                     fg="#888", bg=bg).pack(side="left")

        # 2. Input Widget Row
        if isinstance(value, bool):
            var = tk.BooleanVar(master=self.root, value=value)
            cb = tk.Checkbutton(f, text="", variable=var, bg=bg)
            cb.pack(anchor="w")
            self.entries[config_key][path] = var
        elif label == "which_instrument":
            w = ttk.Combobox(f, values=["SANS-I", "SANS-LLB"], state="readonly")
            w.set(value)
            w.pack(anchor="w", ipady=2)
            self.entries[config_key][path] = w
        else:
            w = tk.Entry(f)
            disp = ", ".join(map(str, value)) if isinstance(value, list) else str(value)
            w.insert(0, disp)
            w.pack(fill="x", padx=(0, 40), ipady=3)
            self.entries[config_key][path] = w

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



    def fill_notebook(self):
        bg = self.root.cget('bg')

        # --- TAB 1: GLOBAL SETUP ---
        s1, f1 = self.create_scrollable_tab(self.notebook, "1. Global Setup")

        # Action Buttons
        tk.Button(f1, text="SAVE SETTINGS", bg="#4CAF50", fg="white", font=("Arial", 10, "bold"),
                  command=self.save_data).pack(side="left", fill="x", expand=True, padx=2)
        tk.Button(f1, text="LIST ALL MEASUREMENTS (Metadata)", bg="#2196F3", fg="white", font=("Arial", 10, "bold"),
                  command=lambda: self.run_script("caller_listing.py")).pack(side="left", fill="x", expand=True, padx=2)

        # This builds everything in analysis_paths, including your new dry_run checkbox
        self.build_config_area(s1, "Analysis Paths", "analysis_paths")

        # Build the instrument section (now includes the dropdown)
        self.build_config_area(s1, "Instrument", "instrument_setup")


        # --- TAB 2 & 3: RENAME & VISUALS (Standard) ---
        s2, f2 = self.create_scrollable_tab(self.notebook, "2. Rename Samples")
        tk.Button(f2, text="RENAME SAMPLES", bg="#2196F3", fg="white", font=("Arial", 10, "bold"),
                  command=lambda: self.run_script("rename_samples.py")).pack(fill="x")
        self.build_config_area(s2, "Rename Settings", "rename_samples")

        s3, f3 = self.create_scrollable_tab(self.notebook, "3. 2D Visualization")
        tk.Button(f3, text="GENERATE 2D PLOTS/GIF", bg="#2196F3", fg="white", font=("Arial", 10, "bold"),
                  command=lambda: self.run_script("plot_2Dpattern.py")).pack(fill="x")
        self.build_config_area(s3, "Plot 2D Settings", "plot_2d")

        # --- TAB 4: MASK & CENTER ---
        s4, f4 = self.create_scrollable_tab(self.notebook, "4. Mask & Center")
        setup_f = tk.LabelFrame(s4, text="Masking Setup", padx=10, pady=10, bg=bg)
        setup_f.pack(fill="x", padx=10, pady=5)
        tk.Label(setup_f, text="sample_name", font=("Arial", 9, "bold"), bg=bg).pack(anchor="w")
        s_ent = tk.Entry(setup_f)
        s_ent.insert(0, self.config_dict.get('beam_center_mask', {}).get('sample_name', 'AgBE'))
        s_ent.pack(fill="x", ipady=3)
        if 'beam_center_mask' not in self.entries: self.entries['beam_center_mask'] = {}
        self.entries['beam_center_mask'][('sample_name',)] = s_ent

        act_f = tk.LabelFrame(s4, text="Distance-Specific Masking", padx=10, pady=10, bg=bg, fg="#2196F3", font=("Arial", 10, "bold"))
        act_f.pack(fill="x", padx=10, pady=5)
        add_r = tk.Frame(act_f, bg=bg); add_r.pack(fill="x", pady=(0,10))
        tk.Label(add_r, text="New Dist:", font=("Arial", 8, "italic"), bg=bg).pack(side="left")
        self.new_dist_entry = tk.Entry(add_r, width=8); self.new_dist_entry.pack(side="left", padx=5)
        tk.Button(add_r, text=" + ", bg="#4CAF50", fg="white", font=("Arial", 8, "bold"), command=self.add_new_distance).pack(side="left")

        scans_dict = self.config_dict.get('beam_center_mask', {}).get('scans', {})
        if isinstance(scans_dict, dict):
            for d in sorted(scans_dict.keys(), key=float):
                r = tk.Frame(act_f, bg=bg); r.pack(fill="x", pady=2)
                tk.Label(r, text=f"{d} m:", width=8, font=("Arial", 9, "bold"), bg=bg, anchor="w").pack(side="left")
                e = tk.Entry(r, width=12); e.insert(0, str(scans_dict[d])); e.pack(side="left", padx=5)
                self.entries['beam_center_mask'][('scans', d)] = e
                tk.Button(r, text="RUN MASKING TOOL", bg="#2196F3", fg="white", font=("Arial", 8, "bold"),
                          command=lambda dist=d: self.run_mask_for_dist(dist)).pack(side="left", fill="x", expand=True, padx=5)
                tk.Button(r, text=" 🗑️ ", bg="#f44336", fg="white", font=("Arial", 8, "bold"),
                          command=lambda dist=d: self.remove_distance(dist)).pack(side="right")

        self.build_config_area(s4, "Detector Geometry", "detector_geometry")
        tk.Button(f4, text="REFRESH FROM YAML", bg="#FF9800", fg="white", font=("Arial", 10, "bold"), pady=8, command=self.refresh_ui).pack(fill="x")

        # --- TAB 5: TRANSMISSION (Step 3) ---
        s5, f5 = self.create_scrollable_tab(self.notebook, "5. Transmission")

        # Action: Run Step 3
        tk.Button(f5, text="RUN TRANSMISSION CALCULATION", bg="#03A9F4", fg="white",
                  font=("Arial", 10, "bold"), pady=12,
                  command=lambda: self.run_script("step3_transmission.py")).pack(fill="x")

        # 1. Display the transmission distance setup
        self.build_config_area(s5, "Transmission Physics", "transmission_setup")

        # 2. ADDED: Display Thickness Editor here too for convenience
        self._build_dict_editor(s5, "Sample Thickness (cm)", "calibration_samples", "thickness")

        # --- TAB 6: RADIAL INTEGRATION (Step 2 & 4) ---
        t6_main = ttk.Frame(self.notebook); self.notebook.add(t6_main, text="6. Radial Integration")

        # Main Action Footer
        f6 = tk.Frame(t6_main, bg=bg); f6.pack(side="bottom", fill="x", padx=20, pady=15)
        tk.Button(f6, text="RUN FULL INTEGRATION PIPELINE", bg="#2196F3", fg="white", font=("Arial", 10, "bold"),
                  pady=10, command=lambda: self.run_script("caller_radial_integration.py")).pack(fill="x")

        # Create the Sub-Notebook
        self.sub_nb = ttk.Notebook(t6_main); self.sub_nb.pack(expand=True, fill="both", padx=10, pady=5)

        # 1. Sub-tab: Calibration Samples (NOW FIRST)
        sc, _ = self.create_scrollable_tab(self.sub_nb, "Calibration Samples")
        if 'calibration_samples' not in self.entries: self.entries['calibration_samples'] = {}

        # ACTION: Step 2 Calibration Check
        tk.Button(sc, text="🔍 CHECK EXISTENCE OF ALL CALIBRATION FILES", bg="#2196F3", fg="white",
                  font=("Arial", 9, "bold"), pady=8, command=lambda: self.run_script("caller_check_calibration.py")).pack(fill="x", pady=10)

        # Calibration UI Components
        cal_f = tk.LabelFrame(sc, text="Primary Standards", padx=10, pady=10, bg=bg); cal_f.pack(fill="x", padx=10, pady=5)
        for field in ['dark_current', 'water', 'water_cell']:
            tk.Label(cal_f, text=field, font=("Arial", 9, "bold"), bg=bg).pack(anchor="w")
            e = tk.Entry(cal_f); e.insert(0, self.config_dict.get('calibration_samples', {}).get(field, "")); e.pack(fill="x", pady=2)
            self.entries['calibration_samples'][(field,)] = e

        self._build_dict_editor(sc, "Empty Cell Mapping", "calibration_samples", "empty_cell")
        self._build_dict_editor(sc, "Sample Thickness (cm)", "calibration_samples", "thickness")

        # 2. Sub-tab: Pipeline & Physics (COMBINED)
        s_combined, _ = self.create_scrollable_tab(self.sub_nb, "Pipeline & Physics Settings")
        # We simply stack the two areas in one scrollable frame
        self.build_config_area(s_combined, "Pipeline Control", "pipeline_control")
        self.build_config_area(s_combined, "Physics Corrections", "physics_corrections")

        # 3. Sub-tab: Analysis Flags (ORIGINAL SUB-TAB 4)
        sd, _ = self.create_scrollable_tab(self.sub_nb, "Analysis Flags")
        self.build_config_area(sd, "Flags", "analysis_flags")

        # --- TAB 7: MERGING ---
        s6, f6 = self.create_scrollable_tab(self.notebook, "6. Merging Curves")
        tk.Button(f6, text="Run Data Merging", bg="#2196F3", fg="white", font=("Arial", 10, "bold"),
                  pady=8, command=lambda: self.run_script("caller_merging.py")).pack(fill="x")

        # 1. Basic Settings (Flags and interp_type)
        # We build these manually to keep the top clean
        top_f = tk.LabelFrame(s6, text="Merging Control", padx=10, pady=10, bg=bg)
        top_f.pack(fill="x", padx=10, pady=5)

        m_data = self.config_dict.get('merging_settings', {})
        if 'merging_settings' not in self.entries: self.entries['merging_settings'] = {}

        # Build checkboxes and interp_type manually
        for k in ['run_step_1_plotting', 'run_step_2_merging', 'run_step_3_interpolation', 'run_step_4_incoherent']:
            val = m_data.get(k, False)
            var = tk.BooleanVar(master=self.root, value=val)
            tk.Checkbutton(top_f, text=k, variable=var, bg=bg).pack(anchor="w")
            self.entries['merging_settings'][(k,)] = var

        tk.Label(top_f, text="interp_type", font=("Arial", 9, "bold"), bg=bg).pack(anchor="w", pady=(5,0))
        i_ent = tk.Entry(top_f); i_ent.insert(0, m_data.get('interp_type', 'log'))
        i_ent.pack(fill="x"); self.entries['merging_settings'][('interp_type',)] = i_ent

        # 2. THE SYNCED SKIP POINTS TABLE
        self._build_merging_table(s6, "Data Clipping (Skip Points)", "merging_settings")

    def _build_merging_table(self, parent, title, m_key):
        bg = self.root.cget('bg')
        group = tk.LabelFrame(parent, text=title, padx=10, pady=10, bg=bg, fg="#e67e22", font=("Arial", 10, "bold"))
        group.pack(fill="x", padx=10, pady=5)

        # Quick-Add Row
        add_f = tk.Frame(group, bg=bg); add_f.pack(fill="x", pady=(0, 10))
        tk.Label(add_f, text="Add Distance:", font=("Arial", 8, "italic"), bg=bg).pack(side="left")
        new_d = tk.Entry(add_f, width=10); new_d.pack(side="left", padx=5)
        tk.Button(add_f, text=" + ", bg="#4CAF50", fg="white",
                  command=lambda: self.add_merging_dist(m_key, new_d)).pack(side="left")

        # Headers
        h_f = tk.Frame(group, bg=bg); h_f.pack(fill="x")
        tk.Label(h_f, text="Distance (m)", width=12, font=("Arial", 8, "bold"), bg=bg).pack(side="left")
        tk.Label(h_f, text="Skip Start", width=15, font=("Arial", 8, "bold"), bg=bg).pack(side="left")
        tk.Label(h_f, text="Skip End", width=15, font=("Arial", 8, "bold"), bg=bg).pack(side="left")

        # Get unique list of distances from both dictionaries
        starts = self.config_dict[m_key].get('skip_start', {})
        ends = self.config_dict[m_key].get('skip_end', {})
        all_dists = sorted(set(list(starts.keys()) + list(ends.keys())), key=float)

        for d in all_dists:
            r = tk.Frame(group, bg=bg); r.pack(fill="x", pady=2)
            tk.Label(r, text=f"{d} m", width=12, anchor="w", bg=bg, font=("Arial", 9, "bold")).pack(side="left")

            # Start Value
            s_ent = tk.Entry(r, width=15); s_ent.insert(0, str(starts.get(d, 0))); s_ent.pack(side="left", padx=2)
            self.entries[m_key][('skip_start', d)] = s_ent

            # End Value
            e_ent = tk.Entry(r, width=15); e_ent.insert(0, str(ends.get(d, 0))); e_ent.pack(side="left", padx=2)
            self.entries[m_key][('skip_end', d)] = e_ent

            # Delete Button
            tk.Button(r, text=" 🗑️ ", bg="#f44336", fg="white", font=("Arial", 7),
                      command=lambda dist=d: self.remove_merging_dist(m_key, dist)).pack(side="right")
    def run_mask_for_dist(self, dist):
        if self.save_data():
            scan = self.entries['beam_center_mask'][('scans', dist)].get()
            self.config_dict['beam_center_mask']['scan_nr'] = int(scan)
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f: yaml.dump(self.config_dict, f)
            self.run_script("mask_beamstop_center.py")

    def remove_distance(self, dist):
        """Removes a distance and ensures the UI doesn't 'save it back' during refresh."""
        if messagebox.askyesno("Confirm", f"Remove {dist}m from all settings?"):
            # 1. CRITICAL: Remove the widgets from self.entries so save_data() ignores them
            if 'beam_center_mask' in self.entries:
                self.entries['beam_center_mask'].pop(('scans', dist), None)

            # 2. Purge from the actual configuration dictionary
            # We try string, float, and int versions of the key to be 100% sure
            bm = self.config_dict.get('beam_center_mask', {})
            dg = self.config_dict.get('detector_geometry', {})

            keys_to_purge = [dist, str(dist)]
            try: keys_to_purge.append(float(dist))
            except: pass

            for k in keys_to_purge:
                if 'scans' in bm: bm['scans'].pop(k, None)
                if 'beam_center_guess' in dg: dg['beam_center_guess'].pop(k, None)
                if 'beamstopper_coordinates' in dg: dg['beamstopper_coordinates'].pop(k, None)
                if 'transmission_coordinates' in dg: dg['transmission_coordinates'].pop(k, None)

            # 3. Save the cleaned dictionary to the YAML file
            # Manually dumping here ensures we save the deletion without the 'save_data' widget loop
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_dict, f)

            # 4. Completely rebuild the UI
            self.log_to_console(f"🗑️ Purged distance {dist}m from all records.")
            self.refresh_ui()

    def add_merging_dist(self, m_key, entry_widget):
        val = entry_widget.get().strip()
        if val:
            try:
                d = float(val)
                if d.is_integer(): d = int(d)
                # Initialize dictionaries if they don't exist
                self.config_dict[m_key].setdefault('skip_start', {})[d] = 0
                self.config_dict[m_key].setdefault('skip_end', {})[d] = 0
                self.save_data()
                self.refresh_ui()
            except: pass

    def remove_merging_dist(self, m_key, dist):
        if messagebox.askyesno("Confirm", f"Remove distance {dist}m from skip points?"):
            # Purge from Registry
            if m_key in self.entries:
                self.entries[m_key].pop(('skip_start', dist), None)
                self.entries[m_key].pop(('skip_end', dist), None)

            # Purge from Config
            self.config_dict[m_key].get('skip_start', {}).pop(dist, None)
            self.config_dict[m_key].get('skip_end', {}).pop(dist, None)

            # Save and Refresh
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_dict, f)
            self.refresh_ui()

    def add_new_distance(self):
        v = self.new_dist_entry.get().strip()
        if v:
            try:
                d = float(v)
                if d.is_integer(): d = int(d)
                scans = self.config_dict['beam_center_mask'].setdefault('scans', {})
                if d not in scans:
                    scans[d] = 0
                    self.save_data(); self.refresh_ui()
            except: pass

    def _build_dict_editor(self, parent, title, m_key, s_key):
        """Builds a two-column editor with a Quick-Add row and a Delete button."""
        bg = self.root.cget('bg')
        if m_key not in self.entries: self.entries[m_key] = {}

        g = tk.LabelFrame(parent, text=title, padx=10, pady=10, bg=bg, fg="#e67e22", font=("Arial", 10, "bold"))
        g.pack(fill="x", padx=10, pady=5)

        # Quick-Add Row
        af = tk.Frame(g, bg=bg); af.pack(fill="x", pady=(0,10))
        nn = tk.Entry(af, width=25); nn.insert(0, "New Sample..."); nn.pack(side="left", padx=2)
        nv = tk.Entry(af, width=15); nv.insert(0, "0.1"); nv.pack(side="left", padx=2)
        tk.Button(af, text=" + ", bg="#4CAF50", fg="white", font=("Arial", 9, "bold"),
                  command=lambda: self.add_dict_item(m_key, s_key, nn, nv)).pack(side="left", padx=5)

        # Data Rows
        d_dict = self.config_dict.get(m_key, {}).get(s_key, {})
        for k, v in d_dict.items():
            r = tk.Frame(g, bg=bg); r.pack(fill="x", pady=2)

            # Key Label
            tk.Label(r, text=k, width=30, anchor="w", bg=bg).pack(side="left")

            # Value Entry
            e = tk.Entry(r, width=20); e.insert(0, str(v)); e.pack(side="left")
            self.entries[m_key][(s_key, k)] = e

            # REMOVE BUTTON (Red 'X')
            tk.Button(r, text=" 🗑️ ", bg="#f44336", fg="white", font=("Arial", 7),
                      command=lambda mk=m_key, sk=s_key, item=k: self.remove_dict_item(mk, sk, item)).pack(side="right")

    def add_dict_item(self, m_key, s_key, n_ent, v_ent):
        n, v = n_ent.get().strip(), v_ent.get().strip()
        if n and v and "New Sample" not in n:
            d = self.config_dict.setdefault(m_key, {}).setdefault(s_key, {})
            d[n] = self._parse_string(v)
            self.save_data(); self.refresh_ui()

    def log_to_console(self, msg):
        self.console.insert(tk.END, f"{msg}\n"); self.console.see(tk.END); self.root.update()

    def run_script(self, name):
        """Runs the script in the active Spyder console."""
        if self.save_data():
            # Check darepy/codes/ then darepy/
            script_path = os.path.normpath(os.path.join(script_dir, "codes", name))
            if not os.path.exists(script_path):
                script_path = os.path.normpath(os.path.join(script_dir, name))

            if os.path.exists(script_path):
                self.log_to_console(f"🚀 RUNNING: {name}")
                try:
                    # The magic connection to Spyder's console
                    from IPython import get_ipython
                    ipy = get_ipython()
                    if ipy:
                        # We use %run to pass the CONFIG_FILE as an argument
                        ipy.run_line_magic('run', f'"{script_path}" "{CONFIG_FILE}"')
                        return
                except:
                    pass

                # Fallback for standard terminal
                import subprocess
                subprocess.Popen([sys.executable, script_path, CONFIG_FILE])
            else:
                self.log_to_console(f"❌ ERROR: {name} not found.")


    def remove_dict_item(self, m_key, s_key, item_name):
        """Generic remover for dictionary items (Empty Cell or Thickness)."""
        if messagebox.askyesno("Confirm", f"Remove '{item_name}' from {s_key}?"):
            # 1. Purge from Registry to prevent 'save_data' from resurrecting it
            if m_key in self.entries:
                self.entries[m_key].pop((s_key, item_name), None)

            # 2. Purge from config dictionary
            if s_key in self.config_dict.get(m_key, {}):
                self.config_dict[m_key][s_key].pop(item_name, None)

            # 3. Direct Save to disk (bypassing the widget loop)
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_dict, f)

            # 4. Refresh to update the UI
            self.log_to_console(f"🗑️ REMOVED from {s_key}: {item_name}")
            self.refresh_ui()

if __name__ == "__main__":
    root = tk.Tk(); app = DarePyGUI(root); root.mainloop()
