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
        self.root.geometry("1000x850")

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
                    target[path[-1]] = actual

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

        # Description text
        if "_desc" in data:
            desc_text = str(data["_desc"]).strip()
            desc_box = tk.Message(parent, text=desc_text, font=("Arial", 9, "italic"),
                                  fg="#555", bg="#f8f9fa", width=800,
                                  justify="left", padx=10, pady=5)
            desc_box.pack(fill="x", pady=(0, 10), anchor="w")

        for k, v in data.items():
            if str(k).startswith("_"): continue

            # Extract comment
            inline_comment = ""
            if hasattr(data, 'ca') and k in data.ca.items:
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
                self._create_field(parent, str(k), v, config_key, current_path, inline_comment)

    def _create_field(self, parent, label, value, config_key, path, comment=""):
        bg = parent.cget('bg')
        f = tk.Frame(parent, bg=bg)
        f.pack(fill="x", pady=4, anchor="w")

        label_frame = tk.Frame(f, bg=bg)
        label_frame.pack(fill="x", anchor="w")
        tk.Label(label_frame, text=label, font=("Arial", 9, "bold"), bg=bg).pack(side="left")

        if comment:
            tk.Label(label_frame, text=f"  # {comment}", font=("Arial", 8, "italic"),
                     fg="#888", bg=bg).pack(side="left")

        if isinstance(value, bool):
            var = tk.BooleanVar(master=self.root, value=value)
            tk.Checkbutton(f, text="", variable=var, bg=bg).pack(anchor="w")
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

        # --- TAB 2: RENAME SAMPLES ---
        s2, f2 = self.create_scrollable_tab(self.notebook, "2. Rename Samples")
        tk.Button(f2, text="RENAME SAMPLES", bg="#2196F3", fg="white", font=("Arial", 10, "bold"),
                  command=lambda: self.run_script("caller_rename_samples.py")).pack(fill="x")
        self.build_config_area(s2, "Rename Settings", "rename_samples")

        # --- TAB 3: 2D VISUALIZATION ---
        s3, f3 = self.create_scrollable_tab(self.notebook, "3. 2D Visualization")
        tk.Button(f3, text="GENERATE 2D PLOTS/GIF", bg="#2196F3", fg="white", font=("Arial", 10, "bold"),
                  command=lambda: self.run_script("caller_plot_2Dpattern.py")).pack(fill="x")
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

        # --- TAB 5: TRANSMISSION ---
        s5, f5 = self.create_scrollable_tab(self.notebook, "5. Transmission")
        tk.Button(f5, text="RUN TRANSMISSION CALCULATION", bg="#03A9F4", fg="white",
                  font=("Arial", 10, "bold"), pady=12,
                  command=lambda: self.run_script("caller_transmission.py")).pack(fill="x")
        self.build_config_area(s5, "Transmission Physics", "transmission_setup")
        self._build_dict_editor(s5, "Sample Thickness (cm)", "calibration_samples", "thickness")

        # --- TAB 6: RADIAL INTEGRATION ---
        t6_main = ttk.Frame(self.notebook); self.notebook.add(t6_main, text="6. Radial Integration")
        f6 = tk.Frame(t6_main, bg=bg); f6.pack(side="bottom", fill="x", padx=20, pady=15)
        tk.Button(f6, text="RUN FULL INTEGRATION PIPELINE", bg="#2196F3", fg="white", font=("Arial", 10, "bold"),
                  pady=10, command=lambda: self.run_script("caller_radial_integration.py")).pack(fill="x")

        self.sub_nb = ttk.Notebook(t6_main); self.sub_nb.pack(expand=True, fill="both", padx=10, pady=5)

        sc, _ = self.create_scrollable_tab(self.sub_nb, "Calibration Samples")
        if 'calibration_samples' not in self.entries: self.entries['calibration_samples'] = {}
        tk.Button(sc, text="🔍 CHECK EXISTENCE OF ALL CALIBRATION FILES", bg="#2196F3", fg="white",
                  font=("Arial", 9, "bold"), pady=8, command=lambda: self.run_script("caller_check_calibration.py")).pack(fill="x", pady=10)

        cal_f = tk.LabelFrame(sc, text="Primary Standards", padx=10, pady=10, bg=bg); cal_f.pack(fill="x", padx=10, pady=5)
        for field in ['dark_current', 'water', 'water_cell']:
            tk.Label(cal_f, text=field, font=("Arial", 9, "bold"), bg=bg).pack(anchor="w")
            e = tk.Entry(cal_f); e.insert(0, self.config_dict.get('calibration_samples', {}).get(field, "")); e.pack(fill="x", pady=2)
            self.entries['calibration_samples'][(field,)] = e

        self._build_dict_editor(sc, "Empty Cell Mapping", "calibration_samples", "empty_cell")
        self._build_dict_editor(sc, "Sample Thickness (cm)", "calibration_samples", "thickness")

        s_combined, _ = self.create_scrollable_tab(self.sub_nb, "Pipeline & Physics Settings")
        self.build_config_area(s_combined, "Pipeline Control", "pipeline_control")
        self.build_config_area(s_combined, "Physics Corrections", "physics_corrections")

        sd, _ = self.create_scrollable_tab(self.sub_nb, "Analysis Flags")
        self.build_config_area(sd, "Flags", "analysis_flags")

        # --- TAB 7: MERGING CURVES ---
        s7, f7 = self.create_scrollable_tab(self.notebook, "7. Merging Curves")
        tk.Button(f7, text="Run Data Merging", bg="#2196F3", fg="white", font=("Arial", 10, "bold"),
                  pady=8, command=lambda: self.run_script("caller_merging.py")).pack(fill="x")

        top_f = tk.LabelFrame(s7, text="Merging Control", padx=10, pady=10, bg=bg)
        top_f.pack(fill="x", padx=10, pady=5)

        m_data = self.config_dict.get('merging_settings', {})
        if 'merging_settings' not in self.entries: self.entries['merging_settings'] = {}

        for k in ['run_step_1_plotting', 'run_step_2_merging', 'run_step_3_interpolation', 'run_step_4_incoherent']:
            val = m_data.get(k, False)
            var = tk.BooleanVar(master=self.root, value=val)
            tk.Checkbutton(top_f, text=k, variable=var, bg=bg).pack(anchor="w")
            self.entries['merging_settings'][(k,)] = var

        tk.Label(top_f, text="interp_type", font=("Arial", 9, "bold"), bg=bg).pack(anchor="w", pady=(5,0))
        i_ent = tk.Entry(top_f); i_ent.insert(0, m_data.get('interp_type', 'log'))
        i_ent.pack(fill="x"); self.entries['merging_settings'][('interp_type',)] = i_ent

        self._build_merging_table(s7, "Data Clipping (Skip Points)", "merging_settings")


    # --- CUSTOM UI LOGIC (Tables & Dictionaries) ---
    def _build_merging_table(self, parent, title, m_key):
        bg = self.root.cget('bg')
        group = tk.LabelFrame(parent, text=title, padx=10, pady=10, bg=bg, fg="#e67e22", font=("Arial", 10, "bold"))
        group.pack(fill="x", padx=10, pady=5)

        add_f = tk.Frame(group, bg=bg); add_f.pack(fill="x", pady=(0, 10))
        tk.Label(add_f, text="Add Distance:", font=("Arial", 8, "italic"), bg=bg).pack(side="left")
        new_d = tk.Entry(add_f, width=10); new_d.pack(side="left", padx=5)
        tk.Button(add_f, text=" + ", bg="#4CAF50", fg="white",
                  command=lambda: self.add_merging_dist(m_key, new_d)).pack(side="left")

        h_f = tk.Frame(group, bg=bg); h_f.pack(fill="x")
        tk.Label(h_f, text="Distance (m)", width=12, font=("Arial", 8, "bold"), bg=bg).pack(side="left")
        tk.Label(h_f, text="Skip Start", width=15, font=("Arial", 8, "bold"), bg=bg).pack(side="left")
        tk.Label(h_f, text="Skip End", width=15, font=("Arial", 8, "bold"), bg=bg).pack(side="left")

        starts = self.config_dict[m_key].get('skip_start', {})
        ends = self.config_dict[m_key].get('skip_end', {})
        all_dists = sorted(set(list(starts.keys()) + list(ends.keys())), key=float)

        for d in all_dists:
            r = tk.Frame(group, bg=bg); r.pack(fill="x", pady=2)
            tk.Label(r, text=f"{d} m", width=12, anchor="w", bg=bg, font=("Arial", 9, "bold")).pack(side="left")
            s_ent = tk.Entry(r, width=15); s_ent.insert(0, str(starts.get(d, 0))); s_ent.pack(side="left", padx=2)
            self.entries[m_key][('skip_start', d)] = s_ent
            e_ent = tk.Entry(r, width=15); e_ent.insert(0, str(ends.get(d, 0))); e_ent.pack(side="left", padx=2)
            self.entries[m_key][('skip_end', d)] = e_ent
            tk.Button(r, text=" 🗑️ ", bg="#f44336", fg="white", font=("Arial", 7),
                      command=lambda dist=d: self.remove_merging_dist(m_key, dist)).pack(side="right")

    def _build_dict_editor(self, parent, title, m_key, s_key):
        bg = self.root.cget('bg')
        if m_key not in self.entries: self.entries[m_key] = {}
        g = tk.LabelFrame(parent, text=title, padx=10, pady=10, bg=bg, fg="#e67e22", font=("Arial", 10, "bold"))
        g.pack(fill="x", padx=10, pady=5)
        af = tk.Frame(g, bg=bg); af.pack(fill="x", pady=(0,10))
        nn = tk.Entry(af, width=25); nn.insert(0, "New Sample..."); nn.pack(side="left", padx=2)
        nv = tk.Entry(af, width=15); nv.insert(0, "0.1"); nv.pack(side="left", padx=2)
        tk.Button(af, text=" + ", bg="#4CAF50", fg="white", font=("Arial", 9, "bold"),
                  command=lambda: self.add_dict_item(m_key, s_key, nn, nv)).pack(side="left", padx=5)

        d_dict = self.config_dict.get(m_key, {}).get(s_key, {})
        for k, v in d_dict.items():
            r = tk.Frame(g, bg=bg); r.pack(fill="x", pady=2)
            tk.Label(r, text=k, width=30, anchor="w", bg=bg).pack(side="left")
            e = tk.Entry(r, width=20); e.insert(0, str(v)); e.pack(side="left")
            self.entries[m_key][(s_key, k)] = e
            tk.Button(r, text=" 🗑️ ", bg="#f44336", fg="white", font=("Arial", 7),
                      command=lambda mk=m_key, sk=s_key, item=k: self.remove_dict_item(mk, sk, item)).pack(side="right")

    # --- ACTION METHODS ---
    def run_mask_for_dist(self, dist):
        if self.save_data():
            scan = self.entries['beam_center_mask'][('scans', dist)].get()
            self.config_dict['beam_center_mask']['scan_nr'] = int(scan)
            with open(self.config_file, 'w', encoding='utf-8') as f: yaml.dump(self.config_dict, f)
            self.run_script("caller_mask_beamstop_center.py")

    def remove_distance(self, dist):
        if messagebox.askyesno("Confirm", f"Remove {dist}m from all settings?"):
            if 'beam_center_mask' in self.entries:
                self.entries['beam_center_mask'].pop(('scans', dist), None)
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
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_dict, f)
            self.log_to_console(f"🗑️ Purged distance {dist}m from all records.")
            self.refresh_ui()

    def add_merging_dist(self, m_key, entry_widget):
        val = entry_widget.get().strip()
        if val:
            try:
                d = float(val)
                if d.is_integer(): d = int(d)
                self.config_dict[m_key].setdefault('skip_start', {})[d] = 0
                self.config_dict[m_key].setdefault('skip_end', {})[d] = 0
                self.save_data()
                self.refresh_ui()
            except: pass

    def remove_merging_dist(self, m_key, dist):
        if messagebox.askyesno("Confirm", f"Remove distance {dist}m from skip points?"):
            if m_key in self.entries:
                self.entries[m_key].pop(('skip_start', dist), None)
                self.entries[m_key].pop(('skip_end', dist), None)
            self.config_dict[m_key].get('skip_start', {}).pop(dist, None)
            self.config_dict[m_key].get('skip_end', {}).pop(dist, None)
            with open(self.config_file, 'w', encoding='utf-8') as f:
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
        """Runs the script in the active Spyder console."""
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
                        ipy.run_line_magic('run', f'"{script_path}" "{self.config_file}"')
                        return
                except:
                    pass
                import subprocess
                subprocess.Popen([sys.executable, script_path, self.config_file])
            else:
                self.log_to_console(f"❌ ERROR: {name} not found.")

if __name__ == "__main__":
    root = tk.Tk()
    app = DarePyGUI(root)
    root.mainloop()
