import tkinter as tk
from tkinter import ttk, messagebox
import os, ast, shutil
from datetime import datetime             # Added datetime here
from ruamel.yaml import YAML

# --- SETUP ---
CONFIG_FILE = "config_experiment.yaml"
yaml = YAML()
yaml.preserve_quotes = True
yaml.boolean_representation = ['false', 'true']

class DarePyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DarePy-SANS Control Panel")
        self.root.geometry("800x900")

        # --- NEW: APPLY STYLES ---
        self.setup_styles()

        self.entries = {}
        # 1. Create the persistent notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)

        # ... (rest of your init code)
        self.setup_console()
        if self.load_data():
            self.fill_notebook()
        else:
            self.log_to_console("CRITICAL: Application started with empty/broken config.")

    def setup_styles(self):
        """Configures the visual 'Theme' of the tabs for high contrast."""
        style = ttk.Style()

        # 'clam' allows for better color customization than the default Windows theme
        style.theme_use('clam')

        # Configure the Notebook background
        style.configure("TNotebook", background="#f0f0f0", borderwidth=0)

        # Configure the Tab appearance (Inactive state)
        style.configure("TNotebook.Tab",
                        padding=[15, 5],
                        font=('Arial', 9, 'bold'),
                        background="#dcdcdc",
                        foreground="#333333",
                        borderwidth=1)

        # Configure the 'Map' (What happens when a tab is Selected or Hovered)
        style.map("TNotebook.Tab",
                  background=[("selected", "#2196F3"), ("active", "#cce5ff")], # Blue when selected, light blue on hover
                  foreground=[("selected", "white"), ("active", "black")],      # White text when selected
                  lightcolor=[("selected", "#2196F3")],                         # Remove the thin border line
                  borderwidth=[("selected", 0)])

    def setup_console(self):
        """Creates the permanent console at the bottom."""
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
        """Loads YAML with UTF-8 encoding to support special characters."""
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                self.config_dict = yaml.load(f)
            return True
        except Exception as e:
            if hasattr(self, 'console'):
                self.log_to_console(f"[YAML LOAD ERROR] {e}")
            messagebox.showerror("YAML Error", f"Could not load config file:\n{e}")
            return False

    def save_data(self):
        """Saves current GUI state and creates a timestamped archive copy named by project."""
        try:
            # 1. Update the internal dictionary from GUI entries
            for s_key, keys in self.entries.items():
                for path, entry in keys.items():
                    raw = entry.get()
                    actual = raw if isinstance(raw, bool) else self._parse_string(raw)
                    target = self.config_dict[s_key]
                    for p in path[:-1]:
                        if p not in target: target[p] = {}
                        target = target[p]
                    target[path[-1]] = actual

            # 2. Save the primary 'Working' file (UTF-8)
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_dict, f)

            # 3. EXTRACT PROJECT NAME
            # Pull the folder name from the project_base path
            p_path = self.config_dict.get('analysis_paths', {}).get('project_base', 'UnknownProject')
            # normpath handles trailing slashes, basename gets the last folder name
            project_name = os.path.basename(os.path.normpath(p_path))

            # 4. CREATE THE SNAPSHOT
            history_dir = os.path.join(os.path.dirname(__file__), "config_history")
            if not os.path.exists(history_dir):
                os.makedirs(history_dir)

            # Generate filename: config_experiment_Jasper_2026-04-10_16-50-00.yaml
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            archive_name = f"config_experiment_{project_name}_{timestamp}.yaml"
            archive_path = os.path.join(history_dir, archive_name)

            # Copy the newly saved file to the archive
            shutil.copy2(CONFIG_FILE, archive_path)

            # 5. LOG TO CONSOLE
            self.log_to_console("------------------------------------------")
            self.log_to_console("✅ SUCCESS: Configuration saved and archived.")
            self.log_to_console(f"📁 Project: {project_name}")
            self.log_to_console(f"📜 Snapshot: {archive_name}")
            self.log_to_console("------------------------------------------")
            return True

        except Exception as e:
            self.log_to_console(f"❌ SAVE ERROR: {e}")
            messagebox.showerror("Save Error", f"Failed to save and archive:\n{e}")
            return False

    def _parse_string(self, raw_val):
        """Helper to convert entry text to appropriate Python types."""
        v = str(raw_val).strip()
        if v == "": return ""
        if v.lower() == 'true': return True
        if v.lower() == 'false': return False
        if v.lower() == 'none': return None
        if v.startswith(('[', '{')):
            try: return ast.literal_eval(v)
            except: return v
        try:
            return float(v) if '.' in v else int(v)
        except:
            return v

    def refresh_ui(self):
        """Reloads YAML and restores Main-Tab and Sub-Tab positions."""
        # 1. Capture Main Tab Index
        try:
            main_idx = self.notebook.index("current")
        except:
            main_idx = 0

        # 2. Capture Sub-Tab Index (if it exists)
        sub_idx = 0
        if hasattr(self, 'sub_nb'):
            try:
                sub_idx = self.sub_nb.index("current")
            except:
                sub_idx = 0

        self.log_to_console("------------------------------------------")
        self.log_to_console("🔄 ACTION: Refreshing GUI and restoring position...")

        if self.load_data():
            # Clear everything
            self.entries = {}
            for child in self.notebook.winfo_children():
                child.destroy()

            # Rebuild the entire UI
            self.fill_notebook()

            # IMPORTANT: Force the UI to calculate all new widget positions
            self.root.update_idletasks()

            # 3. Step One: Restore the Main Tab
            try:
                self.notebook.select(main_idx)
                # Force another update so the Sub-Notebook becomes 'visible' to the OS
                self.root.update_idletasks()

                # 4. Step Two: Restore the Sub-Tab
                # We check main_idx == 4 because Tab 5 is the 5th tab (index 4)
                if main_idx == 4 and hasattr(self, 'sub_nb'):
                    self.sub_nb.select(sub_idx)

            except Exception as e:
                self.log_to_console(f"⚠️ Navigation restore failed: {e}")

            self.log_to_console("✅ SUCCESS: GUI refreshed at previous position.")
            self.log_to_console("------------------------------------------")

    def create_scrollable_tab(self, notebook, text):
        """Creates a tab with a fixed footer for buttons and a scrolling top for inputs."""
        tab = ttk.Frame(notebook)
        notebook.add(tab, text=text)
        bg_color = self.root.cget('bg')

        # 1. THE RESERVED FOOTER (Stays at bottom, doesn't scroll)
        footer_f = tk.Frame(tab, bg=bg_color)
        footer_f.pack(side="bottom", fill="x", padx=20, pady=15)

        # 2. THE SCROLLABLE CONTENT (Takes all space above the footer)
        container = tk.Frame(tab, bg=bg_color)
        container.pack(side="top", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(container, orient="vertical")
        scrollbar.pack(side="right", fill="y")

        canvas = tk.Canvas(container, highlightthickness=0, bg=bg_color)
        canvas.pack(side="left", fill="both", expand=True)

        scroll_f = tk.Frame(canvas, bg=bg_color)
        cw = canvas.create_window((0, 0), window=scroll_f, anchor="nw")

        # Standard linkages
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.config(command=canvas.yview)

        # Resize and Scrollregion logic
        scroll_f.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(cw, width=e.width))

        # Mousewheel support
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", lambda ev: self._on_mousewheel(ev, canvas)))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        return scroll_f, footer_f

    def _on_mousewheel(self, event, canvas):
        """Standard mouse wheel support."""
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def fill_notebook(self):
        """Builds tabs with a guaranteed fixed footer for buttons."""
        bg_color = self.root.cget('bg')

        # --- TAB 1: GLOBAL SETUP ---
        scroll_f1, footer_f1 = self.create_scrollable_tab(self.notebook, "1. Global Setup")

        tk.Button(footer_f1, text="SAVE SETTINGS", bg="#4CAF50", fg="white", font=("Arial", 10, "bold"),
                  pady=8, command=self.save_data).pack(side="left", fill="x", expand=True, padx=(0,5))

        tk.Button(footer_f1, text="REFRESH FROM YAML", bg="#FF9800", fg="white", font=("Arial", 10, "bold"),
                  pady=8, command=self.refresh_ui).pack(side="left", fill="x", expand=True, padx=(5,0))

        self.build_config_area(scroll_f1, "Analysis Paths", "analysis_paths")
        self.build_config_area(scroll_f1, "Instrument", "instrument_setup")

        # --- TAB 2: RENAME ---
        scroll_f2, footer_f2 = self.create_scrollable_tab(self.notebook, "2. Rename Samples")
        tk.Button(footer_f2, text="Execute Sample Renaming", bg="#2196F3", fg="white", font=("Arial", 10, "bold"),
                  pady=8, command=lambda: self.run_script("rename_samples.py")).pack(fill="x")
        self.build_config_area(scroll_f2, "Rename Settings", "rename_samples")

        # --- TAB 3: VISUALIZATION ---
        scroll_f3, footer_f3 = self.create_scrollable_tab(self.notebook, "3. 2D Visualization")
        tk.Button(footer_f3, text="Generate 2D Plots/GIF", bg="#2196F3", fg="white", font=("Arial", 10, "bold"),
                  pady=8, command=lambda: self.run_script("plot_2Dpattern.py")).pack(fill="x")
        self.build_config_area(scroll_f3, "Plot 2D Settings", "plot_2d")


        # --- TAB 4: MASK & CENTER ---
        scroll_f4, footer_f4 = self.create_scrollable_tab(self.notebook, "4. Mask & Center")

        # 1. CLEAN MASKING SETUP (Manually built to avoid redundant 'scans' boxes)
        setup_group = tk.LabelFrame(scroll_f4, text="Masking Setup", padx=10, pady=10, bg=bg_color)
        setup_group.pack(fill="x", padx=10, pady=5, anchor="nw")

        # Add the description/help text from YAML if it exists
        desc = self.config_dict.get('beam_center_mask', {}).get('_desc', '').strip()
        if desc:
            tk.Label(setup_group, text=desc, font=("Arial", 9, "italic"), fg="#555",
                     bg="#f8f9fa", justify="left", wraplength=650, padx=10, pady=10).pack(fill="x", pady=(0, 10))

        # Only show the Sample Name field here
        tk.Label(setup_group, text="sample_name", font=("Arial", 9, "bold"), bg=bg_color).pack(anchor="w")
        s_name_val = self.config_dict.get('beam_center_mask', {}).get('sample_name', 'AgBE')
        s_ent = tk.Entry(setup_group)
        s_ent.insert(0, str(s_name_val))
        s_ent.pack(fill="x", expand=True, ipady=3)

        # Register for saving: Path is ('sample_name',)
        if 'beam_center_mask' not in self.entries: self.entries['beam_center_mask'] = {}
        self.entries['beam_center_mask'][('sample_name',)] = s_ent

        # 2. THE ACTION AREA (The rows with blue buttons)
        action_group = tk.LabelFrame(scroll_f4, text="Distance-Specific Masking", padx=10, pady=10,
                                    bg=bg_color, fg="#2196F3", font=("Arial", 10, "bold"))
        action_group.pack(fill="x", padx=10, pady=5, anchor="nw")

        scans_dict = self.config_dict.get('beam_center_mask', {}).get('scans', {})

        # ADD NEW DISTANCE ROW (The "Management Row")
        add_row = tk.Frame(action_group, bg=bg_color)
        add_row.pack(fill="x", pady=(0, 15))

        tk.Label(add_row, text="Add Distance:", font=("Arial", 9, "italic"), bg=bg_color).pack(side="left")
        self.new_dist_entry = tk.Entry(add_row, width=10)
        self.new_dist_entry.pack(side="left", padx=5)

        tk.Button(add_row, text=" + ", bg="#4CAF50", fg="white", font=("Arial", 9, "bold"),
                  command=self.add_new_distance).pack(side="left")

        # Horizontal line separator
        ttk.Separator(action_group, orient='horizontal').pack(fill='x', pady=5)

        if isinstance(scans_dict, dict):
            sorted_distances = sorted(scans_dict.keys(), key=float)
            for dist in sorted_distances:
                row = tk.Frame(action_group, bg=bg_color)
                row.pack(fill="x", pady=5)

                tk.Label(row, text=f"{dist} m:", width=10, font=("Arial", 9, "bold"),
                         anchor="w", bg=bg_color).pack(side="left")

                scan_val = scans_dict.get(dist, "")
                ent = tk.Entry(row, width=15)
                ent.insert(0, str(scan_val))
                ent.pack(side="left", padx=5)

                # Register for saving: Path is ('scans', dist)
                self.entries['beam_center_mask'][('scans', dist)] = ent

                tk.Button(row, text=f"RUN MASKING TOOL", bg="#2196F3", fg="white",
                          font=("Arial", 8, "bold"), padx=10,
                          command=lambda d=dist: self.run_mask_for_dist(d)).pack(side="right", fill="x", expand=True)

        # 3. Detector Geometry Display (Automatic builder is fine here)
        self.build_config_area(scroll_f4, "Current Detector Geometry", "detector_geometry")

        # 4. FOOTER
        btn_f4 = tk.Frame(footer_f4, bg=bg_color)
        btn_f4.pack(fill="x")
        tk.Button(btn_f4, text="REFRESH FROM YAML", bg="#FF9800", fg="white",
                  font=("Arial", 10, "bold"), pady=8, command=self.refresh_ui).pack(fill="x")

        # --- TAB 5: RADIAL INTEGRATION ---
        t5_main = ttk.Frame(self.notebook)
        self.notebook.add(t5_main, text="5. Radial Integration")

        # Pinned Footer
        footer_f5 = tk.Frame(t5_main, bg=self.root.cget('bg'))
        footer_f5.pack(side="bottom", fill="x", padx=20, pady=15)
        tk.Button(footer_f5, text="Run Radial Integration Pipeline", bg="#2196F3", fg="white",
                  font=("Arial", 10, "bold"), pady=8,
                  command=lambda: self.run_script("caller_radial_integration.py")).pack(fill="x")

        self.sub_nb = ttk.Notebook(t5_main)
        self.sub_nb.pack(expand=True, fill="both", padx=10, pady=5)

        # Sub-tabs 1, 2, 4 use the standard builder
        s1, _ = self.create_scrollable_tab(self.sub_nb, "Pipeline Control"); self.build_config_area(s1, "Control", "pipeline_control")
        s2, _ = self.create_scrollable_tab(self.sub_nb, "Physics Corrections"); self.build_config_area(s2, "Physics", "physics_corrections")

        # --- CUSTOM SUB-TAB 3: CALIBRATION SAMPLES ---
        s3_scroll, _ = self.create_scrollable_tab(self.sub_nb, "Calibration Samples")

        # !! THE FIX: Initialize the dictionary key for this tab !!
        if 'calibration_samples' not in self.entries:
            self.entries['calibration_samples'] = {}

        # A. Basic Calibration (Standard entries)
        basic_cal = tk.LabelFrame(s3_scroll, text="Primary Standards", padx=10, pady=10, bg=self.root.cget('bg'))
        basic_cal.pack(fill="x", padx=10, pady=5)

        for field in ['dark_current', 'water', 'water_cell']:
            tk.Label(basic_cal, text=field, font=("Arial", 9, "bold"), bg=self.root.cget('bg')).pack(anchor="w")
            # Safety get to avoid crashes if YAML is missing a field
            val = self.config_dict.get('calibration_samples', {}).get(field, "")
            ent = tk.Entry(basic_cal)
            ent.insert(0, str(val))
            ent.pack(fill="x", pady=(0, 5))

            # This is where it crashed; now it has a place to live
            self.entries['calibration_samples'][(field,)] = ent

        # B. Two-Column Dictionary Editors
        self._build_dict_editor(s3_scroll, "Empty Cell Mapping (Backgrounds)", "calibration_samples", "empty_cell")
        self._build_dict_editor(s3_scroll, "Sample Thickness (cm)", "calibration_samples", "thickness")

        # Sub-tab 4
        s4, _ = self.create_scrollable_tab(self.sub_nb, "Analysis Flags"); self.build_config_area(s4, "Flags", "analysis_flags")

        # --- TAB 6: MERGING ---
        scroll_f6, footer_f6 = self.create_scrollable_tab(self.notebook, "6. Merging Curves")
        tk.Button(footer_f6, text="Run Data Merging", bg="#2196F3", fg="white", font=("Arial", 10, "bold"),
                  pady=8, command=lambda: self.run_script("caller_merging.py")).pack(fill="x")
        self.build_config_area(scroll_f6, "Merging Settings", "merging_settings")



    def build_config_area(self, parent, title, config_key):
        """Packs LabelFrames so they look like a clean, continuous column."""
        # Ensure the group matches the parent background
        bg_color = self.root.cget('bg')
        group = tk.LabelFrame(parent, text=title, padx=10, pady=10, bg=bg_color)

        # Packing with anchor="w" and fill="x" creates the clean column look
        group.pack(fill="x", padx=10, pady=5, anchor="nw")

        section = self.config_dict.get(config_key, {})
        if config_key not in self.entries:
            self.entries[config_key] = {}

        self._build_nested_ui(group, section, config_key, path=())

    def run_mask_for_dist(self, dist):
        """Saves the specific scan number to the main 'scan_nr' field and runs the script."""
        # Save all current UI entries to the dictionary/YAML first
        if self.save_data():
            # Extract the scan number specifically for this distance
            target_scan = self.entries['beam_center_mask'][('scans', dist)].get()

            # Inject this into the main 'scan_nr' field the script expects
            self.config_dict['beam_center_mask']['scan_nr'] = int(target_scan)

            # Final silent save to ensure the script sees this scan_nr
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_dict, f)

            self.log_to_console(f"🎯 Reduction Target: Distance {dist}m | Scan {target_scan}")
            self.run_script("mask_beamstop_center.py")

    def _build_nested_ui(self, parent, data, config_key, path):
        if not isinstance(data, dict): return
        if "_desc" in data:
            tk.Label(parent, text=data["_desc"].strip(), font=("Arial", 9, "italic"),
                     fg="#555", bg="#f8f9fa", justify="left", wraplength=650, padx=10, pady=10).pack(fill="x", pady=(0, 10))
        ca = getattr(data, 'ca', None)
        for k, v in data.items():
            if str(k).startswith("_"): continue
            help_t = ""
            if ca and k in ca.items:
                o = ca.items[k]
                if len(o) > 2 and o[2]: help_t = str(o[2].value).strip().lstrip('#').strip()
            if isinstance(v, dict):
                sub = tk.LabelFrame(parent, text=str(k), padx=10, pady=5, font=("Arial", 9, "bold"), fg="#e67e22")
                sub.pack(fill="x", padx=5, pady=5)
                self._build_nested_ui(sub, v, config_key, path + (k,))
            else:
                self._create_field(parent, str(k), v, config_key, path + (k,), help_t)

    def _create_field(self, parent, label, value, config_key, path, help_t=""):
        """Creates the appropriate UI widget for a setting and links it to the root master."""
        bg_color = parent.cget('bg') # Keep the clean look

        if label == "which_instrument":
            tk.Label(parent, text=label, font=("Arial", 9, "bold"), bg=bg_color).pack(anchor="w")
            w = ttk.Combobox(parent, values=["SANS-I", "SANS-LLB"], state="readonly")
            w.set(value); w.pack(fill="x", expand=True, ipady=3)
            self.entries[config_key][path] = w

        elif isinstance(value, bool):
            # THE FIX: We explicitly pass master=self.root so Tkinter doesn't panic
            var = tk.BooleanVar(master=self.root, value=value)
            tk.Checkbutton(parent, text=label, variable=var, font=("Arial", 9, "bold"),
                           anchor="w", bg=bg_color, activebackground=bg_color).pack(fill="x")
            self.entries[config_key][path] = var

        else:
            tk.Label(parent, text=label, font=("Arial", 9, "bold"), bg=bg_color).pack(anchor="w")
            w = tk.Entry(parent)
            disp = ", ".join(map(str, value)) if isinstance(value, list) else str(value)
            w.insert(0, disp); w.pack(fill="x", expand=True, ipady=3)
            self.entries[config_key][path] = w

        if help_t:
            tk.Label(parent, text=f"  ? {help_t}", font=("Arial", 8, "italic"),
                     fg="#7f8c8d", bg=bg_color).pack(anchor="w", pady=(0,5))

    def log_to_console(self, msg):
        self.console.insert(tk.END, f"{msg}\n"); self.console.see(tk.END); self.root.update()

    def run_script(self, name):
        """Handoff with full instructional logging."""
        if self.save_data():
            self.log_to_console("\n🚀 [HANDOFF INITIATED]")
            self.log_to_console(f"1. Opening: {name}")
            self.log_to_console("2. ⚠️  ACTION: Switch to Spyder window.")
            self.log_to_console("3. 💡 TIP: Open a NEW console (Ctrl+T) for a clean run.")
            self.log_to_console("4. ⚠️  ACTION: Press 'F5' to execute.")
            self.log_to_console("------------------------------------------")
            os.startfile(os.path.join(os.path.dirname(__file__), name))

    def add_new_distance(self):
        """Adds a new distance key to the dictionary and refreshes the UI."""
        new_val = self.new_dist_entry.get().strip()

        if not new_val:
            return

        try:
            # Ensure it's a valid number (integer or float)
            dist_key = float(new_val)
            if dist_key.is_integer():
                dist_key = int(dist_key)

            # Access the dictionary
            scans = self.config_dict.get('beam_center_mask', {}).get('scans', {})

            # Add the new key with a placeholder scan number (or 0)
            if dist_key not in scans:
                scans[dist_key] = 0
                self.config_dict['beam_center_mask']['scans'] = scans

                # Save and Refresh to show the new row
                self.save_data()
                self.refresh_ui()
                self.log_to_console(f"➕ Added new detector distance: {dist_key}m")
            else:
                messagebox.showinfo("Exists", f"Distance {dist_key}m is already in the list.")

        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for the distance.")

    def _build_dict_editor(self, parent, title, main_key, sub_key):
        """Builds a two-column editor for dictionaries with a Quick-Add row."""
        bg_color = self.root.cget('bg')
        group = tk.LabelFrame(parent, text=title, padx=10, pady=10, bg=bg_color, fg="#e67e22", font=("Arial", 10, "bold"))
        group.pack(fill="x", padx=10, pady=5)

        # 1. THE QUICK-ADD ROW
        add_f = tk.Frame(group, bg=bg_color)
        add_f.pack(fill="x", pady=(0, 10))

        new_name = tk.Entry(add_f, width=25); new_name.insert(0, "New Sample Name...")
        new_name.pack(side="left", padx=2)
        new_val = tk.Entry(add_f, width=15); new_val.insert(0, "Value")
        new_val.pack(side="left", padx=2)

        tk.Button(add_f, text="+", bg="#4CAF50", fg="white", font=("Arial", 9, "bold"),
                  command=lambda n=new_name, v=new_val, mk=main_key, sk=sub_key:
                  self.add_dict_item(mk, sk, n, v)).pack(side="left", padx=5)

        # 2. THE COLUMN HEADERS
        header_f = tk.Frame(group, bg=bg_color)
        header_f.pack(fill="x")
        tk.Label(header_f, text="Sample Name", width=28, font=("Arial", 8, "bold"), bg=bg_color, anchor="w").pack(side="left")
        tk.Label(header_f, text="Value", font=("Arial", 8, "bold"), bg=bg_color, anchor="w").pack(side="left")

        # 3. THE DATA ROWS (Side-by-Side)
        data_dict = self.config_dict.get(main_key, {}).get(sub_key, {})
        for name, value in data_dict.items():
            row = tk.Frame(group, bg=bg_color)
            row.pack(fill="x", pady=2)

            # Label (Sample Name)
            tk.Label(row, text=name, width=30, anchor="w", bg=bg_color, font=("Arial", 9)).pack(side="left")

            # Entry (Value)
            ent = tk.Entry(row, width=20)
            ent.insert(0, str(value))
            ent.pack(side="left")

            # Register for saving: Path is (sub_key, name)
            if main_key not in self.entries: self.entries[main_key] = {}
            self.entries[main_key][(sub_key, name)] = ent

    def add_dict_item(self, main_key, sub_key, name_ent, val_ent):
        """Adds a new key-value pair to the calibration dictionaries."""
        name = name_ent.get().strip()
        val = val_ent.get().strip()

        if name and val and "New Sample" not in name:
            # Add to dictionary
            if sub_key not in self.config_dict[main_key]:
                self.config_dict[main_key][sub_key] = {}

            self.config_dict[main_key][sub_key][name] = self._parse_string(val)

            # Save, log, and Refresh
            self.save_data()
            self.refresh_ui()
            self.log_to_console(f"📝 Added to {sub_key}: {name} = {val}")

if __name__ == "__main__":
    root = tk.Tk(); app = DarePyGUI(root); root.mainloop()
