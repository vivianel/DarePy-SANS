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
        """Wipes tabs and rebuilds from file."""
        self.log_to_console("------------------------------------------")
        self.log_to_console("🔄 ACTION: Reloading from YAML file...")
        if self.load_data():
            self.entries = {}
            for child in self.notebook.winfo_children():
                child.destroy()
            self.fill_notebook()
            self.log_to_console("✅ SUCCESS: GUI updated.")
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

        tk.Button(footer_f4, text="Open Masking Tool", bg="#2196F3", fg="white", font=("Arial", 10, "bold"),
                  pady=8, command=lambda: self.run_script("mask_beamstop_center.py")).pack(side="left", fill="x", expand=True, padx=(0,5))

        tk.Button(footer_f4, text="REFRESH FROM YAML", bg="#FF9800", fg="white", font=("Arial", 10, "bold"),
                  pady=8, command=self.refresh_ui).pack(side="left", fill="x", expand=True, padx=(5,0))

        self.build_config_area(scroll_f4, "Beam Center Masking", "beam_center_mask")
        self.build_config_area(scroll_f4, "Detector Geometry", "detector_geometry")

        # --- TAB 5: RADIAL INTEGRATION ---
        # (This one is special because it has sub-tabs, but we follow the footer rule)
        t5_main = ttk.Frame(self.notebook)
        self.notebook.add(t5_main, text="5. Radial Integration")

        # Pinned Footer for Tab 5
        footer_f5 = tk.Frame(t5_main, bg=bg_color)
        footer_f5.pack(side="bottom", fill="x", padx=20, pady=15)
        tk.Button(footer_f5, text="Run Radial Integration Pipeline", bg="#2196F3", fg="white", font=("Arial", 10, "bold"),
                  pady=8, command=lambda: self.run_script("caller_radial_integration.py")).pack(fill="x")

        # Sub-notebook fills the space above footer
        sub_nb = ttk.Notebook(t5_main)
        sub_nb.pack(expand=True, fill="both", padx=10, pady=5)

        s1, _ = self.create_scrollable_tab(sub_nb, "Pipeline Control"); self.build_config_area(s1, "Control", "pipeline_control")
        s2, _ = self.create_scrollable_tab(sub_nb, "Physics Corrections"); self.build_config_area(s2, "Physics", "physics_corrections")
        s3, _ = self.create_scrollable_tab(sub_nb, "Calibration Samples"); self.build_config_area(s3, "Calibration", "calibration_samples")
        s4, _ = self.create_scrollable_tab(sub_nb, "Analysis Flags"); self.build_config_area(s4, "Flags", "analysis_flags")

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
        if label == "which_instrument":
            tk.Label(parent, text=label, font=("Arial", 9, "bold")).pack(anchor="w")
            w = ttk.Combobox(parent, values=["SANS-I", "SANS-LLB"], state="readonly")
            w.set(value); w.pack(fill="x", expand=True, ipady=3)
            self.entries[config_key][path] = w
        elif isinstance(value, bool):
            var = tk.BooleanVar(value=value)
            tk.Checkbutton(parent, text=label, variable=var, font=("Arial", 9, "bold"), anchor="w").pack(fill="x")
            self.entries[config_key][path] = var
        else:
            tk.Label(parent, text=label, font=("Arial", 9, "bold")).pack(anchor="w")
            w = tk.Entry(parent)
            disp = ", ".join(map(str, value)) if isinstance(value, list) else str(value)
            w.insert(0, disp); w.pack(fill="x", expand=True, ipady=3)
            self.entries[config_key][path] = w
        if help_t:
            tk.Label(parent, text=f"  ? {help_t}", font=("Arial", 8, "italic"), fg="#7f8c8d").pack(anchor="w", pady=(0,5))

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

if __name__ == "__main__":
    root = tk.Tk(); app = DarePyGUI(root); root.mainloop()
