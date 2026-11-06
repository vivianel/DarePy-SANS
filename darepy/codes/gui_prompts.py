"""Small confirmation helper for GUI-launched and command-line workflows."""
import os


def confirm_update(message, title="Confirm configuration update"):
    """Use a desktop dialog in the DarePy GUI, otherwise keep CLI confirmation."""
    if os.environ.get("DAREPY_GUI_MODE") == "1":
        import tkinter as tk
        from tkinter import messagebox

        dialog_root = tk.Tk()
        dialog_root.withdraw()
        dialog_root.attributes("-topmost", True)
        try:
            return messagebox.askyesno(title, message, parent=dialog_root)
        finally:
            dialog_root.destroy()

    return input(f"{message} (y/n): ").strip().lower() in {"y", "yes"}
