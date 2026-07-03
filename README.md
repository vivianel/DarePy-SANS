# DarePy-SANS

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![Version](https://img.shields.io/badge/version-v2026.1.0-brightgreen)

**DarePy-SANS** is a Python-based data reduction pipeline and graphical editor interface tailored for Small-Angle Neutron Scattering (SANS) analysis, optimized for SANS-LLB and SANS-I at the Paul Scherrer Institute (PSI). This repository provides the core processing libraries alongside an interactive visual environment to streamline experiment configuration and data reduction.

---

## Table of Contents
- [Prerequisites](#prerequisites)
- [A – Installing for the First Time](#a--installing-for-the-first-time)
- [B – Using the Code After the First Installation](#b--using-the-code-after-the-first-installation)
- [Project Structure & Data Management](#project-structure--data-management)
- [Contributing](#contributing)
- [License](#license)

---

## Prerequisites

Before you begin, ensure you have the following installed on your system:
* **Git**: [Download and install Git](https://git-scm.com/) if you haven't already.
* **Python**: Version 3.9 or higher is highly recommended.

---

## A – Installing for the First Time

### Step 0: Creating the Right Structure
1. Choose or determine a directory on your computer.
2. Create a folder named `SANS_analysis` (or a similar preferred name).
3. Open your terminal or command line interface and navigate into this folder. You will create the Python environment here (**Step 1**) and clone the Git repository (**Step 2**).

### Step 1: Create and Activate a Python Environment
Using standard Python (`venv`), initialize and enable an isolated environment to prevent dependency conflicts.

1. **Create a virtual environment** named `darepy-env`:
```bash
   python -m venv darepy-env
   
```

2. **Activate the virtual environment**:
* **Linux/macOS:**
```bash
source darepy-env/bin/activate

```


* **Windows (Command Prompt):**
```cmd
darepy-env\Scripts\activate.bat

```


* **Windows (PowerShell):**
```powershell
.\darepy-env\Scripts\Activate.ps1

```





### Step 2: Clone the Repository (main branch is often updated)

Open your terminal (Linux/macOS) or Command Prompt/PowerShell (Windows) and run the following commands:

1. **Clone the repository** using Git:
```bash
git clone https://github.com/vivianel/DarePy-SANS.git

```


2. **Navigate into the cloned directory**:
```bash
cd DarePy-SANS

```



### Step 3: Install Package Dependencies

Once your environment is active, ensure your core package installer (`pip`) is up to date, and then install the pipeline's dependencies using the provided `requirements.txt` file.

1. **Upgrade pip**:
```bash
python -m pip install --upgrade pip

```


2. **Install the required packages**:
```bash
pip install -r requirements.txt

```



### Step 4: Verify the Installation

To verify that the dependencies (such as `ruamel.yaml` or any SANS processing libraries) were installed successfully into your environment, you can test-import python or look at the package list:

1. **List installed packages**:
```bash
pip list

```



You are now ready to run the data reduction pipeline.

### Step 5: Run the codes

Start the visual environment for script control.

1. **Launch spyder**:
```bash
spyder

```



In the `experiments` folder, you can create multiple experiment folders. To add a new experiment, make a new folder, call it the proposal number and a short ID. From `default_exp`, make sure to copy the `config_experiment.yaml` and `instrument_registry.yaml` files in the new folder. The hdf data files from the beamtime should be copied within your experiment folder as `raw_data`. The `default_exp` cloned from the git repository is an example of the expected structure for the experiment.

2. **Start the darePy gui**
  
Navigate to the right folder in `spyder` and the execute the code `launch_gui.py` in the experiment folder. Alternatively, you can run it directly in the command line.

```bash
python launch_gui.py

```



Then follow the steps for running the data reduction.

> **Note:** If the gui is used, the codes within the `darepy` folder should not be changed. They should only be needed in case we run in the command line for debugging reasons.

### Step 6: Deactivate the environment

You should deactivate your environment when you're finished working. Simply run in the command line:

1. **Deactivate the virtual environment**:
```bash
deactivate

```



---

## B – Using the code after the first installation

Go to the directory where the environment and the git repository are saved.

> **Tip:** Because the main branch is constantly updated, you can pull the latest changes at any time from inside this directory.

1. **Pull the latest changes**:
```bash
git pull origin main

```


2. **Activate the virtual environment**:
*(Use the appropriate activation command for your OS from Step 1)*
3. **Run spyder**:
```bash
spyder

```



Find your codes and have fun!

##   Contributing
Contributions to DarePy-SANS are welcome! If you find a bug or have a feature request, please open an issue in the repository. If you would like to contribute code, please open a Pull Request.

## Acknowledgments
Acknowledgment to beamline scientists, contributors, or specific libraries used in the pipeline.

## License
This project is licensed under the [Insert License Name, e.g., MIT License] - see the LICENSE file for details.
"""
