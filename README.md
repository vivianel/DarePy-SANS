DarePy-SANS
===========

"Homemade" solutions to run the radial integration and reduction of SINQ instruments SANS-I and SANS-LLB at PSI.

Installation
============

```
pip install darepy-sans
```

Usage
=====

General
-------

```
darepy ACTION [-h] [-c CONFIG] ...
```

To list all possible actions just call `darepy` without areguments.

Setup new project
-----------------

Navigate to the folder you want to work for analysis and run:

```
darepy init [-r RAW_DATA_PATH] [-c CONFIG_FILE_NAME] [-i INSTRUMENT_CONFIG]
```

All flags are optional, default values use raw_data as path, 
darepy_config.yaml as config file name and do not overwrite
any instrument configuration.

