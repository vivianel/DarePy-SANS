"""
Package for actions that the executable can perform. Each action defines the possible command line
artuments and documentation shown.
"""
import pkgutil
import os
from argparse import ArgumentParser
from typing import Protocol, List

class Action(Protocol):
    priority = 100 # used to sort actions for help

    @classmethod
    def get_parser(cls) -> ArgumentParser:
        ...

    def __init__(self, arguments):
        ...

    def run(self):
        ...

actions:List[pkgutil.ModuleInfo]=[]

for module in pkgutil.iter_modules([os.path.dirname(__file__)]):
    actions.append(module)
