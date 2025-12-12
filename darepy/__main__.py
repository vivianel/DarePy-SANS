"""
Executabel for the darepy package. Used to dispatch calls from the command line.
"""
import importlib
import sys
from . import actions
from .config import base


def print_help():
    print("Usage: darepy ACTION [-h] [...]\n    Available actions:")
    for action in actions.actions:
        module = importlib.import_module(f'darepy.actions.{action.name}', package='darepy.actions')
        print(f"        {action.name} - {module.__doc__.strip().splitlines()[0]}")

def add_global_options(parser):
    parser.add_argument('-c', '--config', dest='config_file', default=base.ConfigObject.config_file,
                        help='Name of config file to be used')

def eval_global_options(arguments):
    base.ConfigObject.config_file = arguments.config_file

def main():
    if len(sys.argv) < 2:
        print_help()
    else:
        action = sys.argv[1]
        module = importlib.import_module(f'darepy.actions.{action}', package='darepy.actions')
        parser = module.Action.get_parser()
        add_global_options(parser)
        arguments = parser.parse_args(sys.argv[2:])
        runner = module.Action(arguments)
        runner.run()

if __name__ == "__main__":
    main()