"""
Basic classes used for darepy configurations. These are used to define constands and default values
and can be stored/loaded from a json config file.
"""
import os

from pathlib import Path
from dataclasses import dataclass, fields, field
from ruamel.yaml import YAML, CommentedMap

def cf(default=None, doc=None, default_factory=None):
    # convenience function to define a field with doc-string
    if doc is None:
        metadata = None
    else:
        metadata = {'doc': doc}
    if default_factory is None:
        return field(default=default, metadata=metadata)
    else:
        return field(default_factory=default_factory, metadata=metadata)

@dataclass
class ConfigObject:
    """
    A parent class used with dataclasses and allows loading/saving to a common configuration.

    Copnfig files use YAML format.
    """
    config_file = 'darepy_config.yaml'

    def __post_init__(self):
        # perform some type conversions automatically
        for field in fields(self):
            if field.type is Path:
                setattr(self, field.name, Path(getattr(self, field.name)))

    @classmethod
    def load(cls):
        config_file = ConfigObject.config_file
        if not os.path.exists(config_file):
            # if the config file does not exist, create default object
            return cls()

        yaml = YAML()
        with open(config_file, "rb") as f:
            data = yaml.load(f)
        if cls.__name__ in data:
            kwargs = data[cls.__name__]
        else:
            kwargs = {}
        return cls(**kwargs)

    def save(self):
        config_file = ConfigObject.config_file
        yaml = YAML()
        yaml.width=120
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                data = yaml.load(f) or CommentedMap()
        else:
            data = CommentedMap()
            ini_comment="Configuration file for DarePy SANS reduction"
            data.yaml_set_start_comment("="*len(ini_comment)+f'\n{ini_comment}\n'+"="*len(ini_comment))
        # remove comments of the section that will be overwritten
        data.ca.items.pop(self.__class__.__name__, None)

        # generate a new section for this config with help strings
        this = CommentedMap()
        lc = self.__doc__.strip()
        lw = max([len(line) for line in lc.splitlines()])
        this.yaml_set_start_comment("-"*lw+f'\n{lc}\n'+"-"*lw, indent=2)
        for field in fields(self):
            this[field.name] = getattr(self, field.name)
            if 'doc' in field.metadata:
                this.yaml_set_comment_before_after_key(field.name, before=field.metadata['doc'],
                                                       indent=4)
        data[self.__class__.__name__] = this
        with open(config_file, "w") as f:
            yaml.dump(data, f)

