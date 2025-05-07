import logging.config
import os

import yaml
import importlib.resources as importlib_resources

import clemcore.backends as backends

BANNER = \
    r"""
      _                                   
     | |                                  
  ___| | ___ _ __ ___   ___ ___  _ __ ___ 
 / __| |/ _ \ '_ ` _ \ / __/ _ \| '__/ _ \
| (__| |  __/ | | | | | (_| (_) | | |  __/
 \___|_|\___|_| |_| |_|\___\___/|_|  \___|
"""  # doom font, thanks to http://patorjk.com/software/taag/

if os.getenv("CLEM_DISABLE_BANNER", "0") not in ("1", "true", "yes", "on"):
    print(BANNER)


def load_logging_config():
    pkg_file_path = "utils/logging.yaml"
    with importlib_resources.files(__package__).joinpath(pkg_file_path).open("r") as f:
        default_config = yaml.safe_load(f)

    custom_file = os.path.join(os.getcwd(), "logging.yaml")
    if os.path.exists(custom_file):
        with open(custom_file) as f:
            custom_config = yaml.safe_load(f)
        return {**default_config, **custom_config}
    return default_config


try:
    import logging

    logging.config.dictConfig(load_logging_config())
except Exception as e:
    print(f"Failed to load logging config: {e}")
