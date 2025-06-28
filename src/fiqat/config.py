"""This module handles configuration data loaded from a single `.toml <https://toml.io/en/>`_ file.
This configuration file is required to specify local dependency paths (e.g. to model files) for various methods that are
included in the toolkit.
"""

# pylint: disable=global-statement

# Standard imports:
from pathlib import Path
import os
from typing import Optional

# External imports:
import toml

config_path_loaded: Path = None
"""The config file path corresponding to the loaded config_data."""

config_data: dict = None
"""The loaded config data"""


def get_config_path_default():
  """Returns the default config file path (`<fiqat_package_directory>/local/fiqat.toml`)."""
  return (Path(__file__) / '../../../local/fiqat.toml').resolve()


def load_config_data(config_path: Optional[Path] = None) -> dict:
  """Loads or reloads the config data and returns it.

  Parameters
  ----------
  config_path : Optional[Path]
    The path to the .toml config file.
    If this is None, the last loaded path will be used (``config_path_loaded``).
    If there is no last loaded path, the path specified via the environment variable ``'FIQAT_CONFIG'`` is used
    (``os.environ['FIQAT_CONFIG']``).
    If that environment variable doesn't exist, then the default path is used (:meth:`get_config_path_default`).

  Returns
  -------
  dict
    The loaded config data, which will also set the global :class:`fiqat.config.config_data`.
  """
  global config_data
  global config_path_loaded

  if config_path is None:
    if config_path_loaded is None:
      config_path = os.environ.get('FIQAT_CONFIG', get_config_path_default())
    else:
      config_path = config_path_loaded

  with open(config_path, 'r', encoding='utf-8') as file:
    config_data = toml.load(file)
  config_path_loaded = config_path
  return config_data


def get_config_data():
  """Returns the loaded config data (:class:`fiqat.config.config_data`).
  If it hasn't been loaded yet, :meth:`load_config_data()` will be called first.
  """
  if config_data is None:
    load_config_data()
  return config_data
