"""
Internal utility functions.
"""

# Standard imports:
from pathlib import Path
from typing import Union

# Local imports:
from .config import get_config_data


def get_config_path_entry(entry_key: str, fallback_path: Union[Path, str]) -> Path:
  toolkit_config = get_config_data()

  entry_value = toolkit_config
  entry_key_parts = entry_key.split('/')
  for entry_key_part in entry_key_parts:
    if isinstance(entry_value, dict) and (entry_key_part in entry_value):
      entry_value = entry_value[entry_key_part]
    else:
      entry_value = None
      break

  if entry_value is None:
    entry_value = Path(toolkit_config['models']) / fallback_path

  return Path(entry_value).expanduser()
