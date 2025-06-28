"""Utility functionality to handle temporary files and directories."""

# Standard imports:
from pathlib import Path
import tempfile
import contextlib
import time

# Local imports:
from .config import get_config_data


@contextlib.contextmanager
def get_temp_dir() -> Path:
  """Creates a
  `with statement context manager <https://docs.python.org/3/reference/datamodel.html#context-managers>`_
  that returns a directory meant for temporary files.

  If a 'temp_dir' path is set in the :class:`fiqat.config.config_data` (see :meth:`get_config_data()`),
  then this path will be returned (the directory will be created first if it doesn't exist already).
  Otherwise
  `tempfile.TemporaryDirectory() <https://docs.python.org/3/library/tempfile.html#tempfile.TemporaryDirectory>`_
  will be used.

  Examples
  --------
  >>> with get_temp_dir() as temp_dir:
  ...   # ... create and process temporary files within temp_dir ...
  """
  _config_data = get_config_data()
  temp_dir = _config_data.get('temp_dir', None)
  if temp_dir is not None:
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    yield temp_dir
  else:
    with tempfile.TemporaryDirectory() as temp_dir:
      yield Path(temp_dir)


@contextlib.contextmanager
def get_temp_path(suffix: str, unlink: bool = True) -> Path:
  """Creates a
  `with statement context manager <https://docs.python.org/3/reference/datamodel.html#context-managers>`_
  that returns a path meant for a temporary file that does not exist yet.

  The path will be created within a temporary directory given by a :meth:`get_temp_dir()` call.

  Parameters
  ----------
  suffix : str
    The filename is formatted as ``f'{time.monotonic_ns()}{suffix}'``.
  unlink : bool
    If ``True``, then on the exit of the context manager
    `.unlink()` will be called on the path, if it `.exists()`.
    I.e. this will automatically remove a created temporary file.

  Examples
  --------
  >>> with get_temp_path(suffix='test', unlink=True) as temp_path:
  ...   with open(temp_path, 'w') as file:
  ...     # ... write something to the file ...
  ...   # ... use the created temporary file, e.g. as input for some command line tool ...
  ...
  ... # Since unlink=True, the temp_path will be automatically removed after the with statement.
  """
  with get_temp_dir() as temp_dir:
    while True:
      temp_path = temp_dir / f'{time.monotonic_ns()}{suffix}'
      if not temp_path.exists():
        yield temp_path
        if unlink:
          if temp_path.exists():
            temp_path.unlink()
        break
