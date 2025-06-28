"""Terminal support functionality.

This attempts to import ``colorama`` and ``termcolor``,
calling ``colorama.init()``,
and re-exports ``termcolor.colored`` and ``termcolor.cprint``.
For this the module will catch any ``ImportError`` and provide fallback functions that omit coloring.

Besides coloring the module also provides a ``tqdm`` helper function for progress bars.
"""

# Standard imports:
from typing import Iterable, Union, Optional, Any

# External imports:
import tqdm as tqdm_module

# Local imports:
from .config import get_config_data

try:
  # External imports:
  from termcolor import colored, cprint  # pylint: disable=unused-import
  import colorama

  colorama.init()
except ImportError:

  def colored(text: str, *_args, **_kwargs) -> str:
    """Fallback for a failed termcolor.colored import that just returns ``text`` without change."""
    return text

  def cprint(text: str, *_args, **_kwargs):
    """Fallback for a failed termcolor.cprint import that just prints ``text`` using ``print`` without color."""
    print(text)


def tqdm(output_items: Iterable,
         *tqdm_args,
         input_items: Optional[Any] = None,
         tqdm_config: Union[bool, tuple, list, dict] = False,
         **tqdm_kwargs) -> Iterable:
  """Helper function that wraps the `tqdm.tqdm <https://tqdm.github.io/>`_ progress bar function.

  This is mostly intended for the main API function implementation.
  Besides the described parameters, it will set a default value for the ``colour`` keyword argument
  if a ``tqdm.colour`` value is set in the toolkit config file (:mod:`fiqat.config`).

  Parameters
  ----------
  output_items : Iterable
    The output items that will be passed as the first argument to `tqdm.tqdm`.
  *tqdm_args : tuple
    Other positional arguments for `tqdm.tqdm` (starting with the second argument, ``output_items`` being the first).
  input_items : Optional[Any]
    If set the ``total`` parameter value will be ``len(input_items)`` (``TypeError``s are silently caught).
    This can be useful if the total would not be automatically inferred from the ``output_items`` parameter.
  tqdm_config : Union[bool, tuple, list, dict]
    If set to a tuple or list, this will be appended to the ``tqdm_args``.
    If set to a dict, this will be merged into the ``tqdm_kwargs``.
  **tqdm_kwargs : dict
    Other keyword arguments for `tqdm.tqdm`.
  """
  # Set the default colour value from the toolkit config file (if set):
  toolkit_config = get_config_data()
  if ('colour' not in tqdm_kwargs) and ('tqdm' in toolkit_config) and ('colour' in toolkit_config['tqdm']):
    tqdm_kwargs['colour'] = str(toolkit_config['tqdm']['colour'])

  # Set the tqdm total based on the "input_items" if specified:
  if (input_items is not None) and ('total' not in tqdm_kwargs):
    try:
      tqdm_kwargs['total'] = len(input_items)
    except TypeError:
      pass

  # If a tqdm_config dict or tuple/list is passed, add that to the kwargs or args (respectively):
  if isinstance(tqdm_config, dict):
    tqdm_kwargs = {**tqdm_kwargs, **tqdm_config}
  elif isinstance(tqdm_config, (tuple, list)):
    tqdm_args = (*tqdm_args, *tqdm_config)

  # Call the actual tqdm function:
  return tqdm_module.tqdm(output_items, *tqdm_args, **tqdm_kwargs)
