"""This module contains utility functionality for iteration."""

# Standard imports:
from typing import Iterable


def iterate_as_batches(iterable: Iterable, batch_size: int) -> Iterable[Iterable]:
  """Iterate the input in batches.

  Parameters
  ----------
  iterable : `Iterable <https://docs.python.org/3/library/typing.html#typing.Iterable>`_
    The iterable that should be iterated through in batches.
  batch_size : int
    The length of each batch.
    The final batch may have a smaller length if the input ``iterable`` length isn't exactly divisible by this.

  Returns
  -------
  Iterable[Iterable]
    This function is a generator that yields batches obtained from the input ``iterable``.
    If the ``iterable`` supports getting the length (``len(iterable)``) and slicing (``iterable[i:j]``),
    each batch will be a slice. Otherwise each batch will be a list constructed by iterating over the ``iterable``.
    The order of the batches and items therein should remain identical to the input ``iterable``,
    unless if the slicing behavior differs (e.g. for some custom type).
  """
  try:
    length = len(iterable)
    for i in range(0, length, batch_size):
      yield iterable[i:min(i + batch_size, length)]
  except TypeError:
    next_batch = []
    for item in iterable:
      next_batch.append(item)
      if len(next_batch) == batch_size:
        yield next_batch
        next_batch = []
    if len(next_batch) > 0:
      yield next_batch
