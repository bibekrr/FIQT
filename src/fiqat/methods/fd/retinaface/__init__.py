"""See main.py for the implementation."""

from .... import registry  # pylint: disable=relative-beyond-top-level


def _loader():
  try:
    from . import main  # pylint: disable=unused-import,import-outside-toplevel
    return registry.MethodRegistryStatus.AVAILABLE, None
  except Exception as error:  # pylint: disable=broad-exception-caught
    return registry.MethodRegistryStatus.UNAVAILABLE, error


registry.register_method_loader('fd/retinaface', _loader)
