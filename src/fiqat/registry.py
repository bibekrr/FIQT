"""The toolkit's registry system that contains entries for the various available methods."""

# Standard imports:
from typing import Optional

# Local imports:
from .types import MethodId, MethodIdStr, MethodType
from .types import RegistryEntry, MethodRegistryLoader, MethodRegistryStatus, MethodRegistryEntry

registry_data = {}
"""The toolkit's method registry data.

Each key is a :class:`fiqat.types.MethodId` and each value is a :class:`fiqat.types.RegistryEntry`.

By default it will contain entries for the toolkit's internally provided methods.
Custom methods registered via :meth:`register_method` or :meth:`register_method_loader` are also added herein.
"""


def get_method_type_from_id(method_id: MethodId) -> MethodType:
  """Reads the :class:`fiqat.types.MethodType` from the given :class:`fiqat.types.MethodId`.

  Parameters
  ----------
  method_id : :class:`fiqat.types.MethodId`
    A valid :class:`fiqat.types.MethodId`, meaning that

    1. the :class:`fiqat.types.MethodId` must have two or more parts (e.g. the two parts "<method-type>/<method-name>"),
       and
    2. the first part, which specifies the :class:`fiqat.types.MethodType`,
       must be a valid :class:`fiqat.types.MethodType`.

  Returns
  -------
  MethodType
    The :class:`fiqat.types.MethodType` read from the given :class:`fiqat.types.MethodId`.
  """
  parts = method_id.parts
  assert len(parts) >= 2, ('The method ID path (', method_id, ') is too short.'
                           ' It needs to at least consist out of a MethodType'
                           ' followed by the concrete method name.'
                           ' I.e. "<method-type>/<method-name>".')
  method_type = MethodType(parts[0])
  return method_type


def _get_or_create_entry(method_id: MethodId) -> RegistryEntry:
  # The get_method_type_from_id checks whether the method_id is valid w.r.t. the structure and MethodType.
  get_method_type_from_id(method_id)
  if method_id not in registry_data:
    registry_data[method_id] = RegistryEntry(
        status=MethodRegistryStatus.UNKNOWN,
        loader=None,
        info=None,
        method_entry=None,
    )
  return registry_data[method_id]


def register_method_loader(method_id: MethodIdStr, loader: MethodRegistryLoader):
  """Register a method by specifying a :class:`fiqat.types.MethodRegistryLoader` function.
  
  This is meant for methods for which the dependencies aren't loaded yet,
  and for which the availability of those dependencies isn't known in advance,
  which is relevant for the toolkit's internally provided methods.
  Custom methods with already loaded (or known to be available) dependencies should usually be registered via
  :meth:`register_method` instead.

  The initial :class:`fiqat.types.RegistryEntry.status` will be :class:`fiqat.types.MethodRegistryStatus.UNKNOWN`.
  See :meth:`get_method` for the loading procedure that is executed when the method is requested.
  
  Parameters
  ----------
  method_id : :class:`fiqat.types.MethodIdStr`
    The ID of the new method.
    This function will assert that this ID hasn't been used to register a method previously.
  loader : :class:`fiqat.types.MethodRegistryLoader`
    The loader function used to initialize or check the method dependencies once the method is requested via
    :meth:`get_method`. See :meth:`get_method` for the loading procedure details.
  """
  method_id = MethodId(method_id)
  # The get_method_type_from_id checks whether the method_id is valid w.r.t. the structure and MethodType.
  assert method_id not in registry_data, (method_id, registry_data[method_id])
  entry = _get_or_create_entry(method_id)
  entry['loader'] = loader


def register_method(method_entry: MethodRegistryEntry):
  """Register a method that is known to be available for use
  by specifying the :class:`fiqat.types.MethodRegistryEntry` data.
  
  The initial :class:`fiqat.types.RegistryEntry.status` will be :class:`fiqat.types.MethodRegistryStatus.AVAILABLE`.

  Parameters
  ----------
  method_entry : :class:`fiqat.types.MethodRegistryEntry`
    The new method entry data.
    This function will assert that the :class:`fiqat.types.MethodRegistryEntry.method_id` hasn't been used to register
    a method previously.
  """
  method_id = method_entry['method_id']
  entry = _get_or_create_entry(method_id)
  entry['method_entry'] = method_entry
  entry['status'] = MethodRegistryStatus.AVAILABLE


def _load_method(method_id: MethodId):
  entry: RegistryEntry = registry_data[method_id]
  if entry['status'] == MethodRegistryStatus.UNKNOWN:
    loader = entry['loader']
    status, info = loader()
    entry['status'] = status
    entry['info'] = info


def get_method(method_id: MethodIdStr, check_method_type: Optional[MethodType] = None) -> MethodRegistryEntry:
  """Returns the :class:`fiqat.types.MethodRegistryEntry` for the requested :class:`fiqat.types.MethodId`.

  If the method has not yet been loaded
  (i.e. :class:`fiqat.types.RegistryEntry.status` is :class:`fiqat.types.MethodRegistryStatus.UNKNOWN`),
  then the :class:`fiqat.types.MethodRegistryLoader` function will be called.
  If it the loading was successful,
  the :class:`fiqat.types.RegistryEntry.status` will become :class:`fiqat.types.MethodRegistryStatus.AVAILABLE`
  and the :class:`fiqat.types.MethodRegistryEntry` is returned.
  Otherwise the :class:`fiqat.types.RegistryEntry.status` will become
  :class:`fiqat.types.MethodRegistryStatus.UNAVAILABLE` and a ``RuntimeError`` is raised.

  Note that if you want to get the :class:`fiqat.types.RegistryEntry` instead of the
  :class:`fiqat.types.MethodRegistryEntry`, you can simply inspect the :class:`registry_data` dictionary directly.

  Parameters
  ----------
  method_id : :class:`fiqat.types.MethodIdStr`
    The ID of the requested method.
  check_method_type : Optional[MethodType]
    If specified, the type readable from the ``method_id`` will be asserted using this explicitly set
    :class:`fiqat.types.MethodType`.

  Returns
  -------
  :class:`fiqat.types.MethodRegistryEntry`
    The entry for the :class:`fiqat.types.MethodRegistryStatus.AVAILABLE` method.
  
  Raises
  ------
  RuntimeError
    If the :class:`fiqat.types.RegistryEntry.status` is :class:`fiqat.types.MethodRegistryStatus.UNAVAILABLE`.
  """
  method_id = MethodId(method_id)
  method_type = get_method_type_from_id(method_id)
  if check_method_type is not None:
    assert method_type == check_method_type, ('Expected MethodType "', check_method_type,
                                              '" but received a method_id with type ', method_type, ': ', method_id)
  _load_method(method_id)
  entry: RegistryEntry = registry_data[method_id]
  if entry['status'] != MethodRegistryStatus.AVAILABLE:
    raise RuntimeError('Method is not available', method_id, entry)
  return entry['method_entry']
