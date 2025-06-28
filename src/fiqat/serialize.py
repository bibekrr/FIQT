"""This module defines a simple schema-less binary serialization format used by the
:class:`fiqat.storage.StorageSqlite` utility.
Using this serialization format is completely optional, the other parts of the toolkit do not require it.

Like JSON this format is schema-less, meaning that the object type and structure is part of the encoding.
Unlike JSON it is a "binary" format that e.g. directly encodes
integers in network byte order (i.e. big-endian), instead of formatting them as text.

This makes it conceptually similar to e.g. `MsgPack <https://msgpack.org/>`_.
Except in contrast to general formats, such as `MsgPack <https://msgpack.org/>`_,
this specialized format natively supports encoding of various common Python types,
as well as common ``numpy`` types
(e.g. `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_),
and some toolkit-specific types (:class:`fiqat.types.namedtuple_types` and :class:`fiqat.types.enum_types`).
See :class:`PackableItem` and :class:`PackType`.
It also supports the direct encoding of circular object references for the types `dict`, `list`, and `set`.
Additionally, each `np.ndarray` and each distinct string and path can likewise be stored only once,
further occurrences being encoded as references.
Path parts are optionally treated as separate strings too.

There are three reasons for why this format was written:

- Since Python's `pickle <https://docs.python.org/3/library/pickle.html>`_ serialization is not secure,
  a different serialization format should be used for any data that might be shared between users.
  This constrained format should be comparatively secure (i.e. it should not be possible to execute arbitrary code).
- Existing general schema-less formats such as JSON or `MsgPack <https://msgpack.org/>`_ still require additional
  specialization functions to handle (de)serialization of various Python types.
  This specialized format can support such types natively.
- Albeit perhaps less important, this custom format supports the automatic encoding of references to already encoded
  objects, including support for circular references. Other formats such as JSON or `MsgPack <https://msgpack.org/>`_
  do not support this directly.

Additionally the format may have these advantages:

- This binary format should usually be more compact than textual formats such as JSON,
  which may be beneficial when larger quantities of data are stored.
- The (de)serialization is simple and written in pure Python code. It consists only of a single, short file,
  which means that it should be fairly straightforward to check or customize it if necessary.

Possible disadvantages:

- There currently is no comprehensive automatic testing of this module.
- Some features reduce (de)serialization speed (e.g. string deduplication).
- The code for newer versions has become a bit more complicated due to various features.
- No other language implementations are planned.
"""

# Standard imports:
from typing import Union, BinaryIO
import struct
from enum import IntEnum
from pathlib import Path, PurePath, PosixPath, WindowsPath, PurePosixPath, PureWindowsPath
import io

# External imports:
import numpy as np

# Local imports:
from .types import namedtuple_types, enum_types

namedtuple_type_by_name = {namedtuple_type.__name__: namedtuple_type for namedtuple_type in namedtuple_types}
enum_type_by_name = {enum_type.__name__: enum_type for enum_type in enum_types}

PackableItem = Union[dict, list, str, int, float, np.ndarray, Path, PurePath]
"""The union of the (de)serializable types.

The :class:`fiqat.types.namedtuple_types` and :class:`fiqat.types.enum_types` are also (de)serializable.

For `int` values, note that the serialization format currently only supports values that still fit into an unsigned
or signed 64bit integer at most.
"""


class PackType(IntEnum):
  """The distinguished serialization type IDs, stored as 8bit unsigned integers."""
  DICT = 0
  """A dictionary."""
  LIST = 1
  """A list."""
  TUPLE = 2
  """A tuple."""
  STR = 3
  """A UTF-8 encoded string (``str``)."""
  FLOAT_32 = 4
  """A IEEE 754 binary32 floating-point number.

  This is currently never encoded by :meth:`pack`, but supported by :meth:`unpack`
  (decoded as a Python ``float``, i.e. converted to IEEE 754 binary64).
  """
  FLOAT_64 = 5
  """A Python ``float``, i.e. a IEEE 754 binary64 floating-point number."""
  INT_8 = 6
  """A Python ``int`` packed as an 8bit signed integer."""
  INT_16 = 7
  """A Python ``int`` packed as a 16bit signed integer."""
  INT_32 = 8
  """A Python ``int`` packed as a 32bit signed integer."""
  INT_64 = 9
  """A Python ``int`` packed as a 64bit signed integer."""
  UINT_8 = 10
  """A Python ``int`` packed as an 8bit unsigned integer."""
  UINT_16 = 11
  """A Python ``int`` packed as a 16bit unsigned integer."""
  UINT_32 = 12
  """A Python ``int`` packed as a 32bit unsigned integer."""
  UINT_64 = 13
  """A Python ``int`` packed as a 64bit unsigned integer."""
  NUMPY_ARRAY = 14
  """An ``np.ndarray``
  packed via ``np.save(..., allow_pickle=False)``
  and unpacked via ``np.load(..., allow_pickle=False)``.
  """
  NUMPY_FLOAT_32 = 15
  """An ``np.float32`` (IEEE 754 binary32 floating-point number)."""
  NUMPY_FLOAT_64 = 16
  """An ``np.float64`` (IEEE 754 binary64 floating-point number)."""
  NUMPY_INT_8 = 17
  """An ``np.int8`` (8bit signed integer)."""
  NUMPY_INT_16 = 18
  """An ``np.int16`` (16bit signed integer)."""
  NUMPY_INT_32 = 19
  """An ``np.int32`` (32bit signed integer)."""
  NUMPY_INT_64 = 20
  """An ``np.int64`` (64bit signed integer)."""
  NUMPY_UINT_8 = 21
  """An ``np.uint8`` (8bit unsigned integer)."""
  NUMPY_UINT_16 = 22
  """An ``np.uint16`` (16bit unsigned integer)."""
  NUMPY_UINT_32 = 23
  """An ``np.uint32`` (32bit unsigned integer)."""
  NUMPY_UINT_64 = 24
  """An ``np.uint64`` (64bit unsigned integer)."""
  POSIX_PATH = 25
  """A ``pathlib.PosixPath``.

  Beginning with format version 4:
  `Concrete path types <https://docs.python.org/3/library/pathlib.html#concrete-paths>`_ can only be instantiated on the
  matching system. :meth:`unpack` will load this as a ``pathlib.Path``,
  which depending on the system is either a ``pathlib.PosixPath`` or a ``pathlib.WindowsPath``.
  """
  WINDOWS_PATH = 26
  """A ``pathlib.WindowsPath``.

  Beginning with format version 4:
  `Concrete path types <https://docs.python.org/3/library/pathlib.html#concrete-paths>`_ can only be instantiated on the
  matching system. :meth:`unpack` will load this as a ``pathlib.Path``,
  which depending on the system is either a ``pathlib.PosixPath`` or a ``pathlib.WindowsPath``.
  """
  PURE_POSIX_PATH = 27
  """A ``pathlib.PurePosixPath``."""
  PURE_WINDOWS_PATH = 28
  """A ``pathlib.PureWindowsPath``."""
  NONE = 29
  """A ``None`` value."""
  FIQAT_NAMEDTUPLE = 30
  """One of the :class:`fiqat.types.namedtuple_types`. The types are packed using their UTF-8 encoded string name."""
  FIQAT_ENUM = 31
  """One of the :class:`fiqat.types.enum_types`. The types are packed using their UTF-8 encoded string name."""
  REFERENCE = 32
  """A reference to a previously encoded DICT, LIST, NUMPY_ARRAY, STR, or *_PATH.
  During encoding, the DICT, LIST, and NUMPY_ARRAY objects are tracked
  via their Python `id <https://docs.python.org/3/library/functions.html#id>`_.
  STR and *_PATH instances are tracked by their string value (i.e. each will be stored only once).
  During decoding, an already decoded object will be used here.
  The encoded reference is a 64bit unsigned integer that represents the index of the object in order of encoding.
  """
  BYTES = 33
  """A Python ``bytes`` string."""
  SET = 34
  """A set.

  Added in format version 5.
  """


serializer_version = 5
"""The current version of the serialization format, encoded at the start as a 64bit unsigned integer."""


def _pack_value(file: BinaryIO, struct_fmt: str, item: PackableItem):
  file.write(struct.pack(struct_fmt, item))


def _pack_type(file: BinaryIO, pack_type: PackType):
  _pack_value(file, '!B', pack_type)


def _pack_type_value(file: BinaryIO, pack_type: PackType, struct_fmt: str, item: PackableItem):
  _pack_type(file, pack_type)
  _pack_value(file, struct_fmt, item)


def _pack_len(file: BinaryIO, item: PackableItem):
  assert len(item) <= ((2**64) - 1)
  _pack_value(file, '!Q', len(item))


def _pack_len_16(file: BinaryIO, item: PackableItem):
  assert len(item) <= ((2**16) - 1)
  _pack_value(file, '!H', len(item))


def _pack_reference_sub(file: BinaryIO, item_id: Union[int, str, Path, PurePath], reference_indices: dict) -> bool:
  reference_index = reference_indices.get(item_id, None)
  if reference_index is None:
    reference_indices[item_id] = len(reference_indices)
    return False
  else:
    _pack_type_value(file, PackType.REFERENCE, '!q', reference_index)
    return True


def _pack_reference_id(file: BinaryIO, item: Union[dict, list, set, np.ndarray], reference_indices: dict) -> bool:
  """For DICT, LIST, SET, and NUMPY_ARRAY."""
  return _pack_reference_sub(file, id(item), reference_indices)


def _pack_reference_str(file: BinaryIO, item: str, reference_indices: dict) -> bool:
  """For STR."""
  return _pack_reference_sub(file, item, reference_indices)


def _pack_reference_path(file: BinaryIO, item: Union[Path, PurePath], reference_indices: dict) -> bool:
  """For *_PATH."""
  return _pack_reference_sub(file, item, reference_indices)


class _Packer:
  __slots__ = ['file', 'reference_indices', 'paths_as_parts', 'str_deduplication']

  def __init__(self, file: BinaryIO, paths_as_parts: bool, str_deduplication: bool) -> None:
    self.file = file
    self.reference_indices = {}
    self.paths_as_parts = paths_as_parts
    self.str_deduplication = str_deduplication

  def pack(self, item):
    file = self.file
    if item is None:
      _pack_type(file, PackType.NONE)
    elif isinstance(item, enum_types):
      _pack_type(file, PackType.FIQAT_ENUM)
      self.pack(type(item).__name__)
      self.pack(item.value)
    elif isinstance(item, dict):
      if not _pack_reference_id(file, item, self.reference_indices):
        _pack_type(file, PackType.DICT)
        _pack_len(file, item)
        for key, value in item.items():
          self.pack(key)
          self.pack(value)
    elif isinstance(item, list):
      if not _pack_reference_id(file, item, self.reference_indices):
        self._pack_list_tuple_set(item, PackType.LIST)
    elif isinstance(item, set):
      if not _pack_reference_id(file, item, self.reference_indices):
        self._pack_list_tuple_set(item, PackType.SET)
    elif isinstance(item, tuple):
      if isinstance(item, namedtuple_types):
        _pack_type(file, PackType.FIQAT_NAMEDTUPLE)
        self.pack(type(item).__name__)
      self._pack_list_tuple_set(item, PackType.TUPLE)
    elif isinstance(item, str):
      self._pack_str(item)
    elif isinstance(item, bytes):
      _pack_type(file, PackType.BYTES)
      _pack_len(file, item)
      file.write(item)
    elif isinstance(item, PosixPath):
      self._pack_other_path(PackType.POSIX_PATH, item)
    elif isinstance(item, WindowsPath):
      self._pack_other_path(PackType.WINDOWS_PATH, item)
    elif isinstance(item, PurePosixPath):
      self._pack_pure_posix_path(item)  # i.e. PackType.PURE_POSIX_PATH
    elif isinstance(item, PureWindowsPath):
      self._pack_other_path(PackType.PURE_WINDOWS_PATH, item)
    elif isinstance(item, np.float32):
      _pack_type_value(file, PackType.NUMPY_FLOAT_32, '!f', item)
    elif isinstance(item, np.float64):
      _pack_type_value(file, PackType.NUMPY_FLOAT_64, '!d', item)
    elif isinstance(item, np.uint8):
      _pack_type_value(file, PackType.NUMPY_UINT_8, '!B', item)
    elif isinstance(item, np.uint16):
      _pack_type_value(file, PackType.NUMPY_UINT_16, '!H', item)
    elif isinstance(item, np.uint32):
      _pack_type_value(file, PackType.NUMPY_UINT_32, '!I', item)
    elif isinstance(item, np.uint64):
      _pack_type_value(file, PackType.NUMPY_UINT_64, '!Q', item)
    elif isinstance(item, np.int8):
      _pack_type_value(file, PackType.NUMPY_INT_8, '!b', item)
    elif isinstance(item, np.int16):
      _pack_type_value(file, PackType.NUMPY_INT_16, '!h', item)
    elif isinstance(item, np.int32):
      _pack_type_value(file, PackType.NUMPY_INT_32, '!i', item)
    elif isinstance(item, np.int64):
      _pack_type_value(file, PackType.NUMPY_INT_64, '!q', item)
    elif isinstance(item, float):
      _pack_type_value(file, PackType.FLOAT_64, '!d', item)
    elif isinstance(item, int):
      if item > 0:
        if item <= ((2**8) - 1):
          _pack_type_value(file, PackType.UINT_8, '!B', item)
        elif item <= ((2**16) - 1):
          _pack_type_value(file, PackType.UINT_16, '!H', item)
        elif item <= ((2**32) - 1):
          _pack_type_value(file, PackType.UINT_32, '!I', item)
        elif item <= ((2**64) - 1):
          _pack_type_value(file, PackType.UINT_64, '!Q', item)
        else:
          raise ValueError('Unsupported integer value', item)
      else:
        if item >= (-(2**8) // 2):
          _pack_type_value(file, PackType.INT_8, '!b', item)
        elif item >= (-(2**16) // 2):
          _pack_type_value(file, PackType.INT_16, '!h', item)
        elif item >= (-(2**32) // 2):
          _pack_type_value(file, PackType.INT_32, '!i', item)
        elif item >= (-(2**64) // 2):
          _pack_type_value(file, PackType.INT_64, '!q', item)
        else:
          raise ValueError('Unsupported integer value', item)
    elif isinstance(item, np.ndarray):
      if not _pack_reference_id(file, item, self.reference_indices):
        _pack_type(file, PackType.NUMPY_ARRAY)
        # NOTE A simpler "np.save(file, item, allow_pickle=False)" may work too,
        #      but a "sub-file" is used just to be sure.
        sub_file = io.BytesIO()
        np.save(sub_file, item, allow_pickle=False)
        sub_buffer = sub_file.getbuffer()
        _pack_len(file, sub_buffer)
        file.write(sub_buffer)
    else:
      raise TypeError('Unsupported item type', type(item), item)

  def _pack_list_tuple_set(self, item: Union[list, tuple, set], pack_type: PackType):
    file = self.file
    _pack_type(file, pack_type)
    _pack_len(file, item)
    for sub_item in item:
      self.pack(sub_item)

  def _pack_str(self, item: str):
    file = self.file
    if not (self.str_deduplication and _pack_reference_str(file, item, self.reference_indices)):
      _pack_type(file, PackType.STR)
      encoded = item.encode('utf-8')
      _pack_len(file, encoded)
      file.write(encoded)

  def _pack_pure_posix_path(self, item: PurePosixPath):
    """Only for PURE_POSIX_PATH."""
    file = self.file
    if self.paths_as_parts:
      if not _pack_reference_path(file, item, self.reference_indices):
        _pack_type(file, PackType.PURE_POSIX_PATH)
        parts = item.parts
        _pack_len_16(file, parts)
        for part in parts:
          self._pack_str(part)
    else:
      _pack_type(file, PackType.PURE_POSIX_PATH)
      self._pack_str(str(item))

  def _pack_other_path(self, pack_type: PackType, item: Union[Path, PurePath]):
    """For POSIX_PATH, WINDOWS_PATH, and PURE_WINDOWS_PATH."""
    _pack_type(self.file, pack_type)
    if self.paths_as_parts:
      self._pack_pure_posix_path(PurePosixPath(item))
    else:
      self._pack_str(str(PurePosixPath(item)))


def pack(file: BinaryIO, item: PackableItem, paths_as_parts: bool = False):
  """Serializes the ``item`` to the ``file``.

  Parameters
  ----------
  file : BinaryIO
    The destination for the serialized data.
  item : PackableItem
    The object that is to be serialized.
  paths_as_parts : bool
    If ``True``, the parts of ``pathlib`` paths will be serialized as individual strings,
    which allows for deduplication for these string parts.
    This can result in smaller serialized data, but can also affect the serialization & deserialization speed.
    If ``False``, one path is serialized as one string.
  """
  _pack_value(file, '!Q', serializer_version)  # Write the format version.
  str_deduplication = not isinstance(item, (str, Path, PurePath))
  serialization_settings = 0
  if paths_as_parts:
    serialization_settings |= 0b01
  if str_deduplication:
    serialization_settings |= 0b10
  _pack_value(file, '!B', serialization_settings)  # Write the serialization settings.
  packer = _Packer(file, paths_as_parts=paths_as_parts, str_deduplication=str_deduplication)
  packer.pack(item)


def _unpack_type(file: BinaryIO) -> PackType:
  value = struct.unpack('!B', file.read(1))[0]
  return PackType(value)  # NOTE Will raise a ValueError if the value doesn't map to a PackType.


def _unpack_value(file: BinaryIO, size: int, struct_fmt: str) -> PackableItem:
  return struct.unpack(struct_fmt, file.read(size))[0]


def _unpack_len(file: BinaryIO) -> int:
  return _unpack_value(file, 8, '!Q')


def _unpack_len_16(file: BinaryIO) -> int:
  return _unpack_value(file, 2, '!H')


def _unpack_str(file: BinaryIO) -> str:
  length = _unpack_len(file)
  value = file.read(length)
  return value.decode('utf-8')


class _Unpacker:
  __slots__ = ['file', 'reference_items', 'version', 'paths_as_parts', 'str_deduplication']

  def __init__(self, file: BinaryIO, version: int, paths_as_parts: bool, str_deduplication: bool) -> None:
    self.file = file
    self.reference_items = []
    self.version = version
    self.paths_as_parts = paths_as_parts
    self.str_deduplication = str_deduplication
    if version < 3:
      assert not paths_as_parts, (version, paths_as_parts)
      assert not str_deduplication, (version, str_deduplication)

  def unpack(self) -> PackableItem:
    file = self.file
    pack_type = _unpack_type(file)
    if pack_type == PackType.NONE:
      return None
    elif pack_type == PackType.DICT:
      length = _unpack_len(file)
      item = {}
      self.reference_items.append(item)
      for _ in range(length):
        key = self.unpack()
        value = self.unpack()
        item[key] = value
      return item
    elif pack_type == PackType.LIST:
      length = _unpack_len(file)
      item = []
      self.reference_items.append(item)
      for _ in range(length):
        item.append(self.unpack())
      return item
    elif pack_type == PackType.SET:
      length = _unpack_len(file)
      item = set()
      self.reference_items.append(item)
      for _ in range(length):
        item.add(self.unpack())
      if length != len(item):
        raise ValueError(
            'fiqat.serialize set decoding length mismatch (encoding error?)',
            ('expected length', length),
            ('len(set_item)', len(item)),
            ('set_item', item),
        )
      return item
    elif pack_type == PackType.TUPLE:
      length = _unpack_len(file)
      return tuple((self.unpack() for _ in range(length)))
    elif pack_type == PackType.STR:
      value = _unpack_str(file)
      if self.str_deduplication:
        self.reference_items.append(value)
      return value
    elif pack_type == PackType.BYTES:
      length = _unpack_len(file)
      return file.read(length)
    elif pack_type == PackType.POSIX_PATH:
      return self._unpack_path(Path)
    elif pack_type == PackType.WINDOWS_PATH:
      return self._unpack_path(Path)
    elif pack_type == PackType.PURE_POSIX_PATH:
      return self._unpack_path(PurePosixPath)
    elif pack_type == PackType.PURE_WINDOWS_PATH:
      return self._unpack_path(PureWindowsPath)
    elif pack_type == PackType.FLOAT_32:
      return _unpack_value(file, 4, '!f')
    elif pack_type == PackType.FLOAT_64:
      return _unpack_value(file, 8, '!d')
    elif pack_type == PackType.INT_8:
      return _unpack_value(file, 1, '!b')
    elif pack_type == PackType.INT_16:
      return _unpack_value(file, 2, '!h')
    elif pack_type == PackType.INT_32:
      return _unpack_value(file, 4, '!i')
    elif pack_type == PackType.INT_64:
      return _unpack_value(file, 8, '!q')
    elif pack_type == PackType.UINT_8:
      return _unpack_value(file, 1, '!B')
    elif pack_type == PackType.UINT_16:
      return _unpack_value(file, 2, '!H')
    elif pack_type == PackType.UINT_32:
      return _unpack_value(file, 4, '!I')
    elif pack_type == PackType.UINT_64:
      return _unpack_value(file, 8, '!Q')
    elif pack_type == PackType.NUMPY_ARRAY:
      # NOTE A simpler "return np.load(file, allow_pickle=False)" may work too,
      #      but a "sub-file" is used just to be sure.
      length = _unpack_len(file)
      sub_file = io.BytesIO(file.read(length))
      item = np.load(sub_file, allow_pickle=False)
      self.reference_items.append(item)
      return item
    elif pack_type == PackType.NUMPY_FLOAT_32:
      return np.float32(_unpack_value(file, 4, '!f'))
    elif pack_type == PackType.NUMPY_FLOAT_64:
      return np.float64(_unpack_value(file, 8, '!d'))
    elif pack_type == PackType.NUMPY_INT_8:
      return np.int8(_unpack_value(file, 1, '!b'))
    elif pack_type == PackType.NUMPY_INT_16:
      return np.int16(_unpack_value(file, 2, '!h'))
    elif pack_type == PackType.NUMPY_INT_32:
      return np.int32(_unpack_value(file, 4, '!i'))
    elif pack_type == PackType.NUMPY_INT_64:
      return np.int64(_unpack_value(file, 8, '!q'))
    elif pack_type == PackType.NUMPY_UINT_8:
      return np.uint8(_unpack_value(file, 1, '!B'))
    elif pack_type == PackType.NUMPY_UINT_16:
      return np.uint16(_unpack_value(file, 2, '!H'))
    elif pack_type == PackType.NUMPY_UINT_32:
      return np.uint32(_unpack_value(file, 4, '!I'))
    elif pack_type == PackType.NUMPY_UINT_64:
      return np.uint64(_unpack_value(file, 8, '!Q'))
    elif pack_type == PackType.FIQAT_NAMEDTUPLE:
      namedtuple_name = self.unpack()
      namedtuple_type = namedtuple_type_by_name[namedtuple_name]
      namedtuple_data = self.unpack()
      return namedtuple_type(*namedtuple_data)
    elif pack_type == PackType.FIQAT_ENUM:
      enum_name = self.unpack()
      enum_type = enum_type_by_name[enum_name]
      enum_data = self.unpack()
      return enum_type(enum_data)
    elif pack_type == PackType.REFERENCE:
      reference_index = _unpack_value(file, 8, '!Q')
      reference_item = self.reference_items[reference_index]
      return reference_item
    else:
      # NOTE _unpack_type already checks the PackType by constructing it. I.e. this would be an implementation error.
      raise NotImplementedError('Implementation error - Invalid type value', pack_type)

  def _unpack_path_before_version_3(self, path_type) -> Union[Path, PurePath]:
    path_str = _unpack_str(self.file)
    return path_type(path_str)

  def _unpack_path_after_version_3(self, path_type) -> Union[Path, PurePath]:
    if self.paths_as_parts:
      reference_index = len(self.reference_items)
      self.reference_items.append(None)

      length = _unpack_len_16(self.file)
      parts = []
      for i in range(length):
        part = self.unpack()
        if not isinstance(part, str):
          raise ValueError('fiqat.serialize path decoding failed due to non-str part type', part, i, length, parts)
        parts.append(part)
      path = path_type('/'.join(parts))  # NOTE: '/'.join(parts) seems to be faster than *parts.

      self.reference_items[reference_index] = path
    else:
      path_str = self.unpack()
      if not isinstance(path_str, str):
        raise ValueError('fiqat.serialize path decoding failed due to non-str content', path_str)
      path = path_type(path_str)
    return path

  def _unpack_path(self, path_type) -> Union[Path, PurePath]:
    if self.version >= 4:
      if self.paths_as_parts:
        if path_type == PurePosixPath:
          return self._unpack_path_after_version_3(path_type)
        pure_posix_path = self.unpack()
        if not isinstance(pure_posix_path, PurePosixPath):
          raise ValueError('fiqat.serialize path decoding failed due to non-PurePosixPath type', pure_posix_path)
        return path_type(pure_posix_path)
      return self._unpack_path_after_version_3(path_type)
    elif self.version >= 3:
      return self._unpack_path_after_version_3(path_type)
    return self._unpack_path_before_version_3(path_type)


def unpack(file: BinaryIO) -> PackableItem:
  """Deserializes an object read from the ``file``.

  Parameters
  ----------
  file : BinaryIO
    The source of the serialized data.

  Returns
  -------
  PackableItem
    The deserialized object.
  """
  version = _unpack_value(file, 8, '!Q')  # Read the format version.
  assert version <= serializer_version, (version, serializer_version)
  if version >= 5:
    serialization_settings = _unpack_value(file, 1, '!B')  # Read the serialization settings.
    paths_as_parts = bool(serialization_settings & 0b01)
    str_deduplication = bool(serialization_settings & 0b10)
  elif version >= 4:
    serialization_settings = _unpack_value(file, 1, '!B')  # Read the serialization settings.
    paths_as_parts = bool(serialization_settings)
  else:
    paths_as_parts = version >= 3
    str_deduplication = version >= 3
  unpacker = _Unpacker(file, version, paths_as_parts=paths_as_parts, str_deduplication=str_deduplication)
  return unpacker.unpack()
