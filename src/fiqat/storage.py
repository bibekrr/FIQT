"""Utility functionality to store and load computed experiment data."""

# Standard imports:
from typing import Iterable, Any, Optional
from abc import ABC, abstractmethod
from pathlib import Path
import sqlite3
import io
import time

# Local imports:
from . import serialize


class StorageBase(ABC):
  """Abstract base class for the data storage utility functionality."""

  @abstractmethod
  def open(self, path: Path):
    """Initializes the storage class.
    This method has to be called before any of the other methods may be used.

    Parameters
    ----------
    path : Path
      The path at which the data is or will be stored.
    """
    pass

  @abstractmethod
  def load_items(self, item_type_id: str) -> Iterable[Any]:
    """Iteratively load all items for the given type.

    Parameters
    ----------
    item_type_id : str
      The type of the items that should be loaded.

    Returns
    -------
    Iterable[Any]
      The loaded items.
    """
    pass

  @abstractmethod
  def update_item(self, item_type_id: str, item_key: str, item: Any):
    """Add or overwrite an item in the storage.
    Note that the storage implementation may not immediately write the update to disk - see the :meth:`save` method.

    Parameters
    ----------
    item_type_id : str
      The type of the item.
    item_key : str
      The identifier key of the item.
    item : Any
      The item data itself.
    """
    pass

  @abstractmethod
  def save(self):
    """Save any pending updates to disk.

    An implementation may also write to disk immediately in the :meth:`update_item`,
    in which case this method may do nothing.
    """
    pass

  @abstractmethod
  def close(self):
    """Calls :meth:`save` and then closes any open file handles.
    The :meth:`open` method has to be called again before any of the other methods may be used.
    """
    pass


class StorageSqlite(StorageBase):
  """A :class:`StorageBase` implementation that stores data in a
  `sqlite3 <https://docs.python.org/3/library/sqlite3.html>`_ database.

  The stored items are serialized using the :mod:`fiqat.serialize` module.

  Parameters
  ----------
  save_on_update_after_sec : float
    If set to a duration in seconds >= 0,
    then :meth:`save` will be called at the end of a :meth:`update_item` call,
    when the elapsed time since the last :meth:`save` call is greater than or equal to this setting.
    I.e. :meth:`save` is always called in :meth:`update_item` if this is 0.
  """

  __slots__ = ['con', 'last_save_time', 'save_on_update_after_sec']

  def __init__(self, save_on_update_after_sec: float = 60) -> None:
    super().__init__()
    self.con = None
    """The ``sqlite3`` database connection obtained by ``sqlite3.connect`` in the :meth:`open` method."""
    self.last_save_time = None
    """The time of the last :meth:`save` call (using the value from ``time.monotonic()``)."""
    self.save_on_update_after_sec = save_on_update_after_sec
    """See the corresponding :class:`StorageSqlite` constructor parameter description."""

  def open(self, path: Path):
    """Initializes the storage class by opening the SQLite database (``sqlite3.connect``).
    This method has to be called before any of the other methods may be used.

    Parameters
    ----------
    path : Path
      The path at which the SQLite database is or will be stored (e.g. with the suffix '.sqlite3').
    """
    assert self.con is None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    self.con = sqlite3.connect(path)

  def _assert_con_open(self):
    assert self.con is not None, 'Need to call "open" first'

  def _does_table_exist(self, table_id: str):
    return (self.con.execute('SELECT name FROM sqlite_schema WHERE type="table" AND name=?', (table_id,)).fetchone()
            is not None)

  def _init_table(self, table_id: str):
    with self.con:
      # NOTE Placeholders apparently can't be used for table (or column) names, hence string interpolation.
      self.con.execute(f'CREATE TABLE IF NOT EXISTS {table_id} (key TEXT PRIMARY KEY, value BLOB)')

  def count_items(self, item_type_id: str) -> int:
    """Counts the number of items for a given item type ID.

    Parameters
    ----------
    item_type_id : str
      The type of the items that should be counted.

    Returns
    -------
    int
      The number of the stored items.
    """
    self._assert_con_open()
    if self._does_table_exist(item_type_id):
      cursor = self.con.execute(f'SELECT COUNT(*) FROM {item_type_id}')
      return cursor.fetchone()[0]
    else:
      return 0

  def load_items(self, item_type_id: str) -> Iterable[Any]:
    self._assert_con_open()
    if self._does_table_exist(item_type_id):
      self.con.row_factory = sqlite3.Row
      # NOTE Placeholders apparently can't be used for table (or column) names, hence string interpolation.
      for row in self.con.execute(f'SELECT * FROM {item_type_id}'):
        key = row['key']
        value = row['value']
        value = io.BytesIO(value)
        value = serialize.unpack(value)
        yield key, value

  def load_item_keys(self, item_type_id: str) -> Iterable[str]:
    """Like :meth:`load_items`, but only loads the item keys.
    If only the keys are needed, this can be faster since the item values aren't deserialized.

    Parameters
    ----------
    item_type_id : str
      The type of the items for which the keys should be loaded.

    Returns
    -------
    Iterable[str]
      The loaded item keys.
    """
    self._assert_con_open()
    if self._does_table_exist(item_type_id):
      self.con.row_factory = sqlite3.Row
      # NOTE Placeholders apparently can't be used for table (or column) names, hence string interpolation.
      for row in self.con.execute(f'SELECT key FROM {item_type_id}'):
        key = row['key']
        yield key

  def load_item(self, item_type_id: str, item_key: str) -> Optional[Any]:
    """Loads a single item with a specific key.

    Parameters
    ----------
    item_type_id : str
      The type of the item
    item_key : str
      The identifier key of the item.

    Returns
    -------
    Optional[Any]
      The loaded item or ``None`` if no entry exists.
    """
    self._assert_con_open()
    if self._does_table_exist(item_type_id):
      self.con.row_factory = sqlite3.Row
      for row in self.con.execute(f'SELECT value FROM {item_type_id} WHERE key=?', (item_key,)):
        value = row['value']
        value = io.BytesIO(value)
        value = serialize.unpack(value)
        return value
    return None

  def _auto_save(self):
    if self.save_on_update_after_sec >= 0:
      current_time = time.monotonic()
      if self.last_save_time is None:
        self.last_save_time = current_time
      if (current_time - self.last_save_time) >= self.save_on_update_after_sec:
        self.save()

  def _insert_or_replace(self, item_type_id: str, item_key: str, buffer: memoryview):
    # NOTE Placeholders apparently can't be used for table (or column) names, hence string interpolation.
    self.con.execute(f'INSERT OR REPLACE INTO {item_type_id} (key, value) VALUES (:key, :value)', {
        'key': item_key,
        'value': buffer,
    })

  def update_item(self, item_type_id: str, item_key: str, item: Any):
    """Add or overwrite an item in the storage.

    The :meth:`save` method may automatically be called at the end;
    see the ``save_on_update_after_sec`` constructor setting of :class:`StorageSqlite`.

    Parameters
    ----------
    item_type_id : str
      The type of the item.
      This is used as the table name in the SQLite database.
      The table will be created if doesn't exist yet.
      Each table has a "key" (string content, i.e. SQLite TEXT) and a "value" (binary content, i.e. SQLite BLOB) column.
    item_key : str
      The identifier key of the item.
      This corresponds to the "key" column of the SQLite table.
    item : Any
      The item data itself.
      It is serialized via :meth:`fiqat.serialize.pack` and stored in the "value" column of the SQLite table.
    """
    self._assert_con_open()
    value = io.BytesIO()
    serialize.pack(value, item)
    buffer = value.getbuffer()
    try:
      self._insert_or_replace(item_type_id, item_key, buffer)
    except sqlite3.OperationalError as error:
      if str(error).startswith('no such table: '):
        self._init_table(item_type_id)
        self._insert_or_replace(item_type_id, item_key, buffer)
      else:
        raise error
    self._auto_save()

  def save(self):
    """Commits any pending updates to the database."""
    self.last_save_time = time.monotonic()
    self._assert_con_open()
    self.con.commit()

  def close(self):
    """Calls :meth:`save` and then closes the connection to the database.
    The :meth:`open` method will have to be called again before any of the other methods can be used again.
    """
    self._assert_con_open()
    self.save()
    self.con.close()
    self.con = None
