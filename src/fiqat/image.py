"""Utility functionality to load and save image files."""

# Standard imports:
from typing import Union
from pathlib import Path
import bz2
import subprocess

# External imports:
import numpy as np
import cv2

# Local imports:
from .types import InputImage, PillowImage, TypedNpImage, NpImage, ImageChannelType
from .temp import get_temp_path


def _get_np_image_channel_type(image: Union[TypedNpImage, NpImage]) -> ImageChannelType:
  """For :class:`fiqat.types.TypedNpImage` this just returns the stored :class:`fiqat.types.ImageChannelType` value.
  For :class:`fiqat.types.NpImage`
  - :class:`fiqat.types.ImageChannelType.GRAY` is returned if the shape is of length 2 or if the shape at index 2 is 1
    (for one channel),
  - otherwise :class:`fiqat.types.ImageChannelType.BGR` is returned if the shape indicates three channels,
    simply assuming that the data is stored in the ``cv2`` default load order.
  """
  if isinstance(image, TypedNpImage):
    return image.channels
  elif isinstance(image, NpImage):
    assert (len(image.shape) == 3) or (len(image.shape) == 2), ('Unexpected image shape length', image.shape, image)
    if len(image.shape) < 3:
      return ImageChannelType.GRAY
    elif image.shape[2] == 1:
      return ImageChannelType.GRAY
    elif image.shape[2] == 3:
      # Expect BGR order as loaded by cv2 by default.
      return ImageChannelType.BGR
    else:
      raise Exception('Invalid image shape in terms of channel count', image.shape[2], image)  # pylint: disable=broad-exception-raised
  else:
    raise Exception('Invalid image type', type(image), image)  # pylint: disable=broad-exception-raised


def load_image_from_path(path: Path, channels: ImageChannelType) -> TypedNpImage:
  """Loads an image file from the given path,
  and converts the loaded image data to the given :class:`fiqat.types.ImageChannelType` if necessary.

  - Uses either ``cv2.imread`` or ``cv2.imdecode`` to load an image file.
  - Can load .bz2 compressed images (e.g. .ppm.bz2 files).
  - Can load .jp2 compressed images via ``matplotlib.image.imread``.
  - Can load .jxl (JPEG XL) compressed images if the `djxl` command-line tool is available,
    which currently first decodes the image file to a temporary file (using :meth:`fiqat.temp.get_temp_path`).

  Parameters
  ----------
  path : Path
    The path to the image file.
  channels : ImageChannelType
    The requested channel type for the returned image data.

  Returns
  -------
  TypedNpImage
    The loaded image data with the requested :class:`fiqat.types.ImageChannelType`.
  """
  path = Path(path)
  if path.suffix.lower() == '.jxl':
    _assert_djxl()
    with get_temp_path('djxl_output.png') as temp_path:  # TODO A different format may be faster.
      # The JPEG XL image first is converted to a temporary image file that can be loaded:
      subprocess.check_output(['djxl', str(path), str(temp_path), '--quiet'])
      image = _load_image_from_path(temp_path, channels)
    return image
  else:
    return _load_image_from_path(path, channels)


def _load_image_from_path(path: Path, channels: ImageChannelType) -> TypedNpImage:
  channels = ImageChannelType(channels)
  cv2_flags: int = cv2.IMREAD_GRAYSCALE if channels == ImageChannelType.GRAY else cv2.IMREAD_COLOR
  suffix = path.suffix.lower()
  if suffix == ".jp2":
    import matplotlib.image  # pylint: disable=import-outside-toplevel
    image = matplotlib.image.imread(path)
    loaded_color_channel_order = ImageChannelType.RGB
  elif suffix == ".bz2":
    with bz2.open(path) as bz2_file:
      buffer = bz2_file.read()
      data = np.frombuffer(buffer, dtype=np.uint8)
      image = cv2.imdecode(data, flags=cv2_flags)
      loaded_color_channel_order = ImageChannelType.BGR
  else:
    image = cv2.imread(str(path), flags=cv2_flags)
    loaded_color_channel_order = ImageChannelType.BGR
  if (channels != ImageChannelType.GRAY) and (loaded_color_channel_order != channels):
    image = cv2.cvtColor(
        image,
        # NOTE: COLOR_BGR2RGB / COLOR_RGB2BGR should be equivalent, so the check is superfluous.
        cv2.COLOR_BGR2RGB if loaded_color_channel_order == ImageChannelType.BGR else cv2.COLOR_RGB2BGR)
  return TypedNpImage(image, channels)


def _assert_input_np_image_dtype(image: NpImage):
  assert image.dtype == np.uint8, ('Invalid numpy data type for the input-image array', image.dtype, image)


def _convert_input_np_image(input_image: NpImage, from_type: ImageChannelType, to_type: ImageChannelType,
                            always_copy: bool) -> NpImage:
  _assert_input_np_image_dtype(input_image)
  if from_type == to_type:
    if always_copy:
      input_image = input_image.copy()
    return input_image
  from_is_color = from_type != ImageChannelType.GRAY
  to_is_color = to_type != ImageChannelType.GRAY
  if from_is_color != to_is_color:
    if to_is_color:
      image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR if to_type == ImageChannelType.BGR else cv2.COLOR_GRAY2RGB)
    else:
      image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY if from_type == ImageChannelType.BGR else cv2.COLOR_RGB2GRAY)
  else:
    image = cv2.cvtColor(
        input_image,
        # NOTE: COLOR_BGR2RGB / COLOR_RGB2BGR should be equivalent, so the check is superfluous.
        cv2.COLOR_BGR2RGB if from_type == ImageChannelType.BGR else cv2.COLOR_RGB2BGR)
  return image


def load_input_image(input_image: InputImage, channels: ImageChannelType, always_copy: bool = False) -> TypedNpImage:
  """Loads and/or converts the ``input_image`` as an 8bit-per-channel image with the given ``channels`` value.

  Parameters
  ----------
  input_image : InputImage
    If the ``input_image`` is a ``Path``, :meth:`load_image_from_path` will be called.
    Otherwise the ``input_image`` will be converted to fit the given ``channels`` value if necessary.
  channels : ImageChannelType
    The requested channel type for the returned image data.
  always_copy : bool
    If ``True``, a copy of the input image data will be returned even if no file loading or channel conversion was
    necessary.

  Returns
  -------
  TypedNpImage
    The image data with the requested :class:`fiqat.types.ImageChannelType`.
  """
  channels = ImageChannelType(channels)
  if isinstance(input_image, Path):
    path = input_image
    image = load_image_from_path(path, channels=channels).image
  elif isinstance(input_image, PillowImage):
    if channels == ImageChannelType.GRAY:
      input_image = input_image.convert("L")
    else:
      input_image = input_image.convert("RGB")
    image = np.asarray(input_image)
    if channels == ImageChannelType.BGR:
      image = _convert_input_np_image(image, ImageChannelType.RGB, ImageChannelType.BGR, always_copy)
  elif isinstance(input_image, TypedNpImage):
    image = _convert_input_np_image(input_image.image, input_image.channels, channels, always_copy)
  elif isinstance(input_image, NpImage):
    input_channels = _get_np_image_channel_type(input_image)
    image = _convert_input_np_image(input_image, input_channels, channels, always_copy)
  else:
    raise Exception('Invalid input_image type', type(input_image), input_image)  # pylint: disable=broad-exception-raised
  _assert_input_np_image_dtype(image)
  return TypedNpImage(image, channels)


def save_image(path: Path, input_image: InputImage):
  """Saves the ``input_image`` as a file to the given ``path``.

  Parameters
  ----------
  path : Path
    The path at which the image should be saved.

    The suffix determines the image format.
    Images are saved using ``cv2.imwrite``.

    Saving images as lossless .jxl (JPEG XL) files is also supported.
    This requires the `cjxl` command-line tool.
    Unless if ``input_image`` is a path to an image file supported by `cjxl`,
    a temporary file (using :meth:`fiqat.temp.get_temp_path`) will first be saved via ``cv2.imwrite``
    to serve as `cjxl` input for the actual file destination.
  input_image : :class:`fiqat.types.InputImage`
    The image that is to be saved.
    Besides image data, this can also be a path to an existing image file, e.g. to convert .png to .jxl images
    or vice versa.
  """
  path.parent.mkdir(parents=True, exist_ok=True)
  if path.suffix.lower() == '.jxl':
    _assert_cjxl()
    if isinstance(input_image, Path) and (input_image.suffix.lower()
                                          in {'.png', '.apng', '.gif', '.jpeg', '.jpg', '.ppm', '.pfm', '.pgx'}):
      # Image files supported by cjxl can be loaded by it without a temporary conversion step.
      _run_cjxl(input_image, path)
    else:
      # Otherwise a temporary image file is created first:
      with get_temp_path('cjxl_input.png') as temp_path:  # TODO A different format may be faster.
        _save_image(temp_path, input_image)
        _run_cjxl(temp_path, path)
  else:
    _save_image(path, input_image)


def _run_cjxl(input_path: Path, output_path: Path):
  subprocess.check_output(['cjxl', str(input_path), str(output_path), '--distance=0', '--quiet'])


def _save_image(path: Path, input_image: InputImage):
  image_data = load_input_image(input_image, ImageChannelType.BGR).image
  cv2.imwrite(str(path), image_data)


def _assert_cjxl():
  _assert_jxl('cjxl', 'encoding')


def _assert_djxl():
  _assert_jxl('djxl', 'decoding')


def _assert_jxl(tool: str, usage: str):
  try:
    output = subprocess.check_output([tool, '--version'], encoding='utf-8')
  except FileNotFoundError as error:
    raise RuntimeError(f'Could not find "{tool}". {tool} is needed for JPEG XL (.jxl) {usage} support.') from error
  if not output.startswith(f'{tool} v'):
    raise RuntimeError(
        f'Unexpected output for "{tool} --version". {tool} is needed for JPEG XL (.jxl) {usage} support.',
        ('output', output))
