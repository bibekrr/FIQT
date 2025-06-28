"""Implements a FIQA method using CR-FIQA(L) or CR-FIQA(S).

The images are expected to be already preprocessed, i.e. cropped & aligned,
but will be resized if necessary using `cv2.resize` with default options.

External file dependencies:
The `pretrained models <https://github.com/fdbtrs/CR-FIQA#pretrained-model>`_.
"""

# Standard imports:
from typing import Iterable, TypedDict
from pathlib import Path

# External imports:
import numpy as np
import cv2

# Local imports:
# pylint: disable=relative-beyond-top-level
from .... import registry
from ....types import MethodRegistryEntry, MethodId
from ....types import InputImage, ImageChannelType, NpImage
from ....types import DeviceConfig, QualityScore
from ....iterate import iterate_as_batches
from ....image import load_input_image
from ....internal import get_config_path_entry
from .quality_model import QualityModel


class CrfiqaConfig(TypedDict):
  device_config: DeviceConfig
  batch_size: int
  model_type: str
  """Either 'CR-FIQA(S)' or 'CR-FIQA(L)'."""


def _load_image(input_image: InputImage) -> NpImage:
  image = load_input_image(input_image, ImageChannelType.RGB).image
  image = cv2.resize(image, (112, 112), interpolation=cv2.INTER_LINEAR)
  # Adjust the range:
  if image.dtype not in {np.float32, np.float64}:
    # Integer range [0,N] is converted to floating-point range [0,1]:
    iinfo = np.iinfo(image.dtype)
    assert iinfo.min == 0
    image = image / iinfo.max  # Output will have dtype np.float64
  # Assumed range [0,1] need to be converted to [-1,+1]:
  image = image.astype(np.float32)
  image = (image * 2) - 1
  # Transpose to separate the RGB channels:
  image = np.transpose(image, (2, 0, 1))
  return image


def _get_dependency_path(model_type: str) -> Path:
  dependency_path = get_config_path_entry('crfiqa/models', 'crfiqa')
  if model_type == 'CR-FIQA(S)':
    model_path = dependency_path / 'CR-FIQA(S)/32572backbone.pth'
  elif model_type == 'CR-FIQA(L)':
    model_path = dependency_path / 'CR-FIQA(L)/181952backbone.pth'
  return model_path


def _load_model(model_type: str, device_config: DeviceConfig) -> QualityModel:
  model_path = _get_dependency_path(model_type)
  if device_config.type == 'cpu':
    ctx = 'cpu'
  elif device_config.type == 'gpu':
    ctx = f'cuda:{device_config.index}'
  else:
    raise RuntimeError('Invalid device_config', device_config)
  if model_type == 'CR-FIQA(S)':
    backbone = 'iresnet50'
  elif model_type == 'CR-FIQA(L)':
    backbone = 'iresnet100'
  else:
    raise RuntimeError('Invalid model_type', model_type)
  return QualityModel(model_path, backbone, ctx)


def _process_image_batch(model: QualityModel, images: list[NpImage]) -> list[QualityScore]:
  quality_scores = model.get_batch_feature(
      images,
      return_features=False,
      return_quality_scores=True,
      quality_scores_as_list=True,
  )
  return quality_scores


def _process(
    _method_registry_entry: MethodRegistryEntry,
    config: CrfiqaConfig,
    input_images: Iterable[InputImage],
) -> Iterable[QualityScore]:
  model = _load_model(config['model_type'], config['device_config'])
  for input_image_batch in iterate_as_batches(input_images, config.get('batch_size', 1)):
    images = [_load_image(input_image) for input_image in input_image_batch]
    for quality_score in _process_image_batch(model, images):
      yield quality_score


method_registry_entry = MethodRegistryEntry(
    method_id=MethodId('fiqa/crfiqa'),
    process=_process,
    default_config=CrfiqaConfig(device_config=DeviceConfig('cpu', 0),),
)

registry.register_method(method_registry_entry)
