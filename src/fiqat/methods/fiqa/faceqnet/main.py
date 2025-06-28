"""Implements a FIQA method using FaceQnet v0 and v1.

External file dependencies:
This uses models from the `FaceQnet repository <https://github.com/uam-biometrics/FaceQnet.git>`_.

- https://github.com/uam-biometrics/FaceQnet/releases/download/v0/FaceQnet.h5 and rename it to FaceQnet_v0.h5.
- https://github.com/uam-biometrics/FaceQnet/releases/download/v1.0/FaceQnet_v1.h5 (keep the filename).
"""

# Standard imports:
from typing import Iterable, TypedDict
from pathlib import Path

# External imports:
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Local imports:
# pylint: disable=relative-beyond-top-level
from .... import registry
from ....types import FiqaMethodRegistryEntry, MethodId
from ....types import InputImage, ImageChannelType, NpImage
from ....types import DeviceConfig, QualityScore, QualityScoreRange
from ....iterate import iterate_as_batches
from ....image import load_input_image
from ....config import get_config_data


class FaceqnetConfig(TypedDict):
  device_config: DeviceConfig
  batch_size: int
  model_type: str
  """Either 'FaceQnet-v0' or 'FaceQnet-v1'."""


def _load_image(input_image: InputImage) -> NpImage:
  image = load_input_image(input_image, ImageChannelType.BGR).image
  image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
  return image


def _get_dependency_path(model_type: str) -> Path:
  toolkit_config = get_config_data()
  if ('faceqnet' in toolkit_config) and ('models' in toolkit_config['faceqnet']):
    dependency_path = Path(toolkit_config['faceqnet']['models'])
  else:
    dependency_path = Path(toolkit_config['models']) / 'faceqnet'
  if model_type == 'FaceQnet-v0':
    model_path = dependency_path / 'FaceQnet_v0.h5'
  elif model_type == 'FaceQnet-v1':
    model_path = dependency_path / 'FaceQnet_v1.h5'
  return model_path


def _clamp_normalized_scores(scores: list[QualityScore]) -> list[QualityScore]:
  return [max(0, min(1, float(score))) for score in scores]


def _process(
    _method_registry_entry: FiqaMethodRegistryEntry,
    config: FaceqnetConfig,
    input_images: Iterable[InputImage],
) -> Iterable[QualityScore]:
  model_path = _get_dependency_path(config['model_type'])
  model = load_model(model_path)
  for input_image_batch in iterate_as_batches(input_images, config.get('batch_size', 1)):
    images = [_load_image(input_image) for input_image in input_image_batch]
    batch_size = len(images)
    image_array = np.array(images, copy=False, dtype=np.float32)
    quality_scores = model.predict(image_array, batch_size=batch_size, verbose=0)
    quality_scores = _clamp_normalized_scores(quality_scores)
    for quality_score in quality_scores:
      yield quality_score


method_registry_entry = FiqaMethodRegistryEntry(
    method_id=MethodId('fiqa/faceqnet'),
    process=_process,
    default_config=FaceqnetConfig(device_config=DeviceConfig('cpu', 0),),
    quality_score_range=QualityScoreRange(0, 1),
)

registry.register_method(method_registry_entry)
