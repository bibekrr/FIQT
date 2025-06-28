"""Implements face recognition methods using ArcFace.

External file dependencies:
This uses a model from the `InsightFace repository <https://github.com/deepinsight/insightface.git>`_.

-  https://github.com/deepinsight/insightface/wiki/Model-Zoo/6633390634bcf907c383cc6c90b62b6700df2a8e#31-lresnet100e-irarcfacems1m-refine-v2
   placed e.g. in .../insightface-60bb5829b1/models (i.e. model-symbol.json & model-0000.params).

   sha256sum:
   6e152989cbbcc8f254fcbbca5b99810f0d77ec8de647ece124ff01597ea91179  log
   931257c0b7174254fd81314706f2591cc6d1dd7299275bb8cf01c774ed0da8be  model-0000.params
   5074cee9924f56566f5735ef2bcf5dd50e72ef53938c50af328d5b0c7df3d80c  model-symbol.json
"""

# Standard imports:
from typing import Iterable, TypedDict
from pathlib import Path

# External imports:
import numpy as np
import cv2
import mxnet
from sklearn.preprocessing import normalize

# Local imports:
# pylint: disable=relative-beyond-top-level
from .... import registry
from ....types import MethodId, MethodRegistryEntry, CscMethodRegistryEntry
from ....types import InputImage, ImageChannelType, NpImage
from ....types import DeviceConfig, FeatureVector
from ....types import FeatureVectorPair, ComparisonScoreType, ComparisonScore, ComparisonScoreRange
from ....iterate import iterate_as_batches
from ....image import load_input_image
from ....internal import get_config_path_entry
from .insight_face import InsightFace


class ArcfaceConfig(TypedDict):
  device_config: DeviceConfig
  batch_size: int


def _get_dependency_path() -> Path:
  insightface_path = get_config_path_entry('arcface/insightface_path', 'insightface-60bb5829b1')
  return insightface_path


def _device_config_to_mxnet(device_config: DeviceConfig) -> mxnet.Context:
  device_config = DeviceConfig(*device_config)
  if device_config.type == 'cpu':
    return mxnet.cpu(device_config.index)
  elif device_config.type == 'gpu':
    return mxnet.gpu(device_config.index)
  else:
    raise RuntimeError('Invalid device_config', device_config)


def _load_model(device_config: DeviceConfig) -> InsightFace:
  insightface_path = _get_dependency_path()
  context = _device_config_to_mxnet(device_config)
  model = InsightFace(insightface_path=insightface_path, context=context)
  return model


def _load_image(input_image: InputImage) -> NpImage:
  image = load_input_image(input_image, ImageChannelType.RGB).image
  image = cv2.resize(image, (112, 112), interpolation=cv2.INTER_LINEAR)
  image = np.transpose(image, (2, 0, 1))
  return image


def _fr_process(
    _method_registry_entry: MethodRegistryEntry,
    config: ArcfaceConfig,
    input_images: Iterable[InputImage],
) -> Iterable[FeatureVector]:
  model = _load_model(config['device_config'])
  for input_image_batch in iterate_as_batches(input_images, config.get('batch_size', 1)):
    images = [_load_image(input_image) for input_image in input_image_batch]
    feature_vectors = model.get_feature(images)
    for feature_vector in feature_vectors:
      yield feature_vector


fr_method_registry_entry = MethodRegistryEntry(
    method_id=MethodId('fr/arcface'),
    process=_fr_process,
    default_config=ArcfaceConfig(device_config=DeviceConfig('cpu', 0),),
)

registry.register_method(fr_method_registry_entry)


def _normalize_features(feature_vector: FeatureVector) -> FeatureVector:
  features = np.array(feature_vector)
  features = features.reshape(1, -1)
  features = normalize(features)
  features = features.flatten()
  return features


def _csc_process(
    _method_registry_entry: CscMethodRegistryEntry,
    _config: dict,
    feature_vector_pairs: Iterable[FeatureVectorPair],
) -> Iterable[ComparisonScore]:
  for feature_vector_pair in feature_vector_pairs:
    feature_vector_0 = _normalize_features(feature_vector_pair[0])
    feature_vector_1 = _normalize_features(feature_vector_pair[1])
    cosine_score = feature_vector_0 @ feature_vector_1.T
    cosine_score = float(cosine_score)
    assert cosine_score <= 1.000001, cosine_score
    assert cosine_score >= -1.000001, cosine_score
    cosine_score = max(-1, min(+1, cosine_score))
    yield cosine_score


csc_method_registry_entry = CscMethodRegistryEntry(
    method_id=MethodId('csc/arcface'),
    process=_csc_process,
    comparison_score_type=ComparisonScoreType.SIMILARITY,
    comparison_score_range=ComparisonScoreRange(-1, +1),
)

registry.register_method(csc_method_registry_entry)
