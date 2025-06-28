"""Implements a face detector method using MTCNN.

External file dependencies:
This uses the "insightface" repository.
$ git clone https://github.com/deepinsight/insightface.git
$ git checkout 60bb5829b1d76bfcec7930ce61c41dde26413279
"""

# Standard imports:
import sys
from pathlib import Path
from typing import Iterable, TypedDict, Optional
from collections import namedtuple

# External imports:
import cv2
import mxnet

# Local imports:
# pylint: disable=relative-beyond-top-level
from .... import registry
from ....types import FdMethodRegistryEntry, MethodId, InputImages, InputImage, FaceDetectorOutput, ImageSize
from ....types import DeviceConfig, DetectedFace, FaceRoi, FacialLandmark, ImageChannelType
from ....iterate import iterate_as_batches
from ....image import load_input_image
from ....internal import get_config_path_entry


class MtcnnConfig(TypedDict):
  resize_to: Optional[ImageSize]
  device_config: DeviceConfig


IntermediateData = namedtuple('IntermediateData', ['adjusted_image', 'scale_tuple', 'image'])


def _get_dependency_paths() -> tuple[Path, Path]:
  insightface_path = get_config_path_entry('mtcnn/insightface_path', 'insightface-60bb5829b1')
  deploy_path = insightface_path / 'deploy'
  mtcnn_path = deploy_path / 'mtcnn-model'
  return deploy_path, mtcnn_path


def _extend_sys_path(retinaface_path: Path) -> int:
  sys.path.append(str(retinaface_path))
  return len(sys.path) - 1


def _restore_sys_path(retinaface_path: Path, pop_index: int):
  retinaface_path_str = str(retinaface_path)
  if sys.path[pop_index] == retinaface_path_str:
    sys.path.pop(pop_index)
  else:
    try:
      pop_index = list(reversed(sys.path)).index(retinaface_path_str)
      sys.path.pop(pop_index)
    except ValueError:
      pass


def _device_config_to_mxnet(device_config: DeviceConfig) -> mxnet.Context:
  device_config = DeviceConfig(*device_config)
  if device_config.type == 'cpu':
    return mxnet.cpu(device_config.index)
  elif device_config.type == 'gpu':
    return mxnet.gpu(device_config.index)
  else:
    raise RuntimeError('Invalid device_config', device_config)


def _load_model(config: MtcnnConfig):
  deploy_path, mtcnn_path = _get_dependency_paths()
  sys_path_index = _extend_sys_path(deploy_path)

  from mtcnn_detector import MtcnnDetector  # pylint: disable=import-outside-toplevel, import-error
  mtcnn_model = MtcnnDetector(
      model_folder=mtcnn_path,
      minsize=config['minsize'],
      threshold=config['thresholds'],
      factor=config['factor'],
      num_worker=config['num_worker'],
      accurate_landmark=config['accurate_landmark'],
      ctx=_device_config_to_mxnet(config['device_config']),
  )

  _restore_sys_path(deploy_path, sys_path_index)
  return mtcnn_model


def _load_image(input_image: InputImage, resize_to: Optional[ImageSize]) -> IntermediateData:
  image = load_input_image(input_image, ImageChannelType.BGR).image
  adjusted_image = image if resize_to is None else cv2.resize(image, resize_to, interpolation=cv2.INTER_LINEAR)
  scale_tuple = (image.shape[1] / adjusted_image.shape[1], image.shape[0] / adjusted_image.shape[0])
  return IntermediateData(adjusted_image, scale_tuple, image)


def _load_images(input_image_batch: InputImages, resize_to: ImageSize) -> list[IntermediateData]:
  images = [_load_image(input_image, resize_to) for input_image in input_image_batch]
  return images


def _run_model_for_batch(mtcnn_model, model_batch_input: list[IntermediateData], config: MtcnnConfig) -> list:
  # If det_type is not 0, mtcnn_model.detect_face skips the first MTCNN stage.
  det_type = 0 if config['skip_stage_1'] else 1
  model_batch_output = [
      # NOTE This implementation currently doesn't support real batched execution of the model.
      mtcnn_model.detect_face(intermediate.adjusted_image, det_type=det_type) for intermediate in model_batch_input
  ]
  return model_batch_output


def _produce_face_detector_output(intermediate: IntermediateData, model_batch_output) -> FaceDetectorOutput:
  detected_faces = []
  input_image_size = ImageSize(intermediate.image.shape[1], intermediate.image.shape[0])
  face_detector_output = FaceDetectorOutput(input_image_size=input_image_size, detected_faces=detected_faces)

  detect_data = model_batch_output.pop(0)
  if detect_data is not None:
    for total_box, points in zip(detect_data[0], detect_data[1]):
      box = total_box[:4].tolist()
      box[0] *= intermediate.scale_tuple[0]
      box[1] *= intermediate.scale_tuple[1]
      box[2] *= intermediate.scale_tuple[0]
      box[3] *= intermediate.scale_tuple[1]
      roi = FaceRoi(x=box[0], y=box[1], width=box[2] - box[0], height=box[3] - box[1])

      confidence = float(total_box[4])

      landmarks = [
          FacialLandmark(landmark[0] * intermediate.scale_tuple[0], landmark[1] * intermediate.scale_tuple[1])
          for landmark in points.reshape((2, 5)).T
      ]

      detected_face = DetectedFace(
          roi=roi,
          landmarks=landmarks,
          confidence=confidence,
      )
      detected_faces.append(detected_face)

  return face_detector_output


def _process(
    _method_registry_entry: FdMethodRegistryEntry,
    config: MtcnnConfig,
    input_images: Iterable[InputImage],
) -> Iterable[FaceDetectorOutput]:
  config = {
      'minsize': 20,
      'thresholds': [0.6, 0.7, 0.8],
      'factor': 0.709,
      'num_worker': 1,
      'accurate_landmark': True,
      'skip_stage_1': True,
      **config,
  }
  mtcnn_model = _load_model(config)
  for input_image_batch in iterate_as_batches(input_images, config.get('batch_size', 1)):
    model_batch_input = _load_images(input_image_batch, config['resize_to'])
    model_batch_output = _run_model_for_batch(mtcnn_model, model_batch_input, config)
    for intermediate in model_batch_input:
      face_detector_output = _produce_face_detector_output(intermediate, model_batch_output)
      yield face_detector_output


method_registry_entry = FdMethodRegistryEntry(
    method_id=MethodId('fd/mtcnn'),
    process=_process,
    default_config=MtcnnConfig(
        resize_to=None,
        device_config=DeviceConfig('cpu', 0),
    ),
    landmark_names=[
        'Image-left eye',
        'Image-right eye',
        'Tip of the nose',
        'Image-left mouth corner',
        'Image-right mouth corner',
    ],
)

registry.register_method(method_registry_entry)
