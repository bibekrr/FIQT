"""Implements a face detector method using RetinaFace.

The RetinaFace class code is customized to enable batch processing, see `custom_retinaface.py`.
(While the original code could also take image list input, it processed the images one by one.)

External file dependencies:
This uses the RetinaFace model and code from the "insightface" repository.
$ git clone https://github.com/deepinsight/insightface.git insightface-f89ecaaa54
$ cd insightface-f89ecaaa54
$ git checkout f89ecaaa547f12127165fc5b5aefca6d979b228a

Possibly required special installation step:
cd ./insightface-f89ecaaa54/detection/RetinaFace
pip install setuptools==63.4.1
pip install Cython
make
"""

# Standard imports:
import sys
from pathlib import Path
from typing import Iterable, TypedDict, NamedTuple, Optional

# External imports:
import cv2

# Local imports:
# pylint: disable=relative-beyond-top-level
from .... import registry
from ....types import FdMethodRegistryEntry, MethodId, InputImages, InputImage, FaceDetectorOutput, ImageSize
from ....types import DeviceConfig, DetectedFace, FaceRoi, FacialLandmark, NpImage, ImageChannelType
from ....iterate import iterate_as_batches
from ....image import load_input_image
from ....config import get_config_data


class RetinaFaceConfig(TypedDict):
  resize_to: Optional[ImageSize]
  device_config: DeviceConfig
  batch_size: int


class IntermediateData(NamedTuple):
  adjusted_image: NpImage
  scale_tuple: tuple[float, float]
  image: NpImage


def _get_dependency_paths() -> tuple[Path, Path]:
  toolkit_config = get_config_data()
  if ('retinaface' in toolkit_config) and ('insightface_path' in toolkit_config['retinaface']):
    insightface_path = toolkit_config['retinaface']['insightface_path']
  else:
    insightface_path = Path(toolkit_config['models']) / 'insightface-f89ecaaa54'
  insightface_path = Path(insightface_path)
  retinaface_path = insightface_path / 'detection/RetinaFace'
  retinaface_model_prefix = insightface_path / 'models/retinaface-R50/R50'
  return retinaface_path, retinaface_model_prefix


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


def _load_model(config: RetinaFaceConfig):
  retinaface_path, retinaface_model_prefix = _get_dependency_paths()
  sys_path_index = _extend_sys_path(retinaface_path)

  from .custom_retinaface import RetinaFace  # pylint: disable=import-outside-toplevel
  retinaface_model = RetinaFace(
      prefix=retinaface_model_prefix,
      epoch=0,
      device_config=config['device_config'],
      network="net3",  # Default
      nms=0.4,  # Default
      nocrop=False,  # Default
      decay4=0.5,  # Default
      vote=False,  # Default
  )

  _restore_sys_path(retinaface_path, sys_path_index)
  return retinaface_model


def _load_image(input_image: InputImage, resize_to: Optional[ImageSize]) -> IntermediateData:
  image = load_input_image(input_image, ImageChannelType.BGR).image
  adjusted_image = image if resize_to is None else cv2.resize(image, resize_to, interpolation=cv2.INTER_LINEAR)
  scale_tuple = (image.shape[1] / adjusted_image.shape[1], image.shape[0] / adjusted_image.shape[0])
  return IntermediateData(adjusted_image, scale_tuple, image)


def _load_images(input_image_batch: InputImages, resize_to: ImageSize) -> list[IntermediateData]:
  images = [_load_image(input_image, resize_to) for input_image in input_image_batch]
  return images


def _run_model_for_batch(retinaface_model, model_batch_input: list[IntermediateData]) -> list:
  model_batch_output = retinaface_model.detect(
      [intermediate.adjusted_image for intermediate in model_batch_input],
      threshold=0.8,
      scales=[1],
      do_flip=False,
  )
  model_batch_output = list(model_batch_output)
  return model_batch_output


def _produce_face_detector_output(intermediate: IntermediateData, model_batch_output) -> FaceDetectorOutput:
  detected_faces = []
  input_image_size = ImageSize(intermediate.image.shape[1], intermediate.image.shape[0])
  face_detector_output = FaceDetectorOutput(input_image_size=input_image_size, detected_faces=detected_faces)

  faces, landmarks_list = model_batch_output.pop(0)
  for face in faces:
    face[0] *= intermediate.scale_tuple[0]
    face[1] *= intermediate.scale_tuple[1]
    face[2] *= intermediate.scale_tuple[0]
    face[3] *= intermediate.scale_tuple[1]
  for landmarks in landmarks_list:
    for i in range(len(landmarks)):  # pylint: disable=consider-using-enumerate
      landmarks[i] *= intermediate.scale_tuple
  for face, landmarks in zip(faces, landmarks_list):
    box = face[:4].tolist()
    roi = FaceRoi(x=box[0], y=box[1], width=box[2] - box[0], height=box[3] - box[1])

    confidence = float(face[4])

    landmarks = [FacialLandmark(*landmark) for landmark in landmarks]

    detected_face = DetectedFace(
        roi=roi,
        landmarks=landmarks,
        confidence=confidence,
    )
    detected_faces.append(detected_face)

  return face_detector_output


def _process(
    _method_registry_entry: FdMethodRegistryEntry,
    config: RetinaFaceConfig,
    input_images: Iterable[InputImage],
) -> Iterable[FaceDetectorOutput]:
  retinaface_model = _load_model(config)
  for input_image_batch in iterate_as_batches(input_images, config.get('batch_size', 1)):
    model_batch_input = _load_images(input_image_batch, config['resize_to'])
    model_batch_output = _run_model_for_batch(retinaface_model, model_batch_input)
    for intermediate in model_batch_input:
      face_detector_output = _produce_face_detector_output(intermediate, model_batch_output)
      yield face_detector_output


method_registry_entry = FdMethodRegistryEntry(
    method_id=MethodId('fd/retinaface'),
    process=_process,
    default_config=RetinaFaceConfig(
        resize_to=ImageSize(250, 250),
        device_config=DeviceConfig('cpu', 0),
        batch_size=1,
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
