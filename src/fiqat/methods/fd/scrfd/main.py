"""Implements a face detector method using SCRFD.

This is using the `insightface Python package <https://github.com/deepinsight/insightface/tree/master/python-package>`_.

External file dependencies:
By default this will use the 'buffalo_l' model,
which can be downloaded from <https://github.com/deepinsight/insightface/tree/master/python-package#model-zoo>.
The `insightface` package supports automatic downloads of this model as well.
"""

# Standard imports:
from typing import Iterable, TypedDict, NamedTuple, Optional

# External imports:
import cv2
import torch  # pylint: disable=unused-import # https://github.com/deepinsight/insightface/issues/2344
from insightface.app import FaceAnalysis

# Local imports:
# pylint: disable=relative-beyond-top-level
from .... import registry
from ....types import FdMethodRegistryEntry, MethodId, InputImages, InputImage, FaceDetectorOutput, ImageSize
from ....types import DeviceConfig, DetectedFace, FaceRoi, FacialLandmark, NpImage, ImageChannelType
from ....iterate import iterate_as_batches
from ....image import load_input_image
from ....internal import get_config_path_entry

################


class ScrfdConfig(TypedDict):
  resize_to: Optional[ImageSize]
  device_config: DeviceConfig
  model_name: str


class IntermediateData(NamedTuple):
  adjusted_image: NpImage
  scale_tuple: tuple[float, float]
  image: NpImage


def _load_model(config: ScrfdConfig):
  insightface_root = get_config_path_entry('insightface/root', 'insightface')
  model_name = config['model_name']
  device_config = config['device_config']
  model_app = FaceAnalysis(
      name=model_name,
      root=insightface_root,
      allowed_modules=['detection'],
      providers=['CPUExecutionProvider' if device_config.type == 'cpu' else 'CUDAExecutionProvider'],
  )
  model_app.prepare(
      # NOTE It seems that ctx_id only has an effect by setting the provider to CPUExecutionProvider if it is below 0.
      # <https://github.com/deepinsight/insightface/blob/a8746be394fcc14652822550ecde3658fb82714a/python-package/insightface/model_zoo/retinaface.py#L131>
      # (The RetinaFace class appears to be instantiated for all face detection models by
      # <https://github.com/deepinsight/insightface/blob/a8746be394fcc14652822550ecde3658fb82714a/python-package/insightface/model_zoo/model_zoo.py#L47>.)
      ctx_id=device_config.index,
      det_size=config['resize_to'],
  )

  return model_app


def _load_image(input_image: InputImage, resize_to: Optional[ImageSize]) -> IntermediateData:
  image = load_input_image(input_image, ImageChannelType.BGR).image
  adjusted_image = image if resize_to is None else cv2.resize(image, resize_to, interpolation=cv2.INTER_LINEAR)
  scale_tuple = (image.shape[1] / adjusted_image.shape[1], image.shape[0] / adjusted_image.shape[0])
  return IntermediateData(adjusted_image, scale_tuple, image)


def _load_images(input_image_batch: InputImages, resize_to: ImageSize) -> list[IntermediateData]:
  images = [_load_image(input_image, resize_to) for input_image in input_image_batch]
  return images


def _run_model_for_batch(model_app: FaceAnalysis, model_batch_input: list[IntermediateData]) -> list:
  model_batch_output = [
      # NOTE This implementation currently doesn't support real batched execution of the model.
      model_app.get(intermediate.adjusted_image) for intermediate in model_batch_input
  ]
  return model_batch_output


def _produce_face_detector_output(intermediate: IntermediateData, model_batch_output) -> FaceDetectorOutput:
  detected_faces = []
  input_image_size = ImageSize(intermediate.image.shape[1], intermediate.image.shape[0])
  face_detector_output = FaceDetectorOutput(input_image_size=input_image_size, detected_faces=detected_faces)

  detect_list = model_batch_output.pop(0)
  if detect_list is not None:
    for detect_item in detect_list:
      box = detect_item['bbox']
      box[0] *= intermediate.scale_tuple[0]
      box[1] *= intermediate.scale_tuple[1]
      box[2] *= intermediate.scale_tuple[0]
      box[3] *= intermediate.scale_tuple[1]
      roi = FaceRoi(x=box[0], y=box[1], width=box[2] - box[0], height=box[3] - box[1])

      confidence = detect_item['det_score']

      landmarks = [
          FacialLandmark(landmark[0] * intermediate.scale_tuple[0], landmark[1] * intermediate.scale_tuple[1])
          for landmark in detect_item['kps']
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
    config: ScrfdConfig,
    input_images: Iterable[InputImage],
) -> Iterable[FaceDetectorOutput]:
  model_app = _load_model(config)
  for input_image_batch in iterate_as_batches(input_images, config.get('batch_size', 1)):
    model_batch_input = _load_images(input_image_batch, config['resize_to'])
    model_batch_output = _run_model_for_batch(model_app, model_batch_input)
    for intermediate in model_batch_input:
      face_detector_output = _produce_face_detector_output(intermediate, model_batch_output)
      yield face_detector_output


method_registry_entry = FdMethodRegistryEntry(
    method_id=MethodId('fd/scrfd'),
    process=_process,
    default_config=ScrfdConfig(
        resize_to=ImageSize(160, 160),
        device_config=DeviceConfig('cpu', 0),
        model_name='buffalo_l',
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
