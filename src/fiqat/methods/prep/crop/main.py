"""Simple preprocessing method that crops the image to the face ROI,
then resizes the cropped region to the output size (if specified).
"""

# Standard imports:
from typing import Iterable, TypedDict, Optional

# External imports:
import numpy as np
import cv2

# Local imports:
# pylint: disable=relative-beyond-top-level
from .... import registry
from ....types import MethodRegistryEntry, MethodId
from ....types import ImageSize, ImageChannelType, TypedNpImage
from ....types import DetectedFace, PrimaryFaceEstimate, FaceDetectorOutput
from ....image import load_input_image


class CropConfig(TypedDict):
  image_size: Optional[ImageSize]
  """Size of the output image of the preprocessing method."""


def _process(
    _method_registry_entry: MethodRegistryEntry,
    config: CropConfig,
    primary_face_estimates: Iterable[PrimaryFaceEstimate],
) -> Iterable[TypedNpImage]:
  output_image_size: Optional[ImageSize] = config.get('image_size', None)
  for primary_face_estimate in primary_face_estimates:
    face_detector_output: FaceDetectorOutput = primary_face_estimate.face_detector_output
    detected_faces = face_detector_output['detected_faces']
    detected_face: DetectedFace = detected_faces[primary_face_estimate.index]
    face_roi = detected_face.get('roi', None)
    assert face_roi is not None, 'The preprocessing method expects face ROI information.'

    input_image = face_detector_output['input_image']
    image = load_input_image(input_image, ImageChannelType.BGR)

    y_start = int(face_roi.y)
    y_end = y_start + int(np.ceil(face_roi.height))
    x_start = int(face_roi.x)
    x_end = x_start + int(np.ceil(face_roi.width))
    cropped_image_part = image.image[y_start:y_end, x_start:x_end]

    if output_image_size is None:
      output_image = cropped_image_part.copy()
    else:
      output_image = cv2.resize(cropped_image_part, output_image_size, interpolation=cv2.INTER_LINEAR)

    yield TypedNpImage(output_image, image.channels)


method_registry_entry = MethodRegistryEntry(
    method_id=MethodId('prep/crop'),
    process=_process,
    default_config=CropConfig(image_size=None),
)

registry.register_method(method_registry_entry)
