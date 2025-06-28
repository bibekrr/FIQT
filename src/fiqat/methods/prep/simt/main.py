"""'Similarity transformation', as used by i.a.:
- ArcFace: Additive Angular Margin Loss for Deep Face Recognition
- CosFace: Large Margin Cosine Loss for Deep Face Recognition
- SphereFace: Deep Hypersphere Embedding for Face Recognition

It crops and aligns the facial image to five facial landmarks,
two for the eyes, one of the tip of the nose, and two for the mouth corners,
as produced by i.a. RetinaFace.
"""

# Standard imports:
from typing import Iterable, TypedDict, Optional

# External imports:
import numpy as np

# Local imports:
# pylint: disable=relative-beyond-top-level
from .... import registry
from ....types import MethodRegistryEntry, MethodId
from ....types import ImageSize, ImageChannelType, TypedNpImage
from ....types import DetectedFace, PrimaryFaceEstimate, FaceDetectorOutput
from ....image import load_input_image
from . import face_align


class SimtConfig(TypedDict):
  image_size: Optional[ImageSize]
  """Size of the output image of the preprocessing method."""


def _process(
    _method_registry_entry: MethodRegistryEntry,
    config: SimtConfig,
    primary_face_estimates: Iterable[PrimaryFaceEstimate],
) -> Iterable[TypedNpImage]:
  for primary_face_estimate in primary_face_estimates:
    face_detector_output: FaceDetectorOutput = primary_face_estimate.face_detector_output
    detected_faces = face_detector_output['detected_faces']
    detected_face: DetectedFace = detected_faces[primary_face_estimate.index]
    landmarks = detected_face['landmarks']
    assert len(landmarks) == 5, ('The preprocessing method expects five landmarks, as produced by e.g. RetinaFace:'
                                 ' Two for the eyes, one of the tip of the nose, and two for the mouth corners.',
                                 landmarks)
    landmarks = np.array(landmarks)
    input_image = face_detector_output['input_image']
    image = load_input_image(input_image, ImageChannelType.BGR)
    image_size = config.get('image_size', None)
    if image_size is None:
      roi = detected_face['roi']
      image_size = ImageSize(round(roi.width), round(roi.height))
    output_image = face_align.norm_crop(image.image, landmarks, image_size)
    yield TypedNpImage(output_image, image.channels)


method_registry_entry = MethodRegistryEntry(
    method_id=MethodId('prep/simt'),
    process=_process,
    default_config=SimtConfig(image_size=None),
)

registry.register_method(method_registry_entry)
