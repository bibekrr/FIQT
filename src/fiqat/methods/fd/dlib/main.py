"""Implements
`face detection <http://dlib.net/face_detector.py.html>`_
and `facial landmark detection <http://dlib.net/face_landmark_detection.py.html>`_ using `dlib`.

External file dependencies:
For landmark detection, this requires the 
`shape_predictor_68_face_landmarks.dat <http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2">`_
as linked in the `facial landmark detection <http://dlib.net/face_landmark_detection.py.html>`_ example.
"""

# Standard imports:
from typing import Iterable, TypedDict, NamedTuple, Optional
from pathlib import Path

# External imports:
import cv2
import dlib

# Local imports:
# pylint: disable=relative-beyond-top-level
from .... import registry
from ....types import FdMethodRegistryEntry, MethodId, InputImages, InputImage, FaceDetectorOutput, ImageSize
from ....types import DetectedFace, FaceRoi, FacialLandmark, NpImage, ImageChannelType
from ....iterate import iterate_as_batches
from ....image import load_input_image
from ....internal import get_config_path_entry


class DlibConfig(TypedDict):
  resize_to: Optional[ImageSize]
  detect_faces: bool
  detect_landmarks: bool


class IntermediateData(NamedTuple):
  adjusted_image: NpImage
  scale_tuple: tuple[float, float]
  image: NpImage


def _load_model(config: DlibConfig):
  if config['detect_faces']:
    face_detector = dlib.get_frontal_face_detector()
  else:
    face_detector = None
  if config['detect_landmarks']:
    predictor_path = get_config_path_entry('dlib/landmark_predictor', 'dlib/shape_predictor_68_face_landmarks.dat')
    landmark_predictor = dlib.shape_predictor(str(predictor_path))
  else:
    landmark_predictor = None
  return face_detector, landmark_predictor


def _load_image(input_image: InputImage, resize_to: Optional[ImageSize]) -> IntermediateData:
  image = load_input_image(input_image, ImageChannelType.RGB).image
  adjusted_image = image if resize_to is None else cv2.resize(image, resize_to, interpolation=cv2.INTER_LINEAR)
  scale_tuple = (image.shape[1] / adjusted_image.shape[1], image.shape[0] / adjusted_image.shape[0])
  return IntermediateData(adjusted_image, scale_tuple, image)


def _load_images(input_image_batch: InputImages, resize_to: ImageSize) -> list[IntermediateData]:
  images = [_load_image(input_image, resize_to) for input_image in input_image_batch]
  return images


def _run_models_for_batch(dlib_models: tuple, model_batch_input: list[IntermediateData]) -> list[list]:
  """See http://dlib.net/face_landmark_detection.py.html & http://dlib.net/face_detector.py.html"""
  # NOTE This implementation currently doesn't support real batched execution of the model.
  face_detector, landmark_predictor = dlib_models
  model_batch_output = []
  for intermediate in model_batch_input:
    if face_detector:
      dets, scores, _ = face_detector.run(intermediate.adjusted_image, 0)
      detect_list = [{'det': det, 'score': score} for det, score in zip(dets, scores)]
    else:
      detect_list = [{
          'det': dlib.rectangle(0, 0, intermediate.adjusted_image.shape[1], intermediate.adjusted_image.shape[0]),
          'score': None,
      }]
    if landmark_predictor:
      for detect_item in detect_list:
        detect_item['shape'] = landmark_predictor(intermediate.adjusted_image, detect_item['det'])
    model_batch_output.append(detect_list)
  return model_batch_output


def _produce_face_detector_output(intermediate: IntermediateData, model_batch_output) -> FaceDetectorOutput:
  detected_faces = []
  input_image_size = ImageSize(intermediate.image.shape[1], intermediate.image.shape[0])
  face_detector_output = FaceDetectorOutput(input_image_size=input_image_size, detected_faces=detected_faces)

  detect_list = model_batch_output.pop(0)
  if detect_list is not None:
    for detect_item in detect_list:
      box = detect_item['det']
      roi = FaceRoi(
          x=box.left() * intermediate.scale_tuple[0],
          y=box.top() * intermediate.scale_tuple[1],
          width=box.width() * intermediate.scale_tuple[0],
          height=box.height() * intermediate.scale_tuple[1],
      )

      detected_face = DetectedFace(roi=roi,)

      if 'shape' in detect_item:
        landmarks = [
            FacialLandmark(landmark.x * intermediate.scale_tuple[0], landmark.y * intermediate.scale_tuple[1])
            for landmark in detect_item['shape'].parts()
        ]
        detected_face['landmarks'] = landmarks

      confidence = detect_item['score']
      if confidence is not None:
        detected_face['confidence'] = confidence
      
      detected_faces.append(detected_face)

  return face_detector_output


def _process(
    _method_registry_entry: FdMethodRegistryEntry,
    config: DlibConfig,
    input_images: Iterable[InputImage],
) -> Iterable[FaceDetectorOutput]:
  dlib_models = _load_model(config)
  for input_image_batch in iterate_as_batches(input_images, config.get('batch_size', 1)):
    model_batch_input = _load_images(input_image_batch, config['resize_to'])
    model_batch_output = _run_models_for_batch(dlib_models, model_batch_input)
    for intermediate in model_batch_input:
      face_detector_output = _produce_face_detector_output(intermediate, model_batch_output)
      yield face_detector_output


method_registry_entry = FdMethodRegistryEntry(
    method_id=MethodId('fd/dlib'),
    process=_process,
    default_config=DlibConfig(
        resize_to=None,
        detect_faces=True,
        detect_landmarks=True,
    ),
    landmark_names=[
        'C-8',  # 'Chin -8',
        'C-7',  # 'Chin -7',
        'C-6',  # 'Chin -6',
        'C-5',  # 'Chin -5',
        'C-4',  # 'Chin -4',
        'C-3',  # 'Chin -3',
        'C-2',  # 'Chin -2',
        'C-1',  # 'Chin -1',
        'C_0',  # 'Chin  0',
        'C+1',  # 'Chin +1',
        'C+2',  # 'Chin +2',
        'C+3',  # 'Chin +3',
        'C+4',  # 'Chin +4',
        'C+5',  # 'Chin +5',
        'C+6',  # 'Chin +6',
        'C+7',  # 'Chin +7',
        'C+8',  # 'Chin +8',
        'EB-5',  # 'Eyebrow -5',
        'EB-4',  # 'Eyebrow -4',
        'EB-3',  # 'Eyebrow -3',
        'EB-2',  # 'Eyebrow -2',
        'EB-1',  # 'Eyebrow -1',
        'EB+1',  # 'Eyebrow +1',
        'EB+2',  # 'Eyebrow +2',
        'EB+3',  # 'Eyebrow +3',
        'EB+4',  # 'Eyebrow +4',
        'EB+5',  # 'Eyebrow +5',
        'NR_1',  # 'Nasal ridge 1',
        'NR_2',  # 'Nasal ridge 2',
        'NR_3',  # 'Nasal ridge 3',
        'NR_4',  # 'Nasal ridge 4',
        'NB-2',  # 'Nasal base -2',
        'NB-1',  # 'Nasal base -1',
        'NB_0',  # 'Nasal base  0',
        'NB+1',  # 'Nasal base +1',
        'NB+2',  # 'Nasal base +2',
        'LE-CL',  # 'Image-left eye corner left',
        'LE-TL',  # 'Image-left eye top left',
        'LE-TR',  # 'Image-left eye top right',
        'LE-CR',  # 'Image-left eye corner right',
        'LE-BR',  # 'Image-left eye bottom right',
        'LE-BL',  # 'Image-left eye bottom left',
        'RE-CL',  # 'Image-right eye corner left',
        'RE-TL',  # 'Image-right eye top left',
        'RE-TR',  # 'Image-right eye top right',
        'RE-CR',  # 'Image-right eye corner right',
        'RE-BR',  # 'Image-right eye bottom right',
        'RE-BL',  # 'Image-right eye bottom left',
        'MOT-3',  # 'Mouth outer top -3',
        'MOT-2',  # 'Mouth outer top -2',
        'MOT-1',  # 'Mouth outer top -1',
        'MOT_0',  # 'Mouth outer top  0',
        'MOT+1',  # 'Mouth outer top +1',
        'MOT+2',  # 'Mouth outer top +2',
        'MOT+3',  # 'Mouth outer top +3',
        'MOB+2',  # 'Mouth outer bottom +2',
        'MOB+1',  # 'Mouth outer bottom +1',
        'MOB_0',  # 'Mouth outer bottom  0',
        'MOB-1',  # 'Mouth outer bottom -1',
        'MOB-2',  # 'Mouth outer bottom -2',
        'MIT-2',  # 'Mouth inner top -2',
        'MIT-1',  # 'Mouth inner top -1',
        'MIT_0',  # 'Mouth inner top  0',
        'MIT+1',  # 'Mouth inner top +1',
        'MIT+2',  # 'Mouth inner top +2',
        'MIB+1',  # 'Mouth inner bottom +1',
        'MIB_0',  # 'Mouth inner bottom  0',
        'MIB-1',  # 'Mouth inner bottom -1',
    ],
    long_landmark_names=[
        'Chin -8',
        'Chin -7',
        'Chin -6',
        'Chin -5',
        'Chin -4',
        'Chin -3',
        'Chin -2',
        'Chin -1',
        'Chin  0',
        'Chin +1',
        'Chin +2',
        'Chin +3',
        'Chin +4',
        'Chin +5',
        'Chin +6',
        'Chin +7',
        'Chin +8',
        'Eyebrow -5',
        'Eyebrow -4',
        'Eyebrow -3',
        'Eyebrow -2',
        'Eyebrow -1',
        'Eyebrow +1',
        'Eyebrow +2',
        'Eyebrow +3',
        'Eyebrow +4',
        'Eyebrow +5',
        'Nasal ridge 1',
        'Nasal ridge 2',
        'Nasal ridge 3',
        'Nasal ridge 4',
        'Nasal base -2',
        'Nasal base -1',
        'Nasal base  0',
        'Nasal base +1',
        'Nasal base +2',
        'Image-left eye corner left',
        'Image-left eye top left',
        'Image-left eye top right',
        'Image-left eye corner right',
        'Image-left eye bottom right',
        'Image-left eye bottom left',
        'Image-right eye corner left',
        'Image-right eye top left',
        'Image-right eye top right',
        'Image-right eye corner right',
        'Image-right eye bottom right',
        'Image-right eye bottom left',
        'Mouth outer top -3',
        'Mouth outer top -2',
        'Mouth outer top -1',
        'Mouth outer top  0',
        'Mouth outer top +1',
        'Mouth outer top +2',
        'Mouth outer top +3',
        'Mouth outer bottom +2',
        'Mouth outer bottom +1',
        'Mouth outer bottom  0',
        'Mouth outer bottom -1',
        'Mouth outer bottom -2',
        'Mouth inner top -2',
        'Mouth inner top -1',
        'Mouth inner top  0',
        'Mouth inner top +1',
        'Mouth inner top +2',
        'Mouth inner bottom +1',
        'Mouth inner bottom  0',
        'Mouth inner bottom -1',
    ],
)

registry.register_method(method_registry_entry)
