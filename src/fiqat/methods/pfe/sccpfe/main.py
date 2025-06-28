"""Size- and Center- and Confidence-based Primary Face Estimation"""

# Standard imports:
from typing import Iterable, Optional, TypedDict

# Local imports:
# pylint: disable=relative-beyond-top-level
from .... import registry
from ....types import MethodRegistryEntry, MethodId
from ....types import FaceDetectorOutput, DetectedFace, FaceRoi, FacialLandmark, PrimaryFaceEstimate


class ScpfeConfig(TypedDict):
  use_roi: bool
  use_landmarks: bool
  use_confidence: bool


def _has_roi(detected_face: DetectedFace) -> bool:
  """Checks whether the :class:`fiqat.types.DetectedFace` either has ``roi`` information."""
  return 'roi' in detected_face


def _has_landmarks(detected_face: DetectedFace) -> bool:
  """Checks whether the :class:`fiqat.types.DetectedFace` either has ``landmarks`` information"""
  return len(detected_face.get('landmarks', [])) > 0


def _has_roi_or_landmarks(detected_face: DetectedFace) -> bool:
  return _has_roi(detected_face) or _has_landmarks(detected_face)


_Point = FacialLandmark
_Box = FaceRoi


def _get_bounding_box_for_points(points: Iterable[_Point]) -> _Box:
  pmin = None
  pmax = None
  for point in points:
    if pmin is None:
      pmin = pmax = point
    else:
      pmin = _Point(min(pmin.x, point.x), min(pmin.y, point.y))
      pmax = _Point(max(pmax.x, point.x), max(pmax.y, point.y))
  roi = _Box(pmin.x, pmin.y, pmax.x - pmin.x, pmax.y - pmin.y)
  return roi


def _derive_roi_from_landmarks(detected_face: DetectedFace) -> FaceRoi:
  """Computes a ROI (Region Of Interest) as the bounding box of the ``landmarks``
  in the :class:`fiqat.types.DetectedFace`.
  """
  assert len(detected_face['landmarks']) > 0
  roi = _get_bounding_box_for_points(detected_face['landmarks'])
  return roi


def _get_or_derive_roi(detected_face: DetectedFace) -> FaceRoi:
  if 'roi' in detected_face:
    roi = detected_face['roi']
  else:
    roi = _derive_roi_from_landmarks(detected_face)
  return roi


def _select_face(
    face_detector_output: FaceDetectorOutput,
    use_roi: bool,
    use_landmarks: bool,
    use_confidence: bool,
) -> Optional[int]:
  """This selects a primary face from the :class:`fiqat.types.FaceDetectorOutput.detected_faces` list.

  Parameters
  ----------
  face_detector_output : FaceDetectorOutput
    The face detector information for which the primary face is to be estimated.
  use_roi : bool
    If ``True``, the ``roi`` data of the :class:`fiqat.types.FaceDetectorOutput.detected_faces` will be used for the
    estimation.
    The first ``roi`` estimation score factor for each candidate face is the minimum of the ROI's width and height.
    The second factor is only computed if ``input_image_size`` information is available in the
    :class:`fiqat.FaceDetectorOutput`,
    and favors ROIs that are closer to the image center.
    This second factor is meant to help with cases where multiple face ROIs
    with similar sizes and ``confidence`` values are detected.
  use_landmarks : bool
    If ``True``, the bounding box for the ``landmarks`` of the :class:`fiqat.types.FaceDetectorOutput.detected_faces` will be
    used as ROI information,
    with score factor computation as described for ``use_roi``.
    If both ``use_roi`` and ``use_landmarks`` is ``True``, then ``roi`` data will be used whenever available,
    and ``landmarks``-based ROIs are used as fallback.
  use_confidence : bool
    If ``True``, the stored ``confidence`` values are used as an estimation score factor,
    normalized relative to the maximum value among the :class:`fiqat.types.FaceDetectorOutput.detected_faces`.
    If either ``use_roi`` or ``use_landmarks`` is ``True`` as well, all factors are combined by multiplication.

  Returns
  -------
  Optional[int]
    The index of the primary face, or ``None`` if the ``face_detector_output`` has an empty ``detected_faces`` list.

  Raises
  ------
  ValueError
    Will be raised if:
    * ``use_roi`` and ``use_landmarks`` and ``use_confidence`` are all False (invalid configuration).
    * ``use_roi`` is True, ``use_landmarks`` is ``True``, but not all of the ``detected_faces`` have ``roi`` or
      ``landmarks`` information.
    * ``use_roi`` is ``True``, ``use_landmarks`` is ``False``, but not all of the ``detected_faces`` have ``roi``
      information.
    * ``use_roi`` is ``False``, ``use_landmarks`` is ``True``, but not all of the ``detected_faces`` have ``landmarks``
      information.
    * ``use_confidence`` is ``True``, but not all of the ``detected_faces`` have ``confidence`` information.
  """
  if not (use_roi or use_landmarks or use_confidence):
    raise ValueError('Invalid configuration: Either use_roi or use_landmarks or use_confidence needs to be True.')

  if len(face_detector_output['detected_faces']) <= 0:
    return None
  if len(face_detector_output['detected_faces']) == 1:
    return 0

  with_roi = use_roi or use_landmarks
  if with_roi:
    if use_roi and use_landmarks:
      check_func = _has_roi_or_landmarks
      error_msg = 'Not all detected_faces have "roi" or "landmarks" data.'
    elif use_roi:
      check_func = _has_roi
      error_msg = 'Not all detected_faces have "roi" data (and "landmarks" data fallback is disabled).'
    else:
      assert use_landmarks
      check_func = _has_landmarks
      error_msg = 'Not all detected_faces have "landmarks" data (and "roi" data use is disabled).'
    all_roi = all((check_func(detected_face) for detected_face in face_detector_output['detected_faces']))
    if not all_roi:
      raise ValueError('Cannot reliably estimate a primary face index using ROI information:', error_msg,
                       face_detector_output)
    image_size = face_detector_output.get('input_image_size', None)

  with_confidence = use_confidence
  if with_confidence and not all(
      ('confidence' in detected_face for detected_face in face_detector_output['detected_faces'])):
    raise ValueError('Cannot reliably estimate a primary face index using confidence information:',
                     'Not all detected_faces have "confidence" data.', face_detector_output)
  if with_confidence:
    confidence_min = 0
    confidence_max = max((detected_face['confidence'] for detected_face in face_detector_output['detected_faces']))
    confidence_range = confidence_max - confidence_min

  best_face_index = None
  best_score = None
  for face_index, detected_face in enumerate(face_detector_output['detected_faces']):
    score = 1

    if with_roi:
      if use_roi and use_landmarks:
        roi = _get_or_derive_roi(detected_face)
      elif use_roi:
        roi = detected_face['roi']
      else:
        assert use_landmarks
        roi = _derive_roi_from_landmarks(detected_face)
      size_factor = min(roi.width, roi.height)
      score *= size_factor
      if image_size is not None:
        roi_cx, roi_cy = roi.x + roi.width * 0.5, roi.y + roi.height * 0.5  # ROI center x/y
        center_factor = 1.0 - 0.5 * (abs((roi_cx / image_size.x) - 0.5) + abs((roi_cy / image_size.y) - 0.5))
        score *= center_factor

    if with_confidence and (confidence_range > 0):
      confidence = detected_face['confidence']
      confidence = max(confidence, confidence_min)
      confidence = (confidence - confidence_min) / confidence_range
      score *= confidence

    if (best_score is None) or (best_score < score):
      best_score = score
      best_face_index = face_index
  return best_face_index


def _process(
    _method_registry_entry: MethodRegistryEntry,
    config: ScpfeConfig,
    face_detector_outputs: Iterable[FaceDetectorOutput],
) -> Iterable[PrimaryFaceEstimate]:
  for face_detector_output in face_detector_outputs:
    index = _select_face(
        face_detector_output,
        use_roi=config['use_roi'],
        use_landmarks=config['use_landmarks'],
        use_confidence=config['use_confidence'],
    )
    primary_face_estimate = PrimaryFaceEstimate(face_detector_output, index)
    yield primary_face_estimate


method_registry_entry = MethodRegistryEntry(
    method_id=MethodId('pfe/sccpfe'),
    process=_process,
    default_config=ScpfeConfig(
        use_roi=True,
        use_landmarks=True,
        use_confidence=True,
    ),
)

registry.register_method(method_registry_entry)
