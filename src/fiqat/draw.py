"""Drawing utility functionality."""

# Standard imports:
from typing import Optional

# External imports:
import cv2

# Local imports:
from .types import InputImage, TypedNpImage, ImageChannelType, Cv2ColorTuple, ImageSize
from .types import FaceDetectorOutput
from .types import FdMethodRegistryEntry
from .image import load_input_image


def draw_face_detector_output(
    input_image: InputImage,
    face_detector_output: FaceDetectorOutput,
    resize_to: Optional[ImageSize] = None,
    draw_landmarks: bool = True,
    draw_roi: bool = True,
    draw_confidence: bool = False,
    draw_face_labels: bool = False,
    draw_landmark_labels: bool = False,
    landmark_names: Optional[list[str]] = None,
    method_registry_entry: Optional[FdMethodRegistryEntry] = None,
    always_show_landmark_indices: bool = True,
    roi_color: Cv2ColorTuple = Cv2ColorTuple(0, 0, 255),
    roi_thickness: int = 2,
    landmark_color: Cv2ColorTuple = Cv2ColorTuple(0, 0, 255),
    landmark_thickness: int = 2,
    landmark_radius: int = 1,
    font_name: Optional[str] = None,
    font_color: Cv2ColorTuple = Cv2ColorTuple(0, 0, 255),
    font_scale: float = 1,
    landmark_font_scale: Optional[float] = None,
    primary_face_index: Optional[int] = None,
    primary_face_color: Cv2ColorTuple = Cv2ColorTuple(0, 255, 0),
) -> TypedNpImage:
  """Draws :class:`fiqat.types.FaceDetectorOutput` data, e.g. facial landmarks, onto a copy of the given image.
  
  Parameters
  ----------
  input_image : InputImage
    The input image.
    The function internally uses `cv2 <https://pypi.org/project/opencv-python/>`_ for drawing.
    If a :class:`fiqat.types.TypedNpImage` is passed a copy will be converted to
    :class:`fiqat.types.ImageChannelType.BGR` if necessary.
    If an :class:`fiqat.types.NpImage` is passed it is assumed to be of :class:`fiqat.types.ImageChannelType.BGR`.
  face_detector_output : FaceDetectorOutput
    The face detector output that includes the detected faces with the data that should be drawn.
  resize_to : Optional[ImageSize]
    If set, the input image is resized to this size before drawing, using ``cv2.resize``.
    The ``face_detector_output`` coordinates will be adjusted automatically.
    You can alternatively resize the image yourself prior to calling this function, since the coordinates will
    also be adjusted if the :class:`fiqat.types.FaceDetectorOutput.input_image_size` deviates from the ``image``'s size.
  draw_landmarks : bool
    If ``True``, the landmarks will be drawn.
    The drawing function is ``cv2.circle``.
  draw_roi : bool
    If ``True``, the detected face ROI i.e. box will be drawn (if available in the ``face_detector_output`` data).
    The drawing function is ``cv2.rectangle``.
  draw_confidence : bool
    If ``True``, the detector's confidence value will be drawn (if available in the ``face_detector_output`` data).
    The confidence value will be shown with two digits after the decimal point.
    The drawing function is ``cv2.putText``.
    This currently only works if ROI data is available to determine the text position.
  draw_face_labels : bool
    If ``True``, faces are labeled as "Face i", i being the index starting at 1.
    The drawing function is ``cv2.putText``.
    This currently only works if ROI data is available to determine the text position.
  draw_landmark_labels : bool
    If ``True``, the landmarks will be labeled.
    If specified, the ``landmark_names`` will be used as labels.
    Otherwise, if specified, ``method_registry_entry``'s ``landmark_names`` will be used.
    If no landmark name is available at a landmark index, the indices starting at 1 will be used as labels.
    The drawing function is ``cv2.putText``.
  landmark_names : Optional[list[str]]
    If specified, these names will be used to label the drawn landmarks.
    Setting this has no effect if ``draw_landmark_labels`` is ``False``.
  method_registry_entry : Optional[FdMethodRegistryEntry]
    If specified, the :class:`fiqat.types.FdMethodRegistryEntry.landmark_names` will be used to label the drawn
    landmarks. Setting this has no effect if ``landmark_names`` is set or if ``draw_landmark_labels`` is ``False``.
  always_show_landmark_indices : bool
    If landmark labels are drawn and if landmark names are provided,
    show the indices (starting at 1) as a prefix to the names.
  roi_color : Cv2ColorTuple
    The color used to draw the face ROI rectangles.
  roi_thickness : int
    The line thickness used to draw the face ROI rectangles.
  landmark_color : Cv2ColorTuple
    The color used to draw the landmark dots.
  landmark_thickness : int
    The line thickness used to draw the landmark dots.
  landmark_radius : int
    The circle radius used to draw the landmark dots.
  font_name : Optional[str]
    Name of the font that should be used to draw the labels.
  font_color : Cv2ColorTuple
    Color of the font that should be used to draw the labels.
  font_scale : float
    Scale of the font that should be used to draw the labels.
  landmark_font_scale : Optional[float]
    Scale of the font that should be used to draw the landmark labels.
    If None, the font_scale will be used.
  primary_face_index : Optional[int]
    Optional index of a "primary" detected face, which will be highlighted in the ``primary_face_color``.
  primary_face_color : Cv2ColorTuple
    This color will be used for the face with the ``primary_face_index`` for all drawing operations.

  Returns
  -------
  :class:`fiqat.types.TypedNpImage`
    The new image with drawn landmarks, with :class:`fiqat.types.ImageChannelType.BGR`.
  """
  if (landmark_names is None) and (method_registry_entry is not None):
    landmark_names = method_registry_entry.get('landmark_names', [])
  if landmark_names is None:
    landmark_names = []
  if landmark_font_scale is None:
    landmark_font_scale = font_scale
  image, channels = load_input_image(input_image, ImageChannelType.BGR, always_copy=True)
  input_image_size = face_detector_output.get('input_image_size', ImageSize(image.shape[1], image.shape[0]))
  output_image_size = resize_to
  if output_image_size is not None:
    output_image_size = ImageSize(*output_image_size)
    image = cv2.resize(image, output_image_size, interpolation=cv2.INTER_LINEAR)
  else:
    output_image_size = ImageSize(image.shape[1], image.shape[0])
  scale_x, scale_y = (output_image_size.x / input_image_size.x, output_image_size.y / input_image_size.y)
  for face_i, detected_face in enumerate(face_detector_output['detected_faces']):
    has_roi = 'roi' in detected_face
    has_confidence = 'confidence' in detected_face
    if draw_roi and has_roi:
      roi = detected_face['roi']
      roi_start = (int(roi.x * scale_x), int(roi.y * scale_y))
      roi_end = (int((roi.x + roi.width) * scale_x), int((roi.y + roi.height) * scale_y))
      cv2.rectangle(
          image,
          roi_start,
          roi_end,
          roi_color if face_i != primary_face_index else primary_face_color,
          roi_thickness,
      )
    if draw_landmarks:
      for landmark_i, landmark in enumerate(detected_face['landmarks']):
        landmark_center = (int(landmark.x * scale_x), int(landmark.y * scale_y))
        cv2.circle(
            image,
            landmark_center,
            landmark_radius,
            landmark_color if face_i != primary_face_index else primary_face_color,
            landmark_thickness,
        )
        if draw_landmark_labels:
          label = f'{landmark_i + 1}'
          if landmark_i < len(landmark_names):
            landmark_name = landmark_names[landmark_i]
            label = f'{label}: {landmark_name}' if always_show_landmark_indices else landmark_name
          text_position = (landmark_center[0], landmark_center[1] - (1 + landmark_radius + (landmark_thickness // 2)))
          cv2.putText(
              image,
              label,
              text_position,
              font_name,
              landmark_font_scale,
              font_color if face_i != primary_face_index else primary_face_color,
          )
    if (draw_face_labels or draw_confidence) and has_roi:
      roi = detected_face['roi']
      text_position = (int(roi.x * scale_x), int((roi.y * scale_y) - (1 + roi_thickness)))
      label = ''
      if draw_face_labels:
        label = f'Face {face_i + 1}'
        if (draw_confidence and has_confidence):
          label += ', '
      if (draw_confidence and has_confidence):
        label += f"{detected_face['confidence']:.2f}"
      cv2.putText(
          image,
          label,
          text_position,
          font_name,
          font_scale,
          font_color if face_i != primary_face_index else primary_face_color,
      )

  return TypedNpImage(image, channels)
