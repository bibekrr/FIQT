"""The common types used in the toolkit."""

# Standard imports:
from pathlib import Path, PurePath
from typing import Iterable, TypedDict, NamedTuple, Callable, Optional, Union

try:
  from typing import TypeAlias
except ImportError:
  # Fallback since TypeAlias is only supported starting with Python 3.10:
  from typing import Type
  TypeAlias = Type

try:
  from enum import StrEnum
except ImportError:
  # Fallback to regular Enum since StrEnum is only supported starting with Python 3.11:
  from enum import Enum
  StrEnum: TypeAlias = Enum

# External imports:
import numpy as np
from PIL import Image

MethodId: TypeAlias = PurePath
"""A method in the toolkit is identified by a path that starts with the corresponding :class:`MethodType` string value,
and continues with one or more arbitrary name parts.

For example, MethodId('fd/abc') would be a :class:`MethodType.FACE_DETECTOR` (i.e. 'fd') method called 'abc'.

Using a path for such IDs allows for canonically nested identifier groups, e.g. 'fiqa/abc/xyz1' and 'fiqa/abc/xyz2',
analgous to files and directories in a filesystem.

alias of :class:`PurePath`
"""

MethodIdStr = Union[MethodId, str]
"""For convenience, a :class:`MethodId` can also be passed as a string to some functions,
such as the functions that comprise the toolkit's main API (:mod:`fiqat.main_api`).
"""


class DeviceConfig(NamedTuple):
  """Specifies the device type (often 'cpu') and index (often 0) for method execution.
  While methods typically support CPU execution, not all methods support GPU execution.
  """
  type: str
  """The device type, either 'cpu' or 'gpu'."""
  index: int
  """The index of the device, e.g. 0 for first device of the type."""


class MethodType(StrEnum):
  """String-enum identifiers for the method types that are supported by the toolkit."""
  FACE_DETECTOR = 'fd'
  """Main API function:
  :meth:`fiqat.main_api.detect_faces`

  Face detector methods detect face bounding-boxes (aka the face region-of-interest) and/or facial landmarks within
  an image. Face detectors may detect zero, one, or multiple faces within an image, and may also provide confidence
  values for each detected face.

  See the :class:`FaceDetectorOutput` type for the canonical face detector output structure of the toolkit.
  """
  FD = 'fd'  # Alias.
  """Alias for :class:`MethodType.FACE_DETECTOR`."""
  PRIMARY_FACE_ESTIMATOR = 'pfe'
  """Main API function:
  :meth:`fiqat.main_api.estimate_primary_faces`

  Primary face estimator methods take :class:`FaceDetectorOutput` and heuristically select one "primary" face,
  e.g. by choosing the one with the largest bounding-box area.
  This can help with images which are supposed to depict one primary face, but may accidentally show other faces e.g.
  in the background. It can also help when face detectors erroneously detected faces.

  See the :class:`PrimaryFaceEstimate` type for the canonical primary face estimator output structure of the toolkit.
  """
  PFE = 'pfe'  # Alias.
  """Alias for :class:`MethodType.PRIMARY_FACE_ESTIMATOR`."""
  PREPROCESSOR = 'prep'
  """Main API function:
  :meth:`fiqat.main_api.preprocess_images`

  Face image preprocessors adjust images based on :class:`PrimaryFaceEstimate`,
  usually by aligning the image to certain landmarks in the :class:`FaceDetectorOutput`,
  and by cropping and resizing the image.

  The preprocessed image is canonically returned as a :class:`TypedNpImage` and can be saved and used for further
  processing, e.g. as input for :class:`MethodType.FACE_IMAGE_QUALITY_ASSESSMENT_ALGORITHM`
  or :class:`MethodType.FACE_RECOGNITION_FEATURE_EXTRACTOR` methods.
  """
  PREP = 'prep'  # Alias.
  """Alias for :class:`MethodType.PREPROCESSOR`."""
  FACE_IMAGE_QUALITY_ASSESSMENT_ALGORITHM = 'fiqa'
  """Main API function:
  :meth:`fiqat.main_api.assess_quality`

  Face image quality assessment algorithms take a (usually preprocessed) image as input
  and assess the quality (often in terms of the image's utility for face recognition).

  Canonically the result is a scalar :class:`QualityScore`, higher values indicating better quality.
  """
  QUALITY_ASSESSOR = 'fiqa'  # Alias.
  """Alias for :class:`MethodType.FACE_IMAGE_QUALITY_ASSESSMENT_ALGORITHM`."""
  FIQA = 'fiqa'  # Alias.
  """Alias for :class:`MethodType.FACE_IMAGE_QUALITY_ASSESSMENT_ALGORITHM`."""
  FACE_RECOGNITION_FEATURE_EXTRACTOR = 'fr'
  """Main API function:
  :meth:`fiqat.main_api.extract_face_recognition_features`

  Feature extractors take a (usually preprocessed) image as input and canonically return a :class:`FeatureVector`.
  These can be used for face recognition, by comparing pairs (:class:`FeatureVectorPair`)."""
  FEATURE_EXTRACTOR = 'fr'  # Alias.
  """Alias for :class:`MethodType.FACE_RECOGNITION_FEATURE_EXTRACTOR`."""
  FR = 'fr'  # Alias.
  """Alias for :class:`MethodType.FACE_RECOGNITION_FEATURE_EXTRACTOR`."""
  COMPARISON_SCORE_COMPUTATION = 'csc'
  """Main API function:
  :meth:`fiqat.main_api.compute_comparison_scores`

  This method type computes a :class:`ComparisonScore` for a :class:`FeatureVectorPair`.
  The toolkit's provided methods will canonically return similarity scores (:class:`ComparisonScoreType.SIMILARITY`),
  meaning that higher values indicate higher similarity between the faces represented by the feature vectors.
  """
  CSC = 'csc'  # Alias.
  """Alias for :class:`MethodType.COMPARISON_SCORE_COMPUTATION`."""


class MethodRegistryStatus(StrEnum):
  """Modules for methods included in the toolkit are lazily loaded,
  and different methods are allowed to require incompatible Python environment setups.
  This enum states the status of the methods.
  """
  UNKNOWN = 'unknown'
  """There was no attempt to load the method yet."""
  AVAILABLE = 'available'
  """The method has been loaded and is available (and should remain available)."""
  UNAVAILABLE = 'unavailable'
  """The method could not been loaded, usually due to missing or incompatible dependencies."""


MethodRegistryInfo = Union[str, Exception]
"""Information provided after a method load attempt."""

MethodRegistryLoader = Callable[[], tuple[MethodRegistryStatus, Optional[MethodRegistryInfo]]]
"""Returns the new :class:`MethodRegistryStatus` and an optional :class:`MethodRegistryInfo` (e.g. about missing
dependencies).
"""

MethodProcessFunction = Callable[[TypedDict, dict, Iterable], Iterable]
"""A :class:`MethodRegistryEntry.process` function.
This can be called by the toolkit's main API functions (:mod:`fiqat.main_api`).

Thr first parameter is the :class:`MethodRegistryEntry`.

The second parameter is a dictionary with configuration options.
The default values of this dictionary are specified by :class:`MethodRegistryEntry.default_config`.
Overwriting values or other values are specified by the keyword arguments (``**kwargs``) to the main API functions
(:mod:`fiqat.main_api`).

The third paramter is an `Iterable <https://docs.python.org/3/library/typing.html#typing.Iterable>`_ over the
input items.

The result is an `Iterable <https://docs.python.org/3/library/typing.html#typing.Iterable>`_ over the output items,
in the same order as the input items (each output item corresponding to one input item).
Typically this function is implemented as a generator, i.e. using ``yield`` to iteratively produce output items.
"""


class MethodRegistryEntry(TypedDict):
  """The toolkit tracks methods in a registry (:mod:`fiqat.registry`) with the general :class:`RegistryEntry` type.
  This :class:`MethodRegistryEntry` type contains further method-specific data for loaded
  (i.e. :class:`MethodRegistryStatus.AVAILABLE`) methods.
  """
  method_id: MethodId
  """The identifier of the method."""
  process: MethodProcessFunction
  """The :class:`MethodProcessFunction` that can be called by the toolkit's main API function (:mod:`fiqat.main_api`)
  that corresponds to the method's type."""
  default_config: dict
  """The method's available configuration options with their default values."""


class RegistryEntry(TypedDict):
  """The toolkit tracks methods in a registry (:mod:`fiqat.registry`) using this type for registry entries."""
  status: MethodRegistryStatus
  """The availability of the method.

  This is mainly relevant for the methods that are included with the toolkit.
  External custom methods will likely be already loaded when they are registered
  (and thus :class:`MethodRegistryStatus.AVAILABLE`).
  """
  loader: Optional[MethodRegistryLoader]
  """The loader function for the method.
  Methods with status :class:`MethodRegistryStatus.UNKNOWN` aren't loaded yet,
  in which case this function will be used to (attempt to) load them.

  This is mainly relevant for the methods that are included with the toolkit.
  External custom methods will likely be already loaded when they are registered,
  and thus do not need to specify a loader function.
  """
  info: Optional[MethodRegistryInfo]
  """Information that may be provided by a :class:`MethodRegistryLoader` call,
  usually when the loading failed for some reason, such as missing or incompatible dependencies."""
  method_entry: Optional[MethodRegistryEntry]
  """The ``MethodRegistryEntry`` for methods with :class:`MethodRegistryStatus.AVAILABLE` status."""


class FdMethodRegistryEntry(MethodRegistryEntry):
  """:class:`MethodRegistryEntry` specialization for face detectors (:class:`MethodType.FACE_DETECTOR`)."""
  landmark_names: list[str]
  """Names for the landmarks, in the same order as the :class:`DetectedFace` landmarks lists.
  These names can be used e.g. to draw labeled landmarks.

  Adding names to this list is optional.
  If there is no landmark name specified at a certain index in this list,
  then the toolkit's drawing code (:meth:`fiqat.draw.draw_face_detector_output`) will fall back to using the
  index as the label.
  """


class QualityScoreRange(NamedTuple):
  """The minimum and maximum :class:`QualityScore` that can be produced by a
  :class:`MethodType.FACE_IMAGE_QUALITY_ASSESSMENT_ALGORITHM` method.
  """
  min: float
  """The minimum value that may be produced."""
  max: float
  """The maximum value that may be produced."""


class FiqaMethodRegistryEntry(MethodRegistryEntry):
  """Face image quality assessment :class:`MethodRegistryEntry`."""
  quality_score_range: Optional[QualityScoreRange]
  """If specified, the minimum/maximum comparison score that can be produced.
  If the method is used via the main API, this will be automatically asserted.
  """


class ComparisonScoreType(StrEnum):
  """The type of a :class:`ComparisonScore` that defines how the value should be interpreted."""
  SIMILARITY = 'similarity'
  """Higher values indicate higher similarity between faces.
  This is the type canonically used by the toolkit,
  which is why there also is a :class:`SimilarityScore` type alias to clarify this usage.
  """
  DISSIMILARITY = 'dissimilarity'
  """Lower values indicate higher similarity between faces."""


class ComparisonScoreRange(NamedTuple):
  """The minimum and maximum :class:`ComparisonScore` value that can be produced by a
  :class:`MethodType.COMPARISON_SCORE_COMPUTATION` method."""
  min: float
  """The minimum value that may be produced."""
  max: float
  """The maximum value that may be produced."""


class CscMethodRegistryEntry(MethodRegistryEntry):
  """:class:`MethodRegistryEntry` specialization for comparison score computation
  (:class:`MethodType.COMPARISON_SCORE_COMPUTATION`).
  """
  comparison_score_type: ComparisonScoreType
  """The method either produces similarity or dissimilarity scores.
  Methods provided by the toolkit will canonically produce :class:`ComparisonScoreType.SIMILARITY` scores.
  """
  comparison_score_range: Optional[ComparisonScoreRange]
  """If specified, the minimum/maximum comparison score that can be produced.
  If the method is used via the main API, i.e. :meth:`fiqat.main_api.compute_comparison_scores`, this will be
  asserted automatically.
  """


class ImageChannelType(StrEnum):
  """The common image channel types that are used within the toolkit."""
  GRAY = 'gray'
  """For grayscale images with a single channel."""
  RGB = 'rgb'
  """For color images with the RGB (Red,Green,Blue) channel order."""
  BGR = 'bgr'
  """For color images with the BGR (Blue,Green,Red) channel order.
  This order is by `cv2 <https://pypi.org/project/opencv-python/>`_ functions used within parts of the toolkit.
  """


class Cv2ColorTuple(NamedTuple):
  """A BGR color tuple, for use by various `cv2 <https://pypi.org/project/opencv-python/>`_ functions."""
  b: int  # pylint: disable=invalid-name
  """Blue channel value, usually in the range [0,255]."""
  g: int  # pylint: disable=invalid-name
  """Green channel value, usually in the range [0,255]."""
  r: int  # pylint: disable=invalid-name
  """Red channel value, usually in the range [0,255]."""


NpImage: TypeAlias = np.ndarray
"""Image represented as a ``numpy`` array.

alias of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
"""


class TypedNpImage(NamedTuple):
  """Tuple of an :class:`NpImage` and the associated :class:`ImageChannelType`."""
  image: NpImage
  """The image data."""
  channels: ImageChannelType
  """The channel type of the image data."""


PillowImage: TypeAlias = Image.Image
"""Image type from the ``Pillow`` (PIL) library.

alias of `PIL.Image.Image <https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image>`_
"""

InputImage = Union[Path, NpImage, PillowImage]
"""Either a
`pathlib.Path <https://docs.python.org/3/library/pathlib.html#pathlib.Path>`_ to an image
or image data (:class:`NpImage` or :class:`PillowImage`)
that can be used as input for the toolkit's main API functions (:mod:`fiqat.main_api`).
"""


class ImageSize(NamedTuple):
  """The integer pixel width(``x``),height(``y``) of an image."""
  x: int  # pylint: disable=invalid-name
  """The width in pixels as an integer."""
  y: int  # pylint: disable=invalid-name
  """The height in pixels as an integer."""


class FaceRoi(NamedTuple):
  """x,y,width,height pixels of a face ROI (Region Of Interest) rectangle, aka bounding-box.
  x,y specifies the top-left start coordinates of the rectangle.
  x+width,y+height specifies the bottom-right end coordinates of the rectangle.

  Note that the values are allowed to be floating-point numbers (i.e. not exact pixel integer values).
  """
  x: float  # pylint: disable=invalid-name
  """The horizontal (left) pixel coordinate."""
  y: float  # pylint: disable=invalid-name
  """The vertical (top) pixel coordinate."""
  width: float
  """The width."""
  height: float
  """The height."""


class FacialLandmark(NamedTuple):
  """The floating-point or integer x,y pixel coordinates of a facial landmark in an image."""
  x: float  # pylint: disable=invalid-name
  """The horizontal pixel coordinate."""
  y: float  # pylint: disable=invalid-name
  """The vertical pixel coordinate."""


class DetectedFace(TypedDict):
  """Face detector information for a single face.

  Note that the actual face detector output structure can vary between the face detector methods.
  """
  roi: Optional[FaceRoi]
  """The ROI (Region Of Interest) bounding-box rectangle around the face."""
  landmarks: Optional[list[FacialLandmark]]
  """A list of facial landmark points.

  Note that different face detectors can provide different facial landmark sets.
  """
  confidence: Optional[float]
  """A confidence value provided by the face detector.
  Higher values mean higher confidence that a face has been detected.

  Note that the range may differ depending on the used face detector.
  """


class FaceDetectorOutput(TypedDict):
  """The output of a face detector for one image, which may include multiple :class:`DetectedFace` instances.

  Note that the actual face detector output structure can vary between the face detector methods.
  For example, different methods can provide different facial landmark sets.
  """
  detected_faces: list[DetectedFace]
  """The detected faces."""
  input_image_size: ImageSize
  """The width/height of the input image that was processed by the face detector.
  This can be used e.g. to normalize the ``roi`` and ``landmarks`` (which are specified in pixel coordinates).
  """
  input_image: InputImage
  """The input image that was processed by the face detector."""


class PrimaryFaceEstimate(NamedTuple):
  """Specifies a primary face by an ``index`` for the :class:`FaceDetectorOutput.detected_faces` list."""
  face_detector_output: FaceDetectorOutput
  """The face detector data which is being indexed.
  Usually the estimate is based on this data too.
  """
  index: Optional[int]
  """The index of the primary face.
  If there is no primary face (e.g. because the :class:`FaceDetectorOutput.detected_faces` list is empty),
  then this shall be ``None``.
  """


FeatureVector: TypeAlias = np.ndarray
"""Features usable for face recognition.

alias of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
"""

FeatureVectorPair: TypeAlias = tuple[FeatureVector, FeatureVector]
""":class:`FeatureVector` pair that can be used to compute a :class:`ComparisonScore`.

alias of :class:`tuple` [ :class:`FeatureVector`, :class:`FeatureVector` ]
"""

ComparisonScore: TypeAlias = float
"""The result of a :class:`FeatureVectorPair` comparison.

alias of :class:`float`
"""
SimilarityScore: TypeAlias = ComparisonScore
"""A :class:`ComparisonScore` that more specifically is a similarity score (:class:`ComparisonScoreType.SIMILARITY`),
meaning that higher values indicate higher similarity between the faces represented by the feature vectors.

alias of :class:`ComparisonScore`
"""

QualityScore: TypeAlias = float
"""A quality score produced by a :class:`MethodType.FACE_IMAGE_QUALITY_ASSESSMENT_ALGORITHM`.

If the score itself is supposed to indicate the utility of the image for face recognition,
then higher values should canonically indicate higher utility (i.e. higher should mean better).

alias of :class:`float`
"""

InputImages = Union[InputImage, Iterable[InputImage]]
"""One or multiple :class:`InputImage` items."""

FaceDetectorOutputs = Union[FaceDetectorOutput, Iterable[FaceDetectorOutput]]
"""One or multiple :class:`FaceDetectorOutput` items."""

PrimaryFaceEstimates = Union[PrimaryFaceEstimate, Iterable[PrimaryFaceEstimate]]
"""One or multiple :class:`PrimaryFaceEstimate` items."""

TypedNpImages = Union[TypedNpImage, Iterable[TypedNpImage]]
"""One or multiple :class:`TypedNpImage` items."""

FeatureVectors = Union[FeatureVector, Iterable[FeatureVector]]
"""One or multiple :class:`FeatureVector` items."""

FeatureVectorPairs = Union[FeatureVectorPair, Iterable[FeatureVectorPair]]
"""One or multiple :class:`FeatureVectorPair` items."""

SimilarityScores = Union[SimilarityScore, Iterable[SimilarityScore]]
"""One or multiple :class:`SimilarityScore` items."""

ComparisonScores = Union[ComparisonScore, Iterable[ComparisonScore]]
"""One or multiple :class:`ComparisonScore` items."""

QualityScores = Union[QualityScore, Iterable[QualityScore]]
"""One or multiple :class:`QualityScore` items."""

namedtuple_types = (DeviceConfig, QualityScoreRange, ComparisonScoreRange, Cv2ColorTuple, TypedNpImage, ImageSize,
                    FaceRoi, FacialLandmark, PrimaryFaceEstimate)
"""All :class:`NamedTuple` types in the :mod:`fiqat.types` module."""

enum_types = (MethodType, ComparisonScoreType, ImageChannelType, MethodRegistryStatus)
"""All enumeration types in the :mod:`fiqat.types` module."""
