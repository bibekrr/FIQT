"""This module contains the main API functions of the toolkit.

All main API functions support processing of either single or multiple input items
(i.e. an `Iterable <https://docs.python.org/3/library/typing.html#typing.Iterable>`_).
The reason for supporting iterable input is to enable method implementations to support possibly
more computationally efficient batched processing of the input items (e.g. for deep learning models executed on a GPU).

If one input item is given, one output item is returned.
If multiple input items are given, then multiple output items will be returned in the same order.
For multiple inputs, note that methods canonically act as iterators, so that results will be computed during iteration.
I.e. for a long list of inputs not all outputs are immediately computed when the function is called,
since the functions act as generators when multiple inputs are used.

Every function takes arbitrary keyword arguments (``**kwargs``) that are passed to the concrete method implementations,
to allow for different configurations.
E.g. included methods that support running on a CPU or GPU can be configured by passing a ``device_config`` keyword
argument of type :class:`fiqat.types.DeviceConfig`.
Available configuration options and their default values can be queried via :meth:`fiqat.registry.get_method`.

Other configuration data specific to your local installation is handled via the :mod:`fiqat.config` module.
E.g. the local paths to required model files.

While the toolkit does define canonical input/output types for the included methods of each
:class:`fiqat.types.MethodType`, custom methods can easily use different types if needed.
"""

# Standard imports:
from typing import Iterable, Union, Any

# Local imports:
from .types import MethodId, MethodType, InputImages, MethodRegistryEntry
from .types import MethodIdStr, FaceDetectorOutputs, PrimaryFaceEstimates, TypedNpImages
from .types import QualityScores, ComparisonScores, FeatureVectors, FeatureVectorPairs
from . import registry
from . import term


def detect_faces(method_id: MethodIdStr,
                 images: InputImages,
                 tqdm_config: Union[bool, tuple, list, dict] = True,
                 **kwargs) -> FaceDetectorOutputs:
  """Detect faces using the specified face detector.

  Parameters
  ----------
  method_id : :class:`fiqat.types.MethodIdStr`
    Specifies the face detector.
    The IDs start with 'fd/' (see :class:`fiqat.types.MethodType.FACE_DETECTOR`).
  images : :class:`fiqat.types.InputImages`
    One image or multiple images that should be processed.
    Note that you can also pass paths to image files (see :class:`fiqat.types.InputImage`).
  tqdm_config : Union[bool, tuple, list, dict]
    If not ``False`` and if the input is iterable, :meth:`fiqat.term.tqdm` will be called to show a progress bar.
    The default progress bar description is the API function name followed by the ``method_id``.
  **kwargs
    Keyword arguments passed to the method.

  Returns
  -------
  :class:`fiqat.types.FaceDetectorOutputs`
    One or multiple output items. See the :mod:`fiqat.main_api` module description for general method output behavior.
  """
  iterable_input, images = _check_iterable_mode(images)
  return _return_result(iterable_input, tqdm_config, f'fiqat.detect_faces: {method_id}', images,
                        _detect_faces(method_id, images, kwargs))


def _detect_faces(method_id: MethodIdStr, images: InputImages, kwargs) -> FaceDetectorOutputs:
  for image, face_detector_output in zip(images, _run_method_processing(method_id, kwargs, 'fd', images)):
    face_detector_output['input_image'] = image
    yield face_detector_output


def estimate_primary_faces(method_id: MethodIdStr,
                           face_detector_output: FaceDetectorOutputs,
                           tqdm_config: Union[bool, tuple, list, dict] = True,
                           **kwargs) -> PrimaryFaceEstimates:
  """Select a primary face for :class:`fiqat.types.FaceDetectorOutput` data (e.g. from a :meth:`detect_faces` call),
  to handle cases in which multiple faces were detected in an image with only one relevant face.

  Parameters
  ----------
  method_id : :class:`fiqat.types.MethodIdStr`
    Specifies the primary face estimator method.
    The IDs start with 'pfe/' (see :class:`fiqat.types.MethodType.PRIMARY_FACE_ESTIMATOR`).
  face_detector_output : :class:`fiqat.types.FaceDetectorOutputs`
    One or multiple :class:`fiqat.types.FaceDetectorOutput` items that should be processed.
  tqdm_config : Union[bool, tuple, list, dict]
    If not ``False`` and if the input is iterable, :meth:`fiqat.term.tqdm` will be called to show a progress bar.
    The default progress bar description is the API function name followed by the ``method_id``.
  **kwargs
    Keyword arguments passed to the method.

  Returns
  -------
  :class:`fiqat.types.PrimaryFaceEstimates`
    One or multiple output items. See the :mod:`fiqat.main_api` module description for general method output behavior.
  """
  iterable_input, face_detector_output = _check_iterable_mode(face_detector_output)
  return _return_result(iterable_input, tqdm_config, f'fiqat.estimate_primary_faces: {method_id}', face_detector_output,
                        _run_method_processing(method_id, kwargs, 'pfe', face_detector_output))


def preprocess_images(method_id: MethodIdStr,
                      primary_face_estimates: PrimaryFaceEstimates,
                      tqdm_config: Union[bool, tuple, list, dict] = True,
                      **kwargs) -> TypedNpImages:
  """Preprocess an image based on :class:`fiqat.types.PrimaryFaceEstimate` data
  (e.g. from a :meth:`estimate_primary_faces` call).
  Each :class:`fiqat.types.PrimaryFaceEstimate` points to a specific detected face in a
  :class:`fiqat.types.FaceDetectorOutput`, and each :class:`fiqat.types.FaceDetectorOutput` also contains the
  corresponding :class:`fiqat.types.InputImage`.

  Parameters
  ----------
  method_id : :class:`fiqat.types.MethodIdStr`
    Specifies the preprocessor method.
    The IDs start with 'prep/' (see :class:`fiqat.types.MethodType.PREPROCESSOR`).
  primary_face_estimates : :class:`fiqat.types.PrimaryFaceEstimates`
    One or multiple :class:`fiqat.types.PrimaryFaceEstimate` items that should be processed.
  tqdm_config : Union[bool, tuple, list, dict]
    If not ``False`` and if the input is iterable, :meth:`fiqat.term.tqdm` will be called to show a progress bar.
    The default progress bar description is the API function name followed by the ``method_id``.
  **kwargs
    Keyword arguments passed to the method.

  Returns
  -------
  :class:`fiqat.types.TypedNpImages`
    One or multiple output items. See the :mod:`fiqat.main_api` module description for general method output behavior.
  """
  iterable_input, primary_face_estimates = _check_iterable_mode(primary_face_estimates)
  return _return_result(iterable_input, tqdm_config, f'fiqat.preprocess_images: {method_id}', primary_face_estimates,
                        _run_method_processing(method_id, kwargs, 'prep', primary_face_estimates))


def assess_quality(method_id: MethodIdStr,
                   images: InputImages,
                   tqdm_config: Union[bool, tuple, list, dict] = True,
                   **kwargs) -> QualityScores:
  """Assess the quality of a face image.
  Usually the image should be preprocessed (e.g. from a :meth:`preprocess_images` call).

  Parameters
  ----------
  method_id : :class:`fiqat.types.MethodIdStr`
    Specifies the face image quality assessment method.
    The IDs start with 'fiqa/' (see :class:`fiqat.types.MethodType.FACE_IMAGE_QUALITY_ASSESSMENT_ALGORITHM`).
  images : :class:`fiqat.types.InputImages`
    One image or multiple images that should be processed.
    Note that you can also pass paths to image files (see :class:`fiqat.types.InputImage`).
  tqdm_config : Union[bool, tuple, list, dict]
    If not ``False`` and if the input is iterable, :meth:`fiqat.term.tqdm` will be called to show a progress bar.
    The default progress bar description is the API function name followed by the ``method_id``.
  **kwargs
    Keyword arguments passed to the method.

  Returns
  -------
  :class:`fiqat.types.QualityScores`
    One or multiple output items. See the :mod:`fiqat.main_api` module description for general method output behavior.
  """
  iterable_input, images = _check_iterable_mode(images)
  return _return_result(
      iterable_input, tqdm_config, f'fiqat.assess_quality: {method_id}', images,
      _run_method_processing_with_score_check(method_id, kwargs, 'fiqa', 'quality_score_range', images))


def extract_face_recognition_features(method_id: MethodIdStr,
                                      images: InputImages,
                                      tqdm_config: Union[bool, tuple, list, dict] = True,
                                      **kwargs) -> FeatureVectors:
  """Extract a feature vector for face recognition from a face image.
  Usually the image should be preprocessed (e.g. from a :meth:`preprocess_images` call).

  Parameters
  ----------
  method_id : :class:`fiqat.types.MethodIdStr`
    Specifies the face recognition feature extractor method.
    The IDs start with 'fr/' (see :class:`fiqat.types.MethodType.FACE_RECOGNITION_FEATURE_EXTRACTOR`).
  images : :class:`fiqat.types.InputImages`
    One image or multiple images that should be processed.
    Note that you can also pass paths to image files (see :class:`fiqat.types.InputImage`).
  tqdm_config : Union[bool, tuple, list, dict]
    If not ``False`` and if the input is iterable, :meth:`fiqat.term.tqdm` will be called to show a progress bar.
    The default progress bar description is the API function name followed by the ``method_id``.
  **kwargs
    Keyword arguments passed to the method.

  Returns
  -------
  :class:`fiqat.types.FeatureVectors`
    One or multiple output items. See the :mod:`fiqat.main_api` module description for general method output behavior.
  """
  iterable_input, images = _check_iterable_mode(images)
  return _return_result(iterable_input, tqdm_config, f'fiqat.extract_face_recognition_features: {method_id}', images,
                        _run_method_processing(method_id, kwargs, 'fr', images))


def compute_comparison_scores(method_id: MethodIdStr,
                              feature_vector_pairs: FeatureVectorPairs,
                              tqdm_config: Union[bool, tuple, list, dict] = True,
                              **kwargs) -> ComparisonScores:
  """Compute a comparison score for a pair of feature vectors for face recognition
  (each e.g. from a :meth:`extract_face_recognition_features` call).
  Note that the comparison score computation method must fit to the used feature vector extration method to obtain
  sensible results.

  Parameters
  ----------
  method_id : :class:`fiqat.types.MethodIdStr`
    Specifies the comparison score computation method.
    The IDs start with 'csc/' (see :class:`fiqat.types.MethodType.COMPARISON_SCORE_COMPUTATION`).
  feature_vector_pairs : :class:`fiqat.types.FeatureVectorPairs`
    One or multiple :class:`FeatureVectorPair` items that should be processed.
  tqdm_config : Union[bool, tuple, list, dict]
    If not ``False`` and if the input is iterable, :meth:`fiqat.term.tqdm` will be called to show a progress bar.
    The default progress bar description is the API function name followed by the ``method_id``.
  **kwargs
    Keyword arguments passed to the method.

  Returns
  -------
  :class:`fiqat.types.ComparisonScores`
    One or multiple output items. See the :mod:`fiqat.main_api` module description for general method output behavior.
    Note that the methods included in the toolkit all canonically return :class:`fiqat.types.SimilarityScores`,
    i.e. :class:`fiqat.types.ComparisonScores` of type :class:`fiqat.types.ComparisonScoreType.SIMILARITY`.
  """
  iterable_input, feature_vector_pairs = _check_iterable_mode(feature_vector_pairs)
  return _return_result(
      iterable_input, tqdm_config, f'fiqat.compute_comparison_scores: {method_id}', feature_vector_pairs,
      _run_method_processing_with_score_check(method_id, kwargs, 'csc', 'comparison_score_range', feature_vector_pairs))


def _run_method_processing_with_score_check(method_id: MethodIdStr, kwargs: dict, method_type_str: str, range_key: str,
                                            *process_args):
  method_id = MethodId(method_id)
  method_registry_entry = registry.get_method(method_id, registry.MethodType(method_type_str))
  scores = _run_method_processing(method_id, kwargs, method_type_str, *process_args)
  score_range = method_registry_entry.get(range_key, None)
  for score in scores:
    if score_range is not None:
      assert (score_range.min <= score) and (score_range.max >= score), (method_id, range_key, score_range, score)
    yield score


main_api_functions_by_method_type = {
    MethodType.FACE_DETECTOR: detect_faces,
    MethodType.PRIMARY_FACE_ESTIMATOR: estimate_primary_faces,
    MethodType.PREPROCESSOR: preprocess_images,
    MethodType.FACE_IMAGE_QUALITY_ASSESSMENT_ALGORITHM: assess_quality,
    MethodType.FACE_RECOGNITION_FEATURE_EXTRACTOR: extract_face_recognition_features,
    MethodType.COMPARISON_SCORE_COMPUTATION: compute_comparison_scores,
}


def process(method_id: MethodIdStr,
            input_items: Union[Any, Iterable[Any]],
            tqdm_config: Union[bool, tuple, list, dict] = True,
            **kwargs) -> Union[Any, Iterable[Any]]:
  """This general processing function will call one of the other main API functions based on the
  :class:`fiqat.types.MethodType` that can be read from the :class:`fiqat.types.MethodIdStr`.

  Parameters
  ----------
  method_id : :class:`fiqat.types.MethodIdStr`
    Specifies the method.
  input_items : Union[Any, Iterable[Any]]
    One or multiple items that should be processed.
  tqdm_config : Union[bool, tuple, list, dict]
    If not ``False`` and if the input is iterable, :meth:`fiqat.term.tqdm` will be called to show a progress bar.
    The default progress bar description is the API function name followed by the ``method_id``.
  **kwargs
    Keyword arguments passed to the method.

  Returns
  -------
  Union[Any, Iterable[Any]]
    One or multiple output items. See the :mod:`fiqat.main_api` module description for general method output behavior.
  """
  method_id = MethodId(method_id)
  method_type_str = method_id.parts[0]
  method_type = MethodType(method_type_str)
  main_api_function = main_api_functions_by_method_type[method_type]
  return main_api_function(method_id, input_items, tqdm_config, **kwargs)


def _check_iterable_mode(item_or_items) -> tuple[bool, Iterable]:
  return (True, item_or_items) if (isinstance(item_or_items, Iterable) and
                                   not isinstance(item_or_items, (dict, tuple))) else (False, [item_or_items])


def _run_method_processing(method_id: MethodIdStr, kwargs: dict, method_type_str: str, *process_args):
  method_id = MethodId(method_id)
  method_registry_entry = registry.get_method(method_id, MethodType(method_type_str))
  config = _assemble_method_config(method_registry_entry, kwargs)
  result = method_registry_entry['process'](method_registry_entry, config, *process_args)
  return result


def _return_result(iterable_input: bool, tqdm_config: Union[bool, tuple, list, dict], tqdm_desc: str, input_items,
                   result: Any) -> Any:
  if iterable_input:
    if tqdm_config is not False:
      result = term.tqdm(result, input_items=input_items, tqdm_config=tqdm_config, desc=tqdm_desc)
    return result
  else:
    return next(result)


def _assemble_method_config(method_registry_entry: MethodRegistryEntry, kwargs):
  default_config = method_registry_entry.get('default_config', {})
  config = type(default_config)(**{**default_config, **kwargs})
  return config
