"""This is a complete example that goes through all steps required to compute an
"Error versus Discard Characteristic" (EDC) plot that compares Face Image Quality Assessment (FIQA) algorithms
relative to how effectively they reduce a face recognition error value (here the FNMR).

Example usage:
python create_edc_plot.py -i /path/to/input_images/ -o /path/to/example_output/
"""

# Standard imports:
from typing import Iterable, Optional
from pathlib import Path, PosixPath
import itertools
import argparse

# External imports:
import numpy as np  # Only used for the _custom_fiqa_method example part.
import cv2  # Only used for the _custom_fiqa_method example part.
import plotly.graph_objects as go

# Toolkit import:
import fiqat


def main():
  # Fixed example settings:
  accepted_input_extensions = {'.png', '.jpg', '.jpeg', '.jp2', '.jxl', '.bmp'}

  prep_image_size = fiqat.ImageSize(224, 224)  # The size of preprocessed image output.

  fiqa_configs = [  # Face Image Quality Assessment (FIQA) configurations that will be used.
      {
          'method_id': fiqat.MethodId('fiqa/custom_test'),  # A custom example method registered below.
          'key': 'custom_test',
      },
      {
          'method_id': fiqat.MethodId('fiqa/faceqnet'),  # Provided by the toolkit.
          'key': 'faceqnet_v0',
          'process_kwargs': {
              'model_type': 'FaceQnet-v0'
          },
      },
      {
          'method_id': fiqat.MethodId('fiqa/faceqnet'),  # Provided by the toolkit.
          'key': 'faceqnet_v1',
          'process_kwargs': {
              'model_type': 'FaceQnet-v1'
          },
      },
      # {
      #     'method_id': fiqat.MethodId('fiqa/magface'),  # Provided by the toolkit.
      #     'key': 'magface',
      # },
  ]

  # Register a custom example method for quality assessment:
  def _custom_fiqa_method(
      _method_registry_entry: fiqat.FiqaMethodRegistryEntry,
      _config: dict,
      input_images: Iterable[fiqat.InputImage],
  ):
    """To roughly assess the (face) image quality, this example method first blurs the image,
    and then compares the distance between the original and the blurred image data.
    """
    for input_image in input_images:
      # First load the image from a file and/or convert the image data to BGR channels:
      image_data = fiqat.load_input_image(input_image, fiqat.ImageChannelType.BGR).image
      # Then convert the image pixel values to floating point values in the range [0,1]:
      if image_data.dtype not in {np.float32, np.float64}:
        iinfo = np.iinfo(image_data.dtype)
        assert iinfo.min == 0, ("iinfo.min not 0", iinfo)
        image_data = image_data / iinfo.max
      # Then create a blurred version of the image data:
      blurred_image_data = cv2.blur(  # pylint: disable=no-member
          image_data,
          ksize=(3, 3),
          borderType=cv2.BORDER_REFLECT_101,  # pylint: disable=no-member
      )
      # Then compute a quality score as the mean distance between the blurred image data and the original image data.
      quality_score = np.mean(np.abs(image_data - blurred_image_data))
      # The output range is [0,1], and higher values imply better quality (in terms of less blur):
      yield quality_score

  fiqat.registry.register_method(
      fiqat.FiqaMethodRegistryEntry(
          method_id=fiqat.MethodId('fiqa/custom_test'),
          process=_custom_fiqa_method,
          default_config={},
          quality_score_range=fiqat.QualityScoreRange(0, 1),
      ))

  # Parse command-line arguments:
  parser = argparse.ArgumentParser(prog='EDC example.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '-i',
      '--input',
      type=str,
      required=True,
      help=('A path to a directory containing input face images.'
            ' Note that the directory name will be used as the subject identifier for each image.'
            f' Accepted formats are (case insensitive): {accepted_input_extensions}'))
  parser.add_argument(
      '-o',
      '--output',
      type=str,
      required=True,
      help='Both computed intermediate data and the final EDC plot will be saved to this directory.')
  parser.add_argument(
      '-c',
      '--toolkit_config',
      type=str,
      help='Optional path to the local toolkit .toml configuration file. See the fiqat.config documentation.')
  parser.add_argument(
      '-q',
      '--similarity_score_quantile',
      type=float,
      default=0.05,
      help='Used to compute the similarity score threshold for the EDC plot.')
  parser.add_argument(
      '-p',
      '--pauc_discard_fraction_limit',
      type=float,
      default=0.20,
      help='Used to compute pAUC values on the EDC curves.')
  parser.add_argument(
      '-l',
      '--input_image_limit',
      type=int,
      help=('If specified, the number of input image paths will be limited to this number.'
            ' The paths will be sorted to select the input up to this limit.'
            ' This option is only relevant if you want to conduct a quick test run'
            ' while there are many images under the --input directory.'))
  args = parser.parse_args()

  input_image_dir = Path(args.input)
  output_dir = Path(args.output)

  similarity_score_quantile = args.similarity_score_quantile
  pauc_discard_fraction_limit = args.pauc_discard_fraction_limit

  input_image_limit = args.input_image_limit
  assert (input_image_limit is None) or (input_image_limit > 0), 'If --input_image_limit is set it must be above 0'

  if args.toolkit_config is not None:
    fiqat.config.load_config_data(args.toolkit_config)

  # Gather input image paths:
  input_image_paths = [
      path for path in fiqat.term.tqdm(input_image_dir.rglob('*'), desc='Gather potential input image paths')
      if path.is_file() and (path.suffix.lower() in accepted_input_extensions)
  ]
  print(fiqat.term.colored('Found input image paths:', 'cyan'), len(input_image_paths))
  if input_image_limit is not None:
    input_image_paths.sort()
    input_image_paths = input_image_paths[:input_image_limit]
    print(fiqat.term.colored('--input_image_limit is set to:', 'cyan'), input_image_limit)
  if len(input_image_paths) == 0:
    fiqat.term.cprint('No input images found, aborting program.', 'red')
    return

  # Establish output paths:
  prep_image_dir = output_dir / 'prep_images'
  storage_sqlite_path = output_dir / 'storage.sqlite3'
  plot_output_path = output_dir / 'EDC-example.html'
  print(fiqat.term.colored('Preprocessed images will be stored under:', 'cyan'), prep_image_dir)
  print(fiqat.term.colored('Intermediate data (e.g. quality scores) will be stored at:', 'cyan'), storage_sqlite_path)
  print(fiqat.term.colored('The final plot will be stored at:', 'cyan'), plot_output_path)

  # Create dictionaries for image data based on the input image paths,
  # using the files' directory names as the subject IDs:
  image_dicts = [{
      'subject': input_path.parent.name,
      'path': input_path.relative_to(input_image_dir),
  } for input_path in input_image_paths]

  # Only keep subjects with at least two images, since this example will look at mated comparisons
  # (i.e. comparisons of images from the same subject):
  subject_image_lists = _create_subject_image_lists(image_dicts)
  subject_image_lists = _filter_subject_image_lists(subject_image_lists)
  print(
      fiqat.term.colored('Used subject count, with at least two images each, before face detection:', 'cyan'),
      len(subject_image_lists))
  image_dicts = _combine_subject_image_lists(subject_image_lists)
  print(fiqat.term.colored('Used image count before face detection:', 'cyan'), len(image_dicts))

  # Initialize the toolkit's storage helper that this example will use to store intermediate data,
  # so that this data doesn't have to be re-computed for every run of the program
  # (note that your code could use a different approach to store data, this is not strictly required by the toolkit):
  storage = fiqat.StorageSqlite()
  storage.open(storage_sqlite_path)
  _load_stored_data(storage, image_dicts, input_image_dir)

  # Detect faces in the input images:
  _detect_faces(storage, image_dicts)

  # Select a primary face for each image's face detector output:
  _estimate_primary_face(storage, image_dicts)

  # Filter out image_dicts for which no (primary) face was detected:
  image_dicts_prior_len = len(image_dicts)
  image_dicts = [image_dict for image_dict in image_dicts if image_dict['primary_face_estimate'].index is not None]
  filter_count = image_dicts_prior_len - len(image_dicts)
  print(
      fiqat.term.colored('Images without any detected face:', 'cyan'),
      filter_count if filter_count == 0 else fiqat.term.colored(str(filter_count), 'yellow'))

  # Recreate and re-filter the subject_image_lists with the remaining image_dicts:
  subject_image_lists = _create_subject_image_lists(image_dicts)
  subject_image_lists = _filter_subject_image_lists(subject_image_lists)
  print(
      fiqat.term.colored('Used subject count, with at least two images each, after face detection:', 'cyan'),
      len(subject_image_lists))
  image_dicts = _combine_subject_image_lists(subject_image_lists)
  print(fiqat.term.colored('Used image count after face detection:', 'cyan'), len(image_dicts))

  # Preprocess images:
  _preprocess_images(image_dicts, prep_image_dir, prep_image_size)

  # Assess image quality (using multiple methods):
  _assess_quality(storage, image_dicts, fiqa_configs)

  # Compute face recognition feature vectors (for one method):
  _compute_face_recognition_features(storage, image_dicts)

  # Close the storage helper since no new intermediate data will be computed from this point on:
  storage.close()

  # Create the EDC plot:
  _create_edc_plot(
      image_dicts,
      subject_image_lists,
      fiqa_configs,
      plot_output_path,
      similarity_score_quantile,
      pauc_discard_fraction_limit,
  )


def _create_subject_image_lists(image_dicts: list) -> dict:
  """Creates lists of image_dicts for each 'subject' ID."""
  subject_image_lists = {}
  for image_dict in image_dicts:
    subject_image_lists.setdefault(image_dict['subject'], []).append(image_dict)
  return subject_image_lists


def _filter_subject_image_lists(subject_image_lists: dict) -> dict:
  """Only keeps subjects with at least two images, since this example will look at mated comparisons
  (i.e. comparisons of images from the same subject).
  """
  return {
      subject_id: subject_image_list
      for subject_id, subject_image_list in subject_image_lists.items()
      if len(subject_image_list) >= 2
  }


def _combine_subject_image_lists(subject_image_lists: dict) -> list:
  """Collects the image dictionaries (usually after filtering the subject_image_lists)."""
  image_dicts = []
  for subject_image_list in subject_image_lists.values():
    image_dicts.extend(subject_image_list)
  return image_dicts


def _load_stored_data(storage: fiqat.StorageBase, image_dicts: list, input_image_dir: Path):
  """Load previously stored intermediate data."""
  image_dicts_by_key = {image_dict['path']: image_dict for image_dict in image_dicts}
  for loaded_key, loaded_data in fiqat.term.tqdm(storage.load_items('image_dicts'), desc='Load already computed data'):
    # Only restore loaded data if there is a image_dict for the matching Path key:
    image_dict_key = Path(loaded_key)
    image_dict = image_dicts_by_key.get(image_dict_key, None)
    if image_dict is not None:
      _restore_loaded_data_to_image_dict(loaded_data, image_dict, input_image_dir)

  # Restore the full input image paths for the image_dicts:
  for image_dict in image_dicts:
    full_input_path = image_dict['full_input_path'] = input_image_dir / image_dict['path']
    if 'face_detector_output' in image_dict:
      fdo_input_image = image_dict['face_detector_output']['input_image']
      assert fdo_input_image == full_input_path, (fdo_input_image, full_input_path)


def _detect_faces(storage: fiqat.StorageBase, image_dicts: list):
  """Detect faces in the input images."""
  for image_dict, face_detector_output in _process(image_dicts, 'fd/scrfd', 'full_input_path',
                                                   ('face_detector_output',)):
    image_dict['face_detector_output'] = face_detector_output
    _update_image_dict(storage, image_dict)
  storage.save()


def _estimate_primary_face(storage: fiqat.StorageBase, image_dicts: list):
  """Select a primary face for each image's face detector output."""
  for image_dict, primary_face_estimate in _process(image_dicts, 'pfe/sccpfe', 'face_detector_output',
                                                    ('primary_face_estimate',)):
    image_dict['primary_face_estimate'] = primary_face_estimate
    _update_image_dict(storage, image_dict)
  storage.save()


def _preprocess_images(image_dicts: list, prep_image_dir: Path, prep_image_size: fiqat.ImageSize):
  """Preprocess images."""
  # Check for existing preprocessed images:
  for image_dict in image_dicts:
    image_dict['prep_image'] = (prep_image_dir / image_dict['path']).with_suffix('.png')
    if image_dict['prep_image'].exists():
      image_dict['prep_image_exists'] = True
    elif 'prep_image_exists' in image_dict:
      del image_dict['prep_image_exists']

  # Preprocess images:
  for image_dict, prep_image in _process(
      image_dicts, 'prep/simt', 'primary_face_estimate', ('prep_image_exists',), image_size=prep_image_size):
    fiqat.save_image(image_dict['prep_image'], prep_image)
    image_dict['prep_image_exists'] = True


def _assess_quality(storage: fiqat.StorageBase, image_dicts: list, fiqa_configs: list):
  """Assess image quality (using multiple methods)."""
  for fiqa_config in fiqa_configs:
    for image_dict, quality_score in _process(image_dicts, fiqa_config['method_id'], 'prep_image',
                                              ('quality_scores', fiqa_config['key']),
                                              **fiqa_config.get('process_kwargs', {})):
      image_dict.setdefault('quality_scores', {})[fiqa_config['key']] = quality_score
      _update_image_dict(storage, image_dict)
    storage.save()


def _compute_face_recognition_features(storage: fiqat.StorageBase, image_dicts: list):
  """Compute face recognition feature vectors (for one method)."""
  for image_dict, feature_vector in _process(image_dicts, 'fr/arcface', 'prep_image', ('feature_vector',)):
    image_dict['feature_vector'] = feature_vector
    _update_image_dict(storage, image_dict)
  storage.save()


def _is_output_already_computed(data: dict, output_key_path: Optional[tuple]) -> bool:
  """Checks whether `data` already contains a previously computed value under the `output_key_path`."""
  if output_key_path is None:
    return False
  for key_part in output_key_path:
    if key_part in data:
      data = data[key_part]
    else:
      return False
  return True


def _process(image_dicts: Iterable, method_id_str: str, image_dict_key: str, output_key_path: Optional[tuple],
             **kwargs) -> Iterable:
  """This helper generator function will call the toolkit's main API via the general fiqat.process function,
  but only for the image_dicts that still need to be processed.
  """
  remaining_image_dicts = [
      image_dict for image_dict in image_dicts if not _is_output_already_computed(image_dict, output_key_path)
  ]
  input_items = [image_dict[image_dict_key] for image_dict in remaining_image_dicts]
  if len(input_items) == 0:
    return []
  else:
    return zip(remaining_image_dicts, fiqat.process(method_id_str, input_items, **kwargs))


def _restore_loaded_data_to_image_dict(loaded_data: dict, image_dict: dict, input_image_dir: Path):
  """Restore data loaded from storage to the corresponding image_dict:"""
  for key in ['quality_scores', 'feature_vector']:
    if key in loaded_data:
      image_dict[key] = loaded_data[key]
  if 'subject' in loaded_data:
    assert image_dict['subject'] == loaded_data['subject'], (
        'Subject changed from loaded data',
        ('new', image_dict['subject']),
        ('loaded', loaded_data['subject']),
    )
  if 'face_detector_output' in loaded_data:
    face_detector_output = loaded_data['face_detector_output']
    face_detector_output['input_image'] = input_image_dir / face_detector_output['input_image']
    image_dict['face_detector_output'] = face_detector_output
    if 'primary_face_estimate.index' in loaded_data:
      image_dict['primary_face_estimate'] = fiqat.PrimaryFaceEstimate(loaded_data['face_detector_output'],
                                                                      loaded_data['primary_face_estimate.index'])


def _update_image_dict(storage: fiqat.StorageBase, image_dict: dict):
  """This helper function will update the intermediate data storage with new image_dict data."""
  item_key = str(PosixPath(image_dict['path']))

  # Only store the data that needs to be stored:
  data_to_be_stored = {}
  for key in ['subject', 'quality_scores', 'feature_vector']:
    if key in image_dict:
      data_to_be_stored[key] = image_dict[key]
  if 'face_detector_output' in image_dict:
    face_detector_output = data_to_be_stored['face_detector_output'] = image_dict['face_detector_output']
    fdo_input_image = face_detector_output['input_image']
    face_detector_output['input_image'] = item_key
  if 'primary_face_estimate' in image_dict:
    data_to_be_stored['primary_face_estimate.index'] = image_dict['primary_face_estimate'].index

  storage.update_item('image_dicts', item_key, data_to_be_stored)

  if 'face_detector_output' in image_dict:
    face_detector_output['input_image'] = fdo_input_image


def _create_edc_plot(
    image_dicts: list,
    subject_image_lists: dict,
    fiqa_configs: list,
    plot_output_path: Path,
    similarity_score_quantile: float,
    pauc_discard_fraction_limit: float,
):
  """Create the EDC plot"""
  # Create an empty fiqat.EdcSample for each image_dict:
  for image_dict in image_dicts:
    image_dict['edc_sample'] = fiqat.EdcSample(quality_score=None)

  # Create a fiqat.EdcSamplePair for every mated sample combination, computing a similarity score for each
  # (i.e. compare every image of a subject against all other images of the same subject):
  edc_sample_pairs = {}
  for subject_image_list in subject_image_lists.values():
    for image_dict_1, image_dict_2 in itertools.combinations(subject_image_list, 2):
      path_1, path_2 = image_dict_1['path'], image_dict_2['path']
      if path_2 < path_1:
        path_1, path_2 = path_2, path_1
        image_dict_1, image_dict_2 = image_dict_2, image_dict_1
      sample_pair_key = (path_1, path_2)

      feature_vector_pair: fiqat.FeatureVectorPair = (image_dict_1['feature_vector'], image_dict_2['feature_vector'])
      similarity_score: fiqat.SimilarityScore = fiqat.compute_comparison_scores('csc/arcface', feature_vector_pair)

      assert sample_pair_key not in edc_sample_pairs
      edc_sample_pairs[sample_pair_key] = fiqat.EdcSamplePair(
          samples=(image_dict_1['edc_sample'], image_dict_2['edc_sample']),
          similarity_score=similarity_score,
      )
  edc_sample_pairs = list(edc_sample_pairs.values())

  # Create the EDC plot figure:
  figure = go.Figure()
  for fiqa_config in fiqa_configs:  # Each iteration computes a curve for one of the FIQA configurations.
    # Set the fiqat.EdcSample.quality_score for the FIQA configurations:
    for image_dict in image_dicts:
      image_dict['edc_sample']['quality_score'] = image_dict['quality_scores'][fiqa_config['key']]
    # Compute the EDC curve:
    edc_output = fiqat.compute_edc(
        error_type=fiqat.EdcErrorType.FNMR,  # FNMR because mated sample pairs are compared.
        sample_pairs=edc_sample_pairs,
        similarity_score_quantile=similarity_score_quantile,
    )
    # Then also compute a pAUC value on the computed EDC curve, by which the FIQA configurations could be ranked:
    pauc_value = fiqat.compute_edc_pauc(edc_output, pauc_discard_fraction_limit)
    # Add the EDC curve to the plot:
    label = f"{fiqa_config['key']} - [0,{pauc_discard_fraction_limit:.2f}]-pAUC value: {pauc_value:.5f}"
    figure.add_trace(
        go.Scatter(
            x=edc_output['discard_fractions'],
            y=edc_output['error_fractions'],  # I.e. here the FNMR.
            name=label,  # For real experiment plots you may want to map this ID to more readable names.
            opacity=0.7,  # Unimportant.
            # "Stepwise" interpolation is recommended, since that reflects how comparisons are
            # actually discarded during the EDC computation. For plotly that can be set via line_shape='hv':
            line_shape='hv',
        ))
  # Finalize & save the plot in the output directory:
  figure.update_layout(
      title='EDC example',
      xaxis_title='Discarded',
      yaxis_title='FNMR',
  )
  plot_output_path.parent.mkdir(parents=True, exist_ok=True)
  figure.write_html(plot_output_path)


if __name__ == '__main__':
  main()
