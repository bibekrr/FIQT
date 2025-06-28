"""This example will run all available methods of these types

- fiqat.MethodType.FACE_DETECTOR
- fiqat.MethodType.FACE_IMAGE_QUALITY_ASSESSMENT_ALGORITHM
- fiqat.MethodType.FACE_RECOGNITION_FEATURE_EXTRACTOR

using a provided example input image, and check that their output remains consistent (i.e. "deterministic") across
multiple executions with this same input.
This example will also output the mean execution time for each tested method,
but note that this can include the time required to load any required models.

Some of the code is identical to the load_all_methods.py example.
"""

# Standard imports:
import argparse
from pathlib import Path
import time
from enum import Enum
import traceback

# External imports:
import numpy as np

# Toolkit import:
import fiqat


def main():
  # Parse command-line arguments:
  parser = argparse.ArgumentParser(prog='Method check.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-i', '--input', type=str, required=True, help='A path to a test face image.')
  parser.add_argument(
      '-c',
      '--batch_count',
      type=int,
      default=2,
      help='Number of "batches" to be executed per method. Each "batch" corresponds to an invocation of fiqat.process.')
  parser.add_argument('-s', '--batch_size', type=int, default=50, help='Number of images per "batch".')
  args = parser.parse_args()

  input_image_path = Path(args.input)
  batch_count = args.batch_count
  batch_size = args.batch_size

  assert batch_count > 0, '--batch_count must be above 0'
  assert batch_size > 0, '--batch_size must be above 0'

  # Load the input image in advance, so that it won't be loaded separately for each method execution:
  input_image: fiqat.InputImage = fiqat.load_input_image(input_image_path, fiqat.ImageChannelType.BGR)

  # Method configurations that should be tested (if they are available), by fiqat.MethodType:
  methods_to_be_tested = {
      fiqat.MethodType.FACE_DETECTOR: [
          {
              'method_id': fiqat.MethodId('fd/retinaface'),
          },
          {
              'method_id': fiqat.MethodId('fd/mtcnn'),
          },
          {
              'method_id': fiqat.MethodId('fd/scrfd'),
          },
          {
              'method_id': fiqat.MethodId('fd/dlib'),
          },
      ],
      fiqat.MethodType.FACE_IMAGE_QUALITY_ASSESSMENT_ALGORITHM: [
          {
              'method_id': fiqat.MethodId('fiqa/crfiqa'),
              'process_kwargs': {
                  'model_type': 'CR-FIQA(S)'
              },
          },
          {
              'method_id': fiqat.MethodId('fiqa/crfiqa'),
              'process_kwargs': {
                  'model_type': 'CR-FIQA(L)'
              },
          },
          {
              'method_id': fiqat.MethodId('fiqa/faceqnet'),
              'process_kwargs': {
                  'model_type': 'FaceQnet-v0'
              },
          },
          {
              'method_id': fiqat.MethodId('fiqa/faceqnet'),
              'process_kwargs': {
                  'model_type': 'FaceQnet-v1'
              },
          },
          {
              'method_id': fiqat.MethodId('fiqa/magface'),
          },
      ],
      fiqat.MethodType.FACE_RECOGNITION_FEATURE_EXTRACTOR: [{
          'method_id': fiqat.MethodId('fr/arcface'),
      },],
  }

  # Try to load the methods categorized by MethodType and print information:
  for method_type, method_configs in methods_to_be_tested.items():
    fiqat.term.cprint(method_type, 'cyan')
    for method_config in method_configs:
      method_id = method_config['method_id']
      process_kwargs = method_config.get('process_kwargs', {})
      registry_entry: fiqat.RegistryEntry = fiqat.registry.registry_data[method_id]
      print(f"- Trying to check {fiqat.term.colored(method_id, 'cyan')}"
            f" with config {fiqat.term.colored(str(process_kwargs), 'cyan')}:")
      try:
        # Try to load the method:
        method_registry_entry: fiqat.MethodRegistryEntry = fiqat.registry.get_method(method_id)
        # Method could be loaded:
        fiqat.term.cprint(f"  - Status: {registry_entry['status']}", 'cyan')
        assert registry_entry['method_entry'] == method_registry_entry  # Just for information.
        fiqat.term.cprint(f"  - Default config: {method_registry_entry.get('default_config', {})}", 'cyan')
        is_available = True
      except RuntimeError:
        # Method couldn't be loaded:
        fiqat.term.cprint(f"  - Status: {registry_entry['status']}", 'yellow')
        fiqat.term.cprint(f"  - Info: {registry_entry['info']}", 'yellow')
        is_available = False
      if is_available:
        # Check the method:
        _check_method(method_id, input_image, batch_count, batch_size, process_kwargs=process_kwargs)
      print()
      print()


def _check_method(method_id: fiqat.MethodId, input_image: fiqat.InputImage, batch_count: int, batch_size: int,
                  process_kwargs: dict):
  # Execute method:
  start_time_ns = time.monotonic_ns()
  results = []
  for i in fiqat.term.tqdm(range(batch_count), desc=f'Batch: {method_id}'):
    input_batch = [input_image] * batch_size
    try:
      for result in fiqat.process(method_id, input_batch, **process_kwargs):
        results.append(result)
    except Exception:  # pylint: disable=broad-except
      fiqat.term.cprint(f'Execution raised an exception (at input batch {i}):\n\n{traceback.format_exc()}', 'red')
      return
  end_time_ns = time.monotonic_ns()

  # Print mean execution time:
  total_time_ns = end_time_ns - start_time_ns
  mean_time_ns = total_time_ns / (batch_size * batch_count)
  mean_time_ms = mean_time_ns / 1_000_000
  fiqat.term.cprint(f"  - Mean execution time: {mean_time_ms}ms", 'cyan')

  # Check results for consistency:
  all_equal = _check_result_consistency(results)
  if all_equal:
    fiqat.term.cprint('  - All results for the same input were identical.', 'green')


def _check_result_consistency(results: list) -> bool:
  for i, result1 in enumerate(results):
    for result2 in results[i + 1:]:
      if not _is_equal(result1, result2):
        return False
  return True


def _is_equal(value1, value2, location=None) -> bool:
  if type(value1) != type(value2):  # pylint: disable=unidiomatic-typecheck
    _print_difference_info('Type', type(value1), type(value2), location)
    return False
  if isinstance(value1, (int, float, str, Path, Enum, np.float32, np.float64)):
    if value1 != value2:
      _print_difference_info('Value', value1, value2, location)
      return False
    else:
      return True
  elif isinstance(value1, (list, tuple)):
    if len(value1) != len(value2):
      _print_difference_info('List or tuple length', len(value1), len(value2), location)
      return False
    if location is None:
      location = []
    for i, (sub1, sub2) in enumerate(zip(value1, value2)):
      if not _is_equal(sub1, sub2, location + [i]):
        return False
    return True
  elif isinstance(value1, dict):
    if len(value1) != len(value2):
      _print_difference_info('Dict length', len(value1), len(value2), location)
      return False
    keys1 = set(value1.keys())
    keys2 = set(value2.keys())
    if keys1 != keys2:
      _print_difference_info('Different dict keys', keys1 - keys2, keys2 - keys1, location)
      return False
    if location is None:
      location = []
    for key in keys1:
      if not _is_equal(value1[key], value2[key], location + [key]):
        return False
    return True
  elif isinstance(value1, np.ndarray):
    if not np.array_equal(value1, value2):
      _print_difference_info('np.ndarray content', value1, value2, location)
      return False
    else:
      return True
  else:
    raise NotImplementedError('Unsupported type', type(value1))


def _print_difference_info(difference_type, difference_value1, difference_value2, location=None):
  fiqat.term.cprint('  - WARNING: Different results for the same input detected!', 'red')
  if location is not None:
    fiqat.term.cprint(f'    - Location in the data structure: {location}', 'red')
  fiqat.term.cprint(f'    - Difference: {difference_type}', 'red')
  fiqat.term.cprint(f'    - Value 1: {difference_value1}', 'red')
  fiqat.term.cprint(f'    - Value 2: {difference_value2}', 'red')


if __name__ == '__main__':
  main()
