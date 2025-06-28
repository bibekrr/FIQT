""""Error versus Discard Characteristic" (EDC) functionality.
See e.g.
`"Considerations on the Evaluation of Biometric Quality Assessment Algorithms"
<https://ieeexplore.ieee.org/document/10330743>`_.
"""

# Standard imports:
from typing import Optional, Union, TypedDict, Callable

# External imports:
import numpy as np

# Local imports:
from .types import StrEnum, QualityScore, SimilarityScore, SimilarityScores


class EdcSample(TypedDict):
  """The input data for one sample that is required to compute an EDC curve,
  which is just the sample's :class:`fiqat.types.QualityScore`.
  """
  quality_score: QualityScore
  """The sample's :class:`fiqat.types.QualityScore`.
  In context of the EDC curve computation, a higher quality score is assumed to indicate higher utility.
  For a typical EDC setup, samples with lower quality will therefore be discarded first.
  """


class EdcSamplePair(TypedDict):
  """The input data for one sample pair that is required to compute an EDC curve."""
  samples: tuple[EdcSample, EdcSample]
  """Data for the individual samples (:class:`EdcSample`)."""
  similarity_score: SimilarityScore
  """The :class:`fiqat.types.SimilarityScore` for the sample pair."""


class EdcErrorType(StrEnum):
  """An error type for an EDC plot. The error is typically plotted on the Y-axis."""
  FNMR = 'FNMR'
  """The
  `FNMR (False Non-Match Rate) <https://www.iso.org/obp/ui/#iso:std:iso-iec:2382:-37:ed-3:v1:en:term:37.09.11>`_.
  """
  FMR = 'FMR'
  """The
  `FMR (False Match Rate) <https://www.iso.org/obp/ui/#iso:std:iso-iec:2382:-37:ed-3:v1:en:term:37.09.09>`_.
  """


PairQualityScoreFunction = Callable[[QualityScore, QualityScore], QualityScore]
"""A function that combines the two quality scores of the :class:`EdcSamplePair.samples`
(:class:`EdcSample.quality_score`) into a pairwise quality score.

Typically this should be the ``min`` function, since this will correspond to discarding samples (and their sample pairs)
in the ascending order of their individual quality scores.
See also
`"Considerations on the Evaluation of Biometric Quality Assessment Algorithms" <https://arxiv.org/abs/2303.13294>`_
section I.A.
"""


class _EdcInputBase(TypedDict):
  """Input required to compute an EDC curve.
  Note that this is just a base class for :class:`_EdcInputSamplePair` and :class:`_EdcInputNumpy`,
  either of which specify the complete input information that is required.
  """
  error_type: EdcErrorType
  """The :class:`EdcErrorType` of the EDC curve."""
  similarity_score_threshold: SimilarityScore
  """The :class:`fiqat.types.SimilarityScore` threshold used to decide which comparisons are errors or not,
  which also depends on the :class:`EdcErrorType`.
  """
  pair_quality_score_function: Optional[PairQualityScoreFunction]
  """The :class:`PairQualityScoreFunction`.
  If this isn't specified, it will be ``min``, which is typically the correct function to use
  (see the :class:`PairQualityScoreFunction` description).
  """


class _EdcInputSamplePair(_EdcInputBase):
  """Input required to compute an EDC curve,
  in the form of a :class:`EdcSamplePair` list in addition to the :class:`_EdcInputBase` data.
  """
  sample_pairs: list[EdcSamplePair]


class _EdcInputNumpy(_EdcInputBase):
  """Input required to compute an EDC curve,
  in the form of two numpy arrays
  (:class:`_EdcInputNumpy.similarity_scores` and :class:`_EdcInputNumpy.pair_quality_scores`)
  in addition to the :class:`_EdcInputBase` data.
  """
  similarity_scores: np.ndarray
  """A numpy array containing :class:`fiqat.types.SimilarityScore` values.
  Each index corresponds to a :class:`EdcSamplePair` and the :class:`_EdcInputNumpy.pair_quality_scores` value at the
  same index.
  """
  pair_quality_scores: np.ndarray
  """A numpy array containing pairwise :class:`QualityScore` values (produced by a :class:`PairQualityScoreFunction`).
  Each index corresponds to a :class:`EdcSamplePair` and the :class:`_EdcInputNumpy.similarity_scores` value at the
  same index.

  These pairwise quality scores must be sorted in ascending order.
  """


_EdcInput = Union[_EdcInputSamplePair, _EdcInputNumpy]
"""Input required to compute an EDC curve.
Either a :class:`_EdcInputSamplePair` or a :class:`_EdcInputNumpy`.
"""


class EdcOutput(TypedDict):
  """The output of :meth:`compute_edc` that represents a computed EDC curve.
  
  Only the :class:`EdcOutput.error_type`, :class:`EdcOutput.error_fractions`, and
  the :class:`EdcOutput.discard_fractions` are required to plot the EDC curve.
  """
  error_type: EdcErrorType
  """The :class:`EdcErrorType` of the EDC curve."""
  error_fractions: np.ndarray
  """The error fraction values of the EDC curve.
  Typically plotted on the Y-axis.
  At each index the value corresponds to the :class:`EdcOutput.discard_fractions` value at the same index.
  """
  discard_fractions: np.ndarray
  """The discard fraction values of the EDC curve.
  Typically plotted on the X-axis.
  At each index the value corresponds to the :class:`EdcOutput.error_fractions` value at the same index.
  """
  error_counts: np.ndarray
  """The discrete integer counts of remaining errors.
  This is used to compute the :class:`EdcOutput.error_fractions`.
  At each index the value corresponds to the :class:`EdcOutput.discard_counts` value at the same index.
  """
  discard_counts: np.ndarray
  """The discrete integer discard counts.
  This is used to compute the :class:`EdcOutput.discard_fractions`.
  At each index the value corresponds to the :class:`EdcOutput.error_counts` value at the same index.
  """
  comparison_count: int
  """The total number of comparisons."""


def compute_edc(
    error_type: EdcErrorType,
    sample_pairs: list[EdcSamplePair],
    similarity_score_threshold: Optional[SimilarityScore] = None,
    similarity_score_quantile: Optional[float] = None,
    starting_error: Optional[float] = None,
    pair_quality_score_function: PairQualityScoreFunction = min,
) -> EdcOutput:
  """Computes an EDC curve.

  Parameters
  ----------
  error_type : EdcErrorType
    The :class:`EdcErrorType` of the EDC curve.
  sample_pairs : list[EdcSamplePair]
    An :class:`EdcSamplePair` list used to compute the EDC curve.
    This specifies the similarity scores and the quality scores.
  similarity_score_threshold : Optional[:class:`fiqat.types.SimilarityScore`]
    The :class:`fiqat.types.SimilarityScore` threshold used to decide which comparisons are errors or not,
    which also depends on the :class:`EdcErrorType`.
    The parameters ``similarity_score_quantile`` or ``starting_error`` can be used instead.
  similarity_score_quantile : Optional[float]
    If set, the ``similarity_score_threshold`` will be computed as
    `np.quantile(similarity_scores, similarity_score_quantile)
    <https://numpy.org/doc/stable/reference/generated/numpy.quantile.html>`_.
    The parameters ``similarity_score_threshold`` or ``starting_error`` can be used instead.
  starting_error : Optional[float]
    If set, the ``similarity_score_threshold`` will be computed as
    `np.quantile(similarity_scores, starting_error)
    <https://numpy.org/doc/stable/reference/generated/numpy.quantile.html>`_ for :class:`EdcErrorType.FNMR`
    or `np.quantile(similarity_scores, 1 - starting_error)
    <https://numpy.org/doc/stable/reference/generated/numpy.quantile.html>`_ for :class:`EdcErrorType.FMR`.
    It is called "starting error" because the actual starting error at the 0% discard fraction will approximate
    this value, depending on the given similarity score distribution.
    The parameters ``similarity_score_threshold`` or ``similarity_score_quantile`` can be used instead.
  pair_quality_score_function : :class:`PairQualityScoreFunction`
    The function used to get pairwise quality scores for each each :class:`EdcSamplePair`
    from the two samples' individual quality scores.
    If this isn't specified, it will be ``min``, which is typically the correct function to use
    (see the :class:`PairQualityScoreFunction` description).

  Returns
  -------
  EdcOutput
    The data for the computed EDC curve.
  """
  edc_input = _EdcInputSamplePair(
      error_type=error_type,
      similarity_score_threshold=similarity_score_threshold,
      sample_pairs=sample_pairs,
      pair_quality_score_function=pair_quality_score_function,
  )
  edc_input_numpy = _get_edc_input_numpy(edc_input)

  assert 1 == sum(
      (0 if value is None else 1 for value in (similarity_score_threshold, similarity_score_quantile, starting_error)
      )), ('Excactly one of the parameters similarity_score_threshold, similarity_score_quantile, or starting_error'
           ' has to be specified.')
  if starting_error is not None:
    similarity_score_quantile = starting_error
    if error_type == EdcErrorType.FMR:
      similarity_score_quantile = 1 - similarity_score_quantile
  if similarity_score_threshold is None:
    similarity_score_threshold = np.quantile(edc_input_numpy['similarity_scores'], similarity_score_quantile)
    edc_input_numpy['similarity_score_threshold'] = similarity_score_threshold

  return _compute_edc(
      error_type=edc_input_numpy['error_type'],
      pair_quality_scores=edc_input_numpy['pair_quality_scores'],
      similarity_scores=edc_input_numpy['similarity_scores'],
      similarity_score_threshold=edc_input_numpy['similarity_score_threshold'],
  )


def _get_edc_input_numpy(edc_input: _EdcInput) -> _EdcInputNumpy:
  if ('pair_quality_scores' in edc_input) or ('pair_similarity_scores' in edc_input):
    assert 'sample_pairs' not in edc_input
    assert ('pair_quality_scores' in edc_input) and ('pair_similarity_scores' in edc_input)
    return edc_input
  else:
    assert 'sample_pairs' in edc_input
    sample_pairs = edc_input['sample_pairs']

    similarity_scores = np.zeros(len(sample_pairs), dtype=np.float64)
    pair_quality_scores = np.zeros(len(sample_pairs), dtype=np.float64)
    pair_quality_score_function = edc_input.get('pair_quality_score_function', min)
    for i, sample_pair in enumerate(sample_pairs):
      similarity_scores[i] = sample_pair['similarity_score']
      sample1, sample2 = sample_pair['samples']
      pair_quality_scores[i] = pair_quality_score_function(
          sample1['quality_score'],
          sample2['quality_score'],
      )

    order_scores = np.argsort(pair_quality_scores)
    pair_quality_scores = pair_quality_scores[order_scores]
    similarity_scores = similarity_scores[order_scores]

    edc_input_numpy: _EdcInputNumpy = {key: value for key, value in edc_input.items() if key != 'sample_pairs'}
    edc_input_numpy['similarity_scores'] = similarity_scores
    edc_input_numpy['pair_quality_scores'] = pair_quality_scores

    return edc_input_numpy


def _form_error_comparison_decision(error_type: EdcErrorType,
                                    similarity_score_or_scores: SimilarityScores,
                                    similarity_score_threshold: SimilarityScore,
                                    out: Optional[np.ndarray] = None):
  if error_type == EdcErrorType.FNMR:
    # FNMR, so non-matches are errors
    return np.less(similarity_score_or_scores, similarity_score_threshold, out=out)
  elif error_type == EdcErrorType.FMR:
    # FMR, so matches are errors
    return np.greater_equal(similarity_score_or_scores, similarity_score_threshold, out=out)


def _compute_edc(
    error_type: EdcErrorType,
    pair_quality_scores: np.ndarray,
    similarity_scores: np.ndarray,
    similarity_score_threshold: SimilarityScore,
) -> EdcOutput:
  """This contains the actual EDC computation.
  The ``similarity_scores`` are linked to the ``pair_quality_scores``,
  and the scores must be sorted by the quality scores.
  """
  assert len(pair_quality_scores) == len(similarity_scores), "Input quality/comparison score count mismatch"

  # The array indices correspond to the discard counts, so 0 comparisons are discarded at index 0.
  comparison_count = len(pair_quality_scores)

  # Compute the (binary) per-comparison errors by comparing the comparison scores against the comparison_threshold:
  error_counts = np.zeros(comparison_count, dtype=np.uint32)
  _form_error_comparison_decision(error_type, similarity_scores, similarity_score_threshold, out=error_counts)
  # Then compute the cumulative error_counts sum:
  # The total error count will be at index 0, which corresponds to 0 discarded comparisons (or samples).
  # Conversely, at index comparison_count-1 only one comparison isn't discarded and the error count remains 0 or 1.
  error_counts = np.flipud(np.cumsum(np.flipud(error_counts), out=error_counts))

  # Usually the EDC should model the effect of discarding samples (instead of individual comparisons) based on
  # a progressively increasing quality threshold. This means that sequences of identical quality scores have to be
  # skipped at once. In this implementation the discard counts are equivalent to the array indices, so computing
  # the relevant array indices for the quality sequence starting points also obtains the corresponding discard counts:
  discard_counts = np.where(pair_quality_scores[:-1] != pair_quality_scores[1:])[0] + 1
  discard_counts = np.concatenate(([0], discard_counts))

  # Subtracting the discard_counts from the total comparison_count results in the remaining_counts:
  remaining_counts = comparison_count - discard_counts
  # Divide the relevant error_counts by the remaining_counts to compute the error_fractions
  error_fractions = error_counts[discard_counts] / remaining_counts
  # Divide the discard_counts by the total comparison_count to compute the discard_fractions:
  discard_fractions = discard_counts / comparison_count

  if discard_fractions[-1] != 1:  # NOTE Add a point at 100% discard for plotting edge cases.
    discard_fractions = np.concatenate((discard_fractions, [1]))
    error_fractions = np.concatenate((error_fractions, [0]))

  edc_output = EdcOutput(
      error_fractions=error_fractions,
      discard_fractions=discard_fractions,
      error_counts=error_counts,
      discard_counts=discard_counts,
      comparison_count=comparison_count,
  )
  return edc_output


def compute_edc_pauc(edc_output: EdcOutput, discard_fraction_limit: float) -> float:
  """This computes the pAUC value for the given EDC curve (with stepwise interpolation).

  Note that this does not automatically subtract the "area under theoretical best" value,
  as done in the paper "Finger image quality assessment features - definitions and evaluation".
  You can get that value by calling :meth:`compute_edc_area_under_theoretical_best` with the same parameters,
  and subtract the result thereof from the pAUC value result of this function
  (``compute_edc_pauc(edc_output, discard_fraction_limit) -
  compute_edc_area_under_theoretical_best(edc_output, discard_fraction_limit)``).

  Parameters
  ----------
  edc_output : EdcOutput
    The EDC curve data as returned by the :meth:`compute_edc` function.
    The required parts are the :class:`EdcOutput.error_fractions` and the :class:`EdcOutput.discard_fractions`.
  discard_fraction_limit : float
    The pAUC value for that discard fraction limit will be computed.
    I.e. if this is 1, the full AUC value is computed.
    Must be in [0,1].

  Returns
  -------
  float
    The computed pAUC value.
  """
  error_fractions, discard_fractions = edc_output['error_fractions'], edc_output['discard_fractions']
  assert len(error_fractions) == len(discard_fractions), 'error_fractions/discard_fractions length mismatch'
  assert discard_fraction_limit >= 0 and discard_fraction_limit <= 1, 'Invalid discard_fraction_limit'
  if discard_fraction_limit == 0:
    return 0
  pauc = 0
  for i in range(len(discard_fractions)):  # pylint: disable=consider-using-enumerate
    if i == (len(discard_fractions) - 1) or discard_fractions[i + 1] >= discard_fraction_limit:
      pauc += error_fractions[i] * (discard_fraction_limit - discard_fractions[i])
      break
    else:
      pauc += error_fractions[i] * (discard_fractions[i + 1] - discard_fractions[i])
  return pauc


def compute_edc_area_under_theoretical_best(edc_output: EdcOutput, discard_fraction_limit: float) -> float:
  """Computes the (partial) "area under theoretical best" up to the specified ``discard_fraction_limit``.

  The "theoretical best" line in an EDC plot is the (straight) line that decreases the
  starting error (the error at the 0% discard fraction, i.e. the :class:`EdcOutput.error_fractions` value at index 0)
  exactly by the increasing discard fraction value,
  stopping once the error becomes 0 (i.e. once the discard fraction is equal to the starting error).
  This is commonly subtracted from the :meth:`compute_edc_pauc` value returned for the same parameters,
  since pAUC values for real curves cannot be lower than this value (hence "theoretical best").

  In other words, this is the area of an isosceles right triangle
  with leg length ("height" & "width") equal to the starting error.
  And if the ``discard_fraction_limit`` happens to be lower than the starting error,
  then this area is "cut off" via the subtraction of the area of another isosceles right triangle,
  with leg length equal to the starting error minus the ``discard_fraction_limit``.

  Parameters
  ----------
  edc_output : EdcOutput
    The EDC curve data as returned by the :meth:`compute_edc` function.
    The only required part is the :class:`EdcOutput.error_fractions` array.
  discard_fraction_limit : float
    The discard fraction cutoff point for the area.

  Returns
  -------
  float
    The computed "area under theoretical best" value.
  """
  starting_error = edc_output['error_fractions'][0]
  area = (starting_error**2) / 2
  if discard_fraction_limit < starting_error:
    area -= ((starting_error - discard_fraction_limit)**2) / 2
  return area


def compute_error_per_discard_count(edc_output: EdcOutput) -> np.ndarray:
  """Compute the error fractions for all discard counts from 0 to the :class:`EdcOutput.comparison_count` - 1
  (with stepwise interpolation).
  This data isn't usually needed for EDC plots, but it can be used e.g. to compute curve combinations.

  Parameters
  ----------
  edc_output: EdcOutput
    The EDC data as returned by the :meth:`compute_edc` function.
    The required parts are the :class:`EdcOutput.comparison_count`, :class:`EdcOutput.error_fractions`,
    and the :class:`EdcOutput.discard_counts`.

  Returns
  -------
  np.ndarray
    An array containing the error fractions with each index being a discard count.
    The size of the array is equal to the :class:`EdcOutput.comparison_count`.
  """
  comparison_count = edc_output['comparison_count']
  error_fractions = edc_output['error_fractions']
  discard_counts = edc_output['discard_counts']
  error_per_discard_count = np.full(comparison_count, -1, dtype=np.float64)
  index = 0
  last_error_fraction = error_fractions[0]
  for next_index, next_error_fraction in zip(discard_counts, error_fractions):
    while index < next_index:
      error_per_discard_count[index] = last_error_fraction
      index += 1
    last_error_fraction = next_error_fraction
  while index < comparison_count:
    error_per_discard_count[index] = last_error_fraction
    index += 1
  return error_per_discard_count
