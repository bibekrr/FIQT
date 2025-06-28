"""Modified code based on:
<https://github.com/fdbtrs/CR-FIQA/blob/8ada2ca36b6020d5aff97cf959330e52ea241843/evaluation/FaceModel.py>

Modifications:
- Auto-formatting.
- A complete model file path is used instead of a "model_prefix" & "model_epoch".
- get_feature: Disabled (unused).
- get_batch_feature: Added parameters to control what is returned.
- get_batch_feature: Changed image_path_list to image_list (i.e. it now takes loaded images instead of paths).
- get_batch_feature: batch_size parameter value now defaults to len(image_list).
- __init__: Replace gpu_id attribute with ctx (full pytorch device string).
- __init__: Change ctx_id parameter to ctx (full pytorch device string) and set the default to "cpu".

License of the original file:
Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
(See <https://github.com/fdbtrs/CR-FIQA/tree/8ada2ca36b6020d5aff97cf959330e52ea241843#license>.)
"""

import numpy as np
from sklearn.preprocessing import normalize


class FaceModel():

  def __init__(self, model_path, ctx="cpu", backbone="iresnet50"):
    self.ctx = ctx
    self.image_size = (112, 112)
    self.model_path = model_path
    self.model = self._get_model(
        ctx=ctx, image_size=self.image_size, path=self.model_path, layer="fc1", backbone=backbone)

  def _get_model(self, ctx, image_size, path, layer):
    pass

  def _getFeatureBlob(self, input_blob):
    pass

  # def get_feature(self, image_path):
  #   image = cv2.imread(image_path)
  #   image = cv2.resize(image, (112, 112))
  #   a = np.transpose(image, (2, 0, 1))
  #   input_blob = np.expand_dims(a, axis=0)
  #   emb = self._getFeatureBlob(input_blob)
  #   emb = normalize(emb.reshape(1, -1))
  #   return emb

  def get_batch_feature(
      self,
      image_list,
      batch_size=None,
      return_features=True,
      return_quality_scores=True,
      quality_scores_as_list=True,
  ):
    if batch_size is None:
      batch_size = len(image_list)
    count = 0
    num_batch = int(len(image_list) / batch_size)
    if return_features:
      features = []
    if return_quality_scores:
      quality_score = []
    for i in range(0, len(image_list), batch_size):
      if count < num_batch:
        images = image_list[i:i + batch_size]
      else:
        images = image_list[i:]
      count += 1
      input_blob = np.array(images)

      emb, qs = self._getFeatureBlob(input_blob)

      if return_quality_scores:
        quality_score.append(qs)
      if return_features:
        features.append(emb)
    if return_features:
      features = np.vstack(features)
      features = normalize(features)
    if return_quality_scores:
      quality_score = np.vstack(quality_score)
      if quality_scores_as_list:
        quality_score = quality_score.flatten().tolist()
    if return_features and return_quality_scores:
      return features, quality_score
    elif return_features:
      return features
    elif return_quality_scores:
      return quality_score
