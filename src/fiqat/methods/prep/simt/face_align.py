"""This is based on insightface's face_align.py, thus the separate file. The original is here:
https://github.com/deepinsight/insightface/blob/f89ecaaa547f12127165fc5b5aefca6d979b228a/recognition/common/face_align.py
Changes from the original:
- Project specific auto-formatting.
- Minor refactoring, mostly to appease pylint (i.a. _init_src_map vs. "src" in the body).
- Allow arbitrary image_size values (src_map & arcface_src were fixed to specific sizes in the original).
"""

import cv2
import numpy as np
from skimage import transform as trans


def _init_src_map():
  src1 = np.array([[51.642, 50.115], [57.617, 49.990], [35.740, 69.007], [51.157, 89.050], [57.025, 89.702]],
                  dtype=np.float32)
  #<--left
  src2 = np.array([[45.031, 50.118], [65.568, 50.872], [39.677, 68.111], [45.177, 86.190], [64.246, 86.758]],
                  dtype=np.float32)
  #---frontal
  src3 = np.array([[39.730, 51.138], [72.270, 51.138], [56.000, 68.493], [42.463, 87.010], [69.537, 87.010]],
                  dtype=np.float32)
  #-->right
  src4 = np.array([[46.845, 50.872], [67.382, 50.118], [72.737, 68.111], [48.167, 86.758], [67.236, 86.190]],
                  dtype=np.float32)
  #-->right profile
  src5 = np.array([[54.796, 49.990], [60.771, 50.115], [76.673, 69.007], [55.388, 89.702], [61.257, 89.050]],
                  dtype=np.float32)
  src = np.array([src1, src2, src3, src4, src5])
  return src / 112


src_map = _init_src_map()

arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32) / 112

arcface_src = np.expand_dims(arcface_src, axis=0)


# lmk is prediction; src is template
def estimate_norm(lmk, image_size=(112, 112), mode="arcface"):
  assert lmk.shape == (5, 2)
  tform = trans.SimilarityTransform()
  lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
  min_transformation_matrix = []
  min_index = []
  min_error = float("inf")
  if mode == "arcface":
    src = arcface_src * image_size
  else:
    src = src_map * image_size
  for i in np.arange(src.shape[0]):
    tform.estimate(lmk, src[i])
    transformation_matrix = tform.params[0:2, :]
    results = np.dot(transformation_matrix, lmk_tran.T)
    results = results.T
    error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
    #         print(error)
    if error < min_error:
      min_error = error
      min_transformation_matrix = transformation_matrix
      min_index = i
  return min_transformation_matrix, min_index


def norm_crop(img, landmark, image_size=(112, 112), mode="arcface"):
  min_transformation_matrix, pose_index = estimate_norm(landmark, image_size, mode)  # pylint: disable=unused-variable
  warped = cv2.warpAffine(img, min_transformation_matrix, image_size, borderValue=0.0)
  return warped
