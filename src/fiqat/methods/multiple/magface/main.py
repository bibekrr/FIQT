"""Implements a FIQA method using MagFace.
MagFace can also be used for face recognition,
and feature vector computation can be used by enabling it via the
`MagfaceConfig.return_features_and_quality_score` setting,
but it is not explicitly supported via a method registry entry yet.

The implementation is in part based on
<https://github.com/IrvingMeng/MagFace/blob/0722687e2efe911d1c960d4d7c3cbc615970a63b/inference/gen_feat.py>
and <https://github.com/IrvingMeng/MagFace/blob/main/inference/examples.ipynb>.
The MagFace repository code has the Apache License 2.0.

External file dependencies:
The current implementation is using the magface_epoch_00025.pth file
linked in the <https://github.com/IrvingMeng/MagFace> repository
as <https://drive.google.com/file/d/1Bd87admxOZvbIOAyTkGEntsEz3fyMt7H/>.
"""

# Standard imports:
from typing import Iterable, TypedDict, Union
from pathlib import Path

# External imports:
import numpy as np
import cv2
from torchvision import transforms
import torch.utils.data as data
import torch.utils.data.distributed
import torch

# Local imports:
# pylint: disable=relative-beyond-top-level
from .... import registry
from ....types import MethodRegistryEntry, MethodId
from ....types import InputImage, ImageChannelType
from ....types import DeviceConfig, QualityScore, FeatureVector
from ....iterate import iterate_as_batches
from ....image import load_input_image
from ....internal import get_config_path_entry
from .network_inf import builder_inf


class MagfaceConfig(TypedDict):
  device_config: DeviceConfig
  batch_size: int
  return_features_and_quality_score: bool
  """Per input image, return both the features usable for face recognition as well as the quality score,
  as a MagfaceResult."""


class MagfaceResult(TypedDict):
  features: FeatureVector
  """Feature vector usable for face recognition."""
  quality_score: float
  """Quality score."""


class Args:
  """Replaces data read via argparse in the original gen_feat.py code."""
  __slots__ = [
      'resume',  # path to latest checkpoint
      'arch',  # backbone architechture
      'workers',  # number of data loading workers
      'cpu_mode',  # This isn't part of the original code / arguments.
      'embedding_size',  # The embedding feature size
  ]

  def __init__(self, resume, arch, workers, cpu_mode):
    self.resume = resume
    self.arch = arch
    self.workers = workers
    self.cpu_mode = cpu_mode
    self.embedding_size = 512


def _get_model_path() -> Path:
  dependency_path = get_config_path_entry('magface/models', 'MagFace')
  model_path = dependency_path / 'magface_epoch_00025.pth'
  return model_path


def _load_model(args: Args):
  model = builder_inf(args)
  model = torch.nn.DataParallel(model)
  if not args.cpu_mode:
    model = model.cuda()
  model.eval()  # switch to evaluate mode
  return model


class ImgLoader(data.Dataset):

  def __init__(self, image_load_data, transform=None):
    self.image_load_data = image_load_data
    self.transform = transform

  def __getitem__(self, index):
    input_image = self.image_load_data[index]
    image = load_input_image(input_image, ImageChannelType.BGR).image
    image = cv2.resize(image, (112, 112), interpolation=cv2.INTER_LINEAR)
    return self.transform(image)

  def __len__(self):
    return len(self.image_load_data)


def _process(
    _method_registry_entry: MethodRegistryEntry,
    config: MagfaceConfig,
    input_images: Iterable[InputImage],
) -> Iterable[Union[QualityScore, MagfaceResult]]:
  return_features_and_qs = config['return_features_and_quality_score']

  model_path = _get_model_path()
  cpu_mode = config['device_config'].type == 'cpu'
  args = Args(
      resume=model_path,
      arch='iresnet100',
      workers=4,
      cpu_mode=cpu_mode,
  )
  model = _load_model(args)

  trans = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.]),
  ])
  batch_size = config.get('batch_size', 1)
  for input_image_batch in iterate_as_batches(input_images, batch_size):
    img_loader = ImgLoader(image_load_data=input_image_batch, transform=trans)
    data_loader = torch.utils.data.DataLoader(
        img_loader, batch_size=batch_size, num_workers=args.workers, pin_memory=True, shuffle=False)
    with torch.no_grad():
      for image_batch in data_loader:
        embedding_feat_batch = model(image_batch)
        for embedding_feat in embedding_feat_batch:
          features = embedding_feat.data.cpu().numpy()
          magnitude = np.linalg.norm(features)  # i.e. the quality score
          if return_features_and_qs:
            yield MagfaceResult(features=features, quality_score=magnitude)
          else:
            yield magnitude


method_registry_entry = MethodRegistryEntry(
    method_id=MethodId('fiqa/magface'),
    process=_process,
    default_config=MagfaceConfig(
        device_config=DeviceConfig('cpu', 0),
        return_features_and_quality_score=False,
    ),
)

registry.register_method(method_registry_entry)
