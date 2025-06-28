"""Modified code based on:
<https://github.com/fdbtrs/CR-FIQA/blob/8ada2ca36b6020d5aff97cf959330e52ea241843/evaluation/QualityModel.py>

Modifications:
- Auto-formatting.
- Import order & location.
- A complete model file path is used instead of a "model_prefix" & "model_epoch".
- Disabled the "imgs.div_(255).sub_(0.5).div_(0.5)" [-1,+1] range adjustment (that will be done in external code).
- __init__: Change gpu_id parameter to ctx (full pytorch device string) and set the default to "cpu".
- __init__: Added backbone parameter.

License of the original file:
Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
(See <https://github.com/fdbtrs/CR-FIQA/tree/8ada2ca36b6020d5aff97cf959330e52ea241843#license>.)
"""

import torch

from .iresnet import iresnet100, iresnet50
from .face_model import FaceModel


class QualityModel(FaceModel):

  def __init__(self, model_path, backbone, ctx="cpu"):
    super(QualityModel, self).__init__(model_path, backbone=backbone, ctx=ctx)

  def _get_model(self, ctx, image_size, path, layer, backbone):
    weight = torch.load(path, map_location=ctx)
    if (backbone == "iresnet50"):
      backbone = iresnet50(num_features=512, qs=1, use_se=False).to(ctx)
    else:
      backbone = iresnet100(num_features=512, qs=1, use_se=False).to(ctx)

    backbone.load_state_dict(weight)
    model = torch.nn.DataParallel(backbone, device_ids=[torch.device(ctx)])
    model.eval()
    return model

  @torch.no_grad()
  def _getFeatureBlob(self, input_blob):
    imgs = torch.Tensor(input_blob).to(self.ctx)
    # imgs.div_(255).sub_(0.5).div_(0.5)
    feat, qs = self.model(imgs)
    return feat.cpu().numpy(), qs.cpu().numpy()
