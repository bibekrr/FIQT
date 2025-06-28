"""Derived from the for SER-FIQ repository implementation for ArcFace (InsightFace)."""

import os
import numpy as np
import mxnet as mx

# from sklearn.preprocessing import normalize # NOTE Normalization is currently done later on demand.


class InsightFace:

  def __init__(
      self,
      context=mx.cpu(),
      insightface_path: str = "./insightface/",
      return_dropout: bool = False,
  ):
    """
        Reimplementing a simplified version of Insightface's FaceModel class.

        Parameters
        ----------
        context : mxnet.context.Context, optional
            The mxnet device context. The default is mxnet.cpu().
        insightface_path : str, optional
            The path to the insightface repository.
        return_dropout : bool, optional
            Whether dropout is to be returned by get_feature for SER-FIQ use.

        Returns
        -------
        None.

        """

    self.return_dropout = return_dropout

    sym, arg_params, aux_params = mx.model.load_checkpoint(os.path.join(insightface_path, "models/model"), 0)

    all_layers = sym.get_internals()
    if return_dropout:
      sym_dropout = all_layers['dropout0_output']
    sym_fc1 = all_layers["fc1_output"]

    sym_grouped = mx.symbol.Group([sym_dropout, sym_fc1] if return_dropout else [sym_fc1])

    self.model = mx.mod.Module(symbol=sym_grouped, context=context, label_names=None)
    self.model.bind(data_shapes=[("data", (1, 3, 112, 112))])
    self.model.set_params(arg_params, aux_params)

  def get_feature(self, images):
    """
        Runs the given aligned image on the Mxnet Insightface NN.
        Returns the embedding output.

        Parameters
        ----------
        images : list of numpy ndarrays
            The aligned input image list.

        Returns
        -------
        embedding : numpy ndarray, (512,)
            The arcface embedding of the image.
        dropout : numpy ndarray, (1, 512, 7, 7)
            The output of the dropout0 layer as numpy array.
            Only returned if self.return_dropout is True.
        """
    # input_blob = np.expand_dims(aligned_img, axis=0)
    # data = mx.nd.array(input_blob)
    data = np.array(images, copy=False, dtype=np.float32)
    data = mx.nd.array(data)
    db = mx.io.DataBatch(data=(data,))  # pylint: disable=invalid-name
    self.model.forward(db, is_train=False)
    if self.return_dropout:
      dropouts, embeddings = self.model.get_outputs()
    else:
      embeddings = self.model.get_outputs()
      embeddings = embeddings[0]

    embeddings = list(map(lambda embedding: embedding.asnumpy(), embeddings))
    # embeddings = list(map(lambda embedding: normalize(embedding.asnumpy()).flatten(), embeddings))

    if self.return_dropout:
      dropouts = list(map(lambda dropout: dropout.asnumpy(), dropouts))
      return embeddings, dropouts
    else:
      return embeddings
