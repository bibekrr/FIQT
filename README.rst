=============================================
Face Image Quality Assessment Toolkit (fiqat)
=============================================

The intended purpose of this toolkit is to facilitate face image quality assessment or face recognition experiments with a simple API.
While e.g. the face detection and image preprocessing parts could be used to help with the training of new models, this toolkit doesn't provide any training-specific utilities.

Toolkit features include:

* **Face detection** (:meth:`fiqat.main_api.detect_faces`): Including face bounding box detection and facial landmark detection.
   
   * **Primary face estimation** (:meth:`fiqat.main_api.estimate_primary_faces`): When multiple faces are detected in an image, but only one is relevant.
   * **Drawing face detector output** (:meth:`fiqat.draw.draw_face_detector_output`): To manually examine the face detector output.
* **Face image preprocessing** (:meth:`fiqat.main_api.preprocess_images`): Cropping and landmark-based alignment of face images for further processing.
* **Face image quality assessment** (:meth:`fiqat.main_api.assess_quality`): Usually a preprocessed image's quality in terms of utility for face recognition.
* **Face recognition feature extraction** (:meth:`fiqat.main_api.extract_face_recognition_features`): Extracting a feature vector from a preprocessed image.
   
   * **Comparison** (:meth:`fiqat.main_api.compute_comparison_scores`): Comparing feature vector pairs for face recognition.
* **"Error versus Discard Characteristic"** (EDC, :mod:`fiqat.edc`): Including "partial Area Under Curve" (pAUC) computation. Used to evaluate quality assessment algorithms with respect to specific face recognition models, see e.g. `"Considerations on the Evaluation of Biometric Quality Assessment Algorithms" <https://ieeexplore.ieee.org/document/10330743>`_ for details.
* **Basic JPEG XL support:** Currently only via the `djxl`/`cjxl` command line tools from https://github.com/libjxl/libjxl/releases. Both for loading & saving, the latter e.g. to facilitate more efficient lossless storage of preprocessed images.
* **Storage helper utility for computed data** (:mod:`fiqat.storage`, fully optional): Using SQLite and a simple custom binary Python object serialization format with included `numpy array <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_ support (:mod:`fiqat.serialize`).
   
   * This is used in example code, but it can easily be replaced with a custom implementation or be fully removed in your own code.
* **Example code for quick experiment setups:** See the :ref:`readme-examples` section.
* **Custom method support:** Methods can easily be registered for all main processing functions, see the `create_edc_plot.py` example in the :ref:`readme-examples` section.

The setup isn't completely automatic, but the default setup without download times should take less than ten minutes.


Download and default configuration file setup
=============================================

#. Download this repository.
#. Download and extract the external dependency package from https://cloud.h-da.de/s/a7pHZqBHptHaHRc (password ``7W4Ei1FjhlV``), which mainly consists of the model files for various included methods, see the :ref:`readme-external-dependency-locations` section. You could omit these dependencies if you don't want to use these methods.
#. In the repository, create a copy of `fiqat_example.toml` at `local/fiqat.toml`. If you want to store this config file at another location, see :meth:`fiqat.config.load_config_data`.
#. Edit the `local/fiqat.toml` by changing the ``models = "/path/to/fiqat/dependencies/"`` path to your local path for the external method file dependency directory.


Python setup using an Anaconda environment
==========================================

First create a new Anaconda environment. The following example names it ``fiqae`` for "Face Image Quality Assessment Environment":

.. code-block:: bash

   conda create -n fiqae python=3.9
   conda activate fiqae

Then install the default Python requirements used by the toolkit's included methods (run this in the ``fiqat`` repository directory):

.. code-block:: bash

   pip install -r requirements.txt
   pip install -r requirements_torch.txt

**Note:** The default `requirements*.txt` (`requirements.txt` & `requirements_torch.txt`) will install dependencies for CPU method execution.
If you want to run certain methods on a GPU, you may have to modify the package installation.
You could also omit certain dependencies (e.g. the ``insightface`` and ``onnxruntime`` packages) if you don't need the corresponding methods.
The `requirements.txt` contains dependencies for some of the toolkit's examples as well.
That's why these dependencies currently are specified by these separate `requirements*.txt` files,
instead of all dependencies being specified as part of the ``fiqat`` package setup (in `pyproject.toml`).

Finally install ``fiqat`` as an `"editable" mode package <https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs>`_ in the new environment (run this in the ``fiqat`` repository directory):

.. code-block:: bash

   pip install --editable .

The ``fiqat`` package should now be available within the ``fiqae`` environment:

.. code-block:: Python

   import fiqat

   # ...


Special ``fd/retinaface`` setup step
------------------------------------

One additional setup step is required if you want to use the ``fd/retinaface`` face detector method:
Run ``make`` in `insightface-f89ecaaa54/detection/RetinaFace` of the external dependency package, using your ``fiqae`` Python environment.


Documentation
=============

You can build the documentation by running ``make html`` in `docs/`, using your previously created Python environment,
which should create HTML documentation files in `docs/_build/html/`.
This uses the ``Sphinx`` Python package, which is specified in the `requirements.txt`.


.. _readme-examples:

Examples
========

These standalone example scripts can be found in the repository's `examples` directory:

* `load_all_methods.py`: This example tries to initialize all methods included in the toolkit. If you successfully installed all dependencies, then all methods should be shown as available.
* `check_methods.py`: This example will run all available face detector, face image quality assessment, and face recognition feature extractor methods. It needs an example image as input for the test runs. With the default configuration, the output of each method will be tested for consistency across multiple runs.
* `face_detection.py`: This example will run the ``fd/scrfd`` face detector for a given input image, and save a new output image with drawn face detector information (such as the facial landmarks).
* `create_edc_plot.py`: This example goes through all the steps necessary to create a simple "Error versus Discard Characteristic" (EDC) plot that compares mutliple face image quality assessment algorithms with respect to face recognition performance. I.e. this example uses all method types of the :mod:`fiqat.main_api` (but not every included method).

   * Intermediate data is stored during computation using the :class:`fiqat.storage.StorageSqlite` helper class, and the script can be aborted and continued using the already computed data. 
   * The example expects input images stored so that the images of each subject are contained in a directory with a subject-specific name (i.e. the images' directory names are the subject identifiers). For instance the extracted `LFW "All images as gzipped tar file (173MB, md5sum a17d05bd522c52d84eca14327a23d494)" <https://vis-www.cs.umass.edu/lfw/>`_ can be used as input.
   * A simple custom example method for quality assessment is defined (``_custom_fiqa_method``), registered (:meth:`fiqat.registry.register_method`), and used (``_assess_quality(storage, image_dicts, fiqa_configs)``) in the example.


Included methods
================

The toolkit currently includes these methods:

* :class:`fiqat.types.MethodType.FACE_DETECTOR`:
   
   * ``fd/dlib``: Separate `face detection <http://dlib.net/face_detector.py.html>`_ and `facial landmark detection (shape_predictor_68_face_landmarks.dat.bz2) <http://dlib.net/face_landmark_detection.py.html>`_ using `dlib <http://dlib.net/>`_.
        
      * Configuration option ``resize_to: Optional[ImageSize]``: Described below.
      * Configuration option  ``detect_faces: bool``: If ``True``, detect faces (i.e. face bounding boxes). Otherwise the whole image area is assumed to depict one face. ``True`` by default.
      * Configuration option ``detect_landmarks: bool``: If ``True``, detect facial landmarks for the detected faces. ``True`` by default.
   
   * ``fd/mtcnn``: MTCNN from https://github.com/deepinsight/insightface/tree/60bb5829b1d76bfcec7930ce61c41dde26413279/deploy (of the "multi-task CNN" approach from `"Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks" <https://arxiv.org/abs/1604.02878>`_).

      * Configuration option ``resize_to: Optional[ImageSize]``: Described below. ``None`` by default.
      * Configuration option ``device_config: DeviceConfig``: Described below.
   
   * ``fd/retinaface``: RetinaFace from https://github.com/deepinsight/insightface/tree/f89ecaaa547f12127165fc5b5aefca6d979b228a/detection/RetinaFace using the "RetinaFace-R50" model.

      * Configuration option ``resize_to: Optional[ImageSize]``: Described below. ``ImageSize(250, 250)`` by default.
      * Configuration option ``device_config: DeviceConfig``: Described below.
      * Configuration option ``batch_size: int``: Described below.

   * ``fd/scrfd``: SCRFD from the `insightface Python package <https://github.com/deepinsight/insightface/tree/48282789fa2e440868b971a4b72fbec7fbc3c049/python-package>`_.

      * Configuration option ``resize_to: Optional[ImageSize]``: Described below. ``ImageSize(160, 160)`` by default.
      * Configuration option ``device_config: DeviceConfig``: Described below.
      * Configuration option ``model_name: str``: Which SCRFD model should be used, see https://github.com/deepinsight/insightface/tree/48282789fa2e440868b971a4b72fbec7fbc3c049/python-package#model-zoo. ``'buffalo_l'`` by default.

* :class:`fiqat.types.MethodType.PRIMARY_FACE_ESTIMATOR`:
   
   * ``pfe/sccpfe``: "Size- and Center- and Confidence-based Primary Face Estimation", which selects the primary face based on the size of the face ROI (:class:`fiqat.types.DetectedFace.roi`), the position of the ROI relative to the image center, and based on the face detector's confidence values.

      * Configuration option ``use_roi: bool``: 
        If ``True``, the ``roi`` data of the :class:`fiqat.types.FaceDetectorOutput.detected_faces` will be used for the estimation.
        The first ``roi`` estimation score factor for each candidate face is the minimum of the ROI's width and height.
        The second factor is only computed if ``input_image_size`` information is available in the :class:`fiqat.FaceDetectorOutput`,
        and favors ROIs that are closer to the image center.
        This second factor is meant to help with cases where multiple face ROIs
        with similar sizes and ``confidence`` values are detected.
        ``True`` by default.
      * Configuration option ``use_landmarks: bool``:
        If ``True``, the bounding box for the ``landmarks`` of the :class:`fiqat.types.FaceDetectorOutput.detected_faces` will be
        used as ROI information,
        with score factor computation as described for ``use_roi``.
        If both ``use_roi`` and ``use_landmarks`` is ``True``, then ``roi`` data will be used whenever available,
        and ``landmarks``-based ROIs are used as fallback.
        ``True`` by default.
      * Configuration option ``use_confidence: bool``:
        If ``True``, the stored ``confidence`` values are used as an estimation score factor,
        normalized relative to the maximum value among the :class:`fiqat.types.FaceDetectorOutput.detected_faces`.
        If either ``use_roi`` or ``use_landmarks`` is ``True`` as well, all factors are combined by multiplication.
        ``True`` by default.

* :class:`fiqat.types.MethodType.PREPROCESSOR`:
   
   * ``prep/crop``: Simple preprocessing method that crops the image to the :class:`fiqat.types.DetectedFace.roi`, then resizes the cropped region to the output size (if specified).

      * Configuration option ``image_size: Optional[ImageSize]``: The size of the output image. If this is ``None``, the cropped region will not be resized. ``None`` by default.
   
   * ``prep/simt``: "Similarity transformation" face image preprocessing/alignment. It crops and aligns the facial image to five facial landmarks, two for the eyes, one of the tip of the nose, and two for the mouth corners, as produced e.g. by ``fd/retinaface``. This approach has been used e.g. in "ArcFace: Additive Angular Margin Loss for Deep Face Recognition", "CosFace: Large Margin Cosine Loss for Deep Face Recognition", "SphereFace: Deep Hypersphere Embedding for Face Recognition".

      * Configuration option ``image_size: Optional[ImageSize]``: The size of the output image. If this is ``None``, the :class:`fiqat.types.DetectedFace.roi` width/height will be used. ``None`` by default.

* :class:`fiqat.types.MethodType.FACE_IMAGE_QUALITY_ASSESSMENT_ALGORITHM`:
   
   * ``fiqa/crfiqa``: CR-FIQA from https://github.com/fdbtrs/CR-FIQA using the "CR-FIQA(S)" or "CR-FIQA(L)" model.

      * Configuration option ``device_config: DeviceConfig``: Described below.
      * Configuration option ``batch_size: int``: Described below.
      * Configuration option ``model_type: str``: Specifies the model that should be used, which must be either ``'CR-FIQA(S)'`` or ``'CR-FIQA(L)'``. **This must be set explicitly.**
      * The model image input size is 112x112. Images are resized via ``cv2.resize(image, (112, 112), interpolation=cv2.INTER_LINEAR)``.

   * ``fiqa/faceqnet``: FaceQnet from https://github.com/javier-hernandezo/FaceQnet using the "FaceQnet v0" or "FaceQnet v1" model.

      * Configuration option ``device_config: DeviceConfig``: Described below.
      * Configuration option ``batch_size: int``: Described below.
      * Configuration option ``model_type: str``: Specifies the model that should be used, which must be either ``'FaceQnet-v0'`` or ``'FaceQnet-v1'``. **This must be set explicitly.**
      * The model image input size is 224x224. Images are resized via ``cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)``.

   * ``fiqa/magface``: MagFace for quality assessment from https://github.com/IrvingMeng/MagFace using the iResNet100-MS1MV2 model (283MB ``magface_epoch_00025.pth`` with sha256sum ``cfeba792dada6f1f30d1e118aff077d493dd95dd76c77c30f57f90fd0164ad58``).

      * Configuration option ``device_config: DeviceConfig``: Described below.
      * Configuration option ``batch_size: int``: Described below.
      * Configuration option ``return_features_and_quality_score: bool``: If ``True``, the method will output dictionaries with a 'features' (:class:`fiqat.types.FeatureVector`) and a `quality_score` entry each, instead of only returning a :class:`fiqat.types.QualityScore` per input. This is an "experimental" option, proper MagFace face recognition method support will be added to the toolkit in a future version. ``False`` by default.
      * The model image input size is 112x112. Images are resized via ``cv2.resize(image, (112, 112), interpolation=cv2.INTER_LINEAR)``.

* :class:`fiqat.types.MethodType.FACE_RECOGNITION_FEATURE_EXTRACTOR`:
   
   * ``fr/arcface``: ArcFace from https://github.com/deepinsight/insightface using the `LResNet100E-IR,ArcFace@ms1m-refine-v2 <https://github.com/deepinsight/insightface/wiki/Model-Zoo/6633390634bcf907c383cc6c90b62b6700df2a8e#31-lresnet100e-irarcfacems1m-refine-v2>`_ model.

      * Configuration option ``device_config: DeviceConfig``: Described below.
      * Configuration option ``batch_size: int``: Described below.
      * The model image input size is 112x112. Images are resized via ``cv2.resize(image, (112, 112), interpolation=cv2.INTER_LINEAR)``.

* :class:`fiqat.types.MethodType.COMPARISON_SCORE_COMPUTATION`:
   
   * ``csc/arcface``: Computes :class:`fiqat.types.SimilarityScore` output for features computed by ``fr/arcface`` (i.e. the cosine score in the range [-1, +1]).

Configuration options can be passed as keyword arguments to the :meth:`fiqat.main_api` functions.

Common options are:

* ``resize_to:`` Optional[:class:`fiqat.types.ImageSize`], for :class:`fiqat.types.MethodType.FACE_DETECTOR` methods: If set, the input images are resized to this size using ``cv2.resize(..., interpolation=cv2.INTER_LINEAR)``, prior to detection.
* ``device_config:`` :class:`fiqat.types.DeviceConfig`: The method supports both CPU and GPU execution. Note that you may need to install Python packages that differ from the `requirements*.txt` to enable GPU support. ``DeviceConfig('cpu', 0)`` by default.
* ``batch_size: int``: The method supports processing input in batches. Larger batch sizes may accelerate processing, but may also require more memory (especially for GPU execution). ``1`` by default.


License information
===================

This "Face Image Quality Assessment Toolkit (fiqat)" itself is released under the MIT License (see the ``LICENSE`` file).

Please note that many of the included methods are based on external projects with their own licenses:

* ``fd/dlib``:
   
   * Dlib itself (http://dlib.net/license.html): Boost Software License - Version 1.0
   * The model used for facial landmark detection (`shape_predictor_68_face_landmarks.dat.bz2`) has additional restrictions according to http://dlib.net/face_landmark_detection.py.html:

      .. code-block:: Python

         #   You can get the trained model file from:
         #   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2.
         #   Note that the license for the iBUG 300-W dataset excludes commercial use.
         #   So you should contact Imperial College London to find out if it's OK for
         #   you to use this model file in a commercial product.

* ``fd/mtcnn`` & ``fr/arcface`` (from the `InsightFace GitHub repository README.md at commit 60bb5829b1d76bfcec7930ce61c41dde26413279 <https://github.com/deepinsight/insightface/tree/60bb5829b1d76bfcec7930ce61c41dde26413279#license>`_), as well as ``fd/retinaface`` (from the `InsightFace GitHub repository README.md at commit f89ecaaa547f12127165fc5b5aefca6d979b228a <https://github.com/deepinsight/insightface/tree/f89ecaaa547f12127165fc5b5aefca6d979b228a#license>`_):

   "The code of InsightFace is released under the MIT License. There is no limitation for both acadmic and commercial usage.

   The training data containing the annotation (and the models trained with these data) are available for non-commercial research purposes only."

* ``fd/scrfd`` (from the `InsightFace Python Library GitHub repository README.md <https://github.com/deepinsight/insightface/tree/48282789fa2e440868b971a4b72fbec7fbc3c049/python-package#license>`_):

   "The code of InsightFace Python Library is released under the MIT License. There is no limitation for both academic and commercial usage.

   The pretrained models we provided with this library are available for non-commercial research purposes only, including both auto-downloading models and manual-downloading models."

* ``fiqa/crfiqa`` (from the `CR-FIQA GitHub repository README.md <https://github.com/fdbtrs/CR-FIQA/tree/d93936b3d65ac957b758bea0735e8d2f2e32a807#license>`_):

   "This project is licensed under the terms of the Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license.
   Copyright (c) 2021 Fraunhofer Institute for Computer Graphics Research IGD Darmstadt"

* ``fiqa/faceqnet`` (according to `the author Javier Hernandez-Ortega in GitHub <https://github.com/javier-hernandezo/FaceQnet/issues/11#issuecomment-631303532>`_):

   "FaceQnet is free of use for non-profit projects. If you wish to use it in your project, just please give us the proper recognition as authors, for example citing the FaceQnet ICB paper."

* ``fiqa/magface`` (`according to the GitHub repository <https://github.com/IrvingMeng/MagFace/tree/1c22ba423fe768e06c3700296dc971e0b9d27e77>`_): Apache License 2.0

Directly used Python packages that are available on https://pypi.org/:

* In ``pyproject.toml``:

   * `colorama <https://pypi.org/project/colorama/>`_ (state 2023-12-19): `BSD-3-Clause License <https://github.com/tartley/colorama/blob/master/LICENSE.txt>`_
   * `opencv-python <https://pypi.org/project/opencv-python/>`_ (state 2023-12-19): `MIT License, Apache License 2.0, and other licenses for various parts (see link) <https://github.com/opencv/opencv-python/tree/4.x#licensing>`_
   * `Pillow <https://pypi.org/project/Pillow/>`_ (state 2023-12-19): `"Historical Permission Notice and Disclaimer" (HPND) License <https://github.com/python-pillow/Pillow/blob/main/LICENSE>`_
   * `termcolor <https://pypi.org/project/termcolor/>`_ (state 2023-12-19): `MIT License <https://github.com/termcolor/termcolor/blob/main/COPYING.txt>`_
   * `toml <https://pypi.org/project/toml/>`_ (state 2023-12-19): `MIT License <https://github.com/uiri/toml/blob/master/LICENSE>`_
   * `tqdm <https://pypi.org/project/tqdm/>`_ (state 2023-12-19): `MIT License, Mozilla Public License 2.0 (MPL 2.0) <https://github.com/tqdm/tqdm/blob/master/LICENCE>`_

* In ``requirements.txt``:

   * `dlib <https://pypi.org/project/dlib/>`_ (state 2023-12-19): `Boost Software License <http://dlib.net/license.html>`_
   * `insightface <https://pypi.org/project/insightface/>`_ (state 2023-12-19): `MIT License, with restrictions regarding pretrained models (see link and quotes in the method-specific list above) <https://github.com/deepinsight/insightface/tree/master/python-package#license>`_
   * `mxnet-native <https://pypi.org/project/mxnet-native/>`_ (state 2023-12-19): `Apache License 2.0 <https://github.com/apache/mxnet/blob/master/LICENSE>`_
   * `numpy <https://pypi.org/project/numpy/>`_ (state 2023-12-19): `BSD-3-Clause License <https://github.com/numpy/numpy/blob/main/LICENSE.txt>`_ and `various for bundled parts (see link) <https://github.com/numpy/numpy/blob/main/LICENSES_bundled.txt>`_
   * `onnxruntime <https://pypi.org/project/onnxruntime/>`_ (state 2023-12-19): `MIT License <https://github.com/microsoft/onnxruntime/blob/main/LICENSE>`_
   * `plotly <https://pypi.org/project/plotly/>`_ (state 2023-12-19): `MIT License <https://github.com/plotly/plotly.py/blob/master/LICENSE.txt>`_
   * `scikit-image <https://pypi.org/project/scikit-image/>`_ (state 2023-12-19): `Various (see link) <https://github.com/scikit-image/scikit-image/blob/main/LICENSE.txt>`_
   * `scikit-learn <https://pypi.org/project/scikit-learn/>`_ (state 2023-12-19): `BSD-3-Clause License <https://github.com/scikit-learn/scikit-learn/blob/main/COPYING>`_
   * `Sphinx <https://pypi.org/project/Sphinx/>`_ (state 2023-12-19): `Various (see link) <https://github.com/sphinx-doc/sphinx/blob/master/LICENSE.rst>`_
   * `sphinx-rtd-theme <https://pypi.org/project/sphinx-rtd-theme/>`_ (state 2023-12-19): `MIT License <https://github.com/readthedocs/sphinx_rtd_theme/blob/master/LICENSE>`_
   * `tensorflow-cpu <https://pypi.org/project/tensorflow-cpu/>`_ (state 2023-12-19): `Apache License 2.0 <https://github.com/tensorflow/tensorflow/blob/master/LICENSE>`_

* In ``requirements_torch.txt``:

   * `torch <https://pypi.org/project/torch/>`_ (state 2023-12-19): `BSD-style license (see link) <https://github.com/pytorch/pytorch/blob/main/LICENSE>`_
   * `torchvision <https://pypi.org/project/torchvision/>`_ (state 2023-12-19): `BSD-3-Clause License <https://github.com/pytorch/vision/blob/main/LICENSE>`_


Known issues and limitations
============================

* The toolkit is currently patching a few deprecated `numpy` Python type aliases, in :mod:`fiqat.patch`, due to the dependencies of some of the included methods. But this probably should not be an issue for any experiment code.
* As noted above, the default `requirements*.txt` only installs dependencies to support CPU execution of the included methods. The dependencies need to be manually adjusted if you want to run methods with GPU support on a GPU.
* Included methods may print internal information as they run.
* The documentation currently does not provide recommendations on which methods may be preferable.


.. _readme-external-dependency-locations:

External dependency package locations
=====================================

For the included methods, the relevant files are located as follows within the external dependency package:

* ``fd/dlib``: `dlib/shape_predictor_68_face_landmarks.dat`
* ``fd/mtcnn``: `insightface-60bb5829b1/deploy`
* ``fd/retinaface``: `insightface-f89ecaaa54/detection/RetinaFace` & `insightface-f89ecaaa54/models/retinaface-R50`
* ``fd/scrfd``: `insightface/models/buffalo_l`
* ``fiqa/crfiqa``: `crfiqa`
* ``fiqa/faceqnet``: `faceqnet`
* ``fiqa/magface``: `MagFace`
* ``fr/arcface``: `insightface-60bb5829b1/models`
