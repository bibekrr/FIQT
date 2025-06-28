"""This module patches numpy by reactivating
`aliases deprecated in version 1.20.0 <https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations>`_,
since some of the included methods still rely on older dependencies which in turn use these aliases:

- np.bool
- np.float
- np.int
"""

import numpy as np

np.bool = bool
np.float = float
np.int = int
