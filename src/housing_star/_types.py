"""Shared type aliases."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
"""A float64 NumPy array of any shape."""

IntArray = NDArray[np.intp]
"""An array of integer indices."""
