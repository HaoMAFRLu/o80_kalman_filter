"""
Defines the types used over the package
(for typing hints)
"""

import numpy as np
import typing
from nptyping import NDArray, Shape, Float32



State3d = NDArray[Shape["3"], Float32]
State6d = NDArray[Shape["6"], Float32]  # 3d position and 3d velocity
State9d = NDArray[Shape["9"], Float32]  # 3d position and 3d velocity

Matrix = NDArray[Shape["Any, Any"], Float32]
Vector = NDArray[Shape["Any"], Float32]
Array = NDArray[Shape["Any, ..."], Float32]
