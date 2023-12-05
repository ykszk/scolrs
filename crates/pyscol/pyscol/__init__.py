import numpy as np

from .pyscol import trimming_param_with_resample


def trimming_param(arr2d: np.ndarray):
    shape = np.array(arr2d.shape)
    minmax = arr2d.min(), arr2d.max()
    if arr2d.dtype != np.uint8:
        int16arr = np.round(255 / (minmax[1] - minmax[0]) * arr2d).astype(np.int16)
    else:
        int16arr = arr2d.astype(np.int16)
    resample_step = shape.max() // 1000 + 1
    return trimming_param_with_resample(int16arr, resample_step)
