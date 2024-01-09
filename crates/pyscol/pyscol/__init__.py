import numpy as np

from .pyscol import clahe_u8_u8, clahe_u16_u8, trimming_box_with_resample


def trimming_param(arr2d: np.ndarray, thresh_quantile: float):
    shape = np.array(arr2d.shape)
    minmax = arr2d.min(), arr2d.max()
    if arr2d.dtype != np.uint8:
        int16arr = np.round(255 / (minmax[1] - minmax[0]) * arr2d).astype(np.int16)
    else:
        int16arr = arr2d.astype(np.int16)
    resample_step = shape.max() // 1000 + 1
    return trimming_box_with_resample(int16arr, thresh_quantile, resample_step)


def clahe(arr: np.ndarray, tile_width: int, tile_height: int, clip_limit: int, tile_sample: float) -> np.ndarray:
    '''
    Contrast Limited Adaptive Histogram Equalization

    Input is a gray scale image with dtype uint8 or uint16.
    '''
    if arr.ndim == 2:
        pass
    elif arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        else:
            raise ValueError(f"Unsupported shape: {arr.shape}")
    else:
        raise ValueError(f"Unsupported shape: {arr.shape}")
    arr = np.ascontiguousarray(arr)
    if arr.dtype == np.uint16:
        return clahe_u16_u8(arr, tile_width, tile_height, clip_limit, tile_sample)
    elif arr.dtype == np.uint8:
        return clahe_u8_u8(arr, tile_width, tile_height, clip_limit, tile_sample)
    else:
        raise ValueError(f"Unsupported dtype: {arr.dtype}")
