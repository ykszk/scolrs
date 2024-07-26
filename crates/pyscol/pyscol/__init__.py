import numpy as np
from pydantic import BaseModel
from enum import Enum
from typing import Optional

from .pyscol import clahe_u8_u8, clahe_u16_u16, clahe_u16_u8, trimming_box_with_resample
class TrimImageFilter(str, Enum):
    Original = 'original',
    SobelX = 'sobel_x'
    SobelY = 'sobel_y'
    Laplacian = 'laplacian'

class TrimPredicateSource(BaseModel):
    filter: TrimImageFilter
    quantile_min: Optional[float]
    quantile_max: Optional[float]
    raw: bool

    @staticmethod
    def from_tuple(t: tuple[str, Optional[float], Optional[float], bool]):
        return TrimPredicateSource(filter=TrimImageFilter(t[0]), quantile_min=t[1], quantile_max=t[2], raw=t[3])


def trimming_param(arr2d: np.ndarray, pred_source: list[TrimPredicateSource]):
    shape = np.array(arr2d.shape)
    minmax = arr2d.min(), arr2d.max()
    if arr2d.dtype != np.uint8:
        int16arr = np.round(255 / (minmax[1] - minmax[0]) * arr2d).astype(np.int16)
    else:
        int16arr = arr2d.astype(np.int16)
    resample_step = shape.max() // 1000 + 1
    pred_source_json = [t.model_dump_json() for t in pred_source]
    return trimming_box_with_resample(int16arr, pred_source_json, resample_step)


def clahe(arr: np.ndarray, tile_width: int, tile_height: int, clip_limit: int, tile_sample: float, u8_out: bool) -> np.ndarray:
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
        if u8_out:
            return clahe_u16_u8(arr, tile_width, tile_height, clip_limit, tile_sample)
        else:
            return clahe_u16_u16(arr, tile_width, tile_height, clip_limit, tile_sample)
    elif arr.dtype == np.uint8:
        return clahe_u8_u8(arr, tile_width, tile_height, clip_limit, tile_sample)
    else:
        raise ValueError(f"Unsupported dtype: {arr.dtype}")


def ada_minmax(arr: np.ndarray, tile_width: int, tile_height: int, tile_sample: float) -> np.ndarray:
    '''
    Adaptive minmax normalization

    Input is a gray scale image with dtype uint8 or uint16.
    '''
    from .pyscol import ada_minmax_u8_u8, ada_minmax_u16_u16

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
        return ada_minmax_u16_u16(arr, tile_width, tile_height, tile_sample)
    elif arr.dtype == np.uint8:
        return ada_minmax_u8_u8(arr, tile_width, tile_height, tile_sample)
    else:
        raise ValueError(f"Unsupported dtype: {arr.dtype}")
