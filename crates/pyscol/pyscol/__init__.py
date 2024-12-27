from pathlib import Path
import numpy as np
from pydantic import BaseModel
from enum import Enum
from typing import Optional

from .pyscol import (
    clahe_u8_u8,
    clahe_u16_u16,
    clahe_u16_u8,
    trimming_box_with_resample,
    py_draw_coronal,
    py_draw_sagittal,
    py_wrap_in_html,
)


class TrimImageFilter(str, Enum):
    Original = ('original',)
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


def clahe(
    arr: np.ndarray, tile_width: int, tile_height: int, clip_limit: int, tile_sample: float, u8_out: bool
) -> np.ndarray:
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


def default_coronal_hide() -> list[str]:
    return [
        "CSVL",
        "T1TiltAngle",
        "CoronalBalance",
        "ClavicleAngle",
        "ShoulderHeight",
        "PelvicObliquity",
        "SacralObliquity",
        "LegLengthDiscrepancy",
    ]


def default_sagittal_hide() -> list[str]:
    return [
        'SagittalBalance',
        'PelvicTilt',
        'SacralSlope',
        'L5IncidenceAngle',
        'PelvicRadiusAngle',
        'LumbosacralAngle',
    ]


def draw_coronal(
    coronal_points_and_curve_json: str,
    json_path: str | Path,
    draws: list[str],
    hide: Optional[list[str]],
    draw_param_json: str,
    label_colors: dict[str, str],
    line_colors: dict[str, str],
    resize: Optional[str],
    svg_size: Optional[tuple[int, int]],
    overlay: Optional[np.ndarray],
    label_and_point_set: Optional[list[tuple[str, np.ndarray]]],
) -> str:
    json_path = str(json_path)
    if hide is None:
        hide = default_coronal_hide()
    return py_draw_coronal(
        coronal_points_and_curve_json,
        json_path,
        draws,
        hide,
        draw_param_json,
        label_colors,
        line_colors,
        resize,
        svg_size,
        overlay,
        label_and_point_set,
    )


def draw_sagittal(
    coronal_points_json: str,
    json_path: str | Path,
    draws: list[str],
    hide: Optional[list[str]],
    draw_param_json: str,
    label_colors: dict[str, str],
    line_colors: dict[str, str],
    resize: Optional[str],
    svg_size: Optional[tuple[int, int]],
    overlay: Optional[np.ndarray],
    label_and_point_set: Optional[list[tuple[str, np.ndarray]]],
) -> str:
    json_path = str(json_path)
    if hide is None:
        hide = default_sagittal_hide()
    return py_draw_sagittal(
        coronal_points_json,
        json_path,
        draws,
        hide,
        draw_param_json,
        label_colors,
        line_colors,
        resize,
        svg_size,
        overlay,
        label_and_point_set,
    )


def wrap_in_html(svg: str, title: str) -> str:
    return py_wrap_in_html(svg, title)


def calc_resize(image_shape: tuple[int, int], resize_param: str) -> tuple[int, int]:
    from .pyscol import py_calc_resize

    return py_calc_resize(image_shape, resize_param)
