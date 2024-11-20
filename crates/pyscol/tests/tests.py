import os
import unittest
from pathlib import Path
from typing import Optional

import numpy as np

# import pydicom
import pyscol
from logzero import logger
from PIL import Image


def set_loglevel():
    logger.setLevel(os.environ.get('LOGLEVEL', 'INFO').upper())


# class TestTrim(unittest.TestCase):
#     def setUp(self):
#         set_loglevel()

#     def test_trim(self):
#         indir = Path("../../tests/data/dicom")
#         for fn in indir.glob("*.dcm"):
#             logger.debug("%s", fn)
#             dcm = pydicom.dcmread(fn)
#             arr = dcm.pixel_array
#             logger.debug("calc")
#             logger.debug("param: %s", pyscol.trimming_param(arr, 0.3))


# class TestClahe(unittest.TestCase):
#     def setUp(self):
#         set_loglevel()

#     def test_clahe(self):
#         indir = Path("../../tests/data/dicom")
#         for fn in indir.glob("*.dcm"):
#             logger.debug("%s", fn)
#             dcm = pydicom.dcmread(fn)
#             arr = dcm.pixel_array
#             pyscol.clahe(arr, 8, 8, 40, 1)

#     def test_clahe_error(self):
#         invalid_shape = np.zeros((512, 512, 3), dtype=np.uint8)
#         with self.assertRaises(ValueError):
#             pyscol.clahe(invalid_shape, 8, 8, 40, 1)
#         invalid_dtype = np.zeros((512, 512), dtype=np.float32)
#         with self.assertRaises(ValueError):
#             pyscol.clahe(invalid_dtype, 8, 8, 40, 1)


def test_output_dir() -> Optional[Path]:
    import os

    dir = os.environ.get("TEST_OUTPUT_DIR", None)
    if dir is None:
        return None
    return Path(dir)


class TestDrawCoronal(unittest.TestCase):
    def setUp(self):
        set_loglevel()

    def test_draw_coronal(self):
        data_dir = Path("../../tests/data")
        indir = data_dir / "case2"
        image = np.array(Image.open(indir / "frontal.jpg"))
        with open(indir / "frontal_native.json") as f:
            coronal_points_json = f.read()
        draws = []
        hide = []
        draw_param_json = "{}"
        label_colors = {}
        line_colors = {}
        svg = pyscol.draw_coronal(
            image, coronal_points_json, draws, hide, draw_param_json, label_colors, line_colors, None, None
        )

        out_dir = test_output_dir()
        if out_dir:
            out_dir = out_dir / "python/case2"
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "coronal.svg", "w") as f:
                f.write(svg)
