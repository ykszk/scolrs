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

    def test_draw_case2(self):
        data_dir = Path("../../tests/data")
        indir = data_dir / "case2"
        for stem in ["frontal", "lateral"]:
            json_path = indir / f"{stem}_native.json"
            with open(json_path) as f:
                coronal_points_json = f.read()
            draws = []
            hide = None
            draw_param_json = "{}"
            label_colors = {}
            line_colors = {}
            resize = "1024x1024"
            svg_size = None
            overlay = None
            label_and_point_set = []
            args = (
                coronal_points_json,
                str(json_path),
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
            if stem == "frontal":
                svg = pyscol.draw_coronal(*args)
            else:
                svg = pyscol.draw_sagittal(*args)

            out_dir = test_output_dir()
            if out_dir:
                out_dir = out_dir / "python/case2"
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_dir / f"{stem}.svg", "w") as f:
                    f.write(svg)

                html = pyscol.wrap_in_html(svg, "test case2")
                with open(out_dir / f"{stem}.html", "w") as f:
                    f.write(html)


class TestResize(unittest.TestCase):
    def setUp(self):
        set_loglevel()

    def test_resize(self):
        resize_param = "1024x1024"
        image_shape = (512, 768)
        self.assertEqual(pyscol.calc_resize(image_shape, resize_param), (683, 1024))
