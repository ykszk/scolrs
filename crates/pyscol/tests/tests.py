import os
import unittest
from pathlib import Path

import numpy as np
import pydicom
import pyscol
from logzero import logger


def set_loglevel():
    logger.setLevel(os.environ.get('LOGLEVEL', 'INFO').upper())


class TestTrim(unittest.TestCase):
    def setUp(self):
        set_loglevel()

    def test_trim(self):
        indir = Path("../../tests/data/dicom")
        for fn in indir.glob("*.dcm"):
            logger.debug("%s", fn)
            dcm = pydicom.dcmread(fn)
            arr = dcm.pixel_array
            logger.debug("calc")
            logger.debug("param: %s", pyscol.trimming_param(arr, 0.3))


class TestClahe(unittest.TestCase):
    def setUp(self):
        set_loglevel()

    def test_clahe(self):
        indir = Path("../../tests/data/dicom")
        for fn in indir.glob("*.dcm"):
            logger.debug("%s", fn)
            dcm = pydicom.dcmread(fn)
            arr = dcm.pixel_array
            pyscol.clahe(arr, 8, 8, 40, 1)

    def test_clahe_error(self):
        invalid_shape = np.zeros((512, 512, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            pyscol.clahe(invalid_shape, 8, 8, 40, 1)
        invalid_dtype = np.zeros((512, 512), dtype=np.float32)
        with self.assertRaises(ValueError):
            pyscol.clahe(invalid_dtype, 8, 8, 40, 1)
