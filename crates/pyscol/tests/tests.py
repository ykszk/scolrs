import unittest
from pathlib import Path

import pydicom
import pyscol
from logzero import logger


class TestTrim(unittest.TestCase):
    def test_trim(self):
        indir = Path("../../../tests/data/dicom")
        for fn in indir.glob("*.dcm"):
            logger.info("%s", fn)
            dcm = pydicom.dcmread(fn)
            arr = dcm.pixel_array
            logger.info("calc")
            print(pyscol.trimming_param(arr))
