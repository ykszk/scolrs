import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pydicom
import pyscol
from logzero import logger


class Timer:
    def __init__(self, verbose=True):
        self.start = time.perf_counter()
        self.verbose = verbose

    def __call__(self):
        return time.perf_counter() - self.start

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.end = time.perf_counter()
        if self.verbose:
            print(self.end - self.start)


indir = Path("../../../tests/data/dicom")
for fn in indir.glob("*.dcm"):
    logger.info("%s", fn)
    dcm = pydicom.dcmread(fn)
    arr = dcm.pixel_array
    logger.info("calc")
    with Timer() as t:
        print(pyscol.trimming_param(arr))
