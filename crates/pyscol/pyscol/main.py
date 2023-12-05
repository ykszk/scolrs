import argparse
import json
import sys
from pathlib import Path

import numpy as np
import piexif
import pydicom
import pyscol
from logzero import logger
from piexif import helper
from PIL import Image
from szkmipy import boundingbox


def normalize(x, percentile):
    minmax = np.percentile(x, [percentile, 100 - percentile])

    x = np.clip(x, minmax[0], minmax[1]).astype(np.float32)
    return np.round(255 * (x - minmax[0]) / (minmax[1] - minmax[0])).astype(np.uint8), minmax


def main():
    parser = argparse.ArgumentParser(description='pyscol')
    parser.add_argument('input', help='Input dicom filename')
    parser.add_argument('output', help='Output jpeg filename')
    parser.add_argument(
        '-p',
        '--percentile',
        help='Percentile for pixel value normalization. default: %(default)s',
        type=float,
        default=5,
    )
    args = parser.parse_args()

    dcm = pydicom.dcmread(args.input)
    arr = dcm.pixel_array
    if dcm.PhotometricInterpretation == 'MONOCHROME1':
        arr = arr.max() - arr
    d_exif = {'original_size': list(arr.shape)}
    bbox = pyscol.trimming_param(arr)
    trimmed = boundingbox.crop(arr, bbox, margin=0)
    u8trimemd, minmax = normalize(trimmed, args.percentile)
    d_exif['window_minmax'] = minmax.tolist()
    d_exif['crop_params'] = np.stack(bbox).ravel().tolist()
    img = Image.fromarray(u8trimemd)
    uc = helper.UserComment.dump(json.dumps(d_exif, ensure_ascii=False), encoding='unicode')
    exif = piexif.dump({'Exif': {piexif.ExifIFD.UserComment: uc}})
    img.save(args.output, exif=exif)


if __name__ == '__main__':
    sys.exit(main())
