import argparse
import json
import sys
from datetime import datetime
from importlib.metadata import version
from typing import Any, Dict

import numpy as np
import piexif
import pydicom
import pydicom.datadict
import pydicom.tag
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
    parser.add_argument(
        '--tags', nargs='*', help="Dicom tags to save in the comment", default=['PixelSpacing', 'ImagerPixelSpacing']
    )
    args = parser.parse_args()
    tags = [(t, pydicom.datadict.tag_for_keyword(t)) for t in args.tags]
    invalid_tags = [key for (key, tag) in tags if tag is None]
    if invalid_tags:
        print("Invalid tag(s) were specified:", invalid_tags)

    tags = [tag for (_, tag) in tags if tag is not None]

    dcm = pydicom.dcmread(args.input)
    arr = dcm.pixel_array
    if dcm.PhotometricInterpretation == 'MONOCHROME1':
        arr = arr.max() - arr
    d_exif: Dict[str, Any] = {
        'pydicom_version': version('pydicom'),
        'pyscol_version': version('pyscol'),
        'created_at': datetime.now().isoformat(),
        'original_size': list(arr.shape),
    }
    bbox = pyscol.trimming_param(arr)
    trimmed = boundingbox.crop(arr, bbox, margin=0)
    u8trimemd, minmax = normalize(trimmed, args.percentile)
    d_exif['window_minmax'] = minmax.tolist()
    d_exif['crop_params'] = np.stack(bbox).ravel().tolist()
    if tags:
        ds = pydicom.Dataset()
        for tag in tags:
            try:
                v = dcm.get(tag)
                ds.add(v)
            except Exception as e:
                logger.warning('Ignoring exception:%s', e)
        d_exif['dicom_tags'] = ds.to_json_dict()

    img = Image.fromarray(u8trimemd)
    uc = helper.UserComment.dump(json.dumps(d_exif, ensure_ascii=False), encoding='unicode')
    exif = piexif.dump({'Exif': {piexif.ExifIFD.UserComment: uc}})
    img.save(args.output, exif=exif)


if __name__ == '__main__':
    sys.exit(main())
