import argparse
import json
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict

import numpy as np
from logzero import logger


def normalize(x, percentile):
    minmax = np.percentile(x, [percentile, 100 - percentile])

    x = np.clip(x, minmax[0], minmax[1]).astype(np.float32)
    return np.round(255 * (x - minmax[0]) / (minmax[1] - minmax[0])).astype(np.uint8), minmax


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('input', help='Input dicom filename or a directory', type=Path)
    parser.add_argument('output', help='Output jpeg filename or directory if input is a directory', type=Path)
    parser.add_argument(
        '-p',
        '--percentile',
        help='Percentile for pixel value normalization. default: %(default)s',
        type=float,
        default=5,
    )
    parser.add_argument(
        '-q',
        '--quantile',
        help='Quantile for trimming. default: %(default)s',
        type=float,
        default=0.2,
    )

    parser.add_argument('--ext', help='Input file extension. default: %(default)s', default='.dcm')

    parser.add_argument(
        '--tags', nargs='*', help="Dicom tags to save in the comment", default=['PixelSpacing', 'ImagerPixelSpacing']
    )


def process_file(input: Path, output: Path, tags: list, quantile: float, percentile: float):
    import piexif
    import pydicom
    import pydicom.datadict
    import pydicom.tag
    import pyscol
    from piexif import helper
    from PIL import Image
    from szkmipy import boundingbox

    d_exif: Dict[str, Any] = {
        'pyscol_version': version('pyscol'),
        'created_at': datetime.now().isoformat(),
    }

    if input.suffix == '.dcm':
        dcm = pydicom.dcmread(input)
        arr = dcm.pixel_array
        if dcm.PhotometricInterpretation == 'MONOCHROME1':
            arr = arr.max() - arr
        d_exif['pydicom_version'] = version('pydicom')
    else:
        dcm = None
        arr = np.array(Image.open(input))
    d_exif['original_size'] = list(arr.shape)
    bbox = pyscol.trimming_param(arr, quantile)
    logger.debug('bbox:%s', bbox)
    trimmed = boundingbox.crop(arr, bbox, margin=0)
    u8trimemd, minmax = normalize(trimmed, percentile)
    d_exif['window_minmax'] = minmax.tolist()
    d_exif['crop_params'] = np.stack(bbox).ravel().tolist()
    if tags and dcm:
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
    img.save(output, exif=exif)


def main(args: argparse.Namespace):
    import pydicom.datadict

    tags = [(t, pydicom.datadict.tag_for_keyword(t)) for t in args.tags]
    invalid_tags = [key for (key, tag) in tags if tag is None]
    if invalid_tags:
        print("Invalid tag(s) were specified:", invalid_tags)

    tags = [tag for (_, tag) in tags if tag is not None]

    if args.input.is_dir():
        if not args.output.is_dir():
            print('Output {} is not a directory. Exiting', args.output)
            return 1
        for input in args.input.glob('*' + args.ext):
            output = args.output / input.with_suffix('.jpg').name
            logger.debug("%s, %s", input, output)
            process_file(input, output, tags, args.quantile, args.percentile)
    else:
        if args.output.is_dir():
            print('Output directory exist. Exiting', args.output)
            return 1
        process_file(args.input, args.output, tags, args.quantile, args.percentile)
