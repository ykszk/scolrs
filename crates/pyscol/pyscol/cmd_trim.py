import argparse
import json
from tqdm import tqdm
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel
import numpy as np
from logzero import logger
import pyscol
import os

logger.setLevel(os.environ.get('LOGLEVEL', 'INFO').upper())


def normalize(x, percentile):
    minmax = np.percentile(x, [percentile, 100 - percentile])

    x = np.clip(x, minmax[0], minmax[1]).astype(np.float32)
    return np.round(255 * (x - minmax[0]) / (minmax[1] - minmax[0])).astype(np.uint8), minmax.astype(np.int64)


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
        '--pred',
        help='Sets of filter, quantie_min, quantile_max, raw. e.g. "sobel_x,None,0.9,False", "sobel_y,0.1" ,. default: %(default)s',
        type=str,
        nargs='*',
        default=['original,0.1,0.9,True', 'laplacian,0.1,0.9,True'],
    )
    parser.add_argument('-j', '--jobs', help='Number of jobs. default: %(default)s', type=int, default=-1)

    parser.add_argument('--input_ext', help='Input file extension. default: %(default)s', default='.dcm')
    parser.add_argument('--output_ext', help='Output file extension. default: %(default)s', default='.jpg')

    parser.add_argument(
        '--tags', nargs='*', help="Dicom tags to save in the comment", default=['PixelSpacing', 'ImagerPixelSpacing']
    )


class CropParams(BaseModel):
    original_size: list[int]
    window_minmax: list[int]
    crop_tlbr: list[int]


class PydicomDataset(BaseModel):
    version: str
    dataset: dict[str, Any]


class ExifComment(BaseModel):
    software: str
    version: str
    created_at: str
    crop_params: Optional[CropParams]
    misc: Dict[str, Any]


def process_file(
    input: Path, output: Path, tags: list, predicate_sources: list[pyscol.TrimPredicateSource], percentile: float
):
    import piexif
    import pydicom
    import pydicom.datadict
    import pydicom.tag
    from piexif import helper
    from PIL import Image
    from szkmipy import boundingbox

    exif_comment = ExifComment(
        software='pyscol.trim',
        version=version('pyscol'),
        created_at=datetime.now().isoformat(),
        crop_params=None,
        misc={},
    )

    if input.suffix == '.dcm':
        dcm = pydicom.dcmread(input)
        arr = dcm.pixel_array
        if dcm.PhotometricInterpretation == 'MONOCHROME1':
            arr = arr.max() - arr
    else:
        dcm = None
        arr = np.array(Image.open(input))
    bbox = pyscol.trimming_param(arr, predicate_sources)
    logger.debug('bbox:%s', bbox)
    trimmed = boundingbox.crop(arr, bbox, margin=0)
    if output.suffix.lower() == '.dcm':
        if dcm is None:
            print(f'Input {input} is not a DICOM file. Cannot save as DICOM.')
            return

        dcm.PixelData = trimmed.tobytes()
        dcm.Rows, dcm.Columns = trimmed.shape
        dcm.PhotometricInterpretation = 'MONOCHROME2'
        dcm.SoftwareVersions = f"pyscol {pyscol.__version__}"
        dcm.ImageComments = json.dumps({'original_size': list(arr.shape), 'crop_tlbr': np.stack(bbox).ravel().tolist()})
        dcm.compress(pydicom.uid.RLELossless)
        dcm.save_as(output)
    else:
        u8trimemd, minmax = normalize(trimmed, percentile)
        crop_param = CropParams(
            original_size=list(arr.shape), window_minmax=minmax.tolist(), crop_tlbr=np.stack(bbox).ravel().tolist()
        )
        exif_comment.crop_params = crop_param
        if tags and dcm:
            ds = pydicom.Dataset()
            for tag in tags:
                v = dcm.get(tag)
                if v is None:
                    continue
                ds.add(v)
            exif_comment.misc['pydicom'] = PydicomDataset(version=pydicom.__version__, dataset=ds.to_json_dict())

        img = Image.fromarray(u8trimemd)
        uc = helper.UserComment.dump(exif_comment.model_dump_json())
        exif = piexif.dump({'Exif': {piexif.ExifIFD.UserComment: uc}})
        img.save(output, exif=exif)


def parse_trim_predicate_source(s: str) -> tuple[str, Optional[float], Optional[float], bool]:
    if len(s) == 0:
        raise ValueError(f"Invalid predicate source: {s}")
    parts: list[Any] = s.split(',')
    for _ in range(len(parts), 4):
        parts.append('None')
    parts[1] = float(parts[1]) if parts[1] != 'None' else None
    parts[2] = float(parts[2]) if parts[2] != 'None' else None
    parts[3] = bool(parts[3]) if parts[3] != 'None' else False
    return tuple(parts)


def main(args: argparse.Namespace):
    import pydicom.datadict

    tags = [(t, pydicom.datadict.tag_for_keyword(t)) for t in args.tags]
    invalid_tags = [key for (key, tag) in tags if tag is None]
    if invalid_tags:
        print("Invalid tag(s) were specified:", invalid_tags)

    tags = [tag for (_, tag) in tags if tag is not None]

    pred_sources = [pyscol.TrimPredicateSource.from_tuple(parse_trim_predicate_source(t)) for t in args.pred]
    logger.debug('pred_sources:%s', pred_sources)

    if args.input.is_dir():
        if not args.output.is_dir():
            print('Output {} is not a directory. Exiting', args.output)
            return 1
        from joblib import Parallel, delayed

        Parallel(n_jobs=args.jobs)(
            delayed(process_file)(
                input, args.output / input.with_suffix(args.output_ext).name, tags, pred_sources, args.percentile
            )
            for input in tqdm(sorted(args.input.glob('*' + args.input_ext)))
        )
        # for input in tqdm(sorted(args.input.glob('*' + args.ext))):
        #     output = args.output / input.with_suffix('.jpg').name
        #     logger.debug("%s, %s", input, output)
        #     process_file(input, output, tags, pred_sources, args.percentile)
    else:
        if args.output.is_dir():
            print('Output directory exist. Exiting', args.output)
            return 1
        process_file(args.input, args.output, tags, pred_sources, args.percentile)
