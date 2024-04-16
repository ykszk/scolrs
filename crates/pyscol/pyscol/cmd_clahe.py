import argparse
from pathlib import Path


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("input", help="Input DICOM directory")
    parser.add_argument("output", help="Output directory")
    parser.add_argument("--ext", help="Output file extension. default: %(default)s", default="jpg")
    parser.add_argument("--trim", action="store_true", help="Trimming")
    parser.add_argument("--compress", action="store_true", help="Compress dicom with RLELossless")
    parser.add_argument("--u8", action="store_true", help="Output as uint8")
    # parser.add_argument("--minmax", action="store_true", help="Apply adaptive minmax-normalization instead of clahe")

    parser.add_argument("--tile_width", type=int, default=8, help="Tile width. default: %(default)s")
    parser.add_argument("--tile_height", type=int, default=8, help="Tile height. default: %(default)s")
    parser.add_argument("--clip_limit", type=int, default=40, help="Clip limit. default: %(default)s")
    parser.add_argument("--tile_sample", type=float, default=1, help="Tile sample. default: %(default)s")


def main(args: argparse.Namespace):
    import numpy as np
    import pydicom
    import pydicom.datadict
    import pydicom.tag
    import pydicom.uid
    import pyscol
    from logzero import logger
    from PIL import Image
    from szkmipy import boundingbox as bb

    for filename in sorted(Path(args.input).glob("*.dcm")):
        logger.debug("%s", filename)
        dcm = pydicom.dcmread(filename)
        arr = dcm.pixel_array
        if dcm.PhotometricInterpretation == "MONOCHROME1":
            arr = 2**dcm.BitsStored - 1 - arr
        if args.trim:
            logger.debug("trimming_param")
            box = pyscol.trimming_param(arr, 0.3)
            logger.debug("boundingbox")
            arr = bb.crop(arr, box)
        if arr.dtype == "int16":
            arr = (arr - arr.min()).astype("uint16")
        if False:  ##args.minmax:
            result = pyscol.ada_minmax(arr, args.tile_width, args.tile_height, args.tile_sample)
        else:
            result = pyscol.clahe(arr, args.tile_width, args.tile_height, args.clip_limit, args.tile_sample, args.u8)
        outdir = Path(args.output)
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / (filename.stem + "." + args.ext)
        logger.debug(f"save: {outpath}")
        if args.ext == 'dcm':
            if result.dtype == 'uint16':
                result = np.round(result * (arr.max() / result.max())).astype('uint16')
            else:
                dcm.BitsStored = 8
            dcm.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
            dcm.PhotometricInterpretation = "MONOCHROME2"
            dcm.SoftwareVersions = 'pydicom suzuki'
            dcm.PixelRepresentation = 0
            dcm.PixelData = result.tobytes()
            if args.compress:
                dcm.compress(pydicom.uid.RLELossless)
            dcm.save_as(outpath)
        else:
            Image.fromarray(result).save(outpath)
