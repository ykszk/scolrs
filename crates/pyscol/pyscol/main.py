import argparse
import os
import sys

from logzero import logger

from . import cmd_clahe, cmd_trim


def main():
    logger.setLevel(os.environ.get('LOGLEVEL', 'INFO').upper())
    parser = argparse.ArgumentParser(description='pyscol')
    subparsers = parser.add_subparsers()
    trim_parser = subparsers.add_parser('trim', help='Trimming')
    cmd_trim.add_arguments(trim_parser)
    trim_parser.set_defaults(func=cmd_trim.main)
    clahe_parser = subparsers.add_parser('clahe', help='CLAHE')
    cmd_clahe.add_arguments(clahe_parser)
    clahe_parser.set_defaults(func=cmd_clahe.main)

    args = parser.parse_args()
    if not hasattr(args, 'func'):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
