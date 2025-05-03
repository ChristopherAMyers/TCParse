import argparse
import json
from .tcparse import parse_from_file, TCJobData

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-t', type=str, required=True,  default=None, help='TeraChem output file')
    arg_parser.add_argument('-j', type=str, required=True,  default=None, help='.json file to save parsed output to')
    arg_parser.add_argument('-x', type=str, required=False, default=None, help='.xyz coordinate file')
    args = arg_parser.parse_args()

    data = parse_from_file(args.t, args.x, args.j)

    new_obj = TCJobData.parse_obj(data)
    return new_obj

if __name__ == '__main__':
    main()