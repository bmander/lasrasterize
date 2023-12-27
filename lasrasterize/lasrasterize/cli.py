import argparse
from .lib import convert_las_to_geotiff

def main():
    parser = argparse.ArgumentParser(description='Convert lidar LAS files into geotiff raster files.')
    parser.add_argument('input', type=str, help='Input LAS file')
    parser.add_argument('output', type=str, help='Output geotiff file')
    args = parser.parse_args()

    convert_las_to_geotiff(args.input, args.output)

if __name__ == '__main__':
    main()