import argparse
from .lib import las_to_raster


def main():
    """Main function."""

    parser = argparse.ArgumentParser(description="Convert LAS file to GeoTIFF raster.")
    parser.add_argument("file_in", help="Input LAS filename.")
    parser.add_argument("file_out", help="Output GeoTIFF filename.")
    parser.add_argument(
        "--crs",
        required=True,
        help="Coordinate reference system in whatever format"
        " is accepted by the rasterio file constuctor. E.g. 'epsg:2926'.",
    )
    parser.add_argument(
        "-n",
        "--return_num",
        required=True,
        action="append",
        type=int,
        help="Return number(s) to rasterize, each in their own layer. "
        "Negative numbers indicate "
        "position relative to last return; e.g. -1 is the last return.",
    )
    parser.add_argument(
        "-t",
        "--theme",
        required=True,
        action="append",
        type=str,
        help="Theme(s) to inclide. Choices are 'elev' and 'intensity'",
    )
    parser.add_argument(
        "--xres",
        type=float,
        help="Width of one pixel in output GeoTIFF, in the horizontal"
        " units of the CRS. If omitted, the LAS file will be used to"
        " make a reasonable guess.",
    )
    parser.add_argument(
        "--yres",
        type=float,
        help="Height of one pizel in output GeoTIFF, in the horizontal"
        " units of the CRS. If omitted, the LAS file will be used to"
        " make a reasonable guess.",
    )
    parser.add_argument(
        "--fill_radius",
        "-r",
        type=int,
        default=2,
        help="Fill raster holes with average values within FILL_RADIUS pixels.",
    )

    args = parser.parse_args()

    las_to_raster(
        args.file_in,
        args.file_out,
        args.crs,
        args.return_num,
        args.theme,
        args.xres,
        args.yres,
        args.fill_radius,
    )


if __name__ == "__main__":
    main()
