import argparse
from .lib import lasfile_to_geotiff, Layerdef


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
        default=None,
        help="Width of one pixel in output GeoTIFF, in the horizontal"
        " units of the CRS. If omitted, the LAS file will be used to"
        " make a reasonable guess.",
    )
    parser.add_argument(
        "--yres",
        type=float,
        default=None,
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

    # make a list of layer definitions
    if len(args.return_num) != len(args.theme):
        raise ValueError(
            "The number of return numbers must match the number of themes."
        )

    layer_defs = []
    for return_num, theme in zip(args.return_num, args.theme):
        if theme not in ("elev", "intensity"):
            raise ValueError("Theme must be 'elev' or 'intensity'.")

        layer_defs.append(
            Layerdef(
                pulse_return=return_num,
                intensity=theme == "intensity",
            )
        )

    lasfile_to_geotiff(
        args.file_in,
        args.file_out,
        layer_defs,
        args.xres,
        args.yres,
        args.fill_radius,
        args.crs,
    )


if __name__ == "__main__":
    main()
